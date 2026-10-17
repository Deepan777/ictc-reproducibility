"""Immutable N06 development-pilot runner and post-manifest outcome sealing."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from ictc.ns3.stochastic import (
    PathAction,
    StochasticScenario,
    classify_physical_benefit,
    load_stochastic_matrix,
    read_action_evidence,
    sha256,
    validate_paired_pre_decision,
)

REGISTRY_FIELDS = (
    "scenario_id",
    "action",
    "seed",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "config_sha256",
    "executable_sha256",
    "manifest_path",
    "failure_classification",
)


def _utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _canonical_digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest().upper()


def _files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "action_manifest.json"
    }


def _write_new_json(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _timeout_text(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value or ""


class N06PilotRunner:
    """Execute only pre-reviewed N06 development seeds with immutable evidence."""

    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
        scenarios: tuple[StochasticScenario, ...] | None = None,
        source_path: Path | None = None,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.executable = executable.resolve()
        self.runtime_directories = tuple(path.resolve() for path in runtime_directories)
        if not self.executable.is_file() or any(
            not path.is_dir() for path in self.runtime_directories
        ):
            raise FileNotFoundError("N06 executable or runtime directory is missing")
        self.scenarios = (
            load_stochastic_matrix(self.matrix_path) if scenarios is None else scenarios
        )
        if not self.scenarios:
            raise ValueError("N06 runner requires at least one planned scenario")
        self.executable_sha256 = sha256(self.executable)
        self.source_path = (
            self.repository / "ns3" / "scratch" / "ictc_stochastic_paths.cc"
            if source_path is None
            else source_path.resolve()
        )
        if not self.source_path.is_file():
            raise FileNotFoundError("N06 native source is missing")
        self.source_sha256 = sha256(self.source_path)
        self.config_sha256 = _canonical_digest(
            {
                "matrix_sha256": sha256(self.matrix_path),
                "source_sha256": self.source_sha256,
                "executable_sha256": self.executable_sha256,
            }
        )
        self.registry_path = self.output_root / "EXPERIMENT_REGISTRY.csv"

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        metadata = {
            "schema_version": "n06-pilot-journal-v1",
            "phase": "development",
            "config_sha256": self.config_sha256,
            "matrix_sha256": sha256(self.matrix_path),
            "source_sha256": self.source_sha256,
            "executable_sha256": self.executable_sha256,
            "scenario_groups": len(self.scenarios),
            "action_executions": sum(len(item.actions) for item in self.scenarios),
            "reserved_confirmatory_seeds_executed": False,
        }
        manifest = self.output_root / "n06_manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != metadata:
                raise ValueError("refusing to mix a different N06 freeze into this journal")
        else:
            _write_new_json(manifest, metadata)
        if not self.registry_path.exists():
            with self.registry_path.open("x", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writeheader()

    def _append(self, **values: object) -> None:
        row = {field: str(values.get(field, "")) for field in REGISTRY_FIELDS}
        with self.registry_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writerow(row)

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        prefix = os.pathsep.join(str(path) for path in self.runtime_directories)
        environment["PATH"] = f"{prefix}{os.pathsep}{environment['PATH']}"
        return environment

    def _valid_action(
        self,
        target: Path,
        scenario: StochasticScenario,
        action: PathAction,
        decision_manifest_sha256: str | None = None,
    ) -> bool:
        manifest_path = target / "action_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
            valid = (
                manifest["schema_version"] == "n06-action-v1"
                and manifest["scenario_id"] == scenario.scenario_id
                and manifest["action"] == action.value
                and manifest["config_sha256"] == self.config_sha256
                and manifest["executable_sha256"] == self.executable_sha256
                and (
                    decision_manifest_sha256 is None
                    or manifest.get("decision_manifest_sha256") == decision_manifest_sha256
                )
                and isinstance(files, dict)
                and bool(files)
                and all(
                    isinstance(name, str)
                    and isinstance(digest, str)
                    and (target / name).is_file()
                    and sha256(target / name) == digest
                    for name, digest in files.items()
                )
            )
            if valid:
                read_action_evidence(target / "raw", scenario, action)
            return bool(valid)
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _failed_target(self, scenario: StochasticScenario, action: PathAction) -> Path:
        root = self.output_root / "failures" / scenario.scenario_id
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (root / f"{action.value}-attempt-{index:03d}").exists():
            index += 1
        return root / f"{action.value}-attempt-{index:03d}"

    def _retain_failure(
        self,
        *,
        staged: Path,
        scenario: StochasticScenario,
        action: PathAction,
        common: Mapping[str, object],
        classification: str,
        detail: object,
    ) -> Path:
        _write_new_json(
            staged / "failure.json",
            {"classification": classification, "detail": str(detail)},
        )
        failed = self._failed_target(scenario, action)
        shutil.move(str(staged), str(failed))
        self._append(
            **common,
            status="FAILED",
            completed_at_utc=_utcnow(),
            manifest_path=failed,
            failure_classification=classification,
        )
        return failed

    def _execute_action(
        self,
        scenario: StochasticScenario,
        action: PathAction,
        *,
        decision_manifest_sha256: str | None = None,
    ) -> Path:
        group = self.output_root / "scenarios" / scenario.scenario_id
        group.mkdir(parents=True, exist_ok=True)
        target = group / action.value
        if target.exists():
            if self._valid_action(target, scenario, action, decision_manifest_sha256):
                return target
            raise FileExistsError(f"refusing to replace invalid existing N06 action: {target}")
        started = _utcnow()
        common = {
            "scenario_id": scenario.scenario_id,
            "action": action.value,
            "seed": scenario.seed,
            "started_at_utc": started,
            "config_sha256": self.config_sha256,
            "executable_sha256": self.executable_sha256,
        }
        self._append(**common, status="STARTED")
        with tempfile.TemporaryDirectory(prefix=f".{action.value}.", dir=group) as temporary:
            staged = Path(temporary)
            raw = staged / "raw"
            command = [str(self.executable), *scenario.command_arguments(action, raw)]
            before = time.perf_counter_ns()
            try:
                completed = subprocess.run(
                    command,
                    env=self._environment(),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                (staged / "stdout.txt").write_text(_timeout_text(error.stdout), encoding="utf-8")
                (staged / "stderr.txt").write_text(_timeout_text(error.stderr), encoding="utf-8")
                failed = self._retain_failure(
                    staged=staged,
                    scenario=scenario,
                    action=action,
                    common=common,
                    classification="SIMULATOR_TIMEOUT",
                    detail=error,
                )
                raise RuntimeError(f"N06 simulator timeout retained at {failed}") from error
            elapsed_ns = time.perf_counter_ns() - before
            (staged / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (staged / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                failed = self._retain_failure(
                    staged=staged,
                    scenario=scenario,
                    action=action,
                    common=common,
                    classification="SIMULATOR_NONZERO_EXIT",
                    detail={"returncode": completed.returncode, "elapsed_ns": elapsed_ns},
                )
                raise RuntimeError(f"N06 simulator failed; evidence retained at {failed}")
            try:
                read_action_evidence(raw, scenario, action)
            except (OSError, ValueError) as error:
                failed = self._retain_failure(
                    staged=staged,
                    scenario=scenario,
                    action=action,
                    common=common,
                    classification="EVIDENCE_VALIDATION_ERROR",
                    detail=error,
                )
                raise RuntimeError(f"N06 invalid raw evidence retained at {failed}") from error
            manifest_payload: dict[str, object] = {
                "schema_version": "n06-action-v1",
                "scenario_id": scenario.scenario_id,
                "action": action.value,
                "seed": scenario.seed,
                "run": scenario.run,
                "config_sha256": self.config_sha256,
                "executable_sha256": self.executable_sha256,
                "command": command,
                "elapsed_ns": elapsed_ns,
                "returncode": completed.returncode,
                "files": _files(staged),
            }
            if decision_manifest_sha256 is not None:
                manifest_payload["decision_manifest_sha256"] = decision_manifest_sha256
            _write_new_json(staged / "action_manifest.json", manifest_payload)
            shutil.move(str(staged), str(target))
        self._append(
            **common,
            status="COMPLETE",
            completed_at_utc=_utcnow(),
            manifest_path=target / "action_manifest.json",
        )
        return target

    def _seal_group(self, scenario: StochasticScenario, targets: dict[PathAction, Path]) -> None:
        group = self.output_root / "scenarios" / scenario.scenario_id
        evidence = {
            action: read_action_evidence(target / "raw", scenario, action)
            for action, target in targets.items()
        }
        action_manifests = {
            action.value: sha256(target / "action_manifest.json")
            for action, target in targets.items()
        }
        expected_group = {
            "schema_version": "n06-group-v1",
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "run": scenario.run,
            "config_sha256": self.config_sha256,
            "pre_decision_sha256": validate_paired_pre_decision(evidence),
            "action_manifest_sha256": action_manifests,
        }
        group_manifest = group / "group_manifest.json"
        if group_manifest.exists():
            if json.loads(group_manifest.read_text(encoding="utf-8")) != expected_group:
                raise ValueError("N06 group manifest does not match validated action evidence")
        else:
            _write_new_json(group_manifest, expected_group)
        keep = evidence[PathAction.KEEP].outcome
        label_payload = {
            "schema_version": "n06-physical-benefit-v1",
            "scenario_id": scenario.scenario_id,
            "group_manifest_sha256": sha256(group_manifest),
            "action_manifest_sha256": action_manifests,
            "delay_margin_ns": 500_000,
            "delivery_ratio_tolerance": 0.01,
            "labels": {
                action.value: classify_physical_benefit(keep, item.outcome).value
                for action, item in evidence.items()
                if action is not PathAction.KEEP
            },
            "semantic_boundary": (
                "Empirical paired physical benefit only; not graphical identification, "
                "causal-admissibility, or an individual controller certificate."
            ),
        }
        label_path = group / "physical_benefit.json"
        if label_path.exists():
            if json.loads(label_path.read_text(encoding="utf-8")) != label_payload:
                raise ValueError("N06 physical-benefit seal does not match current raw evidence")
        else:
            _write_new_json(label_path, label_payload)

    def run(self, *, limit: int | None = None) -> dict[str, int]:
        self.initialise()
        selected = self.scenarios if limit is None else self.scenarios[:limit]
        for scenario in selected:
            targets = {
                action: self._execute_action(scenario, action) for action in scenario.actions
            }
            self._seal_group(scenario, targets)
        return {
            "scenario_groups": len(selected),
            "action_executions": sum(len(item.actions) for item in selected),
            "reserved_confirmatory_seeds_executed": 0,
        }

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete_groups = 0
        complete_actions = 0
        for scenario in self.scenarios:
            targets = {
                action: self.output_root / "scenarios" / scenario.scenario_id / action.value
                for action in scenario.actions
            }
            present = {action: target for action, target in targets.items() if target.exists()}
            if not present:
                continue
            if len(present) != len(targets) or any(
                not self._valid_action(target, scenario, action)
                for action, target in present.items()
            ):
                raise ValueError(f"N06 scenario is incomplete or invalid: {scenario.scenario_id}")
            self._seal_group(scenario, present)
            complete_groups += 1
            complete_actions += len(present)
        if require_complete and complete_groups != len(self.scenarios):
            raise ValueError("N06 development matrix is incomplete")
        return {
            "complete_scenario_groups": complete_groups,
            "complete_action_executions": complete_actions,
            "planned_scenario_groups": len(self.scenarios),
            "planned_action_executions": sum(len(item.actions) for item in self.scenarios),
        }


def load_registry_latest(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    """Return the latest append-only row per scenario/action for independent audit."""
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if rows and set(rows[0]) != set(REGISTRY_FIELDS):
        raise ValueError("N06 registry schema is invalid")
    return {(row["scenario_id"], row["action"]): cast(dict[str, str], row) for row in rows}
