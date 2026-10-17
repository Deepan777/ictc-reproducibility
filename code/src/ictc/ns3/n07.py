"""N07 probe-first method decisions with byte-bound outcome execution."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n06 import N06PilotRunner
from ictc.ns3.stochastic import (
    PathAction,
    StochasticScenario,
    read_action_evidence,
    sha256,
    validate_paired_pre_decision,
)


class MethodId(StrEnum):
    C0_CONTRACT_QUEUE = "C0_CONTRACT_QUEUE"
    B0_KEEP = "B0_KEEP"
    B1_REROUTE = "B1_REROUTE"
    B2_PROPAGATION = "B2_PROPAGATION"
    B3_CAPACITY = "B3_CAPACITY"
    B4_QUEUE = "B4_QUEUE"


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    scenario_id: str
    seed: int
    path_ids: tuple[str, ...]
    latest_timestamp_ns: int
    latest_queues: dict[str, int]
    pre_decision_digest: str
    telemetry_sha256: str
    background_sha256: str


PATH_SPECS: dict[str, dict[str, tuple[float, float]]] = {
    "two-balanced": {"current": (5.0, 5.0), "candidate": (5.0, 1.0)},
    "two-tradeoff": {"current": (8.0, 4.0), "candidate": (3.0, 1.0)},
    "three-heterogeneous": {
        "current": (6.0, 5.0),
        "candidate": (4.0, 1.0),
        "alternate": (8.0, 3.0),
    },
}
ACTION_BY_PATH = {
    "current": PathAction.KEEP,
    "candidate": PathAction.REROUTE,
    "alternate": PathAction.ALTERNATE,
}
_SUMMARY_FIELDS = [
    "scenario_id",
    "topology_family",
    "workload",
    "event",
    "action",
    "seed",
    "run",
    "tx_packets",
    "rx_packets",
    "lost_packets",
    "mean_delay_ns",
    "jitter_ns",
    "throughput_bps",
    "first_tx_ns",
    "last_rx_ns",
]


def _digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest().upper()


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_probe_evidence(raw: Path, scenario: StochasticScenario) -> ProbeEvidence:
    """Expose only the native warm-up view; reject every foreground/post-decision row."""
    required = ("summary.csv", "telemetry.csv", "background.csv", "parameters.json")
    if any(not (raw / name).is_file() for name in required):
        raise ValueError("N07 probe output is incomplete")
    with (raw / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = csv.DictReader(handle)
        if summary.fieldnames != _SUMMARY_FIELDS or list(summary):
            raise ValueError("N07 probe must contain no foreground outcome row")
    parameters = json.loads((raw / "parameters.json").read_text(encoding="utf-8"))
    if not isinstance(parameters, dict) or parameters.get("probe_only") is not True:
        raise ValueError("N07 probe declaration is missing")
    expected = {
        "scenario_id": scenario.scenario_id,
        "topology_family": scenario.topology_family.value,
        "workload": scenario.workload.value,
        "event": scenario.event.value,
        "action": PathAction.KEEP.value,
        "seed": scenario.seed,
        "run": scenario.run,
    }
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError("N07 probe identity does not match its planned scenario")
    with (raw / "telemetry.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"timestamp_ns", "phase", "path_id", "queue_packets"}:
        raise ValueError("N07 probe telemetry schema is invalid")
    values: list[tuple[int, str, int]] = []
    for row in rows:
        try:
            timestamp = int(row["timestamp_ns"])
            queue = int(row["queue_packets"])
        except (KeyError, ValueError) as error:
            raise ValueError("N07 probe telemetry value is invalid") from error
        if row["phase"] != "PRE_DECISION" or timestamp < 0 or queue < 0:
            raise ValueError("N07 probe exposes post-decision or invalid telemetry")
        values.append((timestamp, row["path_id"], queue))
    path_ids = tuple(PATH_SPECS[scenario.topology_family.value])
    latest = max(item[0] for item in values)
    latest_rows = [item for item in values if item[0] == latest]
    if tuple(item[1] for item in latest_rows) != path_ids:
        raise ValueError("N07 probe does not expose one ordered latest row per declared path")
    with (raw / "background.csv").open(newline="", encoding="utf-8") as handle:
        background = list(csv.DictReader(handle))
    if not background or any(row.get("path_id") not in path_ids for row in background):
        raise ValueError("N07 probe background evidence is incomplete")
    return ProbeEvidence(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        path_ids=path_ids,
        latest_timestamp_ns=latest,
        latest_queues={item[1]: item[2] for item in latest_rows},
        pre_decision_digest=_digest(
            [
                {
                    "timestamp_ns": timestamp,
                    "phase": "PRE_DECISION",
                    "path_id": path_id,
                    "queue_packets": queue,
                }
                for timestamp, path_id, queue in values
            ]
        ),
        telemetry_sha256=sha256(raw / "telemetry.csv"),
        background_sha256=sha256(raw / "background.csv"),
    )


def build_method_decisions(
    scenario: StochasticScenario, probe: ProbeEvidence, probe_manifest_sha256: str
) -> dict[str, Any]:
    """Produce six deterministic decisions from probe/declarations only."""
    if probe.scenario_id != scenario.scenario_id or len(probe_manifest_sha256) != 64:
        raise ValueError("N07 probe binding is invalid")
    specs = PATH_SPECS[scenario.topology_family.value]
    if set(probe.latest_queues) != set(specs):
        raise ValueError("N07 telemetry contract is not satisfied")
    estimated_ms = {
        path: 2 * delay_ms + probe.latest_queues[path] * 8.0 / capacity_mbps
        for path, (capacity_mbps, delay_ms) in specs.items()
    }
    best_estimate = min(estimated_ms, key=lambda path: (estimated_ms[path], path))
    c0_path = best_estimate
    if estimated_ms["current"] - estimated_ms[best_estimate] < 0.5:
        c0_path = "current"
    propagation = min(specs, key=lambda path: (specs[path][1], path))
    capacity = max(specs, key=lambda path: (specs[path][0], path))
    queue = min(probe.latest_queues, key=lambda path: (probe.latest_queues[path], path))
    selected = {
        MethodId.C0_CONTRACT_QUEUE: c0_path,
        MethodId.B0_KEEP: "current",
        MethodId.B1_REROUTE: "candidate",
        MethodId.B2_PROPAGATION: propagation,
        MethodId.B3_CAPACITY: capacity,
        MethodId.B4_QUEUE: queue,
    }
    return {
        "schema_version": "n07-decisions-v1",
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "probe_manifest_sha256": probe_manifest_sha256,
        "allowed_input_digest": _digest(
            {
                "topology_family": scenario.topology_family.value,
                "path_specs": specs,
                "latest_timestamp_ns": probe.latest_timestamp_ns,
                "latest_queues": probe.latest_queues,
                "pre_decision_digest": probe.pre_decision_digest,
                "telemetry_sha256": probe.telemetry_sha256,
                "background_sha256": probe.background_sha256,
            }
        ),
        "contract_ready": True,
        "estimated_path_delay_ms": estimated_ms,
        "decisions": [
            {
                "method_id": method.value,
                "selected_path": path,
                "selected_action": ACTION_BY_PATH[path].value,
            }
            for method, path in selected.items()
        ],
        "forbidden_inputs": {
            "foreground_outcome": False,
            "post_decision_telemetry": False,
            "physical_benefit_label": False,
            "oracle_state": False,
        },
    }


class N07DevelopmentRunner:
    """Sequence native probes, sealed decisions, then byte-bound action outcomes."""

    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.action_runner = N06PilotRunner(
            repository=self.repository,
            output_root=self.output_root / "outcomes",
            matrix_path=matrix_path,
            executable=executable,
            runtime_directories=runtime_directories,
        )
        self.scenarios = self.action_runner.scenarios
        self.executable = self.action_runner.executable
        self.runtime_directories = self.action_runner.runtime_directories
        self.config_sha256 = _digest(
            {
                "action_config_sha256": self.action_runner.config_sha256,
                "n07_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n07.py"),
            }
        )

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        prefix = os.pathsep.join(str(path) for path in self.runtime_directories)
        environment["PATH"] = f"{prefix}{os.pathsep}{environment['PATH']}"
        return environment

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "n07-development-journal-v1",
            "config_sha256": self.config_sha256,
            "outcome_config_sha256": self.action_runner.config_sha256,
            "scenario_groups": len(self.scenarios),
            "method_decisions": len(self.scenarios) * len(MethodId),
            "outcome_actions": sum(len(item.actions) for item in self.scenarios),
            "reserved_confirmatory_seeds_executed": False,
        }
        path = self.output_root / "n07_manifest.json"
        if path.exists():
            if json.loads(path.read_text(encoding="utf-8")) != payload:
                raise ValueError("refusing to mix a different N07 freeze into this journal")
        else:
            _write_new(path, payload)
        self.action_runner.initialise()

    def _valid_probe(self, target: Path, scenario: StochasticScenario) -> bool:
        manifest_path = target / "probe_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest["files"]
            valid = (
                manifest["schema_version"] == "n07-probe-manifest-v1"
                and manifest["scenario_id"] == scenario.scenario_id
                and manifest["config_sha256"] == self.config_sha256
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
                read_probe_evidence(target / "raw", scenario)
            return bool(valid)
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _retain_probe_failure(
        self, staged: Path, scenario: StochasticScenario, classification: str, detail: object
    ) -> Path:
        root = self.output_root / "failures" / "probes" / scenario.scenario_id
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (root / f"attempt-{index:03d}").exists():
            index += 1
        target = root / f"attempt-{index:03d}"
        _write_new(
            staged / "failure.json",
            {"classification": classification, "detail": str(detail)},
        )
        shutil.move(str(staged), str(target))
        return target

    def _probe(self, scenario: StochasticScenario) -> Path:
        target = self.output_root / "probes" / scenario.scenario_id
        if target.exists():
            if self._valid_probe(target, scenario):
                return target
            raise FileExistsError("refusing to replace invalid N07 probe")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".probe.", dir=target.parent) as temporary:
            staged = Path(temporary)
            raw = staged / "raw"
            command = [
                str(self.executable),
                *scenario.command_arguments(PathAction.KEEP, raw),
                "--probeOnly=true",
            ]
            before = time.perf_counter_ns()
            try:
                completed = subprocess.run(
                    command,
                    env=self._environment(),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
            except subprocess.TimeoutExpired as error:
                failed = self._retain_probe_failure(staged, scenario, "PROBE_TIMEOUT", error)
                raise RuntimeError(f"N07 probe timeout retained at {failed}") from error
            elapsed = time.perf_counter_ns() - before
            (staged / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (staged / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                failed = self._retain_probe_failure(
                    staged, scenario, "PROBE_NONZERO_EXIT", completed.returncode
                )
                raise RuntimeError(f"N07 probe failure retained at {failed}")
            try:
                read_probe_evidence(raw, scenario)
            except (OSError, ValueError) as error:
                failed = self._retain_probe_failure(
                    staged, scenario, "PROBE_VALIDATION_ERROR", error
                )
                raise RuntimeError(f"N07 invalid probe retained at {failed}") from error
            files = {
                str(path.relative_to(staged)).replace("\\", "/"): sha256(path)
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            }
            _write_new(
                staged / "probe_manifest.json",
                {
                    "schema_version": "n07-probe-manifest-v1",
                    "scenario_id": scenario.scenario_id,
                    "config_sha256": self.config_sha256,
                    "command": command,
                    "elapsed_ns": elapsed,
                    "files": files,
                },
            )
            shutil.move(str(staged), str(target))
        return target

    def _decide(self, scenario: StochasticScenario, probe: Path) -> Path:
        target = self.output_root / "decisions" / scenario.scenario_id / "decisions.json"
        payload = build_method_decisions(
            scenario,
            read_probe_evidence(probe / "raw", scenario),
            sha256(probe / "probe_manifest.json"),
        )
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise ValueError("N07 persisted decision differs from its probe")
        else:
            _write_new(target, payload)
        return target

    def _score(self, scenario: StochasticScenario, decisions: Path) -> Path:
        group = self.action_runner.output_root / "scenarios" / scenario.scenario_id
        label_path = group / "physical_benefit.json"
        labels = json.loads(label_path.read_text(encoding="utf-8"))["labels"]
        decision_payload = json.loads(decisions.read_text(encoding="utf-8"))
        rows = []
        for decision in decision_payload["decisions"]:
            action = decision["selected_action"]
            label = "KEEP"
            if action != PathAction.KEEP.value:
                label = labels[action]
            rows.append({**decision, "physical_label": label})
        payload = {
            "schema_version": "n07-development-score-v1",
            "scenario_id": scenario.scenario_id,
            "decisions_sha256": sha256(decisions),
            "physical_benefit_sha256": sha256(label_path),
            "rows": rows,
            "development_only": True,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if json.loads(target.read_text(encoding="utf-8")) != payload:
                raise ValueError("N07 score does not match its sealed inputs")
        else:
            _write_new(target, payload)
        return target

    def run(self, *, limit: int | None = None) -> dict[str, int]:
        self.initialise()
        selected = self.scenarios if limit is None else self.scenarios[:limit]
        for scenario in selected:
            probe = self._probe(scenario)
            decisions = self._decide(scenario, probe)
            decision_digest = sha256(decisions)
            targets = {
                action: self.action_runner._execute_action(
                    scenario, action, decision_manifest_sha256=decision_digest
                )
                for action in scenario.actions
            }
            probe_evidence = read_probe_evidence(probe / "raw", scenario)
            action_pre_decision = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe_evidence.pre_decision_digest != action_pre_decision:
                raise ValueError("N07 probe and action pre-decision traces differ")
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decisions)
        return {
            "scenario_groups": len(selected),
            "method_decisions": len(selected) * len(MethodId),
            "outcome_actions": sum(len(item.actions) for item in selected),
            "reserved_confirmatory_seeds_executed": 0,
        }

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete = 0
        complete_actions = 0
        for scenario in self.scenarios:
            probe = self.output_root / "probes" / scenario.scenario_id
            decisions = self.output_root / "decisions" / scenario.scenario_id / "decisions.json"
            score = self.output_root / "scores" / scenario.scenario_id / "score.json"
            if not probe.exists() and not decisions.exists() and not score.exists():
                continue
            if (
                not self._valid_probe(probe, scenario)
                or not decisions.is_file()
                or not score.is_file()
            ):
                raise ValueError(f"N07 group is incomplete: {scenario.scenario_id}")
            expected_decisions = build_method_decisions(
                scenario,
                read_probe_evidence(probe / "raw", scenario),
                sha256(probe / "probe_manifest.json"),
            )
            if json.loads(decisions.read_text(encoding="utf-8")) != expected_decisions:
                raise ValueError(f"N07 decision seal is invalid: {scenario.scenario_id}")
            decision_digest = sha256(decisions)
            targets = {
                action: self.action_runner.output_root
                / "scenarios"
                / scenario.scenario_id
                / action.value
                for action in scenario.actions
            }
            if any(
                not self.action_runner._valid_action(target, scenario, action, decision_digest)
                for action, target in targets.items()
            ):
                raise ValueError(f"N07 outcome binding is invalid: {scenario.scenario_id}")
            probe_digest = read_probe_evidence(probe / "raw", scenario).pre_decision_digest
            action_digest = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe_digest != action_digest:
                raise ValueError(f"N07 pre-decision binding is invalid: {scenario.scenario_id}")
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decisions)
            complete += 1
            complete_actions += len(scenario.actions)
        if require_complete and complete != len(self.scenarios):
            raise ValueError("N07 development matrix is incomplete")
        return {
            "complete_scenario_groups": complete,
            "complete_method_decisions": complete * len(MethodId),
            "complete_outcome_actions": complete_actions,
            "planned_scenario_groups": len(self.scenarios),
        }


def aggregate_development_scores(
    root: Path, scenarios: tuple[StochasticScenario, ...]
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {
        method.value: {"KEEP": 0, "BENEFICIAL": 0, "HARMFUL": 0, "NO_MATERIAL_DIFFERENCE": 0}
        for method in MethodId
    }
    for scenario in scenarios:
        path = root / "scores" / scenario.scenario_id / "score.json"
        payload = cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))
        for row in payload["rows"]:
            counts[row["method_id"]][row["physical_label"]] += 1
    return {
        "schema_version": "n07-development-aggregate-v1",
        "scenario_groups": len(scenarios),
        "method_counts": counts,
        "development_only": True,
    }
