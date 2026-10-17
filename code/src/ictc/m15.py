"""Immutable M15 execution journal for the frozen ns-3 and M12 matrices.

Raw evidence lives outside the repository.  This module never overwrites a completed or
failed raw directory: the CSV registry is an append-only transition journal and a run can be
resumed only when its existing manifest still validates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ictc.ns3.bridge import read_paired_run, write_pair_manifest
from ictc.simulation import M14SimulationScenario, load_m14_simulation_matrix, run_paired

REGISTRY_FIELDS = (
    "run_id",
    "experiment_id",
    "scenario_id",
    "method_id",
    "seed",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "config_sha256",
    "git_commit",
    "stdout_path",
    "stderr_path",
    "output_manifest_path",
    "failure_classification",
    "notes",
)


@dataclass(frozen=True, slots=True)
class Ns3Condition:
    """One frozen topology/traffic condition with all paired actions."""

    topology_id: str
    traffic_id: str
    interval_ms: float
    packet_size: int
    actions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Ns3Run:
    """One seed-specific immutable ns-3 group."""

    run_id: str
    seed: int
    condition: Ns3Condition
    duration: float
    max_packets: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def load_ns3_matrix(path: Path) -> tuple[Ns3Run, ...]:
    """Expand and validate the exact 20-seed final ns-3 matrix."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("unexpected M14 ns-3 matrix schema")
    seeds = raw.get("seeds")
    conditions = raw.get("conditions")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 20
        or len(set(seeds)) != 20
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("M14 ns-3 matrix must declare exactly 20 distinct non-negative seeds")
    if not isinstance(conditions, list) or len(conditions) != 6:
        raise ValueError("M14 ns-3 matrix must declare six topology/traffic conditions")
    try:
        duration = float(raw["duration"])
        max_packets = int(raw["max_packets"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("M14 ns-3 matrix has invalid global traffic controls") from error
    if duration <= 1 or max_packets < 1:
        raise ValueError("M14 ns-3 global traffic controls are invalid")
    parsed: list[Ns3Condition] = []
    identities: set[tuple[str, str]] = set()
    for value in conditions:
        if not isinstance(value, dict):
            raise ValueError("M14 ns-3 condition must be an object")
        try:
            condition = Ns3Condition(
                topology_id=str(value["topology_id"]),
                traffic_id=str(value["traffic_id"]),
                interval_ms=float(value["interval_ms"]),
                packet_size=int(value["packet_size"]),
                actions=tuple(str(action) for action in value["actions"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("M14 ns-3 condition is invalid") from error
        if condition.topology_id not in {"two-path", "three-path"}:
            raise ValueError("M14 ns-3 topology is invalid")
        if condition.traffic_id not in {"low", "medium", "high"}:
            raise ValueError("M14 ns-3 traffic identifier is invalid")
        if condition.interval_ms <= 0 or condition.packet_size < 1:
            raise ValueError("M14 ns-3 condition traffic controls are invalid")
        expected_actions: tuple[str, ...] = ("keep", "reroute")
        if condition.topology_id == "three-path":
            expected_actions = (*expected_actions, "alternate")
        if condition.actions != expected_actions:
            raise ValueError("M14 ns-3 condition actions do not match its topology")
        identity = (condition.topology_id, condition.traffic_id)
        if identity in identities:
            raise ValueError("M14 ns-3 condition identity is duplicated")
        identities.add(identity)
        parsed.append(condition)
    if identities != {
        (topology, traffic)
        for topology in ("two-path", "three-path")
        for traffic in ("low", "medium", "high")
    }:
        raise ValueError("M14 ns-3 matrix does not cover every frozen topology/traffic cell")
    return tuple(
        Ns3Run(
            run_id=f"ns3-{condition.topology_id}-{condition.traffic_id}-seed-{seed}",
            seed=seed,
            condition=condition,
            duration=duration,
            max_packets=max_packets,
        )
        for condition in parsed
        for seed in seeds
    )


def planned_row_count(matrix: Path) -> int:
    """Return action executions plus the twenty M12 repetitions required by M14."""
    return sum(len(item.condition.actions) for item in load_ns3_matrix(matrix)) + 20


class M15Orchestrator:
    """Execute frozen M15 ns-3 and packet-prototype rows without replacing raw evidence."""

    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        simulation_matrix_path: Path,
        protocol_path: Path,
        executable: Path,
        runtime_bin: Path,
        ns3_library_bin: Path,
        python: Path | None = None,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.simulation_matrix_path = simulation_matrix_path.resolve()
        self.protocol_path = protocol_path.resolve()
        self.executable = executable.resolve()
        self.runtime_bin = runtime_bin.resolve()
        self.ns3_library_bin = ns3_library_bin.resolve()
        self.python = (python or Path(sys.executable)).resolve()
        if not self.executable.is_file() or not self.python.is_file():
            raise FileNotFoundError("M15 executable or Python runtime is missing")
        if not self.runtime_bin.is_dir() or not self.ns3_library_bin.is_dir():
            raise FileNotFoundError("M15 ns-3 runtime directories are missing")
        self.matrix = load_ns3_matrix(self.matrix_path)
        self.simulation_matrix = load_m14_simulation_matrix(self.simulation_matrix_path)
        self.config_sha256 = self._configuration_digest()
        self.git_commit = _git_commit(self.repository)
        self.registry_path = self.output_root / "EXPERIMENT_REGISTRY.csv"

    def _configuration_digest(self) -> str:
        digest = hashlib.sha256()
        for path in (self.protocol_path, self.matrix_path, self.simulation_matrix_path):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        return digest.hexdigest().upper()

    def _metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "generator": "ictc.m15.M15Orchestrator",
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
            "matrix_sha256": _sha256(self.matrix_path),
            "protocol_sha256": _sha256(self.protocol_path),
            "planned_ns3_groups": len(self.matrix),
            "planned_ns3_action_executions": sum(
                len(item.condition.actions) for item in self.matrix
            ),
            "planned_m12_repetitions": 20,
            "planned_simulation_pairs": len(self.simulation_matrix),
        }

    def initialise(self) -> None:
        """Create a new raw-evidence journal or verify it belongs to this exact freeze."""
        self.output_root.mkdir(parents=True, exist_ok=True)
        metadata_path = self.output_root / "m15_manifest.json"
        metadata = self._metadata()
        if metadata_path.exists():
            recorded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if recorded != metadata:
                raise ValueError("refusing to mix a different M14 freeze into this M15 journal")
        else:
            metadata_path.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        if not self.registry_path.exists():
            with self.registry_path.open("w", newline="", encoding="utf-8") as handle:
                csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writeheader()

    def _append(self, **row: object) -> None:
        rendered = {field: str(row.get(field, "")) for field in REGISTRY_FIELDS}
        with self.registry_path.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writerow(rendered)

    def _latest_statuses(self) -> dict[str, str]:
        if not self.registry_path.exists():
            return {}
        with self.registry_path.open(newline="", encoding="utf-8") as handle:
            return {row["run_id"]: row["status"] for row in csv.DictReader(handle)}

    def _row(
        self, *, run_id: str, scenario_id: str, method_id: str, seed: int
    ) -> dict[str, object]:
        return {
            "run_id": run_id,
            "experiment_id": "M15",
            "scenario_id": scenario_id,
            "method_id": method_id,
            "seed": seed,
            "config_sha256": self.config_sha256,
            "git_commit": self.git_commit,
        }

    def _valid_manifest(self, target: Path) -> bool:
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_sha256") != self.config_sha256:
                return False
            files = manifest.get("files")
            if not isinstance(files, dict):
                return False
            return all(
                isinstance(relative, str)
                and isinstance(expected, str)
                and (target / relative).is_file()
                and _sha256(target / relative) == expected
                for relative, expected in files.items()
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    @staticmethod
    def _files(directory: Path) -> dict[str, str]:
        return {
            str(path.relative_to(directory)).replace("\\", "/"): _sha256(path)
            for path in sorted(directory.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        environment["PATH"] = (
            f"{self.ns3_library_bin}{os.pathsep}{self.runtime_bin}{os.pathsep}{environment['PATH']}"
        )
        return environment

    def _commit(self, staged: Path, target: Path, manifest: dict[str, object]) -> None:
        (staged / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shutil.move(str(staged), str(target))

    def _run_ns3_group(self, item: Ns3Run) -> None:
        target = self.output_root / "ns3" / item.run_id
        statuses = self._latest_statuses()
        action_ids = [f"{item.run_id}-{action}" for action in item.condition.actions]
        if target.exists():
            if self._valid_manifest(target) and all(
                statuses.get(run_id) == "COMPLETE" for run_id in action_ids
            ):
                return
            raise FileExistsError(f"raw ns-3 output is not resumable: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = _utcnow()
        for action, run_id in zip(item.condition.actions, action_ids, strict=True):
            self._append(
                **self._row(
                    run_id=run_id, scenario_id=item.run_id, method_id=action, seed=item.seed
                ),
                status="PENDING",
                started_at_utc=started,
                notes="immutable raw-output execution planned",
            )
        with tempfile.TemporaryDirectory(prefix=f".{item.run_id}.", dir=target.parent) as temporary:
            staged = Path(temporary)
            failure: tuple[str, str] | None = None
            resource: dict[str, float] = {}
            for action in item.condition.actions:
                action_root = staged / action
                command = [
                    str(self.executable),
                    f"--topology={item.condition.topology_id}",
                    f"--action={action}",
                    f"--seed={item.seed}",
                    "--run=1",
                    f"--duration={item.duration}",
                    f"--intervalMs={item.condition.interval_ms}",
                    f"--maxPackets={item.max_packets}",
                    f"--packetSize={item.condition.packet_size}",
                    f"--output={action_root}",
                ]
                begun = time.monotonic()
                completed = subprocess.run(
                    command, capture_output=True, text=True, env=self._environment()
                )
                resource[action] = round(time.monotonic() - begun, 6)
                (staged / f"{action}.stdout.txt").write_text(completed.stdout, encoding="utf-8")
                (staged / f"{action}.stderr.txt").write_text(completed.stderr, encoding="utf-8")
                if completed.returncode != 0:
                    failure = (action, f"ns3_exit_{completed.returncode}")
                    break
            if failure is None and {"keep", "reroute"}.issubset(item.condition.actions):
                pair = read_paired_run(staged)
                write_pair_manifest(pair, staged / "pair_manifest.json")
            files = self._files(staged)
            manifest: dict[str, object] = {
                "schema_version": "1.0.0",
                "kind": "ns3",
                "config_sha256": self.config_sha256,
                "git_commit": self.git_commit,
                "run_id": item.run_id,
                "seed": item.seed,
                "topology_id": item.condition.topology_id,
                "traffic_id": item.condition.traffic_id,
                "actions": list(item.condition.actions),
                "resource_seconds": resource,
                "files": files,
                "status": "COMPLETE" if failure is None else "FAILED",
                "failure_classification": "" if failure is None else failure[1],
            }
            self._commit(staged, target, manifest)
        completed_at = _utcnow()
        for action, run_id in zip(item.condition.actions, action_ids, strict=True):
            is_failure = failure is not None and action >= failure[0]
            self._append(
                **self._row(
                    run_id=run_id, scenario_id=item.run_id, method_id=action, seed=item.seed
                ),
                status="FAILED" if is_failure else "COMPLETE",
                started_at_utc=started,
                completed_at_utc=completed_at,
                stdout_path=str(target / f"{action}.stdout.txt"),
                stderr_path=str(target / f"{action}.stderr.txt"),
                output_manifest_path=str(target / "manifest.json"),
                failure_classification=failure[1] if is_failure and failure else "",
                notes="raw output retained; latest status is authoritative",
            )

    def _run_m12(self, repetition: int) -> None:
        run_id = f"m12-loopback-repetition-{repetition:02d}"
        target = self.output_root / "m12" / run_id
        statuses = self._latest_statuses()
        if target.exists():
            if self._valid_manifest(target) and statuses.get(run_id) == "COMPLETE":
                return
            raise FileExistsError(f"raw M12 output is not resumable: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = _utcnow()
        row = self._row(
            run_id=run_id, scenario_id="windows-loopback", method_id="M12", seed=repetition
        )
        self._append(**row, status="PENDING", started_at_utc=started, notes="Npcap capture planned")
        with tempfile.TemporaryDirectory(prefix=f".{run_id}.", dir=target.parent) as temporary:
            staged = Path(temporary)
            begun = time.monotonic()
            command = [
                str(self.python),
                "scripts/run_m12_capture.py",
                "--output",
                str(staged / "capture"),
            ]
            completed = subprocess.run(
                command,
                cwd=self.repository,
                capture_output=True,
                text=True,
                env=self._environment(),
            )
            elapsed = round(time.monotonic() - begun, 6)
            (staged / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (staged / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            files = self._files(staged)
            manifest = {
                "schema_version": "1.0.0",
                "kind": "m12",
                "config_sha256": self.config_sha256,
                "git_commit": self.git_commit,
                "run_id": run_id,
                "repetition": repetition,
                "resource_seconds": elapsed,
                "files": files,
                "status": "COMPLETE" if completed.returncode == 0 else "FAILED",
                "failure_classification": ""
                if completed.returncode == 0
                else f"m12_exit_{completed.returncode}",
            }
            self._commit(staged, target, manifest)
        self._append(
            **row,
            status="COMPLETE" if completed.returncode == 0 else "FAILED",
            started_at_utc=started,
            completed_at_utc=_utcnow(),
            stdout_path=str(target / "stdout.txt"),
            stderr_path=str(target / "stderr.txt"),
            output_manifest_path=str(target / "manifest.json"),
            failure_classification=""
            if completed.returncode == 0
            else f"m12_exit_{completed.returncode}",
            notes="raw capture retained; latest status is authoritative",
        )

    def _run_simulation(self, item: M14SimulationScenario) -> None:
        run_id = f"simulation-{item.scenario_id}"
        target = self.output_root / "simulation" / item.scenario_id
        statuses = self._latest_statuses()
        row = self._row(
            run_id=run_id,
            scenario_id=item.scenario_id,
            method_id="PAIRED_SIMULATION",
            seed=item.seed,
        )
        if target.exists():
            if self._valid_manifest(target):
                if statuses.get(run_id) != "COMPLETE":
                    self._append(
                        **row,
                        status="COMPLETE",
                        completed_at_utc=_utcnow(),
                        output_manifest_path=str(target / "manifest.json"),
                        notes="validated output recovered after interrupted registry append",
                    )
                return
            raise FileExistsError(f"raw simulation output is not resumable: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        started = _utcnow()
        self._append(
            **row,
            status="PENDING",
            started_at_utc=started,
            notes=f"condition={item.condition_id};contract_state={item.contract_state}",
        )
        with tempfile.TemporaryDirectory(
            prefix=f".{item.scenario_id}.", dir=target.parent
        ) as temporary:
            staged = Path(temporary)
            begun = time.monotonic()
            failure = ""
            try:
                pair = run_paired(item.config)
                payload = {
                    "scenario_id": item.scenario_id,
                    "condition_id": item.condition_id,
                    "contract_state": item.contract_state,
                    "topology_family_id": item.topology_family_id,
                    "topology_split": item.topology_split,
                    "traffic_id": item.traffic_id,
                    "seed": item.seed,
                    "config_sha256": item.config.config_hash,
                    "schedule_sha256": pair.schedule.digest,
                    "keep": {
                        "latency_ms": pair.baseline.outcome.latency_ms,
                        "loss_ratio": pair.baseline.outcome.packet_loss_ratio,
                        "jitter_ms": pair.baseline.outcome.jitter_ms,
                        "deadline_violation_rate": pair.baseline.outcome.deadline_violation_rate,
                    },
                    "reroute": {
                        "latency_ms": pair.intervention.outcome.latency_ms,
                        "loss_ratio": pair.intervention.outcome.packet_loss_ratio,
                        "jitter_ms": pair.intervention.outcome.jitter_ms,
                        "deadline_violation_rate": (
                            pair.intervention.outcome.deadline_violation_rate
                        ),
                    },
                }
                (staged / "outcomes.json").write_text(
                    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
            except Exception as error:
                failure = type(error).__name__
                (staged / "stderr.txt").write_text(str(error) + "\n", encoding="utf-8")
            elapsed = round(time.monotonic() - begun, 6)
            files = self._files(staged)
            manifest = {
                "schema_version": "1.0.0",
                "kind": "simulation",
                "config_sha256": self.config_sha256,
                "git_commit": self.git_commit,
                "run_id": run_id,
                "condition_id": item.condition_id,
                "resource_seconds": elapsed,
                "files": files,
                "status": "COMPLETE" if not failure else "FAILED",
                "failure_classification": failure,
            }
            self._commit(staged, target, manifest)
        self._append(
            **row,
            status="COMPLETE" if not failure else "FAILED",
            started_at_utc=started,
            completed_at_utc=_utcnow(),
            stderr_path=str(target / "stderr.txt") if failure else "",
            output_manifest_path=str(target / "manifest.json"),
            failure_classification=failure,
            notes="raw paired simulation retained; latest status is authoritative",
        )

    def run(
        self,
        *,
        include_m12: bool = True,
        include_simulation: bool = True,
        include_ns3: bool = True,
    ) -> None:
        """Execute every ns-3 row and, unless disabled, all twenty M12 repetitions."""
        self.initialise()
        if include_simulation:
            for item in self.simulation_matrix:
                self._run_simulation(item)
        if include_ns3:
            for ns3_item in self.matrix:
                self._run_ns3_group(ns3_item)
        if include_m12:
            for repetition in range(1, 21):
                self._run_m12(repetition)

    def validate(
        self,
        *,
        include_m12: bool = True,
        include_simulation: bool = True,
        include_ns3: bool = True,
    ) -> dict[str, object]:
        """Validate completion and integrity for the explicitly executed scope."""
        self.initialise()
        statuses = self._latest_statuses()
        expected: set[str] = set()
        if include_ns3:
            expected |= {
                f"{item.run_id}-{action}"
                for item in self.matrix
                for action in item.condition.actions
            }
        if include_m12:
            expected |= {f"m12-loopback-repetition-{index:02d}" for index in range(1, 21)}
        if include_simulation:
            expected |= {f"simulation-{item.scenario_id}" for item in self.simulation_matrix}
        missing = sorted(expected - set(statuses))
        incomplete = sorted(run_id for run_id in expected if statuses.get(run_id) != "COMPLETE")
        invalid: list[str] = []
        if include_ns3:
            for item in self.matrix:
                target = self.output_root / "ns3" / item.run_id
                if not self._valid_manifest(target):
                    invalid.append(item.run_id)
        if include_m12:
            for index in range(1, 21):
                run_id = f"m12-loopback-repetition-{index:02d}"
                if not self._valid_manifest(self.output_root / "m12" / run_id):
                    invalid.append(run_id)
        if include_simulation:
            for simulation_item in self.simulation_matrix:
                if not self._valid_manifest(
                    self.output_root / "simulation" / simulation_item.scenario_id
                ):
                    invalid.append(f"simulation-{simulation_item.scenario_id}")
        return {
            "scope": {
                "m12": include_m12,
                "simulation": include_simulation,
                "ns3": include_ns3,
            },
            "expected_rows": len(expected),
            "complete": not missing and not incomplete and not invalid,
            "missing_rows": missing,
            "non_complete_rows": incomplete,
            "invalid_outputs": invalid,
        }
