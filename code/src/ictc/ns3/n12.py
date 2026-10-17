"""N12 immutable realistic Abilene ns-3 experiment and decision journal."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

TRAFFIC_PROFILES = ("web", "bulk", "realtime")
EVENTS = ("stable", "failure", "capacity", "cross-shift")
MATRICES = (0, 1, 2)
ACTIONS = ("route0", "route1", "route2")
METHODS = (
    "S1_ICTC_CONTRACT",
    "B0_KEEP",
    "B1_SHORTEST_DELAY",
    "B2_MAX_CAPACITY",
    "B3_MIN_QUEUE",
    "B4_GREEDY_TELEMETRY",
    "B5_ECMP_HASH",
)
REGISTRY_FIELDS = (
    "run_id",
    "scenario_id",
    "kind",
    "action",
    "seed",
    "status",
    "started_at_utc",
    "completed_at_utc",
    "config_digest",
    "manifest_path",
    "failure_classification",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode()).hexdigest().upper()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], value)


def write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def utcnow() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True, slots=True)
class N12Scenario:
    scenario_id: str
    phase: str
    traffic: str
    event: str
    matrix: int
    seed: int
    run: int
    duration_seconds: float
    decision_at_seconds: float
    event_at_seconds: float

    def command(self, action: str, output: Path, *, probe: bool) -> list[str]:
        return [
            f"--scenarioId={self.scenario_id}",
            f"--traffic={self.traffic}",
            f"--event={self.event}",
            f"--matrix={self.matrix}",
            f"--action={action}",
            f"--output={output}",
            f"--seed={self.seed}",
            f"--run={self.run}",
            f"--duration={self.duration_seconds}",
            f"--decisionAt={self.decision_at_seconds}",
            f"--eventAt={self.event_at_seconds}",
            f"--probeOnly={1 if probe else 0}",
        ]


def load_matrix(path: Path) -> tuple[dict[str, Any], tuple[N12Scenario, ...]]:
    raw = read_object(path)
    if raw.get("schema_version") != "n12-realistic-abilene-matrix-v1":
        raise ValueError("unexpected N12 matrix schema")
    phase = raw.get("phase")
    if phase not in {"development", "confirmatory"}:
        raise ValueError("N12 phase must be development or confirmatory")
    seeds = raw.get("seeds")
    if not isinstance(seeds, list) or not seeds or any(not isinstance(item, int) for item in seeds):
        raise ValueError("N12 seeds are invalid")
    if phase == "development" and seeds != list(range(84000, 84004)):
        raise ValueError("N12 development seeds differ from the reviewed set")
    if phase == "confirmatory" and seeds != list(range(85000, 85020)):
        raise ValueError("N12 confirmatory seeds differ from the frozen set")
    if raw.get("traffic_profiles") != list(TRAFFIC_PROFILES):
        raise ValueError("N12 traffic profiles are incomplete")
    if raw.get("events") != list(EVENTS) or raw.get("od_matrices") != list(MATRICES):
        raise ValueError("N12 event or OD matrix declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N12 controls are missing")
    duration = float(controls["duration_seconds"])
    decision = float(controls["decision_at_seconds"])
    event_at = float(controls["event_at_seconds"])
    if not 0.5 < event_at + 0.5 < decision < duration - 1.0:
        raise ValueError("N12 timing boundary is invalid")
    scenarios = tuple(
        N12Scenario(
            scenario_id=f"n12-{phase}-{traffic}-{event}-od{matrix}-seed-{seed}",
            phase=phase,
            traffic=traffic,
            event=event,
            matrix=matrix,
            seed=seed,
            run=1,
            duration_seconds=duration,
            decision_at_seconds=decision,
            event_at_seconds=event_at,
        )
        for traffic in TRAFFIC_PROFILES
        for event in EVENTS
        for matrix in MATRICES
        for seed in seeds
    )
    if len(scenarios) != raw.get("expected_scenario_groups"):
        raise ValueError("N12 matrix scenario count differs from its declaration")
    if len(scenarios) * 4 != raw.get("expected_native_runs"):
        raise ValueError("N12 native run count differs from its declaration")
    if len(scenarios) * len(METHODS) != raw.get("expected_controller_decisions"):
        raise ValueError("N12 decision count differs from its declaration")
    return raw, scenarios


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pairing_payload(raw: Path, scenario: N12Scenario) -> dict[str, Any]:
    plan = read_object(raw / "plan.json")
    plan.pop("action", None)
    decision_ns = int(scenario.decision_at_seconds * 1e9)
    telemetry = [
        row
        for row in _csv_rows(raw / "telemetry.csv")
        if int(row["available_timestamp_ns"]) < decision_ns
    ]
    audit = [
        row
        for row in _csv_rows(raw / "telemetry_delivery.csv")
        if int(row["available_timestamp_ns"]) < decision_ns
    ]
    if not telemetry or not audit:
        raise ValueError("N12 evidence has no controller-visible pre-decision telemetry")
    return {"plan": plan, "telemetry": telemetry, "delivery_audit": audit}


def pairing_digest(raw: Path, scenario: N12Scenario) -> str:
    return digest(_pairing_payload(raw, scenario))


def _latest_snapshot(raw: Path, scenario: N12Scenario) -> dict[str, Any]:
    payload = _pairing_payload(raw, scenario)
    telemetry = cast(list[dict[str, str]], payload["telemetry"])
    audit = cast(list[dict[str, str]], payload["delivery_audit"])
    latest: dict[int, dict[str, str]] = {}
    for row in telemetry:
        route = int(row["route_id"])
        if route not in latest or int(row["available_timestamp_ns"]) >= int(
            latest[route]["available_timestamp_ns"]
        ):
            latest[route] = row
    if set(latest) != set(MATRICES):
        raise ValueError("N12 latest telemetry does not cover all candidate routes")
    decision_ns = int(scenario.decision_at_seconds * 1e9)
    delivered_batches = len({row["sample_timestamp_ns"] for row in telemetry})
    audit_batches = len(audit)
    dropped_batches = sum(int(row["dropped"]) for row in audit)
    snapshot = {
        "latest": {
            f"route{route}": {
                "available_timestamp_ns": int(row["available_timestamp_ns"]),
                "bottleneck_capacity_bps": int(row["bottleneck_capacity_bps"]),
                "propagation_delay_ms": float(row["propagation_delay_ms"]),
                "queue_packets": int(row["queue_packets"]),
                "link_up": bool(int(row["link_up"])),
            }
            for route, row in latest.items()
        },
        "delivered_batches": delivered_batches,
        "audited_batches": audit_batches,
        "dropped_batches": dropped_batches,
        "empirical_loss_rate": dropped_batches / audit_batches,
        "latest_age_ms": max(
            (decision_ns - int(row["available_timestamp_ns"])) / 1e6 for row in latest.values()
        ),
        "maximum_delivery_delay_ms": max(int(row["delay_ns"]) / 1e6 for row in audit),
        "pairing_digest": digest(payload),
    }
    return snapshot


def _argbest(values: dict[str, float], *, minimize: bool = False) -> str:
    ordered = sorted(values)
    return (min if minimize else max)(ordered, key=lambda key: (values[key], -int(key[-1])))


def build_decision(
    scenario: N12Scenario, probe_raw: Path, thresholds: dict[str, Any]
) -> dict[str, Any]:
    snapshot = _latest_snapshot(probe_raw, scenario)
    latest = cast(dict[str, dict[str, Any]], snapshot["latest"])
    capacities = {route: float(row["bottleneck_capacity_bps"]) for route, row in latest.items()}
    delays = {route: float(row["propagation_delay_ms"]) for route, row in latest.items()}
    queues = {route: float(row["queue_packets"]) for route, row in latest.items()}
    up = {route: bool(row["link_up"]) for route, row in latest.items()}
    eligible = [route for route in sorted(latest) if up[route]]
    if not eligible:
        eligible = ["route0"]
    visible_score = {
        route: capacities[route] / 1e6
        - float(thresholds["queue_penalty_per_packet"]) * queues[route]
        - float(thresholds["delay_penalty_per_ms"]) * delays[route]
        for route in eligible
    }
    greedy = _argbest(visible_score)
    contract_ready = (
        int(snapshot["delivered_batches"]) >= int(thresholds["minimum_delivered_batches"])
        and float(snapshot["empirical_loss_rate"]) <= float(thresholds["maximum_loss_rate"])
        and float(snapshot["latest_age_ms"]) <= float(thresholds["maximum_latest_age_ms"])
        and float(snapshot["maximum_delivery_delay_ms"])
        <= float(thresholds["maximum_delivery_delay_ms"])
    )
    keep_score = visible_score.get("route0", float("-inf"))
    improvement = visible_score[greedy] - keep_score
    ictc = (
        greedy
        if contract_ready
        and greedy != "route0"
        and (not up["route0"] or improvement >= float(thresholds["minimum_score_improvement"]))
        else "route0"
    )
    decisions = {
        "S1_ICTC_CONTRACT": ictc,
        "B0_KEEP": "route0",
        "B1_SHORTEST_DELAY": _argbest({route: delays[route] for route in eligible}, minimize=True),
        "B2_MAX_CAPACITY": _argbest({route: capacities[route] for route in eligible}),
        "B3_MIN_QUEUE": _argbest({route: queues[route] for route in eligible}, minimize=True),
        "B4_GREEDY_TELEMETRY": greedy,
        "B5_ECMP_HASH": f"route{scenario.seed % 3}",
    }
    return {
        "schema_version": "n12-controller-decisions-v1",
        "scenario": asdict(scenario),
        "created_before_outcomes": True,
        "controller_information_boundary": (
            "Only delivered pre-decision telemetry and frozen route metadata; no post-decision "
            "outcome, hidden event variable, or oracle label."
        ),
        "thresholds": thresholds,
        "snapshot": snapshot,
        "contract_ready": contract_ready,
        "visible_scores": visible_score,
        "decisions": decisions,
    }


def utility(row: dict[str, str]) -> float:
    delivery = float(row["delivery_ratio"])
    throughput = min(1.0, float(row["throughput_mbps"]) / 5.0)
    delay = max(0.0, 1.0 - float(row["p95_delay_ms"]) / 200.0)
    completion = float(row["completion_ratio"])
    drop_penalty = min(1.0, float(row["queue_drops"]) / 100.0)
    return (
        0.50 * delivery + 0.20 * throughput + 0.20 * delay + 0.10 * completion - 0.05 * drop_penalty
    )


def _files(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "run_manifest.json"
    }


class N12Runner:
    """Run a frozen N12 matrix with append-only registry and immutable artifacts."""

    study_id = "N12"
    journal_prefix = "n12"
    journal_schema = "n12-immutable-journal-v1"
    native_run_schema = "n12-native-run-v1"
    resource_schema = "n12-resource-metrics-v1"
    paired_score_schema = "n12-paired-score-v1"
    group_schema = "n12-group-manifest-v1"
    matrix_loader = staticmethod(load_matrix)
    decision_builder = staticmethod(build_decision)
    selected_policy_relative = Path("configs/experiments/n12_selected_policy.json")
    native_source_relative = Path("ns3/scratch/ictc_n12_abilene.cc")
    input_manifest_relative = Path("configs/experiments/n12_inputs/n12_input_manifest.json")
    confirmatory_phases = ("confirmatory",)

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
        self.matrix_path = matrix_path.resolve()
        self.executable = executable.resolve()
        self.runtime_directories = tuple(path.resolve() for path in runtime_directories)
        self.matrix, self.scenarios = self.matrix_loader(self.matrix_path)
        self.source = self.repository / self.native_source_relative
        self.inputs_manifest = self.repository / self.input_manifest_relative
        if (
            not self.executable.is_file()
            or not self.source.is_file()
            or not self.inputs_manifest.is_file()
        ):
            raise FileNotFoundError(
                f"{self.study_id} executable, source, or input manifest is missing"
            )
        if any(not item.is_dir() for item in self.runtime_directories):
            raise FileNotFoundError(f"{self.study_id} runtime directory is missing")
        if sha256(self.source) != self.matrix["native_source_sha256"]:
            raise ValueError(f"{self.study_id} native source differs from its matrix freeze")
        if sha256(self.inputs_manifest) != self.matrix["input_manifest_sha256"]:
            raise ValueError(f"{self.study_id} transformed inputs differ from their matrix freeze")
        self.selected_policy: Path | None = None
        if self.matrix["phase"] in self.confirmatory_phases:
            self.selected_policy = self.repository / self.selected_policy_relative
            policy = read_object(self.selected_policy)
            if sha256(self.selected_policy) != self.matrix["selected_policy_sha256"]:
                raise ValueError(
                    f"{self.study_id} selected policy differs from the confirmation freeze"
                )
            matrix_thresholds = self.matrix.get("controller_thresholds")
            if matrix_thresholds is None and self.matrix.get(
                "controller_thresholds_from_selected_policy"
            ):
                self.matrix["controller_thresholds"] = policy.get("controller_thresholds")
            elif policy.get("controller_thresholds") != matrix_thresholds:
                raise ValueError(
                    f"{self.study_id} confirmation thresholds differ from the selected policy"
                )
        self.executable_sha256 = sha256(self.executable)
        self.repository_commit = self._commit()
        self.config_digest = digest(
            {
                "matrix_sha256": sha256(self.matrix_path),
                "source_sha256": sha256(self.source),
                "input_manifest_sha256": sha256(self.inputs_manifest),
                "executable_sha256": self.executable_sha256,
                "repository_commit": self.repository_commit,
                "selected_policy_sha256": (
                    sha256(self.selected_policy) if self.selected_policy is not None else None
                ),
            }
        )
        self.registry = self.output_root / "EXPERIMENT_REGISTRY.csv"
        self.lock = threading.Lock()

    def _commit(self) -> str:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.repository,
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip().lower()

    def _environment(self) -> dict[str, str]:
        environment = os.environ.copy()
        prefix = os.pathsep.join(str(item) for item in self.runtime_directories)
        environment["PATH"] = f"{prefix}{os.pathsep}{environment['PATH']}"
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["OMP_NUM_THREADS"] = "1"
        return environment

    def _append(self, **values: object) -> None:
        row = {field: str(values.get(field, "")) for field in REGISTRY_FIELDS}
        with self.lock, self.registry.open("a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS).writerow(row)

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "schema_version": self.journal_schema,
            "phase": self.matrix["phase"],
            "config_digest": self.config_digest,
            "matrix_sha256": sha256(self.matrix_path),
            "native_source_sha256": sha256(self.source),
            "input_manifest_sha256": sha256(self.inputs_manifest),
            "executable_sha256": self.executable_sha256,
            "repository_commit": self.repository_commit,
            "selected_policy_sha256": (
                sha256(self.selected_policy) if self.selected_policy is not None else None
            ),
            "planned_scenario_groups": len(self.scenarios),
            "planned_native_runs": len(self.scenarios) * 4,
            "planned_controller_decisions": len(self.scenarios) * len(METHODS),
        }
        manifest = self.output_root / f"{self.journal_prefix}_manifest.json"
        if manifest.exists():
            if read_object(manifest) != manifest_payload:
                raise ValueError(
                    f"refusing to mix a different {self.study_id} freeze into this journal"
                )
        else:
            write_new(manifest, manifest_payload)
        if not self.registry.exists():
            with self.registry.open("x", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=REGISTRY_FIELDS)
                writer.writeheader()
                for scenario in self.scenarios:
                    planned = (("probe", "route0"), *(("action", item) for item in ACTIONS))
                    for kind, action in planned:
                        writer.writerow(
                            {
                                "run_id": f"{scenario.scenario_id}-{kind}-{action}",
                                "scenario_id": scenario.scenario_id,
                                "kind": kind,
                                "action": action,
                                "seed": scenario.seed,
                                "status": "PLANNED",
                                "started_at_utc": "",
                                "completed_at_utc": "",
                                "config_digest": self.config_digest,
                                "manifest_path": "",
                                "failure_classification": "",
                            }
                        )

    def _valid_run(self, target: Path, scenario: N12Scenario, kind: str, action: str) -> bool:
        manifest_path = target / "run_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = read_object(manifest_path)
            files = manifest["files"]
            valid = (
                manifest["schema_version"] == self.native_run_schema
                and manifest["scenario_id"] == scenario.scenario_id
                and manifest["kind"] == kind
                and manifest["action"] == action
                and manifest["config_digest"] == self.config_digest
                and manifest["executable_sha256"] == self.executable_sha256
                and isinstance(files, dict)
                and bool(files)
                and all(
                    isinstance(name, str)
                    and isinstance(value, str)
                    and (target / name).is_file()
                    and sha256(target / name) == value
                    for name, value in files.items()
                )
            )
            if valid:
                _pairing_payload(target / "raw", scenario)
                rows = _csv_rows(target / "raw" / "summary.csv")
                if (kind == "probe" and rows) or (kind == "action" and len(rows) != 1):
                    return False
            return bool(valid)
        except (KeyError, OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _target(self, scenario: N12Scenario, kind: str, action: str) -> Path:
        return self.output_root / "runs" / scenario.scenario_id / f"{kind}-{action}"

    def _failure_target(self, scenario: N12Scenario, kind: str, action: str) -> Path:
        root = self.output_root / "failures" / scenario.scenario_id
        root.mkdir(parents=True, exist_ok=True)
        index = 1
        while (root / f"{kind}-{action}-attempt-{index:03d}").exists():
            index += 1
        return root / f"{kind}-{action}-attempt-{index:03d}"

    def _execute(self, scenario: N12Scenario, kind: str, action: str) -> Path:
        target = self._target(scenario, kind, action)
        if target.exists():
            if self._valid_run(target, scenario, kind, action):
                return target
            raise FileExistsError(f"refusing to replace invalid {self.study_id} output: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        run_id = f"{scenario.scenario_id}-{kind}-{action}"
        started = utcnow()
        self._append(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            kind=kind,
            action=action,
            seed=scenario.seed,
            status="STARTED",
            started_at_utc=started,
            config_digest=self.config_digest,
        )
        staged = Path(tempfile.mkdtemp(prefix=f".{kind}-{action}.", dir=target.parent))
        raw = staged / "raw"
        command = [str(self.executable), *scenario.command(action, raw, probe=kind == "probe")]
        before = time.perf_counter_ns()
        completed = subprocess.run(
            command,
            env=self._environment(),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        elapsed = time.perf_counter_ns() - before
        (staged / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (staged / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        write_new(
            staged / "resource_metrics.json",
            {
                "schema_version": self.resource_schema,
                "wall_time_ns": elapsed,
                "stdout_bytes": len(completed.stdout.encode()),
                "stderr_bytes": len(completed.stderr.encode()),
                "measurement_boundary": "orchestrator wall clock and captured process output",
            },
        )
        try:
            if completed.returncode != 0:
                raise RuntimeError(f"native return code {completed.returncode}")
            _pairing_payload(raw, scenario)
            rows = _csv_rows(raw / "summary.csv")
            if (kind == "probe" and rows) or (kind == "action" and len(rows) != 1):
                raise ValueError(f"unexpected {self.study_id} summary row count")
        except (OSError, ValueError, RuntimeError) as error:
            write_new(
                staged / "failure.json",
                {"classification": "NATIVE_OR_EVIDENCE_FAILURE", "detail": str(error)},
            )
            failed = self._failure_target(scenario, kind, action)
            shutil.move(str(staged), str(failed))
            self._append(
                run_id=run_id,
                scenario_id=scenario.scenario_id,
                kind=kind,
                action=action,
                seed=scenario.seed,
                status="FAILED",
                started_at_utc=started,
                completed_at_utc=utcnow(),
                config_digest=self.config_digest,
                manifest_path=failed,
                failure_classification="NATIVE_OR_EVIDENCE_FAILURE",
            )
            raise RuntimeError(f"{self.study_id} failure retained at {failed}") from error
        manifest = {
            "schema_version": self.native_run_schema,
            "run_id": run_id,
            "scenario_id": scenario.scenario_id,
            "kind": kind,
            "action": action,
            "seed": scenario.seed,
            "run": scenario.run,
            "command": command,
            "returncode": completed.returncode,
            "config_digest": self.config_digest,
            "executable_sha256": self.executable_sha256,
            "repository_commit": self.repository_commit,
            "elapsed_ns": elapsed,
            "files": _files(staged),
        }
        write_new(staged / "run_manifest.json", manifest)
        shutil.move(str(staged), str(target))
        self._append(
            run_id=run_id,
            scenario_id=scenario.scenario_id,
            kind=kind,
            action=action,
            seed=scenario.seed,
            status="COMPLETE",
            started_at_utc=started,
            completed_at_utc=utcnow(),
            config_digest=self.config_digest,
            manifest_path=target / "run_manifest.json",
        )
        return target

    def _decision(self, scenario: N12Scenario, probe: Path) -> Path:
        payload = self.decision_builder(
            scenario,
            probe / "raw",
            cast(dict[str, Any], self.matrix["controller_thresholds"]),
        )
        payload["probe_manifest_sha256"] = sha256(probe / "run_manifest.json")
        target = self.output_root / "decisions" / scenario.scenario_id / "decisions.json"
        if target.exists():
            if read_object(target) != payload:
                raise ValueError(
                    f"{self.study_id} decision differs from sealed probe: {scenario.scenario_id}"
                )
        else:
            write_new(target, payload)
        return target

    def _score(self, scenario: N12Scenario, decision: Path, actions: dict[str, Path]) -> Path:
        outcome_rows = {
            action: _csv_rows(path / "raw" / "summary.csv")[0] for action, path in actions.items()
        }
        utilities = {action: utility(row) for action, row in outcome_rows.items()}
        keep = utilities["route0"]
        best = _argbest(utilities)
        decisions = cast(dict[str, str], read_object(decision)["decisions"])
        rows = []
        for method, action in decisions.items():
            row = outcome_rows[action]
            rows.append(
                {
                    "method": method,
                    "selected_action": action,
                    "utility": utilities[action],
                    "harmful": utilities[action] < keep - 0.02,
                    "beneficial": utilities[action] > keep + 0.02,
                    "opportunity_present": utilities[best] > keep + 0.02,
                    "opportunity_retained": (
                        utilities[best] > keep + 0.02
                        and utilities[action] >= utilities[best] - 0.02
                    ),
                    "metrics": row,
                }
            )
        rows.append(
            {
                "method": "O0_OFFLINE_ORACLE_UPPER_BOUND",
                "selected_action": best,
                "utility": utilities[best],
                "harmful": False,
                "beneficial": utilities[best] > keep + 0.02,
                "opportunity_present": utilities[best] > keep + 0.02,
                "opportunity_retained": utilities[best] > keep + 0.02,
                "metrics": outcome_rows[best],
            }
        )
        payload = {
            "schema_version": self.paired_score_schema,
            "scenario": asdict(scenario),
            "decision_sha256": sha256(decision),
            "action_manifest_sha256": {
                action: sha256(path / "run_manifest.json") for action, path in actions.items()
            },
            "pairing_digest": pairing_digest(actions["route0"] / "raw", scenario),
            "utility_definition": self.matrix["analysis"]["utility_definition"],
            "rows": rows,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if read_object(target) != payload:
                raise ValueError(
                    f"{self.study_id} score differs from sealed evidence: {scenario.scenario_id}"
                )
        else:
            write_new(target, payload)
        return target

    def _group(self, scenario: N12Scenario) -> str:
        probe = self._execute(scenario, "probe", "route0")
        decision = self._decision(scenario, probe)
        actions = {action: self._execute(scenario, "action", action) for action in ACTIONS}
        digests = {pairing_digest(probe / "raw", scenario)} | {
            pairing_digest(path / "raw", scenario) for path in actions.values()
        }
        if len(digests) != 1:
            raise ValueError(
                f"{self.study_id} paired pre-decision evidence differs: {scenario.scenario_id}"
            )
        score = self._score(scenario, decision, actions)
        group_payload = {
            "schema_version": self.group_schema,
            "scenario_id": scenario.scenario_id,
            "probe_manifest_sha256": sha256(probe / "run_manifest.json"),
            "decision_sha256": sha256(decision),
            "action_manifest_sha256": {
                action: sha256(path / "run_manifest.json") for action, path in actions.items()
            },
            "score_sha256": sha256(score),
            "pairing_digest": next(iter(digests)),
        }
        target = self.output_root / "groups" / scenario.scenario_id / "group_manifest.json"
        if target.exists():
            if read_object(target) != group_payload:
                raise ValueError(f"{self.study_id} group seal differs: {scenario.scenario_id}")
        else:
            write_new(target, group_payload)
        return scenario.scenario_id

    def run(self, *, limit: int | None = None, workers: int = 4) -> dict[str, int]:
        self.initialise()
        selected = self.scenarios if limit is None else self.scenarios[:limit]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(self._group, scenario): scenario for scenario in selected}
            for future in as_completed(futures):
                future.result()
        return self.validate(require_complete=limit is None)

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete = 0
        for scenario in self.scenarios:
            group = self.output_root / "groups" / scenario.scenario_id / "group_manifest.json"
            if not group.exists():
                continue
            probe = self._target(scenario, "probe", "route0")
            actions = {action: self._target(scenario, "action", action) for action in ACTIONS}
            decision = self.output_root / "decisions" / scenario.scenario_id / "decisions.json"
            score = self.output_root / "scores" / scenario.scenario_id / "score.json"
            if (
                not self._valid_run(probe, scenario, "probe", "route0")
                or any(
                    not self._valid_run(path, scenario, "action", action)
                    for action, path in actions.items()
                )
                or not decision.is_file()
                or not score.is_file()
            ):
                raise ValueError(
                    f"{self.study_id} group is incomplete or invalid: {scenario.scenario_id}"
                )
            digests = {pairing_digest(probe / "raw", scenario)} | {
                pairing_digest(path / "raw", scenario) for path in actions.values()
            }
            if len(digests) != 1:
                raise ValueError(
                    f"{self.study_id} pairing validation failed: {scenario.scenario_id}"
                )
            expected = {
                "schema_version": self.group_schema,
                "scenario_id": scenario.scenario_id,
                "probe_manifest_sha256": sha256(probe / "run_manifest.json"),
                "decision_sha256": sha256(decision),
                "action_manifest_sha256": {
                    action: sha256(path / "run_manifest.json") for action, path in actions.items()
                },
                "score_sha256": sha256(score),
                "pairing_digest": next(iter(digests)),
            }
            if read_object(group) != expected:
                raise ValueError(
                    f"{self.study_id} group manifest validation failed: {scenario.scenario_id}"
                )
            complete += 1
        if require_complete and complete != len(self.scenarios):
            raise ValueError(f"{self.study_id} matrix incomplete: {complete}/{len(self.scenarios)}")
        return {
            "complete_scenario_groups": complete,
            "complete_native_runs": complete * 4,
            "complete_controller_decisions": complete * len(METHODS),
            "planned_scenario_groups": len(self.scenarios),
            "planned_native_runs": len(self.scenarios) * 4,
        }
