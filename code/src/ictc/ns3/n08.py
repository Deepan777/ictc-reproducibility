"""N08 post-change observation, immutable redecision, and paired outcomes."""

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
from ictc.ns3.n07 import ACTION_BY_PATH, PATH_SPECS
from ictc.ns3.stochastic import (
    EventRegime,
    PathAction,
    StochasticScenario,
    TopologyFamily,
    WorkloadRegime,
    read_action_evidence,
    sha256,
    validate_paired_pre_decision,
)


class N08MethodId(StrEnum):
    D0_CHANGE_RESYNTHESIS = "D0_CHANGE_RESYNTHESIS"
    B0_KEEP = "B0_KEEP"
    B1_REROUTE = "B1_REROUTE"
    B2_PROPAGATION = "B2_PROPAGATION"
    B3_CAPACITY = "B3_CAPACITY"
    B4_QUEUE = "B4_QUEUE"


@dataclass(frozen=True, slots=True)
class StateRow:
    timestamp_ns: int
    phase: str
    path_id: str
    link_up: bool
    rate_bps: int


@dataclass(frozen=True, slots=True)
class ChangeProbeEvidence:
    scenario_id: str
    seed: int
    path_ids: tuple[str, ...]
    latest_timestamp_ns: int
    latest_queues: dict[str, int]
    latest_link_up: dict[str, bool]
    latest_rate_bps: dict[str, int]
    pre_decision_digest: str
    state_digest: str
    observation_digest: str
    event_detected: bool
    telemetry_sha256: str
    state_sha256: str
    background_sha256: str


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
_TELEMETRY_FIELDS = {"timestamp_ns", "phase", "path_id", "queue_packets"}
_STATE_FIELDS = {"timestamp_ns", "phase", "path_id", "link_up", "rate_bps"}


def _digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest().upper()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _write_new(path: Path, payload: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_n08_matrix(path: Path) -> tuple[StochasticScenario, ...]:
    raw = _object(path)
    if raw.get("schema_version") != "n08-post-change-ns3-v1" or raw.get("phase") != "development":
        raise ValueError("unexpected N08 development matrix schema or phase")
    seeds = raw.get("seeds")
    reserved = raw.get("reserved_confirmatory_seeds")
    if seeds != [81000, 81001, 81002, 81003]:
        raise ValueError("N08 development seeds are not the authorized set")
    if reserved != list(range(82000, 82020)) or set(seeds) & set(reserved):
        raise ValueError("N08 confirmatory seed reservation is invalid")
    if raw.get("topology_families") != [item.value for item in TopologyFamily]:
        raise ValueError("N08 topology declaration is incomplete")
    if raw.get("workloads") != [item.value for item in WorkloadRegime]:
        raise ValueError("N08 workload declaration is incomplete")
    if raw.get("events") != [item.value for item in EventRegime]:
        raise ValueError("N08 event declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N08 controls are missing")
    try:
        duration = float(controls["duration_seconds"])
        decision = float(controls["decision_at_seconds"])
        event_at = float(controls["event_at_seconds"])
        interval = float(controls["interval_ms"])
        max_packets = int(controls["max_packets"])
        packet_size = int(controls["packet_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N08 controls are invalid") from error
    if not 0.5 < event_at < decision < duration or min(interval, max_packets, packet_size) <= 0:
        raise ValueError("N08 requires a valid event-before-redecision boundary")
    scenarios = tuple(
        StochasticScenario(
            scenario_id=f"n08-{family.value}-{workload.value}-{event.value}-seed-{seed}",
            topology_family=family,
            workload=workload,
            event=event,
            seed=seed,
            run=1,
            duration_seconds=duration,
            decision_at_seconds=decision,
            event_at_seconds=event_at,
            interval_ms=interval,
            max_packets=max_packets,
            packet_size=packet_size,
        )
        for family in TopologyFamily
        for workload in WorkloadRegime
        for event in EventRegime
        for seed in seeds
    )
    if len(scenarios) != raw.get("expected_scenario_groups"):
        raise ValueError("N08 scenario count differs from its declaration")
    if sum(len(item.actions) for item in scenarios) != raw.get("expected_action_executions"):
        raise ValueError("N08 action count differs from its declaration")
    source_digest = raw.get("source_n07_manifest_sha256")
    if not isinstance(source_digest, str) or len(source_digest) != 64:
        raise ValueError("N08 source N07 manifest binding is invalid")
    return scenarios


def _read_state(path: Path, path_ids: tuple[str, ...]) -> tuple[StateRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows or set(raw_rows[0]) != _STATE_FIELDS:
        raise ValueError("N08 state schema is invalid")
    rows: list[StateRow] = []
    last_time = -1
    for raw in raw_rows:
        try:
            timestamp = int(raw["timestamp_ns"])
            link_value = int(raw["link_up"])
            rate = int(raw["rate_bps"])
        except (KeyError, ValueError) as error:
            raise ValueError("N08 state value is invalid") from error
        row = StateRow(timestamp, raw["phase"], raw["path_id"], link_value == 1, rate)
        if (
            timestamp < last_time
            or row.phase not in {"PRE_DECISION", "POST_DECISION"}
            or row.path_id not in path_ids
            or link_value not in {0, 1}
            or rate <= 0
        ):
            raise ValueError("N08 state ordering or value is invalid")
        last_time = timestamp
        rows.append(row)
    return tuple(rows)


def _state_digest(raw: Path, path_ids: tuple[str, ...]) -> str:
    rows = [row for row in _read_state(raw / "state.csv", path_ids) if row.phase == "PRE_DECISION"]
    if not rows:
        raise ValueError("N08 state has no pre-decision rows")
    return _digest(
        [
            {
                "timestamp_ns": row.timestamp_ns,
                "phase": row.phase,
                "path_id": row.path_id,
                "link_up": row.link_up,
                "rate_bps": row.rate_bps,
            }
            for row in rows
        ]
    )


def read_change_probe(raw: Path, scenario: StochasticScenario) -> ChangeProbeEvidence:
    required = ("summary.csv", "telemetry.csv", "state.csv", "background.csv", "parameters.json")
    if any(not (raw / name).is_file() for name in required):
        raise ValueError("N08 probe output is incomplete")
    with (raw / "summary.csv").open(newline="", encoding="utf-8") as handle:
        summary = csv.DictReader(handle)
        if summary.fieldnames != _SUMMARY_FIELDS or list(summary):
            raise ValueError("N08 probe must contain no foreground outcome row")
    parameters = _object(raw / "parameters.json")
    expected = {
        "scenario_id": scenario.scenario_id,
        "topology_family": scenario.topology_family.value,
        "workload": scenario.workload.value,
        "event": scenario.event.value,
        "action": PathAction.KEEP.value,
        "seed": scenario.seed,
        "run": scenario.run,
        "probe_only": True,
    }
    if any(parameters.get(key) != value for key, value in expected.items()):
        raise ValueError("N08 probe identity or declaration is invalid")
    path_ids = tuple(PATH_SPECS[scenario.topology_family.value])
    with (raw / "telemetry.csv").open(newline="", encoding="utf-8") as handle:
        telemetry = list(csv.DictReader(handle))
    if not telemetry or set(telemetry[0]) != _TELEMETRY_FIELDS:
        raise ValueError("N08 probe telemetry schema is invalid")
    queue_rows: list[tuple[int, str, int]] = []
    for row in telemetry:
        try:
            timestamp = int(row["timestamp_ns"])
            queue = int(row["queue_packets"])
        except (KeyError, ValueError) as error:
            raise ValueError("N08 probe telemetry value is invalid") from error
        if (
            row["phase"] != "PRE_DECISION"
            or timestamp < 0
            or timestamp >= int(scenario.decision_at_seconds * 1e9)
            or row["path_id"] not in path_ids
            or queue < 0
        ):
            raise ValueError("N08 probe exposes post-decision or invalid telemetry")
        queue_rows.append((timestamp, row["path_id"], queue))
    states = _read_state(raw / "state.csv", path_ids)
    if any(row.phase != "PRE_DECISION" for row in states):
        raise ValueError("N08 probe exposes post-decision state")
    if [(item[0], item[1]) for item in queue_rows] != [
        (item.timestamp_ns, item.path_id) for item in states
    ]:
        raise ValueError("N08 queue and link-state sample layouts differ")
    latest = max(item[0] for item in queue_rows)
    latest_queue_rows = [item for item in queue_rows if item[0] == latest]
    latest_states = [item for item in states if item.timestamp_ns == latest]
    if (
        tuple(item[1] for item in latest_queue_rows) != path_ids
        or tuple(item.path_id for item in latest_states) != path_ids
    ):
        raise ValueError("N08 probe lacks one ordered latest sample per path")
    latest_queues = {item[1]: item[2] for item in latest_queue_rows}
    latest_up = {item.path_id: item.link_up for item in latest_states}
    latest_rates = {item.path_id: item.rate_bps for item in latest_states}
    initial_rates = {
        path_id: int(PATH_SPECS[scenario.topology_family.value][path_id][0] * 1_000_000)
        for path_id in path_ids
    }
    event_detected = {
        EventRegime.STABLE: latest_up == dict.fromkeys(path_ids, True)
        and latest_rates == initial_rates,
        EventRegime.CANDIDATE_CAPACITY_DROP: latest_rates["candidate"] < initial_rates["candidate"],
        EventRegime.CURRENT_FAILURE: not latest_up["current"],
        EventRegime.CANDIDATE_CROSS_SHIFT: latest_queues["candidate"] > 0,
    }[scenario.event]
    with (raw / "background.csv").open(newline="", encoding="utf-8") as handle:
        background = list(csv.DictReader(handle))
    if not background or any(row.get("path_id") not in path_ids for row in background):
        raise ValueError("N08 probe background evidence is incomplete")
    pre_payload = [
        {
            "timestamp_ns": timestamp,
            "phase": "PRE_DECISION",
            "path_id": path_id,
            "queue_packets": queue,
        }
        for timestamp, path_id, queue in queue_rows
    ]
    state_digest = _state_digest(raw, path_ids)
    observation = {
        "latest_timestamp_ns": latest,
        "latest_queues": latest_queues,
        "latest_link_up": latest_up,
        "latest_rate_bps": latest_rates,
        "declared_event": scenario.event.value,
        "event_detected": event_detected,
    }
    return ChangeProbeEvidence(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        path_ids=path_ids,
        latest_timestamp_ns=latest,
        latest_queues=latest_queues,
        latest_link_up=latest_up,
        latest_rate_bps=latest_rates,
        pre_decision_digest=_digest(pre_payload),
        state_digest=state_digest,
        observation_digest=_digest(observation),
        event_detected=event_detected,
        telemetry_sha256=sha256(raw / "telemetry.csv"),
        state_sha256=sha256(raw / "state.csv"),
        background_sha256=sha256(raw / "background.csv"),
    )


def _source_n07_id(scenario: StochasticScenario) -> str:
    return scenario.scenario_id.replace("n08-", "n06-", 1)


def build_redecision_generations(
    scenario: StochasticScenario,
    probe: ChangeProbeEvidence,
    probe_manifest_sha256: str,
    source_n07_decision: dict[str, Any],
    source_n07_decision_sha256: str,
) -> dict[str, Any]:
    if (
        probe.scenario_id != scenario.scenario_id
        or len(probe_manifest_sha256) != 64
        or len(source_n07_decision_sha256) != 64
        or source_n07_decision.get("scenario_id") != _source_n07_id(scenario)
    ):
        raise ValueError("N08 probe or N07 decision binding is invalid")
    prior_rows = source_n07_decision.get("decisions")
    if not isinstance(prior_rows, list):
        raise ValueError("N08 source N07 decisions are invalid")
    prior = next((row for row in prior_rows if row.get("method_id") == "C0_CONTRACT_QUEUE"), None)
    if not isinstance(prior, dict):
        raise ValueError("N08 source N07 controller decision is missing")
    generation_zero_payload = {
        "generation": 0,
        "trigger": "INITIAL_SNAPSHOT",
        "previous_generation_digest": None,
        "source_n07_scenario_id": _source_n07_id(scenario),
        "source_n07_decision_sha256": source_n07_decision_sha256,
        "selected_path": prior.get("selected_path"),
        "selected_action": prior.get("selected_action"),
    }
    generation_zero = {
        **generation_zero_payload,
        "recorded_digest": _digest(generation_zero_payload),
    }
    specs = PATH_SPECS[scenario.topology_family.value]
    available = [path for path in probe.path_ids if probe.latest_link_up[path]]
    contract_ready = set(probe.latest_queues) == set(specs) and set(probe.latest_rate_bps) == set(
        specs
    )
    estimated_ms: dict[str, float] = {}
    for path in available:
        rate = probe.latest_rate_bps[path]
        estimated_ms[path] = 2 * specs[path][1] + probe.latest_queues[path] * 8_000_000 / rate
    d0_path = "current"
    if contract_ready and available:
        best = min(available, key=lambda path: (estimated_ms[path], path))
        if "current" not in available or estimated_ms["current"] - estimated_ms[best] >= 0.5:
            d0_path = best
    propagation = min(specs, key=lambda path: (specs[path][1], path))
    capacity = max(available, key=lambda path: (probe.latest_rate_bps[path], path))
    queue = min(available, key=lambda path: (probe.latest_queues[path], path))
    selected = {
        N08MethodId.D0_CHANGE_RESYNTHESIS: d0_path,
        N08MethodId.B0_KEEP: "current",
        N08MethodId.B1_REROUTE: "candidate",
        N08MethodId.B2_PROPAGATION: propagation,
        N08MethodId.B3_CAPACITY: capacity,
        N08MethodId.B4_QUEUE: queue,
    }
    decisions = [
        {
            "method_id": method.value,
            "selected_path": path,
            "selected_action": ACTION_BY_PATH[path].value,
        }
        for method, path in selected.items()
    ]
    trigger = {
        EventRegime.STABLE: "NO_DECLARED_CHANGE",
        EventRegime.CANDIDATE_CAPACITY_DROP: "CAPABILITY_CHANGE",
        EventRegime.CURRENT_FAILURE: "TOPOLOGY_CHANGE",
        EventRegime.CANDIDATE_CROSS_SHIFT: "LOAD_SHIFT",
    }[scenario.event]
    generation_one_payload = {
        "generation": 1,
        "trigger": trigger,
        "previous_generation_digest": generation_zero["recorded_digest"],
        "probe_manifest_sha256": probe_manifest_sha256,
        "change_observation_digest": probe.observation_digest,
        "contract_ready": contract_ready,
        "event_detected": probe.event_detected,
        "estimated_path_delay_ms": estimated_ms,
        "decisions": decisions,
    }
    generation_one = {**generation_one_payload, "recorded_digest": _digest(generation_one_payload)}
    return {
        "schema_version": "n08-redecision-ledger-v1",
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "generations": [generation_zero, generation_one],
        "current_generation": 1,
        "ledger_digest": _digest(
            {
                "scenario_id": scenario.scenario_id,
                "generation_digests": [
                    generation_zero["recorded_digest"],
                    generation_one["recorded_digest"],
                ],
            }
        ),
        "forbidden_inputs": {
            "foreground_outcome": False,
            "post_decision_telemetry": False,
            "physical_benefit_label": False,
            "oracle_state": False,
        },
    }


class N08DevelopmentRunner:
    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
        source_n07_root: Path,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.source_n07_root = source_n07_root.resolve()
        self.matrix = _object(self.matrix_path)
        self.scenarios = load_n08_matrix(self.matrix_path)
        source_manifest = self.source_n07_root / "n07_manifest.json"
        if (
            not source_manifest.is_file()
            or sha256(source_manifest) != self.matrix["source_n07_manifest_sha256"]
        ):
            raise ValueError("N08 source N07 root manifest does not match the freeze")
        self.action_runner = N06PilotRunner(
            repository=self.repository,
            output_root=self.output_root / "outcomes",
            matrix_path=self.matrix_path,
            executable=executable,
            runtime_directories=runtime_directories,
            scenarios=self.scenarios,
        )
        self.executable = self.action_runner.executable
        self.runtime_directories = self.action_runner.runtime_directories
        self.config_sha256 = _digest(
            {
                "action_config_sha256": self.action_runner.config_sha256,
                "n08_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n08.py"),
                "source_n07_manifest_sha256": sha256(source_manifest),
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
            "schema_version": "n08-development-journal-v1",
            "config_sha256": self.config_sha256,
            "scenario_groups": len(self.scenarios),
            "method_decisions": len(self.scenarios) * len(N08MethodId),
            "outcome_actions": sum(len(item.actions) for item in self.scenarios),
            "source_n07_manifest_sha256": self.matrix["source_n07_manifest_sha256"],
            "reserved_confirmatory_seeds_executed": False,
        }
        manifest = self.output_root / "n08_manifest.json"
        if manifest.exists():
            if _object(manifest) != payload:
                raise ValueError("refusing to mix a different N08 freeze into this journal")
        else:
            _write_new(manifest, payload)
        self.action_runner.initialise()

    def _valid_probe(self, target: Path, scenario: StochasticScenario) -> bool:
        manifest_path = target / "probe_manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = _object(manifest_path)
            files = manifest["files"]
            valid = (
                manifest["schema_version"] == "n08-probe-manifest-v1"
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
                read_change_probe(target / "raw", scenario)
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
            staged / "failure.json", {"classification": classification, "detail": str(detail)}
        )
        shutil.move(str(staged), str(target))
        return target

    def _probe(self, scenario: StochasticScenario) -> Path:
        target = self.output_root / "probes" / scenario.scenario_id
        if target.exists():
            if self._valid_probe(target, scenario):
                return target
            raise FileExistsError("refusing to replace invalid N08 probe")
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
                raise RuntimeError(f"N08 probe timeout retained at {failed}") from error
            elapsed = time.perf_counter_ns() - before
            (staged / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (staged / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            if completed.returncode != 0:
                failed = self._retain_probe_failure(
                    staged, scenario, "PROBE_NONZERO_EXIT", completed.returncode
                )
                raise RuntimeError(f"N08 probe failure retained at {failed}")
            try:
                read_change_probe(raw, scenario)
            except (OSError, ValueError) as error:
                failed = self._retain_probe_failure(
                    staged, scenario, "PROBE_VALIDATION_ERROR", error
                )
                raise RuntimeError(f"N08 invalid probe retained at {failed}") from error
            files = {
                str(path.relative_to(staged)).replace("\\", "/"): sha256(path)
                for path in sorted(staged.rglob("*"))
                if path.is_file()
            }
            _write_new(
                staged / "probe_manifest.json",
                {
                    "schema_version": "n08-probe-manifest-v1",
                    "scenario_id": scenario.scenario_id,
                    "config_sha256": self.config_sha256,
                    "command": command,
                    "elapsed_ns": elapsed,
                    "files": files,
                },
            )
            shutil.move(str(staged), str(target))
        return target

    def _source_decision(self, scenario: StochasticScenario) -> Path:
        return self.source_n07_root / "decisions" / _source_n07_id(scenario) / "decisions.json"

    def _decide(self, scenario: StochasticScenario, probe: Path) -> Path:
        source = self._source_decision(scenario)
        if not source.is_file():
            raise ValueError(f"N08 source N07 decision is missing: {_source_n07_id(scenario)}")
        payload = build_redecision_generations(
            scenario,
            read_change_probe(probe / "raw", scenario),
            sha256(probe / "probe_manifest.json"),
            _object(source),
            sha256(source),
        )
        target = self.output_root / "decisions" / scenario.scenario_id / "redecision.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N08 persisted redecision differs from sealed inputs")
        else:
            _write_new(target, payload)
        return target

    def _verify_state_pairing(
        self,
        scenario: StochasticScenario,
        probe: ChangeProbeEvidence,
        targets: dict[PathAction, Path],
    ) -> None:
        digests = {_state_digest(target / "raw", probe.path_ids) for target in targets.values()}
        if digests != {probe.state_digest}:
            raise ValueError(
                f"N08 probe/action state differs before decision: {scenario.scenario_id}"
            )

    def _score(self, scenario: StochasticScenario, decision: Path) -> Path:
        label_path = (
            self.action_runner.output_root
            / "scenarios"
            / scenario.scenario_id
            / "physical_benefit.json"
        )
        labels = _object(label_path)["labels"]
        generations = _object(decision)["generations"]
        rows = []
        for item in generations[1]["decisions"]:
            action = item["selected_action"]
            rows.append({**item, "physical_label": "KEEP" if action == "keep" else labels[action]})
        payload = {
            "schema_version": "n08-development-score-v1",
            "scenario_id": scenario.scenario_id,
            "redecision_sha256": sha256(decision),
            "physical_benefit_sha256": sha256(label_path),
            "event": scenario.event.value,
            "event_detected": generations[1]["event_detected"],
            "rows": rows,
            "development_only": True,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N08 score does not match its sealed inputs")
        else:
            _write_new(target, payload)
        return target

    def run(self, *, limit: int | None = None) -> dict[str, int]:
        self.initialise()
        selected = self.scenarios if limit is None else self.scenarios[:limit]
        for scenario in selected:
            probe_path = self._probe(scenario)
            probe = read_change_probe(probe_path / "raw", scenario)
            decision = self._decide(scenario, probe_path)
            decision_digest = sha256(decision)
            targets = {
                action: self.action_runner._execute_action(
                    scenario, action, decision_manifest_sha256=decision_digest
                )
                for action in scenario.actions
            }
            action_digest = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe.pre_decision_digest != action_digest:
                raise ValueError("N08 probe and action queue traces differ before decision")
            self._verify_state_pairing(scenario, probe, targets)
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decision)
        return {
            "scenario_groups": len(selected),
            "method_decisions": len(selected) * len(N08MethodId),
            "outcome_actions": sum(len(item.actions) for item in selected),
            "reserved_confirmatory_seeds_executed": 0,
        }

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete = 0
        actions = 0
        for scenario in self.scenarios:
            probe_path = self.output_root / "probes" / scenario.scenario_id
            decision = self.output_root / "decisions" / scenario.scenario_id / "redecision.json"
            score = self.output_root / "scores" / scenario.scenario_id / "score.json"
            if not probe_path.exists() and not decision.exists() and not score.exists():
                continue
            if (
                not self._valid_probe(probe_path, scenario)
                or not decision.is_file()
                or not score.is_file()
            ):
                raise ValueError(f"N08 group is incomplete: {scenario.scenario_id}")
            expected = build_redecision_generations(
                scenario,
                read_change_probe(probe_path / "raw", scenario),
                sha256(probe_path / "probe_manifest.json"),
                _object(self._source_decision(scenario)),
                sha256(self._source_decision(scenario)),
            )
            if _object(decision) != expected:
                raise ValueError(f"N08 redecision seal is invalid: {scenario.scenario_id}")
            targets = {
                action: self.action_runner.output_root
                / "scenarios"
                / scenario.scenario_id
                / action.value
                for action in scenario.actions
            }
            decision_digest = sha256(decision)
            if any(
                not self.action_runner._valid_action(target, scenario, action, decision_digest)
                for action, target in targets.items()
            ):
                raise ValueError(f"N08 outcome binding is invalid: {scenario.scenario_id}")
            probe = read_change_probe(probe_path / "raw", scenario)
            action_digest = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe.pre_decision_digest != action_digest:
                raise ValueError(f"N08 queue-trace binding is invalid: {scenario.scenario_id}")
            self._verify_state_pairing(scenario, probe, targets)
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decision)
            complete += 1
            actions += len(targets)
        if require_complete and complete != len(self.scenarios):
            raise ValueError("N08 development matrix is incomplete")
        return {
            "complete_scenario_groups": complete,
            "complete_method_decisions": complete * len(N08MethodId),
            "complete_outcome_actions": actions,
            "planned_scenario_groups": len(self.scenarios),
        }


def aggregate_n08_scores(root: Path, scenarios: tuple[StochasticScenario, ...]) -> dict[str, Any]:
    labels = ("KEEP", "BENEFICIAL", "HARMFUL", "NO_MATERIAL_DIFFERENCE")
    counts = {method.value: {label: 0 for label in labels} for method in N08MethodId}
    event_counts = {event.value: {"detected": 0, "total": 0} for event in EventRegime}
    for scenario in scenarios:
        payload = _object(root / "scores" / scenario.scenario_id / "score.json")
        event_counts[scenario.event.value]["total"] += 1
        event_counts[scenario.event.value]["detected"] += int(payload["event_detected"])
        for row in payload["rows"]:
            counts[row["method_id"]][row["physical_label"]] += 1
    return {
        "schema_version": "n08-development-aggregate-v1",
        "scenario_groups": len(scenarios),
        "method_counts": counts,
        "event_detection": event_counts,
        "development_only": True,
    }
