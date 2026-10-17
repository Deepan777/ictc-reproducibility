"""N11 genuinely stochastic native ns-3 validation over a new immutable seed set."""

from __future__ import annotations

import csv
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n06 import N06PilotRunner
from ictc.ns3.n07 import ACTION_BY_PATH, PATH_SPECS
from ictc.ns3.n08 import (
    ChangeProbeEvidence,
    _digest,
    _object,
    _write_new,
    read_change_probe,
)
from ictc.ns3.n09 import N09DevelopmentRunner
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


class N11MethodId(StrEnum):
    S1_STOCHASTIC_ICTC = "S1_STOCHASTIC_ICTC"
    B0_KEEP = "B0_KEEP"
    B1_REROUTE = "B1_REROUTE"
    B2_PROPAGATION = "B2_PROPAGATION"
    B3_CAPACITY = "B3_CAPACITY"
    B4_QUEUE = "B4_QUEUE"


@dataclass(frozen=True, slots=True)
class RealizedPath:
    path_id: str
    initial_capacity_bps: int
    propagation_delay_ns: int
    queue_limit_packets: int
    cross_traffic_rate_bps: int


@dataclass(frozen=True, slots=True)
class StochasticPlan:
    scenario_id: str
    seed: int
    run: int
    foreground_mean_interval_ms: float
    telemetry_loss_probability: float
    telemetry_delay_mean_ms: float
    telemetry_delay_maximum_ms: float
    failure_start_seconds: float
    failure_duration_seconds: float
    capacity_event_start_seconds: float
    capacity_drop_rate_bps: int
    cross_shift_start_seconds: float
    cross_shift_duration_seconds: float
    cross_shift_rate_bps: int
    paths: tuple[RealizedPath, ...]


@dataclass(frozen=True, slots=True)
class TelemetryDelivery:
    sample_timestamp_ns: int
    available_timestamp_ns: int
    phase: str
    dropped: bool
    delay_ns: int
    path_count: int


@dataclass(frozen=True, slots=True)
class ProbeQuality:
    sampled_batches: int
    delivered_batches: int
    dropped_batches: int
    empirical_loss_rate: float
    mean_delivery_delay_ns: float
    maximum_delivery_delay_ns: int
    latest_age_ns: int


@dataclass(frozen=True, slots=True)
class N11Probe:
    base: ChangeProbeEvidence
    plan: StochasticPlan
    quality: ProbeQuality
    plan_sha256: str
    pre_delivery_digest: str
    event_detected: bool


_PLAN_FIELDS = {
    "schema_version",
    "scenario_id",
    "seed",
    "run",
    "foreground_mean_interval_ms",
    "telemetry_loss_probability",
    "telemetry_delay_mean_ms",
    "telemetry_delay_maximum_ms",
    "failure_start_seconds",
    "failure_duration_seconds",
    "capacity_event_start_seconds",
    "capacity_drop_rate_bps",
    "cross_shift_start_seconds",
    "cross_shift_duration_seconds",
    "cross_shift_rate_bps",
    "paths",
}
_PATH_FIELDS = {
    "path_id",
    "initial_capacity_bps",
    "propagation_delay_ns",
    "queue_limit_packets",
    "cross_traffic_rate_bps",
}
_DELIVERY_FIELDS = {
    "sample_timestamp_ns",
    "available_timestamp_ns",
    "phase",
    "dropped",
    "delay_ns",
    "path_count",
}

N11_STOCHASTIC_FACTORS = {
    "packet_arrivals": (
        "Poisson process; exponential inter-arrival time with per-seed mean multiplier "
        "Uniform(0.65,1.35)"
    ),
    "cross_traffic_intensity": (
        "per-path nominal rate multiplier Uniform(0.65,1.35) with independent "
        "exponential On/Off durations"
    ),
    "failure_time": "nominal event time plus Uniform(-0.20,0.20) seconds",
    "failure_duration": "Uniform(1.0,1.6) seconds with explicit recovery",
    "link_capacity": "per-path nominal capacity multiplier Uniform(0.82,1.18)",
    "capacity_change": "candidate capacity multiplier Uniform(0.12,0.38)",
    "propagation_delay": "per-path nominal delay multiplier Uniform(0.78,1.22)",
    "queue_behavior": "per-path DropTail limit discrete Uniform[48,128] packets",
    "telemetry_loss": "per-scenario batch-loss probability Uniform(0.02,0.18)",
    "telemetry_delay": (
        "per-scenario exponential delay with mean Uniform(5,80) ms, capped at 200 ms"
    ),
}

N11_ANALYSIS_FREEZE = {
    "utility_scores": {
        "BENEFICIAL": 1,
        "HARMFUL": -1,
        "KEEP": 0,
        "NO_MATERIAL_DIFFERENCE": 0,
    },
    "independent_block": "seed",
    "confidence_level": 0.95,
    "bootstrap_resamples": 10000,
    "bootstrap_seed": 20260716,
    "paired_test": "exact one-sided seed-block sign-flip",
    "primary_contrast": "S1_STOCHASTIC_ICTC_minus_B3_CAPACITY",
    "continuous_endpoints": [
        "delivery_ratio",
        "mean_delay_ms",
        "jitter_ms",
        "throughput_mbps",
        "loss_ratio",
    ],
    "require_unique_seed_plans": True,
    "require_unique_seed_outcome_vectors": True,
    "require_condition_level_outcome_variability": True,
}


def _repository_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise ValueError("N11 repository commit identity is invalid")
    return commit


def read_stochastic_plan(path: Path, scenario: StochasticScenario) -> StochasticPlan:
    raw = _object(path)
    if set(raw) != _PLAN_FIELDS or raw.get("schema_version") != "n11-stochastic-plan-v1":
        raise ValueError("N11 stochastic plan schema is invalid")
    raw_paths = raw.get("paths")
    if not isinstance(raw_paths, list) or any(
        not isinstance(item, dict) or set(item) != _PATH_FIELDS for item in raw_paths
    ):
        raise ValueError("N11 realized path schema is invalid")
    try:
        paths = tuple(
            RealizedPath(
                path_id=str(item["path_id"]),
                initial_capacity_bps=int(item["initial_capacity_bps"]),
                propagation_delay_ns=int(item["propagation_delay_ns"]),
                queue_limit_packets=int(item["queue_limit_packets"]),
                cross_traffic_rate_bps=int(item["cross_traffic_rate_bps"]),
            )
            for item in raw_paths
        )
        plan = StochasticPlan(
            scenario_id=str(raw["scenario_id"]),
            seed=int(raw["seed"]),
            run=int(raw["run"]),
            foreground_mean_interval_ms=float(raw["foreground_mean_interval_ms"]),
            telemetry_loss_probability=float(raw["telemetry_loss_probability"]),
            telemetry_delay_mean_ms=float(raw["telemetry_delay_mean_ms"]),
            telemetry_delay_maximum_ms=float(raw["telemetry_delay_maximum_ms"]),
            failure_start_seconds=float(raw["failure_start_seconds"]),
            failure_duration_seconds=float(raw["failure_duration_seconds"]),
            capacity_event_start_seconds=float(raw["capacity_event_start_seconds"]),
            capacity_drop_rate_bps=int(raw["capacity_drop_rate_bps"]),
            cross_shift_start_seconds=float(raw["cross_shift_start_seconds"]),
            cross_shift_duration_seconds=float(raw["cross_shift_duration_seconds"]),
            cross_shift_rate_bps=int(raw["cross_shift_rate_bps"]),
            paths=paths,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N11 stochastic plan values are invalid") from error
    expected_paths = tuple(PATH_SPECS[scenario.topology_family.value])
    if (
        (plan.scenario_id, plan.seed, plan.run)
        != (scenario.scenario_id, scenario.seed, scenario.run)
        or tuple(item.path_id for item in plan.paths) != expected_paths
        or not 0.0 < plan.telemetry_loss_probability < 1.0
        or not 0 < plan.telemetry_delay_mean_ms <= plan.telemetry_delay_maximum_ms
        or min(
            plan.foreground_mean_interval_ms,
            plan.failure_start_seconds,
            plan.failure_duration_seconds,
            plan.capacity_event_start_seconds,
            plan.cross_shift_start_seconds,
            plan.cross_shift_duration_seconds,
            plan.cross_shift_rate_bps,
            plan.capacity_drop_rate_bps,
        )
        <= 0
        or any(
            min(
                item.initial_capacity_bps,
                item.propagation_delay_ns,
                item.queue_limit_packets,
                item.cross_traffic_rate_bps,
            )
            <= 0
            for item in plan.paths
        )
    ):
        raise ValueError("N11 stochastic plan violates its declared bounds or identity")
    return plan


def read_telemetry_delivery(path: Path) -> tuple[TelemetryDelivery, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows or set(raw_rows[0]) != _DELIVERY_FIELDS:
        raise ValueError("N11 telemetry-delivery schema is invalid")
    rows: list[TelemetryDelivery] = []
    last_sample = -1
    for raw in raw_rows:
        try:
            dropped_value = int(raw["dropped"])
            row = TelemetryDelivery(
                sample_timestamp_ns=int(raw["sample_timestamp_ns"]),
                available_timestamp_ns=int(raw["available_timestamp_ns"]),
                phase=raw["phase"],
                dropped=dropped_value == 1,
                delay_ns=int(raw["delay_ns"]),
                path_count=int(raw["path_count"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError("N11 telemetry-delivery value is invalid") from error
        if (
            dropped_value not in {0, 1}
            or row.sample_timestamp_ns <= last_sample
            or row.available_timestamp_ns < row.sample_timestamp_ns
            or row.delay_ns != row.available_timestamp_ns - row.sample_timestamp_ns
            or row.phase not in {"PRE_DECISION", "POST_DECISION"}
            or row.path_count < 2
        ):
            raise ValueError("N11 telemetry-delivery ordering or bounds are invalid")
        last_sample = row.sample_timestamp_ns
        rows.append(row)
    return tuple(rows)


def _pre_delivery_rows(raw: Path, scenario: StochasticScenario) -> tuple[TelemetryDelivery, ...]:
    decision_ns = int(scenario.decision_at_seconds * 1e9)
    return tuple(
        row
        for row in read_telemetry_delivery(raw / "telemetry_delivery.csv")
        if row.sample_timestamp_ns < decision_ns
    )


def stochastic_pairing_digest(raw: Path, scenario: StochasticScenario) -> str:
    plan = read_stochastic_plan(raw / "stochastic_plan.json", scenario)
    rows = _pre_delivery_rows(raw, scenario)
    return _digest(
        {
            "plan": asdict(plan),
            "pre_decision_delivery_schedule": [asdict(row) for row in rows],
        }
    )


def read_n11_probe(raw: Path, scenario: StochasticScenario) -> N11Probe:
    base = read_change_probe(raw, scenario)
    plan_path = raw / "stochastic_plan.json"
    plan = read_stochastic_plan(plan_path, scenario)
    rows = _pre_delivery_rows(raw, scenario)
    decision_ns = int(scenario.decision_at_seconds * 1e9)
    visible = tuple(
        row for row in rows if not row.dropped and row.available_timestamp_ns < decision_ns
    )
    dropped = sum(row.dropped for row in rows)
    delays = [row.delay_ns for row in visible]
    if not rows or not delays:
        raise ValueError("N11 telemetry delivery has no controller-visible pre-decision batch")
    quality = ProbeQuality(
        sampled_batches=len(rows),
        delivered_batches=len(visible),
        dropped_batches=dropped,
        empirical_loss_rate=dropped / len(rows),
        mean_delivery_delay_ns=sum(delays) / len(delays),
        maximum_delivery_delay_ns=max(delays),
        latest_age_ns=decision_ns - base.latest_timestamp_ns,
    )
    initial_rates = {item.path_id: item.initial_capacity_bps for item in plan.paths}
    event_detected = {
        EventRegime.STABLE: base.latest_link_up == dict.fromkeys(base.path_ids, True)
        and base.latest_rate_bps == initial_rates,
        EventRegime.CANDIDATE_CAPACITY_DROP: base.latest_rate_bps["candidate"]
        < initial_rates["candidate"],
        EventRegime.CURRENT_FAILURE: not base.latest_link_up["current"],
        EventRegime.CANDIDATE_CROSS_SHIFT: base.latest_queues["candidate"] > 0,
    }[scenario.event]
    return N11Probe(
        base=base,
        plan=plan,
        quality=quality,
        plan_sha256=sha256(plan_path),
        pre_delivery_digest=_digest([asdict(row) for row in rows]),
        event_detected=event_detected,
    )


def load_n11_matrix(path: Path) -> tuple[StochasticScenario, ...]:
    raw = _object(path)
    if (
        raw.get("schema_version") != "n11-genuinely-stochastic-ns3-v1"
        or raw.get("phase") != "confirmatory-validation"
    ):
        raise ValueError("unexpected N11 matrix schema or phase")
    seeds = raw.get("seeds")
    if seeds != list(range(83000, 83020)):
        raise ValueError("N11 confirmatory seeds are not the frozen set")
    if raw.get("topology_families") != [item.value for item in TopologyFamily]:
        raise ValueError("N11 topology declaration is incomplete")
    if raw.get("workloads") != [item.value for item in WorkloadRegime] or raw.get("events") != [
        item.value for item in EventRegime
    ]:
        raise ValueError("N11 workload/event declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N11 controls are missing")
    try:
        duration = float(controls["duration_seconds"])
        decision = float(controls["decision_at_seconds"])
        event_at = float(controls["event_at_seconds"])
        interval = float(controls["interval_ms"])
        max_packets = int(controls["max_packets"])
        packet_size = int(controls["packet_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N11 controls are invalid") from error
    if not 0.5 < event_at < decision < duration or min(interval, max_packets, packet_size) <= 0:
        raise ValueError("N11 timing or traffic controls are invalid")
    required_hashes = (
        "source_n10_manifest_sha256",
        "source_n10_analysis_sha256",
        "support_design_file_sha256",
        "support_design_digest",
        "native_source_sha256",
    )
    if any(not isinstance(raw.get(name), str) or len(raw[name]) != 64 for name in required_hashes):
        raise ValueError("N11 source-design binding is invalid")
    thresholds = raw.get("controller_thresholds")
    if thresholds != {
        "maximum_latest_age_ms": 300.0,
        "maximum_empirical_loss_rate": 0.30,
        "maximum_delivery_delay_ms": 200.0,
        "minimum_delivered_batches": 10,
        "minimum_candidate_capacity_ratio": 0.85,
    }:
        raise ValueError("N11 controller thresholds differ from the freeze")
    if raw.get("stochastic_factors") != N11_STOCHASTIC_FACTORS:
        raise ValueError("N11 stochastic-factor declaration differs from the freeze")
    if raw.get("analysis") != N11_ANALYSIS_FREEZE:
        raise ValueError("N11 analysis declaration differs from the freeze")
    if raw.get("supported_contexts") != [
        "two-balanced|light|stable",
        "two-balanced|saturated|stable",
    ]:
        raise ValueError("N11 supported contexts differ from the inherited freeze")
    scenarios = tuple(
        StochasticScenario(
            scenario_id=f"n11-{family.value}-{workload.value}-{event.value}-seed-{seed}",
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
    if len(scenarios) != raw.get("expected_scenario_groups") or sum(
        len(item.actions) for item in scenarios
    ) != raw.get("expected_action_executions"):
        raise ValueError("N11 matrix counts differ from the freeze")
    if len(scenarios) * len(N11MethodId) != raw.get("expected_method_decisions"):
        raise ValueError("N11 method-decision count differs from the freeze")
    return scenarios


def build_n11_generations(
    scenario: StochasticScenario,
    probe: N11Probe,
    probe_manifest_sha256: str,
    *,
    source_n10_manifest_sha256: str,
    source_n10_analysis_sha256: str,
    support_design_digest: str,
    support_design_file_sha256: str,
    supported_contexts: tuple[str, ...],
    thresholds: dict[str, float | int],
) -> dict[str, Any]:
    bindings = (
        probe_manifest_sha256,
        source_n10_manifest_sha256,
        source_n10_analysis_sha256,
        support_design_digest,
        support_design_file_sha256,
    )
    if probe.base.scenario_id != scenario.scenario_id or any(len(item) != 64 for item in bindings):
        raise ValueError("N11 probe or source binding is invalid")
    path_ids = probe.base.path_ids
    contract_ready = (
        set(probe.base.latest_queues) == set(path_ids)
        and set(probe.base.latest_rate_bps) == set(path_ids)
        and set(probe.base.latest_link_up) == set(path_ids)
        and probe.quality.latest_age_ns <= float(thresholds["maximum_latest_age_ms"]) * 1e6
        and probe.quality.empirical_loss_rate <= float(thresholds["maximum_empirical_loss_rate"])
        and probe.quality.maximum_delivery_delay_ns
        <= float(thresholds["maximum_delivery_delay_ms"]) * 1e6
        and probe.quality.delivered_batches >= int(thresholds["minimum_delivered_batches"])
    )
    available = [path for path in path_ids if probe.base.latest_link_up[path]]
    b3_path = (
        min(
            available,
            key=lambda path: (
                -probe.base.latest_rate_bps[path],
                0 if path == "current" else 1,
                path,
            ),
        )
        if available
        else "current"
    )
    by_path = {item.path_id: item for item in probe.plan.paths}
    context = "|".join(
        (scenario.topology_family.value, scenario.workload.value, scenario.event.value)
    )
    support_applied = (
        contract_ready
        and context in supported_contexts
        and b3_path == "current"
        and probe.base.latest_link_up.get("current") is True
        and probe.base.latest_link_up.get("candidate") is True
        and by_path["candidate"].propagation_delay_ns < by_path["current"].propagation_delay_ns
        and probe.base.latest_rate_bps["candidate"]
        >= float(thresholds["minimum_candidate_capacity_ratio"])
        * probe.base.latest_rate_bps["current"]
    )
    s1_path = "candidate" if support_applied else b3_path
    propagation = min(path_ids, key=lambda path: (by_path[path].propagation_delay_ns, path))
    queue = min(
        available or ["current"], key=lambda path: (probe.base.latest_queues.get(path, 10**9), path)
    )
    selected = {
        N11MethodId.S1_STOCHASTIC_ICTC: s1_path,
        N11MethodId.B0_KEEP: "current",
        N11MethodId.B1_REROUTE: "candidate",
        N11MethodId.B2_PROPAGATION: propagation,
        N11MethodId.B3_CAPACITY: b3_path,
        N11MethodId.B4_QUEUE: queue,
    }
    decisions = [
        {
            "method_id": method.value,
            "selected_path": path,
            "selected_action": ACTION_BY_PATH[path].value,
        }
        for method, path in selected.items()
    ]
    generation_zero_payload = {
        "generation": 0,
        "trigger": "N10_FROZEN_CONTROLLER_STOCHASTIC_VALIDATION",
        "previous_generation_digest": None,
        "source_n10_manifest_sha256": source_n10_manifest_sha256,
        "source_n10_analysis_sha256": source_n10_analysis_sha256,
        "support_design_digest": support_design_digest,
        "support_design_file_sha256": support_design_file_sha256,
        "supported_contexts": list(supported_contexts),
        "controller_thresholds": thresholds,
    }
    generation_zero = {
        **generation_zero_payload,
        "recorded_digest": _digest(generation_zero_payload),
    }
    generation_one_payload = {
        "generation": 1,
        "trigger": "STOCHASTIC_POST_EVENT_REDECISION",
        "previous_generation_digest": generation_zero["recorded_digest"],
        "probe_manifest_sha256": probe_manifest_sha256,
        "stochastic_plan_sha256": probe.plan_sha256,
        "pre_delivery_digest": probe.pre_delivery_digest,
        "change_observation_digest": probe.base.observation_digest,
        "telemetry_quality": asdict(probe.quality),
        "event_detected": probe.event_detected,
        "contract_ready": contract_ready,
        "context": context,
        "support_applied": support_applied,
        "decisions": decisions,
    }
    generation_one = {
        **generation_one_payload,
        "recorded_digest": _digest(generation_one_payload),
    }
    return {
        "schema_version": "n11-stochastic-redecision-ledger-v1",
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
            "physical_benefit_label": False,
            "post_decision_telemetry": False,
            "oracle_state": False,
        },
    }


class N11ValidationRunner(N09DevelopmentRunner):
    """Execute the frozen N11 stochastic matrix without touching N10 evidence."""

    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
        support_design_path: Path,
        source_n10_root: Path,
        source_n10_analysis: Path,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.support_design_path = support_design_path.resolve()
        self.source_n10_root = source_n10_root.resolve()
        self.source_n10_analysis = source_n10_analysis.resolve()
        self.matrix = _object(self.matrix_path)
        self.design = _object(self.support_design_path)
        self.scenarios = load_n11_matrix(self.matrix_path)
        n10_manifest = self.source_n10_root / "n10_manifest.json"
        if sha256(n10_manifest) != self.matrix["source_n10_manifest_sha256"]:
            raise ValueError("N11 source N10 manifest does not match the freeze")
        if sha256(self.source_n10_analysis) != self.matrix["source_n10_analysis_sha256"]:
            raise ValueError("N11 source N10 analysis does not match the freeze")
        if sha256(self.support_design_path) != self.matrix["support_design_file_sha256"]:
            raise ValueError("N11 support design file does not match the freeze")
        if self.design.get("frozen_design_digest") != self.matrix["support_design_digest"]:
            raise ValueError("N11 support design digest does not match the freeze")
        self.supported_contexts = tuple(self.matrix["supported_contexts"])
        self.repository_commit = _repository_commit(self.repository)
        source_path = self.repository / "ns3" / "scratch" / "ictc_stochastic_validation.cc"
        if sha256(source_path) != self.matrix["native_source_sha256"]:
            raise ValueError("N11 native simulator source does not match the freeze")
        self.action_runner = N06PilotRunner(
            repository=self.repository,
            output_root=self.output_root / "outcomes",
            matrix_path=self.matrix_path,
            executable=executable,
            runtime_directories=runtime_directories,
            scenarios=self.scenarios,
            source_path=source_path,
        )
        self.executable = self.action_runner.executable
        self.runtime_directories = self.action_runner.runtime_directories
        self.config_sha256 = _digest(
            {
                "action_config_sha256": self.action_runner.config_sha256,
                "n11_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n11.py"),
                "source_n10_manifest_sha256": sha256(n10_manifest),
                "source_n10_analysis_sha256": sha256(self.source_n10_analysis),
                "support_design_file_sha256": sha256(self.support_design_path),
            }
        )

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "n11-stochastic-validation-journal-v1",
            "config_sha256": self.config_sha256,
            "scenario_groups": len(self.scenarios),
            "method_decisions": len(self.scenarios) * len(N11MethodId),
            "outcome_actions": sum(len(item.actions) for item in self.scenarios),
            "confirmatory_seeds": list(range(83000, 83020)),
            "repository_commit": self.repository_commit,
            "source_n10_manifest_sha256": self.matrix["source_n10_manifest_sha256"],
            "source_n10_analysis_sha256": self.matrix["source_n10_analysis_sha256"],
            "no_n10_overwrite": True,
        }
        target = self.output_root / "n11_manifest.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("refusing to mix another N11 freeze into this journal")
        else:
            _write_new(target, payload)
        self.action_runner.initialise()

    def _expected_decision(self, scenario: StochasticScenario, probe_path: Path) -> dict[str, Any]:
        return build_n11_generations(
            scenario,
            read_n11_probe(probe_path / "raw", scenario),
            sha256(probe_path / "probe_manifest.json"),
            source_n10_manifest_sha256=self.matrix["source_n10_manifest_sha256"],
            source_n10_analysis_sha256=self.matrix["source_n10_analysis_sha256"],
            support_design_digest=self.matrix["support_design_digest"],
            support_design_file_sha256=self.matrix["support_design_file_sha256"],
            supported_contexts=self.supported_contexts,
            thresholds=cast(dict[str, float | int], self.matrix["controller_thresholds"]),
        )

    def _verify_state_pairing(
        self,
        scenario: StochasticScenario,
        probe: ChangeProbeEvidence,
        targets: dict[PathAction, Path],
    ) -> None:
        super()._verify_state_pairing(scenario, probe, targets)
        probe_root = self.output_root / "probes" / scenario.scenario_id / "raw"
        digests = {stochastic_pairing_digest(probe_root, scenario)} | {
            stochastic_pairing_digest(target / "raw", scenario) for target in targets.values()
        }
        if len(digests) != 1:
            raise ValueError("N11 stochastic plan/delivery schedule differs across paired actions")

    def _score(self, scenario: StochasticScenario, decision: Path) -> Path:
        group = self.action_runner.output_root / "scenarios" / scenario.scenario_id
        label_path = group / "physical_benefit.json"
        labels = _object(label_path)["labels"]
        generation = _object(decision)["generations"][1]
        evidence = {
            action.value: read_action_evidence(group / action.value / "raw", scenario, action)
            for action in scenario.actions
        }
        rows = []
        for item in generation["decisions"]:
            action = item["selected_action"]
            outcome = evidence[action].outcome
            rows.append(
                {
                    **item,
                    "physical_label": "KEEP" if action == "keep" else labels[action],
                    "outcome": {
                        "tx_packets": outcome.tx_packets,
                        "rx_packets": outcome.rx_packets,
                        "delivery_ratio": outcome.delivery_ratio,
                        "mean_delay_ns": outcome.mean_delay_ns,
                        "jitter_ns": outcome.jitter_ns,
                        "throughput_bps": outcome.throughput_bps,
                    },
                }
            )
        payload = {
            "schema_version": "n11-stochastic-validation-score-v1",
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "redecision_sha256": sha256(decision),
            "physical_benefit_sha256": sha256(label_path),
            "stochastic_plan_sha256": sha256(group / "keep" / "raw" / "stochastic_plan.json"),
            "event": scenario.event.value,
            "contract_ready": generation["contract_ready"],
            "support_applied": generation["support_applied"],
            "rows": rows,
            "confirmatory_validation": True,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N11 score does not match sealed decision/outcomes")
        else:
            _write_new(target, payload)
        return target

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete = 0
        actions = 0
        support = 0
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
                raise ValueError(f"N11 group is incomplete: {scenario.scenario_id}")
            expected = self._expected_decision(scenario, probe_path)
            if _object(decision) != expected:
                raise ValueError(f"N11 decision seal is invalid: {scenario.scenario_id}")
            support += int(expected["generations"][1]["support_applied"])
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
                raise ValueError(f"N11 outcome binding is invalid: {scenario.scenario_id}")
            probe = read_change_probe(probe_path / "raw", scenario)
            action_digest = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe.pre_decision_digest != action_digest:
                raise ValueError(f"N11 queue-trace pairing is invalid: {scenario.scenario_id}")
            self._verify_state_pairing(scenario, probe, targets)
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decision)
            complete += 1
            actions += len(targets)
        if require_complete and complete != len(self.scenarios):
            raise ValueError("N11 genuinely stochastic matrix is incomplete")
        return {
            "complete_scenario_groups": complete,
            "complete_method_decisions": complete * len(N11MethodId),
            "complete_outcome_actions": actions,
            "support_applied": support,
            "planned_scenario_groups": len(self.scenarios),
        }
