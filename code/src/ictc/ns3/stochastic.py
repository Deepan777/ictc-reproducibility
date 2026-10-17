"""Strict N06 stochastic ns-3 matrix, raw bridge, and physical-benefit labels."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast


class TopologyFamily(StrEnum):
    TWO_BALANCED = "two-balanced"
    TWO_TRADEOFF = "two-tradeoff"
    THREE_HETEROGENEOUS = "three-heterogeneous"


class WorkloadRegime(StrEnum):
    LIGHT = "light"
    BURSTY = "bursty"
    SATURATED = "saturated"


class EventRegime(StrEnum):
    STABLE = "stable"
    CANDIDATE_CAPACITY_DROP = "candidate-capacity-drop"
    CURRENT_FAILURE = "current-failure"
    CANDIDATE_CROSS_SHIFT = "candidate-cross-shift"


class PathAction(StrEnum):
    KEEP = "keep"
    REROUTE = "reroute"
    ALTERNATE = "alternate"


class PhysicalBenefit(StrEnum):
    BENEFICIAL = "BENEFICIAL"
    HARMFUL = "HARMFUL"
    NO_MATERIAL_DIFFERENCE = "NO_MATERIAL_DIFFERENCE"


@dataclass(frozen=True, slots=True)
class StochasticScenario:
    scenario_id: str
    topology_family: TopologyFamily
    workload: WorkloadRegime
    event: EventRegime
    seed: int
    run: int
    duration_seconds: float
    decision_at_seconds: float
    event_at_seconds: float
    interval_ms: float
    max_packets: int
    packet_size: int

    @property
    def actions(self) -> tuple[PathAction, ...]:
        actions = (PathAction.KEEP, PathAction.REROUTE)
        if self.topology_family is TopologyFamily.THREE_HETEROGENEOUS:
            return (*actions, PathAction.ALTERNATE)
        return actions

    def command_arguments(self, action: PathAction, output: Path) -> tuple[str, ...]:
        if action not in self.actions:
            raise ValueError("action does not exist in this topology family")
        return (
            f"--scenarioId={self.scenario_id}",
            f"--topologyFamily={self.topology_family.value}",
            f"--workload={self.workload.value}",
            f"--event={self.event.value}",
            f"--action={action.value}",
            f"--seed={self.seed}",
            f"--run={self.run}",
            f"--duration={self.duration_seconds}",
            f"--decisionAt={self.decision_at_seconds}",
            f"--eventAt={self.event_at_seconds}",
            f"--intervalMs={self.interval_ms}",
            f"--maxPackets={self.max_packets}",
            f"--packetSize={self.packet_size}",
            f"--output={output}",
        )


@dataclass(frozen=True, slots=True)
class ForegroundOutcome:
    scenario_id: str
    topology_family: str
    workload: str
    event: str
    action: str
    seed: int
    run: int
    tx_packets: int
    rx_packets: int
    lost_packets: int
    mean_delay_ns: int
    jitter_ns: int
    throughput_bps: float
    first_tx_ns: int
    last_rx_ns: int

    @property
    def delivery_ratio(self) -> float:
        return self.rx_packets / self.tx_packets


@dataclass(frozen=True, slots=True)
class TelemetryRow:
    timestamp_ns: int
    phase: str
    path_id: str
    queue_packets: int


@dataclass(frozen=True, slots=True)
class ActionEvidence:
    outcome: ForegroundOutcome
    telemetry: tuple[TelemetryRow, ...]
    background_rows: tuple[tuple[str, int, int, int, int, int], ...]
    raw_sha256: dict[str, str]

    @property
    def pre_decision(self) -> tuple[TelemetryRow, ...]:
        return tuple(row for row in self.telemetry if row.phase == "PRE_DECISION")


_SUMMARY_FIELDS = {
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
}
_TELEMETRY_FIELDS = {"timestamp_ns", "phase", "path_id", "queue_packets"}
_BACKGROUND_FIELDS = {
    "path_id",
    "flow_id",
    "tx_packets",
    "rx_packets",
    "lost_packets",
    "mean_delay_ns",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def load_stochastic_matrix(path: Path) -> tuple[StochasticScenario, ...]:
    """Expand the development matrix while rejecting reserved confirmatory seeds."""
    raw = _object(path)
    if raw.get("schema_version") != "n06-stochastic-ns3-v1":
        raise ValueError("unexpected N06 matrix schema")
    if raw.get("phase") != "development":
        raise ValueError("N06 implementation accepts only a development matrix")
    seeds = raw.get("seeds")
    reserved = raw.get("reserved_confirmatory_seeds")
    if seeds != [81000, 81001, 81002, 81003]:
        raise ValueError("N06 development seeds are not the authorized set")
    if reserved != list(range(82000, 82020)) or set(seeds) & set(reserved):
        raise ValueError("N06 confirmatory seed reservation is invalid")
    expected_families = [item.value for item in TopologyFamily]
    expected_workloads = [item.value for item in WorkloadRegime]
    expected_events = [item.value for item in EventRegime]
    if raw.get("topology_families") != expected_families:
        raise ValueError("N06 topology family declaration is incomplete")
    if raw.get("workloads") != expected_workloads or raw.get("events") != expected_events:
        raise ValueError("N06 workload/event declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N06 controls are missing")
    try:
        duration = float(controls["duration_seconds"])
        decision = float(controls["decision_at_seconds"])
        event_at = float(controls["event_at_seconds"])
        interval = float(controls["interval_ms"])
        max_packets = int(controls["max_packets"])
        packet_size = int(controls["packet_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N06 controls are invalid") from error
    if not 0.5 < decision < event_at < duration or min(interval, max_packets, packet_size) <= 0:
        raise ValueError("N06 timing or traffic controls are invalid")
    scenarios = tuple(
        StochasticScenario(
            scenario_id=(f"n06-{family.value}-{workload.value}-{event.value}-seed-{seed}"),
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
    if len(scenarios) != raw.get("expected_scenario_groups", -1):
        raise ValueError("N06 scenario count differs from its declaration")
    if sum(len(item.actions) for item in scenarios) != raw.get("expected_action_executions", -1):
        raise ValueError("N06 action count differs from its declaration")
    return scenarios


def _read_summary(path: Path) -> ForegroundOutcome:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1 or set(rows[0]) != _SUMMARY_FIELDS:
        raise ValueError("N06 summary must contain one exact-schema foreground row")
    row = rows[0]
    try:
        outcome = ForegroundOutcome(
            scenario_id=row["scenario_id"],
            topology_family=row["topology_family"],
            workload=row["workload"],
            event=row["event"],
            action=row["action"],
            seed=int(row["seed"]),
            run=int(row["run"]),
            tx_packets=int(row["tx_packets"]),
            rx_packets=int(row["rx_packets"]),
            lost_packets=int(row["lost_packets"]),
            mean_delay_ns=int(row["mean_delay_ns"]),
            jitter_ns=int(row["jitter_ns"]),
            throughput_bps=float(row["throughput_bps"]),
            first_tx_ns=int(row["first_tx_ns"]),
            last_rx_ns=int(row["last_rx_ns"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError("N06 foreground row contains invalid values") from error
    if (
        outcome.tx_packets <= 0
        or not 0 <= outcome.rx_packets <= outcome.tx_packets
        or not 0 <= outcome.lost_packets <= outcome.tx_packets - outcome.rx_packets
        or min(outcome.mean_delay_ns, outcome.jitter_ns, outcome.throughput_bps) < 0
        or outcome.first_tx_ns < 0
        or (outcome.rx_packets > 0 and outcome.last_rx_ns < outcome.first_tx_ns)
    ):
        raise ValueError("N06 foreground packet accounting is invalid")
    return outcome


def _read_telemetry(path: Path) -> tuple[TelemetryRow, ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != _TELEMETRY_FIELDS:
        raise ValueError("N06 telemetry schema is invalid")
    result: list[TelemetryRow] = []
    last_time = -1
    for raw in rows:
        try:
            row = TelemetryRow(
                timestamp_ns=int(raw["timestamp_ns"]),
                phase=raw["phase"],
                path_id=raw["path_id"],
                queue_packets=int(raw["queue_packets"]),
            )
        except (KeyError, ValueError) as error:
            raise ValueError("N06 telemetry contains invalid values") from error
        if (
            row.timestamp_ns < last_time
            or row.phase not in {"PRE_DECISION", "POST_DECISION"}
            or not row.path_id
            or row.queue_packets < 0
        ):
            raise ValueError("N06 telemetry ordering or value is invalid")
        last_time = row.timestamp_ns
        result.append(row)
    if not any(row.phase == "PRE_DECISION" for row in result):
        raise ValueError("N06 telemetry has no pre-decision rows")
    return tuple(result)


def _read_background(path: Path) -> tuple[tuple[str, int, int, int, int, int], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != _BACKGROUND_FIELDS:
        raise ValueError("N06 background schema is invalid")
    result = tuple(
        (
            row["path_id"],
            int(row["flow_id"]),
            int(row["tx_packets"]),
            int(row["rx_packets"]),
            int(row["lost_packets"]),
            int(row["mean_delay_ns"]),
        )
        for row in rows
    )
    if any(
        item[2] <= 0 or not 0 <= item[3] <= item[2] or not 0 <= item[4] <= item[2] - item[3]
        for item in result
    ):
        raise ValueError("N06 background packet accounting is invalid")
    return result


def read_action_evidence(
    raw: Path, expected: StochasticScenario, action: PathAction
) -> ActionEvidence:
    """Read raw simulator output and bind it to the expected action-independent scenario."""
    required = ("summary.csv", "telemetry.csv", "background.csv", "parameters.json")
    if any(not (raw / name).is_file() for name in required):
        raise ValueError("N06 raw action evidence is incomplete")
    outcome = _read_summary(raw / "summary.csv")
    parameters = _object(raw / "parameters.json")
    identity = (
        outcome.scenario_id,
        outcome.topology_family,
        outcome.workload,
        outcome.event,
        outcome.action,
        outcome.seed,
        outcome.run,
    )
    expected_identity = (
        expected.scenario_id,
        expected.topology_family.value,
        expected.workload.value,
        expected.event.value,
        action.value,
        expected.seed,
        expected.run,
    )
    if identity != expected_identity:
        raise ValueError("N06 raw summary does not match its planned scenario/action")
    for key, value in {
        "scenario_id": expected.scenario_id,
        "topology_family": expected.topology_family.value,
        "workload": expected.workload.value,
        "event": expected.event.value,
        "action": action.value,
        "seed": expected.seed,
        "run": expected.run,
    }.items():
        if parameters.get(key) != value:
            raise ValueError("N06 parameters do not match the planned scenario/action")
    return ActionEvidence(
        outcome=outcome,
        telemetry=_read_telemetry(raw / "telemetry.csv"),
        background_rows=_read_background(raw / "background.csv"),
        raw_sha256={name: sha256(raw / name) for name in required},
    )


def validate_paired_pre_decision(evidence: dict[PathAction, ActionEvidence]) -> str:
    """Require action-independent pre-decision bytes across every potential outcome."""
    if PathAction.KEEP not in evidence or PathAction.REROUTE not in evidence:
        raise ValueError("N06 pair requires KEEP and REROUTE")
    layouts = {item.pre_decision for item in evidence.values()}
    if len(layouts) != 1:
        raise ValueError("N06 paired actions differ before the decision boundary")
    payload = [asdict(row) for row in next(iter(layouts))]
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest().upper()


def classify_physical_benefit(
    keep: ForegroundOutcome,
    candidate: ForegroundOutcome,
    *,
    delay_margin_ns: int = 500_000,
    delivery_tolerance: float = 0.01,
) -> PhysicalBenefit:
    """Classify an empirical paired physical outcome, never causal identifiability."""
    if delay_margin_ns <= 0 or not 0 <= delivery_tolerance <= 1:
        raise ValueError("physical-benefit thresholds are invalid")
    if keep.action != PathAction.KEEP.value or candidate.action == PathAction.KEEP.value:
        raise ValueError("physical-benefit comparison requires KEEP and a candidate action")
    if keep.scenario_id != candidate.scenario_id or (keep.seed, keep.run) != (
        candidate.seed,
        candidate.run,
    ):
        raise ValueError("physical-benefit outcomes are not paired")
    if candidate.rx_packets == 0 < keep.rx_packets:
        return PhysicalBenefit.HARMFUL
    if keep.rx_packets == 0 < candidate.rx_packets:
        return PhysicalBenefit.BENEFICIAL
    if candidate.delivery_ratio < keep.delivery_ratio - delivery_tolerance:
        return PhysicalBenefit.HARMFUL
    delay_change = candidate.mean_delay_ns - keep.mean_delay_ns
    if delay_change <= -delay_margin_ns:
        return PhysicalBenefit.BENEFICIAL
    if delay_change >= delay_margin_ns:
        return PhysicalBenefit.HARMFUL
    return PhysicalBenefit.NO_MATERIAL_DIFFERENCE
