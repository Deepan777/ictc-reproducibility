"""Leakage-separated observational benchmark derived from paired M08 simulations."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ictc.baselines import (
    AccessPhase,
    BaselineId,
    CorpusAccessGuard,
    CorpusTable,
    load_baseline_input,
    load_outcomes,
)
from ictc.baselines.data import OutcomeMetrics
from ictc.causal import BackdoorIdentifier, CausalDAG, IdentifiabilityResult
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.pilot.authorization import TestAccessAuthorization

M10_GRAPH_ID = "m10_observational_selection_v1"
M10_GRAPH_NODES = ("EndToEndDelay", "QueuePressureBin", "RouteAction")
M10_GRAPH_EDGES = (
    ("QueuePressureBin", "EndToEndDelay"),
    ("QueuePressureBin", "RouteAction"),
    ("RouteAction", "EndToEndDelay"),
)
ASSIGNMENT_POLICY_ID = "m10_queue_pressure_assignment_v1"
ACQUISITION_DEADLINE_TICKS = 5


def _digest(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


M10_GRAPH_DIGEST = _digest(
    {"graph_id": M10_GRAPH_ID, "nodes": M10_GRAPH_NODES, "edges": M10_GRAPH_EDGES}
)


def _graph() -> CausalDAG:
    return CausalDAG(M10_GRAPH_NODES, M10_GRAPH_EDGES, M10_GRAPH_NODES)


def _identifier() -> BackdoorIdentifier:
    return BackdoorIdentifier(_graph(), graph_id=M10_GRAPH_ID)


@dataclass(frozen=True, slots=True)
class PreActionEvidence:
    """Controller-safe tick-zero evidence delivered before the acquisition deadline."""

    queue_pressure_bin: int | None
    utilization_difference: float | None
    delivered_at_tick: int | None
    freshness_ticks: int | None
    clock_skew_span_ticks: int | None
    source_ids: tuple[str, ...]
    provenance_digests: tuple[str, ...]
    telemetry_items: int
    telemetry_bytes: int

    def __post_init__(self) -> None:
        paired_values = (
            self.queue_pressure_bin,
            self.utilization_difference,
            self.delivered_at_tick,
            self.freshness_ticks,
            self.clock_skew_span_ticks,
        )
        if any(value is None for value in paired_values) != all(
            value is None for value in paired_values
        ):
            raise ValueError("derived pre-action evidence must be wholly present or absent")
        if self.queue_pressure_bin not in {None, -1, 0, 1}:
            raise ValueError("pre-action queue-pressure bin is invalid")
        if self.utilization_difference is not None and not math.isfinite(
            self.utilization_difference
        ):
            raise ValueError("pre-action utilization difference must be finite")
        counters = (
            self.delivered_at_tick,
            self.freshness_ticks,
            self.clock_skew_span_ticks,
            self.telemetry_items,
            self.telemetry_bytes,
        )
        if any(value is not None and value < 0 for value in counters):
            raise ValueError("pre-action evidence counters cannot be negative")
        if len(self.source_ids) != len(set(self.source_ids)):
            raise ValueError("pre-action source IDs must be unique")
        if len(self.provenance_digests) != len(set(self.provenance_digests)):
            raise ValueError("pre-action provenance digests must be unique")
        if len(self.source_ids) != self.telemetry_items:
            raise ValueError("one source ID is required per telemetry item")
        if len(self.provenance_digests) != self.telemetry_items:
            raise ValueError("one provenance digest is required per telemetry item")


@dataclass(frozen=True, slots=True)
class ObservationalDecision:
    """Controller-safe observational row; paired potential outcomes are excluded."""

    pair_id: str
    scenario_id: str
    split: DatasetSplit
    seed: int
    topology_family_id: str
    traffic_level_id: str
    observation_regime_id: str
    assigned_action: ActionKind
    observed_latency_ms: float | None
    visible_queue_pressure_bin: int | None
    pre_action_evidence: PreActionEvidence
    reference_identification: IdentifiabilityResult
    assignment_probability: float
    assignment_uniform: float
    assignment_policy_id: str = ASSIGNMENT_POLICY_ID

    def __post_init__(self) -> None:
        if self.assigned_action not in {
            ActionKind.KEEP_CURRENT_PATH,
            ActionKind.REROUTE_TO_PATH,
        }:
            raise ValueError("observational assignment must be KEEP or REROUTE")
        if self.observed_latency_ms is not None and (
            not math.isfinite(self.observed_latency_ms) or self.observed_latency_ms < 0
        ):
            raise ValueError("observed latency must be finite and nonnegative")
        if self.visible_queue_pressure_bin not in {None, -1, 0, 1}:
            raise ValueError("visible queue-pressure bin is invalid")
        if not self.topology_family_id:
            raise ValueError("topology family ID is required")
        if not 0 <= self.assignment_probability <= 1 or not 0 <= self.assignment_uniform < 1:
            raise ValueError("assignment probability and noise must be valid probabilities")


@dataclass(frozen=True, slots=True)
class PilotOutcomeTruth:
    """Privileged network outcome for one executed action."""

    mean_latency_ms: float
    packet_delivery_ratio: float
    packet_loss_ratio: float
    jitter_ms: float
    throughput_packets_per_tick: float
    deadline_violation_rate: float
    route_changes: int

    def __post_init__(self) -> None:
        values = (
            self.mean_latency_ms,
            self.packet_delivery_ratio,
            self.packet_loss_ratio,
            self.jitter_ms,
            self.throughput_packets_per_tick,
            self.deadline_violation_rate,
        )
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("pilot network outcomes must be finite and nonnegative")
        ratios = (
            self.packet_delivery_ratio,
            self.packet_loss_ratio,
            self.deadline_violation_rate,
        )
        if any(not 0 <= value <= 1 for value in ratios):
            raise ValueError("pilot network outcome ratios must lie between zero and one")
        if self.route_changes not in {0, 1}:
            raise ValueError("pilot route-change count must be zero or one")


@dataclass(frozen=True, slots=True)
class PotentialOutcomeTruth:
    """Privileged paired truth used only after decisions are persisted."""

    pair_id: str
    true_queue_pressure_bin: int
    keep: PilotOutcomeTruth
    reroute: PilotOutcomeTruth

    def __post_init__(self) -> None:
        if self.true_queue_pressure_bin not in {-1, 0, 1}:
            raise ValueError("true queue-pressure bin is invalid")

    @property
    def keep_latency_ms(self) -> float:
        return self.keep.mean_latency_ms

    @property
    def reroute_latency_ms(self) -> float:
        return self.reroute.mean_latency_ms

    @property
    def latency_effect_ms(self) -> float:
        return self.reroute_latency_ms - self.keep_latency_ms

    @property
    def reroute_beneficial(self) -> bool:
        return self.latency_effect_ms < 0


def _pilot_outcome(
    outcome: OutcomeMetrics,
    *,
    run_id: str,
    accounting_rows: list[dict[str, object]],
    duration_ticks: int,
    route_changes: int,
) -> PilotOutcomeTruth:
    selected = [row for row in accounting_rows if row.get("run_id") == run_id]
    if not selected:
        raise ValueError("pilot outcome accounting is unavailable for one action")
    generated_values = [row.get("foreground_generated") for row in selected]
    delivered_values = [row.get("foreground_delivered") for row in selected]
    if not all(isinstance(value, int) for value in (*generated_values, *delivered_values)):
        raise ValueError("pilot outcome accounting contains invalid packet counts")
    generated = sum(cast(int, value) for value in generated_values)
    delivered = sum(cast(int, value) for value in delivered_values)
    if generated <= 0 or delivered > generated or duration_ticks <= 0:
        raise ValueError("pilot outcome accounting totals are invalid")
    return PilotOutcomeTruth(
        outcome.mean_latency_ms,
        delivered / generated,
        outcome.packet_loss_ratio,
        outcome.jitter_ms,
        delivered / duration_ticks,
        outcome.deadline_violation_rate,
        route_changes,
    )


@dataclass(frozen=True, slots=True)
class PilotBenchmarkRecord:
    decision: ObservationalDecision
    truth: PotentialOutcomeTruth

    def __post_init__(self) -> None:
        if self.decision.pair_id != self.truth.pair_id:
            raise ValueError("decision and truth pair IDs differ")


def _pressure_bin(current_queue: int, candidate_queue: int) -> int:
    difference = current_queue - candidate_queue
    return -1 if difference < 0 else 1 if difference > 0 else 0


def _assignment(pair_id: str, pressure_bin: int) -> tuple[ActionKind, float, float]:
    probability = {-1: 0.25, 0: 0.5, 1: 0.75}[pressure_bin]
    encoded = f"{ASSIGNMENT_POLICY_ID}\0{pair_id}".encode()
    integer = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    uniform = integer / 2**64
    action = ActionKind.REROUTE_TO_PATH if uniform < probability else ActionKind.KEEP_CURRENT_PATH
    return action, probability, uniform


def reference_identification(pressure_available: bool) -> IdentifiabilityResult:
    available = ["RouteAction", "EndToEndDelay"]
    if pressure_available:
        available.append("QueuePressureBin")
    return _identifier().identify("RouteAction", "EndToEndDelay", available)


def _pre_action_evidence(
    rows: list[dict[str, object]],
    *,
    baseline_run_id: str,
    current_path_id: str,
    candidate_path_id: str,
) -> PreActionEvidence:
    """Derive only from action-independent tick-zero samples delivered by the deadline."""
    selected = [
        row
        for row in rows
        if row.get("run_id") == baseline_run_id
        and row.get("sample_tick") == 0
        and row.get("missing") is False
        and isinstance(row.get("available_tick"), int)
        and cast(int, row["available_tick"]) <= ACQUISITION_DEADLINE_TICKS
    ]
    by_path = {cast(str, row.get("path_id")): row for row in selected}
    expected = {current_path_id, candidate_path_id}
    if not expected <= set(by_path):
        sources = tuple(
            sorted(
                cast(str, row["source_id"])
                for row in selected
                if isinstance(row.get("source_id"), str)
            )
        )
        digests = tuple(
            sorted(
                cast(str, row["provenance_digest"])
                for row in selected
                if isinstance(row.get("provenance_digest"), str)
            )
        )
        return PreActionEvidence(
            None,
            None,
            None,
            None,
            None,
            sources,
            digests,
            len(selected),
            len(selected) * 64,
        )
    if set(by_path) != expected:
        raise ValueError("tick-zero pre-action evidence contains an unexpected path")
    current = by_path[current_path_id]
    candidate = by_path[candidate_path_id]
    numeric_fields = ("queue_occupancy", "utilization", "available_tick", "observed_timestamp_tick")
    if any(
        isinstance(row.get(field), bool) or not isinstance(row.get(field), int | float)
        for row in (current, candidate)
        for field in numeric_fields
    ):
        raise ValueError("tick-zero pre-action evidence contains invalid numeric fields")
    sources = tuple(sorted(cast(str, row["source_id"]) for row in (current, candidate)))
    digests = tuple(sorted(cast(str, row["provenance_digest"]) for row in (current, candidate)))
    if any(not value for value in (*sources, *digests)):
        raise ValueError("tick-zero pre-action evidence lacks source provenance")
    delivered = max(cast(int, current["available_tick"]), cast(int, candidate["available_tick"]))
    observed_ticks = (
        cast(int, current["observed_timestamp_tick"]),
        cast(int, candidate["observed_timestamp_tick"]),
    )
    return PreActionEvidence(
        _pressure_bin(
            cast(int, current["queue_occupancy"]),
            cast(int, candidate["queue_occupancy"]),
        ),
        cast(float, current["utilization"]) - cast(float, candidate["utilization"]),
        delivered,
        delivered,
        max(observed_ticks) - min(observed_ticks),
        sources,
        digests,
        2,
        128,
    )


def _build_repair_components(
    directory: Path,
    *,
    split: DatasetSplit,
    read_truth: bool,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> tuple[ObservationalDecision, PotentialOutcomeTruth | None]:
    """Build one repair row; test requires an explicit GO authorization."""
    if split is DatasetSplit.TEST:
        if test_authorization is None:
            raise PermissionError("M10 test benchmark construction requires GO authorization")
        test_authorization.assert_allows(split)
    access = guard or CorpusAccessGuard()
    item = load_baseline_input(
        directory,
        baseline_id=BaselineId.B0,
        split=split,
        guard=access,
    )
    scenario_rows = cast(
        list[dict[str, object]],
        access.read_table(
            directory,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.DECISION,
            split=split,
            table=CorpusTable.SCENARIO,
        ).to_pylist(),
    )
    proposal_rows = cast(
        list[dict[str, object]],
        access.read_table(
            directory,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.DECISION,
            split=split,
            table=CorpusTable.ACTION_PROPOSALS,
        ).to_pylist(),
    )
    if len(scenario_rows) != 1:
        raise ValueError("repair benchmark requires one scenario row")
    baseline_proposals = [row for row in proposal_rows if row.get("is_baseline") is True]
    if len(baseline_proposals) != 1 or not isinstance(baseline_proposals[0].get("run_id"), str):
        raise ValueError("repair benchmark requires one baseline run ID")
    baseline_run_id = cast(str, baseline_proposals[0]["run_id"])
    candidate_run_ids = [
        row.get("run_id") for row in proposal_rows if row.get("is_baseline") is False
    ]
    if len(candidate_run_ids) != 1 or not isinstance(candidate_run_ids[0], str):
        raise ValueError("repair benchmark requires one candidate run ID")
    candidate_run_id = candidate_run_ids[0]
    visible_rows = cast(
        list[dict[str, object]],
        access.read_table(
            directory,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.DECISION,
            split=split,
            table=CorpusTable.VISIBLE,
        ).to_pylist(),
    )
    oracle_rows = cast(
        list[dict[str, object]],
        access.read_table(
            directory,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.TRAINING,
            split=split,
            table=CorpusTable.ORACLE,
            columns=["run_id", "tick", "path_id", "true_queue_occupancy"],
        ).to_pylist(),
    )
    tick_zero = {
        cast(str, row["path_id"]): cast(int, row["true_queue_occupancy"])
        for row in oracle_rows
        if row.get("run_id") == baseline_run_id and row.get("tick") == 0
    }
    if set(tick_zero) != {item.current_path_id, item.candidate_path_id}:
        raise ValueError("oracle tick zero does not contain exactly the paired paths")
    true_bin = _pressure_bin(tick_zero[item.current_path_id], tick_zero[item.candidate_path_id])
    current = item.snapshot(item.current_path_id)
    candidate = item.snapshot(item.candidate_path_id)
    visible_bin = (
        _pressure_bin(current.queue_occupancy, candidate.queue_occupancy)
        if current is not None and candidate is not None
        else None
    )
    if visible_bin is not None and visible_bin != true_bin:
        raise ValueError("visible and oracle queue-pressure bins disagree")
    pre_action = _pre_action_evidence(
        visible_rows,
        baseline_run_id=baseline_run_id,
        current_path_id=item.current_path_id,
        candidate_path_id=item.candidate_path_id,
    )
    if pre_action.queue_pressure_bin is not None and pre_action.queue_pressure_bin != true_bin:
        raise ValueError("acquired and oracle queue-pressure bins disagree")
    assigned, probability, uniform = _assignment(item.pair_id, true_bin)
    scenario = scenario_rows[0]
    traffic = scenario.get("traffic_level_id")
    regime = scenario.get("observation_regime_id")
    topology = scenario.get("topology_family_id")
    if not isinstance(traffic, str) or not isinstance(regime, str) or not isinstance(topology, str):
        raise ValueError("scenario has invalid topology/traffic/regime identifiers")
    observed: float | None = None
    truth: PotentialOutcomeTruth | None = None
    if read_truth:
        outcomes = load_outcomes(
            directory,
            item=item,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.TRAINING,
            guard=access,
        )
        accounting_rows = cast(
            list[dict[str, object]],
            access.read_table(
                directory,
                baseline_id=BaselineId.B0,
                phase=AccessPhase.TRAINING,
                split=split,
                table=CorpusTable.ACCOUNTING,
            ).to_pylist(),
        )
        schedule_rows = cast(
            list[dict[str, object]],
            access.read_table(
                directory,
                baseline_id=BaselineId.B0,
                phase=AccessPhase.TRAINING,
                split=split,
                table=CorpusTable.SCHEDULE,
                columns=["tick"],
            ).to_pylist(),
        )
        ticks = [row.get("tick") for row in schedule_rows]
        if not ticks or not all(isinstance(tick, int) for tick in ticks):
            raise ValueError("repair benchmark schedule has invalid ticks")
        duration_ticks = max(cast(int, tick) for tick in ticks) + 1
        observed = (
            outcomes.reroute.mean_latency_ms
            if assigned is ActionKind.REROUTE_TO_PATH
            else outcomes.keep.mean_latency_ms
        )
        truth = PotentialOutcomeTruth(
            item.pair_id,
            true_bin,
            _pilot_outcome(
                outcomes.keep,
                run_id=baseline_run_id,
                accounting_rows=accounting_rows,
                duration_ticks=duration_ticks,
                route_changes=0,
            ),
            _pilot_outcome(
                outcomes.reroute,
                run_id=candidate_run_id,
                accounting_rows=accounting_rows,
                duration_ticks=duration_ticks,
                route_changes=1,
            ),
        )
    decision = ObservationalDecision(
        item.pair_id,
        item.scenario_id,
        split,
        item.seed,
        topology,
        traffic,
        regime,
        assigned,
        observed,
        visible_bin,
        pre_action,
        reference_identification(pre_action.queue_pressure_bin is not None),
        probability,
        uniform,
    )
    return decision, truth


def build_repair_decision(
    directory: Path,
    *,
    split: DatasetSplit,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> ObservationalDecision:
    """Build a controller row without reading paired outcomes or accounting."""
    decision, truth = _build_repair_components(
        directory,
        split=split,
        read_truth=False,
        guard=guard,
        test_authorization=test_authorization,
    )
    assert truth is None
    return decision


def build_repair_record(
    directory: Path,
    *,
    split: DatasetSplit,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> PilotBenchmarkRecord:
    """Build one privileged train/development repair row for fitting or scoring."""
    decision, truth = _build_repair_components(
        directory,
        split=split,
        read_truth=True,
        guard=guard,
        test_authorization=test_authorization,
    )
    assert truth is not None
    return PilotBenchmarkRecord(decision, truth)


def build_repair_benchmark(
    directories: tuple[Path, ...],
    *,
    split: DatasetSplit,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> tuple[PilotBenchmarkRecord, ...]:
    if not directories:
        raise ValueError("repair benchmark requires at least one directory")
    access = guard or CorpusAccessGuard()
    return tuple(
        build_repair_record(
            directory,
            split=split,
            guard=access,
            test_authorization=test_authorization,
        )
        for directory in sorted(directories)
    )


def summarize_repair_benchmark(
    records: tuple[PilotBenchmarkRecord, ...],
) -> dict[str, object]:
    if not records:
        raise ValueError("repair summary requires at least one record")
    if any(record.decision.split is DatasetSplit.TEST for record in records):
        raise PermissionError("repair summary cannot contain test records")
    by_regime: dict[str, dict[str, int]] = {}
    for regime in sorted({record.decision.observation_regime_id for record in records}):
        selected = [record for record in records if record.decision.observation_regime_id == regime]
        by_regime[regime] = {
            "pair_count": len(selected),
            "identified_count": sum(
                record.decision.reference_identification.identified for record in selected
            ),
            "non_identified_count": sum(
                not record.decision.reference_identification.identified for record in selected
            ),
            "visible_pressure_count": sum(
                record.decision.visible_queue_pressure_bin is not None for record in selected
            ),
            "acquired_pressure_count": sum(
                record.decision.pre_action_evidence.queue_pressure_bin is not None
                for record in selected
            ),
        }
    support = Counter(
        (record.truth.true_queue_pressure_bin, record.decision.assigned_action.value)
        for record in records
    )
    return {
        "schema_version": "1.0.0",
        "graph_id": M10_GRAPH_ID,
        "graph_digest": M10_GRAPH_DIGEST,
        "assignment_policy_id": ASSIGNMENT_POLICY_ID,
        "pair_count": len(records),
        "split_counts": dict(
            sorted(Counter(record.decision.split.value for record in records).items())
        ),
        "regime_counts": by_regime,
        "assignment_support": {
            f"pressure_{pressure}:{action}": count
            for (pressure, action), count in sorted(support.items())
        },
        "reroute_beneficial_count": sum(record.truth.reroute_beneficial for record in records),
        "reroute_harmful_or_null_count": sum(
            not record.truth.reroute_beneficial for record in records
        ),
    }
