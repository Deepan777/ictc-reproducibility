"""Deterministic paired two-path network environment for M03."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ictc.domain.enums import ActionKind
from ictc.domain.models import InterventionOutcome, NetworkAction, PairedCounterfactualOutcome
from ictc.simulation.models import SimulationConfig
from ictc.simulation.queueing import PacketAccounting, PacketQueue, ServiceResult
from ictc.simulation.schedule import ExogenousSchedule, TelemetryImpairment, generate_schedule


@dataclass(frozen=True, slots=True)
class OracleRecord:
    """Ground truth that must never be passed to a tested controller."""

    run_id: str
    tick: int
    phase: str
    path_id: str
    selected_path: bool
    true_foreground_arrivals: int
    true_cross_arrivals: int
    true_service_capacity: int
    true_queue_occupancy: int
    true_foreground_served: int
    true_cross_served: int
    true_foreground_dropped: int
    true_cross_dropped: int


@dataclass(frozen=True, slots=True)
class VisibleTelemetryRecord:
    """Controller-safe observation with explicit quality metadata."""

    run_id: str
    sample_tick: int
    available_tick: int
    observed_timestamp_tick: int
    path_id: str
    queue_occupancy: int | None
    utilization: float | None
    sample_age_ticks: int
    measurement_delay_ticks: int
    clock_skew_ticks: int
    stale: bool
    missing: bool
    source_id: str
    provenance_digest: str


@dataclass(frozen=True, slots=True)
class ControllerTrace:
    """The only simulation trace type accepted by controller-facing code."""

    run_id: str
    records: tuple[VisibleTelemetryRecord, ...]

    def available_at(self, tick: int) -> tuple[VisibleTelemetryRecord, ...]:
        """Return samples delivered to the controller by a decision tick."""
        return tuple(
            sorted(
                (record for record in self.records if record.available_tick <= tick),
                key=lambda record: (record.available_tick, record.sample_tick, record.path_id),
            )
        )


@dataclass(frozen=True, slots=True)
class OracleTrace:
    """Separate ground-truth trace for validation and later corpus generation."""

    run_id: str
    records: tuple[OracleRecord, ...]


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """One fixed-action run under a pre-generated exogenous schedule."""

    run_id: str
    config_hash: str
    schedule_digest: str
    action: NetworkAction
    selected_path_id: str
    outcome: InterventionOutcome
    accounting: tuple[PacketAccounting, ...]
    controller_trace: ControllerTrace
    oracle_trace: OracleTrace

    def controller_input(self) -> ControllerTrace:
        """Return the controller-safe projection without an oracle back-reference."""
        return self.controller_trace

    def experimental_input_payload(self) -> dict[str, str]:
        """Expose the fields used to verify paired-run input identity."""
        return {
            "config_hash": self.config_hash,
            "schedule_digest": self.schedule_digest,
            "action": self.action.kind.value,
        }


@dataclass(frozen=True, slots=True)
class PairedSimulation:
    """KEEP and REROUTE runs that share every exogenous input."""

    pair_id: str
    config: SimulationConfig
    schedule: ExogenousSchedule
    baseline: SimulationRun
    intervention: SimulationRun

    def __post_init__(self) -> None:
        baseline_input = self.baseline.experimental_input_payload()
        intervention_input = self.intervention.experimental_input_payload()
        baseline_action = baseline_input.pop("action")
        intervention_action = intervention_input.pop("action")
        if baseline_input != intervention_input:
            raise ValueError("paired runs differ in an input other than action")
        if baseline_action != ActionKind.KEEP_CURRENT_PATH.value:
            raise ValueError("paired baseline must keep the current path")
        if intervention_action != ActionKind.REROUTE_TO_PATH.value:
            raise ValueError("paired intervention must reroute")
        if self.schedule.digest != self.baseline.schedule_digest:
            raise ValueError("paired schedule does not match run digest")

    @property
    def paired_outcome(self) -> PairedCounterfactualOutcome:
        """Return the validated domain-level paired outcome."""
        return PairedCounterfactualOutcome(
            pair_id=self.pair_id,
            baseline=self.baseline.outcome,
            intervention=self.intervention.outcome,
        )

    @property
    def latency_effect_ms(self) -> float:
        """Return intervention minus baseline mean foreground latency."""
        return self.intervention.outcome.latency_ms - self.baseline.outcome.latency_ms


def _telemetry_record(
    *,
    run_id: str,
    tick: int,
    path_id: str,
    occupancy: int,
    service: ServiceResult,
    capacity: int,
    impairment: TelemetryImpairment,
    stale_after_ticks: int,
    malicious_queue_offset_packets: int,
) -> VisibleTelemetryRecord:
    missing = impairment.missing
    queue_value = None if missing else occupancy
    if queue_value is not None and impairment.malicious:
        queue_value += malicious_queue_offset_packets
    utilization = None if missing else service.total / capacity
    payload = {
        "run_id": run_id,
        "sample_tick": tick,
        "path_id": path_id,
        "queue_occupancy": queue_value,
        "utilization": utilization,
        "missing": missing,
    }
    provenance = (
        hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
        .hexdigest()
        .upper()
    )
    return VisibleTelemetryRecord(
        run_id=run_id,
        sample_tick=tick,
        available_tick=tick + impairment.delay_ticks,
        observed_timestamp_tick=tick + impairment.clock_skew_ticks,
        path_id=path_id,
        queue_occupancy=queue_value,
        utilization=utilization,
        sample_age_ticks=impairment.delay_ticks,
        measurement_delay_ticks=impairment.delay_ticks,
        clock_skew_ticks=impairment.clock_skew_ticks,
        stale=impairment.delay_ticks > stale_after_ticks,
        missing=missing,
        source_id=f"telemetry:{path_id}",
        provenance_digest=provenance,
    )


def _network_action(
    config: SimulationConfig, kind: ActionKind, schedule_digest: str
) -> NetworkAction:
    action_suffix = kind.value.lower()
    return NetworkAction(
        action_id=f"{config.scenario_id}:{action_suffix}:{schedule_digest[:12]}",
        kind=kind,
        path_id=config.candidate_path_id if kind is ActionKind.REROUTE_TO_PATH else None,
    )


def run_action(
    config: SimulationConfig, schedule: ExogenousSchedule, kind: ActionKind,
    *, foreground_start_tick: int = 0,
) -> SimulationRun:
    """Run one fixed action without consuming any new random number."""
    if kind not in {ActionKind.KEEP_CURRENT_PATH, ActionKind.REROUTE_TO_PATH}:
        raise ValueError("M03 supports only KEEP and REROUTE actions")
    if not 0 <= foreground_start_tick < config.duration_ticks:
        raise ValueError("foreground start must be inside the simulated episode")
    if schedule.config_hash != config.config_hash or schedule.scenario_id != config.scenario_id:
        raise ValueError("schedule does not belong to the supplied configuration")
    if len(schedule.ticks) != config.duration_ticks:
        raise ValueError("schedule duration differs from configuration")

    selected_path = (
        config.current_path_id if kind is ActionKind.KEEP_CURRENT_PATH else config.candidate_path_id
    )
    action = _network_action(config, kind, schedule.digest)
    run_id = f"{config.scenario_id}:{kind.value.lower()}:{schedule.digest[:12]}"
    queues = {
        path.path_id: PacketQueue(
            path_id=path.path_id,
            service_capacity=path.service_packets_per_tick,
            queue_limit=path.queue_limit_packets,
        )
        for path in config.paths
    }
    oracle_records: list[OracleRecord] = []
    visible_records: list[VisibleTelemetryRecord] = []

    for tick, tick_schedule in enumerate(schedule.ticks):
        for path_index, path in enumerate(config.paths):
            queue = queues[path.path_id]
            cross_arrivals = tick_schedule.cross_arrivals[path_index]
            foreground_arrivals = (
                tick_schedule.foreground_arrivals
                if tick >= foreground_start_tick and path.path_id == selected_path else 0
            )
            dropped_cross = queue.enqueue("cross", cross_arrivals, tick)
            dropped_foreground = queue.enqueue("foreground", foreground_arrivals, tick)
            capacity = tick_schedule.service_capacities[path_index]
            service = queue.service(tick, capacity)
            oracle_records.append(
                OracleRecord(
                    run_id=run_id,
                    tick=tick,
                    phase="arrivals",
                    path_id=path.path_id,
                    selected_path=path.path_id == selected_path,
                    true_foreground_arrivals=foreground_arrivals,
                    true_cross_arrivals=cross_arrivals,
                    true_service_capacity=capacity,
                    true_queue_occupancy=queue.occupancy,
                    true_foreground_served=service.foreground,
                    true_cross_served=service.cross,
                    true_foreground_dropped=dropped_foreground,
                    true_cross_dropped=dropped_cross,
                )
            )
            if tick % config.telemetry.sample_every_ticks == 0:
                visible_records.append(
                    _telemetry_record(
                        run_id=run_id,
                        tick=tick,
                        path_id=path.path_id,
                        occupancy=queue.occupancy,
                        service=service,
                        capacity=capacity,
                        impairment=tick_schedule.telemetry_impairments[path_index],
                        stale_after_ticks=config.telemetry.stale_after_ticks,
                        malicious_queue_offset_packets=config.telemetry.malicious_queue_offset_packets,
                    )
                )

    drain_tick = config.duration_ticks
    while any(not queue.empty for queue in queues.values()):
        for path in config.paths:
            queue = queues[path.path_id]
            service = queue.service(drain_tick)
            oracle_records.append(
                OracleRecord(
                    run_id=run_id,
                    tick=drain_tick,
                    phase="drain",
                    path_id=path.path_id,
                    selected_path=path.path_id == selected_path,
                    true_foreground_arrivals=0,
                    true_cross_arrivals=0,
                    true_service_capacity=path.service_packets_per_tick,
                    true_queue_occupancy=queue.occupancy,
                    true_foreground_served=service.foreground,
                    true_cross_served=service.cross,
                    true_foreground_dropped=0,
                    true_cross_dropped=0,
                )
            )
        drain_tick += 1

    accounting = tuple(queues[path.path_id].accounting() for path in config.paths)
    selected_accounting = next(item for item in accounting if item.path_id == selected_path)
    latency_ticks = queues[selected_path].foreground_latency_ticks
    selected_config = config.path(selected_path)
    latencies_ms = tuple(
        ticks * config.tick_duration_ms + selected_config.propagation_delay_ms
        for ticks in latency_ticks
    )
    mean_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    jitter = (
        sum(abs(right - left) for left, right in zip(latencies_ms, latencies_ms[1:], strict=False))
        / (len(latencies_ms) - 1)
        if len(latencies_ms) > 1
        else 0.0
    )
    generated = selected_accounting.foreground_generated
    dropped = selected_accounting.foreground_dropped
    loss_ratio = dropped / generated if generated else 0.0
    late_deliveries = sum(latency > config.deadline_ms for latency in latencies_ms)
    deadline_violation_rate = (late_deliveries + dropped) / generated if generated else 0.0
    outcome = InterventionOutcome(
        scenario_id=config.scenario_id,
        run_id=run_id,
        action=action,
        random_schedule_digest=schedule.digest,
        latency_ms=mean_latency,
        packet_loss_ratio=loss_ratio,
        jitter_ms=jitter,
        deadline_violation_rate=deadline_violation_rate,
    )
    return SimulationRun(
        run_id=run_id,
        config_hash=config.config_hash,
        schedule_digest=schedule.digest,
        action=action,
        selected_path_id=selected_path,
        outcome=outcome,
        accounting=accounting,
        controller_trace=ControllerTrace(run_id=run_id, records=tuple(visible_records)),
        oracle_trace=OracleTrace(run_id=run_id, records=tuple(oracle_records)),
    )


def run_paired(config: SimulationConfig, *, foreground_start_tick: int = 0) -> PairedSimulation:
    """Run KEEP and REROUTE against one shared exogenous schedule."""
    schedule = generate_schedule(config)
    baseline = run_action(config, schedule, ActionKind.KEEP_CURRENT_PATH,
                          foreground_start_tick=foreground_start_tick)
    intervention = run_action(config, schedule, ActionKind.REROUTE_TO_PATH,
                              foreground_start_tick=foreground_start_tick)
    return PairedSimulation(
        pair_id=f"{config.scenario_id}:pair:{schedule.digest[:12]}",
        config=config,
        schedule=schedule,
        baseline=baseline,
        intervention=intervention,
    )
