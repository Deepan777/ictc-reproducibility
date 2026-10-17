"""Determinism, analytics, accounting, and leakage tests for M03."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import pytest

from ictc.domain.enums import ActionKind
from ictc.simulation import (
    ArrivalMode,
    ArrivalProcess,
    PathConfig,
    PathTrafficConfig,
    SimulationConfig,
    TelemetryConfig,
    generate_schedule,
    load_simulation_config,
    run_action,
    run_paired,
)

ROOT = Path(__file__).resolve().parents[2]
TOY_CONFIG = ROOT / "configs/experiments/m03_toy.json"


def _constant_config(
    *,
    scenario_id: str,
    duration: int,
    foreground: int,
    current_service: int,
    candidate_service: int,
    queue_limit: int = 64,
    current_delay: float = 0.0,
    candidate_delay: float = 0.0,
    telemetry: TelemetryConfig | None = None,
) -> SimulationConfig:
    return SimulationConfig(
        scenario_id=scenario_id,
        seed=71,
        duration_ticks=duration,
        tick_duration_ms=1.0,
        deadline_ms=100.0,
        current_path_id="current",
        candidate_path_id="candidate",
        paths=(
            PathConfig(
                path_id="current",
                service_packets_per_tick=current_service,
                queue_limit_packets=queue_limit,
                propagation_delay_ms=current_delay,
            ),
            PathConfig(
                path_id="candidate",
                service_packets_per_tick=candidate_service,
                queue_limit_packets=queue_limit,
                propagation_delay_ms=candidate_delay,
            ),
        ),
        foreground_process=ArrivalProcess(
            mode=ArrivalMode.CONSTANT, rate_packets_per_tick=float(foreground)
        ),
        cross_traffic=(
            PathTrafficConfig(
                path_id="current",
                process=ArrivalProcess(mode=ArrivalMode.CONSTANT, rate_packets_per_tick=0),
            ),
            PathTrafficConfig(
                path_id="candidate",
                process=ArrivalProcess(mode=ArrivalMode.CONSTANT, rate_packets_per_tick=0),
            ),
        ),
        telemetry=telemetry or TelemetryConfig(),
    )


def test_analytic_propagation_fixture_and_pairing_invariant() -> None:
    config = load_simulation_config(TOY_CONFIG)
    pair = run_paired(config)

    assert pair.baseline.outcome.latency_ms == 11.0
    assert pair.intervention.outcome.latency_ms == 3.0
    assert pair.latency_effect_ms == -8.0
    assert pair.baseline.schedule_digest == pair.intervention.schedule_digest
    assert (
        pair.paired_outcome.baseline.random_schedule_digest
        == pair.paired_outcome.intervention.random_schedule_digest
    )
    assert pair.paired_outcome.baseline.action.kind is ActionKind.KEEP_CURRENT_PATH
    assert pair.paired_outcome.intervention.action.kind is ActionKind.REROUTE_TO_PATH

    baseline_inputs = pair.baseline.experimental_input_payload()
    intervention_inputs = pair.intervention.experimental_input_payload()
    assert baseline_inputs.pop("action") != intervention_inputs.pop("action")
    assert baseline_inputs == intervention_inputs


def test_same_seed_and_action_produce_identical_output() -> None:
    config = load_simulation_config(TOY_CONFIG)
    first_schedule = generate_schedule(config)
    second_schedule = generate_schedule(config)

    assert first_schedule == second_schedule
    assert first_schedule.digest == second_schedule.digest
    assert run_action(config, first_schedule, ActionKind.KEEP_CURRENT_PATH) == run_action(
        config, second_schedule, ActionKind.KEEP_CURRENT_PATH
    )


def test_fifo_queue_effect_matches_hand_calculation() -> None:
    config = _constant_config(
        scenario_id="analytic-queue",
        duration=3,
        foreground=2,
        current_service=1,
        candidate_service=2,
    )
    pair = run_paired(config)

    assert pair.baseline.outcome.latency_ms == pytest.approx(2.5)
    assert pair.intervention.outcome.latency_ms == pytest.approx(1.0)
    assert pair.latency_effect_ms == pytest.approx(-1.5)


def test_tail_drop_and_packet_accounting_invariants() -> None:
    config = _constant_config(
        scenario_id="analytic-loss",
        duration=2,
        foreground=3,
        current_service=1,
        candidate_service=3,
        queue_limit=2,
    )
    pair = run_paired(config)
    current = next(item for item in pair.baseline.accounting if item.path_id == "current")

    assert current.foreground_generated == 6
    assert current.foreground_delivered == 3
    assert current.foreground_dropped == 3
    assert pair.baseline.outcome.packet_loss_ratio == 0.5
    for run in (pair.baseline, pair.intervention):
        for accounting in run.accounting:
            accounting.validate()
            assert accounting.final_queue_occupancy == 0


def test_controller_projection_excludes_oracle_fields_and_models_impairments() -> None:
    config = _constant_config(
        scenario_id="telemetry-boundary",
        duration=3,
        foreground=1,
        current_service=2,
        candidate_service=2,
        telemetry=TelemetryConfig(
            sample_every_ticks=1,
            minimum_delay_ticks=2,
            maximum_delay_ticks=2,
            missing_probability=1.0,
            maximum_clock_skew_ticks=1,
            stale_after_ticks=1,
        ),
    )
    trace = run_paired(config).baseline.controller_input()

    assert not hasattr(trace, "oracle_trace")
    assert not hasattr(trace, "__dict__")
    record_names = {field.name for field in fields(type(trace.records[0]))}
    assert not any(name.startswith("true_") for name in record_names)
    assert not any("cross" in name or "capacity" in name for name in record_names)
    assert all(record.missing for record in trace.records)
    assert all(record.queue_occupancy is None for record in trace.records)
    assert all(record.utilization is None for record in trace.records)
    assert all(record.measurement_delay_ticks == 2 for record in trace.records)
    assert all(record.stale for record in trace.records)
    assert trace.available_at(1) == ()
    assert len(trace.available_at(2)) == 2


def test_source_failure_and_malicious_telemetry_are_deterministic_and_visible() -> None:
    config = _constant_config(
        scenario_id="adversarial-telemetry",
        duration=2,
        foreground=1,
        current_service=2,
        candidate_service=2,
        telemetry=TelemetryConfig(
            source_failure_probability=0.0,
            malicious_probability=1.0,
            malicious_queue_offset_packets=9,
        ),
    )

    trace = run_paired(config).baseline.controller_input()

    assert all(not record.missing for record in trace.records)
    assert all(
        record.queue_occupancy is not None and record.queue_occupancy >= 9
        for record in trace.records
    )


def test_source_failure_hides_telemetry() -> None:
    config = _constant_config(
        scenario_id="source-failure",
        duration=2,
        foreground=1,
        current_service=2,
        candidate_service=2,
        telemetry=TelemetryConfig(source_failure_probability=1.0),
    )

    trace = run_paired(config).baseline.controller_input()

    assert all(record.missing and record.queue_occupancy is None for record in trace.records)


def test_capacity_change_and_cross_traffic_shift_are_precommitted() -> None:
    config = _constant_config(
        scenario_id="dynamic-capacity",
        duration=4,
        foreground=1,
        current_service=4,
        candidate_service=4,
    ).model_copy(
        update={
            "capacity_change_tick": 2,
            "current_service_after_change": 1,
            "candidate_service_after_change": 3,
            "cross_traffic_shift_tick": 2,
            "cross_traffic_shift_multiplier": 2.0,
        }
    )

    schedule = generate_schedule(config)
    run = run_paired(config).baseline

    assert schedule.ticks[1].service_capacities == (4, 4)
    assert schedule.ticks[2].service_capacities == (1, 3)
    assert run.schedule_digest == schedule.digest
    assert any(record.true_service_capacity == 1 for record in run.oracle_trace.records)


def test_poisson_and_telemetry_schedules_are_seeded_and_action_independent() -> None:
    base = _constant_config(
        scenario_id="stochastic",
        duration=20,
        foreground=1,
        current_service=4,
        candidate_service=4,
    )
    payload = base.model_dump()
    payload.update(
        {
            "foreground_process": ArrivalProcess(
                mode=ArrivalMode.POISSON, rate_packets_per_tick=1.5
            ),
            "cross_traffic": (
                PathTrafficConfig(
                    path_id="current",
                    process=ArrivalProcess(mode=ArrivalMode.POISSON, rate_packets_per_tick=0.7),
                ),
                PathTrafficConfig(
                    path_id="candidate",
                    process=ArrivalProcess(mode=ArrivalMode.POISSON, rate_packets_per_tick=0.9),
                ),
            ),
            "telemetry": TelemetryConfig(
                minimum_delay_ticks=0,
                maximum_delay_ticks=3,
                missing_probability=0.25,
                maximum_clock_skew_ticks=2,
                stale_after_ticks=1,
            ),
        }
    )
    config = SimulationConfig.model_validate(payload)
    pair = run_paired(config)

    assert pair.schedule == generate_schedule(config)
    assert pair.baseline.schedule_digest == pair.intervention.schedule_digest
    assert len(pair.schedule.ticks) == 20
    assert any(tick.foreground_arrivals != 1 for tick in pair.schedule.ticks)


def test_schedule_mismatch_is_rejected() -> None:
    config = load_simulation_config(TOY_CONFIG)
    other = config.model_copy(update={"seed": config.seed + 1})
    with pytest.raises(ValueError, match="does not belong"):
        run_action(config, generate_schedule(other), ActionKind.KEEP_CURRENT_PATH)
