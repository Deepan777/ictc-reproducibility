"""M14 final simulation matrix tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ictc.simulation.m14_matrix import load_m14_simulation_matrix
from ictc.simulation.m14_telemetry import reconstruct_pre_action_telemetry

ROOT = Path(__file__).resolve().parents[2]


def test_m14_matrix_covers_all_frozen_conditions_with_twenty_seeds() -> None:
    scenarios = load_m14_simulation_matrix(
        ROOT / "configs" / "experiments" / "m14_final_simulation_matrix.json"
    )

    assert len(scenarios) == 4320
    counts = Counter(item.condition_id for item in scenarios)
    assert counts["unseen_topology"] == 120
    assert all(
        count == 420 for condition, count in counts.items() if condition != "unseen_topology"
    )
    assert {item.seed for item in scenarios} == set(range(60000, 60020))


def test_m14_dynamic_conditions_bind_their_action_independent_inputs() -> None:
    scenarios = load_m14_simulation_matrix(
        ROOT / "configs" / "experiments" / "m14_final_simulation_matrix.json"
    )
    topology_change = next(item for item in scenarios if item.condition_id == "topology_change")
    cross_traffic_shift = next(
        item for item in scenarios if item.condition_id == "cross_traffic_shift"
    )

    assert topology_change.config.capacity_change_tick == 32
    assert topology_change.config.candidate_service_after_change == 9
    assert cross_traffic_shift.config.cross_traffic_shift_tick == 32
    assert cross_traffic_shift.config.cross_traffic_shift_multiplier == 2.0


def test_repair_matrix_uses_twenty_fresh_seeds() -> None:
    scenarios = load_m14_simulation_matrix(
        ROOT / "configs" / "experiments" / "m14_confirmatory_repair_matrix.json"
    )
    assert len(scenarios) == 4320
    assert {scenario.seed for scenario in scenarios} == set(range(70000, 70020))

    malicious = next(
        scenario for scenario in scenarios if scenario.condition_id == "malicious_telemetry"
    )
    malicious_telemetry = reconstruct_pre_action_telemetry(malicious)
    assert all(sample["provenance_valid"] is False for sample in malicious_telemetry.samples)
    infeasible = next(
        scenario for scenario in scenarios if scenario.condition_id == "contract_infeasibility"
    )
    assert reconstruct_pre_action_telemetry(infeasible).contract_feasible is False
