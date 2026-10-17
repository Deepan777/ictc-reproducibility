from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ictc.ns3.n12 import N12Scenario
from ictc.ns3.n13 import (
    CONFIRMATORY_SEEDS,
    DEVELOPMENT_SEEDS,
    RISK_FEATURE_NAMES,
    build_decision,
    load_matrix,
    predict_harm_probability,
)
from ictc.ns3.n13_analysis import _bootstrap_interval
from ictc.ns3.n13_selection import _candidates

REPOSITORY = Path(__file__).resolve().parents[2]


def _thresholds() -> dict[str, float | int]:
    return {
        "minimum_delivered_batches": 20,
        "maximum_loss_rate": 0.30,
        "maximum_latest_age_ms": 300.0,
        "maximum_delivery_delay_ms": 250.0,
        "queue_penalty_per_packet": 0.5,
        "delay_penalty_per_ms": 0.08,
        "lookback_batches": 12,
        "minimum_latest_score_improvement": 0.0,
        "minimum_candidate_up_fraction": 0.90,
        "minimum_candidate_consensus_fraction": 0.75,
        "maximum_candidate_queue_p90": 40.0,
        "maximum_candidate_capacity_cv": 0.20,
        "minimum_downside_margin_p10": 0.0,
    }


def _probe(root: Path, *, unstable_candidate: bool) -> N12Scenario:
    scenario = N12Scenario(
        scenario_id="n13-test",
        phase="development",
        traffic="web",
        event="failure",
        matrix=0,
        seed=86000,
        run=1,
        duration_seconds=8.0,
        decision_at_seconds=3.5,
        event_at_seconds=2.2,
    )
    (root / "plan.json").write_text(
        json.dumps(
            {
                "schema_version": "n12-native-plan-v1",
                "scenario_id": scenario.scenario_id,
                "action": "route0",
                "seed": scenario.seed,
            }
        ),
        encoding="utf-8",
    )
    telemetry_fields = (
        "sample_timestamp_ns",
        "available_timestamp_ns",
        "phase",
        "route_id",
        "bottleneck_capacity_bps",
        "propagation_delay_ms",
        "queue_packets",
        "link_up",
    )
    with (root / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=telemetry_fields)
        writer.writeheader()
        for sample in range(20):
            sample_ns = 1_400_000_000 + sample * 100_000_000
            for route in range(3):
                route2_up = not unstable_candidate or sample >= 15
                writer.writerow(
                    {
                        "sample_timestamp_ns": sample_ns,
                        "available_timestamp_ns": sample_ns + 10_000_000,
                        "phase": "PRE_DECISION",
                        "route_id": route,
                        "bottleneck_capacity_bps": (10 + route * 10) * 1_000_000,
                        "propagation_delay_ms": 50 - route * 10,
                        "queue_packets": 20 - route * 5,
                        "link_up": int(route != 2 or route2_up),
                    }
                )
    delivery_fields = (
        "sample_timestamp_ns",
        "available_timestamp_ns",
        "phase",
        "dropped",
        "delay_ns",
    )
    with (root / "telemetry_delivery.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=delivery_fields)
        writer.writeheader()
        for sample in range(20):
            sample_ns = 1_400_000_000 + sample * 100_000_000
            writer.writerow(
                {
                    "sample_timestamp_ns": sample_ns,
                    "available_timestamp_ns": sample_ns + 10_000_000,
                    "phase": "PRE_DECISION",
                    "dropped": 0,
                    "delay_ns": 10_000_000,
                }
            )
    return scenario


def test_n13_development_matrix_is_complete_and_confirmation_is_reserved() -> None:
    raw, scenarios = load_matrix(
        REPOSITORY / "configs" / "experiments" / "n13_temporal_risk_development.json"
    )
    assert len(scenarios) == 288
    assert {scenario.seed for scenario in scenarios} == set(DEVELOPMENT_SEEDS)
    assert not set(DEVELOPMENT_SEEDS) & set(CONFIRMATORY_SEEDS)
    assert raw["reserved_confirmatory_seeds"] == list(CONFIRMATORY_SEEDS)

    confirmation, held_out = load_matrix(
        REPOSITORY / "configs" / "experiments" / "n13_temporal_risk_confirmation.json"
    )
    assert len(held_out) == 720
    assert {scenario.seed for scenario in held_out} == set(CONFIRMATORY_SEEDS)
    assert confirmation["selected_policy_sha256"] == (
        "3F3F68797122C488D0AB9D48D04B349C671D3AF4B9A8F71E4BD1A0C9D157404C"
    )


def test_n13_temporal_risk_rejects_an_unstable_latest_candidate(tmp_path: Path) -> None:
    scenario = _probe(tmp_path, unstable_candidate=True)
    decision = build_decision(scenario, tmp_path, _thresholds())
    assert decision["greedy_action"] == "route2"
    assert decision["decisions"]["S1_ICTC_CONTRACT"] == "route0"
    assert decision["contract_ready"] is False
    assert "ROUTE_HEALTH_PERSISTENCE" in decision["rejection_reasons"]
    assert decision["created_before_outcomes"] is True


def test_n13_temporal_risk_accepts_a_stable_candidate(tmp_path: Path) -> None:
    scenario = _probe(tmp_path, unstable_candidate=False)
    decision = build_decision(scenario, tmp_path, _thresholds())
    assert decision["greedy_action"] == "route2"
    assert decision["decisions"]["S1_ICTC_CONTRACT"] == "route2"
    assert decision["contract_ready"] is True
    assert decision["rejection_reasons"] == []


def test_n13_candidate_grid_is_prospectively_bounded() -> None:
    protocol = json.loads(
        (REPOSITORY / "configs" / "experiments" / "n13_policy_selection_protocol.json").read_text(
            encoding="utf-8"
        )
    )
    candidates = _candidates(protocol)
    assert len(candidates) == 216
    assert {candidate["candidate_id"] for candidate in candidates} == {
        f"C{index:03d}" for index in range(216)
    }


def test_n13_frozen_logistic_model_scores_without_training_dependency() -> None:
    features = tuple(0.0 for _ in RISK_FEATURE_NAMES)
    model = {
        "feature_names": list(RISK_FEATURE_NAMES),
        "means": [0.0] * len(features),
        "scales": [1.0] * len(features),
        "coefficients": [0.0] * len(features),
        "intercept": 0.0,
    }
    assert predict_harm_probability(features, model) == 0.5


def test_n13_seed_bootstrap_is_deterministic() -> None:
    values = np.array([-0.02, -0.01, -0.03, -0.02])
    first = _bootstrap_interval(values, resamples=1000, seed=12, confidence=0.95)
    second = _bootstrap_interval(values, resamples=1000, seed=12, confidence=0.95)
    assert first == second
    assert first[1] < 0
