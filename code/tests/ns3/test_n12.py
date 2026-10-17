from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ictc.ns3.n12 import (
    METHODS,
    N12Scenario,
    build_decision,
    load_matrix,
    utility,
)
from ictc.ns3.n12_analysis import _exact_sign_flip_pvalue

REPOSITORY = Path(__file__).resolve().parents[2]


def test_n12_development_matrix_is_complete_and_seed_separated() -> None:
    raw, scenarios = load_matrix(
        REPOSITORY / "configs" / "experiments" / "n12_realistic_development.json"
    )
    assert raw["phase"] == "development"
    assert len(scenarios) == 144
    assert {item.seed for item in scenarios} == set(range(84000, 84004))
    assert not {item.seed for item in scenarios} & set(raw["reserved_confirmatory_seeds"])


def test_n12_confirmation_is_twenty_new_seeds_and_policy_bound() -> None:
    raw, scenarios = load_matrix(
        REPOSITORY / "configs" / "experiments" / "n12_realistic_confirmation.json"
    )
    assert raw["phase"] == "confirmatory"
    assert len(scenarios) == 720
    assert {item.seed for item in scenarios} == set(range(85000, 85020))
    policy = json.loads(
        (REPOSITORY / "configs" / "experiments" / "n12_selected_policy.json").read_text(
            encoding="utf-8"
        )
    )
    assert policy["controller_thresholds"] == raw["controller_thresholds"]


def test_n12_native_source_declares_realistic_network_features() -> None:
    source = (REPOSITORY / "ns3" / "scratch" / "ictc_n12_abilene.cc").read_text(encoding="utf-8")
    assert "std::array<EdgeSpec, 14>" in source
    assert "routers.Create(11)" in source
    assert "TcpCubic::GetTypeId()" in source
    assert "FqCoDelQueueDisc" in source
    assert all(event in source for event in ("failure", "capacity", "cross-shift"))


def _synthetic_probe(root: Path) -> N12Scenario:
    scenario = N12Scenario(
        scenario_id="n12-test",
        phase="development",
        traffic="web",
        event="failure",
        matrix=0,
        seed=84000,
        run=1,
        duration_seconds=8.0,
        decision_at_seconds=3.5,
        event_at_seconds=2.2,
    )
    plan = {
        "schema_version": "n12-native-plan-v1",
        "scenario_id": scenario.scenario_id,
        "action": "route0",
        "seed": scenario.seed,
    }
    (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    with (root / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        delivery_fields = (
            "sample_timestamp_ns",
            "available_timestamp_ns",
            "phase",
            "route_id",
            "bottleneck_capacity_bps",
            "propagation_delay_ms",
            "queue_packets",
            "link_up",
        )
        writer = csv.DictWriter(handle, fieldnames=delivery_fields)
        writer.writeheader()
        for sample in range(20):
            for route in range(3):
                writer.writerow(
                    {
                        "sample_timestamp_ns": 1_000_000_000 + sample * 100_000_000,
                        "available_timestamp_ns": 1_010_000_000 + sample * 100_000_000,
                        "phase": "PRE_DECISION",
                        "route_id": route,
                        "bottleneck_capacity_bps": 10_000_000 + route * 10_000_000,
                        "propagation_delay_ms": 50 - route * 10,
                        "queue_packets": 20 - route * 5,
                        "link_up": 0 if route == 0 else 1,
                    }
                )
    with (root / "telemetry_delivery.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = (
            "sample_timestamp_ns",
            "available_timestamp_ns",
            "phase",
            "dropped",
            "delay_ns",
        )
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in range(20):
            writer.writerow(
                {
                    "sample_timestamp_ns": 1_000_000_000 + sample * 100_000_000,
                    "available_timestamp_ns": 1_010_000_000 + sample * 100_000_000,
                    "phase": "PRE_DECISION",
                    "dropped": 0,
                    "delay_ns": 10_000_000,
                }
            )
    return scenario


def test_n12_decisions_use_only_delivered_probe_and_abstain_safely(tmp_path: Path) -> None:
    scenario = _synthetic_probe(tmp_path)
    thresholds = {
        "minimum_delivered_batches": 20,
        "maximum_loss_rate": 0.3,
        "maximum_latest_age_ms": 600.0,
        "maximum_delivery_delay_ms": 250.0,
        "queue_penalty_per_packet": 0.25,
        "delay_penalty_per_ms": 0.08,
        "minimum_score_improvement": 1.5,
    }
    decision = build_decision(scenario, tmp_path, thresholds)
    assert set(decision["decisions"]) == set(METHODS)
    assert decision["decisions"]["S1_ICTC_CONTRACT"] == "route2"
    assert not any("ORACLE" in method for method in decision["decisions"])
    assert decision["created_before_outcomes"] is True


def test_n12_predeclared_utility_rewards_delivery_and_penalizes_delay() -> None:
    good = {
        "delivery_ratio": "1",
        "throughput_mbps": "5",
        "p95_delay_ms": "20",
        "completion_ratio": "1",
        "queue_drops": "0",
    }
    bad = {
        "delivery_ratio": "0.5",
        "throughput_mbps": "1",
        "p95_delay_ms": "180",
        "completion_ratio": "0.5",
        "queue_drops": "100",
    }
    assert utility(good) > utility(bad)


def test_n12_exact_sign_flip_test_has_expected_extremes() -> None:
    assert _exact_sign_flip_pvalue(np.array([1.0, 1.0, 1.0])) == 0.125
    assert _exact_sign_flip_pvalue(np.array([-1.0, -1.0, -1.0])) == 1.0
