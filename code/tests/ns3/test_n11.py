"""Regression tests for N11 stochastic plans, decisions, and frozen matrix."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest

from ictc.ns3.n08 import ChangeProbeEvidence
from ictc.ns3.n11 import (
    N11MethodId,
    N11Probe,
    ProbeQuality,
    RealizedPath,
    StochasticPlan,
    build_n11_generations,
    load_n11_matrix,
    read_stochastic_plan,
    read_telemetry_delivery,
)
from ictc.ns3.stochastic import StochasticScenario

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n11_genuinely_stochastic_validation.json"


def _plan(scenario_id: str, seed: int) -> StochasticPlan:
    return StochasticPlan(
        scenario_id=scenario_id,
        seed=seed,
        run=1,
        foreground_mean_interval_ms=2.1,
        telemetry_loss_probability=0.1,
        telemetry_delay_mean_ms=40.0,
        telemetry_delay_maximum_ms=200.0,
        failure_start_seconds=3.1,
        failure_duration_seconds=1.3,
        capacity_event_start_seconds=3.1,
        capacity_drop_rate_bps=1_000_000,
        cross_shift_start_seconds=3.1,
        cross_shift_duration_seconds=1.5,
        cross_shift_rate_bps=6_000_000,
        paths=(
            RealizedPath("current", 5_000_000, 5_000_000, 64, 1_000_000),
            RealizedPath("candidate", 4_500_000, 1_000_000, 72, 2_000_000),
        ),
    )


def _probe() -> tuple[StochasticScenario, N11Probe]:
    scenario = load_n11_matrix(MATRIX)[0]
    base = ChangeProbeEvidence(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        path_ids=("current", "candidate"),
        latest_timestamp_ns=3_700_000_000,
        latest_queues={"current": 0, "candidate": 0},
        latest_link_up={"current": True, "candidate": True},
        latest_rate_bps={"current": 5_000_000, "candidate": 4_500_000},
        pre_decision_digest="1" * 64,
        state_digest="2" * 64,
        observation_digest="3" * 64,
        event_detected=True,
        telemetry_sha256="4" * 64,
        state_sha256="5" * 64,
        background_sha256="6" * 64,
    )
    quality = ProbeQuality(38, 34, 4, 4 / 38, 40_000_000.0, 100_000_000, 100_000_000)
    return scenario, N11Probe(
        base=base,
        plan=_plan(scenario.scenario_id, scenario.seed),
        quality=quality,
        plan_sha256="7" * 64,
        pre_delivery_digest="8" * 64,
        event_detected=True,
    )


def test_n11_matrix_freezes_twenty_new_seeds_and_all_random_mechanisms(
    tmp_path: Path,
) -> None:
    scenarios = load_n11_matrix(MATRIX)
    assert len(scenarios) == 720
    assert sum(len(item.actions) for item in scenarios) == 1680
    assert len(scenarios) * len(N11MethodId) == 4320
    assert {item.seed for item in scenarios} == set(range(83000, 83020))
    raw = json.loads(MATRIX.read_text(encoding="utf-8"))
    assert set(raw["stochastic_factors"]) == {
        "packet_arrivals",
        "cross_traffic_intensity",
        "failure_time",
        "failure_duration",
        "link_capacity",
        "capacity_change",
        "propagation_delay",
        "queue_behavior",
        "telemetry_loss",
        "telemetry_delay",
    }
    raw["stochastic_factors"]["packet_arrivals"] = "changed after freeze"
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="stochastic-factor"):
        load_n11_matrix(invalid)


def test_n11_decision_uses_delayed_lossy_contract_and_relative_capacity_support() -> None:
    scenario, probe = _probe()
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    ledger = build_n11_generations(
        scenario,
        probe,
        "9" * 64,
        source_n10_manifest_sha256=matrix["source_n10_manifest_sha256"],
        source_n10_analysis_sha256=matrix["source_n10_analysis_sha256"],
        support_design_digest=matrix["support_design_digest"],
        support_design_file_sha256=matrix["support_design_file_sha256"],
        supported_contexts=tuple(matrix["supported_contexts"]),
        thresholds=matrix["controller_thresholds"],
    )
    generation = ledger["generations"][1]
    decisions = {row["method_id"]: row for row in generation["decisions"]}
    assert generation["contract_ready"] is True
    assert generation["support_applied"] is True
    assert decisions[N11MethodId.S1_STOCHASTIC_ICTC]["selected_action"] == "reroute"
    assert decisions[N11MethodId.B3_CAPACITY]["selected_action"] == "keep"
    assert not any(ledger["forbidden_inputs"].values())

    stale = N11Probe(
        base=probe.base,
        plan=probe.plan,
        quality=ProbeQuality(38, 8, 30, 30 / 38, 40_000_000.0, 100_000_000, 100_000_000),
        plan_sha256=probe.plan_sha256,
        pre_delivery_digest=probe.pre_delivery_digest,
        event_detected=True,
    )
    rejected = build_n11_generations(
        scenario,
        stale,
        "9" * 64,
        source_n10_manifest_sha256=matrix["source_n10_manifest_sha256"],
        source_n10_analysis_sha256=matrix["source_n10_analysis_sha256"],
        support_design_digest=matrix["support_design_digest"],
        support_design_file_sha256=matrix["support_design_file_sha256"],
        supported_contexts=tuple(matrix["supported_contexts"]),
        thresholds=matrix["controller_thresholds"],
    )
    assert rejected["generations"][1]["contract_ready"] is False
    assert rejected["generations"][1]["support_applied"] is False


def test_n11_plan_and_telemetry_delivery_parsers_reject_identity_and_order_defects(
    tmp_path: Path,
) -> None:
    scenario = load_n11_matrix(MATRIX)[0]
    plan = _plan(scenario.scenario_id, scenario.seed)
    payload = {
        "schema_version": "n11-stochastic-plan-v1",
        **{key: value for key, value in asdict(plan).items() if key != "paths"},
        "paths": [asdict(path) for path in plan.paths],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert read_stochastic_plan(path, scenario) == plan
    payload["seed"] += 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_stochastic_plan(path, scenario)

    delivery = tmp_path / "delivery.csv"
    with delivery.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "sample_timestamp_ns",
                "available_timestamp_ns",
                "phase",
                "dropped",
                "delay_ns",
                "path_count",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_timestamp_ns": 1_000,
                "available_timestamp_ns": 1_100,
                "phase": "PRE_DECISION",
                "dropped": 0,
                "delay_ns": 100,
                "path_count": 2,
            }
        )
    assert len(read_telemetry_delivery(delivery)) == 1
    delivery.write_text(
        delivery.read_text(encoding="utf-8").replace(",100,2", ",99,2"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ordering or bounds"):
        read_telemetry_delivery(delivery)
