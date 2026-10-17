"""Known-answer test for N11 seed distributions and variability gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n07 import PATH_SPECS
from ictc.ns3.n11 import N11MethodId, load_n11_matrix
from ictc.ns3.n11_analysis import analyze_n11, render_n11_seed_csv
from ictc.ns3.stochastic import sha256

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n11_genuinely_stochastic_validation.json"


def _write_fixture(root: Path) -> None:
    methods = tuple(method.value for method in N11MethodId)
    for scenario_index, scenario in enumerate(load_n11_matrix(MATRIX)):
        seed_offset = scenario.seed - 83000
        paths = [
            {
                "path_id": path_id,
                "initial_capacity_bps": 4_000_000 + seed_offset * 10_000 + index * 100_000,
                "propagation_delay_ns": 1_000_000 + seed_offset * 1_000 + index * 100_000,
                "queue_limit_packets": 48 + seed_offset + index,
                "cross_traffic_rate_bps": 500_000 + seed_offset * 10_000 + index * 50_000,
            }
            for index, path_id in enumerate(PATH_SPECS[scenario.topology_family.value])
        ]
        plan = {
            "schema_version": "n11-stochastic-plan-v1",
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "run": 1,
            "foreground_mean_interval_ms": 1.5 + seed_offset / 100,
            "telemetry_loss_probability": 0.03 + seed_offset / 1000,
            "telemetry_delay_mean_ms": 10.0 + seed_offset,
            "telemetry_delay_maximum_ms": 200.0,
            "failure_start_seconds": 2.8 + seed_offset / 1000,
            "failure_duration_seconds": 1.0 + seed_offset / 100,
            "capacity_event_start_seconds": 2.8 + seed_offset / 1000,
            "capacity_drop_rate_bps": 800_000 + seed_offset * 10_000,
            "cross_shift_start_seconds": 2.8 + seed_offset / 1000,
            "cross_shift_duration_seconds": 1.2 + seed_offset / 100,
            "cross_shift_rate_bps": 5_000_000 + seed_offset * 10_000,
            "paths": paths,
        }
        raw = root / "outcomes" / "scenarios" / scenario.scenario_id / "keep" / "raw"
        raw.mkdir(parents=True)
        plan_path = raw / "stochastic_plan.json"
        plan_path.write_text(json.dumps(plan), encoding="utf-8")

        rows: list[dict[str, Any]] = []
        for method_index, method in enumerate(methods):
            tx = 1000
            rx = 900 + seed_offset
            rows.append(
                {
                    "method_id": method,
                    "physical_label": (
                        "BENEFICIAL" if method == N11MethodId.S1_STOCHASTIC_ICTC.value else "KEEP"
                    ),
                    "outcome": {
                        "tx_packets": tx,
                        "rx_packets": rx,
                        "delivery_ratio": rx / tx,
                        "mean_delay_ns": 1_000_000 + seed_offset * 1_000 + method_index,
                        "jitter_ns": 100_000 + seed_offset * 100 + method_index,
                        "throughput_bps": (
                            1_000_000 + seed_offset * 1_000 + method_index + scenario_index
                        ),
                    },
                }
            )
        score = {
            "schema_version": "n11-stochastic-validation-score-v1",
            "scenario_id": scenario.scenario_id,
            "seed": scenario.seed,
            "stochastic_plan_sha256": sha256(plan_path),
            "support_applied": scenario.event.value == "stable",
            "contract_ready": True,
            "rows": rows,
            "confirmatory_validation": True,
        }
        score_path = root / "scores" / scenario.scenario_id / "score.json"
        score_path.parent.mkdir(parents=True)
        score_path.write_text(json.dumps(score), encoding="utf-8")
        decision = {
            "generations": [
                {},
                {
                    "telemetry_quality": {
                        "empirical_loss_rate": 0.05 + seed_offset / 1000,
                        "mean_delivery_delay_ns": 10_000_000 + seed_offset * 1000,
                        "latest_age_ns": 20_000_000 + seed_offset * 1000,
                    }
                },
            ]
        }
        decision_path = root / "decisions" / scenario.scenario_id / "redecision.json"
        decision_path.parent.mkdir(parents=True)
        decision_path.write_text(json.dumps(decision), encoding="utf-8")


def test_n11_analysis_reports_seed_distributions_cis_and_variability(tmp_path: Path) -> None:
    _write_fixture(tmp_path)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    first = analyze_n11(
        root=tmp_path,
        scenarios=load_n11_matrix(MATRIX),
        analysis=matrix["analysis"],
    )
    second = analyze_n11(
        root=tmp_path,
        scenarios=load_n11_matrix(MATRIX),
        analysis=matrix["analysis"],
    )
    assert first == second
    assert first["stochastic_validation_success"] is True
    assert first["variability_audit"]["unique_seed_plan_fingerprints"] == 20
    assert first["variability_audit"]["unique_seed_outcome_fingerprints"] == 20
    assert first["variability_audit"]["all_condition_method_outcomes_variable"] is True
    factors = first["stochastic_factor_distributions"]
    assert all(item["variable_across_seeds"] for item in factors.values())
    assert all(len(item["seed_distribution"]) == 20 for item in factors.values())
    primary = first["paired_contrasts"]["S1_STOCHASTIC_ICTC_minus_B3_CAPACITY:utility"]
    assert primary["bootstrap_95_ci_for_mean"][0] > 0
    assert primary["exact_one_sided_seed_block_sign_flip_p"] == pytest.approx(1 / 2**20)
    seed_csv = render_n11_seed_csv(first)
    assert len(seed_csv.splitlines()) == 20 * len(N11MethodId) * 6 + 1
