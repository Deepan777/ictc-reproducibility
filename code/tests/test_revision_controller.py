from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

from ictc.ns3.n12 import N12Scenario
from ictc.ns3.n13 import RISK_FEATURE_NAMES
from revision_controller import (
    admit,
    context,
    context_from_rows,
    received_rows,
    risk_features,
    route_features,
)


def scenario() -> N12Scenario:
    return N12Scenario("test", "development", "web", "stable", 0, 100000, 1, 8.0, 3.5, 2.2)


def records() -> list[dict[str, str]]:
    result = []
    for sample in range(400_000_000, 3_500_000_000, 100_000_000):
        for route in range(3):
            result.append(
                {
                    "sample_timestamp_ns": str(sample),
                    "available_timestamp_ns": str(sample + 10_000_000),
                    "phase": "PRE_DECISION",
                    "route_id": str(route),
                    "bottleneck_capacity_bps": "25000000",
                    "propagation_delay_ms": "10",
                    "queue_packets": "0",
                    "link_up": "1",
                }
            )
    return result


def model() -> dict[str, object]:
    return {
        "feature_names": list(RISK_FEATURE_NAMES),
        "means": [0.0] * 23,
        "scales": [1.0] * 23,
        "coefficients": [0.0] * 23,
        "intercept": -10.0,
    }


def test_no_simulator_audit_or_plan_needed(tmp_path: Path) -> None:
    rows = records()
    with (tmp_path / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    first = context(tmp_path, scenario())
    (tmp_path / "plan.json").write_text("NOT A CONTROLLER INPUT")
    (tmp_path / "telemetry_delivery.csv").write_text("INVALID UNAVAILABLE DROP FLAGS")
    assert context(tmp_path, scenario()) == first


def test_reports_received_at_decision_are_excluded(tmp_path: Path) -> None:
    rows = records()
    rows.append({**rows[-1], "available_timestamp_ns": "3500000000", "queue_packets": "999999"})
    with (tmp_path / "telemetry.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert all(
        int(r["available_timestamp_ns"]) < 3500000000 for r in received_rows(tmp_path, 3500000000)
    )


def test_only_expired_missing_reports_count_as_loss() -> None:
    rows = [r for r in records() if int(r["sample_timestamp_ns"]) != 3400000000]
    ctx = context_from_rows(rows, 3500000000)
    assert ctx["snapshot"]["empirical_loss_rate"] == 0.0
    rows = [r for r in rows if int(r["sample_timestamp_ns"]) != 500000000]
    assert context_from_rows(rows, 3500000000)["snapshot"]["empirical_loss_rate"] > 0.0


def test_quality_gate_cannot_be_overridden_by_low_risk() -> None:
    ctx = context_from_rows(records(), 3500000000, budget=8)
    assert not ctx["quality_ready"]
    assert admit(scenario(), ctx, "route1", model(), 0.30) == "route0"


def test_event_labels_never_change_feature_vector() -> None:
    ctx = context_from_rows(records(), 3500000000)
    original = risk_features(scenario(), ctx, "route1")
    changed = risk_features(replace(scenario(), event="failure"), ctx, "route1")
    assert len(original) == 23 and changed == original


def test_learning_representation_has_no_route_index() -> None:
    ctx = context_from_rows(records(), 3500000000)
    assert route_features(scenario(), ctx, "route1") == route_features(scenario(), ctx, "route2")


def test_prospectively_declared_partitions_do_not_overlap() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol = json.loads((root / "configs/revision_protocol.json").read_text())
    sets = [set(protocol[f"{phase}_offsets"]) for phase in ("development", "validation", "test")]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(sets[2]) == 24
    for base in protocol["seed_bases"].values():
        assert not ({base + x for s in sets for x in s} & set(range(86000, 90020)))
