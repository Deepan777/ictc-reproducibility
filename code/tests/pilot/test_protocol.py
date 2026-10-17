"""Frozen M10 metric/statistical protocol completeness tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from ictc.pilot import load_frozen_protocol
from ictc.pilot.protocol import MASTER_METRICS

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/m10_pilot_protocol.json"
AMENDMENT = ROOT / "configs/experiments/m10_pilot_protocol_amendment_001.json"


def test_protocol_classifies_every_master_metric_and_loads_thresholds() -> None:
    frozen = load_frozen_protocol(PROTOCOL)
    metrics = cast(list[dict[str, object]], frozen.payload["metrics"])

    assert frozen.protocol_id == "m10-controlled-pilot-v1"
    assert len(frozen.configuration_digest) == 64
    assert [item["metric_id"] for item in metrics] == list(MASTER_METRICS)
    assert frozen.operational.acquisition_deadline_ticks == 5
    assert frozen.operational.maximum_deadline_violation_rate == 0.05
    assert len(frozen.operational.uncertainty_width_grid_ms) == 5
    assert len(frozen.operational.deadline_budget_grid) == 9


def test_protocol_rejects_missing_metric_and_test_authorization(tmp_path: Path) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["metrics"] = payload["metrics"][:-1]
    missing = tmp_path / "missing.json"
    missing.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="every master-prompt metric"):
        load_frozen_protocol(missing)

    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload["authorized_splits"].append("test")
    leaked = tmp_path / "leaked.json"
    leaked.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="only train and development"):
        load_frozen_protocol(leaked)


def test_protocol_amendment_preserves_base_and_repairs_ate_unit() -> None:
    base = load_frozen_protocol(PROTOCOL)
    amended = load_frozen_protocol(AMENDMENT)
    metrics = cast(list[dict[str, object]], amended.payload["metrics"])
    coverage = next(item for item in metrics if item["metric_id"] == "confidence_interval_coverage")

    assert amended.protocol_id == "m10-controlled-pilot-v1.1"
    assert amended.configuration_digest != base.configuration_digest
    assert "split/topology-family" in cast(str, coverage["denominator"])
    assert "individual" in cast(str, coverage["reason"])
