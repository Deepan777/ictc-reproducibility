"""Tests for the M15 frozen-matrix expansion contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ictc.m15 import load_ns3_matrix, planned_row_count

ROOT = Path(__file__).resolve().parents[2]


def test_final_ns3_matrix_expands_to_exact_frozen_execution_count() -> None:
    matrix = ROOT / "configs" / "experiments" / "m14_final_ns3_matrix.json"
    runs = load_ns3_matrix(matrix)
    assert len(runs) == 120
    assert sum(len(run.condition.actions) for run in runs) == 300
    assert planned_row_count(matrix) == 320
    assert {run.condition.topology_id for run in runs} == {"two-path", "three-path"}
    assert all(run.seed in range(60000, 60020) for run in runs)


def test_final_ns3_matrix_rejects_alternate_in_two_path_topology(tmp_path: Path) -> None:
    source = ROOT / "configs" / "experiments" / "m14_final_ns3_matrix.json"
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["conditions"][0]["actions"].append("alternate")
    target = tmp_path / "broken.json"
    target.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="actions"):
        load_ns3_matrix(target)
