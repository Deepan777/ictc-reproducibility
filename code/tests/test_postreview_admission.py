import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis_tools"))
from offline_admission import admitted, eligible, load, metrics


def group():
    return {
        "utilities": {"route0": 0.8, "route1": 0.9, "route2": 0.75},
        "quality": True,
        "choices": {"Learned": "route1"},
        "margins": {"Learned": 0.05},
        "ensemble_lower": 0.03,
        "history": [{"choice": {"Learned": "route1"}, "margin": {"Learned": 0.04}}] * 4,
    }


def test_metrics_pairing_and_headroom():
    result = metrics([group()], ["route1"])
    assert result["harm"] == 0 and result["retention"] == 1 and result["coverage"] == 1
    assert metrics([group()], ["route2"])["harm"] == 1
    assert metrics([group()], ["route0"])["retention"] == 0


def test_boundary_and_quality():
    g = group()
    assert admitted(g, "Learned", "margin", {"margin": 0.05, "k": 1}) == "route1"
    g["quality"] = False
    assert admitted(g, "Learned", "margin", {"margin": 0, "k": 1}) == "route0"


def test_persistence_and_ensemble():
    g = group()
    assert admitted(g, "Learned", "persistence", {"margin": 0.04, "k": 4}) == "route1"
    assert admitted(g, "Learned", "persistence", {"margin": 0.04, "k": 8}) == "route0"
    assert admitted(g, "Learned", "ensemble", {"margin": 0.031, "k": 1}) == "route0"


def test_eligibility_all_clauses():
    base = {"harm": 0.05, "utility": 0.9}
    stats = {"harm": 0.04, "utility": 0.89, "retention": 0.45}
    assert eligible(stats, base)
    for key, value in [
        ("harm", 0.041),
        ("utility", 0.889),
        ("retention", 0.449),
        ("retention", None),
    ]:
        assert not eligible(dict(stats, **{key: value}), base)


def test_outcome_access_barrier():
    with pytest.raises(PermissionError):
        load("abilene", "test")
