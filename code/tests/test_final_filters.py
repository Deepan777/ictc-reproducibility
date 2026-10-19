"""Specified admission boundaries without outcome inputs."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis_tools/admission_filters"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from filters import filter_action, greedy_score


def example():
    return {
        "quality_ready": True,
        "proposers": {
            "Learned": {
                "proposal": "route1",
                "margin": 0.02,
                "history_choices": ["route1", "route1"],
                "history_margins": [0.03, 0.04],
                "tree_differences": [0.01] * 64,
            }
        },
    }


def test_margin_boundary_and_quality_conjunction():
    row = example()
    assert filter_action(row, "Learned", "Margin", {"tau": 0.02}, quality=True) == "route1"
    row["quality_ready"] = False
    assert filter_action(row, "Learned", "Margin", {"tau": 0.02}, quality=True) == "route0"
    assert filter_action(row, "Learned", "Margin", {"tau": 0.02}, quality=False) == "route1"


def test_persistence_insufficient_or_disagreeing_history():
    row = example()
    assert (
        filter_action(row, "Learned", "Persistence", {"k": 2, "m": 0.03}, quality=True) == "route1"
    )
    assert filter_action(row, "Learned", "Persistence", {"k": 3, "m": 0}, quality=True) == "route0"
    row["proposers"]["Learned"]["history_choices"][0] = "route0"
    assert filter_action(row, "Learned", "Persistence", {"k": 2, "m": 0}, quality=True) == "route0"


def test_ensemble_and_finite_incumbent_score():
    assert (
        filter_action(example(), "Learned", "Ensemble", {"q": 0.1, "tau": 0.01}, quality=True)
        == "route1"
    )
    assert greedy_score(
        {
            "bottleneck_capacity_bps": 2e6,
            "queue_packets": 2,
            "propagation_delay_ms": 10,
            "link_up": False,
        }
    ) == pytest.approx(0.2)
