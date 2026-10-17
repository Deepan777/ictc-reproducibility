from __future__ import annotations

import numpy as np

from ictc.evaluation.m16 import (
    binary_comparison,
    bootstrap_mean_ci,
    hodges_lehmann,
    holm_adjust,
)


def test_holm_is_monotone_in_sorted_p_values() -> None:
    adjusted = holm_adjust({"a": 0.01, "b": 0.02, "c": 0.5})
    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.5}


def test_hodges_lehmann_uses_walsh_averages() -> None:
    assert hodges_lehmann(np.array([1.0, 3.0])) == 2.0


def test_binary_comparison_is_paired_and_deterministic() -> None:
    left = {
        "p1": {"false_admission": True},
        "p2": {"false_admission": False},
        "p3": {"false_admission": True},
    }
    right = {
        "p1": {"false_admission": False},
        "p2": {"false_admission": False},
        "p3": {"false_admission": True},
    }
    result = binary_comparison(
        left,
        right,
        "false_admissions",
        seed=17,
        replicates=100,
        label="test",
    )
    assert result["paired_count"] == 3
    assert result["a0_only_count"] == 1
    assert result["comparator_only_count"] == 0


def test_bootstrap_constant_vector_is_exact() -> None:
    assert bootstrap_mean_ci(np.ones(10), seed=7, replicates=100, label="constant") == (1.0, 1.0)
