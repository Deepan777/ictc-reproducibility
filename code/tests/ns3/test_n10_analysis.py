"""Known-answer tests for the frozen N10 seed-blocked analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n09 import N09MethodId
from ictc.ns3.n10 import load_n10_matrix
from ictc.ns3.n10_analysis import _effect, _object, _quantile, _wilson, analyze_n10

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n10_confirmatory_support.json"
METHODS = tuple(method.value for method in N09MethodId)
D1 = N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value


def _fixture_scores(root: Path, *, harmful_support: bool = False) -> None:
    scenarios = load_n10_matrix(MATRIX)
    support_index = 0
    for scenario in scenarios:
        supported = (
            scenario.topology_family.value == "two-balanced"
            and scenario.workload.value in {"light", "saturated"}
            and scenario.event.value == "stable"
        )
        d1_label = "BENEFICIAL"
        if supported:
            support_index += 1
            if harmful_support and support_index == 1:
                d1_label = "HARMFUL"
        payload = {
            "schema_version": "n10-confirmatory-score-v1",
            "scenario_id": scenario.scenario_id,
            "confirmatory": True,
            "support_applied": supported,
            "rows": [
                {
                    "method_id": method,
                    "physical_label": d1_label if method == D1 else "KEEP",
                }
                for method in METHODS
            ],
        }
        target = root / "scores" / scenario.scenario_id / "score.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload), encoding="utf-8")
    assert support_index == 40


def _analyze(root: Path) -> dict[str, Any]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    return analyze_n10(
        root=root,
        scenarios=load_n10_matrix(MATRIX),
        analysis=matrix["analysis"],
    )


def test_known_answer_analysis_passes_all_frozen_gates_deterministically(tmp_path: Path) -> None:
    _fixture_scores(tmp_path)
    first = _analyze(tmp_path)
    second = _analyze(tmp_path)
    assert first == second
    assert first["independent_seed_blocks"] == 20
    assert first["complete_factorial_cells"] == 720
    assert first["confirmatory_success"] is True
    assert first["support_opportunities"] == {
        "total": 40,
        "beneficial": 40,
        "harmful": 0,
        "other": 0,
        "beneficial_wilson_95_ci": [0.912378399, 1.0],
    }
    primary = first["paired_contrasts"][first["primary_contrast"]]
    assert primary["bootstrap_95_ci"][0] > 0
    assert primary["exact_one_sided_sign_flip_p"] == pytest.approx(1 / 2**20)
    assert primary["holm_adjusted_p"] == pytest.approx(5 / 2**20)
    assert primary["effect_size"] == {
        "paired_d_z": "INFINITE",
        "rank_biserial": 1.0,
    }
    b3 = first["method_results"][N09MethodId.B3_CAPACITY.value]
    assert b3["harmful_nonkeep_wilson_95_ci"] is None


def test_harmful_supported_decision_fails_confirmatory_gate(tmp_path: Path) -> None:
    _fixture_scores(tmp_path, harmful_support=True)
    result = _analyze(tmp_path)
    assert result["confirmatory_success"] is False
    assert result["support_opportunities"]["harmful"] == 1
    assert result["method_results"][D1]["counts"]["HARMFUL"] == 1


def test_analysis_rejects_incomplete_or_unbound_scores(tmp_path: Path) -> None:
    _fixture_scores(tmp_path)
    scenario = load_n10_matrix(MATRIX)[0]
    score = tmp_path / "scores" / scenario.scenario_id / "score.json"
    score.unlink()
    with pytest.raises(FileNotFoundError):
        _analyze(tmp_path)

    _fixture_scores(tmp_path)
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["confirmatory"] = False
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="bound confirmatory"):
        _analyze(tmp_path)


def test_analysis_rejects_any_change_to_frozen_statistics(tmp_path: Path) -> None:
    _fixture_scores(tmp_path)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["analysis"]["bootstrap_resamples"] = 9999
    with pytest.raises(ValueError, match="differs from the freeze"):
        analyze_n10(
            root=tmp_path,
            scenarios=load_n10_matrix(MATRIX),
            analysis=matrix["analysis"],
        )


def test_analysis_rejects_matrix_rows_labels_and_support_defects(tmp_path: Path) -> None:
    _fixture_scores(tmp_path)
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    scenarios = load_n10_matrix(MATRIX)
    with pytest.raises(ValueError, match="exact complete confirmatory matrix"):
        analyze_n10(root=tmp_path, scenarios=scenarios[:-1], analysis=matrix["analysis"])

    score = tmp_path / "scores" / scenarios[0].scenario_id / "score.json"
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["rows"] = []
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="method rows are incomplete"):
        _analyze(tmp_path)

    _fixture_scores(tmp_path)
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["rows"][0]["physical_label"] = "UNKNOWN"
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown physical label"):
        _analyze(tmp_path)

    _fixture_scores(tmp_path)
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["support_applied"] = False
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="support opportunities"):
        _analyze(tmp_path)


def test_analysis_helper_boundaries(tmp_path: Path) -> None:
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        _object(non_object)
    assert _quantile([2.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="Wilson interval inputs"):
        _wilson(1, 0)
    assert _effect([-1.0, -1.0])["paired_d_z"] == "-INFINITE"
    assert _effect([0.0, 0.0])["paired_d_z"] == "UNDEFINED"
