"""Tests for outcome-separated N09 contextual-support design compilation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ictc.ns3.n08 import load_n08_matrix
from ictc.ns3.n09_design import compile_n09_support_design

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n08_post_change_development.json"
SUPPORTED = {
    "two-balanced|light|stable",
    "two-balanced|saturated|stable",
}


def _fixture_root(tmp_path: Path) -> Path:
    root = tmp_path / "n08"
    (root / "n08_manifest.json").parent.mkdir(parents=True)
    (root / "n08_manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
    (root / "outcomes").mkdir()
    (root / "outcomes" / "EXPERIMENT_REGISTRY.csv").write_text(
        "scenario_id,status\n", encoding="utf-8"
    )
    for scenario in load_n08_matrix(MATRIX):
        context = "|".join(
            (scenario.topology_family.value, scenario.workload.value, scenario.event.value)
        )
        score = root / "scores" / scenario.scenario_id / "score.json"
        score.parent.mkdir(parents=True)
        score.write_text(
            json.dumps(
                {
                    "scenario_id": scenario.scenario_id,
                    "development_only": True,
                    "rows": [
                        {
                            "method_id": "B1_REROUTE",
                            "selected_action": "reroute",
                            "physical_label": "BENEFICIAL" if context in SUPPORTED else "HARMFUL",
                        },
                        {
                            "method_id": "B3_CAPACITY",
                            "selected_action": "keep",
                            "physical_label": "KEEP",
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
    return root


def test_design_selects_only_exact_four_seed_supported_contexts(tmp_path: Path) -> None:
    design = compile_n09_support_design(n08_root=_fixture_root(tmp_path), matrix_path=MATRIX)
    assert set(design["frozen_design"]["supported_contexts"]) == SUPPORTED
    assert design["frozen_design"]["fresh_validation_seeds"] == list(range(81100, 81108))
    assert design["frozen_design"]["reserved_confirmatory_seeds"] == list(range(82000, 82020))
    assert len(design["frozen_design_digest"]) == 64
    assert "not confirmatory" in design["selection_warning"]


def test_design_rejects_threshold_and_score_tampering(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    with pytest.raises(ValueError, match="four-seed"):
        compile_n09_support_design(n08_root=root, matrix_path=MATRIX, minimum_seeds=3)
    scenario = load_n08_matrix(MATRIX)[0]
    score = root / "scores" / scenario.scenario_id / "score.json"
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["development_only"] = False
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not finalized"):
        compile_n09_support_design(n08_root=root, matrix_path=MATRIX)


def test_design_rejects_non_object_score(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    scenario = load_n08_matrix(MATRIX)[0]
    (root / "scores" / scenario.scenario_id / "score.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        compile_n09_support_design(n08_root=root, matrix_path=MATRIX)


def test_design_rejects_non_list_score_rows(tmp_path: Path) -> None:
    root = _fixture_root(tmp_path)
    scenario = load_n08_matrix(MATRIX)[0]
    score = root / "scores" / scenario.scenario_id / "score.json"
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["rows"] = None
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="score rows are invalid"):
        compile_n09_support_design(n08_root=root, matrix_path=MATRIX)
