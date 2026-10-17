"""M08 corpus-matrix validation and deterministic expansion tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from ictc.datasets import CorpusMatrixConfig, DatasetSplit, load_corpus_matrix

ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "configs/experiments/m08_micro_matrix.json"
PILOT = ROOT / "configs/experiments/m08_pilot_matrix.json"


def _raw_matrix() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(MICRO.read_text(encoding="utf-8")))


def test_declared_matrices_have_preassigned_isolated_splits() -> None:
    micro = load_corpus_matrix(MICRO)
    pilot = load_corpus_matrix(PILOT)

    assert micro.pair_count == 12
    assert len(micro.scenarios()) == 12
    assert pilot.pair_count == 1980
    assert len({scenario.run_id for scenario in micro.scenarios()}) == 12
    assert {scenario.split for scenario in micro.scenarios()} == set(DatasetSplit)
    assert all(
        scenario.topology_abstraction == "abstract-two-path-parameter-family"
        for scenario in micro.scenarios()
    )
    assert len(micro.config_hash) == 64
    assert micro.canonical_json() == load_corpus_matrix(MICRO).canonical_json()


@pytest.mark.parametrize(
    ("section", "identifier_key", "message"),
    [
        ("topology_families", "family_id", "duplicate topology family"),
        ("traffic_levels", "traffic_id", "duplicate traffic level"),
        ("observation_regimes", "regime_id", "duplicate observation regime"),
        ("seed_groups", "group_id", "duplicate seed group"),
    ],
)
def test_duplicate_matrix_identifiers_are_rejected(
    section: str, identifier_key: str, message: str
) -> None:
    raw = _raw_matrix()
    items = cast(list[dict[str, Any]], raw[section])
    items[1][identifier_key] = items[0][identifier_key]
    with pytest.raises(ValidationError, match=message):
        CorpusMatrixConfig.model_validate(raw)


def test_family_and_seed_split_coverage_is_required() -> None:
    raw = _raw_matrix()
    families = cast(list[dict[str, Any]], raw["topology_families"])
    families[2]["split"] = "train"
    with pytest.raises(ValidationError, match="topology families must declare"):
        CorpusMatrixConfig.model_validate(raw)

    raw = _raw_matrix()
    groups = cast(list[dict[str, Any]], raw["seed_groups"])
    groups[2]["split"] = "train"
    with pytest.raises(ValidationError, match="seed groups must declare"):
        CorpusMatrixConfig.model_validate(raw)


def test_seed_leakage_and_duplicates_are_rejected() -> None:
    raw = _raw_matrix()
    groups = cast(list[dict[str, Any]], raw["seed_groups"])
    groups[1]["seeds"] = groups[0]["seeds"]
    with pytest.raises(ValidationError, match="appears in multiple splits"):
        CorpusMatrixConfig.model_validate(raw)

    raw = _raw_matrix()
    groups = cast(list[dict[str, Any]], raw["seed_groups"])
    groups[0]["seeds"] = [801, 801]
    with pytest.raises(ValidationError, match="duplicate seeds"):
        CorpusMatrixConfig.model_validate(raw)

    groups[0]["seeds"] = [-1]
    with pytest.raises(ValidationError, match="non-negative"):
        CorpusMatrixConfig.model_validate(raw)


def test_invalid_constant_traffic_and_regime_delay_are_rejected() -> None:
    raw = _raw_matrix()
    traffic = cast(list[dict[str, Any]], raw["traffic_levels"])[0]
    traffic["arrival_mode"] = "CONSTANT"
    with pytest.raises(ValidationError, match="integer packet counts"):
        CorpusMatrixConfig.model_validate(raw)

    raw = _raw_matrix()
    regime = cast(list[dict[str, Any]], raw["observation_regimes"])[0]
    regime["minimum_delay_ticks"] = 2
    regime["maximum_delay_ticks"] = 1
    with pytest.raises(ValidationError, match="maximum telemetry delay"):
        CorpusMatrixConfig.model_validate(raw)


def test_materialized_simulation_preserves_declared_parameters() -> None:
    scenario = load_corpus_matrix(MICRO).scenarios()[0]
    simulation = scenario.simulation

    assert simulation.scenario_id.startswith("ictc-m08-micro-train-balanced-low-complete")
    assert simulation.seed == 8000
    assert simulation.path("path-current").service_packets_per_tick == 7
    assert simulation.cross_process("path-candidate").rate_packets_per_tick == 1.0
