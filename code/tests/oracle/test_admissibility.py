"""Adversarial checks for the independent finite-model causal oracle."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from ictc.oracle.admissibility import (
    CausalAdmissibilityOracle,
    EffectDirection,
    FinitePotentialOutcomeModel,
    OracleQuery,
    OracleReason,
    PotentialOutcomeUnit,
)
from ictc.oracle.benchmark import (
    ambiguous_oracle,
    build_n01_evidence,
    identified_oracle,
    render_n01_evidence,
    routing_query,
)
from ictc.oracle.boundary import link_post_decision_labels, seal_persisted_decisions

ROOT = Path(__file__).resolve().parents[2]


def test_observational_twins_with_opposite_effects_fail_closed() -> None:
    oracle = ambiguous_oracle()
    left, right = oracle.models

    assert left.observational_distribution() == right.observational_distribution()
    assert {left.effect(), right.effect()} == {Fraction(-10), Fraction(10)}

    result = oracle.evaluate(left.observational_distribution())
    assert result.causally_admissible is False
    assert result.reason is OracleReason.NON_IDENTIFIABLE_EFFECT_DISAGREEMENT
    assert result.effect_direction is EffectDirection.UNRESOLVED
    assert result.identified_effect is None
    assert set(result.witness_model_ids or ()) == {left.model_id, right.model_id}


def test_bounded_model_class_can_identify_zero_effect_without_claiming_benefit() -> None:
    oracle = identified_oracle()
    left, right = oracle.models

    assert left.to_payload() != right.to_payload()
    assert left.observational_distribution() == right.observational_distribution()
    assert left.effect() == right.effect() == 0

    result = oracle.evaluate(left.observational_distribution())
    assert result.causally_admissible is True
    assert result.reason is OracleReason.IDENTIFIED_UNIQUE_EFFECT
    assert result.effect_direction is EffectDirection.ZERO
    assert result.identified_effect == 0
    assert result.witness_model_ids is None


def test_identified_negative_effect_remains_identified_but_not_beneficial() -> None:
    base = ambiguous_oracle().models[1]
    oracle = CausalAdmissibilityOracle("negative-singleton", base.query, (base,))

    result = oracle.evaluate(base.observational_distribution())
    assert result.causally_admissible is True
    assert result.effect_direction is EffectDirection.NEGATIVE
    assert result.identified_effect == -10

    positive = ambiguous_oracle().models[0]
    positive_result = CausalAdmissibilityOracle(
        "positive-singleton", positive.query, (positive,)
    ).evaluate(positive.observational_distribution())
    assert positive_result.effect_direction is EffectDirection.POSITIVE
    assert positive_result.identified_effect == 10


def test_observation_outside_declared_class_fails_closed() -> None:
    oracle = ambiguous_oracle()
    outside = (((0, Fraction(99)), Fraction(1)),)

    result = oracle.evaluate(outside)
    assert result.causally_admissible is False
    assert result.reason is OracleReason.OBSERVATION_OUTSIDE_MODEL_CLASS
    assert result.compatible_effects == ()
    assert result.witness_model_ids is None


def test_certificate_and_evidence_are_deterministic() -> None:
    oracle = ambiguous_oracle()
    observation = oracle.models[0].observational_distribution()
    first = oracle.evaluate(observation)
    second = oracle.evaluate(observation)

    assert first.to_payload() == second.to_payload()
    assert first.certificate_digest == second.certificate_digest
    assert render_n01_evidence() == render_n01_evidence()
    assert json.loads(render_n01_evidence()) == build_n01_evidence()


def test_decisions_are_sealed_before_complete_oracle_labels_are_linked() -> None:
    decisions = b'{"pair_id":"pair-b","action":"hold"}\n{"pair_id":"pair-a","action":"reroute"}\n'
    seal = seal_persisted_decisions(decisions)
    oracle = ambiguous_oracle()
    certificate = oracle.evaluate(oracle.models[0].observational_distribution())

    linked = link_post_decision_labels(
        decisions,
        seal,
        {"pair-a": certificate, "pair-b": certificate},
    )
    assert seal.pair_ids == ("pair-a", "pair-b")
    assert linked.decision_seal.sha256 == seal.sha256
    assert [pair_id for pair_id, _ in linked.labels] == ["pair-a", "pair-b"]


def test_post_decision_boundary_rejects_mutation_missing_labels_and_truth_leakage() -> None:
    decisions = b'{"pair_id":"pair-a","action":"hold"}\n'
    seal = seal_persisted_decisions(decisions)
    oracle = ambiguous_oracle()
    certificate = oracle.evaluate(oracle.models[0].observational_distribution())

    with pytest.raises(ValueError, match="changed after"):
        link_post_decision_labels(
            b'{"pair_id":"pair-a","action":"reroute"}\n',
            seal,
            {"pair-a": certificate},
        )
    with pytest.raises(ValueError, match="match the sealed"):
        link_post_decision_labels(decisions, seal, {})
    with pytest.raises(ValueError, match="oracle-only truth"):
        seal_persisted_decisions(b'{"pair_id":"pair-a","oracle_reason":"identified"}\n')


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"", "must not be empty"),
        (b"not-json\n", "not valid JSON"),
        (b"[]\n", "must be a JSON object"),
        (b'{"action":"hold"}\n', "non-empty pair_id"),
        (b'{"pair_id":"a"}\n{"pair_id":"a"}\n', "must be unique"),
        (b"\n\n", "contains no decisions"),
        (
            b'{"pair_id":"a","nested":[{"compatible_effects":[]}]}\n',
            "oracle-only truth",
        ),
    ],
)
def test_decision_seal_rejects_malformed_or_leaky_artifacts(content: bytes, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        seal_persisted_decisions(content)


def test_evidence_contains_the_byte_sealed_post_decision_linkage() -> None:
    boundary = build_n01_evidence()["post_decision_boundary"]
    assert isinstance(boundary, dict)
    assert boundary["controller_exposure"] == "none; labels linked after the decision-byte seal"
    seal = boundary["decision_seal"]
    assert isinstance(seal, dict)
    byte_count = seal["byte_count"]
    assert isinstance(byte_count, int)
    assert byte_count > 0
    labels = boundary["labels"]
    assert isinstance(labels, list)
    assert len(labels) == 2


def test_tracked_n01_evidence_matches_fresh_rendering() -> None:
    evidence = ROOT / "docs/milestones/evidence/n01_independent_oracle.json"
    assert evidence.read_text(encoding="utf-8") == render_n01_evidence()


def test_oracle_package_has_no_controller_or_reference_contract_imports() -> None:
    oracle_root = ROOT / "src/ictc/oracle"
    forbidden = (
        "ictc.controller",
        "ictc.contract",
        "ictc.pilot",
        "m14_scoring",
        "reference_admissible",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(oracle_root.glob("*.py"))
    )
    for token in forbidden:
        assert token not in source


@pytest.mark.parametrize(
    "distribution",
    [
        (),
        (((2, Fraction(1)), Fraction(1)),),
        (((0, Fraction(1)), Fraction(0)), ((1, Fraction(2)), Fraction(1))),
        (((0, Fraction(1)), Fraction(1, 2)),),
        (((0, Fraction(1)), Fraction(1, 2)), ((0, Fraction(1)), Fraction(1, 2))),
        (((1, Fraction(1)), Fraction(1, 2)), ((0, Fraction(1)), Fraction(1, 2))),
    ],
)
def test_malformed_observational_distributions_are_rejected(
    distribution: tuple[tuple[tuple[int, Fraction], Fraction], ...],
) -> None:
    with pytest.raises(ValueError):
        ambiguous_oracle().evaluate(distribution)


def test_invalid_query_units_models_and_oracles_are_rejected() -> None:
    with pytest.raises(ValueError, match="query fields"):
        OracleQuery("", "zero", "one", "Y", "unit")
    with pytest.raises(ValueError, match="unit ID"):
        PotentialOutcomeUnit("", Fraction(1), 0, Fraction(0), Fraction(1))
    with pytest.raises(ValueError, match="positive"):
        PotentialOutcomeUnit("u", Fraction(0), 0, Fraction(0), Fraction(1))
    with pytest.raises(ValueError, match="binary"):
        PotentialOutcomeUnit("u", Fraction(1), 3, Fraction(0), Fraction(1))

    unit = PotentialOutcomeUnit("u", Fraction(1), 0, Fraction(0), Fraction(1))
    query = routing_query()
    with pytest.raises(ValueError, match="model ID"):
        FinitePotentialOutcomeModel("", query, (unit,))
    with pytest.raises(ValueError, match="at least one"):
        FinitePotentialOutcomeModel("empty", query, ())
    with pytest.raises(ValueError, match="unique"):
        FinitePotentialOutcomeModel("duplicate", query, (unit, unit))
    half = PotentialOutcomeUnit("half", Fraction(1, 2), 0, Fraction(0), Fraction(1))
    with pytest.raises(ValueError, match="sum exactly"):
        FinitePotentialOutcomeModel("bad-mass", query, (half,))

    model = FinitePotentialOutcomeModel("valid", query, (unit,))
    with pytest.raises(ValueError, match="oracle ID"):
        CausalAdmissibilityOracle("", query, (model,))
    with pytest.raises(ValueError, match="must not be empty"):
        CausalAdmissibilityOracle("empty", query, ())
    with pytest.raises(ValueError, match="unique"):
        CausalAdmissibilityOracle("duplicate", query, (model, model))
    other_query = OracleQuery("other", "zero", "one", "Y", "unit")
    with pytest.raises(ValueError, match="declared query"):
        CausalAdmissibilityOracle("mismatch", other_query, (model,))
