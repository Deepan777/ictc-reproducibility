"""Executable scientific checks for the M02 causal counterexamples."""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from ictc.causal.counterexamples import (
    CounterexamplePair,
    FiniteSCM,
    PotentialOutcomeRow,
    build_m02_evidence,
    hidden_cross_traffic_pair,
    observational_equivalence_pair,
    render_m02_evidence,
)

ROOT = Path(__file__).resolve().parents[2]


def test_generic_models_are_observationally_equal_with_opposite_effects() -> None:
    pair = observational_equivalence_pair()

    assert pair.observationally_equivalent
    assert pair.opposite_effect_signs
    assert pair.model_left.average_treatment_effect() == Fraction(1, 2)
    assert pair.model_right.average_treatment_effect() == Fraction(-1, 2)
    assert pair.model_left.observational_distribution() == (
        ((0, Fraction(0)), Fraction(1, 4)),
        ((0, Fraction(1)), Fraction(1, 4)),
        ((1, Fraction(0)), Fraction(1, 4)),
        ((1, Fraction(1)), Fraction(1, 4)),
    )
    assert pair.model_left.observed_correlation() == pytest.approx(0.0)
    assert pair.model_left.modal_exact_prediction_accuracy() == Fraction(1, 2)


def test_hidden_cross_traffic_gives_perfect_prediction_but_ambiguous_effect() -> None:
    pair = hidden_cross_traffic_pair()

    assert pair.observationally_equivalent
    assert pair.opposite_effect_signs
    assert pair.model_left.observational_distribution() == (
        ((0, Fraction(50)), Fraction(1, 2)),
        ((1, Fraction(70)), Fraction(1, 2)),
    )
    assert pair.model_left.conditional_mean_mse() == 0
    assert pair.model_left.modal_exact_prediction_accuracy() == 1
    assert pair.model_left.observed_correlation() == pytest.approx(1.0)
    assert pair.model_left.average_treatment_effect() == -10
    assert pair.model_right.average_treatment_effect() == 10


def test_pair_payload_validates_and_retains_evidence_obligations() -> None:
    pair = hidden_cross_traffic_pair()
    pair.validate_witness()
    payload = pair.to_payload()

    assert payload["observationally_equivalent"] is True
    assert payload["opposite_effect_signs"] is True
    obligations = payload["evidence_obligations"]
    assert isinstance(obligations, list)
    assert any("action support" in str(obligation) for obligation in obligations)
    assert "not sufficient" in pair.validity_scope


def test_complete_evidence_is_deterministic_and_narrowly_scoped() -> None:
    first = render_m02_evidence()
    second = render_m02_evidence()
    payload = json.loads(first)

    assert first == second
    assert payload == build_m02_evidence()
    assert payload["random_sampling_used"] is False
    assert len(payload["counterexamples"]) == 2
    assert "does not establish empirical ICTC performance" in payload["claim_boundary"]


def test_tracked_evidence_matches_fresh_rendering() -> None:
    evidence = ROOT / "docs/milestones/evidence/m02_counterexamples.json"
    assert evidence.read_text(encoding="utf-8") == render_m02_evidence()


@pytest.mark.parametrize(
    ("latent_state", "probability", "action", "message"),
    [
        ("", Fraction(1), 0, "named"),
        ("u", Fraction(0), 0, "positive"),
        ("u", Fraction(1), 2, "binary"),
    ],
)
def test_potential_outcome_row_rejects_invalid_state(
    latent_state: str, probability: Fraction, action: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PotentialOutcomeRow(latent_state, probability, action, Fraction(0), Fraction(1))


def test_finite_scm_rejects_invalid_enumerations() -> None:
    row = PotentialOutcomeRow("u", Fraction(1), 0, Fraction(0), Fraction(1))
    with pytest.raises(ValueError, match="model ID"):
        FiniteSCM("", "zero", "one", "Y", "unit", (row,))
    with pytest.raises(ValueError, match="at least one"):
        FiniteSCM("empty", "zero", "one", "Y", "unit", ())
    with pytest.raises(ValueError, match="unique"):
        FiniteSCM("duplicate", "zero", "one", "Y", "unit", (row, row))
    with pytest.raises(ValueError, match="sum exactly"):
        FiniteSCM(
            "bad-mass",
            "zero",
            "one",
            "Y",
            "unit",
            (PotentialOutcomeRow("u", Fraction(1, 2), 0, Fraction(0), Fraction(1)),),
        )


def test_binary_action_validation_and_constant_outcome_metric() -> None:
    row = PotentialOutcomeRow("u", Fraction(1), 0, Fraction(3), Fraction(3))
    model = FiniteSCM("constant", "zero", "one", "Y", "unit", (row,))

    with pytest.raises(ValueError, match="binary"):
        row.outcome(5)
    with pytest.raises(ValueError, match="binary"):
        model.interventional_mean(-1)
    assert model.average_treatment_effect() == 0
    assert model.observed_correlation() is None


def test_pair_validation_rejects_a_non_witness() -> None:
    pair = observational_equivalence_pair()
    invalid = CounterexamplePair(
        counterexample_id="invalid",
        title="invalid",
        statement="invalid",
        model_left=pair.model_left,
        model_right=hidden_cross_traffic_pair().model_left,
        evidence_obligations=("none",),
        validity_scope="test only",
    )
    with pytest.raises(ValueError, match="observationally distinct"):
        invalid.validate_witness()

    zero_effect = FiniteSCM(
        "zero-effect",
        "A=0",
        "A=1",
        "binary_utility",
        "indicator",
        (
            PotentialOutcomeRow("u0", Fraction(1, 2), 0, Fraction(0), Fraction(0)),
            PotentialOutcomeRow("u1", Fraction(1, 2), 1, Fraction(1), Fraction(1)),
        ),
    )
    no_opposite_sign = CounterexamplePair(
        counterexample_id="zero",
        title="zero",
        statement="zero",
        model_left=zero_effect,
        model_right=zero_effect,
        evidence_obligations=("none",),
        validity_scope="test only",
    )
    with pytest.raises(ValueError, match="opposite signs"):
        no_opposite_sign.validate_witness()
