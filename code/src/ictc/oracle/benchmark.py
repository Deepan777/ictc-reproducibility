"""Deterministic N01 benchmark cases for the independent admissibility oracle."""

from __future__ import annotations

import json
from fractions import Fraction

from ictc.oracle.admissibility import (
    CausalAdmissibilityOracle,
    FinitePotentialOutcomeModel,
    JsonValue,
    OracleQuery,
    PotentialOutcomeUnit,
)
from ictc.oracle.boundary import link_post_decision_labels, seal_persisted_decisions


def routing_query() -> OracleQuery:
    return OracleQuery(
        query_id="reroute_effect_on_goodput_v1",
        action_zero_label="retain_primary_path",
        action_one_label="reroute_to_backup_path",
        outcome_name="post_action_goodput",
        outcome_unit="Mbps",
    )


def _model(
    model_id: str,
    *,
    outcome_one_for_natural_zero: int,
    outcome_zero_for_natural_one: int,
) -> FinitePotentialOutcomeModel:
    return FinitePotentialOutcomeModel(
        model_id=model_id,
        query=routing_query(),
        units=(
            PotentialOutcomeUnit(
                "low_cross_traffic",
                Fraction(1, 2),
                0,
                Fraction(50),
                Fraction(outcome_one_for_natural_zero),
            ),
            PotentialOutcomeUnit(
                "high_cross_traffic",
                Fraction(1, 2),
                1,
                Fraction(outcome_zero_for_natural_one),
                Fraction(70),
            ),
        ),
    )


def ambiguous_oracle() -> CausalAdmissibilityOracle:
    """Two observationally equal models with -10 and +10 Mbps effects."""
    return CausalAdmissibilityOracle(
        oracle_id="n01-hidden-cross-traffic-ambiguous-v1",
        query=routing_query(),
        models=(
            _model(
                "beneficial-observational-twin",
                outcome_one_for_natural_zero=60,
                outcome_zero_for_natural_one=60,
            ),
            _model(
                "harmful-observational-twin",
                outcome_one_for_natural_zero=40,
                outcome_zero_for_natural_one=80,
            ),
        ),
    )


def identified_oracle() -> CausalAdmissibilityOracle:
    """A bounded class with distinct counterfactuals but a uniquely zero effect."""
    return CausalAdmissibilityOracle(
        oracle_id="n01-bounded-zero-effect-v1",
        query=routing_query(),
        models=(
            _model(
                "zero-effect-flat-counterfactuals",
                outcome_one_for_natural_zero=50,
                outcome_zero_for_natural_one=70,
            ),
            _model(
                "zero-effect-offset-counterfactuals",
                outcome_one_for_natural_zero=40,
                outcome_zero_for_natural_one=60,
            ),
        ),
    )


def build_n01_evidence() -> dict[str, JsonValue]:
    ambiguous = ambiguous_oracle()
    ambiguous_observation = ambiguous.models[0].observational_distribution()
    ambiguous_certificate = ambiguous.evaluate(ambiguous_observation)
    identified = identified_oracle()
    identified_observation = identified.models[0].observational_distribution()
    identified_certificate = identified.evaluate(identified_observation)
    persisted_decisions = (
        b'{"pair_id":"ambiguous-case","action":"reroute"}\n'
        b'{"pair_id":"identified-case","action":"hold"}\n'
    )
    decision_seal = seal_persisted_decisions(persisted_decisions)
    post_decision_labels = link_post_decision_labels(
        persisted_decisions,
        decision_seal,
        {
            "ambiguous-case": ambiguous_certificate,
            "identified-case": identified_certificate,
        },
    )
    return {
        "schema_version": "n01-independent-oracle-evidence-v1",
        "milestone": "N01",
        "oracle_boundary": {
            "controller_inputs_used": False,
            "a0_reference_flags_used": False,
            "contract_reason_codes_used": False,
            "arithmetic": "exact fractions",
            "label_semantics": ("All compatible finite models agree on E[Y(1)-Y(0)]."),
        },
        "ambiguous_case": {
            "model_effects": [
                {
                    "model_id": model.model_id,
                    "effect": {
                        "numerator": model.effect().numerator,
                        "denominator": model.effect().denominator,
                    },
                }
                for model in ambiguous.models
            ],
            "observationally_equal": (
                ambiguous.models[0].observational_distribution()
                == ambiguous.models[1].observational_distribution()
            ),
            "certificate": ambiguous_certificate.to_payload(),
        },
        "identified_case": {
            "counterfactual_models_distinct": (
                identified.models[0].to_payload() != identified.models[1].to_payload()
            ),
            "certificate": identified_certificate.to_payload(),
        },
        "post_decision_boundary": post_decision_labels.to_payload(),
        "random_sampling_used": False,
        "claim_boundary": (
            "This benchmark validates an independent exact finite-model oracle. It does not "
            "retroactively validate M16, prove a real network SCM, or establish publication "
            "novelty."
        ),
    }


def render_n01_evidence() -> str:
    return json.dumps(build_n01_evidence(), indent=2, sort_keys=True) + "\n"
