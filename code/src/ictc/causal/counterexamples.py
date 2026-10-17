"""Deterministic finite-SCM counterexamples for the M02 scientific gate.

These constructions demonstrate non-identifiability in narrowly declared observational
regimes. They do not implement the general graph algorithms reserved for M04.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type DistributionKey = tuple[int, Fraction]
type Distribution = tuple[tuple[DistributionKey, Fraction], ...]


def _fraction_payload(value: Fraction) -> dict[str, JsonValue]:
    """Serialize an exact rational value together with its decimal rendering."""
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "decimal": float(value),
    }


def _effect_sign(value: Fraction) -> int:
    """Return the sign of an exact effect."""
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


@dataclass(frozen=True, slots=True)
class PotentialOutcomeRow:
    """One latent state and its two potential outcomes."""

    latent_state: str
    probability: Fraction
    natural_action: int
    outcome_if_zero: Fraction
    outcome_if_one: Fraction

    def __post_init__(self) -> None:
        if not self.latent_state:
            raise ValueError("latent state must be named")
        if self.probability <= 0:
            raise ValueError("latent-state probability must be positive")
        if self.natural_action not in {0, 1}:
            raise ValueError("natural action must be binary")

    def outcome(self, action: int) -> Fraction:
        """Return the potential outcome under a binary intervention."""
        if action == 0:
            return self.outcome_if_zero
        if action == 1:
            return self.outcome_if_one
        raise ValueError("intervention action must be binary")


@dataclass(frozen=True, slots=True)
class FiniteSCM:
    """A fully enumerable structural causal model with binary treatment."""

    model_id: str
    action_zero_label: str
    action_one_label: str
    outcome_name: str
    outcome_unit: str
    rows: tuple[PotentialOutcomeRow, ...]

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model ID must be non-empty")
        if not self.rows:
            raise ValueError("finite SCM requires at least one latent state")
        states = [row.latent_state for row in self.rows]
        if len(states) != len(set(states)):
            raise ValueError("latent-state names must be unique")
        if sum((row.probability for row in self.rows), start=Fraction()) != 1:
            raise ValueError("latent-state probabilities must sum exactly to one")

    def observational_distribution(self) -> Distribution:
        """Enumerate the observed joint distribution P(A, Y)."""
        masses: dict[DistributionKey, Fraction] = {}
        for row in self.rows:
            key = (row.natural_action, row.outcome(row.natural_action))
            masses[key] = masses.get(key, Fraction()) + row.probability
        return tuple(sorted(masses.items(), key=lambda item: (item[0][0], item[0][1])))

    def interventional_mean(self, action: int) -> Fraction:
        """Calculate E[Y | do(A=action)] by exact finite enumeration."""
        if action not in {0, 1}:
            raise ValueError("intervention action must be binary")
        return sum((row.probability * row.outcome(action) for row in self.rows), start=Fraction())

    def average_treatment_effect(self) -> Fraction:
        """Calculate E[Y(1) - Y(0)] exactly."""
        return self.interventional_mean(1) - self.interventional_mean(0)

    def observed_correlation(self) -> float | None:
        """Return the population correlation in the observational distribution."""
        mean_action = sum(
            (mass * action for (action, _), mass in self.observational_distribution()),
            start=Fraction(),
        )
        mean_outcome = sum(
            (mass * outcome for (_, outcome), mass in self.observational_distribution()),
            start=Fraction(),
        )
        covariance = sum(
            (
                mass * (action - mean_action) * (outcome - mean_outcome)
                for (action, outcome), mass in self.observational_distribution()
            ),
            start=Fraction(),
        )
        action_variance = sum(
            (
                mass * (action - mean_action) ** 2
                for (action, _), mass in self.observational_distribution()
            ),
            start=Fraction(),
        )
        outcome_variance = sum(
            (
                mass * (outcome - mean_outcome) ** 2
                for (_, outcome), mass in self.observational_distribution()
            ),
            start=Fraction(),
        )
        if action_variance == 0 or outcome_variance == 0:
            return None
        return float(covariance) / math.sqrt(float(action_variance * outcome_variance))

    def conditional_mean_mse(self) -> Fraction:
        """Return the optimal in-distribution conditional-mean prediction error."""
        action_mass: dict[int, Fraction] = {}
        action_weighted_outcome: dict[int, Fraction] = {}
        distribution = self.observational_distribution()
        for (action, outcome), mass in distribution:
            action_mass[action] = action_mass.get(action, Fraction()) + mass
            action_weighted_outcome[action] = (
                action_weighted_outcome.get(action, Fraction()) + mass * outcome
            )
        conditional_mean = {
            action: action_weighted_outcome[action] / mass for action, mass in action_mass.items()
        }
        return sum(
            (
                mass * (outcome - conditional_mean[action]) ** 2
                for (action, outcome), mass in distribution
            ),
            start=Fraction(),
        )

    def modal_exact_prediction_accuracy(self) -> Fraction:
        """Return the Bayes exact-match accuracy under observed action values."""
        by_action: dict[int, dict[Fraction, Fraction]] = {}
        for (action, outcome), mass in self.observational_distribution():
            outcomes = by_action.setdefault(action, {})
            outcomes[outcome] = outcomes.get(outcome, Fraction()) + mass
        return sum((max(outcomes.values()) for outcomes in by_action.values()), start=Fraction())

    def observational_metrics(self) -> dict[str, JsonValue]:
        """Return prediction/correlation summaries that depend only on P(A, Y)."""
        correlation = self.observed_correlation()
        return {
            "correlation": correlation,
            "conditional_mean_mse": _fraction_payload(self.conditional_mean_mse()),
            "modal_exact_prediction_accuracy": _fraction_payload(
                self.modal_exact_prediction_accuracy()
            ),
        }

    def to_payload(self) -> dict[str, JsonValue]:
        """Serialize the complete finite SCM and its exact calculated effect."""
        observed: list[JsonValue] = []
        for (action, outcome), mass in self.observational_distribution():
            observed.append(
                {
                    "action": action,
                    "outcome": _fraction_payload(outcome),
                    "probability": _fraction_payload(mass),
                }
            )

        latent_rows: list[JsonValue] = []
        for row in self.rows:
            latent_rows.append(
                {
                    "latent_state": row.latent_state,
                    "probability": _fraction_payload(row.probability),
                    "natural_action": row.natural_action,
                    "outcome_if_zero": _fraction_payload(row.outcome_if_zero),
                    "outcome_if_one": _fraction_payload(row.outcome_if_one),
                }
            )

        return {
            "model_id": self.model_id,
            "action_labels": {"0": self.action_zero_label, "1": self.action_one_label},
            "outcome_name": self.outcome_name,
            "outcome_unit": self.outcome_unit,
            "latent_potential_outcomes": latent_rows,
            "observational_distribution": observed,
            "observational_metrics": self.observational_metrics(),
            "interventional_mean_if_zero": _fraction_payload(self.interventional_mean(0)),
            "interventional_mean_if_one": _fraction_payload(self.interventional_mean(1)),
            "average_treatment_effect": _fraction_payload(self.average_treatment_effect()),
            "effect_sign": _effect_sign(self.average_treatment_effect()),
        }


@dataclass(frozen=True, slots=True)
class CounterexamplePair:
    """Two observationally equivalent SCMs with incompatible causal answers."""

    counterexample_id: str
    title: str
    statement: str
    model_left: FiniteSCM
    model_right: FiniteSCM
    evidence_obligations: tuple[str, ...]
    validity_scope: str

    @property
    def observationally_equivalent(self) -> bool:
        """Whether the two SCMs imply exactly the same P(A, Y)."""
        return (
            self.model_left.observational_distribution()
            == self.model_right.observational_distribution()
        )

    @property
    def opposite_effect_signs(self) -> bool:
        """Whether the two nonzero effects point in opposite directions."""
        left_sign = _effect_sign(self.model_left.average_treatment_effect())
        right_sign = _effect_sign(self.model_right.average_treatment_effect())
        return left_sign != 0 and left_sign == -right_sign

    def validate_witness(self) -> None:
        """Fail if the pair does not establish the intended non-identifiability witness."""
        if not self.observationally_equivalent:
            raise ValueError(f"{self.counterexample_id} models are observationally distinct")
        if not self.opposite_effect_signs:
            raise ValueError(f"{self.counterexample_id} effects do not have opposite signs")
        if self.model_left.observational_metrics() != self.model_right.observational_metrics():
            raise ValueError(f"{self.counterexample_id} observational metrics differ")

    def to_payload(self) -> dict[str, JsonValue]:
        """Serialize the validated witness and its narrow scientific implication."""
        self.validate_witness()
        return {
            "counterexample_id": self.counterexample_id,
            "title": self.title,
            "statement": self.statement,
            "observationally_equivalent": self.observationally_equivalent,
            "opposite_effect_signs": self.opposite_effect_signs,
            "shared_observational_metrics": self.model_left.observational_metrics(),
            "models": [self.model_left.to_payload(), self.model_right.to_payload()],
            "evidence_obligations": list(self.evidence_obligations),
            "validity_scope": self.validity_scope,
        }


def observational_equivalence_pair() -> CounterexamplePair:
    """Create a generic binary witness with ATE +1/2 versus -1/2."""
    quarter = Fraction(1, 4)
    positive = FiniteSCM(
        model_id="binary-positive-effect",
        action_zero_label="A=0",
        action_one_label="A=1",
        outcome_name="binary_utility",
        outcome_unit="indicator",
        rows=(
            PotentialOutcomeRow("u0", quarter, 0, Fraction(0), Fraction(1)),
            PotentialOutcomeRow("u1", quarter, 0, Fraction(1), Fraction(1)),
            PotentialOutcomeRow("u2", quarter, 1, Fraction(0), Fraction(0)),
            PotentialOutcomeRow("u3", quarter, 1, Fraction(0), Fraction(1)),
        ),
    )
    negative = FiniteSCM(
        model_id="binary-negative-effect",
        action_zero_label="A=0",
        action_one_label="A=1",
        outcome_name="binary_utility",
        outcome_unit="indicator",
        rows=(
            PotentialOutcomeRow("u0", quarter, 0, Fraction(0), Fraction(0)),
            PotentialOutcomeRow("u1", quarter, 0, Fraction(1), Fraction(0)),
            PotentialOutcomeRow("u2", quarter, 1, Fraction(1), Fraction(0)),
            PotentialOutcomeRow("u3", quarter, 1, Fraction(1), Fraction(1)),
        ),
    )
    return CounterexamplePair(
        counterexample_id="CE1_OBSERVATIONAL_EQUIVALENCE",
        title="Identical observed distribution, opposite intervention effects",
        statement=(
            "Both models induce the uniform joint distribution over binary A and Y, while "
            "their average treatment effects are +1/2 and -1/2."
        ),
        model_left=positive,
        model_right=negative,
        evidence_obligations=(
            "Declare a causal graph and validity scope that rule out incompatible SCMs.",
            "Provide a valid identification strategy with its exchangeability and support "
            "assumptions.",
            "Retain provenance showing that the evidence was collected under the declared regime.",
        ),
        validity_scope=(
            "Finite binary SCMs under the displayed observational policy; this is a witness of "
            "non-identifiability, not a claim about every causal model."
        ),
    )


def hidden_cross_traffic_pair() -> CounterexamplePair:
    """Create a rerouting witness confounded by unobserved cross traffic."""
    half = Fraction(1, 2)
    helpful = FiniteSCM(
        model_id="reroute-helpful-hidden-cross-traffic",
        action_zero_label="KEEP_CURRENT_PATH",
        action_one_label="REROUTE_TO_PATH",
        outcome_name="end_to_end_latency",
        outcome_unit="milliseconds",
        rows=(
            PotentialOutcomeRow("cross_traffic_low", half, 0, Fraction(50), Fraction(40)),
            PotentialOutcomeRow("cross_traffic_high", half, 1, Fraction(80), Fraction(70)),
        ),
    )
    harmful = FiniteSCM(
        model_id="reroute-harmful-hidden-cross-traffic",
        action_zero_label="KEEP_CURRENT_PATH",
        action_one_label="REROUTE_TO_PATH",
        outcome_name="end_to_end_latency",
        outcome_unit="milliseconds",
        rows=(
            PotentialOutcomeRow("cross_traffic_low", half, 0, Fraction(50), Fraction(60)),
            PotentialOutcomeRow("cross_traffic_high", half, 1, Fraction(60), Fraction(70)),
        ),
    )
    return CounterexamplePair(
        counterexample_id="CE2_HIDDEN_CROSS_TRAFFIC",
        title="Perfect latency prediction does not identify rerouting impact",
        statement=(
            "The observed policy yields (KEEP, 50 ms) or (REROUTE, 70 ms) with equal "
            "probability in both models. The conditional predictor has zero error, but rerouting "
            "changes mean latency by -10 ms in one model and +10 ms in the other."
        ),
        model_left=helpful,
        model_right=harmful,
        evidence_obligations=(
            "Measure path-relevant cross traffic and capacity with declared source identity.",
            "Satisfy freshness and synchronization bounds for action-time and outcome-time "
            "evidence.",
            "Establish action support through randomized or justified exogenous variation, or use "
            "another valid identification strategy.",
            "Reject or defer when hidden common causes or the required action support remain "
            "unresolved.",
        ),
        validity_scope=(
            "Two-path rerouting with binary hidden cross-traffic state and the displayed "
            "deterministic observational policy. Measuring cross traffic alone is not sufficient "
            "if action overlap remains absent."
        ),
    )


def counterexample_pairs() -> tuple[CounterexamplePair, ...]:
    """Return every deterministic M02 counterexample in stable order."""
    return observational_equivalence_pair(), hidden_cross_traffic_pair()


def build_m02_evidence() -> dict[str, JsonValue]:
    """Build the complete deterministic M02 evidence object."""
    pairs = counterexample_pairs()
    payloads: list[JsonValue] = []
    for pair in pairs:
        payloads.append(pair.to_payload())
    return {
        "schema_version": "1.0.0",
        "generator": "ictc.causal.counterexamples.build_m02_evidence",
        "method": "complete finite enumeration with exact rational arithmetic",
        "random_sampling_used": False,
        "counterexamples": payloads,
        "formal_conclusion": (
            "Within each declared model class, the observed distribution and all statistics "
            "computed only from it are compatible with opposite intervention effects. Extra "
            "causal assumptions or evidence obligations are therefore required before admission."
        ),
        "claim_boundary": (
            "This artifact establishes the displayed non-identifiability witnesses only. It does "
            "not establish empirical ICTC performance, broad literature novelty, or estimator "
            "accuracy."
        ),
    }


def render_m02_evidence() -> str:
    """Render deterministic UTF-8 JSON with stable key and list ordering."""
    return json.dumps(build_m02_evidence(), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
