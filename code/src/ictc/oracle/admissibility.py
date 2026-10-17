"""Independent exact finite-model oracle for causal-effect identifiability.

This evaluation-only implementation intentionally does not use the ICTC admission rule,
contract synthesis, graph identifier, controller reasons, or M14 reference flags.  Its answer is
model-class relative: an effect is identified exactly when every enumerated model compatible with
the supplied observational distribution implies the same interventional contrast.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type ObservationKey = tuple[int, Fraction]
type ObservationDistribution = tuple[tuple[ObservationKey, Fraction], ...]


class OracleReason(StrEnum):
    """Closed set of exact finite-model oracle outcomes."""

    IDENTIFIED_UNIQUE_EFFECT = "identified_unique_effect"
    NON_IDENTIFIABLE_EFFECT_DISAGREEMENT = "non_identifiable_effect_disagreement"
    OBSERVATION_OUTSIDE_MODEL_CLASS = "observation_outside_model_class"


class EffectDirection(StrEnum):
    """Direction of an identified action-one minus action-zero contrast."""

    POSITIVE = "positive"
    ZERO = "zero"
    NEGATIVE = "negative"
    UNRESOLVED = "unresolved"


def _fraction_payload(value: Fraction) -> dict[str, JsonValue]:
    return {"numerator": value.numerator, "denominator": value.denominator}


def _canonical_digest(payload: JsonValue) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution_payload(distribution: ObservationDistribution) -> list[JsonValue]:
    return [
        {
            "action": action,
            "outcome": _fraction_payload(outcome),
            "probability": _fraction_payload(probability),
        }
        for (action, outcome), probability in distribution
    ]


def _validate_distribution(distribution: ObservationDistribution) -> None:
    if not distribution:
        raise ValueError("observational distribution must not be empty")
    keys: list[ObservationKey] = []
    total = Fraction()
    for key, probability in distribution:
        action, _ = key
        if action not in {0, 1}:
            raise ValueError("observational actions must be binary")
        if probability <= 0:
            raise ValueError("observational probabilities must be positive")
        keys.append(key)
        total += probability
    if len(keys) != len(set(keys)):
        raise ValueError("observational outcomes must have unique support keys")
    if tuple(keys) != tuple(sorted(keys, key=lambda key: (key[0], key[1]))):
        raise ValueError("observational distribution must use canonical sorted order")
    if total != 1:
        raise ValueError("observational probabilities must sum exactly to one")


@dataclass(frozen=True, slots=True)
class OracleQuery:
    """Binary intervention query evaluated by the independent oracle."""

    query_id: str
    action_zero_label: str
    action_one_label: str
    outcome_name: str
    outcome_unit: str

    def __post_init__(self) -> None:
        values = (
            self.query_id,
            self.action_zero_label,
            self.action_one_label,
            self.outcome_name,
            self.outcome_unit,
        )
        if any(not value.strip() for value in values):
            raise ValueError("oracle query fields must be non-empty")

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "query_id": self.query_id,
            "action_zero_label": self.action_zero_label,
            "action_one_label": self.action_one_label,
            "outcome_name": self.outcome_name,
            "outcome_unit": self.outcome_unit,
        }


@dataclass(frozen=True, slots=True)
class PotentialOutcomeUnit:
    """One exact latent unit used only by the evaluation oracle."""

    unit_id: str
    probability: Fraction
    natural_action: int
    outcome_if_zero: Fraction
    outcome_if_one: Fraction

    def __post_init__(self) -> None:
        if not self.unit_id.strip():
            raise ValueError("oracle unit ID must be non-empty")
        if self.probability <= 0:
            raise ValueError("oracle unit probability must be positive")
        if self.natural_action not in {0, 1}:
            raise ValueError("oracle natural action must be binary")

    def observed_outcome(self) -> Fraction:
        return self.outcome_if_one if self.natural_action == 1 else self.outcome_if_zero


@dataclass(frozen=True, slots=True)
class FinitePotentialOutcomeModel:
    """A fully enumerable candidate model for one oracle query."""

    model_id: str
    query: OracleQuery
    units: tuple[PotentialOutcomeUnit, ...]

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("oracle model ID must be non-empty")
        if not self.units:
            raise ValueError("oracle model must contain at least one unit")
        unit_ids = [unit.unit_id for unit in self.units]
        if len(unit_ids) != len(set(unit_ids)):
            raise ValueError("oracle unit IDs must be unique within a model")
        if sum((unit.probability for unit in self.units), start=Fraction()) != 1:
            raise ValueError("oracle unit probabilities must sum exactly to one")

    def observational_distribution(self) -> ObservationDistribution:
        masses: dict[ObservationKey, Fraction] = {}
        for unit in self.units:
            key = (unit.natural_action, unit.observed_outcome())
            masses[key] = masses.get(key, Fraction()) + unit.probability
        return tuple(sorted(masses.items(), key=lambda item: (item[0][0], item[0][1])))

    def effect(self) -> Fraction:
        """Return E[Y(1)-Y(0)] by exact finite enumeration."""
        return sum(
            (
                unit.probability * (unit.outcome_if_one - unit.outcome_if_zero)
                for unit in self.units
            ),
            start=Fraction(),
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "model_id": self.model_id,
            "query": self.query.to_payload(),
            "units": [
                {
                    "unit_id": unit.unit_id,
                    "probability": _fraction_payload(unit.probability),
                    "natural_action": unit.natural_action,
                    "outcome_if_zero": _fraction_payload(unit.outcome_if_zero),
                    "outcome_if_one": _fraction_payload(unit.outcome_if_one),
                }
                for unit in self.units
            ],
        }


@dataclass(frozen=True, slots=True)
class AdmissibilityCertificate:
    """Deterministic oracle result, suitable for post-decision evaluation evidence."""

    oracle_id: str
    query: OracleQuery
    reason: OracleReason
    causally_admissible: bool
    effect_direction: EffectDirection
    identified_effect: Fraction | None
    compatible_effects: tuple[tuple[str, Fraction], ...]
    witness_model_ids: tuple[str, str] | None
    observation_digest: str
    model_class_digest: str

    def _payload_without_digest(self) -> dict[str, JsonValue]:
        identified_effect: JsonValue = None
        if self.identified_effect is not None:
            identified_effect = _fraction_payload(self.identified_effect)
        witness: JsonValue = None
        if self.witness_model_ids is not None:
            witness = list(self.witness_model_ids)
        return {
            "schema_version": "n01-admissibility-certificate-v1",
            "oracle_id": self.oracle_id,
            "query": self.query.to_payload(),
            "reason": self.reason.value,
            "causally_admissible": self.causally_admissible,
            "effect_direction": self.effect_direction.value,
            "identified_effect": identified_effect,
            "compatible_effects": [
                {"model_id": model_id, "effect": _fraction_payload(effect)}
                for model_id, effect in self.compatible_effects
            ],
            "witness_model_ids": witness,
            "observation_digest": self.observation_digest,
            "model_class_digest": self.model_class_digest,
            "claim_boundary": (
                "Model-class-relative identifiability only; this certificate does not establish "
                "effect benefit, feasibility, graph validity, transportability, or estimability."
            ),
        }

    @property
    def certificate_digest(self) -> str:
        return _canonical_digest(self._payload_without_digest())

    def to_payload(self) -> dict[str, JsonValue]:
        payload = self._payload_without_digest()
        payload["certificate_digest"] = self.certificate_digest
        return payload


@dataclass(frozen=True, slots=True)
class CausalAdmissibilityOracle:
    """Exact compatibility oracle over a declared finite potential-outcome model class."""

    oracle_id: str
    query: OracleQuery
    models: tuple[FinitePotentialOutcomeModel, ...]

    def __post_init__(self) -> None:
        if not self.oracle_id.strip():
            raise ValueError("oracle ID must be non-empty")
        if not self.models:
            raise ValueError("oracle model class must not be empty")
        model_ids = [model.model_id for model in self.models]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError("oracle model IDs must be unique")
        if any(model.query != self.query for model in self.models):
            raise ValueError("every oracle model must use the declared query")

    @property
    def model_class_digest(self) -> str:
        payload: JsonValue = {
            "oracle_id": self.oracle_id,
            "query": self.query.to_payload(),
            "models": [
                model.to_payload() for model in sorted(self.models, key=lambda item: item.model_id)
            ],
        }
        return _canonical_digest(payload)

    def evaluate(self, observation: ObservationDistribution) -> AdmissibilityCertificate:
        """Label exact identification without consulting any controller-produced field."""
        _validate_distribution(observation)
        observation_digest = _canonical_digest(_distribution_payload(observation))
        compatible = tuple(
            sorted(
                (
                    model
                    for model in self.models
                    if model.observational_distribution() == observation
                ),
                key=lambda model: model.model_id,
            )
        )
        compatible_effects = tuple((model.model_id, model.effect()) for model in compatible)
        if not compatible:
            return AdmissibilityCertificate(
                oracle_id=self.oracle_id,
                query=self.query,
                reason=OracleReason.OBSERVATION_OUTSIDE_MODEL_CLASS,
                causally_admissible=False,
                effect_direction=EffectDirection.UNRESOLVED,
                identified_effect=None,
                compatible_effects=(),
                witness_model_ids=None,
                observation_digest=observation_digest,
                model_class_digest=self.model_class_digest,
            )

        effects = {effect for _, effect in compatible_effects}
        if len(effects) == 1:
            effect = next(iter(effects))
            direction = EffectDirection.ZERO
            if effect > 0:
                direction = EffectDirection.POSITIVE
            elif effect < 0:
                direction = EffectDirection.NEGATIVE
            return AdmissibilityCertificate(
                oracle_id=self.oracle_id,
                query=self.query,
                reason=OracleReason.IDENTIFIED_UNIQUE_EFFECT,
                causally_admissible=True,
                effect_direction=direction,
                identified_effect=effect,
                compatible_effects=compatible_effects,
                witness_model_ids=None,
                observation_digest=observation_digest,
                model_class_digest=self.model_class_digest,
            )

        first_model_id, first_effect = compatible_effects[0]
        second_model_id = next(
            model_id for model_id, effect in compatible_effects[1:] if effect != first_effect
        )
        return AdmissibilityCertificate(
            oracle_id=self.oracle_id,
            query=self.query,
            reason=OracleReason.NON_IDENTIFIABLE_EFFECT_DISAGREEMENT,
            causally_admissible=False,
            effect_direction=EffectDirection.UNRESOLVED,
            identified_effect=None,
            compatible_effects=compatible_effects,
            witness_model_ids=(first_model_id, second_model_id),
            observation_digest=observation_digest,
            model_class_digest=self.model_class_digest,
        )
