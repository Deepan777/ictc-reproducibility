"""Transparent finite-sample estimation for an identified backdoor functional."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from statistics import NormalDist, fmean, variance

from ictc.causal.identification import IdentifiabilityResult
from ictc.domain.enums import ActionKind
from ictc.domain.models import CausalEffectEstimate, UncertaintyInterval


class EstimationReason(StrEnum):
    ESTIMATED = "ESTIMATED"
    IDENTIFICATION_REQUIRED = "IDENTIFICATION_REQUIRED"
    ADJUSTMENT_SET_NOT_IDENTIFIED = "ADJUSTMENT_SET_NOT_IDENTIFIED"
    NO_OBSERVATIONS = "NO_OBSERVATIONS"
    COVARIATE_SCHEMA_MISMATCH = "COVARIATE_SCHEMA_MISMATCH"
    INSUFFICIENT_ACTION_SUPPORT = "INSUFFICIENT_ACTION_SUPPORT"
    INSUFFICIENT_STRATUM_SAMPLE = "INSUFFICIENT_STRATUM_SAMPLE"
    CONTEXT_MISMATCH = "CONTEXT_MISMATCH"
    INSUFFICIENT_INDEPENDENT_CLUSTERS = "INSUFFICIENT_INDEPENDENT_CLUSTERS"


@dataclass(frozen=True, slots=True)
class AdjustmentObservation:
    """One controller-visible pre-action covariate row and observed outcome."""

    treatment: ActionKind
    outcome: float
    covariates: tuple[tuple[str, float], ...] = ()
    cluster_id: str | None = None

    def __post_init__(self) -> None:
        if self.treatment not in {
            ActionKind.KEEP_CURRENT_PATH,
            ActionKind.REROUTE_TO_PATH,
        }:
            raise ValueError("estimation supports only KEEP and REROUTE treatments")
        if not math.isfinite(self.outcome):
            raise ValueError("outcome must be finite")
        names = [name for name, _value in self.covariates]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("covariates must have unique names in canonical order")
        if any(not name or not math.isfinite(value) for name, value in self.covariates):
            raise ValueError("covariate names and values must be finite and nonempty")

    @classmethod
    def from_mapping(
        cls,
        *,
        treatment: ActionKind,
        outcome: float,
        covariates: dict[str, float] | None = None,
        cluster_id: str | None = None,
    ) -> AdjustmentObservation:
        return cls(
            treatment=treatment,
            outcome=outcome,
            covariates=tuple(sorted((covariates or {}).items())),
            cluster_id=cluster_id,
        )


@dataclass(frozen=True, slots=True)
class StratumDiagnostic:
    values: tuple[float, ...]
    baseline_count: int
    intervention_count: int
    weight: float
    effect: float | None


@dataclass(frozen=True, slots=True)
class EstimationResult:
    estimable: bool
    reason: EstimationReason
    adjustment_set: tuple[str, ...]
    sample_size: int
    strata: tuple[StratumDiagnostic, ...]
    estimate: CausalEffectEstimate | None = None
    standard_error: float | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.estimable != (self.estimate is not None):
            raise ValueError("estimable result and estimate presence must agree")
        if self.estimable and self.reason is not EstimationReason.ESTIMATED:
            raise ValueError("successful estimation requires the ESTIMATED reason")
        if not self.estimable and self.standard_error is not None:
            raise ValueError("failed estimation cannot report a standard error")


def _failure(
    reason: EstimationReason,
    adjustment_set: tuple[str, ...],
    observations: tuple[AdjustmentObservation, ...],
    detail: str,
    strata: tuple[StratumDiagnostic, ...] = (),
) -> EstimationResult:
    return EstimationResult(
        estimable=False,
        reason=reason,
        adjustment_set=adjustment_set,
        sample_size=len(observations),
        strata=strata,
        detail=detail,
    )


def estimate_stratified_backdoor(
    observations: tuple[AdjustmentObservation, ...],
    identification: IdentifiabilityResult,
    adjustment_set: tuple[str, ...],
    *,
    estimand: str,
    unit: str,
    confidence_level: float = 0.95,
    context_values: tuple[tuple[str, float], ...] = (),
) -> EstimationResult:
    """Estimate intervention-minus-baseline effect by exact empirical stratification.

    The approximate interval holds empirical stratum weights fixed. With cluster IDs,
    it uses cluster-summed influence scores (CR1); without them it assumes independent
    completed episodes. It is not the held-out algorithm-comparison procedure.
    """
    if not 0 < confidence_level < 1:
        raise ValueError("confidence level must lie strictly between zero and one")
    canonical_adjustment = tuple(sorted(adjustment_set))
    context = dict(context_values)
    if (
        len(context) != len(context_values)
        or tuple(sorted(context)) != identification.conditioning_variables
    ):
        return _failure(
            EstimationReason.CONTEXT_MISMATCH,
            canonical_adjustment,
            observations,
            "Context values must match the variables used for conditional identification.",
        )
    if len(canonical_adjustment) != len(set(canonical_adjustment)):
        raise ValueError("adjustment variables must be unique")
    if not identification.identified:
        return _failure(
            EstimationReason.IDENTIFICATION_REQUIRED,
            canonical_adjustment,
            observations,
            "effect estimation is disabled because the estimand is not identified",
        )
    valid_sets = {tuple(sorted(item)) for item in identification.valid_adjustment_sets}
    if canonical_adjustment not in valid_sets:
        return _failure(
            EstimationReason.ADJUSTMENT_SET_NOT_IDENTIFIED,
            canonical_adjustment,
            observations,
            "selected adjustment set is not one of the identified backdoor sets",
        )
    if not observations:
        return _failure(
            EstimationReason.NO_OBSERVATIONS,
            canonical_adjustment,
            observations,
            "no finite-sample observations are available",
        )
    expected_names = tuple(sorted((*canonical_adjustment, *context)))
    if set(canonical_adjustment) & context.keys():
        raise ValueError("additional adjustment variables must exclude mandatory context")
    for row in observations:
        if tuple(name for name, _value in row.covariates) != expected_names:
            return _failure(
                EstimationReason.COVARIATE_SCHEMA_MISMATCH,
                canonical_adjustment,
                observations,
                "every row must contain exactly the selected pre-action adjustment variables",
            )

    observations = tuple(
        row
        for row in observations
        if all(dict(row.covariates)[name] == value for name, value in context.items())
    )
    if not observations:
        return _failure(
            EstimationReason.NO_OBSERVATIONS,
            canonical_adjustment,
            observations,
            "No completed observations match the requested context.",
        )
    grouped: defaultdict[tuple[float, ...], dict[ActionKind, list[float]]] = defaultdict(
        lambda: {
            ActionKind.KEEP_CURRENT_PATH: [],
            ActionKind.REROUTE_TO_PATH: [],
        }
    )
    for row in observations:
        key = tuple(dict(row.covariates)[name] for name in canonical_adjustment)
        grouped[key][row.treatment].append(row.outcome)

    diagnostics: list[StratumDiagnostic] = []
    total = len(observations)
    for values, arms in sorted(grouped.items()):
        baseline = arms[ActionKind.KEEP_CURRENT_PATH]
        intervention = arms[ActionKind.REROUTE_TO_PATH]
        weight = (len(baseline) + len(intervention)) / total
        effect = fmean(intervention) - fmean(baseline) if baseline and intervention else None
        diagnostics.append(
            StratumDiagnostic(values, len(baseline), len(intervention), weight, effect)
        )
    frozen_diagnostics = tuple(diagnostics)
    if any(item.baseline_count == 0 or item.intervention_count == 0 for item in diagnostics):
        return _failure(
            EstimationReason.INSUFFICIENT_ACTION_SUPPORT,
            canonical_adjustment,
            observations,
            "both actions must occur in every empirical adjustment stratum",
            frozen_diagnostics,
        )
    if any(item.baseline_count < 2 or item.intervention_count < 2 for item in diagnostics):
        return _failure(
            EstimationReason.INSUFFICIENT_STRATUM_SAMPLE,
            canonical_adjustment,
            observations,
            "each action requires at least two observations per stratum for uncertainty",
            frozen_diagnostics,
        )

    point = 0.0
    sampling_variance = 0.0
    for diagnostic in diagnostics:
        arms = grouped[diagnostic.values]
        baseline = arms[ActionKind.KEEP_CURRENT_PATH]
        intervention = arms[ActionKind.REROUTE_TO_PATH]
        assert diagnostic.effect is not None
        point += diagnostic.weight * diagnostic.effect
        sampling_variance += diagnostic.weight**2 * (
            variance(intervention) / len(intervention) + variance(baseline) / len(baseline)
        )
    standard_error = math.sqrt(sampling_variance)
    clustered = any(row.cluster_id is not None for row in observations)
    if clustered:
        if any(row.cluster_id is None for row in observations):
            raise ValueError("cluster IDs must be supplied for every observation or none")
        scores: defaultdict[str, float] = defaultdict(float)
        by_values = {item.values: item for item in diagnostics}
        for row in observations:
            values = tuple(dict(row.covariates)[name] for name in canonical_adjustment)
            diagnostic = by_values[values]
            arm = grouped[values][row.treatment]
            sign = 1 if row.treatment is ActionKind.REROUTE_TO_PATH else -1
            assert row.cluster_id is not None
            scores[row.cluster_id] += (
                sign * diagnostic.weight * (row.outcome - fmean(arm)) / len(arm)
            )
        if len(scores) < 2:
            return _failure(
                EstimationReason.INSUFFICIENT_INDEPENDENT_CLUSTERS,
                canonical_adjustment,
                observations,
                "At least two independent clusters are needed for variance.",
                frozen_diagnostics,
            )
        sampling_variance = len(scores) / (len(scores) - 1) * sum(v * v for v in scores.values())
        standard_error = math.sqrt(sampling_variance)
    critical = NormalDist().inv_cdf((1 + confidence_level) / 2)
    margin = critical * standard_error
    estimate = CausalEffectEstimate(
        estimand=estimand,
        point_estimate=point,
        unit=unit,
        uncertainty=UncertaintyInterval(
            lower=point - margin,
            upper=point + margin,
            confidence_level=confidence_level,
        ),
        estimator_name=(
            "empirical-stratified-backdoor-seed-cluster-v2"
            if clustered
            else "empirical-stratified-backdoor-standardization-v1"
        ),
    )
    return EstimationResult(
        estimable=True,
        reason=EstimationReason.ESTIMATED,
        adjustment_set=canonical_adjustment,
        sample_size=total,
        strata=frozen_diagnostics,
        estimate=estimate,
        standard_error=standard_error,
        detail=(
            "approximate cluster-robust normal interval; empirical stratum weights fixed"
            if clustered
            else "approximate independent-episode normal interval; empirical stratum weights fixed"
        ),
    )
