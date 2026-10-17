"""Finite-sample stratified backdoor estimation and explicit-failure tests."""

from __future__ import annotations

import math

import pytest

from ictc.causal.estimation import (
    AdjustmentObservation,
    EstimationReason,
    EstimationResult,
    estimate_stratified_backdoor,
)
from ictc.causal.identification import IdentifiabilityReason, IdentifiabilityResult
from ictc.domain.enums import ActionKind


def identification(*, identified: bool = True) -> IdentifiabilityResult:
    return IdentifiabilityResult(
        identified=identified,
        reason=(
            IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
            if identified
            else IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
        ),
        treatment="RouteAction",
        outcome="Latency",
        available_variables=("Latency", "RouteAction", "Z"),
        valid_adjustment_sets=(("Z",),) if identified else (),
        minimal_adjustment_sets=(("Z",),) if identified else (),
        missing_variable_sets=() if identified else (("Z",),),
    )


def row(action: ActionKind, outcome: float, z: float) -> AdjustmentObservation:
    return AdjustmentObservation.from_mapping(
        treatment=action, outcome=outcome, covariates={"Z": z}
    )


def complete_rows() -> tuple[AdjustmentObservation, ...]:
    return (
        row(ActionKind.KEEP_CURRENT_PATH, 10, 0),
        row(ActionKind.KEEP_CURRENT_PATH, 12, 0),
        row(ActionKind.REROUTE_TO_PATH, 8, 0),
        row(ActionKind.REROUTE_TO_PATH, 10, 0),
        row(ActionKind.KEEP_CURRENT_PATH, 20, 1),
        row(ActionKind.KEEP_CURRENT_PATH, 22, 1),
        row(ActionKind.REROUTE_TO_PATH, 17, 1),
        row(ActionKind.REROUTE_TO_PATH, 19, 1),
    )


def estimate(rows: tuple[AdjustmentObservation, ...]) -> EstimationResult:
    return estimate_stratified_backdoor(
        rows,
        identification(),
        ("Z",),
        estimand="reroute-minus-keep latency",
        unit="ms",
    )


def test_known_stratified_effect_and_interval() -> None:
    result = estimate(complete_rows())
    assert result.estimable
    assert result.reason is EstimationReason.ESTIMATED
    assert result.sample_size == 8
    assert result.estimate is not None
    assert result.estimate.point_estimate == pytest.approx(-2.5)
    assert result.standard_error == pytest.approx(1.0)
    assert result.estimate.uncertainty.lower == pytest.approx(-4.4599639845)
    assert result.estimate.uncertainty.upper == pytest.approx(-0.5400360155)
    assert tuple(item.weight for item in result.strata) == (0.5, 0.5)
    assert tuple(item.effect for item in result.strata) == (-2.0, -3.0)


def test_identification_and_adjustment_set_dominate_estimation() -> None:
    nonidentified = estimate_stratified_backdoor(
        complete_rows(),
        identification(identified=False),
        ("Z",),
        estimand="effect",
        unit="ms",
    )
    assert not nonidentified.estimable
    assert nonidentified.reason is EstimationReason.IDENTIFICATION_REQUIRED

    wrong_set = estimate_stratified_backdoor(
        complete_rows(), identification(), (), estimand="effect", unit="ms"
    )
    assert not wrong_set.estimable
    assert wrong_set.reason is EstimationReason.ADJUSTMENT_SET_NOT_IDENTIFIED


def test_no_data_schema_positivity_and_sample_failures_are_distinct() -> None:
    no_data = estimate(())
    assert no_data.reason is EstimationReason.NO_OBSERVATIONS

    mismatch = estimate(
        (
            AdjustmentObservation.from_mapping(
                treatment=ActionKind.KEEP_CURRENT_PATH,
                outcome=1,
                covariates={"Other": 0},
            ),
        )
    )
    assert mismatch.reason is EstimationReason.COVARIATE_SCHEMA_MISMATCH

    no_support = estimate(
        (
            row(ActionKind.KEEP_CURRENT_PATH, 1, 0),
            row(ActionKind.KEEP_CURRENT_PATH, 2, 0),
        )
    )
    assert no_support.reason is EstimationReason.INSUFFICIENT_ACTION_SUPPORT
    assert no_support.strata[0].intervention_count == 0

    too_small = estimate(
        (
            row(ActionKind.KEEP_CURRENT_PATH, 1, 0),
            row(ActionKind.REROUTE_TO_PATH, 0, 0),
        )
    )
    assert too_small.reason is EstimationReason.INSUFFICIENT_STRATUM_SAMPLE


def test_observation_result_and_confidence_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        AdjustmentObservation.from_mapping(treatment=ActionKind.KEEP_CURRENT_PATH, outcome=math.inf)
    with pytest.raises(ValueError, match="canonical"):
        AdjustmentObservation(
            treatment=ActionKind.KEEP_CURRENT_PATH,
            outcome=1,
            covariates=(("Z", 0), ("A", 1)),
        )
    with pytest.raises(ValueError, match="confidence"):
        estimate_stratified_backdoor(
            complete_rows(),
            identification(),
            ("Z",),
            estimand="effect",
            unit="ms",
            confidence_level=1,
        )
    with pytest.raises(ValueError, match="unique"):
        estimate_stratified_backdoor(
            complete_rows(),
            identification(),
            ("Z", "Z"),
            estimand="effect",
            unit="ms",
        )
    with pytest.raises(ValueError, match="must agree"):
        EstimationResult(
            estimable=True,
            reason=EstimationReason.ESTIMATED,
            adjustment_set=(),
            sample_size=0,
            strata=(),
        )
