"""M10 A0-A16 orchestration and post-persistence truth-boundary tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ictc.causal.estimation import EstimationReason, EstimationResult
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind, AdmissionDecision
from ictc.domain.models import CausalEffectEstimate, UncertaintyInterval
from ictc.pilot import (
    ObservationalDecision,
    PilotOutcomeTruth,
    PotentialOutcomeTruth,
    PreActionEvidence,
)
from ictc.pilot.ablation import (
    AblationId,
    FixtureKind,
    PilotDecision,
    PilotModel,
    PilotOpportunity,
    build_opportunities,
    fit_pilot_model,
    run_ablation_suite,
    score_persisted_decisions,
    score_population_estimands,
)
from ictc.pilot.benchmark import reference_identification


def _estimate(point: float = -2.0, lower: float = -3.0, upper: float = -1.0) -> EstimationResult:
    return EstimationResult(
        True,
        EstimationReason.ESTIMATED,
        ("QueuePressureBin",),
        24,
        (),
        CausalEffectEstimate(
            estimand="reroute-minus-keep latency",
            point_estimate=point,
            unit="ms",
            uncertainty=UncertaintyInterval(
                lower=lower,
                upper=upper,
                confidence_level=0.95,
            ),
            estimator_name="test",
        ),
        0.5,
    )


def _model() -> PilotModel:
    crossing = _estimate(-1.0, -2.1, 0.1)
    return PilotModel(
        _estimate(),
        (("candidate-constrained", crossing),),
        _estimate(-1.5, -2.5, -0.5),
        _estimate(-1.2, -2.2, -0.2),
    )


def _outcome(latency: float, route_changes: int) -> PilotOutcomeTruth:
    return PilotOutcomeTruth(latency, 0.9, 0.1, 1.0, 2.0, 0.1, route_changes)


def _decision(
    index: int = 0,
    *,
    split: DatasetSplit = DatasetSplit.DEVELOPMENT,
    pressure: int = 1,
    action: ActionKind = ActionKind.REROUTE_TO_PATH,
    outcome: float = 5.0,
    topology: str = "development-topology",
    regime: str = "complete",
) -> ObservationalDecision:
    evidence = PreActionEvidence(
        pressure,
        float(pressure),
        0,
        0,
        0,
        ("source-a", "source-b"),
        ("A" * 64, "B" * 64),
        2,
        128,
    )
    return ObservationalDecision(
        f"pair-{index}",
        f"scenario-{index}",
        split,
        index,
        topology,
        "medium",
        regime,
        action,
        outcome,
        pressure,
        evidence,
        reference_identification(True),
        0.75,
        0.25,
    )


def _select(
    rows: tuple[PilotDecision, ...],
    ablation: AblationId,
    *,
    configuration: str = "default",
) -> PilotDecision:
    return next(
        row for row in rows if row.ablation_id is ablation and row.configuration_id == configuration
    )


def test_fit_uses_training_observations_without_truth() -> None:
    rows = tuple(
        _decision(
            index,
            split=DatasetSplit.TRAIN,
            pressure=pressure,
            action=action,
            outcome=10.0 + pressure - (2.0 if action is ActionKind.REROUTE_TO_PATH else 0),
            topology="training-topology",
        )
        for index, (pressure, action) in enumerate(
            (pressure, action)
            for pressure in (-1, 0, 1)
            for action in (ActionKind.KEEP_CURRENT_PATH, ActionKind.REROUTE_TO_PATH)
            for _repeat in range(2)
        )
    )
    model = fit_pilot_model(rows)

    assert model.global_causal.estimable
    assert model.b9_association.estimable
    assert model.misspecified_unadjusted.estimable
    with pytest.raises(PermissionError, match="training rows only"):
        fit_pilot_model((replace(rows[0], split=DatasetSplit.DEVELOPMENT),))


def test_suite_covers_all_ablations_and_frozen_grids() -> None:
    opportunity = PilotOpportunity("opportunity", _decision())
    rows = run_ablation_suite((opportunity,), _model())

    assert len(rows) == 29
    assert {row.ablation_id for row in rows} == set(AblationId)
    assert sum(row.ablation_id is AblationId.A15 for row in rows) == 5
    assert sum(row.ablation_id is AblationId.A16 for row in rows) == 9
    assert "keep_latency_ms" not in PilotOpportunity.__dataclass_fields__
    assert "keep_latency_ms" not in rows[0].__dataclass_fields__


@pytest.mark.parametrize(
    ("fixture", "ablation"),
    [
        (FixtureKind.INVALID_PROVENANCE, AblationId.A5),
        (FixtureKind.LOW_TRUST, AblationId.A6),
        (FixtureKind.SINGLE_DOMAIN, AblationId.A7),
        (FixtureKind.QUEUE_FIELD_REDACTED, AblationId.A9),
    ],
)
def test_clause_and_association_ablations_change_only_exercised_fixtures(
    fixture: FixtureKind, ablation: AblationId
) -> None:
    opportunity = PilotOpportunity(f"opportunity:{fixture.value}", _decision(), fixture)
    rows = run_ablation_suite((opportunity,), _model())

    assert _select(rows, AblationId.A0).executable_action is ActionKind.KEEP_CURRENT_PATH
    assert _select(rows, ablation).executable_action is ActionKind.REROUTE_TO_PATH


def test_identification_estimation_monitoring_and_fallback_ablations() -> None:
    redacted = PilotOpportunity(
        "redacted", _decision(topology="candidate-constrained"), FixtureKind.QUEUE_FIELD_REDACTED
    )
    rows = run_ablation_suite((redacted,), _model())
    a0 = _select(rows, AblationId.A0)
    a1 = _select(rows, AblationId.A1)
    a2 = _select(rows, AblationId.A2)
    a13 = _select(rows, AblationId.A13)
    a14 = _select(rows, AblationId.A14)

    assert a0.reference_identified is False
    assert a1.model_identified and a1.executable_action is ActionKind.KEEP_CURRENT_PATH
    assert a2.executable_action is ActionKind.KEEP_CURRENT_PATH
    assert a13.executable_action is None
    assert a14.executable_action is ActionKind.REROUTE_TO_PATH

    complete = PilotOpportunity("candidate", _decision(topology="candidate-constrained"))
    complete_rows = run_ablation_suite((complete,), _model())
    assert _select(complete_rows, AblationId.A0).decision is AdmissionDecision.MODIFY
    assert _select(complete_rows, AblationId.A2).decision is AdmissionDecision.ADMIT

    breach = PilotOpportunity("breach", _decision(), FixtureKind.POST_COMMIT_BREACH)
    breach_rows = run_ablation_suite((breach,), _model())
    assert _select(breach_rows, AblationId.A0).fallback_executed
    assert _select(breach_rows, AblationId.A12).executable_action is ActionKind.REROUTE_TO_PATH

    infeasible = PilotOpportunity("infeasible", _decision(), FixtureKind.CONTRACT_INFEASIBLE)
    infeasible_rows = run_ablation_suite((infeasible,), _model())
    assert all(row.reference_contract_satisfied is False for row in infeasible_rows)
    assert all(row.executable_action is not ActionKind.REROUTE_TO_PATH for row in infeasible_rows)


def test_freshness_synchronization_cost_and_static_contract_paths() -> None:
    base = _decision()
    delayed_evidence = replace(
        base.pre_action_evidence,
        delivered_at_tick=3,
        freshness_ticks=3,
    )
    delayed = PilotOpportunity(
        "delayed",
        replace(base, pre_action_evidence=delayed_evidence, observation_regime_id="delayed"),
    )
    delayed_rows = run_ablation_suite((delayed,), _model())
    assert _select(delayed_rows, AblationId.A0).fallback_executed
    assert _select(delayed_rows, AblationId.A3).executable_action is ActionKind.REROUTE_TO_PATH

    skewed_evidence = replace(base.pre_action_evidence, clock_skew_span_ticks=4)
    skewed = PilotOpportunity("skewed", replace(base, pre_action_evidence=skewed_evidence))
    skewed_rows = run_ablation_suite((skewed,), _model())
    assert _select(skewed_rows, AblationId.A4).executable_action is ActionKind.REROUTE_TO_PATH

    complete_rows = run_ablation_suite((PilotOpportunity("cost", base),), _model())
    assert _select(complete_rows, AblationId.A10).telemetry_bytes == 640
    assert not _select(complete_rows, AblationId.A10).contract_satisfied
    assert not _select(complete_rows, AblationId.A11).contract_satisfied
    assert _select(complete_rows, AblationId.A8).contract_satisfied


def test_controlled_fault_expansion_and_post_persistence_scoring() -> None:
    decision = _decision()
    opportunities = build_opportunities((decision,), include_controlled_faults=True)
    assert len(opportunities) == len(FixtureKind)

    rows = run_ablation_suite((opportunities[0],), _model())
    a0 = _select(rows, AblationId.A0)
    truth = PotentialOutcomeTruth(decision.pair_id, 1, _outcome(10.0, 0), _outcome(5.0, 1))
    scores = score_persisted_decisions((a0,), {decision.pair_id: truth})
    assert scores[0].harmful_action is False
    assert scores[0].selected_latency_ms == 5.0
    estimands = score_population_estimands(
        (a0,),
        {opportunities[0].opportunity_id: opportunities[0]},
        {decision.pair_id: truth},
    )
    assert estimands[0].target_ate_ms == -5.0
    assert estimands[0].estimation_error_ms == 3.0
    assert estimands[0].interval_covered is False

    with pytest.raises(KeyError, match="paired truth"):
        score_persisted_decisions((a0,), {})
    with pytest.raises(PermissionError, match="require GO authorization"):
        build_opportunities(
            (replace(decision, split=DatasetSplit.TEST),), include_controlled_faults=False
        )
