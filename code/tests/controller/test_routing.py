"""Visible-only routing proposals, gating order, actuation, and fallback tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest

from ictc.causal.estimation import EstimationReason, EstimationResult
from ictc.causal.identification import IdentifiabilityReason, IdentifiabilityResult
from ictc.controller.admission import AdmissionGate, AdmissionRequest
from ictc.controller.routing import (
    GatedRoutingController,
    RecordingRoutingActuator,
    ThresholdRoutingPolicy,
)
from ictc.domain.enums import ActionKind, ContractState, ReasonCode, SafetyDisposition
from ictc.domain.models import (
    CausalEffectEstimate,
    NetworkAction,
    SafetyDecision,
    UncertaintyInterval,
)
from ictc.simulation.environment import (
    ControllerTrace,
    OracleRecord,
    OracleTrace,
    VisibleTelemetryRecord,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


def visible(
    path_id: str, occupancy: int | None, *, missing: bool = False
) -> VisibleTelemetryRecord:
    return VisibleTelemetryRecord(
        run_id="run-1",
        sample_tick=4,
        available_tick=5,
        observed_timestamp_tick=4,
        path_id=path_id,
        queue_occupancy=occupancy,
        utilization=0.5 if occupancy is not None else None,
        sample_age_ticks=1,
        measurement_delay_ticks=1,
        clock_skew_ticks=0,
        stale=False,
        missing=missing,
        source_id=f"source:{path_id}",
        provenance_digest="a" * 64,
    )


def trace(*, missing_candidate: bool = False) -> ControllerTrace:
    return ControllerTrace(
        run_id="run-1",
        records=(
            visible("path-a", 10),
            visible("path-b", None if missing_candidate else 2, missing=missing_candidate),
        ),
    )


def identified(value: bool = True) -> IdentifiabilityResult:
    return IdentifiabilityResult(
        identified=value,
        reason=(
            IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
            if value
            else IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
        ),
        treatment="RouteAction",
        outcome="Latency",
        available_variables=("RouteAction", "Latency"),
        valid_adjustment_sets=((),) if value else (),
    )


def successful_estimation() -> EstimationResult:
    return EstimationResult(
        estimable=True,
        reason=EstimationReason.ESTIMATED,
        adjustment_set=(),
        sample_size=10,
        strata=(),
        estimate=CausalEffectEstimate(
            estimand="effect",
            point_estimate=-2,
            unit="ms",
            uncertainty=UncertaintyInterval(lower=-2.2, upper=-1.8, confidence_level=0.95),
            estimator_name="fixture",
        ),
        standard_error=0.1,
    )


def admission_for(proposal: NetworkAction, *, identifiable: bool = True) -> AdmissionRequest:
    return AdmissionRequest(
        proposed_action=proposal,
        fallback_action=NetworkAction(action_id="fallback", kind=ActionKind.KEEP_CURRENT_PATH),
        graph_id="graph-v1",
        identification=identified(identifiable),
        contract_id="contract-1",
        contract_state=ContractState.SATISFIED,
        estimation=successful_estimation(),
        maximum_uncertainty_width=1,
        safety=SafetyDecision(disposition=SafetyDisposition.SAFE),
        policy_permitted=True,
        remediation_possible=False,
        deadline=NOW + timedelta(seconds=1),
        evaluated_at=NOW,
    )


def test_policy_proposes_reroute_from_visible_queues_and_keep_on_missing() -> None:
    policy = ThresholdRoutingPolicy("path-a", "path-b", minimum_queue_improvement=3)
    proposed = policy.propose(trace(), decision_tick=5)
    assert proposed.kind is ActionKind.REROUTE_TO_PATH
    assert proposed.path_id == "path-b"
    assert policy.propose(trace(), decision_tick=5) == proposed
    assert (
        policy.propose(trace(missing_candidate=True), decision_tick=5).kind
        is ActionKind.KEEP_CURRENT_PATH
    )


def test_gate_is_between_proposal_and_execution_and_fallback_is_explicit() -> None:
    actuator = RecordingRoutingActuator()
    controller = GatedRoutingController(
        policy=ThresholdRoutingPolicy("path-a", "path-b"),
        gate=AdmissionGate(),
        actuator=actuator,
    )
    rejected = controller.decide_and_execute(
        trace(),
        decision_tick=5,
        build_admission_request=lambda proposal: admission_for(proposal, identifiable=False),
    )
    assert rejected.proposal.kind is ActionKind.REROUTE_TO_PATH
    assert rejected.executed_action.kind is ActionKind.KEEP_CURRENT_PATH
    assert rejected.fallback_used
    assert ReasonCode.FALLBACK_EXECUTED in rejected.execution_reason_codes
    assert actuator.executed_actions == [rejected.executed_action]

    admitted = controller.decide_and_execute(
        trace(),
        decision_tick=5,
        build_admission_request=admission_for,
    )
    assert admitted.executed_action.kind is ActionKind.REROUTE_TO_PATH
    assert not admitted.fallback_used


def test_oracle_trace_cannot_satisfy_controller_policy_interface() -> None:
    oracle = OracleTrace(
        run_id="oracle",
        records=(
            OracleRecord(
                run_id="oracle",
                tick=0,
                phase="arrivals",
                path_id="path-a",
                selected_path=True,
                true_foreground_arrivals=1,
                true_cross_arrivals=1,
                true_service_capacity=1,
                true_queue_occupancy=1,
                true_foreground_served=1,
                true_cross_served=0,
                true_foreground_dropped=0,
                true_cross_dropped=0,
            ),
        ),
    )
    policy = ThresholdRoutingPolicy("path-a", "path-b")
    with pytest.raises(AttributeError):
        policy.propose(cast(ControllerTrace, oracle), decision_tick=0)


def test_mismatched_admission_binding_is_rejected() -> None:
    controller = GatedRoutingController(
        policy=ThresholdRoutingPolicy("path-a", "path-b"),
        gate=AdmissionGate(),
        actuator=RecordingRoutingActuator(),
    )
    with pytest.raises(ValueError, match="bind"):
        controller.decide_and_execute(
            trace(),
            decision_tick=5,
            build_admission_request=lambda proposal: replace(
                admission_for(proposal),
                proposed_action=NetworkAction(
                    action_id="different", kind=ActionKind.KEEP_CURRENT_PATH
                ),
            ),
        )
