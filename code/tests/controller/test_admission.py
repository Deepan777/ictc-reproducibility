"""Executable-admission separation and evidence tests."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from ictc.causal.estimation import EstimationReason, EstimationResult
from ictc.causal.identification import IdentifiabilityReason, IdentifiabilityResult
from ictc.controller.admission import AdmissionGate, AdmissionRequest
from ictc.domain.enums import (
    ActionKind,
    AdmissionDecision,
    ContractState,
    ReasonCode,
    SafetyDisposition,
)
from ictc.domain.models import (
    CausalEffectEstimate,
    NetworkAction,
    SafetyDecision,
    UncertaintyInterval,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=UTC)


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
        minimal_adjustment_sets=((),) if value else (),
    )


def estimation(
    *, point: float = -2.0, lower: float = -2.2, upper: float = -1.8
) -> EstimationResult:
    estimate = CausalEffectEstimate(
        estimand="reroute-minus-keep latency",
        point_estimate=point,
        unit="ms",
        uncertainty=UncertaintyInterval(lower=lower, upper=upper, confidence_level=0.95),
        estimator_name="fixture",
    )
    return EstimationResult(
        estimable=True,
        reason=EstimationReason.ESTIMATED,
        adjustment_set=(),
        sample_size=20,
        strata=(),
        estimate=estimate,
        standard_error=0.1,
    )


def failed_estimation(
    reason: EstimationReason = EstimationReason.INSUFFICIENT_ACTION_SUPPORT,
) -> EstimationResult:
    return EstimationResult(
        estimable=False,
        reason=reason,
        adjustment_set=(),
        sample_size=2,
        strata=(),
    )


def action(kind: ActionKind) -> NetworkAction:
    return NetworkAction(
        action_id=f"action:{kind.value}",
        kind=kind,
        path_id="path-b" if kind is ActionKind.REROUTE_TO_PATH else None,
    )


def request() -> AdmissionRequest:
    return AdmissionRequest(
        proposed_action=action(ActionKind.REROUTE_TO_PATH),
        fallback_action=action(ActionKind.KEEP_CURRENT_PATH),
        graph_id="graph-v1",
        identification=identified(),
        contract_id="contract-1",
        contract_state=ContractState.SATISFIED,
        estimation=estimation(),
        maximum_uncertainty_width=1.0,
        safety=SafetyDecision(disposition=SafetyDisposition.SAFE),
        policy_permitted=True,
        remediation_possible=True,
        deadline=NOW + timedelta(seconds=5),
        evaluated_at=NOW,
        evidence_digests=("digest-a",),
        maximum_acceptable_effect=0,
    )


def test_all_gates_pass_admits_with_reproducible_evidence() -> None:
    gate = AdmissionGate()
    first = gate.evaluate(request())
    second = gate.evaluate(request())
    assert first.decision is AdmissionDecision.ADMIT
    assert first.executable_action == first.proposed_action
    assert first.reason_codes[-1] is ReasonCode.ACTION_ADMITTED
    assert first.audit_digest == second.audit_digest
    assert first.evidence_record == second.evidence_record
    assert first.evidence_record.evidence_digests == ("digest-a",)


def test_narrow_interval_cannot_override_nonidentifiability_or_contract() -> None:
    gate = AdmissionGate()
    nonidentified = gate.evaluate(replace(request(), identification=identified(False)))
    assert nonidentified.decision is AdmissionDecision.DEFER
    assert ReasonCode.NOT_IDENTIFIABLE in nonidentified.reason_codes
    assert nonidentified.executable_action == request().fallback_action

    unsatisfied = gate.evaluate(replace(request(), contract_state=ContractState.ACTIVE))
    assert unsatisfied.decision is AdmissionDecision.DEFER
    assert ReasonCode.MISSING_REQUIRED_TELEMETRY in unsatisfied.reason_codes

    breached = gate.evaluate(replace(request(), contract_state=ContractState.BREACHED))
    assert breached.decision is AdmissionDecision.REJECT
    assert ReasonCode.CONTRACT_BREACHED in breached.reason_codes


def test_safety_policy_deadline_and_uncertainty_are_non_overridable() -> None:
    gate = AdmissionGate()
    unsafe = gate.evaluate(
        replace(
            request(),
            safety=SafetyDecision(
                disposition=SafetyDisposition.UNSAFE,
                invariant_ids=("max-loss",),
                reason_codes=(ReasonCode.SAFETY_INVARIANT_VIOLATED,),
            ),
        )
    )
    assert unsafe.decision is AdmissionDecision.REJECT
    assert ReasonCode.SAFETY_INVARIANT_VIOLATED in unsafe.reason_codes

    denied = gate.evaluate(replace(request(), policy_permitted=False))
    assert denied.decision is AdmissionDecision.REJECT
    assert ReasonCode.POLICY_VIOLATION in denied.reason_codes

    expired = gate.evaluate(replace(request(), evaluated_at=NOW + timedelta(seconds=6)))
    assert expired.decision is AdmissionDecision.REJECT
    assert ReasonCode.DEADLINE_EXCEEDED in expired.reason_codes

    wide = gate.evaluate(replace(request(), estimation=estimation(lower=-10, upper=10)))
    assert wide.decision is AdmissionDecision.DEFER
    assert ReasonCode.UNCERTAINTY_TOO_WIDE in wide.reason_codes


def test_estimation_failure_modify_and_unknown_safety_paths() -> None:
    gate = AdmissionGate()
    unsupported = gate.evaluate(replace(request(), estimation=failed_estimation()))
    assert unsupported.decision is AdmissionDecision.DEFER
    assert ReasonCode.INSUFFICIENT_ACTION_SUPPORT in unsupported.reason_codes

    modified = gate.evaluate(
        replace(request(), estimation=estimation(point=1, lower=0.8, upper=1.2))
    )
    assert modified.decision is AdmissionDecision.MODIFY
    assert modified.executable_action == request().fallback_action
    assert ReasonCode.ACTION_MODIFIED in modified.reason_codes

    unknown = gate.evaluate(
        replace(
            request(),
            safety=SafetyDecision(disposition=SafetyDisposition.UNKNOWN),
        )
    )
    assert unknown.decision is AdmissionDecision.DEFER


def test_admission_request_validation() -> None:
    with pytest.raises(ValueError, match="fallback"):
        replace(request(), fallback_action=action(ActionKind.REROUTE_TO_PATH))
    with pytest.raises(ValueError, match="uncertainty"):
        replace(request(), maximum_uncertainty_width=-1)
    with pytest.raises(ValueError, match="timezone"):
        replace(request(), evaluated_at=datetime(2026, 7, 11, 12, 0))
    with pytest.raises(ValueError, match="unique"):
        replace(request(), evidence_digests=("same", "same"))
