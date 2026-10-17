"""Admission gate that preserves identification, estimation, safety, and policy boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from ictc.causal.estimation import EstimationReason, EstimationResult
from ictc.causal.identification import IdentifiabilityResult
from ictc.domain.enums import (
    ActionKind,
    AdmissionDecision,
    ContractState,
    ReasonCode,
    SafetyDisposition,
)
from ictc.domain.models import CausalEffectEstimate, EvidenceRecord, NetworkAction, SafetyDecision


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    proposed_action: NetworkAction
    fallback_action: NetworkAction
    graph_id: str
    identification: IdentifiabilityResult
    contract_id: str | None
    contract_state: ContractState
    estimation: EstimationResult
    maximum_uncertainty_width: float
    safety: SafetyDecision
    policy_permitted: bool
    remediation_possible: bool
    deadline: datetime
    evaluated_at: datetime
    evidence_digests: tuple[str, ...] = ()
    maximum_acceptable_effect: float | None = None

    def __post_init__(self) -> None:
        if not self.graph_id:
            raise ValueError("graph ID is required")
        if self.fallback_action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("routing fallback must retain the current path")
        if self.maximum_uncertainty_width < 0:
            raise ValueError("maximum uncertainty width cannot be negative")
        for timestamp in (self.deadline, self.evaluated_at):
            if timestamp.tzinfo is None or timestamp.utcoffset() is None:
                raise ValueError("admission timestamps must be timezone-aware")
        if len(self.evidence_digests) != len(set(self.evidence_digests)):
            raise ValueError("evidence digests must be unique")


@dataclass(frozen=True, slots=True)
class AdmissionOutcome:
    decision: AdmissionDecision
    proposed_action: NetworkAction
    executable_action: NetworkAction | None
    reason_codes: tuple[ReasonCode, ...]
    evidence_record: EvidenceRecord
    estimate: CausalEffectEstimate | None
    contract_state: ContractState
    audit_digest: str


def _estimation_reason(result: EstimationResult) -> ReasonCode:
    if result.reason is EstimationReason.INSUFFICIENT_ACTION_SUPPORT:
        return ReasonCode.INSUFFICIENT_ACTION_SUPPORT
    if result.reason is EstimationReason.INSUFFICIENT_STRATUM_SAMPLE:
        return ReasonCode.INSUFFICIENT_SAMPLE_SIZE
    return ReasonCode.ESTIMATION_UNAVAILABLE


def _contract_reason(state: ContractState) -> ReasonCode:
    if state is ContractState.EXPIRED:
        return ReasonCode.CONTRACT_EXPIRED
    if state in {ContractState.BREACHED, ContractState.BREACHED_BEFORE_EXECUTION}:
        return ReasonCode.CONTRACT_BREACHED
    if state is ContractState.INFEASIBLE:
        return ReasonCode.CONTRACT_INFEASIBLE
    if state is ContractState.CANCELLED:
        return ReasonCode.CANCELLED
    return ReasonCode.MISSING_REQUIRED_TELEMETRY


def _canonical_payload(
    request: AdmissionRequest,
    decision: AdmissionDecision,
    reasons: tuple[ReasonCode, ...],
    executable_action: NetworkAction | None,
) -> dict[str, object]:
    estimate = request.estimation.estimate
    return {
        "action": request.proposed_action.model_dump(mode="json"),
        "contract_id": request.contract_id,
        "contract_state": request.contract_state.value,
        "decision": decision.value,
        "deadline": request.deadline.astimezone(UTC).isoformat(),
        "estimate": estimate.model_dump(mode="json") if estimate else None,
        "estimation": {
            "adjustment_set": list(request.estimation.adjustment_set),
            "detail": request.estimation.detail,
            "estimable": request.estimation.estimable,
            "reason": request.estimation.reason.value,
            "sample_size": request.estimation.sample_size,
            "strata": [
                {
                    "baseline_count": item.baseline_count,
                    "effect": item.effect,
                    "intervention_count": item.intervention_count,
                    "values": list(item.values),
                    "weight": item.weight,
                }
                for item in request.estimation.strata
            ],
        },
        "evaluated_at": request.evaluated_at.astimezone(UTC).isoformat(),
        "evidence_digests": list(request.evidence_digests),
        "executable_action": (
            executable_action.model_dump(mode="json") if executable_action else None
        ),
        "graph_id": request.graph_id,
        "fallback_action": request.fallback_action.model_dump(mode="json"),
        "identification": {
            "available_variables": list(request.identification.available_variables),
            "detail": request.identification.detail,
            "identified": request.identification.identified,
            "outcome": request.identification.outcome,
            "reason": request.identification.reason.value,
            "treatment": request.identification.treatment,
            "valid_adjustment_sets": [
                list(item) for item in request.identification.valid_adjustment_sets
            ],
        },
        "maximum_acceptable_effect": request.maximum_acceptable_effect,
        "maximum_uncertainty_width": request.maximum_uncertainty_width,
        "policy_permitted": request.policy_permitted,
        "remediation_possible": request.remediation_possible,
        "reasons": [reason.value for reason in reasons],
        "safety": request.safety.model_dump(mode="json"),
    }


class AdmissionGate:
    """Apply executable-admission predicates in a fixed, auditable order."""

    def evaluate(self, request: AdmissionRequest) -> AdmissionOutcome:
        decision: AdmissionDecision
        reasons: tuple[ReasonCode, ...]
        executable: NetworkAction | None
        terminal_contracts = {
            ContractState.INFEASIBLE,
            ContractState.EXPIRED,
            ContractState.BREACHED,
            ContractState.BREACHED_BEFORE_EXECUTION,
            ContractState.CANCELLED,
            ContractState.RELEASED,
        }
        if request.evaluated_at > request.deadline:
            decision = AdmissionDecision.REJECT
            reasons = (ReasonCode.DEADLINE_EXCEEDED, ReasonCode.ACTION_REJECTED)
            executable = request.fallback_action
        elif request.safety.disposition is SafetyDisposition.UNSAFE:
            decision = AdmissionDecision.REJECT
            reasons = (
                ReasonCode.SAFETY_INVARIANT_VIOLATED,
                ReasonCode.ACTION_REJECTED,
            )
            executable = request.fallback_action
        elif not request.policy_permitted:
            decision = AdmissionDecision.REJECT
            reasons = (ReasonCode.POLICY_VIOLATION, ReasonCode.ACTION_REJECTED)
            executable = request.fallback_action
        elif not request.identification.identified:
            decision = (
                AdmissionDecision.DEFER
                if request.remediation_possible
                else AdmissionDecision.REJECT
            )
            reasons = (
                ReasonCode.NOT_IDENTIFIABLE,
                ReasonCode.ACTION_DEFERRED
                if decision is AdmissionDecision.DEFER
                else ReasonCode.ACTION_REJECTED,
            )
            executable = request.fallback_action
        elif request.contract_state is not ContractState.SATISFIED:
            decision = (
                AdmissionDecision.REJECT
                if request.contract_state in terminal_contracts or not request.remediation_possible
                else AdmissionDecision.DEFER
            )
            reasons = (
                _contract_reason(request.contract_state),
                ReasonCode.ACTION_REJECTED
                if decision is AdmissionDecision.REJECT
                else ReasonCode.ACTION_DEFERRED,
            )
            executable = request.fallback_action
        elif not request.estimation.estimable:
            decision = (
                AdmissionDecision.DEFER
                if request.remediation_possible
                else AdmissionDecision.REJECT
            )
            reasons = (
                _estimation_reason(request.estimation),
                ReasonCode.ACTION_DEFERRED
                if decision is AdmissionDecision.DEFER
                else ReasonCode.ACTION_REJECTED,
            )
            executable = request.fallback_action
        else:
            assert request.estimation.estimate is not None
            estimate = request.estimation.estimate
            if estimate.uncertainty.width > request.maximum_uncertainty_width:
                decision = (
                    AdmissionDecision.DEFER
                    if request.remediation_possible
                    else AdmissionDecision.REJECT
                )
                reasons = (
                    ReasonCode.UNCERTAINTY_TOO_WIDE,
                    ReasonCode.ACTION_DEFERRED
                    if decision is AdmissionDecision.DEFER
                    else ReasonCode.ACTION_REJECTED,
                )
                executable = request.fallback_action
            elif request.safety.disposition is SafetyDisposition.UNKNOWN:
                decision = (
                    AdmissionDecision.DEFER
                    if request.remediation_possible
                    else AdmissionDecision.REJECT
                )
                reasons = (
                    ReasonCode.SAFETY_INVARIANT_VIOLATED,
                    ReasonCode.ACTION_DEFERRED
                    if decision is AdmissionDecision.DEFER
                    else ReasonCode.ACTION_REJECTED,
                )
                executable = request.fallback_action
            elif (
                request.maximum_acceptable_effect is not None
                and estimate.point_estimate > request.maximum_acceptable_effect
            ):
                decision = AdmissionDecision.MODIFY
                reasons = (
                    ReasonCode.IDENTIFIABLE,
                    ReasonCode.CONTRACT_SATISFIED,
                    ReasonCode.ESTIMABLE,
                    ReasonCode.UNCERTAINTY_ACCEPTABLE,
                    ReasonCode.SAFETY_PASSED,
                    ReasonCode.POLICY_PASSED,
                    ReasonCode.ACTION_MODIFIED,
                )
                executable = request.fallback_action
            else:
                decision = AdmissionDecision.ADMIT
                reasons = (
                    ReasonCode.IDENTIFIABLE,
                    ReasonCode.CONTRACT_SATISFIED,
                    ReasonCode.ESTIMABLE,
                    ReasonCode.UNCERTAINTY_ACCEPTABLE,
                    ReasonCode.SAFETY_PASSED,
                    ReasonCode.POLICY_PASSED,
                    ReasonCode.ACTION_ADMITTED,
                )
                executable = request.proposed_action

        if executable == request.fallback_action and decision is not AdmissionDecision.ADMIT:
            reasons = (*reasons, ReasonCode.FALLBACK_REQUIRED)
        payload = _canonical_payload(request, decision, reasons, executable)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        evidence = EvidenceRecord(
            evidence_record_id=f"admission:{digest[:24]}",
            action_id=request.proposed_action.action_id,
            contract_id=request.contract_id,
            graph_id=request.graph_id,
            decision=decision,
            reason_codes=reasons,
            safety_decision=request.safety,
            recorded_at=request.evaluated_at,
            evidence_digests=request.evidence_digests,
        )
        return AdmissionOutcome(
            decision=decision,
            proposed_action=request.proposed_action,
            executable_action=executable,
            reason_codes=reasons,
            evidence_record=evidence,
            estimate=request.estimation.estimate,
            contract_state=request.contract_state,
            audit_digest=digest,
        )
