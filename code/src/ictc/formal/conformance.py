"""Adapters that exhaustively compare production boundaries with the pure reference semantics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from ictc.causal.estimation import EstimationReason, EstimationResult
from ictc.causal.identification import IdentifiabilityReason, IdentifiabilityResult
from ictc.controller.admission import AdmissionGate, AdmissionRequest
from ictc.domain.enums import ActionKind, ContractState, ReasonCode, SafetyDisposition
from ictc.domain.models import (
    CausalEffectEstimate,
    NetworkAction,
    SafetyDecision,
    UncertaintyInterval,
)
from ictc.formal.semantics import (
    REFERENCE_TRANSITIONS,
    ActionDisposition,
    AdmissionFacts,
    all_admission_facts,
    reference_admission,
    validate_reference_invariants,
)
from ictc.protocol.state_machine import ProtocolStateMachine, ProtocolTransitionError

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ConformanceSummary:
    admission_cases: int
    admission_mismatches: int
    invariant_violations: int
    deterministic_digest_mismatches: int
    lifecycle_cases: int
    lifecycle_mismatches: int
    decision_counts: tuple[tuple[str, int], ...]
    decisive_gate_counts: tuple[tuple[str, int], ...]


def _action(kind: ActionKind) -> NetworkAction:
    return NetworkAction(
        action_id=f"n03:{kind.value}",
        kind=kind,
        path_id="backup" if kind is ActionKind.REROUTE_TO_PATH else None,
    )


def _identification(identified: bool) -> IdentifiabilityResult:
    return IdentifiabilityResult(
        identified=identified,
        reason=(
            IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
            if identified
            else IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
        ),
        treatment="A",
        outcome="Y",
        available_variables=("A", "Y"),
        valid_adjustment_sets=((),) if identified else (),
        minimal_adjustment_sets=((),) if identified else (),
    )


def _estimation(facts: AdmissionFacts) -> EstimationResult:
    if not facts.estimable:
        return EstimationResult(
            estimable=False,
            reason=EstimationReason.INSUFFICIENT_ACTION_SUPPORT,
            adjustment_set=(),
            sample_size=0,
            strata=(),
        )
    point = 1.0 if facts.effect_within_limit is False else -1.0
    width = 0.5 if facts.uncertainty_acceptable else 2.0
    return EstimationResult(
        estimable=True,
        reason=EstimationReason.ESTIMATED,
        adjustment_set=(),
        sample_size=20,
        strata=(),
        estimate=CausalEffectEstimate(
            estimand="reroute-minus-keep latency",
            point_estimate=point,
            unit="ms",
            uncertainty=UncertaintyInterval(
                lower=point - width / 2,
                upper=point + width / 2,
                confidence_level=0.95,
            ),
            estimator_name="n03-conformance-fixture",
        ),
        standard_error=0.1,
    )


def admission_request(facts: AdmissionFacts) -> AdmissionRequest:
    safety = SafetyDecision(
        disposition=facts.safety,
        invariant_ids=("n03-invariant",) if facts.safety is SafetyDisposition.UNSAFE else (),
        reason_codes=(ReasonCode.SAFETY_INVARIANT_VIOLATED,)
        if facts.safety is SafetyDisposition.UNSAFE
        else (),
    )
    deadline = NOW + timedelta(seconds=1)
    return AdmissionRequest(
        proposed_action=_action(ActionKind.REROUTE_TO_PATH),
        fallback_action=_action(ActionKind.KEEP_CURRENT_PATH),
        graph_id="n03-graph-v1",
        identification=_identification(facts.identified),
        contract_id="n03-contract-v1",
        contract_state=facts.contract_state,
        estimation=_estimation(facts),
        maximum_uncertainty_width=1.0,
        safety=safety,
        policy_permitted=facts.policy_permitted,
        remediation_possible=facts.remediation_possible,
        deadline=deadline,
        evaluated_at=NOW if facts.deadline_met else deadline + timedelta(milliseconds=1),
        evidence_digests=("n03-evidence",),
        maximum_acceptable_effect=(None if facts.effect_within_limit is None else 0.0),
    )


def run_conformance() -> ConformanceSummary:
    gate = AdmissionGate()
    admission_mismatches = 0
    invariant_violations = 0
    digest_mismatches = 0
    decision_counts: Counter[str] = Counter()
    gate_counts: Counter[str] = Counter()
    facts_space = all_admission_facts()
    for facts in facts_space:
        expected = reference_admission(facts)
        request = admission_request(facts)
        actual = gate.evaluate(request)
        repeated = gate.evaluate(request)
        actual_action = (
            ActionDisposition.PROPOSED
            if actual.executable_action == request.proposed_action
            else ActionDisposition.FALLBACK
        )
        expected_fallback_reason = expected.decision.value != "ADMIT"
        actual_fallback_reason = ReasonCode.FALLBACK_REQUIRED in actual.reason_codes
        if (
            actual.decision is not expected.decision
            or actual_action is not expected.action
            or expected.required_reason not in actual.reason_codes
            or actual_fallback_reason != expected_fallback_reason
        ):
            admission_mismatches += 1
        if actual.audit_digest != repeated.audit_digest:
            digest_mismatches += 1
        invariant_violations += len(validate_reference_invariants(facts, expected))
        decision_counts[expected.decision.value] += 1
        gate_counts[expected.decisive_gate.value] += 1

    lifecycle_cases = 0
    lifecycle_mismatches = 0
    for source in ContractState:
        for target in ContractState:
            lifecycle_cases += 1
            expected_allowed = target in REFERENCE_TRANSITIONS[source]
            machine = ProtocolStateMachine(state=source)
            try:
                machine.transition(target, occurred_at=NOW)
            except ProtocolTransitionError:
                actual_allowed = False
            else:
                actual_allowed = True
            if actual_allowed != expected_allowed:
                lifecycle_mismatches += 1

    return ConformanceSummary(
        admission_cases=len(facts_space),
        admission_mismatches=admission_mismatches,
        invariant_violations=invariant_violations,
        deterministic_digest_mismatches=digest_mismatches,
        lifecycle_cases=lifecycle_cases,
        lifecycle_mismatches=lifecycle_mismatches,
        decision_counts=tuple(sorted(decision_counts.items())),
        decisive_gate_counts=tuple(sorted(gate_counts.items())),
    )
