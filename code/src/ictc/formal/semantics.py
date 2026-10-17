"""Implementation-independent finite semantics for ICTC admission and evaluation order.

This module deliberately imports domain enums only. It does not import production controller,
protocol, planner, pilot, outcome, or oracle implementations. Exhaustive tests compare this
reference relation with those components at their public boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from itertools import product

from ictc.domain.enums import (
    AdmissionDecision,
    ContractState,
    ReasonCode,
    SafetyDisposition,
)


class ActionDisposition(StrEnum):
    PROPOSED = "PROPOSED"
    FALLBACK = "FALLBACK"


class DecisiveGate(StrEnum):
    DEADLINE = "DEADLINE"
    SAFETY = "SAFETY"
    POLICY = "POLICY"
    IDENTIFICATION = "IDENTIFICATION"
    CONTRACT = "CONTRACT"
    ESTIMATION = "ESTIMATION"
    UNCERTAINTY = "UNCERTAINTY"
    EFFECT_THRESHOLD = "EFFECT_THRESHOLD"
    ALL_EXECUTABLE_GATES = "ALL_EXECUTABLE_GATES"


_TERMINAL_CONTRACTS = frozenset(
    {
        ContractState.INFEASIBLE,
        ContractState.EXPIRED,
        ContractState.BREACHED,
        ContractState.BREACHED_BEFORE_EXECUTION,
        ContractState.CANCELLED,
        ContractState.RELEASED,
    }
)


@dataclass(frozen=True, slots=True)
class AdmissionFacts:
    """Finite abstraction of every predicate used by executable admission."""

    deadline_met: bool
    safety: SafetyDisposition
    policy_permitted: bool
    identified: bool
    contract_state: ContractState
    estimable: bool
    uncertainty_acceptable: bool
    effect_within_limit: bool | None
    remediation_possible: bool


@dataclass(frozen=True, slots=True)
class ReferenceAdmission:
    decision: AdmissionDecision
    action: ActionDisposition
    decisive_gate: DecisiveGate
    required_reason: ReasonCode


def _defer_or_reject(remediation_possible: bool) -> AdmissionDecision:
    return AdmissionDecision.DEFER if remediation_possible else AdmissionDecision.REJECT


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


def reference_admission(facts: AdmissionFacts) -> ReferenceAdmission:
    """Evaluate the declared gate priority as a pure total relation."""
    if not facts.deadline_met:
        return ReferenceAdmission(
            AdmissionDecision.REJECT,
            ActionDisposition.FALLBACK,
            DecisiveGate.DEADLINE,
            ReasonCode.DEADLINE_EXCEEDED,
        )
    if facts.safety is SafetyDisposition.UNSAFE:
        return ReferenceAdmission(
            AdmissionDecision.REJECT,
            ActionDisposition.FALLBACK,
            DecisiveGate.SAFETY,
            ReasonCode.SAFETY_INVARIANT_VIOLATED,
        )
    if not facts.policy_permitted:
        return ReferenceAdmission(
            AdmissionDecision.REJECT,
            ActionDisposition.FALLBACK,
            DecisiveGate.POLICY,
            ReasonCode.POLICY_VIOLATION,
        )
    if not facts.identified:
        return ReferenceAdmission(
            _defer_or_reject(facts.remediation_possible),
            ActionDisposition.FALLBACK,
            DecisiveGate.IDENTIFICATION,
            ReasonCode.NOT_IDENTIFIABLE,
        )
    if facts.contract_state is not ContractState.SATISFIED:
        terminal = facts.contract_state in _TERMINAL_CONTRACTS
        return ReferenceAdmission(
            AdmissionDecision.REJECT if terminal else _defer_or_reject(facts.remediation_possible),
            ActionDisposition.FALLBACK,
            DecisiveGate.CONTRACT,
            _contract_reason(facts.contract_state),
        )
    if not facts.estimable:
        return ReferenceAdmission(
            _defer_or_reject(facts.remediation_possible),
            ActionDisposition.FALLBACK,
            DecisiveGate.ESTIMATION,
            ReasonCode.INSUFFICIENT_ACTION_SUPPORT,
        )
    if not facts.uncertainty_acceptable:
        return ReferenceAdmission(
            _defer_or_reject(facts.remediation_possible),
            ActionDisposition.FALLBACK,
            DecisiveGate.UNCERTAINTY,
            ReasonCode.UNCERTAINTY_TOO_WIDE,
        )
    if facts.safety is SafetyDisposition.UNKNOWN:
        return ReferenceAdmission(
            _defer_or_reject(facts.remediation_possible),
            ActionDisposition.FALLBACK,
            DecisiveGate.SAFETY,
            ReasonCode.SAFETY_INVARIANT_VIOLATED,
        )
    if facts.effect_within_limit is False:
        return ReferenceAdmission(
            AdmissionDecision.MODIFY,
            ActionDisposition.FALLBACK,
            DecisiveGate.EFFECT_THRESHOLD,
            ReasonCode.ACTION_MODIFIED,
        )
    return ReferenceAdmission(
        AdmissionDecision.ADMIT,
        ActionDisposition.PROPOSED,
        DecisiveGate.ALL_EXECUTABLE_GATES,
        ReasonCode.ACTION_ADMITTED,
    )


def all_admission_facts() -> tuple[AdmissionFacts, ...]:
    """Enumerate the complete declared finite admission abstraction."""
    return tuple(
        AdmissionFacts(*values)
        for values in product(
            (False, True),
            tuple(SafetyDisposition),
            (False, True),
            (False, True),
            tuple(ContractState),
            (False, True),
            (False, True),
            (None, False, True),
            (False, True),
        )
    )


def validate_reference_invariants(
    facts: AdmissionFacts, outcome: ReferenceAdmission
) -> tuple[str, ...]:
    """Return named violations of the executable-set and fallback definitions."""
    executable = (
        facts.deadline_met
        and facts.safety is SafetyDisposition.SAFE
        and facts.policy_permitted
        and facts.identified
        and facts.contract_state is ContractState.SATISFIED
        and facts.estimable
        and facts.uncertainty_acceptable
        and facts.effect_within_limit is not False
    )
    violations: list[str] = []
    if (outcome.decision is AdmissionDecision.ADMIT) != executable:
        violations.append("ADMIT_IFF_EXECUTABLE")
    expected_action = (
        ActionDisposition.PROPOSED
        if outcome.decision is AdmissionDecision.ADMIT
        else ActionDisposition.FALLBACK
    )
    if outcome.action is not expected_action:
        violations.append("NON_ADMIT_REQUIRES_FALLBACK")
    if outcome.decision is AdmissionDecision.DEFER and not facts.remediation_possible:
        violations.append("DEFER_REQUIRES_REMEDIATION")
    return tuple(violations)


REFERENCE_TRANSITIONS: dict[ContractState, frozenset[ContractState]] = {
    ContractState.PROPOSED: frozenset(
        {ContractState.REQUIREMENTS_GENERATED, ContractState.CANCELLED}
    ),
    ContractState.REQUIREMENTS_GENERATED: frozenset(
        {ContractState.OFFERS_RECEIVED, ContractState.CANCELLED}
    ),
    ContractState.OFFERS_RECEIVED: frozenset(
        {ContractState.COMMITTED, ContractState.INFEASIBLE, ContractState.CANCELLED}
    ),
    ContractState.COMMITTED: frozenset(
        {ContractState.ACTIVE, ContractState.EXPIRED, ContractState.CANCELLED}
    ),
    ContractState.ACTIVE: frozenset(
        {ContractState.SATISFIED, ContractState.BREACHED, ContractState.CANCELLED}
    ),
    ContractState.SATISFIED: frozenset(
        {
            ContractState.RELEASED,
            ContractState.BREACHED_BEFORE_EXECUTION,
            ContractState.CANCELLED,
        }
    ),
    ContractState.BREACHED: frozenset({ContractState.RELEASED, ContractState.CANCELLED}),
    ContractState.RELEASED: frozenset(),
    ContractState.INFEASIBLE: frozenset(),
    ContractState.EXPIRED: frozenset(),
    ContractState.BREACHED_BEFORE_EXECUTION: frozenset(),
    ContractState.CANCELLED: frozenset(),
}


def maximal_lifecycle_paths() -> tuple[tuple[ContractState, ...], ...]:
    """Enumerate every maximal legal lifecycle path from PROPOSED to a terminal state."""
    paths: list[tuple[ContractState, ...]] = []

    def visit(path: tuple[ContractState, ...]) -> None:
        targets = REFERENCE_TRANSITIONS[path[-1]]
        if not targets:
            paths.append(path)
            return
        for target in sorted(targets, key=lambda item: item.value):
            visit((*path, target))

    visit((ContractState.PROPOSED,))
    return tuple(paths)


class TraceEvent(StrEnum):
    QUERY_DECLARED = "QUERY_DECLARED"
    CONTRACT_PLANNED = "CONTRACT_PLANNED"
    CONTRACT_SATISFIED = "CONTRACT_SATISFIED"
    DECISION_PERSISTED = "DECISION_PERSISTED"
    BREACH_RECORDED = "BREACH_RECORDED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    ORACLE_LABEL_ATTACHED = "ORACLE_LABEL_ATTACHED"


class TraceViolation(StrEnum):
    MISSING_OR_DUPLICATE_DECISION = "MISSING_OR_DUPLICATE_DECISION"
    MISSING_OR_DUPLICATE_ACTION = "MISSING_OR_DUPLICATE_ACTION"
    ACTION_BEFORE_DECISION = "ACTION_BEFORE_DECISION"
    ORACLE_BEFORE_DECISION = "ORACLE_BEFORE_DECISION"
    ADMIT_WITHOUT_PRIOR_CONTRACT = "ADMIT_WITHOUT_PRIOR_CONTRACT"
    PROPOSED_ACTION_WITHOUT_ADMIT = "PROPOSED_ACTION_WITHOUT_ADMIT"
    PROPOSED_ACTION_AFTER_BREACH = "PROPOSED_ACTION_AFTER_BREACH"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"
    ACTION_FIELD_ON_NON_EXECUTION_EVENT = "ACTION_FIELD_ON_NON_EXECUTION_EVENT"


@dataclass(frozen=True, slots=True)
class TraceRecord:
    event: TraceEvent
    action: ActionDisposition | None = None

    def __post_init__(self) -> None:
        if self.event is TraceEvent.ACTION_EXECUTED:
            if self.action is None:
                raise ValueError("action execution event requires an action disposition")
        elif self.action is not None:
            raise ValueError("only action execution events may carry an action disposition")


@dataclass(frozen=True, slots=True)
class CrossLayerTrace:
    decision: AdmissionDecision
    records: tuple[TraceRecord, ...]


def validate_trace(trace: CrossLayerTrace) -> tuple[TraceViolation, ...]:
    """Check temporal leakage and fail-closed action invariants on a persisted trace."""
    events = tuple(record.event for record in trace.records)
    decision_indexes = [
        index for index, event in enumerate(events) if event is TraceEvent.DECISION_PERSISTED
    ]
    action_indexes = [
        index for index, event in enumerate(events) if event is TraceEvent.ACTION_EXECUTED
    ]
    violations: list[TraceViolation] = []
    if len(decision_indexes) != 1:
        violations.append(TraceViolation.MISSING_OR_DUPLICATE_DECISION)
    if len(action_indexes) != 1:
        violations.append(TraceViolation.MISSING_OR_DUPLICATE_ACTION)
    if not decision_indexes or not action_indexes:
        return tuple(violations)
    decision_index = decision_indexes[0]
    action_index = action_indexes[0]
    if action_index < decision_index:
        violations.append(TraceViolation.ACTION_BEFORE_DECISION)
    oracle_indexes = [
        index for index, event in enumerate(events) if event is TraceEvent.ORACLE_LABEL_ATTACHED
    ]
    if any(index < decision_index for index in oracle_indexes):
        violations.append(TraceViolation.ORACLE_BEFORE_DECISION)
    contract_indexes = [
        index for index, event in enumerate(events) if event is TraceEvent.CONTRACT_SATISFIED
    ]
    if trace.decision is AdmissionDecision.ADMIT and not any(
        index < decision_index for index in contract_indexes
    ):
        violations.append(TraceViolation.ADMIT_WITHOUT_PRIOR_CONTRACT)
    executed = trace.records[action_index].action
    breach_between = any(
        trace.records[index].event is TraceEvent.BREACH_RECORDED
        for index in range(decision_index + 1, action_index)
    )
    if executed is ActionDisposition.PROPOSED:
        if trace.decision is not AdmissionDecision.ADMIT:
            violations.append(TraceViolation.PROPOSED_ACTION_WITHOUT_ADMIT)
        if breach_between:
            violations.append(TraceViolation.PROPOSED_ACTION_AFTER_BREACH)
    elif trace.decision is not AdmissionDecision.ADMIT or breach_between:
        pass
    else:
        violations.append(TraceViolation.FALLBACK_REQUIRED)
    return tuple(dict.fromkeys(violations))
