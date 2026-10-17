"""Validated ICTC protocol lifecycle transitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from ictc.domain.enums import ContractState, ReasonCode


class ProtocolTransitionError(ValueError):
    """Raised when a requested ICTC lifecycle transition is not admissible."""


_ALLOWED_TRANSITIONS: dict[ContractState, frozenset[ContractState]] = {
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
        {ContractState.RELEASED, ContractState.BREACHED_BEFORE_EXECUTION, ContractState.CANCELLED}
    ),
    ContractState.BREACHED: frozenset({ContractState.RELEASED, ContractState.CANCELLED}),
    ContractState.RELEASED: frozenset(),
    ContractState.INFEASIBLE: frozenset(),
    ContractState.EXPIRED: frozenset(),
    ContractState.BREACHED_BEFORE_EXECUTION: frozenset(),
    ContractState.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class ProtocolTransition:
    """An auditable transition between two protocol states."""

    source: ContractState
    target: ContractState
    occurred_at: datetime
    reason_code: ReasonCode | None = None


@dataclass(slots=True)
class ProtocolStateMachine:
    """Small, explicit state machine used by later protocol implementations."""

    state: ContractState = ContractState.PROPOSED
    history: list[ProtocolTransition] = field(default_factory=list)

    @property
    def terminal(self) -> bool:
        """Whether no further lifecycle transition is admissible."""
        return not _ALLOWED_TRANSITIONS[self.state]

    def allowed_targets(self) -> frozenset[ContractState]:
        """Return the immutable set of legal targets from the current state."""
        return _ALLOWED_TRANSITIONS[self.state]

    def transition(
        self,
        target: ContractState,
        *,
        reason_code: ReasonCode | None = None,
        occurred_at: datetime | None = None,
    ) -> ProtocolTransition:
        """Transition once or raise without changing state for an invalid request."""
        if target not in self.allowed_targets():
            raise ProtocolTransitionError(
                f"invalid ICTC transition: {self.state.value} -> {target.value}"
            )
        transition = ProtocolTransition(
            source=self.state,
            target=target,
            occurred_at=occurred_at or datetime.now(UTC),
            reason_code=reason_code,
        )
        self.state = target
        self.history.append(transition)
        return transition
