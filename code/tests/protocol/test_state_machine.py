"""Lifecycle tests for legal and illegal ICTC protocol transitions."""

from __future__ import annotations

import pytest

from ictc.domain.enums import ContractState, ReasonCode
from ictc.protocol.state_machine import ProtocolStateMachine, ProtocolTransitionError


def test_happy_path_reaches_release() -> None:
    machine = ProtocolStateMachine()
    for target in (
        ContractState.REQUIREMENTS_GENERATED,
        ContractState.OFFERS_RECEIVED,
        ContractState.COMMITTED,
        ContractState.ACTIVE,
        ContractState.SATISFIED,
        ContractState.RELEASED,
    ):
        machine.transition(target)
    assert machine.state is ContractState.RELEASED
    assert machine.terminal
    assert len(machine.history) == 6


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (ContractState.OFFERS_RECEIVED, ContractState.INFEASIBLE),
        (ContractState.COMMITTED, ContractState.EXPIRED),
        (ContractState.ACTIVE, ContractState.BREACHED),
        (ContractState.SATISFIED, ContractState.BREACHED_BEFORE_EXECUTION),
        (ContractState.PROPOSED, ContractState.CANCELLED),
        (ContractState.REQUIREMENTS_GENERATED, ContractState.CANCELLED),
        (ContractState.OFFERS_RECEIVED, ContractState.CANCELLED),
        (ContractState.COMMITTED, ContractState.CANCELLED),
        (ContractState.ACTIVE, ContractState.CANCELLED),
        (ContractState.SATISFIED, ContractState.CANCELLED),
        (ContractState.BREACHED, ContractState.CANCELLED),
    ],
)
def test_all_declared_alternative_transitions_are_valid(
    source: ContractState,
    target: ContractState,
) -> None:
    machine = ProtocolStateMachine(state=source)
    transition = machine.transition(target, reason_code=ReasonCode.CANCELLED)
    assert transition.source is source
    assert machine.state is target


def test_breach_can_only_release_after_active_path() -> None:
    machine = ProtocolStateMachine(state=ContractState.BREACHED)
    machine.transition(ContractState.RELEASED, reason_code=ReasonCode.CONTRACT_BREACHED)
    assert machine.terminal


def test_invalid_transition_does_not_change_state() -> None:
    machine = ProtocolStateMachine()
    with pytest.raises(ProtocolTransitionError, match="invalid ICTC transition"):
        machine.transition(ContractState.ACTIVE)
    assert machine.state is ContractState.PROPOSED
    assert machine.history == []
