"""Property tests for the contract lifecycle state machine."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ictc.domain.enums import ContractState
from ictc.protocol.state_machine import ProtocolStateMachine, ProtocolTransitionError


@given(st.sampled_from(list(ContractState)), st.sampled_from(list(ContractState)))
def test_invalid_transition_never_mutates_state(
    source: ContractState,
    target: ContractState,
) -> None:
    machine = ProtocolStateMachine(state=source)
    if target in machine.allowed_targets():
        machine.transition(target)
        assert machine.state is target
    else:
        with pytest.raises(ProtocolTransitionError):
            machine.transition(target)
        assert machine.state is source
        assert machine.history == []
