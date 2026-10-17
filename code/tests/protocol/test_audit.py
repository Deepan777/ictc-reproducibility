"""Integrity and lifecycle reconstruction tests for the M06 audit log."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ictc.domain.enums import ContractState
from ictc.protocol.audit import AuditEventType, AuditIntegrityError, AuditLog

NOW = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)


def append_transition(
    log: AuditLog, source: ContractState, target: ContractState, sequence: int
) -> None:
    log.append(
        event_type=AuditEventType.MESSAGE_APPLIED,
        lifecycle_id="contract-1",
        message_id=f"message-{sequence}",
        message_kind="TEST",
        occurred_at=NOW,
        state_before=source,
        state_after=target,
        payload_digest=f"{sequence:064x}",
    )


def test_audit_chain_round_trip_and_reconstruction(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = AuditLog(path)
    append_transition(log, ContractState.PROPOSED, ContractState.REQUIREMENTS_GENERATED, 1)
    append_transition(log, ContractState.REQUIREMENTS_GENERATED, ContractState.OFFERS_RECEIVED, 2)
    log.append(
        event_type=AuditEventType.DUPLICATE_RETURNED,
        lifecycle_id="contract-1",
        message_id="message-2",
        message_kind="TEST",
        occurred_at=NOW,
        state_before=ContractState.OFFERS_RECEIVED,
        state_after=ContractState.OFFERS_RECEIVED,
        payload_digest=f"{2:064x}",
    )

    events = AuditLog(path).read_verified()
    assert [event.sequence for event in events] == [1, 2, 3]
    assert events[0].previous_event_digest == "0" * 64
    assert events[1].previous_event_digest == events[0].event_digest
    assert log.reconstruct_lifecycle("contract-1") == (
        ContractState.PROPOSED,
        ContractState.REQUIREMENTS_GENERATED,
        ContractState.OFFERS_RECEIVED,
    )
    assert log.reconstruct_lifecycle("unknown") == ()
    assert all("payload" not in event.to_dict() for event in events)


def test_audit_tamper_and_truncation_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = AuditLog(path)
    append_transition(log, ContractState.PROPOSED, ContractState.REQUIREMENTS_GENERATED, 1)
    original = path.read_text(encoding="utf-8")
    value = json.loads(original)
    value["message_kind"] = "TAMPERED"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(AuditIntegrityError, match="event digest"):
        log.read_verified()

    path.write_text(original.rstrip("\n"), encoding="utf-8")
    with pytest.raises(AuditIntegrityError, match="incomplete"):
        AuditLog(path)


def test_audit_schema_sequence_and_state_discontinuity_are_detected(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    log = AuditLog(path)
    append_transition(log, ContractState.PROPOSED, ContractState.REQUIREMENTS_GENERATED, 1)
    append_transition(log, ContractState.ACTIVE, ContractState.SATISFIED, 2)
    with pytest.raises(AuditIntegrityError, match="discontinuous"):
        log.reconstruct_lifecycle("contract-1")

    value = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    value["unexpected"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(AuditIntegrityError, match="fields"):
        AuditLog(path)
