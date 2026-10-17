"""Append-only, digest-chained protocol audit events."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from ictc.domain.enums import ContractState

GENESIS_DIGEST = "0" * 64


class AuditIntegrityError(ValueError):
    """Raised when an audit log is malformed or its digest chain is broken."""


class AuditEventType(StrEnum):
    MESSAGE_APPLIED = "MESSAGE_APPLIED"
    DUPLICATE_RETURNED = "DUPLICATE_RETURNED"
    MESSAGE_REJECTED = "MESSAGE_REJECTED"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value with stable, compact ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class AuditEvent:
    sequence: int
    event_type: str
    lifecycle_id: str
    message_id: str
    message_kind: str
    occurred_at: str
    state_before: str | None
    state_after: str | None
    payload_digest: str
    response_digest: str | None
    reason: str | None
    previous_event_digest: str
    event_digest: str

    def unsigned_dict(self) -> dict[str, object]:
        value = asdict(self)
        value.pop("event_digest")
        return value

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def build(
        cls,
        *,
        sequence: int,
        event_type: AuditEventType,
        lifecycle_id: str,
        message_id: str,
        message_kind: str,
        occurred_at: datetime,
        state_before: ContractState | None,
        state_after: ContractState | None,
        payload_digest: str,
        response_digest: str | None,
        reason: str | None,
        previous_event_digest: str,
    ) -> AuditEvent:
        unsigned: dict[str, object] = {
            "sequence": sequence,
            "event_type": event_type.value,
            "lifecycle_id": lifecycle_id,
            "message_id": message_id,
            "message_kind": message_kind,
            "occurred_at": occurred_at.astimezone(UTC).isoformat(),
            "state_before": state_before.value if state_before else None,
            "state_after": state_after.value if state_after else None,
            "payload_digest": payload_digest,
            "response_digest": response_digest,
            "reason": reason,
            "previous_event_digest": previous_event_digest,
        }
        digest = sha256_hex(canonical_json(unsigned).encode("utf-8"))
        return cls(**unsigned, event_digest=digest)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> AuditEvent:
        expected = {field.name for field in cls.__dataclass_fields__.values()}
        if set(value) != expected:
            raise AuditIntegrityError("audit event fields do not match the schema")
        try:
            return cls(**value)
        except TypeError as error:
            raise AuditIntegrityError("audit event has invalid field types") from error


class AuditLog:
    """Single-writer JSON Lines log with a verifiable SHA-256 digest chain."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        events = self.read_verified()
        self._sequence = len(events)
        self._last_digest = events[-1].event_digest if events else GENESIS_DIGEST

    def read_verified(self) -> tuple[AuditEvent, ...]:
        if not self.path.exists():
            return ()
        events: list[AuditEvent] = []
        previous = GENESIS_DIGEST
        with self.path.open("r", encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, start=1):
                if not raw_line.endswith("\n"):
                    raise AuditIntegrityError(f"audit line {line_number} is incomplete")
                try:
                    value = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    raise AuditIntegrityError(
                        f"audit line {line_number} is invalid JSON"
                    ) from error
                if not isinstance(value, dict):
                    raise AuditIntegrityError(f"audit line {line_number} is not an object")
                event = AuditEvent.from_dict(value)
                if event.sequence != line_number:
                    raise AuditIntegrityError("audit sequence is not contiguous")
                if event.previous_event_digest != previous:
                    raise AuditIntegrityError("audit previous-event digest does not match")
                expected = sha256_hex(canonical_json(event.unsigned_dict()).encode("utf-8"))
                if event.event_digest != expected:
                    raise AuditIntegrityError("audit event digest does not match")
                events.append(event)
                previous = event.event_digest
        return tuple(events)

    def append(
        self,
        *,
        event_type: AuditEventType,
        lifecycle_id: str,
        message_id: str,
        message_kind: str,
        occurred_at: datetime,
        state_before: ContractState | None,
        state_after: ContractState | None,
        payload_digest: str,
        response_digest: str | None = None,
        reason: str | None = None,
    ) -> AuditEvent:
        with self._lock:
            event = AuditEvent.build(
                sequence=self._sequence + 1,
                event_type=event_type,
                lifecycle_id=lifecycle_id,
                message_id=message_id,
                message_kind=message_kind,
                occurred_at=occurred_at,
                state_before=state_before,
                state_after=state_after,
                payload_digest=payload_digest,
                response_digest=response_digest,
                reason=reason,
                previous_event_digest=self._last_digest,
            )
            encoded = (canonical_json(event.to_dict()) + "\n").encode("utf-8")
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | os.O_BINARY)
            try:
                written = os.write(descriptor, encoded)
                if written != len(encoded):
                    raise OSError("incomplete audit event append")
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            self._sequence = event.sequence
            self._last_digest = event.event_digest
            return event

    def reconstruct_lifecycle(self, lifecycle_id: str) -> tuple[ContractState, ...]:
        relevant = [event for event in self.read_verified() if event.lifecycle_id == lifecycle_id]
        if not relevant:
            return ()
        states: list[ContractState] = []
        for event in relevant:
            if event.state_before is None or event.state_after is None:
                continue
            source = ContractState(event.state_before)
            target = ContractState(event.state_after)
            if not states:
                states.append(source)
            if states[-1] != source:
                raise AuditIntegrityError("audit lifecycle states are discontinuous")
            if target != source:
                states.append(target)
        return tuple(states)
