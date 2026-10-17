"""Contract-scoped ICTC coordinator runtime with replay-safe processing."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol, cast

from ictc.domain.enums import ContractState, ReasonCode
from ictc.proto import ictc_pb2
from ictc.protocol.audit import AuditEventType, AuditLog, sha256_hex
from ictc.protocol.state_machine import ProtocolStateMachine, ProtocolTransitionError

PROTOCOL_VERSION = "1.0"


class ProtoMessage(Protocol):
    """Structural subset shared by generated protocol payloads."""

    metadata: ictc_pb2.ProtocolMetadata

    def HasField(self, field_name: str) -> bool: ...

    def SerializeToString(self, *, deterministic: bool = False) -> bytes: ...

    def ParseFromString(self, serialized: bytes) -> int: ...


class ProtocolErrorCode(StrEnum):
    INVALID_METADATA = "INVALID_METADATA"
    UNSUPPORTED_VERSION = "UNSUPPORTED_VERSION"
    REPLAY_REJECTED = "REPLAY_REJECTED"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    UNKNOWN_LIFECYCLE = "UNKNOWN_LIFECYCLE"
    LIFECYCLE_CONFLICT = "LIFECYCLE_CONFLICT"
    INVALID_TRANSITION = "INVALID_TRANSITION"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"


class ProtocolRequestError(ValueError):
    """A stable, transport-independent rejection of one protocol request."""

    def __init__(self, code: ProtocolErrorCode, detail: str) -> None:
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True, slots=True)
class ProcessedMessage:
    message_kind: str
    payload_digest: str
    response: ProtoMessage


@dataclass(slots=True)
class LifecycleRecord:
    lifecycle_id: str
    machine: ProtocolStateMachine = field(default_factory=ProtocolStateMachine)
    offer_ids: set[str] = field(default_factory=set)
    contract_id: str | None = None
    valid_until: datetime | None = None


def utc_now() -> datetime:
    return datetime.now(UTC)


def protobuf_digest(message: ProtoMessage, *, ignore_attempt: bool = False) -> str:
    """Digest deterministic Protobuf bytes, optionally normalizing retry attempt count."""
    canonical = clone_message(message)
    if ignore_attempt:
        canonical.metadata.delivery_attempt = 0
    return sha256_hex(canonical.SerializeToString(deterministic=True))


def clone_message[MessageT: ProtoMessage](message: MessageT) -> MessageT:
    clone = type(message)()
    clone.ParseFromString(message.SerializeToString(deterministic=True))
    return clone


def state_to_proto(state: ContractState) -> ictc_pb2.ContractState:
    return cast(
        ictc_pb2.ContractState,
        ictc_pb2.ContractState.Value(f"CONTRACT_STATE_{state.value}"),
    )


def state_from_proto(value: ictc_pb2.ContractState) -> ContractState:
    name = ictc_pb2.ContractState.Name(value)
    prefix = "CONTRACT_STATE_"
    if not name.startswith(prefix) or name == "CONTRACT_STATE_UNSPECIFIED":
        raise ProtocolRequestError(
            ProtocolErrorCode.INVALID_TRANSITION, "contract state is unspecified"
        )
    return ContractState(name.removeprefix(prefix))


def build_metadata(
    *, lifecycle_id: str, message_id: str, sent_at: datetime, delivery_attempt: int = 1
) -> ictc_pb2.ProtocolMetadata:
    return ictc_pb2.ProtocolMetadata(
        protocol_version=PROTOCOL_VERSION,
        message_id=message_id,
        lifecycle_id=lifecycle_id,
        sent_at_unix_ms=int(sent_at.timestamp() * 1000),
        delivery_attempt=delivery_attempt,
    )


def default_requirement(proposal: ictc_pb2.IctcPropose, *, now: datetime) -> ictc_pb2.IctcRequire:
    """Build a minimal transport fixture; causal requirements remain an injected concern."""
    lifecycle_id = proposal.metadata.lifecycle_id
    return ictc_pb2.IctcRequire(
        requirement_id=f"requirement:{proposal.proposal_id}",
        target_estimand=proposal.target_outcome.metric,
        max_freshness_ms=max(proposal.evaluation_horizon.horizon_ms, 1),
        max_clock_skew_ms=max(proposal.evaluation_horizon.horizon_ms, 1),
        minimum_trust=0.0,
        minimum_distinct_sources=1,
        maximum_uncertainty_width=0.0,
        metadata=build_metadata(
            lifecycle_id=lifecycle_id,
            message_id=f"response:{proposal.metadata.message_id}",
            sent_at=now,
        ),
    )


class CoordinatorRuntime:
    """Own lifecycle state while keeping transport, causal logic, and admission separate."""

    def __init__(
        self,
        audit_log: AuditLog,
        *,
        clock: Callable[[], datetime] = utc_now,
        replay_window: timedelta = timedelta(minutes=5),
        requirement_factory: Callable[[ictc_pb2.IctcPropose, datetime], ictc_pb2.IctcRequire]
        | None = None,
    ) -> None:
        if replay_window <= timedelta(0):
            raise ValueError("replay window must be positive")
        self.audit_log = audit_log
        self.clock = clock
        self.replay_window = replay_window
        self.requirement_factory = requirement_factory
        self._records: dict[str, LifecycleRecord] = {}
        self._processed: dict[tuple[str, str], ProcessedMessage] = {}
        self._lock = threading.RLock()

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("protocol clock must return a timezone-aware value")
        return now.astimezone(UTC)

    def _metadata(self, request: ProtoMessage) -> tuple[str, str, datetime]:
        if not request.HasField("metadata"):
            raise ProtocolRequestError(
                ProtocolErrorCode.INVALID_METADATA, "protocol metadata is required"
            )
        metadata = request.metadata
        if not metadata.message_id or not metadata.lifecycle_id or not metadata.protocol_version:
            raise ProtocolRequestError(
                ProtocolErrorCode.INVALID_METADATA, "metadata identifiers and version are required"
            )
        if metadata.protocol_version.split(".", maxsplit=1)[0] != "1":
            raise ProtocolRequestError(
                ProtocolErrorCode.UNSUPPORTED_VERSION,
                f"major version {metadata.protocol_version!r} is not supported",
            )
        if metadata.sent_at_unix_ms == 0 or metadata.delivery_attempt == 0:
            raise ProtocolRequestError(
                ProtocolErrorCode.INVALID_METADATA, "timestamp and delivery attempt are required"
            )
        sent_at = datetime.fromtimestamp(metadata.sent_at_unix_ms / 1000, tz=UTC)
        return metadata.lifecycle_id, metadata.message_id, sent_at

    def _record_for(self, lifecycle_id: str) -> LifecycleRecord:
        try:
            return self._records[lifecycle_id]
        except KeyError as error:
            raise ProtocolRequestError(ProtocolErrorCode.UNKNOWN_LIFECYCLE, lifecycle_id) from error

    def _status(
        self, record: LifecycleRecord, request_id: str, now: datetime
    ) -> ictc_pb2.IctcStatus:
        return ictc_pb2.IctcStatus(
            contract_id=record.contract_id or record.lifecycle_id,
            state=state_to_proto(record.machine.state),
            source_available=True,
            metadata=build_metadata(
                lifecycle_id=record.lifecycle_id,
                message_id=f"response:{request_id}",
                sent_at=now,
            ),
        )

    def _audit_rejection(
        self,
        *,
        lifecycle_id: str,
        message_id: str,
        message_kind: str,
        payload_digest: str,
        reason: str,
        now: datetime,
    ) -> None:
        record = self._records.get(lifecycle_id)
        state = record.machine.state if record else None
        self.audit_log.append(
            event_type=AuditEventType.MESSAGE_REJECTED,
            lifecycle_id=lifecycle_id,
            message_id=message_id,
            message_kind=message_kind,
            occurred_at=now,
            state_before=state,
            state_after=state,
            payload_digest=payload_digest,
            reason=reason,
        )

    def _process[RequestT: ProtoMessage, ResponseT: ProtoMessage](
        self,
        request: RequestT,
        message_kind: str,
        handler: Callable[[str, str, datetime], ResponseT],
    ) -> ResponseT:
        with self._lock:
            now = self._now()
            payload_digest = protobuf_digest(request, ignore_attempt=True)
            try:
                lifecycle_id, message_id, sent_at = self._metadata(request)
            except ProtocolRequestError as error:
                self._audit_rejection(
                    lifecycle_id="unbound",
                    message_id="missing",
                    message_kind=message_kind,
                    payload_digest=payload_digest,
                    reason=str(error),
                    now=now,
                )
                raise
            key = (lifecycle_id, message_id)
            prior = self._processed.get(key)
            if prior is not None:
                record = self._records.get(lifecycle_id)
                state = record.machine.state if record else None
                if prior.message_kind != message_kind or prior.payload_digest != payload_digest:
                    conflict_error = ProtocolRequestError(
                        ProtocolErrorCode.IDEMPOTENCY_CONFLICT,
                        "message ID was reused with different content or kind",
                    )
                    self._audit_rejection(
                        lifecycle_id=lifecycle_id,
                        message_id=message_id,
                        message_kind=message_kind,
                        payload_digest=payload_digest,
                        reason=str(conflict_error),
                        now=now,
                    )
                    raise conflict_error
                response = cast(ResponseT, clone_message(prior.response))
                self.audit_log.append(
                    event_type=AuditEventType.DUPLICATE_RETURNED,
                    lifecycle_id=lifecycle_id,
                    message_id=message_id,
                    message_kind=message_kind,
                    occurred_at=now,
                    state_before=state,
                    state_after=state,
                    payload_digest=payload_digest,
                    response_digest=protobuf_digest(response),
                )
                return response
            if abs(now - sent_at) > self.replay_window:
                replay_error = ProtocolRequestError(
                    ProtocolErrorCode.REPLAY_REJECTED,
                    "message timestamp is outside the replay window",
                )
                self._audit_rejection(
                    lifecycle_id=lifecycle_id,
                    message_id=message_id,
                    message_kind=message_kind,
                    payload_digest=payload_digest,
                    reason=str(replay_error),
                    now=now,
                )
                raise replay_error

            record_before = self._records.get(lifecycle_id)
            history_length = len(record_before.machine.history) if record_before else 0
            state_before = record_before.machine.state if record_before else None
            try:
                response = handler(lifecycle_id, message_id, now)
            except (ProtocolRequestError, ProtocolTransitionError) as raw_error:
                request_error = (
                    raw_error
                    if isinstance(raw_error, ProtocolRequestError)
                    else ProtocolRequestError(ProtocolErrorCode.INVALID_TRANSITION, str(raw_error))
                )
                self._audit_rejection(
                    lifecycle_id=lifecycle_id,
                    message_id=message_id,
                    message_kind=message_kind,
                    payload_digest=payload_digest,
                    reason=str(request_error),
                    now=now,
                )
                raise request_error from raw_error

            record_after = self._records.get(lifecycle_id)
            response_digest = protobuf_digest(response)
            transitions = (
                record_after.machine.history[history_length:] if record_after is not None else []
            )
            if transitions:
                for index, transition in enumerate(transitions):
                    self.audit_log.append(
                        event_type=AuditEventType.MESSAGE_APPLIED,
                        lifecycle_id=lifecycle_id,
                        message_id=message_id if index == 0 else f"{message_id}:internal:{index}",
                        message_kind=message_kind if index == 0 else "INTERNAL_TRANSITION",
                        occurred_at=transition.occurred_at,
                        state_before=transition.source,
                        state_after=transition.target,
                        payload_digest=payload_digest,
                        response_digest=response_digest,
                        reason=transition.reason_code.value if transition.reason_code else None,
                    )
            else:
                final_state = record_after.machine.state if record_after else state_before
                self.audit_log.append(
                    event_type=AuditEventType.MESSAGE_APPLIED,
                    lifecycle_id=lifecycle_id,
                    message_id=message_id,
                    message_kind=message_kind,
                    occurred_at=now,
                    state_before=final_state,
                    state_after=final_state,
                    payload_digest=payload_digest,
                    response_digest=response_digest,
                )
            self._processed[key] = ProcessedMessage(
                message_kind, payload_digest, clone_message(response)
            )
            return response

    def propose(self, request: ictc_pb2.IctcPropose) -> ictc_pb2.IctcRequire:
        def handler(lifecycle_id: str, _message_id: str, now: datetime) -> ictc_pb2.IctcRequire:
            if lifecycle_id in self._records:
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT, "lifecycle already exists"
                )
            record = LifecycleRecord(lifecycle_id=lifecycle_id)
            self._records[lifecycle_id] = record
            record.machine.transition(ContractState.REQUIREMENTS_GENERATED, occurred_at=now)
            if self.requirement_factory is None:
                return default_requirement(request, now=now)
            response = self.requirement_factory(request, now)
            response.metadata.CopyFrom(
                build_metadata(
                    lifecycle_id=lifecycle_id,
                    message_id=f"response:{request.metadata.message_id}",
                    sent_at=now,
                )
            )
            return response

        return self._process(request, "ICTC_PROPOSE", handler)

    def offer(self, request: ictc_pb2.IctcOffer) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            record = self._record_for(lifecycle_id)
            if request.offer_id in record.offer_ids:
                raise ProtocolRequestError(
                    ProtocolErrorCode.IDEMPOTENCY_CONFLICT,
                    "offer ID was reused under a different message ID",
                )
            if record.machine.state is ContractState.REQUIREMENTS_GENERATED:
                record.machine.transition(ContractState.OFFERS_RECEIVED, occurred_at=now)
            elif record.machine.state is not ContractState.OFFERS_RECEIVED:
                raise ProtocolRequestError(
                    ProtocolErrorCode.INVALID_TRANSITION,
                    f"offer is invalid from {record.machine.state.value}",
                )
            record.offer_ids.add(request.offer_id)
            return self._status(record, message_id, now)

        return self._process(request, "ICTC_OFFER", handler)

    def commit(self, request: ictc_pb2.IctcCommit) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            if request.contract_id != lifecycle_id:
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT,
                    "commit contract ID does not match lifecycle metadata",
                )
            valid_until = datetime.fromtimestamp(request.valid_until_unix_ms / 1000, tz=UTC)
            if valid_until <= now:
                raise ProtocolRequestError(
                    ProtocolErrorCode.DEADLINE_EXCEEDED, "commit validity already expired"
                )
            record = self._record_for(lifecycle_id)
            record.machine.transition(ContractState.COMMITTED, occurred_at=now)
            record.contract_id = request.contract_id
            record.valid_until = valid_until
            return self._status(record, message_id, now)

        return self._process(request, "ICTC_COMMIT", handler)

    def report_status(self, request: ictc_pb2.IctcStatus) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            if request.contract_id != lifecycle_id:
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT,
                    "status contract ID does not match lifecycle metadata",
                )
            record = self._record_for(lifecycle_id)
            target = state_from_proto(request.state)
            if target is not record.machine.state:
                record.machine.transition(target, occurred_at=now)
            return self._status(record, message_id, now)

        return self._process(request, "ICTC_STATUS", handler)

    def report_breach(self, request: ictc_pb2.IctcBreach) -> ictc_pb2.IctcRelease:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcRelease:
            if request.contract_id != lifecycle_id:
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT,
                    "breach contract ID does not match lifecycle metadata",
                )
            record = self._record_for(lifecycle_id)
            if record.machine.state is ContractState.ACTIVE:
                record.machine.transition(ContractState.BREACHED, occurred_at=now)
                record.machine.transition(ContractState.RELEASED, occurred_at=now)
            elif record.machine.state is ContractState.SATISFIED:
                record.machine.transition(ContractState.BREACHED_BEFORE_EXECUTION, occurred_at=now)
            else:
                raise ProtocolRequestError(
                    ProtocolErrorCode.INVALID_TRANSITION,
                    f"breach is invalid from {record.machine.state.value}",
                )
            return ictc_pb2.IctcRelease(
                contract_id=lifecycle_id,
                release_reason=f"breach:{request.severity}",
                final_state=state_to_proto(record.machine.state),
                execution_outcome="FALLBACK_REQUIRED",
                audit_digest=request.audit_digest,
                metadata=build_metadata(
                    lifecycle_id=lifecycle_id,
                    message_id=f"response:{message_id}",
                    sent_at=now,
                ),
            )

        return self._process(request, "ICTC_BREACH", handler)

    def expire_due(self, *, now: datetime | None = None) -> tuple[str, ...]:
        instant = (now or self._now()).astimezone(UTC)
        expired: list[str] = []
        with self._lock:
            for lifecycle_id, record in sorted(self._records.items()):
                if (
                    record.machine.state is ContractState.COMMITTED
                    and record.valid_until is not None
                    and record.valid_until <= instant
                ):
                    before = record.machine.state
                    transition = record.machine.transition(
                        ContractState.EXPIRED,
                        reason_code=ReasonCode.DEADLINE_EXCEEDED,
                        occurred_at=instant,
                    )
                    self.audit_log.append(
                        event_type=AuditEventType.CONTRACT_EXPIRED,
                        lifecycle_id=lifecycle_id,
                        message_id=f"timeout:{int(instant.timestamp() * 1000)}",
                        message_kind="INTERNAL_TIMEOUT",
                        occurred_at=instant,
                        state_before=before,
                        state_after=transition.target,
                        payload_digest=sha256_hex(lifecycle_id.encode("utf-8")),
                        reason=ReasonCode.DEADLINE_EXCEEDED.value,
                    )
                    expired.append(lifecycle_id)
        return tuple(expired)

    def state(self, lifecycle_id: str) -> ContractState:
        with self._lock:
            return self._record_for(lifecycle_id).machine.state
