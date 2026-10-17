"""Local gRPC services, telemetry agent, retries, and deterministic fault injection."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import TypeVar, cast

import grpc  # type: ignore[import-untyped]

from ictc.domain.enums import ContractState
from ictc.proto import ictc_pb2, ictc_pb2_grpc
from ictc.protocol.audit import AuditEventType, AuditLog
from ictc.protocol.runtime import (
    CoordinatorRuntime,
    ProtocolErrorCode,
    ProtocolRequestError,
    ProtoMessage,
    build_metadata,
    clone_message,
    protobuf_digest,
    state_to_proto,
    utc_now,
)

_RequestT = TypeVar("_RequestT", bound=ProtoMessage)
_ResponseT = TypeVar("_ResponseT", bound=ProtoMessage)


class FaultAction(StrEnum):
    PASS = "PASS"
    LOSS = "LOSS"
    DUPLICATE = "DUPLICATE"
    DELAY = "DELAY"


class FaultPlan:
    """Consume a deterministic per-method delivery-fault schedule."""

    def __init__(
        self,
        actions: dict[str, tuple[FaultAction, ...]] | None = None,
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        if delay_seconds < 0:
            raise ValueError("fault delay cannot be negative")
        self._actions: defaultdict[str, deque[FaultAction]] = defaultdict(deque)
        for method, values in (actions or {}).items():
            self._actions[method].extend(values)
        self.delay_seconds = delay_seconds
        self._lock = threading.Lock()

    def next(self, method: str) -> FaultAction:
        with self._lock:
            queue = self._actions[method]
            return queue.popleft() if queue else FaultAction.PASS


_GRPC_CODES: dict[ProtocolErrorCode, grpc.StatusCode] = {
    ProtocolErrorCode.INVALID_METADATA: grpc.StatusCode.INVALID_ARGUMENT,
    ProtocolErrorCode.UNSUPPORTED_VERSION: grpc.StatusCode.UNIMPLEMENTED,
    ProtocolErrorCode.REPLAY_REJECTED: grpc.StatusCode.PERMISSION_DENIED,
    ProtocolErrorCode.IDEMPOTENCY_CONFLICT: grpc.StatusCode.ALREADY_EXISTS,
    ProtocolErrorCode.UNKNOWN_LIFECYCLE: grpc.StatusCode.NOT_FOUND,
    ProtocolErrorCode.LIFECYCLE_CONFLICT: grpc.StatusCode.FAILED_PRECONDITION,
    ProtocolErrorCode.INVALID_TRANSITION: grpc.StatusCode.FAILED_PRECONDITION,
    ProtocolErrorCode.DEADLINE_EXCEEDED: grpc.StatusCode.DEADLINE_EXCEEDED,
}


class CoordinatorGrpcServicer(ictc_pb2_grpc.IctcCoordinatorServicer):
    def __init__(self, runtime: CoordinatorRuntime, *, faults: FaultPlan | None = None) -> None:
        self.runtime = runtime
        self.faults = faults or FaultPlan()

    def _invoke(
        self,
        method_name: str,
        handler: Callable[[_RequestT], _ResponseT],
        request: _RequestT,
        context: grpc.ServicerContext,
    ) -> _ResponseT:
        action = self.faults.next(method_name)
        if action is FaultAction.LOSS:
            context.abort(grpc.StatusCode.UNAVAILABLE, "injected message loss")
        if action is FaultAction.DELAY:
            time.sleep(self.faults.delay_seconds)
        try:
            response = handler(request)
            if action is FaultAction.DUPLICATE:
                response = handler(request)
            return response
        except ProtocolRequestError as error:
            context.abort(_GRPC_CODES[error.code], str(error))
            raise AssertionError("grpc abort returned unexpectedly") from error

    def Propose(
        self, request: ictc_pb2.IctcPropose, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcRequire:
        return self._invoke("Propose", self.runtime.propose, request, context)

    def Offer(
        self, request: ictc_pb2.IctcOffer, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke("Offer", self.runtime.offer, request, context)

    def Commit(
        self, request: ictc_pb2.IctcCommit, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke("Commit", self.runtime.commit, request, context)

    def ReportStatus(
        self, request: ictc_pb2.IctcStatus, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke("ReportStatus", self.runtime.report_status, request, context)

    def ReportBreach(
        self, request: ictc_pb2.IctcBreach, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcRelease:
        return self._invoke("ReportBreach", self.runtime.report_breach, request, context)


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    timeout_seconds: float = 1.0
    backoff_seconds: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max attempts must be positive")
        if self.timeout_seconds <= 0 or self.backoff_seconds < 0:
            raise ValueError("retry timing values are invalid")


class RetryingCoordinatorClient:
    """Retry transient unary RPC failures without changing message identity."""

    def __init__(
        self,
        stub: ictc_pb2_grpc.IctcCoordinatorStub,
        *,
        policy: RetryPolicy | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.stub = stub
        self.policy = policy or RetryPolicy()
        self.sleeper = sleeper

    def _call(
        self,
        rpc: Callable[..., _ResponseT],
        request: _RequestT,
    ) -> _ResponseT:
        last_error: grpc.RpcError | None = None
        for attempt in range(1, self.policy.max_attempts + 1):
            outgoing = clone_message(request)
            outgoing.metadata.delivery_attempt = attempt
            try:
                return rpc(outgoing, timeout=self.policy.timeout_seconds)
            except grpc.RpcError as error:
                last_error = error
                if (
                    error.code()
                    not in {
                        grpc.StatusCode.UNAVAILABLE,
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                    }
                    or attempt == self.policy.max_attempts
                ):
                    raise
                self.sleeper(self.policy.backoff_seconds)
        assert last_error is not None
        raise last_error

    def propose(self, request: ictc_pb2.IctcPropose) -> ictc_pb2.IctcRequire:
        rpc = cast(Callable[..., ictc_pb2.IctcRequire], self.stub.Propose)
        return self._call(rpc, request)

    def offer(self, request: ictc_pb2.IctcOffer) -> ictc_pb2.IctcStatus:
        rpc = cast(Callable[..., ictc_pb2.IctcStatus], self.stub.Offer)
        return self._call(rpc, request)

    def commit(self, request: ictc_pb2.IctcCommit) -> ictc_pb2.IctcStatus:
        rpc = cast(Callable[..., ictc_pb2.IctcStatus], self.stub.Commit)
        return self._call(rpc, request)

    def report_status(self, request: ictc_pb2.IctcStatus) -> ictc_pb2.IctcStatus:
        rpc = cast(Callable[..., ictc_pb2.IctcStatus], self.stub.ReportStatus)
        return self._call(rpc, request)

    def report_breach(self, request: ictc_pb2.IctcBreach) -> ictc_pb2.IctcRelease:
        rpc = cast(Callable[..., ictc_pb2.IctcRelease], self.stub.ReportBreach)
        return self._call(rpc, request)


class TelemetryAgentRuntime:
    """A source-local contract participant with replay-safe activation and release."""

    def __init__(
        self,
        *,
        source_id: str,
        offer_template: ictc_pb2.IctcOffer,
        audit_log: AuditLog,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        if not source_id:
            raise ValueError("source ID is required")
        if offer_template.source_id != source_id:
            raise ValueError("offer template source does not match the agent")
        self.source_id = source_id
        self.offer_template = clone_message(offer_template)
        self.audit_log = audit_log
        self.clock = clock
        self._states: dict[str, ContractState] = {}
        self._processed: dict[tuple[str, str], tuple[str, ProtoMessage]] = {}
        self._lock = threading.RLock()

    def _process(
        self,
        request: _RequestT,
        kind: str,
        handler: Callable[[str, str, datetime], _ResponseT],
    ) -> _ResponseT:
        with self._lock:
            if not request.HasField("metadata"):
                raise ProtocolRequestError(
                    ProtocolErrorCode.INVALID_METADATA, "protocol metadata is required"
                )
            lifecycle_id = request.metadata.lifecycle_id
            message_id = request.metadata.message_id
            if not lifecycle_id or not message_id:
                raise ProtocolRequestError(
                    ProtocolErrorCode.INVALID_METADATA, "metadata identifiers are required"
                )
            digest = protobuf_digest(request, ignore_attempt=True)
            key = (lifecycle_id, message_id)
            if key in self._processed:
                prior_digest, prior_response = self._processed[key]
                if digest != prior_digest:
                    raise ProtocolRequestError(
                        ProtocolErrorCode.IDEMPOTENCY_CONFLICT,
                        "agent message ID was reused with different content",
                    )
                response = cast(_ResponseT, clone_message(prior_response))
                state = self._states.get(lifecycle_id)
                self.audit_log.append(
                    event_type=AuditEventType.DUPLICATE_RETURNED,
                    lifecycle_id=lifecycle_id,
                    message_id=message_id,
                    message_kind=kind,
                    occurred_at=self.clock(),
                    state_before=state,
                    state_after=state,
                    payload_digest=digest,
                    response_digest=protobuf_digest(response),
                )
                return response
            now = self.clock().astimezone(UTC)
            before = self._states.get(lifecycle_id)
            response = handler(lifecycle_id, message_id, now)
            after = self._states.get(lifecycle_id)
            self.audit_log.append(
                event_type=AuditEventType.MESSAGE_APPLIED,
                lifecycle_id=lifecycle_id,
                message_id=message_id,
                message_kind=kind,
                occurred_at=now,
                state_before=before,
                state_after=after,
                payload_digest=digest,
                response_digest=protobuf_digest(response),
            )
            self._processed[key] = (digest, clone_message(response))
            return response

    def get_offer(self, request: ictc_pb2.IctcRequire) -> ictc_pb2.IctcOffer:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcOffer:
            if lifecycle_id in self._states:
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT, "agent lifecycle already offered"
                )
            response = clone_message(self.offer_template)
            response.metadata.CopyFrom(
                build_metadata(
                    lifecycle_id=lifecycle_id,
                    message_id=f"response:{message_id}",
                    sent_at=now,
                )
            )
            self._states[lifecycle_id] = ContractState.OFFERS_RECEIVED
            return response

        return self._process(request, "AGENT_GET_OFFER", handler)

    def commit(self, request: ictc_pb2.IctcCommit) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            if (
                request.contract_id != lifecycle_id
                or self.source_id not in request.selected_source_ids
            ):
                raise ProtocolRequestError(
                    ProtocolErrorCode.LIFECYCLE_CONFLICT,
                    "commit does not select this agent for the lifecycle",
                )
            if self._states.get(lifecycle_id) is not ContractState.OFFERS_RECEIVED:
                raise ProtocolRequestError(
                    ProtocolErrorCode.INVALID_TRANSITION, "agent commit arrived out of order"
                )
            self._states[lifecycle_id] = ContractState.ACTIVE
            return self._status(lifecycle_id, message_id, now)

        return self._process(request, "AGENT_COMMIT", handler)

    def report_status(self, request: ictc_pb2.IctcCommit) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            if lifecycle_id not in self._states:
                raise ProtocolRequestError(ProtocolErrorCode.UNKNOWN_LIFECYCLE, lifecycle_id)
            return self._status(lifecycle_id, message_id, now)

        return self._process(request, "AGENT_STATUS", handler)

    def release(self, request: ictc_pb2.IctcRelease) -> ictc_pb2.IctcStatus:
        def handler(lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
            if request.contract_id != lifecycle_id or lifecycle_id not in self._states:
                raise ProtocolRequestError(ProtocolErrorCode.UNKNOWN_LIFECYCLE, lifecycle_id)
            self._states[lifecycle_id] = ContractState.RELEASED
            return self._status(lifecycle_id, message_id, now)

        return self._process(request, "AGENT_RELEASE", handler)

    def _status(self, lifecycle_id: str, message_id: str, now: datetime) -> ictc_pb2.IctcStatus:
        return ictc_pb2.IctcStatus(
            contract_id=lifecycle_id,
            state=state_to_proto(self._states[lifecycle_id]),
            source_available=True,
            metadata=build_metadata(
                lifecycle_id=lifecycle_id,
                message_id=f"response:{message_id}",
                sent_at=now,
            ),
        )


class TelemetryAgentGrpcServicer(ictc_pb2_grpc.IctcTelemetryAgentServicer):
    def __init__(self, runtime: TelemetryAgentRuntime) -> None:
        self.runtime = runtime

    @staticmethod
    def _invoke(
        handler: Callable[[_RequestT], _ResponseT],
        request: _RequestT,
        context: grpc.ServicerContext,
    ) -> _ResponseT:
        try:
            return handler(request)
        except ProtocolRequestError as error:
            context.abort(_GRPC_CODES[error.code], str(error))
            raise AssertionError("grpc abort returned unexpectedly") from error

    def GetOffer(
        self, request: ictc_pb2.IctcRequire, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcOffer:
        return self._invoke(self.runtime.get_offer, request, context)

    def Commit(
        self, request: ictc_pb2.IctcCommit, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke(self.runtime.commit, request, context)

    def ReportStatus(
        self, request: ictc_pb2.IctcCommit, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke(self.runtime.report_status, request, context)

    def Release(
        self, request: ictc_pb2.IctcRelease, context: grpc.ServicerContext
    ) -> ictc_pb2.IctcStatus:
        return self._invoke(self.runtime.release, request, context)
