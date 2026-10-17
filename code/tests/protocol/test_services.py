"""Telemetry-agent and transport-policy branch tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ictc.proto import ictc_pb2
from ictc.protocol.audit import AuditLog
from ictc.protocol.runtime import (
    ProtocolErrorCode,
    ProtocolRequestError,
    build_metadata,
    clone_message,
)
from ictc.protocol.services import FaultAction, FaultPlan, RetryPolicy, TelemetryAgentRuntime

NOW = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
LIFECYCLE = "agent-contract"


def template(source_id: str = "agent-a") -> ictc_pb2.IctcOffer:
    return ictc_pb2.IctcOffer(
        offer_id="offer-a",
        source_id=source_id,
        measurable_variables=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
    )


def requirement(message_id: str = "require") -> ictc_pb2.IctcRequire:
    return ictc_pb2.IctcRequire(
        requirement_id="requirement-a",
        target_estimand="latency-effect",
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=NOW),
    )


def commit(*, selected_source: str = "agent-a", message_id: str = "commit") -> ictc_pb2.IctcCommit:
    return ictc_pb2.IctcCommit(
        contract_id=LIFECYCLE,
        selected_source_ids=[selected_source],
        selected_fields=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
        synchronization_epoch_unix_ms=int(NOW.timestamp() * 1000),
        contract_start_unix_ms=int(NOW.timestamp() * 1000),
        valid_until_unix_ms=int((NOW + timedelta(seconds=10)).timestamp() * 1000),
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=NOW),
    )


def agent(tmp_path: Path) -> TelemetryAgentRuntime:
    return TelemetryAgentRuntime(
        source_id="agent-a",
        offer_template=template(),
        audit_log=AuditLog(tmp_path / "agent.jsonl"),
        clock=lambda: NOW,
    )


def test_fault_and_retry_policy_validation() -> None:
    with pytest.raises(ValueError, match="fault delay"):
        FaultPlan(delay_seconds=-0.1)
    plan = FaultPlan({"Offer": (FaultAction.LOSS,)})
    assert plan.next("Offer") is FaultAction.LOSS
    assert plan.next("Offer") is FaultAction.PASS
    assert plan.next("Unknown") is FaultAction.PASS

    with pytest.raises(ValueError, match="max attempts"):
        RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="timing"):
        RetryPolicy(timeout_seconds=0)
    with pytest.raises(ValueError, match="timing"):
        RetryPolicy(backoff_seconds=-1)


def test_agent_constructor_metadata_and_idempotency_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source ID"):
        TelemetryAgentRuntime(
            source_id="", offer_template=template(), audit_log=AuditLog(tmp_path / "a.jsonl")
        )
    with pytest.raises(ValueError, match="does not match"):
        TelemetryAgentRuntime(
            source_id="agent-b",
            offer_template=template(),
            audit_log=AuditLog(tmp_path / "b.jsonl"),
        )

    runtime = agent(tmp_path)
    missing = requirement()
    missing.ClearField("metadata")
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.get_offer(missing)
    assert captured.value.code is ProtocolErrorCode.INVALID_METADATA

    original = requirement()
    runtime.get_offer(original)
    changed = clone_message(original)
    changed.requirement_id = "changed"
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.get_offer(changed)
    assert captured.value.code is ProtocolErrorCode.IDEMPOTENCY_CONFLICT
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.get_offer(requirement(message_id="second-require"))
    assert captured.value.code is ProtocolErrorCode.LIFECYCLE_CONFLICT


def test_agent_reordering_status_and_release_errors(tmp_path: Path) -> None:
    runtime = agent(tmp_path)
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.commit(commit())
    assert captured.value.code is ProtocolErrorCode.INVALID_TRANSITION

    runtime.get_offer(requirement())
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.commit(commit(selected_source="other"))
    assert captured.value.code is ProtocolErrorCode.LIFECYCLE_CONFLICT

    valid_commit = commit(message_id="valid-commit")
    active = runtime.commit(valid_commit)
    status_request = clone_message(valid_commit)
    status_request.metadata.message_id = "status"
    assert runtime.report_status(status_request).state == active.state

    unknown = clone_message(valid_commit)
    unknown.contract_id = "unknown"
    unknown.metadata.lifecycle_id = "unknown"
    unknown.metadata.message_id = "unknown-status"
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.report_status(unknown)
    assert captured.value.code is ProtocolErrorCode.UNKNOWN_LIFECYCLE

    release = ictc_pb2.IctcRelease(
        contract_id="unknown",
        metadata=build_metadata(lifecycle_id="unknown", message_id="release", sent_at=NOW),
    )
    with pytest.raises(ProtocolRequestError) as captured:
        runtime.release(release)
    assert captured.value.code is ProtocolErrorCode.UNKNOWN_LIFECYCLE
