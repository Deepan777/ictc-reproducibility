"""Coordinator lifecycle, idempotency, replay, timeout, and breach tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from ictc.domain.enums import ContractState
from ictc.proto import ictc_pb2
from ictc.protocol.audit import AuditEventType, AuditLog
from ictc.protocol.runtime import (
    CoordinatorRuntime,
    ProtocolErrorCode,
    ProtocolRequestError,
    build_metadata,
    clone_message,
    state_to_proto,
)


@dataclass
class MutableClock:
    current: datetime

    def __call__(self) -> datetime:
        return self.current


NOW = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
LIFECYCLE = "contract-1"


def action(action_id: str, kind: ictc_pb2.ActionKind) -> ictc_pb2.NetworkAction:
    return ictc_pb2.NetworkAction(
        action_id=action_id,
        action_class=ictc_pb2.ACTION_CLASS_ROUTING,
        action_kind=kind,
        path_id="path-a",
    )


def proposal(*, message_id: str = "propose-1", sent_at: datetime = NOW) -> ictc_pb2.IctcPropose:
    return ictc_pb2.IctcPropose(
        proposal_id="proposal-1",
        baseline_action=action("baseline", ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH),
        proposed_action=action("reroute", ictc_pb2.ACTION_KIND_REROUTE_TO_PATH),
        target_outcome=ictc_pb2.TargetOutcome(metric="latency", unit="ms", weight=1.0),
        evaluation_horizon=ictc_pb2.EvaluationHorizon(horizon_ms=100),
        action_deadline_unix_ms=int((NOW + timedelta(seconds=30)).timestamp() * 1000),
        causal_graph_scope="routing-v1",
        safety_policy_scope="safety-v1",
        fallback_action=action("fallback", ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH),
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=sent_at),
    )


def offer(*, message_id: str = "offer-1") -> ictc_pb2.IctcOffer:
    return ictc_pb2.IctcOffer(
        offer_id="offer-a",
        source_id="source-a",
        measurable_variables=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
        expected_delay_ms=5,
        max_clock_skew_ms=1,
        cost=1.0,
        bandwidth_kbps=4.0,
        trust_score=1.0,
        provenance_mechanism="sha256",
        administrative_domain="domain-a",
        available_from_unix_ms=int(NOW.timestamp() * 1000),
        available_until_unix_ms=int((NOW + timedelta(minutes=1)).timestamp() * 1000),
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=NOW),
    )


def commitment(*, valid_until: datetime | None = None) -> ictc_pb2.IctcCommit:
    return ictc_pb2.IctcCommit(
        contract_id=LIFECYCLE,
        selected_source_ids=["source-a"],
        selected_fields=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
        synchronization_epoch_unix_ms=int(NOW.timestamp() * 1000),
        contract_start_unix_ms=int(NOW.timestamp() * 1000),
        valid_until_unix_ms=int((valid_until or NOW + timedelta(seconds=10)).timestamp() * 1000),
        evidence_digests=["evidence"],
        expected_total_cost=1.0,
        fallback_action=action("fallback", ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH),
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id="commit-1", sent_at=NOW),
    )


def status(target: ContractState, message_id: str) -> ictc_pb2.IctcStatus:
    return ictc_pb2.IctcStatus(
        contract_id=LIFECYCLE,
        state=state_to_proto(target),
        source_available=True,
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=NOW),
    )


def breach(message_id: str = "breach-1") -> ictc_pb2.IctcBreach:
    return ictc_pb2.IctcBreach(
        contract_id=LIFECYCLE,
        violated_clause_ids=["freshness"],
        affected_estimand="latency-effect",
        severity="HIGH",
        occurred_at_unix_ms=int(NOW.timestamp() * 1000),
        recommended_fallback=action("fallback", ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH),
        audit_digest="agent-audit-anchor",
        metadata=build_metadata(lifecycle_id=LIFECYCLE, message_id=message_id, sent_at=NOW),
    )


def runtime(tmp_path: Path, clock: MutableClock | None = None) -> CoordinatorRuntime:
    selected_clock = clock or MutableClock(NOW)
    return CoordinatorRuntime(AuditLog(tmp_path / "events.jsonl"), clock=selected_clock)


def advance_to_active(engine: CoordinatorRuntime) -> None:
    engine.propose(proposal())
    engine.offer(offer())
    engine.commit(commitment())
    engine.report_status(status(ContractState.ACTIVE, "status-active"))


def test_valid_lifecycle_duplicate_and_breach_before_execution(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    requirement = engine.propose(proposal())
    duplicate = proposal()
    duplicate.metadata.delivery_attempt = 2
    assert engine.propose(duplicate) == requirement
    engine.offer(offer())
    engine.commit(commitment())
    engine.report_status(status(ContractState.ACTIVE, "status-active"))
    engine.report_status(status(ContractState.SATISFIED, "status-satisfied"))
    release = engine.report_breach(breach())

    assert engine.state(LIFECYCLE) is ContractState.BREACHED_BEFORE_EXECUTION
    assert release.final_state == state_to_proto(ContractState.BREACHED_BEFORE_EXECUTION)
    assert release.execution_outcome == "FALLBACK_REQUIRED"
    assert engine.audit_log.reconstruct_lifecycle(LIFECYCLE) == (
        ContractState.PROPOSED,
        ContractState.REQUIREMENTS_GENERATED,
        ContractState.OFFERS_RECEIVED,
        ContractState.COMMITTED,
        ContractState.ACTIVE,
        ContractState.SATISFIED,
        ContractState.BREACHED_BEFORE_EXECUTION,
    )
    assert any(
        event.event_type == AuditEventType.DUPLICATE_RETURNED
        for event in engine.audit_log.read_verified()
    )


def test_active_breach_records_breach_and_release(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    advance_to_active(engine)
    release = engine.report_breach(breach())
    assert engine.state(LIFECYCLE) is ContractState.RELEASED
    assert release.final_state == state_to_proto(ContractState.RELEASED)
    assert engine.audit_log.reconstruct_lifecycle(LIFECYCLE)[-3:] == (
        ContractState.ACTIVE,
        ContractState.BREACHED,
        ContractState.RELEASED,
    )


def test_reordered_status_is_rejected_without_state_corruption(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    engine.propose(proposal())
    engine.offer(offer())
    with pytest.raises(ProtocolRequestError) as captured:
        engine.report_status(status(ContractState.ACTIVE, "reordered-active"))
    assert captured.value.code is ProtocolErrorCode.INVALID_TRANSITION
    assert engine.state(LIFECYCLE) is ContractState.OFFERS_RECEIVED
    assert engine.commit(commitment()).state == state_to_proto(ContractState.COMMITTED)


def test_idempotency_conflicts_and_semantic_offer_duplicates_are_rejected(
    tmp_path: Path,
) -> None:
    engine = runtime(tmp_path)
    engine.propose(proposal())
    conflict = clone_message(proposal())
    conflict.proposal_id = "changed"
    with pytest.raises(ProtocolRequestError) as captured:
        engine.propose(conflict)
    assert captured.value.code is ProtocolErrorCode.IDEMPOTENCY_CONFLICT
    assert engine.state(LIFECYCLE) is ContractState.REQUIREMENTS_GENERATED

    engine.offer(offer())
    with pytest.raises(ProtocolRequestError, match="offer ID"):
        engine.offer(offer(message_id="different-message"))
    assert engine.state(LIFECYCLE) is ContractState.OFFERS_RECEIVED


def test_replay_metadata_and_lifecycle_validation(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    with pytest.raises(ProtocolRequestError) as captured:
        engine.propose(proposal(sent_at=NOW - timedelta(minutes=6)))
    assert captured.value.code is ProtocolErrorCode.REPLAY_REJECTED

    missing = proposal()
    missing.ClearField("metadata")
    with pytest.raises(ProtocolRequestError) as captured:
        engine.propose(missing)
    assert captured.value.code is ProtocolErrorCode.INVALID_METADATA

    wrong_version = proposal(message_id="wrong-version")
    wrong_version.metadata.protocol_version = "2.0"
    with pytest.raises(ProtocolRequestError) as captured:
        engine.propose(wrong_version)
    assert captured.value.code is ProtocolErrorCode.UNSUPPORTED_VERSION

    unknown_offer = offer(message_id="unknown")
    unknown_offer.metadata.lifecycle_id = "unknown"
    with pytest.raises(ProtocolRequestError) as captured:
        engine.offer(unknown_offer)
    assert captured.value.code is ProtocolErrorCode.UNKNOWN_LIFECYCLE


def test_commit_timeout_and_expired_commit_rejection(tmp_path: Path) -> None:
    clock = MutableClock(NOW)
    engine = runtime(tmp_path, clock)
    engine.propose(proposal())
    engine.offer(offer())
    engine.commit(commitment(valid_until=NOW + timedelta(seconds=2)))
    clock.current = NOW + timedelta(seconds=3)
    assert engine.expire_due() == (LIFECYCLE,)
    assert engine.expire_due() == ()
    assert engine.state(LIFECYCLE) is ContractState.EXPIRED
    assert engine.audit_log.reconstruct_lifecycle(LIFECYCLE)[-1] is ContractState.EXPIRED

    second = CoordinatorRuntime(AuditLog(tmp_path / "second.jsonl"), clock=clock)
    stale_proposal = proposal(sent_at=clock.current)
    second.propose(stale_proposal)
    stale_offer = offer()
    stale_offer.metadata.sent_at_unix_ms = int(clock.current.timestamp() * 1000)
    second.offer(stale_offer)
    stale_commit = commitment(valid_until=NOW + timedelta(seconds=1))
    stale_commit.metadata.sent_at_unix_ms = int(clock.current.timestamp() * 1000)
    with pytest.raises(ProtocolRequestError) as captured:
        second.commit(stale_commit)
    assert captured.value.code is ProtocolErrorCode.DEADLINE_EXCEEDED


def test_same_state_status_is_audited_without_transition(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    advance_to_active(engine)
    response = engine.report_status(status(ContractState.ACTIVE, "heartbeat"))
    assert response.state == state_to_proto(ContractState.ACTIVE)
    last = engine.audit_log.read_verified()[-1]
    assert last.state_before == last.state_after == ContractState.ACTIVE.value


def test_runtime_configuration_metadata_and_custom_requirement_boundaries(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="replay window"):
        CoordinatorRuntime(AuditLog(tmp_path / "invalid.jsonl"), replay_window=timedelta(0))
    naive = CoordinatorRuntime(
        AuditLog(tmp_path / "naive.jsonl"), clock=lambda: datetime(2026, 7, 11, 10, 0)
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        naive.propose(proposal())

    invalid_metadata = proposal()
    invalid_metadata.metadata.message_id = ""
    with pytest.raises(ProtocolRequestError) as captured:
        runtime(tmp_path).propose(invalid_metadata)
    assert captured.value.code is ProtocolErrorCode.INVALID_METADATA

    invalid_timestamp = proposal()
    invalid_timestamp.metadata.sent_at_unix_ms = 0
    with pytest.raises(ProtocolRequestError) as captured:
        runtime(tmp_path).propose(invalid_timestamp)
    assert captured.value.code is ProtocolErrorCode.INVALID_METADATA

    def factory(request: ictc_pb2.IctcPropose, _now: datetime) -> ictc_pb2.IctcRequire:
        return ictc_pb2.IctcRequire(
            requirement_id=f"custom:{request.proposal_id}", target_estimand="custom-estimand"
        )

    custom = CoordinatorRuntime(
        AuditLog(tmp_path / "custom.jsonl"), clock=lambda: NOW, requirement_factory=factory
    )
    response = custom.propose(proposal())
    assert response.requirement_id == "custom:proposal-1"
    assert response.metadata.lifecycle_id == LIFECYCLE


def test_contract_binding_and_unspecified_status_are_rejected(tmp_path: Path) -> None:
    engine = runtime(tmp_path)
    engine.propose(proposal())
    engine.offer(offer())
    wrong_commit = commitment()
    wrong_commit.contract_id = "other"
    with pytest.raises(ProtocolRequestError) as captured:
        engine.commit(wrong_commit)
    assert captured.value.code is ProtocolErrorCode.LIFECYCLE_CONFLICT

    engine.commit(commitment())
    unspecified = status(ContractState.ACTIVE, "unspecified")
    unspecified.state = ictc_pb2.CONTRACT_STATE_UNSPECIFIED
    with pytest.raises(ProtocolRequestError) as captured:
        engine.report_status(unspecified)
    assert captured.value.code is ProtocolErrorCode.INVALID_TRANSITION

    wrong_breach = breach("wrong-contract")
    wrong_breach.contract_id = "other"
    with pytest.raises(ProtocolRequestError) as captured:
        engine.report_breach(wrong_breach)
    assert captured.value.code is ProtocolErrorCode.LIFECYCLE_CONFLICT
