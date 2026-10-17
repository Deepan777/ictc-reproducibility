"""Windows-loopback gRPC lifecycle with loss, duplication, delay, and reordering."""

from __future__ import annotations

from concurrent import futures
from datetime import UTC, datetime, timedelta
from pathlib import Path

import grpc  # type: ignore[import-untyped]
import pytest

from ictc.domain.enums import ContractState
from ictc.proto import ictc_pb2, ictc_pb2_grpc
from ictc.protocol.audit import AuditEventType, AuditLog
from ictc.protocol.runtime import CoordinatorRuntime, build_metadata, state_to_proto
from ictc.protocol.services import (
    CoordinatorGrpcServicer,
    FaultAction,
    FaultPlan,
    RetryingCoordinatorClient,
    RetryPolicy,
    TelemetryAgentGrpcServicer,
    TelemetryAgentRuntime,
)

NOW = datetime(2026, 7, 11, 10, 0, tzinfo=UTC)
CONTRACT = "grpc-contract"


def metadata(message_id: str) -> ictc_pb2.ProtocolMetadata:
    return build_metadata(lifecycle_id=CONTRACT, message_id=message_id, sent_at=NOW)


def keep_action() -> ictc_pb2.NetworkAction:
    return ictc_pb2.NetworkAction(
        action_id="keep",
        action_class=ictc_pb2.ACTION_CLASS_ROUTING,
        action_kind=ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH,
        path_id="path-a",
    )


def make_proposal() -> ictc_pb2.IctcPropose:
    return ictc_pb2.IctcPropose(
        proposal_id="proposal",
        baseline_action=keep_action(),
        proposed_action=ictc_pb2.NetworkAction(
            action_id="reroute",
            action_class=ictc_pb2.ACTION_CLASS_ROUTING,
            action_kind=ictc_pb2.ACTION_KIND_REROUTE_TO_PATH,
            path_id="path-b",
        ),
        target_outcome=ictc_pb2.TargetOutcome(metric="latency", unit="ms", weight=1.0),
        evaluation_horizon=ictc_pb2.EvaluationHorizon(horizon_ms=100),
        action_deadline_unix_ms=int((NOW + timedelta(seconds=30)).timestamp() * 1000),
        causal_graph_scope="routing-v1",
        safety_policy_scope="safety-v1",
        fallback_action=keep_action(),
        metadata=metadata("propose"),
    )


def offer_template() -> ictc_pb2.IctcOffer:
    return ictc_pb2.IctcOffer(
        offer_id="agent-offer",
        source_id="agent-a",
        measurable_variables=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
        expected_delay_ms=1,
        max_clock_skew_ms=1,
        cost=1.0,
        bandwidth_kbps=4.0,
        trust_score=1.0,
        provenance_mechanism="sha256",
        administrative_domain="domain-a",
        available_from_unix_ms=int(NOW.timestamp() * 1000),
        available_until_unix_ms=int((NOW + timedelta(minutes=1)).timestamp() * 1000),
    )


def make_commit() -> ictc_pb2.IctcCommit:
    return ictc_pb2.IctcCommit(
        contract_id=CONTRACT,
        selected_source_ids=["agent-a"],
        selected_fields=["CrossTraffic"],
        sampling_rates_hz={"CrossTraffic": 10.0},
        synchronization_epoch_unix_ms=int(NOW.timestamp() * 1000),
        contract_start_unix_ms=int(NOW.timestamp() * 1000),
        valid_until_unix_ms=int((NOW + timedelta(seconds=10)).timestamp() * 1000),
        evidence_digests=["evidence"],
        expected_total_cost=1.0,
        fallback_action=keep_action(),
        metadata=metadata("commit"),
    )


def test_loopback_services_survive_deterministic_delivery_faults(tmp_path: Path) -> None:
    coordinator_log = AuditLog(tmp_path / "coordinator.jsonl")
    agent_log = AuditLog(tmp_path / "agent.jsonl")
    coordinator = CoordinatorRuntime(coordinator_log, clock=lambda: NOW)
    agent = TelemetryAgentRuntime(
        source_id="agent-a",
        offer_template=offer_template(),
        audit_log=agent_log,
        clock=lambda: NOW,
    )
    faults = FaultPlan(
        {
            "Propose": (FaultAction.LOSS, FaultAction.PASS),
            "Offer": (FaultAction.DUPLICATE,),
            "Commit": (FaultAction.DELAY, FaultAction.PASS),
        },
        delay_seconds=0.05,
    )
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    ictc_pb2_grpc.add_IctcCoordinatorServicer_to_server(
        CoordinatorGrpcServicer(coordinator, faults=faults), server
    )  # type: ignore[no-untyped-call]
    ictc_pb2_grpc.add_IctcTelemetryAgentServicer_to_server(
        TelemetryAgentGrpcServicer(agent), server
    )  # type: ignore[no-untyped-call]
    port = server.add_insecure_port("127.0.0.1:0")
    assert port > 0
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    grpc.channel_ready_future(channel).result(timeout=2)
    coordinator_stub = ictc_pb2_grpc.IctcCoordinatorStub(channel)  # type: ignore[no-untyped-call]
    agent_stub = ictc_pb2_grpc.IctcTelemetryAgentStub(channel)  # type: ignore[no-untyped-call]
    client = RetryingCoordinatorClient(
        coordinator_stub,
        # Remain below the injected 50 ms delay while tolerating Windows suite-load jitter.
        policy=RetryPolicy(max_attempts=3, timeout_seconds=0.03),
    )
    try:
        requirement = client.propose(make_proposal())
        agent_offer = agent_stub.GetOffer(requirement, timeout=1)

        reordered = ictc_pb2.IctcStatus(
            contract_id=CONTRACT,
            state=state_to_proto(ContractState.ACTIVE),
            source_available=True,
            metadata=metadata("reordered-status"),
        )
        with pytest.raises(grpc.RpcError) as captured:
            coordinator_stub.ReportStatus(reordered, timeout=1)
        assert captured.value.code() is grpc.StatusCode.FAILED_PRECONDITION
        assert coordinator.state(CONTRACT) is ContractState.REQUIREMENTS_GENERATED

        assert client.offer(agent_offer).state == state_to_proto(ContractState.OFFERS_RECEIVED)
        commit = make_commit()
        assert client.commit(commit).state == state_to_proto(ContractState.COMMITTED)
        agent_status = agent_stub.Commit(commit, timeout=1)
        assert agent_stub.Commit(commit, timeout=1) == agent_status
        assert client.report_status(agent_status).state == state_to_proto(ContractState.ACTIVE)

        breach = ictc_pb2.IctcBreach(
            contract_id=CONTRACT,
            violated_clause_ids=["freshness"],
            affected_estimand="latency-effect",
            severity="HIGH",
            occurred_at_unix_ms=int(NOW.timestamp() * 1000),
            recommended_fallback=keep_action(),
            audit_digest=agent_log.read_verified()[-1].event_digest,
            metadata=metadata("breach"),
        )
        release = client.report_breach(breach)
        assert release.final_state == state_to_proto(ContractState.RELEASED)
        assert agent_stub.Release(release, timeout=1).state == state_to_proto(
            ContractState.RELEASED
        )
    finally:
        channel.close()
        server.stop(grace=0).wait(timeout=2)

    events = coordinator_log.read_verified()
    assert coordinator_log.reconstruct_lifecycle(CONTRACT) == (
        ContractState.PROPOSED,
        ContractState.REQUIREMENTS_GENERATED,
        ContractState.OFFERS_RECEIVED,
        ContractState.COMMITTED,
        ContractState.ACTIVE,
        ContractState.BREACHED,
        ContractState.RELEASED,
    )
    assert any(event.event_type == AuditEventType.DUPLICATE_RETURNED for event in events)
    assert any(event.event_type == AuditEventType.MESSAGE_REJECTED for event in events)
    assert any(
        event.event_type == AuditEventType.DUPLICATE_RETURNED for event in agent_log.read_verified()
    )
