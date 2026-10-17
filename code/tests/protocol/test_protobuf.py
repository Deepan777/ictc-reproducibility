"""Round-trip and reproducibility tests for generated ICTC Protobuf bindings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction
from ictc.proto import ictc_pb2
from ictc.protocol.serialization import network_action_from_proto, network_action_to_proto

ROOT = Path(__file__).resolve().parents[2]
LEGACY_V1_STATUS_HEX = "0a0f6c65676163792d636f6e7472616374100520074801"


def test_propose_envelope_round_trip() -> None:
    message = ictc_pb2.IctcEnvelope(
        propose=ictc_pb2.IctcPropose(
            proposal_id="proposal-1",
            baseline_action=ictc_pb2.NetworkAction(
                action_id="keep-1",
                action_class=ictc_pb2.ACTION_CLASS_ROUTING,
                action_kind=ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH,
            ),
            proposed_action=ictc_pb2.NetworkAction(
                action_id="reroute-1",
                action_class=ictc_pb2.ACTION_CLASS_ROUTING,
                action_kind=ictc_pb2.ACTION_KIND_REROUTE_TO_PATH,
                path_id="path-b",
            ),
            target_outcome=ictc_pb2.TargetOutcome(metric="latency", unit="ms", weight=1.0),
            evaluation_horizon=ictc_pb2.EvaluationHorizon(horizon_ms=100),
            action_deadline_unix_ms=1_700_000_000_000,
            causal_graph_scope="routing_telemetry_v1",
            safety_policy_scope="default",
        )
    )
    restored = ictc_pb2.IctcEnvelope.FromString(message.SerializeToString(deterministic=True))
    assert restored.WhichOneof("payload") == "propose"
    assert restored.propose.proposed_action.path_id == "path-b"


def test_domain_action_proto_round_trip() -> None:
    original = NetworkAction(
        action_id="reroute-1",
        kind=ActionKind.REROUTE_TO_PATH,
        path_id="path-b",
    )
    restored = network_action_from_proto(network_action_to_proto(original))
    assert restored == original


def test_unknown_proto_action_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="action class"):
        network_action_from_proto(
            ictc_pb2.NetworkAction(
                action_id="unknown-class",
                action_class=ictc_pb2.ACTION_CLASS_UNSPECIFIED,
                action_kind=ictc_pb2.ACTION_KIND_KEEP_CURRENT_PATH,
            )
        )
    with pytest.raises(ValueError, match="action kind"):
        network_action_from_proto(
            ictc_pb2.NetworkAction(
                action_id="unknown-kind",
                action_class=ictc_pb2.ACTION_CLASS_ROUTING,
                action_kind=ictc_pb2.ACTION_KIND_UNSPECIFIED,
            )
        )


def test_generated_bindings_are_current() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_proto.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_m06_additions_preserve_legacy_v1_wire_payload() -> None:
    """Bytes captured before metadata fields were added remain parseable and unchanged."""
    legacy = bytes.fromhex(LEGACY_V1_STATUS_HEX)
    status = ictc_pb2.IctcStatus.FromString(legacy)
    assert status.contract_id == "legacy-contract"
    assert status.state == ictc_pb2.CONTRACT_STATE_ACTIVE
    assert status.freshness_ms == 7
    assert status.source_available
    assert not status.HasField("metadata")
    assert status.SerializeToString(deterministic=True) == legacy

    fields = ictc_pb2.IctcStatus.DESCRIPTOR.fields_by_name
    assert fields["contract_id"].number == 1
    assert fields["uncertainty_width"].number == 10
    assert fields["metadata"].number == 11
