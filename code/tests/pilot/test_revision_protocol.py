"""Frozen M10 revision protocol and bounded assignment tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ictc.datasets import sha256_file
from ictc.pilot import (
    M10R_ASSIGNMENT_POLICY_ID,
    load_revision_development_protocol,
    load_revision_evaluation_protocol,
    load_revision_protocol,
    revision_assignment,
)

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / "configs/experiments/m10r_revision_protocol.json"
DEVELOPMENT_PROTOCOL = ROOT / "configs/experiments/m10r_revision_development_protocol.json"
EVALUATION_PROTOCOL = ROOT / "configs/experiments/m10r_revision_evaluation_protocol.json"
INHERITED = ROOT / "configs/experiments/m10_pilot_protocol_amendment_001.json"


def test_revision_protocol_freezes_development_boundary_and_support() -> None:
    protocol = load_revision_protocol(PROTOCOL)

    assert protocol.protocol_id == "m10r-contextual-revision-development-v1"
    assert protocol.assignment_policy_id == M10R_ASSIGNMENT_POLICY_ID
    assert protocol.assignment_probabilities == ((-1, 0.25), (0, 0.5), (1, 0.75))
    assert protocol.minimum_per_arm == 10
    assert protocol.analysis_seed == 61013
    assert len(protocol.configuration_digest) == 64
    inherited = protocol.payload["prior_metric_dispositions"]
    assert isinstance(inherited, dict)
    assert inherited["source_file_sha256"] == sha256_file(INHERITED)


def test_revision_assignment_is_deterministic_and_bounded() -> None:
    protocol = load_revision_protocol(PROTOCOL)

    first = revision_assignment("pair-a", -1, protocol)
    assert revision_assignment("pair-a", -1, protocol) == first
    assert first[1] == 0.25
    assert 0 <= first[2] < 1
    assert revision_assignment("pair-a", 1, protocol)[1] == 0.75
    with pytest.raises(ValueError, match="-1, 0, or 1"):
        revision_assignment("pair-a", 2, protocol)


def test_revision_development_protocol_freezes_pretruth_grid_and_selection_rule() -> None:
    protocol = load_revision_development_protocol(DEVELOPMENT_PROTOCOL)

    assert (
        protocol.base.configuration_digest
        == "A0F3FFCE5B8860E73191C78428F864B7FE13D911D8581355325C98F019678D7B"
    )
    assert protocol.upper_interval_threshold_grid_ms == (0.0, -0.25, -0.5, -1.0, -2.0)
    assert protocol.minimum_retained_base_reroute_fraction == 0.25
    assert len(protocol.configuration_digest) == 64


def test_revision_evaluation_protocol_freezes_selected_gate_without_granting_access() -> None:
    protocol = load_revision_evaluation_protocol(EVALUATION_PROTOCOL, repo_root=ROOT)

    assert protocol.protocol_id == "m10r-contextual-revision-evaluation-v1"
    assert protocol.selected_configuration_id == "context-local:upper<=-0.25ms"
    assert protocol.selected_scope == "context-local"
    assert protocol.upper_interval_threshold_ms == -0.25
    assert protocol.expected_pair_count == 720
    assert len(protocol.configuration_digest) == 64


def test_revision_development_protocol_rejects_evaluation_and_degenerate_selection(
    tmp_path: Path,
) -> None:
    (tmp_path / PROTOCOL.name).write_text(PROTOCOL.read_text(encoding="utf-8"), encoding="utf-8")
    payload = json.loads(DEVELOPMENT_PROTOCOL.read_text(encoding="utf-8"))
    payload["sealed_evaluation"]["outcome_access_allowed"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PermissionError, match="sealed evaluation"):
        load_revision_development_protocol(invalid)

    payload = json.loads(DEVELOPMENT_PROTOCOL.read_text(encoding="utf-8"))
    payload["selection_rule"]["minimum_retained_base_reroute_fraction"] = 0.0
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        load_revision_development_protocol(invalid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("authorized_splits", ["train", "development", "test"], "only train"),
        (
            "estimators",
            {
                "scopes": ["global", "traffic-only", "context-local"],
                "adjustment_set": ["QueuePressureBin"],
                "minimum_per_arm": 10,
                "unsupported_context_action": "DEFER",
                "broader_fallback_for_unsupported_context": True,
            },
            "fallback",
        ),
        (
            "sealed_evaluation",
            {
                "split": "test",
                "status": "AUTHORIZED",
                "output_generated": False,
                "authorization_required": "invalid",
            },
            "unauthorized",
        ),
    ],
)
def test_revision_protocol_rejects_boundary_violations(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    payload = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((ValueError, PermissionError), match=message):
        load_revision_protocol(path)
