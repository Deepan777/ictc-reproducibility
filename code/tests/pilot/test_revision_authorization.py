"""Machine-bound single-use M10R evaluation authorization tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ictc.datasets import manifest_tree_digest, sha256_file
from ictc.pilot import (
    RevisionAuthorizationPhase,
    load_revision_evaluation_authorization,
    load_revision_evaluation_protocol,
)

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / "docs/milestones/M10_revision_repeated_pre_review.md"
PROTOCOL = ROOT / "configs/experiments/m10r_revision_evaluation_protocol.json"
MATRIX = ROOT / "configs/experiments/m10r_revision_matrix.json"
TRAINING = ROOT / "data/raw/m10r_revision_train_development"
REVIEW_COMMIT = "2e836b17fa8a36733faef801f4adfb515fc38474"


def _payload(tmp_path: Path) -> dict[str, object]:
    protocol = load_revision_evaluation_protocol(PROTOCOL, repo_root=ROOT)
    return {
        "schema_version": "1.0.0",
        "authorization_id": "m10r-evaluation-unit-test",
        "authorized_split": "test",
        "review_path": "docs/milestones/M10_revision_repeated_pre_review.md",
        "review_sha256": sha256_file(REVIEW),
        "review_commit": REVIEW_COMMIT,
        "execution_logic_commit": REVIEW_COMMIT,
        "evaluation_protocol_digest": protocol.configuration_digest,
        "matrix_file_sha256": sha256_file(MATRIX),
        "training_corpus": "data/raw/m10r_revision_train_development",
        "training_manifest_tree_digest": manifest_tree_digest(TRAINING),
        "expected_test_pair_count": 720,
        "corpus_output_directory": str(tmp_path / "corpus"),
        "result_output_directory": str(tmp_path / "results"),
    }


def test_revision_evaluation_authorization_binds_generation_outputs(tmp_path: Path) -> None:
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(_payload(tmp_path)), encoding="utf-8")

    authorization = load_revision_evaluation_authorization(
        path,
        repo_root=ROOT,
        evaluation_protocol_path=PROTOCOL,
        matrix_path=MATRIX,
        phase=RevisionAuthorizationPhase.GENERATION,
    )

    assert authorization.authorization_id == "m10r-evaluation-unit-test"
    assert authorization.expected_pair_count == 720
    assert authorization.test_access.authorization_id == authorization.authorization_id


def test_revision_evaluation_authorization_rejects_changed_review(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["review_sha256"] = "0" * 64
    path = tmp_path / "authorization.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="review digest"):
        load_revision_evaluation_authorization(
            path,
            repo_root=ROOT,
            evaluation_protocol_path=PROTOCOL,
            matrix_path=MATRIX,
            phase=RevisionAuthorizationPhase.GENERATION,
        )
