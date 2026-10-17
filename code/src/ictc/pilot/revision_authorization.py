"""Single-use authorization for fresh M10R evaluation generation and execution."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from subprocess import run
from typing import cast

from ictc.datasets import DatasetSplit, sha256_file
from ictc.pilot.authorization import TestAccessAuthorization
from ictc.pilot.revision_protocol import load_revision_evaluation_protocol


class RevisionAuthorizationPhase(StrEnum):
    GENERATION = "generation"
    EXECUTION = "execution"


@dataclass(frozen=True, slots=True)
class RevisionEvaluationAuthorization:
    """Review-, commit-, protocol-, matrix-, and output-bound authorization."""

    authorization_id: str
    review_commit: str
    execution_logic_commit: str
    evaluation_protocol_digest: str
    expected_pair_count: int
    corpus_output: Path
    result_output: Path

    def __post_init__(self) -> None:
        if not self.authorization_id:
            raise ValueError("M10R evaluation authorization ID is required")
        if len(self.review_commit) != 40 or len(self.execution_logic_commit) != 40:
            raise ValueError("M10R authorization commits must be full Git hashes")
        if self.expected_pair_count <= 0:
            raise ValueError("M10R authorization requires a positive pair count")
        if not self.corpus_output.is_absolute() or not self.result_output.is_absolute():
            raise ValueError("M10R authorization output paths must be absolute")

    def assert_allows(self, split: DatasetSplit) -> None:
        if split is not DatasetSplit.TEST:
            raise PermissionError("M10R evaluation authorization permits only test")

    @property
    def test_access(self) -> TestAccessAuthorization:
        return TestAccessAuthorization(
            self.authorization_id,
            self.review_commit,
            self.execution_logic_commit,
            self.evaluation_protocol_digest,
            "0" * 64,
            self.expected_pair_count,
            self.result_output,
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _is_ancestor(repo_root: Path, commit: str) -> bool:
    completed = run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def load_revision_evaluation_authorization(
    path: Path,
    *,
    repo_root: Path,
    evaluation_protocol_path: Path,
    matrix_path: Path,
    phase: RevisionAuthorizationPhase,
) -> RevisionEvaluationAuthorization:
    """Load an authorization before generation or after generation but before execution."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("M10R evaluation authorization must be a JSON object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema_version") != "1.0.0" or payload.get("authorized_split") != "test":
        raise ValueError("M10R evaluation authorization schema or split is invalid")

    review_path = (repo_root / cast(str, payload["review_path"])).resolve()
    if _sha256(review_path) != payload.get("review_sha256"):
        raise ValueError("M10R evaluation GO review digest differs")
    review_commit = cast(str, payload["review_commit"])
    execution_commit = cast(str, payload["execution_logic_commit"])
    if not _is_ancestor(repo_root, review_commit) or not _is_ancestor(repo_root, execution_commit):
        raise ValueError("M10R authorization commits are not ancestors of HEAD")

    protocol = load_revision_evaluation_protocol(evaluation_protocol_path, repo_root=repo_root)
    if protocol.configuration_digest != payload.get("evaluation_protocol_digest"):
        raise ValueError("M10R evaluation authorization protocol digest differs")
    if sha256_file(matrix_path) != payload.get("matrix_file_sha256"):
        raise ValueError("M10R evaluation authorization matrix digest differs")
    training_corpus = (repo_root / cast(str, payload["training_corpus"])).resolve()
    from ictc.datasets import manifest_tree_digest

    if manifest_tree_digest(training_corpus) != payload.get("training_manifest_tree_digest"):
        raise ValueError("M10R authorization training corpus digest differs")

    corpus_output = (repo_root / cast(str, payload["corpus_output_directory"])).resolve()
    result_output = (repo_root / cast(str, payload["result_output_directory"])).resolve()
    if phase is RevisionAuthorizationPhase.GENERATION:
        if corpus_output.exists() or result_output.exists():
            raise FileExistsError("M10R evaluation authorization is already consumed")
    else:
        if not corpus_output.is_dir():
            raise FileNotFoundError("M10R authorized evaluation corpus is absent")
        if result_output.exists():
            raise FileExistsError("M10R evaluation execution authorization is consumed")

    authorization_id = payload.get("authorization_id")
    expected_count = payload.get("expected_test_pair_count")
    if not isinstance(authorization_id, str) or not authorization_id:
        raise ValueError("M10R evaluation authorization ID is invalid")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise ValueError("M10R evaluation pair count must be an integer")
    if expected_count != protocol.expected_pair_count:
        raise ValueError("M10R evaluation authorization pair count differs")
    return RevisionEvaluationAuthorization(
        authorization_id,
        review_commit,
        execution_commit,
        protocol.configuration_digest,
        expected_count,
        corpus_output,
        result_output,
    )
