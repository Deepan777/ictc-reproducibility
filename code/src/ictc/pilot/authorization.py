"""Machine-bound, single-output authorization for the one-time M10 test run."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from subprocess import run
from typing import ClassVar, cast

from ictc.datasets import DatasetSplit, manifest_tree_digest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True, slots=True)
class TestAccessAuthorization:
    __test__: ClassVar[bool] = False

    authorization_id: str
    review_commit: str
    execution_logic_commit: str
    protocol_configuration_digest: str
    source_manifest_tree_digest: str
    expected_test_pair_count: int
    output_directory: Path

    def __post_init__(self) -> None:
        if not self.authorization_id:
            raise ValueError("test authorization ID is required")
        if len(self.review_commit) != 40 or len(self.execution_logic_commit) != 40:
            raise ValueError("test authorization commits must be full Git hashes")
        if self.expected_test_pair_count <= 0:
            raise ValueError("test authorization requires a positive pair count")
        if not self.output_directory.is_absolute():
            raise ValueError("test authorization output directory must be absolute")

    def assert_allows(self, split: DatasetSplit) -> None:
        if split is not DatasetSplit.TEST:
            raise PermissionError("test authorization is valid only for the test split")


def _is_ancestor(repo_root: Path, commit: str) -> bool:
    completed = run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode == 0


def load_test_authorization(
    path: Path,
    *,
    repo_root: Path,
    corpus: Path,
    protocol_path: Path,
) -> TestAccessAuthorization:
    from ictc.pilot.protocol import load_frozen_protocol

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("test authorization must be a JSON object")
    raw = cast(dict[str, object], payload)
    if raw.get("schema_version") != "1.0.0" or raw.get("authorized_split") != "test":
        raise ValueError("test authorization schema or split is invalid")
    review_path = (repo_root / cast(str, raw["review_path"])).resolve()
    if _sha256(review_path) != raw.get("review_sha256"):
        raise ValueError("test authorization GO review digest differs")
    review_commit = cast(str, raw["review_commit"])
    execution_commit = cast(str, raw["execution_logic_commit"])
    if not _is_ancestor(repo_root, review_commit) or not _is_ancestor(repo_root, execution_commit):
        raise ValueError("test authorization commits are not ancestors of HEAD")
    protocol = load_frozen_protocol(protocol_path)
    if protocol.configuration_digest != raw.get("protocol_configuration_digest"):
        raise ValueError("test authorization protocol digest differs")
    tree_digest = manifest_tree_digest(corpus)
    if tree_digest != raw.get("source_manifest_tree_digest"):
        raise ValueError("test authorization source manifest digest differs")
    output = (repo_root / cast(str, raw["output_directory"])).resolve()
    if output.exists():
        raise FileExistsError("test authorization is already consumed by its output directory")
    authorization_id = cast(str, raw["authorization_id"])
    expected_count = raw.get("expected_test_pair_count")
    if not isinstance(expected_count, int) or isinstance(expected_count, bool):
        raise ValueError("test authorization pair count must be an integer")
    return TestAccessAuthorization(
        authorization_id,
        review_commit,
        execution_commit,
        protocol.configuration_digest,
        tree_digest,
        expected_count,
        output,
    )
