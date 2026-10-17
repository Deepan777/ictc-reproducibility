"""Authorization checks for the one-shot M14 controller held-out evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from ictc.datasets import DatasetSplit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


@dataclass(frozen=True, slots=True)
class M14ControllerAuthorization:
    """Immutable user-authorized scope for decision-before-truth execution."""

    authorization_id: str
    raw_matrix_root: Path
    expected_test_pair_count: int
    decision_output_root: Path
    training_matrix_root: Path | None = None

    def __post_init__(self) -> None:
        if not self.authorization_id or self.expected_test_pair_count <= 0:
            raise ValueError("M14 authorization identity or test count is invalid")
        if not self.raw_matrix_root.is_absolute() or not self.decision_output_root.is_absolute():
            raise ValueError("M14 authorization paths must be absolute")
        if self.training_matrix_root is not None and not self.training_matrix_root.is_absolute():
            raise ValueError("M14 training matrix root must be absolute")

    def assert_allows(self, split: DatasetSplit) -> None:
        if split is not DatasetSplit.TEST:
            raise PermissionError("M14 controller authorization permits test rows only")


def load_m14_controller_authorization(
    path: Path, *, repository: Path
) -> M14ControllerAuthorization:
    """Validate the recorded user authorization before the one-shot output is created."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("M14 controller authorization schema is invalid")
    if raw.get("authorized_split") != "test":
        raise PermissionError("M14 authorization does not permit held-out test execution")
    review = raw.get("review_path")
    expected_hash = raw.get("review_sha256")
    if not isinstance(review, str) or not isinstance(expected_hash, str):
        raise ValueError("M14 authorization review binding is invalid")
    review_path = (repository / review).resolve()
    if _sha256(review_path) != expected_hash:
        raise ValueError("M14 authorization review digest differs")
    required = (
        "authorization_id",
        "raw_matrix_root",
        "expected_test_pair_count",
        "decision_output_root",
    )
    if any(key not in raw for key in required):
        raise ValueError("M14 authorization is incomplete")
    count = raw["expected_test_pair_count"]
    if not isinstance(count, int) or isinstance(count, bool):
        raise ValueError("M14 authorization test count is invalid")
    authorization = M14ControllerAuthorization(
        str(raw["authorization_id"]),
        Path(str(raw["raw_matrix_root"])),
        count,
        Path(str(raw["decision_output_root"])),
        (
            Path(str(raw["training_matrix_root"]))
            if raw.get("training_matrix_root") is not None
            else None
        ),
    )
    if authorization.decision_output_root.exists():
        raise FileExistsError("M14 controller authorization has already been consumed")
    return authorization
