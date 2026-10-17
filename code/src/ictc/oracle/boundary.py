"""Decision-before-oracle boundary for independent evaluation labels.

The functions operate on opaque persisted JSONL bytes and never import controller schemas.  A
decision artifact is sealed first; labels can be linked only while those exact bytes and pair IDs
remain unchanged and only when every decision has exactly one independent certificate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ictc.oracle.admissibility import AdmissibilityCertificate, JsonValue

_FORBIDDEN_ORACLE_FIELDS = frozenset(
    {
        "causally_admissible",
        "compatible_effects",
        "identified_effect",
        "oracle_admissible",
        "oracle_reason",
        "potential_outcomes",
        "witness_model_ids",
    }
)


def _contains_forbidden_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            str(key) in _FORBIDDEN_ORACLE_FIELDS or _contains_forbidden_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def _decision_pair_ids(content: bytes) -> tuple[str, ...]:
    if not content:
        raise ValueError("persisted decision artifact must not be empty")
    pair_ids: list[str] = []
    for line_number, raw_line in enumerate(content.splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"decision line {line_number} is not valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError(f"decision line {line_number} must be a JSON object")
        if _contains_forbidden_field(value):
            raise ValueError(f"decision line {line_number} contains oracle-only truth")
        pair_id = value.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise ValueError(f"decision line {line_number} requires a non-empty pair_id")
        pair_ids.append(pair_id)
    if not pair_ids:
        raise ValueError("persisted decision artifact contains no decisions")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("persisted decision pair IDs must be unique")
    return tuple(sorted(pair_ids))


@dataclass(frozen=True, slots=True)
class DecisionArtifactSeal:
    """Digest and membership committed before independent labels are attached."""

    sha256: str
    byte_count: int
    pair_ids: tuple[str, ...]

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "n01-decision-seal-v1",
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "pair_ids": list(self.pair_ids),
        }


def seal_persisted_decisions(content: bytes) -> DecisionArtifactSeal:
    """Validate and hash decisions before any oracle certificate is linked."""
    return DecisionArtifactSeal(
        sha256=hashlib.sha256(content).hexdigest(),
        byte_count=len(content),
        pair_ids=_decision_pair_ids(content),
    )


@dataclass(frozen=True, slots=True)
class PostDecisionOracleLabels:
    """Complete independent labels linked to one unchanged decision artifact."""

    decision_seal: DecisionArtifactSeal
    labels: tuple[tuple[str, AdmissibilityCertificate], ...]

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "n01-post-decision-labels-v1",
            "decision_seal": self.decision_seal.to_payload(),
            "labels": [
                {"pair_id": pair_id, "certificate": certificate.to_payload()}
                for pair_id, certificate in self.labels
            ],
            "controller_exposure": "none; labels linked after the decision-byte seal",
        }


def link_post_decision_labels(
    sealed_decision_content: bytes,
    seal: DecisionArtifactSeal,
    labels_by_pair: dict[str, AdmissibilityCertificate],
) -> PostDecisionOracleLabels:
    """Link a complete label set only if the sealed decisions are byte-identical."""
    current = seal_persisted_decisions(sealed_decision_content)
    if current != seal:
        raise ValueError("persisted decisions changed after the decision seal")
    if set(labels_by_pair) != set(seal.pair_ids):
        raise ValueError("oracle labels must match the sealed decision pair IDs exactly")
    labels = tuple((pair_id, labels_by_pair[pair_id]) for pair_id in seal.pair_ids)
    return PostDecisionOracleLabels(decision_seal=seal, labels=labels)
