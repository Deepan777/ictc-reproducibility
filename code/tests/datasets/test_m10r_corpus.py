"""M10 revision corpus context, integrity, resume, and sealed-split tests."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from ictc.datasets import (
    M10R_CONTEXT_SCHEMA,
    CorpusProvenance,
    DatasetSplit,
    RevisionCorpusProvenance,
    generate_m10r_scenario,
    load_corpus_matrix,
    validate_m10r_corpus,
    validate_m10r_corpus_pair,
)
from ictc.pilot import load_revision_protocol

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiments/m10r_revision_matrix.json"
PROTOCOL = ROOT / "configs/experiments/m10r_revision_protocol.json"


def _provenance() -> RevisionCorpusProvenance:
    protocol = load_revision_protocol(PROTOCOL)
    return RevisionCorpusProvenance(
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        protocol.configuration_digest,
    )


def test_m10r_pair_contains_explicit_controller_safe_context(tmp_path: Path) -> None:
    scenario = next(
        item for item in load_corpus_matrix(MATRIX).scenarios() if item.split is DatasetSplit.TRAIN
    )
    result = generate_m10r_scenario(scenario, tmp_path, _provenance())
    pair = tmp_path / result.output_relpath
    manifest = validate_m10r_corpus_pair(pair)
    context = pq.read_table(pair / "topology_context.parquet")

    assert result.status == "COMPLETED"
    assert context.schema == M10R_CONTEXT_SCHEMA
    assert context.num_rows == 1
    row = context.to_pylist()[0]
    assert row["topology_context_id"] == manifest["topology_context_id"]
    assert row["source"] == "frozen pre-simulation path configuration"
    assert row["fixed_before_simulation"] is True
    assert row["outcome_or_post_action_fields_used"] is False
    assert not any("outcome" in name for name in context.schema.names[:-1])


def test_m10r_generation_resumes_valid_pair_and_detects_tampering(tmp_path: Path) -> None:
    scenario = next(
        item
        for item in load_corpus_matrix(MATRIX).scenarios()
        if item.split is DatasetSplit.DEVELOPMENT
    )
    first = generate_m10r_scenario(scenario, tmp_path, _provenance())
    second = generate_m10r_scenario(scenario, tmp_path, _provenance())
    assert first.status == "COMPLETED"
    assert second.status == "SKIPPED_VALID"

    pair = tmp_path / first.output_relpath
    with (pair / "topology_context.parquet").open("ab") as handle:
        handle.write(b"tamper")
    third = generate_m10r_scenario(scenario, tmp_path, _provenance())
    assert third.status == "CORRUPT"
    assert third.failure_classification == "ValueError"


def test_m10r_corpus_validator_checks_digests_counts_and_authorized_splits(
    tmp_path: Path,
) -> None:
    matrix = load_corpus_matrix(MATRIX)
    scenarios = tuple(
        next(item for item in matrix.scenarios() if item.split is split)
        for split in (DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT)
    )
    provenance = _provenance()
    for scenario in scenarios:
        generate_m10r_scenario(scenario, tmp_path, provenance)

    assert validate_m10r_corpus(
        tmp_path,
        expected_split_counts={DatasetSplit.TRAIN: 1, DatasetSplit.DEVELOPMENT: 1},
        matrix_config_sha256="C" * 64,
        revision_protocol_sha256=provenance.revision_protocol_sha256,
    )


def test_m10r_generation_rejects_sealed_evaluation_before_output(tmp_path: Path) -> None:
    scenario = next(
        item for item in load_corpus_matrix(MATRIX).scenarios() if item.split is DatasetSplit.TEST
    )

    with pytest.raises(PermissionError, match="not authorized"):
        generate_m10r_scenario(scenario, tmp_path, _provenance())
    assert not any(tmp_path.iterdir())


def test_m10r_evaluation_generation_requires_and_preserves_authorization(
    tmp_path: Path,
) -> None:
    scenario = next(
        item for item in load_corpus_matrix(MATRIX).scenarios() if item.split is DatasetSplit.TEST
    )

    result = generate_m10r_scenario(
        scenario,
        tmp_path,
        _provenance(),
        evaluation_authorization_id="m10r-evaluation-once-test",
    )
    manifest = validate_m10r_corpus_pair(
        tmp_path / result.output_relpath,
        evaluation_authorization_id="m10r-evaluation-once-test",
    )

    assert result.status == "COMPLETED"
    assert manifest["split"] == "test"
    assert manifest["evaluation_authorization_id"] == "m10r-evaluation-once-test"
    with pytest.raises(PermissionError, match="evaluation output"):
        validate_m10r_corpus_pair(tmp_path / result.output_relpath)


def test_m10r_provenance_requires_protocol_digest() -> None:
    with pytest.raises(ValueError, match="64-character"):
        RevisionCorpusProvenance(
            CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
            "bad",
        )
