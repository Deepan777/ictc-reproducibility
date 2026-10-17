"""M08 corpus immutability, validation, registry, split, and resource tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from ictc.datasets import (
    RUN_REGISTRY_SCHEMA,
    SCENARIO_SCHEMA,
    CorpusProvenance,
    DatasetSplit,
    export_corpus_pair,
    generate_corpus,
    generate_scenario,
    load_corpus_matrix,
    manifest_tree_digest,
    project_resource_need,
    read_attempt_registry,
    sha256_file,
    validate_corpus_pair,
    validate_split_isolation,
)
from ictc.simulation import PairedSimulation, SimulationConfig, run_paired

ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "configs/experiments/m08_micro_matrix.json"


def _provenance() -> CorpusProvenance:
    return CorpusProvenance(
        git_commit="A" * 40,
        environment_sha256="B" * 64,
        matrix_config_sha256="C" * 64,
    )


def _bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def _rewrite_scenario_field(run_directory: Path, field: str, value: object) -> None:
    table_path = run_directory / "scenario.parquet"
    rows = pq.read_table(table_path).to_pylist()
    rows[0][field] = value
    pq.write_table(
        pa.Table.from_pylist(rows, schema=SCENARIO_SCHEMA),
        table_path,
        compression=None,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )
    manifest_path = run_directory / "manifest.json"
    raw = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    raw[field] = value
    file_metadata = cast(dict[str, Any], raw["files"])
    cast(dict[str, Any], file_metadata["scenario.parquet"])["sha256"] = sha256_file(table_path)
    if field in RUN_REGISTRY_SCHEMA.names:
        registry_path = run_directory / "run_registry.parquet"
        registry_rows = pq.read_table(registry_path).to_pylist()
        registry_rows[0][field] = value
        pq.write_table(
            pa.Table.from_pylist(registry_rows, schema=RUN_REGISTRY_SCHEMA),
            registry_path,
            compression=None,
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        )
        cast(dict[str, Any], file_metadata["run_registry.parquet"])["sha256"] = sha256_file(
            registry_path
        )
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")


def test_export_is_complete_valid_and_byte_deterministic(tmp_path: Path) -> None:
    scenario = load_corpus_matrix(MICRO).scenarios()[0]
    pair = run_paired(scenario.simulation)
    first = tmp_path / "first"
    second = tmp_path / "second"

    manifest = export_corpus_pair(scenario, pair, _provenance(), first)
    export_corpus_pair(scenario, pair, _provenance(), second)

    assert validate_corpus_pair(first) == manifest
    assert _bytes(first) == _bytes(second)
    assert manifest["synthetic_origin"] is True
    assert manifest["topology_abstraction"] == "abstract-two-path-parameter-family"
    assert len(cast(dict[str, object], manifest["files"])) == 11
    with pytest.raises(FileExistsError, match="overwrite"):
        export_corpus_pair(scenario, pair, _provenance(), first)


def test_export_rejects_mismatched_pair_and_bad_provenance(tmp_path: Path) -> None:
    scenarios = load_corpus_matrix(MICRO).scenarios()
    with pytest.raises(ValueError, match="does not match"):
        export_corpus_pair(
            scenarios[0], run_paired(scenarios[1].simulation), _provenance(), tmp_path / "bad"
        )
    with pytest.raises(ValueError, match="git_commit"):
        CorpusProvenance("not-a-digest", "B" * 64, "C" * 64)


def test_checksum_and_manifest_tampering_are_detected(tmp_path: Path) -> None:
    scenario = load_corpus_matrix(MICRO).scenarios()[0]
    target = tmp_path / "pair"
    export_corpus_pair(scenario, run_paired(scenario.simulation), _provenance(), target)
    with (target / "visible.parquet").open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_corpus_pair(target)

    target = tmp_path / "wrong-origin"
    export_corpus_pair(scenario, run_paired(scenario.simulation), _provenance(), target)
    manifest_path = target / "manifest.json"
    raw = cast(dict[str, Any], json.loads(manifest_path.read_text(encoding="utf-8")))
    raw["synthetic_origin"] = False
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="synthetic origin"):
        validate_corpus_pair(target)


def test_resume_skip_corruption_incomplete_and_failure_are_retained(tmp_path: Path) -> None:
    scenarios = load_corpus_matrix(MICRO).scenarios()
    root = tmp_path / "corpus"
    completed = generate_scenario(scenarios[0], root, _provenance())
    skipped = generate_scenario(scenarios[0], root, _provenance())
    assert completed.status == "COMPLETED"
    assert skipped.status == "SKIPPED_VALID"

    target = root / "runs" / scenarios[0].run_id
    with (target / "outcomes.parquet").open("ab") as handle:
        handle.write(b"corrupt")
    corrupt = generate_scenario(scenarios[0], root, _provenance())
    assert corrupt.status == "CORRUPT"
    assert (target / "outcomes.parquet").read_bytes().endswith(b"corrupt")

    incomplete_target = root / "runs" / scenarios[1].run_id
    incomplete_target.mkdir(parents=True)
    incomplete = generate_scenario(scenarios[1], root, _provenance())
    assert incomplete.status == "INCOMPLETE"

    def fail(_: SimulationConfig) -> PairedSimulation:
        raise RuntimeError("injected runner failure")

    failed = generate_scenario(scenarios[2], root, _provenance(), runner=fail)
    assert failed.status == "FAILED"
    assert failed.failure_classification == "RuntimeError"
    registry = read_attempt_registry(root / "attempt_registry.jsonl")
    assert [item.status for item in registry] == [
        "COMPLETED",
        "SKIPPED_VALID",
        "CORRUPT",
        "INCOMPLETE",
        "FAILED",
    ]


def test_registry_rejects_invalid_json_lines(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    registry.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_attempt_registry(registry)
    registry.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        read_attempt_registry(registry)


def test_complete_micro_corpus_has_family_and_seed_isolation(tmp_path: Path) -> None:
    matrix = load_corpus_matrix(MICRO)
    results = generate_corpus(matrix.scenarios(), tmp_path, _provenance())

    assert all(item.status == "COMPLETED" for item in results)
    assert validate_split_isolation(tmp_path) == {
        DatasetSplit.TRAIN.value: 4,
        DatasetSplit.DEVELOPMENT.value: 4,
        DatasetSplit.TEST.value: 4,
    }


def test_split_validator_detects_family_and_seed_leakage(tmp_path: Path) -> None:
    scenarios = load_corpus_matrix(MICRO).scenarios()
    selected = (scenarios[0], next(item for item in scenarios if item.split is DatasetSplit.TEST))
    generate_corpus(selected, tmp_path, _provenance())
    test_run = tmp_path / "runs" / selected[1].run_id
    _rewrite_scenario_field(test_run, "topology_family_id", selected[0].topology_family_id)
    with pytest.raises(ValueError, match="topology family .* leaks"):
        validate_split_isolation(tmp_path)

    _rewrite_scenario_field(test_run, "topology_family_id", selected[1].topology_family_id)
    _rewrite_scenario_field(test_run, "seed", selected[0].seed)
    with pytest.raises(ValueError, match="seed .* leaks"):
        validate_split_isolation(tmp_path)


def test_split_validator_requires_runs_and_test_family(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run directory"):
        validate_split_isolation(tmp_path)
    scenario = load_corpus_matrix(MICRO).scenarios()[0]
    generate_scenario(scenario, tmp_path, _provenance())
    with pytest.raises(ValueError, match="no isolated test"):
        validate_split_isolation(tmp_path)


def test_resource_projection_enforces_policy_formula() -> None:
    accepted = project_resource_need(
        measured_pairs=2,
        measured_bytes=200,
        measured_seconds=4.0,
        projected_pairs=10,
        observed_free_bytes=6 * 1024**3,
    )
    rejected = project_resource_need(
        measured_pairs=2,
        measured_bytes=2 * 1024**3,
        measured_seconds=4.0,
        projected_pairs=10,
        observed_free_bytes=6 * 1024**3,
    )
    assert accepted.bytes_per_pair == 100
    assert accepted.projected_incremental_bytes == 1000
    assert accepted.authorized is True
    assert rejected.authorized is False

    with pytest.raises(ValueError, match="measurements"):
        project_resource_need(
            measured_pairs=0,
            measured_bytes=1,
            measured_seconds=1,
            projected_pairs=1,
            observed_free_bytes=1,
        )
    with pytest.raises(ValueError, match="projection inputs"):
        project_resource_need(
            measured_pairs=1,
            measured_bytes=1,
            measured_seconds=1,
            projected_pairs=0,
            observed_free_bytes=1,
        )
    with pytest.raises(ValueError, match="policy"):
        project_resource_need(
            measured_pairs=1,
            measured_bytes=1,
            measured_seconds=1,
            projected_pairs=1,
            observed_free_bytes=1,
            multiplier=0.5,
        )


def test_sha256_file_matches_known_digest(tmp_path: Path) -> None:
    path = tmp_path / "value"
    path.write_bytes(b"abc")
    assert sha256_file(path) == ("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD")


def test_manifest_tree_digest_is_ordered_and_requires_manifests(tmp_path: Path) -> None:
    scenarios = load_corpus_matrix(MICRO).scenarios()
    with pytest.raises(ValueError, match="run directory"):
        manifest_tree_digest(tmp_path)
    (tmp_path / "runs").mkdir()
    with pytest.raises(ValueError, match="no run directories"):
        manifest_tree_digest(tmp_path)

    generate_corpus((scenarios[1], scenarios[0]), tmp_path, _provenance())
    first = manifest_tree_digest(tmp_path)
    assert first == manifest_tree_digest(tmp_path)
    (tmp_path / "runs" / scenarios[0].run_id / "manifest.json").unlink()
    with pytest.raises(ValueError, match="run has no manifest"):
        manifest_tree_digest(tmp_path)
