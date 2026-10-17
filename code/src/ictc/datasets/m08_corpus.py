"""Immutable generation, validation, resume, and resource gates for M08."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ictc.datasets.m03_export import (
    ACCOUNTING_SCHEMA,
    ORACLE_SCHEMA,
    OUTCOME_SCHEMA,
    SCHEDULE_SCHEMA,
    VISIBLE_SCHEMA,
    export_pair,
)
from ictc.datasets.m08_config import CorpusScenario, DatasetSplit
from ictc.simulation import PairedSimulation, SimulationConfig, run_paired

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

CORPUS_SCHEMA_VERSION = "1.0.0"
GENERATOR_VERSION = "0.8.0"

SCENARIO_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("scenario_id", pa.string()),
        ("split", pa.string()),
        ("topology_family_id", pa.string()),
        ("topology_abstraction", pa.string()),
        ("traffic_level_id", pa.string()),
        ("observation_regime_id", pa.string()),
        ("seed_group_id", pa.string()),
        ("seed", pa.int64()),
        ("simulation_config_hash", pa.string()),
        ("schedule_digest", pa.string()),
    ]
)

ACTION_PROPOSAL_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("run_id", pa.string()),
        ("proposal_id", pa.string()),
        ("action_kind", pa.string()),
        ("target_path_id", pa.string()),
        ("is_baseline", pa.bool_()),
    ]
)

CONTRACT_SCHEMA = pa.schema(
    [
        ("contract_id", pa.string()),
        ("proposal_id", pa.string()),
        ("run_id", pa.string()),
        ("record_kind", pa.string()),
        ("required_variables_json", pa.string()),
        ("freshness_limit_ticks", pa.int64()),
        ("synchronization_limit_ticks", pa.int64()),
        ("minimum_source_diversity", pa.int64()),
    ]
)

PROTOCOL_EVENT_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("contract_id", pa.string()),
        ("run_id", pa.string()),
        ("sequence_number", pa.int64()),
        ("event_type", pa.string()),
        ("record_kind", pa.string()),
    ]
)

FAILURE_INJECTION_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("observation_regime_id", pa.string()),
        ("failure_classification", pa.string()),
        ("minimum_delay_ticks", pa.int64()),
        ("maximum_delay_ticks", pa.int64()),
        ("missing_probability", pa.float64()),
        ("maximum_clock_skew_ticks", pa.int64()),
    ]
)

RUN_REGISTRY_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("scenario_id", pa.string()),
        ("split", pa.string()),
        ("seed", pa.int64()),
        ("status", pa.string()),
        ("generator_version", pa.string()),
        ("git_commit", pa.string()),
        ("environment_sha256", pa.string()),
        ("matrix_config_sha256", pa.string()),
    ]
)

EXPECTED_SCHEMAS = {
    "scenario.parquet": SCENARIO_SCHEMA,
    "schedule.parquet": SCHEDULE_SCHEMA,
    "oracle.parquet": ORACLE_SCHEMA,
    "visible.parquet": VISIBLE_SCHEMA,
    "action_proposals.parquet": ACTION_PROPOSAL_SCHEMA,
    "contracts.parquet": CONTRACT_SCHEMA,
    "protocol_events.parquet": PROTOCOL_EVENT_SCHEMA,
    "outcomes.parquet": OUTCOME_SCHEMA,
    "failure_injections.parquet": FAILURE_INJECTION_SCHEMA,
    "accounting.parquet": ACCOUNTING_SCHEMA,
    "run_registry.parquet": RUN_REGISTRY_SCHEMA,
}


@dataclass(frozen=True, slots=True)
class CorpusProvenance:
    """Stable code, environment, and matrix provenance recorded in each run."""

    git_commit: str
    environment_sha256: str
    matrix_config_sha256: str
    generator_version: str = GENERATOR_VERSION

    def __post_init__(self) -> None:
        for label, value in (
            ("git_commit", self.git_commit),
            ("environment_sha256", self.environment_sha256),
            ("matrix_config_sha256", self.matrix_config_sha256),
        ):
            if not value or any(character not in "0123456789abcdefABCDEF" for character in value):
                raise ValueError(f"{label} must be a hexadecimal digest")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """One append-only generation-attempt result."""

    run_id: str
    scenario_id: str
    split: str
    seed: int
    status: str
    failure_classification: str
    output_relpath: str
    detail: str
    started_at_utc: str
    completed_at_utc: str
    elapsed_seconds: float


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    """Measured micro-pilot projection for a larger immutable matrix."""

    measured_pairs: int
    measured_bytes: int
    measured_seconds: float
    bytes_per_pair: float
    seconds_per_pair: float
    projected_pairs: int
    projected_incremental_bytes: int
    projected_seconds: float
    required_free_bytes: int
    observed_free_bytes: int
    authorized: bool


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _schema_descriptor(schema: pa.Schema) -> list[JsonValue]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _write_parquet(path: Path, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression=None,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )


def _extra_rows(
    scenario: CorpusScenario,
    pair: PairedSimulation,
    provenance: CorpusProvenance,
) -> dict[str, list[dict[str, object]]]:
    runs = (pair.baseline, pair.intervention)
    proposals: list[dict[str, object]] = []
    contracts: list[dict[str, object]] = []
    events: list[dict[str, object]] = []
    requirements = json.dumps(
        ["queue_occupancy", "utilization", "sample_age_ticks", "provenance_digest"],
        separators=(",", ":"),
    )
    for run_index, run in enumerate(runs):
        proposal_id = f"proposal-{scenario.run_id}-{run_index}"
        contract_id = f"contract-{scenario.run_id}-{run_index}"
        proposals.append(
            {
                "pair_id": pair.pair_id,
                "run_id": run.run_id,
                "proposal_id": proposal_id,
                "action_kind": run.action.kind.value,
                "target_path_id": run.action.path_id,
                "is_baseline": run_index == 0,
            }
        )
        contracts.append(
            {
                "contract_id": contract_id,
                "proposal_id": proposal_id,
                "run_id": run.run_id,
                "record_kind": "synthetic-mechanism-declaration",
                "required_variables_json": requirements,
                "freshness_limit_ticks": scenario.simulation.telemetry.stale_after_ticks,
                "synchronization_limit_ticks": (
                    scenario.simulation.telemetry.maximum_clock_skew_ticks
                ),
                "minimum_source_diversity": 1,
            }
        )
        for sequence, event_type in enumerate(
            ("ACTION_PROPOSED", "CONTRACT_DECLARED", "COMMIT_RECORDED", "RUN_COMPLETED"),
            start=1,
        ):
            events.append(
                {
                    "event_id": f"event-{scenario.run_id}-{run_index}-{sequence}",
                    "contract_id": contract_id,
                    "run_id": run.run_id,
                    "sequence_number": sequence,
                    "event_type": event_type,
                    "record_kind": "synthetic-deterministic-protocol-trace",
                }
            )

    telemetry = scenario.simulation.telemetry
    return {
        "scenario.parquet": [
            {
                "pair_id": pair.pair_id,
                "scenario_id": scenario.simulation.scenario_id,
                "split": scenario.split.value,
                "topology_family_id": scenario.topology_family_id,
                "topology_abstraction": scenario.topology_abstraction,
                "traffic_level_id": scenario.traffic_level_id,
                "observation_regime_id": scenario.observation_regime_id,
                "seed_group_id": scenario.seed_group_id,
                "seed": scenario.seed,
                "simulation_config_hash": scenario.simulation.config_hash,
                "schedule_digest": pair.schedule.digest,
            }
        ],
        "action_proposals.parquet": proposals,
        "contracts.parquet": contracts,
        "protocol_events.parquet": events,
        "failure_injections.parquet": [
            {
                "pair_id": pair.pair_id,
                "observation_regime_id": scenario.observation_regime_id,
                "failure_classification": scenario.failure_classification,
                "minimum_delay_ticks": telemetry.minimum_delay_ticks,
                "maximum_delay_ticks": telemetry.maximum_delay_ticks,
                "missing_probability": telemetry.missing_probability,
                "maximum_clock_skew_ticks": telemetry.maximum_clock_skew_ticks,
            }
        ],
        "run_registry.parquet": [
            {
                "pair_id": pair.pair_id,
                "scenario_id": scenario.simulation.scenario_id,
                "split": scenario.split.value,
                "seed": scenario.seed,
                "status": "COMPLETED",
                "generator_version": provenance.generator_version,
                "git_commit": provenance.git_commit,
                "environment_sha256": provenance.environment_sha256,
                "matrix_config_sha256": provenance.matrix_config_sha256,
            }
        ],
    }


def _write_corpus_export(
    scenario: CorpusScenario,
    pair: PairedSimulation,
    provenance: CorpusProvenance,
    directory: Path,
) -> dict[str, JsonValue]:
    base_manifest_path = directory / "manifest.json"
    base_manifest = cast(
        dict[str, JsonValue], json.loads(base_manifest_path.read_text(encoding="utf-8"))
    )
    base_manifest_path.unlink()
    rows_by_file = _extra_rows(scenario, pair, provenance)
    for name, rows in rows_by_file.items():
        _write_parquet(directory / name, EXPECTED_SCHEMAS[name], rows)

    files: dict[str, JsonValue] = {}
    for name, schema in EXPECTED_SCHEMAS.items():
        path = directory / name
        files[name] = {
            "sha256": sha256_file(path),
            "row_count": pq.read_metadata(path).num_rows,
            "schema": _schema_descriptor(schema),
        }

    manifest: dict[str, JsonValue] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "generator": "ictc.datasets.m08_corpus.export_corpus_pair",
        "generator_version": provenance.generator_version,
        "run_id": scenario.run_id,
        "pair_id": pair.pair_id,
        "scenario_id": scenario.simulation.scenario_id,
        "split": scenario.split.value,
        "topology_family_id": scenario.topology_family_id,
        "topology_abstraction": scenario.topology_abstraction,
        "traffic_level_id": scenario.traffic_level_id,
        "observation_regime_id": scenario.observation_regime_id,
        "seed_group_id": scenario.seed_group_id,
        "seed": scenario.seed,
        "simulation_config": cast(JsonValue, scenario.simulation.model_dump(mode="json")),
        "simulation_config_hash": scenario.simulation.config_hash,
        "matrix_config_sha256": provenance.matrix_config_sha256,
        "environment_sha256": provenance.environment_sha256,
        "git_commit": provenance.git_commit,
        "schedule_digest": pair.schedule.digest,
        "paired_inputs_identical_except_action": True,
        "synthetic_origin": True,
        "m03_export_schema_version": base_manifest["schema_version"],
        "files": files,
    }
    base_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def export_corpus_pair(
    scenario: CorpusScenario,
    pair: PairedSimulation,
    provenance: CorpusProvenance,
    target: Path,
) -> dict[str, JsonValue]:
    """Atomically create one immutable M08 pair directory; never overwrite."""
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite corpus pair: {target}")
    if pair.config != scenario.simulation:
        raise ValueError("paired simulation does not match the corpus scenario")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        staging = Path(temporary) / "pair"
        export_pair(pair, staging)
        manifest = _write_corpus_export(scenario, pair, provenance, staging)
        os.rename(staging, target)
    validate_corpus_pair(target)
    return manifest


def validate_corpus_pair(
    target: Path, *, additional_schemas: dict[str, pa.Schema] | None = None
) -> dict[str, JsonValue]:
    """Validate checksums, schemas, pairing, provenance, and optional declared extensions."""
    expected_schemas = dict(EXPECTED_SCHEMAS)
    if additional_schemas:
        overlap = set(expected_schemas) & set(additional_schemas)
        if overlap:
            raise ValueError(f"extension schemas cannot replace base tables: {sorted(overlap)}")
        expected_schemas.update(additional_schemas)
    manifest_path = target / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("corpus manifest must be a JSON object")
    manifest = cast(dict[str, JsonValue], raw)
    if manifest.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ValueError("unexpected M08 corpus schema version")
    if manifest.get("generator_version") != GENERATOR_VERSION:
        raise ValueError("unexpected M08 generator version")
    if manifest.get("paired_inputs_identical_except_action") is not True:
        raise ValueError("manifest does not preserve paired input identity")
    if manifest.get("synthetic_origin") is not True:
        raise ValueError("manifest must declare synthetic origin")
    if manifest.get("topology_abstraction") != "abstract-two-path-parameter-family":
        raise ValueError("manifest overstates the supported topology abstraction")
    if manifest.get("split") not in {item.value for item in DatasetSplit}:
        raise ValueError("invalid corpus split")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(expected_schemas):
        raise ValueError("manifest file set does not match the M08 schema")
    actual_files = {path.name for path in target.iterdir() if path.is_file()}
    if actual_files != {*expected_schemas, "manifest.json"}:
        raise ValueError("corpus directory contains an undeclared or missing file")

    for name, expected_schema in expected_schemas.items():
        metadata = files[name]
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid file metadata for {name}")
        path = target / name
        if not path.is_file():
            raise ValueError(f"missing corpus table: {name}")
        if metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"checksum mismatch for {name}")
        if metadata.get("row_count") != pq.read_metadata(path).num_rows:
            raise ValueError(f"row-count mismatch for {name}")
        if pq.read_schema(path) != expected_schema:
            raise ValueError(f"schema mismatch for {name}")
        if metadata.get("schema") != _schema_descriptor(expected_schema):
            raise ValueError(f"schema descriptor mismatch for {name}")

    visible_names = set(pq.read_schema(target / "visible.parquet").names)
    forbidden = {
        name
        for name in visible_names
        if name.startswith("true_")
        or "oracle" in name.lower()
        or "cross" in name.lower()
        or "capacity" in name.lower()
    }
    if forbidden:
        raise ValueError(f"visible table exposes oracle fields: {sorted(forbidden)}")

    scenarios = pq.read_table(target / "scenario.parquet").to_pylist()
    if len(scenarios) != 1:
        raise ValueError("corpus pair must contain exactly one scenario row")
    scenario_row = scenarios[0]
    scenario_manifest_keys = (
        "pair_id",
        "scenario_id",
        "split",
        "topology_family_id",
        "topology_abstraction",
        "traffic_level_id",
        "observation_regime_id",
        "seed_group_id",
        "seed",
        "simulation_config_hash",
        "schedule_digest",
    )
    if any(scenario_row[key] != manifest.get(key) for key in scenario_manifest_keys):
        raise ValueError("scenario row disagrees with the manifest")

    outcomes = pq.read_table(target / "outcomes.parquet").to_pylist()
    if len(outcomes) != 2:
        raise ValueError("paired corpus export must contain two outcomes")
    actions = {row["action_kind"] for row in outcomes}
    if actions != {"KEEP_CURRENT_PATH", "REROUTE_TO_PATH"}:
        raise ValueError("paired outcome actions are invalid")
    schedule_digests = {row["schedule_digest"] for row in outcomes}
    if schedule_digests != {manifest.get("schedule_digest")}:
        raise ValueError("paired outcomes do not share the manifest schedule")
    proposals = pq.read_table(target / "action_proposals.parquet").to_pylist()
    if {row["run_id"] for row in proposals} != {row["run_id"] for row in outcomes}:
        raise ValueError("action proposals and outcomes refer to different runs")
    registry_rows = pq.read_table(target / "run_registry.parquet").to_pylist()
    if len(registry_rows) != 1:
        raise ValueError("corpus pair must contain exactly one run-registry row")
    registry_row = registry_rows[0]
    registry_manifest_keys = (
        "pair_id",
        "scenario_id",
        "split",
        "seed",
        "generator_version",
        "git_commit",
        "environment_sha256",
        "matrix_config_sha256",
    )
    if registry_row["status"] != "COMPLETED" or any(
        registry_row[key] != manifest.get(key) for key in registry_manifest_keys
    ):
        raise ValueError("run-registry row disagrees with the manifest")
    return manifest


def validate_split_isolation(corpus_root: Path) -> dict[str, int]:
    """Validate family and seed isolation across all completed corpus runs."""
    family_split: dict[str, str] = {}
    seed_split: dict[int, str] = {}
    split_counts = {item.value: 0 for item in DatasetSplit}
    run_root = corpus_root / "runs"
    if not run_root.is_dir():
        raise ValueError("corpus run directory is missing")
    for run_directory in sorted(path for path in run_root.iterdir() if path.is_dir()):
        manifest = validate_corpus_pair(run_directory)
        family = manifest.get("topology_family_id")
        split = manifest.get("split")
        seed = manifest.get("seed")
        if not isinstance(family, str) or not isinstance(split, str) or not isinstance(seed, int):
            raise ValueError("manifest split keys have invalid types")
        if family in family_split and family_split[family] != split:
            raise ValueError(f"topology family {family} leaks across splits")
        if seed in seed_split and seed_split[seed] != split:
            raise ValueError(f"seed {seed} leaks across splits")
        family_split[family] = split
        seed_split[seed] = split
        split_counts[split] += 1
    if split_counts[DatasetSplit.TEST.value] == 0:
        raise ValueError("corpus has no isolated test-family run")
    return split_counts


def _append_attempt(registry: Path, result: GenerationResult) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with registry.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_attempt_registry(registry: Path) -> tuple[GenerationResult, ...]:
    """Read and validate the append-only JSON-lines attempt registry."""
    results: list[GenerationResult] = []
    for line_number, line in enumerate(registry.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TypeError("record is not an object")
            results.append(GenerationResult(**raw))
        except (json.JSONDecodeError, TypeError) as error:
            raise ValueError(f"invalid attempt registry record at line {line_number}") from error
    return tuple(results)


def generate_scenario(
    scenario: CorpusScenario,
    corpus_root: Path,
    provenance: CorpusProvenance,
    *,
    runner: Callable[[SimulationConfig], PairedSimulation] | None = None,
) -> GenerationResult:
    """Generate or classify one immutable scenario and append every attempt."""
    started_clock = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    target = corpus_root / "runs" / scenario.run_id
    registry = corpus_root / "attempt_registry.jsonl"
    status = "COMPLETED"
    classification = "NONE"
    detail = "created immutable corpus pair"
    execute = run_paired if runner is None else runner
    if target.exists():
        try:
            validate_corpus_pair(target)
            status = "SKIPPED_VALID"
            detail = "existing immutable output validated"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            status = "CORRUPT" if (target / "manifest.json").is_file() else "INCOMPLETE"
            classification = type(error).__name__
            detail = str(error)
    else:
        try:
            pair = execute(scenario.simulation)
            export_corpus_pair(scenario, pair, provenance, target)
        except Exception as error:
            status = "FAILED"
            classification = type(error).__name__
            detail = str(error)
    completed = datetime.now(UTC).isoformat()
    result = GenerationResult(
        run_id=scenario.run_id,
        scenario_id=scenario.simulation.scenario_id,
        split=scenario.split.value,
        seed=scenario.seed,
        status=status,
        failure_classification=classification,
        output_relpath=f"runs/{scenario.run_id}",
        detail=detail,
        started_at_utc=started,
        completed_at_utc=completed,
        elapsed_seconds=time.perf_counter() - started_clock,
    )
    _append_attempt(registry, result)
    return result


def generate_corpus(
    scenarios: Iterable[CorpusScenario],
    corpus_root: Path,
    provenance: CorpusProvenance,
) -> tuple[GenerationResult, ...]:
    """Generate a bounded sequence of scenarios while retaining all failures."""
    return tuple(generate_scenario(item, corpus_root, provenance) for item in scenarios)


def directory_size(path: Path) -> int:
    """Return total regular-file bytes below a directory."""
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def manifest_tree_digest(corpus_root: Path) -> str:
    """Hash the ordered run-ID/manifest-digest index for a corpus directory."""
    run_root = corpus_root / "runs"
    if not run_root.is_dir():
        raise ValueError("corpus run directory is missing")
    digest = hashlib.sha256()
    run_directories = sorted(path for path in run_root.iterdir() if path.is_dir())
    if not run_directories:
        raise ValueError("corpus contains no run directories")
    for run_directory in run_directories:
        manifest = run_directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"run has no manifest: {run_directory.name}")
        record = f"{run_directory.name}\0{sha256_file(manifest)}\n"
        digest.update(record.encode("ascii"))
    return digest.hexdigest().upper()


def project_resource_need(
    *,
    measured_pairs: int,
    measured_bytes: int,
    measured_seconds: float,
    projected_pairs: int,
    observed_free_bytes: int,
    reserve_bytes: int = 5 * 1024**3,
    multiplier: float = 1.30,
) -> ResourceEstimate:
    """Project immutable storage/runtime and enforce the repository formula."""
    if measured_pairs <= 0 or measured_bytes <= 0 or measured_seconds < 0:
        raise ValueError("micro-pilot measurements must be positive")
    if projected_pairs <= 0 or observed_free_bytes < 0:
        raise ValueError("projection inputs must be positive")
    if reserve_bytes < 0 or multiplier < 1:
        raise ValueError("resource policy is invalid")
    bytes_per_pair = measured_bytes / measured_pairs
    seconds_per_pair = measured_seconds / measured_pairs
    projected_bytes = math.ceil(bytes_per_pair * projected_pairs)
    required = math.ceil(projected_bytes * multiplier) + reserve_bytes
    return ResourceEstimate(
        measured_pairs=measured_pairs,
        measured_bytes=measured_bytes,
        measured_seconds=measured_seconds,
        bytes_per_pair=bytes_per_pair,
        seconds_per_pair=seconds_per_pair,
        projected_pairs=projected_pairs,
        projected_incremental_bytes=projected_bytes,
        projected_seconds=seconds_per_pair * projected_pairs,
        required_free_bytes=required,
        observed_free_bytes=observed_free_bytes,
        authorized=observed_free_bytes >= required,
    )


def measure_projection(
    corpus_root: Path,
    results: Iterable[GenerationResult],
    *,
    measured_seconds: float,
    projected_pairs: int,
) -> ResourceEstimate:
    """Measure a micro-pilot directory and project the declared full matrix."""
    completed = sum(result.status == "COMPLETED" for result in results)
    free_bytes = shutil.disk_usage(corpus_root.resolve()).free
    return project_resource_need(
        measured_pairs=completed,
        measured_bytes=directory_size(corpus_root),
        measured_seconds=measured_seconds,
        projected_pairs=projected_pairs,
        observed_free_bytes=free_bytes,
    )
