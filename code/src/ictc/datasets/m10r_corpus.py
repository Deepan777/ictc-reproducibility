"""Immutable M10 revision corpus extension with controller-safe topology context."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ictc.datasets.m08_config import CorpusScenario, DatasetSplit
from ictc.datasets.m08_corpus import (
    CorpusProvenance,
    GenerationResult,
    export_corpus_pair,
    sha256_file,
    validate_corpus_pair,
)
from ictc.simulation import PairedSimulation, SimulationConfig, run_paired

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

M10R_EXTENSION_SCHEMA_VERSION = "1.0.0"
M10R_CONTEXT_SOURCE = "frozen pre-simulation path configuration"
M10R_CONTEXT_SCHEMA = pa.schema(
    [
        ("pair_id", pa.string()),
        ("topology_context_id", pa.string()),
        ("service_relation", pa.string()),
        ("propagation_delay_relation", pa.string()),
        ("queue_limit_relation", pa.string()),
        ("source", pa.string()),
        ("fixed_before_simulation", pa.bool_()),
        ("outcome_or_post_action_fields_used", pa.bool_()),
    ]
)
M10R_ADDITIONAL_SCHEMAS = {"topology_context.parquet": M10R_CONTEXT_SCHEMA}


@dataclass(frozen=True, slots=True)
class RevisionCorpusProvenance:
    """Base corpus provenance plus the frozen M10 revision protocol digest."""

    base: CorpusProvenance
    revision_protocol_sha256: str

    def __post_init__(self) -> None:
        value = self.revision_protocol_sha256
        if len(value) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in value
        ):
            raise ValueError("revision protocol SHA-256 must be a 64-character hexadecimal digest")


@dataclass(frozen=True, slots=True)
class _SafeContext:
    service_relation: str
    propagation_delay_relation: str
    queue_limit_relation: str

    @property
    def context_id(self) -> str:
        return (
            f"service={self.service_relation};"
            f"delay={self.propagation_delay_relation};"
            f"queue={self.queue_limit_relation}"
        )


def _schema_descriptor(schema: pa.Schema) -> list[JsonValue]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _relative(candidate: int | float, current: int | float) -> str:
    return "LOWER" if candidate < current else "HIGHER" if candidate > current else "EQUAL"


def _context_for(scenario: CorpusScenario) -> _SafeContext:
    paths = {path.path_id: path for path in scenario.simulation.paths}
    current = paths[scenario.simulation.current_path_id]
    candidate = paths[scenario.simulation.candidate_path_id]
    return _SafeContext(
        _relative(candidate.service_packets_per_tick, current.service_packets_per_tick),
        _relative(candidate.propagation_delay_ms, current.propagation_delay_ms),
        _relative(candidate.queue_limit_packets, current.queue_limit_packets),
    )


def _write_context(path: Path, pair_id: str, context: _SafeContext) -> None:
    table = pa.Table.from_pylist(
        [
            {
                "pair_id": pair_id,
                "topology_context_id": context.context_id,
                "service_relation": context.service_relation,
                "propagation_delay_relation": context.propagation_delay_relation,
                "queue_limit_relation": context.queue_limit_relation,
                "source": M10R_CONTEXT_SOURCE,
                "fixed_before_simulation": True,
                "outcome_or_post_action_fields_used": False,
            }
        ],
        schema=M10R_CONTEXT_SCHEMA,
    )
    pq.write_table(
        table,
        path,
        compression=None,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
    )


def _extend_export(
    scenario: CorpusScenario,
    pair: PairedSimulation,
    provenance: RevisionCorpusProvenance,
    directory: Path,
    *,
    evaluation_authorization_id: str | None,
) -> dict[str, JsonValue]:
    context = _context_for(scenario)
    context_path = directory / "topology_context.parquet"
    _write_context(context_path, pair.pair_id, context)
    manifest_path = directory / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("base corpus manifest must be an object")
    manifest = cast(dict[str, JsonValue], raw)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("base corpus manifest files must be an object")
    files["topology_context.parquet"] = {
        "sha256": sha256_file(context_path),
        "row_count": pq.read_metadata(context_path).num_rows,
        "schema": _schema_descriptor(M10R_CONTEXT_SCHEMA),
    }
    manifest.update(
        {
            "generator": "ictc.datasets.m10r_corpus.export_m10r_corpus_pair",
            "corpus_id": scenario.corpus_id,
            "m10r_extension_schema_version": M10R_EXTENSION_SCHEMA_VERSION,
            "revision_protocol_sha256": provenance.revision_protocol_sha256,
            "topology_context_id": context.context_id,
            "topology_context_source": M10R_CONTEXT_SOURCE,
            "topology_context_fixed_before_simulation": True,
            "topology_context_outcome_or_post_action_fields_used": False,
            "evaluation_authorization_id": evaluation_authorization_id,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def export_m10r_corpus_pair(
    scenario: CorpusScenario,
    pair: PairedSimulation,
    provenance: RevisionCorpusProvenance,
    target: Path,
    *,
    evaluation_authorization_id: str | None = None,
) -> dict[str, JsonValue]:
    """Create one immutable M10R pair, requiring explicit authorization for evaluation."""
    if scenario.split is DatasetSplit.TEST and not evaluation_authorization_id:
        raise PermissionError("sealed M10 revision evaluation generation is not authorized")
    if scenario.split is not DatasetSplit.TEST and evaluation_authorization_id is not None:
        raise PermissionError("M10 revision evaluation authorization cannot label non-test data")
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite revision corpus pair: {target}")
    if pair.config != scenario.simulation:
        raise ValueError("paired simulation does not match the revision corpus scenario")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        staging = Path(temporary) / "pair"
        export_corpus_pair(scenario, pair, provenance.base, staging)
        manifest = _extend_export(
            scenario,
            pair,
            provenance,
            staging,
            evaluation_authorization_id=evaluation_authorization_id,
        )
        os.rename(staging, target)
    validate_m10r_corpus_pair(target, evaluation_authorization_id=evaluation_authorization_id)
    return manifest


def validate_m10r_corpus_pair(
    target: Path, *, evaluation_authorization_id: str | None = None
) -> dict[str, JsonValue]:
    """Validate base integrity and the controller-safe M10R context declaration."""
    manifest = validate_corpus_pair(target, additional_schemas=M10R_ADDITIONAL_SCHEMAS)
    if manifest.get("m10r_extension_schema_version") != M10R_EXTENSION_SCHEMA_VERSION:
        raise ValueError("unexpected M10 revision corpus extension version")
    if manifest.get("corpus_id") != "ictc-m10r-contextual":
        raise ValueError("unexpected M10 revision corpus ID")
    is_test = manifest.get("split") == DatasetSplit.TEST.value
    if is_test and not evaluation_authorization_id:
        raise PermissionError("sealed M10 revision evaluation output is present")
    if is_test and manifest.get("evaluation_authorization_id") != evaluation_authorization_id:
        raise PermissionError("M10 revision evaluation authorization ID differs")
    if not is_test and manifest.get("evaluation_authorization_id") is not None:
        raise PermissionError("non-test M10 revision output has an evaluation authorization")
    protocol_digest = manifest.get("revision_protocol_sha256")
    if not isinstance(protocol_digest, str) or len(protocol_digest) != 64:
        raise ValueError("M10 revision manifest lacks the frozen protocol digest")
    rows = pq.read_table(target / "topology_context.parquet").to_pylist()
    if len(rows) != 1:
        raise ValueError("M10 revision context table must contain exactly one row")
    row = rows[0]
    expected = {
        "pair_id": manifest.get("pair_id"),
        "topology_context_id": manifest.get("topology_context_id"),
        "source": M10R_CONTEXT_SOURCE,
        "fixed_before_simulation": True,
        "outcome_or_post_action_fields_used": False,
    }
    if any(row[key] != value for key, value in expected.items()):
        raise ValueError("M10 revision context row disagrees with its safe manifest declaration")
    relation_values = {
        row[key]
        for key in (
            "service_relation",
            "propagation_delay_relation",
            "queue_limit_relation",
        )
    }
    if not relation_values <= {"LOWER", "EQUAL", "HIGHER"}:
        raise ValueError("M10 revision context contains an invalid relation")
    return manifest


def validate_m10r_corpus(
    corpus_root: Path,
    *,
    expected_split_counts: dict[DatasetSplit, int],
    matrix_config_sha256: str,
    revision_protocol_sha256: str,
) -> dict[str, int]:
    """Validate the complete authorized corpus and prove that no test output is present."""
    if set(expected_split_counts) != {DatasetSplit.TRAIN, DatasetSplit.DEVELOPMENT}:
        raise ValueError("M10 revision corpus validation may require only train and development")
    if any(count <= 0 for count in expected_split_counts.values()):
        raise ValueError("M10 revision expected split counts must be positive")
    run_root = corpus_root / "runs"
    if not run_root.is_dir():
        raise ValueError("M10 revision corpus run directory is missing")
    family_split: dict[str, DatasetSplit] = {}
    seed_split: dict[int, DatasetSplit] = {}
    counts: Counter[DatasetSplit] = Counter()
    for run_directory in sorted(path for path in run_root.iterdir() if path.is_dir()):
        manifest = validate_m10r_corpus_pair(run_directory)
        raw_split = manifest.get("split")
        try:
            split = DatasetSplit(cast(str, raw_split))
        except (TypeError, ValueError) as error:
            raise ValueError("M10 revision manifest contains an invalid split") from error
        if split is DatasetSplit.TEST:
            raise PermissionError("sealed M10 revision evaluation output is present")
        if manifest.get("matrix_config_sha256") != matrix_config_sha256:
            raise ValueError("M10 revision manifest matrix digest mismatch")
        if manifest.get("revision_protocol_sha256") != revision_protocol_sha256:
            raise ValueError("M10 revision manifest protocol digest mismatch")
        family = manifest.get("topology_family_id")
        seed = manifest.get("seed")
        if not isinstance(family, str) or not isinstance(seed, int):
            raise ValueError("M10 revision manifest split keys have invalid types")
        if family in family_split and family_split[family] is not split:
            raise ValueError(f"topology family {family} leaks across revision splits")
        if seed in seed_split and seed_split[seed] is not split:
            raise ValueError(f"seed {seed} leaks across revision splits")
        family_split[family] = split
        seed_split[seed] = split
        counts[split] += 1
    if counts != Counter(expected_split_counts):
        actual = {split.value: counts[split] for split in DatasetSplit}
        raise ValueError(f"M10 revision corpus split counts differ from the matrix: {actual}")
    return {split.value: counts[split] for split in expected_split_counts}


def _append_attempt(registry: Path, result: GenerationResult) -> None:
    registry.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(result), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    with registry.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def generate_m10r_scenario(
    scenario: CorpusScenario,
    corpus_root: Path,
    provenance: RevisionCorpusProvenance,
    *,
    runner: Callable[[SimulationConfig], PairedSimulation] | None = None,
    evaluation_authorization_id: str | None = None,
) -> GenerationResult:
    """Generate or resume one authorized M10R scenario and retain every attempt."""
    if scenario.split is DatasetSplit.TEST and not evaluation_authorization_id:
        raise PermissionError("sealed M10 revision evaluation generation is not authorized")
    if scenario.split is not DatasetSplit.TEST and evaluation_authorization_id is not None:
        raise PermissionError("M10 revision evaluation authorization permits only test")
    started_clock = time.perf_counter()
    started = datetime.now(UTC).isoformat()
    target = corpus_root / "runs" / scenario.run_id
    status = "COMPLETED"
    classification = "NONE"
    detail = "created immutable M10 revision corpus pair"
    execute = run_paired if runner is None else runner
    if target.exists():
        try:
            validate_m10r_corpus_pair(
                target, evaluation_authorization_id=evaluation_authorization_id
            )
            status = "SKIPPED_VALID"
            detail = "existing immutable M10 revision output validated"
        except (OSError, ValueError, json.JSONDecodeError) as error:
            status = "CORRUPT" if (target / "manifest.json").is_file() else "INCOMPLETE"
            classification = type(error).__name__
            detail = str(error)
    else:
        try:
            pair = execute(scenario.simulation)
            export_m10r_corpus_pair(
                scenario,
                pair,
                provenance,
                target,
                evaluation_authorization_id=evaluation_authorization_id,
            )
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
    _append_attempt(corpus_root / "attempt_registry.jsonl", result)
    return result


def generate_m10r_corpus(
    scenarios: Iterable[CorpusScenario],
    corpus_root: Path,
    provenance: RevisionCorpusProvenance,
    *,
    evaluation_authorization_id: str | None = None,
) -> tuple[GenerationResult, ...]:
    """Generate a bounded sequence under the supplied split authorization."""
    return tuple(
        generate_m10r_scenario(
            item,
            corpus_root,
            provenance,
            evaluation_authorization_id=evaluation_authorization_id,
        )
        for item in scenarios
    )
