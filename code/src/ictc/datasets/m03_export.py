"""Immutable Parquet export and schema validation for M03 paired runs."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ictc.simulation.environment import PairedSimulation, SimulationRun

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

EXPORT_SCHEMA_VERSION = "1.0.0"

SCHEDULE_SCHEMA = pa.schema(
    [
        ("tick", pa.int64()),
        ("path_id", pa.string()),
        ("foreground_arrivals", pa.int64()),
        ("cross_arrivals", pa.int64()),
        ("telemetry_missing", pa.bool_()),
        ("telemetry_delay_ticks", pa.int64()),
        ("telemetry_clock_skew_ticks", pa.int64()),
    ]
)

ORACLE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("tick", pa.int64()),
        ("phase", pa.string()),
        ("path_id", pa.string()),
        ("selected_path", pa.bool_()),
        ("true_foreground_arrivals", pa.int64()),
        ("true_cross_arrivals", pa.int64()),
        ("true_service_capacity", pa.int64()),
        ("true_queue_occupancy", pa.int64()),
        ("true_foreground_served", pa.int64()),
        ("true_cross_served", pa.int64()),
        ("true_foreground_dropped", pa.int64()),
        ("true_cross_dropped", pa.int64()),
    ]
)

VISIBLE_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("sample_tick", pa.int64()),
        ("available_tick", pa.int64()),
        ("observed_timestamp_tick", pa.int64()),
        ("path_id", pa.string()),
        ("queue_occupancy", pa.int64()),
        ("utilization", pa.float64()),
        ("sample_age_ticks", pa.int64()),
        ("measurement_delay_ticks", pa.int64()),
        ("clock_skew_ticks", pa.int64()),
        ("stale", pa.bool_()),
        ("missing", pa.bool_()),
        ("source_id", pa.string()),
        ("provenance_digest", pa.string()),
    ]
)

OUTCOME_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("scenario_id", pa.string()),
        ("action_kind", pa.string()),
        ("selected_path_id", pa.string()),
        ("schedule_digest", pa.string()),
        ("mean_latency_ms", pa.float64()),
        ("packet_loss_ratio", pa.float64()),
        ("jitter_ms", pa.float64()),
        ("deadline_violation_rate", pa.float64()),
    ]
)

ACCOUNTING_SCHEMA = pa.schema(
    [
        ("run_id", pa.string()),
        ("path_id", pa.string()),
        ("foreground_generated", pa.int64()),
        ("foreground_delivered", pa.int64()),
        ("foreground_dropped", pa.int64()),
        ("cross_generated", pa.int64()),
        ("cross_delivered", pa.int64()),
        ("cross_dropped", pa.int64()),
        ("final_queue_occupancy", pa.int64()),
    ]
)

EXPECTED_SCHEMAS = {
    "schedule.parquet": SCHEDULE_SCHEMA,
    "oracle.parquet": ORACLE_SCHEMA,
    "visible.parquet": VISIBLE_SCHEMA,
    "outcomes.parquet": OUTCOME_SCHEMA,
    "accounting.parquet": ACCOUNTING_SCHEMA,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _schema_descriptor(schema: pa.Schema) -> list[JsonValue]:
    return [{"name": field.name, "type": str(field.type)} for field in schema]


def _schedule_rows(pair: PairedSimulation) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for tick_index, tick in enumerate(pair.schedule.ticks):
        for path_index, path_id in enumerate(pair.schedule.path_ids):
            impairment = tick.telemetry_impairments[path_index]
            rows.append(
                {
                    "tick": tick_index,
                    "path_id": path_id,
                    "foreground_arrivals": tick.foreground_arrivals,
                    "cross_arrivals": tick.cross_arrivals[path_index],
                    "telemetry_missing": impairment.missing,
                    "telemetry_delay_ticks": impairment.delay_ticks,
                    "telemetry_clock_skew_ticks": impairment.clock_skew_ticks,
                }
            )
    return rows


def _oracle_rows(runs: tuple[SimulationRun, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for record in run.oracle_trace.records:
            rows.append(
                {
                    "run_id": record.run_id,
                    "tick": record.tick,
                    "phase": record.phase,
                    "path_id": record.path_id,
                    "selected_path": record.selected_path,
                    "true_foreground_arrivals": record.true_foreground_arrivals,
                    "true_cross_arrivals": record.true_cross_arrivals,
                    "true_service_capacity": record.true_service_capacity,
                    "true_queue_occupancy": record.true_queue_occupancy,
                    "true_foreground_served": record.true_foreground_served,
                    "true_cross_served": record.true_cross_served,
                    "true_foreground_dropped": record.true_foreground_dropped,
                    "true_cross_dropped": record.true_cross_dropped,
                }
            )
    return rows


def _visible_rows(runs: tuple[SimulationRun, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for record in run.controller_trace.records:
            rows.append(
                {
                    "run_id": record.run_id,
                    "sample_tick": record.sample_tick,
                    "available_tick": record.available_tick,
                    "observed_timestamp_tick": record.observed_timestamp_tick,
                    "path_id": record.path_id,
                    "queue_occupancy": record.queue_occupancy,
                    "utilization": record.utilization,
                    "sample_age_ticks": record.sample_age_ticks,
                    "measurement_delay_ticks": record.measurement_delay_ticks,
                    "clock_skew_ticks": record.clock_skew_ticks,
                    "stale": record.stale,
                    "missing": record.missing,
                    "source_id": record.source_id,
                    "provenance_digest": record.provenance_digest,
                }
            )
    return rows


def _outcome_rows(runs: tuple[SimulationRun, ...]) -> list[dict[str, object]]:
    return [
        {
            "run_id": run.run_id,
            "scenario_id": run.outcome.scenario_id,
            "action_kind": run.action.kind.value,
            "selected_path_id": run.selected_path_id,
            "schedule_digest": run.schedule_digest,
            "mean_latency_ms": run.outcome.latency_ms,
            "packet_loss_ratio": run.outcome.packet_loss_ratio,
            "jitter_ms": run.outcome.jitter_ms,
            "deadline_violation_rate": run.outcome.deadline_violation_rate,
        }
        for run in runs
    ]


def _accounting_rows(runs: tuple[SimulationRun, ...]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in runs:
        for item in run.accounting:
            rows.append(
                {
                    "run_id": run.run_id,
                    "path_id": item.path_id,
                    "foreground_generated": item.foreground_generated,
                    "foreground_delivered": item.foreground_delivered,
                    "foreground_dropped": item.foreground_dropped,
                    "cross_generated": item.cross_generated,
                    "cross_delivered": item.cross_delivered,
                    "cross_dropped": item.cross_dropped,
                    "final_queue_occupancy": item.final_queue_occupancy,
                }
            )
    return rows


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


def _write_export(pair: PairedSimulation, directory: Path) -> dict[str, JsonValue]:
    runs = (pair.baseline, pair.intervention)
    rows_by_file = {
        "schedule.parquet": _schedule_rows(pair),
        "oracle.parquet": _oracle_rows(runs),
        "visible.parquet": _visible_rows(runs),
        "outcomes.parquet": _outcome_rows(runs),
        "accounting.parquet": _accounting_rows(runs),
    }
    files: dict[str, JsonValue] = {}
    for name, schema in EXPECTED_SCHEMAS.items():
        rows = rows_by_file[name]
        path = directory / name
        _write_parquet(path, schema, rows)
        files[name] = {
            "sha256": _sha256(path),
            "row_count": len(rows),
            "schema": _schema_descriptor(schema),
        }

    manifest: dict[str, JsonValue] = {
        "schema_version": EXPORT_SCHEMA_VERSION,
        "generator": "ictc.datasets.m03_export.export_pair",
        "pair_id": pair.pair_id,
        "scenario_id": pair.config.scenario_id,
        "config": cast(JsonValue, pair.config.model_dump(mode="json")),
        "config_hash": pair.config.config_hash,
        "schedule_digest": pair.schedule.digest,
        "baseline_action": pair.baseline.action.kind.value,
        "intervention_action": pair.intervention.action.kind.value,
        "paired_inputs_identical_except_action": True,
        "latency_effect_ms": pair.latency_effect_ms,
        "files": files,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def export_pair(pair: PairedSimulation, target: Path) -> dict[str, JsonValue]:
    """Atomically create a new paired Parquet export; never overwrite."""
    target = target.resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite export: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}.", dir=target.parent) as temporary:
        temporary_path = Path(temporary)
        manifest = _write_export(pair, temporary_path)
        os.rename(temporary_path, target)
    validate_export(target)
    return manifest


def validate_export(target: Path) -> dict[str, JsonValue]:
    """Validate manifest hashes, row counts, schemas, and leakage boundary."""
    manifest_path = target / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("export manifest must be a JSON object")
    manifest = cast(dict[str, JsonValue], raw)
    if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION:
        raise ValueError("unexpected M03 export schema version")
    if manifest.get("paired_inputs_identical_except_action") is not True:
        raise ValueError("manifest does not preserve the paired-input invariant")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_SCHEMAS):
        raise ValueError("manifest file set does not match the M03 schema")

    for name, expected_schema in EXPECTED_SCHEMAS.items():
        metadata = files[name]
        if not isinstance(metadata, dict):
            raise ValueError(f"invalid file metadata for {name}")
        path = target / name
        if not path.is_file():
            raise ValueError(f"missing exported Parquet file: {name}")
        if metadata.get("sha256") != _sha256(path):
            raise ValueError(f"checksum mismatch for {name}")
        if metadata.get("row_count") != pq.read_metadata(path).num_rows:
            raise ValueError(f"row-count mismatch for {name}")
        if pq.read_schema(path) != expected_schema:
            raise ValueError(f"schema mismatch for {name}")
        if metadata.get("schema") != _schema_descriptor(expected_schema):
            raise ValueError(f"manifest schema descriptor mismatch for {name}")

    visible_names = set(VISIBLE_SCHEMA.names)
    forbidden = {
        name
        for name in visible_names
        if name.startswith("true_")
        or "cross" in name.lower()
        or "capacity" in name.lower()
        or "oracle" in name.lower()
    }
    if forbidden:
        raise ValueError(f"controller-visible schema exposes hidden fields: {sorted(forbidden)}")
    return manifest
