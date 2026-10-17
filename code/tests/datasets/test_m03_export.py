"""Parquet schema, immutability, leakage, and reproducibility tests."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from ictc.datasets import VISIBLE_SCHEMA, export_pair, validate_export
from ictc.simulation import load_simulation_config, run_paired

ROOT = Path(__file__).resolve().parents[2]
TOY_CONFIG = ROOT / "configs/experiments/m03_toy.json"
TRACKED_FIXTURE = ROOT / "docs/milestones/evidence/m03_toy"


def _file_bytes(directory: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def test_export_schema_hashes_and_pair_invariant(tmp_path: Path) -> None:
    pair = run_paired(load_simulation_config(TOY_CONFIG))
    target = tmp_path / "export"
    manifest = export_pair(pair, target)

    assert validate_export(target) == manifest
    assert manifest["paired_inputs_identical_except_action"] is True
    assert manifest["latency_effect_ms"] == -8.0
    assert pq.read_table(target / "outcomes.parquet").num_rows == 2
    assert pq.read_table(target / "schedule.parquet").num_rows == 8
    assert set(pq.read_schema(target / "visible.parquet").names) == set(VISIBLE_SCHEMA.names)
    assert not any(name.startswith("true_") for name in VISIBLE_SCHEMA.names)
    assert not any("cross" in name or "capacity" in name for name in VISIBLE_SCHEMA.names)


def test_export_refuses_overwrite_and_is_byte_deterministic(tmp_path: Path) -> None:
    pair = run_paired(load_simulation_config(TOY_CONFIG))
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_pair(pair, first)
    export_pair(pair, second)

    assert _file_bytes(first) == _file_bytes(second)
    with pytest.raises(FileExistsError, match="overwrite"):
        export_pair(pair, first)


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    pair = run_paired(load_simulation_config(TOY_CONFIG))
    target = tmp_path / "tampered"
    export_pair(pair, target)
    manifest_path = target / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["outcomes.parquet"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_export(target)


def test_tracked_analytic_fixture_is_valid() -> None:
    manifest = validate_export(TRACKED_FIXTURE)
    assert manifest["scenario_id"] == "m03-analytic-two-path"
    assert manifest["latency_effect_ms"] == -8.0
