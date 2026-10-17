"""Tests for the M11 raw ns-3 bridge."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ictc.ns3 import read_paired_run, write_pair_manifest
from ictc.ns3.runner import load_matrix

SUMMARY_FIELDS = [
    "action",
    "seed",
    "run",
    "flow_id",
    "source",
    "destination",
    "tx_packets",
    "rx_packets",
    "lost_packets",
    "mean_delay_ns",
    "first_tx_ns",
    "last_rx_ns",
]


def _write_pair(root: Path, *, reroute_run: int = 1) -> None:
    for action, delay, run in (("keep", 17_000_000, 1), ("reroute", 5_000_000, reroute_run)):
        directory = root / action
        directory.mkdir(parents=True)
        with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerow(
                {
                    "action": action,
                    "seed": 713,
                    "run": run,
                    "flow_id": 1,
                    "source": "10.1.1.1",
                    "destination": "10.1.2.2",
                    "tx_packets": 400,
                    "rx_packets": 400,
                    "lost_packets": 0,
                    "mean_delay_ns": delay,
                    "first_tx_ns": 1_000_000_000,
                    "last_rx_ns": 5_000_000_000,
                }
            )
        (directory / "telemetry.csv").write_text(
            "timestamp_ns,path_id,queue_packets\n0,current,0\n0,candidate,0\n100,current,1\n",
            encoding="utf-8",
        )


def test_validated_pair_has_effect_and_immutable_manifest(tmp_path: Path) -> None:
    _write_pair(tmp_path)
    pair = read_paired_run(tmp_path)
    assert pair.latency_effect_ms == -12.0
    manifest = tmp_path / "manifest.json"
    payload = write_pair_manifest(pair, manifest)
    assert payload["paired_inputs_identical_except_action"] is True
    with pytest.raises(FileExistsError):
        write_pair_manifest(pair, manifest)


def test_mixed_seed_or_run_is_rejected(tmp_path: Path) -> None:
    _write_pair(tmp_path, reroute_run=2)
    with pytest.raises(ValueError, match="different seed/run"):
        read_paired_run(tmp_path)


def test_matrix_rejects_reused_action_seed_run(tmp_path: Path) -> None:
    matrix = {
        "schema_version": "1.0.0",
        "runs": [
            {
                "run_id": "one",
                "seed": 1,
                "run": 1,
                "actions": ["keep"],
                "duration": 2,
                "interval_ms": 1,
                "max_packets": 1,
                "packet_size": 1,
            },
            {
                "run_id": "two",
                "seed": 1,
                "run": 1,
                "actions": ["keep"],
                "duration": 2,
                "interval_ms": 1,
                "max_packets": 1,
                "packet_size": 1,
            },
        ],
    }
    path = tmp_path / "matrix.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")
    with pytest.raises(ValueError, match="collision"):
        load_matrix(path)
