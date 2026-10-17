"""Frozen, resumable runner for the M11 native ns-3 matrix."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ictc.ns3.bridge import read_paired_run, write_pair_manifest


@dataclass(frozen=True, slots=True)
class NativeRun:
    run_id: str
    seed: int
    run: int
    actions: tuple[str, ...]
    duration: float
    interval_ms: float
    max_packets: int
    packet_size: int
    topology_id: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_matrix(path: Path) -> tuple[NativeRun, ...]:
    """Load a strict matrix and reject accidental seed/run collisions."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "1.0.0":
        raise ValueError("unexpected M11 matrix schema")
    values = raw.get("runs")
    if not isinstance(values, list) or not values:
        raise ValueError("M11 matrix has no runs")
    parsed: list[NativeRun] = []
    identities: set[tuple[str, int, int]] = set()
    run_ids: set[str] = set()
    for value in values:
        if not isinstance(value, dict):
            raise ValueError("M11 matrix run must be an object")
        actions = value.get("actions")
        if (
            not isinstance(actions, list)
            or not actions
            or any(item not in {"keep", "reroute", "alternate"} for item in actions)
        ):
            raise ValueError("M11 matrix actions are invalid")
        try:
            item = NativeRun(
                run_id=cast(str, value["run_id"]),
                seed=cast(int, value["seed"]),
                run=cast(int, value["run"]),
                actions=tuple(cast(str, action) for action in actions),
                duration=float(value["duration"]),
                interval_ms=float(value["interval_ms"]),
                max_packets=cast(int, value["max_packets"]),
                packet_size=cast(int, value["packet_size"]),
                topology_id=cast(
                    str,
                    value.get(
                        "topology_id",
                        "three-path" if "three-path" in cast(str, value["run_id"]) else "two-path",
                    ),
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid M11 matrix run") from error
        if not item.run_id or item.run_id in run_ids or item.seed < 0 or item.run < 1:
            raise ValueError("M11 run identity is invalid")
        if (
            item.duration <= 1
            or item.interval_ms <= 0
            or item.max_packets < 1
            or item.packet_size < 1
            or item.topology_id not in {"two-path", "three-path"}
        ):
            raise ValueError("M11 traffic values are invalid")
        for action in item.actions:
            identity = (action, item.seed, item.run)
            if identity in identities:
                raise ValueError(f"M11 action seed/run collision: {identity}")
            identities.add(identity)
        run_ids.add(item.run_id)
        parsed.append(item)
    return tuple(parsed)


def run_matrix(
    matrix_path: Path, executable: Path, output_root: Path, runtime_bin: Path
) -> list[dict[str, object]]:
    """Execute or validate immutable M11 output directories without overwriting raw evidence."""
    if not executable.is_file() or not runtime_bin.is_dir():
        raise FileNotFoundError("ns-3 executable or runtime directory is missing")
    matrix = load_matrix(matrix_path)
    matrix_hash = _sha256(matrix_path)
    results: list[dict[str, object]] = []
    environment = os.environ.copy()
    environment["PATH"] = f"{runtime_bin}{os.pathsep}{environment['PATH']}"
    output_root.mkdir(parents=True, exist_ok=True)
    for item in matrix:
        target = output_root / item.run_id
        manifest = target / "runner_manifest.json"
        if target.exists():
            if not manifest.is_file():
                raise FileExistsError(f"refusing to resume unmanifested raw output: {target}")
            recorded = json.loads(manifest.read_text(encoding="utf-8"))
            if recorded.get("matrix_sha256") != matrix_hash:
                raise ValueError(f"resume matrix mismatch: {target}")
            results.append(cast(dict[str, object], recorded))
            continue
        with tempfile.TemporaryDirectory(prefix=f".{item.run_id}.", dir=output_root) as temporary:
            staged = Path(temporary)
            for action in item.actions:
                command = [
                    str(executable),
                    f"--topology={item.topology_id}",
                    f"--action={action}",
                    f"--seed={item.seed}",
                    f"--run={item.run}",
                    f"--duration={item.duration}",
                    f"--intervalMs={item.interval_ms}",
                    f"--maxPackets={item.max_packets}",
                    f"--packetSize={item.packet_size}",
                    f"--output={staged / action}",
                ]
                subprocess.run(command, check=True, env=environment, capture_output=True, text=True)
            files = {
                str(path.relative_to(staged)): _sha256(path)
                for path in sorted(staged.rglob("*.csv"))
            }
            evidence: dict[str, object] = {
                "schema_version": "1.0.0",
                "run_id": item.run_id,
                "matrix_sha256": matrix_hash,
                "seed": item.seed,
                "run": item.run,
                "actions": list(item.actions),
                "files": files,
            }
            if {"keep", "reroute"}.issubset(item.actions):
                pair = read_paired_run(staged)
                write_pair_manifest(pair, staged / "pair_manifest.json")
                evidence["latency_effect_ms"] = pair.latency_effect_ms
            (staged / "runner_manifest.json").write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            shutil.move(str(staged), str(target))
        results.append(evidence)
    return results
