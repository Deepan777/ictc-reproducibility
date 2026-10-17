"""Strict bridge from M11 ns-3 CSV evidence into ICTC-owned manifests.

The bridge deliberately accepts only raw simulator output and never reads a Python oracle.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

_SUMMARY_COLUMNS = {
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
}
_SUMMARY_COLUMNS_WITH_TOPOLOGY = _SUMMARY_COLUMNS | {"topology"}
_TELEMETRY_COLUMNS = {"timestamp_ns", "path_id", "queue_packets"}


@dataclass(frozen=True, slots=True)
class Ns3FlowSummary:
    """One foreground flow reported by an ns-3 FlowMonitor CSV."""

    action: str
    seed: int
    run: int
    flow_id: int
    source: str
    destination: str
    tx_packets: int
    rx_packets: int
    lost_packets: int
    mean_delay_ns: int
    first_tx_ns: int
    last_rx_ns: int


@dataclass(frozen=True, slots=True)
class Ns3PairedRun:
    """Validated raw output from matched KEEP and REROUTE ns-3 invocations."""

    keep: Ns3FlowSummary
    reroute: Ns3FlowSummary
    keep_telemetry_rows: int
    reroute_telemetry_rows: int
    keep_telemetry_sha256: str
    reroute_telemetry_sha256: str

    @property
    def latency_effect_ms(self) -> float:
        """Return reroute minus keep mean packet delay in milliseconds."""
        return (self.reroute.mean_delay_ns - self.keep.mean_delay_ns) / 1_000_000


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _read_single_summary(path: Path, expected_action: str) -> Ns3FlowSummary:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = set(rows[0]) if rows else set()
    if fields not in (_SUMMARY_COLUMNS, _SUMMARY_COLUMNS_WITH_TOPOLOGY):
        raise ValueError(f"{path} does not have the required ns-3 summary schema")
    matching = [row for row in rows if row["action"] == expected_action]
    if len(matching) != 1:
        raise ValueError(f"{path} must contain exactly one {expected_action} flow")
    row = matching[0]
    try:
        summary = Ns3FlowSummary(
            action=row["action"],
            seed=int(row["seed"]),
            run=int(row["run"]),
            flow_id=int(row["flow_id"]),
            source=row["source"],
            destination=row["destination"],
            tx_packets=int(row["tx_packets"]),
            rx_packets=int(row["rx_packets"]),
            lost_packets=int(row["lost_packets"]),
            mean_delay_ns=int(row["mean_delay_ns"]),
            first_tx_ns=int(row["first_tx_ns"]),
            last_rx_ns=int(row["last_rx_ns"]),
        )
    except (KeyError, ValueError) as error:
        raise ValueError(f"invalid ns-3 summary row in {path}") from error
    if summary.tx_packets < summary.rx_packets or summary.rx_packets < 0:
        raise ValueError(f"invalid packet accounting in {path}")
    if summary.lost_packets < 0 or summary.mean_delay_ns < 0:
        raise ValueError(f"invalid loss or delay in {path}")
    if summary.last_rx_ns < summary.first_tx_ns:
        raise ValueError(f"non-monotonic flow times in {path}")
    return summary


def _read_telemetry_layout(path: Path) -> tuple[tuple[int, str], ...]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != _TELEMETRY_COLUMNS:
        raise ValueError(f"{path} does not have the required ns-3 telemetry schema")
    last_timestamp = -1
    layout: list[tuple[int, str]] = []
    for row in rows:
        try:
            timestamp = int(row["timestamp_ns"])
            queue_packets = int(row["queue_packets"])
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid ns-3 telemetry row in {path}") from error
        if timestamp < last_timestamp or queue_packets < 0 or not row["path_id"]:
            raise ValueError(f"invalid telemetry ordering or value in {path}")
        last_timestamp = timestamp
        layout.append((timestamp, row["path_id"]))
    return tuple(layout)


def read_paired_run(root: Path) -> Ns3PairedRun:
    """Read a paired raw run without accepting missing, stale, or mixed-seed inputs."""
    keep_summary = _read_single_summary(root / "keep" / "summary.csv", "keep")
    reroute_summary = _read_single_summary(root / "reroute" / "summary.csv", "reroute")
    if (keep_summary.seed, keep_summary.run) != (reroute_summary.seed, reroute_summary.run):
        raise ValueError("paired ns-3 outputs use different seed/run values")
    if keep_summary.tx_packets == 0 or reroute_summary.tx_packets == 0:
        raise ValueError("paired ns-3 outputs contain no foreground packets")
    keep_telemetry = root / "keep" / "telemetry.csv"
    reroute_telemetry = root / "reroute" / "telemetry.csv"
    keep_layout = _read_telemetry_layout(keep_telemetry)
    reroute_layout = _read_telemetry_layout(reroute_telemetry)
    if keep_layout != reroute_layout:
        raise ValueError("paired ns-3 telemetry timestamps or paths do not align")
    return Ns3PairedRun(
        keep=keep_summary,
        reroute=reroute_summary,
        keep_telemetry_rows=len(keep_layout),
        reroute_telemetry_rows=len(reroute_layout),
        keep_telemetry_sha256=_sha256(keep_telemetry),
        reroute_telemetry_sha256=_sha256(reroute_telemetry),
    )


def write_pair_manifest(pair: Ns3PairedRun, target: Path) -> dict[str, object]:
    """Atomically write a non-overwritable manifest for one validated ns-3 pair."""
    if target.exists():
        raise FileExistsError(f"refusing to overwrite ns-3 manifest: {target}")
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "generator": "ictc.ns3.bridge.write_pair_manifest",
        "paired_inputs_identical_except_action": True,
        "latency_effect_ms": pair.latency_effect_ms,
        "keep": asdict(pair.keep),
        "reroute": asdict(pair.reroute),
        "keep_telemetry_rows": pair.keep_telemetry_rows,
        "reroute_telemetry_rows": pair.reroute_telemetry_rows,
        "keep_telemetry_sha256": pair.keep_telemetry_sha256,
        "reroute_telemetry_sha256": pair.reroute_telemetry_sha256,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=target.parent, delete=False
    ) as temporary:
        json.dump(payload, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, target)
    except FileExistsError:
        temporary_path.unlink(missing_ok=True)
        raise
    return payload
