"""Minimal standard-PCAP parsing and M12 event-alignment checks."""

from __future__ import annotations

import json
import struct
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class UdpPacket:
    timestamp_ns: int
    source_port: int
    destination_port: int
    payload: str


@dataclass(frozen=True, slots=True)
class CaptureAlignment:
    packet_count: int
    payload_counts: dict[str, int]
    maximum_anchor_delta_ns: int
    tolerance_ns: int


def read_loopback_udp_packets(path: Path) -> tuple[UdpPacket, ...]:
    """Read IPv4 UDP packets from the little-endian DLT_NULL PCAP emitted by Npcap."""
    data = path.read_bytes()
    if len(data) < 24:
        raise ValueError("PCAP is shorter than its global header")
    magic, major, minor, _, _, _, linktype = struct.unpack_from("<IHHIIII", data)
    if (magic, major, minor, linktype) != (0xA1B2C3D4, 2, 4, 0):
        raise ValueError("expected a little-endian PCAP 2.4 with DLT_NULL loopback frames")
    packets: list[UdpPacket] = []
    offset = 24
    while offset < len(data):
        if offset + 16 > len(data):
            raise ValueError("truncated PCAP packet header")
        seconds, microseconds, captured, _ = struct.unpack_from("<IIII", data, offset)
        frame_start = offset + 16
        frame_end = frame_start + captured
        if frame_end > len(data):
            raise ValueError("truncated PCAP packet frame")
        frame = data[frame_start:frame_end]
        offset = frame_end
        if len(frame) < 32 or frame[:4] != b"\x02\x00\x00\x00":
            continue
        ip = frame[4:]
        header_length = (ip[0] & 0x0F) * 4
        if header_length < 20 or len(ip) < header_length + 8 or ip[9] != 17:
            continue
        source_port, destination_port, length, _ = struct.unpack_from("!HHHH", ip, header_length)
        if length < 8 or len(ip) < header_length + length:
            raise ValueError("invalid UDP length in PCAP frame")
        payload = ip[header_length + 8 : header_length + length]
        try:
            payload_text = payload.decode("ascii")
        except UnicodeDecodeError:
            # Npcap's loopback device observes unrelated local UDP as well as the prototype.
            # It is not an M12 packet and must not make a valid capture unparsable.
            continue
        packets.append(
            UdpPacket(
                timestamp_ns=seconds * 1_000_000_000 + microseconds * 1_000,
                source_port=source_port,
                destination_port=destination_port,
                payload=payload_text,
            )
        )
    return tuple(packets)


def audit_m12_alignment(
    events_path: Path, capture_path: Path, *, tolerance_ns: int = 10_000_000
) -> CaptureAlignment:
    """Verify expected M12 payloads, ordering, and event-to-capture timestamp anchors."""
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
    packets = read_loopback_udp_packets(capture_path)
    expected = {
        "TELEMETRY_READY": 1,
        "TELEMETRY_BREACH": 1,
        "background:0": 1,
        "background:1": 1,
        "background:2": 1,
        **{f"ictc:{sequence}": 2 for sequence in range(6)},
        **{f"ACK:ictc:{sequence}": 1 for sequence in range(6)},
    }
    payload_times: dict[str, list[int]] = {}
    for packet in packets:
        if packet.payload in expected:
            payload_times.setdefault(packet.payload, []).append(packet.timestamp_ns)
    observed = Counter(packet.payload for packet in packets if packet.payload in expected)
    if dict(observed) != expected:
        raise ValueError(f"unexpected M12 capture payloads: {dict(observed)}")
    if payload_times["TELEMETRY_BREACH"][0] >= payload_times["ictc:4"][0]:
        raise ValueError("telemetry breach was not captured before fallback packet 4")
    if payload_times["ictc:1"][-1] >= payload_times["ictc:2"][0]:
        raise ValueError("route-change packet order is invalid")

    anchor_events = {
        "TELEMETRY_READY": next(event for event in events if event["event"] == "TELEMETRY_READY"),
        "TELEMETRY_BREACH": next(event for event in events if event["event"] == "TELEMETRY_BREACH"),
    }
    maximum_delta = 0
    for payload, event in anchor_events.items():
        maximum_delta = max(maximum_delta, abs(event["timestamp_ns"] - payload_times[payload][0]))
    deliveries = [
        event
        for event in events
        if event["event"] == "PACKET_DELIVERED" and event["component"] == "foreground"
    ]
    if len(deliveries) != 6:
        raise ValueError("expected six foreground delivery events")
    for sequence, event in enumerate(deliveries):
        maximum_delta = max(
            maximum_delta,
            abs(event["timestamp_ns"] - payload_times[f"ictc:{sequence}"][-1]),
        )
    if maximum_delta > tolerance_ns:
        raise ValueError(f"event/capture timestamp delta {maximum_delta} exceeds {tolerance_ns}")
    return CaptureAlignment(
        packet_count=len(packets),
        payload_counts=dict(sorted(observed.items())),
        maximum_anchor_delta_ns=maximum_delta,
        tolerance_ns=tolerance_ns,
    )


def write_alignment_immutable(
    events_path: Path, capture_path: Path, output: Path
) -> CaptureAlignment:
    """Persist an audit result without overwriting existing evidence."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite alignment output: {output}")
    result = audit_m12_alignment(events_path, capture_path)
    output.write_text(json.dumps(asdict(result), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
