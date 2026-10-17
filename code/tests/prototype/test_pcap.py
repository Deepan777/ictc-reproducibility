import struct
from pathlib import Path

from ictc.prototype.pcap import read_loopback_udp_packets


def _record(timestamp: int, payload: bytes) -> bytes:
    ip = bytes([0x45]) + b"\x00" * 8 + bytes([17]) + b"\x00" * 10
    udp = struct.pack("!HHHH", 1000, 2000, 8 + len(payload), 0) + payload
    frame = b"\x02\x00\x00\x00" + ip + udp
    return struct.pack("<IIII", timestamp, 123, len(frame), len(frame)) + frame


def test_reads_loopback_udp_payloads(tmp_path: Path) -> None:
    capture = tmp_path / "capture.pcap"
    capture.write_bytes(
        struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 0) + _record(10, b"ictc:0")
    )

    packets = read_loopback_udp_packets(capture)

    assert packets[0].timestamp_ns == 10_000_123_000
    assert packets[0].source_port == 1000
    assert packets[0].destination_port == 2000
    assert packets[0].payload == "ictc:0"


def test_ignores_unrelated_non_ascii_loopback_udp_payloads(tmp_path: Path) -> None:
    capture = tmp_path / "capture.pcap"
    capture.write_bytes(
        struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 0)
        + _record(10, b"\xff\x00")
        + _record(11, b"ictc:0")
    )

    packets = read_loopback_udp_packets(capture)

    assert [packet.payload for packet in packets] == ["ictc:0"]
