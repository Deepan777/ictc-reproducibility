"""Synchronous Npcap loopback capture for the adapted Windows M12 prototype."""

from __future__ import annotations

import ctypes
import json
import subprocess
import sys
import time
from ctypes import (
    POINTER,
    Structure,
    byref,
    c_char_p,
    c_int,
    c_long,
    c_ubyte,
    c_uint,
    c_void_p,
    create_string_buffer,
)
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ictc.prototype.windows import PrototypeEvent


class _PcapIf(Structure):
    pass


_PcapIf._fields_ = [
    ("next", POINTER(_PcapIf)),
    ("name", c_char_p),
    ("description", c_char_p),
    ("addresses", c_void_p),
    ("flags", c_uint),
]


class _Timeval(Structure):
    _fields_ = [("tv_sec", c_long), ("tv_usec", c_long)]


class _PcapHeader(Structure):
    _fields_ = [("ts", _Timeval), ("caplen", c_uint), ("len", c_uint)]


@dataclass(frozen=True, slots=True)
class NpcapCaptureRecord:
    device: str
    packet_count: int
    started_ns: int
    stopped_ns: int


def _load_pcap() -> ctypes.WinDLL:
    library = ctypes.WinDLL(r"C:\Windows\System32\Npcap\wpcap.dll")
    library.pcap_findalldevs.argtypes = [POINTER(POINTER(_PcapIf)), c_char_p]
    library.pcap_findalldevs.restype = c_int
    library.pcap_freealldevs.argtypes = [POINTER(_PcapIf)]
    library.pcap_open_live.argtypes = [c_char_p, c_int, c_int, c_int, c_char_p]
    library.pcap_open_live.restype = c_void_p
    library.pcap_setnonblock.argtypes = [c_void_p, c_int, c_char_p]
    library.pcap_setnonblock.restype = c_int
    library.pcap_next_ex.argtypes = [
        c_void_p,
        POINTER(POINTER(_PcapHeader)),
        POINTER(POINTER(c_ubyte)),
    ]
    library.pcap_next_ex.restype = c_int
    library.pcap_geterr.argtypes = [c_void_p]
    library.pcap_geterr.restype = c_char_p
    library.pcap_dump_open.argtypes = [c_void_p, c_char_p]
    library.pcap_dump_open.restype = c_void_p
    library.pcap_dump.argtypes = [c_void_p, POINTER(_PcapHeader), POINTER(c_ubyte)]
    library.pcap_dump_flush.argtypes = [c_void_p]
    library.pcap_dump_flush.restype = c_int
    library.pcap_dump_close.argtypes = [c_void_p]
    library.pcap_close.argtypes = [c_void_p]
    return library


def _loopback_device(library: ctypes.WinDLL) -> bytes:
    devices = POINTER(_PcapIf)()
    error = create_string_buffer(256)
    if library.pcap_findalldevs(byref(devices), error) != 0:
        raise RuntimeError(
            f"Npcap device enumeration failed: {error.value.decode(errors='replace')}"
        )
    try:
        current = devices
        while current:
            name = current.contents.name
            if name == b"\\Device\\NPF_Loopback":
                return bytes(name)
            current = current.contents.next
    finally:
        if devices:
            library.pcap_freealldevs(devices)
    raise RuntimeError("Npcap loopback capture device was not found")


def run_captured_loopback_prototype(
    output: Path, *, breach: bool = True
) -> tuple[PrototypeEvent, ...]:
    """Capture a complete prototype run on Npcap's loopback device into standard PCAP."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prototype output: {output}")
    temporary_capture = output.parent / f"{output.name}.pcap"
    if temporary_capture.exists():
        raise FileExistsError(f"refusing to overwrite capture output: {temporary_capture}")
    temporary_capture.parent.mkdir(parents=True, exist_ok=True)

    library = _load_pcap()
    device = _loopback_device(library)
    error = create_string_buffer(256)
    handle = library.pcap_open_live(device, 65_535, 1, 20, error)
    if not handle:
        raise RuntimeError(f"Npcap open failed: {error.value.decode(errors='replace')}")
    if library.pcap_setnonblock(handle, 1, error) != 0:
        library.pcap_close(handle)
        raise RuntimeError(f"Npcap nonblocking mode failed: {error.value.decode(errors='replace')}")
    dumper = library.pcap_dump_open(handle, str(temporary_capture).encode("utf-8"))
    if not dumper:
        message = library.pcap_geterr(handle).decode(errors="replace")
        library.pcap_close(handle)
        raise RuntimeError(f"Npcap dump open failed: {message}")

    command = (
        "from pathlib import Path; "
        "from ictc.prototype.windows import run_loopback_prototype; "
        f"print(len(run_loopback_prototype(Path(r'{output}'), breach={breach})))"
    )
    worker = subprocess.Popen(
        [sys.executable, "-c", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    packet_count = 0
    started_ns = time.time_ns()
    try:
        while worker.poll() is None:
            header = POINTER(_PcapHeader)()
            packet = POINTER(c_ubyte)()
            result = library.pcap_next_ex(handle, byref(header), byref(packet))
            if result == 1:
                library.pcap_dump(dumper, header, packet)
                packet_count += 1
            elif result < 0:
                raise RuntimeError(
                    f"Npcap read failed: {library.pcap_geterr(handle).decode(errors='replace')}"
                )
            else:
                time.sleep(0.001)
        standard_output, standard_error = worker.communicate(timeout=5)
        if worker.returncode != 0:
            raise RuntimeError(f"prototype subprocess failed: {standard_error.strip()}")
        if standard_output.strip() != "16":
            raise RuntimeError(f"unexpected prototype output: {standard_output.strip()}")
        from ictc.prototype.windows import PrototypeEvent

        events = tuple(
            PrototypeEvent(**json.loads(line))
            for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
        )
    finally:
        stopped_ns = time.time_ns()
        library.pcap_dump_flush(dumper)
        library.pcap_dump_close(dumper)
        library.pcap_close(handle)
        if worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=5)

    temporary_capture.replace(output / "traffic.pcap")
    capture = NpcapCaptureRecord(
        device=device.decode("ascii"),
        packet_count=packet_count,
        started_ns=started_ns,
        stopped_ns=stopped_ns,
    )
    (output / "capture.json").write_text(
        json.dumps(asdict(capture), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return events
