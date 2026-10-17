"""Bounded real-UDP prototype with route changes and breach fallback."""

from __future__ import annotations

import json
import multiprocessing
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path

_PROCESS_TIMEOUT_SECONDS = 15


@dataclass(frozen=True, slots=True)
class PrototypeEvent:
    timestamp_ns: int
    component: str
    event: str
    detail: str


def _event(events: list[PrototypeEvent], component: str, event: str, detail: str) -> None:
    events.append(PrototypeEvent(time.time_ns(), component, event, detail))


def _receive_acknowledgement(sender: socket.socket) -> bytes:
    """Read a UDP acknowledgement, ignoring Windows' asynchronous ICMP reset notification."""
    while True:
        try:
            acknowledgement, _ = sender.recvfrom(1024)
            return acknowledgement
        except ConnectionResetError:
            continue


def _foreground_sender_worker(
    port_queue: multiprocessing.Queue[int],
    commands: multiprocessing.Queue[tuple[bytes, tuple[str, int]]],
    acknowledgements: multiprocessing.Queue[bytes],
    count: int,
) -> None:
    """Send foreground packets and validate receiver acknowledgements in a dedicated process."""
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(("127.0.0.1", 0))
    sender.settimeout(_PROCESS_TIMEOUT_SECONDS)
    port_queue.put(sender.getsockname()[1])
    try:
        for _ in range(count):
            payload, destination = commands.get(timeout=_PROCESS_TIMEOUT_SECONDS)
            sender.sendto(payload, destination)
            acknowledgements.put(_receive_acknowledgement(sender))
    finally:
        sender.close()


def _relay_worker(
    port_queue: multiprocessing.Queue[tuple[int, int]],
    index: int,
    destination: tuple[str, int],
    count: int,
) -> None:
    relay = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    relay.bind(("127.0.0.1", 0))
    port_queue.put((index, relay.getsockname()[1]))
    try:
        for _ in range(count):
            payload, _ = relay.recvfrom(1024)
            relay.sendto(payload, destination)
    finally:
        relay.close()


def _receiver_worker(
    port_queue: multiprocessing.Queue[int], acknowledgement_destination: tuple[str, int], count: int
) -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    port_queue.put(receiver.getsockname()[1])
    try:
        for _ in range(count):
            payload, _ = receiver.recvfrom(1024)
            receiver.sendto(b"ACK:" + payload, acknowledgement_destination)
    finally:
        receiver.close()


def _telemetry_worker(destination: tuple[str, int], commands: multiprocessing.Queue[str]) -> None:
    """Send actual UDP telemetry messages from a dedicated telemetry process."""
    telemetry = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        command = commands.get(timeout=10)
        if command != "SEND_READY":
            raise RuntimeError("telemetry worker was not armed")
        telemetry.sendto(b"TELEMETRY_READY", destination)
        command = commands.get(timeout=10)
        if command == "INJECT_BREACH":
            telemetry.sendto(b"TELEMETRY_BREACH", destination)
    finally:
        telemetry.close()


def _background_worker(
    destination: tuple[str, int], event_queue: multiprocessing.Queue[str], count: int
) -> None:
    """Generate deterministic real UDP background traffic in a child process."""
    event_queue.put("BACKGROUND_STARTED")
    background = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        for sequence in range(count):
            background.sendto(f"background:{sequence}".encode("ascii"), destination)
    finally:
        background.close()
        event_queue.put("BACKGROUND_COMPLETED")


def _background_sink_worker(
    port_queue: multiprocessing.Queue[int], result_queue: multiprocessing.Queue[str], count: int
) -> None:
    """Receive background UDP packets without affecting the foreground path."""
    sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sink.bind(("127.0.0.1", 0))
    port_queue.put(sink.getsockname()[1])
    try:
        for _ in range(count):
            payload, _ = sink.recvfrom(1024)
            result_queue.put(payload.decode("ascii"))
    finally:
        sink.close()


def run_loopback_prototype(output: Path, *, breach: bool = True) -> tuple[PrototypeEvent, ...]:
    """Run real loopback UDP traffic through two relay sockets and persist immutable events."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prototype output: {output}")
    output.mkdir(parents=True)
    events: list[PrototypeEvent] = []
    context = multiprocessing.get_context("spawn")
    ports: multiprocessing.Queue[int] = context.Queue()
    relay_ports: multiprocessing.Queue[tuple[int, int]] = context.Queue()
    sender_ports: multiprocessing.Queue[int] = context.Queue()
    sender_commands: multiprocessing.Queue[tuple[bytes, tuple[str, int]]] = context.Queue()
    sender_acknowledgements: multiprocessing.Queue[bytes] = context.Queue()
    worker_events: multiprocessing.Queue[str] = context.Queue()
    background_results: multiprocessing.Queue[str] = context.Queue()
    telemetry_commands: multiprocessing.Queue[str] = context.Queue()
    telemetry_listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    telemetry_listener.bind(("127.0.0.1", 0))
    telemetry_destination = ("127.0.0.1", telemetry_listener.getsockname()[1])
    foreground_sender = context.Process(
        target=_foreground_sender_worker,
        args=(sender_ports, sender_commands, sender_acknowledgements, 6),
        name="ictc-foreground-sender",
    )
    foreground_sender.start()
    acknowledgement_destination = ("127.0.0.1", sender_ports.get(timeout=_PROCESS_TIMEOUT_SECONDS))
    receiver = context.Process(
        target=_receiver_worker,
        args=(ports, acknowledgement_destination, 6),
        name="ictc-receiver",
    )
    receiver.start()
    destination = ("127.0.0.1", ports.get(timeout=_PROCESS_TIMEOUT_SECONDS))
    background_sink = context.Process(
        target=_background_sink_worker,
        args=(ports, background_results, 3),
        name="ictc-background-sink",
    )
    background_sink.start()
    background_destination = ("127.0.0.1", ports.get(timeout=_PROCESS_TIMEOUT_SECONDS))
    telemetry = context.Process(
        target=_telemetry_worker,
        args=(telemetry_destination, telemetry_commands),
        name="ictc-telemetry",
    )
    background = context.Process(
        target=_background_worker,
        args=(background_destination, worker_events, 3),
        name="ictc-background",
    )
    relay_counts = (4, 2)
    relays = [
        context.Process(
            target=_relay_worker,
            args=(relay_ports, index, destination, relay_counts[index]),
            name=f"ictc-relay-{index}",
        )
        for index in range(2)
    ]
    for relay in relays:
        relay.start()
    telemetry.start()
    background.start()
    relay_addresses_by_index = {
        index: ("127.0.0.1", port)
        for index, port in (relay_ports.get(timeout=_PROCESS_TIMEOUT_SECONDS) for _ in relays)
    }
    relay_addresses = [relay_addresses_by_index[index] for index in range(len(relays))]
    started_detail = ";".join(
        [
            f"receiver={receiver.pid}",
            f"foreground_sender={foreground_sender.pid}",
            f"relays={[relay.pid for relay in relays]}",
            f"telemetry={telemetry.pid}",
            f"background={background.pid}",
            f"background_sink={background_sink.pid}",
        ]
    )
    _event(events, "coordinator", "PROCESSES_STARTED", started_detail)
    telemetry_listener.settimeout(_PROCESS_TIMEOUT_SECONDS)
    telemetry_commands.put("SEND_READY")
    _event(
        events,
        "telemetry",
        telemetry_listener.recvfrom(1024)[0].decode("ascii"),
        "udp-protocol-message",
    )
    try:
        active = 0
        for sequence in range(6):
            if sequence == 2:
                active = 1
                _event(events, "coordinator", "ROUTE_CHANGE", "relay-1")
            if breach and sequence == 4:
                telemetry_commands.put("INJECT_BREACH")
                signal, _ = telemetry_listener.recvfrom(1024)
                if signal != b"TELEMETRY_BREACH":
                    raise RuntimeError("unexpected telemetry signal")
                _event(events, "telemetry", "TELEMETRY_BREACH", "udp-protocol-message")
                active = 0
                _event(events, "coordinator", "BREACH_FALLBACK", "telemetry-breach->relay-0")
            payload = f"ictc:{sequence}".encode()
            sender_commands.put((payload, relay_addresses[active]))
            acknowledgement = sender_acknowledgements.get(timeout=_PROCESS_TIMEOUT_SECONDS)
            if acknowledgement != b"ACK:" + payload:
                raise RuntimeError("unexpected receiver acknowledgement")
            _event(events, "foreground", "PACKET_DELIVERED", f"sequence={sequence};relay={active}")
        for _ in range(3):
            background_payload = background_results.get(timeout=_PROCESS_TIMEOUT_SECONDS)
            _event(events, "background", "PACKET_DELIVERED", background_payload)
        for _ in range(2):
            _event(
                events,
                "background",
                worker_events.get(timeout=_PROCESS_TIMEOUT_SECONDS),
                "separate-process",
            )
        (output / "events.jsonl").write_text(
            "".join(json.dumps(asdict(event), sort_keys=True) + "\n" for event in events),
            encoding="utf-8",
        )
        return tuple(events)
    finally:
        telemetry_listener.close()
        foreground_sender.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        if foreground_sender.is_alive():
            foreground_sender.terminate()
        receiver.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        if receiver.is_alive():
            receiver.terminate()
        if telemetry.is_alive():
            telemetry_commands.put("STOP")
        telemetry.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        if telemetry.is_alive():
            telemetry.terminate()
        background.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        if background.is_alive():
            background.terminate()
        background_sink.join(timeout=_PROCESS_TIMEOUT_SECONDS)
        if background_sink.is_alive():
            background_sink.terminate()
        for relay in relays:
            relay.join(timeout=_PROCESS_TIMEOUT_SECONDS)
            if relay.is_alive():
                relay.terminate()
