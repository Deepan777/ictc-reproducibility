"""Validation and low-level queue tests for the M03 environment."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from ictc.simulation import (
    ArrivalMode,
    ArrivalProcess,
    PacketAccounting,
    PacketQueue,
    PathConfig,
    PathTrafficConfig,
    SimulationConfig,
    TelemetryConfig,
)


def _valid_payload() -> dict[str, object]:
    return {
        "scenario_id": "validation",
        "seed": 1,
        "duration_ticks": 2,
        "tick_duration_ms": 1,
        "deadline_ms": 10,
        "current_path_id": "a",
        "candidate_path_id": "b",
        "paths": [
            {
                "path_id": "a",
                "service_packets_per_tick": 1,
                "queue_limit_packets": 2,
                "propagation_delay_ms": 1,
            },
            {
                "path_id": "b",
                "service_packets_per_tick": 1,
                "queue_limit_packets": 2,
                "propagation_delay_ms": 1,
            },
        ],
        "foreground_process": {"mode": "CONSTANT", "rate_packets_per_tick": 1},
        "cross_traffic": [
            {"path_id": "a", "process": {"mode": "CONSTANT", "rate_packets_per_tick": 0}},
            {"path_id": "b", "process": {"mode": "CONSTANT", "rate_packets_per_tick": 0}},
        ],
        "telemetry": {},
    }


def test_configuration_rejects_invalid_arrivals_and_topology() -> None:
    with pytest.raises(ValidationError, match="integer packet count"):
        ArrivalProcess(mode=ArrivalMode.CONSTANT, rate_packets_per_tick=0.5)
    with pytest.raises(ValidationError, match="maximum telemetry delay"):
        TelemetryConfig(minimum_delay_ticks=2, maximum_delay_ticks=1)

    payload = _valid_payload()
    payload["candidate_path_id"] = "a"
    with pytest.raises(ValidationError, match="must differ"):
        SimulationConfig.model_validate(payload)

    payload = _valid_payload()
    payload["cross_traffic"] = [
        {"path_id": "a", "process": {"mode": "CONSTANT", "rate_packets_per_tick": 0}}
    ]
    with pytest.raises(ValidationError, match="every path"):
        SimulationConfig.model_validate(payload)


def test_path_and_traffic_lookup_reject_unknown_ids() -> None:
    config = SimulationConfig.model_validate(_valid_payload())
    with pytest.raises(KeyError):
        config.path("missing")
    with pytest.raises(KeyError):
        config.cross_process("missing")
    assert len(config.config_hash) == 64
    assert config.canonical_json() == config.canonical_json()


def test_packet_queue_fifo_service_and_negative_arrival_validation() -> None:
    queue = PacketQueue("p", service_capacity=2, queue_limit=5)
    with pytest.raises(ValueError, match="negative"):
        queue.enqueue("foreground", -1, 0)
    assert queue.enqueue("cross", 2, 0) == 0
    assert queue.enqueue("foreground", 4, 0) == 1
    first = queue.service(0)
    second = queue.service(1)
    third = queue.service(2)

    assert (first.cross, first.foreground) == (2, 0)
    assert (second.cross, second.foreground) == (0, 2)
    assert (third.cross, third.foreground) == (0, 1)
    assert queue.foreground_latency_ticks == (2, 2, 3)
    accounting = queue.accounting()
    assert accounting.foreground_generated == 4
    assert accounting.foreground_dropped == 1


@pytest.mark.parametrize(
    "accounting",
    [
        PacketAccounting("p", 2, 1, 0, 0, 0, 0, 0),
        PacketAccounting("p", 0, 0, 0, 2, 1, 0, 0),
        PacketAccounting("p", 0, 0, 0, 0, 0, 0, 1),
    ],
)
def test_accounting_detects_conservation_failures(accounting: PacketAccounting) -> None:
    with pytest.raises(ValueError):
        accounting.validate()


def test_configuration_types_are_immutable() -> None:
    path = PathConfig(
        path_id="p", service_packets_per_tick=1, queue_limit_packets=1, propagation_delay_ms=0
    )
    traffic = PathTrafficConfig(
        path_id="p", process=ArrivalProcess(mode=ArrivalMode.POISSON, rate_packets_per_tick=1)
    )
    assert path.path_id == traffic.path_id
    with pytest.raises(ValidationError, match="frozen"):
        path.path_id = "changed"  # type: ignore[misc]
