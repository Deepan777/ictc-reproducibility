"""Action-independent exogenous schedules for paired M03 runs."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass

from ictc.simulation.models import ArrivalMode, ArrivalProcess, SimulationConfig


def _derived_rng(seed: int, stream_name: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{stream_name}".encode()).digest()
    return random.Random(int.from_bytes(digest[:16], byteorder="big", signed=False))


def _poisson(rng: random.Random, rate: float) -> int:
    if rate == 0:
        return 0
    threshold = math.exp(-rate)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _arrival_count(process: ArrivalProcess, rng: random.Random) -> int:
    if process.mode is ArrivalMode.CONSTANT:
        return int(process.rate_packets_per_tick)
    return _poisson(rng, process.rate_packets_per_tick)


@dataclass(frozen=True, slots=True)
class TelemetryImpairment:
    """Missingness, delay, and signed skew for one path/tick sample."""

    missing: bool
    delay_ticks: int
    clock_skew_ticks: int
    source_failed: bool
    malicious: bool


@dataclass(frozen=True, slots=True)
class ScheduleTick:
    """Every exogenous input for one simulation tick."""

    foreground_arrivals: int
    cross_arrivals: tuple[int, ...]
    telemetry_impairments: tuple[TelemetryImpairment, ...]
    service_capacities: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ExogenousSchedule:
    """Pre-generated schedule shared without mutation by paired actions."""

    scenario_id: str
    config_hash: str
    seed: int
    path_ids: tuple[str, ...]
    ticks: tuple[ScheduleTick, ...]

    def canonical_json(self) -> str:
        """Serialize all exogenous inputs with stable ordering."""
        payload = {
            "scenario_id": self.scenario_id,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "path_ids": list(self.path_ids),
            "ticks": [
                {
                    "foreground_arrivals": tick.foreground_arrivals,
                    "cross_arrivals": list(tick.cross_arrivals),
                    "telemetry_impairments": [
                        {
                            "missing": impairment.missing,
                            "delay_ticks": impairment.delay_ticks,
                            "clock_skew_ticks": impairment.clock_skew_ticks,
                            "source_failed": impairment.source_failed,
                            "malicious": impairment.malicious,
                        }
                        for impairment in tick.telemetry_impairments
                    ],
                    "service_capacities": list(tick.service_capacities),
                }
                for tick in self.ticks
            ],
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    @property
    def digest(self) -> str:
        """Return the SHA-256 identity of the complete schedule."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest().upper()


def generate_schedule(config: SimulationConfig) -> ExogenousSchedule:
    """Generate every random draw before any action is selected."""
    path_ids = tuple(path.path_id for path in config.paths)
    foreground_rng = _derived_rng(config.seed, "foreground")
    cross_rngs = {path_id: _derived_rng(config.seed, f"cross:{path_id}") for path_id in path_ids}
    telemetry_rngs = {
        path_id: _derived_rng(config.seed, f"telemetry:{path_id}") for path_id in path_ids
    }

    ticks: list[ScheduleTick] = []
    for _ in range(config.duration_ticks):
        multiplier = (
            config.cross_traffic_shift_multiplier
            if config.cross_traffic_shift_tick is not None
            and len(ticks) >= config.cross_traffic_shift_tick
            else 1.0
        )
        cross_arrivals = tuple(
            int(
                round(
                    _arrival_count(config.cross_process(path_id), cross_rngs[path_id]) * multiplier
                )
            )
            for path_id in path_ids
        )
        impairments: list[TelemetryImpairment] = []
        for path_id in path_ids:
            rng = telemetry_rngs[path_id]
            impairments.append(
                TelemetryImpairment(
                    source_failed=rng.random() < config.telemetry.source_failure_probability,
                    malicious=rng.random() < config.telemetry.malicious_probability,
                    missing=False,
                    delay_ticks=rng.randint(
                        config.telemetry.minimum_delay_ticks,
                        config.telemetry.maximum_delay_ticks,
                    ),
                    clock_skew_ticks=rng.randint(
                        -config.telemetry.maximum_clock_skew_ticks,
                        config.telemetry.maximum_clock_skew_ticks,
                    ),
                )
            )
            impairment = impairments[-1]
            impairments[-1] = TelemetryImpairment(
                missing=impairment.source_failed
                or rng.random() < config.telemetry.missing_probability,
                delay_ticks=impairment.delay_ticks,
                clock_skew_ticks=impairment.clock_skew_ticks,
                source_failed=impairment.source_failed,
                malicious=impairment.malicious,
            )
        ticks.append(
            ScheduleTick(
                foreground_arrivals=_arrival_count(config.foreground_process, foreground_rng),
                cross_arrivals=cross_arrivals,
                telemetry_impairments=tuple(impairments),
                service_capacities=tuple(
                    int(
                        (
                            config.current_service_after_change
                            if path_id == config.current_path_id
                            else config.candidate_service_after_change
                        )
                        or 0
                    )
                    if config.capacity_change_tick is not None
                    and len(ticks) >= config.capacity_change_tick
                    else config.path(path_id).service_packets_per_tick
                    for path_id in path_ids
                ),
            )
        )
    return ExogenousSchedule(
        scenario_id=config.scenario_id,
        config_hash=config.config_hash,
        seed=config.seed,
        path_ids=path_ids,
        ticks=tuple(ticks),
    )
