"""Validated configuration for the deterministic M03 network environment."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImmutableConfig(BaseModel):
    """Strict immutable configuration base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ArrivalMode(StrEnum):
    """Supported packet-count arrival generators."""

    CONSTANT = "CONSTANT"
    POISSON = "POISSON"


class ArrivalProcess(ImmutableConfig):
    """Packet arrivals per discrete simulation tick."""

    mode: ArrivalMode
    rate_packets_per_tick: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_constant_rate(self) -> Self:
        if self.mode is ArrivalMode.CONSTANT and not self.rate_packets_per_tick.is_integer():
            raise ValueError("constant arrival rate must be an integer packet count")
        return self


class PathConfig(ImmutableConfig):
    """One path represented by a finite FIFO bottleneck queue."""

    path_id: str = Field(min_length=1)
    service_packets_per_tick: int = Field(gt=0, le=10000)
    queue_limit_packets: int = Field(ge=0, le=1_000_000)
    propagation_delay_ms: float = Field(ge=0)


class PathTrafficConfig(ImmutableConfig):
    """Exogenous cross traffic associated with one candidate path."""

    path_id: str = Field(min_length=1)
    process: ArrivalProcess


class TelemetryConfig(ImmutableConfig):
    """Action-independent telemetry impairment regime."""

    sample_every_ticks: int = Field(default=1, gt=0)
    minimum_delay_ticks: int = Field(default=0, ge=0)
    maximum_delay_ticks: int = Field(default=0, ge=0)
    missing_probability: float = Field(default=0.0, ge=0, le=1)
    source_failure_probability: float = Field(default=0.0, ge=0, le=1)
    malicious_probability: float = Field(default=0.0, ge=0, le=1)
    malicious_queue_offset_packets: int = Field(default=0, ge=0, le=1_000_000)
    maximum_clock_skew_ticks: int = Field(default=0, ge=0)
    stale_after_ticks: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_delay_bounds(self) -> Self:
        if self.maximum_delay_ticks < self.minimum_delay_ticks:
            raise ValueError("maximum telemetry delay cannot be below minimum delay")
        if self.malicious_probability > 0 and self.malicious_queue_offset_packets == 0:
            raise ValueError("malicious telemetry requires a non-zero queue offset")
        return self


class SimulationConfig(ImmutableConfig):
    """Complete reproducible configuration for one paired simulation."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    scenario_id: str = Field(min_length=1)
    seed: int = Field(ge=0)
    duration_ticks: int = Field(gt=0, le=100_000)
    tick_duration_ms: float = Field(gt=0)
    deadline_ms: float = Field(gt=0)
    current_path_id: str = Field(min_length=1)
    candidate_path_id: str = Field(min_length=1)
    paths: tuple[PathConfig, ...]
    foreground_process: ArrivalProcess
    cross_traffic: tuple[PathTrafficConfig, ...]
    telemetry: TelemetryConfig
    capacity_change_tick: int | None = Field(default=None, ge=0)
    current_service_after_change: int | None = Field(default=None, gt=0, le=10_000)
    candidate_service_after_change: int | None = Field(default=None, gt=0, le=10_000)
    cross_traffic_shift_tick: int | None = Field(default=None, ge=0)
    cross_traffic_shift_multiplier: float = Field(default=1.0, ge=0, le=100)

    @model_validator(mode="after")
    def validate_topology_and_traffic(self) -> Self:
        if len(self.paths) != 2:
            raise ValueError("M03 requires exactly two candidate paths")
        path_ids = [path.path_id for path in self.paths]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("path IDs must be unique")
        if self.current_path_id == self.candidate_path_id:
            raise ValueError("current and candidate path IDs must differ")
        if {self.current_path_id, self.candidate_path_id} != set(path_ids):
            raise ValueError("current and candidate path IDs must name the two configured paths")
        traffic_ids = [traffic.path_id for traffic in self.cross_traffic]
        if len(traffic_ids) != len(set(traffic_ids)):
            raise ValueError("cross-traffic path IDs must be unique")
        if set(traffic_ids) != set(path_ids):
            raise ValueError("cross traffic must be configured for every path exactly once")
        capacity_values = (self.current_service_after_change, self.candidate_service_after_change)
        if (self.capacity_change_tick is None) != (capacity_values == (None, None)):
            raise ValueError("capacity change requires both post-change service capacities")
        if (
            self.capacity_change_tick is not None
            and self.capacity_change_tick >= self.duration_ticks
        ):
            raise ValueError("capacity change tick must be inside the simulation")
        if (
            self.cross_traffic_shift_tick is not None
            and self.cross_traffic_shift_tick >= self.duration_ticks
        ):
            raise ValueError("cross-traffic shift tick must be inside the simulation")
        return self

    def path(self, path_id: str) -> PathConfig:
        """Return one configured path by ID."""
        for path in self.paths:
            if path.path_id == path_id:
                return path
        raise KeyError(path_id)

    def cross_process(self, path_id: str) -> ArrivalProcess:
        """Return the cross-traffic process for one path."""
        for traffic in self.cross_traffic:
            if traffic.path_id == path_id:
                return traffic.process
        raise KeyError(path_id)

    def canonical_json(self) -> str:
        """Serialize configuration with stable ordering for hashing/manifests."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 of the canonical configuration."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest().upper()


def load_simulation_config(path: Path) -> SimulationConfig:
    """Load and validate a JSON simulation configuration."""
    return SimulationConfig.model_validate_json(path.read_text(encoding="utf-8"))
