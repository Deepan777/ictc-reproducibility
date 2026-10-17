"""Validated configuration for the M08 synthetic corpus matrix."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ictc.simulation import (
    ArrivalMode,
    ArrivalProcess,
    PathConfig,
    PathTrafficConfig,
    SimulationConfig,
    TelemetryConfig,
)


class ImmutableConfig(BaseModel):
    """Strict immutable configuration base for corpus declarations."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetSplit(StrEnum):
    """Pre-materialization dataset partition."""

    TRAIN = "train"
    DEVELOPMENT = "development"
    TEST = "test"


class TopologyFamily(ImmutableConfig):
    """One abstract two-path parameter family, not a general graph topology."""

    family_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    split: DatasetSplit
    current_service_packets_per_tick: int = Field(gt=0, le=10_000)
    candidate_service_packets_per_tick: int = Field(gt=0, le=10_000)
    current_queue_limit_packets: int = Field(ge=0, le=1_000_000)
    candidate_queue_limit_packets: int = Field(ge=0, le=1_000_000)
    current_propagation_delay_ms: float = Field(ge=0)
    candidate_propagation_delay_ms: float = Field(ge=0)


class TrafficLevel(ImmutableConfig):
    """Foreground and path-specific cross-traffic declaration."""

    traffic_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    arrival_mode: ArrivalMode
    foreground_rate_packets_per_tick: float = Field(ge=0, le=100)
    current_cross_rate_packets_per_tick: float = Field(ge=0, le=100)
    candidate_cross_rate_packets_per_tick: float = Field(ge=0, le=100)

    @model_validator(mode="after")
    def validate_constant_rates(self) -> Self:
        if self.arrival_mode is ArrivalMode.CONSTANT:
            rates = (
                self.foreground_rate_packets_per_tick,
                self.current_cross_rate_packets_per_tick,
                self.candidate_cross_rate_packets_per_tick,
            )
            if any(not rate.is_integer() for rate in rates):
                raise ValueError("constant traffic rates must be integer packet counts")
        return self


class ObservationRegime(ImmutableConfig):
    """Action-independent telemetry/failure injection regime."""

    regime_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    failure_classification: str = Field(min_length=1)
    sample_every_ticks: int = Field(default=1, gt=0)
    minimum_delay_ticks: int = Field(default=0, ge=0)
    maximum_delay_ticks: int = Field(default=0, ge=0)
    missing_probability: float = Field(default=0.0, ge=0, le=1)
    maximum_clock_skew_ticks: int = Field(default=0, ge=0)
    stale_after_ticks: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_delay_bounds(self) -> Self:
        if self.maximum_delay_ticks < self.minimum_delay_ticks:
            raise ValueError("maximum telemetry delay cannot be below minimum delay")
        return self


class SeedGroup(ImmutableConfig):
    """Named seed group assigned to exactly one split before generation."""

    group_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    split: DatasetSplit
    seeds: tuple[int, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_seeds(self) -> Self:
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seed group contains duplicate seeds")
        return self


class SimulationDefaults(ImmutableConfig):
    """Simulation values shared across materialized corpus scenarios."""

    duration_ticks: int = Field(gt=0, le=100_000)
    tick_duration_ms: float = Field(gt=0)
    deadline_ms: float = Field(gt=0)


class CorpusScenario(ImmutableConfig):
    """One fully materialized, split-labelled paired scenario."""

    corpus_id: str
    split: DatasetSplit
    topology_family_id: str
    topology_abstraction: Literal["abstract-two-path-parameter-family"] = (
        "abstract-two-path-parameter-family"
    )
    traffic_level_id: str
    observation_regime_id: str
    failure_classification: str
    seed_group_id: str
    seed: int
    simulation: SimulationConfig

    @property
    def run_id(self) -> str:
        """Return a Windows-safe immutable identifier for this paired run."""
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"m08-{hashlib.sha256(payload).hexdigest()[:24]}"


class CorpusMatrixConfig(ImmutableConfig):
    """Complete configuration-driven M08 scenario matrix."""

    schema_version: Literal["1.0.0"] = "1.0.0"
    corpus_id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    defaults: SimulationDefaults
    topology_families: tuple[TopologyFamily, ...] = Field(min_length=1)
    traffic_levels: tuple[TrafficLevel, ...] = Field(min_length=1)
    observation_regimes: tuple[ObservationRegime, ...] = Field(min_length=1)
    seed_groups: tuple[SeedGroup, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_matrix_isolation(self) -> Self:
        self._require_unique("topology family", [item.family_id for item in self.topology_families])
        self._require_unique("traffic level", [item.traffic_id for item in self.traffic_levels])
        self._require_unique(
            "observation regime", [item.regime_id for item in self.observation_regimes]
        )
        self._require_unique("seed group", [item.group_id for item in self.seed_groups])

        family_splits = {item.split for item in self.topology_families}
        seed_splits = {item.split for item in self.seed_groups}
        required = set(DatasetSplit)
        if family_splits != required:
            raise ValueError("topology families must declare train, development, and test splits")
        if seed_splits != required:
            raise ValueError("seed groups must declare train, development, and test splits")

        seen_seeds: dict[int, DatasetSplit] = {}
        for group in self.seed_groups:
            for seed in group.seeds:
                prior = seen_seeds.get(seed)
                if prior is not None and prior is not group.split:
                    raise ValueError(f"seed {seed} appears in multiple splits")
                if prior is not None:
                    raise ValueError(f"seed {seed} appears in multiple seed groups")
                seen_seeds[seed] = group.split
        return self

    @staticmethod
    def _require_unique(label: str, values: list[str]) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate {label} identifier")

    def canonical_json(self) -> str:
        """Serialize the complete matrix with stable ordering."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    @property
    def config_hash(self) -> str:
        """Return the SHA-256 of the declared matrix."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest().upper()

    @property
    def pair_count(self) -> int:
        """Return the number of paired scenarios after split-compatible expansion."""
        return sum(
            len(group.seeds) * len(self.traffic_levels) * len(self.observation_regimes)
            for family in self.topology_families
            for group in self.seed_groups
            if family.split is group.split
        )

    def scenarios(self) -> tuple[CorpusScenario, ...]:
        """Materialize deterministic scenarios without random row-level splitting."""
        scenarios: list[CorpusScenario] = []
        for family in self.topology_families:
            for group in self.seed_groups:
                if family.split is not group.split:
                    continue
                for seed in group.seeds:
                    for traffic in self.traffic_levels:
                        for regime in self.observation_regimes:
                            scenario_id = (
                                f"{self.corpus_id}-{family.family_id}-{traffic.traffic_id}-"
                                f"{regime.regime_id}-{group.group_id}-{seed}"
                            )
                            simulation = self._simulation(
                                scenario_id, seed, family, traffic, regime
                            )
                            scenarios.append(
                                CorpusScenario(
                                    corpus_id=self.corpus_id,
                                    split=family.split,
                                    topology_family_id=family.family_id,
                                    traffic_level_id=traffic.traffic_id,
                                    observation_regime_id=regime.regime_id,
                                    failure_classification=regime.failure_classification,
                                    seed_group_id=group.group_id,
                                    seed=seed,
                                    simulation=simulation,
                                )
                            )
        return tuple(scenarios)

    def _simulation(
        self,
        scenario_id: str,
        seed: int,
        family: TopologyFamily,
        traffic: TrafficLevel,
        regime: ObservationRegime,
    ) -> SimulationConfig:
        current_path = "path-current"
        candidate_path = "path-candidate"
        return SimulationConfig(
            scenario_id=scenario_id,
            seed=seed,
            duration_ticks=self.defaults.duration_ticks,
            tick_duration_ms=self.defaults.tick_duration_ms,
            deadline_ms=self.defaults.deadline_ms,
            current_path_id=current_path,
            candidate_path_id=candidate_path,
            paths=(
                PathConfig(
                    path_id=current_path,
                    service_packets_per_tick=family.current_service_packets_per_tick,
                    queue_limit_packets=family.current_queue_limit_packets,
                    propagation_delay_ms=family.current_propagation_delay_ms,
                ),
                PathConfig(
                    path_id=candidate_path,
                    service_packets_per_tick=family.candidate_service_packets_per_tick,
                    queue_limit_packets=family.candidate_queue_limit_packets,
                    propagation_delay_ms=family.candidate_propagation_delay_ms,
                ),
            ),
            foreground_process=ArrivalProcess(
                mode=traffic.arrival_mode,
                rate_packets_per_tick=traffic.foreground_rate_packets_per_tick,
            ),
            cross_traffic=(
                PathTrafficConfig(
                    path_id=current_path,
                    process=ArrivalProcess(
                        mode=traffic.arrival_mode,
                        rate_packets_per_tick=traffic.current_cross_rate_packets_per_tick,
                    ),
                ),
                PathTrafficConfig(
                    path_id=candidate_path,
                    process=ArrivalProcess(
                        mode=traffic.arrival_mode,
                        rate_packets_per_tick=traffic.candidate_cross_rate_packets_per_tick,
                    ),
                ),
            ),
            telemetry=TelemetryConfig(
                sample_every_ticks=regime.sample_every_ticks,
                minimum_delay_ticks=regime.minimum_delay_ticks,
                maximum_delay_ticks=regime.maximum_delay_ticks,
                missing_probability=regime.missing_probability,
                maximum_clock_skew_ticks=regime.maximum_clock_skew_ticks,
                stale_after_ticks=regime.stale_after_ticks,
            ),
        )


def load_corpus_matrix(path: Path) -> CorpusMatrixConfig:
    """Load and validate one JSON corpus-matrix declaration."""
    return CorpusMatrixConfig.model_validate_json(path.read_text(encoding="utf-8"))
