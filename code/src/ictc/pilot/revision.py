"""Context and graph declarations for the post-M10 bounded revision repair."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from ictc.causal import (
    AdjustmentObservation,
    BackdoorIdentifier,
    CausalDAG,
    EstimationResult,
    IdentifiabilityResult,
    estimate_stratified_backdoor,
)
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind

M10R_GRAPH_ID = "m10_contextual_observational_selection_v2"
M10R_ASSIGNMENT_POLICY_ID = "m10r_bounded_queue_pressure_assignment_v1"
M10R_GRAPH_NODES = (
    "TopologyContext",
    "TrafficLevel",
    "QueuePressureBin",
    "RouteAction",
    "EndToEndDelay",
)
M10R_GRAPH_EDGES = (
    ("TopologyContext", "QueuePressureBin"),
    ("TopologyContext", "EndToEndDelay"),
    ("TrafficLevel", "QueuePressureBin"),
    ("TrafficLevel", "EndToEndDelay"),
    ("QueuePressureBin", "RouteAction"),
    ("QueuePressureBin", "EndToEndDelay"),
    ("RouteAction", "EndToEndDelay"),
)


class RelativeBin(StrEnum):
    """Pre-simulation candidate-minus-current relation."""

    LOWER = "LOWER"
    EQUAL = "EQUAL"
    HIGHER = "HIGHER"


class EffectSupportStatus(StrEnum):
    """Disposition of an estimate under the frozen per-arm support rule."""

    SUPPORTED = "SUPPORTED"
    ESTIMATOR_INESTIMABLE = "ESTIMATOR_INESTIMABLE"
    BELOW_MINIMUM_PER_ARM = "BELOW_MINIMUM_PER_ARM"


def _relative(candidate: float, current: float) -> RelativeBin:
    if candidate < current:
        return RelativeBin.LOWER
    if candidate > current:
        return RelativeBin.HIGHER
    return RelativeBin.EQUAL


@dataclass(frozen=True, slots=True)
class TopologyContext:
    """Controller-safe topology relations fixed before simulation outcomes exist."""

    service_relation: RelativeBin
    propagation_delay_relation: RelativeBin
    queue_limit_relation: RelativeBin

    @property
    def context_id(self) -> str:
        return (
            f"service={self.service_relation.value};"
            f"delay={self.propagation_delay_relation.value};"
            f"queue={self.queue_limit_relation.value}"
        )

    @classmethod
    def from_path_configuration(
        cls,
        *,
        current_service_packets_per_tick: int,
        candidate_service_packets_per_tick: int,
        current_propagation_delay_ms: float,
        candidate_propagation_delay_ms: float,
        current_queue_limit_packets: int,
        candidate_queue_limit_packets: int,
    ) -> TopologyContext:
        values = (
            current_service_packets_per_tick,
            candidate_service_packets_per_tick,
            current_propagation_delay_ms,
            candidate_propagation_delay_ms,
            current_queue_limit_packets,
            candidate_queue_limit_packets,
        )
        if any(value < 0 for value in values):
            raise ValueError("topology context inputs cannot be negative")
        if current_service_packets_per_tick == 0 or candidate_service_packets_per_tick == 0:
            raise ValueError("topology service rates must be positive")
        return cls(
            _relative(candidate_service_packets_per_tick, current_service_packets_per_tick),
            _relative(candidate_propagation_delay_ms, current_propagation_delay_ms),
            _relative(candidate_queue_limit_packets, current_queue_limit_packets),
        )


@dataclass(frozen=True, slots=True, order=True)
class ContextKey:
    """Pre-action context for one supported local causal effect."""

    topology_context_id: str
    traffic_level_id: str
    queue_pressure_bin: int

    def __post_init__(self) -> None:
        if not self.topology_context_id or not self.traffic_level_id:
            raise ValueError("context and traffic identifiers are required")
        if self.queue_pressure_bin not in {-1, 0, 1}:
            raise ValueError("queue pressure context must be -1, 0, or 1")


@dataclass(frozen=True, slots=True)
class ContextObservation:
    """One controller-safe observational training row for a local effect."""

    split: DatasetSplit
    key: ContextKey
    treatment: ActionKind
    observed_latency_ms: float
    cluster_id: str | None = None

    def __post_init__(self) -> None:
        if self.split is not DatasetSplit.TRAIN:
            raise PermissionError("contextual effect fitting accepts training rows only")
        if self.treatment not in {ActionKind.KEEP_CURRENT_PATH, ActionKind.REROUTE_TO_PATH}:
            raise ValueError("contextual effect treatment must be KEEP or REROUTE")
        if self.observed_latency_ms < 0:
            raise ValueError("observed latency cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextEffect:
    """One local estimate plus the stricter revision support disposition."""

    key: ContextKey
    result: EstimationResult
    keep_count: int
    reroute_count: int
    minimum_per_arm: int

    def __post_init__(self) -> None:
        if self.keep_count < 0 or self.reroute_count < 0:
            raise ValueError("context support counts cannot be negative")
        _validate_minimum_per_arm(self.minimum_per_arm)

    @property
    def supported(self) -> bool:
        return (
            self.result.estimable
            and self.keep_count >= self.minimum_per_arm
            and self.reroute_count >= self.minimum_per_arm
        )

    @property
    def support_status(self) -> EffectSupportStatus:
        if not self.result.estimable:
            return EffectSupportStatus.ESTIMATOR_INESTIMABLE
        if not self.supported:
            return EffectSupportStatus.BELOW_MINIMUM_PER_ARM
        return EffectSupportStatus.SUPPORTED


@dataclass(frozen=True, slots=True)
class PopulationEffect:
    """Population effect with support checked in every pressure stratum."""

    result: EstimationResult
    minimum_per_arm: int

    def __post_init__(self) -> None:
        _validate_minimum_per_arm(self.minimum_per_arm)

    @property
    def support_status(self) -> EffectSupportStatus:
        return _stratified_support_status(self.result, self.minimum_per_arm)

    @property
    def supported(self) -> bool:
        return self.support_status is EffectSupportStatus.SUPPORTED


@dataclass(frozen=True, slots=True)
class TrafficEffect:
    """One traffic-conditional effect with pressure-stratum support diagnostics."""

    traffic_level_id: str
    result: EstimationResult
    minimum_per_arm: int

    def __post_init__(self) -> None:
        if not self.traffic_level_id:
            raise ValueError("traffic identifier is required")
        _validate_minimum_per_arm(self.minimum_per_arm)

    @property
    def support_status(self) -> EffectSupportStatus:
        return _stratified_support_status(self.result, self.minimum_per_arm)

    @property
    def supported(self) -> bool:
        return self.support_status is EffectSupportStatus.SUPPORTED


@dataclass(frozen=True, slots=True)
class ContextualEffectModel:
    """Local estimates with no implicit fallback to broader population effects."""

    effects: tuple[ContextEffect, ...]

    def effect_for(self, key: ContextKey) -> ContextEffect | None:
        return next((effect for effect in self.effects if effect.key == key), None)

    def admissible_estimate_for(self, key: ContextKey) -> EstimationResult | None:
        effect = self.effect_for(key)
        return effect.result if effect is not None and effect.supported else None


@dataclass(frozen=True, slots=True)
class RevisionEffectModel:
    """Frozen global, traffic, and local comparisons without cross-scope fallback."""

    population_effect: PopulationEffect
    traffic_effects: tuple[TrafficEffect, ...]
    contextual_effects: ContextualEffectModel

    def __post_init__(self) -> None:
        traffic_ids = [effect.traffic_level_id for effect in self.traffic_effects]
        if traffic_ids != sorted(traffic_ids) or len(traffic_ids) != len(set(traffic_ids)):
            raise ValueError("traffic effects must have unique identifiers in canonical order")

    def traffic_effect_for(self, traffic_level_id: str) -> TrafficEffect | None:
        return next(
            (
                effect
                for effect in self.traffic_effects
                if effect.traffic_level_id == traffic_level_id
            ),
            None,
        )

    def admissible_population_estimate(self) -> EstimationResult | None:
        effect = self.population_effect
        return effect.result if effect.supported else None

    def admissible_traffic_estimate_for(self, traffic_level_id: str) -> EstimationResult | None:
        effect = self.traffic_effect_for(traffic_level_id)
        return effect.result if effect is not None and effect.supported else None

    def admissible_context_estimate_for(self, key: ContextKey) -> EstimationResult | None:
        """Return only the requested local estimate; broader estimates are never substituted."""
        return self.contextual_effects.admissible_estimate_for(key)


def _validate_minimum_per_arm(minimum_per_arm: int) -> None:
    if minimum_per_arm < 2:
        raise ValueError("effect minimum support must be at least two per arm")


def _stratified_support_status(
    result: EstimationResult, minimum_per_arm: int
) -> EffectSupportStatus:
    if not result.estimable:
        return EffectSupportStatus.ESTIMATOR_INESTIMABLE
    if not result.strata or any(
        diagnostic.baseline_count < minimum_per_arm
        or diagnostic.intervention_count < minimum_per_arm
        for diagnostic in result.strata
    ):
        return EffectSupportStatus.BELOW_MINIMUM_PER_ARM
    return EffectSupportStatus.SUPPORTED


def _estimate_rows(rows: tuple[ContextObservation, ...], *, estimand: str) -> EstimationResult:
    adjustment_rows = tuple(
        AdjustmentObservation.from_mapping(
            treatment=row.treatment,
            outcome=row.observed_latency_ms,
            covariates={"QueuePressureBin": float(row.key.queue_pressure_bin)},
            cluster_id=row.cluster_id,
        )
        for row in rows
    )
    return estimate_stratified_backdoor(
        adjustment_rows,
        revision_identification(True),
        ("QueuePressureBin",),
        estimand=estimand,
        unit="ms",
    )


def fit_contextual_effect_model(
    observations: tuple[ContextObservation, ...], *, minimum_per_arm: int = 10
) -> ContextualEffectModel:
    """Fit exact local effects and retain unsupported cells without broad fallback."""
    if not observations:
        raise ValueError("contextual effect fitting requires observations")
    _validate_minimum_per_arm(minimum_per_arm)
    grouped: defaultdict[ContextKey, list[ContextObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.key].append(observation)
    effects: list[ContextEffect] = []
    for key, rows in sorted(grouped.items()):
        result = _estimate_rows(
            tuple(rows),
            estimand=(
                "context-local reroute-minus-keep latency: "
                f"{key.topology_context_id};{key.traffic_level_id};"
                f"pressure={key.queue_pressure_bin}"
            ),
        )
        keep_count = sum(row.treatment is ActionKind.KEEP_CURRENT_PATH for row in rows)
        reroute_count = len(rows) - keep_count
        effects.append(ContextEffect(key, result, keep_count, reroute_count, minimum_per_arm))
    return ContextualEffectModel(tuple(effects))


def fit_revision_effect_model(
    observations: tuple[ContextObservation, ...], *, minimum_per_arm: int = 10
) -> RevisionEffectModel:
    """Fit the three predeclared comparison scopes from training observations only."""
    if not observations:
        raise ValueError("revision effect fitting requires observations")
    _validate_minimum_per_arm(minimum_per_arm)

    population = PopulationEffect(
        _estimate_rows(observations, estimand="population reroute-minus-keep latency"),
        minimum_per_arm,
    )
    grouped: defaultdict[str, list[ContextObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.key.traffic_level_id].append(observation)
    traffic_effects = tuple(
        TrafficEffect(
            traffic_level_id,
            _estimate_rows(
                tuple(rows),
                estimand=f"traffic-conditional reroute-minus-keep latency: {traffic_level_id}",
            ),
            minimum_per_arm,
        )
        for traffic_level_id, rows in sorted(grouped.items())
    )
    return RevisionEffectModel(
        population,
        traffic_effects,
        fit_contextual_effect_model(observations, minimum_per_arm=minimum_per_arm),
    )


def _graph() -> CausalDAG:
    return CausalDAG(M10R_GRAPH_NODES, M10R_GRAPH_EDGES, M10R_GRAPH_NODES)


def revision_identification(queue_pressure_available: bool) -> IdentifiabilityResult:
    """Identify the revision total effect without conflating context with adjustment."""
    available = {"TopologyContext", "TrafficLevel", "RouteAction", "EndToEndDelay"}
    if queue_pressure_available:
        available.add("QueuePressureBin")
    return BackdoorIdentifier(_graph(), graph_id=M10R_GRAPH_ID).identify(
        "RouteAction", "EndToEndDelay", available
    )


def _digest() -> str:
    payload = {
        "graph_id": M10R_GRAPH_ID,
        "nodes": M10R_GRAPH_NODES,
        "edges": M10R_GRAPH_EDGES,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


M10R_GRAPH_DIGEST = _digest()
