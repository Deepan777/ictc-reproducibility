"""Common, immutable M09 baseline inputs, outputs, and observation policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from ictc.datasets.m08_config import DatasetSplit
from ictc.domain.models import NetworkAction


class BaselineId(StrEnum):
    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    B6 = "B6"
    B7 = "B7"
    B8 = "B8"
    B9 = "B9"


class BaselineDisposition(StrEnum):
    EXECUTE = "EXECUTE"
    ABSTAIN = "ABSTAIN"
    FAIL = "FAIL"


class BaselineReason(StrEnum):
    ORACLE_BEST_ACTION = "ORACLE_BEST_ACTION"
    NO_GATE = "NO_GATE"
    FIXED_TELEMETRY_ACCEPTED = "FIXED_TELEMETRY_ACCEPTED"
    ADAPTIVE_UTILITY_ACCEPTED = "ADAPTIVE_UTILITY_ACCEPTED"
    CONFIDENCE_ACCEPTED = "CONFIDENCE_ACCEPTED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    PREDICTED_HARM = "PREDICTED_HARM"
    CAUSAL_EFFECT_SUPPORTED = "CAUSAL_EFFECT_SUPPORTED"
    CAUSAL_EFFECT_UNSUPPORTED = "CAUSAL_EFFECT_UNSUPPORTED"
    CONSTRAINTS_SATISFIED = "CONSTRAINTS_SATISFIED"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    STATIC_CONTRACT_SATISFIED = "STATIC_CONTRACT_SATISFIED"
    STATIC_CONTRACT_UNAVAILABLE = "STATIC_CONTRACT_UNAVAILABLE"
    TWIN_BEST_ACTION = "TWIN_BEST_ACTION"
    FEATURE_SELECTOR_ACCEPTED = "FEATURE_SELECTOR_ACCEPTED"
    REQUIRED_OBSERVATION_MISSING = "REQUIRED_OBSERVATION_MISSING"
    INVALID_INPUT = "INVALID_INPUT"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"


NUMERIC_OBSERVATION_FIELDS = (
    "queue_occupancy",
    "utilization",
    "sample_age_ticks",
    "measurement_delay_ticks",
    "clock_skew_ticks",
)


@dataclass(frozen=True, slots=True)
class PathSnapshot:
    """Latest controller-visible measurement for one path at the decision tick."""

    path_id: str
    sample_tick: int
    available_tick: int
    observed_timestamp_tick: int
    queue_occupancy: int
    utilization: float
    sample_age_ticks: int
    measurement_delay_ticks: int
    clock_skew_ticks: int
    stale: bool
    source_id: str
    provenance_digest: str

    def __post_init__(self) -> None:
        if not self.path_id or not self.source_id:
            raise ValueError("path and source IDs are required")
        if (
            min(
                self.sample_tick,
                self.available_tick,
                self.queue_occupancy,
                self.sample_age_ticks,
                self.measurement_delay_ticks,
            )
            < 0
        ):
            raise ValueError("path snapshot counters cannot be negative")
        if self.available_tick < self.sample_tick:
            raise ValueError("available tick cannot precede the sample tick")
        if not math.isfinite(self.utilization) or self.utilization < 0:
            raise ValueError("utilization must be finite and nonnegative")
        if len(self.provenance_digest) < 16:
            raise ValueError("provenance digest is too short")

    def numeric(self, field: str) -> float:
        if field not in NUMERIC_OBSERVATION_FIELDS:
            raise KeyError(f"unsupported numeric observation field: {field}")
        value = getattr(self, field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"observation field {field} is not numeric")
        return float(value)


@dataclass(frozen=True, slots=True)
class BaselineInput:
    """One paired action opportunity with only pre-action visible observations."""

    pair_id: str
    scenario_id: str
    split: DatasetSplit
    seed: int
    decision_tick: int
    current_path_id: str
    candidate_path_id: str
    proposed_action: NetworkAction
    snapshots: tuple[PathSnapshot, ...]
    scenario_config_digest: str
    schedule_digest: str

    def __post_init__(self) -> None:
        if not self.pair_id or not self.scenario_id:
            raise ValueError("pair and scenario IDs are required")
        if self.seed < 0 or self.decision_tick < 0:
            raise ValueError("seed and decision tick cannot be negative")
        if self.current_path_id == self.candidate_path_id:
            raise ValueError("current and candidate paths must differ")
        if self.proposed_action.path_id != self.candidate_path_id:
            raise ValueError("proposed action must target the candidate path")
        path_ids = [item.path_id for item in self.snapshots]
        if len(path_ids) != len(set(path_ids)):
            raise ValueError("at most one snapshot is allowed per path")
        if any(item.available_tick > self.decision_tick for item in self.snapshots):
            raise ValueError("baseline input contains a future observation")
        if len(self.scenario_config_digest) < 16:
            raise ValueError("scenario configuration digest is too short")
        if len(self.schedule_digest) < 16:
            raise ValueError("random schedule digest is too short")

    def snapshot(self, path_id: str) -> PathSnapshot | None:
        return next((item for item in self.snapshots if item.path_id == path_id), None)


@dataclass(frozen=True, slots=True)
class ObservationPolicy:
    """Field-level observation and communication budget declared before evaluation."""

    fields: tuple[str, ...]
    maximum_items: int
    maximum_bytes: int
    adaptive: bool = False

    def __post_init__(self) -> None:
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("observation-policy fields must be unique")
        unsupported = set(self.fields) - set(NUMERIC_OBSERVATION_FIELDS)
        if unsupported:
            raise ValueError(f"unsupported observation-policy fields: {sorted(unsupported)}")
        if not self.fields:
            if self.maximum_items != 0 or self.maximum_bytes != 0:
                raise ValueError("an empty observation policy requires zero budgets")
        elif self.maximum_items <= 0 or self.maximum_bytes <= 0:
            raise ValueError("nonempty observation-policy budgets must be positive")


@dataclass(frozen=True, slots=True)
class BaselineResult:
    """Common machine-readable output emitted by every M09 baseline."""

    baseline_id: BaselineId
    pair_id: str
    scenario_id: str
    disposition: BaselineDisposition
    reason: BaselineReason
    proposed_action: NetworkAction
    selected_action: NetworkAction | None
    observation_fields: tuple[str, ...]
    telemetry_items: int
    telemetry_bytes: int
    inference_time_ns: int
    configuration_digest: str
    score: float | None = None
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.pair_id or not self.scenario_id:
            raise ValueError("result pair and scenario IDs are required")
        if self.disposition is BaselineDisposition.EXECUTE and self.selected_action is None:
            raise ValueError("executed result requires a selected action")
        if self.disposition is not BaselineDisposition.EXECUTE and self.selected_action is not None:
            raise ValueError("abstained or failed result cannot select an action")
        if min(self.telemetry_items, self.telemetry_bytes, self.inference_time_ns) < 0:
            raise ValueError("cost measurements cannot be negative")
        if len(self.configuration_digest) < 16:
            raise ValueError("configuration digest is too short")
        if self.score is not None and not math.isfinite(self.score):
            raise ValueError("baseline score must be finite")
        detail_names = [name for name, _value in self.details]
        if detail_names != sorted(detail_names) or len(detail_names) != len(set(detail_names)):
            raise ValueError("result details must use unique names in canonical order")


class BaselineController(Protocol):
    @property
    def baseline_id(self) -> BaselineId: ...

    @property
    def observation_policy(self) -> ObservationPolicy: ...

    @property
    def configuration_digest(self) -> str: ...

    def decide(self, item: BaselineInput) -> BaselineResult: ...
