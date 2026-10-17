"""Training examples, outcome parsing, and deterministic visible-feature extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ictc.baselines.access import AccessPhase, CorpusAccessGuard, CorpusTable
from ictc.baselines.features import load_baseline_input
from ictc.baselines.types import NUMERIC_OBSERVATION_FIELDS, BaselineId, BaselineInput
from ictc.datasets.m08_config import DatasetSplit
from ictc.domain.enums import ActionKind


@dataclass(frozen=True, slots=True)
class OutcomeMetrics:
    action_kind: ActionKind
    selected_path_id: str
    schedule_digest: str
    mean_latency_ms: float
    packet_loss_ratio: float
    jitter_ms: float
    deadline_violation_rate: float

    def __post_init__(self) -> None:
        values = (self.mean_latency_ms, self.packet_loss_ratio, self.jitter_ms)
        if not self.selected_path_id or len(self.schedule_digest) < 16:
            raise ValueError("outcome path and schedule digest are required")
        if not all(math.isfinite(value) and value >= 0 for value in values):
            raise ValueError("outcome metrics must be finite and nonnegative")
        if not 0 <= self.packet_loss_ratio <= 1 or not 0 <= self.deadline_violation_rate <= 1:
            raise ValueError("outcome ratios must lie between zero and one")

    def objective(
        self,
        *,
        loss_weight: float,
        jitter_weight: float,
        deadline_weight: float,
    ) -> float:
        return (
            self.mean_latency_ms
            + loss_weight * self.packet_loss_ratio
            + jitter_weight * self.jitter_ms
            + deadline_weight * self.deadline_violation_rate
        )


@dataclass(frozen=True, slots=True)
class PairedOutcomes:
    pair_id: str
    scenario_id: str
    keep: OutcomeMetrics
    reroute: OutcomeMetrics

    def __post_init__(self) -> None:
        if not self.pair_id or not self.scenario_id:
            raise ValueError("paired outcome identifiers are required")
        if self.keep.action_kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("paired keep outcome has the wrong action")
        if self.reroute.action_kind is not ActionKind.REROUTE_TO_PATH:
            raise ValueError("paired reroute outcome has the wrong action")
        if self.keep.schedule_digest != self.reroute.schedule_digest:
            raise ValueError("paired outcomes must share one random schedule")

    @property
    def latency_effect_ms(self) -> float:
        return self.reroute.mean_latency_ms - self.keep.mean_latency_ms

    @property
    def reroute_beneficial(self) -> bool:
        return self.latency_effect_ms < 0


@dataclass(frozen=True, slots=True)
class TrainingExample:
    input: BaselineInput
    outcomes: PairedOutcomes

    def __post_init__(self) -> None:
        if self.input.pair_id != self.outcomes.pair_id:
            raise ValueError("training input and outcomes have different pair IDs")
        if self.input.scenario_id != self.outcomes.scenario_id:
            raise ValueError("training input and outcomes have different scenario IDs")
        if self.input.schedule_digest != self.outcomes.keep.schedule_digest:
            raise ValueError("training input and outcomes have different schedules")


@dataclass(frozen=True, slots=True)
class FeatureVector:
    names: tuple[str, ...]
    values: tuple[float, ...]
    fields: tuple[str, ...]
    complete: bool
    telemetry_items: int
    telemetry_bytes: int

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError("feature names and values must have equal length")
        if len(self.names) != len(set(self.names)):
            raise ValueError("feature names must be unique")
        if not all(math.isfinite(value) for value in self.values):
            raise ValueError("feature values must be finite")
        if min(self.telemetry_items, self.telemetry_bytes) < 0:
            raise ValueError("feature communication costs cannot be negative")


def extract_features(item: BaselineInput, fields: tuple[str, ...]) -> FeatureVector:
    if len(fields) != len(set(fields)):
        raise ValueError("feature fields must be unique")
    unsupported = set(fields) - set(NUMERIC_OBSERVATION_FIELDS)
    if unsupported:
        raise ValueError(f"unsupported feature fields: {sorted(unsupported)}")
    current = item.snapshot(item.current_path_id)
    candidate = item.snapshot(item.candidate_path_id)
    complete = current is not None and candidate is not None
    names: list[str] = []
    values: list[float] = []
    for field_name in fields:
        current_value = current.numeric(field_name) if current is not None else 0.0
        candidate_value = candidate.numeric(field_name) if candidate is not None else 0.0
        present = float(complete)
        names.extend(
            (
                f"current.{field_name}",
                f"candidate.{field_name}",
                f"candidate_minus_current.{field_name}",
                f"both_present.{field_name}",
            )
        )
        values.extend((current_value, candidate_value, candidate_value - current_value, present))
    observed_paths = int(current is not None) + int(candidate is not None)
    items = observed_paths * len(fields)
    return FeatureVector(
        names=tuple(names),
        values=tuple(values),
        fields=fields,
        complete=complete,
        telemetry_items=items,
        telemetry_bytes=items * 8,
    )


def _parse_outcome(row: dict[str, object]) -> OutcomeMetrics:
    action = row.get("action_kind")
    path = row.get("selected_path_id")
    schedule = row.get("schedule_digest")
    latency = row.get("mean_latency_ms")
    loss = row.get("packet_loss_ratio")
    jitter = row.get("jitter_ms")
    deadline = row.get("deadline_violation_rate")
    numeric = (latency, loss, jitter, deadline)
    if (
        not isinstance(action, str)
        or not isinstance(path, str)
        or not isinstance(schedule, str)
        or not all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in numeric
        )
    ):
        raise ValueError("outcome row has invalid fields")
    return OutcomeMetrics(
        action_kind=ActionKind(action),
        selected_path_id=path,
        schedule_digest=schedule,
        mean_latency_ms=float(cast(int | float, latency)),
        packet_loss_ratio=float(cast(int | float, loss)),
        jitter_ms=float(cast(int | float, jitter)),
        deadline_violation_rate=float(cast(int | float, deadline)),
    )


def load_outcomes(
    directory: Path,
    *,
    item: BaselineInput,
    baseline_id: BaselineId,
    phase: AccessPhase,
    guard: CorpusAccessGuard,
) -> PairedOutcomes:
    rows = guard.read_table(
        directory,
        baseline_id=baseline_id,
        phase=phase,
        split=item.split,
        table=CorpusTable.OUTCOMES,
    ).to_pylist()
    parsed = tuple(_parse_outcome(row) for row in cast(list[dict[str, object]], rows))
    keep = [outcome for outcome in parsed if outcome.action_kind is ActionKind.KEEP_CURRENT_PATH]
    reroute = [outcome for outcome in parsed if outcome.action_kind is ActionKind.REROUTE_TO_PATH]
    if len(keep) != 1 or len(reroute) != 1:
        raise ValueError("paired corpus must contain exactly one outcome for each action")
    scenario_ids = {row.get("scenario_id") for row in cast(list[dict[str, object]], rows)}
    if scenario_ids != {item.scenario_id}:
        raise ValueError("outcome scenario does not match the baseline input")
    return PairedOutcomes(item.pair_id, item.scenario_id, keep[0], reroute[0])


def load_training_examples(
    directories: tuple[Path, ...],
    *,
    baseline_id: BaselineId,
    split: DatasetSplit,
    guard: CorpusAccessGuard | None = None,
) -> tuple[TrainingExample, ...]:
    if split is DatasetSplit.TEST:
        raise ValueError("test outcomes cannot create training or tuning examples")
    access = guard or CorpusAccessGuard()
    examples: list[TrainingExample] = []
    for directory in sorted(directories):
        item = load_baseline_input(
            directory,
            baseline_id=baseline_id,
            split=split,
            guard=access,
        )
        outcomes = load_outcomes(
            directory,
            item=item,
            baseline_id=baseline_id,
            phase=AccessPhase.TRAINING,
            guard=access,
        )
        examples.append(TrainingExample(item, outcomes))
    if not examples:
        raise ValueError("at least one training example is required")
    return tuple(examples)
