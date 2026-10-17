"""B0-B9 baseline controllers selected and fidelity-bounded by the M09 review."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter_ns
from typing import cast

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_selection import mutual_info_classif  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression, Ridge  # type: ignore[import-untyped]

from ictc.baselines.access import AccessPhase, CorpusAccessGuard, CorpusTable
from ictc.baselines.data import TrainingExample, extract_features, load_outcomes
from ictc.baselines.features import load_baseline_input
from ictc.baselines.types import (
    NUMERIC_OBSERVATION_FIELDS,
    BaselineDisposition,
    BaselineId,
    BaselineInput,
    BaselineReason,
    BaselineResult,
    ObservationPolicy,
)
from ictc.causal.estimation import AdjustmentObservation, estimate_stratified_backdoor
from ictc.causal.graph import CausalDAG
from ictc.causal.identification import BackdoorIdentifier
from ictc.datasets.m08_config import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]

CONFIDENCE_THRESHOLD_CANDIDATES = (0.5, 0.6, 0.7, 0.8, 0.9)
MAXIMUM_SELECTIVE_ERROR = 0.1
DIGITAL_TWIN_ALPHA_CANDIDATES = (0.0, 0.1, 1.0, 10.0)
CORRELATION_TOP_K_CANDIDATES = (1, 2, 3, 4, 5)

DECLARED_CAUSAL_GRAPH_ID = "routing_telemetry_v1"
DECLARED_CAUSAL_NODES = (
    "ClockSkew",
    "CrossTraffic",
    "EffectiveCapacity",
    "EndToEndDelay",
    "ExposedCrossTraffic",
    "LinkCapacity",
    "LinkDegradation",
    "LinkUtilization",
    "MeasurementDelay",
    "ObservedFreshness",
    "ObservedTelemetryValue",
    "ObservedTimestampQuality",
    "OfferedLoad",
    "PacketLoss",
    "QueueOccupancy",
    "RouteAction",
    "SelectedPath",
    "SourceCompromise",
)
DECLARED_CAUSAL_EDGES = (
    ("ClockSkew", "ObservedTimestampQuality"),
    ("CrossTraffic", "LinkUtilization"),
    ("CrossTraffic", "QueueOccupancy"),
    ("EffectiveCapacity", "QueueOccupancy"),
    ("LinkCapacity", "EndToEndDelay"),
    ("LinkCapacity", "QueueOccupancy"),
    ("LinkDegradation", "EffectiveCapacity"),
    ("MeasurementDelay", "ObservedFreshness"),
    ("OfferedLoad", "QueueOccupancy"),
    ("QueueOccupancy", "EndToEndDelay"),
    ("QueueOccupancy", "PacketLoss"),
    ("RouteAction", "SelectedPath"),
    ("SelectedPath", "EndToEndDelay"),
    ("SelectedPath", "ExposedCrossTraffic"),
    ("SelectedPath", "PacketLoss"),
    ("SourceCompromise", "ObservedTelemetryValue"),
)


def _digest(payload: dict[str, JsonValue]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest().upper()


DECLARED_CAUSAL_GRAPH_DIGEST = _digest(
    {
        "graph_id": DECLARED_CAUSAL_GRAPH_ID,
        "nodes": list(DECLARED_CAUSAL_NODES),
        "edges": [list(edge) for edge in DECLARED_CAUSAL_EDGES],
    }
)


def _keep_action(item: BaselineInput, baseline_id: BaselineId) -> NetworkAction:
    return NetworkAction(
        action_id=f"{baseline_id.value.lower()}:keep:{item.pair_id}",
        kind=ActionKind.KEEP_CURRENT_PATH,
        path_id=None,
    )


def _details(**values: object) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, str(value)) for name, value in values.items()))


def _result(
    *,
    baseline_id: BaselineId,
    item: BaselineInput,
    disposition: BaselineDisposition,
    reason: BaselineReason,
    selected_action: NetworkAction | None,
    fields: tuple[str, ...],
    telemetry_items: int,
    telemetry_bytes: int,
    started_ns: int,
    configuration_digest: str,
    score: float | None = None,
    details: tuple[tuple[str, str], ...] = (),
) -> BaselineResult:
    return BaselineResult(
        baseline_id=baseline_id,
        pair_id=item.pair_id,
        scenario_id=item.scenario_id,
        disposition=disposition,
        reason=reason,
        proposed_action=item.proposed_action,
        selected_action=selected_action,
        observation_fields=fields,
        telemetry_items=telemetry_items,
        telemetry_bytes=telemetry_bytes,
        inference_time_ns=perf_counter_ns() - started_ns,
        configuration_digest=configuration_digest,
        score=score,
        details=details,
    )


def _require_training_split(examples: tuple[TrainingExample, ...]) -> None:
    if not examples:
        raise ValueError("at least one training example is required")
    if any(example.input.split is not DatasetSplit.TRAIN for example in examples):
        raise ValueError("model fitting is restricted to the training split")


def _require_development_split(examples: tuple[TrainingExample, ...]) -> None:
    if not examples:
        raise ValueError("at least one development example is required")
    if any(example.input.split is not DatasetSplit.DEVELOPMENT for example in examples):
        raise ValueError("threshold tuning is restricted to the development split")


@dataclass(frozen=True, slots=True)
class OracleChoice:
    pair_id: str
    reroute: bool
    keep_objective: float
    reroute_objective: float
    oracle_rows: int

    def __post_init__(self) -> None:
        if not self.pair_id or self.oracle_rows <= 0:
            raise ValueError("oracle choice requires a pair ID and full-state rows")
        if not math.isfinite(self.keep_objective) or not math.isfinite(self.reroute_objective):
            raise ValueError("oracle objectives must be finite")


@dataclass(frozen=True, slots=True)
class OracleController:
    choices: tuple[OracleChoice, ...]
    loss_weight: float = 1000.0
    jitter_weight: float = 1.0
    deadline_weight: float = 1000.0
    baseline_id: BaselineId = field(default=BaselineId.B0, init=False)
    observation_policy: ObservationPolicy = field(default=ObservationPolicy((), 0, 0), init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not self.choices or len({choice.pair_id for choice in self.choices}) != len(
            self.choices
        ):
            raise ValueError("oracle choices must be nonempty and pair-unique")
        if min(self.loss_weight, self.jitter_weight, self.deadline_weight) < 0:
            raise ValueError("oracle objective weights cannot be negative")
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": self.baseline_id.value,
                    "loss_weight": self.loss_weight,
                    "jitter_weight": self.jitter_weight,
                    "deadline_weight": self.deadline_weight,
                }
            ),
        )

    @classmethod
    def from_corpus(
        cls,
        directories: tuple[Path, ...],
        *,
        split: DatasetSplit,
        guard: CorpusAccessGuard | None = None,
        loss_weight: float = 1000.0,
        jitter_weight: float = 1.0,
        deadline_weight: float = 1000.0,
    ) -> OracleController:
        access = guard or CorpusAccessGuard()
        choices: list[OracleChoice] = []
        for directory in sorted(directories):
            item = load_baseline_input(
                directory,
                baseline_id=BaselineId.B0,
                split=split,
                guard=access,
            )
            oracle_rows = access.read_table(
                directory,
                baseline_id=BaselineId.B0,
                phase=AccessPhase.DECISION,
                split=split,
                table=CorpusTable.ORACLE,
                columns=["run_id", "tick", "path_id"],
            ).num_rows
            outcomes = load_outcomes(
                directory,
                item=item,
                baseline_id=BaselineId.B0,
                phase=AccessPhase.DECISION,
                guard=access,
            )
            keep = outcomes.keep.objective(
                loss_weight=loss_weight,
                jitter_weight=jitter_weight,
                deadline_weight=deadline_weight,
            )
            reroute = outcomes.reroute.objective(
                loss_weight=loss_weight,
                jitter_weight=jitter_weight,
                deadline_weight=deadline_weight,
            )
            choices.append(OracleChoice(item.pair_id, reroute < keep, keep, reroute, oracle_rows))
        return cls(
            tuple(choices),
            loss_weight=loss_weight,
            jitter_weight=jitter_weight,
            deadline_weight=deadline_weight,
        )

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        choice = next((value for value in self.choices if value.pair_id == item.pair_id), None)
        if choice is None:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.FAIL,
                reason=BaselineReason.INVALID_INPUT,
                selected_action=None,
                fields=(),
                telemetry_items=0,
                telemetry_bytes=0,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        selected = item.proposed_action if choice.reroute else _keep_action(item, self.baseline_id)
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=BaselineDisposition.EXECUTE,
            reason=BaselineReason.ORACLE_BEST_ACTION,
            selected_action=selected,
            fields=(),
            telemetry_items=0,
            telemetry_bytes=0,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=choice.keep_objective - choice.reroute_objective,
            details=_details(oracle_rows=choice.oracle_rows),
        )


@dataclass(frozen=True, slots=True)
class NoGateController:
    baseline_id: BaselineId = field(default=BaselineId.B1, init=False)
    observation_policy: ObservationPolicy = field(default=ObservationPolicy((), 0, 0), init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "configuration_digest", _digest({"baseline": "B1"}))

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=BaselineDisposition.EXECUTE,
            reason=BaselineReason.NO_GATE,
            selected_action=item.proposed_action,
            fields=(),
            telemetry_items=0,
            telemetry_bytes=0,
            started_ns=started,
            configuration_digest=self.configuration_digest,
        )


@dataclass(frozen=True, slots=True)
class FixedTelemetryController:
    minimum_improvement: float = 0.0
    utilization_weight: float = 10.0
    baseline_id: BaselineId = field(default=BaselineId.B2, init=False)
    observation_policy: ObservationPolicy = field(
        default=ObservationPolicy(("queue_occupancy", "utilization"), 4, 32), init=False
    )
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_improvement) or self.utilization_weight < 0:
            raise ValueError("fixed-telemetry thresholds must be finite and nonnegative")
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B2",
                    "minimum_improvement": self.minimum_improvement,
                    "utilization_weight": self.utilization_weight,
                }
            ),
        )

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.observation_policy.fields)
        if not features.complete:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=self.observation_policy.fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        current = item.snapshot(item.current_path_id)
        candidate = item.snapshot(item.candidate_path_id)
        assert current is not None and candidate is not None
        score = (current.queue_occupancy - candidate.queue_occupancy) + self.utilization_weight * (
            current.utilization - candidate.utilization
        )
        accepted = score >= self.minimum_improvement
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.FIXED_TELEMETRY_ACCEPTED
                if accepted
                else BaselineReason.PREDICTED_HARM
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=self.observation_policy.fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=score,
        )


@dataclass(slots=True)
class AdaptiveCoverageController:
    field_weights: tuple[tuple[str, float], ...] = (
        ("queue_occupancy", 1.0),
        ("utilization", 0.9),
        ("sample_age_ticks", 0.5),
        ("measurement_delay_ticks", 0.4),
        ("clock_skew_ticks", 0.3),
    )
    maximum_selected_fields: int = 2
    freshness_penalty: float = 0.25
    switching_penalty: float = 0.1
    minimum_score: float = 0.0
    previous_fields: tuple[str, ...] = field(default=(), init=False)
    baseline_id: BaselineId = field(default=BaselineId.B3, init=False)
    observation_policy: ObservationPolicy = field(init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        names = tuple(name for name, _weight in self.field_weights)
        if not names or len(names) != len(set(names)):
            raise ValueError("adaptive field weights must be nonempty and unique")
        if set(names) - set(NUMERIC_OBSERVATION_FIELDS):
            raise ValueError("adaptive controller contains unsupported fields")
        if any(not math.isfinite(weight) or weight < 0 for _name, weight in self.field_weights):
            raise ValueError("adaptive field weights must be finite and nonnegative")
        if not 1 <= self.maximum_selected_fields <= len(names):
            raise ValueError("adaptive field budget is outside the candidate set")
        if min(self.freshness_penalty, self.switching_penalty) < 0:
            raise ValueError("adaptive penalties cannot be negative")
        self.observation_policy = ObservationPolicy(
            names,
            self.maximum_selected_fields * 2,
            self.maximum_selected_fields * 16,
            adaptive=True,
        )
        self.configuration_digest = _digest(
            {
                "baseline": "B3",
                "field_weights": {name: weight for name, weight in self.field_weights},
                "maximum_selected_fields": self.maximum_selected_fields,
                "freshness_penalty": self.freshness_penalty,
                "switching_penalty": self.switching_penalty,
                "minimum_score": self.minimum_score,
            }
        )

    def select_fields(self, item: BaselineInput) -> tuple[str, ...]:
        current = item.snapshot(item.current_path_id)
        candidate = item.snapshot(item.candidate_path_id)
        utilities: list[tuple[float, str]] = []
        for name, weight in self.field_weights:
            spread = 0.0
            freshness = 1.0
            if current is not None and candidate is not None:
                first = current.numeric(name)
                second = candidate.numeric(name)
                spread = abs(first - second) / (1 + abs(first) + abs(second))
                freshness = (current.sample_age_ticks + candidate.sample_age_ticks) / (
                    2 + current.sample_age_ticks + candidate.sample_age_ticks
                )
            switch_cost = (
                self.switching_penalty
                if self.previous_fields and name not in self.previous_fields
                else 0.0
            )
            utilities.append(
                (weight + spread - self.freshness_penalty * freshness - switch_cost, name)
            )
        utilities.sort(key=lambda item_: (-item_[0], item_[1]))
        return tuple(name for _utility, name in utilities[: self.maximum_selected_fields])

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        selected_fields = self.select_fields(item)
        self.previous_fields = selected_fields
        features = extract_features(item, selected_fields)
        if not features.complete:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=selected_fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        current = item.snapshot(item.current_path_id)
        candidate = item.snapshot(item.candidate_path_id)
        assert current is not None and candidate is not None
        weights = dict(self.field_weights)
        contributions: list[float] = []
        for name in selected_fields:
            first = current.numeric(name)
            second = candidate.numeric(name)
            contributions.append(weights[name] * (first - second) / (1 + abs(first) + abs(second)))
        score = sum(contributions) / sum(weights[name] for name in selected_fields)
        accepted = score >= self.minimum_score
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.ADAPTIVE_UTILITY_ACCEPTED
                if accepted
                else BaselineReason.PREDICTED_HARM
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=selected_fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class LinearClassifier:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        lengths = {len(self.mean), len(self.scale), len(self.coefficients)}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("linear classifier parameter lengths must match and be nonzero")
        if any(value <= 0 or not math.isfinite(value) for value in self.scale):
            raise ValueError("linear classifier scales must be finite and positive")
        if not all(
            math.isfinite(value) for value in (*self.mean, *self.coefficients, self.intercept)
        ):
            raise ValueError("linear classifier parameters must be finite")

    def probability(self, values: tuple[float, ...]) -> float:
        if len(values) != len(self.mean):
            raise ValueError("classifier feature width does not match fitted parameters")
        logit = self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                values, self.mean, self.scale, self.coefficients, strict=True
            )
        )
        if logit >= 0:
            return 1 / (1 + math.exp(-logit))
        exp_logit = math.exp(logit)
        return exp_logit / (1 + exp_logit)


def _fit_classifier(
    examples: tuple[TrainingExample, ...], fields: tuple[str, ...], *, regularization_c: float
) -> LinearClassifier:
    _require_training_split(examples)
    labels = np.asarray(
        [example.outcomes.reroute_beneficial for example in examples], dtype=np.int64
    )
    if len(np.unique(labels)) != 2:
        raise ValueError("confidence fitting requires both beneficial and harmful reroutes")
    matrix = np.asarray(
        [extract_features(example.input, fields).values for example in examples], dtype=np.float64
    )
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0)
    scale[scale == 0] = 1.0
    standardized = (matrix - mean) / scale
    model = LogisticRegression(
        C=regularization_c,
        solver="liblinear",
        random_state=0,
        max_iter=1000,
    ).fit(standardized, labels)
    coefficients = cast(NDArray[np.float64], model.coef_)[0]
    intercept = float(cast(NDArray[np.float64], model.intercept_)[0])
    return LinearClassifier(
        tuple(float(value) for value in mean),
        tuple(float(value) for value in scale),
        tuple(float(value) for value in coefficients),
        intercept,
    )


@dataclass(frozen=True, slots=True)
class SelectiveConfidenceController:
    model: LinearClassifier
    fields: tuple[str, ...]
    confidence_threshold: float = 0.75
    regularization_c: float = 1.0
    baseline_id: BaselineId = field(default=BaselineId.B4, init=False)
    observation_policy: ObservationPolicy = field(init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not 0.5 <= self.confidence_threshold <= 1:
            raise ValueError("confidence threshold must lie in [0.5, 1]")
        if self.regularization_c <= 0:
            raise ValueError("logistic regularization C must be positive")
        width = len(extract_feature_names(self.fields))
        if width != len(self.model.mean):
            raise ValueError("confidence model width does not match its fields")
        object.__setattr__(
            self,
            "observation_policy",
            ObservationPolicy(self.fields, len(self.fields) * 2, len(self.fields) * 16),
        )
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B4",
                    "fields": list(self.fields),
                    "confidence_threshold": self.confidence_threshold,
                    "regularization_c": self.regularization_c,
                    "mean": list(self.model.mean),
                    "scale": list(self.model.scale),
                    "coefficients": list(self.model.coefficients),
                    "intercept": self.model.intercept,
                }
            ),
        )

    @classmethod
    def fit(
        cls,
        examples: tuple[TrainingExample, ...],
        *,
        fields: tuple[str, ...] = ("queue_occupancy", "utilization"),
        confidence_threshold: float = 0.75,
        regularization_c: float = 1.0,
    ) -> SelectiveConfidenceController:
        return cls(
            _fit_classifier(examples, fields, regularization_c=regularization_c),
            fields,
            confidence_threshold,
            regularization_c,
        )

    def probability(self, item: BaselineInput) -> float:
        return self.model.probability(extract_features(item, self.fields).values)

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.fields)
        if not features.complete:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=self.fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        probability = self.model.probability(features.values)
        confidence = max(probability, 1 - probability)
        accepted = probability >= 0.5 and confidence >= self.confidence_threshold
        reason = (
            BaselineReason.CONFIDENCE_ACCEPTED
            if accepted
            else BaselineReason.PREDICTED_HARM
            if probability < 0.5
            else BaselineReason.LOW_CONFIDENCE
        )
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=reason,
            selected_action=item.proposed_action if accepted else None,
            fields=self.fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=probability,
            details=_details(confidence=confidence),
        )


def tune_confidence_threshold(
    controller: SelectiveConfidenceController,
    examples: tuple[TrainingExample, ...],
    *,
    candidates: tuple[float, ...] = CONFIDENCE_THRESHOLD_CANDIDATES,
    maximum_selective_error: float = MAXIMUM_SELECTIVE_ERROR,
) -> SelectiveConfidenceController:
    _require_development_split(examples)
    if not candidates or any(not 0.5 <= value <= 1 for value in candidates):
        raise ValueError("candidate confidence thresholds must lie in [0.5, 1]")
    if not 0 <= maximum_selective_error <= 1:
        raise ValueError("maximum selective error must lie in [0, 1]")
    probabilities = tuple(controller.probability(example.input) for example in examples)
    labels = tuple(example.outcomes.reroute_beneficial for example in examples)
    feasible: list[tuple[int, float]] = []
    for threshold in sorted(set(candidates)):
        selected = [
            index
            for index, probability in enumerate(probabilities)
            if probability >= 0.5 and max(probability, 1 - probability) >= threshold
        ]
        risk = sum(not labels[index] for index in selected) / len(selected) if selected else 0.0
        if risk <= maximum_selective_error:
            feasible.append((len(selected), threshold))
    if not feasible:
        raise ValueError("no confidence threshold satisfies the development risk bound")
    _coverage, threshold = max(feasible, key=lambda value: (value[0], -value[1]))
    return replace(controller, confidence_threshold=threshold)


@dataclass(frozen=True, slots=True)
class FixedCausalController:
    identified: bool
    point_effect_ms: float | None
    upper_effect_ms: float | None
    sample_size: int
    graph_id: str = DECLARED_CAUSAL_GRAPH_ID
    graph_digest: str = DECLARED_CAUSAL_GRAPH_DIGEST
    adjustment_set: tuple[str, ...] = ()
    baseline_id: BaselineId = field(default=BaselineId.B5, init=False)
    observation_policy: ObservationPolicy = field(default=ObservationPolicy((), 0, 0), init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError("causal sample size cannot be negative")
        if self.identified != (
            self.point_effect_ms is not None and self.upper_effect_ms is not None
        ):
            raise ValueError("causal identification and estimate presence must agree")
        if not self.graph_id or len(self.graph_digest) != 64:
            raise ValueError("causal graph ID and digest are required")
        if self.adjustment_set != tuple(sorted(set(self.adjustment_set))):
            raise ValueError("causal adjustment set must be unique and canonical")
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B5",
                    "identified": self.identified,
                    "point_effect_ms": self.point_effect_ms,
                    "upper_effect_ms": self.upper_effect_ms,
                    "sample_size": self.sample_size,
                    "graph_id": self.graph_id,
                    "graph_digest": self.graph_digest,
                    "adjustment_set": list(self.adjustment_set),
                }
            ),
        )

    @classmethod
    def fit(cls, examples: tuple[TrainingExample, ...]) -> FixedCausalController:
        _require_training_split(examples)
        graph = CausalDAG(
            DECLARED_CAUSAL_NODES,
            DECLARED_CAUSAL_EDGES,
            DECLARED_CAUSAL_NODES,
        )
        identification = BackdoorIdentifier(graph).identify(
            "RouteAction", "EndToEndDelay", ("RouteAction", "EndToEndDelay")
        )
        adjustment_set: tuple[str, ...] = ()
        observations = tuple(
            row
            for example in examples
            for row in (
                AdjustmentObservation.from_mapping(
                    treatment=ActionKind.KEEP_CURRENT_PATH,
                    outcome=example.outcomes.keep.mean_latency_ms,
                ),
                AdjustmentObservation.from_mapping(
                    treatment=ActionKind.REROUTE_TO_PATH,
                    outcome=example.outcomes.reroute.mean_latency_ms,
                ),
            )
        )
        estimate = estimate_stratified_backdoor(
            observations,
            identification,
            adjustment_set,
            estimand="reroute-minus-keep latency",
            unit="ms",
        )
        if not estimate.estimable or estimate.estimate is None:
            return cls(
                False,
                None,
                None,
                estimate.sample_size,
                DECLARED_CAUSAL_GRAPH_ID,
                DECLARED_CAUSAL_GRAPH_DIGEST,
                adjustment_set,
            )
        return cls(
            True,
            estimate.estimate.point_estimate,
            estimate.estimate.uncertainty.upper,
            estimate.sample_size,
            DECLARED_CAUSAL_GRAPH_ID,
            DECLARED_CAUSAL_GRAPH_DIGEST,
            adjustment_set,
        )

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        accepted = self.identified and cast(float, self.upper_effect_ms) < 0
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.CAUSAL_EFFECT_SUPPORTED
                if accepted
                else BaselineReason.CAUSAL_EFFECT_UNSUPPORTED
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=(),
            telemetry_items=0,
            telemetry_bytes=0,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=-self.upper_effect_ms if self.upper_effect_ms is not None else None,
            details=_details(
                adjustment_set=",".join(self.adjustment_set),
                graph_id=self.graph_id,
                sample_size=self.sample_size,
            ),
        )


@dataclass(frozen=True, slots=True)
class ConstraintOnlyController:
    peak_utilization_limit: float = 0.95
    average_utilization_limit: float = 0.85
    queue_limit: int = 64
    baseline_id: BaselineId = field(default=BaselineId.B6, init=False)
    observation_policy: ObservationPolicy = field(
        default=ObservationPolicy(("queue_occupancy", "utilization"), 4, 32), init=False
    )
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not 0 <= self.average_utilization_limit <= self.peak_utilization_limit:
            raise ValueError("average utilization limit cannot exceed the peak limit")
        if self.peak_utilization_limit <= 0 or self.queue_limit < 0:
            raise ValueError("constraint limits must be positive or zero as applicable")
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B6",
                    "peak_utilization_limit": self.peak_utilization_limit,
                    "average_utilization_limit": self.average_utilization_limit,
                    "queue_limit": self.queue_limit,
                }
            ),
        )

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.observation_policy.fields)
        current = item.snapshot(item.current_path_id)
        candidate = item.snapshot(item.candidate_path_id)
        if current is None or candidate is None:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=self.observation_policy.fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        average = (current.utilization + candidate.utilization) / 2
        accepted = (
            candidate.utilization <= self.peak_utilization_limit
            and average <= self.average_utilization_limit
            and candidate.queue_occupancy <= self.queue_limit
        )
        margin = min(
            self.peak_utilization_limit - candidate.utilization,
            self.average_utilization_limit - average,
            float(self.queue_limit - candidate.queue_occupancy),
        )
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.CONSTRAINTS_SATISFIED
                if accepted
                else BaselineReason.CONSTRAINT_VIOLATION
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=self.observation_policy.fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=margin,
            details=_details(average_utilization=average),
        )


@dataclass(frozen=True, slots=True)
class StaticContractController:
    required_fields: tuple[str, ...] = ("queue_occupancy", "utilization", "sample_age_ticks")
    maximum_age_ticks: int = 0
    baseline_id: BaselineId = field(default=BaselineId.B7, init=False)
    observation_policy: ObservationPolicy = field(init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.maximum_age_ticks < 0:
            raise ValueError("static-contract age limit cannot be negative")
        object.__setattr__(
            self,
            "observation_policy",
            ObservationPolicy(
                self.required_fields,
                len(self.required_fields) * 2,
                len(self.required_fields) * 16,
            ),
        )
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B7",
                    "required_fields": list(self.required_fields),
                    "maximum_age_ticks": self.maximum_age_ticks,
                }
            ),
        )

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.required_fields)
        current = item.snapshot(item.current_path_id)
        candidate = item.snapshot(item.candidate_path_id)
        accepted = (
            current is not None
            and candidate is not None
            and not current.stale
            and not candidate.stale
            and current.sample_age_ticks <= self.maximum_age_ticks
            and candidate.sample_age_ticks <= self.maximum_age_ticks
        )
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.STATIC_CONTRACT_SATISFIED
                if accepted
                else BaselineReason.STATIC_CONTRACT_UNAVAILABLE
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=self.required_fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
        )


@dataclass(frozen=True, slots=True)
class LinearRegressor:
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float

    def __post_init__(self) -> None:
        lengths = {len(self.mean), len(self.scale), len(self.coefficients)}
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("linear regressor parameter lengths must match and be nonzero")
        if any(value <= 0 or not math.isfinite(value) for value in self.scale):
            raise ValueError("linear regressor scales must be finite and positive")
        if not all(
            math.isfinite(value) for value in (*self.mean, *self.coefficients, self.intercept)
        ):
            raise ValueError("linear regressor parameters must be finite")

    def predict(self, values: tuple[float, ...]) -> float:
        if len(values) != len(self.mean):
            raise ValueError("regressor feature width does not match fitted parameters")
        return self.intercept + sum(
            coefficient * ((value - mean) / scale)
            for value, mean, scale, coefficient in zip(
                values, self.mean, self.scale, self.coefficients, strict=True
            )
        )


def extract_feature_names(fields: tuple[str, ...]) -> tuple[str, ...]:
    names: list[str] = []
    for field_name in fields:
        names.extend(
            (
                f"current.{field_name}",
                f"candidate.{field_name}",
                f"candidate_minus_current.{field_name}",
                f"both_present.{field_name}",
            )
        )
    return tuple(names)


@dataclass(frozen=True, slots=True)
class DigitalTwinController:
    model: LinearRegressor
    fields: tuple[str, ...]
    ridge_alpha: float = 1.0
    baseline_id: BaselineId = field(default=BaselineId.B8, init=False)
    observation_policy: ObservationPolicy = field(init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.ridge_alpha < 0:
            raise ValueError("ridge alpha cannot be negative")
        if len(self.model.mean) != len(extract_feature_names(self.fields)) + 1:
            raise ValueError("digital-twin model width does not match its fields")
        object.__setattr__(
            self,
            "observation_policy",
            ObservationPolicy(self.fields, len(self.fields) * 2, len(self.fields) * 16),
        )
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B8",
                    "fields": list(self.fields),
                    "ridge_alpha": self.ridge_alpha,
                    "mean": list(self.model.mean),
                    "scale": list(self.model.scale),
                    "coefficients": list(self.model.coefficients),
                    "intercept": self.model.intercept,
                }
            ),
        )

    @classmethod
    def fit(
        cls,
        examples: tuple[TrainingExample, ...],
        *,
        fields: tuple[str, ...] = ("queue_occupancy", "utilization"),
        ridge_alpha: float = 1.0,
    ) -> DigitalTwinController:
        _require_training_split(examples)
        rows: list[tuple[float, ...]] = []
        targets: list[float] = []
        for example in examples:
            base = extract_features(example.input, fields).values
            rows.extend(((*base, 0.0), (*base, 1.0)))
            targets.extend(
                (
                    example.outcomes.keep.mean_latency_ms,
                    example.outcomes.reroute.mean_latency_ms,
                )
            )
        matrix = np.asarray(rows, dtype=np.float64)
        target = np.asarray(targets, dtype=np.float64)
        mean = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
        scale[scale == 0] = 1.0
        model = Ridge(alpha=ridge_alpha).fit((matrix - mean) / scale, target)
        coefficients = cast(NDArray[np.float64], model.coef_)
        regressor = LinearRegressor(
            tuple(float(value) for value in mean),
            tuple(float(value) for value in scale),
            tuple(float(value) for value in coefficients),
            float(model.intercept_),
        )
        return cls(regressor, fields, ridge_alpha)

    def predict_pair(self, item: BaselineInput) -> tuple[float, float]:
        base = extract_features(item, self.fields).values
        return self.model.predict((*base, 0.0)), self.model.predict((*base, 1.0))

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.fields)
        if not features.complete:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=self.fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        keep, reroute = self.predict_pair(item)
        selected = item.proposed_action if reroute < keep else _keep_action(item, self.baseline_id)
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=BaselineDisposition.EXECUTE,
            reason=BaselineReason.TWIN_BEST_ACTION,
            selected_action=selected,
            fields=self.fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=keep - reroute,
            details=_details(predicted_keep_ms=keep, predicted_reroute_ms=reroute),
        )


def tune_digital_twin_alpha(
    training: tuple[TrainingExample, ...],
    development: tuple[TrainingExample, ...],
    *,
    candidates: tuple[float, ...] = DIGITAL_TWIN_ALPHA_CANDIDATES,
    fields: tuple[str, ...] = ("queue_occupancy", "utilization"),
) -> DigitalTwinController:
    _require_training_split(training)
    _require_development_split(development)
    if not candidates or any(value < 0 or not math.isfinite(value) for value in candidates):
        raise ValueError("digital-twin alpha candidates must be finite and nonnegative")
    scored: list[tuple[float, float, DigitalTwinController]] = []
    for alpha in sorted(set(candidates)):
        controller = DigitalTwinController.fit(training, fields=fields, ridge_alpha=alpha)
        absolute_errors: list[float] = []
        for example in development:
            keep, reroute = controller.predict_pair(example.input)
            absolute_errors.extend(
                (
                    abs(keep - example.outcomes.keep.mean_latency_ms),
                    abs(reroute - example.outcomes.reroute.mean_latency_ms),
                )
            )
        scored.append((sum(absolute_errors) / len(absolute_errors), alpha, controller))
    return min(scored, key=lambda item: (item[0], item[1]))[2]


@dataclass(frozen=True, slots=True)
class CorrelationSelectorController:
    model: LinearClassifier
    selected_fields: tuple[str, ...]
    mutual_information: tuple[tuple[str, float], ...]
    top_k: int
    baseline_id: BaselineId = field(default=BaselineId.B9, init=False)
    observation_policy: ObservationPolicy = field(init=False)
    configuration_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.top_k != len(self.selected_fields) or self.top_k <= 0:
            raise ValueError("feature-selector top-k and selected fields must agree")
        if tuple(sorted(name for name, _score in self.mutual_information)) != tuple(
            sorted(NUMERIC_OBSERVATION_FIELDS)
        ):
            raise ValueError("mutual-information audit must cover every candidate field")
        if len(self.model.mean) != len(extract_feature_names(self.selected_fields)):
            raise ValueError("feature-selector model width does not match selected fields")
        object.__setattr__(
            self,
            "observation_policy",
            ObservationPolicy(self.selected_fields, self.top_k * 2, self.top_k * 16),
        )
        object.__setattr__(
            self,
            "configuration_digest",
            _digest(
                {
                    "baseline": "B9",
                    "selected_fields": list(self.selected_fields),
                    "mutual_information": {name: score for name, score in self.mutual_information},
                    "top_k": self.top_k,
                    "mean": list(self.model.mean),
                    "scale": list(self.model.scale),
                    "coefficients": list(self.model.coefficients),
                    "intercept": self.model.intercept,
                }
            ),
        )

    @classmethod
    def fit(
        cls,
        examples: tuple[TrainingExample, ...],
        *,
        top_k: int = 2,
        regularization_c: float = 1.0,
    ) -> CorrelationSelectorController:
        _require_training_split(examples)
        candidates = NUMERIC_OBSERVATION_FIELDS
        if not 1 <= top_k <= len(candidates):
            raise ValueError("top-k is outside the feature candidate set")
        matrix = np.asarray(
            [
                tuple(
                    extract_features(example.input, (field_name,)).values[2]
                    for field_name in candidates
                )
                for example in examples
            ],
            dtype=np.float64,
        )
        labels = np.asarray(
            [example.outcomes.reroute_beneficial for example in examples], dtype=np.int64
        )
        if len(np.unique(labels)) != 2:
            raise ValueError("mutual-information selection requires two outcome classes")
        scores = mutual_info_classif(matrix, labels, random_state=0, discrete_features=False)
        audit = tuple((name, float(score)) for name, score in zip(candidates, scores, strict=True))
        selected = tuple(
            name for name, _score in sorted(audit, key=lambda item: (-item[1], item[0]))[:top_k]
        )
        model = _fit_classifier(examples, selected, regularization_c=regularization_c)
        return cls(model, selected, audit, top_k)

    def decide(self, item: BaselineInput) -> BaselineResult:
        started = perf_counter_ns()
        features = extract_features(item, self.selected_fields)
        if not features.complete:
            return _result(
                baseline_id=self.baseline_id,
                item=item,
                disposition=BaselineDisposition.ABSTAIN,
                reason=BaselineReason.REQUIRED_OBSERVATION_MISSING,
                selected_action=None,
                fields=self.selected_fields,
                telemetry_items=features.telemetry_items,
                telemetry_bytes=features.telemetry_bytes,
                started_ns=started,
                configuration_digest=self.configuration_digest,
            )
        probability = self.model.probability(features.values)
        accepted = probability >= 0.5
        return _result(
            baseline_id=self.baseline_id,
            item=item,
            disposition=(BaselineDisposition.EXECUTE if accepted else BaselineDisposition.ABSTAIN),
            reason=(
                BaselineReason.FEATURE_SELECTOR_ACCEPTED
                if accepted
                else BaselineReason.PREDICTED_HARM
            ),
            selected_action=item.proposed_action if accepted else None,
            fields=self.selected_fields,
            telemetry_items=features.telemetry_items,
            telemetry_bytes=features.telemetry_bytes,
            started_ns=started,
            configuration_digest=self.configuration_digest,
            score=probability,
        )


def tune_correlation_top_k(
    training: tuple[TrainingExample, ...],
    development: tuple[TrainingExample, ...],
    *,
    candidates: tuple[int, ...] = CORRELATION_TOP_K_CANDIDATES,
    regularization_c: float = 1.0,
) -> CorrelationSelectorController:
    _require_training_split(training)
    _require_development_split(development)
    if not candidates or any(
        not 1 <= value <= len(NUMERIC_OBSERVATION_FIELDS) for value in candidates
    ):
        raise ValueError("feature-selector top-k candidates are outside the candidate set")
    scored: list[tuple[int, int, CorrelationSelectorController]] = []
    for top_k in sorted(set(candidates)):
        controller = CorrelationSelectorController.fit(
            training,
            top_k=top_k,
            regularization_c=regularization_c,
        )
        correct = 0
        for example in development:
            probability = controller.model.probability(
                extract_features(example.input, controller.selected_fields).values
            )
            correct += (probability >= 0.5) is example.outcomes.reroute_beneficial
        scored.append((correct, top_k, controller))
    return max(scored, key=lambda item: (item[0], -item[1]))[2]
