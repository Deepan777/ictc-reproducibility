"""Frozen B0-B9 suite construction, resource measurement, and common output schema."""

from __future__ import annotations

import json
import os
import tempfile
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns, process_time_ns
from typing import cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ictc.baselines.data import TrainingExample
from ictc.baselines.methods import (
    AdaptiveCoverageController,
    ConstraintOnlyController,
    CorrelationSelectorController,
    DigitalTwinController,
    FixedCausalController,
    FixedTelemetryController,
    NoGateController,
    OracleController,
    SelectiveConfidenceController,
    StaticContractController,
    tune_confidence_threshold,
    tune_correlation_top_k,
    tune_digital_twin_alpha,
)
from ictc.baselines.types import BaselineController, BaselineId, BaselineInput, BaselineResult
from ictc.datasets.m08_config import DatasetSplit

BASELINE_OUTPUT_SCHEMA_VERSION = "1.0.0"
BASELINE_OUTPUT_SCHEMA = pa.schema(
    [
        ("schema_version", pa.string()),
        ("pair_id", pa.string()),
        ("scenario_id", pa.string()),
        ("split", pa.string()),
        ("seed", pa.int64()),
        ("baseline_id", pa.string()),
        ("disposition", pa.string()),
        ("reason", pa.string()),
        ("proposed_action_kind", pa.string()),
        ("selected_action_kind", pa.string()),
        ("selected_path_id", pa.string()),
        ("observation_fields_json", pa.string()),
        ("telemetry_items", pa.int64()),
        ("telemetry_bytes", pa.int64()),
        ("inference_time_ns", pa.int64()),
        ("cpu_time_ns", pa.int64()),
        ("peak_python_bytes", pa.int64()),
        ("gpu_used", pa.bool_()),
        ("deadline_ns", pa.int64()),
        ("deadline_exceeded", pa.bool_()),
        ("parameter_count", pa.int64()),
        ("training_time_ns", pa.int64()),
        ("model_storage_bytes", pa.int64()),
        ("configuration_digest", pa.string()),
        ("score", pa.float64()),
        ("details_json", pa.string()),
    ]
)


@dataclass(frozen=True, slots=True)
class TrainingCost:
    baseline_id: BaselineId
    training_time_ns: int
    parameter_count: int
    model_storage_bytes: int
    cpu_only: bool = True

    def __post_init__(self) -> None:
        if min(self.training_time_ns, self.parameter_count, self.model_storage_bytes) < 0:
            raise ValueError("baseline training costs cannot be negative")
        if not self.cpu_only:
            raise ValueError("M09 frozen baselines are CPU-only")


@dataclass(frozen=True, slots=True)
class BaselineRunRecord:
    result: BaselineResult
    split: DatasetSplit
    seed: int
    cpu_time_ns: int
    peak_python_bytes: int
    gpu_used: bool
    deadline_ns: int
    training_cost: TrainingCost

    def __post_init__(self) -> None:
        if self.result.baseline_id is not self.training_cost.baseline_id:
            raise ValueError("run result and training cost refer to different baselines")
        if self.seed < 0 or min(self.cpu_time_ns, self.peak_python_bytes) < 0:
            raise ValueError("run resource observations cannot be negative")
        if self.deadline_ns <= 0:
            raise ValueError("baseline deadline must be positive")
        if self.gpu_used:
            raise ValueError("M09 baseline run unexpectedly used a GPU")

    @property
    def deadline_exceeded(self) -> bool:
        return self.result.inference_time_ns > self.deadline_ns

    def to_row(self) -> dict[str, object]:
        selected = self.result.selected_action
        return {
            "schema_version": BASELINE_OUTPUT_SCHEMA_VERSION,
            "pair_id": self.result.pair_id,
            "scenario_id": self.result.scenario_id,
            "split": self.split.value,
            "seed": self.seed,
            "baseline_id": self.result.baseline_id.value,
            "disposition": self.result.disposition.value,
            "reason": self.result.reason.value,
            "proposed_action_kind": self.result.proposed_action.kind.value,
            "selected_action_kind": selected.kind.value if selected is not None else None,
            "selected_path_id": selected.path_id if selected is not None else None,
            "observation_fields_json": json.dumps(
                self.result.observation_fields, separators=(",", ":")
            ),
            "telemetry_items": self.result.telemetry_items,
            "telemetry_bytes": self.result.telemetry_bytes,
            "inference_time_ns": self.result.inference_time_ns,
            "cpu_time_ns": self.cpu_time_ns,
            "peak_python_bytes": self.peak_python_bytes,
            "gpu_used": self.gpu_used,
            "deadline_ns": self.deadline_ns,
            "deadline_exceeded": self.deadline_exceeded,
            "parameter_count": self.training_cost.parameter_count,
            "training_time_ns": self.training_cost.training_time_ns,
            "model_storage_bytes": self.training_cost.model_storage_bytes,
            "configuration_digest": self.result.configuration_digest,
            "score": self.result.score,
            "details_json": json.dumps(
                dict(self.result.details), sort_keys=True, separators=(",", ":")
            ),
        }


def _parameter_count(controller: BaselineController) -> int:
    if isinstance(
        controller,
        SelectiveConfidenceController | DigitalTwinController | CorrelationSelectorController,
    ):
        return len(controller.model.coefficients) + 1
    return 0


def _training_cost(controller: BaselineController, training_time_ns: int = 0) -> TrainingCost:
    storage_value: object
    if isinstance(controller, OracleController):
        storage_value = (
            controller.loss_weight,
            controller.jitter_weight,
            controller.deadline_weight,
        )
    else:
        storage_value = controller
    return TrainingCost(
        controller.baseline_id,
        training_time_ns,
        _parameter_count(controller),
        len(repr(storage_value).encode("utf-8")),
    )


@dataclass(frozen=True, slots=True)
class BaselineSuite:
    controllers: tuple[BaselineController, ...]
    training_costs: tuple[TrainingCost, ...]
    deadline_ns: int = 50_000_000

    def __post_init__(self) -> None:
        controller_ids = tuple(controller.baseline_id for controller in self.controllers)
        cost_ids = tuple(cost.baseline_id for cost in self.training_costs)
        expected = tuple(BaselineId)
        if controller_ids != expected or cost_ids != expected:
            raise ValueError("baseline suite must contain ordered B0-B9 controllers and costs")
        if self.deadline_ns <= 0:
            raise ValueError("suite deadline must be positive")
        if len({controller.configuration_digest for controller in self.controllers}) != len(
            self.controllers
        ):
            raise ValueError("baseline configurations must have distinct digests")

    def run(self, item: BaselineInput) -> tuple[BaselineRunRecord, ...]:
        records: list[BaselineRunRecord] = []
        for controller, training_cost in zip(self.controllers, self.training_costs, strict=True):
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()
            else:
                tracemalloc.reset_peak()
            cpu_started = process_time_ns()
            try:
                result = controller.decide(item)
                cpu_time = process_time_ns() - cpu_started
                _current, peak = tracemalloc.get_traced_memory()
            finally:
                if not was_tracing:
                    tracemalloc.stop()
            if result.baseline_id is not controller.baseline_id:
                raise RuntimeError("baseline emitted another method's ID")
            if result.pair_id != item.pair_id or result.scenario_id != item.scenario_id:
                raise RuntimeError("baseline changed the paired scenario identity")
            policy = controller.observation_policy
            if (
                result.telemetry_items > policy.maximum_items
                or result.telemetry_bytes > policy.maximum_bytes
            ):
                raise RuntimeError("baseline exceeded its declared telemetry budget")
            records.append(
                BaselineRunRecord(
                    result,
                    item.split,
                    item.seed,
                    cpu_time,
                    peak,
                    False,
                    self.deadline_ns,
                    training_cost,
                )
            )
        return tuple(records)


def build_baseline_suite(
    *,
    oracle: OracleController,
    training: tuple[TrainingExample, ...],
    development: tuple[TrainingExample, ...],
    deadline_ns: int = 50_000_000,
) -> BaselineSuite:
    training_times: dict[BaselineId, int] = {}

    started = perf_counter_ns()
    confidence = SelectiveConfidenceController.fit(training, confidence_threshold=0.5)
    confidence = tune_confidence_threshold(confidence, development)
    training_times[BaselineId.B4] = perf_counter_ns() - started

    started = perf_counter_ns()
    causal = FixedCausalController.fit(training)
    training_times[BaselineId.B5] = perf_counter_ns() - started

    started = perf_counter_ns()
    twin = tune_digital_twin_alpha(training, development)
    training_times[BaselineId.B8] = perf_counter_ns() - started

    started = perf_counter_ns()
    selector = tune_correlation_top_k(training, development)
    training_times[BaselineId.B9] = perf_counter_ns() - started

    controllers = cast(
        tuple[BaselineController, ...],
        (
            oracle,
            NoGateController(),
            FixedTelemetryController(),
            AdaptiveCoverageController(),
            confidence,
            causal,
            ConstraintOnlyController(),
            StaticContractController(),
            twin,
            selector,
        ),
    )
    costs = tuple(
        _training_cost(controller, training_times.get(controller.baseline_id, 0))
        for controller in controllers
    )
    return BaselineSuite(controllers, costs, deadline_ns)


def write_baseline_results(records: tuple[BaselineRunRecord, ...], target: Path) -> None:
    if not records:
        raise ValueError("at least one baseline result is required")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable baseline output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = [record.to_row() for record in records]
    table = pa.Table.from_pylist(rows, schema=BASELINE_OUTPUT_SCHEMA)
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(
            table,
            temporary,
            compression=None,
            use_dictionary=False,
            write_statistics=True,
            version="2.6",
            data_page_version="1.0",
        )
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_baseline_results(target: Path) -> None:
    """Validate the common persisted schema and B0-B9 row completeness.

    Records are intentionally not reconstructed because the domain result contains nested
    actions. Returning ``None`` prevents this validator from becoming an alternate runtime
    input path for tested controllers.
    """
    if not target.is_file():
        raise FileNotFoundError(target)
    if pq.read_schema(target) != BASELINE_OUTPUT_SCHEMA:
        raise ValueError("baseline output schema does not match the frozen M09 schema")
    rows = cast(list[dict[str, object]], pq.read_table(target).to_pylist())
    if not rows:
        raise ValueError("baseline output is empty")
    by_pair: dict[str, set[str]] = {}
    for row in rows:
        if row.get("schema_version") != BASELINE_OUTPUT_SCHEMA_VERSION:
            raise ValueError("baseline output has an unsupported schema version")
        pair_id = row.get("pair_id")
        baseline_id = row.get("baseline_id")
        if not isinstance(pair_id, str) or not isinstance(baseline_id, str):
            raise ValueError("baseline output contains invalid identifiers")
        by_pair.setdefault(pair_id, set()).add(baseline_id)
    expected = {item.value for item in BaselineId}
    if any(ids != expected for ids in by_pair.values()):
        raise ValueError("each output pair must contain exactly B0-B9")
