"""Frozen suite, resource accounting, and common Parquet output tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from ictc.baselines import (
    BASELINE_OUTPUT_SCHEMA,
    BaselineDisposition,
    BaselineId,
    BaselineInput,
    BaselineReason,
    BaselineResult,
    BaselineSuite,
    ObservationPolicy,
    OracleChoice,
    OracleController,
    OutcomeMetrics,
    PairedOutcomes,
    PathSnapshot,
    TrainingCost,
    TrainingExample,
    build_baseline_suite,
    validate_baseline_results,
    write_baseline_results,
)
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction


def make_input(*, split: DatasetSplit = DatasetSplit.TRAIN, index: int = 0) -> BaselineInput:
    snapshots = tuple(
        PathSnapshot(
            path,
            0,
            0,
            0,
            queue,
            0.5,
            0,
            0,
            0,
            False,
            f"telemetry:{path}",
            f"{index + 1:064X}",
        )
        for path, queue in (("current", 10), ("candidate", index))
    )
    return BaselineInput(
        pair_id=f"pair-{split.value}-{index}",
        scenario_id=f"scenario-{split.value}-{index}",
        split=split,
        seed=index,
        decision_tick=0,
        current_path_id="current",
        candidate_path_id="candidate",
        proposed_action=NetworkAction(
            action_id=f"proposal-{index}",
            kind=ActionKind.REROUTE_TO_PATH,
            path_id="candidate",
        ),
        snapshots=snapshots,
        scenario_config_digest="C" * 64,
        schedule_digest=f"{index + 1:064X}",
    )


def examples(split: DatasetSplit) -> tuple[TrainingExample, ...]:
    rows: list[TrainingExample] = []
    for index in range(20):
        beneficial = index < 10
        item = make_input(
            split=split,
            index=index + (100 if split is DatasetSplit.DEVELOPMENT else 0),
        )
        item = replace(
            item,
            snapshots=(
                item.snapshots[0],
                replace(item.snapshots[1], queue_occupancy=index if beneficial else 20 + index),
            ),
        )
        keep = OutcomeMetrics(
            ActionKind.KEEP_CURRENT_PATH, "current", item.schedule_digest, 10, 0, 1, 0
        )
        reroute = OutcomeMetrics(
            ActionKind.REROUTE_TO_PATH,
            "candidate",
            item.schedule_digest,
            8 if beneficial else 12,
            0,
            1,
            0,
        )
        rows.append(
            TrainingExample(
                item,
                PairedOutcomes(item.pair_id, item.scenario_id, keep, reroute),
            )
        )
    return tuple(rows)


def build_suite(item: BaselineInput, *, deadline_ns: int = 50_000_000) -> BaselineSuite:
    oracle = OracleController((OracleChoice(item.pair_id, True, 10, 5, 10),))
    return build_baseline_suite(
        oracle=oracle,
        training=examples(DatasetSplit.TRAIN),
        development=examples(DatasetSplit.DEVELOPMENT),
        deadline_ns=deadline_ns,
    )


def test_suite_runs_ordered_b0_b9_on_identical_pair_and_measures_costs() -> None:
    item = make_input(index=500)
    suite = build_suite(item, deadline_ns=1)
    records = suite.run(item)

    assert tuple(record.result.baseline_id for record in records) == tuple(BaselineId)
    assert {record.result.pair_id for record in records} == {item.pair_id}
    assert {record.result.scenario_id for record in records} == {item.scenario_id}
    assert all(record.cpu_time_ns >= 0 for record in records)
    assert all(record.peak_python_bytes >= 0 for record in records)
    assert all(record.gpu_used is False for record in records)
    assert all(record.deadline_exceeded for record in records)
    assert records[4].training_cost.parameter_count > 0
    assert records[7].training_cost.parameter_count == 0


def test_common_output_is_immutable_schema_valid_and_complete(tmp_path: Path) -> None:
    item = make_input(index=501)
    records = build_suite(item).run(item)
    target = tmp_path / "results.parquet"

    write_baseline_results(records, target)
    validate_baseline_results(target)
    table = pq.read_table(target)
    assert table.schema == BASELINE_OUTPUT_SCHEMA
    assert table.num_rows == 10
    assert set(table.column("baseline_id").to_pylist()) == {item.value for item in BaselineId}
    assert all(value == item.pair_id for value in table.column("pair_id").to_pylist())
    with pytest.raises(FileExistsError, match="overwrite"):
        write_baseline_results(records, target)
    with pytest.raises(ValueError, match="at least one"):
        write_baseline_results((), tmp_path / "empty.parquet")


def test_output_validator_rejects_missing_wrong_schema_and_incomplete_pair(tmp_path: Path) -> None:
    missing = tmp_path / "missing.parquet"
    with pytest.raises(FileNotFoundError):
        validate_baseline_results(missing)

    wrong = tmp_path / "wrong.parquet"
    pq.write_table(pa.table({"value": [1]}), wrong)
    with pytest.raises(ValueError, match="schema"):
        validate_baseline_results(wrong)

    item = make_input(index=502)
    records = build_suite(item).run(item)
    incomplete = tmp_path / "incomplete.parquet"
    rows = [record.to_row() for record in records[:-1]]
    pq.write_table(pa.Table.from_pylist(rows, schema=BASELINE_OUTPUT_SCHEMA), incomplete)
    with pytest.raises(ValueError, match="exactly B0-B9"):
        validate_baseline_results(incomplete)


def test_training_and_run_cost_validation() -> None:
    item = make_input(index=503)
    suite = build_suite(item)
    record = suite.run(item)[0]

    with pytest.raises(ValueError, match="cannot be negative"):
        TrainingCost(BaselineId.B0, -1, 0, 0)
    with pytest.raises(ValueError, match="CPU-only"):
        TrainingCost(BaselineId.B0, 0, 0, 0, cpu_only=False)
    with pytest.raises(ValueError, match="different baselines"):
        replace(record, training_cost=TrainingCost(BaselineId.B1, 0, 0, 0))
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(record, cpu_time_ns=-1)
    with pytest.raises(ValueError, match="positive"):
        replace(record, deadline_ns=0)
    with pytest.raises(ValueError, match="GPU"):
        replace(record, gpu_used=True)


def test_suite_validation_requires_exact_order_costs_deadline_and_digests() -> None:
    item = make_input(index=504)
    suite = build_suite(item)

    with pytest.raises(ValueError, match="ordered B0-B9"):
        BaselineSuite(suite.controllers[:-1], suite.training_costs[:-1])
    with pytest.raises(ValueError, match="positive"):
        replace(suite, deadline_ns=0)


class OverBudgetController:
    baseline_id = BaselineId.B1
    observation_policy = ObservationPolicy((), 0, 0)
    configuration_digest = "F" * 64

    def decide(self, item: BaselineInput) -> BaselineResult:
        return BaselineResult(
            baseline_id=self.baseline_id,
            pair_id=item.pair_id,
            scenario_id=item.scenario_id,
            disposition=BaselineDisposition.EXECUTE,
            reason=BaselineReason.NO_GATE,
            proposed_action=item.proposed_action,
            selected_action=item.proposed_action,
            observation_fields=(),
            telemetry_items=1,
            telemetry_bytes=8,
            inference_time_ns=1,
            configuration_digest=self.configuration_digest,
        )


def test_suite_detects_budget_breach() -> None:
    item = make_input(index=505)
    suite = build_suite(item)
    controllers = (suite.controllers[0], OverBudgetController(), *suite.controllers[2:])
    broken = BaselineSuite(controllers, suite.training_costs, suite.deadline_ns)

    with pytest.raises(RuntimeError, match="telemetry budget"):
        broken.run(item)
