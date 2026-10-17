"""M09 corpus access-control and leakage-safe input-loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ictc.baselines import (
    AccessPhase,
    BaselineId,
    CorpusAccessError,
    CorpusAccessGuard,
    CorpusTable,
    load_baseline_input,
)
from ictc.datasets import CorpusProvenance, DatasetSplit, export_corpus_pair, load_corpus_matrix
from ictc.simulation import run_paired

ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "configs/experiments/m08_micro_matrix.json"


def _export(tmp_path: Path, split: DatasetSplit) -> Path:
    scenario = next(item for item in load_corpus_matrix(MICRO).scenarios() if item.split is split)
    target = tmp_path / split.value
    export_corpus_pair(
        scenario,
        run_paired(scenario.simulation),
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        target,
    )
    return target


def test_b0_exclusively_reads_oracle_and_decision_outcomes(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.TRAIN)
    guard = CorpusAccessGuard()

    oracle = guard.read_table(
        target,
        baseline_id=BaselineId.B0,
        phase=AccessPhase.DECISION,
        split=DatasetSplit.TRAIN,
        table=CorpusTable.ORACLE,
    )
    outcomes = guard.read_table(
        target,
        baseline_id=BaselineId.B0,
        phase=AccessPhase.DECISION,
        split=DatasetSplit.TRAIN,
        table=CorpusTable.OUTCOMES,
    )

    assert oracle.num_rows > 0
    assert outcomes.num_rows == 2
    guard.assert_no_denied_access()


@pytest.mark.parametrize("baseline_id", [item for item in BaselineId if item is not BaselineId.B0])
def test_non_oracle_baselines_cannot_read_oracle_at_decision_time(
    tmp_path: Path, baseline_id: BaselineId
) -> None:
    target = _export(tmp_path, DatasetSplit.TRAIN)
    guard = CorpusAccessGuard()

    with pytest.raises(CorpusAccessError, match="exclusive to B0"):
        guard.read_table(
            target,
            baseline_id=baseline_id,
            phase=AccessPhase.DECISION,
            split=DatasetSplit.TRAIN,
            table=CorpusTable.ORACLE,
        )

    assert guard.events[-1].allowed is False
    with pytest.raises(CorpusAccessError, match="denied request"):
        guard.assert_no_denied_access()


def test_training_outcomes_are_limited_by_method_and_actual_split(tmp_path: Path) -> None:
    train = _export(tmp_path, DatasetSplit.TRAIN)
    test = _export(tmp_path, DatasetSplit.TEST)

    for baseline_id in (BaselineId.B4, BaselineId.B5, BaselineId.B8, BaselineId.B9):
        guard = CorpusAccessGuard()
        assert (
            guard.read_table(
                train,
                baseline_id=baseline_id,
                phase=AccessPhase.TRAINING,
                split=DatasetSplit.TRAIN,
                table=CorpusTable.OUTCOMES,
            ).num_rows
            == 2
        )
    with pytest.raises(CorpusAccessError, match="test outcomes"):
        CorpusAccessGuard().read_table(
            test,
            baseline_id=BaselineId.B4,
            phase=AccessPhase.TRAINING,
            split=DatasetSplit.TEST,
            table=CorpusTable.OUTCOMES,
        )
    with pytest.raises(CorpusAccessError, match="forbidden"):
        CorpusAccessGuard().read_table(
            train,
            baseline_id=BaselineId.B3,
            phase=AccessPhase.TRAINING,
            split=DatasetSplit.TRAIN,
            table=CorpusTable.OUTCOMES,
        )
    with pytest.raises(CorpusAccessError, match="does not match"):
        CorpusAccessGuard().read_table(
            test,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.DECISION,
            split=DatasetSplit.TRAIN,
            table=CorpusTable.ORACLE,
        )


def test_public_tables_are_visible_but_future_schedule_is_not(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.DEVELOPMENT)
    guard = CorpusAccessGuard()

    for table in (
        CorpusTable.SCENARIO,
        CorpusTable.VISIBLE,
        CorpusTable.ACTION_PROPOSALS,
        CorpusTable.CONTRACTS,
    ):
        assert (
            guard.read_table(
                target,
                baseline_id=BaselineId.B7,
                phase=AccessPhase.DECISION,
                split=DatasetSplit.DEVELOPMENT,
                table=table,
            ).num_rows
            > 0
        )
    with pytest.raises(CorpusAccessError, match="outside"):
        guard.read_table(
            target,
            baseline_id=BaselineId.B7,
            phase=AccessPhase.DECISION,
            split=DatasetSplit.DEVELOPMENT,
            table=CorpusTable.SCHEDULE,
        )


def test_loader_uses_only_tick_zero_baseline_trace(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.TRAIN)
    guard = CorpusAccessGuard()
    item = load_baseline_input(
        target,
        baseline_id=BaselineId.B2,
        split=DatasetSplit.TRAIN,
        guard=guard,
    )

    assert item.split is DatasetSplit.TRAIN
    assert item.current_path_id != item.candidate_path_id
    assert item.proposed_action.path_id == item.candidate_path_id
    assert all(snapshot.available_tick == 0 for snapshot in item.snapshots)
    assert {event.table for event in guard.events} == {
        CorpusTable.SCENARIO,
        CorpusTable.VISIBLE,
        CorpusTable.ACTION_PROPOSALS,
    }
    guard.assert_no_denied_access()
    with pytest.raises(ValueError, match="tick zero"):
        load_baseline_input(
            target,
            baseline_id=BaselineId.B2,
            split=DatasetSplit.TRAIN,
            decision_tick=1,
        )


def test_guard_reports_missing_or_invalid_corpus_metadata(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        CorpusAccessGuard().read_table(
            tmp_path,
            baseline_id=BaselineId.B0,
            phase=AccessPhase.DECISION,
            split=DatasetSplit.TRAIN,
            table=CorpusTable.ORACLE,
        )
