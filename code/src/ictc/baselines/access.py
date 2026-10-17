"""Audited M08 corpus access with an explicit B0-only oracle boundary."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from ictc.baselines.types import BaselineId
from ictc.datasets.m08_config import DatasetSplit


class AccessPhase(StrEnum):
    DECISION = "DECISION"
    TRAINING = "TRAINING"


class CorpusTable(StrEnum):
    SCENARIO = "scenario.parquet"
    VISIBLE = "visible.parquet"
    ACTION_PROPOSALS = "action_proposals.parquet"
    CONTRACTS = "contracts.parquet"
    FAILURE_INJECTIONS = "failure_injections.parquet"
    PROTOCOL_EVENTS = "protocol_events.parquet"
    RUN_REGISTRY = "run_registry.parquet"
    SCHEDULE = "schedule.parquet"
    ORACLE = "oracle.parquet"
    OUTCOMES = "outcomes.parquet"
    ACCOUNTING = "accounting.parquet"


class CorpusAccessError(PermissionError):
    """Raised whenever a baseline requests data outside its frozen observation policy."""


@dataclass(frozen=True, slots=True)
class AccessEvent:
    baseline_id: BaselineId
    phase: AccessPhase
    split: DatasetSplit
    table: CorpusTable
    allowed: bool
    reason: str


_DECISION_PUBLIC = frozenset(
    {
        CorpusTable.SCENARIO,
        CorpusTable.VISIBLE,
        CorpusTable.ACTION_PROPOSALS,
        CorpusTable.CONTRACTS,
    }
)
_OUTCOME_TRAINING_BASELINES = frozenset(
    {BaselineId.B4, BaselineId.B5, BaselineId.B8, BaselineId.B9}
)


@dataclass(slots=True)
class CorpusAccessGuard:
    """Read corpus tables only after checking baseline, phase, and actual split metadata."""

    events: list[AccessEvent] = field(default_factory=list)

    @staticmethod
    def _actual_split(directory: Path) -> DatasetSplit:
        scenario_path = directory / CorpusTable.SCENARIO.value
        if not scenario_path.is_file():
            raise FileNotFoundError(scenario_path)
        values = pq.read_table(scenario_path, columns=["split"]).column("split").to_pylist()
        if len(values) != 1 or not isinstance(values[0], str):
            raise ValueError("scenario table must contain exactly one string split")
        return DatasetSplit(values[0])

    def _record(
        self,
        baseline_id: BaselineId,
        phase: AccessPhase,
        split: DatasetSplit,
        table: CorpusTable,
        *,
        allowed: bool,
        reason: str,
    ) -> None:
        self.events.append(AccessEvent(baseline_id, phase, split, table, allowed, reason))

    def _authorize(
        self,
        baseline_id: BaselineId,
        phase: AccessPhase,
        split: DatasetSplit,
        table: CorpusTable,
    ) -> tuple[bool, str]:
        if table in _DECISION_PUBLIC:
            return True, "controller-visible table"
        if baseline_id is BaselineId.B0:
            return True, "B0 full-state oracle role"
        if table is CorpusTable.OUTCOMES and phase is AccessPhase.TRAINING:
            if split is DatasetSplit.TEST:
                return False, "test outcomes are forbidden during training"
            if baseline_id in _OUTCOME_TRAINING_BASELINES:
                return True, "development-authorized historical outcome fitting"
        if table is CorpusTable.ORACLE:
            return False, "oracle state is exclusive to B0"
        if table in {CorpusTable.OUTCOMES, CorpusTable.ACCOUNTING}:
            return False, "action outcomes are forbidden at baseline decision time"
        return False, "table is outside the frozen baseline observation policy"

    def read_table(
        self,
        directory: Path,
        *,
        baseline_id: BaselineId,
        phase: AccessPhase,
        split: DatasetSplit,
        table: CorpusTable,
        columns: list[str] | None = None,
    ) -> pa.Table:
        resolved = directory.resolve()
        actual_split = self._actual_split(resolved)
        if actual_split is not split:
            reason = (
                f"declared split {split.value} does not match corpus split {actual_split.value}"
            )
            self._record(baseline_id, phase, actual_split, table, allowed=False, reason=reason)
            raise CorpusAccessError(reason)
        allowed, reason = self._authorize(baseline_id, phase, actual_split, table)
        self._record(baseline_id, phase, actual_split, table, allowed=allowed, reason=reason)
        if not allowed:
            raise CorpusAccessError(reason)
        path = resolved / table.value
        if not path.is_file():
            raise FileNotFoundError(path)
        return pq.read_table(path, columns=columns)

    def assert_no_denied_access(self) -> None:
        denied = [event for event in self.events if not event.allowed]
        if denied:
            raise CorpusAccessError(f"access audit contains {len(denied)} denied request(s)")
