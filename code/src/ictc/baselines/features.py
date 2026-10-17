"""Leakage-safe conversion from one M08 pair directory to a common baseline input."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from ictc.baselines.access import AccessPhase, CorpusAccessGuard, CorpusTable
from ictc.baselines.types import BaselineId, BaselineInput, PathSnapshot
from ictc.datasets.m08_config import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction


def _single_row(rows: list[dict[str, object]], table_name: str) -> dict[str, object]:
    if len(rows) != 1:
        raise ValueError(f"{table_name} must contain exactly one row")
    return rows[0]


def load_baseline_input(
    directory: Path,
    *,
    baseline_id: BaselineId,
    split: DatasetSplit,
    decision_tick: int = 0,
    guard: CorpusAccessGuard | None = None,
) -> BaselineInput:
    """Load only baseline-run observations delivered by the declared decision tick.

    M08 outcomes are fixed-action traces, so any visible sample after tick zero can be
    action-dependent. The M09 offline decision contract therefore defaults to tick zero and
    rejects future rows. A later decision tick requires a separately validated pre-action phase.
    """
    if decision_tick != 0:
        raise ValueError(
            "M09 corpus decisions are frozen at tick zero to prevent post-action leakage"
        )
    access = guard or CorpusAccessGuard()
    scenario_rows = access.read_table(
        directory,
        baseline_id=baseline_id,
        phase=AccessPhase.DECISION,
        split=split,
        table=CorpusTable.SCENARIO,
    ).to_pylist()
    scenario = _single_row(cast(list[dict[str, object]], scenario_rows), "scenario")
    proposal_rows = access.read_table(
        directory,
        baseline_id=baseline_id,
        phase=AccessPhase.DECISION,
        split=split,
        table=CorpusTable.ACTION_PROPOSALS,
    ).to_pylist()
    proposals = cast(list[dict[str, object]], proposal_rows)
    baseline_rows = [item for item in proposals if item.get("is_baseline") is True]
    candidate_rows = [item for item in proposals if item.get("is_baseline") is False]
    baseline_proposal = _single_row(baseline_rows, "baseline proposal")
    candidate_proposal = _single_row(candidate_rows, "candidate proposal")
    baseline_run_id = baseline_proposal.get("run_id")
    candidate_path_id = candidate_proposal.get("target_path_id")
    if not isinstance(baseline_run_id, str) or not isinstance(candidate_path_id, str):
        raise ValueError("action proposals contain invalid run or path IDs")
    if candidate_proposal.get("action_kind") != ActionKind.REROUTE_TO_PATH.value:
        raise ValueError("candidate proposal must be a reroute action")

    visible_rows = access.read_table(
        directory,
        baseline_id=baseline_id,
        phase=AccessPhase.DECISION,
        split=split,
        table=CorpusTable.VISIBLE,
    ).to_pylist()
    visible = cast(list[dict[str, object]], visible_rows)
    latest: dict[str, dict[str, object]] = {}
    for row in visible:
        if row.get("run_id") != baseline_run_id:
            continue
        available_tick = row.get("available_tick")
        path_id = row.get("path_id")
        if not isinstance(available_tick, int) or not isinstance(path_id, str):
            raise ValueError("visible row has invalid tick or path ID")
        if available_tick > decision_tick or row.get("missing") is True:
            continue
        previous = latest.get(path_id)
        if previous is None or cast(int, previous["available_tick"]) <= available_tick:
            latest[path_id] = row
    if candidate_path_id not in {cast(str, row.get("path_id")) for row in visible}:
        raise ValueError("candidate path does not occur in the visible table")
    path_ids = sorted({cast(str, row.get("path_id")) for row in visible})
    current_paths = [path_id for path_id in path_ids if path_id != candidate_path_id]
    if len(current_paths) != 1:
        raise ValueError("M09 baseline loader requires exactly two paths")

    snapshots: list[PathSnapshot] = []
    for path_id, row in sorted(latest.items()):
        values = (
            row.get("sample_tick"),
            row.get("available_tick"),
            row.get("observed_timestamp_tick"),
            row.get("queue_occupancy"),
            row.get("utilization"),
            row.get("measurement_delay_ticks"),
            row.get("clock_skew_ticks"),
        )
        if (
            not all(isinstance(value, int) for value in values[:4])
            or not isinstance(values[4], int | float)
            or isinstance(values[4], bool)
            or not all(isinstance(value, int) for value in values[5:])
        ):
            raise ValueError("visible row contains invalid numeric fields")
        source_id = row.get("source_id")
        provenance = row.get("provenance_digest")
        if not isinstance(source_id, str) or not isinstance(provenance, str):
            raise ValueError("visible row contains invalid source provenance")
        observed_tick = cast(int, values[2])
        snapshots.append(
            PathSnapshot(
                path_id=path_id,
                sample_tick=cast(int, values[0]),
                available_tick=cast(int, values[1]),
                observed_timestamp_tick=observed_tick,
                queue_occupancy=cast(int, values[3]),
                utilization=float(values[4]),
                sample_age_ticks=max(0, decision_tick - observed_tick),
                measurement_delay_ticks=cast(int, values[5]),
                clock_skew_ticks=cast(int, values[6]),
                stale=bool(row.get("stale")),
                source_id=source_id,
                provenance_digest=provenance,
            )
        )

    pair_id = scenario.get("pair_id")
    scenario_id = scenario.get("scenario_id")
    seed = scenario.get("seed")
    config_digest = scenario.get("simulation_config_hash")
    schedule_digest = scenario.get("schedule_digest")
    if (
        not isinstance(pair_id, str)
        or not isinstance(scenario_id, str)
        or not isinstance(seed, int)
        or not isinstance(config_digest, str)
        or not isinstance(schedule_digest, str)
    ):
        raise ValueError("scenario row contains invalid identifiers")
    proposed_action = NetworkAction(
        action_id=cast(str, candidate_proposal["proposal_id"]),
        kind=ActionKind.REROUTE_TO_PATH,
        path_id=candidate_path_id,
    )
    return BaselineInput(
        pair_id=pair_id,
        scenario_id=scenario_id,
        split=split,
        seed=seed,
        decision_tick=decision_tick,
        current_path_id=current_paths[0],
        candidate_path_id=candidate_path_id,
        proposed_action=proposed_action,
        snapshots=tuple(snapshots),
        scenario_config_digest=config_digest,
        schedule_digest=schedule_digest,
    )
