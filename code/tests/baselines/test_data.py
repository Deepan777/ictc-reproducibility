"""M09 outcome parsing, training-example, and feature-extraction tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ictc.baselines import (
    BaselineId,
    BaselineInput,
    CorpusAccessGuard,
    OutcomeMetrics,
    PairedOutcomes,
    PathSnapshot,
    TrainingExample,
    extract_features,
    load_training_examples,
)
from ictc.datasets import CorpusProvenance, DatasetSplit, export_corpus_pair, load_corpus_matrix
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction
from ictc.simulation import run_paired

ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "configs/experiments/m08_micro_matrix.json"


def make_input(
    *,
    split: DatasetSplit = DatasetSplit.TRAIN,
    current_queue: int = 5,
    candidate_queue: int = 2,
    current_utilization: float = 0.8,
    candidate_utilization: float = 0.4,
    candidate_present: bool = True,
    stale: bool = False,
    index: int = 0,
) -> BaselineInput:
    def snapshot(path: str, queue: int, utilization: float) -> PathSnapshot:
        return PathSnapshot(
            path_id=path,
            sample_tick=0,
            available_tick=0,
            observed_timestamp_tick=0,
            queue_occupancy=queue,
            utilization=utilization,
            sample_age_ticks=0,
            measurement_delay_ticks=0,
            clock_skew_ticks=0,
            stale=stale,
            source_id=f"telemetry:{path}",
            provenance_digest=f"{index:064X}",
        )

    snapshots = [snapshot("current", current_queue, current_utilization)]
    if candidate_present:
        snapshots.append(snapshot("candidate", candidate_queue, candidate_utilization))
    return BaselineInput(
        pair_id=f"pair-{index}",
        scenario_id=f"scenario-{index}",
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
        snapshots=tuple(snapshots),
        scenario_config_digest="C" * 64,
        schedule_digest=f"{index + 1:064X}",
    )


def make_outcomes(item: BaselineInput, *, beneficial: bool) -> PairedOutcomes:
    keep = OutcomeMetrics(
        ActionKind.KEEP_CURRENT_PATH,
        "current",
        item.schedule_digest,
        10.0,
        0.0,
        1.0,
        0.0,
    )
    reroute = OutcomeMetrics(
        ActionKind.REROUTE_TO_PATH,
        "candidate",
        item.schedule_digest,
        8.0 if beneficial else 12.0,
        0.0,
        1.0,
        0.0,
    )
    return PairedOutcomes(item.pair_id, item.scenario_id, keep, reroute)


def make_examples(
    split: DatasetSplit = DatasetSplit.TRAIN, *, count: int = 20
) -> tuple[TrainingExample, ...]:
    examples: list[TrainingExample] = []
    for index in range(count):
        beneficial = index < count // 2
        item = make_input(
            split=split,
            current_queue=10,
            candidate_queue=index if beneficial else 20 + index,
            current_utilization=0.5,
            candidate_utilization=0.5,
            index=index + (100 if split is DatasetSplit.DEVELOPMENT else 0),
        )
        examples.append(TrainingExample(item, make_outcomes(item, beneficial=beneficial)))
    return tuple(examples)


def test_outcome_metrics_objective_and_pair_properties() -> None:
    item = make_input()
    pair = make_outcomes(item, beneficial=True)

    assert pair.latency_effect_ms == -2
    assert pair.reroute_beneficial is True
    assert pair.keep.objective(loss_weight=100, jitter_weight=2, deadline_weight=100) == 12
    assert TrainingExample(item, pair).outcomes == pair


def test_outcome_and_training_validation_reject_inconsistent_data() -> None:
    item = make_input()
    pair = make_outcomes(item, beneficial=False)

    with pytest.raises(ValueError, match="path and schedule"):
        replace(pair.keep, selected_path_id="")
    with pytest.raises(ValueError, match="finite"):
        replace(pair.keep, mean_latency_ms=float("nan"))
    with pytest.raises(ValueError, match="between"):
        replace(pair.keep, deadline_violation_rate=2)
    with pytest.raises(ValueError, match="wrong action"):
        replace(pair, keep=replace(pair.keep, action_kind=ActionKind.REROUTE_TO_PATH))
    with pytest.raises(ValueError, match="share one"):
        replace(pair, reroute=replace(pair.reroute, schedule_digest="D" * 64))
    with pytest.raises(ValueError, match="pair IDs"):
        TrainingExample(replace(item, pair_id="other"), pair)
    with pytest.raises(ValueError, match="scenario IDs"):
        TrainingExample(replace(item, scenario_id="other"), pair)
    with pytest.raises(ValueError, match="different schedules"):
        TrainingExample(replace(item, schedule_digest="E" * 64), pair)


def test_feature_extraction_is_ordered_costed_and_missing_aware() -> None:
    item = make_input(current_queue=8, candidate_queue=3)
    vector = extract_features(item, ("queue_occupancy", "utilization"))

    assert vector.complete is True
    assert vector.telemetry_items == 4
    assert vector.telemetry_bytes == 32
    assert vector.values[:4] == (8.0, 3.0, -5.0, 1.0)
    assert vector.names[0] == "current.queue_occupancy"

    missing = extract_features(replace(item, snapshots=item.snapshots[:1]), ("utilization",))
    assert missing.complete is False
    assert missing.values == (0.8, 0.0, -0.8, 0.0)
    assert missing.telemetry_items == 1

    with pytest.raises(ValueError, match="unique"):
        extract_features(item, ("utilization", "utilization"))
    with pytest.raises(ValueError, match="unsupported"):
        extract_features(item, ("true_queue",))
    with pytest.raises(ValueError, match="equal length"):
        replace(vector, values=())
    with pytest.raises(ValueError, match="unique"):
        replace(vector, names=(vector.names[0],) * len(vector.names))
    with pytest.raises(ValueError, match="finite"):
        replace(vector, values=(float("nan"), *vector.values[1:]))
    with pytest.raises(ValueError, match="negative"):
        replace(vector, telemetry_bytes=-1)


def test_training_loader_uses_guard_and_rejects_test_or_empty(tmp_path: Path) -> None:
    scenario = next(
        item for item in load_corpus_matrix(MICRO).scenarios() if item.split is DatasetSplit.TRAIN
    )
    target = tmp_path / "pair"
    export_corpus_pair(
        scenario,
        run_paired(scenario.simulation),
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        target,
    )
    guard = CorpusAccessGuard()
    examples = load_training_examples(
        (target,), baseline_id=BaselineId.B4, split=DatasetSplit.TRAIN, guard=guard
    )

    assert len(examples) == 1
    assert examples[0].outcomes.keep.schedule_digest == examples[0].input.schedule_digest
    guard.assert_no_denied_access()
    with pytest.raises(ValueError, match="test outcomes"):
        load_training_examples((), baseline_id=BaselineId.B4, split=DatasetSplit.TEST)
    with pytest.raises(ValueError, match="at least one"):
        load_training_examples((), baseline_id=BaselineId.B4, split=DatasetSplit.TRAIN)
