"""Controlled M10 observational benchmark, leakage, and graph-label tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest
import yaml  # type: ignore[import-untyped]

from ictc.baselines import CorpusAccessGuard, CorpusTable
from ictc.datasets import (
    CorpusProvenance,
    DatasetSplit,
    export_corpus_pair,
    load_corpus_matrix,
)
from ictc.domain.enums import ActionKind
from ictc.pilot import (
    TestAccessAuthorization,
    build_repair_decision,
    build_repair_record,
    summarize_repair_benchmark,
)
from ictc.pilot.benchmark import (
    M10_GRAPH_EDGES,
    M10_GRAPH_ID,
    M10_GRAPH_NODES,
    ObservationalDecision,
    PilotBenchmarkRecord,
    PilotOutcomeTruth,
    PotentialOutcomeTruth,
    PreActionEvidence,
    reference_identification,
)
from ictc.simulation import run_paired

ROOT = Path(__file__).resolve().parents[2]
MICRO = ROOT / "configs/experiments/m08_micro_matrix.json"
GRAPH = ROOT / "configs/causal_graphs/m10_observational_selection.yaml"


def _export(tmp_path: Path, split: DatasetSplit, regime: str) -> Path:
    scenario = next(
        item
        for item in load_corpus_matrix(MICRO).scenarios()
        if item.split is split and item.observation_regime_id == regime
    )
    target = tmp_path / f"{split.value}-{regime}"
    export_corpus_pair(
        scenario,
        run_paired(scenario.simulation),
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        target,
    )
    return target


def test_controlled_graph_constants_match_separate_yaml() -> None:
    payload = yaml.safe_load(GRAPH.read_text(encoding="utf-8"))

    assert payload["graph_id"] == M10_GRAPH_ID
    assert set(payload["variables"]) == set(M10_GRAPH_NODES)
    assert {tuple(edge) for edge in payload["edges"]} == set(M10_GRAPH_EDGES)
    assert "does not revise" in payload["change_control"]


def test_complete_record_is_identified_by_queue_pressure(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.TRAIN, "complete")
    guard = CorpusAccessGuard()
    record = build_repair_record(target, split=DatasetSplit.TRAIN, guard=guard)

    assert record.decision.visible_queue_pressure_bin == record.truth.true_queue_pressure_bin
    assert record.decision.reference_identification.identified is True
    assert record.decision.reference_identification.minimal_adjustment_sets == (
        ("QueuePressureBin",),
    )
    assert record.decision.observed_latency_ms in {
        record.truth.keep_latency_ms,
        record.truth.reroute_latency_ms,
    }
    guard.assert_no_denied_access()


def test_skewed_record_separates_immediate_visibility_from_acquired_identification(
    tmp_path: Path,
) -> None:
    target = _export(tmp_path, DatasetSplit.DEVELOPMENT, "skewed-stale")
    first = build_repair_record(target, split=DatasetSplit.DEVELOPMENT)
    second = build_repair_record(target, split=DatasetSplit.DEVELOPMENT)

    assert first == second
    assert first.decision.visible_queue_pressure_bin is None
    assert (
        first.decision.pre_action_evidence.queue_pressure_bin == first.truth.true_queue_pressure_bin
    )
    assert first.decision.pre_action_evidence.freshness_ticks is not None
    assert first.decision.pre_action_evidence.freshness_ticks > 0
    assert first.decision.reference_identification.identified is True
    expected = (
        first.truth.reroute_latency_ms
        if first.decision.assigned_action is ActionKind.REROUTE_TO_PATH
        else first.truth.keep_latency_ms
    )
    assert first.decision.observed_latency_ms == expected


def test_repair_builder_rejects_test_before_reading_it(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.TEST, "complete")
    guard = CorpusAccessGuard()

    with pytest.raises(PermissionError, match="requires GO authorization"):
        build_repair_record(target, split=DatasetSplit.TEST, guard=guard)
    assert guard.events == []

    authorization = TestAccessAuthorization(
        "synthetic-test-fixture",
        "A" * 40,
        "B" * 40,
        "C" * 64,
        "D" * 64,
        1,
        (tmp_path / "authorized-output").resolve(),
    )
    decision = build_repair_decision(
        target,
        split=DatasetSplit.TEST,
        guard=guard,
        test_authorization=authorization,
    )
    assert decision.split is DatasetSplit.TEST
    assert all(event.table is not CorpusTable.OUTCOMES for event in guard.events)


def test_decision_only_builder_does_not_read_outcomes_or_accounting(tmp_path: Path) -> None:
    target = _export(tmp_path, DatasetSplit.DEVELOPMENT, "complete")
    guard = CorpusAccessGuard()
    decision = build_repair_decision(target, split=DatasetSplit.DEVELOPMENT, guard=guard)

    assert decision.observed_latency_ms is None
    assert all(
        event.table not in {CorpusTable.OUTCOMES, CorpusTable.ACCOUNTING} for event in guard.events
    )


def test_repair_summary_retains_identification_support_and_truth(tmp_path: Path) -> None:
    complete = build_repair_record(
        _export(tmp_path, DatasetSplit.TRAIN, "complete"), split=DatasetSplit.TRAIN
    )
    stress = build_repair_record(
        _export(tmp_path, DatasetSplit.DEVELOPMENT, "skewed-stale"),
        split=DatasetSplit.DEVELOPMENT,
    )
    unavailable = replace(
        stress.decision,
        pre_action_evidence=PreActionEvidence(None, None, None, None, None, (), (), 0, 0),
        reference_identification=reference_identification(False),
    )
    summary = summarize_repair_benchmark(
        (complete, PilotBenchmarkRecord(unavailable, stress.truth))
    )
    regimes = cast(dict[str, dict[str, int]], summary["regime_counts"])
    beneficial = cast(int, summary["reroute_beneficial_count"])
    harmful = cast(int, summary["reroute_harmful_or_null_count"])

    assert summary["pair_count"] == 2
    assert regimes["complete"]["identified_count"] == 1
    assert regimes["skewed-stale"]["non_identified_count"] == 1
    assert beneficial + harmful == 2
    with pytest.raises(ValueError, match="at least one"):
        summarize_repair_benchmark(())
    test_decision = replace(complete.decision, split=DatasetSplit.TEST)
    with pytest.raises(PermissionError, match="test records"):
        summarize_repair_benchmark((PilotBenchmarkRecord(test_decision, complete.truth),))


def test_benchmark_value_validation(tmp_path: Path) -> None:
    record = build_repair_record(
        _export(tmp_path, DatasetSplit.TRAIN, "complete"), split=DatasetSplit.TRAIN
    )
    with pytest.raises(ValueError, match="KEEP or REROUTE"):
        replace(record.decision, assigned_action=cast(ActionKind, "INVALID"))
    with pytest.raises(ValueError, match="latency"):
        replace(record.decision, observed_latency_ms=float("nan"))
    with pytest.raises(ValueError, match="pressure"):
        replace(record.decision, visible_queue_pressure_bin=2)
    with pytest.raises(ValueError, match="probabilities"):
        replace(record.decision, assignment_uniform=1.0)
    with pytest.raises(ValueError, match="pressure"):
        replace(record.truth, true_queue_pressure_bin=2)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        replace(record.truth.keep, mean_latency_ms=-1)
    with pytest.raises(ValueError, match="pair IDs"):
        PilotBenchmarkRecord(
            record.decision,
            PotentialOutcomeTruth(
                "other",
                0,
                PilotOutcomeTruth(1, 1, 0, 0, 1, 0, 0),
                PilotOutcomeTruth(2, 1, 0, 0, 1, 0, 1),
            ),
        )
    with pytest.raises(ValueError, match="wholly present"):
        replace(record.decision.pre_action_evidence, queue_pressure_bin=None)
    with pytest.raises(ValueError, match="one source ID"):
        replace(record.decision.pre_action_evidence, source_ids=())


def test_observational_decision_type_excludes_paired_truth_fields() -> None:
    assert "keep_latency_ms" not in ObservationalDecision.__dataclass_fields__
    assert "reroute_latency_ms" not in ObservationalDecision.__dataclass_fields__
