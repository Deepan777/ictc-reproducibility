"""Frozen M10 revision matrix and sealed-evaluation boundary tests."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ictc.datasets import DatasetSplit, load_corpus_matrix
from ictc.pilot import TopologyContext

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiments/m10r_revision_matrix.json"


def test_m10r_matrix_has_frozen_split_counts_and_disjoint_seeds() -> None:
    matrix = load_corpus_matrix(MATRIX)
    scenarios = matrix.scenarios()
    counts = Counter(item.split for item in scenarios)

    assert matrix.corpus_id == "ictc-m10r-contextual"
    assert matrix.pair_count == 3600
    assert counts == {
        DatasetSplit.TRAIN: 2160,
        DatasetSplit.DEVELOPMENT: 720,
        DatasetSplit.TEST: 720,
    }
    seeds = {
        split: {item.seed for item in scenarios if item.split is split} for split in DatasetSplit
    }
    assert seeds[DatasetSplit.TRAIN].isdisjoint(seeds[DatasetSplit.DEVELOPMENT])
    assert seeds[DatasetSplit.TRAIN].isdisjoint(seeds[DatasetSplit.TEST])
    assert seeds[DatasetSplit.DEVELOPMENT].isdisjoint(seeds[DatasetSplit.TEST])


def test_m10r_contexts_are_pre_simulation_and_registered_before_evaluation() -> None:
    matrix = load_corpus_matrix(MATRIX)
    contexts = {
        family.family_id: TopologyContext.from_path_configuration(
            current_service_packets_per_tick=family.current_service_packets_per_tick,
            candidate_service_packets_per_tick=family.candidate_service_packets_per_tick,
            current_propagation_delay_ms=family.current_propagation_delay_ms,
            candidate_propagation_delay_ms=family.candidate_propagation_delay_ms,
            current_queue_limit_packets=family.current_queue_limit_packets,
            candidate_queue_limit_packets=family.candidate_queue_limit_packets,
        ).context_id
        for family in matrix.topology_families
    }
    training_contexts = {
        context for family, context in contexts.items() if family.startswith("m10r-train-")
    }
    evaluation_contexts = {
        context for family, context in contexts.items() if family.startswith("m10r-evaluation-")
    }

    assert len(training_contexts) == 3
    assert evaluation_contexts <= training_contexts
    assert all(
        "m10r-evaluation" in item.topology_family_id
        for item in matrix.scenarios()
        if item.split is DatasetSplit.TEST
    )
