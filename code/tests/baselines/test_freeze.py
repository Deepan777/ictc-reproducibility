"""Reloadable and tamper-evident M09 baseline configuration tests."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ictc.baselines import (
    BaselineId,
    BaselineInput,
    BaselineSuite,
    ObservationPolicy,
    OracleChoice,
    OracleController,
    OutcomeMetrics,
    PairedOutcomes,
    PathSnapshot,
    TrainingExample,
    build_baseline_suite,
    frozen_baseline_payload,
    load_frozen_baseline_suite,
    write_frozen_baseline_suite,
)
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction


def make_input(split: DatasetSplit, index: int) -> BaselineInput:
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
        for path, queue in (("current", 10), ("candidate", index % 20))
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
    result: list[TrainingExample] = []
    offset = 100 if split is DatasetSplit.DEVELOPMENT else 0
    for index in range(20):
        beneficial = index < 10
        item = make_input(split, index + offset)
        item = replace(
            item,
            snapshots=(
                item.snapshots[0],
                replace(
                    item.snapshots[1],
                    queue_occupancy=index if beneficial else 20 + index,
                ),
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
        result.append(
            TrainingExample(
                item,
                PairedOutcomes(item.pair_id, item.scenario_id, keep, reroute),
            )
        )
    return tuple(result)


def make_payload() -> tuple[dict[str, Any], OracleController, BaselineInput]:
    item = make_input(DatasetSplit.DEVELOPMENT, 999)
    oracle = OracleController((OracleChoice(item.pair_id, True, 10, 5, 20),))
    suite = build_baseline_suite(
        oracle=oracle,
        training=examples(DatasetSplit.TRAIN),
        development=examples(DatasetSplit.DEVELOPMENT),
    )
    payload = frozen_baseline_payload(
        suite,
        implementation_commit="1" * 40,
        corpus_root="data/raw/m08_pilot/runs",
        manifest_tree_digest="A" * 64,
        training_pair_ids=tuple(f"train-{index}" for index in range(20)),
        development_pair_ids=tuple(f"dev-{index}" for index in range(20)),
        smoke_pair_ids=("dev-0",),
    )
    return payload, oracle, item


def test_frozen_suite_round_trip_reconstructs_all_digests_and_runs(tmp_path: Path) -> None:
    payload, oracle, item = make_payload()
    target = tmp_path / "m09_frozen.json"

    write_frozen_baseline_suite(payload, target)
    loaded = load_frozen_baseline_suite(target, oracle=oracle)

    controller_payloads = payload["controllers"]
    assert isinstance(controller_payloads, list)
    assert tuple(controller.configuration_digest for controller in loaded.controllers) == tuple(
        value["configuration_digest"] for value in controller_payloads if isinstance(value, dict)
    )
    assert tuple(record.result.baseline_id for record in loaded.run(item)) == tuple(BaselineId)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_frozen_baseline_suite(payload, target)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda root: root.update(schema_version="2.0.0"), "schema"),
        (lambda root: root.update(milestone="M10"), "milestone"),
        (
            lambda root: root["tuning_policy"].update(forbidden_split="development"),
            "split isolation",
        ),
        (lambda root: root["controllers"].reverse(), "ordered B0-B9"),
        (
            lambda root: root["controllers"][4].update(configuration_digest="F" * 64),
            "digest",
        ),
        (
            lambda root: root["controllers"][2]["observation_policy"].update(maximum_items=99),
            "observation policy",
        ),
    ],
)
def test_frozen_suite_rejects_manifest_tampering(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    message: str,
) -> None:
    payload, oracle, _item = make_payload()
    mutable = json.loads(json.dumps(payload))
    mutation(mutable)
    target = tmp_path / "tampered.json"
    target.write_text(json.dumps(mutable), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_frozen_baseline_suite(target, oracle=oracle)


def test_freeze_provenance_and_oracle_validation() -> None:
    payload, oracle, _item = make_payload()
    with pytest.raises(ValueError, match="Git object"):
        frozen_baseline_payload(
            load_suite(payload, oracle),
            implementation_commit="short",
            corpus_root="root",
            manifest_tree_digest="A" * 64,
            training_pair_ids=("train",),
            development_pair_ids=("dev",),
            smoke_pair_ids=("dev",),
        )
    with pytest.raises(ValueError, match="disjoint"):
        frozen_baseline_payload(
            load_suite(payload, oracle),
            implementation_commit="1" * 40,
            corpus_root="root",
            manifest_tree_digest="A" * 64,
            training_pair_ids=("same",),
            development_pair_ids=("same",),
            smoke_pair_ids=("same",),
        )
    with pytest.raises(ValueError, match="development subset"):
        frozen_baseline_payload(
            load_suite(payload, oracle),
            implementation_commit="1" * 40,
            corpus_root="root",
            manifest_tree_digest="A" * 64,
            training_pair_ids=("train",),
            development_pair_ids=("dev",),
            smoke_pair_ids=("other",),
        )


def load_suite(payload: dict[str, Any], oracle: OracleController) -> BaselineSuite:
    """Use a temporary manifest to exercise the public loader in validation tests."""
    with tempfile.TemporaryDirectory() as directory:
        target = Path(directory) / "frozen.json"
        target.write_text(json.dumps(payload), encoding="utf-8")
        return load_frozen_baseline_suite(target, oracle=oracle)


def test_empty_policy_is_still_available_for_oracle_and_no_gate() -> None:
    assert ObservationPolicy((), 0, 0).fields == ()
