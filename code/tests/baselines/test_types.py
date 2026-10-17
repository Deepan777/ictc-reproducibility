"""M09 common input, output, and observation-policy validation tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ictc.baselines import (
    BaselineDisposition,
    BaselineId,
    BaselineInput,
    BaselineReason,
    BaselineResult,
    ObservationPolicy,
    PathSnapshot,
)
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction


def _snapshot(**changes: object) -> PathSnapshot:
    values: dict[str, object] = {
        "path_id": "current",
        "sample_tick": 0,
        "available_tick": 0,
        "observed_timestamp_tick": 0,
        "queue_occupancy": 2,
        "utilization": 0.5,
        "sample_age_ticks": 0,
        "measurement_delay_ticks": 0,
        "clock_skew_ticks": 0,
        "stale": False,
        "source_id": "telemetry:current",
        "provenance_digest": "A" * 64,
    }
    values.update(changes)
    return PathSnapshot(**values)  # type: ignore[arg-type]


def _action() -> NetworkAction:
    return NetworkAction(
        action_id="proposal-reroute",
        kind=ActionKind.REROUTE_TO_PATH,
        path_id="candidate",
    )


def test_snapshot_exposes_only_declared_numeric_fields() -> None:
    snapshot = _snapshot()

    assert snapshot.numeric("queue_occupancy") == 2.0
    assert snapshot.numeric("utilization") == 0.5
    with pytest.raises(KeyError, match="unsupported"):
        snapshot.numeric("provenance_digest")
    with pytest.raises(ValueError, match="available tick"):
        _snapshot(sample_tick=2, available_tick=1)
    with pytest.raises(ValueError, match="utilization"):
        _snapshot(utilization=float("nan"))
    with pytest.raises(ValueError, match="digest"):
        _snapshot(provenance_digest="short")
    with pytest.raises(ValueError, match="required"):
        _snapshot(path_id="")
    with pytest.raises(ValueError, match="negative"):
        _snapshot(queue_occupancy=-1)
    object.__setattr__(snapshot, "queue_occupancy", True)
    with pytest.raises(TypeError, match="not numeric"):
        snapshot.numeric("queue_occupancy")


def test_input_rejects_future_duplicate_and_mismatched_action_data() -> None:
    item = BaselineInput(
        pair_id="pair",
        scenario_id="scenario",
        split=DatasetSplit.TRAIN,
        seed=1,
        decision_tick=0,
        current_path_id="current",
        candidate_path_id="candidate",
        proposed_action=_action(),
        snapshots=(_snapshot(),),
        scenario_config_digest="B" * 64,
        schedule_digest="S" * 64,
    )

    assert item.snapshot("current") == item.snapshots[0]
    assert item.snapshot("missing") is None
    with pytest.raises(ValueError, match="future"):
        replace(item, snapshots=(_snapshot(available_tick=1),))
    with pytest.raises(ValueError, match="one snapshot"):
        replace(item, snapshots=(_snapshot(), _snapshot()))
    with pytest.raises(ValueError, match="target"):
        replace(
            item,
            proposed_action=NetworkAction(
                action_id="wrong",
                kind=ActionKind.REROUTE_TO_PATH,
                path_id="other",
            ),
        )
    with pytest.raises(ValueError, match="required"):
        replace(item, pair_id="")
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(item, seed=-1)
    with pytest.raises(ValueError, match="must differ"):
        replace(item, current_path_id="candidate")
    with pytest.raises(ValueError, match="too short"):
        replace(item, scenario_config_digest="short")
    with pytest.raises(ValueError, match="schedule digest"):
        replace(item, schedule_digest="short")


def test_observation_policy_enforces_fields_and_budgets() -> None:
    policy = ObservationPolicy(("queue_occupancy", "utilization"), 4, 32)

    assert policy.adaptive is False
    assert ObservationPolicy((), 0, 0).fields == ()
    with pytest.raises(ValueError, match="unique"):
        ObservationPolicy(("utilization", "utilization"), 2, 16)
    with pytest.raises(ValueError, match="unsupported"):
        ObservationPolicy(("true_queue",), 1, 8)
    with pytest.raises(ValueError, match="positive"):
        ObservationPolicy(("utilization",), 0, 8)
    with pytest.raises(ValueError, match="zero budgets"):
        ObservationPolicy((), 1, 8)


def test_result_schema_requires_consistent_disposition_and_costs() -> None:
    executed = BaselineResult(
        baseline_id=BaselineId.B1,
        pair_id="pair",
        scenario_id="scenario",
        disposition=BaselineDisposition.EXECUTE,
        reason=BaselineReason.NO_GATE,
        proposed_action=_action(),
        selected_action=_action(),
        observation_fields=(),
        telemetry_items=0,
        telemetry_bytes=0,
        inference_time_ns=1,
        configuration_digest="C" * 64,
        details=(("policy", "no-gate"),),
    )

    assert executed.selected_action == executed.proposed_action
    with pytest.raises(ValueError, match="requires"):
        replace(executed, selected_action=None)
    with pytest.raises(ValueError, match="cannot select"):
        BaselineResult(
            baseline_id=BaselineId.B4,
            pair_id="pair",
            scenario_id="scenario",
            disposition=BaselineDisposition.ABSTAIN,
            reason=BaselineReason.LOW_CONFIDENCE,
            proposed_action=_action(),
            selected_action=_action(),
            observation_fields=("utilization",),
            telemetry_items=1,
            telemetry_bytes=8,
            inference_time_ns=1,
            configuration_digest="D" * 64,
        )
    with pytest.raises(ValueError, match="canonical"):
        BaselineResult(
            baseline_id=BaselineId.B1,
            pair_id="pair",
            scenario_id="scenario",
            disposition=BaselineDisposition.EXECUTE,
            reason=BaselineReason.NO_GATE,
            proposed_action=_action(),
            selected_action=_action(),
            observation_fields=(),
            telemetry_items=0,
            telemetry_bytes=0,
            inference_time_ns=1,
            configuration_digest="E" * 64,
            details=(("z", "1"), ("a", "2")),
        )
    with pytest.raises(ValueError, match="required"):
        replace(executed, pair_id="")
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(executed, telemetry_bytes=-1)
    with pytest.raises(ValueError, match="too short"):
        replace(executed, configuration_digest="short")
    with pytest.raises(ValueError, match="finite"):
        replace(executed, score=float("nan"))
