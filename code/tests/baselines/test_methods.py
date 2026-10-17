"""B0-B9 behavior, fidelity properties, split guards, and failure tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ictc.baselines import (
    AdaptiveCoverageController,
    BaselineDisposition,
    BaselineId,
    BaselineInput,
    BaselineReason,
    ConstraintOnlyController,
    CorpusAccessGuard,
    CorrelationSelectorController,
    DigitalTwinController,
    FixedCausalController,
    FixedTelemetryController,
    LinearClassifier,
    LinearRegressor,
    NoGateController,
    ObservationPolicy,
    OracleChoice,
    OracleController,
    OutcomeMetrics,
    PairedOutcomes,
    PathSnapshot,
    SelectiveConfidenceController,
    StaticContractController,
    TrainingExample,
    extract_feature_names,
    tune_confidence_threshold,
    tune_correlation_top_k,
    tune_digital_twin_alpha,
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
    current_queue: int = 10,
    candidate_queue: int = 2,
    current_utilization: float = 0.8,
    candidate_utilization: float = 0.4,
    candidate_present: bool = True,
    stale: bool = False,
    age: int = 0,
    index: int = 0,
) -> BaselineInput:
    def snapshot(path: str, queue: int, utilization: float) -> PathSnapshot:
        return PathSnapshot(
            path,
            0,
            0,
            -age,
            queue,
            utilization,
            age,
            age,
            0,
            stale,
            f"telemetry:{path}",
            f"{index + 1:064X}",
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


def example(
    index: int,
    *,
    beneficial: bool,
    split: DatasetSplit = DatasetSplit.TRAIN,
) -> TrainingExample:
    item = make_input(
        split=split,
        current_queue=10,
        candidate_queue=index if beneficial else 20 + index,
        current_utilization=0.5,
        candidate_utilization=0.5,
        index=index + (1000 if split is DatasetSplit.DEVELOPMENT else 0),
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
    return TrainingExample(item, PairedOutcomes(item.pair_id, item.scenario_id, keep, reroute))


def examples(split: DatasetSplit = DatasetSplit.TRAIN) -> tuple[TrainingExample, ...]:
    return tuple(example(index, beneficial=index < 10, split=split) for index in range(20))


def test_b0_oracle_uses_realized_objective_and_handles_unknown_pair(tmp_path: Path) -> None:
    item = make_input(index=1)
    reroute = OracleController((OracleChoice(item.pair_id, True, 10, 5, 20),))
    keep = OracleController((OracleChoice(item.pair_id, False, 5, 10, 20),))

    assert reroute.decide(item).selected_action == item.proposed_action
    keep_result = keep.decide(item)
    assert keep_result.selected_action is not None
    assert keep_result.selected_action.kind is ActionKind.KEEP_CURRENT_PATH
    assert reroute.decide(replace(item, pair_id="unknown")).disposition is BaselineDisposition.FAIL

    scenario = load_corpus_matrix(MICRO).scenarios()[0]
    target = tmp_path / "oracle"
    export_corpus_pair(
        scenario,
        run_paired(scenario.simulation),
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        target,
    )
    guard = CorpusAccessGuard()
    controller = OracleController.from_corpus((target,), split=scenario.split, guard=guard)
    assert len(controller.choices) == 1
    assert any(event.table.value == "oracle.parquet" for event in guard.events)
    guard.assert_no_denied_access()


def test_b0_validation_rejects_bad_choices_and_weights() -> None:
    with pytest.raises(ValueError, match="pair ID"):
        OracleChoice("", True, 1, 2, 1)
    with pytest.raises(ValueError, match="finite"):
        OracleChoice("pair", True, float("nan"), 2, 1)
    with pytest.raises(ValueError, match="nonempty"):
        OracleController(())
    choice = OracleChoice("pair", True, 1, 2, 1)
    with pytest.raises(ValueError, match="pair-unique"):
        OracleController((choice, choice))
    with pytest.raises(ValueError, match="negative"):
        OracleController((choice,), loss_weight=-1)


def test_b1_always_executes_proposal_without_admission_telemetry() -> None:
    result = NoGateController().decide(make_input())

    assert result.baseline_id is BaselineId.B1
    assert result.disposition is BaselineDisposition.EXECUTE
    assert result.reason is BaselineReason.NO_GATE
    assert result.telemetry_items == 0
    assert result.selected_action == result.proposed_action


def test_b2_fixed_telemetry_accepts_rejects_and_abstains_on_missing() -> None:
    controller = FixedTelemetryController(minimum_improvement=1)
    accepted = controller.decide(make_input())
    rejected = controller.decide(
        make_input(current_queue=1, candidate_queue=10, current_utilization=0.1)
    )
    missing = controller.decide(make_input(candidate_present=False))

    assert accepted.reason is BaselineReason.FIXED_TELEMETRY_ACCEPTED
    assert accepted.telemetry_items == 4
    assert rejected.reason is BaselineReason.PREDICTED_HARM
    assert rejected.disposition is BaselineDisposition.ABSTAIN
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    with pytest.raises(ValueError, match="finite"):
        FixedTelemetryController(minimum_improvement=float("nan"))
    with pytest.raises(ValueError, match="nonnegative"):
        FixedTelemetryController(utilization_weight=-1)


def test_b3_adapts_under_budget_and_applies_switching_penalty() -> None:
    controller = AdaptiveCoverageController(maximum_selected_fields=2)
    item = make_input(current_queue=20, candidate_queue=0)
    first = controller.decide(item)
    prior = controller.previous_fields
    second = controller.decide(item)

    assert first.reason is BaselineReason.ADAPTIVE_UTILITY_ACCEPTED
    assert len(first.observation_fields) == 2
    assert first.telemetry_items <= controller.observation_policy.maximum_items
    assert controller.previous_fields == prior == second.observation_fields
    missing = AdaptiveCoverageController().decide(make_input(candidate_present=False))
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    rejected = AdaptiveCoverageController(minimum_score=2).decide(item)
    assert rejected.reason is BaselineReason.PREDICTED_HARM


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"field_weights": ()}, "nonempty"),
        ({"field_weights": (("unknown", 1.0),)}, "unsupported"),
        ({"field_weights": (("utilization", -1.0),)}, "nonnegative"),
        ({"maximum_selected_fields": 99}, "outside"),
        ({"freshness_penalty": -1}, "cannot be negative"),
    ],
)
def test_b3_configuration_validation(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AdaptiveCoverageController(**kwargs)  # type: ignore[arg-type]


def test_linear_classifier_probability_and_validation() -> None:
    model = LinearClassifier((0,), (1,), (2,), 0)

    assert model.probability((1,)) > 0.5
    assert model.probability((-1,)) < 0.5
    with pytest.raises(ValueError, match="lengths"):
        LinearClassifier((), (), (), 0)
    with pytest.raises(ValueError, match="positive"):
        LinearClassifier((0,), (0,), (1,), 0)
    with pytest.raises(ValueError, match="finite"):
        LinearClassifier((0,), (1,), (float("nan"),), 0)
    with pytest.raises(ValueError, match="width"):
        model.probability((1, 2))


def test_b4_fits_training_only_and_selectively_accepts() -> None:
    controller = SelectiveConfidenceController.fit(examples(), confidence_threshold=0.5)
    beneficial = controller.decide(example(1, beneficial=True).input)
    harmful = controller.decide(example(15, beneficial=False).input)
    missing = controller.decide(make_input(candidate_present=False))

    assert beneficial.reason is BaselineReason.CONFIDENCE_ACCEPTED
    assert beneficial.score is not None and beneficial.score >= 0.5
    assert harmful.reason is BaselineReason.PREDICTED_HARM
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    with pytest.raises(ValueError, match="training split"):
        SelectiveConfidenceController.fit(examples(DatasetSplit.DEVELOPMENT))
    same_class = tuple(example(index, beneficial=True) for index in range(4))
    with pytest.raises(ValueError, match="both beneficial"):
        SelectiveConfidenceController.fit(same_class)


def test_b4_development_threshold_tuning_and_validation() -> None:
    controller = SelectiveConfidenceController.fit(examples(), confidence_threshold=0.9)
    tuned = tune_confidence_threshold(
        controller,
        examples(DatasetSplit.DEVELOPMENT),
        candidates=(0.5, 0.7, 0.9),
        maximum_selective_error=0.1,
    )

    assert tuned.confidence_threshold in {0.5, 0.7, 0.9}
    with pytest.raises(ValueError, match="development split"):
        tune_confidence_threshold(controller, examples())
    with pytest.raises(ValueError, match="candidate"):
        tune_confidence_threshold(controller, examples(DatasetSplit.DEVELOPMENT), candidates=(0.4,))
    with pytest.raises(ValueError, match="selective error"):
        tune_confidence_threshold(
            controller,
            examples(DatasetSplit.DEVELOPMENT),
            maximum_selective_error=2,
        )
    with pytest.raises(ValueError, match="threshold"):
        replace(controller, confidence_threshold=0.4)
    with pytest.raises(ValueError, match="positive"):
        replace(controller, regularization_c=0)


def test_b5_fixed_causal_effect_requires_supported_negative_interval() -> None:
    beneficial = tuple(example(index, beneficial=True) for index in range(10))
    harmful = tuple(example(index, beneficial=False) for index in range(10))
    accepted = FixedCausalController.fit(beneficial)
    rejected = FixedCausalController.fit(harmful)
    too_small = FixedCausalController.fit((example(1, beneficial=True),))

    assert accepted.decide(make_input()).reason is BaselineReason.CAUSAL_EFFECT_SUPPORTED
    assert rejected.decide(make_input()).reason is BaselineReason.CAUSAL_EFFECT_UNSUPPORTED
    assert too_small.identified is False
    assert accepted.graph_id == "routing_telemetry_v1"
    assert len(accepted.graph_digest) == 64
    assert accepted.adjustment_set == ()
    with pytest.raises(ValueError, match="presence"):
        FixedCausalController(True, None, None, 2)
    with pytest.raises(ValueError, match="negative"):
        FixedCausalController(False, None, None, -1)
    with pytest.raises(ValueError, match="graph ID"):
        replace(accepted, graph_id="")
    with pytest.raises(ValueError, match="canonical"):
        replace(accepted, adjustment_set=("QueueOccupancy", "QueueOccupancy"))


def test_b6_constraint_only_controller_enforces_peak_average_and_queue() -> None:
    controller = ConstraintOnlyController(
        peak_utilization_limit=0.9,
        average_utilization_limit=0.8,
        queue_limit=10,
    )
    accepted = controller.decide(make_input(candidate_queue=2, candidate_utilization=0.4))
    peak = controller.decide(make_input(candidate_utilization=1.0))
    queue = controller.decide(make_input(candidate_queue=11))
    missing = controller.decide(make_input(candidate_present=False))

    assert accepted.reason is BaselineReason.CONSTRAINTS_SATISFIED
    assert peak.reason is BaselineReason.CONSTRAINT_VIOLATION
    assert queue.reason is BaselineReason.CONSTRAINT_VIOLATION
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    with pytest.raises(ValueError, match="cannot exceed"):
        ConstraintOnlyController(peak_utilization_limit=0.5, average_utilization_limit=0.6)
    with pytest.raises(ValueError, match="positive"):
        ConstraintOnlyController(peak_utilization_limit=0, average_utilization_limit=0)


def test_b7_static_contract_is_invariant_and_availability_gated() -> None:
    controller = StaticContractController(maximum_age_ticks=0)
    accepted = controller.decide(make_input())
    stale = controller.decide(make_input(stale=True))
    old = controller.decide(make_input(age=1))
    missing = controller.decide(make_input(candidate_present=False))

    assert accepted.reason is BaselineReason.STATIC_CONTRACT_SATISFIED
    assert stale.reason is BaselineReason.STATIC_CONTRACT_UNAVAILABLE
    assert old.reason is BaselineReason.STATIC_CONTRACT_UNAVAILABLE
    assert missing.reason is BaselineReason.STATIC_CONTRACT_UNAVAILABLE
    assert accepted.observation_fields == stale.observation_fields
    with pytest.raises(ValueError, match="cannot be negative"):
        StaticContractController(maximum_age_ticks=-1)


def test_linear_regressor_and_feature_name_validation() -> None:
    model = LinearRegressor((0,), (1,), (2,), 1)

    assert model.predict((3,)) == 7
    assert extract_feature_names(("utilization",))[-1] == "both_present.utilization"
    with pytest.raises(ValueError, match="lengths"):
        LinearRegressor((), (), (), 0)
    with pytest.raises(ValueError, match="positive"):
        LinearRegressor((0,), (0,), (1,), 0)
    with pytest.raises(ValueError, match="width"):
        model.predict((1, 2))


def test_b8_twin_fits_what_if_actions_and_selects_best_prediction() -> None:
    beneficial = tuple(example(index, beneficial=True) for index in range(10))
    harmful = tuple(example(index, beneficial=False) for index in range(10))
    reroute_twin = DigitalTwinController.fit(beneficial)
    keep_twin = DigitalTwinController.fit(harmful)

    reroute = reroute_twin.decide(make_input())
    keep = keep_twin.decide(make_input())
    missing = reroute_twin.decide(make_input(candidate_present=False))
    assert reroute.selected_action is not None
    assert reroute.selected_action.kind is ActionKind.REROUTE_TO_PATH
    assert keep.selected_action is not None
    assert keep.selected_action.kind is ActionKind.KEEP_CURRENT_PATH
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    with pytest.raises(ValueError, match="training split"):
        DigitalTwinController.fit(examples(DatasetSplit.DEVELOPMENT))
    with pytest.raises(ValueError, match="cannot be negative"):
        replace(reroute_twin, ridge_alpha=-1)


def test_b8_ridge_alpha_is_selected_on_development_only() -> None:
    tuned = tune_digital_twin_alpha(
        examples(),
        examples(DatasetSplit.DEVELOPMENT),
        candidates=(0.0, 1.0),
    )

    assert tuned.ridge_alpha in {0.0, 1.0}
    with pytest.raises(ValueError, match="development split"):
        tune_digital_twin_alpha(examples(), examples())
    with pytest.raises(ValueError, match="alpha candidates"):
        tune_digital_twin_alpha(examples(), examples(DatasetSplit.DEVELOPMENT), candidates=(-1.0,))


def test_b9_selects_predictive_noncausal_feature_and_freezes_top_k() -> None:
    controller = CorrelationSelectorController.fit(examples(), top_k=1)
    beneficial = controller.decide(example(1, beneficial=True).input)
    harmful = controller.decide(example(15, beneficial=False).input)
    missing = controller.decide(make_input(candidate_present=False))

    assert controller.selected_fields == ("queue_occupancy",)
    assert beneficial.reason is BaselineReason.FEATURE_SELECTOR_ACCEPTED
    assert harmful.reason is BaselineReason.PREDICTED_HARM
    assert missing.reason is BaselineReason.REQUIRED_OBSERVATION_MISSING
    with pytest.raises(ValueError, match="outside"):
        CorrelationSelectorController.fit(examples(), top_k=0)
    with pytest.raises(ValueError, match="two outcome classes"):
        CorrelationSelectorController.fit(
            tuple(example(index, beneficial=True) for index in range(10)), top_k=1
        )


def test_b9_top_k_is_selected_on_development_only() -> None:
    tuned = tune_correlation_top_k(
        examples(),
        examples(DatasetSplit.DEVELOPMENT),
        candidates=(1, 2),
    )

    assert tuned.top_k in {1, 2}
    with pytest.raises(ValueError, match="development split"):
        tune_correlation_top_k(examples(), examples())
    with pytest.raises(ValueError, match="outside"):
        tune_correlation_top_k(examples(), examples(DatasetSplit.DEVELOPMENT), candidates=(0,))


def test_all_controllers_expose_common_policy_and_digest() -> None:
    simple = (
        NoGateController(),
        FixedTelemetryController(),
        AdaptiveCoverageController(),
        ConstraintOnlyController(),
        StaticContractController(),
    )

    assert all(
        isinstance(controller.observation_policy, ObservationPolicy) for controller in simple
    )
    assert all(len(controller.configuration_digest) == 64 for controller in simple)
