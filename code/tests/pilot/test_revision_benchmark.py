"""M10 revision observational-row, leakage-boundary, and safety-gate tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ictc.baselines import CorpusAccessGuard
from ictc.datasets import (
    CorpusMatrixConfig,
    CorpusProvenance,
    DatasetSplit,
    RevisionCorpusProvenance,
    generate_m10r_scenario,
    load_corpus_matrix,
)
from ictc.domain.enums import ActionKind, AdmissionDecision
from ictc.pilot import (
    AblationId,
    ContextKey,
    ContextObservation,
    FixtureKind,
    FrozenRevisionProtocol,
    PilotDecision,
    PilotOutcomeTruth,
    PotentialOutcomeTruth,
    RevisionObservationalRow,
    RevisionSafetyDecision,
    RevisionSafetyReason,
    RevisionSafetyScope,
    TestAccessAuthorization,
    build_revision_benchmark_record,
    build_revision_observational_row,
    fit_revision_effect_model,
    load_revision_protocol,
    run_revision_safety_gate,
    score_revision_safety_decision,
    summarize_revision_safety,
)

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs/experiments/m10r_revision_matrix.json"
PROTOCOL = ROOT / "configs/experiments/m10r_revision_protocol.json"


def _provenance() -> RevisionCorpusProvenance:
    protocol = load_revision_protocol(PROTOCOL)
    return RevisionCorpusProvenance(
        CorpusProvenance("A" * 40, "B" * 64, "C" * 64),
        protocol.configuration_digest,
    )


def _generated_pair(
    tmp_path: Path, split: DatasetSplit
) -> tuple[Path, CorpusMatrixConfig, FrozenRevisionProtocol]:
    matrix = load_corpus_matrix(MATRIX)
    protocol = load_revision_protocol(PROTOCOL)
    scenario = next(item for item in matrix.scenarios() if item.split is split)
    result = generate_m10r_scenario(scenario, tmp_path, _provenance())
    return tmp_path / result.output_relpath, matrix, protocol


def test_revision_decision_row_uses_context_and_reads_no_outcomes(tmp_path: Path) -> None:
    pair, matrix, protocol = _generated_pair(tmp_path, DatasetSplit.DEVELOPMENT)
    guard = CorpusAccessGuard()
    row = build_revision_observational_row(
        pair,
        split=DatasetSplit.DEVELOPMENT,
        matrix=matrix,
        protocol=protocol,
        guard=guard,
    )

    assert row.decision.assignment_policy_id == "m10r_bounded_queue_pressure_assignment_v1"
    assert row.decision.observed_latency_ms is None
    assert row.topology_context.context_id.startswith("service=")
    assert not any(
        event.table.value in {"outcomes.parquet", "accounting.parquet"} for event in guard.events
    )
    assert not any(event.split is DatasetSplit.TEST for event in guard.events)


def test_revision_truth_join_selects_only_the_assigned_observation(tmp_path: Path) -> None:
    pair, matrix, protocol = _generated_pair(tmp_path, DatasetSplit.TRAIN)
    record = build_revision_benchmark_record(
        pair,
        split=DatasetSplit.TRAIN,
        matrix=matrix,
        protocol=protocol,
    )

    expected = (
        record.truth.reroute.mean_latency_ms
        if record.row.decision.assigned_action is ActionKind.REROUTE_TO_PATH
        else record.truth.keep.mean_latency_ms
    )
    assert record.row.decision.observed_latency_ms == expected
    assert record.selected_outcome.mean_latency_ms == expected
    if record.row.context_key is not None:
        assert record.training_observation().observed_latency_ms == expected


def _pilot_decision(row: RevisionObservationalRow, action: ActionKind) -> PilotDecision:
    return PilotDecision(
        opportunity_id=f"{row.decision.pair_id}::observed",
        pair_id=row.decision.pair_id,
        scenario_id=row.decision.scenario_id,
        split=row.decision.split,
        test_authorization_id=row.evaluation_authorization_id,
        ablation_id=AblationId.A0,
        configuration_id="default",
        fixture=FixtureKind.OBSERVED,
        decision=AdmissionDecision.ADMIT
        if action is ActionKind.REROUTE_TO_PATH
        else AdmissionDecision.DEFER,
        executable_action=action,
        reason_codes=(),
        reference_identified=True,
        model_identified=True,
        reference_contract_satisfied=True,
        contract_satisfied=True,
        estimable=True,
        estimate_point_ms=-5.0,
        estimate_lower_ms=-5.0,
        estimate_upper_ms=-5.0,
        telemetry_items=2,
        telemetry_bytes=128,
        contract_acquisition_latency_ms=0.0,
        protocol_messages=5,
        admission_wall_time_ns=1,
        admission_cpu_time_ns=1,
        breach_detection_time_ns=None,
        fallback_execution_time_ns=None,
        breach_detected=False,
        fallback_executed=action is ActionKind.KEEP_CURRENT_PATH,
    )


def test_revision_safety_gate_admits_only_supported_interval_after_base_gate(
    tmp_path: Path,
) -> None:
    pair, matrix, protocol = _generated_pair(tmp_path, DatasetSplit.DEVELOPMENT)
    row = build_revision_observational_row(
        pair,
        split=DatasetSplit.DEVELOPMENT,
        matrix=matrix,
        protocol=protocol,
    )
    key = row.context_key
    assert key is not None
    observations = tuple(
        ContextObservation(DatasetSplit.TRAIN, key, ActionKind.KEEP_CURRENT_PATH, 10.0)
        for _index in range(10)
    ) + tuple(
        ContextObservation(DatasetSplit.TRAIN, key, ActionKind.REROUTE_TO_PATH, 5.0)
        for _index in range(10)
    )
    model = fit_revision_effect_model(observations, minimum_per_arm=10)

    admitted = run_revision_safety_gate(
        _pilot_decision(row, ActionKind.REROUTE_TO_PATH),
        row,
        model,
        scope=RevisionSafetyScope.CONTEXT,
        upper_interval_threshold_ms=0.0,
    )
    strict = run_revision_safety_gate(
        _pilot_decision(row, ActionKind.REROUTE_TO_PATH),
        row,
        model,
        scope=RevisionSafetyScope.CONTEXT,
        upper_interval_threshold_ms=-10.0,
    )
    base_deferred = run_revision_safety_gate(
        _pilot_decision(row, ActionKind.KEEP_CURRENT_PATH),
        row,
        model,
        scope=RevisionSafetyScope.CONTEXT,
        upper_interval_threshold_ms=0.0,
    )

    assert admitted.reason is RevisionSafetyReason.SAFETY_GATE_ADMITTED
    assert admitted.final_executable_action is ActionKind.REROUTE_TO_PATH
    assert strict.reason is RevisionSafetyReason.INTERVAL_ABOVE_THRESHOLD
    assert strict.final_executable_action is ActionKind.KEEP_CURRENT_PATH
    assert base_deferred.reason is RevisionSafetyReason.BASE_CAUSAL_GATE_DEFERRED
    assert base_deferred.final_executable_action is ActionKind.KEEP_CURRENT_PATH


def test_revision_context_gate_never_falls_back_to_supported_global_estimate(
    tmp_path: Path,
) -> None:
    pair, matrix, protocol = _generated_pair(tmp_path, DatasetSplit.DEVELOPMENT)
    row = build_revision_observational_row(
        pair,
        split=DatasetSplit.DEVELOPMENT,
        matrix=matrix,
        protocol=protocol,
    )
    pressure = row.decision.pre_action_evidence.queue_pressure_bin
    assert pressure is not None
    different = ContextKey("different-context", row.decision.traffic_level_id, pressure)
    observations = tuple(
        ContextObservation(DatasetSplit.TRAIN, different, ActionKind.KEEP_CURRENT_PATH, 10.0)
        for _index in range(10)
    ) + tuple(
        ContextObservation(DatasetSplit.TRAIN, different, ActionKind.REROUTE_TO_PATH, 5.0)
        for _index in range(10)
    )
    model = fit_revision_effect_model(observations, minimum_per_arm=10)

    decision = run_revision_safety_gate(
        _pilot_decision(row, ActionKind.REROUTE_TO_PATH),
        row,
        model,
        scope=RevisionSafetyScope.CONTEXT,
        upper_interval_threshold_ms=0.0,
    )

    assert model.population_effect.supported is True
    assert decision.reason is RevisionSafetyReason.CONTEXT_UNAVAILABLE
    assert decision.final_executable_action is ActionKind.KEEP_CURRENT_PATH


def test_revision_builders_reject_sealed_evaluation_without_touching_disk(tmp_path: Path) -> None:
    matrix = load_corpus_matrix(MATRIX)
    protocol = load_revision_protocol(PROTOCOL)
    with pytest.raises(PermissionError, match="sealed"):
        build_revision_observational_row(
            tmp_path / "absent",
            split=DatasetSplit.TEST,
            matrix=matrix,
            protocol=protocol,
        )


def test_revision_builder_accepts_only_explicitly_authorized_evaluation(tmp_path: Path) -> None:
    matrix = load_corpus_matrix(MATRIX)
    protocol = load_revision_protocol(PROTOCOL)
    scenario = next(item for item in matrix.scenarios() if item.split is DatasetSplit.TEST)
    result = generate_m10r_scenario(
        scenario,
        tmp_path,
        _provenance(),
        evaluation_authorization_id="m10r-evaluation-unit-test",
    )
    authorization = TestAccessAuthorization(
        "m10r-evaluation-unit-test",
        "A" * 40,
        "B" * 40,
        "C" * 64,
        "D" * 64,
        1,
        (tmp_path / "result").resolve(),
    )

    row = build_revision_observational_row(
        tmp_path / result.output_relpath,
        split=DatasetSplit.TEST,
        matrix=matrix,
        protocol=protocol,
        test_authorization=authorization,
    )

    assert row.decision.split is DatasetSplit.TEST
    assert row.evaluation_authorization_id == authorization.authorization_id


def _outcome(latency_ms: float, *, reroute: bool) -> PilotOutcomeTruth:
    return PilotOutcomeTruth(
        mean_latency_ms=latency_ms,
        packet_delivery_ratio=1.0,
        packet_loss_ratio=0.0,
        jitter_ms=0.0,
        throughput_packets_per_tick=1.0,
        deadline_violation_rate=0.0,
        route_changes=int(reroute),
    )


def _truth(pair_id: str, *, keep_ms: float, reroute_ms: float) -> PotentialOutcomeTruth:
    return PotentialOutcomeTruth(
        pair_id,
        0,
        _outcome(keep_ms, reroute=False),
        _outcome(reroute_ms, reroute=True),
    )


def test_revision_safety_score_uses_paired_truth_only_after_decision() -> None:
    decision = RevisionSafetyDecision(
        opportunity_id="pair-1::observed",
        pair_id="pair-1",
        scenario_id="scenario-1",
        split=DatasetSplit.DEVELOPMENT,
        scope=RevisionSafetyScope.CONTEXT,
        configuration_id="context-local:upper<=0ms",
        upper_interval_threshold_ms=0.0,
        base_executable_action=ActionKind.REROUTE_TO_PATH,
        final_executable_action=ActionKind.KEEP_CURRENT_PATH,
        reason=RevisionSafetyReason.INTERVAL_ABOVE_THRESHOLD,
        support_status=None,
        estimate_point_ms=1.0,
        estimate_lower_ms=0.5,
        estimate_upper_ms=1.5,
        reference_identified=True,
        reference_contract_satisfied=True,
    )

    score = score_revision_safety_decision(decision, _truth("pair-1", keep_ms=10.0, reroute_ms=5.0))

    assert score.base_reroute is True
    assert score.executed_reroute is False
    assert score.unnecessary_safety_deferral is True
    assert score.selected_latency_ms == 10.0


def test_revision_safety_summary_enforces_admissibility_and_retention() -> None:
    template = RevisionSafetyDecision(
        opportunity_id="pair-0::observed",
        pair_id="pair-0",
        scenario_id="scenario-0",
        split=DatasetSplit.DEVELOPMENT,
        scope=RevisionSafetyScope.CONTEXT,
        configuration_id="context-local:upper<=0ms",
        upper_interval_threshold_ms=0.0,
        base_executable_action=ActionKind.REROUTE_TO_PATH,
        final_executable_action=ActionKind.KEEP_CURRENT_PATH,
        reason=RevisionSafetyReason.INTERVAL_ABOVE_THRESHOLD,
        support_status=None,
        estimate_point_ms=1.0,
        estimate_lower_ms=0.5,
        estimate_upper_ms=1.5,
        reference_identified=True,
        reference_contract_satisfied=True,
    )
    truths = tuple(
        _truth(f"pair-{index}", keep_ms=10.0, reroute_ms=5.0 if index == 0 else 15.0)
        for index in range(4)
    )
    context = tuple(
        replace(
            template,
            opportunity_id=f"pair-{index}::observed",
            pair_id=f"pair-{index}",
            scenario_id=f"scenario-{index}",
            final_executable_action=(
                ActionKind.REROUTE_TO_PATH if index == 0 else ActionKind.KEEP_CURRENT_PATH
            ),
            reason=(
                RevisionSafetyReason.SAFETY_GATE_ADMITTED
                if index == 0
                else RevisionSafetyReason.INTERVAL_ABOVE_THRESHOLD
            ),
        )
        for index in range(4)
    )
    inadmissible = tuple(
        replace(
            decision,
            scope=RevisionSafetyScope.GLOBAL,
            configuration_id="global:upper<=0ms",
            final_executable_action=ActionKind.REROUTE_TO_PATH,
            reason=RevisionSafetyReason.SAFETY_GATE_ADMITTED,
            reference_identified=False,
        )
        for decision in context
    )
    decisions = context + inadmissible
    scores = tuple(
        score_revision_safety_decision(decision, truths[index % len(truths)])
        for index, decision in enumerate(decisions)
    )

    summary = summarize_revision_safety(
        decisions,
        scores,
        minimum_retained_base_reroute_fraction=0.25,
    )
    candidates = summary["candidates"]
    assert isinstance(candidates, dict)
    context_summary = candidates["context-local:upper<=0ms"]
    global_summary = candidates["global:upper<=0ms"]
    assert isinstance(context_summary, dict)
    assert isinstance(global_summary, dict)
    assert context_summary["retained_base_reroute_fraction"] == 0.25
    assert context_summary["harmful_event_burden_rate"] == 0.0
    assert context_summary["eligible"] is True
    assert global_summary["false_causal_admission_count"] == 4
    assert global_summary["eligible"] is False
    assert summary["selected_configuration_id"] == "context-local:upper<=0ms"
