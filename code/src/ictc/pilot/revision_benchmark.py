"""Leakage-separated observational rows and safety gates for the M10 revision."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path

from ictc.baselines import CorpusAccessGuard
from ictc.datasets import CorpusMatrixConfig, DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.pilot.ablation import PilotDecision
from ictc.pilot.authorization import TestAccessAuthorization
from ictc.pilot.benchmark import (
    ASSIGNMENT_POLICY_ID,
    ObservationalDecision,
    PilotOutcomeTruth,
    PotentialOutcomeTruth,
    build_repair_decision,
    build_repair_record,
)
from ictc.pilot.revision import (
    M10R_ASSIGNMENT_POLICY_ID,
    ContextEffect,
    ContextKey,
    ContextObservation,
    EffectSupportStatus,
    PopulationEffect,
    RevisionEffectModel,
    TopologyContext,
    TrafficEffect,
    fit_revision_effect_model,
    revision_identification,
)
from ictc.pilot.revision_protocol import FrozenRevisionProtocol, revision_assignment


class RevisionSafetyScope(StrEnum):
    GLOBAL = "global"
    TRAFFIC = "traffic-only"
    CONTEXT = "context-local"


class RevisionSafetyReason(StrEnum):
    BASE_CAUSAL_GATE_DEFERRED = "BASE_CAUSAL_GATE_DEFERRED"
    CONTEXT_UNAVAILABLE = "CONTEXT_UNAVAILABLE"
    ESTIMATOR_INESTIMABLE = "ESTIMATOR_INESTIMABLE"
    BELOW_MINIMUM_PER_ARM = "BELOW_MINIMUM_PER_ARM"
    INTERVAL_ABOVE_THRESHOLD = "INTERVAL_ABOVE_THRESHOLD"
    SAFETY_GATE_ADMITTED = "SAFETY_GATE_ADMITTED"


@dataclass(frozen=True, slots=True)
class RevisionObservationalRow:
    """Controller-safe context plus one bounded observational assignment row."""

    decision: ObservationalDecision
    topology_context: TopologyContext
    assignment_queue_pressure_bin: int
    evaluation_authorization_id: str | None = None

    def __post_init__(self) -> None:
        if self.decision.split is DatasetSplit.TEST and not self.evaluation_authorization_id:
            raise PermissionError("sealed M10 revision evaluation rows are not authorized")
        if self.decision.split is not DatasetSplit.TEST and self.evaluation_authorization_id:
            raise PermissionError("evaluation authorization cannot label non-test rows")
        if self.decision.assignment_policy_id != M10R_ASSIGNMENT_POLICY_ID:
            raise ValueError("revision row assignment policy ID mismatch")
        if self.assignment_queue_pressure_bin not in {-1, 0, 1}:
            raise ValueError("revision assignment pressure bin must be -1, 0, or 1")

    @property
    def context_key(self) -> ContextKey | None:
        pressure = self.decision.pre_action_evidence.queue_pressure_bin
        if pressure is None:
            return None
        return ContextKey(
            self.topology_context.context_id,
            self.decision.traffic_level_id,
            pressure,
        )


@dataclass(frozen=True, slots=True)
class RevisionBenchmarkRecord:
    """Privileged train/development truth joined only after decision persistence."""

    row: RevisionObservationalRow
    truth: PotentialOutcomeTruth

    def __post_init__(self) -> None:
        if self.row.decision.pair_id != self.truth.pair_id:
            raise ValueError("revision row and truth pair IDs differ")
        if self.row.assignment_queue_pressure_bin != self.truth.true_queue_pressure_bin:
            raise ValueError("revision assignment pressure disagrees with paired truth")

    @property
    def selected_outcome(self) -> PilotOutcomeTruth:
        return (
            self.truth.reroute
            if self.row.decision.assigned_action is ActionKind.REROUTE_TO_PATH
            else self.truth.keep
        )

    def training_observation(self) -> ContextObservation:
        if self.row.decision.split is not DatasetSplit.TRAIN:
            raise PermissionError("revision effect fitting accepts training records only")
        key = self.row.context_key
        if key is None:
            raise ValueError("revision effect fitting requires acquired queue pressure")
        return ContextObservation(
            DatasetSplit.TRAIN,
            key,
            self.row.decision.assigned_action,
            self.selected_outcome.mean_latency_ms,
            cluster_id=str(self.row.decision.seed),
        )


@dataclass(frozen=True, slots=True)
class RevisionSafetyDecision:
    """One pre-truth safety overlay applied only after the causal A0 gate."""

    opportunity_id: str
    pair_id: str
    scenario_id: str
    split: DatasetSplit
    scope: RevisionSafetyScope
    configuration_id: str
    upper_interval_threshold_ms: float
    base_executable_action: ActionKind | None
    final_executable_action: ActionKind | None
    reason: RevisionSafetyReason
    support_status: EffectSupportStatus | None
    estimate_point_ms: float | None
    estimate_lower_ms: float | None
    estimate_upper_ms: float | None
    reference_identified: bool
    reference_contract_satisfied: bool
    test_authorization_id: str | None = None

    def __post_init__(self) -> None:
        if self.split is DatasetSplit.TEST and not self.test_authorization_id:
            raise PermissionError("sealed M10 revision safety decisions are not authorized")
        if self.split is not DatasetSplit.TEST and self.test_authorization_id:
            raise PermissionError("test authorization cannot label non-test safety decisions")
        if not math.isfinite(self.upper_interval_threshold_ms):
            raise ValueError("revision safety threshold must be finite")
        if self.final_executable_action is ActionKind.REROUTE_TO_PATH and (
            self.reason is not RevisionSafetyReason.SAFETY_GATE_ADMITTED
        ):
            raise ValueError("revision reroute requires an admitted safety gate")


@dataclass(frozen=True, slots=True)
class RevisionSafetyScore:
    """Post-persistence paired score with explicit endpoint denominators."""

    opportunity_id: str
    pair_id: str
    scope: RevisionSafetyScope
    configuration_id: str
    base_reroute: bool
    executed_reroute: bool
    retained_base_reroute: bool
    beneficial_base_reroute: bool
    false_causal_admission: bool
    harmful_action: bool
    unnecessary_safety_deferral: bool
    selected_latency_ms: float


def _topology_context(matrix: CorpusMatrixConfig, topology_family_id: str) -> TopologyContext:
    families = {family.family_id: family for family in matrix.topology_families}
    try:
        family = families[topology_family_id]
    except KeyError as error:
        raise ValueError("revision topology family is absent from the frozen matrix") from error
    return TopologyContext.from_path_configuration(
        current_service_packets_per_tick=family.current_service_packets_per_tick,
        candidate_service_packets_per_tick=family.candidate_service_packets_per_tick,
        current_propagation_delay_ms=family.current_propagation_delay_ms,
        candidate_propagation_delay_ms=family.candidate_propagation_delay_ms,
        current_queue_limit_packets=family.current_queue_limit_packets,
        candidate_queue_limit_packets=family.candidate_queue_limit_packets,
    )


def _assignment_pressure(legacy: ObservationalDecision) -> int:
    if legacy.assignment_policy_id != ASSIGNMENT_POLICY_ID:
        raise ValueError("legacy assignment policy compatibility boundary changed")
    by_probability = {0.25: -1, 0.5: 0, 0.75: 1}
    try:
        return by_probability[legacy.assignment_probability]
    except KeyError as error:
        raise ValueError("legacy assignment probability cannot recover the pressure bin") from error


def _convert_decision(
    legacy: ObservationalDecision,
    matrix: CorpusMatrixConfig,
    protocol: FrozenRevisionProtocol,
    *,
    observed_latency_ms: float | None,
    evaluation_authorization_id: str | None = None,
) -> RevisionObservationalRow:
    pressure = _assignment_pressure(legacy)
    action, probability, uniform = revision_assignment(legacy.pair_id, pressure, protocol)
    decision = replace(
        legacy,
        assigned_action=action,
        observed_latency_ms=observed_latency_ms,
        reference_identification=revision_identification(
            legacy.pre_action_evidence.queue_pressure_bin is not None
        ),
        assignment_probability=probability,
        assignment_uniform=uniform,
        assignment_policy_id=M10R_ASSIGNMENT_POLICY_ID,
    )
    return RevisionObservationalRow(
        decision,
        _topology_context(matrix, legacy.topology_family_id),
        pressure,
        evaluation_authorization_id,
    )


def build_revision_observational_row(
    directory: Path,
    *,
    split: DatasetSplit,
    matrix: CorpusMatrixConfig,
    protocol: FrozenRevisionProtocol,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> RevisionObservationalRow:
    """Build a decision-time row without opening outcomes or accounting tables."""
    if split is DatasetSplit.TEST and test_authorization is None:
        raise PermissionError("sealed M10 revision evaluation rows are not authorized")
    legacy = build_repair_decision(
        directory,
        split=split,
        guard=guard,
        test_authorization=test_authorization,
    )
    return _convert_decision(
        legacy,
        matrix,
        protocol,
        observed_latency_ms=None,
        evaluation_authorization_id=(
            test_authorization.authorization_id if test_authorization else None
        ),
    )


def build_revision_benchmark_record(
    directory: Path,
    *,
    split: DatasetSplit,
    matrix: CorpusMatrixConfig,
    protocol: FrozenRevisionProtocol,
    guard: CorpusAccessGuard | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> RevisionBenchmarkRecord:
    """Join privileged truth for training or post-persistence development scoring."""
    if split is DatasetSplit.TEST and test_authorization is None:
        raise PermissionError("sealed M10 revision evaluation truth is not authorized")
    legacy = build_repair_record(
        directory,
        split=split,
        guard=guard,
        test_authorization=test_authorization,
    )
    pressure = _assignment_pressure(legacy.decision)
    action, _probability, _uniform = revision_assignment(
        legacy.decision.pair_id, pressure, protocol
    )
    observed = (
        legacy.truth.reroute.mean_latency_ms
        if action is ActionKind.REROUTE_TO_PATH
        else legacy.truth.keep.mean_latency_ms
    )
    row = _convert_decision(
        legacy.decision,
        matrix,
        protocol,
        observed_latency_ms=observed,
        evaluation_authorization_id=(
            test_authorization.authorization_id if test_authorization else None
        ),
    )
    return RevisionBenchmarkRecord(row, legacy.truth)


def fit_revision_benchmark_model(
    records: tuple[RevisionBenchmarkRecord, ...], *, minimum_per_arm: int
) -> RevisionEffectModel:
    """Fit only complete training rows whose adjustment variable was acquired."""
    observations = tuple(
        record.training_observation()
        for record in records
        if record.row.decision.observation_regime_id == "complete"
        and record.row.context_key is not None
    )
    return fit_revision_effect_model(observations, minimum_per_arm=minimum_per_arm)


def _effect_for(
    row: RevisionObservationalRow,
    model: RevisionEffectModel,
    scope: RevisionSafetyScope,
) -> PopulationEffect | TrafficEffect | ContextEffect | None:
    if scope is RevisionSafetyScope.GLOBAL:
        return model.population_effect
    if scope is RevisionSafetyScope.TRAFFIC:
        return model.traffic_effect_for(row.decision.traffic_level_id)
    key = row.context_key
    return model.contextual_effects.effect_for(key) if key is not None else None


def run_revision_safety_gate(
    base: PilotDecision,
    row: RevisionObservationalRow,
    model: RevisionEffectModel,
    *,
    scope: RevisionSafetyScope,
    upper_interval_threshold_ms: float,
) -> RevisionSafetyDecision:
    """Apply one mean-effect safety candidate after, and never instead of, base A0."""
    if base.pair_id != row.decision.pair_id or base.split is not row.decision.split:
        raise ValueError("base causal decision and revision row differ")
    configuration_id = f"{scope.value}:upper<={upper_interval_threshold_ms:g}ms"
    if base.executable_action is not ActionKind.REROUTE_TO_PATH:
        return RevisionSafetyDecision(
            base.opportunity_id,
            base.pair_id,
            base.scenario_id,
            base.split,
            scope,
            configuration_id,
            upper_interval_threshold_ms,
            base.executable_action,
            base.executable_action,
            RevisionSafetyReason.BASE_CAUSAL_GATE_DEFERRED,
            None,
            None,
            None,
            None,
            base.reference_identified,
            base.reference_contract_satisfied,
            base.test_authorization_id,
        )

    effect = _effect_for(row, model, scope)
    if effect is None:
        reason = RevisionSafetyReason.CONTEXT_UNAVAILABLE
        support_status = None
        estimate = None
    elif not effect.supported:
        support_status = effect.support_status
        reason = (
            RevisionSafetyReason.ESTIMATOR_INESTIMABLE
            if support_status is EffectSupportStatus.ESTIMATOR_INESTIMABLE
            else RevisionSafetyReason.BELOW_MINIMUM_PER_ARM
        )
        estimate = effect.result.estimate
    else:
        support_status = effect.support_status
        estimate = effect.result.estimate
        assert estimate is not None
        reason = (
            RevisionSafetyReason.SAFETY_GATE_ADMITTED
            if estimate.uncertainty.upper <= upper_interval_threshold_ms
            else RevisionSafetyReason.INTERVAL_ABOVE_THRESHOLD
        )
    admitted = reason is RevisionSafetyReason.SAFETY_GATE_ADMITTED
    return RevisionSafetyDecision(
        base.opportunity_id,
        base.pair_id,
        base.scenario_id,
        base.split,
        scope,
        configuration_id,
        upper_interval_threshold_ms,
        base.executable_action,
        ActionKind.REROUTE_TO_PATH if admitted else ActionKind.KEEP_CURRENT_PATH,
        reason,
        support_status,
        estimate.point_estimate if estimate is not None else None,
        estimate.uncertainty.lower if estimate is not None else None,
        estimate.uncertainty.upper if estimate is not None else None,
        base.reference_identified,
        base.reference_contract_satisfied,
        base.test_authorization_id,
    )


def score_revision_safety_decision(
    decision: RevisionSafetyDecision, truth: PotentialOutcomeTruth
) -> RevisionSafetyScore:
    """Score one already-persisted safety decision against paired truth."""
    if decision.pair_id != truth.pair_id:
        raise ValueError("revision safety decision and truth pair IDs differ")
    base_reroute = decision.base_executable_action is ActionKind.REROUTE_TO_PATH
    executed = decision.final_executable_action is ActionKind.REROUTE_TO_PATH
    selected = truth.reroute if executed else truth.keep
    reference_admissible = decision.reference_identified and decision.reference_contract_satisfied
    return RevisionSafetyScore(
        decision.opportunity_id,
        decision.pair_id,
        decision.scope,
        decision.configuration_id,
        base_reroute,
        executed,
        base_reroute and executed,
        base_reroute and truth.reroute_beneficial,
        executed and not reference_admissible,
        executed and not truth.reroute_beneficial,
        base_reroute and not executed and truth.reroute_beneficial,
        selected.mean_latency_ms,
    )


def summarize_revision_safety(
    decisions: tuple[RevisionSafetyDecision, ...],
    scores: tuple[RevisionSafetyScore, ...],
    *,
    minimum_retained_base_reroute_fraction: float,
) -> dict[str, object]:
    """Aggregate frozen endpoints and select a development candidate without truth leakage."""
    if not decisions or len(decisions) != len(scores):
        raise ValueError("revision safety decisions and scores must be nonempty and aligned")
    if not 0 < minimum_retained_base_reroute_fraction <= 1:
        raise ValueError("revision retained-reroute threshold must lie in (0, 1]")
    groups: defaultdict[
        tuple[RevisionSafetyScope, str],
        list[tuple[RevisionSafetyDecision, RevisionSafetyScore]],
    ] = defaultdict(list)
    for decision, score in zip(decisions, scores, strict=True):
        if (
            decision.opportunity_id != score.opportunity_id
            or decision.pair_id != score.pair_id
            or decision.scope is not score.scope
            or decision.configuration_id != score.configuration_id
        ):
            raise ValueError("revision safety decision and score order differs")
        groups[(decision.scope, decision.configuration_id)].append((decision, score))

    candidates: dict[str, dict[str, object]] = {}
    ranked: list[tuple[tuple[object, ...], str]] = []
    scope_rank = {
        RevisionSafetyScope.CONTEXT: 0,
        RevisionSafetyScope.TRAFFIC: 1,
        RevisionSafetyScope.GLOBAL: 2,
    }
    for (scope, configuration_id), rows in sorted(
        groups.items(), key=lambda item: (item[0][0].value, item[0][1])
    ):
        selected_scores = [score for _decision, score in rows]
        base_reroutes = sum(score.base_reroute for score in selected_scores)
        executed = sum(score.executed_reroute for score in selected_scores)
        retained = sum(score.retained_base_reroute for score in selected_scores)
        beneficial_base = sum(score.beneficial_base_reroute for score in selected_scores)
        false_admissions = sum(score.false_causal_admission for score in selected_scores)
        harmful = sum(score.harmful_action for score in selected_scores)
        unnecessary = sum(score.unnecessary_safety_deferral for score in selected_scores)
        retained_fraction = retained / base_reroutes if base_reroutes else 0.0
        conditional_harmful = harmful / executed if executed else None
        unnecessary_rate = unnecessary / beneficial_base if beneficial_base else None
        eligible = (
            false_admissions == 0 and retained_fraction >= minimum_retained_base_reroute_fraction
        )
        reason_counts = Counter(decision.reason.value for decision, _score in rows)
        candidates[configuration_id] = {
            "scope": scope.value,
            "opportunity_count": len(rows),
            "base_reroute_count": base_reroutes,
            "executed_reroute_count": executed,
            "retained_base_reroute_count": retained,
            "retained_base_reroute_fraction": retained_fraction,
            "beneficial_base_reroute_count": beneficial_base,
            "false_causal_admission_count": false_admissions,
            "harmful_event_count": harmful,
            "harmful_event_burden_rate": harmful / len(rows),
            "conditional_harmful_action_rate": conditional_harmful,
            "unnecessary_safety_deferral_count": unnecessary,
            "unnecessary_safety_deferral_rate": unnecessary_rate,
            "mean_selected_latency_ms": sum(score.selected_latency_ms for score in selected_scores)
            / len(selected_scores),
            "reason_counts": dict(sorted(reason_counts.items())),
            "eligible": eligible,
        }
        if eligible:
            ranked.append(
                (
                    (
                        harmful,
                        conditional_harmful if conditional_harmful is not None else math.inf,
                        unnecessary,
                        unnecessary_rate if unnecessary_rate is not None else math.inf,
                        -retained_fraction,
                        scope_rank[scope],
                        configuration_id,
                    ),
                    configuration_id,
                )
            )
    selected_configuration = min(ranked)[1] if ranked else None
    return {
        "minimum_retained_base_reroute_fraction": minimum_retained_base_reroute_fraction,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(ranked),
        "selected_configuration_id": selected_configuration,
        "safety_subclaim_status": "CANDIDATE_SELECTED" if ranked else "REJECTED",
        "candidates": candidates,
    }
