"""Leakage-separated A0-A16 orchestration for the M10 controlled pilot."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from time import perf_counter_ns, process_time_ns

from ictc.causal.estimation import (
    AdjustmentObservation,
    EstimationResult,
    estimate_stratified_backdoor,
)
from ictc.causal.identification import IdentifiabilityReason, IdentifiabilityResult
from ictc.controller import AdmissionGate, AdmissionRequest
from ictc.datasets import DatasetSplit
from ictc.domain.enums import (
    ActionKind,
    AdmissionDecision,
    ContractState,
    ReasonCode,
    SafetyDisposition,
)
from ictc.domain.models import NetworkAction, SafetyDecision
from ictc.pilot.authorization import TestAccessAuthorization
from ictc.pilot.benchmark import (
    M10_GRAPH_ID,
    ObservationalDecision,
    PotentialOutcomeTruth,
)
from ictc.pilot.benchmark import (
    reference_identification as benchmark_reference_identification,
)


class AblationId(StrEnum):
    A0 = "A0"
    A1 = "A1"
    A2 = "A2"
    A3 = "A3"
    A4 = "A4"
    A5 = "A5"
    A6 = "A6"
    A7 = "A7"
    A8 = "A8"
    A9 = "A9"
    A10 = "A10"
    A11 = "A11"
    A12 = "A12"
    A13 = "A13"
    A14 = "A14"
    A15 = "A15"
    A16 = "A16"


class FixtureKind(StrEnum):
    OBSERVED = "OBSERVED"
    QUEUE_FIELD_REDACTED = "QUEUE_FIELD_REDACTED"
    INVALID_PROVENANCE = "INVALID_PROVENANCE"
    LOW_TRUST = "LOW_TRUST"
    SINGLE_DOMAIN = "SINGLE_DOMAIN"
    POST_COMMIT_BREACH = "POST_COMMIT_BREACH"
    CONTRACT_INFEASIBLE = "CONTRACT_INFEASIBLE"


@dataclass(frozen=True, slots=True)
class PilotProtocol:
    freshness_limit_ticks: int = 2
    synchronization_limit_ticks: int = 1
    acquisition_deadline_ticks: int = 5
    telemetry_budget_bytes: int = 256
    minimum_trust_millionths: int = 800_000
    minimum_source_diversity: int = 2
    maximum_uncertainty_width_ms: float = 20.0
    maximum_acceptable_effect_ms: float = 0.0
    maximum_deadline_violation_rate: float = 0.05
    uncertainty_width_grid_ms: tuple[float, ...] = (0.0, 1.0, 5.0, 20.0, 1000.0)
    deadline_budget_grid: tuple[tuple[int, int], ...] = (
        (0, 64),
        (0, 128),
        (0, 640),
        (2, 64),
        (2, 128),
        (2, 640),
        (5, 64),
        (5, 128),
        (5, 640),
    )

    def __post_init__(self) -> None:
        counters = (
            self.freshness_limit_ticks,
            self.synchronization_limit_ticks,
            self.acquisition_deadline_ticks,
            self.telemetry_budget_bytes,
            self.minimum_trust_millionths,
            self.minimum_source_diversity,
        )
        if min(counters) < 0:
            raise ValueError("pilot protocol thresholds cannot be negative")
        if not self.uncertainty_width_grid_ms or any(
            value < 0 for value in self.uncertainty_width_grid_ms
        ):
            raise ValueError("uncertainty grid must contain nonnegative thresholds")
        if not self.deadline_budget_grid or any(
            deadline < 0 or budget <= 0 for deadline, budget in self.deadline_budget_grid
        ):
            raise ValueError("deadline/budget grid values are invalid")
        if not 0 <= self.maximum_deadline_violation_rate <= 1:
            raise ValueError("deadline-violation safety threshold must be a ratio")


@dataclass(frozen=True, slots=True)
class PilotModel:
    global_causal: EstimationResult
    topology_causal: tuple[tuple[str, EstimationResult], ...]
    b9_association: EstimationResult
    misspecified_unadjusted: EstimationResult

    def __post_init__(self) -> None:
        results = (
            self.global_causal,
            self.b9_association,
            self.misspecified_unadjusted,
            *(result for _topology, result in self.topology_causal),
        )
        if any(not result.estimable for result in results):
            raise ValueError("frozen pilot model cannot contain an inestimable result")
        names = tuple(name for name, _result in self.topology_causal)
        if names != tuple(sorted(set(names))):
            raise ValueError("topology-specific pilot models must be unique and sorted")

    def causal_for(self, topology_family_id: str) -> EstimationResult:
        return dict(self.topology_causal).get(topology_family_id, self.global_causal)


@dataclass(frozen=True, slots=True)
class PilotOpportunity:
    opportunity_id: str
    decision: ObservationalDecision
    fixture: FixtureKind = FixtureKind.OBSERVED

    def __post_init__(self) -> None:
        if not self.opportunity_id:
            raise ValueError("pilot opportunity ID is required")
        if self.fixture is not FixtureKind.OBSERVED:
            evidence = self.decision.pre_action_evidence
            if evidence.queue_pressure_bin is None or evidence.utilization_difference is None:
                raise ValueError("controlled faults require a complete pre-action evidence row")


@dataclass(frozen=True, slots=True)
class ContractAssessment:
    satisfied: bool
    required_variable_available: bool
    freshness_satisfied: bool
    synchronization_satisfied: bool
    provenance_satisfied: bool
    trust_satisfied: bool
    diversity_satisfied: bool
    deadline_satisfied: bool
    budget_satisfied: bool
    telemetry_items: int
    telemetry_bytes: int
    failed_clauses: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PilotDecision:
    opportunity_id: str
    pair_id: str
    scenario_id: str
    split: DatasetSplit
    test_authorization_id: str | None
    ablation_id: AblationId
    configuration_id: str
    fixture: FixtureKind
    decision: AdmissionDecision
    executable_action: ActionKind | None
    reason_codes: tuple[str, ...]
    reference_identified: bool
    model_identified: bool
    reference_contract_satisfied: bool
    contract_satisfied: bool
    estimable: bool
    estimate_point_ms: float | None
    estimate_lower_ms: float | None
    estimate_upper_ms: float | None
    telemetry_items: int
    telemetry_bytes: int
    contract_acquisition_latency_ms: float
    protocol_messages: int
    admission_wall_time_ns: int
    admission_cpu_time_ns: int
    breach_detection_time_ns: int | None
    fallback_execution_time_ns: int | None
    breach_detected: bool
    fallback_executed: bool

    def __post_init__(self) -> None:
        if self.split is DatasetSplit.TEST and self.test_authorization_id is None:
            raise PermissionError("M10 test decisions require an authorization ID")
        if self.split is not DatasetSplit.TEST and self.test_authorization_id is not None:
            raise ValueError("non-test decisions cannot carry a test authorization ID")
        if self.executable_action is ActionKind.REROUTE_TO_PATH and self.fallback_executed:
            raise ValueError("reroute execution and fallback execution are mutually exclusive")
        counters = (
            self.telemetry_items,
            self.telemetry_bytes,
            self.protocol_messages,
            self.admission_wall_time_ns,
            self.admission_cpu_time_ns,
        )
        if min(counters) < 0 or self.contract_acquisition_latency_ms < 0:
            raise ValueError("pilot decision resource measurements cannot be negative")


@dataclass(frozen=True, slots=True)
class PilotScore:
    opportunity_id: str
    pair_id: str
    ablation_id: AblationId
    configuration_id: str
    false_admission: bool
    unnecessary_deferral: bool
    harmful_action: bool
    invariant_violation: bool
    individual_effect_residual_ms: float | None
    individual_effect_inside_ate_interval: bool | None
    interval_width_ms: float | None
    selected_latency_ms: float | None
    selected_packet_delivery_ratio: float | None
    selected_packet_loss_ratio: float | None
    selected_jitter_ms: float | None
    selected_throughput_packets_per_tick: float | None
    selected_deadline_violation_rate: float | None
    selected_route_changes: int | None
    fallback_outcome_ms: float | None


@dataclass(frozen=True, slots=True)
class EstimandScore:
    """One population-ATE comparison at the estimator's declared grouping unit."""

    split: DatasetSplit
    topology_family_id: str
    ablation_id: AblationId
    configuration_id: str
    pair_count: int
    target_ate_ms: float
    estimate_point_ms: float
    estimate_lower_ms: float
    estimate_upper_ms: float
    estimation_error_ms: float
    interval_covered: bool
    interval_width_ms: float


def _identified_for(adjustment_set: tuple[str, ...], *, detail: str) -> IdentifiabilityResult:
    reason = (
        IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
        if not adjustment_set
        else IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
    )
    available = ("EndToEndDelay", "RouteAction", *adjustment_set)
    return IdentifiabilityResult(
        True,
        reason,
        "RouteAction",
        "EndToEndDelay",
        tuple(sorted(available)),
        (adjustment_set,),
        (adjustment_set,),
        detail=detail,
    )


def _observations(
    decisions: tuple[ObservationalDecision, ...],
    covariate: str | None,
) -> tuple[AdjustmentObservation, ...]:
    rows: list[AdjustmentObservation] = []
    for decision in decisions:
        evidence = decision.pre_action_evidence
        if decision.observed_latency_ms is None:
            raise ValueError("pilot estimator requires observed training outcomes")
        values: dict[str, float] = {}
        if covariate == "QueuePressureBin":
            if evidence.queue_pressure_bin is None:
                continue
            values[covariate] = float(evidence.queue_pressure_bin)
        elif covariate == "UtilizationDifferenceBin":
            if evidence.utilization_difference is None:
                continue
            difference = evidence.utilization_difference
            values[covariate] = float(-1 if difference < 0 else 1 if difference > 0 else 0)
        rows.append(
            AdjustmentObservation.from_mapping(
                treatment=decision.assigned_action,
                outcome=decision.observed_latency_ms,
                covariates=values,
            )
        )
    return tuple(rows)


def _estimate(
    decisions: tuple[ObservationalDecision, ...],
    adjustment_set: tuple[str, ...],
    identification: IdentifiabilityResult,
) -> EstimationResult:
    covariate = adjustment_set[0] if adjustment_set else None
    return estimate_stratified_backdoor(
        _observations(decisions, covariate),
        identification,
        adjustment_set,
        estimand="reroute-minus-keep end-to-end latency",
        unit="ms",
    )


def fit_pilot_model(training: tuple[ObservationalDecision, ...]) -> PilotModel:
    """Fit only from controller-safe training observations; paired truth is not accepted."""
    if not training:
        raise ValueError("pilot fitting requires training observations")
    if any(decision.split is not DatasetSplit.TRAIN for decision in training):
        raise PermissionError("pilot fitting accepts training rows only")
    complete = tuple(
        decision
        for decision in training
        if decision.observation_regime_id == "complete"
        and decision.pre_action_evidence.queue_pressure_bin is not None
    )
    if not complete:
        raise ValueError("pilot fitting requires complete-observation training rows")
    causal_identification = _identified_for(
        ("QueuePressureBin",), detail="frozen controlled M10 backdoor adjustment"
    )
    global_causal = _estimate(complete, ("QueuePressureBin",), causal_identification)
    association_identification = _identified_for(
        ("UtilizationDifferenceBin",),
        detail="A9 association-only selected field; not a causal graph claim",
    )
    b9_association = _estimate(complete, ("UtilizationDifferenceBin",), association_identification)
    misspecified_identification = _identified_for(
        (), detail="A14 intentionally misspecified graph removes the backdoor path"
    )
    misspecified = _estimate(complete, (), misspecified_identification)
    topology_results: list[tuple[str, EstimationResult]] = []
    for topology in sorted({decision.topology_family_id for decision in complete}):
        selected = tuple(
            decision for decision in complete if decision.topology_family_id == topology
        )
        result = _estimate(selected, ("QueuePressureBin",), causal_identification)
        if result.estimable:
            topology_results.append((topology, result))
    return PilotModel(
        global_causal,
        tuple(topology_results),
        b9_association,
        misspecified,
    )


def build_opportunities(
    decisions: tuple[ObservationalDecision, ...],
    *,
    include_controlled_faults: bool,
    test_authorization: TestAccessAuthorization | None = None,
) -> tuple[PilotOpportunity, ...]:
    if not decisions:
        raise ValueError("pilot opportunities require decision rows")
    contains_test = any(decision.split is DatasetSplit.TEST for decision in decisions)
    if contains_test:
        if test_authorization is None:
            raise PermissionError("M10 test opportunities require GO authorization")
        test_authorization.assert_allows(DatasetSplit.TEST)
    opportunities = [
        PilotOpportunity(f"{decision.pair_id}::observed", decision) for decision in decisions
    ]
    if include_controlled_faults:
        fixtures = tuple(item for item in FixtureKind if item is not FixtureKind.OBSERVED)
        for decision in decisions:
            if (
                decision.observation_regime_id != "complete"
                or decision.pre_action_evidence.queue_pressure_bin is None
            ):
                continue
            opportunities.extend(
                PilotOpportunity(f"{decision.pair_id}::{fixture.value.lower()}", decision, fixture)
                for fixture in fixtures
            )
    return tuple(opportunities)


def _opportunity_reference(opportunity: PilotOpportunity) -> IdentifiabilityResult:
    if opportunity.fixture is FixtureKind.QUEUE_FIELD_REDACTED:
        return _reference_identification_for_available(False)
    return opportunity.decision.reference_identification


def _reference_identification_for_available(available: bool) -> IdentifiabilityResult:
    return benchmark_reference_identification(available)


def _contract(
    opportunity: PilotOpportunity,
    ablation: AblationId,
    protocol: PilotProtocol,
    *,
    deadline_ticks: int | None = None,
    budget_bytes: int | None = None,
) -> ContractAssessment:
    evidence = opportunity.decision.pre_action_evidence
    deadline = protocol.acquisition_deadline_ticks if deadline_ticks is None else deadline_ticks
    budget = protocol.telemetry_budget_bytes if budget_bytes is None else budget_bytes
    no_adjustment_required = ablation in {AblationId.A1, AblationId.A14}
    uses_association = ablation is AblationId.A9
    queue_available = (
        evidence.queue_pressure_bin is not None
        and opportunity.fixture is not FixtureKind.QUEUE_FIELD_REDACTED
    )
    utilization_available = evidence.utilization_difference is not None
    required_available = (
        True
        if no_adjustment_required
        else utilization_available
        if uses_association
        else queue_available
    )
    evidence_used = not no_adjustment_required and required_available
    delivery = evidence.delivered_at_tick if evidence_used else 0
    freshness = evidence.freshness_ticks if evidence_used else 0
    synchronization = evidence.clock_skew_span_ticks if evidence_used else 0
    freshness_limit = 0 if ablation is AblationId.A8 else protocol.freshness_limit_ticks
    freshness_ok = (
        True
        if ablation is AblationId.A3 or not evidence_used
        else freshness is not None and freshness <= freshness_limit
    )
    synchronization_ok = (
        True
        if ablation is AblationId.A4 or not evidence_used
        else synchronization is not None and synchronization <= protocol.synchronization_limit_ticks
    )
    provenance_ok = (
        ablation is AblationId.A5
        or not evidence_used
        or opportunity.fixture is not FixtureKind.INVALID_PROVENANCE
    )
    trust = 500_000 if opportunity.fixture is FixtureKind.LOW_TRUST else 900_000
    trust_threshold = 0 if ablation is AblationId.A6 else protocol.minimum_trust_millionths
    trust_ok = not evidence_used or trust >= trust_threshold
    diversity = 1 if opportunity.fixture is FixtureKind.SINGLE_DOMAIN else len(evidence.source_ids)
    minimum_diversity = 1 if ablation is AblationId.A7 else protocol.minimum_source_diversity
    diversity_ok = not evidence_used or diversity >= minimum_diversity
    deadline_ok = delivery is not None and delivery <= deadline
    items = evidence.telemetry_items if evidence_used else 0
    bytes_used = evidence.telemetry_bytes if evidence_used else 0
    if ablation is AblationId.A10 and evidence_used:
        items *= 5
        bytes_used *= 5
    elif ablation is AblationId.A11 and evidence_used:
        items = 1
        bytes_used = 64
        diversity_ok = minimum_diversity <= 1
    budget_ok = bytes_used <= budget
    clauses = {
        "SYNTHESIS_FEASIBILITY": (opportunity.fixture is not FixtureKind.CONTRACT_INFEASIBLE),
        "REQUIRED_VARIABLE": required_available,
        "FRESHNESS": freshness_ok,
        "SYNCHRONIZATION": synchronization_ok,
        "PROVENANCE": provenance_ok,
        "TRUST": trust_ok,
        "DIVERSITY": diversity_ok,
        "DEADLINE": deadline_ok,
        "BUDGET": budget_ok,
    }
    failed = tuple(name for name, passed in clauses.items() if not passed)
    return ContractAssessment(
        not failed,
        required_available,
        freshness_ok,
        synchronization_ok,
        provenance_ok,
        trust_ok,
        diversity_ok,
        deadline_ok,
        budget_ok,
        items,
        bytes_used,
        failed,
    )


def _model_identification(
    opportunity: PilotOpportunity, ablation: AblationId
) -> IdentifiabilityResult:
    if ablation is AblationId.A1:
        return _identified_for((), detail="A1 forced identification predicate")
    if ablation is AblationId.A14:
        return _identified_for((), detail="A14 misspecified no-confounding graph")
    if ablation is AblationId.A9:
        available = opportunity.decision.pre_action_evidence.utilization_difference is not None
        return (
            _identified_for(
                ("UtilizationDifferenceBin",), detail="A9 B9-selected association field"
            )
            if available
            else _reference_identification_for_available(False)
        )
    return _opportunity_reference(opportunity)


def _estimate_for(
    opportunity: PilotOpportunity, ablation: AblationId, model: PilotModel
) -> EstimationResult:
    if ablation is AblationId.A9:
        return model.b9_association
    if ablation is AblationId.A14:
        return model.misspecified_unadjusted
    return model.causal_for(opportunity.decision.topology_family_id)


def _actions(opportunity: PilotOpportunity) -> tuple[NetworkAction, NetworkAction]:
    proposed = NetworkAction(
        action_id=f"m10:{opportunity.opportunity_id}:reroute",
        kind=ActionKind.REROUTE_TO_PATH,
        path_id="path-candidate",
    )
    fallback = NetworkAction(
        action_id=f"m10:{opportunity.opportunity_id}:keep",
        kind=ActionKind.KEEP_CURRENT_PATH,
    )
    return proposed, fallback


def _run_one(
    opportunity: PilotOpportunity,
    ablation: AblationId,
    configuration_id: str,
    model: PilotModel,
    protocol: PilotProtocol,
    *,
    test_authorization_id: str | None = None,
    uncertainty_width: float | None = None,
    deadline_ticks: int | None = None,
    budget_bytes: int | None = None,
) -> PilotDecision:
    wall_started = perf_counter_ns()
    cpu_started = process_time_ns()
    reference = _opportunity_reference(opportunity)
    reference_contract = _contract(opportunity, AblationId.A0, protocol)
    contract = _contract(
        opportunity,
        ablation,
        protocol,
        deadline_ticks=deadline_ticks,
        budget_bytes=budget_bytes,
    )
    identification = _model_identification(opportunity, ablation)
    estimation = _estimate_for(opportunity, ablation, model)
    estimate = estimation.estimate
    upper_margin = (
        estimate.uncertainty.upper - estimate.point_estimate if estimate is not None else 0.0
    )
    proposed, fallback = _actions(opportunity)
    now = datetime(2026, 7, 13, 0, 0, tzinfo=UTC)
    effect_gate_removed = ablation is AblationId.A2
    request = AdmissionRequest(
        proposed_action=proposed,
        fallback_action=fallback,
        graph_id=M10_GRAPH_ID,
        identification=identification,
        contract_id=f"contract:{opportunity.opportunity_id}:{ablation.value}",
        contract_state=(ContractState.SATISFIED if contract.satisfied else ContractState.ACTIVE),
        estimation=estimation,
        maximum_uncertainty_width=(
            math.inf
            if effect_gate_removed
            else protocol.maximum_uncertainty_width_ms
            if uncertainty_width is None
            else uncertainty_width
        ),
        safety=SafetyDecision(disposition=SafetyDisposition.SAFE),
        policy_permitted=True,
        remediation_possible=True,
        deadline=now + timedelta(seconds=1),
        evaluated_at=now,
        evidence_digests=opportunity.decision.pre_action_evidence.provenance_digests,
        maximum_acceptable_effect=(
            None if effect_gate_removed else protocol.maximum_acceptable_effect_ms - upper_margin
        ),
    )
    outcome = AdmissionGate().evaluate(request)
    executable = outcome.executable_action.kind if outcome.executable_action is not None else None
    reasons = tuple(reason.value for reason in outcome.reason_codes)
    final_decision = outcome.decision
    breach = opportunity.fixture is FixtureKind.POST_COMMIT_BREACH
    monitors_breach = ablation is not AblationId.A12
    breach_time: int | None = None
    if breach and monitors_breach and executable is ActionKind.REROUTE_TO_PATH:
        breach_started = perf_counter_ns()
        final_decision = AdmissionDecision.REJECT
        executable = ActionKind.KEEP_CURRENT_PATH
        reasons = (
            *reasons,
            ReasonCode.CONTRACT_BREACHED.value,
            ReasonCode.FALLBACK_EXECUTED.value,
        )
        breach_time = perf_counter_ns() - breach_started
    fallback_executed = executable is ActionKind.KEEP_CURRENT_PATH
    fallback_time: int | None = None
    if ablation is AblationId.A13 and executable is not ActionKind.REROUTE_TO_PATH:
        executable = None
        fallback_executed = False
    if fallback_executed:
        fallback_started = perf_counter_ns()
        executable = ActionKind.KEEP_CURRENT_PATH
        fallback_time = perf_counter_ns() - fallback_started
    return PilotDecision(
        opportunity.opportunity_id,
        opportunity.decision.pair_id,
        opportunity.decision.scenario_id,
        opportunity.decision.split,
        test_authorization_id,
        ablation,
        configuration_id,
        opportunity.fixture,
        final_decision,
        executable,
        reasons,
        reference.identified,
        identification.identified,
        reference_contract.satisfied,
        contract.satisfied,
        estimation.estimable,
        estimate.point_estimate if estimate is not None else None,
        estimate.uncertainty.lower if estimate is not None else None,
        estimate.uncertainty.upper if estimate is not None else None,
        contract.telemetry_items,
        contract.telemetry_bytes,
        float(opportunity.decision.pre_action_evidence.delivered_at_tick or 0),
        3 + contract.telemetry_items,
        perf_counter_ns() - wall_started,
        process_time_ns() - cpu_started,
        breach_time,
        fallback_time,
        breach,
        fallback_executed,
    )


def run_ablation_suite(
    opportunities: tuple[PilotOpportunity, ...],
    model: PilotModel,
    protocol: PilotProtocol | None = None,
    test_authorization: TestAccessAuthorization | None = None,
) -> tuple[PilotDecision, ...]:
    """Run A0-A16 without accepting paired outcome truth."""
    if not opportunities:
        raise ValueError("ablation suite requires at least one opportunity")
    contains_test = any(item.decision.split is DatasetSplit.TEST for item in opportunities)
    if contains_test:
        if any(item.decision.split is not DatasetSplit.TEST for item in opportunities):
            raise ValueError("test and non-test opportunities cannot share one suite call")
        if test_authorization is None:
            raise PermissionError("M10 test suite requires GO authorization")
        test_authorization.assert_allows(DatasetSplit.TEST)
    authorization_id = (
        test_authorization.authorization_id if test_authorization is not None else None
    )
    frozen = protocol or PilotProtocol()
    output: list[PilotDecision] = []
    single = tuple(AblationId)[0:15]
    for opportunity in opportunities:
        output.extend(
            _run_one(
                opportunity,
                ablation,
                "default",
                model,
                frozen,
                test_authorization_id=authorization_id,
            )
            for ablation in single
        )
        output.extend(
            _run_one(
                opportunity,
                AblationId.A15,
                f"width_ms={width:g}",
                model,
                frozen,
                test_authorization_id=authorization_id,
                uncertainty_width=width,
            )
            for width in frozen.uncertainty_width_grid_ms
        )
        output.extend(
            _run_one(
                opportunity,
                AblationId.A16,
                f"deadline_ticks={deadline};budget_bytes={budget}",
                model,
                frozen,
                test_authorization_id=authorization_id,
                deadline_ticks=deadline,
                budget_bytes=budget,
            )
            for deadline, budget in frozen.deadline_budget_grid
        )
    return tuple(output)


def score_persisted_decisions(
    decisions: tuple[PilotDecision, ...],
    truth_by_pair: dict[str, PotentialOutcomeTruth],
    protocol: PilotProtocol | None = None,
) -> tuple[PilotScore, ...]:
    """Join paired truth only after controller decisions have crossed persistence boundary."""
    if not decisions:
        raise ValueError("pilot scoring requires persisted decisions")
    scores: list[PilotScore] = []
    frozen = protocol or PilotProtocol()
    for decision in decisions:
        truth = truth_by_pair.get(decision.pair_id)
        if truth is None:
            raise KeyError(f"paired truth is unavailable for {decision.pair_id}")
        rerouted = decision.executable_action is ActionKind.REROUTE_TO_PATH
        deferred = not rerouted
        reference_admissible = (
            decision.reference_identified and decision.reference_contract_satisfied
        )
        harmful = rerouted and not truth.reroute_beneficial
        lower = decision.estimate_lower_ms
        upper = decision.estimate_upper_ms
        coverage = (
            lower <= truth.latency_effect_ms <= upper
            if lower is not None and upper is not None
            else None
        )
        width = upper - lower if lower is not None and upper is not None else None
        selected_truth = (
            truth.reroute
            if rerouted
            else truth.keep
            if decision.executable_action is ActionKind.KEEP_CURRENT_PATH
            else None
        )
        scores.append(
            PilotScore(
                decision.opportunity_id,
                decision.pair_id,
                decision.ablation_id,
                decision.configuration_id,
                rerouted and not reference_admissible,
                deferred
                and reference_admissible
                and truth.reroute_beneficial
                and not decision.breach_detected,
                harmful,
                rerouted
                and truth.reroute.deadline_violation_rate > frozen.maximum_deadline_violation_rate,
                (
                    decision.estimate_point_ms - truth.latency_effect_ms
                    if decision.estimate_point_ms is not None
                    else None
                ),
                coverage,
                width,
                selected_truth.mean_latency_ms if selected_truth is not None else None,
                (selected_truth.packet_delivery_ratio if selected_truth is not None else None),
                selected_truth.packet_loss_ratio if selected_truth is not None else None,
                selected_truth.jitter_ms if selected_truth is not None else None,
                (
                    selected_truth.throughput_packets_per_tick
                    if selected_truth is not None
                    else None
                ),
                (selected_truth.deadline_violation_rate if selected_truth is not None else None),
                selected_truth.route_changes if selected_truth is not None else None,
                truth.keep_latency_ms if decision.fallback_executed else None,
            )
        )
    return tuple(scores)


def score_population_estimands(
    decisions: tuple[PilotDecision, ...],
    opportunity_by_id: dict[str, PilotOpportunity],
    truth_by_pair: dict[str, PotentialOutcomeTruth],
) -> tuple[EstimandScore, ...]:
    """Score ATE intervals against paired population means, never individual effects."""
    groups: dict[tuple[DatasetSplit, str, AblationId, str], list[PilotDecision]] = {}
    for decision in decisions:
        opportunity = opportunity_by_id.get(decision.opportunity_id)
        if opportunity is None:
            raise KeyError(f"opportunity is unavailable for {decision.opportunity_id}")
        if opportunity.fixture is not FixtureKind.OBSERVED:
            continue
        key = (
            decision.split,
            opportunity.decision.topology_family_id,
            decision.ablation_id,
            decision.configuration_id,
        )
        groups.setdefault(key, []).append(decision)
    output: list[EstimandScore] = []
    for (split, topology, ablation, configuration), rows in sorted(
        groups.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        unique = {row.pair_id: row for row in rows}
        estimates = {
            (row.estimate_point_ms, row.estimate_lower_ms, row.estimate_upper_ms)
            for row in unique.values()
        }
        if len(estimates) != 1:
            raise ValueError("one estimand group contains inconsistent effect estimates")
        point, lower, upper = estimates.pop()
        if point is None or lower is None or upper is None:
            raise ValueError("one estimand group has no finite effect interval")
        truths = [truth_by_pair[pair_id].latency_effect_ms for pair_id in unique]
        target = sum(truths) / len(truths)
        output.append(
            EstimandScore(
                split,
                topology,
                ablation,
                configuration,
                len(truths),
                target,
                point,
                lower,
                upper,
                point - target,
                lower <= target <= upper,
                upper - lower,
            )
        )
    return tuple(output)
