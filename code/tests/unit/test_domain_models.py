"""Validation tests for ICTC's M01 domain model."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ictc.domain.enums import (
    ActionKind,
    AdmissionDecision,
    BreachSeverity,
    ClauseSatisfaction,
    ContractState,
    IdentificationMethod,
    PrivacyClass,
    ReasonCode,
    SafetyDisposition,
    TelemetrySourceClass,
    VariableRole,
)
from ictc.domain.models import (
    AdjustmentSet,
    BandwidthClause,
    BaselineAction,
    CausalEffectEstimate,
    CausalVariable,
    ClauseAssessment,
    ContractBreach,
    ContractCommitment,
    ContractOffer,
    ContractStatus,
    ControllerVisibleState,
    DeadlineClause,
    EvaluationHorizon,
    EvidenceRecord,
    ExperimentScenario,
    FreshnessClause,
    IdentifiabilityQuery,
    IdentificationStrategy,
    InterventionOutcome,
    NetworkAction,
    ObservationQuality,
    OracleState,
    PairedCounterfactualOutcome,
    PrivacyClause,
    ProvenanceClause,
    RedundancyClause,
    SafetyDecision,
    SynchronizationClause,
    TelemetryCapability,
    TelemetryContract,
    TelemetryObservation,
    TelemetrySource,
    TemporalCausalGraph,
    TemporalEdge,
    TrustClause,
    UncertaintyInterval,
)

NOW = datetime(2026, 7, 10, 7, 0, tzinfo=UTC)


def keep_action() -> NetworkAction:
    return NetworkAction(action_id="keep-1", kind=ActionKind.KEEP_CURRENT_PATH)


def reroute_action() -> NetworkAction:
    return NetworkAction(
        action_id="reroute-1",
        kind=ActionKind.REROUTE_TO_PATH,
        path_id="candidate-path-b",
    )


def source() -> TelemetrySource:
    return TelemetrySource(
        source_id="switch-a",
        source_class=TelemetrySourceClass.SWITCH,
        administrative_domain="domain-a",
        trust_score=0.9,
    )


def capability() -> TelemetryCapability:
    return TelemetryCapability(
        capability_id="cap-a",
        source=source(),
        measurable_variable_ids=("queue_occupancy", "link_utilization"),
        minimum_sampling_rate_hz=1.0,
        maximum_sampling_rate_hz=20.0,
        expected_delay_ms=5,
        clock_accuracy_ms=1,
        cost=1.0,
        bandwidth_kbps=10.0,
        available_from=NOW,
        available_until=NOW + timedelta(minutes=1),
    )


def query() -> IdentifiabilityQuery:
    return IdentifiabilityQuery(
        query_id="query-1",
        graph_id="graph-1",
        treatment_variable_id="route_action",
        outcome_variable_id="end_to_end_delay",
        baseline_action=BaselineAction(action=keep_action()),
        proposed_action=reroute_action(),
        horizon=EvaluationHorizon(duration_ms=100),
    )


def strategy() -> IdentificationStrategy:
    return IdentificationStrategy(
        strategy_id="backdoor-1",
        method=IdentificationMethod.BACKDOOR,
        adjustment_set=AdjustmentSet(
            adjustment_set_id="adjust-1",
            variable_ids=("cross_traffic",),
        ),
        assumptions=("declared graph is valid",),
    )


def observation(*, missing: bool = False) -> TelemetryObservation:
    return TelemetryObservation(
        observation_id="obs-1",
        source_id="switch-a",
        variable_id="queue_occupancy",
        value=None if missing else 0.4,
        observed_at=NOW,
        quality=ObservationQuality(
            sample_age_ms=3,
            measurement_delay_ms=1,
            clock_skew_ms=1,
            sampling_rate_hz=10.0,
            missing=missing,
        ),
        trust_score=0.9,
        provenance_digest="a" * 16,
    )


def clauses() -> tuple[object, ...]:
    return (
        FreshnessClause(clause_id="fresh", maximum_age_ms=10),
        SynchronizationClause(clause_id="sync", maximum_clock_skew_ms=3),
        ProvenanceClause(clause_id="prov", accepted_mechanisms=("sha256",)),
        TrustClause(clause_id="trust", minimum_trust_score=0.8),
        RedundancyClause(clause_id="redundancy", minimum_distinct_sources=2, failure_tolerance=1),
        PrivacyClause(
            clause_id="privacy",
            privacy_class=PrivacyClass.INTERNAL,
            permitted_administrative_domains=("domain-a",),
        ),
        BandwidthClause(clause_id="bandwidth", maximum_bandwidth_kbps=100.0),
        DeadlineClause(clause_id="deadline", deadline=NOW + timedelta(seconds=10)),
    )


def contract() -> TelemetryContract:
    return TelemetryContract(
        contract_id="contract-1",
        proposal_id="proposal-1",
        query=query(),
        required_variable_ids=("cross_traffic", "queue_occupancy"),
        allowed_strategies=(strategy(),),
        clauses=clauses(),  # type: ignore[arg-type]
        fallback_action=keep_action(),
    )


def test_network_action_and_baseline_validation() -> None:
    assert reroute_action().path_id == "candidate-path-b"
    with pytest.raises(ValidationError, match="requires path_id"):
        NetworkAction(action_id="bad", kind=ActionKind.REROUTE_TO_PATH)
    with pytest.raises(ValidationError, match="must not name"):
        NetworkAction(action_id="bad", kind=ActionKind.KEEP_CURRENT_PATH, path_id="path-a")
    with pytest.raises(ValidationError, match="baseline"):
        BaselineAction(action=reroute_action())


def test_graph_and_identification_validation() -> None:
    route = CausalVariable(
        variable_id="route_action",
        display_name="Route action",
        role=VariableRole.TREATMENT,
        observed=True,
    )
    delay = CausalVariable(
        variable_id="end_to_end_delay",
        display_name="Delay",
        role=VariableRole.OUTCOME,
        observed=True,
    )
    graph = TemporalCausalGraph(
        graph_id="graph-1",
        version=1,
        scope="routing",
        variables=(route, delay),
        edges=(
            TemporalEdge(source_variable_id="route_action", target_variable_id="end_to_end_delay"),
        ),
    )
    assert graph.edges[0].lag_steps == 0
    with pytest.raises(ValidationError, match="latent"):
        CausalVariable(
            variable_id="hidden",
            display_name="Hidden",
            role=VariableRole.LATENT,
            observed=True,
        )
    with pytest.raises(ValidationError, match="self loops"):
        TemporalEdge(source_variable_id="route_action", target_variable_id="route_action")
    with pytest.raises(ValidationError, match="undeclared"):
        TemporalCausalGraph(
            graph_id="graph-2",
            version=1,
            scope="routing",
            variables=(route,),
            edges=(TemporalEdge(source_variable_id="route_action", target_variable_id="delay"),),
        )
    with pytest.raises(ValidationError, match="BACKDOOR"):
        IdentificationStrategy(
            strategy_id="bad",
            method=IdentificationMethod.BACKDOOR,
            assumptions=("x",),
        )
    with pytest.raises(ValidationError, match="FRONTDOOR"):
        IdentificationStrategy(
            strategy_id="bad",
            method=IdentificationMethod.FRONTDOOR,
            assumptions=("x",),
        )
    with pytest.raises(ValidationError, match="NONE"):
        IdentificationStrategy(
            strategy_id="bad",
            method=IdentificationMethod.NONE,
            adjustment_set=AdjustmentSet(adjustment_set_id="a"),
            assumptions=("x",),
        )
    with pytest.raises(ValidationError, match="proposed action"):
        IdentifiabilityQuery(
            query_id="bad",
            graph_id="graph-1",
            treatment_variable_id="route_action",
            outcome_variable_id="end_to_end_delay",
            baseline_action=BaselineAction(action=keep_action()),
            proposed_action=keep_action(),
            horizon=EvaluationHorizon(duration_ms=1),
        )


def test_telemetry_visibility_and_capability_validation() -> None:
    assert observation().value == 0.4
    assert observation(missing=True).value is None
    with pytest.raises(ValidationError, match="must not carry"):
        TelemetryObservation(
            observation_id="bad",
            source_id="switch-a",
            variable_id="queue_occupancy",
            value=1.0,
            observed_at=NOW,
            quality=ObservationQuality(
                sample_age_ms=0,
                measurement_delay_ms=0,
                clock_skew_ms=0,
                sampling_rate_hz=1,
                missing=True,
            ),
            trust_score=1,
        )
    with pytest.raises(ValidationError, match="requires a value"):
        TelemetryObservation(
            observation_id="missing-value",
            source_id="switch-a",
            variable_id="queue_occupancy",
            value=None,
            observed_at=NOW,
            quality=ObservationQuality(
                sample_age_ms=0,
                measurement_delay_ms=0,
                clock_skew_ms=0,
                sampling_rate_hz=1,
            ),
            trust_score=1,
        )
    with pytest.raises(ValidationError, match="oracle"):
        TelemetryObservation(
            observation_id="bad",
            source_id="oracle",
            variable_id="true_queue_state",
            value=1.0,
            observed_at=NOW,
            quality=ObservationQuality(
                sample_age_ms=0,
                measurement_delay_ms=0,
                clock_skew_ms=0,
                sampling_rate_hz=1,
            ),
            trust_score=1,
        )
    with pytest.raises(ValidationError, match="minimum sampling"):
        TelemetryCapability(
            capability_id="bad-rate",
            source=source(),
            measurable_variable_ids=("queue_occupancy",),
            minimum_sampling_rate_hz=21.0,
            maximum_sampling_rate_hz=20.0,
            expected_delay_ms=0,
            clock_accuracy_ms=0,
            cost=0,
            bandwidth_kbps=0,
            available_from=NOW,
            available_until=NOW + timedelta(seconds=1),
        )
    with pytest.raises(ValidationError, match="availability interval"):
        TelemetryCapability(
            capability_id="bad-time",
            source=source(),
            measurable_variable_ids=("queue_occupancy",),
            minimum_sampling_rate_hz=1.0,
            maximum_sampling_rate_hz=2.0,
            expected_delay_ms=0,
            clock_accuracy_ms=0,
            cost=0,
            bandwidth_kbps=0,
            available_from=NOW,
            available_until=NOW,
        )
    with pytest.raises(ValidationError, match="unique"):
        TelemetryCapability(
            capability_id="bad-duplicate",
            source=source(),
            measurable_variable_ids=("queue_occupancy", "queue_occupancy"),
            minimum_sampling_rate_hz=1.0,
            maximum_sampling_rate_hz=2.0,
            expected_delay_ms=0,
            clock_accuracy_ms=0,
            cost=0,
            bandwidth_kbps=0,
            available_from=NOW,
            available_until=NOW + timedelta(seconds=1),
        )


def test_contract_offer_commitment_and_status_validation() -> None:
    model = contract()
    assert model.state is ContractState.PROPOSED
    offer = ContractOffer(
        offer_id="offer-1",
        contract_id=model.contract_id,
        capability=capability(),
        offered_variable_ids=("queue_occupancy",),
        sampling_rates_hz={"queue_occupancy": 10.0},
    )
    assert offer.capability.source.source_id == "switch-a"
    with pytest.raises(ValidationError, match="outside source capability"):
        ContractOffer(
            offer_id="bad",
            contract_id=model.contract_id,
            capability=capability(),
            offered_variable_ids=("cross_traffic",),
            sampling_rates_hz={"cross_traffic": 10.0},
        )
    with pytest.raises(ValidationError, match="sampling rates must match"):
        ContractOffer(
            offer_id="bad",
            contract_id=model.contract_id,
            capability=capability(),
            offered_variable_ids=("queue_occupancy",),
            sampling_rates_hz={"link_utilization": 10.0},
        )
    with pytest.raises(ValidationError, match="at most one"):
        TelemetryContract(
            contract_id="duplicate-kind",
            proposal_id="proposal-1",
            query=query(),
            required_variable_ids=("cross_traffic",),
            allowed_strategies=(strategy(),),
            clauses=(
                FreshnessClause(clause_id="fresh", maximum_age_ms=10),
                FreshnessClause(clause_id="fresh-2", maximum_age_ms=2),
            ),
            fallback_action=keep_action(),
        )

    commitment = ContractCommitment(
        commitment_id="commit-1",
        contract_id=model.contract_id,
        selected_source_ids=("switch-a", "switch-b"),
        selected_field_ids=("queue_occupancy",),
        sampling_rates_hz={"queue_occupancy": 10.0},
        synchronization_epoch=NOW,
        valid_from=NOW,
        valid_until=NOW + timedelta(seconds=10),
        evidence_digests=("b" * 16,),
        expected_total_cost=2.0,
        fallback_action=keep_action(),
    )
    assert commitment.expected_total_cost == 2.0
    with pytest.raises(ValidationError, match="validity interval"):
        ContractCommitment(
            commitment_id="bad-time",
            contract_id=model.contract_id,
            selected_source_ids=("switch-a",),
            selected_field_ids=("queue_occupancy",),
            sampling_rates_hz={"queue_occupancy": 10.0},
            synchronization_epoch=NOW,
            valid_from=NOW,
            valid_until=NOW,
            evidence_digests=("b" * 16,),
            expected_total_cost=2.0,
            fallback_action=keep_action(),
        )
    satisfied = ClauseAssessment(clause_id="fresh", state=ClauseSatisfaction.SATISFIED)
    unsatisfied = ClauseAssessment(
        clause_id="trust",
        state=ClauseSatisfaction.UNSATISFIED,
        reason_codes=(ReasonCode.TRUST_BELOW_THRESHOLD,),
    )
    status = ContractStatus(
        contract_id=model.contract_id,
        state=ContractState.ACTIVE,
        clause_assessments=(satisfied, unsatisfied),
        current_quality=observation().quality,
        source_available=True,
        uncertainty_width=0.2,
    )
    assert len(status.clause_assessments) == 2
    with pytest.raises(ValidationError, match="requires a reason"):
        ClauseAssessment(clause_id="bad", state=ClauseSatisfaction.BREACHED)
    with pytest.raises(ValidationError, match="must not carry"):
        ClauseAssessment(
            clause_id="bad",
            state=ClauseSatisfaction.SATISFIED,
            reason_codes=(ReasonCode.IDENTIFIABLE,),
        )


def test_decisions_evidence_and_paired_counterfactual_invariants() -> None:
    interval = UncertaintyInterval(lower=-1, upper=2, confidence_level=0.95)
    assert interval.width == 3
    estimate = CausalEffectEstimate(
        estimand="delay reroute effect",
        point_estimate=-0.4,
        unit="ms",
        uncertainty=interval,
        estimator_name="placeholder",
    )
    assert estimate.uncertainty.width == 3
    with pytest.raises(ValidationError, match="lower"):
        UncertaintyInterval(lower=2, upper=1, confidence_level=0.95)
    safe = SafetyDecision(disposition=SafetyDisposition.SAFE)
    unsafe = SafetyDecision(
        disposition=SafetyDisposition.UNSAFE,
        invariant_ids=("loss_limit",),
        reason_codes=(ReasonCode.SAFETY_INVARIANT_VIOLATED,),
    )
    assert safe.disposition is SafetyDisposition.SAFE
    with pytest.raises(ValidationError, match="unsafe"):
        SafetyDecision(disposition=SafetyDisposition.UNSAFE)
    record = EvidenceRecord(
        evidence_record_id="record-1",
        action_id="reroute-1",
        contract_id="contract-1",
        graph_id="graph-1",
        decision=AdmissionDecision.DEFER,
        reason_codes=(ReasonCode.UNCERTAINTY_TOO_WIDE,),
        safety_decision=unsafe,
        recorded_at=NOW,
    )
    assert record.decision is AdmissionDecision.DEFER
    scenario = ExperimentScenario(
        scenario_id="scenario-1",
        topology_id="two-path",
        topology_family="toy",
        node_count=4,
        link_count=4,
        traffic_model="poisson",
        traffic_intensity=0.5,
        experiment_seed=11,
        code_commit="abcdef0",
        config_hash="a" * 16,
        environment_hash="b" * 16,
    )
    assert scenario.node_count == 4
    oracle = OracleState(
        scenario_id=scenario.scenario_id,
        run_id="run-1",
        true_cross_traffic=0.3,
        true_available_capacity=100,
        true_queue_state=0.2,
        true_link_degradation=0,
        true_failure_state=False,
        true_background_flow_count=2,
        true_shared_resource_state="normal",
    )
    assert oracle.true_queue_state == 0.2
    visible = ControllerVisibleState(
        scenario_id=scenario.scenario_id,
        run_id="run-1",
        observations=(observation(),),
        captured_at=NOW,
    )
    assert visible.observations[0].source_id == "switch-a"
    baseline = InterventionOutcome(
        scenario_id="scenario-1",
        run_id="baseline",
        action=keep_action(),
        random_schedule_digest="c" * 16,
        latency_ms=10,
        packet_loss_ratio=0.1,
        jitter_ms=1,
        deadline_violation_rate=0,
    )
    intervention = InterventionOutcome(
        scenario_id="scenario-1",
        run_id="intervention",
        action=reroute_action(),
        random_schedule_digest="c" * 16,
        latency_ms=8,
        packet_loss_ratio=0.1,
        jitter_ms=1,
        deadline_violation_rate=0,
    )
    pair = PairedCounterfactualOutcome(
        pair_id="pair-1", baseline=baseline, intervention=intervention
    )
    assert pair.intervention.latency_ms < pair.baseline.latency_ms
    with pytest.raises(ValidationError, match="same random schedule"):
        PairedCounterfactualOutcome(
            pair_id="bad",
            baseline=baseline,
            intervention=InterventionOutcome(
                scenario_id=intervention.scenario_id,
                run_id=intervention.run_id,
                action=intervention.action,
                random_schedule_digest="d" * 16,
                latency_ms=intervention.latency_ms,
                packet_loss_ratio=intervention.packet_loss_ratio,
                jitter_ms=intervention.jitter_ms,
                deadline_violation_rate=intervention.deadline_violation_rate,
            ),
        )


def test_breach_validation() -> None:
    breach = ContractBreach(
        breach_id="breach-1",
        contract_id="contract-1",
        violated_clause_ids=("fresh",),
        affected_estimand="delay",
        severity=BreachSeverity.HIGH,
        occurred_at=NOW,
        recommended_fallback=keep_action(),
        audit_digest="d" * 16,
    )
    assert breach.severity is BreachSeverity.HIGH
    with pytest.raises(ValidationError, match="unique"):
        ContractBreach(
            breach_id="duplicate",
            contract_id=breach.contract_id,
            violated_clause_ids=("fresh", "fresh"),
            affected_estimand=breach.affected_estimand,
            severity=breach.severity,
            occurred_at=NOW,
            recommended_fallback=keep_action(),
            audit_digest="d" * 16,
        )


def test_remaining_domain_validation_branches() -> None:
    variable = CausalVariable(
        variable_id="route_action",
        display_name="Route action",
        role=VariableRole.TREATMENT,
        observed=True,
    )
    with pytest.raises(ValidationError, match="variable IDs"):
        TemporalCausalGraph(
            graph_id="duplicate-vars",
            version=1,
            scope="routing",
            variables=(variable, variable),
        )
    delay = CausalVariable(
        variable_id="delay",
        display_name="Delay",
        role=VariableRole.OUTCOME,
        observed=True,
    )
    edge = TemporalEdge(source_variable_id="route_action", target_variable_id="delay")
    with pytest.raises(ValidationError, match="edges must be unique"):
        TemporalCausalGraph(
            graph_id="duplicate-edges",
            version=1,
            scope="routing",
            variables=(variable, delay),
            edges=(edge, edge),
        )
    with pytest.raises(ValidationError, match="nonempty"):
        AdjustmentSet(adjustment_set_id="blank", variable_ids=("",))
    with pytest.raises(ValidationError, match="unique"):
        AdjustmentSet(adjustment_set_id="duplicate", variable_ids=("x", "x"))
    with pytest.raises(ValidationError, match="mediator variable"):
        IdentificationStrategy(
            strategy_id="duplicate-mediator",
            method=IdentificationMethod.FRONTDOOR,
            mediator_variable_ids=("m", "m"),
            assumptions=("x",),
        )
    with pytest.raises(ValidationError, match="timezone-aware"):
        TelemetryObservation(
            observation_id="naive",
            source_id="switch-a",
            variable_id="queue_occupancy",
            value=1.0,
            observed_at=datetime(2026, 7, 10, 7, 0),
            quality=ObservationQuality(
                sample_age_ms=0,
                measurement_delay_ms=0,
                clock_skew_ms=0,
                sampling_rate_hz=1,
            ),
            trust_score=1,
        )
    with pytest.raises(ValidationError, match="failure tolerance"):
        RedundancyClause(clause_id="bad", minimum_distinct_sources=1, failure_tolerance=1)
    with pytest.raises(ValidationError, match="timezone-aware"):
        DeadlineClause(clause_id="naive", deadline=datetime(2026, 7, 10, 7, 0))
    with pytest.raises(ValidationError, match="required variable"):
        TelemetryContract(
            contract_id="duplicate-required",
            proposal_id="proposal-1",
            query=query(),
            required_variable_ids=("x", "x"),
            allowed_strategies=(strategy(),),
            clauses=(FreshnessClause(clause_id="fresh", maximum_age_ms=1),),
            fallback_action=keep_action(),
        )
    with pytest.raises(ValidationError, match="outside source capability"):
        ContractOffer(
            offer_id="bad-rate",
            contract_id="contract-1",
            capability=capability(),
            offered_variable_ids=("queue_occupancy",),
            sampling_rates_hz={"queue_occupancy": 100.0},
        )
    with pytest.raises(ValidationError, match="selected source"):
        ContractCommitment(
            commitment_id="duplicate-source",
            contract_id="contract-1",
            selected_source_ids=("switch-a", "switch-a"),
            selected_field_ids=("queue_occupancy",),
            sampling_rates_hz={"queue_occupancy": 1.0},
            synchronization_epoch=NOW,
            valid_from=NOW,
            valid_until=NOW + timedelta(seconds=1),
            evidence_digests=("x" * 16,),
            expected_total_cost=0,
            fallback_action=keep_action(),
        )
    repeated = ClauseAssessment(clause_id="fresh", state=ClauseSatisfaction.SATISFIED)
    with pytest.raises(ValidationError, match="unique"):
        ContractStatus(
            contract_id="contract-1",
            state=ContractState.ACTIVE,
            clause_assessments=(repeated, repeated),
            current_quality=observation().quality,
            source_available=True,
        )
    with pytest.raises(ValidationError, match="safe"):
        SafetyDecision(disposition=SafetyDisposition.SAFE, invariant_ids=("loss",))
    with pytest.raises(ValidationError, match="timezone-aware"):
        EvidenceRecord(
            evidence_record_id="naive",
            action_id="action",
            graph_id="graph",
            decision=AdmissionDecision.ADMIT,
            reason_codes=(ReasonCode.IDENTIFIABLE,),
            safety_decision=SafetyDecision(disposition=SafetyDisposition.SAFE),
            recorded_at=datetime(2026, 7, 10, 7, 0),
        )
