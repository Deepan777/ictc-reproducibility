"""Immutable, validated models for the initial ICTC research scope."""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ictc.domain.enums import (
    ActionClass,
    ActionKind,
    AdmissionDecision,
    BreachSeverity,
    ClauseKind,
    ClauseSatisfaction,
    ContractState,
    IdentificationMethod,
    PrivacyClass,
    ReasonCode,
    SafetyDisposition,
    TargetOutcomeMetric,
    TelemetrySourceClass,
    VariableRole,
)


class ImmutableModel(BaseModel):
    """Base model that prevents undeclared fields and mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class NetworkAction(ImmutableModel):
    """A proposed or baseline routing action."""

    action_id: str = Field(min_length=1)
    action_class: ActionClass = ActionClass.ROUTING
    kind: ActionKind
    path_id: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_path_requirement(self) -> Self:
        if self.kind is ActionKind.REROUTE_TO_PATH and self.path_id is None:
            raise ValueError("REROUTE_TO_PATH requires path_id")
        if self.kind is ActionKind.KEEP_CURRENT_PATH and self.path_id is not None:
            raise ValueError("KEEP_CURRENT_PATH must not name a replacement path")
        return self


class BaselineAction(ImmutableModel):
    """The no-reroute action used as the initial counterfactual baseline."""

    action: NetworkAction

    @model_validator(mode="after")
    def validate_keep_current_path(self) -> Self:
        if self.action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("initial routing baseline must retain the current path")
        return self


class TargetOutcome(ImmutableModel):
    """A primary outcome metric and optional utility weight."""

    metric: TargetOutcomeMetric
    unit: str = Field(min_length=1)
    weight: float = Field(default=1.0, gt=0)


class EvaluationHorizon(ImmutableModel):
    """Positive forward-looking outcome horizon."""

    duration_ms: int = Field(gt=0)


class CausalVariable(ImmutableModel):
    """A declared variable and its causal role in the scoped graph."""

    variable_id: str = Field(min_length=1)
    display_name: str = Field(min_length=1)
    role: VariableRole
    observed: bool
    unit: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_latent_visibility(self) -> Self:
        if self.role is VariableRole.LATENT and self.observed:
            raise ValueError("a latent variable cannot be declared observed")
        return self


class TemporalEdge(ImmutableModel):
    """A directed causal relation with an explicit nonnegative temporal lag."""

    source_variable_id: str = Field(min_length=1)
    target_variable_id: str = Field(min_length=1)
    lag_steps: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_not_self_loop(self) -> Self:
        if self.source_variable_id == self.target_variable_id:
            raise ValueError("temporal causal edges cannot be self loops")
        return self


class TemporalCausalGraph(ImmutableModel):
    """Explicit, versioned causal graph; M01 does not learn or rewrite it."""

    graph_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    scope: str = Field(min_length=1)
    variables: tuple[CausalVariable, ...] = Field(min_length=1)
    edges: tuple[TemporalEdge, ...] = ()

    @model_validator(mode="after")
    def validate_graph_references(self) -> Self:
        variable_ids = [variable.variable_id for variable in self.variables]
        if len(variable_ids) != len(set(variable_ids)):
            raise ValueError("causal graph variable IDs must be unique")
        known = set(variable_ids)
        edge_pairs = [
            (edge.source_variable_id, edge.target_variable_id, edge.lag_steps)
            for edge in self.edges
        ]
        if len(edge_pairs) != len(set(edge_pairs)):
            raise ValueError("causal graph edges must be unique")
        for edge in self.edges:
            if edge.source_variable_id not in known or edge.target_variable_id not in known:
                raise ValueError("causal graph edge references an undeclared variable")
        return self


class AdjustmentSet(ImmutableModel):
    """Named adjustment-set candidate used for a declared estimand."""

    adjustment_set_id: str = Field(min_length=1)
    variable_ids: tuple[str, ...] = ()

    @field_validator("variable_ids")
    @classmethod
    def validate_unique_variables(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not variable_id for variable_id in value):
            raise ValueError("adjustment-set variable IDs must be nonempty")
        if len(value) != len(set(value)):
            raise ValueError("adjustment-set variable IDs must be unique")
        return value


class IdentificationStrategy(ImmutableModel):
    """A declared route for identifying an intervention effect."""

    strategy_id: str = Field(min_length=1)
    method: IdentificationMethod
    adjustment_set: AdjustmentSet | None = None
    mediator_variable_ids: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_method_requirements(self) -> Self:
        if self.method is IdentificationMethod.BACKDOOR and self.adjustment_set is None:
            raise ValueError("BACKDOOR requires an explicit adjustment set")
        if self.method is IdentificationMethod.FRONTDOOR and not self.mediator_variable_ids:
            raise ValueError("FRONTDOOR requires at least one mediator")
        if self.method is IdentificationMethod.NONE and self.adjustment_set is not None:
            raise ValueError("NONE cannot include an adjustment set")
        if len(self.mediator_variable_ids) != len(set(self.mediator_variable_ids)):
            raise ValueError("mediator variable IDs must be unique")
        return self


class IdentifiabilityQuery(ImmutableModel):
    """Formal query for the proposed-versus-baseline effect over a horizon."""

    query_id: str = Field(min_length=1)
    graph_id: str = Field(min_length=1)
    treatment_variable_id: str = Field(min_length=1)
    outcome_variable_id: str = Field(min_length=1)
    baseline_action: BaselineAction
    proposed_action: NetworkAction
    horizon: EvaluationHorizon

    @model_validator(mode="after")
    def validate_distinct_actions(self) -> Self:
        if self.proposed_action.kind is ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("identifiability query proposed action must be an intervention")
        return self


class TelemetrySource(ImmutableModel):
    """A named telemetry producer with trust and administrative-domain metadata."""

    source_id: str = Field(min_length=1)
    source_class: TelemetrySourceClass
    administrative_domain: str = Field(min_length=1)
    trust_score: float = Field(ge=0, le=1)
    available: bool = True


class ObservationQuality(ImmutableModel):
    """Quality metadata attached to a controller-visible observation."""

    sample_age_ms: int = Field(ge=0)
    measurement_delay_ms: int = Field(ge=0)
    clock_skew_ms: int = Field(ge=0)
    sampling_rate_hz: float = Field(gt=0)
    missing: bool = False


class TelemetryObservation(ImmutableModel):
    """One controller-visible measurement, never an oracle-state field."""

    observation_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    variable_id: str = Field(min_length=1)
    value: float | None = None
    observed_at: datetime
    quality: ObservationQuality
    trust_score: float = Field(ge=0, le=1)
    provenance_digest: str | None = Field(default=None, min_length=1)

    @field_validator("observed_at")
    @classmethod
    def validate_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observation timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_missingness(self) -> Self:
        if self.quality.missing and self.value is not None:
            raise ValueError("missing observation must not carry a value")
        if not self.quality.missing and self.value is None:
            raise ValueError("non-missing observation requires a value")
        if self.variable_id.startswith("true_") or self.source_id.lower() == "oracle":
            raise ValueError("controller-visible observations cannot expose oracle state")
        return self


class TelemetryCapability(ImmutableModel):
    """A source's measurable variables, quality bounds, and resource cost."""

    capability_id: str = Field(min_length=1)
    source: TelemetrySource
    measurable_variable_ids: tuple[str, ...] = Field(min_length=1)
    minimum_sampling_rate_hz: float = Field(gt=0)
    maximum_sampling_rate_hz: float = Field(gt=0)
    expected_delay_ms: int = Field(ge=0)
    clock_accuracy_ms: int = Field(ge=0)
    cost: float = Field(ge=0)
    bandwidth_kbps: float = Field(ge=0)
    available_from: datetime
    available_until: datetime

    @field_validator("available_from", "available_until")
    @classmethod
    def validate_capability_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capability availability timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_capability_bounds(self) -> Self:
        if self.minimum_sampling_rate_hz > self.maximum_sampling_rate_hz:
            raise ValueError("minimum sampling rate cannot exceed maximum sampling rate")
        if self.available_until <= self.available_from:
            raise ValueError("availability interval must be positive")
        if len(self.measurable_variable_ids) != len(set(self.measurable_variable_ids)):
            raise ValueError("measurable variable IDs must be unique")
        return self


class FreshnessClause(ImmutableModel):
    """Maximum accepted sample age."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.FRESHNESS
    maximum_age_ms: int = Field(ge=0)


class SynchronizationClause(ImmutableModel):
    """Maximum tolerated clock skew between evidence sources."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.SYNCHRONIZATION
    maximum_clock_skew_ms: int = Field(ge=0)


class ProvenanceClause(ImmutableModel):
    """Required verifiable provenance mechanisms."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.PROVENANCE
    accepted_mechanisms: tuple[str, ...] = Field(min_length=1)


class TrustClause(ImmutableModel):
    """Minimum trust score for selected evidence sources."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.TRUST
    minimum_trust_score: float = Field(ge=0, le=1)


class RedundancyClause(ImmutableModel):
    """Independent-source and failure-tolerance obligation."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.REDUNDANCY
    minimum_distinct_sources: int = Field(ge=1)
    failure_tolerance: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_failure_tolerance(self) -> Self:
        if self.failure_tolerance >= self.minimum_distinct_sources:
            raise ValueError("failure tolerance must be lower than distinct source count")
        return self


class PrivacyClause(ImmutableModel):
    """Privacy classification and permitted administrative domains."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.PRIVACY
    privacy_class: PrivacyClass
    permitted_administrative_domains: tuple[str, ...] = Field(min_length=1)


class BandwidthClause(ImmutableModel):
    """Maximum telemetry bandwidth allocated to the contract."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.BANDWIDTH
    maximum_bandwidth_kbps: float = Field(gt=0)


class DeadlineClause(ImmutableModel):
    """Deadline by which the evidence obligation must be satisfied."""

    clause_id: str = Field(min_length=1)
    kind: ClauseKind = ClauseKind.DEADLINE
    deadline: datetime

    @field_validator("deadline")
    @classmethod
    def validate_deadline_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("contract deadline must be timezone-aware")
        return value


ContractClause = (
    FreshnessClause
    | SynchronizationClause
    | ProvenanceClause
    | TrustClause
    | RedundancyClause
    | PrivacyClause
    | BandwidthClause
    | DeadlineClause
)


class TelemetryContract(ImmutableModel):
    """Machine-readable pre-actuation evidence obligation."""

    contract_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    query: IdentifiabilityQuery
    required_variable_ids: tuple[str, ...] = Field(min_length=1)
    allowed_strategies: tuple[IdentificationStrategy, ...] = Field(min_length=1)
    clauses: tuple[ContractClause, ...] = Field(min_length=1)
    fallback_action: NetworkAction
    state: ContractState = ContractState.PROPOSED

    @model_validator(mode="after")
    def validate_contract_contents(self) -> Self:
        if len(self.required_variable_ids) != len(set(self.required_variable_ids)):
            raise ValueError("contract required variable IDs must be unique")
        clause_ids = [clause.clause_id for clause in self.clauses]
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("contract clause IDs must be unique")
        clause_kinds = [clause.kind for clause in self.clauses]
        if len(clause_kinds) != len(set(clause_kinds)):
            raise ValueError("a contract may contain at most one clause per clause kind")
        if self.fallback_action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("initial routing fallback must retain the current path")
        return self


class ContractOffer(ImmutableModel):
    """A source's offer to satisfy part of a telemetry contract."""

    offer_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    capability: TelemetryCapability
    offered_variable_ids: tuple[str, ...] = Field(min_length=1)
    sampling_rates_hz: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_offer_variables(self) -> Self:
        offered = set(self.offered_variable_ids)
        measurable = set(self.capability.measurable_variable_ids)
        if not offered <= measurable:
            raise ValueError("offer cannot include a variable outside source capability")
        if set(self.sampling_rates_hz) != offered:
            raise ValueError("offer sampling rates must match offered variables")
        for rate in self.sampling_rates_hz.values():
            if (
                not self.capability.minimum_sampling_rate_hz
                <= rate
                <= self.capability.maximum_sampling_rate_hz
            ):
                raise ValueError("offered sampling rate is outside source capability")
        return self


class ContractCommitment(ImmutableModel):
    """Selected sources and evidence settings for one committed contract."""

    commitment_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    selected_source_ids: tuple[str, ...] = Field(min_length=1)
    selected_field_ids: tuple[str, ...] = Field(min_length=1)
    sampling_rates_hz: dict[str, float] = Field(min_length=1)
    synchronization_epoch: datetime
    valid_from: datetime
    valid_until: datetime
    evidence_digests: tuple[str, ...] = Field(min_length=1)
    expected_total_cost: float = Field(ge=0)
    fallback_action: NetworkAction

    @field_validator("synchronization_epoch", "valid_from", "valid_until")
    @classmethod
    def validate_commitment_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("commitment timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_commitment_contents(self) -> Self:
        if len(self.selected_source_ids) != len(set(self.selected_source_ids)):
            raise ValueError("selected source IDs must be unique")
        if len(self.selected_field_ids) != len(set(self.selected_field_ids)):
            raise ValueError("selected field IDs must be unique")
        if set(self.sampling_rates_hz) != set(self.selected_field_ids):
            raise ValueError("commitment sampling rates must match selected fields")
        if self.valid_until <= self.valid_from:
            raise ValueError("commitment validity interval must be positive")
        if self.fallback_action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("initial routing fallback must retain the current path")
        return self


class ClauseAssessment(ImmutableModel):
    """Current satisfaction result for a clause."""

    clause_id: str = Field(min_length=1)
    state: ClauseSatisfaction
    reason_codes: tuple[ReasonCode, ...] = ()

    @model_validator(mode="after")
    def validate_clause_reasoning(self) -> Self:
        if self.state is ClauseSatisfaction.SATISFIED and self.reason_codes:
            raise ValueError("satisfied clause must not carry failure reason codes")
        if (
            self.state in {ClauseSatisfaction.UNSATISFIED, ClauseSatisfaction.BREACHED}
            and not self.reason_codes
        ):
            raise ValueError("unsatisfied or breached clause requires a reason code")
        return self


class ContractStatus(ImmutableModel):
    """Protocol status including clause-level evidence health."""

    contract_id: str = Field(min_length=1)
    state: ContractState
    clause_assessments: tuple[ClauseAssessment, ...] = Field(min_length=1)
    current_quality: ObservationQuality
    source_available: bool
    uncertainty_width: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_assessment_ids(self) -> Self:
        clause_ids = [assessment.clause_id for assessment in self.clause_assessments]
        if len(clause_ids) != len(set(clause_ids)):
            raise ValueError("contract status clause IDs must be unique")
        return self


class ContractBreach(ImmutableModel):
    """A breach that invalidates evidence or requires a fallback."""

    breach_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    violated_clause_ids: tuple[str, ...] = Field(min_length=1)
    affected_estimand: str = Field(min_length=1)
    severity: BreachSeverity
    occurred_at: datetime
    recommended_fallback: NetworkAction
    audit_digest: str = Field(min_length=1)

    @field_validator("occurred_at")
    @classmethod
    def validate_breach_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("breach timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_breach_fallback(self) -> Self:
        if len(self.violated_clause_ids) != len(set(self.violated_clause_ids)):
            raise ValueError("violated clause IDs must be unique")
        if self.recommended_fallback.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("initial routing breach fallback must retain the current path")
        return self


class UncertaintyInterval(ImmutableModel):
    """Finite-sample uncertainty separate from identifiability."""

    lower: float
    upper: float
    confidence_level: float = Field(gt=0, lt=1)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        if self.lower > self.upper:
            raise ValueError("uncertainty lower bound cannot exceed upper bound")
        return self

    @property
    def width(self) -> float:
        """Return the nonnegative interval width."""
        return self.upper - self.lower


class CausalEffectEstimate(ImmutableModel):
    """Estimated intervention effect and the uncertainty attached to it."""

    estimand: str = Field(min_length=1)
    point_estimate: float
    unit: str = Field(min_length=1)
    uncertainty: UncertaintyInterval
    estimator_name: str = Field(min_length=1)


class SafetyDecision(ImmutableModel):
    """Safety result independent from admission and identification decisions."""

    disposition: SafetyDisposition
    invariant_ids: tuple[str, ...] = ()
    reason_codes: tuple[ReasonCode, ...] = ()

    @model_validator(mode="after")
    def validate_safety_result(self) -> Self:
        if self.disposition is SafetyDisposition.SAFE and self.invariant_ids:
            raise ValueError("safe decision must not report violated invariants")
        if self.disposition is SafetyDisposition.UNSAFE and not self.invariant_ids:
            raise ValueError("unsafe decision requires at least one violated invariant")
        return self


class EvidenceRecord(ImmutableModel):
    """Machine-readable audit record for every admission outcome."""

    evidence_record_id: str = Field(min_length=1)
    action_id: str = Field(min_length=1)
    contract_id: str | None = Field(default=None, min_length=1)
    graph_id: str = Field(min_length=1)
    decision: AdmissionDecision
    reason_codes: tuple[ReasonCode, ...] = Field(min_length=1)
    safety_decision: SafetyDecision
    recorded_at: datetime
    evidence_digests: tuple[str, ...] = ()

    @field_validator("recorded_at")
    @classmethod
    def validate_record_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("evidence record timestamp must be timezone-aware")
        return value


class ExperimentScenario(ImmutableModel):
    """Deterministic scenario metadata shared by paired counterfactual runs."""

    scenario_id: str = Field(min_length=1)
    topology_id: str = Field(min_length=1)
    topology_family: str = Field(min_length=1)
    node_count: int = Field(ge=2)
    link_count: int = Field(ge=1)
    traffic_model: str = Field(min_length=1)
    traffic_intensity: float = Field(ge=0)
    experiment_seed: int = Field(ge=0)
    code_commit: str = Field(min_length=7)
    config_hash: str = Field(min_length=16)
    environment_hash: str = Field(min_length=16)


class OracleState(ImmutableModel):
    """Ground-truth simulator state that is forbidden to tested controllers."""

    scenario_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    true_cross_traffic: float = Field(ge=0)
    true_available_capacity: float = Field(ge=0)
    true_queue_state: float = Field(ge=0)
    true_link_degradation: float = Field(ge=0, le=1)
    true_failure_state: bool
    true_background_flow_count: int = Field(ge=0)
    true_shared_resource_state: str = Field(min_length=1)


class ControllerVisibleState(ImmutableModel):
    """Controller input restricted to visible telemetry and declared metadata."""

    scenario_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    observations: tuple[TelemetryObservation, ...]
    captured_at: datetime

    @field_validator("captured_at")
    @classmethod
    def validate_visible_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("controller-visible state timestamp must be timezone-aware")
        return value


class InterventionOutcome(ImmutableModel):
    """Outcome from one fixed-action run under a shared random schedule."""

    scenario_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    action: NetworkAction
    random_schedule_digest: str = Field(min_length=16)
    latency_ms: float = Field(ge=0)
    packet_loss_ratio: float = Field(ge=0, le=1)
    jitter_ms: float = Field(ge=0)
    deadline_violation_rate: float = Field(ge=0, le=1)


class PairedCounterfactualOutcome(ImmutableModel):
    """Baseline and intervention outcomes sharing topology, seed, and schedules."""

    pair_id: str = Field(min_length=1)
    baseline: InterventionOutcome
    intervention: InterventionOutcome

    @model_validator(mode="after")
    def validate_pairing(self) -> Self:
        if self.baseline.scenario_id != self.intervention.scenario_id:
            raise ValueError("paired outcomes must use the same scenario")
        if self.baseline.random_schedule_digest != self.intervention.random_schedule_digest:
            raise ValueError("paired outcomes must share the same random schedule")
        if self.baseline.action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("paired baseline must retain the current path")
        if self.intervention.action.kind is not ActionKind.REROUTE_TO_PATH:
            raise ValueError("paired intervention must reroute to a path")
        return self
