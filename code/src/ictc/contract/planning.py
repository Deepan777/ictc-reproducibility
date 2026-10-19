"""End-to-end compilation from a declared causal query to a telemetry contract.

The planner composes the accepted finite-DAG backdoor identifier with the accepted M05 capability
optimizer.  It compiles declared premises; it does not validate the causal graph, estimate an
effect, read outcomes, consult the evaluation oracle, or admit an action.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from math import isfinite
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ictc.causal.graph import CausalDAG, InvalidCausalGraphError
from ictc.causal.identification import (
    BackdoorIdentifier,
    IdentifiabilityReason,
    IdentifiabilityResult,
)
from ictc.contract.synthesis import (
    CapabilityAdvertisement,
    ContractSelection,
    ContractSynthesizer,
    SynthesisRequirements,
    SynthesisStatus,
)
from ictc.domain.enums import ContractState, IdentificationMethod, PrivacyClass, VariableRole
from ictc.domain.models import (
    AdjustmentSet,
    BandwidthClause,
    DeadlineClause,
    FreshnessClause,
    IdentifiabilityQuery,
    IdentificationStrategy,
    PrivacyClause,
    ProvenanceClause,
    RedundancyClause,
    SynchronizationClause,
    TelemetryContract,
    TemporalCausalGraph,
    TrustClause,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class PlanningMode(StrEnum):
    EXACT = "EXACT"
    HEURISTIC = "HEURISTIC"


class PlanningStatus(StrEnum):
    EXACT_CONTRACT = "EXACT_CONTRACT"
    HEURISTIC_CONTRACT = "HEURISTIC_CONTRACT"
    IDENTIFICATION_FAILED = "IDENTIFICATION_FAILED"
    NO_ADJUSTMENT_REQUIRED = "NO_ADJUSTMENT_REQUIRED"
    CAPABILITY_INFEASIBLE = "CAPABILITY_INFEASIBLE"
    INVALID_INPUT = "INVALID_INPUT"


class PlanningReason(StrEnum):
    CONTRACT_SYNTHESIZED = "CONTRACT_SYNTHESIZED"
    GRAPH_SCOPE_MISMATCH = "GRAPH_SCOPE_MISMATCH"
    QUERY_ENDPOINT_MISSING = "QUERY_ENDPOINT_MISSING"
    QUERY_ROLE_MISMATCH = "QUERY_ROLE_MISMATCH"
    INVALID_CAUSAL_GRAPH = "INVALID_CAUSAL_GRAPH"
    EFFECT_NOT_IDENTIFIED = "EFFECT_NOT_IDENTIFIED"
    IDENTIFIED_WITHOUT_ADJUSTMENT = "IDENTIFIED_WITHOUT_ADJUSTMENT"
    CAPABILITY_SELECTION_INFEASIBLE = "CAPABILITY_SELECTION_INFEASIBLE"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _digest(payload: JsonValue) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _canonicalize(value: object) -> JsonValue:
    """Convert model payloads to stable JSON, sorting only semantically unordered sets."""
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump(mode="python"))
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in sorted(value.items())}
    if isinstance(value, set | frozenset):
        canonical = [_canonicalize(item) for item in value]
        return sorted(canonical, key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, list | tuple):
        return [_canonicalize(item) for item in value]
    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise TypeError(f"unsupported canonical payload value: {type(value).__name__}")


class ContractPlanningPolicy(_ImmutableModel):
    """Explicit noncausal evidence-quality and resource policy for contract generation."""

    policy_id: str = Field(min_length=1)
    mode: PlanningMode = PlanningMode.EXACT
    required_sampling_rate_hz: int = Field(gt=0)
    maximum_freshness_ms: int = Field(ge=0)
    maximum_clock_skew_ms: int = Field(ge=0)
    minimum_trust_millionths: int = Field(ge=0, le=1_000_000)
    privacy_class: PrivacyClass
    permitted_administrative_domains: tuple[str, ...] = Field(min_length=1)
    required_provenance_mechanisms: tuple[str, ...] = Field(min_length=1)
    collection_start: datetime
    deadline: datetime
    maximum_bandwidth_bps: int = Field(gt=0)
    minimum_distinct_domains_per_variable: int = Field(ge=1)
    failure_tolerance: int = Field(ge=0)
    max_adjustment_variables: int = Field(default=20, ge=0)

    @field_validator("collection_start", "deadline")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("planning times must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_policy(self) -> Self:
        if self.deadline <= self.collection_start:
            raise ValueError("planning deadline must follow collection start")
        if len(self.permitted_administrative_domains) != len(
            set(self.permitted_administrative_domains)
        ):
            raise ValueError("permitted administrative domains must be unique")
        if len(self.required_provenance_mechanisms) != len(
            set(self.required_provenance_mechanisms)
        ):
            raise ValueError("required provenance mechanisms must be unique")
        if any(not value for value in self.permitted_administrative_domains):
            raise ValueError("permitted administrative domains must be non-empty")
        if any(not value for value in self.required_provenance_mechanisms):
            raise ValueError("required provenance mechanisms must be non-empty")
        return self

    @property
    def digest(self) -> str:
        return _digest(_canonicalize(self))

    def requirements(
        self, requirement_id: str, alternatives: tuple[tuple[str, ...], ...]
    ) -> SynthesisRequirements:
        return SynthesisRequirements(
            requirement_id=requirement_id,
            variable_alternatives=alternatives,
            required_sampling_rate_hz=self.required_sampling_rate_hz,
            maximum_freshness_ms=self.maximum_freshness_ms,
            maximum_clock_skew_ms=self.maximum_clock_skew_ms,
            minimum_trust_millionths=self.minimum_trust_millionths,
            allowed_privacy_classes=frozenset((self.privacy_class,)),
            permitted_administrative_domains=frozenset(self.permitted_administrative_domains),
            required_provenance_mechanisms=frozenset(self.required_provenance_mechanisms),
            collection_start=self.collection_start,
            deadline=self.deadline,
            maximum_bandwidth_bps=self.maximum_bandwidth_bps,
            minimum_distinct_domains_per_variable=(self.minimum_distinct_domains_per_variable),
            failure_tolerance=self.failure_tolerance,
        )


class ContractPlan(_ImmutableModel):
    """Typed and digest-bound result of one end-to-end planning request."""

    plan_id: str = Field(min_length=1)
    status: PlanningStatus
    reason: PlanningReason
    query: IdentifiabilityQuery
    graph_id: str = Field(min_length=1)
    graph_version: int = Field(ge=1)
    graph_digest: str = Field(min_length=64, max_length=64)
    policy_digest: str = Field(min_length=64, max_length=64)
    capability_registry_digest: str = Field(min_length=64, max_length=64)
    identification_reason: IdentifiabilityReason | None = None
    minimal_adjustment_sets: tuple[tuple[str, ...], ...] = ()
    missing_variable_sets: tuple[tuple[str, ...], ...] = ()
    open_backdoor_paths: tuple[tuple[str, ...], ...] = ()
    requirements: SynthesisRequirements | None = None
    selection: ContractSelection | None = None
    contract: TelemetryContract | None = None
    detail: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent_result(self) -> Self:
        successful = {
            PlanningStatus.EXACT_CONTRACT,
            PlanningStatus.HEURISTIC_CONTRACT,
        }
        if self.status in successful:
            if self.requirements is None or self.selection is None or self.contract is None:
                raise ValueError(
                    "successful planning requires requirements, selection, and contract"
                )
            if self.selection.selected_alternative != self.contract.required_variable_ids:
                raise ValueError("selected alternative must equal contract required variables")
            if not self.minimal_adjustment_sets:
                raise ValueError("successful planning requires identification alternatives")
            if (
                self.status is PlanningStatus.EXACT_CONTRACT
                and self.selection.status is not SynthesisStatus.EXACT_OPTIMAL
            ):
                raise ValueError("exact contract requires an exact-optimal selection")
            if (
                self.status is PlanningStatus.HEURISTIC_CONTRACT
                and self.selection.status is not SynthesisStatus.HEURISTIC_FEASIBLE
            ):
                raise ValueError("heuristic contract requires a heuristic-feasible selection")
        elif self.contract is not None:
            raise ValueError("unsuccessful planning cannot emit a contract")
        if self.status is PlanningStatus.CAPABILITY_INFEASIBLE:
            if self.requirements is None or self.selection is None:
                raise ValueError("capability infeasibility must retain requirements and selection")
            if self.selection.status is not SynthesisStatus.INFEASIBLE:
                raise ValueError("capability infeasibility requires an infeasible selection")
        if self.status is PlanningStatus.NO_ADJUSTMENT_REQUIRED and (
            self.requirements is not None or self.selection is not None
        ):
            raise ValueError("no-adjustment result cannot fabricate capability requirements")
        return self

    def _payload_without_digest(self) -> dict[str, JsonValue]:
        payload = _canonicalize(self)
        if not isinstance(payload, dict):
            raise TypeError("contract plan must serialize to a JSON object")
        return payload

    @property
    def plan_digest(self) -> str:
        return _digest(self._payload_without_digest())

    def to_payload(self) -> dict[str, JsonValue]:
        payload = self._payload_without_digest()
        payload["plan_digest"] = self.plan_digest
        payload["claim_boundary"] = (
            "Compiles a declared finite-DAG backdoor query and explicit policy; it does not "
            "validate causal assumptions, estimate effects, or admit the action."
        )
        return payload


class QueryContractPlanner:
    """Compose exact identification and accepted M05 selection into one typed plan."""

    def __init__(self, advertisements: tuple[CapabilityAdvertisement, ...]) -> None:
        self.advertisements = tuple(sorted(advertisements, key=lambda item: item.capability_id))
        self.synthesizer = ContractSynthesizer(self.advertisements)

    @property
    def capability_registry_digest(self) -> str:
        payload = _canonicalize(self.advertisements)
        return _digest(payload)

    @staticmethod
    def _graph_digest(graph: TemporalCausalGraph) -> str:
        return _digest(_canonicalize(graph))

    def _base(
        self,
        *,
        plan_id: str,
        status: PlanningStatus,
        reason: PlanningReason,
        query: IdentifiabilityQuery,
        graph: TemporalCausalGraph,
        policy: ContractPlanningPolicy,
        detail: str,
        identification: IdentifiabilityResult | None = None,
        requirements: SynthesisRequirements | None = None,
        selection: ContractSelection | None = None,
        contract: TelemetryContract | None = None,
    ) -> ContractPlan:
        return ContractPlan(
            plan_id=plan_id,
            status=status,
            reason=reason,
            query=query,
            graph_id=graph.graph_id,
            graph_version=graph.version,
            graph_digest=self._graph_digest(graph),
            policy_digest=policy.digest,
            capability_registry_digest=self.capability_registry_digest,
            identification_reason=(None if identification is None else identification.reason),
            minimal_adjustment_sets=(
                () if identification is None else identification.minimal_adjustment_sets
            ),
            missing_variable_sets=(
                () if identification is None else identification.missing_variable_sets
            ),
            open_backdoor_paths=(
                () if identification is None else identification.open_backdoor_paths
            ),
            requirements=requirements,
            selection=selection,
            contract=contract,
            detail=detail,
        )

    @staticmethod
    def _contract(
        *,
        proposal_id: str,
        contract_id: str,
        query: IdentifiabilityQuery,
        policy: ContractPlanningPolicy,
        selection: ContractSelection,
    ) -> TelemetryContract:
        alternative = selection.selected_alternative
        if alternative is None:
            raise ValueError("feasible selection requires an adjustment alternative")
        adjustment = AdjustmentSet(
            adjustment_set_id=f"{contract_id}:adjustment",
            variable_ids=alternative,
        )
        strategy = IdentificationStrategy(
            strategy_id=f"{contract_id}:backdoor",
            method=IdentificationMethod.BACKDOOR,
            adjustment_set=adjustment,
            assumptions=(
                "declared finite DAG is causally valid for the query scope",
                "conditional exchangeability holds after the selected adjustment set",
                "consistency and action support hold in the target population",
            ),
        )
        clauses = (
            FreshnessClause(
                clause_id=f"{contract_id}:freshness",
                maximum_age_ms=policy.maximum_freshness_ms,
            ),
            SynchronizationClause(
                clause_id=f"{contract_id}:synchronization",
                maximum_clock_skew_ms=policy.maximum_clock_skew_ms,
            ),
            ProvenanceClause(
                clause_id=f"{contract_id}:provenance",
                accepted_mechanisms=tuple(sorted(policy.required_provenance_mechanisms)),
            ),
            TrustClause(
                clause_id=f"{contract_id}:trust",
                minimum_trust_score=policy.minimum_trust_millionths / 1_000_000,
            ),
            RedundancyClause(
                clause_id=f"{contract_id}:redundancy",
                minimum_distinct_sources=policy.failure_tolerance + 1,
                failure_tolerance=policy.failure_tolerance,
            ),
            PrivacyClause(
                clause_id=f"{contract_id}:privacy",
                privacy_class=policy.privacy_class,
                permitted_administrative_domains=tuple(
                    sorted(policy.permitted_administrative_domains)
                ),
            ),
            BandwidthClause(
                clause_id=f"{contract_id}:bandwidth",
                maximum_bandwidth_kbps=policy.maximum_bandwidth_bps / 1000,
            ),
            DeadlineClause(
                clause_id=f"{contract_id}:deadline",
                deadline=policy.deadline,
            ),
        )
        return TelemetryContract(
            contract_id=contract_id,
            proposal_id=proposal_id,
            query=query,
            required_variable_ids=alternative,
            allowed_strategies=(strategy,),
            clauses=clauses,
            fallback_action=query.baseline_action.action,
            state=ContractState.REQUIREMENTS_GENERATED,
        )

    def plan(
        self,
        *,
        plan_id: str,
        proposal_id: str,
        contract_id: str,
        query: IdentifiabilityQuery,
        graph: TemporalCausalGraph,
        policy: ContractPlanningPolicy,
        conditioning_variables: tuple[str, ...] = (),
        identification_mode: str = "exhaustive",
        path_diagnostics: bool = True,
    ) -> ContractPlan:
        if identification_mode not in ("exhaustive", "single"):
            raise ValueError("identification mode must be 'exhaustive' or 'single'")
        for label, value in (
            ("plan_id", plan_id),
            ("proposal_id", proposal_id),
            ("contract_id", contract_id),
        ):
            if not value.strip():
                raise ValueError(f"{label} must be non-empty")
        if query.graph_id != graph.graph_id:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.INVALID_INPUT,
                reason=PlanningReason.GRAPH_SCOPE_MISMATCH,
                query=query,
                graph=graph,
                policy=policy,
                detail=f"query graph {query.graph_id!r} does not match {graph.graph_id!r}",
            )
        variables = {variable.variable_id: variable for variable in graph.variables}
        missing = tuple(
            sorted({query.treatment_variable_id, query.outcome_variable_id} - set(variables))
        )
        if missing:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.INVALID_INPUT,
                reason=PlanningReason.QUERY_ENDPOINT_MISSING,
                query=query,
                graph=graph,
                policy=policy,
                detail=f"query endpoints are absent from graph: {list(missing)}",
            )
        if (
            variables[query.treatment_variable_id].role is not VariableRole.TREATMENT
            or variables[query.outcome_variable_id].role is not VariableRole.OUTCOME
        ):
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.INVALID_INPUT,
                reason=PlanningReason.QUERY_ROLE_MISMATCH,
                query=query,
                graph=graph,
                policy=policy,
                detail="query endpoints must have TREATMENT and OUTCOME roles",
            )
        try:
            dag = CausalDAG.from_temporal_graph(graph)
        except InvalidCausalGraphError as error:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.INVALID_INPUT,
                reason=PlanningReason.INVALID_CAUSAL_GRAPH,
                query=query,
                graph=graph,
                policy=policy,
                detail=str(error),
            )
        identifier = BackdoorIdentifier(
            dag,
            graph_id=graph.graph_id,
            max_adjustment_variables=policy.max_adjustment_variables,
        )
        identification = (
            identifier.identify(
                query.treatment_variable_id,
                query.outcome_variable_id,
                mode="single",
                conditioning_variables=conditioning_variables,
                path_diagnostics=path_diagnostics,
            )
            if identification_mode == "single"
            else identifier.identify_conditional(
                query.treatment_variable_id, query.outcome_variable_id, conditioning_variables
            )
            if conditioning_variables
            else identifier.identify_query(query, path_diagnostics=path_diagnostics)
        )
        if not identification.identified:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.IDENTIFICATION_FAILED,
                reason=PlanningReason.EFFECT_NOT_IDENTIFIED,
                query=query,
                graph=graph,
                policy=policy,
                detail=identification.detail,
                identification=identification,
            )
        alternatives = tuple(
            tuple(sorted(set(alternative) | set(conditioning_variables)))
            for alternative in identification.minimal_adjustment_sets
            if alternative or conditioning_variables
        )
        if not alternatives:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.NO_ADJUSTMENT_REQUIRED,
                reason=PlanningReason.IDENTIFIED_WITHOUT_ADJUSTMENT,
                query=query,
                graph=graph,
                policy=policy,
                detail=(
                    "The declared graph identifies the effect without a pre-action adjustment "
                    "set; no empty telemetry contract was generated."
                ),
                identification=identification,
            )
        requirements = policy.requirements(f"{plan_id}:requirements", alternatives)
        selection = (
            self.synthesizer.synthesize_exact(requirements)
            if policy.mode is PlanningMode.EXACT
            else self.synthesizer.synthesize_heuristic(requirements)
        )
        if selection.status is SynthesisStatus.INFEASIBLE:
            return self._base(
                plan_id=plan_id,
                status=PlanningStatus.CAPABILITY_INFEASIBLE,
                reason=PlanningReason.CAPABILITY_SELECTION_INFEASIBLE,
                query=query,
                graph=graph,
                policy=policy,
                detail=f"capability selection failed: {selection.reason.value}",
                identification=identification,
                requirements=requirements,
                selection=selection,
            )
        contract = self._contract(
            proposal_id=proposal_id,
            contract_id=contract_id,
            query=query,
            policy=policy,
            selection=selection,
        )
        status = (
            PlanningStatus.EXACT_CONTRACT
            if selection.status is SynthesisStatus.EXACT_OPTIMAL
            else PlanningStatus.HEURISTIC_CONTRACT
        )
        return self._base(
            plan_id=plan_id,
            status=status,
            reason=PlanningReason.CONTRACT_SYNTHESIZED,
            query=query,
            graph=graph,
            policy=policy,
            detail=(
                "Compiled the selected declared backdoor obligation into a telemetry contract; "
                "causal assumptions remain unvalidated premises."
            ),
            identification=identification,
            requirements=requirements,
            selection=selection,
            contract=contract,
        )
