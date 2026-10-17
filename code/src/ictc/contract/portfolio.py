"""Bounded multi-arm routing portfolio planning over intervention-specific contracts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ictc.contract.planning import (
    ContractPlan,
    ContractPlanningPolicy,
    PlanningStatus,
    QueryContractPlanner,
)
from ictc.contract.synthesis import CapabilityAdvertisement
from ictc.domain.enums import ActionKind
from ictc.domain.models import IdentifiabilityQuery, TemporalCausalGraph

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class PortfolioStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    NO_CONTRACT = "NO_CONTRACT"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class InterventionCandidate(_ImmutableModel):
    """One reroute arm and its intervention-specific planning premises."""

    candidate_id: str = Field(min_length=1)
    proposal_id: str = Field(min_length=1)
    contract_id: str = Field(min_length=1)
    query: IdentifiabilityQuery
    graph: TemporalCausalGraph
    policy: ContractPlanningPolicy

    @model_validator(mode="after")
    def routing_arm(self) -> Self:
        if self.query.proposed_action.kind is not ActionKind.REROUTE_TO_PATH:
            raise ValueError("portfolio candidates must be reroute interventions")
        if self.query.baseline_action.action.kind is not ActionKind.KEEP_CURRENT_PATH:
            raise ValueError("portfolio candidates require the keep-current-path baseline")
        return self


class CandidateContractPlan(_ImmutableModel):
    candidate_id: str = Field(min_length=1)
    plan: ContractPlan


class InterventionPortfolio(_ImmutableModel):
    """Complete, order-independent result retaining every candidate plan."""

    portfolio_id: str = Field(min_length=1)
    status: PortfolioStatus
    capability_registry_digest: str = Field(min_length=64, max_length=64)
    candidate_plans: tuple[CandidateContractPlan, ...] = Field(min_length=1)
    contract_candidate_ids: tuple[str, ...]
    no_contract_candidate_ids: tuple[str, ...]
    resource_pareto_candidate_ids: tuple[str, ...]

    @model_validator(mode="after")
    def consistent_membership(self) -> Self:
        candidate_ids = tuple(item.candidate_id for item in self.candidate_plans)
        if candidate_ids != tuple(sorted(candidate_ids)):
            raise ValueError("portfolio candidate plans must use canonical order")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("portfolio candidate IDs must be unique")
        if set(self.contract_candidate_ids) | set(self.no_contract_candidate_ids) != set(
            candidate_ids
        ):
            raise ValueError("portfolio contract membership must cover every candidate")
        if set(self.contract_candidate_ids) & set(self.no_contract_candidate_ids):
            raise ValueError("portfolio contract membership must be disjoint")
        if not set(self.resource_pareto_candidate_ids) <= set(self.contract_candidate_ids):
            raise ValueError("resource Pareto candidates must have generated contracts")
        expected_status = PortfolioStatus.NO_CONTRACT
        if len(self.contract_candidate_ids) == len(candidate_ids):
            expected_status = PortfolioStatus.COMPLETE
        elif self.contract_candidate_ids:
            expected_status = PortfolioStatus.PARTIAL
        if self.status is not expected_status:
            raise ValueError("portfolio status does not match retained candidate plans")
        return self

    def _payload_without_digest(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "n04-intervention-portfolio-v1",
            "portfolio_id": self.portfolio_id,
            "status": self.status.value,
            "capability_registry_digest": self.capability_registry_digest,
            "candidate_plans": [
                {"candidate_id": item.candidate_id, "plan": item.plan.to_payload()}
                for item in self.candidate_plans
            ],
            "contract_candidate_ids": list(self.contract_candidate_ids),
            "no_contract_candidate_ids": list(self.no_contract_candidate_ids),
            "resource_pareto_candidate_ids": list(self.resource_pareto_candidate_ids),
            "pareto_semantics": (
                "Nondominance on declared telemetry cost and bandwidth only; no causal benefit, "
                "safety, or networking-performance ranking."
            ),
        }

    @property
    def portfolio_digest(self) -> str:
        canonical = json.dumps(
            self._payload_without_digest(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_payload(self) -> dict[str, JsonValue]:
        payload = self._payload_without_digest()
        payload["portfolio_digest"] = self.portfolio_digest
        return payload


def _validate_candidates(candidates: tuple[InterventionCandidate, ...]) -> None:
    if not candidates:
        raise ValueError("intervention portfolio requires at least one candidate")
    unique_fields: dict[str, list[str | None]] = {
        "candidate IDs": [item.candidate_id for item in candidates],
        "query IDs": [item.query.query_id for item in candidates],
        "action IDs": [item.query.proposed_action.action_id for item in candidates],
        "path IDs": [item.query.proposed_action.path_id for item in candidates],
        "proposal IDs": [item.proposal_id for item in candidates],
        "contract IDs": [item.contract_id for item in candidates],
    }
    for label, values in unique_fields.items():
        if len(values) != len(set(values)):
            raise ValueError(f"portfolio {label} must be unique")
    first = candidates[0].query
    for candidate in candidates[1:]:
        query = candidate.query
        if query.baseline_action != first.baseline_action:
            raise ValueError("portfolio candidates must share one baseline action")
        if query.treatment_variable_id != first.treatment_variable_id:
            raise ValueError("portfolio candidates must share one treatment variable")
        if query.outcome_variable_id != first.outcome_variable_id:
            raise ValueError("portfolio candidates must share one outcome variable")
        if query.horizon != first.horizon:
            raise ValueError("portfolio candidates must share one evaluation horizon")


def _resource_pareto(plans: tuple[CandidateContractPlan, ...]) -> tuple[str, ...]:
    resources: list[tuple[str, int, int]] = []
    for item in plans:
        selection = item.plan.selection
        if item.plan.contract is None or selection is None:
            continue
        cost = selection.total_cost_microunits
        bandwidth = selection.total_bandwidth_bps
        if cost is None or bandwidth is None:
            continue
        resources.append((item.candidate_id, cost, bandwidth))
    nondominated = []
    for candidate_id, cost, bandwidth in resources:
        dominated = any(
            other_cost <= cost
            and other_bandwidth <= bandwidth
            and (other_cost < cost or other_bandwidth < bandwidth)
            for other_id, other_cost, other_bandwidth in resources
            if other_id != candidate_id
        )
        if not dominated:
            nondominated.append(candidate_id)
    return tuple(sorted(nondominated))


class InterventionPortfolioPlanner:
    """Plan each comparable reroute arm independently and retain every result."""

    def __init__(self, advertisements: tuple[CapabilityAdvertisement, ...]) -> None:
        self.query_planner = QueryContractPlanner(advertisements)

    def plan(
        self,
        *,
        portfolio_id: str,
        candidates: tuple[InterventionCandidate, ...],
    ) -> InterventionPortfolio:
        if not portfolio_id.strip():
            raise ValueError("portfolio ID must be non-empty")
        _validate_candidates(candidates)
        planned = tuple(
            CandidateContractPlan(
                candidate_id=candidate.candidate_id,
                plan=self.query_planner.plan(
                    plan_id=f"{portfolio_id}:{candidate.candidate_id}:plan",
                    proposal_id=candidate.proposal_id,
                    contract_id=candidate.contract_id,
                    query=candidate.query,
                    graph=candidate.graph,
                    policy=candidate.policy,
                ),
            )
            for candidate in sorted(candidates, key=lambda item: item.candidate_id)
        )
        contract_ids = tuple(
            item.candidate_id
            for item in planned
            if item.plan.status
            in {PlanningStatus.EXACT_CONTRACT, PlanningStatus.HEURISTIC_CONTRACT}
        )
        no_contract_ids = tuple(
            item.candidate_id for item in planned if item.candidate_id not in contract_ids
        )
        status = PortfolioStatus.NO_CONTRACT
        if len(contract_ids) == len(planned):
            status = PortfolioStatus.COMPLETE
        elif contract_ids:
            status = PortfolioStatus.PARTIAL
        return InterventionPortfolio(
            portfolio_id=portfolio_id,
            status=status,
            capability_registry_digest=self.query_planner.capability_registry_digest,
            candidate_plans=planned,
            contract_candidate_ids=contract_ids,
            no_contract_candidate_ids=no_contract_ids,
            resource_pareto_candidate_ids=_resource_pareto(planned),
        )
