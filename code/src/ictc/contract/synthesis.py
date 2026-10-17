"""Minimum-cost robust telemetry selection for M05."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from itertools import combinations
from typing import Self

from ortools.sat.python import cp_model
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ictc.domain.enums import PrivacyClass


class SynthesisStatus(StrEnum):
    EXACT_OPTIMAL = "EXACT_OPTIMAL"
    HEURISTIC_FEASIBLE = "HEURISTIC_FEASIBLE"
    INFEASIBLE = "INFEASIBLE"


class SynthesisReason(StrEnum):
    SELECTED = "SELECTED"
    NO_IDENTIFICATION_ALTERNATIVE = "NO_IDENTIFICATION_ALTERNATIVE"
    NO_QUALIFYING_CAPABILITY = "NO_QUALIFYING_CAPABILITY"
    INSUFFICIENT_FAILURE_REDUNDANCY = "INSUFFICIENT_FAILURE_REDUNDANCY"
    INSUFFICIENT_DOMAIN_DIVERSITY = "INSUFFICIENT_DOMAIN_DIVERSITY"
    BANDWIDTH_BUDGET_EXCEEDED = "BANDWIDTH_BUDGET_EXCEEDED"
    SOLVER_INFEASIBLE = "SOLVER_INFEASIBLE"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CapabilityAdvertisement(_ImmutableModel):
    """One source's independently selectable capability for one causal variable."""

    capability_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    administrative_domain: str = Field(min_length=1)
    variable_id: str = Field(min_length=1)
    minimum_sampling_rate_hz: int = Field(gt=0)
    maximum_sampling_rate_hz: int = Field(gt=0)
    expected_delay_ms: int = Field(ge=0)
    clock_accuracy_ms: int = Field(ge=0)
    trust_millionths: int = Field(ge=0, le=1_000_000)
    privacy_class: PrivacyClass
    provenance_mechanisms: tuple[str, ...] = Field(min_length=1)
    cost_microunits: int = Field(ge=0)
    bandwidth_bps: int = Field(ge=0)
    available_from: datetime
    available_until: datetime

    @field_validator("available_from", "available_until")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("capability availability must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_bounds(self) -> Self:
        if self.minimum_sampling_rate_hz > self.maximum_sampling_rate_hz:
            raise ValueError("minimum sampling rate cannot exceed maximum sampling rate")
        if self.available_until <= self.available_from:
            raise ValueError("capability availability interval must be positive")
        if len(self.provenance_mechanisms) != len(set(self.provenance_mechanisms)):
            raise ValueError("provenance mechanisms must be unique")
        return self


class SynthesisRequirements(_ImmutableModel):
    """Machine-readable constraints for one intervention-conditioned selection."""

    requirement_id: str = Field(min_length=1)
    variable_alternatives: tuple[tuple[str, ...], ...] = Field(min_length=1)
    required_sampling_rate_hz: int = Field(gt=0)
    maximum_freshness_ms: int = Field(ge=0)
    maximum_clock_skew_ms: int = Field(ge=0)
    minimum_trust_millionths: int = Field(ge=0, le=1_000_000)
    allowed_privacy_classes: frozenset[PrivacyClass] = Field(min_length=1)
    permitted_administrative_domains: frozenset[str] = Field(min_length=1)
    required_provenance_mechanisms: frozenset[str] = Field(min_length=1)
    collection_start: datetime
    deadline: datetime
    maximum_bandwidth_bps: int = Field(gt=0)
    minimum_distinct_domains_per_variable: int = Field(ge=1)
    failure_tolerance: int = Field(ge=0)

    @field_validator("collection_start", "deadline")
    @classmethod
    def timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("contract times must be timezone-aware")
        return value

    @model_validator(mode="after")
    def valid_requirements(self) -> Self:
        if self.deadline <= self.collection_start:
            raise ValueError("contract deadline must follow collection start")
        normalized = tuple(tuple(sorted(alternative)) for alternative in self.variable_alternatives)
        if any(not alternative for alternative in normalized):
            raise ValueError("identification alternatives must be nonempty")
        if any(len(item) != len(set(item)) for item in normalized):
            raise ValueError("variables in each alternative must be unique")
        if len(normalized) != len(set(normalized)):
            raise ValueError("identification alternatives must be unique")
        return self

    def clauses(self) -> tuple[dict[str, object], ...]:
        """Return every enforced clause in a serialization-ready form."""
        return (
            {"kind": "SAMPLING_RATE", "minimum_hz": self.required_sampling_rate_hz},
            {"kind": "FRESHNESS", "maximum_ms": self.maximum_freshness_ms},
            {"kind": "SYNCHRONIZATION", "maximum_skew_ms": self.maximum_clock_skew_ms},
            {"kind": "TRUST", "minimum_millionths": self.minimum_trust_millionths},
            {"kind": "PRIVACY", "allowed": sorted(self.allowed_privacy_classes)},
            {
                "kind": "PROVENANCE",
                "required": sorted(self.required_provenance_mechanisms),
            },
            {
                "kind": "DOMAIN",
                "permitted": sorted(self.permitted_administrative_domains),
                "minimum_distinct_per_variable": self.minimum_distinct_domains_per_variable,
            },
            {"kind": "REDUNDANCY", "failure_tolerance": self.failure_tolerance},
            {"kind": "BANDWIDTH", "maximum_bps": self.maximum_bandwidth_bps},
            {"kind": "DEADLINE", "at": self.deadline.isoformat()},
        )


class ContractSelection(_ImmutableModel):
    status: SynthesisStatus
    reason: SynthesisReason
    requirement_id: str
    selected_alternative: tuple[str, ...] | None = None
    selected_capability_ids: tuple[str, ...] = ()
    selected_source_ids: tuple[str, ...] = ()
    total_cost_microunits: int | None = Field(default=None, ge=0)
    total_bandwidth_bps: int | None = Field(default=None, ge=0)
    clauses: tuple[dict[str, object], ...]
    unsatisfied_variables: tuple[str, ...] = ()
    alternatives: tuple[tuple[str, ...], ...] = ()
    optimality_claimed: bool = False

    @model_validator(mode="after")
    def consistent_status(self) -> Self:
        if self.status is SynthesisStatus.EXACT_OPTIMAL and not self.optimality_claimed:
            raise ValueError("exact optimal result must claim optimality")
        if self.status is not SynthesisStatus.EXACT_OPTIMAL and self.optimality_claimed:
            raise ValueError("only an exact optimal result may claim optimality")
        if self.status is SynthesisStatus.INFEASIBLE and self.selected_capability_ids:
            raise ValueError("infeasible result cannot select capabilities")
        return self


def _qualifies(item: CapabilityAdvertisement, req: SynthesisRequirements) -> bool:
    acquisition_ms = int((req.deadline - req.collection_start).total_seconds() * 1000)
    return (
        item.maximum_sampling_rate_hz >= req.required_sampling_rate_hz
        and item.expected_delay_ms <= req.maximum_freshness_ms
        and item.expected_delay_ms <= acquisition_ms
        and item.clock_accuracy_ms <= req.maximum_clock_skew_ms
        and item.trust_millionths >= req.minimum_trust_millionths
        and item.privacy_class in req.allowed_privacy_classes
        and item.administrative_domain in req.permitted_administrative_domains
        and req.required_provenance_mechanisms <= set(item.provenance_mechanisms)
        and item.available_from <= req.collection_start
        and item.available_until >= req.deadline
    )


class ContractSynthesizer:
    def __init__(self, advertisements: Iterable[CapabilityAdvertisement]) -> None:
        items = tuple(sorted(advertisements, key=lambda item: item.capability_id))
        ids = [item.capability_id for item in items]
        pairs = [(item.source_id, item.variable_id) for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("capability IDs must be unique")
        if len(pairs) != len(set(pairs)):
            raise ValueError("a source may advertise a variable only once")
        self.advertisements = items

    def _candidates(
        self, alternative: tuple[str, ...], req: SynthesisRequirements
    ) -> tuple[CapabilityAdvertisement, ...]:
        return tuple(
            item
            for item in self.advertisements
            if item.variable_id in alternative and _qualifies(item, req)
        )

    @staticmethod
    def _diagnose(
        alternative: tuple[str, ...],
        candidates: tuple[CapabilityAdvertisement, ...],
        req: SynthesisRequirements,
    ) -> tuple[SynthesisReason, tuple[str, ...]] | None:
        by_variable = {
            variable: tuple(item for item in candidates if item.variable_id == variable)
            for variable in alternative
        }
        empty = tuple(variable for variable, items in by_variable.items() if not items)
        if empty:
            return SynthesisReason.NO_QUALIFYING_CAPABILITY, empty
        required_count = req.failure_tolerance + 1
        weak = tuple(
            variable
            for variable, items in by_variable.items()
            if len({item.source_id for item in items}) < required_count
        )
        if weak:
            return SynthesisReason.INSUFFICIENT_FAILURE_REDUNDANCY, weak
        undiverse = tuple(
            variable
            for variable, items in by_variable.items()
            if len({item.administrative_domain for item in items})
            < req.minimum_distinct_domains_per_variable
        )
        if undiverse:
            return SynthesisReason.INSUFFICIENT_DOMAIN_DIVERSITY, undiverse
        minimum_bandwidth = sum(
            sum(sorted(item.bandwidth_bps for item in items)[:required_count])
            for items in by_variable.values()
        )
        if minimum_bandwidth > req.maximum_bandwidth_bps:
            return SynthesisReason.BANDWIDTH_BUDGET_EXCEEDED, alternative
        return None

    @staticmethod
    def _selection(
        status: SynthesisStatus,
        req: SynthesisRequirements,
        alternative: tuple[str, ...],
        selected: tuple[CapabilityAdvertisement, ...],
        feasible_alternatives: tuple[tuple[str, ...], ...],
    ) -> ContractSelection:
        return ContractSelection(
            status=status,
            reason=SynthesisReason.SELECTED,
            requirement_id=req.requirement_id,
            selected_alternative=alternative,
            selected_capability_ids=tuple(item.capability_id for item in selected),
            selected_source_ids=tuple(sorted({item.source_id for item in selected})),
            total_cost_microunits=sum(item.cost_microunits for item in selected),
            total_bandwidth_bps=sum(item.bandwidth_bps for item in selected),
            clauses=req.clauses(),
            alternatives=feasible_alternatives,
            optimality_claimed=status is SynthesisStatus.EXACT_OPTIMAL,
        )

    def _solve_alternative_exact(
        self, alternative: tuple[str, ...], req: SynthesisRequirements
    ) -> tuple[CapabilityAdvertisement, ...] | None:
        candidates = self._candidates(alternative, req)
        if self._diagnose(alternative, candidates, req) is not None:
            return None
        model = cp_model.CpModel()
        variables = [model.new_bool_var(f"select_{index}") for index in range(len(candidates))]
        required_count = req.failure_tolerance + 1
        for variable in alternative:
            indexes = [i for i, item in enumerate(candidates) if item.variable_id == variable]
            model.add(sum(variables[i] for i in indexes) >= required_count)
            domains = sorted({candidates[i].administrative_domain for i in indexes})
            domain_used = []
            for domain in domains:
                marker = model.new_bool_var(f"domain_{variable}_{domain}")
                domain_indexes = [
                    i for i in indexes if candidates[i].administrative_domain == domain
                ]
                model.add_max_equality(marker, [variables[i] for i in domain_indexes])
                domain_used.append(marker)
            model.add(sum(domain_used) >= req.minimum_distinct_domains_per_variable)
        model.add(
            sum(item.bandwidth_bps * variables[i] for i, item in enumerate(candidates))
            <= req.maximum_bandwidth_bps
        )
        secondary_bound = (len(candidates) * (len(candidates) + 1) // 2) + 1
        model.minimize(
            sum(
                (item.cost_microunits * secondary_bound + i + 1) * variables[i]
                for i, item in enumerate(candidates)
            )
        )
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        status = solver.solve(model)
        if status != cp_model.OPTIMAL:
            return None
        return tuple(item for i, item in enumerate(candidates) if solver.value(variables[i]))

    def synthesize_exact(self, req: SynthesisRequirements) -> ContractSelection:
        solutions = [
            (tuple(sorted(alternative)), selected)
            for alternative in req.variable_alternatives
            if (selected := self._solve_alternative_exact(tuple(sorted(alternative)), req))
            is not None
        ]
        if not solutions:
            diagnostics = [
                self._diagnose(tuple(sorted(alt)), self._candidates(tuple(sorted(alt)), req), req)
                for alt in req.variable_alternatives
            ]
            reason, missing = next(
                (item for item in diagnostics if item is not None),
                (SynthesisReason.SOLVER_INFEASIBLE, ()),
            )
            return ContractSelection(
                status=SynthesisStatus.INFEASIBLE,
                reason=reason,
                requirement_id=req.requirement_id,
                clauses=req.clauses(),
                unsatisfied_variables=missing,
            )
        solutions.sort(
            key=lambda pair: (
                sum(item.cost_microunits for item in pair[1]),
                tuple(item.capability_id for item in pair[1]),
            )
        )
        alternative, selected = solutions[0]
        return self._selection(
            SynthesisStatus.EXACT_OPTIMAL,
            req,
            alternative,
            selected,
            tuple(pair[0] for pair in solutions[1:]),
        )

    def synthesize_heuristic(self, req: SynthesisRequirements) -> ContractSelection:
        feasible: list[tuple[tuple[str, ...], tuple[CapabilityAdvertisement, ...]]] = []
        required_count = req.failure_tolerance + 1
        for raw_alternative in req.variable_alternatives:
            alternative = tuple(sorted(raw_alternative))
            candidates = self._candidates(alternative, req)
            if self._diagnose(alternative, candidates, req) is not None:
                continue
            selected_items: list[CapabilityAdvertisement] = []
            for variable in alternative:
                options = sorted(
                    (item for item in candidates if item.variable_id == variable),
                    key=lambda item: (item.cost_microunits, item.bandwidth_bps, item.capability_id),
                )
                chosen: list[CapabilityAdvertisement] = []
                for item in options:
                    if (
                        len(chosen) >= required_count
                        and len({entry.administrative_domain for entry in chosen})
                        >= req.minimum_distinct_domains_per_variable
                    ):
                        break
                    chosen.append(item)
                selected_items.extend(chosen)
            canonical = tuple(sorted(selected_items, key=lambda item: item.capability_id))
            if sum(item.bandwidth_bps for item in canonical) <= req.maximum_bandwidth_bps:
                feasible.append((alternative, canonical))
        if not feasible:
            return ContractSelection(
                status=SynthesisStatus.INFEASIBLE,
                reason=SynthesisReason.SOLVER_INFEASIBLE,
                requirement_id=req.requirement_id,
                clauses=req.clauses(),
            )
        feasible.sort(
            key=lambda pair: (
                sum(item.cost_microunits for item in pair[1]),
                tuple(item.capability_id for item in pair[1]),
            )
        )
        alternative, chosen_selection = feasible[0]
        return self._selection(
            SynthesisStatus.HEURISTIC_FEASIBLE,
            req,
            alternative,
            chosen_selection,
            tuple(pair[0] for pair in feasible[1:]),
        )


def survives_source_failures(
    selection: ContractSelection,
    advertisements: Iterable[CapabilityAdvertisement],
    failure_tolerance: int,
) -> bool:
    """Exhaustively verify that every required variable survives any declared source failures."""
    if selection.selected_alternative is None:
        return False
    selected_ids = set(selection.selected_capability_ids)
    selected = tuple(item for item in advertisements if item.capability_id in selected_ids)
    sources = sorted({item.source_id for item in selected})
    for failed_count in range(failure_tolerance + 1):
        for failed in combinations(sources, failed_count):
            surviving_variables = {
                item.variable_id for item in selected if item.source_id not in failed
            }
            if not set(selection.selected_alternative) <= surviving_variables:
                return False
    return True
