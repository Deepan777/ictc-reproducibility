"""Immutable hash-chained generations for declared-input contract resynthesis."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ictc.contract.portfolio import (
    InterventionCandidate,
    InterventionPortfolio,
    InterventionPortfolioPlanner,
    PortfolioStatus,
)
from ictc.contract.synthesis import CapabilityAdvertisement

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


class ResynthesisTrigger(StrEnum):
    INITIAL = "INITIAL"
    CAPABILITY_CHANGE = "CAPABILITY_CHANGE"
    TOPOLOGY_CHANGE = "TOPOLOGY_CHANGE"
    POLICY_CHANGE = "POLICY_CHANGE"
    CONTRACT_BREACH = "CONTRACT_BREACH"


class GenerationDisposition(StrEnum):
    READY = "READY"
    PARTIAL_READY = "PARTIAL_READY"
    FAIL_CLOSED = "FAIL_CLOSED"


class _ImmutableModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


def _digest(payload: JsonValue) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_digests(candidates: tuple[InterventionCandidate, ...]) -> tuple[str, str, str]:
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    graph_payload: JsonValue = [
        {
            "candidate_id": item.candidate_id,
            "query": item.query.model_dump(mode="json"),
            "graph": item.graph.model_dump(mode="json"),
        }
        for item in ordered
    ]
    policy_payload: JsonValue = [
        {"candidate_id": item.candidate_id, "policy_digest": item.policy.digest} for item in ordered
    ]
    source_payload: JsonValue = [
        {
            "candidate_id": item.candidate_id,
            "proposal_id": item.proposal_id,
            "contract_id": item.contract_id,
            "query": item.query.model_dump(mode="json"),
            "graph": item.graph.model_dump(mode="json"),
            "policy_digest": item.policy.digest,
        }
        for item in ordered
    ]
    return _digest(source_payload), _digest(graph_payload), _digest(policy_payload)


def _generation_candidates(
    candidates: tuple[InterventionCandidate, ...], generation: int
) -> tuple[InterventionCandidate, ...]:
    return tuple(
        item.model_copy(
            update={
                "proposal_id": f"{item.proposal_id}:g{generation}",
                "contract_id": f"{item.contract_id}:g{generation}",
            }
        )
        for item in candidates
    )


class ContractGeneration(_ImmutableModel):
    generation: int = Field(ge=0)
    trigger: ResynthesisTrigger
    previous_generation_digest: str | None = Field(default=None, min_length=64, max_length=64)
    source_candidate_digest: str = Field(min_length=64, max_length=64)
    graph_bundle_digest: str = Field(min_length=64, max_length=64)
    policy_bundle_digest: str = Field(min_length=64, max_length=64)
    capability_registry_digest: str = Field(min_length=64, max_length=64)
    portfolio: InterventionPortfolio
    disposition: GenerationDisposition
    execution_ready_candidate_ids: tuple[str, ...]
    recorded_digest: str = Field(min_length=64, max_length=64)

    def _digest_payload(self) -> dict[str, JsonValue]:
        return {
            "generation": self.generation,
            "trigger": self.trigger.value,
            "previous_generation_digest": self.previous_generation_digest,
            "source_candidate_digest": self.source_candidate_digest,
            "graph_bundle_digest": self.graph_bundle_digest,
            "policy_bundle_digest": self.policy_bundle_digest,
            "capability_registry_digest": self.capability_registry_digest,
            "portfolio": self.portfolio.to_payload(),
            "disposition": self.disposition.value,
            "execution_ready_candidate_ids": list(self.execution_ready_candidate_ids),
        }

    @property
    def computed_digest(self) -> str:
        return _digest(self._digest_payload())

    @model_validator(mode="after")
    def valid_generation(self) -> Self:
        if self.recorded_digest != self.computed_digest:
            raise ValueError("generation recorded digest does not match its immutable payload")
        if self.capability_registry_digest != self.portfolio.capability_registry_digest:
            raise ValueError("generation capability digest must match its portfolio")
        expected = GenerationDisposition.FAIL_CLOSED
        if self.portfolio.status is PortfolioStatus.COMPLETE:
            expected = GenerationDisposition.READY
        elif self.portfolio.status is PortfolioStatus.PARTIAL:
            expected = GenerationDisposition.PARTIAL_READY
        if self.disposition is not expected:
            raise ValueError("generation disposition does not match portfolio status")
        if self.execution_ready_candidate_ids != self.portfolio.contract_candidate_ids:
            raise ValueError("execution readiness must come only from current generated contracts")
        return self

    @classmethod
    def build(
        cls,
        *,
        generation: int,
        trigger: ResynthesisTrigger,
        previous_generation_digest: str | None,
        source_candidate_digest: str,
        graph_bundle_digest: str,
        policy_bundle_digest: str,
        portfolio: InterventionPortfolio,
    ) -> ContractGeneration:
        disposition = GenerationDisposition.FAIL_CLOSED
        if portfolio.status is PortfolioStatus.COMPLETE:
            disposition = GenerationDisposition.READY
        elif portfolio.status is PortfolioStatus.PARTIAL:
            disposition = GenerationDisposition.PARTIAL_READY
        payload: dict[str, JsonValue] = {
            "generation": generation,
            "trigger": trigger.value,
            "previous_generation_digest": previous_generation_digest,
            "source_candidate_digest": source_candidate_digest,
            "graph_bundle_digest": graph_bundle_digest,
            "policy_bundle_digest": policy_bundle_digest,
            "capability_registry_digest": portfolio.capability_registry_digest,
            "portfolio": portfolio.to_payload(),
            "disposition": disposition.value,
            "execution_ready_candidate_ids": list(portfolio.contract_candidate_ids),
        }
        return cls(
            generation=generation,
            trigger=trigger,
            previous_generation_digest=previous_generation_digest,
            source_candidate_digest=source_candidate_digest,
            graph_bundle_digest=graph_bundle_digest,
            policy_bundle_digest=policy_bundle_digest,
            capability_registry_digest=portfolio.capability_registry_digest,
            portfolio=portfolio,
            disposition=disposition,
            execution_ready_candidate_ids=portfolio.contract_candidate_ids,
            recorded_digest=_digest(payload),
        )

    def to_payload(self) -> dict[str, JsonValue]:
        payload = self._digest_payload()
        payload["recorded_digest"] = self.recorded_digest
        return payload


class DynamicContractLedger(_ImmutableModel):
    ledger_id: str = Field(min_length=1)
    generations: tuple[ContractGeneration, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def valid_chain(self) -> Self:
        contract_ids: list[str] = []
        for index, generation in enumerate(self.generations):
            if generation.generation != index:
                raise ValueError("contract generations must be consecutive from zero")
            if index == 0:
                if generation.trigger is not ResynthesisTrigger.INITIAL:
                    raise ValueError("generation zero must use the INITIAL trigger")
                if generation.previous_generation_digest is not None:
                    raise ValueError("generation zero cannot have a previous digest")
            else:
                if generation.trigger is ResynthesisTrigger.INITIAL:
                    raise ValueError("later generations cannot use the INITIAL trigger")
                if (
                    generation.previous_generation_digest
                    != self.generations[index - 1].recorded_digest
                ):
                    raise ValueError("contract generation hash chain is broken")
            contract_ids.extend(
                item.plan.contract.contract_id
                for item in generation.portfolio.candidate_plans
                if item.plan.contract is not None
            )
        if len(contract_ids) != len(set(contract_ids)):
            raise ValueError("contract IDs must never be reused across generations")
        return self

    @property
    def current(self) -> ContractGeneration:
        return self.generations[-1]

    @property
    def ledger_digest(self) -> str:
        return _digest(
            {
                "ledger_id": self.ledger_id,
                "generation_digests": [item.recorded_digest for item in self.generations],
            }
        )

    def to_payload(self) -> dict[str, JsonValue]:
        return {
            "schema_version": "n05-dynamic-contract-ledger-v1",
            "ledger_id": self.ledger_id,
            "generations": [item.to_payload() for item in self.generations],
            "current_generation": self.current.generation,
            "current_execution_ready_candidate_ids": list(
                self.current.execution_ready_candidate_ids
            ),
            "ledger_digest": self.ledger_digest,
            "execution_boundary": (
                "Only the latest validated generation is execution-ready; superseded contracts "
                "remain audit evidence and are not reused."
            ),
        }


class DynamicContractResynthesizer:
    @staticmethod
    def _generation(
        *,
        ledger_id: str,
        generation: int,
        trigger: ResynthesisTrigger,
        previous_digest: str | None,
        candidates: tuple[InterventionCandidate, ...],
        advertisements: tuple[CapabilityAdvertisement, ...],
    ) -> ContractGeneration:
        source_digest, graph_digest, policy_digest = _candidate_digests(candidates)
        generated_candidates = _generation_candidates(candidates, generation)
        portfolio = InterventionPortfolioPlanner(advertisements).plan(
            portfolio_id=f"{ledger_id}:g{generation}",
            candidates=generated_candidates,
        )
        return ContractGeneration.build(
            generation=generation,
            trigger=trigger,
            previous_generation_digest=previous_digest,
            source_candidate_digest=source_digest,
            graph_bundle_digest=graph_digest,
            policy_bundle_digest=policy_digest,
            portfolio=portfolio,
        )

    def start(
        self,
        *,
        ledger_id: str,
        candidates: tuple[InterventionCandidate, ...],
        advertisements: tuple[CapabilityAdvertisement, ...],
    ) -> DynamicContractLedger:
        if not ledger_id.strip():
            raise ValueError("ledger ID must be non-empty")
        initial = self._generation(
            ledger_id=ledger_id,
            generation=0,
            trigger=ResynthesisTrigger.INITIAL,
            previous_digest=None,
            candidates=candidates,
            advertisements=advertisements,
        )
        return DynamicContractLedger(ledger_id=ledger_id, generations=(initial,))

    def resynthesize(
        self,
        ledger: DynamicContractLedger,
        *,
        trigger: ResynthesisTrigger,
        candidates: tuple[InterventionCandidate, ...],
        advertisements: tuple[CapabilityAdvertisement, ...],
    ) -> DynamicContractLedger:
        if trigger is ResynthesisTrigger.INITIAL:
            raise ValueError("INITIAL trigger is valid only for generation zero")
        generation = len(ledger.generations)
        next_generation = self._generation(
            ledger_id=ledger.ledger_id,
            generation=generation,
            trigger=trigger,
            previous_digest=ledger.current.recorded_digest,
            candidates=candidates,
            advertisements=advertisements,
        )
        if (
            trigger is ResynthesisTrigger.CAPABILITY_CHANGE
            and next_generation.capability_registry_digest
            == ledger.current.capability_registry_digest
        ):
            raise ValueError("CAPABILITY_CHANGE requires a changed capability registry digest")
        if (
            trigger is ResynthesisTrigger.TOPOLOGY_CHANGE
            and next_generation.graph_bundle_digest == ledger.current.graph_bundle_digest
        ):
            raise ValueError("TOPOLOGY_CHANGE requires a changed graph bundle digest")
        if (
            trigger is ResynthesisTrigger.POLICY_CHANGE
            and next_generation.policy_bundle_digest == ledger.current.policy_bundle_digest
        ):
            raise ValueError("POLICY_CHANGE requires a changed policy bundle digest")
        return DynamicContractLedger(
            ledger_id=ledger.ledger_id,
            generations=(*ledger.generations, next_generation),
        )
