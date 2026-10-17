"""Immutable generation, trigger, fail-closed, and recovery tests for N05."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ictc.contract import (
    ContractGeneration,
    DynamicContractLedger,
    DynamicContractResynthesizer,
    GenerationDisposition,
    ResynthesisTrigger,
)
from ictc.contract.dynamic_benchmark import (
    build_n05_evidence,
    build_n05_ledger,
    capability_loss_registry,
    recovery_registry,
    render_n05_evidence,
    topology_changed_candidates,
)
from ictc.contract.portfolio_benchmark import portfolio_candidates, portfolio_registry

ROOT = Path(__file__).resolve().parents[2]


def test_four_generations_cover_loss_fail_closed_and_recovery() -> None:
    ledger = build_n05_ledger()
    assert tuple(item.trigger for item in ledger.generations) == (
        ResynthesisTrigger.INITIAL,
        ResynthesisTrigger.CAPABILITY_CHANGE,
        ResynthesisTrigger.TOPOLOGY_CHANGE,
        ResynthesisTrigger.CAPABILITY_CHANGE,
    )
    assert tuple(item.disposition for item in ledger.generations) == (
        GenerationDisposition.PARTIAL_READY,
        GenerationDisposition.PARTIAL_READY,
        GenerationDisposition.FAIL_CLOSED,
        GenerationDisposition.PARTIAL_READY,
    )
    assert tuple(item.execution_ready_candidate_ids for item in ledger.generations) == (
        ("path-b", "path-c"),
        ("path-c",),
        (),
        ("path-c", "path-d"),
    )
    assert ledger.current.generation == 3
    assert ledger.current.execution_ready_candidate_ids == ("path-c", "path-d")


def test_hash_chain_and_contract_ids_are_unique_across_generations() -> None:
    ledger = build_n05_ledger()
    for previous, current in zip(ledger.generations, ledger.generations[1:], strict=False):
        assert current.previous_generation_digest == previous.recorded_digest
        assert current.recorded_digest == current.computed_digest
    contract_ids = [
        item.plan.contract.contract_id
        for generation in ledger.generations
        for item in generation.portfolio.candidate_plans
        if item.plan.contract is not None
    ]
    assert len(contract_ids) == len(set(contract_ids))
    assert all(
        f":g{generation.generation}" in item.plan.contract.contract_id
        for generation in ledger.generations
        for item in generation.portfolio.candidate_plans
        if item.plan.contract is not None
    )


def test_resynthesis_returns_new_ledger_without_mutating_prior_generations() -> None:
    engine = DynamicContractResynthesizer()
    initial = engine.start(
        ledger_id="immutable",
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    updated = engine.resynthesize(
        initial,
        trigger=ResynthesisTrigger.CAPABILITY_CHANGE,
        candidates=portfolio_candidates(),
        advertisements=capability_loss_registry(),
    )
    assert len(initial.generations) == 1
    assert len(updated.generations) == 2
    assert updated.generations[0] == initial.generations[0]


def test_breach_with_unchanged_inputs_still_creates_fresh_contract_generation() -> None:
    engine = DynamicContractResynthesizer()
    initial = engine.start(
        ledger_id="breach",
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    breached = engine.resynthesize(
        initial,
        trigger=ResynthesisTrigger.CONTRACT_BREACH,
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    assert breached.current.source_candidate_digest == initial.current.source_candidate_digest
    assert breached.current.capability_registry_digest == initial.current.capability_registry_digest
    old_ids = {
        item.plan.contract.contract_id
        for item in initial.current.portfolio.candidate_plans
        if item.plan.contract is not None
    }
    new_ids = {
        item.plan.contract.contract_id
        for item in breached.current.portfolio.candidate_plans
        if item.plan.contract is not None
    }
    assert old_ids.isdisjoint(new_ids)


@pytest.mark.parametrize(
    "trigger",
    [
        ResynthesisTrigger.CAPABILITY_CHANGE,
        ResynthesisTrigger.TOPOLOGY_CHANGE,
        ResynthesisTrigger.POLICY_CHANGE,
    ],
)
def test_declared_change_triggers_require_the_corresponding_digest_change(
    trigger: ResynthesisTrigger,
) -> None:
    engine = DynamicContractResynthesizer()
    initial = engine.start(
        ledger_id="unchanged",
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    with pytest.raises(ValueError, match="requires a changed"):
        engine.resynthesize(
            initial,
            trigger=trigger,
            candidates=portfolio_candidates(),
            advertisements=portfolio_registry(),
        )
    with pytest.raises(ValueError, match="generation zero"):
        engine.resynthesize(
            initial,
            trigger=ResynthesisTrigger.INITIAL,
            candidates=portfolio_candidates(),
            advertisements=portfolio_registry(),
        )


def test_policy_change_accepts_a_changed_policy_bundle() -> None:
    engine = DynamicContractResynthesizer()
    initial = engine.start(
        ledger_id="policy",
        candidates=portfolio_candidates(),
        advertisements=portfolio_registry(),
    )
    changed = tuple(
        item.model_copy(
            update={"policy": item.policy.model_copy(update={"maximum_bandwidth_bps": 200})}
        )
        for item in portfolio_candidates()
    )
    updated = engine.resynthesize(
        initial,
        trigger=ResynthesisTrigger.POLICY_CHANGE,
        candidates=changed,
        advertisements=portfolio_registry(),
    )
    assert updated.current.policy_bundle_digest != initial.current.policy_bundle_digest


def test_generation_and_chain_tampering_are_rejected() -> None:
    ledger = build_n05_ledger()
    payload = ledger.current.model_dump()
    payload["execution_ready_candidate_ids"] = ()
    with pytest.raises(ValidationError, match="recorded digest"):
        ContractGeneration.model_validate(payload)

    original = ledger.generations[1]
    broken = ContractGeneration.build(
        generation=original.generation,
        trigger=original.trigger,
        previous_generation_digest="0" * 64,
        source_candidate_digest=original.source_candidate_digest,
        graph_bundle_digest=original.graph_bundle_digest,
        policy_bundle_digest=original.policy_bundle_digest,
        portfolio=original.portfolio,
    )
    with pytest.raises(ValidationError, match="hash chain"):
        DynamicContractLedger(
            ledger_id=ledger.ledger_id,
            generations=(ledger.generations[0], broken),
        )


def test_dynamic_ledger_is_deterministic_and_evidence_is_tracked() -> None:
    first = build_n05_ledger()
    second = build_n05_ledger()
    assert first.to_payload() == second.to_payload()
    assert first.ledger_digest == second.ledger_digest
    rendered = render_n05_evidence()
    assert json.loads(rendered) == build_n05_evidence()
    evidence = ROOT / "docs/milestones/evidence/n05_dynamic_resynthesis.json"
    assert evidence.read_text(encoding="utf-8") == rendered


def test_invalid_ledger_id_and_dynamic_truth_imports_are_rejected() -> None:
    with pytest.raises(ValueError, match="ledger ID"):
        DynamicContractResynthesizer().start(
            ledger_id="",
            candidates=portfolio_candidates(),
            advertisements=portfolio_registry(),
        )
    source = (ROOT / "src/ictc/contract/dynamic.py").read_text(encoding="utf-8")
    for token in (
        "ictc.oracle",
        "ictc.pilot",
        "m14_scoring",
        "post_action_outcome",
        "reference_admissible",
        "AdmissionGate",
    ):
        assert token not in source


def test_benchmark_change_inputs_are_genuinely_distinct() -> None:
    assert capability_loss_registry() != portfolio_registry()
    assert topology_changed_candidates() != portfolio_candidates()
    assert recovery_registry() != capability_loss_registry()
