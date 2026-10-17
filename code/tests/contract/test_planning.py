"""End-to-end and negative tests for N02 query-to-contract planning."""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from ictc.contract import (
    ContractPlan,
    ContractPlanningPolicy,
    PlanningMode,
    PlanningReason,
    PlanningStatus,
    QueryContractPlanner,
    SynthesisStatus,
    survives_source_failures,
)
from ictc.contract.planning_benchmark import (
    NOW,
    build_n02_evidence,
    capability_registry,
    planning_policy,
    render_n02_evidence,
    routing_graph,
    routing_query,
)
from ictc.domain.enums import VariableRole
from ictc.domain.models import CausalVariable, TemporalCausalGraph, TemporalEdge

ROOT = Path(__file__).resolve().parents[2]


def plan_exact() -> ContractPlan:
    return QueryContractPlanner(capability_registry()).plan(
        plan_id="plan",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=planning_policy(),
    )


def test_conditioning_context_is_mandatory_even_with_no_extra_adjustment() -> None:
    result = QueryContractPlanner(capability_registry()).plan(
        plan_id="conditional-plan",proposal_id="proposal",contract_id="contract",
        query=routing_query(),graph=routing_graph(),policy=planning_policy(),
        conditioning_variables=("U",),
    )
    assert result.contract is not None
    assert "U" in result.contract.required_variable_ids


def test_query_graph_policy_and_registry_compile_to_minimum_cost_contract() -> None:
    result = plan_exact()

    assert result.status is PlanningStatus.EXACT_CONTRACT
    assert result.reason is PlanningReason.CONTRACT_SYNTHESIZED
    assert result.minimal_adjustment_sets == (("U",), ("V",))
    assert result.selection is not None
    assert result.selection.status is SynthesisStatus.EXACT_OPTIMAL
    assert result.selection.selected_alternative == ("V",)
    assert result.selection.selected_capability_ids == ("observe-v-a", "observe-v-b")
    assert result.selection.total_cost_microunits == 5
    assert result.contract is not None
    assert result.contract.required_variable_ids == ("V",)
    assert result.contract.fallback_action == result.query.baseline_action.action
    strategy = result.contract.allowed_strategies[0]
    assert strategy.adjustment_set is not None
    assert strategy.adjustment_set.variable_ids == ("V",)
    assert len(result.contract.clauses) == 8
    assert result.requirements is not None
    assert result.requirements.required_sampling_rate_hz == 5
    assert survives_source_failures(result.selection, capability_registry(), 1)


def test_plan_digests_and_evidence_are_deterministic_and_order_independent() -> None:
    first = plan_exact()
    reversed_registry = tuple(reversed(capability_registry()))
    second = QueryContractPlanner(reversed_registry).plan(
        plan_id="plan",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=planning_policy(),
    )

    assert first.to_payload() == second.to_payload()
    assert first.plan_digest == second.plan_digest
    assert json.loads(render_n02_evidence()) == build_n02_evidence()
    assert render_n02_evidence() == render_n02_evidence()


def test_tracked_n02_evidence_matches_fresh_rendering() -> None:
    evidence = ROOT / "docs/milestones/evidence/n02_query_to_contract.json"
    assert evidence.read_text(encoding="utf-8") == render_n02_evidence()


def test_heuristic_plan_is_never_labeled_exact_or_optimal() -> None:
    result = QueryContractPlanner(capability_registry()).plan(
        plan_id="heuristic",
        proposal_id="proposal",
        contract_id="heuristic-contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=planning_policy(mode=PlanningMode.HEURISTIC),
    )

    assert result.status is PlanningStatus.HEURISTIC_CONTRACT
    assert result.selection is not None
    assert result.selection.status is SynthesisStatus.HEURISTIC_FEASIBLE
    assert result.selection.optimality_claimed is False

    exact_payload = plan_exact().model_dump()
    exact_payload["selection"] = result.selection.model_dump()
    with pytest.raises(ValidationError, match="exact-optimal"):
        ContractPlan.model_validate(exact_payload)

    missing_identification = plan_exact().model_dump()
    missing_identification["minimal_adjustment_sets"] = ()
    with pytest.raises(ValidationError, match="identification alternatives"):
        ContractPlan.model_validate(missing_identification)


def test_capability_infeasibility_retains_requirements_and_specific_selection_reason() -> None:
    result = QueryContractPlanner(()).plan(
        plan_id="infeasible",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=planning_policy(),
    )

    assert result.status is PlanningStatus.CAPABILITY_INFEASIBLE
    assert result.reason is PlanningReason.CAPABILITY_SELECTION_INFEASIBLE
    assert result.requirements is not None
    assert result.requirements.variable_alternatives == (("U",), ("V",))
    assert result.selection is not None
    assert result.selection.status is SynthesisStatus.INFEASIBLE
    assert result.contract is None


def test_unconfounded_graph_reports_no_adjustment_without_empty_contract() -> None:
    graph = TemporalCausalGraph(
        graph_id="direct",
        version=1,
        scope="direct effect",
        variables=(routing_graph().variables[0], routing_graph().variables[1]),
        edges=(TemporalEdge(source_variable_id="A", target_variable_id="Y"),),
    )
    result = QueryContractPlanner(()).plan(
        plan_id="direct",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(graph_id="direct"),
        graph=graph,
        policy=planning_policy(),
    )

    assert result.status is PlanningStatus.NO_ADJUSTMENT_REQUIRED
    assert result.reason is PlanningReason.IDENTIFIED_WITHOUT_ADJUSTMENT
    assert result.minimal_adjustment_sets == ((),)
    assert result.requirements is None
    assert result.selection is None
    assert result.contract is None


def test_latent_confounder_and_enumeration_bound_fail_identification_explicitly() -> None:
    source_variables = routing_graph().variables
    latent_variable = CausalVariable(
        variable_id="U",
        display_name="latent ingress pressure",
        role=VariableRole.LATENT,
        observed=False,
    )
    latent_graph = TemporalCausalGraph(
        graph_id="latent",
        version=1,
        scope="latent confounding",
        variables=(source_variables[0], source_variables[1], latent_variable),
        edges=(
            TemporalEdge(source_variable_id="U", target_variable_id="A"),
            TemporalEdge(source_variable_id="U", target_variable_id="Y"),
            TemporalEdge(source_variable_id="A", target_variable_id="Y"),
        ),
    )
    latent = QueryContractPlanner(capability_registry()).plan(
        plan_id="latent",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(graph_id="latent"),
        graph=latent_graph,
        policy=planning_policy(),
    )
    bounded_policy = planning_policy().model_copy(update={"max_adjustment_variables": 0})
    bounded = QueryContractPlanner(capability_registry()).plan(
        plan_id="bounded",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=bounded_policy,
    )

    assert latent.status is PlanningStatus.IDENTIFICATION_FAILED
    assert latent.missing_variable_sets == (("U",),)
    assert bounded.status is PlanningStatus.IDENTIFICATION_FAILED
    assert bounded.identification_reason is not None
    assert bounded.identification_reason.value == "ENUMERATION_LIMIT_EXCEEDED"


@pytest.mark.parametrize(
    ("case", "reason"),
    [
        ("graph", PlanningReason.GRAPH_SCOPE_MISMATCH),
        ("missing", PlanningReason.QUERY_ENDPOINT_MISSING),
        ("role", PlanningReason.QUERY_ROLE_MISMATCH),
        ("cycle", PlanningReason.INVALID_CAUSAL_GRAPH),
    ],
)
def test_invalid_query_graph_combinations_return_typed_failures(
    case: str, reason: PlanningReason
) -> None:
    graph = routing_graph()
    query = routing_query()
    if case == "graph":
        query = routing_query(graph_id="other")
    elif case == "missing":
        query = query.model_copy(update={"outcome_variable_id": "missing"})
    elif case == "role":
        changed = list(graph.variables)
        changed[0] = changed[0].model_copy(update={"role": VariableRole.CONFOUNDER})
        graph = graph.model_copy(update={"variables": tuple(changed)})
    else:
        graph = graph.model_copy(
            update={
                "edges": graph.edges
                + (TemporalEdge(source_variable_id="Y", target_variable_id="A"),)
            }
        )
    result = QueryContractPlanner(capability_registry()).plan(
        plan_id=case,
        proposal_id="proposal",
        contract_id="contract",
        query=query,
        graph=graph,
        policy=planning_policy(),
    )

    assert result.status is PlanningStatus.INVALID_INPUT
    assert result.reason is reason
    assert result.contract is None


def test_planner_rejects_empty_identifiers() -> None:
    planner = QueryContractPlanner(capability_registry())
    for field in ("plan_id", "proposal_id", "contract_id"):
        arguments = {
            "plan_id": "plan",
            "proposal_id": "proposal",
            "contract_id": "contract",
            field: "",
        }
        with pytest.raises(ValueError, match=field):
            planner.plan(
                **arguments,
                query=routing_query(),
                graph=routing_graph(),
                policy=planning_policy(),
            )


def test_policy_validation_rejects_ambiguous_or_temporally_invalid_inputs() -> None:
    payload = planning_policy().model_dump()
    cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"collection_start": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"deadline": NOW}, "follow collection"),
        (
            {"permitted_administrative_domains": ("domain-a", "domain-a")},
            "domains must be unique",
        ),
        ({"required_provenance_mechanisms": ("sha256", "sha256")}, "must be unique"),
        ({"permitted_administrative_domains": ("",)}, "must be non-empty"),
        ({"required_provenance_mechanisms": ("",)}, "must be non-empty"),
    )
    for update, message in cases:
        with pytest.raises(ValidationError, match=message):
            ContractPlanningPolicy.model_validate(payload | update)

    later = NOW + timedelta(seconds=5)
    valid = ContractPlanningPolicy.model_validate(
        payload | {"collection_start": later, "deadline": later + timedelta(seconds=1)}
    )
    assert valid.deadline > valid.collection_start


def test_planning_module_cannot_read_oracle_outcome_or_controller_decisions() -> None:
    planning_source = (ROOT / "src/ictc/contract/planning.py").read_text(encoding="utf-8")
    forbidden = (
        "ictc.oracle",
        "ictc.pilot",
        "m14_scoring",
        "reference_admissible",
        "post_action_outcome",
    )
    for token in forbidden:
        assert token not in planning_source
