"""Deterministic three-arm fixtures for N04 portfolio planning."""

from __future__ import annotations

import json

from ictc.contract.planning_benchmark import capability, planning_policy
from ictc.contract.portfolio import (
    InterventionCandidate,
    InterventionPortfolioPlanner,
    JsonValue,
)
from ictc.contract.synthesis import CapabilityAdvertisement
from ictc.domain.enums import ActionKind, VariableRole
from ictc.domain.models import (
    BaselineAction,
    CausalVariable,
    EvaluationHorizon,
    IdentifiabilityQuery,
    NetworkAction,
    TemporalCausalGraph,
    TemporalEdge,
)


def arm_graph(graph_id: str, confounder: str) -> TemporalCausalGraph:
    return TemporalCausalGraph(
        graph_id=graph_id,
        version=1,
        scope=f"reroute arm adjusted by {confounder}",
        variables=(
            CausalVariable(
                variable_id="A",
                display_name="route action",
                role=VariableRole.TREATMENT,
                observed=True,
            ),
            CausalVariable(
                variable_id="Y",
                display_name="post-action goodput",
                role=VariableRole.OUTCOME,
                observed=True,
                unit="Mbps",
            ),
            CausalVariable(
                variable_id=confounder,
                display_name=f"pre-action {confounder}",
                role=VariableRole.CONFOUNDER,
                observed=True,
            ),
        ),
        edges=(
            TemporalEdge(source_variable_id=confounder, target_variable_id="A"),
            TemporalEdge(source_variable_id=confounder, target_variable_id="Y"),
            TemporalEdge(source_variable_id="A", target_variable_id="Y"),
        ),
    )


def arm_candidate(candidate_id: str, path_id: str, confounder: str) -> InterventionCandidate:
    graph_id = f"n04-{candidate_id}-graph-v1"
    return InterventionCandidate(
        candidate_id=candidate_id,
        proposal_id=f"n04-{candidate_id}-proposal-v1",
        contract_id=f"n04-{candidate_id}-contract-v1",
        query=IdentifiabilityQuery(
            query_id=f"n04-{candidate_id}-query-v1",
            graph_id=graph_id,
            treatment_variable_id="A",
            outcome_variable_id="Y",
            baseline_action=BaselineAction(
                action=NetworkAction(
                    action_id="n04-common-baseline",
                    kind=ActionKind.KEEP_CURRENT_PATH,
                )
            ),
            proposed_action=NetworkAction(
                action_id=f"n04-{candidate_id}-action",
                kind=ActionKind.REROUTE_TO_PATH,
                path_id=path_id,
            ),
            horizon=EvaluationHorizon(duration_ms=1000),
        ),
        graph=arm_graph(graph_id, confounder),
        policy=planning_policy(),
    )


def portfolio_candidates() -> tuple[InterventionCandidate, ...]:
    return (
        arm_candidate("path-b", "path-b", "U"),
        arm_candidate("path-c", "path-c", "W"),
        arm_candidate("path-d", "path-d", "Z"),
    )


def portfolio_registry() -> tuple[CapabilityAdvertisement, ...]:
    return (
        capability("u-a", "u-source-a", "domain-a", "U", 2),
        capability("u-b", "u-source-b", "domain-b", "U", 3),
        capability("w-a", "w-source-a", "domain-a", "W", 4),
        capability("w-b", "w-source-b", "domain-b", "W", 5),
    )


def build_n04_evidence() -> dict[str, JsonValue]:
    portfolio = InterventionPortfolioPlanner(portfolio_registry()).plan(
        portfolio_id="n04-three-arm-portfolio-v1",
        candidates=portfolio_candidates(),
    )
    return {
        "schema_version": "n04-multi-arm-evidence-v1",
        "milestone": "N04",
        "portfolio": portfolio.to_payload(),
        "hand_checked_expectations": {
            "retained_candidates": ["path-b", "path-c", "path-d"],
            "contract_candidates": ["path-b", "path-c"],
            "no_contract_candidates": ["path-d"],
            "path_b_required_variable": ["U"],
            "path_b_cost": 5,
            "path_c_required_variable": ["W"],
            "path_c_cost": 9,
            "resource_pareto_candidates": ["path-b"],
        },
        "ranking_boundary": (
            "Pareto membership uses declared telemetry cost and bandwidth only. It does not rank "
            "causal effect, safety, utility, or network performance."
        ),
        "forbidden_inputs": {
            "oracle_labels": False,
            "post_action_outcomes": False,
            "controller_decisions": False,
            "m14_reference_flags": False,
        },
        "random_sampling_used": False,
    }


def render_n04_evidence() -> str:
    return json.dumps(build_n04_evidence(), indent=2, sort_keys=True) + "\n"
