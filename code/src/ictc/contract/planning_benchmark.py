"""Hand-checkable deterministic fixtures for the N02 query-to-contract bridge."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

from ictc.contract.planning import (
    ContractPlanningPolicy,
    JsonValue,
    PlanningMode,
    QueryContractPlanner,
)
from ictc.contract.synthesis import CapabilityAdvertisement, survives_source_failures
from ictc.domain.enums import ActionKind, PrivacyClass, VariableRole
from ictc.domain.models import (
    BaselineAction,
    CausalVariable,
    EvaluationHorizon,
    IdentifiabilityQuery,
    NetworkAction,
    TemporalCausalGraph,
    TemporalEdge,
)

NOW = datetime(2026, 7, 14, 12, 0, tzinfo=UTC)


def routing_graph(*, graph_id: str = "n02-routing-graph-v1") -> TemporalCausalGraph:
    """One backdoor path A <- U -> V -> Y, blockable by U or V."""
    return TemporalCausalGraph(
        graph_id=graph_id,
        version=1,
        scope="single reroute with pre-action path pressure",
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
                variable_id="U",
                display_name="ingress pressure",
                role=VariableRole.CONFOUNDER,
                observed=True,
            ),
            CausalVariable(
                variable_id="V",
                display_name="candidate queue pressure",
                role=VariableRole.CONFOUNDER,
                observed=True,
            ),
        ),
        edges=(
            TemporalEdge(source_variable_id="U", target_variable_id="A"),
            TemporalEdge(source_variable_id="U", target_variable_id="V"),
            TemporalEdge(source_variable_id="V", target_variable_id="Y"),
            TemporalEdge(source_variable_id="A", target_variable_id="Y"),
        ),
    )


def routing_query(*, graph_id: str = "n02-routing-graph-v1") -> IdentifiabilityQuery:
    return IdentifiabilityQuery(
        query_id="n02-reroute-goodput-query-v1",
        graph_id=graph_id,
        treatment_variable_id="A",
        outcome_variable_id="Y",
        baseline_action=BaselineAction(
            action=NetworkAction(action_id="keep", kind=ActionKind.KEEP_CURRENT_PATH)
        ),
        proposed_action=NetworkAction(
            action_id="reroute", kind=ActionKind.REROUTE_TO_PATH, path_id="backup"
        ),
        horizon=EvaluationHorizon(duration_ms=1000),
    )


def planning_policy(*, mode: PlanningMode = PlanningMode.EXACT) -> ContractPlanningPolicy:
    return ContractPlanningPolicy(
        policy_id="n02-quality-policy-v1",
        mode=mode,
        required_sampling_rate_hz=5,
        maximum_freshness_ms=20,
        maximum_clock_skew_ms=5,
        minimum_trust_millionths=800_000,
        privacy_class=PrivacyClass.INTERNAL,
        permitted_administrative_domains=("domain-a", "domain-b"),
        required_provenance_mechanisms=("sha256",),
        collection_start=NOW,
        deadline=NOW + timedelta(seconds=1),
        maximum_bandwidth_bps=100,
        minimum_distinct_domains_per_variable=2,
        failure_tolerance=1,
    )


def capability(
    capability_id: str,
    source_id: str,
    domain: str,
    variable: str,
    cost: int,
) -> CapabilityAdvertisement:
    return CapabilityAdvertisement(
        capability_id=capability_id,
        source_id=source_id,
        administrative_domain=domain,
        variable_id=variable,
        minimum_sampling_rate_hz=1,
        maximum_sampling_rate_hz=10,
        expected_delay_ms=5,
        clock_accuracy_ms=2,
        trust_millionths=900_000,
        privacy_class=PrivacyClass.INTERNAL,
        provenance_mechanisms=("sha256",),
        cost_microunits=cost,
        bandwidth_bps=10,
        available_from=NOW - timedelta(seconds=1),
        available_until=NOW + timedelta(seconds=2),
    )


def capability_registry() -> tuple[CapabilityAdvertisement, ...]:
    return (
        capability("observe-u-a", "source-u-a", "domain-a", "U", 8),
        capability("observe-u-b", "source-u-b", "domain-b", "U", 9),
        capability("observe-v-a", "source-v-a", "domain-a", "V", 2),
        capability("observe-v-b", "source-v-b", "domain-b", "V", 3),
    )


def build_n02_evidence() -> dict[str, JsonValue]:
    graph = routing_graph()
    query = routing_query()
    policy = planning_policy()
    registry = capability_registry()
    exact = QueryContractPlanner(registry).plan(
        plan_id="n02-exact-plan-v1",
        proposal_id="n02-proposal-v1",
        contract_id="n02-contract-v1",
        query=query,
        graph=graph,
        policy=policy,
    )
    infeasible = QueryContractPlanner(()).plan(
        plan_id="n02-infeasible-plan-v1",
        proposal_id="n02-proposal-infeasible-v1",
        contract_id="n02-contract-infeasible-v1",
        query=query,
        graph=graph,
        policy=policy,
    )
    return {
        "schema_version": "n02-query-to-contract-evidence-v1",
        "milestone": "N02",
        "exact_plan": exact.to_payload(),
        "infeasible_plan": infeasible.to_payload(),
        "hand_checked_expectations": {
            "minimal_adjustment_sets": [["U"], ["V"]],
            "selected_alternative": ["V"],
            "selected_capability_ids": ["observe-v-a", "observe-v-b"],
            "minimum_cost_microunits": 5,
            "survives_declared_source_failures": survives_source_failures(
                exact.selection, registry, policy.failure_tolerance
            )
            if exact.selection is not None
            else False,
        },
        "forbidden_inputs": {
            "evaluation_oracle": False,
            "post_action_outcomes": False,
            "controller_decisions": False,
            "m14_reference_flags": False,
        },
        "random_sampling_used": False,
        "claim_boundary": (
            "N02 demonstrates deterministic compilation of declared premises, not empirical "
            "validation of the graph, assumptions, advertisements, or intervention benefit."
        ),
    }


def render_n02_evidence() -> str:
    return json.dumps(build_n02_evidence(), indent=2, sort_keys=True) + "\n"
