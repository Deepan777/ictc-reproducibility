from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
from hypothesis import given, settings
from hypothesis import strategies as st

from ictc.causal import BackdoorIdentifier, CausalDAG


@st.composite
def dag_cases(
    draw: st.DrawFn,
) -> tuple[tuple[str, ...], tuple[tuple[str, str], ...], str, str, tuple[str, ...]]:
    size = draw(st.integers(min_value=2, max_value=7))
    nodes = tuple(f"N{index}" for index in range(size))
    possible_edges = [(nodes[i], nodes[j]) for i in range(size) for j in range(i + 1, size)]
    edges = tuple(
        edge
        for edge, selected in zip(
            possible_edges,
            draw(
                st.lists(st.booleans(), min_size=len(possible_edges), max_size=len(possible_edges))
            ),
            strict=True,
        )
        if selected
    )
    treatment_index = draw(st.integers(min_value=0, max_value=size - 2))
    outcome_index = draw(st.integers(min_value=treatment_index + 1, max_value=size - 1))
    remaining = [
        node for node in nodes if node not in {nodes[treatment_index], nodes[outcome_index]}
    ]
    conditioned = tuple(
        node
        for node, selected in zip(
            remaining,
            draw(st.lists(st.booleans(), min_size=len(remaining), max_size=len(remaining))),
            strict=True,
        )
        if selected
    )
    return nodes, edges, nodes[treatment_index], nodes[outcome_index], conditioned


@given(dag_cases())
@settings(max_examples=200, deadline=None)
def test_d_separation_matches_networkx_for_generated_dags(
    case: tuple[tuple[str, ...], tuple[tuple[str, str], ...], str, str, tuple[str, ...]],
) -> None:
    nodes, edges, treatment, outcome, conditioned = case
    ours = CausalDAG(nodes, edges).is_d_separated(treatment, outcome, conditioned)
    independent = nx.DiGraph(edges)
    independent.add_nodes_from(nodes)

    assert ours is nx.is_d_separator(independent, {treatment}, {outcome}, set(conditioned))


@given(dag_cases())
@settings(max_examples=100, deadline=None)
def test_backdoor_criterion_matches_networkx_for_generated_dags(
    case: tuple[tuple[str, ...], tuple[tuple[str, str], ...], str, str, tuple[str, ...]],
) -> None:
    nodes, edges, treatment, outcome, conditioned = case
    graph = CausalDAG(nodes, edges)
    identifier = BackdoorIdentifier(graph)
    ours = identifier.is_valid_adjustment_set(treatment, outcome, conditioned)

    independent = nx.DiGraph(edges)
    independent.add_nodes_from(nodes)
    independent.remove_edges_from(list(independent.out_edges(treatment)))
    expected = set(conditioned).isdisjoint(graph.descendants(treatment)) and nx.is_d_separator(
        independent, {treatment}, {outcome}, set(conditioned)
    )

    assert ours is expected
