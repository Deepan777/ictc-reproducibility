from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import pytest

from ictc.causal import (
    BackdoorIdentifier,
    CausalDAG,
    IdentifiabilityReason,
    IdentifiabilityResult,
)
from ictc.domain.enums import ActionKind
from ictc.domain.models import (
    BaselineAction,
    EvaluationHorizon,
    IdentifiabilityQuery,
    NetworkAction,
)


def test_randomized_treatment_needs_no_adjustment() -> None:
    graph = CausalDAG(("X", "M", "Y"), (("X", "M"), ("M", "Y")))
    result = BackdoorIdentifier(graph).identify("X", "Y")

    assert result.identified is True
    assert result.reason is IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
    assert result.minimal_adjustment_sets == ((),)
    assert result.open_backdoor_paths == ()


def test_single_confounder_has_one_minimal_adjustment_set() -> None:
    graph = CausalDAG(
        ("U", "X", "M", "Y"),
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
    )
    identifier = BackdoorIdentifier(graph)
    result = identifier.identify("X", "Y")

    assert result.identified is True
    assert result.reason is IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
    assert result.minimal_adjustment_sets == (("U",),)
    assert ("U",) in result.valid_adjustment_sets
    assert identifier.is_valid_adjustment_set("X", "Y", ("U",)) is True
    assert identifier.is_valid_adjustment_set("X", "Y", ("M",)) is False


def test_two_independent_confounders_must_both_be_adjusted() -> None:
    graph = CausalDAG(
        ("U", "V", "X", "Y"),
        (("U", "X"), ("U", "Y"), ("V", "X"), ("V", "Y")),
    )
    result = BackdoorIdentifier(graph).identify("X", "Y")

    assert result.minimal_adjustment_sets == (("U", "V"),)
    assert result.open_backdoor_paths == (("X", "U", "Y"), ("X", "V", "Y"))


def test_alternative_adjustment_sets_are_enumerated_and_minimized() -> None:
    graph = CausalDAG(
        ("U", "W", "X", "Y"),
        (("U", "W"), ("W", "X"), ("U", "Y")),
    )
    result = BackdoorIdentifier(graph).identify("X", "Y")

    assert result.minimal_adjustment_sets == (("U",), ("W",))
    assert result.valid_adjustment_sets == (("U",), ("W",), ("U", "W"))


def test_latent_confounder_returns_missing_variable_and_never_uses_proxy() -> None:
    graph = CausalDAG(
        ("U", "P", "X", "Y"),
        (("U", "P"), ("U", "X"), ("U", "Y")),
        observed_nodes=("P", "X", "Y"),
    )
    result = BackdoorIdentifier(graph).identify("X", "Y")

    assert result.identified is False
    assert result.reason is IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
    assert result.valid_adjustment_sets == ()
    assert result.missing_variable_sets == (("U",),)
    assert "P" not in result.missing_variable_sets[0]
    assert result.open_backdoor_paths == (("X", "U", "Y"),)
    with pytest.raises(ValueError, match="not declared observable"):
        BackdoorIdentifier(graph).identify("X", "Y", ("U", "X", "Y"))


def test_missing_treatment_or_outcome_is_not_reported_as_identifiable() -> None:
    graph = CausalDAG(("X", "Y"), (("X", "Y"),))
    result = BackdoorIdentifier(graph).identify("X", "Y", ("X",))

    assert result.identified is False
    assert result.reason is IdentifiabilityReason.TREATMENT_OR_OUTCOME_UNAVAILABLE
    assert result.missing_variable_sets == (("Y",),)


def test_enumeration_limit_is_an_explicit_failure() -> None:
    graph = CausalDAG(("A", "B", "X", "Y"), (("A", "X"), ("A", "Y")))
    result = BackdoorIdentifier(graph, max_adjustment_variables=1).identify("X", "Y")

    assert result.identified is False
    assert result.reason is IdentifiabilityReason.ENUMERATION_LIMIT_EXCEEDED
    assert "2^2" in result.detail


def test_theoretical_missing_set_search_has_the_same_exact_bound() -> None:
    graph = CausalDAG(
        ("U", "V", "X", "Y"),
        (("U", "X"), ("U", "Y"), ("V", "X"), ("V", "Y")),
        observed_nodes=("X", "Y"),
    )
    result = BackdoorIdentifier(graph, max_adjustment_variables=1).identify("X", "Y")

    assert result.identified is False
    assert result.reason is IdentifiabilityReason.ENUMERATION_LIMIT_EXCEEDED
    assert "Theoretical" in result.detail


def test_adjustment_benchmarks_match_networkx_independent_validation() -> None:
    edges = (
        ("A", "X"),
        ("A", "B"),
        ("B", "Y"),
        ("C", "X"),
        ("C", "Y"),
        ("X", "M"),
        ("M", "Y"),
    )
    nodes = ("A", "B", "C", "X", "M", "Y")
    graph = CausalDAG(nodes, edges)
    identifier = BackdoorIdentifier(graph)
    ours = identifier.enumerate_adjustment_sets("X", "Y")

    backdoor = nx.DiGraph(edges)
    backdoor.add_nodes_from(nodes)
    backdoor.remove_edges_from(list(backdoor.out_edges("X")))
    candidates = identifier.eligible_adjustment_variables("X", "Y", nodes)
    independently_valid = tuple(
        candidate
        for candidate in (
            (),
            *((node,) for node in candidates),
            *(
                (first, second)
                for index, first in enumerate(candidates)
                for second in candidates[index + 1 :]
            ),
            candidates,
        )
        if set(candidate).isdisjoint(graph.descendants("X"))
        and nx.is_d_separator(backdoor, {"X"}, {"Y"}, set(candidate))
    )

    assert set(ours) == set(independently_valid)
    assert identifier.minimal_adjustment_sets(ours) == (("A", "C"), ("B", "C"))


def test_minimal_set_helper_handles_nonmonotone_valid_collection() -> None:
    valid = ((), ("A",), ("B",), ("A", "B"))
    assert BackdoorIdentifier.minimal_adjustment_sets(valid) == ((),)


def test_invalid_engine_requests_fail_closed() -> None:
    graph = CausalDAG(("X", "Y"), (("X", "Y"),))
    with pytest.raises(ValueError, match="nonnegative"):
        BackdoorIdentifier(graph, max_adjustment_variables=-1)
    with pytest.raises(ValueError, match="distinct"):
        BackdoorIdentifier(graph).identify("X", "X")
    with pytest.raises(ValueError, match="not declared observable"):
        hidden = CausalDAG(("U", "X", "Y"), (("U", "X"),), ("X", "Y"))
        BackdoorIdentifier(hidden).enumerate_adjustment_sets("X", "Y", ("U",))
    assert BackdoorIdentifier(graph).is_valid_adjustment_set("X", "Y", ("X",)) is False

    with pytest.raises(ValueError, match="identified result"):
        IdentifiabilityResult(
            identified=True,
            reason=IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR,
            treatment="X",
            outcome="Y",
            available_variables=("X", "Y"),
        )
    with pytest.raises(ValueError, match="failed result"):
        IdentifiabilityResult(
            identified=False,
            reason=IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES,
            treatment="X",
            outcome="Y",
            available_variables=("X", "Y"),
            valid_adjustment_sets=((),),
        )


def _query(graph_id: str) -> IdentifiabilityQuery:
    return IdentifiabilityQuery(
        query_id="query-1",
        graph_id=graph_id,
        treatment_variable_id="X",
        outcome_variable_id="Y",
        baseline_action=BaselineAction(
            action=NetworkAction(action_id="keep", kind=ActionKind.KEEP_CURRENT_PATH)
        ),
        proposed_action=NetworkAction(
            action_id="reroute", kind=ActionKind.REROUTE_TO_PATH, path_id="path-b"
        ),
        horizon=EvaluationHorizon(duration_ms=100),
    )


def test_domain_query_enforces_graph_scope_before_identification() -> None:
    identifier = BackdoorIdentifier(CausalDAG(("X", "Y"), (("X", "Y"),)), graph_id="routing-v1")

    mismatch = identifier.identify_query(_query("wrong-graph"))
    assert mismatch.identified is False
    assert mismatch.reason is IdentifiabilityReason.GRAPH_SCOPE_MISMATCH
    assert "wrong-graph" in mismatch.detail

    matched = identifier.identify_query(_query("routing-v1"))
    assert matched.identified is True
    assert matched.reason is IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
