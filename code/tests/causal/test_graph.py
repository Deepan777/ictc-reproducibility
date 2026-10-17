from __future__ import annotations

import pytest

from ictc.causal import CausalDAG, GraphComplexityError, InvalidCausalGraphError
from ictc.domain.enums import VariableRole
from ictc.domain.models import CausalVariable, TemporalCausalGraph, TemporalEdge


def test_rejects_invalid_graphs_and_unknown_nodes() -> None:
    with pytest.raises(InvalidCausalGraphError, match="nonempty"):
        CausalDAG((), ())
    with pytest.raises(InvalidCausalGraphError, match="unique"):
        CausalDAG(("A", "A"), ())
    with pytest.raises(InvalidCausalGraphError, match="edges must be unique"):
        CausalDAG(("A", "B"), (("A", "B"), ("A", "B")))
    with pytest.raises(InvalidCausalGraphError, match="acyclic"):
        CausalDAG(("A", "B"), (("A", "B"), ("B", "A")))
    with pytest.raises(InvalidCausalGraphError, match="self loop"):
        CausalDAG(("A",), (("A", "A"),))
    with pytest.raises(InvalidCausalGraphError, match="unknown"):
        CausalDAG(("A",), (("A", "B"),))
    with pytest.raises(KeyError, match="unknown causal graph"):
        CausalDAG(("A",), ()).parents("missing")
    with pytest.raises(InvalidCausalGraphError, match="observed-node"):
        CausalDAG(("A",), (), ("missing",))


def test_builds_from_the_validated_domain_graph() -> None:
    source = CausalVariable(
        variable_id="X", display_name="Treatment", role=VariableRole.TREATMENT, observed=True
    )
    outcome = CausalVariable(
        variable_id="Y", display_name="Outcome", role=VariableRole.OUTCOME, observed=False
    )
    domain_graph = TemporalCausalGraph(
        graph_id="g",
        version=1,
        scope="test",
        variables=(source, outcome),
        edges=(TemporalEdge(source_variable_id="X", target_variable_id="Y"),),
    )

    graph = CausalDAG.from_temporal_graph(domain_graph)
    assert graph.nodes == ("X", "Y")
    assert graph.edges == (("X", "Y"),)
    assert graph.observed_nodes == {"X"}
    assert tuple(graph.iter_edges()) == graph.edges


def test_ancestors_descendants_and_mutilation_are_explicit() -> None:
    graph = CausalDAG(
        ("U", "X", "M", "Y"),
        (("U", "X"), ("X", "M"), ("M", "Y")),
        ("X", "M", "Y"),
    )

    assert graph.parents("X") == {"U"}
    assert graph.children("X") == {"M"}
    assert graph.ancestors("Y") == {"U", "X", "M"}
    assert graph.descendants("U") == {"X", "M", "Y"}
    assert graph.ancestors(("X", "Y")) == {"U", "M"}
    assert graph.topological_order() == ("U", "X", "M", "Y")

    assert graph.intervene("X").edges == (("M", "Y"), ("X", "M"))
    assert graph.remove_outgoing("X").edges == (("M", "Y"), ("U", "X"))
    assert graph.observed_nodes == {"X", "M", "Y"}


@pytest.mark.parametrize(
    ("edges", "conditioned", "expected"),
    [
        ((("X", "M"), ("M", "Y")), (), False),
        ((("X", "M"), ("M", "Y")), ("M",), True),
        ((("U", "X"), ("U", "Y")), (), False),
        ((("U", "X"), ("U", "Y")), ("U",), True),
        ((("X", "C"), ("Y", "C")), (), True),
        ((("X", "C"), ("Y", "C")), ("C",), False),
    ],
)
def test_known_chain_fork_and_collider_d_separation(
    edges: tuple[tuple[str, str], tuple[str, str]],
    conditioned: tuple[str, ...],
    expected: bool,
) -> None:
    nodes = {node for edge in edges for node in edge}
    graph = CausalDAG(nodes, edges)
    assert graph.is_d_separated("X", "Y", conditioned) is expected


def test_conditioning_on_collider_descendant_opens_path() -> None:
    graph = CausalDAG(
        ("X", "Y", "C", "D"),
        (("X", "C"), ("Y", "C"), ("C", "D")),
    )
    path = ("X", "C", "Y")

    assert graph.is_path_active(path) is False
    assert graph.is_path_active(path, ("C",)) is True
    assert graph.is_path_active(path, ("D",)) is True
    assert graph.is_d_separated("X", "Y") is True
    assert graph.is_d_separated("X", "Y", ("D",)) is False


def test_backdoor_diagnostics_report_only_paths_entering_treatment() -> None:
    graph = CausalDAG(
        ("U", "X", "M", "Y"),
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
    )

    assert graph.backdoor_paths("X", "Y") == (("X", "U", "Y"),)
    assert graph.open_backdoor_paths("X", "Y") == (("X", "U", "Y"),)
    assert graph.open_backdoor_paths("X", "Y", ("U",)) == ()


def test_d_separation_rejects_overlapping_sets_and_invalid_paths() -> None:
    graph = CausalDAG(("X", "Y", "Z"), (("X", "Z"), ("Z", "Y")))
    with pytest.raises(ValueError, match="disjoint"):
        graph.is_d_separated("X", "Y", ("X",))
    with pytest.raises(ValueError, match="non-edge"):
        graph.is_path_active(("X", "Y"))
    with pytest.raises(ValueError, match="simple path"):
        graph.is_path_active(("X", "Z", "X"))


def test_simple_path_diagnostic_has_a_declared_bound() -> None:
    graph = CausalDAG(
        ("X", "A", "B", "Y"),
        (("X", "A"), ("X", "B"), ("A", "Y"), ("B", "Y")),
    )
    with pytest.raises(GraphComplexityError, match="max_paths=1"):
        graph.simple_paths("X", "Y", max_paths=1)
