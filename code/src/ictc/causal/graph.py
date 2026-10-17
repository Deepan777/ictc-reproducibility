"""Inspectible finite-DAG operations used by the ICTC identifiability engine."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from ictc.domain.models import TemporalCausalGraph


class InvalidCausalGraphError(ValueError):
    """Raised when a graph is not a finite directed acyclic graph."""


class GraphComplexityError(RuntimeError):
    """Raised when a diagnostic path enumeration exceeds its declared bound."""


def _node_set(nodes: str | Iterable[str]) -> frozenset[str]:
    return frozenset((nodes,)) if isinstance(nodes, str) else frozenset(nodes)


@dataclass(frozen=True, slots=True)
class CausalDAG:
    """A deterministic finite DAG with optional observed-node declarations.

    This class deliberately implements the graph operations used by ICTC rather than
    delegating its scientific decision to a third-party causal package. NetworkX is used
    only in tests as an independent cross-check.
    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    observed_nodes: frozenset[str]

    def __init__(
        self,
        nodes: Iterable[str],
        edges: Iterable[tuple[str, str]],
        observed_nodes: Iterable[str] | None = None,
    ) -> None:
        canonical_nodes = tuple(sorted(nodes))
        canonical_edges = tuple(sorted(edges))
        if not canonical_nodes or any(not node for node in canonical_nodes):
            raise InvalidCausalGraphError("a causal DAG requires nonempty node names")
        if len(canonical_nodes) != len(set(canonical_nodes)):
            raise InvalidCausalGraphError("causal DAG node names must be unique")
        if len(canonical_edges) != len(set(canonical_edges)):
            raise InvalidCausalGraphError("causal DAG edges must be unique")

        known = set(canonical_nodes)
        for source, target in canonical_edges:
            if source not in known or target not in known:
                raise InvalidCausalGraphError("causal DAG edge references an unknown node")
            if source == target:
                raise InvalidCausalGraphError("causal DAG cannot contain a self loop")

        canonical_observed = frozenset(known if observed_nodes is None else observed_nodes)
        if not canonical_observed <= known:
            raise InvalidCausalGraphError("observed-node declaration references an unknown node")

        object.__setattr__(self, "nodes", canonical_nodes)
        object.__setattr__(self, "edges", canonical_edges)
        object.__setattr__(self, "observed_nodes", canonical_observed)
        if not self._is_acyclic():
            raise InvalidCausalGraphError("causal graph must be acyclic")

    @classmethod
    def from_temporal_graph(cls, graph: TemporalCausalGraph) -> CausalDAG:
        """Build the finite unrolled-DAG view supported by M04."""
        return cls(
            nodes=(variable.variable_id for variable in graph.variables),
            edges=((edge.source_variable_id, edge.target_variable_id) for edge in graph.edges),
            observed_nodes=(
                variable.variable_id for variable in graph.variables if variable.observed
            ),
        )

    def _is_acyclic(self) -> bool:
        indegree = {node: 0 for node in self.nodes}
        children: dict[str, set[str]] = {node: set() for node in self.nodes}
        for source, target in self.edges:
            indegree[target] += 1
            children[source].add(target)
        ready = deque(sorted(node for node, degree in indegree.items() if degree == 0))
        visited = 0
        while ready:
            node = ready.popleft()
            visited += 1
            for child in sorted(children[node]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
        return visited == len(self.nodes)

    def _validate_nodes(self, nodes: Iterable[str]) -> frozenset[str]:
        selected = frozenset(nodes)
        unknown = selected - set(self.nodes)
        if unknown:
            raise KeyError(f"unknown causal graph nodes: {sorted(unknown)}")
        return selected

    def parents(self, nodes: str | Iterable[str]) -> frozenset[str]:
        selected = self._validate_nodes(_node_set(nodes))
        return frozenset(source for source, target in self.edges if target in selected)

    def children(self, nodes: str | Iterable[str]) -> frozenset[str]:
        selected = self._validate_nodes(_node_set(nodes))
        return frozenset(target for source, target in self.edges if source in selected)

    def ancestors(
        self, nodes: str | Iterable[str], *, include_self: bool = False
    ) -> frozenset[str]:
        selected = self._validate_nodes(_node_set(nodes))
        result = set(selected) if include_self else set()
        frontier = list(selected)
        while frontier:
            parents = self.parents(frontier.pop())
            new_nodes = parents - result - selected
            result.update(parents - selected)
            frontier.extend(sorted(new_nodes))
        return frozenset(result)

    def descendants(
        self, nodes: str | Iterable[str], *, include_self: bool = False
    ) -> frozenset[str]:
        selected = self._validate_nodes(_node_set(nodes))
        result = set(selected) if include_self else set()
        frontier = list(selected)
        while frontier:
            children = self.children(frontier.pop())
            new_nodes = children - result - selected
            result.update(children - selected)
            frontier.extend(sorted(new_nodes))
        return frozenset(result)

    def induced_subgraph(self, nodes: Iterable[str]) -> CausalDAG:
        selected = self._validate_nodes(nodes)
        return CausalDAG(
            selected,
            (
                (source, target)
                for source, target in self.edges
                if source in selected and target in selected
            ),
            self.observed_nodes & selected,
        )

    def remove_incoming(self, nodes: str | Iterable[str]) -> CausalDAG:
        """Return G with incoming edges to intervention nodes removed (G_bar_X)."""
        selected = self._validate_nodes(_node_set(nodes))
        return CausalDAG(
            self.nodes,
            ((source, target) for source, target in self.edges if target not in selected),
            self.observed_nodes,
        )

    def remove_outgoing(self, nodes: str | Iterable[str]) -> CausalDAG:
        """Return G with outgoing edges from nodes removed (the backdoor graph G_under_X)."""
        selected = self._validate_nodes(_node_set(nodes))
        return CausalDAG(
            self.nodes,
            ((source, target) for source, target in self.edges if source not in selected),
            self.observed_nodes,
        )

    def intervene(self, nodes: str | Iterable[str]) -> CausalDAG:
        """Apply a do-intervention by deleting incoming arrows."""
        return self.remove_incoming(nodes)

    def is_d_separated(
        self,
        x: str | Iterable[str],
        y: str | Iterable[str],
        conditioned: Iterable[str] = (),
    ) -> bool:
        """Test d-separation using ancestral moralization.

        X and Y must be disjoint and neither may be conditioned. Empty X or Y is rejected
        because it would hide an invalid identifiability query.
        """
        x_nodes = self._validate_nodes(_node_set(x))
        y_nodes = self._validate_nodes(_node_set(y))
        z_nodes = self._validate_nodes(conditioned)
        if not x_nodes or not y_nodes:
            raise ValueError("d-separation endpoint sets must be nonempty")
        if x_nodes & y_nodes or (x_nodes | y_nodes) & z_nodes:
            raise ValueError("d-separation endpoint and conditioning sets must be disjoint")

        relevant = x_nodes | y_nodes | z_nodes
        ancestral_nodes = relevant | self.ancestors(relevant)
        ancestral = self.induced_subgraph(ancestral_nodes)
        adjacency: dict[str, set[str]] = {node: set() for node in ancestral.nodes}
        for source, target in ancestral.edges:
            adjacency[source].add(target)
            adjacency[target].add(source)
        for child in ancestral.nodes:
            parents = sorted(ancestral.parents(child))
            for index, first in enumerate(parents):
                for second in parents[index + 1 :]:
                    adjacency[first].add(second)
                    adjacency[second].add(first)

        reachable = set(x_nodes - z_nodes)
        frontier = list(reachable)
        while frontier:
            node = frontier.pop()
            for neighbour in adjacency[node] - z_nodes - reachable:
                reachable.add(neighbour)
                frontier.append(neighbour)
        return not bool(reachable & y_nodes)

    def simple_paths(
        self, source: str, target: str, *, max_paths: int = 10_000
    ) -> tuple[tuple[str, ...], ...]:
        """Enumerate deterministic simple paths in the undirected skeleton."""
        self._validate_nodes((source, target))
        if source == target:
            return ((source,),)
        adjacency: dict[str, set[str]] = {node: set() for node in self.nodes}
        for first, second in self.edges:
            adjacency[first].add(second)
            adjacency[second].add(first)

        paths: list[tuple[str, ...]] = []

        def visit(node: str, path: tuple[str, ...]) -> None:
            for neighbour in sorted(adjacency[node]):
                if neighbour in path:
                    continue
                candidate = (*path, neighbour)
                if neighbour == target:
                    paths.append(candidate)
                    if len(paths) > max_paths:
                        raise GraphComplexityError(
                            f"simple-path diagnostic exceeded max_paths={max_paths}"
                        )
                else:
                    visit(neighbour, candidate)

        visit(source, (source,))
        return tuple(paths)

    def is_path_active(self, path: tuple[str, ...], conditioned: Iterable[str] = ()) -> bool:
        """Return whether one skeleton path is d-connecting given the conditioning set."""
        if len(path) < 2 or len(path) != len(set(path)):
            raise ValueError("an active-path query requires a simple path with two endpoints")
        self._validate_nodes(path)
        edge_set = set(self.edges)
        for first, second in zip(path, path[1:], strict=False):
            if (first, second) not in edge_set and (second, first) not in edge_set:
                raise ValueError("active-path query contains a non-edge")

        z_nodes = self._validate_nodes(conditioned)
        conditioned_or_ancestor = z_nodes | self.ancestors(z_nodes)
        for previous, node, following in zip(path, path[1:], path[2:], strict=False):
            is_collider = (previous, node) in edge_set and (following, node) in edge_set
            if is_collider and node not in conditioned_or_ancestor:
                return False
            if not is_collider and node in z_nodes:
                return False
        return True

    def backdoor_paths(self, treatment: str, outcome: str) -> tuple[tuple[str, ...], ...]:
        """Return paths whose first edge has an arrow into the treatment."""
        incoming = self.parents(treatment)
        return tuple(
            path
            for path in self.simple_paths(treatment, outcome)
            if len(path) >= 2 and path[1] in incoming
        )

    def open_backdoor_paths(
        self, treatment: str, outcome: str, conditioned: Iterable[str] = ()
    ) -> tuple[tuple[str, ...], ...]:
        return tuple(
            path
            for path in self.backdoor_paths(treatment, outcome)
            if self.is_path_active(path, conditioned)
        )

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic topological ordering."""
        indegree = {node: 0 for node in self.nodes}
        for _, target in self.edges:
            indegree[target] += 1
        ready = sorted(node for node, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node = ready.pop(0)
            order.append(node)
            for child in sorted(self.children(node)):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort()
        return tuple(order)

    def iter_edges(self) -> Iterator[tuple[str, str]]:
        """Expose the canonical edges without a mutable graph object."""
        return iter(self.edges)
