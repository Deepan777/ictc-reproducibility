"""Backdoor identifiability analysis with explicit success and failure results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

from ictc.causal.graph import CausalDAG, GraphComplexityError
from ictc.domain.models import IdentifiabilityQuery


class IdentifiabilityReason(StrEnum):
    """Machine-readable reasons emitted by the scoped M04 engine."""

    IDENTIFIED_WITHOUT_ADJUSTMENT = "IDENTIFIED_WITHOUT_ADJUSTMENT"
    IDENTIFIED_BY_BACKDOOR = "IDENTIFIED_BY_BACKDOOR"
    GRAPH_SCOPE_MISMATCH = "GRAPH_SCOPE_MISMATCH"
    TREATMENT_OR_OUTCOME_UNAVAILABLE = "TREATMENT_OR_OUTCOME_UNAVAILABLE"
    MISSING_ADJUSTMENT_VARIABLES = "MISSING_ADJUSTMENT_VARIABLES"
    ENUMERATION_LIMIT_EXCEEDED = "ENUMERATION_LIMIT_EXCEEDED"


@dataclass(frozen=True, slots=True)
class IdentifiabilityResult:
    """Complete, immutable outcome of one graphical identification request."""

    identified: bool
    reason: IdentifiabilityReason
    treatment: str
    outcome: str
    available_variables: tuple[str, ...]
    valid_adjustment_sets: tuple[tuple[str, ...], ...] = ()
    minimal_adjustment_sets: tuple[tuple[str, ...], ...] = ()
    open_backdoor_paths: tuple[tuple[str, ...], ...] = ()
    missing_variable_sets: tuple[tuple[str, ...], ...] = ()
    detail: str = ""
    conditioning_variables: tuple[str, ...] = ()
    diagnostic_truncated: bool = False

    def __post_init__(self) -> None:
        if self.identified and not self.valid_adjustment_sets:
            raise ValueError("an identified result must include at least one adjustment set")
        if not self.identified and self.valid_adjustment_sets:
            raise ValueError("a failed result cannot contain a valid adjustment set")


def _ordered_subsets(nodes: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    for size in range(len(nodes) + 1):
        yield from combinations(nodes, size)


class BackdoorIdentifier:
    """Exhaustive adjustment-set engine for finite observed DAGs.

    The enumeration is intentionally exact and inspectable. Its exponential bound is explicit;
    callers must use a different, separately validated method for graphs beyond that bound.
    """

    def __init__(
        self,
        graph: CausalDAG,
        *,
        graph_id: str | None = None,
        max_adjustment_variables: int = 20,
    ) -> None:
        if max_adjustment_variables < 0:
            raise ValueError("max_adjustment_variables must be nonnegative")
        self.graph = graph
        self.graph_id = graph_id
        self.max_adjustment_variables = max_adjustment_variables

    def _validate_endpoints(self, treatment: str, outcome: str) -> None:
        self.graph._validate_nodes((treatment, outcome))
        if treatment == outcome:
            raise ValueError("treatment and outcome must be distinct")

    def eligible_adjustment_variables(
        self, treatment: str, outcome: str, available_variables: Iterable[str]
    ) -> tuple[str, ...]:
        """Return available non-endpoint variables that are not treatment descendants."""
        self._validate_endpoints(treatment, outcome)
        available = self.graph._validate_nodes(available_variables)
        forbidden = {treatment, outcome} | set(self.graph.descendants(treatment))
        return tuple(sorted(available - forbidden))

    def is_valid_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        adjustment: Iterable[str],
    ) -> bool:
        """Evaluate Pearl's backdoor criterion for a candidate set."""
        self._validate_endpoints(treatment, outcome)
        selected = self.graph._validate_nodes(adjustment)
        if treatment in selected or outcome in selected:
            return False
        if selected & self.graph.descendants(treatment):
            return False
        backdoor_graph = self.graph.remove_outgoing(treatment)
        return backdoor_graph.is_d_separated(treatment, outcome, selected)

    def enumerate_adjustment_sets(
        self,
        treatment: str,
        outcome: str,
        available_variables: Iterable[str] | None = None,
    ) -> tuple[tuple[str, ...], ...]:
        available = (
            self.graph.observed_nodes
            if available_variables is None
            else frozenset(available_variables)
        )
        if not available <= self.graph.observed_nodes:
            unavailable = sorted(available - self.graph.observed_nodes)
            raise ValueError(f"variables are not declared observable: {unavailable}")
        candidates = self.eligible_adjustment_variables(treatment, outcome, available)
        if len(candidates) > self.max_adjustment_variables:
            raise OverflowError(
                "exact adjustment enumeration requires "
                f"2^{len(candidates)} candidates, above the configured "
                f"limit of {self.max_adjustment_variables} variables"
            )
        return tuple(
            candidate
            for candidate in _ordered_subsets(candidates)
            if self.is_valid_adjustment_set(treatment, outcome, candidate)
        )

    @staticmethod
    def minimal_adjustment_sets(
        valid_sets: Iterable[tuple[str, ...]],
    ) -> tuple[tuple[str, ...], ...]:
        canonical = tuple(dict.fromkeys(tuple(sorted(item)) for item in valid_sets))
        set_views = tuple(frozenset(item) for item in canonical)
        minimal = [
            item
            for index, item in enumerate(canonical)
            if not any(other < set_views[index] for other in set_views)
        ]
        return tuple(sorted(minimal, key=lambda item: (len(item), item)))

    def _theoretical_minimal_sets(
        self, treatment: str, outcome: str
    ) -> tuple[tuple[str, ...], ...]:
        candidates = self.eligible_adjustment_variables(treatment, outcome, self.graph.nodes)
        if len(candidates) > self.max_adjustment_variables:
            raise OverflowError
        valid = tuple(
            candidate
            for candidate in _ordered_subsets(candidates)
            if self.is_valid_adjustment_set(treatment, outcome, candidate)
        )
        return self.minimal_adjustment_sets(valid)

    def identify(
        self,
        treatment: str,
        outcome: str,
        available_variables: Iterable[str] | None = None,
        *,
        path_diagnostics: bool = True,
        mode: str = "exhaustive",
        conditioning_variables: Iterable[str] = (),
    ) -> IdentifiabilityResult:
        """Identify a total effect by adjustment or explain why current evidence cannot."""
        self._validate_endpoints(treatment, outcome)
        if mode not in ("exhaustive", "single"):
            raise ValueError("identification mode must be 'exhaustive' or 'single'")
        context = tuple(conditioning_variables)
        available = frozenset(
            self.graph.observed_nodes if available_variables is None else available_variables
        )
        self.graph._validate_nodes(available)
        if not available <= self.graph.observed_nodes:
            unavailable = sorted(available - self.graph.observed_nodes)
            raise ValueError(f"variables are not declared observable: {unavailable}")

        canonical_available = tuple(sorted(available))
        missing_endpoints = tuple(sorted({treatment, outcome} - available))
        truncated = False
        open_paths: tuple[tuple[str, ...], ...] = ()
        if path_diagnostics:
            try:
                open_paths = self.graph.open_backdoor_paths(treatment, outcome)
            except GraphComplexityError:
                truncated = True
        if mode == "single":
            selected = self.find_minimal_adjustment_set(treatment, outcome, available, context)
            single_valid = () if selected is None else (selected,)
            return IdentifiabilityResult(
                identified=selected is not None,
                reason=(
                    IdentifiabilityReason.TREATMENT_OR_OUTCOME_UNAVAILABLE
                    if missing_endpoints
                    else IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
                    if selected is None
                    else IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
                    if not selected
                    else IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
                ),
                treatment=treatment,
                outcome=outcome,
                available_variables=canonical_available,
                valid_adjustment_sets=single_valid,
                minimal_adjustment_sets=single_valid,
                open_backdoor_paths=open_paths,
                diagnostic_truncated=truncated,
                conditioning_variables=tuple(sorted(context)),
                detail=(
                    "One inclusion-minimal available back-door adjustment set."
                    if selected is not None
                    else "Not identified by back-door adjustment with the available variables."
                ),
            )
        if context:
            # Keep the existing exhaustive conditional result and semantics.
            return self.identify_conditional(treatment, outcome, context, available)
        if missing_endpoints:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.TREATMENT_OR_OUTCOME_UNAVAILABLE,
                treatment=treatment,
                outcome=outcome,
                available_variables=canonical_available,
                open_backdoor_paths=open_paths,
                diagnostic_truncated=truncated,
                missing_variable_sets=(missing_endpoints,),
                detail="Treatment assignment and outcome measurement must both be available.",
            )

        try:
            valid = self.enumerate_adjustment_sets(treatment, outcome, available)
        except OverflowError as error:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.ENUMERATION_LIMIT_EXCEEDED,
                treatment=treatment,
                outcome=outcome,
                available_variables=canonical_available,
                open_backdoor_paths=open_paths,
                diagnostic_truncated=truncated,
                detail=str(error),
            )

        if valid:
            minimal = self.minimal_adjustment_sets(valid)
            reason = (
                IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
                if () in valid
                else IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
            )
            return IdentifiabilityResult(
                identified=True,
                reason=reason,
                treatment=treatment,
                outcome=outcome,
                available_variables=canonical_available,
                valid_adjustment_sets=valid,
                minimal_adjustment_sets=minimal,
                open_backdoor_paths=open_paths,
                diagnostic_truncated=truncated,
                detail=(
                    "No open backdoor path requires adjustment."
                    if reason is IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
                    else "At least one available set satisfies the backdoor criterion."
                ),
            )

        try:
            theoretical = self._theoretical_minimal_sets(treatment, outcome)
        except OverflowError:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.ENUMERATION_LIMIT_EXCEEDED,
                treatment=treatment,
                outcome=outcome,
                available_variables=canonical_available,
                open_backdoor_paths=open_paths,
                diagnostic_truncated=truncated,
                detail="Theoretical missing-variable search exceeded the configured exact bound.",
            )

        missing_sets = tuple(
            sorted(
                {
                    tuple(sorted(set(candidate) - available))
                    for candidate in theoretical
                    if set(candidate) - available
                },
                key=lambda item: (len(item), item),
            )
        )
        return IdentifiabilityResult(
            identified=False,
            reason=IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES,
            treatment=treatment,
            outcome=outcome,
            available_variables=canonical_available,
            open_backdoor_paths=open_paths,
            diagnostic_truncated=truncated,
            missing_variable_sets=missing_sets,
            detail=(
                "No set of currently available observed variables blocks every backdoor path; "
                "missing-variable sets list exact graph variables, never proxies."
            ),
        )

    def find_minimal_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        available_variables: Iterable[str],
        conditioning_variables: Iterable[str] = (),
    ) -> tuple[str, ...] | None:
        """Find one inclusion-minimal set, or None, under Pearl's back-door test.

        Mandatory context is not returned as optional adjustment. The candidate
        construction and deletion use O(K) d-separation calls, not subset search.
        This neither lists all alternatives nor minimizes acquisition cost.
        """
        self._validate_endpoints(treatment, outcome)
        available = self.graph._validate_nodes(available_variables)
        context = self.graph._validate_nodes(conditioning_variables)
        forbidden = {treatment, outcome} | set(self.graph.descendants(treatment))
        if context & forbidden:
            raise ValueError(
                "conditioning context must exclude endpoints and treatment descendants"
            )
        if not available <= self.graph.observed_nodes:
            raise ValueError("available variables must be declared observable")
        if not (context | {treatment, outcome}) <= available:
            return None
        ancestors = self.graph.ancestors({treatment, outcome} | context)
        selected = set((ancestors & available) - forbidden - context)
        backdoor = self.graph.remove_outgoing(treatment)
        if not backdoor.is_d_separated(treatment, outcome, selected | context):
            return None
        for node in sorted(selected):
            if backdoor.is_d_separated(treatment, outcome, (selected - {node}) | context):
                selected.remove(node)
        return tuple(sorted(selected))

    def identify_query(
        self,
        query: IdentifiabilityQuery,
        available_variables: Iterable[str] | None = None,
        *,
        path_diagnostics: bool = True,
        mode: str = "exhaustive",
    ) -> IdentifiabilityResult:
        """Evaluate a domain query while enforcing its declared graph scope."""
        if self.graph_id is not None and query.graph_id != self.graph_id:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.GRAPH_SCOPE_MISMATCH,
                treatment=query.treatment_variable_id,
                outcome=query.outcome_variable_id,
                available_variables=tuple(
                    sorted(
                        self.graph.observed_nodes
                        if available_variables is None
                        else available_variables
                    )
                ),
                detail=f"query graph {query.graph_id!r} does not match {self.graph_id!r}",
            )
        return self.identify(
            query.treatment_variable_id,
            query.outcome_variable_id,
            available_variables,
            path_diagnostics=path_diagnostics,
            mode=mode,
        )

    def identify_conditional(
        self,
        treatment: str,
        outcome: str,
        conditioning_variables: Iterable[str],
        available_variables: Iterable[str] | None = None,
    ) -> IdentifiabilityResult:
        """Identify a conditional total effect with mandatory pre-treatment context.

        The context is part of every blocking set, not an optional adjustment.
        This is a sufficient back-door test, not general conditional do-calculus.
        Positivity is assessed separately in the context-filtered observations.
        """
        self._validate_endpoints(treatment, outcome)
        context = self.graph._validate_nodes(conditioning_variables)
        available = self.graph._validate_nodes(
            self.graph.observed_nodes if available_variables is None else available_variables
        )
        forbidden = {treatment, outcome} | set(self.graph.descendants(treatment))
        if context & forbidden:
            raise ValueError(
                "conditioning context must exclude endpoints and treatment descendants"
            )
        if not available <= self.graph.observed_nodes:
            raise ValueError("available variables must be declared observable")
        required = context | {treatment, outcome}
        if not required <= available:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES,
                missing_variable_sets=(tuple(sorted(required - available)),),
                detail="Mandatory conditioning context and both endpoints must be observable.",
                treatment=treatment,
                outcome=outcome,
                available_variables=tuple(sorted(available)),
                conditioning_variables=tuple(sorted(context)),
            )
        candidates = tuple(
            node
            for node in self.eligible_adjustment_variables(treatment, outcome, available)
            if node not in context
        )
        if len(candidates) > self.max_adjustment_variables:
            return IdentifiabilityResult(
                identified=False,
                reason=IdentifiabilityReason.ENUMERATION_LIMIT_EXCEEDED,
                detail="Conditional exact adjustment search exceeded its configured bound.",
                treatment=treatment,
                outcome=outcome,
                available_variables=tuple(sorted(available)),
                conditioning_variables=tuple(sorted(context)),
            )
        valid = tuple(
            extra
            for extra in _ordered_subsets(candidates)
            if self.is_valid_adjustment_set(treatment, outcome, (*context, *extra))
        )
        return IdentifiabilityResult(
            identified=bool(valid),
            reason=(
                IdentifiabilityReason.IDENTIFIED_WITHOUT_ADJUSTMENT
                if () in valid
                else IdentifiabilityReason.IDENTIFIED_BY_BACKDOOR
                if valid
                else IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
            ),
            valid_adjustment_sets=valid,
            minimal_adjustment_sets=self.minimal_adjustment_sets(valid),
            detail="Back-door blocking checked conditional on mandatory context and adjustment.",
            treatment=treatment,
            outcome=outcome,
            available_variables=tuple(sorted(available)),
            conditioning_variables=tuple(sorted(context)),
        )
