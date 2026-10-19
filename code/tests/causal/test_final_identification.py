"""Regressions for nonfatal diagnostics and the optional single-set mode."""

from dataclasses import asdict

import pytest

from ictc.causal.graph import CausalDAG, GraphComplexityError
from ictc.causal.identification import BackdoorIdentifier


def test_default_identification_keeps_previous_fields_and_paths() -> None:
    graph = CausalDAG(("A", "Y", "U"), (("U", "A"), ("U", "Y"), ("A", "Y")))
    result = BackdoorIdentifier(graph).identify("A", "Y")
    assert result.open_backdoor_paths == (("A", "U", "Y"),)
    assert result.valid_adjustment_sets == (("U",),)
    assert not result.diagnostic_truncated
    skipped = BackdoorIdentifier(graph).identify("A", "Y", path_diagnostics=False)
    baseline, other = asdict(result), asdict(skipped)
    baseline.pop("open_backdoor_paths")
    other.pop("open_backdoor_paths")
    assert baseline == other


def test_diagnostic_limit_is_nonfatal_and_skip_never_enumerates(monkeypatch) -> None:
    graph = CausalDAG(("A", "Y", "U"), (("U", "A"), ("U", "Y")))

    def fail(*args, **kwargs):
        raise GraphComplexityError("test limit")

    monkeypatch.setattr(CausalDAG, "open_backdoor_paths", fail)
    result = BackdoorIdentifier(graph).identify("A", "Y")
    assert result.identified and result.diagnostic_truncated
    assert result.open_backdoor_paths == ()
    skipped = BackdoorIdentifier(graph).identify("A", "Y", path_diagnostics=False)
    assert skipped.identified and not skipped.diagnostic_truncated


def test_real_dense_graph_exceeds_path_cap_without_aborting() -> None:
    nodes = tuple(f"V{i}" for i in range(10))
    graph = CausalDAG(
        nodes, tuple((nodes[i], nodes[j]) for i in range(10) for j in range(i + 1, 10))
    )
    result = BackdoorIdentifier(graph).identify("V8", "V9")
    assert result.identified and result.diagnostic_truncated
    assert result.open_backdoor_paths == ()


def test_single_minimal_set_and_unobserved_confounding() -> None:
    graph = CausalDAG(("A", "Y", "U", "V"), (("U", "V"), ("V", "A"), ("U", "Y")))
    identifier = BackdoorIdentifier(graph, max_adjustment_variables=0)
    result = identifier.identify("A", "Y", mode="single", path_diagnostics=False)
    assert result.identified
    assert len(result.valid_adjustment_sets) == 1
    selected = result.minimal_adjustment_sets[0]
    assert identifier.is_valid_adjustment_set("A", "Y", selected)
    assert selected in (("U",), ("V",))
    assert identifier.find_minimal_adjustment_set("A", "Y", ("A", "Y")) is None


def test_single_empty_adjustment_is_not_failure() -> None:
    result = BackdoorIdentifier(CausalDAG(("A", "Y"), (("A", "Y"),))).identify(
        "A", "Y", mode="single"
    )
    assert result.identified and result.valid_adjustment_sets == ((),)


def test_single_mandatory_collider_context_matches_exhaustive() -> None:
    nodes = ("B1", "B2", "A", "I", "Y")
    identifier = BackdoorIdentifier(
        CausalDAG(nodes, (("B1", "A"), ("B1", "I"), ("B2", "I"), ("B2", "Y")))
    )
    restricted = identifier.identify(
        "A", "Y", ("A", "Y", "I"), mode="single", conditioning_variables=("I",)
    )
    assert not restricted.identified
    full = identifier.identify("A", "Y", mode="single", conditioning_variables=("I",))
    exhaustive = identifier.identify_conditional("A", "Y", ("I",))
    assert full.identified and full.minimal_adjustment_sets[0] in exhaustive.minimal_adjustment_sets
    assert full.conditioning_variables == ("I",)


def test_single_rejects_invalid_context_and_mode() -> None:
    identifier = BackdoorIdentifier(CausalDAG(("A", "Y"), (("A", "Y"),)))
    with pytest.raises(ValueError, match="descendants"):
        identifier.find_minimal_adjustment_set("A", "Y", ("A", "Y"), ("Y",))
    with pytest.raises(ValueError, match="mode"):
        identifier.identify("A", "Y", mode="unknown")


def test_contract_single_mode_supplies_only_one_alternative() -> None:
    from ictc.contract import QueryContractPlanner
    from ictc.contract.planning_benchmark import (
        capability_registry,
        planning_policy,
        routing_graph,
        routing_query,
    )

    result = QueryContractPlanner(capability_registry()).plan(
        plan_id="single",
        proposal_id="proposal",
        contract_id="contract",
        query=routing_query(),
        graph=routing_graph(),
        policy=planning_policy(),
        identification_mode="single",
        path_diagnostics=False,
    )
    assert result.contract is not None and result.requirements is not None
    assert len(result.minimal_adjustment_sets) == 1
    assert result.requirements.variable_alternatives == result.minimal_adjustment_sets


def test_legacy_result_fields_are_byte_identical_below_path_limit() -> None:
    import json

    # Synthetic expected values captured from the public v4 implementation.
    # No Git history, separately held records or local installation is required.
    graph = CausalDAG(("A", "Y", "U", "V"), (("U", "A"), ("U", "V"), ("V", "Y"), ("A", "Y")))
    cases = (
        (graph.nodes, True, (("U",), ("V",), ("U", "V")), (("U",), ("V",)), ()),
        (frozenset(("A", "Y")), False, (), (), (("U",), ("V",))),
        (frozenset(("A", "Y", "U")), True, (("U",),), (("U",),), ()),
    )
    for available, identified, valid, minimal, missing in cases:
        old = {
            "identified": identified,
            "reason": "IDENTIFIED_BY_BACKDOOR" if identified else "MISSING_ADJUSTMENT_VARIABLES",
            "treatment": "A",
            "outcome": "Y",
            "available_variables": tuple(sorted(available)),
            "valid_adjustment_sets": valid,
            "minimal_adjustment_sets": minimal,
            "open_backdoor_paths": (("A", "U", "V", "Y"),),
            "missing_variable_sets": missing,
            "conditioning_variables": (),
            "detail": (
                "At least one available set satisfies the backdoor criterion."
                if identified
                else "No set of currently available observed variables blocks every backdoor path; "
                "missing-variable sets list exact graph variables, never proxies."
            ),
        }
        new = asdict(BackdoorIdentifier(graph).identify("A", "Y", available))
        assert new.pop("diagnostic_truncated") is False
        assert json.dumps(old, sort_keys=True).encode() == json.dumps(new, sort_keys=True).encode()
