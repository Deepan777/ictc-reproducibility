from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from ictc.contract import (
    CapabilityAdvertisement,
    ContractSelection,
    ContractSynthesizer,
    SynthesisReason,
    SynthesisRequirements,
    SynthesisStatus,
    survives_source_failures,
)
from ictc.domain.enums import PrivacyClass

NOW = datetime(2026, 1, 1, tzinfo=UTC)


def advertisement(
    capability_id: str,
    source: str,
    domain: str,
    variable: str = "U",
    *,
    cost: int = 1,
    bandwidth: int = 10,
    trust: int = 900_000,
) -> CapabilityAdvertisement:
    return CapabilityAdvertisement(
        capability_id=capability_id,
        source_id=source,
        administrative_domain=domain,
        variable_id=variable,
        minimum_sampling_rate_hz=1,
        maximum_sampling_rate_hz=10,
        expected_delay_ms=5,
        clock_accuracy_ms=2,
        trust_millionths=trust,
        privacy_class=PrivacyClass.INTERNAL,
        provenance_mechanisms=("sha256",),
        cost_microunits=cost,
        bandwidth_bps=bandwidth,
        available_from=NOW - timedelta(seconds=1),
        available_until=NOW + timedelta(seconds=10),
    )


def requirements(
    alternatives: tuple[tuple[str, ...], ...] = (("U",),),
    *,
    failure_tolerance: int = 0,
    domains: int = 1,
    bandwidth: int = 100,
) -> SynthesisRequirements:
    return SynthesisRequirements(
        requirement_id="r1",
        variable_alternatives=alternatives,
        required_sampling_rate_hz=5,
        maximum_freshness_ms=20,
        maximum_clock_skew_ms=5,
        minimum_trust_millionths=800_000,
        allowed_privacy_classes=frozenset((PrivacyClass.INTERNAL,)),
        permitted_administrative_domains=frozenset(("A", "B", "C")),
        required_provenance_mechanisms=frozenset(("sha256",)),
        collection_start=NOW,
        deadline=NOW + timedelta(seconds=1),
        maximum_bandwidth_bps=bandwidth,
        minimum_distinct_domains_per_variable=domains,
        failure_tolerance=failure_tolerance,
    )


def test_exact_solver_returns_hand_computed_minimum() -> None:
    items = (
        advertisement("expensive", "s1", "A", cost=5),
        advertisement("cheap", "s2", "B", cost=3),
    )
    result = ContractSynthesizer(items).synthesize_exact(requirements())

    assert result.status is SynthesisStatus.EXACT_OPTIMAL
    assert result.selected_capability_ids == ("cheap",)
    assert result.total_cost_microunits == 3
    assert result.optimality_claimed is True


def test_exact_beats_greedy_but_heuristic_never_claims_optimality() -> None:
    items = (
        advertisement("a", "s1", "A", cost=2),
        advertisement("b", "s2", "B", cost=3),
        advertisement("c", "s3", "A", cost=1),
    )
    req = requirements(failure_tolerance=1, domains=2)
    synthesizer = ContractSynthesizer(items)
    exact = synthesizer.synthesize_exact(req)
    heuristic = synthesizer.synthesize_heuristic(req)

    assert exact.selected_capability_ids == ("b", "c")
    assert exact.total_cost_microunits == 4
    assert heuristic.status is SynthesisStatus.HEURISTIC_FEASIBLE
    assert heuristic.total_cost_microunits == 6
    assert heuristic.optimality_claimed is False
    assert survives_source_failures(exact, items, 1)
    assert survives_source_failures(heuristic, items, 1)


def test_exact_selects_cheapest_identification_alternative() -> None:
    items = (
        advertisement("u", "s1", "A", "U", cost=8),
        advertisement("v", "s2", "B", "V", cost=2),
    )
    result = ContractSynthesizer(items).synthesize_exact(requirements((("U",), ("V",))))

    assert result.selected_alternative == ("V",)
    assert result.selected_capability_ids == ("v",)
    assert result.alternatives == (("U",),)


def test_all_enforced_clauses_are_machine_readable() -> None:
    clauses = requirements().clauses()
    kinds = {clause["kind"] for clause in clauses}
    assert kinds == {
        "SAMPLING_RATE",
        "FRESHNESS",
        "SYNCHRONIZATION",
        "TRUST",
        "PRIVACY",
        "PROVENANCE",
        "DOMAIN",
        "REDUNDANCY",
        "BANDWIDTH",
        "DEADLINE",
    }
    json.dumps(clauses)


@pytest.mark.parametrize(
    ("items", "req", "reason"),
    [
        (
            (advertisement("low", "s1", "A", trust=1),),
            requirements(),
            SynthesisReason.NO_QUALIFYING_CAPABILITY,
        ),
        (
            (advertisement("one", "s1", "A"),),
            requirements(failure_tolerance=1),
            SynthesisReason.INSUFFICIENT_FAILURE_REDUNDANCY,
        ),
        (
            (advertisement("one", "s1", "A"), advertisement("two", "s2", "A")),
            requirements(failure_tolerance=1, domains=2),
            SynthesisReason.INSUFFICIENT_DOMAIN_DIVERSITY,
        ),
        (
            (advertisement("one", "s1", "A", bandwidth=80),),
            requirements(bandwidth=50),
            SynthesisReason.BANDWIDTH_BUDGET_EXCEEDED,
        ),
    ],
)
def test_infeasible_instances_return_specific_reasons(
    items: tuple[CapabilityAdvertisement, ...],
    req: SynthesisRequirements,
    reason: SynthesisReason,
) -> None:
    result = ContractSynthesizer(items).synthesize_exact(req)
    assert result.status is SynthesisStatus.INFEASIBLE
    assert result.reason is reason
    assert result.selected_capability_ids == ()


def test_invalid_advertisements_and_duplicate_source_variable_are_rejected() -> None:
    payload = advertisement("bad", "s", "A").model_dump()
    payload.update({"minimum_sampling_rate_hz": 20, "maximum_sampling_rate_hz": 10})
    with pytest.raises(ValidationError, match="sampling rate"):
        CapabilityAdvertisement.model_validate(payload)
    first = advertisement("a", "s", "A")
    second = advertisement("b", "s", "B")
    with pytest.raises(ValueError, match="only once"):
        ContractSynthesizer((first, second))


def test_failure_audit_rejects_nonrobust_or_infeasible_selection() -> None:
    item = advertisement("a", "s", "A")
    selection = ContractSynthesizer((item,)).synthesize_exact(requirements())
    assert survives_source_failures(selection, (item,), 1) is False
    infeasible = ContractSynthesizer(()).synthesize_exact(requirements())
    assert survives_source_failures(infeasible, (), 0) is False


def test_combined_constraints_can_be_solver_infeasible_after_coarse_diagnostics() -> None:
    items = (
        advertisement("a", "s1", "A", bandwidth=1),
        advertisement("b", "s2", "A", bandwidth=1),
        advertisement("c", "s3", "B", bandwidth=100),
    )
    req = requirements(failure_tolerance=1, domains=2, bandwidth=50)
    synthesizer = ContractSynthesizer(items)
    exact = synthesizer.synthesize_exact(req)
    heuristic = synthesizer.synthesize_heuristic(req)
    assert exact.reason is SynthesisReason.SOLVER_INFEASIBLE
    assert heuristic.status is SynthesisStatus.INFEASIBLE


def test_model_validation_rejects_ambiguous_or_inconsistent_states() -> None:
    capability_payload = advertisement("x", "s", "A").model_dump()
    capability_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"available_from": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"available_until": NOW - timedelta(seconds=2)}, "interval"),
        ({"provenance_mechanisms": ("x", "x")}, "unique"),
    )
    for update, message in capability_cases:
        with pytest.raises(ValidationError, match=message):
            CapabilityAdvertisement.model_validate(capability_payload | update)

    requirement_payload = requirements().model_dump()
    requirement_cases: tuple[tuple[dict[str, object], str], ...] = (
        ({"deadline": datetime(2026, 1, 1)}, "timezone-aware"),
        ({"deadline": NOW}, "follow"),
        ({"variable_alternatives": ((),)}, "nonempty"),
        ({"variable_alternatives": (("U", "U"),)}, "unique"),
        ({"variable_alternatives": (("U",), ("U",))}, "alternatives must be unique"),
    )
    for update, message in requirement_cases:
        with pytest.raises(ValidationError, match=message):
            SynthesisRequirements.model_validate(requirement_payload | update)

    clauses = requirements().clauses()
    with pytest.raises(ValidationError, match="must claim"):
        ContractSelection(
            status=SynthesisStatus.EXACT_OPTIMAL,
            reason=SynthesisReason.SELECTED,
            requirement_id="r",
            clauses=clauses,
        )
    with pytest.raises(ValidationError, match="only an exact"):
        ContractSelection(
            status=SynthesisStatus.HEURISTIC_FEASIBLE,
            reason=SynthesisReason.SELECTED,
            requirement_id="r",
            clauses=clauses,
            optimality_claimed=True,
        )
    with pytest.raises(ValidationError, match="cannot select"):
        ContractSelection(
            status=SynthesisStatus.INFEASIBLE,
            reason=SynthesisReason.SELECTED,
            requirement_id="r",
            clauses=clauses,
            selected_capability_ids=("x",),
        )

    with pytest.raises(ValueError, match="capability IDs"):
        item = advertisement("same", "s1", "A")
        ContractSynthesizer((item, item))
