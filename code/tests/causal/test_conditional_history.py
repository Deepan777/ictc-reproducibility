"""Conditional collider, support, temporal order and dependent-episode regressions."""

from dataclasses import replace

import pytest

from ictc.causal.estimation import (
    AdjustmentObservation,
    EstimationReason,
    estimate_stratified_backdoor,
)
from ictc.causal.graph import CausalDAG
from ictc.causal.history import CompletedHistoricalEpisode, history_available_before
from ictc.causal.identification import BackdoorIdentifier
from ictc.domain.enums import ActionKind
from ictc.ns3.n12 import utility


def collider_identifier() -> BackdoorIdentifier:
    nodes = ("B1", "B2", "A", "I", "Y")
    return BackdoorIdentifier(
        CausalDAG(nodes, (("B1", "A"), ("B1", "I"), ("B2", "I"), ("B2", "Y")), nodes)
    )


def test_conditioning_opens_collider_and_unconditional_certificate_is_rejected() -> None:
    identifier = collider_identifier()
    assert () in identifier.identify("A", "Y").valid_adjustment_sets
    conditional = identifier.identify_conditional("A", "Y", ("I",), ("A", "Y", "I"))
    assert not conditional.identified
    # Within I=1: observational difference is 1, but do(A) leaves Y=B2 unchanged.
    states = [(b1, b2) for b1 in (0, 1) for b2 in (0, 1)]
    selected = [(a, y) for a, y in states if a == y]
    assert selected == [(0, 0), (1, 1)]
    assert sum(b2 for _b1, b2 in states) / 4 == 0.5  # under either intervention
    rows = tuple(
        AdjustmentObservation.from_mapping(
            treatment=ActionKind.REROUTE_TO_PATH if b1 else ActionKind.KEEP_CURRENT_PATH,
            outcome=b2,
            covariates={"I": float(b1 == b2)},
        )
        for b1, b2 in states
        for _ in range(3)
    )
    result = estimate_stratified_backdoor(
        rows,
        identifier.identify("A", "Y"),
        (),
        estimand="conditional effect",
        unit="1",
        context_values=(("I", 1.0),),
    )
    assert result.reason is EstimationReason.CONTEXT_MISMATCH


def test_conditionally_valid_adjustment_fails_conditional_positivity() -> None:
    identifier = collider_identifier()
    identified = identifier.identify_conditional("A", "Y", ("I",))
    assert identified.identified and ("B1",) in identified.valid_adjustment_sets
    rows = tuple(
        AdjustmentObservation.from_mapping(
            treatment=ActionKind.REROUTE_TO_PATH if b1 else ActionKind.KEEP_CURRENT_PATH,
            outcome=b2,
            covariates={"B1": float(b1), "I": float(b1 == b2)},
        )
        for b1 in (0, 1)
        for b2 in (0, 1)
        for _ in range(3)
    )
    result = estimate_stratified_backdoor(
        rows,
        identified,
        ("B1",),
        estimand="conditional effect",
        unit="1",
        context_values=(("I", 1.0),),
    )
    assert result.reason is EstimationReason.INSUFFICIENT_ACTION_SUPPORT


def test_valid_empty_adjustment_and_conditional_weights_are_preserved() -> None:
    graph = CausalDAG(("A", "C", "Y"), (("A", "Y"), ("C", "Y")), ("A", "C", "Y"))
    identifier = BackdoorIdentifier(graph)
    identified = identifier.identify_conditional("A", "Y", ("C",))
    assert () in identified.valid_adjustment_sets
    rows = tuple(
        AdjustmentObservation.from_mapping(
            treatment=action, outcome=100 * c + a, covariates={"C": float(c)}
        )
        for c in (0, 1)
        for a, action in enumerate((ActionKind.KEEP_CURRENT_PATH, ActionKind.REROUTE_TO_PATH))
        for _ in range(3)
    )
    result = estimate_stratified_backdoor(
        rows, identified, (), estimand="effect", unit="1", context_values=(("C", 1.0),)
    )
    assert result.sample_size == 6 and result.estimate is not None
    assert result.estimate.point_estimate == 1
    with pytest.raises(ValueError, match="descendants"):
        identifier.identify_conditional("A", "Y", ("Y",))


def test_cluster_variance_does_not_treat_copied_records_as_new_episodes() -> None:
    identified = BackdoorIdentifier(CausalDAG(("A", "Y"), (("A", "Y"),), ("A", "Y"))).identify(
        "A", "Y"
    )
    rows = tuple(
        AdjustmentObservation.from_mapping(treatment=action, outcome=value, cluster_id=str(g))
        for g in range(4)
        for action, value in (
            (ActionKind.KEEP_CURRENT_PATH, float(g)),
            (ActionKind.REROUTE_TO_PATH, float(2 * g)),
        )
    )
    one = estimate_stratified_backdoor(rows, identified, (), estimand="effect", unit="1")
    two = estimate_stratified_backdoor(rows + rows, identified, (), estimand="effect", unit="1")
    assert one.standard_error == pytest.approx(two.standard_error)
    assert one.estimate is not None and two.estimate is not None
    assert one.estimate.point_estimate == two.estimate.point_estimate
    no_clusters = tuple(replace(row, cluster_id=None) for row in rows)
    assert estimate_stratified_backdoor(
        no_clusters, identified, (), estimand="effect", unit="1"
    ).estimable


def test_historical_outcomes_must_be_complete_received_and_pre_action_adjusted() -> None:
    row = AdjustmentObservation.from_mapping(treatment=ActionKind.KEEP_CURRENT_PATH, outcome=1)
    episode = CompletedHistoricalEpisode(row, 0, 1, 3, 4)
    assert history_available_before((episode,), 5) == (row,)
    for cutoff in (2, 3, 4):
        with pytest.raises(PermissionError):
            history_available_before((episode,), cutoff)
    with pytest.raises(ValueError, match="precede"):
        CompletedHistoricalEpisode(row, 1, 1, 3, 4)


def test_zero_receptions_receive_no_latency_reward() -> None:
    values = {
        "delivery_ratio": "0",
        "throughput_mbps": "0",
        "p95_delay_ms": "0",
        "completion_ratio": "0",
        "queue_drops": "0",
        "rx_packets": "0",
    }
    assert utility(values) == 0
    assert utility(dict(values, rx_packets="1")) == 0.2
