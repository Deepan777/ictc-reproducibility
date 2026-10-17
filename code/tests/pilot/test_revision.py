"""Post-M10 contextual graph and controller-safe context tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from ictc.causal import EstimationReason, IdentifiabilityReason
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.pilot import (
    M10R_GRAPH_EDGES,
    M10R_GRAPH_ID,
    M10R_GRAPH_NODES,
    ContextKey,
    ContextObservation,
    EffectSupportStatus,
    RelativeBin,
    TopologyContext,
    fit_contextual_effect_model,
    fit_revision_effect_model,
    revision_identification,
)

ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "configs/causal_graphs/m10_contextual_observational_selection.yaml"


def test_revision_graph_constants_match_separate_yaml() -> None:
    payload = yaml.safe_load(GRAPH.read_text(encoding="utf-8"))

    assert payload["graph_id"] == M10R_GRAPH_ID
    assert set(payload["variables"]) == set(M10R_GRAPH_NODES)
    assert {tuple(edge) for edge in payload["edges"]} == set(M10R_GRAPH_EDGES)
    assert "preserves" in payload["change_control"]


def test_queue_pressure_remains_the_minimal_adjustment() -> None:
    complete = revision_identification(True)
    missing = revision_identification(False)

    assert complete.identified is True
    assert complete.minimal_adjustment_sets == (("QueuePressureBin",),)
    assert missing.identified is False
    assert missing.reason is IdentifiabilityReason.MISSING_ADJUSTMENT_VARIABLES
    assert missing.missing_variable_sets == (("QueuePressureBin",),)


def test_topology_context_is_pre_simulation_and_deterministic() -> None:
    context = TopologyContext.from_path_configuration(
        current_service_packets_per_tick=5,
        candidate_service_packets_per_tick=8,
        current_propagation_delay_ms=7.0,
        candidate_propagation_delay_ms=5.0,
        current_queue_limit_packets=96,
        candidate_queue_limit_packets=160,
    )

    assert context.service_relation is RelativeBin.HIGHER
    assert context.propagation_delay_relation is RelativeBin.LOWER
    assert context.queue_limit_relation is RelativeBin.HIGHER
    assert context.context_id == "service=HIGHER;delay=LOWER;queue=HIGHER"


@pytest.mark.parametrize(
    "override",
    [
        {"current_service_packets_per_tick": 0},
        {"candidate_propagation_delay_ms": -1.0},
        {"current_queue_limit_packets": -1},
    ],
)
def test_topology_context_rejects_invalid_configuration(override: dict[str, int | float]) -> None:
    values: dict[str, int | float] = {
        "current_service_packets_per_tick": 5,
        "candidate_service_packets_per_tick": 8,
        "current_propagation_delay_ms": 7.0,
        "candidate_propagation_delay_ms": 5.0,
        "current_queue_limit_packets": 96,
        "candidate_queue_limit_packets": 160,
    }
    values.update(override)
    with pytest.raises(ValueError, match="positive|negative"):
        TopologyContext.from_path_configuration(**values)  # type: ignore[arg-type]


def _context_rows(key: ContextKey, keep: int, reroute: int) -> tuple[ContextObservation, ...]:
    return tuple(
        ContextObservation(DatasetSplit.TRAIN, key, ActionKind.KEEP_CURRENT_PATH, 10.0)
        for _index in range(keep)
    ) + tuple(
        ContextObservation(DatasetSplit.TRAIN, key, ActionKind.REROUTE_TO_PATH, 5.0)
        for _index in range(reroute)
    )


def test_contextual_model_enforces_stricter_support_without_global_fallback() -> None:
    supported = ContextKey("registered", "high", 1)
    sparse = ContextKey("registered", "low", 1)
    unseen = ContextKey("unseen", "high", 1)
    model = fit_contextual_effect_model(
        (*_context_rows(supported, 10, 10), *_context_rows(sparse, 10, 2)),
        minimum_per_arm=10,
    )

    supported_effect = model.effect_for(supported)
    sparse_effect = model.effect_for(sparse)
    assert supported_effect is not None and supported_effect.supported is True
    assert sparse_effect is not None and sparse_effect.supported is False
    assert model.admissible_estimate_for(supported) is supported_effect.result
    assert model.admissible_estimate_for(sparse) is None
    assert model.admissible_estimate_for(unseen) is None


def test_contextual_model_rejects_nontraining_and_invalid_support() -> None:
    key = ContextKey("registered", "medium", 0)
    with pytest.raises(PermissionError, match="training"):
        ContextObservation(DatasetSplit.DEVELOPMENT, key, ActionKind.KEEP_CURRENT_PATH, 1.0)
    with pytest.raises(ValueError, match="at least two"):
        fit_contextual_effect_model(_context_rows(key, 2, 2), minimum_per_arm=1)


def test_revision_model_fits_global_traffic_and_local_scopes() -> None:
    high_low_pressure = ContextKey("registered", "high", -1)
    high_high_pressure = ContextKey("registered", "high", 1)
    low_sparse = ContextKey("registered", "low", 0)
    model = fit_revision_effect_model(
        (
            *_context_rows(high_low_pressure, 10, 10),
            *_context_rows(high_high_pressure, 10, 10),
            *_context_rows(low_sparse, 10, 2),
        ),
        minimum_per_arm=10,
    )

    high_effect = model.traffic_effect_for("high")
    low_effect = model.traffic_effect_for("low")
    assert model.population_effect.result.estimable is True
    assert model.population_effect.support_status is EffectSupportStatus.BELOW_MINIMUM_PER_ARM
    assert high_effect is not None and high_effect.supported is True
    assert low_effect is not None
    assert low_effect.result.estimable is True
    assert low_effect.support_status is EffectSupportStatus.BELOW_MINIMUM_PER_ARM
    assert model.admissible_population_estimate() is None
    assert model.admissible_traffic_estimate_for("high") is high_effect.result
    assert model.admissible_traffic_estimate_for("low") is None
    assert model.admissible_traffic_estimate_for("unknown") is None


def test_revision_model_preserves_local_inestimability_without_broad_fallback() -> None:
    unsupported_local = ContextKey("topology-a", "high", 1)
    supported_local = ContextKey("topology-b", "high", 1)
    model = fit_revision_effect_model(
        (
            *_context_rows(unsupported_local, 10, 0),
            *_context_rows(supported_local, 10, 20),
        ),
        minimum_per_arm=10,
    )

    traffic_effect = model.traffic_effect_for("high")
    local_effect = model.contextual_effects.effect_for(unsupported_local)
    assert model.population_effect.supported is True
    assert traffic_effect is not None and traffic_effect.supported is True
    assert local_effect is not None
    assert local_effect.result.reason is EstimationReason.INSUFFICIENT_ACTION_SUPPORT
    assert local_effect.support_status is EffectSupportStatus.ESTIMATOR_INESTIMABLE
    assert model.admissible_context_estimate_for(unsupported_local) is None
    assert model.admissible_context_estimate_for(supported_local) is not None


def test_revision_model_rejects_empty_rows_and_invalid_support() -> None:
    with pytest.raises(ValueError, match="requires observations"):
        fit_revision_effect_model(())
    key = ContextKey("registered", "medium", 0)
    with pytest.raises(ValueError, match="at least two"):
        fit_revision_effect_model(_context_rows(key, 2, 2), minimum_per_arm=1)
