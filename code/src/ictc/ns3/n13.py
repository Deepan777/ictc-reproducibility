"""Prospective N13 temporal-risk telemetry contract on the N12 Abilene test bed."""

from __future__ import annotations

import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n12 import (
    ACTIONS,
    EVENTS,
    MATRICES,
    METHODS,
    TRAFFIC_PROFILES,
    N12Runner,
    N12Scenario,
    _argbest,
    _latest_snapshot,
    _pairing_payload,
    digest,
    read_object,
)

DEVELOPMENT_SEEDS = tuple(range(86000, 86008))
CONFIRMATORY_SEEDS = tuple(range(87000, 87020))
RISK_FEATURE_NAMES = (
    "traffic_bulk",
    "traffic_realtime",
    "matrix_1",
    "matrix_2",
    "candidate_route_1",
    "candidate_route_2",
    "telemetry_loss_rate",
    "latest_age_scaled",
    "maximum_delivery_delay_scaled",
    "delivered_batches_scaled",
    "candidate_up_fraction",
    "candidate_consensus_fraction",
    "candidate_queue_p90_scaled",
    "candidate_capacity_cv",
    "downside_margin_scaled",
    "latest_improvement_scaled",
    "candidate_capacity_scaled",
    "candidate_queue_scaled",
    "candidate_delay_scaled",
    "keep_capacity_scaled",
    "keep_queue_scaled",
    "keep_delay_scaled",
    "keep_latest_up",
)


def load_matrix(path: Path) -> tuple[dict[str, Any], tuple[N12Scenario, ...]]:
    """Load only the prospectively declared N13 development or confirmation matrix."""
    raw = read_object(path)
    if raw.get("schema_version") != "n13-temporal-risk-matrix-v1":
        raise ValueError("unexpected N13 matrix schema")
    phase = raw.get("phase")
    expected_seeds = DEVELOPMENT_SEEDS if phase == "development" else CONFIRMATORY_SEEDS
    if phase not in {"development", "confirmatory"}:
        raise ValueError("N13 phase must be development or confirmatory")
    if raw.get("seeds") != list(expected_seeds):
        raise ValueError("N13 seeds differ from the prospectively declared set")
    if raw.get("traffic_profiles") != list(TRAFFIC_PROFILES):
        raise ValueError("N13 traffic profiles are incomplete")
    if raw.get("events") != list(EVENTS) or raw.get("od_matrices") != list(MATRICES):
        raise ValueError("N13 event or OD matrix declaration is incomplete")
    if raw.get("actions") != list(ACTIONS):
        raise ValueError("N13 action declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N13 controls are missing")
    duration = float(controls["duration_seconds"])
    decision = float(controls["decision_at_seconds"])
    event_at = float(controls["event_at_seconds"])
    if not 0.5 < event_at + 0.5 < decision < duration - 1.0:
        raise ValueError("N13 timing boundary is invalid")
    scenarios = tuple(
        N12Scenario(
            scenario_id=f"n13-{phase}-{traffic}-{event}-od{matrix}-seed-{seed}",
            phase=phase,
            traffic=traffic,
            event=event,
            matrix=matrix,
            seed=seed,
            run=1,
            duration_seconds=duration,
            decision_at_seconds=decision,
            event_at_seconds=event_at,
        )
        for traffic in TRAFFIC_PROFILES
        for event in EVENTS
        for matrix in MATRICES
        for seed in expected_seeds
    )
    if len(scenarios) != raw.get("expected_scenario_groups"):
        raise ValueError("N13 matrix scenario count differs from its declaration")
    if len(scenarios) * 4 != raw.get("expected_native_runs"):
        raise ValueError("N13 native run count differs from its declaration")
    if len(scenarios) * len(METHODS) != raw.get("expected_controller_decisions"):
        raise ValueError("N13 decision count differs from its declaration")
    return raw, scenarios


def _nearest_rank(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def build_risk_context(
    scenario: N12Scenario, probe_raw: Path, thresholds: dict[str, Any]
) -> dict[str, Any]:
    """Derive temporal risk features from delivered pre-decision telemetry only."""
    snapshot = _latest_snapshot(probe_raw, scenario)
    payload = _pairing_payload(probe_raw, scenario)
    telemetry = cast(list[dict[str, str]], payload["telemetry"])
    latest = cast(dict[str, dict[str, Any]], snapshot["latest"])
    capacities = {route: float(row["bottleneck_capacity_bps"]) for route, row in latest.items()}
    delays = {route: float(row["propagation_delay_ms"]) for route, row in latest.items()}
    queues = {route: float(row["queue_packets"]) for route, row in latest.items()}
    up = {route: bool(row["link_up"]) for route, row in latest.items()}
    eligible = [route for route in sorted(latest) if up[route]] or ["route0"]
    visible_scores = {
        route: capacities[route] / 1e6
        - float(thresholds["queue_penalty_per_packet"]) * queues[route]
        - float(thresholds["delay_penalty_per_ms"]) * delays[route]
        for route in eligible
    }
    greedy = _argbest(visible_scores)

    by_sample: dict[int, dict[str, dict[str, str]]] = {}
    for row in telemetry:
        sample = int(row["sample_timestamp_ns"])
        by_sample.setdefault(sample, {})[f"route{int(row['route_id'])}"] = row
    complete_samples = [sample for sample in sorted(by_sample) if len(by_sample[sample]) == 3]
    lookback = int(thresholds["lookback_batches"])
    recent_samples = complete_samples[-lookback:]
    if not recent_samples:
        raise ValueError("N13 has no complete delivered telemetry batch")

    candidate_rows = [by_sample[sample][greedy] for sample in recent_samples]
    candidate_up_fraction = statistics.mean(float(row["link_up"]) for row in candidate_rows)
    candidate_queue_p90 = _nearest_rank(
        [float(row["queue_packets"]) for row in candidate_rows], 0.90
    )
    candidate_capacities = [float(row["bottleneck_capacity_bps"]) for row in candidate_rows]
    capacity_mean = statistics.mean(candidate_capacities)
    candidate_capacity_cv = (
        statistics.pstdev(candidate_capacities) / capacity_mean if capacity_mean > 0 else 1e9
    )

    historical_choices: list[str] = []
    downside_margins: list[float] = []
    for sample in recent_samples:
        rows = by_sample[sample]
        sample_eligible = [route for route in sorted(rows) if bool(int(rows[route]["link_up"]))]
        if not sample_eligible:
            sample_eligible = ["route0"]
        sample_scores = {
            route: float(rows[route]["bottleneck_capacity_bps"]) / 1e6
            - float(thresholds["queue_penalty_per_packet"]) * float(rows[route]["queue_packets"])
            - float(thresholds["delay_penalty_per_ms"]) * float(rows[route]["propagation_delay_ms"])
            for route in sample_eligible
        }
        historical_choices.append(_argbest(sample_scores))
        if greedy in sample_scores and "route0" in sample_scores:
            downside_margins.append(sample_scores[greedy] - sample_scores["route0"])
    candidate_consensus_fraction = statistics.mean(
        float(choice == greedy) for choice in historical_choices
    )
    latest_improvement = (
        visible_scores[greedy] - visible_scores["route0"] if "route0" in visible_scores else 1e9
    )
    downside_margin_p10 = (
        _nearest_rank(downside_margins, 0.10) if downside_margins else latest_improvement
    )
    quality_ready = (
        int(snapshot["delivered_batches"]) >= int(thresholds["minimum_delivered_batches"])
        and float(snapshot["empirical_loss_rate"]) <= float(thresholds["maximum_loss_rate"])
        and float(snapshot["latest_age_ms"]) <= float(thresholds["maximum_latest_age_ms"])
        and float(snapshot["maximum_delivery_delay_ms"])
        <= float(thresholds["maximum_delivery_delay_ms"])
    )
    return {
        "snapshot": snapshot,
        "latest": latest,
        "capacities": capacities,
        "delays": delays,
        "queues": queues,
        "up": up,
        "eligible": eligible,
        "visible_scores": visible_scores,
        "greedy_action": greedy,
        "quality_ready": quality_ready,
        "complete_recent_batches": len(recent_samples),
        "candidate_up_fraction": candidate_up_fraction,
        "candidate_consensus_fraction": candidate_consensus_fraction,
        "candidate_queue_p90": candidate_queue_p90,
        "candidate_capacity_cv": candidate_capacity_cv,
        "candidate_downside_margin_p10": downside_margin_p10,
        "latest_score_improvement": latest_improvement,
    }


def evaluate_risk_policy(
    context: dict[str, Any], thresholds: dict[str, Any]
) -> tuple[str, bool, tuple[str, ...]]:
    """Apply a frozen temporal-risk admission policy to a controller-safe context."""
    candidate = str(context["greedy_action"])
    reasons: list[str] = []
    if candidate == "route0":
        reasons.append("NO_ROUTE_CHANGE_PROPOSED")
    if not bool(context["quality_ready"]):
        reasons.append("GLOBAL_TELEMETRY_QUALITY")
    if float(context["candidate_up_fraction"]) < float(thresholds["minimum_candidate_up_fraction"]):
        reasons.append("ROUTE_HEALTH_PERSISTENCE")
    if float(context["candidate_consensus_fraction"]) < float(
        thresholds["minimum_candidate_consensus_fraction"]
    ):
        reasons.append("CHOICE_INSTABILITY")
    if float(context["candidate_queue_p90"]) > float(thresholds["maximum_candidate_queue_p90"]):
        reasons.append("QUEUE_TAIL_RISK")
    if float(context["candidate_capacity_cv"]) > float(thresholds["maximum_candidate_capacity_cv"]):
        reasons.append("CAPACITY_INSTABILITY")
    if float(context["candidate_downside_margin_p10"]) < float(
        thresholds["minimum_downside_margin_p10"]
    ):
        reasons.append("DOWNSIDE_MARGIN")
    if float(context["latest_score_improvement"]) < float(
        thresholds["minimum_latest_score_improvement"]
    ):
        reasons.append("LATEST_IMPROVEMENT")
    ready = not reasons
    return (candidate if ready else "route0"), ready, tuple(reasons)


def risk_feature_vector(scenario: N12Scenario, context: dict[str, Any]) -> tuple[float, ...]:
    """Return the frozen ordered feature vector available before actuation."""

    def clip(value: object, bound: float) -> float:
        return max(-bound, min(bound, float(cast(Any, value))))

    candidate = str(context["greedy_action"])
    latest = cast(dict[str, dict[str, Any]], context["latest"])
    snapshot = cast(dict[str, Any], context["snapshot"])
    candidate_row = latest[candidate]
    keep = latest["route0"]
    values = (
        float(scenario.traffic == "bulk"),
        float(scenario.traffic == "realtime"),
        float(scenario.matrix == 1),
        float(scenario.matrix == 2),
        float(candidate == "route1"),
        float(candidate == "route2"),
        float(snapshot["empirical_loss_rate"]),
        float(snapshot["latest_age_ms"]) / 300.0,
        float(snapshot["maximum_delivery_delay_ms"]) / 250.0,
        float(snapshot["delivered_batches"]) / 30.0,
        float(context["candidate_up_fraction"]),
        float(context["candidate_consensus_fraction"]),
        float(context["candidate_queue_p90"]) / 100.0,
        float(context["candidate_capacity_cv"]),
        clip(context["candidate_downside_margin_p10"], 20.0) / 20.0,
        clip(context["latest_score_improvement"], 20.0) / 20.0,
        float(candidate_row["bottleneck_capacity_bps"]) / 25_000_000.0,
        float(candidate_row["queue_packets"]) / 100.0,
        float(candidate_row["propagation_delay_ms"]) / 100.0,
        float(keep["bottleneck_capacity_bps"]) / 25_000_000.0,
        float(keep["queue_packets"]) / 100.0,
        float(keep["propagation_delay_ms"]) / 100.0,
        float(bool(keep["link_up"])),
    )
    if len(values) != len(RISK_FEATURE_NAMES):
        raise AssertionError("N13 risk feature declaration differs from its vector")
    return values


def predict_harm_probability(features: tuple[float, ...], model: dict[str, Any]) -> float:
    """Score a frozen standardized logistic model without fitting-time dependencies."""
    if model.get("feature_names") != list(RISK_FEATURE_NAMES):
        raise ValueError("N13 risk model feature order differs from the frozen declaration")
    means = cast(list[float], model["means"])
    scales = cast(list[float], model["scales"])
    coefficients = cast(list[float], model["coefficients"])
    if not len(features) == len(means) == len(scales) == len(coefficients):
        raise ValueError("N13 risk model coefficient dimensions are inconsistent")
    logit = float(model["intercept"]) + sum(
        coefficient * ((value - mean) / scale if scale > 0 else 0.0)
        for value, mean, scale, coefficient in zip(
            features, means, scales, coefficients, strict=True
        )
    )
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def build_decision(
    scenario: N12Scenario, probe_raw: Path, thresholds: dict[str, Any]
) -> dict[str, Any]:
    context = build_risk_context(scenario, probe_raw, thresholds)
    predicted_harm_probability: float | None = None
    model = thresholds.get("risk_model")
    if isinstance(model, dict):
        features = risk_feature_vector(scenario, context)
        predicted_harm_probability = predict_harm_probability(features, model)
        model_reasons: list[str] = []
        if str(context["greedy_action"]) == "route0":
            model_reasons.append("NO_ROUTE_CHANGE_PROPOSED")
        if not bool(context["quality_ready"]):
            model_reasons.append("GLOBAL_TELEMETRY_QUALITY")
        if predicted_harm_probability > float(model["maximum_predicted_harm_probability"]):
            model_reasons.append("PREDICTED_HARM_RISK")
        contract_ready = not model_reasons
        action = str(context["greedy_action"]) if contract_ready else "route0"
        reasons = tuple(model_reasons)
    else:
        action, contract_ready, reasons = evaluate_risk_policy(context, thresholds)
    capacities = cast(dict[str, float], context["capacities"])
    delays = cast(dict[str, float], context["delays"])
    queues = cast(dict[str, float], context["queues"])
    eligible = cast(list[str], context["eligible"])
    decisions = {
        "S1_ICTC_CONTRACT": action,
        "B0_KEEP": "route0",
        "B1_SHORTEST_DELAY": _argbest({route: delays[route] for route in eligible}, minimize=True),
        "B2_MAX_CAPACITY": _argbest({route: capacities[route] for route in eligible}),
        "B3_MIN_QUEUE": _argbest({route: queues[route] for route in eligible}, minimize=True),
        "B4_GREEDY_TELEMETRY": str(context["greedy_action"]),
        "B5_ECMP_HASH": f"route{scenario.seed % 3}",
    }
    risk_features = {
        name: context[name]
        for name in (
            "complete_recent_batches",
            "candidate_up_fraction",
            "candidate_consensus_fraction",
            "candidate_queue_p90",
            "candidate_capacity_cv",
            "candidate_downside_margin_p10",
            "latest_score_improvement",
        )
    }
    return {
        "schema_version": "n13-controller-decisions-v1",
        "scenario": asdict(scenario),
        "created_before_outcomes": True,
        "controller_information_boundary": (
            "Only delivered pre-decision telemetry and frozen route metadata; no hidden event, "
            "post-decision outcome, paired action summary, or oracle label."
        ),
        "thresholds": thresholds,
        "snapshot": context["snapshot"],
        "visible_scores": context["visible_scores"],
        "greedy_action": context["greedy_action"],
        "risk_features": risk_features,
        "risk_context_digest": digest(risk_features),
        "predicted_harm_probability": predicted_harm_probability,
        "contract_ready": contract_ready,
        "rejection_reasons": list(reasons),
        "decisions": decisions,
    }


class N13Runner(N12Runner):
    """N13 journal using the unchanged native Abilene experiment engine."""

    study_id = "N13"
    journal_prefix = "n13"
    journal_schema = "n13-immutable-journal-v1"
    native_run_schema = "n13-native-run-v1"
    resource_schema = "n13-resource-metrics-v1"
    paired_score_schema = "n13-paired-score-v1"
    group_schema = "n13-group-manifest-v1"
    matrix_loader = staticmethod(load_matrix)
    decision_builder = staticmethod(build_decision)
    selected_policy_relative = Path("configs/experiments/n13_selected_policy.json")
