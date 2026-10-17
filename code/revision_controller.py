"""Decision-time controller used only in the isolated COMNET major revision.

This module reads received telemetry, never simulator delivery audit or outcome files.
All policies here make one route decision. They are not sequential RL reproductions.
"""
from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any

from ictc.ns3.n13 import RISK_FEATURE_NAMES, predict_harm_probability, risk_feature_vector
from ictc.ns3.n12 import N12Scenario


def received_rows(raw: Path, decision_ns: int) -> list[dict[str, str]]:
    with (raw / "telemetry.csv").open(encoding="utf-8", newline="") as stream:
        return [r for r in csv.DictReader(stream) if int(r["available_timestamp_ns"]) < decision_ns]


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))]


def context_from_rows(
    rows: list[dict[str, str]], decision_ns: int, *, window: int = 12,
    budget: int = 32, deadline_ms: float = 300.0,
) -> dict[str, Any]:
    # Sequence generation follows the declared source schedule. Only expired
    # deadlines can reveal a missing report; true simulator drop flags are absent.
    all_rows = [r for r in rows if int(r["available_timestamp_ns"]) < decision_ns]
    samples: dict[int, dict[str, dict[str, str]]] = {}
    for row in all_rows:
        samples.setdefault(int(row["sample_timestamp_ns"]), {})[f"route{row['route_id']}"] = row
    complete = sorted(s for s, batch in samples.items() if set(batch) == {"route0", "route1", "route2"})
    if not complete:
        raise ValueError("no complete delivered batch available before actuation")
    retained_samples = complete[-budget:]
    retained = [samples[s][r] for s in retained_samples for r in ("route0", "route1", "route2")]
    # A latest route record is chosen by receipt order, consistent with the original
    # implementation. Generation time is additionally checked for freshness.
    latest_rows: dict[str, dict[str, str]] = {}
    for row in retained:
        route = f"route{row['route_id']}"
        if route not in latest_rows or int(row["available_timestamp_ns"]) > int(latest_rows[route]["available_timestamp_ns"]):
            latest_rows[route] = row
    latest = {
        route: {
            "available_timestamp_ns": int(row["available_timestamp_ns"]),
            "sample_timestamp_ns": int(row["sample_timestamp_ns"]),
            "bottleneck_capacity_bps": int(row["bottleneck_capacity_bps"]),
            "propagation_delay_ms": float(row["propagation_delay_ms"]),
            "queue_packets": int(row["queue_packets"]),
            "link_up": bool(int(row["link_up"])),
        } for route, row in latest_rows.items()
    }
    expired = [s for s in range(400_000_000, decision_ns, 100_000_000) if s + 250_000_000 < decision_ns]
    received = set(samples)
    missing = sum(s not in received for s in expired)
    snapshot = {
        "latest": latest,
        "delivered_batches": len(retained_samples),
        "audited_batches": len(expired),
        "dropped_batches": missing,
        "empirical_loss_rate": missing / len(expired) if expired else 1.0,
        "latest_age_ms": max((decision_ns - row["available_timestamp_ns"]) / 1e6 for row in latest.values()),
        "maximum_sample_age_ms": max((decision_ns - row["sample_timestamp_ns"]) / 1e6 for row in latest.values()),
        "maximum_delivery_delay_ms": max((int(r["available_timestamp_ns"]) - int(r["sample_timestamp_ns"])) / 1e6 for r in retained),
    }
    eligible = [r for r in sorted(latest) if latest[r]["link_up"]] or ["route0"]
    scores = {r: latest[r]["bottleneck_capacity_bps"] / 1e6 - 0.5 * latest[r]["queue_packets"] - 0.08 * latest[r]["propagation_delay_ms"] for r in eligible}
    greedy = max(eligible, key=lambda r: (scores[r], -int(r[-1])))
    ready = (len(retained_samples) >= 20 and snapshot["empirical_loss_rate"] <= 0.30
             and snapshot["maximum_sample_age_ms"] <= deadline_ms
             and snapshot["maximum_delivery_delay_ms"] <= 250.0)
    context = {"snapshot": snapshot, "latest": latest, "eligible": eligible, "visible_scores": scores,
               "greedy_action": greedy, "quality_ready": ready, "samples": samples,
               "recent_samples": retained_samples[-window:]}
    return candidate_context(context, greedy)


def candidate_context(context: dict[str, Any], candidate: str) -> dict[str, Any]:
    value = dict(context)
    sample_ids = context["recent_samples"]
    samples = context["samples"]
    candidate_rows = [samples[s][candidate] for s in sample_ids]
    capacities = [float(r["bottleneck_capacity_bps"]) for r in candidate_rows]
    mean_capacity = statistics.mean(capacities)
    choices, margins = [], []
    for sample in sample_ids:
        batch = samples[sample]
        eligible = [r for r in sorted(batch) if int(batch[r]["link_up"])] or ["route0"]
        scores = {r: float(batch[r]["bottleneck_capacity_bps"]) / 1e6 - 0.5 * float(batch[r]["queue_packets"]) - 0.08 * float(batch[r]["propagation_delay_ms"]) for r in eligible}
        choices.append(max(eligible, key=lambda r: (scores[r], -int(r[-1]))))
        if candidate in scores and "route0" in scores:
            margins.append(scores[candidate] - scores["route0"])
    latest = context["latest"]
    def score(r: str) -> float:
        return latest[r]["bottleneck_capacity_bps"] / 1e6 - 0.5 * latest[r]["queue_packets"] - 0.08 * latest[r]["propagation_delay_ms"]
    value.update({
        "greedy_action": candidate,
        "complete_recent_batches": len(sample_ids),
        "candidate_up_fraction": statistics.mean(float(r["link_up"]) for r in candidate_rows),
        "candidate_consensus_fraction": statistics.mean(float(r == candidate) for r in choices),
        "candidate_queue_p90": quantile([float(r["queue_packets"]) for r in candidate_rows], 0.90),
        "candidate_capacity_cv": statistics.pstdev(capacities) / mean_capacity if mean_capacity else 1e9,
        "candidate_downside_margin_p10": quantile(margins, 0.10) if margins else score(candidate) - score("route0"),
        "latest_score_improvement": score(candidate) - score("route0"),
    })
    return value


def context(raw: Path, scenario: N12Scenario, **options: Any) -> dict[str, Any]:
    return context_from_rows(received_rows(raw, int(scenario.decision_at_seconds * 1e9)), int(scenario.decision_at_seconds * 1e9), **options)


def risk_features(scenario: N12Scenario, ctx: dict[str, Any], candidate: str) -> tuple[float, ...]:
    return risk_feature_vector(scenario, candidate_context(ctx, candidate))


def route_features(scenario: N12Scenario, ctx: dict[str, Any], route: str) -> list[float]:
    """A shared route representation: no OD identity, topology name, or route index."""
    row, keep = ctx["latest"][route], ctx["latest"]["route0"]
    candidate = candidate_context(ctx, route)
    return [float(scenario.traffic == "bulk"), float(scenario.traffic == "realtime"),
            row["bottleneck_capacity_bps"] / 25e6, row["queue_packets"] / 100,
            row["propagation_delay_ms"] / 100, float(row["link_up"]),
            keep["bottleneck_capacity_bps"] / 25e6, keep["queue_packets"] / 100,
            keep["propagation_delay_ms"] / 100, float(keep["link_up"]),
            candidate["candidate_up_fraction"], candidate["candidate_queue_p90"] / 100,
            candidate["candidate_capacity_cv"], candidate["latest_score_improvement"] / 20,
            ctx["snapshot"]["empirical_loss_rate"], ctx["snapshot"]["maximum_sample_age_ms"] / 300,
            ctx["snapshot"]["maximum_delivery_delay_ms"] / 250]


def fluid_proposal(scenario: N12Scenario, ctx: dict[str, Any]) -> str:
    payload_bytes = 2_000_000 if scenario.traffic == "web" else 8_000_000 if scenario.traffic == "bulk" else 5e6 * (scenario.duration_seconds - scenario.decision_at_seconds) / 8
    def cost(route: str) -> float:
        row = ctx["latest"][route]
        return row["propagation_delay_ms"] / 1000 + 8 * (payload_bytes + 1219 * row["queue_packets"]) / max(1, row["bottleneck_capacity_bps"])
    return min(ctx["eligible"], key=lambda r: (cost(r), int(r[-1])))


def learned_proposal(scenario: N12Scenario, ctx: dict[str, Any], model: Any) -> str:
    eligible = ctx["eligible"]
    predictions = model.predict([route_features(scenario, ctx, r) for r in eligible])
    return max(zip(eligible, predictions, strict=True), key=lambda value: (float(value[1]), -int(value[0][-1])))[0]


def admit(scenario: N12Scenario, ctx: dict[str, Any], proposal: str, model: dict[str, Any], threshold: float) -> str:
    if proposal == "route0" or not ctx["quality_ready"]:
        return "route0"
    score = predict_harm_probability(risk_features(scenario, ctx, proposal), model)
    return proposal if score <= threshold else "route0"


def decisions(scenario: N12Scenario, ctx: dict[str, Any], original: dict[str, Any], bundle: Any = None, freeze: Any = None) -> dict[str, str]:
    latest, eligible = ctx["latest"], ctx["eligible"]
    greedy = str(ctx["greedy_action"])
    result = {"Keep": "route0", "Greedy": greedy,
              "MinimumDelay": min(eligible, key=lambda r: (latest[r]["propagation_delay_ms"], int(r[-1]))),
              "MaximumCapacity": max(eligible, key=lambda r: (latest[r]["bottleneck_capacity_bps"], -int(r[-1]))),
              "MinimumQueue": min(eligible, key=lambda r: (latest[r]["queue_packets"], int(r[-1]))),
              "OriginalICTCObservable": admit(scenario, ctx, greedy, original, 0.30),
              "GreedyQuality": greedy if ctx["quality_ready"] else "route0",
              "Fluid": fluid_proposal(scenario, ctx)}
    result["FluidQuality"] = result["Fluid"] if ctx["quality_ready"] else "route0"
    if bundle is None:
        return result
    depths = [freeze["depth"]] if freeze else [2, 4, 8]
    for depth in depths:
        candidate = learned_proposal(scenario, ctx, bundle["proposers"][depth])
        label = "Learned" if freeze else f"LearnedDepth{depth}"
        result[label] = candidate
        result[label + "Quality"] = candidate if ctx["quality_ready"] else "route0"
        model = bundle["risk"][f"learned{depth}"]
        thresholds = [freeze["thresholds"][f"learned{depth}"]] if freeze else [i / 20 for i in range(1, 20)]
        for threshold in thresholds:
            name = label + "ICTC" if freeze else f"{label}ICTC{threshold:.2f}"
            result[name] = admit(scenario, ctx, candidate, model, threshold)
    for count in (3, 6, 12):
        key = f"greedy{count}"
        thresholds = [freeze["thresholds"][key]] if freeze else [i / 20 for i in range(1, 20)]
        for threshold in thresholds:
            name = f"AdaptedICTC{count}" if freeze else f"AdaptedICTC{count}T{threshold:.2f}"
            result[name] = admit(scenario, ctx, greedy, bundle["risk"][key], threshold)
    for key, proposal, name in (("original", greedy, "ThresholdICTC"), ("fluid", result["Fluid"], "FluidICTC")):
        thresholds = [freeze["thresholds"][key]] if freeze else [i / 20 for i in range(1, 20)]
        model = original if key == "original" else bundle["risk"][key]
        for threshold in thresholds:
            result[name if freeze else f"{name}T{threshold:.2f}"] = admit(scenario, ctx, proposal, model, threshold)
    return result
