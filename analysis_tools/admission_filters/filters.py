"""Outcome-free margin, persistence and ensemble admission comparators."""
from __future__ import annotations

from typing import Any
import numpy as np

from revision_controller import context_from_rows, fluid_proposal, route_features

ROUTES = ("route0", "route1", "route2")
PROPOSERS = ("Greedy", "Learned", "Fluid")


def service_time(scenario: Any, ctx: dict, route: str) -> float:
    payload = (2_000_000 if scenario.traffic == "web" else 8_000_000
               if scenario.traffic == "bulk" else 5e6 * (scenario.duration_seconds - scenario.decision_at_seconds) / 8)
    row = ctx["latest"][route]
    return row["propagation_delay_ms"] / 1000 + 8 * (payload + 1219 * row["queue_packets"]) / max(1, row["bottleneck_capacity_bps"])


def greedy_score(row: dict) -> float:
    return row["bottleneck_capacity_bps"] / 1e6 - 0.5 * row["queue_packets"] - 0.08 * row["propagation_delay_ms"]


def evidence(scenario: Any, ctx: dict, rows: list[dict], model: Any) -> dict:
    """Evaluate all proposers on current information and historical receipt prefixes.

    Historical complete batches are ordered by generation time, as in the existing
    context. Each history prefix ends one nanosecond after that batch's last
    receipt, and never at the final decision time. No outcome argument exists.
    """
    complete = sorted(s for s, batch in ctx["samples"].items() if set(batch) == set(ROUTES))
    histories = []
    for sample in complete[-12:]:
        cutoff = max(int(r["available_timestamp_ns"]) for r in ctx["samples"][sample].values()) + 1
        histories.append(context_from_rows(rows, cutoff))
    contexts = [ctx, *histories]
    features = np.asarray([route_features(scenario, c, route) for c in contexts for route in ROUTES])
    trees = np.asarray([tree.predict(features) for tree in model.estimators_]).reshape(len(model.estimators_), len(contexts), 3)
    means = trees.mean(axis=0)
    result = {}
    for proposer in PROPOSERS:
        choices, margins = [], []
        for i, c in enumerate(contexts):
            if proposer == "Greedy":
                choice = c["greedy_action"]
                margin = greedy_score(c["latest"][choice]) - greedy_score(c["latest"]["route0"])
            elif proposer == "Fluid":
                choice = fluid_proposal(scenario, c)
                t0 = service_time(scenario, c, "route0")
                margin = (t0 - service_time(scenario, c, choice)) / t0
            else:
                choice = max(c["eligible"], key=lambda r: (float(means[i, int(r[-1])]), -int(r[-1])))
                margin = float(means[i, int(choice[-1])] - means[i, 0])
            choices.append(choice)
            margins.append(float(margin))
        result[proposer] = {"proposal": choices[0], "margin": margins[0],
                            "history_choices": choices[1:], "history_margins": margins[1:]}
    g = result["Learned"]["proposal"]
    result["Learned"]["tree_differences"] = (trees[:, 0, int(g[-1])] - trees[:, 0, 0]).tolist()
    return {"quality_ready": bool(ctx["quality_ready"]), "proposers": result}


def filter_action(value: dict, proposer: str, family: str, params: dict, *, quality: bool) -> str:
    row = value["proposers"][proposer]
    proposal = row["proposal"]
    if proposal == "route0" or (quality and not value["quality_ready"]):
        return "route0"
    if family == "Margin":
        admit = row["margin"] >= params["tau"]
    elif family == "Persistence":
        k = params["k"]
        admit = len(row["history_choices"]) >= k and all(
            action == proposal and margin >= params["m"]
            for action, margin in zip(row["history_choices"][-k:], row["history_margins"][-k:], strict=True))
    elif family == "Ensemble":
        admit = float(np.quantile(row["tree_differences"], params["q"])) >= params["tau"]
    else:
        raise ValueError(family)
    return proposal if admit else "route0"


def grids(values: list[dict], proposer: str) -> dict:
    margins = [v["proposers"][proposer]["margin"] for v in values if v["proposers"][proposer]["proposal"] != "route0"]
    if not margins:
        raise ValueError(f"No validation nonincumbent proposals for {proposer}; percentile grid is undefined")
    percentiles = np.quantile(margins, np.arange(1, 10) / 10).tolist()
    tau = sorted(set([0.0, *percentiles]))
    median = float(np.median(margins))
    output = {"Margin": [{"tau": t} for t in tau],
              "Persistence": [{"k": k, "m": m} for k in (2, 3, 5, 8, 12) for m in sorted(set([0.0, median]))],
              "margin_percentiles_10_to_90": percentiles, "median_margin": median,
              "validation_nonincumbent_proposals": len(margins)}
    if proposer == "Learned":
        output["Ensemble"] = []
        output["ensemble_percentiles"] = {}
        for q in (0.10, 0.25, 0.50):
            qs = [float(np.quantile(v["proposers"][proposer]["tree_differences"], q))
                  for v in values if v["proposers"][proposer]["proposal"] != "route0"]
            ps = np.quantile(qs, np.arange(1, 10) / 10).tolist()
            output["ensemble_percentiles"][str(q)] = ps
            output["Ensemble"].extend({"q": q, "tau": t} for t in sorted(set([0.0, *ps])))
    return output


def suffix(params: dict) -> str:
    return ",".join(f"{key}={value:.17g}" if isinstance(value, float) else f"{key}={value}" for key, value in params.items())
