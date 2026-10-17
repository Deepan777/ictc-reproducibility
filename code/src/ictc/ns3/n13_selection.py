"""Development-only prospective selection for the N13 temporal-risk contract."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, TypedDict, cast

from ictc.ns3.n12 import _csv_rows, digest, read_object, sha256, utility, write_new
from ictc.ns3.n13 import build_risk_context, evaluate_risk_policy, load_matrix


class _Aggregate(TypedDict):
    utilities: list[float]
    harmful: int
    opportunities: int
    retained: int
    disagreements: int


def _write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite N13 selection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _candidates(protocol: dict[str, Any]) -> list[dict[str, Any]]:
    fixed = cast(dict[str, Any], protocol["fixed_thresholds"])
    grid = cast(dict[str, list[float]], protocol["candidate_grid"])
    names = [
        "minimum_candidate_up_fraction",
        "minimum_candidate_consensus_fraction",
        "maximum_candidate_queue_p90",
        "maximum_candidate_capacity_cv",
        "minimum_downside_margin_p10",
    ]
    return [
        {"candidate_id": f"C{index:03d}", **fixed, **dict(zip(names, values, strict=True))}
        for index, values in enumerate(itertools.product(*(grid[name] for name in names)))
    ]


def select_policy(
    journal: Path, matrix_path: Path, protocol_path: Path, output: Path
) -> dict[str, Any]:
    """Persist probe-only decisions first, then score them on development outcomes."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N13 policy selection: {output}")
    matrix, scenarios = load_matrix(matrix_path)
    protocol = read_object(protocol_path)
    if matrix["phase"] != "development":
        raise ValueError("N13 selection may use only the development matrix")
    if sorted({scenario.seed for scenario in scenarios}) != protocol["source_seeds"]:
        raise ValueError("N13 selection seeds differ from the prospective protocol")
    if set(matrix["seeds"]) & set(protocol["forbidden_confirmatory_seeds"]):
        raise ValueError("N13 selection touched a forbidden confirmation seed")
    candidates = _candidates(protocol)
    output.mkdir(parents=True)

    contexts: dict[str, dict[str, Any]] = {}
    decision_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        probe = journal / "runs" / scenario.scenario_id / "probe-route0" / "raw"
        context = build_risk_context(scenario, probe, candidates[0])
        contexts[scenario.scenario_id] = context
        context_digest = digest(
            {
                name: context[name]
                for name in (
                    "snapshot",
                    "visible_scores",
                    "greedy_action",
                    "quality_ready",
                    "candidate_up_fraction",
                    "candidate_consensus_fraction",
                    "candidate_queue_p90",
                    "candidate_capacity_cv",
                    "candidate_downside_margin_p10",
                    "latest_score_improvement",
                )
            }
        )
        for candidate in candidates:
            action, ready, reasons = evaluate_risk_policy(context, candidate)
            decision_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "seed": scenario.seed,
                    "candidate_id": candidate["candidate_id"],
                    "selected_action": action,
                    "greedy_action": context["greedy_action"],
                    "contract_ready": int(ready),
                    "rejection_reasons": "|".join(reasons),
                    "context_digest": context_digest,
                }
            )
    decisions_path = output / "n13_candidate_decisions.csv"
    _write_csv_new(
        decisions_path,
        [
            "scenario_id",
            "seed",
            "candidate_id",
            "selected_action",
            "greedy_action",
            "contract_ready",
            "rejection_reasons",
            "context_digest",
        ],
        decision_rows,
    )

    # Outcome files are opened only after the complete candidate-decision artifact is sealed.
    decisions_by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in decision_rows:
        decisions_by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    aggregates: dict[str, _Aggregate] = {
        str(candidate["candidate_id"]): {
            "utilities": [],
            "harmful": 0,
            "opportunities": 0,
            "retained": 0,
            "disagreements": 0,
        }
        for candidate in candidates
    }
    greedy_utilities: list[float] = []
    greedy_harmful = 0
    for scenario in scenarios:
        action_rows = {
            action: _csv_rows(
                journal / "runs" / scenario.scenario_id / f"action-{action}" / "raw" / "summary.csv"
            )[0]
            for action in ("route0", "route1", "route2")
        }
        utilities = {action: utility(row) for action, row in action_rows.items()}
        keep = utilities["route0"]
        best = max(sorted(utilities), key=utilities.__getitem__)
        greedy_action = str(contexts[scenario.scenario_id]["greedy_action"])
        greedy_utilities.append(utilities[greedy_action])
        greedy_harmful += int(utilities[greedy_action] < keep - 0.02)
        for decision in decisions_by_scenario[scenario.scenario_id]:
            aggregate = aggregates[str(decision["candidate_id"])]
            action = str(decision["selected_action"])
            aggregate["utilities"].append(utilities[action])
            aggregate["harmful"] += int(utilities[action] < keep - 0.02)
            opportunity = utilities[best] > keep + 0.02
            aggregate["opportunities"] += int(opportunity)
            aggregate["retained"] += int(
                opportunity and utilities[action] >= utilities[best] - 0.02
            )
            aggregate["disagreements"] += int(action != greedy_action)

    count = len(scenarios)
    greedy_mean_utility = sum(greedy_utilities) / count
    greedy_harm_rate = greedy_harmful / count
    gate = cast(dict[str, float], protocol["development_gate"])
    metric_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        aggregate = aggregates[candidate_id]
        candidate_utilities = aggregate["utilities"]
        opportunities = aggregate["opportunities"]
        harm_rate = aggregate["harmful"] / count
        mean_utility = sum(candidate_utilities) / count
        retention = aggregate["retained"] / opportunities if opportunities else 0.0
        disagreement = aggregate["disagreements"] / count
        eligible = (
            harm_rate <= float(gate["maximum_harmful_action_rate"])
            and harm_rate <= greedy_harm_rate * float(gate["maximum_harm_rate_ratio_to_greedy"])
            and mean_utility
            >= greedy_mean_utility - float(gate["maximum_mean_utility_loss_vs_greedy"])
            and retention >= float(gate["minimum_opportunity_retention_rate"])
            and disagreement >= float(gate["minimum_greedy_disagreement_rate"])
        )
        metric_rows.append(
            {
                "candidate_id": candidate_id,
                "mean_utility": mean_utility,
                "utility_difference_vs_greedy": mean_utility - greedy_mean_utility,
                "harmful_actions": aggregate["harmful"],
                "harmful_action_rate": harm_rate,
                "harm_rate_difference_vs_greedy": harm_rate - greedy_harm_rate,
                "opportunity_retention_rate": retention,
                "greedy_disagreement_rate": disagreement,
                "development_gate_pass": int(eligible),
            }
        )
    metrics_path = output / "n13_candidate_metrics.csv"
    _write_csv_new(metrics_path, list(metric_rows[0]), metric_rows)
    eligible_rows = [row for row in metric_rows if bool(row["development_gate_pass"])]
    selected_metrics = min(
        eligible_rows,
        key=lambda row: (
            float(row["harmful_action_rate"]),
            -float(row["mean_utility"]),
            -float(row["opportunity_retention_rate"]),
            str(row["candidate_id"]),
        ),
        default=None,
    )
    selected_thresholds = None
    if selected_metrics is not None:
        selected_thresholds = next(
            {key: value for key, value in candidate.items() if key != "candidate_id"}
            for candidate in candidates
            if candidate["candidate_id"] == selected_metrics["candidate_id"]
        )
    payload = {
        "schema_version": "n13-selected-policy-v1",
        "development_gate_pass": selected_metrics is not None,
        "selected_candidate_id": (
            selected_metrics["candidate_id"] if selected_metrics is not None else None
        ),
        "controller_thresholds": selected_thresholds,
        "selected_development_metrics": selected_metrics,
        "greedy_development_metrics": {
            "mean_utility": greedy_mean_utility,
            "harmful_actions": greedy_harmful,
            "harmful_action_rate": greedy_harm_rate,
        },
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(eligible_rows),
        "protocol_sha256": sha256(protocol_path),
        "development_matrix_sha256": sha256(matrix_path),
        "development_manifest_sha256": sha256(journal / "n13_manifest.json"),
        "candidate_decisions_sha256": sha256(decisions_path),
        "candidate_metrics_sha256": sha256(metrics_path),
        "confirmatory_seeds_accessed": False,
    }
    policy_path = output / "n13_selected_policy.json"
    write_new(policy_path, payload)
    write_new(
        output / "selection_manifest.json",
        {
            "schema_version": "n13-selection-manifest-v1",
            "selected_policy_sha256": sha256(policy_path),
            "candidate_decisions_sha256": sha256(decisions_path),
            "candidate_metrics_sha256": sha256(metrics_path),
        },
    )
    return payload
