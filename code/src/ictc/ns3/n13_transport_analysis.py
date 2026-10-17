"""Prospectively frozen per-topology analysis for the N13 external transport study."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import binomtest  # type: ignore[import-untyped]

from ictc.ns3.n12 import read_object, sha256, write_new
from ictc.ns3.n13_analysis import _bootstrap_interval, _write_csv_new
from ictc.ns3.n13_transport import load_matrix


def _load_frames(
    journal: Path, matrix_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, float]:
    matrix, scenarios = load_matrix(matrix_path)
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    native_wall_ns = 0
    for scenario in scenarios:
        group_path = journal / "groups" / scenario.scenario_id / "group_manifest.json"
        score_path = journal / "scores" / scenario.scenario_id / "score.json"
        decision_path = journal / "decisions" / scenario.scenario_id / "decisions.json"
        group = read_object(group_path)
        if group.get("score_sha256") != sha256(score_path) or group.get(
            "decision_sha256"
        ) != sha256(decision_path):
            raise ValueError(f"N13 transport group hash mismatch: {scenario.scenario_id}")
        score, decision = read_object(score_path), read_object(decision_path)
        snapshot = cast(dict[str, Any], decision["snapshot"])
        decisions.append(
            {
                "scenario_id": scenario.scenario_id,
                "topology": scenario.topology,
                "seed": scenario.seed,
                "traffic": scenario.traffic,
                "event": scenario.event,
                "matrix": scenario.matrix,
                "contract_ready": int(bool(decision["contract_ready"])),
                "predicted_harm_probability": float(decision["predicted_harm_probability"]),
                "ictc_action": decision["decisions"]["S1_ICTC_CONTRACT"],
                "greedy_action": decision["decisions"]["B4_GREEDY_TELEMETRY"],
                "ictc_greedy_disagreement": int(
                    decision["decisions"]["S1_ICTC_CONTRACT"]
                    != decision["decisions"]["B4_GREEDY_TELEMETRY"]
                ),
                "delivered_batches": int(snapshot["delivered_batches"]),
                "audited_batches": int(snapshot["audited_batches"]),
                "empirical_loss_rate": float(snapshot["empirical_loss_rate"]),
                "latest_age_ms": float(snapshot["latest_age_ms"]),
                "maximum_delivery_delay_ms": float(snapshot["maximum_delivery_delay_ms"]),
                "rejection_reasons": "|".join(decision["rejection_reasons"]),
            }
        )
        for action in ("route0", "route1", "route2"):
            manifest = read_object(
                journal / "runs" / scenario.scenario_id / f"action-{action}" / "run_manifest.json"
            )
            native_wall_ns += int(manifest["elapsed_ns"])
        probe_manifest = read_object(
            journal / "runs" / scenario.scenario_id / "probe-route0" / "run_manifest.json"
        )
        native_wall_ns += int(probe_manifest["elapsed_ns"])
        for row in cast(list[dict[str, Any]], score["rows"]):
            metrics = cast(dict[str, str], row["metrics"])
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "topology": scenario.topology,
                    "seed": scenario.seed,
                    "traffic": scenario.traffic,
                    "event": scenario.event,
                    "matrix": scenario.matrix,
                    "method": row["method"],
                    "selected_action": row["selected_action"],
                    "utility": float(row["utility"]),
                    "harmful": int(bool(row["harmful"])),
                    "beneficial": int(bool(row["beneficial"])),
                    "opportunity_present": int(bool(row["opportunity_present"])),
                    "opportunity_retained": int(bool(row["opportunity_retained"])),
                    **{
                        name: float(metrics[name])
                        for name in (
                            "delivery_ratio",
                            "mean_delay_ms",
                            "p95_delay_ms",
                            "p99_delay_ms",
                            "jitter_ms",
                            "throughput_mbps",
                            "completion_ratio",
                            "queue_drops",
                            "observed_transfer_seconds",
                        )
                    },
                }
            )
    frame, decision_frame = pd.DataFrame.from_records(records), pd.DataFrame.from_records(decisions)
    if len(frame) != len(scenarios) * 8 or len(decision_frame) != len(scenarios):
        raise ValueError("N13 transport analysis input is incomplete")
    return matrix, frame, decision_frame, native_wall_ns / 1e9


def _paired_effect(values: np.ndarray) -> float | None:
    standard = float(values.std(ddof=1))
    return float(values.mean() / standard) if standard > 0 else None


def analyze(journal: Path, matrix_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N13 transport analysis directory: {output}")
    matrix, frame, decision_frame, native_wall_seconds = _load_frames(journal, matrix_path)
    analysis = cast(dict[str, Any], matrix["analysis"])
    resamples, bootstrap_seed = (
        int(analysis["bootstrap_resamples"]),
        int(analysis["bootstrap_seed"]),
    )
    confidence = float(analysis["confidence_level"])
    metric_names = [
        "utility",
        "harmful",
        "beneficial",
        "delivery_ratio",
        "p95_delay_ms",
        "throughput_mbps",
        "completion_ratio",
        "queue_drops",
    ]
    seed_frame = frame.groupby(["topology", "seed", "method"], as_index=False)[metric_names].mean()
    summaries: list[dict[str, Any]] = []
    topology_results: dict[str, Any] = {}
    for topology_index, (topology, topology_seed) in enumerate(
        seed_frame.groupby("topology", sort=True)
    ):
        method_rows = []
        for method_index, (method, group) in enumerate(topology_seed.groupby("method", sort=True)):
            interval = _bootstrap_interval(
                group["utility"].to_numpy(float),
                resamples=resamples,
                seed=bootstrap_seed + topology_index * 100 + method_index,
                confidence=confidence,
            )
            method_rows.append(
                {
                    "topology": topology,
                    "method": method,
                    "seeds": len(group),
                    "mean_utility": float(group["utility"].mean()),
                    "utility_ci_low": interval[0],
                    "utility_ci_high": interval[1],
                    "harmful_action_rate": float(group["harmful"].mean()),
                    "beneficial_action_rate": float(group["beneficial"].mean()),
                    "mean_delivery_ratio": float(group["delivery_ratio"].mean()),
                    "mean_p95_delay_ms": float(group["p95_delay_ms"].mean()),
                    "mean_throughput_mbps": float(group["throughput_mbps"].mean()),
                    "mean_completion_ratio": float(group["completion_ratio"].mean()),
                    "mean_queue_drops": float(group["queue_drops"].mean()),
                }
            )
        summaries.extend(method_rows)
        utility = topology_seed.pivot(index="seed", columns="method", values="utility")
        harm = topology_seed.pivot(index="seed", columns="method", values="harmful")
        utility_diff = (utility["S1_ICTC_CONTRACT"] - utility["B4_GREEDY_TELEMETRY"]).to_numpy(
            float
        )
        harm_diff = (harm["S1_ICTC_CONTRACT"] - harm["B4_GREEDY_TELEMETRY"]).to_numpy(float)
        utility_ci = _bootstrap_interval(
            utility_diff,
            resamples=resamples,
            seed=bootstrap_seed + 1000 + topology_index * 2,
            confidence=confidence,
        )
        harm_ci = _bootstrap_interval(
            harm_diff,
            resamples=resamples,
            seed=bootstrap_seed + 1001 + topology_index * 2,
            confidence=confidence,
        )
        scenario_pair = frame[
            (frame["topology"] == topology)
            & frame["method"].isin(["S1_ICTC_CONTRACT", "B4_GREEDY_TELEMETRY"])
        ].pivot(index="scenario_id", columns="method", values="harmful")
        greedy_only = int(
            (
                (scenario_pair["S1_ICTC_CONTRACT"] == 0)
                & (scenario_pair["B4_GREEDY_TELEMETRY"] == 1)
            ).sum()
        )
        ictc_only = int(
            (
                (scenario_pair["S1_ICTC_CONTRACT"] == 1)
                & (scenario_pair["B4_GREEDY_TELEMETRY"] == 0)
            ).sum()
        )
        discordant = greedy_only + ictc_only
        exact_p = (
            float(binomtest(greedy_only, discordant, 0.5, alternative="greater").pvalue)
            if discordant
            else 1.0
        )
        ictc = frame[(frame["topology"] == topology) & (frame["method"] == "S1_ICTC_CONTRACT")]
        opportunities = int(ictc["opportunity_present"].sum())
        retention = (
            float(ictc["opportunity_retained"].sum() / opportunities) if opportunities else 0.0
        )
        topology_results[str(topology)] = {
            "seed_blocks": len(utility_diff),
            "harm_rate_difference": float(harm_diff.mean()),
            "harm_rate_difference_ci": list(harm_ci),
            "harm_superiority_pass": harm_ci[1] < 0,
            "utility_difference": float(utility_diff.mean()),
            "utility_difference_ci": list(utility_ci),
            "utility_noninferiority_margin": -0.01,
            "utility_noninferiority_pass": utility_ci[0] > -0.01,
            "opportunity_retention_rate": retention,
            "opportunity_retention_pass": retention >= 0.45,
            "paired_utility_effect_size_dz": _paired_effect(utility_diff),
            "greedy_only_harmful_discordances": greedy_only,
            "ictc_only_harmful_discordances": ictc_only,
            "exact_one_sided_discordant_harm_pvalue": exact_p,
        }
        topology_results[str(topology)]["all_reference_gates_pass"] = all(
            bool(topology_results[str(topology)][key])
            for key in (
                "harm_superiority_pass",
                "utility_noninferiority_pass",
                "opportunity_retention_pass",
            )
        )
    summary_frame = pd.DataFrame.from_records(summaries)
    condition_frame = frame.groupby(["topology", "traffic", "event", "method"], as_index=False)[
        [
            "utility",
            "harmful",
            "beneficial",
            "delivery_ratio",
            "p95_delay_ms",
            "throughput_mbps",
            "queue_drops",
        ]
    ].mean()
    evidence_cost = {
        "mean_delivered_batches": float(decision_frame["delivered_batches"].mean()),
        "mean_audited_batches": float(decision_frame["audited_batches"].mean()),
        "mean_empirical_loss_rate": float(decision_frame["empirical_loss_rate"].mean()),
        "mean_latest_age_ms": float(decision_frame["latest_age_ms"].mean()),
        "mean_maximum_delivery_delay_ms": float(decision_frame["maximum_delivery_delay_ms"].mean()),
        "summed_native_wall_seconds": native_wall_seconds,
    }
    result = {
        "schema_version": "n13-external-transport-analysis-v1",
        "scenario_groups": int(decision_frame.shape[0]),
        "native_runs": int(decision_frame.shape[0] * 4),
        "independent_seeds_per_topology": 20,
        "journal_manifest_sha256": sha256(journal / "n13_transport_manifest.json"),
        "matrix_sha256": sha256(matrix_path),
        "policy_sha256": matrix["selected_policy_sha256"],
        "policy_retuned": False,
        "topology_results": topology_results,
        "both_topologies_pass_reference_gates": all(
            bool(value["all_reference_gates_pass"]) for value in topology_results.values()
        ),
        "evidence_cost": evidence_cost,
    }
    output.mkdir(parents=True)
    _write_csv_new(output / "n13_transport_long_metrics.csv", frame)
    _write_csv_new(output / "n13_transport_seed_level_metrics.csv", seed_frame)
    _write_csv_new(output / "n13_transport_method_summary.csv", summary_frame)
    _write_csv_new(output / "n13_transport_condition_summary.csv", condition_frame)
    _write_csv_new(output / "n13_transport_decision_audit.csv", decision_frame)
    write_new(output / "n13_transport_analysis.json", result)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    topology_order = sorted(topology_results)
    harm_means = [topology_results[item]["harm_rate_difference"] for item in topology_order]
    harm_errors = np.array(
        [
            [
                harm_means[index] - topology_results[item]["harm_rate_difference_ci"][0]
                for index, item in enumerate(topology_order)
            ],
            [
                topology_results[item]["harm_rate_difference_ci"][1] - harm_means[index]
                for index, item in enumerate(topology_order)
            ],
        ]
    )
    utility_means = [topology_results[item]["utility_difference"] for item in topology_order]
    utility_errors = np.array(
        [
            [
                utility_means[index] - topology_results[item]["utility_difference_ci"][0]
                for index, item in enumerate(topology_order)
            ],
            [
                topology_results[item]["utility_difference_ci"][1] - utility_means[index]
                for index, item in enumerate(topology_order)
            ],
        ]
    )
    axes[0].errorbar(harm_means, topology_order, xerr=harm_errors, fmt="o", capsize=4)
    axes[0].axvline(0, color="black", linewidth=1, linestyle="--")
    axes[0].set_xlabel("ICTC − greedy harmful-action rate (95% CI)")
    axes[1].errorbar(utility_means, topology_order, xerr=utility_errors, fmt="o", capsize=4)
    axes[1].axvline(-0.01, color="black", linewidth=1, linestyle="--")
    axes[1].set_xlabel("ICTC − greedy utility (95% CI)")
    figure.tight_layout()
    figure.savefig(output / "n13_transport_primary_contrasts.png", dpi=180)
    plt.close(figure)
    lines = [
        "# N13 external transport confirmation",
        "",
        "M005 was transported without retuning.",
        "",
    ]
    for topology in topology_order:
        value = topology_results[topology]
        lines.extend(
            [
                f"## {topology}",
                "",
                (
                    f"- Harm difference: {value['harm_rate_difference']:.6f} "
                    f"(95% CI {value['harm_rate_difference_ci']})"
                ),
                (
                    f"- Utility difference: {value['utility_difference']:.6f} "
                    f"(95% CI {value['utility_difference_ci']})"
                ),
                f"- Opportunity retention: {value['opportunity_retention_rate']:.6f}",
                f"- Reference gates: {'PASS' if value['all_reference_gates_pass'] else 'FAIL'}",
                "",
            ]
        )
    (output / "n13_transport_summary.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )
    files = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    write_new(
        output / "analysis_manifest.json",
        {"schema_version": "n13-external-transport-analysis-manifest-v1", "files": files},
    )
    if not all(
        math.isfinite(float(value["utility_difference"])) for value in topology_results.values()
    ):
        raise ValueError("N13 transport utility contrast contains a non-finite value")
    return result
