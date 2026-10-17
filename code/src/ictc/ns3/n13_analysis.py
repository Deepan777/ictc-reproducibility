"""Predeclared seed-blocked confirmatory analysis for N13."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.stats import binomtest  # type: ignore[import-untyped]

from ictc.ns3.n12 import read_object, sha256, write_new
from ictc.ns3.n13 import load_matrix


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite N13 analysis artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _bootstrap_interval(
    values: np.ndarray, *, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    if values.ndim != 1 or not len(values):
        raise ValueError("N13 bootstrap requires a non-empty one-dimensional seed vector")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return float(np.quantile(draws, alpha)), float(np.quantile(draws, 1.0 - alpha))


def _load_frames(
    journal: Path, matrix_path: Path
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    matrix, scenarios = load_matrix(matrix_path)
    records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for scenario in scenarios:
        group_path = journal / "groups" / scenario.scenario_id / "group_manifest.json"
        score_path = journal / "scores" / scenario.scenario_id / "score.json"
        decision_path = journal / "decisions" / scenario.scenario_id / "decisions.json"
        group = read_object(group_path)
        if group.get("score_sha256") != sha256(score_path):
            raise ValueError(f"N13 score checksum mismatch: {scenario.scenario_id}")
        if group.get("decision_sha256") != sha256(decision_path):
            raise ValueError(f"N13 decision checksum mismatch: {scenario.scenario_id}")
        score = read_object(score_path)
        decision = read_object(decision_path)
        snapshot = cast(dict[str, Any], decision["snapshot"])
        decisions.append(
            {
                "scenario_id": scenario.scenario_id,
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
        for row in cast(list[dict[str, Any]], score["rows"]):
            metrics = cast(dict[str, str], row["metrics"])
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
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
    frame = pd.DataFrame.from_records(records)
    decision_frame = pd.DataFrame.from_records(decisions)
    if len(frame) != len(scenarios) * 8 or len(decision_frame) != len(scenarios):
        raise ValueError("N13 analysis input is incomplete")
    return matrix, frame, decision_frame


def _seed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "utility",
        "harmful",
        "beneficial",
        "opportunity_present",
        "opportunity_retained",
        "delivery_ratio",
        "p95_delay_ms",
        "throughput_mbps",
        "completion_ratio",
        "queue_drops",
    ]
    return frame.groupby(["seed", "method"], as_index=False)[metrics].mean()


def _method_summary(
    seed_frame: pd.DataFrame, *, resamples: int, bootstrap_seed: int, confidence: float
) -> pd.DataFrame:
    rows = []
    for index, (method, group) in enumerate(seed_frame.groupby("method", sort=True)):
        utility = group["utility"].to_numpy(float)
        low, high = _bootstrap_interval(
            utility,
            resamples=resamples,
            seed=bootstrap_seed + 100 + index,
            confidence=confidence,
        )
        rows.append(
            {
                "method": method,
                "seeds": len(group),
                "mean_utility": float(utility.mean()),
                "utility_ci_low": low,
                "utility_ci_high": high,
                "harmful_action_rate": float(group["harmful"].mean()),
                "beneficial_action_rate": float(group["beneficial"].mean()),
                "mean_delivery_ratio": float(group["delivery_ratio"].mean()),
                "mean_p95_delay_ms": float(group["p95_delay_ms"].mean()),
                "mean_throughput_mbps": float(group["throughput_mbps"].mean()),
                "mean_completion_ratio": float(group["completion_ratio"].mean()),
                "mean_queue_drops": float(group["queue_drops"].mean()),
            }
        )
    return pd.DataFrame(rows)


def analyze(journal: Path, matrix_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N13 analysis directory: {output}")
    matrix, frame, decision_frame = _load_frames(journal, matrix_path)
    analysis = cast(dict[str, Any], matrix["analysis"])
    resamples = int(analysis["bootstrap_resamples"])
    bootstrap_seed = int(analysis["bootstrap_seed"])
    confidence = float(analysis["confidence_level"])
    seed_frame = _seed_frame(frame)
    method_summary = _method_summary(
        seed_frame,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    condition_summary = (
        frame.groupby(["traffic", "event", "method"], as_index=False)[
            ["utility", "harmful", "beneficial", "delivery_ratio", "p95_delay_ms"]
        ]
        .mean()
        .sort_values(["traffic", "event", "method"])
    )
    utility_pivot = seed_frame.pivot(index="seed", columns="method", values="utility")
    harm_pivot = seed_frame.pivot(index="seed", columns="method", values="harmful")
    utility_difference = (
        utility_pivot["S1_ICTC_CONTRACT"] - utility_pivot["B4_GREEDY_TELEMETRY"]
    ).to_numpy(float)
    harm_difference = (harm_pivot["S1_ICTC_CONTRACT"] - harm_pivot["B4_GREEDY_TELEMETRY"]).to_numpy(
        float
    )
    utility_interval = _bootstrap_interval(
        utility_difference,
        resamples=resamples,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    harm_interval = _bootstrap_interval(
        harm_difference,
        resamples=resamples,
        seed=bootstrap_seed + 1,
        confidence=confidence,
    )
    paired = frame[frame["method"].isin(["S1_ICTC_CONTRACT", "B4_GREEDY_TELEMETRY"])].pivot(
        index="scenario_id", columns="method", values="harmful"
    )
    greedy_only_harm = int(
        ((paired["S1_ICTC_CONTRACT"] == 0) & (paired["B4_GREEDY_TELEMETRY"] == 1)).sum()
    )
    ictc_only_harm = int(
        ((paired["S1_ICTC_CONTRACT"] == 1) & (paired["B4_GREEDY_TELEMETRY"] == 0)).sum()
    )
    discordant = greedy_only_harm + ictc_only_harm
    exact_harm_pvalue = (
        float(binomtest(greedy_only_harm, discordant, 0.5, alternative="greater").pvalue)
        if discordant
        else 1.0
    )
    ictc_rows = frame[frame["method"] == "S1_ICTC_CONTRACT"]
    opportunities = int(ictc_rows["opportunity_present"].sum())
    retention = (
        float(ictc_rows["opportunity_retained"].sum() / opportunities) if opportunities else 0.0
    )
    harm_superiority = harm_interval[1] < 0.0
    utility_noninferiority = utility_interval[0] > -0.01
    retention_pass = retention >= float(analysis["minimum_opportunity_retention_rate"])
    accepted = harm_superiority and utility_noninferiority and retention_pass
    primary = {
        "seed_blocks": len(utility_difference),
        "harm_rate_difference": float(harm_difference.mean()),
        "harm_rate_difference_ci": list(harm_interval),
        "harm_superiority_pass": harm_superiority,
        "utility_difference": float(utility_difference.mean()),
        "utility_difference_ci": list(utility_interval),
        "utility_noninferiority_margin": -0.01,
        "utility_noninferiority_pass": utility_noninferiority,
        "opportunity_retention_rate": retention,
        "opportunity_retention_pass": retention_pass,
        "greedy_only_harmful_discordances": greedy_only_harm,
        "ictc_only_harmful_discordances": ictc_only_harm,
        "exact_one_sided_discordant_harm_pvalue": exact_harm_pvalue,
        "all_confirmatory_gates_pass": accepted,
    }
    result = {
        "schema_version": "n13-confirmatory-analysis-v1",
        "scenario_groups": int(decision_frame.shape[0]),
        "native_runs": int(decision_frame.shape[0] * 4),
        "independent_seeds": int(seed_frame["seed"].nunique()),
        "journal_manifest_sha256": sha256(journal / "n13_manifest.json"),
        "matrix_sha256": sha256(matrix_path),
        "ictc_greedy_disagreement_rate": float(decision_frame["ictc_greedy_disagreement"].mean()),
        "contract_ready_rate": float(decision_frame["contract_ready"].mean()),
        "mean_predicted_harm_probability": float(
            decision_frame["predicted_harm_probability"].mean()
        ),
        "evidence_cost": {
            "mean_delivered_batches": float(decision_frame["delivered_batches"].mean()),
            "mean_audited_batches": float(decision_frame["audited_batches"].mean()),
            "mean_empirical_loss_rate": float(decision_frame["empirical_loss_rate"].mean()),
            "mean_latest_age_ms": float(decision_frame["latest_age_ms"].mean()),
            "mean_maximum_delivery_delay_ms": float(
                decision_frame["maximum_delivery_delay_ms"].mean()
            ),
        },
        "primary_confirmatory_result": primary,
    }
    output.mkdir(parents=True)
    _write_csv_new(output / "n13_long_metrics.csv", frame)
    _write_csv_new(output / "n13_seed_level_metrics.csv", seed_frame)
    _write_csv_new(output / "n13_method_summary.csv", method_summary)
    _write_csv_new(output / "n13_condition_summary.csv", condition_summary)
    _write_csv_new(output / "n13_decision_audit.csv", decision_frame)
    write_new(output / "n13_analysis.json", result)
    ordered = method_summary.sort_values("mean_utility")
    figure = output / "n13_utility_by_method.png"
    plt.figure(figsize=(10, 5.5))
    plt.errorbar(
        ordered["mean_utility"],
        ordered["method"],
        xerr=np.vstack(
            [
                ordered["mean_utility"] - ordered["utility_ci_low"],
                ordered["utility_ci_high"] - ordered["mean_utility"],
            ]
        ),
        fmt="o",
        capsize=3,
    )
    plt.xlabel("Mean predeclared utility (95% seed-bootstrap CI)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    summary = output / "n13_summary.md"
    summary.write_text(
        "\n".join(
            [
                "# N13 temporal-risk confirmation",
                "",
                f"- Scenario groups: {result['scenario_groups']}",
                f"- Independent seeds: {result['independent_seeds']}",
                f"- Harm-rate difference: {primary['harm_rate_difference']:.6f}",
                (f"- Harm 95% CI: [{harm_interval[0]:.6f}, {harm_interval[1]:.6f}]"),
                f"- Utility difference: {primary['utility_difference']:.6f}",
                (f"- Utility 95% CI: [{utility_interval[0]:.6f}, {utility_interval[1]:.6f}]"),
                f"- Opportunity retention: {retention:.6f}",
                f"- All confirmatory gates: {'PASS' if accepted else 'FAIL'}",
                "",
                "Null, harmful, and negative results are retained. The offline oracle is not a",
                "fair controller baseline.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    files = {
        path.name: sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    write_new(
        output / "analysis_manifest.json",
        {
            "schema_version": "n13-analysis-manifest-v1",
            "files": files,
        },
    )
    if not all(math.isfinite(float(value)) for value in utility_difference):
        raise ValueError("N13 primary utility contrast contains a non-finite value")
    return result
