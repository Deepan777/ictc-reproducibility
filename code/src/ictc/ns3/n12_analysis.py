"""Seed-blocked analysis for the N12 realistic Abilene experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from ictc.ns3.n12 import digest, load_matrix, read_object, sha256, write_new


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv_new(path: Path, frame: pd.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def load_long_frame(journal: Path, matrix_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    matrix, scenarios = load_matrix(matrix_path)
    records: list[dict[str, Any]] = []
    for scenario in scenarios:
        group_path = journal / "groups" / scenario.scenario_id / "group_manifest.json"
        score_path = journal / "scores" / scenario.scenario_id / "score.json"
        if not group_path.is_file() or not score_path.is_file():
            raise ValueError(f"missing N12 score evidence: {scenario.scenario_id}")
        group = read_object(group_path)
        if group.get("score_sha256") != sha256(score_path):
            raise ValueError(f"N12 score checksum mismatch: {scenario.scenario_id}")
        score = read_object(score_path)
        for row in cast(list[dict[str, Any]], score["rows"]):
            metrics = cast(dict[str, str], row["metrics"])
            records.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "phase": scenario.phase,
                    "traffic": scenario.traffic,
                    "event": scenario.event,
                    "matrix": scenario.matrix,
                    "seed": scenario.seed,
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
    expected = len(scenarios) * 8
    if len(frame) != expected:
        raise ValueError(f"N12 long frame has {len(frame)} rows, expected {expected}")
    return matrix, frame


def _seed_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "utility",
        "harmful",
        "beneficial",
        "opportunity_present",
        "opportunity_retained",
        "delivery_ratio",
        "mean_delay_ms",
        "p95_delay_ms",
        "p99_delay_ms",
        "jitter_ms",
        "throughput_mbps",
        "completion_ratio",
        "queue_drops",
        "observed_transfer_seconds",
    ]
    return frame.groupby(["seed", "method"], as_index=False)[numeric].mean()


def _bootstrap_interval(
    values: np.ndarray, resamples: int, seed: int, confidence: float
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(resamples, len(values)), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(samples, [alpha, 1.0 - alpha])
    return float(low), float(high)


def _exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    observed = float(differences.mean())
    count = 0
    total = 1 << len(differences)
    chunk = 1 << min(16, len(differences))
    indices = np.arange(len(differences), dtype=np.uint64)
    for start in range(0, total, chunk):
        masks = np.arange(start, min(total, start + chunk), dtype=np.uint64)[:, None]
        signs = 1.0 - 2.0 * ((masks >> indices) & 1).astype(float)
        count += int(np.count_nonzero(signs @ differences / len(differences) >= observed - 1e-15))
    return count / total


def _method_summary(
    seed_frame: pd.DataFrame,
    long_frame: pd.DataFrame,
    *,
    resamples: int,
    bootstrap_seed: int,
    confidence: float,
) -> pd.DataFrame:
    rows = []
    for index, (method, group) in enumerate(seed_frame.groupby("method", sort=True)):
        utility_values = group["utility"].to_numpy(float)
        low, high = _bootstrap_interval(
            utility_values, resamples, bootstrap_seed + index, confidence
        )
        opportunities = group["opportunity_present"].to_numpy(float)
        retained = group["opportunity_retained"].to_numpy(float)
        retention = float(retained.sum() / opportunities.sum()) if opportunities.sum() else math.nan
        rows.append(
            {
                "method": method,
                "seeds": len(group),
                "mean_utility": float(utility_values.mean()),
                "utility_ci_low": low,
                "utility_ci_high": high,
                "harmful_action_rate": float(group["harmful"].mean()),
                "beneficial_action_rate": float(group["beneficial"].mean()),
                "opportunity_retention_rate": retention,
                "mean_delivery_ratio": float(group["delivery_ratio"].mean()),
                "mean_p95_delay_ms": float(group["p95_delay_ms"].mean()),
                "mean_throughput_mbps": float(group["throughput_mbps"].mean()),
                "mean_completion_ratio": float(group["completion_ratio"].mean()),
                "mean_queue_drops": float(group["queue_drops"].mean()),
                "route_change_rate": float(
                    (
                        long_frame.loc[long_frame["method"] == method, "selected_action"]
                        != "route0"
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def _stochastic_frame(journal: Path, scenarios: tuple[Any, ...]) -> pd.DataFrame:
    rows = []
    for scenario in scenarios:
        plan = read_object(
            journal / "runs" / scenario.scenario_id / "probe-route0" / "raw" / "plan.json"
        )
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "traffic": scenario.traffic,
                "event": scenario.event,
                "matrix": scenario.matrix,
                "seed": scenario.seed,
                **{
                    name: plan[name]
                    for name in (
                        "capacity_multiplier",
                        "load_multiplier",
                        "event_start_seconds",
                        "event_duration_seconds",
                        "telemetry_loss_probability",
                        "telemetry_mean_delay_ms",
                        "degraded_rate_bps",
                        "cross_shift_rate_mbps",
                    )
                },
            }
        )
    return pd.DataFrame(rows)


def _outcome_variability(journal: Path, scenarios: tuple[Any, ...]) -> tuple[bool, int]:
    records = []
    for scenario in scenarios:
        for action in ("route0", "route1", "route2"):
            rows = _csv_rows(
                journal / "runs" / scenario.scenario_id / f"action-{action}" / "raw" / "summary.csv"
            )
            if len(rows) != 1:
                raise ValueError(f"invalid N12 action summary: {scenario.scenario_id}/{action}")
            row = rows[0]
            signature = digest(
                {
                    name: row[name]
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
                }
            )
            records.append(
                {
                    "traffic": scenario.traffic,
                    "event": scenario.event,
                    "matrix": scenario.matrix,
                    "action": action,
                    "signature": signature,
                }
            )
    counts = (
        pd.DataFrame(records)
        .groupby(["traffic", "event", "matrix", "action"])["signature"]
        .nunique()
    )
    return bool((counts > 1).all()), int((counts <= 1).sum())


def _decision_audit(journal: Path, scenarios: tuple[Any, ...]) -> pd.DataFrame:
    records = []
    for scenario in scenarios:
        decision = read_object(journal / "decisions" / scenario.scenario_id / "decisions.json")
        snapshot = cast(dict[str, Any], decision["snapshot"])
        choices = cast(dict[str, str], decision["decisions"])
        records.append(
            {
                "scenario_id": scenario.scenario_id,
                "traffic": scenario.traffic,
                "event": scenario.event,
                "matrix": scenario.matrix,
                "seed": scenario.seed,
                "contract_ready": int(bool(decision["contract_ready"])),
                "delivered_batches": int(snapshot["delivered_batches"]),
                "audited_batches": int(snapshot["audited_batches"]),
                "dropped_batches": int(snapshot["dropped_batches"]),
                "empirical_loss_rate": float(snapshot["empirical_loss_rate"]),
                "latest_age_ms": float(snapshot["latest_age_ms"]),
                "maximum_delivery_delay_ms": float(snapshot["maximum_delivery_delay_ms"]),
                "ictc_action": choices["S1_ICTC_CONTRACT"],
                "greedy_action": choices["B4_GREEDY_TELEMETRY"],
                "ictc_equals_greedy": int(
                    choices["S1_ICTC_CONTRACT"] == choices["B4_GREEDY_TELEMETRY"]
                ),
            }
        )
    return pd.DataFrame(records)


def _paired_contrasts(
    seed_frame: pd.DataFrame, *, resamples: int, bootstrap_seed: int, confidence: float
) -> pd.DataFrame:
    pivot = seed_frame.pivot(index="seed", columns="method", values="utility")
    rows = []
    baselines = sorted(method for method in pivot if method != "S1_ICTC_CONTRACT")
    for index, baseline in enumerate(baselines):
        differences = (pivot["S1_ICTC_CONTRACT"] - pivot[baseline]).to_numpy(float)
        low, high = _bootstrap_interval(
            differences, resamples, bootstrap_seed + 2000 + index, confidence
        )
        standard_deviation = float(differences.std(ddof=1))
        rows.append(
            {
                "contrast": f"S1_ICTC_CONTRACT_minus_{baseline}",
                "seed_blocks": len(differences),
                "mean_difference": float(differences.mean()),
                "ci_low": low,
                "ci_high": high,
                "paired_effect_size_dz": (
                    float(differences.mean() / standard_deviation)
                    if standard_deviation > 0
                    else math.nan
                ),
                "confirmatory_test_predeclared": int(baseline == "B4_GREEDY_TELEMETRY"),
            }
        )
    return pd.DataFrame(rows)


def analyze(journal: Path, matrix_path: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N12 analysis directory: {output}")
    output.mkdir(parents=True)
    matrix, scenarios = load_matrix(matrix_path)
    loaded_matrix, frame = load_long_frame(journal, matrix_path)
    if loaded_matrix != matrix:
        raise ValueError("N12 matrix changed during analysis")
    seed_frame = _seed_frame(frame)
    analysis = cast(dict[str, Any], matrix["analysis"])
    resamples = int(analysis["bootstrap_resamples"])
    bootstrap_seed = int(analysis["bootstrap_seed"])
    confidence = float(analysis["confidence_level"])
    method_summary = _method_summary(
        seed_frame,
        frame,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    condition_summary = (
        frame.groupby(["traffic", "event", "method"], as_index=False)
        .agg(
            mean_utility=("utility", "mean"),
            harmful_action_rate=("harmful", "mean"),
            mean_delivery_ratio=("delivery_ratio", "mean"),
            mean_p95_delay_ms=("p95_delay_ms", "mean"),
            mean_throughput_mbps=("throughput_mbps", "mean"),
        )
        .sort_values(["traffic", "event", "method"])
    )
    stochastic = _stochastic_frame(journal, scenarios)
    decision_audit = _decision_audit(journal, scenarios)
    paired_contrasts = _paired_contrasts(
        seed_frame,
        resamples=resamples,
        bootstrap_seed=bootstrap_seed,
        confidence=confidence,
    )
    variability = {
        column: int(stochastic[column].nunique())
        for column in (
            "capacity_multiplier",
            "load_multiplier",
            "event_start_seconds",
            "event_duration_seconds",
            "telemetry_loss_probability",
            "telemetry_mean_delay_ms",
        )
    }
    outcomes_vary, invariant_action_conditions = _outcome_variability(journal, scenarios)
    variability_pass = all(value > 1 for value in variability.values()) and outcomes_vary
    primary: dict[str, Any] | None = None
    if matrix["phase"] == "confirmatory":
        pivot = seed_frame.pivot(index="seed", columns="method", values="utility")
        differences = (pivot["S1_ICTC_CONTRACT"] - pivot["B4_GREEDY_TELEMETRY"]).to_numpy(float)
        low, high = _bootstrap_interval(differences, resamples, bootstrap_seed + 1000, confidence)
        standard_deviation = float(differences.std(ddof=1))
        primary = {
            "contrast": analysis["primary_confirmatory_contrast"],
            "seed_blocks": len(differences),
            "mean_difference": float(differences.mean()),
            "confidence_interval": [low, high],
            "paired_effect_size_dz": (
                float(differences.mean() / standard_deviation) if standard_deviation > 0 else None
            ),
            "exact_one_sided_sign_flip_pvalue": _exact_sign_flip_pvalue(differences),
            "direction": "greater",
        }
    _write_csv_new(output / "n12_long_metrics.csv", frame)
    _write_csv_new(output / "n12_seed_level_metrics.csv", seed_frame)
    _write_csv_new(output / "n12_method_summary.csv", method_summary)
    _write_csv_new(output / "n12_condition_summary.csv", condition_summary)
    _write_csv_new(output / "n12_stochastic_plans.csv", stochastic)
    _write_csv_new(output / "n12_decision_audit.csv", decision_audit)
    _write_csv_new(output / "n12_paired_contrasts.csv", paired_contrasts)
    figure = output / "n12_utility_by_method.png"
    ordered = method_summary.sort_values("mean_utility")
    plt.figure(figsize=(10, 5.5))
    errors = np.vstack(
        (
            ordered["mean_utility"] - ordered["utility_ci_low"],
            ordered["utility_ci_high"] - ordered["mean_utility"],
        )
    )
    plt.errorbar(
        ordered["mean_utility"],
        ordered["method"],
        xerr=errors,
        fmt="o",
        capsize=3,
    )
    plt.xlabel("Mean predeclared utility (95% seed-bootstrap CI)")
    plt.ylabel("Method")
    plt.tight_layout()
    plt.savefig(figure, dpi=180)
    plt.close()
    result = {
        "schema_version": "n12-realistic-analysis-v1",
        "phase": matrix["phase"],
        "scenario_groups": len(scenarios),
        "native_runs": len(scenarios) * 4,
        "method_rows": len(frame),
        "independent_seeds": len(matrix["seeds"]),
        "variability_unique_counts": variability,
        "all_condition_action_outcome_vectors_vary_across_seeds": outcomes_vary,
        "invariant_condition_action_vectors": invariant_action_conditions,
        "variability_gate_pass": variability_pass,
        "evidence_cost": {
            "mean_delivered_batches": float(decision_audit["delivered_batches"].mean()),
            "mean_audited_batches": float(decision_audit["audited_batches"].mean()),
            "mean_empirical_loss_rate": float(decision_audit["empirical_loss_rate"].mean()),
            "mean_latest_age_ms": float(decision_audit["latest_age_ms"].mean()),
            "mean_maximum_delivery_delay_ms": float(
                decision_audit["maximum_delivery_delay_ms"].mean()
            ),
            "contract_ready_rate": float(decision_audit["contract_ready"].mean()),
        },
        "ictc_greedy_action_agreement_rate": float(decision_audit["ictc_equals_greedy"].mean()),
        "primary_confirmatory_result": primary,
        "matrix_sha256": sha256(matrix_path),
        "journal_manifest_sha256": sha256(journal / "n12_manifest.json"),
    }
    write_new(output / "n12_analysis.json", result)
    summary_lines = [
        f"# N12 {matrix['phase']} realistic-network analysis",
        "",
        f"- Scenario groups: {len(scenarios)}",
        f"- Native runs: {len(scenarios) * 4}",
        f"- Independent seeds: {len(matrix['seeds'])}",
        f"- Stochastic variability gate: {'PASS' if variability_pass else 'FAIL'}",
        (
            "- ICTC/greedy action agreement: "
            f"{float(decision_audit['ictc_equals_greedy'].mean()):.6f}"
        ),
        f"- Contract-ready rate: {float(decision_audit['contract_ready'].mean()):.6f}",
        (
            "- Mean delivered telemetry batches per decision: "
            f"{float(decision_audit['delivered_batches'].mean()):.3f}"
        ),
        "",
        "The offline oracle is an upper bound, not a fair controller baseline. All fair methods",
        "use the same delivered pre-decision telemetry. Negative and null results are retained.",
    ]
    if primary is not None:
        summary_lines.extend(
            [
                "",
                "## Primary confirmatory contrast",
                "",
                f"- Mean utility difference: {primary['mean_difference']:.6f}",
                (
                    "- 95% seed-bootstrap CI: "
                    f"[{primary['confidence_interval'][0]:.6f}, "
                    f"{primary['confidence_interval'][1]:.6f}]"
                ),
                (
                    "- Exact one-sided seed sign-flip p-value: "
                    f"{primary['exact_one_sided_sign_flip_pvalue']:.8f}"
                ),
            ]
        )
    summary = output / "n12_summary.md"
    summary.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    manifest = {
        str(path.relative_to(output)).replace("\\", "/"): sha256(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "analysis_manifest.json"
    }
    write_new(
        output / "analysis_manifest.json",
        {
            "schema_version": "n12-analysis-manifest-v1",
            "analysis_digest": digest(result),
            "files": manifest,
        },
    )
    return result
