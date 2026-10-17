"""Deterministic paired inference for the frozen M14/M15 controller study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
from scipy.stats import binomtest, wilcoxon  # type: ignore[import-untyped]


def jsonl_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path} contains a non-object row")
        rows.append(cast(dict[str, Any], value))
    if not rows:
        raise ValueError(f"{path} is empty")
    return tuple(rows)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def child_seed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def bootstrap_mean_ci(
    values: np.ndarray, *, seed: int, replicates: int, label: str
) -> tuple[float, float]:
    """Percentile CI for the paired mean difference."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("paired bootstrap requires a nonempty vector")
    generator = np.random.default_rng(child_seed(seed, label))
    unique, counts = np.unique(values, return_counts=True)
    probabilities = counts.astype(np.float64) / values.size
    if unique.size <= 8:
        sampled = generator.multinomial(values.size, probabilities, size=replicates)
        means = sampled @ unique.astype(np.float64) / values.size
    else:
        means = np.empty(replicates, dtype=np.float64)
        for start in range(0, replicates, 250):
            count = min(250, replicates - start)
            indices = generator.integers(0, values.size, size=(count, values.size))
            means[start : start + count] = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for index, (label, value) in enumerate(ordered):
        running = max(running, min(1.0, value * (total - index)))
        adjusted[label] = running
    return adjusted


def hodges_lehmann(values: np.ndarray) -> float:
    """One-sample Hodges-Lehmann pseudomedian from Walsh averages."""
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Hodges-Lehmann requires a nonempty vector")
    upper = np.triu_indices(values.size)
    return float(np.median((values[upper[0]] + values[upper[1]]) / 2.0))


def _binary_values(rows: dict[str, dict[str, Any]], metric: str) -> np.ndarray:
    if metric == "admission_fraction":
        return np.array(
            [
                bool(row.get("executed_reroute", row.get("selected_route_changes") == 1))
                for _pair, row in sorted(rows.items())
            ],
            dtype=np.int8,
        )
    key = {
        "false_admissions": "false_admission",
        "harmful_action_rate_among_admissions": "harmful_action",
        "unnecessary_deferrals": "unnecessary_deferral",
    }[metric]
    return np.array([bool(row[key]) for _pair, row in sorted(rows.items())], dtype=np.int8)


def binary_comparison(
    reference: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    metric: str,
    *,
    seed: int,
    replicates: int,
    label: str,
) -> dict[str, object]:
    keys = sorted(set(reference) & set(comparator))
    if len(keys) != len(reference) or len(keys) != len(comparator):
        raise ValueError(f"{label} is not completely paired")
    left_rows = {key: reference[key] for key in keys}
    right_rows = {key: comparator[key] for key in keys}
    left = _binary_values(left_rows, metric)
    right = _binary_values(right_rows, metric)
    left_only = int(np.sum((left == 1) & (right == 0)))
    right_only = int(np.sum((left == 0) & (right == 1)))
    discordant = left_only + right_only
    p_value = float(binomtest(left_only, discordant, 0.5).pvalue) if discordant else 1.0
    difference = left.astype(np.float64) - right.astype(np.float64)
    lower, upper = bootstrap_mean_ci(difference, seed=seed, replicates=replicates, label=label)
    output: dict[str, object] = {
        "paired_count": len(keys),
        "a0_event_count": int(left.sum()),
        "comparator_event_count": int(right.sum()),
        "a0_rate": float(left.mean()),
        "comparator_rate": float(right.mean()),
        "risk_difference_a0_minus_comparator": float(difference.mean()),
        "paired_bootstrap_95_ci": [lower, upper],
        "a0_only_count": left_only,
        "comparator_only_count": right_only,
        "discordant_count": discordant,
        "exact_mcnemar_p_value": p_value,
    }
    if metric == "harmful_action_rate_among_admissions":
        left_admissions = _binary_values(left_rows, "admission_fraction")
        right_admissions = _binary_values(right_rows, "admission_fraction")
        output["a0_admission_count"] = int(left_admissions.sum())
        output["comparator_admission_count"] = int(right_admissions.sum())
        output["a0_conditional_harmful_rate"] = (
            float(left.sum() / left_admissions.sum()) if left_admissions.sum() else None
        )
        output["comparator_conditional_harmful_rate"] = (
            float(right.sum() / right_admissions.sum()) if right_admissions.sum() else None
        )
    return output


def continuous_comparison(
    reference: dict[str, dict[str, Any]],
    comparator: dict[str, dict[str, Any]],
    metric: str,
    *,
    seed: int,
    replicates: int,
    label: str,
) -> dict[str, object]:
    common = sorted(set(reference) & set(comparator))
    applicable = [
        key
        for key in common
        if reference[key].get(metric) is not None and comparator[key].get(metric) is not None
    ]
    if not applicable:
        return {
            "paired_count": 0,
            "inapplicable_pair_count": len(common),
            "reason": "no pair has a measured value for both methods",
        }
    differences = np.array(
        [float(reference[key][metric]) - float(comparator[key][metric]) for key in applicable],
        dtype=np.float64,
    )
    if np.all(differences == 0):
        p_value = 1.0
        method = "Wilcoxon signed-rank (all paired differences zero)"
    else:
        try:
            p_value = float(
                wilcoxon(differences, alternative="two-sided", zero_method="pratt").pvalue
            )
            method = "Wilcoxon signed-rank with Pratt zero handling"
        except ValueError:
            generator = np.random.default_rng(child_seed(seed, f"permutation:{label}"))
            observed = abs(float(differences.mean()))
            exceed = 0
            for _ in range(replicates):
                signs = generator.choice((-1.0, 1.0), size=differences.size)
                exceed += abs(float(np.mean(differences * signs))) >= observed
            p_value = (exceed + 1) / (replicates + 1)
            method = "deterministic paired sign-permutation fallback"
    lower, upper = bootstrap_mean_ci(differences, seed=seed, replicates=replicates, label=label)
    return {
        "paired_count": len(applicable),
        "inapplicable_pair_count": len(common) - len(applicable),
        "a0_mean": float(np.mean([float(reference[key][metric]) for key in applicable])),
        "comparator_mean": float(np.mean([float(comparator[key][metric]) for key in applicable])),
        "mean_difference_a0_minus_comparator": float(differences.mean()),
        "median_difference_a0_minus_comparator": float(np.median(differences)),
        "hodges_lehmann_shift_a0_minus_comparator": hodges_lehmann(differences),
        "paired_bootstrap_95_ci": [lower, upper],
        "paired_test": method,
        "paired_test_p_value": p_value,
    }


def _index(rows: tuple[dict[str, Any], ...], key: str) -> dict[str, dict[str, Any]]:
    output = {str(row[key]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate {key} in persisted rows")
    return output


def _merge(
    scores: dict[str, dict[str, Any]], decisions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    if set(scores) != set(decisions):
        raise ValueError("score and decision pair sets differ")
    return {key: scores[key] | decisions[key] for key in scores}


def _method_summary(rows: dict[str, dict[str, Any]]) -> dict[str, object]:
    admissions = _binary_values(rows, "admission_fraction")
    harmful = _binary_values(rows, "harmful_action_rate_among_admissions")
    return {
        "pair_count": len(rows),
        "false_admission_rate": float(_binary_values(rows, "false_admissions").mean()),
        "harmful_action_risk": float(harmful.mean()),
        "admission_fraction": float(admissions.mean()),
        "conditional_harmful_rate": (
            float(harmful.sum() / admissions.sum()) if admissions.sum() else None
        ),
        "unnecessary_deferral_rate": float(_binary_values(rows, "unnecessary_deferrals").mean()),
    }


def calculate_m16(decision_root: Path, amendment_path: Path) -> dict[str, object]:
    amendment = cast(dict[str, Any], json.loads(amendment_path.read_text(encoding="utf-8")))
    seed = int(amendment["analysis_seed"])
    replicates = int(amendment["bootstrap_replicates"])
    ablation_scores = jsonl_rows(decision_root / "ablation_scores.jsonl")
    ablation_decisions = jsonl_rows(decision_root / "ablation_decisions.jsonl")
    score_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    decision_groups: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for row in ablation_scores:
        group = (str(row["ablation_id"]), str(row["configuration_id"]))
        score_groups.setdefault(group, {})[str(row["pair_id"])] = row
    for row in ablation_decisions:
        group = (str(row["ablation_id"]), str(row["configuration_id"]))
        decision_groups.setdefault(group, {})[str(row["pair_id"])] = row
    groups = {
        group: _merge(scores, decision_groups[group]) for group, scores in score_groups.items()
    }
    a0 = groups[("A0", "default")]

    baseline_scores = jsonl_rows(decision_root / "baseline_scores.jsonl")
    baseline_decisions = (
        *jsonl_rows(decision_root / "baseline_b0_oracle_decisions.jsonl"),
        *jsonl_rows(decision_root / "baseline_b1_b8_decisions.jsonl"),
    )
    baselines: dict[str, dict[str, dict[str, Any]]] = {}
    for baseline in cast(list[str], amendment["primary_comparators"]):
        scores = _index(
            tuple(row for row in baseline_scores if row["baseline_id"] == baseline),
            "pair_id",
        )
        decisions = _index(
            tuple(row for row in baseline_decisions if row["baseline_id"] == baseline),
            "pair_id",
        )
        baselines[baseline] = _merge(scores, decisions)

    primary_metrics = (
        "false_admissions",
        "harmful_action_rate_among_admissions",
        "admission_fraction",
    )
    comparisons: dict[str, dict[str, object]] = {}
    p_values: dict[str, dict[str, float]] = {metric: {} for metric in primary_metrics}
    continuous_metrics = (
        "selected_latency_ms",
        "selected_packet_loss_ratio",
        "selected_jitter_ms",
        "selected_deadline_violation_rate",
        "telemetry_bytes",
    )
    for baseline, rows in baselines.items():
        primary = {
            metric: binary_comparison(
                a0,
                rows,
                metric,
                seed=seed,
                replicates=replicates,
                label=f"primary:{baseline}:{metric}",
            )
            for metric in primary_metrics
        }
        for metric in primary_metrics:
            p_values[metric][baseline] = float(
                cast(float, primary[metric]["exact_mcnemar_p_value"])
            )
        comparisons[baseline] = {
            "summary": _method_summary(rows),
            "primary": primary,
            "unnecessary_deferrals": binary_comparison(
                a0,
                rows,
                "unnecessary_deferrals",
                seed=seed,
                replicates=replicates,
                label=f"secondary:{baseline}:unnecessary_deferrals",
            ),
            "continuous": {
                metric: continuous_comparison(
                    a0,
                    rows,
                    metric,
                    seed=seed,
                    replicates=replicates,
                    label=f"secondary:{baseline}:{metric}",
                )
                for metric in continuous_metrics
            },
        }
    for metric in primary_metrics:
        adjusted = holm_adjust(p_values[metric])
        for baseline in baselines:
            primary = cast(dict[str, dict[str, object]], comparisons[baseline]["primary"])
            primary[metric]["holm_adjusted_p_value"] = adjusted[baseline]

    ablations: dict[str, object] = {}
    for (ablation, configuration), rows in sorted(groups.items()):
        if ablation == "A0":
            continue
        label = f"{ablation}:{configuration}"
        ablations[label] = {
            "summary": _method_summary(rows),
            "paired_primary": {
                metric: binary_comparison(
                    a0,
                    rows,
                    metric,
                    seed=seed,
                    replicates=replicates,
                    label=f"ablation:{label}:{metric}",
                )
                for metric in primary_metrics
            },
        }

    condition_summaries: dict[str, object] = {}
    for condition in sorted({str(row["condition_id"]) for row in a0.values()}):
        selected = {pair: row for pair, row in a0.items() if row["condition_id"] == condition}
        condition_summaries[condition] = _method_summary(selected)

    input_files = (
        "ablation_decisions.jsonl",
        "ablation_scores.jsonl",
        "baseline_b0_oracle_decisions.jsonl",
        "baseline_b1_b8_decisions.jsonl",
        "baseline_scores.jsonl",
        "scoring_audit.json",
    )
    return {
        "schema_version": "1.0.0",
        "milestone": "M16",
        "analysis": "frozen confirmatory paired analysis",
        "amendment_id": amendment["amendment_id"],
        "amendment_sha256": sha256_file(amendment_path),
        "analysis_seed": seed,
        "bootstrap_replicates": replicates,
        "pair_count": len(a0),
        "excluded_pair_count": 0,
        "a0_summary": _method_summary(a0),
        "baseline_comparisons": comparisons,
        "ablation_comparisons": ablations,
        "a0_condition_summaries": condition_summaries,
        "input_sha256": {name: sha256_file(decision_root / name) for name in input_files},
        "limitations": [
            "Throughput was not measured in immutable M15 summaries and is excluded.",
            (
                "Continuous controller outcomes use explicitly reported paired complete "
                "cases; no missing outcome is imputed."
            ),
            (
                "Conditional harmful-action rates are descriptive because admissions "
                "create unequal post-decision denominators; exact McNemar inference uses "
                "unconditional pair-level harmful-action risk."
            ),
        ],
    }


def primary_table_rows(results: dict[str, Any]) -> tuple[dict[str, object], ...]:
    rows: list[dict[str, object]] = []
    for baseline, comparison in results["baseline_comparisons"].items():
        for endpoint, value in comparison["primary"].items():
            rows.append(
                {
                    "comparator": baseline,
                    "endpoint": endpoint,
                    "n": value["paired_count"],
                    "a0_rate": value["a0_rate"],
                    "comparator_rate": value["comparator_rate"],
                    "risk_difference": value["risk_difference_a0_minus_comparator"],
                    "ci_lower": value["paired_bootstrap_95_ci"][0],
                    "ci_upper": value["paired_bootstrap_95_ci"][1],
                    "mcnemar_p": value["exact_mcnemar_p_value"],
                    "holm_p": value["holm_adjusted_p_value"],
                }
            )
    return tuple(rows)
