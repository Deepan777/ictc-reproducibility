"""Frozen seed-blocked confirmatory analysis for N10."""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n09 import N09MethodId
from ictc.ns3.stochastic import StochasticScenario

_METHODS = tuple(method.value for method in N09MethodId)
_D1 = N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value
_BASELINES = tuple(method for method in _METHODS if method != _D1)
_UTILITY = {"BENEFICIAL": 1, "HARMFUL": -1, "KEEP": 0, "NO_MATERIAL_DIFFERENCE": 0}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _interval(values: list[float]) -> list[float]:
    return [round(_quantile(values, 0.025), 9), round(_quantile(values, 0.975), 9)]


def _sign_flip_p(seed_sum_differences: list[int]) -> float:
    observed = sum(seed_sum_differences)
    distribution = [0]
    for value in seed_sum_differences:
        distribution = [item + value for item in distribution] + [
            item - value for item in distribution
        ]
    return sum(value >= observed for value in distribution) / len(distribution)


def _holm(raw: dict[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=lambda name: raw[name])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, name in enumerate(ordered):
        running = max(running, raw[name] * (count - rank))
        adjusted[name] = min(1.0, running)
    return adjusted


def _wilson(successes: int, total: int) -> list[float]:
    if total <= 0 or not 0 <= successes <= total:
        raise ValueError("Wilson interval inputs are invalid")
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return [round(max(0.0, centre - half), 9), round(min(1.0, centre + half), 9)]


def _effect(differences: list[float]) -> dict[str, float | str]:
    mean = statistics.mean(differences)
    deviation = statistics.stdev(differences)
    if deviation > 0:
        dz: float | str = round(mean / deviation, 9)
    elif mean > 0:
        dz = "INFINITE"
    elif mean < 0:
        dz = "-INFINITE"
    else:
        dz = "UNDEFINED"
    positive = sum(value > 0 for value in differences)
    negative = sum(value < 0 for value in differences)
    denominator = positive + negative
    rank_biserial = (positive - negative) / denominator if denominator else 0.0
    return {"paired_d_z": dz, "rank_biserial": round(rank_biserial, 9)}


def analyze_n10(
    *, root: Path, scenarios: tuple[StochasticScenario, ...], analysis: dict[str, Any]
) -> dict[str, Any]:
    expected_analysis = {
        "utility_scores": _UTILITY,
        "independent_block": "seed",
        "confidence_level": 0.95,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260715,
        "paired_test": "exact one-sided seed-block sign-flip",
        "multiplicity": "Holm across D1 versus B0-B4",
        "family_alpha": 0.05,
        "minimum_supported_beneficial": 32,
        "required_d1_harmful": 0,
    }
    if analysis != expected_analysis:
        raise ValueError("N10 analysis configuration differs from the freeze")
    if len(scenarios) != 720 or {scenario.seed for scenario in scenarios} != set(
        range(82000, 82020)
    ):
        raise ValueError("N10 analysis requires the exact complete confirmatory matrix")
    counts = {method: {label: 0 for label in _UTILITY} for method in _METHODS}
    seed_sums = {seed: {method: 0 for method in _METHODS} for seed in range(82000, 82020)}
    seed_cells = {seed: 0 for seed in range(82000, 82020)}
    support = {"total": 0, "beneficial": 0, "harmful": 0, "other": 0}
    for scenario in scenarios:
        payload = _object(root / "scores" / scenario.scenario_id / "score.json")
        if (
            payload.get("scenario_id") != scenario.scenario_id
            or payload.get("confirmatory") is not True
        ):
            raise ValueError("N10 score is not bound confirmatory evidence")
        rows = payload.get("rows")
        if not isinstance(rows, list) or {row.get("method_id") for row in rows} != set(_METHODS):
            raise ValueError("N10 score method rows are incomplete")
        seed_cells[scenario.seed] += 1
        by_method = {row["method_id"]: row for row in rows}
        for method, row in by_method.items():
            label = row["physical_label"]
            if label not in _UTILITY:
                raise ValueError("N10 score contains an unknown physical label")
            counts[method][label] += 1
            seed_sums[scenario.seed][method] += _UTILITY[label]
        if payload["support_applied"]:
            support["total"] += 1
            label = by_method[_D1]["physical_label"]
            key = label.lower() if label in {"BENEFICIAL", "HARMFUL"} else "other"
            support[key] += 1
    if any(value != 36 for value in seed_cells.values()) or support["total"] != 40:
        raise ValueError("N10 analysis found incomplete seed blocks or support opportunities")
    seeds = list(range(82000, 82020))
    seed_means = {method: [seed_sums[seed][method] / 36 for seed in seeds] for method in _METHODS}
    rng = random.Random(analysis["bootstrap_seed"])
    samples = [
        [rng.randrange(len(seeds)) for _ in seeds] for _ in range(analysis["bootstrap_resamples"])
    ]
    method_results: dict[str, Any] = {}
    for method in _METHODS:
        values = seed_means[method]
        boot = [statistics.mean(values[index] for index in sample) for sample in samples]
        non_keep = sum(counts[method][label] for label in _UTILITY if label != "KEEP")
        harmful = counts[method]["HARMFUL"]
        method_results[method] = {
            "mean_seed_utility": round(statistics.mean(values), 9),
            "bootstrap_95_ci": _interval(boot),
            "counts": counts[method],
            "harmful_nonkeep_rate": round(harmful / non_keep, 9) if non_keep else 0.0,
            "harmful_nonkeep_wilson_95_ci": (_wilson(harmful, non_keep) if non_keep else None),
        }
    raw_p: dict[str, float] = {}
    contrast_work: dict[str, tuple[list[int], list[float], list[float]]] = {}
    for baseline in _BASELINES:
        name = f"{_D1}_minus_{baseline}"
        integer_differences = [seed_sums[seed][_D1] - seed_sums[seed][baseline] for seed in seeds]
        differences = [value / 36 for value in integer_differences]
        boot = [statistics.mean(differences[index] for index in sample) for sample in samples]
        raw_p[name] = _sign_flip_p(integer_differences)
        contrast_work[name] = (integer_differences, differences, boot)
    adjusted = _holm(raw_p)
    contrasts: dict[str, dict[str, Any]] = {}
    for name, (_integers, differences, boot) in contrast_work.items():
        contrasts[name] = {
            "mean_paired_seed_difference": round(statistics.mean(differences), 9),
            "bootstrap_95_ci": _interval(boot),
            "exact_one_sided_sign_flip_p": raw_p[name],
            "holm_adjusted_p": adjusted[name],
            "effect_size": _effect(differences),
        }
    primary = f"{_D1}_minus_{N09MethodId.B3_CAPACITY.value}"
    supported_ci = _wilson(support["beneficial"], support["total"])
    success = (
        contrasts[primary]["bootstrap_95_ci"][0] > 0
        and contrasts[primary]["holm_adjusted_p"] < analysis["family_alpha"]
        and counts[_D1]["HARMFUL"] == analysis["required_d1_harmful"]
        and counts[N09MethodId.B3_CAPACITY.value]["HARMFUL"] == 0
        and support["beneficial"] >= analysis["minimum_supported_beneficial"]
        and support["harmful"] == 0
    )
    result = {
        "schema_version": "n10-confirmatory-analysis-v1",
        "analysis_freeze": analysis,
        "independent_seed_blocks": 20,
        "complete_factorial_cells": len(scenarios),
        "method_results": method_results,
        "paired_contrasts": contrasts,
        "primary_contrast": primary,
        "support_opportunities": {
            **support,
            "beneficial_wilson_95_ci": supported_ci,
        },
        "confirmatory_success": success,
        "claim_boundary": (
            "Conditional on the frozen native ns-3 factorial; not broad-topology, real-network, "
            "individual-causal-benefit, or deployment evidence."
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["analysis_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
    return result


def render_n10_analysis(
    *, root: Path, scenarios: tuple[StochasticScenario, ...], analysis: dict[str, Any]
) -> str:
    return (
        json.dumps(
            analyze_n10(root=root, scenarios=scenarios, analysis=analysis), indent=2, sort_keys=True
        )
        + "\n"
    )
