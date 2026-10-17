"""Seed-blocked distributions and confidence intervals for N11."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n11 import (
    N11_ANALYSIS_FREEZE,
    N11MethodId,
    read_stochastic_plan,
)
from ictc.ns3.stochastic import EventRegime, StochasticScenario, sha256

_METHODS = tuple(method.value for method in N11MethodId)
_S1 = N11MethodId.S1_STOCHASTIC_ICTC.value
_B3 = N11MethodId.B3_CAPACITY.value
_UTILITY = cast(dict[str, int], N11_ANALYSIS_FREEZE["utility_scores"])
_ENDPOINTS = tuple(cast(list[str], N11_ANALYSIS_FREEZE["continuous_endpoints"]))


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("a distribution cannot be empty")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _rounded(value: float) -> float:
    return round(value, 9)


def _interval(values: list[float]) -> list[float]:
    return [_rounded(_quantile(values, 0.025)), _rounded(_quantile(values, 0.975))]


def _distribution(values: list[float]) -> dict[str, float | int]:
    return {
        "count": len(values),
        "minimum": _rounded(min(values)),
        "q1": _rounded(_quantile(values, 0.25)),
        "median": _rounded(_quantile(values, 0.5)),
        "q3": _rounded(_quantile(values, 0.75)),
        "maximum": _rounded(max(values)),
        "mean": _rounded(statistics.mean(values)),
        "standard_deviation": _rounded(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def _sign_flip_p(differences: list[int]) -> float:
    observed = sum(differences)
    distribution = [0]
    for value in differences:
        distribution = [item + value for item in distribution] + [
            item - value for item in distribution
        ]
    return sum(value >= observed for value in distribution) / len(distribution)


def _endpoint(outcome: dict[str, Any]) -> dict[str, float]:
    required = {
        "tx_packets",
        "rx_packets",
        "delivery_ratio",
        "mean_delay_ns",
        "jitter_ns",
        "throughput_bps",
    }
    if set(outcome) != required:
        raise ValueError("N11 score outcome schema is invalid")
    tx = int(outcome["tx_packets"])
    rx = int(outcome["rx_packets"])
    delivery = float(outcome["delivery_ratio"])
    if tx < 0 or rx < 0 or rx > tx or not 0.0 <= delivery <= 1.0:
        raise ValueError("N11 score outcome values are invalid")
    return {
        "delivery_ratio": delivery,
        "mean_delay_ms": float(outcome["mean_delay_ns"]) / 1e6,
        "jitter_ms": float(outcome["jitter_ns"]) / 1e6,
        "throughput_mbps": float(outcome["throughput_bps"]) / 1e6,
        "loss_ratio": 1.0 - delivery,
    }


def _bootstrap_samples(*, count: int, resamples: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    return [[rng.randrange(count) for _ in range(count)] for _ in range(resamples)]


def _bootstrap_summary(
    values_by_seed: dict[int, float], seeds: list[int], samples: list[list[int]]
) -> dict[str, Any]:
    values = [values_by_seed[seed] for seed in seeds]
    boot = [statistics.mean(values[index] for index in sample) for sample in samples]
    return {
        "seed_distribution": {str(seed): _rounded(values_by_seed[seed]) for seed in seeds},
        "distribution_across_seed_means": _distribution(values),
        "bootstrap_95_ci_for_mean": _interval(boot),
    }


def analyze_n11(
    *, root: Path, scenarios: tuple[StochasticScenario, ...], analysis: dict[str, Any]
) -> dict[str, Any]:
    """Analyze only a complete N11 journal with seed as the independent block."""

    if analysis != N11_ANALYSIS_FREEZE:
        raise ValueError("N11 analysis configuration differs from the freeze")
    seeds = list(range(83000, 83020))
    if len(scenarios) != 720 or {scenario.seed for scenario in scenarios} != set(seeds):
        raise ValueError("N11 analysis requires the exact complete 20-seed matrix")
    if any(sum(item.seed == seed for item in scenarios) != 36 for seed in seeds):
        raise ValueError("N11 analysis requires 36 factorial cells in every seed block")

    method_values: dict[str, dict[str, dict[int, list[float]]]] = {
        method: {endpoint: {seed: [] for seed in seeds} for endpoint in (*_ENDPOINTS, "utility")}
        for method in _METHODS
    }
    counts = {method: {label: 0 for label in _UTILITY} for method in _METHODS}
    factor_values: dict[str, dict[int, list[float]]] = {
        name: {seed: [] for seed in seeds}
        for name in (
            "foreground_mean_interval_ms",
            "cross_traffic_rate_mbps",
            "failure_start_seconds",
            "failure_duration_seconds",
            "initial_link_capacity_mbps",
            "capacity_drop_rate_mbps",
            "propagation_delay_ms",
            "queue_limit_packets",
            "cross_shift_rate_mbps",
            "cross_shift_duration_seconds",
            "configured_telemetry_loss_probability",
            "configured_telemetry_delay_mean_ms",
            "empirical_telemetry_loss_rate",
            "empirical_telemetry_delay_mean_ms",
            "latest_telemetry_age_ms",
        )
    }
    condition_vectors: dict[str, dict[str, dict[int, tuple[float, ...]]]] = {}
    plan_hashes: dict[int, list[str]] = {seed: [] for seed in seeds}
    outcome_vectors: dict[int, list[Any]] = {seed: [] for seed in seeds}
    support = 0
    contract_ready = 0

    for scenario in scenarios:
        score_path = root / "scores" / scenario.scenario_id / "score.json"
        score = _object(score_path)
        if (
            score.get("scenario_id") != scenario.scenario_id
            or score.get("seed") != scenario.seed
            or score.get("confirmatory_validation") is not True
        ):
            raise ValueError("N11 score is not bound confirmatory-validation evidence")
        rows = score.get("rows")
        if not isinstance(rows, list) or {row.get("method_id") for row in rows} != set(_METHODS):
            raise ValueError("N11 score method rows are incomplete")
        support += int(score.get("support_applied") is True)
        contract_ready += int(score.get("contract_ready") is True)

        raw = root / "outcomes" / "scenarios" / scenario.scenario_id / "keep" / "raw"
        plan_path = raw / "stochastic_plan.json"
        plan = read_stochastic_plan(plan_path, scenario)
        if score.get("stochastic_plan_sha256") != sha256(plan_path):
            raise ValueError("N11 score is not bound to its stochastic plan")
        plan_hashes[scenario.seed].append(sha256(plan_path))
        factor_values["foreground_mean_interval_ms"][scenario.seed].append(
            plan.foreground_mean_interval_ms
        )
        factor_values["configured_telemetry_loss_probability"][scenario.seed].append(
            plan.telemetry_loss_probability
        )
        factor_values["configured_telemetry_delay_mean_ms"][scenario.seed].append(
            plan.telemetry_delay_mean_ms
        )
        for path in plan.paths:
            factor_values["cross_traffic_rate_mbps"][scenario.seed].append(
                path.cross_traffic_rate_bps / 1e6
            )
            factor_values["initial_link_capacity_mbps"][scenario.seed].append(
                path.initial_capacity_bps / 1e6
            )
            factor_values["propagation_delay_ms"][scenario.seed].append(
                path.propagation_delay_ns / 1e6
            )
            factor_values["queue_limit_packets"][scenario.seed].append(
                float(path.queue_limit_packets)
            )
        if scenario.event is EventRegime.CURRENT_FAILURE:
            factor_values["failure_start_seconds"][scenario.seed].append(plan.failure_start_seconds)
            factor_values["failure_duration_seconds"][scenario.seed].append(
                plan.failure_duration_seconds
            )
        if scenario.event is EventRegime.CANDIDATE_CAPACITY_DROP:
            factor_values["capacity_drop_rate_mbps"][scenario.seed].append(
                plan.capacity_drop_rate_bps / 1e6
            )
        if scenario.event is EventRegime.CANDIDATE_CROSS_SHIFT:
            factor_values["cross_shift_rate_mbps"][scenario.seed].append(
                plan.cross_shift_rate_bps / 1e6
            )
            factor_values["cross_shift_duration_seconds"][scenario.seed].append(
                plan.cross_shift_duration_seconds
            )

        decision = _object(root / "decisions" / scenario.scenario_id / "redecision.json")
        generations = decision.get("generations")
        if not isinstance(generations, list) or len(generations) != 2:
            raise ValueError("N11 decision generations are invalid")
        quality = generations[1].get("telemetry_quality")
        if not isinstance(quality, dict):
            raise ValueError("N11 decision telemetry quality is missing")
        factor_values["empirical_telemetry_loss_rate"][scenario.seed].append(
            float(quality["empirical_loss_rate"])
        )
        factor_values["empirical_telemetry_delay_mean_ms"][scenario.seed].append(
            float(quality["mean_delivery_delay_ns"]) / 1e6
        )
        factor_values["latest_telemetry_age_ms"][scenario.seed].append(
            float(quality["latest_age_ns"]) / 1e6
        )

        condition = "|".join(
            (scenario.topology_family.value, scenario.workload.value, scenario.event.value)
        )
        condition_vectors.setdefault(condition, {method: {} for method in _METHODS})
        for row in sorted(rows, key=lambda item: item["method_id"]):
            method = row["method_id"]
            label = row.get("physical_label")
            if label not in _UTILITY:
                raise ValueError("N11 score contains an unknown physical label")
            endpoint_values = _endpoint(cast(dict[str, Any], row.get("outcome")))
            counts[method][label] += 1
            method_values[method]["utility"][scenario.seed].append(float(_UTILITY[label]))
            for name in _ENDPOINTS:
                method_values[method][name][scenario.seed].append(endpoint_values[name])
            vector = tuple(_rounded(endpoint_values[name]) for name in _ENDPOINTS)
            condition_vectors[condition][method][scenario.seed] = vector
            outcome_vectors[scenario.seed].extend((method, *vector, label))

    samples = _bootstrap_samples(
        count=len(seeds),
        resamples=int(analysis["bootstrap_resamples"]),
        seed=int(analysis["bootstrap_seed"]),
    )
    method_results: dict[str, Any] = {}
    seed_means: dict[str, dict[str, dict[int, float]]] = {}
    for method in _METHODS:
        seed_means[method] = {}
        summaries: dict[str, Any] = {}
        for endpoint in (*_ENDPOINTS, "utility"):
            per_seed = {
                seed: statistics.mean(method_values[method][endpoint][seed]) for seed in seeds
            }
            seed_means[method][endpoint] = per_seed
            raw_values = [
                value for seed in seeds for value in method_values[method][endpoint][seed]
            ]
            summaries[endpoint] = {
                **_bootstrap_summary(per_seed, seeds, samples),
                "distribution_across_factorial_cells": _distribution(raw_values),
            }
        method_results[method] = {"label_counts": counts[method], "metrics": summaries}

    paired_contrasts: dict[str, Any] = {}
    for endpoint in (*_ENDPOINTS, "utility"):
        differences = {
            seed: seed_means[_S1][endpoint][seed] - seed_means[_B3][endpoint][seed]
            for seed in seeds
        }
        paired_contrasts[f"{_S1}_minus_{_B3}:{endpoint}"] = _bootstrap_summary(
            differences, seeds, samples
        )
    utility_integers = [
        int(round((seed_means[_S1]["utility"][seed] - seed_means[_B3]["utility"][seed]) * 36))
        for seed in seeds
    ]
    paired_contrasts[f"{_S1}_minus_{_B3}:utility"]["exact_one_sided_seed_block_sign_flip_p"] = (
        _sign_flip_p(utility_integers)
    )

    factor_results: dict[str, Any] = {}
    factor_variability = True
    for name, by_seed in factor_values.items():
        if any(not by_seed[seed] for seed in seeds):
            raise ValueError(f"N11 stochastic factor has an empty seed block: {name}")
        per_seed = {seed: statistics.mean(by_seed[seed]) for seed in seeds}
        raw_values = [value for seed in seeds for value in by_seed[seed]]
        distinct = len({_rounded(per_seed[seed]) for seed in seeds})
        variable = distinct > 1
        factor_variability = factor_variability and variable
        factor_results[name] = {
            **_bootstrap_summary(per_seed, seeds, samples),
            "distribution_across_realizations": _distribution(raw_values),
            "distinct_seed_means": distinct,
            "variable_across_seeds": variable,
        }

    plan_fingerprints = {
        seed: hashlib.sha256("".join(plan_hashes[seed]).encode("ascii")).hexdigest().upper()
        for seed in seeds
    }
    outcome_fingerprints = {
        seed: hashlib.sha256(
            json.dumps(outcome_vectors[seed], separators=(",", ":")).encode("utf-8")
        )
        .hexdigest()
        .upper()
        for seed in seeds
    }
    unique_plans = len(set(plan_fingerprints.values())) == len(seeds)
    unique_outcomes = len(set(outcome_fingerprints.values())) == len(seeds)
    condition_audit: dict[str, dict[str, bool]] = {}
    condition_variability = True
    for condition, methods in condition_vectors.items():
        condition_audit[condition] = {}
        for method, vectors in methods.items():
            variable = len(set(vectors.values())) > 1 and set(vectors) == set(seeds)
            condition_audit[condition][method] = variable
            condition_variability = condition_variability and variable

    success = factor_variability and unique_plans and unique_outcomes and condition_variability
    result: dict[str, Any] = {
        "schema_version": "n11-genuinely-stochastic-analysis-v1",
        "analysis_freeze": analysis,
        "independent_seed_blocks": 20,
        "complete_factorial_cells": len(scenarios),
        "method_decisions": len(scenarios) * len(_METHODS),
        "support_applied": support,
        "contract_ready": contract_ready,
        "method_results": method_results,
        "paired_contrasts": paired_contrasts,
        "stochastic_factor_distributions": factor_results,
        "variability_audit": {
            "unique_seed_plan_fingerprints": len(set(plan_fingerprints.values())),
            "unique_seed_outcome_fingerprints": len(set(outcome_fingerprints.values())),
            "all_stochastic_factors_variable": factor_variability,
            "all_condition_method_outcomes_variable": condition_variability,
            "seed_plan_fingerprints": {str(seed): plan_fingerprints[seed] for seed in seeds},
            "seed_outcome_fingerprints": {str(seed): outcome_fingerprints[seed] for seed in seeds},
            "condition_method_audit": condition_audit,
        },
        "stochastic_validation_success": success,
        "claim_boundary": (
            "Native ns-3 simulation evidence under the frozen factorial and declared stochastic "
            "mechanisms; not field-deployment or unrestricted-topology evidence."
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":"))
    result["analysis_digest"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()
    return result


def render_n11_analysis(
    *, root: Path, scenarios: tuple[StochasticScenario, ...], analysis: dict[str, Any]
) -> str:
    return (
        json.dumps(
            analyze_n11(root=root, scenarios=scenarios, analysis=analysis),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def render_n11_seed_csv(result: dict[str, Any]) -> str:
    """Render one row per seed, method, and outcome endpoint for paper-side reuse."""

    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(("seed", "method_id", "metric", "seed_mean"))
    methods = cast(dict[str, Any], result["method_results"])
    for method in _METHODS:
        metrics = methods[method]["metrics"]
        for metric in (*_ENDPOINTS, "utility"):
            distribution = metrics[metric]["seed_distribution"]
            for seed in range(83000, 83020):
                writer.writerow((seed, method, metric, distribution[str(seed)]))
    return output.getvalue()
