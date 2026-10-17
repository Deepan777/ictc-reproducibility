"""Seed-cross-fitted harm-risk model selection for the N13 development repair."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from ictc.ns3.n12 import _csv_rows, read_object, sha256, utility, write_new
from ictc.ns3.n13 import (
    RISK_FEATURE_NAMES,
    build_risk_context,
    load_matrix,
    risk_feature_vector,
)


def _write_csv_new(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite N13 model-selection artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fit_model(features: np.ndarray, labels: np.ndarray, regularization: float) -> tuple[Any, Any]:
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        C=regularization,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=20260720,
    ).fit(scaler.transform(features), labels)
    return scaler, model


def select_model(
    journal: Path, matrix_path: Path, protocol_path: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N13 model selection: {output}")
    matrix, scenarios = load_matrix(matrix_path)
    protocol = read_object(protocol_path)
    if matrix["phase"] != "development":
        raise ValueError("N13 risk-model selection may use only development data")
    if sorted({scenario.seed for scenario in scenarios}) != protocol["source_seeds"]:
        raise ValueError("N13 risk-model seeds differ from the repair protocol")
    if set(matrix["seeds"]) & set(protocol["forbidden_confirmatory_seeds"]):
        raise ValueError("N13 risk-model selection touched a confirmation seed")
    fixed = cast(dict[str, Any], protocol["fixed_thresholds"])
    output.mkdir(parents=True)

    contexts: dict[str, dict[str, Any]] = {}
    feature_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        probe = journal / "runs" / scenario.scenario_id / "probe-route0" / "raw"
        context = build_risk_context(scenario, probe, fixed)
        contexts[scenario.scenario_id] = context
        values = risk_feature_vector(scenario, context)
        feature_rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "seed": scenario.seed,
                "greedy_action": context["greedy_action"],
                **dict(zip(RISK_FEATURE_NAMES, values, strict=True)),
            }
        )
    features_path = output / "n13_controller_safe_features.csv"
    _write_csv_new(
        features_path,
        ["scenario_id", "seed", "greedy_action", *RISK_FEATURE_NAMES],
        feature_rows,
    )

    # The complete controller-safe feature matrix is sealed before outcomes are opened here.
    labels: list[int] = []
    utilities_by_scenario: dict[str, dict[str, float]] = {}
    best_by_scenario: dict[str, str] = {}
    for scenario in scenarios:
        action_rows = {
            action: _csv_rows(
                journal / "runs" / scenario.scenario_id / f"action-{action}" / "raw" / "summary.csv"
            )[0]
            for action in ("route0", "route1", "route2")
        }
        utilities = {action: utility(row) for action, row in action_rows.items()}
        utilities_by_scenario[scenario.scenario_id] = utilities
        best_by_scenario[scenario.scenario_id] = max(sorted(utilities), key=utilities.__getitem__)
        greedy = str(contexts[scenario.scenario_id]["greedy_action"])
        labels.append(int(utilities[greedy] < utilities["route0"] - 0.02))

    features = np.asarray(
        [[float(row[name]) for name in RISK_FEATURE_NAMES] for row in feature_rows], dtype=float
    )
    label_array = np.asarray(labels, dtype=int)
    seeds = np.asarray([scenario.seed for scenario in scenarios], dtype=int)
    model_family = cast(dict[str, Any], protocol["model_family"])
    regularizations = [float(value) for value in model_family["regularization_C"]]
    probability_thresholds = [
        float(value) for value in model_family["maximum_predicted_harm_probability"]
    ]
    probabilities: dict[float, np.ndarray] = {}
    for regularization in regularizations:
        predicted = np.zeros(len(scenarios), dtype=float)
        for held_out_seed in sorted(set(seeds.tolist())):
            train = seeds != held_out_seed
            test = ~train
            scaler, model = _fit_model(features[train], label_array[train], regularization)
            predicted[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
        probabilities[regularization] = predicted

    greedy_utilities = [
        utilities_by_scenario[scenario.scenario_id][
            str(contexts[scenario.scenario_id]["greedy_action"])
        ]
        for scenario in scenarios
    ]
    greedy_harmful = sum(labels)
    greedy_harm_rate = greedy_harmful / len(scenarios)
    greedy_mean_utility = sum(greedy_utilities) / len(scenarios)
    gate = cast(dict[str, float], protocol["development_gate"])
    metric_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for model_index, (regularization, threshold) in enumerate(
        itertools.product(regularizations, probability_thresholds)
    ):
        candidate_id = f"M{model_index:03d}"
        selected_utilities: list[float] = []
        harmful = 0
        opportunities = 0
        retained = 0
        disagreements = 0
        for index, scenario in enumerate(scenarios):
            greedy = str(contexts[scenario.scenario_id]["greedy_action"])
            probability = float(probabilities[regularization][index])
            action = greedy if probability <= threshold else "route0"
            utilities = utilities_by_scenario[scenario.scenario_id]
            keep = utilities["route0"]
            best = best_by_scenario[scenario.scenario_id]
            selected_utilities.append(utilities[action])
            harmful += int(utilities[action] < keep - 0.02)
            opportunity = utilities[best] > keep + 0.02
            opportunities += int(opportunity)
            retained += int(opportunity and utilities[action] >= utilities[best] - 0.02)
            disagreements += int(action != greedy)
            decision_rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "seed": scenario.seed,
                    "candidate_id": candidate_id,
                    "regularization_C": regularization,
                    "risk_threshold": threshold,
                    "cross_fitted_harm_probability": probability,
                    "greedy_action": greedy,
                    "selected_action": action,
                }
            )
        count = len(scenarios)
        harm_rate = harmful / count
        mean_utility = sum(selected_utilities) / count
        retention = retained / opportunities if opportunities else 0.0
        disagreement = disagreements / count
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
                "regularization_C": regularization,
                "maximum_predicted_harm_probability": threshold,
                "mean_utility": mean_utility,
                "utility_difference_vs_greedy": mean_utility - greedy_mean_utility,
                "harmful_actions": harmful,
                "harmful_action_rate": harm_rate,
                "harm_rate_difference_vs_greedy": harm_rate - greedy_harm_rate,
                "opportunity_retention_rate": retention,
                "greedy_disagreement_rate": disagreement,
                "development_gate_pass": int(eligible),
            }
        )
    decisions_path = output / "n13_cross_fitted_decisions.csv"
    _write_csv_new(decisions_path, list(decision_rows[0]), decision_rows)
    metrics_path = output / "n13_model_candidates.csv"
    _write_csv_new(metrics_path, list(metric_rows[0]), metric_rows)
    eligible_rows = [row for row in metric_rows if bool(row["development_gate_pass"])]
    selected = min(
        eligible_rows,
        key=lambda row: (
            float(row["harmful_action_rate"]),
            -float(row["mean_utility"]),
            -float(row["opportunity_retention_rate"]),
            str(row["candidate_id"]),
        ),
        default=None,
    )
    controller_thresholds = None
    if selected is not None:
        regularization = float(selected["regularization_C"])
        scaler, model = _fit_model(features, label_array, regularization)
        controller_thresholds = {
            **fixed,
            "risk_model": {
                "schema_version": "n13-standardized-logistic-risk-v1",
                "feature_names": list(RISK_FEATURE_NAMES),
                "means": [float(value) for value in scaler.mean_],
                "scales": [float(value) for value in scaler.scale_],
                "coefficients": [float(value) for value in model.coef_[0]],
                "intercept": float(model.intercept_[0]),
                "regularization_C": regularization,
                "maximum_predicted_harm_probability": float(
                    selected["maximum_predicted_harm_probability"]
                ),
                "training_rows": len(scenarios),
                "harmful_training_rows": int(label_array.sum()),
                "selection_validation": "leave-one-seed-out cross-fitting",
            },
        }
    payload = {
        "schema_version": "n13-selected-risk-model-v1",
        "development_gate_pass": selected is not None,
        "selected_candidate_id": selected["candidate_id"] if selected is not None else None,
        "controller_thresholds": controller_thresholds,
        "selected_development_metrics": selected,
        "greedy_development_metrics": {
            "mean_utility": greedy_mean_utility,
            "harmful_actions": greedy_harmful,
            "harmful_action_rate": greedy_harm_rate,
        },
        "candidate_count": len(metric_rows),
        "eligible_candidate_count": len(eligible_rows),
        "protocol_sha256": sha256(protocol_path),
        "development_matrix_sha256": sha256(matrix_path),
        "development_manifest_sha256": sha256(journal / "n13_manifest.json"),
        "controller_safe_features_sha256": sha256(features_path),
        "cross_fitted_decisions_sha256": sha256(decisions_path),
        "candidate_metrics_sha256": sha256(metrics_path),
        "confirmatory_seeds_accessed": False,
    }
    policy_path = output / "n13_selected_policy.json"
    write_new(policy_path, payload)
    write_new(
        output / "selection_manifest.json",
        {
            "schema_version": "n13-model-selection-manifest-v1",
            "selected_policy_sha256": sha256(policy_path),
            "controller_safe_features_sha256": sha256(features_path),
            "cross_fitted_decisions_sha256": sha256(decisions_path),
            "candidate_metrics_sha256": sha256(metrics_path),
        },
    )
    return payload
