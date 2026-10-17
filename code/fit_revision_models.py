"""Development-only fitting and validation-only selection for the revision."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from ictc.ns3.n12 import N12Scenario, sha256
from ictc.ns3.n13 import RISK_FEATURE_NAMES
from revision_controller import context, fluid_proposal, learned_proposal, risk_features, route_features
from run_revision_native import ROOT, new, read


def dataset(topology: str, phase: str) -> list[dict[str, Any]]:
    values = []
    matrix = read(ROOT / f"configs/{topology}-{phase}-matrix.json")
    for config in matrix["scenarios"]:
        root = ROOT / f"experiments/{topology}/{phase}/groups/{config['scenario_id']}"
        if not (root / "group_manifest.json").exists():
            raise ValueError(f"incomplete {phase} group: {root}")
        scenario = N12Scenario(**config)
        ctx = context(root / "probe-route0/raw", scenario)
        outcome = read(root / "score.json")
        values.append({"scenario": scenario, "ctx": ctx, "utilities": outcome["utilities"],
                       "score": outcome, "root": root})
    return values


def regression_data(groups: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for group in groups:
        for route in ("route0", "route1", "route2"):
            x.append(route_features(group["scenario"], group["ctx"], route))
            y.append(group["utilities"][route])
    return np.asarray(x), np.asarray(y)


def forest(groups: list[dict[str, Any]], depth: int) -> ExtraTreesRegressor:
    x, y = regression_data(groups)
    return ExtraTreesRegressor(n_estimators=64, max_depth=depth, min_samples_leaf=4,
                               random_state=20261014, n_jobs=1).fit(x, y)


def logistic(x: np.ndarray, y: np.ndarray, c: float) -> dict[str, Any]:
    scaler = StandardScaler().fit(x)
    if len(set(y.tolist())) == 1:
        # This fitted constant is recorded explicitly when a development subset
        # has no class overlap. It does not assert calibrated population risk.
        prevalence = (float(y.sum()) + 1) / (len(y) + 2)
        coefficients = [0.0] * len(RISK_FEATURE_NAMES)
        intercept = float(np.log(prevalence / (1 - prevalence)))
        status = "constant score because development labels contain one class"
    else:
        estimator = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                                        solver="lbfgs", random_state=20261014).fit(scaler.transform(x), y)
        coefficients = estimator.coef_[0].tolist()
        intercept = float(estimator.intercept_[0])
        status = "class-balanced standardized logistic risk score"
    return {"feature_names": list(RISK_FEATURE_NAMES), "means": scaler.mean_.tolist(),
            "scales": scaler.scale_.tolist(), "coefficients": coefficients,
            "intercept": intercept, "regularization_C": c, "status": status,
            "training_groups": len(y), "training_harm_labels": int(y.sum())}


def fit_risk(groups: list[dict[str, Any]], proposals: list[str]) -> dict[str, Any]:
    from ictc.ns3.n13 import predict_harm_probability
    x = np.asarray([risk_features(g["scenario"], g["ctx"], p) for g, p in zip(groups, proposals, strict=True)])
    y = np.asarray([int(g["utilities"][p] < g["utilities"]["route0"] - 0.02) for g, p in zip(groups, proposals, strict=True)])
    seeds = np.asarray([g["scenario"].seed for g in groups])
    candidate_loss = {}
    for c in (0.05, 0.2, 1.0, 5.0):
        pred = np.empty(len(y))
        for seed in sorted(set(seeds.tolist())):
            train, test = seeds != seed, seeds == seed
            model = logistic(x[train], y[train], c)
            pred[test] = [predict_harm_probability(tuple(v), model) for v in x[test]]
        candidate_loss[c] = float(log_loss(y, np.column_stack((1 - pred, pred)), labels=[0, 1]))
    chosen = min(candidate_loss, key=lambda c: (candidate_loss[c], c))
    model = logistic(x, y, chosen)
    model["leave_one_seed_out_log_loss"] = candidate_loss
    return model


def fit(topology: str) -> None:
    output = ROOT / f"analysis/models/{topology}"
    if (output / "bundle.joblib").exists():
        raise FileExistsError("model fitting already completed; do not refit after validation")
    started = time.perf_counter()
    cpu_started = time.process_time()
    groups = dataset(topology, "development")
    seed_values = sorted({g["scenario"].seed for g in groups})
    proposers, risk = {}, {}
    for depth in (2, 4, 8):
        proposers[depth] = forest(groups, depth)
        out_of_fold_proposals = [""] * len(groups)
        for seed in seed_values:
            train = [g for g in groups if g["scenario"].seed != seed]
            model = forest(train, depth)
            for index, group in enumerate(groups):
                if group["scenario"].seed == seed:
                    out_of_fold_proposals[index] = learned_proposal(group["scenario"], group["ctx"], model)
        risk[f"learned{depth}"] = fit_risk(groups, out_of_fold_proposals)
    for count in (3, 6, 12):
        subset = [g for g in groups if g["scenario"].seed in seed_values[:count]]
        risk[f"greedy{count}"] = fit_risk(subset, [str(g["ctx"]["greedy_action"]) for g in subset])
    risk["fluid"] = fit_risk(groups, [fluid_proposal(g["scenario"], g["ctx"]) for g in groups])
    bundle = {"proposers": proposers, "risk": risk, "topology": topology,
              "development_seeds": seed_values, "development_groups": len(groups)}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output / "bundle.joblib", compress=3)
    new(output / "training_record.json", {
        "topology": topology, "development_seeds": seed_values, "development_groups": len(groups),
        "route_utility_training_rows": len(groups) * 3,
        "bundle_sha256": sha256(output / "bundle.joblib"),
        "wall_seconds": time.perf_counter() - started, "cpu_seconds": time.process_time() - cpu_started,
        "training_code_sha256": sha256(Path(__file__)), "test_data_opened": False,
        "proposer_risk_labels": "leave-one-development-seed-out proposer predictions",
        "risk_models": risk,
    })
    print(json.dumps({"fitted": topology, "groups": len(groups), "seconds": time.perf_counter()-started}), flush=True)


def metrics(groups: list[dict[str, Any]], method: str, baseline: str) -> dict[str, float]:
    selected, comparator, harm, base_harm, opportunities, retained = [], [], 0, 0, 0, 0
    for group in groups:
        rows = {row["method"]: row for row in group["score"]["rows"]}
        row, base = rows[method], rows[baseline]
        selected.append(row["utility"])
        comparator.append(base["utility"])
        harm += int(row["harmful"])
        base_harm += int(base["harmful"])
        opportunities += int(row["opportunity"])
        retained += int(row["retained"])
    n = len(groups)
    return {"harm": harm / n, "base_harm": base_harm / n,
            "utility": float(np.mean(selected)), "utility_difference": float(np.mean(np.asarray(selected)-comparator)),
            "retention": retained / opportunities if opportunities else 0.0,
            "opportunities": opportunities, "retained": retained}


def select(topology: str) -> None:
    output = ROOT / f"analysis/models/{topology}"
    if (output / "selection.json").exists():
        raise FileExistsError("selection already frozen")
    groups = dataset(topology, "validation")
    bundle = joblib.load(output / "bundle.joblib")
    x, y = regression_data(groups)
    errors = {depth: float(np.mean((bundle["proposers"][depth].predict(x) - y)**2)) for depth in (2, 4, 8)}
    depth = min(errors, key=lambda d: (errors[d], d))
    families = {"original": ("ThresholdICTCT", "Greedy"),
                "fluid": ("FluidICTCT", "Fluid"),
                **{f"greedy{c}": (f"AdaptedICTC{c}T", "Greedy") for c in (3, 6, 12)},
                **{f"learned{d}": (f"LearnedDepth{d}ICTC", f"LearnedDepth{d}") for d in (2, 4, 8)}}
    thresholds, records = {}, {}
    for key, (prefix, baseline) in families.items():
        candidates = []
        for threshold in [i / 20 for i in range(1, 20)]:
            metric = metrics(groups, f"{prefix}{threshold:.2f}", baseline)
            metric.update({"threshold": threshold,
                "eligible": metric["harm"] <= 0.05 and metric["harm"] <= 0.8 * metric["base_harm"]
                and metric["utility_difference"] >= -0.01 and metric["retention"] >= 0.45})
            candidates.append(metric)
        eligible = [m for m in candidates if m["eligible"]]
        chosen = min(eligible, key=lambda m: (m["harm"], -m["utility"], -m["retention"], m["threshold"])) if eligible else next(m for m in candidates if m["threshold"] == 0.30)
        thresholds[key] = chosen["threshold"]
        records[key] = {"selected": chosen, "candidate_results": candidates,
                        "selection_failed": not bool(eligible)}
    new(output / "selection.json", {
        "topology": topology, "depth": depth, "validation_mse": errors, "thresholds": thresholds,
        "threshold_selection": records,
        "development_seeds": bundle["development_seeds"],
        "validation_seeds": sorted({g["scenario"].seed for g in groups}),
        "bundle_sha256": sha256(output / "bundle.joblib"),
        "validation_matrix_sha256": sha256(ROOT / f"configs/{topology}-validation-matrix.json"),
        "test_outcomes_opened": False, "frozen_at_unix_seconds": time.time(),
    })
    print(json.dumps({"selected": topology, "depth": depth, "thresholds": thresholds,
                      "validation_failures": [k for k,v in records.items() if v["selection_failed"]]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("fit", "select"))
    parser.add_argument("--topology", required=True)
    args = parser.parse_args()
    (fit if args.action == "fit" else select)(args.topology)
