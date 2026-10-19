# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Deepan777
# ruff: noqa: E402
"""New admission comparisons with a hard selection-before-test barrier."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import shutil
import sys
import time

ROOT = Path(os.environ.get("ICTC_ANALYSIS_ROOT", Path(__file__).resolve().parents[1]))
CODE_ROOT = Path(
    os.environ.get(
        "ICTC_CODE_ROOT",
        ROOT / "software/code" if (ROOT / "software/code").exists() else ROOT / "code",
    )
)
sys.path[:0] = [str(CODE_ROOT), str(CODE_ROOT / "src")]
import joblib
import numpy as np
from ictc.ns3.n12 import N12Scenario
from ictc.ns3.n13 import predict_harm_probability
from revision_controller import (
    context,
    context_from_rows,
    received_rows,
    route_features,
    fluid_proposal,
    risk_features,
    decisions,
)

PROJECT = ROOT.parents[1]
ARCHIVE = Path(os.environ.get("ICTC_CAMPAIGN_ROOT", ROOT / "reproduction-inputs"))
GIT = shutil.which("git") or "git"
PROTOCOL = json.loads((ROOT / "protocols/offline_admission.json").read_text())
PROPOSERS = ("Greedy", "Learned", "Fluid")


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def head():
    return subprocess.check_output(
        [GIT, "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def write_new(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def models(topology):
    name = "geant-2012-03" if topology == "geant-stress" else topology
    path = ARCHIVE / f"analysis/models/{name}"
    freeze = read(path / "selection.json")
    assert sha(path / "bundle.joblib").upper() == freeze["bundle_sha256"].upper()
    bundle = joblib.load(path / "bundle.joblib")
    original = read(ARCHIVE / "code/configs/experiments/n13_selected_policy.json")[
        "controller_thresholds"
    ]["risk_model"]
    return bundle, freeze, original


def fluid_cost(scenario, ctx, route):
    row = ctx["latest"][route]
    payload = (
        2e6
        if scenario.traffic == "web"
        else 8e6
        if scenario.traffic == "bulk"
        else 5e6 * (scenario.duration_seconds - scenario.decision_at_seconds) / 8
    )
    return row["propagation_delay_ms"] / 1000 + 8 * (
        payload + 1219 * row["queue_packets"]
    ) / max(1, row["bottleneck_capacity_bps"])


def snapshot_values(scenario, ctx, model):
    routes = ctx["eligible"]
    x = [route_features(scenario, ctx, r) for r in ("route0", "route1", "route2")]
    prediction = model.predict(x)
    learned = max(routes, key=lambda r: (prediction[int(r[-1])], -int(r[-1])))
    fluid = fluid_proposal(scenario, ctx)
    greedy = ctx["greedy_action"]
    scores = {
        r: ctx["latest"][r]["bottleneck_capacity_bps"] / 1e6
        - 0.5 * ctx["latest"][r]["queue_packets"]
        - 0.08 * ctx["latest"][r]["propagation_delay_ms"]
        for r in ("route0", "route1", "route2")
    }
    choices = {"Greedy": greedy, "Learned": learned, "Fluid": fluid}
    margins = {
        "Greedy": scores[greedy] - scores["route0"],
        "Learned": float(prediction[int(learned[-1])] - prediction[0]),
        "Fluid": fluid_cost(scenario, ctx, "route0") - fluid_cost(scenario, ctx, fluid),
    }
    return choices, margins, x


def covariates(scenario, group, bundle, freeze, original):
    raw = group / "probe-route0/raw"
    t = int(scenario.decision_at_seconds * 1e9)
    rows = received_rows(raw, t)
    ctx = context(raw, scenario)
    model = bundle["proposers"][freeze["depth"]]
    choices, margins, x = snapshot_values(scenario, ctx, model)
    history = []
    for sample in ctx["recent_samples"][-12:]:
        batch = ctx["samples"][sample]
        cutoff = max(int(r["available_timestamp_ns"]) for r in batch.values()) + 1
        prefix = [
            r
            for r in rows
            if int(r["sample_timestamp_ns"]) <= sample
            and int(r["available_timestamp_ns"]) < cutoff
        ]
        hc = context_from_rows(prefix, cutoff)
        c, m, _ = snapshot_values(scenario, hc, model)
        history.append({"choice": c, "margin": m})
    per_tree = np.asarray(
        [
            tree.predict([x[0], x[int(choices["Learned"][-1])]])
            for tree in model.estimators_
        ]
    )
    lower = float(
        np.quantile(
            per_tree[:, 1] - per_tree[:, 0], PROTOCOL["ensemble_lower_quantile"]
        )
    )
    scores = {}
    for p in PROPOSERS:
        risk = (
            original
            if p == "Greedy"
            else bundle["risk"][
                "fluid" if p == "Fluid" else f"learned{freeze['depth']}"
            ]
        )
        scores[p] = predict_harm_probability(
            risk_features(scenario, ctx, choices[p]), risk
        )
    existing = decisions(scenario, ctx, original, bundle, freeze)
    return {
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "traffic": scenario.traffic,
        "event": scenario.event,
        "matrix": scenario.matrix,
        "quality": bool(ctx["quality_ready"]),
        "choices": choices,
        "margins": margins,
        "ensemble_lower": lower,
        "history": history,
        "risk_scores": scores,
        "existing_decisions": existing,
    }


def load(topology, phase, *, permit_test=False):
    if phase == "test" and not permit_test:
        raise PermissionError(
            "Test outcome access requires committed selection attestation"
        )
    assert phase in ("development", "validation", "test")
    bundle, freeze, original = models(topology)
    matrix = read(ARCHIVE / f"configs/{topology}-{phase}-matrix.json")
    groups = []
    for i, config in enumerate(matrix["scenarios"]):
        assert config["phase"] == phase
        scenario = N12Scenario(**config)
        group = (
            ARCHIVE / f"experiments/{topology}/{phase}/groups/{scenario.scenario_id}"
        )
        cov = covariates(scenario, group, bundle, freeze, original)
        # Controller inputs are computed first. Paired outcomes enter evaluation only here.
        outcome = read(group / "score.json")
        cov["utilities"] = outcome["utilities"]
        cov["input_digest"] = sha(group / "probe-route0/raw/telemetry.csv")
        cov["outcome_digest"] = sha(group / "score.json")
        groups.append(cov)
        if (i + 1) % 24 == 0:
            print(topology, phase, i + 1, flush=True)
    return groups, freeze


def admitted(g, p, f, param):
    choice = g["choices"][p]
    if choice == "route0" or not g["quality"]:
        return "route0"
    if f == "margin":
        ok = g["margins"][p] >= param["margin"]
    elif f == "ensemble":
        ok = g["ensemble_lower"] >= param["margin"]
    else:
        k = param["k"]
        hist = g["history"][-k:]
        ok = len(hist) == k and all(
            v["choice"][p] == choice and v["margin"][p] >= param["margin"] for v in hist
        )
    return choice if ok else "route0"


def metrics(groups, actions):
    n = len(groups)
    harm = 0
    utility = 0.0
    opp = ret = change = 0
    for g, a in zip(groups, actions, strict=True):
        u = g["utilities"]
        best = max(u.values())
        utility += u[a]
        harm += u[a] < u["route0"] - 0.02
        change += a != "route0"
        opportunity = best > u["route0"] + 0.02
        opp += opportunity
        ret += opportunity and u[a] >= best - 0.02
    return {
        "harm": harm / n,
        "harm_events": int(harm),
        "utility": utility / n,
        "retention": ret / opp if opp else None,
        "retained": int(ret),
        "opportunities": int(opp),
        "coverage": change / n,
        "groups": n,
    }


def eligible(stats, base):
    return (
        stats["harm"] <= 0.05 + 1e-12
        and stats["harm"] <= 0.8 * base["harm"] + 1e-12
        and stats["utility"] - base["utility"] >= -0.01 - 1e-12
        and stats["retention"] is not None
        and stats["retention"] >= 0.45 - 1e-12
    )


def select():
    assert not (ROOT / "analysis/selection.json").exists()
    output = {
        "protocol_commit": head(),
        "test_outcomes_opened": False,
        "topologies": {},
    }
    for topology in PROTOCOL["topologies"]:
        dev, _ = load(topology, "development")
        val, freeze = load(topology, "validation")
        record = {"existing_selection": freeze, "new_filters": {}}
        for p in PROPOSERS:
            base = metrics(val, [g["choices"][p] for g in val])
            for f in ("margin", "persistence", "ensemble"):
                if f == "ensemble" and p != "Learned":
                    continue
                values = [
                    g["ensemble_lower"] if f == "ensemble" else g["margins"][p]
                    for g in dev
                ]
                grid = sorted(
                    set(
                        [0.0]
                        + [
                            float(np.quantile(values, q))
                            for q in PROTOCOL["margin_quantiles_from_development"]
                        ]
                    )
                )
                candidates = []
                for margin in grid:
                    for k in PROTOCOL["persistence_k"] if f == "persistence" else [1]:
                        param = {"margin": margin, "k": k}
                        stats = metrics(val, [admitted(g, p, f, param) for g in val])
                        candidates.append(
                            {
                                "parameters": param,
                                "metrics": stats,
                                "eligible": eligible(stats, base),
                            }
                        )
                valid = [v for v in candidates if v["eligible"]]
                best = (
                    min(
                        valid,
                        key=lambda v: (
                            v["metrics"]["harm"],
                            -v["metrics"]["utility"],
                            -v["metrics"]["retention"],
                            v["parameters"]["margin"],
                            v["parameters"]["k"],
                        ),
                    )
                    if valid
                    else None
                )
                record["new_filters"][p + "/" + f] = {
                    "selected": best,
                    "candidates": candidates,
                    "proposer_validation": base,
                    "status": "selected"
                    if best
                    else "no eligible development/validation candidate",
                }
        output["topologies"][topology] = record
        write_new(
            ROOT / f"analysis/{topology}-selection-inputs.json",
            {"development": dev, "validation": val},
        )
    write_new(ROOT / "analysis/selection.json", output)
    print("SELECTION_WRITTEN; TEST OUTCOMES NOT OPENED", flush=True)


def attest():
    path = ROOT / "analysis/selection.json"
    data = read(path)
    assert not data.get("selection_commit")
    commit = head()
    before = subprocess.check_output(
        [GIT, "-C", str(ROOT), "show", commit + ":analysis/selection.json"]
    )
    assert json.loads(before) == data
    data["selection_commit"] = commit
    path.write_text(json.dumps(data, indent=2) + "\n")
    print("Selection annotation added; commit this metadata before evaluation:", commit)


def check_freeze():
    path = ROOT / "analysis/selection.json"
    data = read(path)
    assert "selection_commit" in data and not data["test_outcomes_opened"]
    assert (
        subprocess.run(
            [
                GIT,
                "-C",
                str(ROOT),
                "diff",
                "--quiet",
                "HEAD",
                "--",
                "analysis/selection.json",
            ]
        ).returncode
        == 0
    )
    original = json.loads(
        subprocess.check_output(
            [
                GIT,
                "-C",
                str(ROOT),
                "show",
                data["selection_commit"] + ":analysis/selection.json",
            ]
        )
    )
    assert {k: v for k, v in data.items() if k != "selection_commit"} == original
    return data


def evaluate():
    selection = check_freeze()
    assert not (ROOT / "analysis/test-access.json").exists()
    write_new(
        ROOT / "analysis/test-access.json",
        {
            "selection_commit": selection["selection_commit"],
            "attestation_commit": head(),
            "timestamp_unix": time.time(),
            "scope": "New offline evaluation; existing test populations were used in earlier published analyses.",
        },
    )
    for topology in (*PROTOCOL["topologies"], "geant-stress"):
        groups, freeze = load(topology, "test", permit_test=True)
        selected = selection["topologies"][
            "geant-2012-03" if topology == "geant-stress" else topology
        ]
        for g in groups:
            for key, value in selected["new_filters"].items():
                if value["selected"]:
                    p, f = key.split("/")
                    g["existing_decisions"][p + "_" + f] = admitted(
                        g, p, f, value["selected"]["parameters"]
                    )
            for p in PROPOSERS:
                for eta in PROTOCOL["frontier_eta"]:
                    g["existing_decisions"][f"{p}_frontier_{eta:.2f}"] = (
                        g["choices"][p]
                        if g["quality"] and g["risk_scores"][p] <= eta
                        else "route0"
                    )
            g["existing_decisions"]["OfflineOracle"] = max(
                g["utilities"], key=g["utilities"].get
            )
        write_new(
            ROOT / f"analysis/{topology}-test-offline.json",
            {"groups": groups, "existing_selection": freeze},
        )
        print("EVALUATED", topology, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["select", "attest", "evaluate"])
    args = p.parse_args()
    {"select": select, "attest": attest, "evaluate": evaluate}[args.phase]()
