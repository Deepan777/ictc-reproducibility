"""Separate validation, outcome-free test decisions and outcome-join stages.

Run --help. No fit method or simulator invocation is used anywhere in this file.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import joblib
import numpy as np
from scipy.stats import binomtest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "code"), str(REPO / "code/src")]
from ictc.ns3.n12 import N12Scenario, utility
from ictc.ns3.n13 import predict_harm_probability
from revision_controller import context_from_rows, received_rows, risk_features
from filters import PROPOSERS, evidence, filter_action, grids, suffix

TOPOS = ("abilene", "geant-2012-03", "garr-2011-09", "geant-stress")
MEANS = ("utility", "delivery_ratio", "throughput_mbps", "p95_delay_ms", "completion_ratio", "queue_drops")
COUNTS = ("harmful_count", "beneficial_count", "opportunity_count", "retained_count", "change_count")
BASELINES = ("Keep", "OfflineOracle", "Greedy", "Learned", "Fluid", "GreedyQuality", "LearnedQuality", "FluidQuality", "LearnedICTC", "FluidICTC", "AdaptedICTC12")
SCHEMA = ["topology", "method", "seed", "groups", *MEANS, *COUNTS]
GIT = shutil.which("git")


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, payload):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def csv_new(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def tracked_commit(path):
    if GIT is None:
        raise RuntimeError("Git must be installed and available on PATH")
    relative = str(path.relative_to(REPO)).replace("\\", "/")
    value = subprocess.check_output([GIT, "log", "-1", "--format=%H", "--", relative], cwd=REPO, text=True).strip()
    if not value:
        raise ValueError(f"Required evidence not committed: {path}")
    stored = subprocess.check_output([GIT, "show", f"{value}:{relative}"], cwd=REPO)
    if stored != path.read_bytes():
        raise ValueError("Evidence differs from committed bytes")
    return value


def model_inputs(archive, topology):
    model_topo = "geant-2012-03" if topology == "geant-stress" else topology
    base = archive / f"analysis/models/{model_topo}"
    freeze = read(base / "selection.json")
    if sha(base / "bundle.joblib").lower() != freeze["bundle_sha256"].lower():
        raise ValueError("Existing model checksum mismatch")
    bundle = joblib.load(base / "bundle.joblib")
    return bundle, freeze, bundle["proposers"][freeze["depth"]]


def inputs(archive, topology, phase, model):
    matrix = read(archive / f"configs/{topology}-{phase}-matrix.json")
    result = []
    for i, config in enumerate(matrix["scenarios"]):
        scenario = N12Scenario(**config)
        group = archive / f"experiments/{topology}/{phase}/groups/{scenario.scenario_id}"
        raw_rows = received_rows(group / "probe-route0/raw", int(scenario.decision_at_seconds * 1e9))
        ctx = context_from_rows(raw_rows, int(scenario.decision_at_seconds * 1e9))
        result.append({"config": config, "root": group, "ctx": ctx, "scenario": scenario,
                       "evidence": evidence(scenario, ctx, raw_rows, model)})
        if (i + 1) % 72 == 0:
            print(f"{phase} received-only features: {topology} {i+1}/{len(matrix['scenarios'])}", flush=True)
    return result


def metric(actions, values, proposer_actions):
    selected = [v[a] for a, v in zip(actions, values, strict=True)]
    baseline = [v[a] for a, v in zip(proposer_actions, values, strict=True)]
    n = len(selected)
    opportunities = [max(v.values()) > v["route0"] + .02 for v in values]
    retained = sum(o and value >= max(v.values()) - .02 for o, value, v in zip(opportunities, selected, values, strict=True))
    harm = sum(value < v["route0"] - .02 for value, v in zip(selected, values, strict=True)) / n
    base_harm = sum(value < v["route0"] - .02 for value, v in zip(baseline, values, strict=True)) / n
    retention = retained / sum(opportunities) if sum(opportunities) else 0.0
    difference = float(np.mean(np.asarray(selected) - baseline))
    return {"harm_rate": harm, "proposer_harm_rate": base_harm, "mean_utility": float(np.mean(selected)),
            "utility_difference": difference, "retention": retention,
            "eligible": bool(harm <= .05 and harm <= .8 * base_harm and difference >= -.01 and retention >= .45)}


def rank(row):
    p = row["parameters"]
    return (row["harm_rate"], -row["mean_utility"], -row["retention"], p.get("tau", 0), p.get("k", 0), p.get("m", 0), p.get("q", 0))


def selection(args):
    output = {"protocol": "author-prespecified validation-only grids and eligibility; no new fitting", "topologies": {}}
    for topology in TOPOS[:3]:
        _, _, model = model_inputs(args.archive, topology)
        groups = inputs(args.archive, topology, "validation", model)
        if len(groups) != 72 or len({g['config']['seed'] for g in groups}) != 6:
            raise ValueError("Validation population completeness failed")
        values = [read(g["root"] / "score.json")["utilities"] for g in groups]
        data = {"seeds": sorted({g["config"]["seed"] for g in groups}), "groups": 72, "proposers": {}}
        for proposer in PROPOSERS:
            observations = [g["evidence"] for g in groups]
            grid = grids(observations, proposer)
            own = [o["proposers"][proposer]["proposal"] for o in observations]
            families = {}
            for family in ("Margin", "Persistence", "Ensemble"):
                if family not in grid:
                    continue
                for quality in (True, False):
                    name = proposer + ("Quality" if quality else "") + family
                    candidates = []
                    for params in grid[family]:
                        actions = [filter_action(o, proposer, family, params, quality=quality) for o in observations]
                        candidates.append({"parameters": params, **metric(actions, values, own)})
                    eligible = [c for c in candidates if c["eligible"]]
                    chosen = min(eligible or candidates, key=rank)
                    families[name] = {"family": family, "quality_gate": quality,
                                      "validation_failed": not bool(eligible), "selected": chosen, "candidates": candidates}
            data["proposers"][proposer] = {"grids": grid, "methods": families}
        output["topologies"][topology] = data
    write(args.output / "b1/selection_validation.json", output)


def decide(args):
    selections_path = args.output / "b1/selection_validation.json"
    selection_commit = tracked_commit(selections_path)
    selections = read(selections_path)["topologies"]
    records, sweep_records = [], []
    selected_points = {}
    for topology in TOPOS:
        bundle, freeze, model = model_inputs(args.archive, topology)
        groups = inputs(args.archive, topology, "test", model)
        selected = selections["geant-2012-03" if topology == "geant-stress" else topology]
        selected_points[topology] = {}
        for group in groups:
            common = {"population": topology, "group": group["config"]["scenario_id"], "seed": group["config"]["seed"]}
            persisted = read(group["root"] / "decisions.json")["decisions"]
            for proposer in PROPOSERS:
                if persisted[proposer] != group["evidence"]["proposers"][proposer]["proposal"]:
                    raise ValueError(f"Existing proposer changed: {topology}/{common['group']}/{proposer}")
            for proposer in PROPOSERS:
                for name, spec in selected["proposers"][proposer]["methods"].items():
                    action = filter_action(group["evidence"], proposer, spec["family"], spec["selected"]["parameters"], quality=spec["quality_gate"])
                    records.append({**common, "method": name, "action": action})
                    if spec["quality_gate"]:
                        for params in selected["proposers"][proposer]["grids"][spec["family"]]:
                            label = name + "@" + suffix(params)
                            action = filter_action(group["evidence"], proposer, spec["family"], params, quality=True)
                            sweep_records.append({**common, "method": label, "action": action})
                    selected_points[topology][name] = spec["selected"]["parameters"]
                key = {"Greedy": "greedy12", "Learned": f"learned{freeze['depth']}", "Fluid": "fluid"}[proposer]
                candidate = group["evidence"]["proposers"][proposer]["proposal"]
                risk = predict_harm_probability(risk_features(group["scenario"], group["ctx"], candidate), bundle["risk"][key])
                label = {"Greedy": "AdaptedICTC12", "Learned": "LearnedICTC", "Fluid": "FluidICTC"}[proposer]
                selected_points[topology][label] = {"eta": freeze["thresholds"][key]}
                for eta in np.arange(1, 20) / 20:
                    action = candidate if group["ctx"]["quality_ready"] and risk <= eta else "route0"
                    sweep_records.append({**common, "method": f"{label}@{eta:.2f}", "action": action})
    path = args.output / "b1/test_decisions.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    sweep_path = REPO / "analysis_tools/final_analyses_sweep_decisions.jsonl"
    with sweep_path.open("x", encoding="utf-8", newline="\n") as stream:
        for record in sweep_records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    write(args.output / "b1/test_decisions_digest.json", {"sha256": sha(path), "decisions": len(records), "selection_commit": selection_commit,
                                                          "descriptive_sweep_decisions_sha256": sha(sweep_path), "descriptive_sweep_decisions": len(sweep_records)})
    write(args.output / "b3/selected_points.json", selected_points)


def aggregate(records, condition=False):
    blocks = defaultdict(list)
    for row in records:
        key = (row["topology"], row["method"], row["traffic"], row["event"]) if condition else (row["topology"], row["method"], row["seed"])
        blocks[key].append(row)
    output = []
    for key, rows in sorted(blocks.items()):
        first = {"topology": key[0], "method": key[1], **({"traffic": key[2], "event": key[3]} if condition else {"seed": key[2]}), "groups": len(rows)}
        output.append({**first, **{m: sum(r[m] for r in rows)/len(rows) for m in MEANS}, **{m: sum(r[m] for r in rows) for m in COUNTS}})
    return output


def joined(topology, config, action, method, outcomes, values):
    best, keep = max(values.values()), values["route0"]
    opportunity = best > keep + .02
    return {"topology": topology, "method": method, "seed": config["seed"], "traffic": config["traffic"], "event": config["event"],
            "utility": values[action], **{m: float(outcomes[action][m]) for m in MEANS if m != "utility"},
            "harmful_count": int(values[action] < keep - .02), "beneficial_count": int(values[action] > keep + .02),
            "opportunity_count": int(opportunity), "retained_count": int(opportunity and values[action] >= best - .02), "change_count": int(action != "route0")}


def contrasts(seed_rows):
    indices = np.random.default_rng(20261014).integers(0, 24, size=(10000, 24))
    outputs, families = [], defaultdict(list)
    for topology in TOPOS:
        by_method = defaultdict(dict)
        for row in seed_rows:
            if row["topology"] == topology:
                by_method[row["method"]][row["seed"]] = row
        for method in sorted(by_method):
            if method in BASELINES:
                continue
            proposer = next(p for p in PROPOSERS if method.startswith(p))
            comparators = [proposer]
            if "Quality" in method:
                comparators.append({"Greedy": "AdaptedICTC12", "Learned": "LearnedICTC", "Fluid": "FluidICTC"}[proposer])
            for comparator in comparators:
                for endpoint in ("harm_rate", "utility", "retention"):
                    def val(r):
                        return (r["harmful_count"] / r["groups"] if endpoint == "harm_rate" else r["utility"] if endpoint == "utility"
                                else r["retained_count"] / r["opportunity_count"] if r["opportunity_count"] else 0.0)
                    seeds = sorted(by_method[method])
                    if len(seeds) != 24 or seeds != sorted(by_method[comparator]):
                        raise ValueError("Pairing mismatch")
                    d = np.asarray([val(by_method[method][s]) - val(by_method[comparator][s]) for s in seeds])
                    low, high = np.quantile(d[indices].mean(axis=1), [.025, .975])
                    nonzero = d[d != 0]
                    p = float(binomtest(int((nonzero > 0).sum()), len(nonzero), p=.5).pvalue) if len(nonzero) else 1.0
                    row = {"topology": topology, "comparison": f"{method}-minus-{comparator}", "endpoint": endpoint,
                           "mean_diff": float(d.mean()), "ci_low": float(low), "ci_high": float(high), "exact_sign_p": p, "holm_p": None}
                    outputs.append(row)
                    families[(topology, proposer, endpoint)].append(row)
    for family in families.values():
        previous = 0.0
        ordered = sorted(family, key=lambda r: r["exact_sign_p"])
        for rank_, row in enumerate(ordered):
            previous = min(1.0, max(previous, (len(ordered)-rank_) * row["exact_sign_p"]))
            row["holm_p"] = previous
    return outputs


def evaluate(args):
    decisions_path = args.output / "b1/test_decisions.jsonl"
    decision_commit = tracked_commit(decisions_path)
    if sha(decisions_path) != read(args.output / "b1/test_decisions_digest.json")["sha256"]:
        raise ValueError("Decision digest mismatch")
    sweep_path = REPO / "analysis_tools/final_analyses_sweep_decisions.jsonl"
    tracked_commit(sweep_path)
    actions = defaultdict(dict)
    for line in decisions_path.read_text().splitlines():
        row = json.loads(line)
        actions[(row["population"], row["group"])][row["method"]] = row["action"]
    sweeps = defaultdict(dict)
    for line in sweep_path.read_text().splitlines():
        row = json.loads(line)
        sweeps[(row["population"], row["group"])][row["method"]] = row["action"]
    primary_records, sweep_records = [], []
    for topology in TOPOS:
        matrix = read(args.archive / f"configs/{topology}-test-matrix.json")
        for config in matrix["scenarios"]:
            root = args.archive / f"experiments/{topology}/test/groups/{config['scenario_id']}"
            outcomes = {}
            for action in ("route0", "route1", "route2"):
                with (root / f"action-{action}/raw/summary.csv").open(newline="") as stream:
                    outcomes[action] = next(csv.DictReader(stream))
            values = {action: utility(row) for action, row in outcomes.items()}
            saved = read(root / "decisions.json")["decisions"]
            for method in BASELINES:
                action = max(values, key=values.__getitem__) if method == "OfflineOracle" else saved[method]
                primary_records.append(joined(topology, config, action, method, outcomes, values))
            for method, action in actions[(topology, config["scenario_id"])].items():
                primary_records.append(joined(topology, config, action, method, outcomes, values))
            for method, action in sweeps[(topology, config["scenario_id"])].items():
                sweep_records.append(joined(topology, config, action, method, outcomes, values))
    seeds = aggregate(primary_records)
    with zipfile.ZipFile(args.attachment) as zipped:
        import io
        published = list(csv.DictReader(io.StringIO(zipped.read("seed_level_metrics.csv").decode())))
        condition_schema = next(csv.reader(io.StringIO(zipped.read("condition_level_metrics.csv").decode())))
    index = {(r["topology"], r["method"], int(r["seed"])): r for r in published}
    for row in seeds:
        if row["method"] in BASELINES:
            old = index[(row["topology"], row["method"], row["seed"])]
            for field in MEANS:
                if abs(float(old[field]) - row[field]) > 1e-9:
                    raise ValueError("Duplicate baseline join check failed")
            for field in ("groups", *COUNTS):
                if int(old[field]) != row[field]:
                    raise ValueError("Duplicate baseline count check failed")
    csv_new(args.output / "b1/seed_level_metrics_filters.csv", seeds, SCHEMA)
    csv_new(args.output / "b1/condition_level_metrics_filters.csv", aggregate([r for r in primary_records if r["method"] not in BASELINES], condition=True), condition_schema)
    csv_new(args.output / "b1/paired_contrasts_filters.csv", contrasts(seeds), ["topology", "comparison", "endpoint", "mean_diff", "ci_low", "ci_high", "exact_sign_p", "holm_p"])
    eta = [r for r in sweep_records if "ICTC@" in r["method"] or "ICTC12@" in r["method"]]
    filters = [r for r in sweep_records if not ("ICTC@" in r["method"] or "ICTC12@" in r["method"])]
    csv_new(args.output / "b3/eta_sweep_seed_level.csv", aggregate(eta), SCHEMA)
    csv_new(args.output / "b3/filter_sweep_seed_level.csv", aggregate(filters), SCHEMA)
    write(REPO / "analysis_tools/final_analyses_join_record.json", {"test_decisions_commit": decision_commit, "selected_seed_rows": len(seeds), "raw_packet_campaign_rerun": False})
    print(f"Joined {len(primary_records)} selected/baseline and {len(sweep_records)} descriptive records", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("validation", "decisions", "evaluate"))
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--attachment", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "codex_outputs")
    arguments = parser.parse_args()
    {"validation": selection, "decisions": decide, "evaluate": evaluate}[arguments.stage](arguments)
