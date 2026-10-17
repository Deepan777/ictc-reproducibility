"""Seed-block inference and explicit metadata correction audit."""
from __future__ import annotations
import argparse
import csv
import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any
import numpy as np
from scipy.stats import binomtest
from ictc.ns3.n12 import N12Scenario, _csv_rows, sha256, utility
from revision_controller import context, decisions
from run_revision_native import ROOT, new, read


def csv_new(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary(records: list[dict[str, Any]], *, baseline: str = "Greedy") -> dict[str, Any]:
    seeds = sorted({r["seed"] for r in records})
    methods = sorted({r["method"] for r in records})
    rng = np.random.default_rng(20261014)
    indices = rng.integers(0, len(seeds), size=(10000, len(seeds)))
    result = {}
    blocks: dict[str, dict[str, np.ndarray]] = {}
    for method in methods:
        selected = [r for r in records if r["method"] == method]
        metrics = ["utility", "harmful", "beneficial", "change", "delivery_ratio", "throughput_mbps", "p95_delay_ms", "completion_ratio", "queue_drops"]
        block = {metric: np.asarray([np.mean([float(r[metric]) for r in selected if r["seed"] == seed]) for seed in seeds]) for metric in metrics}
        blocks[method] = block
        opportunities = np.asarray([sum(int(r["opportunity"]) for r in selected if r["seed"] == seed) for seed in seeds])
        retained = np.asarray([sum(int(r["retained"]) for r in selected if r["seed"] == seed) for seed in seeds])
        output: dict[str, Any] = {"groups": len(selected), "seeds": len(seeds),
            "harm_events": sum(int(r["harmful"]) for r in selected),
            "opportunities": int(opportunities.sum()), "retained": int(retained.sum())}
        for metric, values in block.items():
            samples = values[indices].mean(axis=1)
            output[metric] = {"mean": float(values.mean()), "ci": np.quantile(samples, [0.025, 0.975]).tolist()}
        den = opportunities[indices].sum(axis=1)
        ratios = np.divide(retained[indices].sum(axis=1), den, out=np.zeros(len(den), dtype=float), where=den > 0)
        output["retention"] = {"mean": float(retained.sum() / opportunities.sum()) if opportunities.sum() else None,
                               "ci": np.quantile(ratios, [0.025, 0.975]).tolist()}
        if not output["harm_events"]:
            output["upper95_probability_any_harm_in_independent_seed_block"] = 1 - 0.05**(1 / len(seeds))
        result[method] = output
    tests = []
    for method in methods:
        if method in (baseline, "OfflineOracle"):
            continue
        for metric in ("utility", "harmful"):
            values = blocks[method][metric] - blocks[baseline][metric]
            samples = values[indices].mean(axis=1)
            ci = np.quantile(samples, [0.025, 0.975]).tolist()
            positive, negative = int(np.sum(values > 1e-12)), int(np.sum(values < -1e-12))
            p = float(binomtest(positive, positive+negative, p=0.5).pvalue) if positive+negative else 1.0
            result[method][f"{metric}_minus_{baseline}"] = {"difference": float(values.mean()), "ci": ci,
                "exact_seed_block_sign_p_two_sided": p,
                "paired_effect_dz": float(values.mean()/values.std(ddof=1)) if values.std(ddof=1) else None,
                "positive_seed_blocks": positive, "negative_seed_blocks": negative}
            tests.append((method, metric, p))
    for metric in ("utility", "harmful"):
        subset = sorted([v for v in tests if v[1] == metric], key=lambda v: v[2])
        previous = 0.0
        for rank, (method, _, p) in enumerate(subset):
            adjusted = min(1.0, max(previous, (len(subset)-rank) * p))
            previous = adjusted
            result[method][f"{metric}_minus_{baseline}"]["holm_adjusted_p"] = adjusted
    return {"independent_seed_blocks": seeds, "methods": result,
            "bootstrap_resamples": 10000, "bootstrap_seed": 20261014,
            "retention_denominator": "all scenarios where the best of three realized routes exceeds KEEP by >0.02"}


def row_record(seed: int, method: str, action: str, values: dict[str, float], outcomes: dict[str, dict[str, str]]) -> dict[str, Any]:
    best = max(values.values())
    opportunity = best > values["route0"] + 0.02
    return {"seed": seed, "method": method, "action": action, "utility": values[action],
            "harmful": values[action] < values["route0"]-0.02,
            "beneficial": values[action] > values["route0"]+0.02,
            "change": action != "route0", "opportunity": opportunity,
            "retained": opportunity and values[action] >= best-0.02,
            **{m: float(outcomes[action][m]) for m in ("delivery_ratio", "throughput_mbps", "p95_delay_ms", "completion_ratio", "queue_drops")}}


def original_audit() -> None:
    output = ROOT / "analysis/original_metadata_audit"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    original = read(ROOT / "code/configs/experiments/n13_selected_policy.json")["controller_thresholds"]["risk_model"]
    portable = ROOT / 'original-evidence'
    abilene_default = portable / 'n13-confirmation'
    transport_default = portable / 'n13-transport'
    journals = [Path(os.environ.get('ICTC_ORIGINAL_ABILENE_JOURNAL', str(abilene_default))),
                Path(os.environ.get('ICTC_ORIGINAL_TRANSPORT_JOURNAL', str(transport_default)))]
    populations: dict[str, list[dict[str, Any]]] = {}
    changes, source_refs = [], []
    names = {field.name for field in fields(N12Scenario)}
    for journal in journals:
        for path in sorted((journal / "scores").glob("*/score.json")):
            score = read(path)
            raw_scenario = score["scenario"]
            scenario = N12Scenario(**{key: value for key,value in raw_scenario.items() if key in names})
            topology = raw_scenario.get("topology", "abilene")
            population = populations.setdefault(topology, [])
            group = journal / "runs" / scenario.scenario_id
            ctx = context(group / "probe-route0/raw", scenario)
            selected = decisions(scenario, ctx, original)
            outcomes = {a: _csv_rows(group / f"action-{a}/raw/summary.csv")[0] for a in ("route0", "route1", "route2")}
            utilities = {a: utility(r) for a,r in outcomes.items()}
            original_rows = {r["method"]: r for r in score["rows"]}
            for label, old in (("Greedy", "B4_GREEDY_TELEMETRY"), ("OriginallyReportedICTC", "S1_ICTC_CONTRACT")):
                population.append(row_record(scenario.seed, label, original_rows[old]["selected_action"], utilities, outcomes))
            population.append(row_record(scenario.seed, "ObservableMetadataICTC", selected["OriginalICTCObservable"], utilities, outcomes))
            old_action = original_rows["S1_ICTC_CONTRACT"]["selected_action"]
            if old_action != selected["OriginalICTCObservable"]:
                changes.append({"topology": topology, "scenario_id": scenario.scenario_id, "original_action": old_action,
                                "observable_metadata_action": selected["OriginalICTCObservable"]})
            source_refs.append({"path": str(path), "sha256": sha256(path)})
    for topology, records in populations.items():
        csv_new(output / f"{topology}-records.csv", records)
        new(output / f"{topology}-analysis.json", summary(records))
    csv_new(output / "decision_changes.csv", changes or [{"topology": "none", "scenario_id": "none", "original_action": "none", "observable_metadata_action": "none"}])
    new(output / "audit.json", {"source_scores": source_refs, "changed_decisions": len(changes),
        "interpretation": "restricted correction audit on already-inspected original outcomes; not an independent confirmation; original artifacts retained",
        "correction": "loss only after timeout, delivered delay only, sample-generation freshness; original logistic coefficients and threshold 0.30 unchanged"})
    print(json.dumps({"metadata_correction_groups": sum(len(v)//3 for v in populations.values()), "changed_decisions": len(changes)}), flush=True)


def revision(topology: str) -> None:
    output = ROOT / f"analysis/test/{topology}"
    if output.exists():
        raise FileExistsError(output)
    matrix = read(ROOT / f"configs/{topology}-test-matrix.json")
    records, manifests = [], []
    for config in matrix["scenarios"]:
        root = ROOT / f"experiments/{topology}/test/groups/{config['scenario_id']}"
        group_seal = read(root / "group_manifest.json")
        if sha256(root / "decisions.json") != group_seal["decision_sha256"] or sha256(root / "score.json") != group_seal["score_sha256"]:
            raise ValueError("test decision/outcome integrity failed")
        seal_time = read(root / "decisions.json")["created_at_unix_seconds"]
        for action in ("route0", "route1", "route2"):
            manifest = read(root / f"action-{action}/manifest.json")
            if manifest["completed_at_unix_seconds"] < seal_time:
                raise ValueError("test outcome preceded persisted decision")
            for relative, expected in manifest["files"].items():
                if sha256(root / f"action-{action}" / relative) != expected:
                    raise ValueError("native evidence checksum failure")
        score = read(root / "score.json")
        outcomes = {a: _csv_rows(root / f"action-{a}/raw/summary.csv")[0] for a in ("route0", "route1", "route2")}
        for row in score["rows"]:
            record = row_record(config["seed"], row["method"], row["action"], score["utilities"], outcomes)
            record.update({"scenario_id": config["scenario_id"], "traffic": config["traffic"], "event": config["event"], "matrix": config["matrix"]})
            records.append(record)
        manifests.append({"path": str(root / "group_manifest.json"), "sha256": sha256(root / "group_manifest.json")})
    output.mkdir(parents=True)
    csv_new(output / "records.csv", records)
    analysis = summary(records)
    # Matched admission contrasts additionally use their own proposer.
    for proposer, admitted in (("Learned", "LearnedICTC"), ("Fluid", "FluidICTC")):
        paired = summary([r for r in records if r["method"] in (proposer, admitted)], baseline=proposer)
        analysis[f"matched_{proposer}"] = paired
    new(output / "analysis.json", analysis)
    new(output / "integrity.json", {"groups": len(matrix["scenarios"]), "native_runs": len(matrix["scenarios"])*4,
        "seed_blocks": len({r["seed"] for r in records}), "failures_excluded": 0,
        "group_manifests": manifests, "matrix_sha256": sha256(ROOT / f"configs/{topology}-test-matrix.json")})
    print(json.dumps({"analyzed": topology, "groups": len(matrix["scenarios"]), "methods": len(analysis["methods"])}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("original", "revision"))
    parser.add_argument("--topology")
    args = parser.parse_args()
    original_audit() if args.action == "original" else revision(args.topology)
