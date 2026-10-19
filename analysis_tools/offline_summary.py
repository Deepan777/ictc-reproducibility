# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Deepan777
# ruff: noqa: E402
"""Seed-paired analysis of the newly frozen filters and descriptive frontiers."""

from pathlib import Path
import csv
import sys
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis_tools"))
from offline_admission import PROTOCOL, ARCHIVE, read, write_new, metrics


def summarise(topology):
    data = read(ROOT / f"analysis/{topology}-test-offline.json")
    groups = data["groups"]
    seeds = sorted({g["seed"] for g in groups})
    assert len(seeds) == 24
    assert all(sum(g["seed"] == s for g in groups) == 12 for s in seeds)
    ix = np.random.default_rng(20261014).integers(0, len(seeds), (10000, len(seeds)))
    methods = sorted(groups[0]["existing_decisions"])
    oracle = np.array([max(g["utilities"].values()) for g in groups])
    keep = np.array([g["utilities"]["route0"] for g in groups])

    def seedmean(x):
        return np.array([x[[g["seed"] == s for g in groups]].mean() for s in seeds])

    denom = seedmean(oracle - keep)
    denboot = denom[ix].mean(1)
    blocks = {}
    summary = {}
    records = []
    for g in groups:
        source = (
            ARCHIVE
            / f"experiments/{topology}/test/groups/{g['scenario_id']}/decisions.json"
        )
        recorded = read(source)["decisions"]
        assert all(g["existing_decisions"][m] == a for m, a in recorded.items()), g[
            "scenario_id"
        ]
    for m in methods:
        actions = [g["existing_decisions"][m] for g in groups]
        u = np.array([g["utilities"][a] for g, a in zip(groups, actions)])
        harm = (u < keep - 0.02).astype(float)
        opp = oracle > keep + 0.02
        ret = opp & (u >= oracle - 0.02)
        change = np.array([a != "route0" for a in actions], dtype=float)
        hb, ub, cb = map(seedmean, (harm, u, change))
        rb = np.array([ret[[g["seed"] == s for g in groups]].sum() for s in seeds])
        ob = np.array([opp[[g["seed"] == s for g in groups]].sum() for s in seeds])
        hcb = (ub - seedmean(keep))[ix].mean(1) / denboot

        def stat(values):
            return {
                "mean": float(values.mean()),
                "ci": np.quantile(values[ix].mean(1), [0.025, 0.975]).tolist(),
            }

        out = metrics(groups, actions)
        out.update(
            {
                "harm_ci": stat(hb)["ci"],
                "utility_ci": stat(ub)["ci"],
                "coverage_ci": stat(cb)["ci"],
                "headroom_captured": float((ub.mean() - keep.mean()) / denom.mean()),
                "headroom_ci": np.quantile(hcb, [0.025, 0.975]).tolist(),
                "retention_ci": np.quantile(
                    rb[ix].sum(1) / ob[ix].sum(1), [0.025, 0.975]
                ).tolist(),
            }
        )
        summary[m] = out
        blocks[m] = {"harm": hb, "utility": ub}
        for g, a, v, h, c, o, r in zip(groups, actions, u, harm, change, opp, ret):
            if "_frontier_" not in m:
                records.append(
                    {
                        "topology": topology,
                        "scenario_id": g["scenario_id"],
                        "seed": g["seed"],
                        "traffic": g["traffic"],
                        "event": g["event"],
                        "method": m,
                        "action": a,
                        "utility": float(v),
                        "harmful": int(h),
                        "change": int(c),
                        "opportunity": int(o),
                        "retained": int(r),
                    }
                )
    contrasts = {}
    for m in methods:
        if not any(m.endswith("_" + f) for f in ("margin", "persistence", "ensemble")):
            continue
        proposer = m.split("_")[0]
        ictc = {
            "Greedy": "OriginalICTCObservable",
            "Learned": "LearnedICTC",
            "Fluid": "FluidICTC",
        }[proposer]
        for reference in (proposer, ictc):
            for endpoint in ("harm", "utility"):
                values = blocks[m][endpoint] - blocks[reference][endpoint]
                nonzero = values[np.abs(values) > 1e-12]
                p = (
                    float(binomtest(int((nonzero > 0).sum()), len(nonzero)).pvalue)
                    if len(nonzero)
                    else 1.0
                )
                contrasts[f"{m}_minus_{reference}/{endpoint}"] = {
                    "difference": float(values.mean()),
                    "ci": np.quantile(values[ix].mean(1), [0.025, 0.975]).tolist(),
                    "p_exact_sign": p,
                    "effect_dz": float(values.mean() / values.std(ddof=1))
                    if values.std(ddof=1)
                    else None,
                }
    for endpoint in ("harm", "utility"):
        names = sorted(
            [n for n in contrasts if n.endswith("/" + endpoint)],
            key=lambda n: contrasts[n]["p_exact_sign"],
        )
        previous = 0.0
        for rank, name in enumerate(names):
            previous = max(
                previous,
                min(1.0, (len(names) - rank) * contrasts[name]["p_exact_sign"]),
            )
            contrasts[name]["p_holm"] = previous
    primary = {}
    for proposer in ("Learned", "Fluid"):
        primary[proposer] = {}
        for endpoint in ("harm", "utility"):
            values = blocks[proposer + "ICTC"][endpoint] - blocks[proposer][endpoint]
            nz = values[np.abs(values) > 1e-12]
            p = (
                float(binomtest(int((nz > 0).sum()), len(nz)).pvalue)
                if len(nz)
                else 1.0
            )
            primary[proposer][endpoint] = {
                "difference": float(values.mean()),
                "ci": np.quantile(values[ix].mean(1), [0.025, 0.975]).tolist(),
                "p_exact_sign": p,
            }
    for endpoint in ("harm", "utility"):
        previous = 0.0
        for rank, p in enumerate(
            sorted(primary, key=lambda k: primary[k][endpoint]["p_exact_sign"])
        ):
            previous = max(
                previous, min(1.0, (2 - rank) * primary[p][endpoint]["p_exact_sign"])
            )
            primary[p][endpoint]["p_holm"] = previous
    result = {
        "topology": topology,
        "seed_blocks": seeds,
        "methods": summary,
        "new_filter_contrasts": contrasts,
        "primary_contrasts": primary,
        "oracle_minus_incumbent": float(denom.mean()),
        "noninferiority_margin_fraction_of_headroom": float(0.01 / denom.mean()),
        "all_original_decisions_reproduced": True,
        "bootstrap_seed": 20261014,
        "resamples": 10000,
        "undefined_headroom_bootstraps": int((denboot == 0).sum()),
        "existing_selection": data["existing_selection"],
    }
    write_new(ROOT / f"analysis/{topology}-new-summary-v2.json", result)
    with (ROOT / f"analysis/{topology}-new-records-v2.csv").open("x", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print(
        topology,
        "HEADROOM",
        result["oracle_minus_incumbent"],
        "MARGIN FRACTION",
        result["noninferiority_margin_fraction_of_headroom"],
        flush=True,
    )
    for m in methods:
        if any(m.endswith("_" + f) for f in ("margin", "persistence", "ensemble")):
            print(
                m,
                summary[m]["harm_events"],
                summary[m]["retention"],
                summary[m]["headroom_captured"],
                flush=True,
            )
    return result


def projections():
    protocol = read(ROOT / "protocols/scaling.json")["projection"]
    rows = []
    for r in protocol["R"]:
        for n in protocol["N"]:
            rows.append(
                {
                    "routes": r,
                    "flows": n,
                    "per_flow_evidence_bytes": 32 * r * 49,
                    "aggregate_evidence_bytes": n * 32 * r * 49,
                    "total_payload_bytes_per_second": n * r * 10 * 41,
                    "additional_metadata_bytes_per_second": n * r * 10 * 16,
                    "serial_decisions_per_second_from_ictc_median": 1 / 0.0009458,
                }
            )
    write_new(
        ROOT / "analysis/analytical_projections-v2.json",
        {
            "formulae": "record=25 shared+16 transmitted metadata+8 local receipt bytes; 32 batches; 10 Hz",
            "scope": "Analytical memory/payload projections and reciprocal host-local wall median, not measured concurrency or wire traffic.",
            "rows": rows,
        },
    )


if __name__ == "__main__":
    for topology in (*PROTOCOL["topologies"], "geant-stress"):
        summarise(topology)
    projections()
