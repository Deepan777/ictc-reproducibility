# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Deepan777
# ruff: noqa: E402
"""Structured synthetic scaling of the existing exact identifier and planner."""

from pathlib import Path
import argparse
from datetime import timedelta
import json
import os
import subprocess
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
import numpy as np
from ictc.causal.graph import CausalDAG
from ictc.causal.identification import BackdoorIdentifier
from ictc.contract.planning_benchmark import capability, NOW
from ictc.contract.synthesis import ContractSynthesizer, SynthesisRequirements
from ictc.domain.enums import PrivacyClass


def instance_graph(n, k):
    if k + 2 > n:
        raise ValueError("Structurally impossible K+2 > |V|")
    u = [f"U{i:03d}" for i in range(k)]
    d = [f"D{i:03d}" for i in range(n - k - 2)]
    graph = CausalDAG(
        ["A", "Y", *u, *d],
        [("U000", "A"), ("U000", "Y"), ("A", "Y"), *[("Y", v) for v in d]],
    )
    return BackdoorIdentifier(graph, max_adjustment_variables=20), ["A", "Y", *u]


def instance_catalogue(j, f):
    rng = np.random.default_rng(20261019 + j * 10 + f)
    ads = [
        capability(
            f"c{i:03d}",
            f"s{i:03d}",
            "domain-a" if i % 2 else "domain-b",
            ("U000", "A", "Y")[i % 3],
            int(rng.integers(1, 100)),
        )
        for i in range(j)
    ]
    req = SynthesisRequirements(
        requirement_id="synthetic-scaling",
        variable_alternatives=(("A", "U000", "Y"),),
        required_sampling_rate_hz=5,
        maximum_freshness_ms=20,
        maximum_clock_skew_ms=5,
        minimum_trust_millionths=800000,
        allowed_privacy_classes=frozenset([PrivacyClass.INTERNAL]),
        permitted_administrative_domains=frozenset(["domain-a", "domain-b"]),
        required_provenance_mechanisms=frozenset(["sha256"]),
        collection_start=NOW,
        deadline=NOW + timedelta(seconds=1),
        maximum_bandwidth_bps=100000,
        minimum_distinct_domains_per_variable=1,
        failure_tolerance=f,
    )
    return ContractSynthesizer(ads), req


def timed(fn):
    start = time.perf_counter_ns()
    result = fn()
    return (time.perf_counter_ns() - start) / 1e6, result


def worker(kind, a, b):
    if kind == "identification":
        identifier, available = instance_graph(a, b)

        def fn():
            return identifier.identify("A", "Y", available)
    else:
        planner, req = instance_catalogue(a, b)

        def fn():
            return planner.synthesize_exact(req)

    pilot, result = timed(fn)
    warm, reps = (50, 1000) if pilot <= 10 else (3, 10) if pilot <= 1000 else (0, 3)
    for _ in range(warm):
        fn()
    samples = [timed(fn)[0] for _ in range(reps)]
    cache = {"query": result}
    cache_samples = []
    for _ in range(50):
        cache["query"]
    for _ in range(1000):
        cache_samples.append(timed(lambda: cache["query"])[0])
    payload = {
        "kind": kind,
        "a": a,
        "b": b,
        "pilot_ms": pilot,
        "warmups": warm,
        "repetitions": reps,
        "samples_ms": samples,
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.quantile(samples, 0.95)),
        "cache_median_ms": float(np.median(cache_samples)),
        "cache_p95_ms": float(np.quantile(cache_samples, 0.95)),
        "status": str(getattr(result, "status", getattr(result, "reason", "unknown"))),
        "eligible_sets": list(getattr(result, "minimal_adjustment_sets", [])),
    }
    print(json.dumps(payload), flush=True)


def run():
    protocol = json.loads((ROOT / "protocols/scaling.json").read_text())
    output = ROOT / "analysis/scaling"
    output.mkdir(exist_ok=False)
    cases = [
        ("identification", n, k)
        for n in protocol["nodes"]
        for k in protocol["eligible_variables"]
        if k + 2 <= n
    ]
    cases += [
        ("synthesis", j, f)
        for j in protocol["catalogue_sizes"]
        for f in protocol["failure_tolerances"]
    ]
    results = []
    for kind, a, b in cases:
        print("BENCHMARK", kind, a, b, flush=True)
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", kind, str(a), str(b)],
                capture_output=True,
                text=True,
                timeout=65,
                check=True,
            )
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            result = {
                "kind": kind,
                "a": a,
                "b": b,
                "status": "case-time-budget-censored",
                "case_limit_seconds": 65,
                "note": "Whole-case watchdog; no fabricated per-call 60-s timing.",
            }
        except subprocess.CalledProcessError as exc:
            result = {
                "kind": kind,
                "a": a,
                "b": b,
                "status": "failed",
                "stderr": exc.stderr,
            }
        (output / f"{kind}-{a}-{b}.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        results.append(result)
        print(
            "RESULT", kind, a, b, result.get("median_ms", result["status"]), flush=True
        )
    report = {"protocol": protocol, "cases": results, "full_grid": []}
    for n in protocol["nodes"]:
        for k in protocol["eligible_variables"]:
            for j in protocol["catalogue_sizes"]:
                for f in protocol["failure_tolerances"]:
                    report["full_grid"].append(
                        {
                            "nodes": n,
                            "eligible_variables": k,
                            "catalogue_size": j,
                            "failure_tolerance": f,
                            "status": "structurally invalid"
                            if k + 2 > n
                            else "factorised benchmark",
                            "identifier_record": f"identification-{n}-{k}.json"
                            if k + 2 <= n
                            else None,
                            "synthesis_record": f"synthesis-{j}-{f}.json",
                            "scope": "Identifier and synthesis measured separately; their sum is not an end-to-end measurement.",
                        }
                    )
    (ROOT / "analysis/synthesis_scaling.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--worker", nargs=3)
    args = p.parse_args()
    worker(
        args.worker[0], int(args.worker[1]), int(args.worker[2])
    ) if args.worker else run()
