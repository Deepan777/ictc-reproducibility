# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Deepan777
# ruff: noqa: E402
"""One isolated slow-case call, retaining the earlier censored multi-call case."""

from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if "--worker" in sys.argv:
    from synthesis_scaling import instance_graph, timed

    identifier, available = instance_graph(80, 16)
    value, result = timed(lambda: identifier.identify("A", "Y", available))
    print(
        json.dumps(
            {
                "nodes": 80,
                "eligible_variables": 16,
                "elapsed_ms": value,
                "repetitions": 1,
                "warmups": 0,
                "status": str(result.reason),
                "sets": result.minimal_adjustment_sets,
            }
        ),
        flush=True,
    )
else:
    try:
        p = subprocess.run(
            [sys.executable, __file__, "--worker"],
            capture_output=True,
            text=True,
            check=True,
            timeout=65,
        )
        result = json.loads(p.stdout.strip())
    except subprocess.TimeoutExpired:
        result = {
            "nodes": 80,
            "eligible_variables": 16,
            "status": "single-call process censored after 65 s including imports",
            "repetitions": 0,
            "individual_elapsed_ms": None,
        }
    target = ROOT / "analysis/scaling/identification-80-16-single-pilot.json"
    with target.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result), flush=True)
