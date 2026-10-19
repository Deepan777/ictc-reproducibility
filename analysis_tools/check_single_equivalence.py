"""Prespecified 10,000-DAG check against exhaustive back-door enumeration."""
from __future__ import annotations
from itertools import combinations
import json
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code/src"))
from ictc.causal.graph import CausalDAG
from ictc.causal.identification import BackdoorIdentifier


def main():
    output = REPO / "codex_outputs/idfix/equivalence_check.json"
    if output.exists():
        raise FileExistsError(output)
    rng = np.random.default_rng(20261019)
    counts = {"graphs": 0, "identified_both": 0, "not_identified_both": 0, "existence_disagreements": 0,
              "invalid_sets": 0, "nonminimal_sets": 0, "with_context": 0, "with_unobserved": 0}
    examples = []
    started = time.perf_counter()
    for trial in range(10000):
        n = int(rng.integers(6, 31))
        density = float(rng.uniform(.1, .5))
        nodes = tuple(f"V{i:02}" for i in range(n))
        order = list(rng.permutation(nodes))
        edges = [(order[i], order[j]) for i in range(n) for j in range(i+1, n) if rng.random() < density]
        a, y = (str(s) for s in rng.choice(nodes, 2, replace=False))
        chosen = set(str(s) for s in rng.choice([s for s in nodes if s not in (a, y)], min(n-2, int(rng.integers(0, 11))), replace=False))
        available = chosen | {a, y}
        observed = available | {node for node in nodes if rng.random() < .5}
        graph = CausalDAG(nodes, edges, observed)
        eligible_context = sorted(chosen - graph.descendants(a))
        context = set(str(s) for s in rng.choice(eligible_context, min(len(eligible_context), int(rng.integers(0, 4))), replace=False))
        identifier = BackdoorIdentifier(graph)
        candidates = tuple(s for s in identifier.eligible_adjustment_variables(a, y, available) if s not in context)
        valid = tuple(candidate for size in range(len(candidates)+1) for candidate in combinations(candidates, size)
                      if identifier.is_valid_adjustment_set(a, y, (*context, *candidate)))
        result = identifier.identify(a, y, available, mode="single", conditioning_variables=context, path_diagnostics=False)
        selected = result.minimal_adjustment_sets[0] if result.identified else None
        mismatch = result.identified != bool(valid)
        invalid = selected is not None and selected not in valid
        nonminimal = selected is not None and selected not in identifier.minimal_adjustment_sets(valid)
        counts["graphs"] += 1
        counts["identified_both"] += int(bool(valid) and result.identified)
        counts["not_identified_both"] += int(not valid and not result.identified)
        counts["existence_disagreements"] += int(mismatch)
        counts["invalid_sets"] += int(invalid)
        counts["nonminimal_sets"] += int(nonminimal)
        counts["with_context"] += int(bool(context))
        counts["with_unobserved"] += int(len(observed) < n)
        if (mismatch or invalid or nonminimal) and len(examples) < 20:
            examples.append({"trial": trial, "edges": edges, "treatment": a, "outcome": y,
                             "available": sorted(available), "context": sorted(context), "selected": selected, "valid_sets": valid})
        if (trial+1) % 1000 == 0:
            print(f"Equivalence {trial+1}/10000; disagreements={counts['existence_disagreements']}; invalid={counts['invalid_sets']}; nonminimal={counts['nonminimal_sets']}", flush=True)
    passed = not any(counts[key] for key in ("existence_disagreements", "invalid_sets", "nonminimal_sets"))
    payload = {**counts, "seed": 20261019, "node_range": [6, 30], "edge_density_range": [.1, .5],
               "maximum_optional_available_variables": 10, "wall_seconds": time.perf_counter()-started,
               "passed": passed, "single_mode_permitted_for_benchmark": passed, "counterexamples": examples}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
