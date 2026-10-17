"""Compile the N09 contextual support repair from finalized N08 development scores."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n08 import load_n08_matrix
from ictc.ns3.stochastic import StochasticScenario, sha256


def _digest(payload: object) -> str:
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest().upper()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _context(scenario: StochasticScenario) -> str:
    return "|".join((scenario.topology_family.value, scenario.workload.value, scenario.event.value))


def compile_n09_support_design(
    *, n08_root: Path, matrix_path: Path, minimum_seeds: int = 4
) -> dict[str, Any]:
    if minimum_seeds != 4:
        raise ValueError("N09 design freezes exact four-seed N08 support")
    scenarios = load_n08_matrix(matrix_path)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    decisions: list[tuple[str, str]] = []
    for scenario in scenarios:
        score_path = n08_root / "scores" / scenario.scenario_id / "score.json"
        score = _object(score_path)
        if (
            score.get("scenario_id") != scenario.scenario_id
            or score.get("development_only") is not True
        ):
            raise ValueError("N09 design input is not finalized N08 development evidence")
        rows = score.get("rows")
        if not isinstance(rows, list):
            raise ValueError("N09 design score rows are invalid")
        by_method = {row["method_id"]: row for row in rows}
        b1 = by_method["B1_REROUTE"]
        b3 = by_method["B3_CAPACITY"]
        grouped[_context(scenario)].append(
            {
                "seed": str(scenario.seed),
                "b1_label": b1["physical_label"],
                "b3_action": b3["selected_action"],
            }
        )
        decisions.append((b3["selected_action"], b3["physical_label"]))
    supported = []
    for context, rows in sorted(grouped.items()):
        if len(rows) != minimum_seeds or len({row["seed"] for row in rows}) != minimum_seeds:
            raise ValueError("N09 context does not have exact independent-seed support")
        if all(row["b1_label"] == "BENEFICIAL" and row["b3_action"] == "keep" for row in rows):
            supported.append(context)
    retrospective = {"BENEFICIAL": 0, "HARMFUL": 0, "KEEP": 0, "NO_MATERIAL_DIFFERENCE": 0}
    for scenario, (b3_action, b3_label) in zip(scenarios, decisions, strict=True):
        score = _object(n08_root / "scores" / scenario.scenario_id / "score.json")
        b1 = next(row for row in score["rows"] if row["method_id"] == "B1_REROUTE")
        label = b1["physical_label"] if _context(scenario) in supported else b3_label
        if b3_action == "keep" and _context(scenario) not in supported:
            label = "KEEP"
        retrospective[label] += 1
    source_manifest = n08_root / "n08_manifest.json"
    source_registry = n08_root / "outcomes" / "EXPERIMENT_REGISTRY.csv"
    frozen_payload = {
        "algorithm": (
            "Use B3 capacity-safe action; add candidate reroute only in an exact topology/workload/"
            "event context where all four N08 seeds were beneficial and B3 kept."
        ),
        "minimum_independent_seeds": minimum_seeds,
        "supported_contexts": supported,
        "fallback": "B3_CAPACITY",
        "fresh_validation_seeds": list(range(81100, 81108)),
        "reserved_confirmatory_seeds": list(range(82000, 82020)),
    }
    return {
        "schema_version": "n09-context-support-design-v1",
        "phase": "retrospective N08 development design; prospective N09 fresh-seed evaluation",
        "source": {
            "n08_root": str(n08_root),
            "n08_manifest_sha256": sha256(source_manifest),
            "n08_registry_sha256": sha256(source_registry),
            "n08_matrix_sha256": sha256(matrix_path),
            "scenario_groups": len(scenarios),
        },
        "frozen_design": frozen_payload,
        "frozen_design_digest": _digest(frozen_payload),
        "retrospective_n08_counts": retrospective,
        "selection_warning": (
            "Contexts were selected using N08 outcomes and are not confirmatory evidence; only the "
            "predeclared fresh N09 seeds may evaluate transport of this rule."
        ),
    }


def render_n09_support_design(*, n08_root: Path, matrix_path: Path) -> str:
    return (
        json.dumps(
            compile_n09_support_design(n08_root=n08_root, matrix_path=matrix_path),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
