"""Deterministic summary of checksum-validated N06 development evidence."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ictc.ns3.n06 import N06PilotRunner, load_registry_latest
from ictc.ns3.stochastic import PathAction, sha256


def build_n06_evidence(runner: N06PilotRunner) -> dict[str, Any]:
    validation = runner.validate(require_complete=True)
    label_counts: Counter[str] = Counter()
    reroute_event_counts: Counter[str] = Counter()
    pre_decision_digests: set[str] = set()
    background_by_cell: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    elapsed_seconds: list[float] = []
    evidence_files: list[Path] = [runner.registry_path, runner.output_root / "n06_manifest.json"]
    for scenario in runner.scenarios:
        group = runner.output_root / "scenarios" / scenario.scenario_id
        group_manifest = json.loads((group / "group_manifest.json").read_text(encoding="utf-8"))
        labels = json.loads((group / "physical_benefit.json").read_text(encoding="utf-8"))
        pre_decision_digests.add(group_manifest["pre_decision_sha256"])
        for action, label in labels["labels"].items():
            label_counts[f"{action}:{label}"] += 1
            if action == PathAction.REROUTE.value:
                reroute_event_counts[f"{scenario.event.value}:{label}"] += 1
        evidence_files.extend((group / "group_manifest.json", group / "physical_benefit.json"))
        for action in scenario.actions:
            target = group / action.value
            manifest = json.loads((target / "action_manifest.json").read_text(encoding="utf-8"))
            elapsed_seconds.append(manifest["elapsed_ns"] / 1_000_000_000)
            evidence_files.append(target / "action_manifest.json")
            evidence_files.extend(target / name for name in manifest["files"])
            if action is PathAction.KEEP:
                background_by_cell[
                    (
                        scenario.topology_family.value,
                        scenario.workload.value,
                        scenario.event.value,
                    )
                ].add(sha256(target / "raw" / "background.csv"))
    distinct_background = [len(values) for values in background_by_cell.values()]
    if len(background_by_cell) != 36 or min(distinct_background) < 2:
        raise ValueError("N06 development seeds do not demonstrate stochastic cell variation")
    registry_rows = runner.registry_path.read_text(encoding="utf-8").splitlines()[1:]
    latest = load_registry_latest(runner.registry_path)
    statuses = Counter(row["status"] for row in latest.values())
    total_bytes = sum(path.stat().st_size for path in evidence_files)
    total_seconds = sum(elapsed_seconds)
    return {
        "schema_version": "n06-development-evidence-v1",
        "phase": "development-only; not confirmatory evidence",
        "raw_root": str(runner.output_root),
        "configuration": {
            "config_sha256": runner.config_sha256,
            "matrix_sha256": sha256(runner.matrix_path),
            "source_sha256": runner.source_sha256,
            "executable_sha256": runner.executable_sha256,
            "development_seeds": [81000, 81001, 81002, 81003],
            "reserved_confirmatory_seeds": list(range(82000, 82020)),
            "reserved_confirmatory_seeds_executed": 0,
        },
        "validation": validation,
        "registry": {
            "rows": len(registry_rows),
            "latest_status_counts": dict(sorted(statuses.items())),
            "orphaned_started_attempts": len(registry_rows) - 2 * statuses["COMPLETE"],
            "sha256": sha256(runner.registry_path),
        },
        "stochasticity": {
            "topology_workload_event_cells": len(background_by_cell),
            "minimum_distinct_background_hashes_across_four_seeds": min(distinct_background),
            "maximum_distinct_background_hashes_across_four_seeds": max(distinct_background),
            "distinct_pre_decision_digests": len(pre_decision_digests),
            "paired_pre_decision_differences": 0,
        },
        "physical_benefit_labels": {
            "counts": dict(sorted(label_counts.items())),
            "reroute_by_event": dict(sorted(reroute_event_counts.items())),
            "semantic_boundary": (
                "Empirical paired physical benefit; not graph identification, "
                "causal-admissibility, or controller certification."
            ),
        },
        "resources": {
            "evidence_bytes": total_bytes,
            "mean_action_seconds": round(statistics.mean(elapsed_seconds), 6),
            "maximum_action_seconds": round(max(elapsed_seconds), 6),
            "total_action_seconds": round(total_seconds, 6),
            "projected_twenty_seed_bytes": total_bytes * 5,
            "projected_twenty_seed_action_seconds": round(total_seconds * 5, 3),
        },
        "root_manifest_sha256": sha256(runner.output_root / "n06_manifest.json"),
        "limitations": [
            "four development seeds only",
            "three small wired path families",
            "synthetic on/off traffic and declared disruptions",
            "no controller comparison or confirmatory inference",
            "no real-network, wireless, deployment, or broad-topology claim",
        ],
    }


def render_n06_evidence(runner: N06PilotRunner) -> str:
    return json.dumps(build_n06_evidence(runner), indent=2, sort_keys=True) + "\n"
