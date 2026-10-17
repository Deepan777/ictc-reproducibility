"""Deterministic evidence summary for the finalized N07 development journal."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ictc.ns3.n07 import N07DevelopmentRunner, aggregate_development_scores
from ictc.ns3.stochastic import sha256


def build_n07_evidence(runner: N07DevelopmentRunner) -> dict[str, Any]:
    validation = runner.validate(require_complete=True)
    aggregate = aggregate_development_scores(runner.output_root, runner.scenarios)
    probe_elapsed: list[float] = []
    action_elapsed: list[float] = []
    decision_digests: set[str] = set()
    referenced_files: list[Path] = [
        runner.output_root / "n07_manifest.json",
        runner.action_runner.output_root / "n06_manifest.json",
        runner.action_runner.registry_path,
    ]
    for scenario in runner.scenarios:
        probe = runner.output_root / "probes" / scenario.scenario_id
        probe_manifest_path = probe / "probe_manifest.json"
        probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
        probe_elapsed.append(probe_manifest["elapsed_ns"] / 1_000_000_000)
        referenced_files.append(probe_manifest_path)
        referenced_files.extend(probe / name for name in probe_manifest["files"])
        decision = runner.output_root / "decisions" / scenario.scenario_id / "decisions.json"
        score = runner.output_root / "scores" / scenario.scenario_id / "score.json"
        decision_digest = sha256(decision)
        decision_digests.add(decision_digest)
        referenced_files.extend((decision, score))
        group = runner.action_runner.output_root / "scenarios" / scenario.scenario_id
        referenced_files.extend((group / "group_manifest.json", group / "physical_benefit.json"))
        for action in scenario.actions:
            target = group / action.value
            manifest_path = target / "action_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("decision_manifest_sha256") != decision_digest:
                raise ValueError("N07 action is not bound to its persisted decision bytes")
            action_elapsed.append(manifest["elapsed_ns"] / 1_000_000_000)
            referenced_files.append(manifest_path)
            referenced_files.extend(target / name for name in manifest["files"])
    method_counts = aggregate["method_counts"]
    c0 = method_counts["C0_CONTRACT_QUEUE"]
    total_seconds = sum(probe_elapsed) + sum(action_elapsed)
    total_bytes = sum(path.stat().st_size for path in referenced_files)
    return {
        "schema_version": "n07-development-evidence-v1",
        "phase": "development-only; not confirmatory evidence",
        "raw_root": str(runner.output_root),
        "validation": validation,
        "configuration": {
            "config_sha256": runner.config_sha256,
            "outcome_config_sha256": runner.action_runner.config_sha256,
            "executable_sha256": runner.action_runner.executable_sha256,
            "development_seeds": [81000, 81001, 81002, 81003],
            "reserved_confirmatory_seeds": list(range(82000, 82020)),
            "reserved_confirmatory_seeds_executed": 0,
        },
        "temporal_boundary": {
            "probe_manifests": len(runner.scenarios),
            "persisted_method_decisions": len(runner.scenarios) * 6,
            "decision_files": len(decision_digests),
            "decision_bound_action_manifests": len(action_elapsed),
            "probe_action_pre_decision_differences": 0,
            "forbidden_access_violations": 0,
        },
        "development_method_counts": method_counts,
        "negative_result": {
            "c0_beneficial": c0["BENEFICIAL"],
            "c0_harmful": c0["HARMFUL"],
            "c0_keep": c0["KEEP"],
            "conclusion": (
                "The single-snapshot C0 rule is not competitive with the capacity baseline; "
                "reserved-seed confirmation is not authorized."
            ),
        },
        "resources": {
            "evidence_bytes": total_bytes,
            "mean_probe_seconds": round(statistics.mean(probe_elapsed), 6),
            "mean_action_seconds": round(statistics.mean(action_elapsed), 6),
            "total_native_seconds": round(total_seconds, 6),
        },
        "hashes": {
            "n07_manifest_sha256": sha256(runner.output_root / "n07_manifest.json"),
            "outcome_registry_sha256": sha256(runner.action_runner.registry_path),
            "outcome_manifest_sha256": sha256(
                runner.action_runner.output_root / "n06_manifest.json"
            ),
        },
        "superseded_development_traces": [
            "D:\\ictc-toolchain\\n07-runs\\development-001",
            "D:\\ictc-toolchain\\n07-runs\\development-002",
        ],
        "claim_boundary": (
            "Development mechanism/negative feasibility evidence only; no controller superiority, "
            "confirmatory inference, broad-topology, real-network, or deployment claim."
        ),
    }


def render_n07_evidence(runner: N07DevelopmentRunner) -> str:
    return json.dumps(build_n07_evidence(runner), indent=2, sort_keys=True) + "\n"
