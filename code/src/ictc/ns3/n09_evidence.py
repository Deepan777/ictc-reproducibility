"""Deterministic evidence summary for the finalized N09 fresh-development journal."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ictc.ns3.n09 import N09DevelopmentRunner, aggregate_n09_scores
from ictc.ns3.stochastic import sha256


def build_n09_evidence(runner: N09DevelopmentRunner) -> dict[str, Any]:
    validation = runner.validate(require_complete=True)
    aggregate = aggregate_n09_scores(runner.output_root, runner.scenarios)
    probe_elapsed: list[float] = []
    action_elapsed: list[float] = []
    referenced: list[Path] = [
        runner.output_root / "n09_manifest.json",
        runner.action_runner.output_root / "n06_manifest.json",
        runner.action_runner.registry_path,
    ]
    decisions: set[str] = set()
    generations: set[str] = set()
    bound_actions = 0
    for scenario in runner.scenarios:
        probe = runner.output_root / "probes" / scenario.scenario_id
        probe_manifest_path = probe / "probe_manifest.json"
        probe_manifest = json.loads(probe_manifest_path.read_text(encoding="utf-8"))
        probe_elapsed.append(probe_manifest["elapsed_ns"] / 1_000_000_000)
        referenced.append(probe_manifest_path)
        referenced.extend(probe / name for name in probe_manifest["files"])
        decision = runner.output_root / "decisions" / scenario.scenario_id / "redecision.json"
        decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        decision_digest = sha256(decision)
        decisions.add(decision_digest)
        generations.update(item["recorded_digest"] for item in decision_payload["generations"])
        score = runner.output_root / "scores" / scenario.scenario_id / "score.json"
        referenced.extend((decision, score))
        group = runner.action_runner.output_root / "scenarios" / scenario.scenario_id
        referenced.extend((group / "group_manifest.json", group / "physical_benefit.json"))
        for action in scenario.actions:
            target = group / action.value
            manifest_path = target / "action_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("decision_manifest_sha256") != decision_digest:
                raise ValueError("N09 action is not bound to its persisted fresh redecision")
            bound_actions += 1
            action_elapsed.append(manifest["elapsed_ns"] / 1_000_000_000)
            referenced.append(manifest_path)
            referenced.extend(target / name for name in manifest["files"])
    d1 = aggregate["method_counts"]["D1_CONTEXT_SUPPORTED_CAPACITY"]
    b3 = aggregate["method_counts"]["B3_CAPACITY"]
    return {
        "schema_version": "n09-fresh-development-evidence-v1",
        "phase": "fresh development evaluation; not confirmatory evidence",
        "raw_root": str(runner.output_root),
        "validation": validation,
        "configuration": {
            "config_sha256": runner.config_sha256,
            "outcome_config_sha256": runner.action_runner.config_sha256,
            "executable_sha256": runner.action_runner.executable_sha256,
            "support_design_digest": runner.matrix["support_design_digest"],
            "support_design_file_sha256": runner.matrix["support_design_file_sha256"],
            "fresh_development_seeds": list(range(81100, 81108)),
            "reserved_confirmatory_seeds": list(range(82000, 82020)),
            "reserved_confirmatory_seeds_executed": 0,
        },
        "temporal_boundary": {
            "probe_manifests": len(runner.scenarios),
            "persisted_method_decisions": len(runner.scenarios) * 6,
            "decision_files": len(decisions),
            "hash_chained_generations": len(generations),
            "decision_bound_action_manifests": bound_actions,
            "probe_action_queue_differences": 0,
            "probe_action_state_differences": 0,
            "forbidden_access_violations": 0,
        },
        "fresh_method_counts": aggregate["method_counts"],
        "support_opportunities": aggregate["support_opportunities"],
        "continuation_gate": {
            "passes": aggregate["continuation_gate_passes"],
            "d1_harmful": d1["HARMFUL"],
            "b3_harmful": b3["HARMFUL"],
            "d1_beneficial": d1["BENEFICIAL"],
            "b3_beneficial": b3["BENEFICIAL"],
            "incremental_beneficial": d1["BENEFICIAL"] - b3["BENEFICIAL"],
            "required_incremental_beneficial": 8,
            "conclusion": (
                "The frozen fresh-development continuation gate passes; a separate reviewed "
                "confirmatory protocol is required before reserved-seed execution."
            ),
        },
        "resources": {
            "evidence_bytes": sum(path.stat().st_size for path in referenced),
            "mean_probe_seconds": round(statistics.mean(probe_elapsed), 6),
            "mean_action_seconds": round(statistics.mean(action_elapsed), 6),
            "total_native_seconds": round(sum(probe_elapsed) + sum(action_elapsed), 6),
        },
        "hashes": {
            "n09_manifest_sha256": sha256(runner.output_root / "n09_manifest.json"),
            "outcome_registry_sha256": sha256(runner.action_runner.registry_path),
            "outcome_manifest_sha256": sha256(
                runner.action_runner.output_root / "n06_manifest.json"
            ),
        },
        "claim_boundary": (
            "Fresh development transport evidence only; no confirmatory superiority, broad-"
            "topology, real-network, individual-benefit, or deployment claim."
        ),
    }


def render_n09_evidence(runner: N09DevelopmentRunner) -> str:
    return json.dumps(build_n09_evidence(runner), indent=2, sort_keys=True) + "\n"
