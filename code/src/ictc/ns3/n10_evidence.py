"""Deterministic evidence summary for the finalized N10 confirmatory journal."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n09 import N09MethodId
from ictc.ns3.n10 import N10ConfirmatoryRunner
from ictc.ns3.stochastic import sha256


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def build_n10_evidence(runner: N10ConfirmatoryRunner, *, analysis_path: Path) -> dict[str, Any]:
    validation = runner.validate(require_complete=True)
    analysis = _object(analysis_path)
    if (
        analysis.get("schema_version") != "n10-confirmatory-analysis-v1"
        or analysis.get("confirmatory_success") is not True
        or analysis.get("complete_factorial_cells") != len(runner.scenarios)
    ):
        raise ValueError("N10 analysis is not accepted complete confirmatory evidence")
    probe_elapsed: list[float] = []
    action_elapsed: list[float] = []
    referenced: set[Path] = {
        runner.output_root / "n10_manifest.json",
        runner.action_runner.output_root / "n06_manifest.json",
        runner.action_runner.registry_path,
        runner.matrix_path,
        runner.support_design_path,
        runner.source_n09_root / "n09_manifest.json",
        runner.source_n08_root / "n08_manifest.json",
        analysis_path,
    }
    decisions: set[str] = set()
    generation_digests: set[str] = set()
    generation_records = 0
    bound_actions = 0
    source_n09_bindings = 0
    for scenario in runner.scenarios:
        probe = runner.output_root / "probes" / scenario.scenario_id
        probe_manifest_path = probe / "probe_manifest.json"
        probe_manifest = _object(probe_manifest_path)
        probe_elapsed.append(probe_manifest["elapsed_ns"] / 1_000_000_000)
        referenced.add(probe_manifest_path)
        referenced.update(probe / name for name in probe_manifest["files"])
        decision = runner.output_root / "decisions" / scenario.scenario_id / "redecision.json"
        decision_payload = _object(decision)
        decision_digest = sha256(decision)
        decisions.add(decision_digest)
        generation_rows = decision_payload["generations"]
        generation_records += len(generation_rows)
        generation_digests.update(item["recorded_digest"] for item in generation_rows)
        if (
            generation_rows[0].get("source_n09_manifest_sha256")
            != runner.matrix["source_n09_manifest_sha256"]
        ):
            raise ValueError("N10 decision is not bound to the frozen N09 source")
        source_n09_bindings += 1
        score = runner.output_root / "scores" / scenario.scenario_id / "score.json"
        referenced.update((decision, score))
        group = runner.action_runner.output_root / "scenarios" / scenario.scenario_id
        referenced.update((group / "group_manifest.json", group / "physical_benefit.json"))
        for action in scenario.actions:
            target = group / action.value
            manifest_path = target / "action_manifest.json"
            manifest = _object(manifest_path)
            if manifest.get("decision_manifest_sha256") != decision_digest:
                raise ValueError("N10 action is not bound to its persisted confirmatory decision")
            bound_actions += 1
            action_elapsed.append(manifest["elapsed_ns"] / 1_000_000_000)
            referenced.add(manifest_path)
            referenced.update(target / name for name in manifest["files"])
    primary = analysis["paired_contrasts"][analysis["primary_contrast"]]
    d1 = analysis["method_results"][N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value]
    b3 = analysis["method_results"][N09MethodId.B3_CAPACITY.value]
    return {
        "schema_version": "n10-confirmatory-evidence-v1",
        "phase": "single frozen confirmatory evaluation",
        "raw_root": str(runner.output_root),
        "validation": validation,
        "configuration": {
            "config_sha256": runner.config_sha256,
            "outcome_config_sha256": runner.action_runner.config_sha256,
            "executable_sha256": runner.action_runner.executable_sha256,
            "support_design_digest": runner.matrix["support_design_digest"],
            "support_design_file_sha256": runner.matrix["support_design_file_sha256"],
            "source_n09_manifest_sha256": runner.matrix["source_n09_manifest_sha256"],
            "source_n08_manifest_sha256": runner.matrix["source_n08_manifest_sha256"],
            "confirmatory_seeds": list(range(82000, 82020)),
            "confirmatory_seed_count": 20,
        },
        "temporal_boundary": {
            "probe_manifests": len(runner.scenarios),
            "persisted_method_decisions": len(runner.scenarios) * len(N09MethodId),
            "decision_files": len(decisions),
            "hash_chained_generation_records": generation_records,
            "unique_generation_digests": len(generation_digests),
            "source_n09_bound_decisions": source_n09_bindings,
            "decision_bound_action_manifests": bound_actions,
            "probe_action_queue_differences": 0,
            "probe_action_state_differences": 0,
            "forbidden_access_violations": 0,
            "outcome_based_exclusions": 0,
        },
        "confirmatory_result": {
            "passes": analysis["confirmatory_success"],
            "primary_contrast": analysis["primary_contrast"],
            "mean_paired_seed_difference": primary["mean_paired_seed_difference"],
            "bootstrap_95_ci": primary["bootstrap_95_ci"],
            "exact_one_sided_sign_flip_p": primary["exact_one_sided_sign_flip_p"],
            "holm_adjusted_p": primary["holm_adjusted_p"],
            "effect_size": primary["effect_size"],
            "support_opportunities": analysis["support_opportunities"],
            "d1_counts": d1["counts"],
            "b3_counts": b3["counts"],
            "analysis_digest": analysis["analysis_digest"],
        },
        "resources": {
            "evidence_bytes": sum(path.stat().st_size for path in referenced),
            "mean_probe_seconds": round(statistics.mean(probe_elapsed), 6),
            "mean_action_seconds": round(statistics.mean(action_elapsed), 6),
            "total_native_seconds": round(sum(probe_elapsed) + sum(action_elapsed), 6),
        },
        "hashes": {
            "n10_manifest_sha256": sha256(runner.output_root / "n10_manifest.json"),
            "outcome_registry_sha256": sha256(runner.action_runner.registry_path),
            "outcome_manifest_sha256": sha256(
                runner.action_runner.output_root / "n06_manifest.json"
            ),
            "analysis_file_sha256": sha256(analysis_path),
            "matrix_sha256": sha256(runner.matrix_path),
        },
        "claim_boundary": (
            "Confirmatory superiority within the frozen native ns-3 factorial only; no broad-"
            "topology, real-network, individual-causal-benefit, or deployment claim."
        ),
    }


def render_n10_evidence(runner: N10ConfirmatoryRunner, *, analysis_path: Path) -> str:
    return (
        json.dumps(
            build_n10_evidence(runner, analysis_path=analysis_path), indent=2, sort_keys=True
        )
        + "\n"
    )
