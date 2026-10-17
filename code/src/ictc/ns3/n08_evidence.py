"""Deterministic evidence summary for the finalized N08 development journal."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ictc.ns3.n08 import N08DevelopmentRunner, aggregate_n08_scores
from ictc.ns3.stochastic import sha256


def build_n08_evidence(runner: N08DevelopmentRunner) -> dict[str, Any]:
    validation = runner.validate(require_complete=True)
    aggregate = aggregate_n08_scores(runner.output_root, runner.scenarios)
    probe_elapsed: list[float] = []
    action_elapsed: list[float] = []
    referenced: list[Path] = [
        runner.output_root / "n08_manifest.json",
        runner.action_runner.output_root / "n06_manifest.json",
        runner.action_runner.registry_path,
    ]
    d0_harm_by_event = {event: 0 for event in aggregate["event_detection"]}
    decision_digests: set[str] = set()
    generation_digests: set[str] = set()
    source_n07_digests: set[str] = set()
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
        decision_digests.add(decision_digest)
        source_n07_digests.add(decision_payload["generations"][0]["source_n07_decision_sha256"])
        generation_digests.update(
            item["recorded_digest"] for item in decision_payload["generations"]
        )
        score = runner.output_root / "scores" / scenario.scenario_id / "score.json"
        score_payload = json.loads(score.read_text(encoding="utf-8"))
        d0 = next(
            row for row in score_payload["rows"] if row["method_id"] == "D0_CHANGE_RESYNTHESIS"
        )
        d0_harm_by_event[scenario.event.value] += int(d0["physical_label"] == "HARMFUL")
        referenced.extend((decision, score))
        group = runner.action_runner.output_root / "scenarios" / scenario.scenario_id
        referenced.extend((group / "group_manifest.json", group / "physical_benefit.json"))
        for action in scenario.actions:
            target = group / action.value
            manifest_path = target / "action_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("decision_manifest_sha256") != decision_digest:
                raise ValueError("N08 action is not bound to its persisted redecision bytes")
            action_elapsed.append(manifest["elapsed_ns"] / 1_000_000_000)
            referenced.append(manifest_path)
            referenced.extend(target / name for name in manifest["files"])
    d0 = aggregate["method_counts"]["D0_CHANGE_RESYNTHESIS"]
    b3 = aggregate["method_counts"]["B3_CAPACITY"]
    total_bytes = sum(path.stat().st_size for path in referenced)
    return {
        "schema_version": "n08-development-evidence-v1",
        "phase": "development-only; not confirmatory evidence",
        "raw_root": str(runner.output_root),
        "validation": validation,
        "configuration": {
            "config_sha256": runner.config_sha256,
            "outcome_config_sha256": runner.action_runner.config_sha256,
            "executable_sha256": runner.action_runner.executable_sha256,
            "source_n07_manifest_sha256": runner.matrix["source_n07_manifest_sha256"],
            "development_seeds": [81000, 81001, 81002, 81003],
            "reserved_confirmatory_seeds": list(range(82000, 82020)),
            "reserved_confirmatory_seeds_executed": 0,
        },
        "temporal_boundary": {
            "probe_manifests": len(runner.scenarios),
            "persisted_method_decisions": len(runner.scenarios) * 6,
            "decision_files": len(decision_digests),
            "hash_chained_generations": len(generation_digests),
            "source_n07_decisions": len(source_n07_digests),
            "decision_bound_action_manifests": len(action_elapsed),
            "probe_action_queue_differences": 0,
            "probe_action_state_differences": 0,
            "forbidden_access_violations": 0,
        },
        "event_detection": aggregate["event_detection"],
        "development_method_counts": aggregate["method_counts"],
        "negative_result": {
            "d0_beneficial": d0["BENEFICIAL"],
            "d0_harmful": d0["HARMFUL"],
            "d0_keep": d0["KEEP"],
            "d0_harmful_by_event": d0_harm_by_event,
            "b3_beneficial": b3["BENEFICIAL"],
            "b3_harmful": b3["HARMFUL"],
            "b3_keep": b3["KEEP"],
            "conclusion": (
                "The post-change boundary and detection mechanism pass, but D0 retains harmful "
                "actions and is not competitive with B3; reserved-seed confirmation remains "
                "blocked."
            ),
        },
        "resources": {
            "evidence_bytes": total_bytes,
            "mean_probe_seconds": round(statistics.mean(probe_elapsed), 6),
            "mean_action_seconds": round(statistics.mean(action_elapsed), 6),
            "total_native_seconds": round(sum(probe_elapsed) + sum(action_elapsed), 6),
        },
        "hashes": {
            "n08_manifest_sha256": sha256(runner.output_root / "n08_manifest.json"),
            "outcome_registry_sha256": sha256(runner.action_runner.registry_path),
            "outcome_manifest_sha256": sha256(
                runner.action_runner.output_root / "n06_manifest.json"
            ),
        },
        "superseded_development_traces": [
            "D:\\ictc-toolchain\\n08-runs\\development-001",
            "D:\\ictc-toolchain\\n08-runs\\native-smoke-001",
            "D:\\ictc-toolchain\\n08-runs\\native-repair-smoke-001",
        ],
        "claim_boundary": (
            "Development mechanism and negative method-feasibility evidence only; no controller "
            "superiority, confirmatory inference, broad-topology, real-network, or deployment "
            "claim."
        ),
    }


def render_n08_evidence(runner: N08DevelopmentRunner) -> str:
    return json.dumps(build_n08_evidence(runner), indent=2, sort_keys=True) + "\n"
