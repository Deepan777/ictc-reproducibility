"""N10 immutable confirmatory runner over the frozen N09 controller."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ictc.ns3.n06 import N06PilotRunner
from ictc.ns3.n08 import ChangeProbeEvidence, _digest, _object, _write_new, read_change_probe
from ictc.ns3.n09 import N09DevelopmentRunner, N09MethodId, build_n09_generations
from ictc.ns3.stochastic import (
    EventRegime,
    StochasticScenario,
    TopologyFamily,
    WorkloadRegime,
    sha256,
)


def load_n10_matrix(path: Path) -> tuple[StochasticScenario, ...]:
    raw = _object(path)
    if (
        raw.get("schema_version") != "n10-confirmatory-support-ns3-v1"
        or raw.get("phase") != "confirmatory"
    ):
        raise ValueError("unexpected N10 confirmatory matrix schema or phase")
    seeds = raw.get("seeds")
    if seeds != list(range(82000, 82020)):
        raise ValueError("N10 seeds are not the authorized reserved set")
    if raw.get("topology_families") != [item.value for item in TopologyFamily]:
        raise ValueError("N10 topology declaration is incomplete")
    if raw.get("workloads") != [item.value for item in WorkloadRegime]:
        raise ValueError("N10 workload declaration is incomplete")
    if raw.get("events") != [item.value for item in EventRegime]:
        raise ValueError("N10 event declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N10 controls are missing")
    try:
        duration = float(controls["duration_seconds"])
        decision = float(controls["decision_at_seconds"])
        event_at = float(controls["event_at_seconds"])
        interval = float(controls["interval_ms"])
        max_packets = int(controls["max_packets"])
        packet_size = int(controls["packet_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N10 controls are invalid") from error
    if not 0.5 < event_at < decision < duration or min(interval, max_packets, packet_size) <= 0:
        raise ValueError("N10 requires a valid event-before-redecision boundary")
    scenarios = tuple(
        StochasticScenario(
            scenario_id=f"n10-{family.value}-{workload.value}-{event.value}-seed-{seed}",
            topology_family=family,
            workload=workload,
            event=event,
            seed=seed,
            run=1,
            duration_seconds=duration,
            decision_at_seconds=decision,
            event_at_seconds=event_at,
            interval_ms=interval,
            max_packets=max_packets,
            packet_size=packet_size,
        )
        for family in TopologyFamily
        for workload in WorkloadRegime
        for event in EventRegime
        for seed in seeds
    )
    if len(scenarios) != raw.get("expected_scenario_groups"):
        raise ValueError("N10 scenario count differs from its declaration")
    if sum(len(item.actions) for item in scenarios) != raw.get("expected_action_executions"):
        raise ValueError("N10 action count differs from its declaration")
    if len(scenarios) * len(N09MethodId) != raw.get("expected_method_decisions"):
        raise ValueError("N10 method-decision count differs from its declaration")
    if raw.get("supported_contexts") != [
        "two-balanced|light|stable",
        "two-balanced|saturated|stable",
    ]:
        raise ValueError("N10 supported contexts differ from the frozen design")
    analysis = raw.get("analysis")
    expected_analysis = {
        "utility_scores": {
            "BENEFICIAL": 1,
            "HARMFUL": -1,
            "KEEP": 0,
            "NO_MATERIAL_DIFFERENCE": 0,
        },
        "independent_block": "seed",
        "confidence_level": 0.95,
        "bootstrap_resamples": 10000,
        "bootstrap_seed": 20260715,
        "paired_test": "exact one-sided seed-block sign-flip",
        "multiplicity": "Holm across D1 versus B0-B4",
        "family_alpha": 0.05,
        "minimum_supported_beneficial": 32,
        "required_d1_harmful": 0,
    }
    if analysis != expected_analysis:
        raise ValueError("N10 analysis freeze is invalid")
    for field in (
        "support_design_digest",
        "support_design_file_sha256",
        "source_n09_manifest_sha256",
        "source_n08_manifest_sha256",
    ):
        if not isinstance(raw.get(field), str) or len(raw[field]) != 64:
            raise ValueError(f"N10 {field} is invalid")
    return scenarios


def build_n10_generations(
    scenario: StochasticScenario,
    probe: ChangeProbeEvidence,
    probe_manifest_sha256: str,
    *,
    support_design_digest: str,
    support_design_file_sha256: str,
    source_n09_manifest_sha256: str,
    source_n08_manifest_sha256: str,
    supported_contexts: tuple[str, ...],
) -> dict[str, Any]:
    payload = build_n09_generations(
        scenario,
        probe,
        probe_manifest_sha256,
        support_design_digest=support_design_digest,
        support_design_file_sha256=support_design_file_sha256,
        source_n08_manifest_sha256=source_n08_manifest_sha256,
        supported_contexts=supported_contexts,
    )
    if len(source_n09_manifest_sha256) != 64:
        raise ValueError("N10 source N09 manifest binding is invalid")
    generation_zero = dict(payload["generations"][0])
    generation_zero.pop("recorded_digest")
    generation_zero["source_n09_manifest_sha256"] = source_n09_manifest_sha256
    generation_zero["recorded_digest"] = _digest(generation_zero)
    generation_one = dict(payload["generations"][1])
    generation_one.pop("recorded_digest")
    generation_one["previous_generation_digest"] = generation_zero["recorded_digest"]
    generation_one["recorded_digest"] = _digest(generation_one)
    return {
        **payload,
        "schema_version": "n10-confirmatory-redecision-ledger-v1",
        "generations": [generation_zero, generation_one],
        "ledger_digest": _digest(
            {
                "scenario_id": scenario.scenario_id,
                "generation_digests": [
                    generation_zero["recorded_digest"],
                    generation_one["recorded_digest"],
                ],
            }
        ),
    }


class N10ConfirmatoryRunner(N09DevelopmentRunner):
    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
        support_design_path: Path,
        source_n09_root: Path,
        source_n08_root: Path,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.support_design_path = support_design_path.resolve()
        self.source_n09_root = source_n09_root.resolve()
        self.source_n08_root = source_n08_root.resolve()
        self.matrix = _object(self.matrix_path)
        self.design = _object(self.support_design_path)
        self.scenarios = load_n10_matrix(self.matrix_path)
        n09_manifest = self.source_n09_root / "n09_manifest.json"
        n08_manifest = self.source_n08_root / "n08_manifest.json"
        if sha256(self.support_design_path) != self.matrix["support_design_file_sha256"]:
            raise ValueError("N10 support-design file does not match the freeze")
        if self.design.get("frozen_design_digest") != self.matrix["support_design_digest"]:
            raise ValueError("N10 support-design digest does not match the freeze")
        if (
            not n09_manifest.is_file()
            or sha256(n09_manifest) != self.matrix["source_n09_manifest_sha256"]
        ):
            raise ValueError("N10 source N09 manifest does not match the freeze")
        if (
            not n08_manifest.is_file()
            or sha256(n08_manifest) != self.matrix["source_n08_manifest_sha256"]
        ):
            raise ValueError("N10 source N08 manifest does not match the freeze")
        self.supported_contexts = tuple(self.matrix["supported_contexts"])
        self.action_runner = N06PilotRunner(
            repository=self.repository,
            output_root=self.output_root / "outcomes",
            matrix_path=self.matrix_path,
            executable=executable,
            runtime_directories=runtime_directories,
            scenarios=self.scenarios,
        )
        self.executable = self.action_runner.executable
        self.runtime_directories = self.action_runner.runtime_directories
        self.config_sha256 = _digest(
            {
                "action_config_sha256": self.action_runner.config_sha256,
                "n10_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n10.py"),
                "n09_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n09.py"),
                "support_design_file_sha256": sha256(self.support_design_path),
                "source_n09_manifest_sha256": sha256(n09_manifest),
                "source_n08_manifest_sha256": sha256(n08_manifest),
            }
        )

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "n10-confirmatory-journal-v1",
            "config_sha256": self.config_sha256,
            "scenario_groups": len(self.scenarios),
            "method_decisions": len(self.scenarios) * len(N09MethodId),
            "outcome_actions": sum(len(item.actions) for item in self.scenarios),
            "confirmatory_seeds": list(range(82000, 82020)),
            "single_execution_authorization": True,
        }
        manifest = self.output_root / "n10_manifest.json"
        if manifest.exists():
            if _object(manifest) != payload:
                raise ValueError("refusing to mix a different N10 freeze into this journal")
        else:
            _write_new(manifest, payload)
        self.action_runner.initialise()

    def _expected_decision(self, scenario: StochasticScenario, probe_path: Path) -> dict[str, Any]:
        return build_n10_generations(
            scenario,
            read_change_probe(probe_path / "raw", scenario),
            sha256(probe_path / "probe_manifest.json"),
            support_design_digest=self.matrix["support_design_digest"],
            support_design_file_sha256=self.matrix["support_design_file_sha256"],
            source_n09_manifest_sha256=self.matrix["source_n09_manifest_sha256"],
            source_n08_manifest_sha256=self.matrix["source_n08_manifest_sha256"],
            supported_contexts=self.supported_contexts,
        )

    def _score(self, scenario: StochasticScenario, decision: Path) -> Path:
        label_path = (
            self.action_runner.output_root
            / "scenarios"
            / scenario.scenario_id
            / "physical_benefit.json"
        )
        labels = _object(label_path)["labels"]
        generation = _object(decision)["generations"][1]
        rows = []
        for item in generation["decisions"]:
            action = item["selected_action"]
            rows.append({**item, "physical_label": "KEEP" if action == "keep" else labels[action]})
        payload = {
            "schema_version": "n10-confirmatory-score-v1",
            "scenario_id": scenario.scenario_id,
            "redecision_sha256": sha256(decision),
            "physical_benefit_sha256": sha256(label_path),
            "event": scenario.event.value,
            "support_applied": generation["support_applied"],
            "rows": rows,
            "confirmatory": True,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N10 score does not match its sealed inputs")
        else:
            _write_new(target, payload)
        return target

    def run(self, *, limit: int | None = None) -> dict[str, int]:
        result = super().run(limit=limit)
        selected = self.scenarios if limit is None else self.scenarios[:limit]
        return {
            "scenario_groups": result["scenario_groups"],
            "method_decisions": result["method_decisions"],
            "outcome_actions": result["outcome_actions"],
            "confirmatory_seeds_executed": len({scenario.seed for scenario in selected}),
        }
