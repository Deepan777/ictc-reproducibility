"""N09 fresh-seed contextual-support evaluation over the accepted N08 boundary."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from ictc.ns3.n06 import N06PilotRunner
from ictc.ns3.n07 import ACTION_BY_PATH, PATH_SPECS
from ictc.ns3.n08 import (
    ChangeProbeEvidence,
    N08DevelopmentRunner,
    _digest,
    _object,
    _state_digest,
    _write_new,
    read_change_probe,
)
from ictc.ns3.stochastic import (
    EventRegime,
    StochasticScenario,
    TopologyFamily,
    WorkloadRegime,
    read_action_evidence,
    sha256,
    validate_paired_pre_decision,
)


class N09MethodId(StrEnum):
    D1_CONTEXT_SUPPORTED_CAPACITY = "D1_CONTEXT_SUPPORTED_CAPACITY"
    B0_KEEP = "B0_KEEP"
    B1_REROUTE = "B1_REROUTE"
    B2_PROPAGATION = "B2_PROPAGATION"
    B3_CAPACITY = "B3_CAPACITY"
    B4_QUEUE = "B4_QUEUE"


def load_n09_matrix(path: Path) -> tuple[StochasticScenario, ...]:
    raw = _object(path)
    if (
        raw.get("schema_version") != "n09-fresh-support-ns3-v1"
        or raw.get("phase") != "fresh-development-evaluation"
    ):
        raise ValueError("unexpected N09 fresh-development matrix schema or phase")
    seeds = raw.get("seeds")
    reserved = raw.get("reserved_confirmatory_seeds")
    if seeds != list(range(81100, 81108)):
        raise ValueError("N09 fresh development seeds are not the frozen set")
    if reserved != list(range(82000, 82020)) or set(seeds) & set(reserved):
        raise ValueError("N09 confirmatory seed reservation is invalid")
    if raw.get("topology_families") != [item.value for item in TopologyFamily]:
        raise ValueError("N09 topology declaration is incomplete")
    if raw.get("workloads") != [item.value for item in WorkloadRegime]:
        raise ValueError("N09 workload declaration is incomplete")
    if raw.get("events") != [item.value for item in EventRegime]:
        raise ValueError("N09 event declaration is incomplete")
    controls = raw.get("controls")
    if not isinstance(controls, dict):
        raise ValueError("N09 controls are missing")
    try:
        duration = float(controls["duration_seconds"])
        decision = float(controls["decision_at_seconds"])
        event_at = float(controls["event_at_seconds"])
        interval = float(controls["interval_ms"])
        max_packets = int(controls["max_packets"])
        packet_size = int(controls["packet_size"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("N09 controls are invalid") from error
    if not 0.5 < event_at < decision < duration or min(interval, max_packets, packet_size) <= 0:
        raise ValueError("N09 requires a valid event-before-redecision boundary")
    scenarios = tuple(
        StochasticScenario(
            scenario_id=f"n09-{family.value}-{workload.value}-{event.value}-seed-{seed}",
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
        raise ValueError("N09 scenario count differs from its declaration")
    if sum(len(item.actions) for item in scenarios) != raw.get("expected_action_executions"):
        raise ValueError("N09 action count differs from its declaration")
    if raw.get("supported_contexts") != [
        "two-balanced|light|stable",
        "two-balanced|saturated|stable",
    ]:
        raise ValueError("N09 supported contexts differ from the frozen design")
    for field in (
        "support_design_digest",
        "support_design_file_sha256",
        "source_n08_manifest_sha256",
    ):
        if not isinstance(raw.get(field), str) or len(raw[field]) != 64:
            raise ValueError(f"N09 {field} is invalid")
    return scenarios


def _context(scenario: StochasticScenario) -> str:
    return "|".join((scenario.topology_family.value, scenario.workload.value, scenario.event.value))


def build_n09_generations(
    scenario: StochasticScenario,
    probe: ChangeProbeEvidence,
    probe_manifest_sha256: str,
    *,
    support_design_digest: str,
    support_design_file_sha256: str,
    source_n08_manifest_sha256: str,
    supported_contexts: tuple[str, ...],
) -> dict[str, Any]:
    digests = (
        probe_manifest_sha256,
        support_design_digest,
        support_design_file_sha256,
        source_n08_manifest_sha256,
    )
    if probe.scenario_id != scenario.scenario_id or any(len(item) != 64 for item in digests):
        raise ValueError("N09 probe or frozen-design binding is invalid")
    specs = PATH_SPECS[scenario.topology_family.value]
    contract_ready = (
        set(probe.latest_queues) == set(specs)
        and set(probe.latest_rate_bps) == set(specs)
        and set(probe.latest_link_up) == set(specs)
    )
    available = [path for path in probe.path_ids if probe.latest_link_up[path]]
    if not contract_ready or not available:
        raise ValueError("N09 observation contract has no capacity-safe action")
    b3_path = min(
        available,
        key=lambda path: (
            -probe.latest_rate_bps[path],
            0 if path == "current" else 1,
            path,
        ),
    )
    d1_path = b3_path
    initial_rates = {path: int(specs[path][0] * 1_000_000) for path in probe.path_ids}
    support_applied = (
        _context(scenario) in supported_contexts
        and probe.latest_link_up.get("current") is True
        and probe.latest_link_up.get("candidate") is True
        and probe.latest_rate_bps == initial_rates
        and b3_path == "current"
    )
    if support_applied:
        d1_path = "candidate"
    propagation = min(specs, key=lambda path: (specs[path][1], path))
    queue = min(available, key=lambda path: (probe.latest_queues[path], path))
    selected = {
        N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY: d1_path,
        N09MethodId.B0_KEEP: "current",
        N09MethodId.B1_REROUTE: "candidate",
        N09MethodId.B2_PROPAGATION: propagation,
        N09MethodId.B3_CAPACITY: b3_path,
        N09MethodId.B4_QUEUE: queue,
    }
    decisions = [
        {
            "method_id": method.value,
            "selected_path": path,
            "selected_action": ACTION_BY_PATH[path].value,
        }
        for method, path in selected.items()
    ]
    generation_zero_payload = {
        "generation": 0,
        "trigger": "FROZEN_SUPPORT_DESIGN",
        "previous_generation_digest": None,
        "support_design_digest": support_design_digest,
        "support_design_file_sha256": support_design_file_sha256,
        "source_n08_manifest_sha256": source_n08_manifest_sha256,
        "supported_contexts": list(supported_contexts),
    }
    generation_zero = {
        **generation_zero_payload,
        "recorded_digest": _digest(generation_zero_payload),
    }
    trigger = {
        EventRegime.STABLE: "NO_DECLARED_CHANGE",
        EventRegime.CANDIDATE_CAPACITY_DROP: "CAPABILITY_CHANGE",
        EventRegime.CURRENT_FAILURE: "TOPOLOGY_CHANGE",
        EventRegime.CANDIDATE_CROSS_SHIFT: "LOAD_SHIFT",
    }[scenario.event]
    generation_one_payload = {
        "generation": 1,
        "trigger": trigger,
        "previous_generation_digest": generation_zero["recorded_digest"],
        "probe_manifest_sha256": probe_manifest_sha256,
        "change_observation_digest": probe.observation_digest,
        "contract_ready": contract_ready,
        "event_detected": probe.event_detected,
        "context": _context(scenario),
        "support_applied": support_applied,
        "decisions": decisions,
    }
    generation_one = {
        **generation_one_payload,
        "recorded_digest": _digest(generation_one_payload),
    }
    return {
        "schema_version": "n09-redecision-ledger-v1",
        "scenario_id": scenario.scenario_id,
        "seed": scenario.seed,
        "generations": [generation_zero, generation_one],
        "current_generation": 1,
        "ledger_digest": _digest(
            {
                "scenario_id": scenario.scenario_id,
                "generation_digests": [
                    generation_zero["recorded_digest"],
                    generation_one["recorded_digest"],
                ],
            }
        ),
        "forbidden_inputs": {
            "fresh_foreground_outcome": False,
            "fresh_physical_benefit_label": False,
            "post_decision_telemetry": False,
            "oracle_state": False,
        },
    }


class N09DevelopmentRunner(N08DevelopmentRunner):
    """Reuse the accepted N08 probe/pairing engine with a frozen N09 decision ledger."""

    def __init__(
        self,
        *,
        repository: Path,
        output_root: Path,
        matrix_path: Path,
        executable: Path,
        runtime_directories: tuple[Path, ...],
        support_design_path: Path,
        source_n08_root: Path,
    ) -> None:
        self.repository = repository.resolve()
        self.output_root = output_root.resolve()
        self.matrix_path = matrix_path.resolve()
        self.support_design_path = support_design_path.resolve()
        self.source_n08_root = source_n08_root.resolve()
        self.matrix = _object(self.matrix_path)
        self.design = _object(self.support_design_path)
        self.scenarios = load_n09_matrix(self.matrix_path)
        source_manifest = self.source_n08_root / "n08_manifest.json"
        if sha256(self.support_design_path) != self.matrix["support_design_file_sha256"]:
            raise ValueError("N09 support-design file does not match the freeze")
        if self.design.get("frozen_design_digest") != self.matrix["support_design_digest"]:
            raise ValueError("N09 support-design digest does not match the freeze")
        if (
            not source_manifest.is_file()
            or sha256(source_manifest) != self.matrix["source_n08_manifest_sha256"]
        ):
            raise ValueError("N09 source N08 root manifest does not match the freeze")
        if self.design["frozen_design"]["supported_contexts"] != self.matrix["supported_contexts"]:
            raise ValueError("N09 supported contexts do not match the compiled design")
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
        self.supported_contexts = tuple(self.matrix["supported_contexts"])
        self.config_sha256 = _digest(
            {
                "action_config_sha256": self.action_runner.config_sha256,
                "n09_source_sha256": sha256(self.repository / "src" / "ictc" / "ns3" / "n09.py"),
                "support_design_file_sha256": sha256(self.support_design_path),
                "source_n08_manifest_sha256": sha256(source_manifest),
            }
        )

    def initialise(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "n09-fresh-development-journal-v1",
            "config_sha256": self.config_sha256,
            "scenario_groups": len(self.scenarios),
            "method_decisions": len(self.scenarios) * len(N09MethodId),
            "outcome_actions": sum(len(item.actions) for item in self.scenarios),
            "development_seeds": list(range(81100, 81108)),
            "reserved_confirmatory_seeds_executed": False,
        }
        manifest = self.output_root / "n09_manifest.json"
        if manifest.exists():
            if _object(manifest) != payload:
                raise ValueError("refusing to mix a different N09 freeze into this journal")
        else:
            _write_new(manifest, payload)
        self.action_runner.initialise()

    def _expected_decision(self, scenario: StochasticScenario, probe_path: Path) -> dict[str, Any]:
        return build_n09_generations(
            scenario,
            read_change_probe(probe_path / "raw", scenario),
            sha256(probe_path / "probe_manifest.json"),
            support_design_digest=self.matrix["support_design_digest"],
            support_design_file_sha256=self.matrix["support_design_file_sha256"],
            source_n08_manifest_sha256=self.matrix["source_n08_manifest_sha256"],
            supported_contexts=self.supported_contexts,
        )

    def _decide(self, scenario: StochasticScenario, probe: Path) -> Path:
        payload = self._expected_decision(scenario, probe)
        target = self.output_root / "decisions" / scenario.scenario_id / "redecision.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N09 persisted redecision differs from sealed inputs")
        else:
            _write_new(target, payload)
        return target

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
            "schema_version": "n09-fresh-development-score-v1",
            "scenario_id": scenario.scenario_id,
            "redecision_sha256": sha256(decision),
            "physical_benefit_sha256": sha256(label_path),
            "event": scenario.event.value,
            "support_applied": generation["support_applied"],
            "rows": rows,
            "development_only": True,
        }
        target = self.output_root / "scores" / scenario.scenario_id / "score.json"
        if target.exists():
            if _object(target) != payload:
                raise ValueError("N09 score does not match its sealed inputs")
        else:
            _write_new(target, payload)
        return target

    def validate(self, *, require_complete: bool = True) -> dict[str, int]:
        self.initialise()
        complete = 0
        actions = 0
        support_applied = 0
        for scenario in self.scenarios:
            probe_path = self.output_root / "probes" / scenario.scenario_id
            decision = self.output_root / "decisions" / scenario.scenario_id / "redecision.json"
            score = self.output_root / "scores" / scenario.scenario_id / "score.json"
            if not probe_path.exists() and not decision.exists() and not score.exists():
                continue
            if (
                not self._valid_probe(probe_path, scenario)
                or not decision.is_file()
                or not score.is_file()
            ):
                raise ValueError(f"N09 group is incomplete: {scenario.scenario_id}")
            expected = self._expected_decision(scenario, probe_path)
            if _object(decision) != expected:
                raise ValueError(f"N09 redecision seal is invalid: {scenario.scenario_id}")
            support_applied += int(expected["generations"][1]["support_applied"])
            targets = {
                action: self.action_runner.output_root
                / "scenarios"
                / scenario.scenario_id
                / action.value
                for action in scenario.actions
            }
            decision_digest = sha256(decision)
            if any(
                not self.action_runner._valid_action(target, scenario, action, decision_digest)
                for action, target in targets.items()
            ):
                raise ValueError(f"N09 outcome binding is invalid: {scenario.scenario_id}")
            probe = read_change_probe(probe_path / "raw", scenario)
            action_digest = validate_paired_pre_decision(
                {
                    action: read_action_evidence(target / "raw", scenario, action)
                    for action, target in targets.items()
                }
            )
            if probe.pre_decision_digest != action_digest:
                raise ValueError(f"N09 queue-trace binding is invalid: {scenario.scenario_id}")
            state_digests = {
                _state_digest(target / "raw", probe.path_ids) for target in targets.values()
            }
            if state_digests != {probe.state_digest}:
                raise ValueError(f"N09 state binding is invalid: {scenario.scenario_id}")
            self.action_runner._seal_group(scenario, targets)
            self._score(scenario, decision)
            complete += 1
            actions += len(targets)
        if require_complete and complete != len(self.scenarios):
            raise ValueError("N09 fresh development matrix is incomplete")
        return {
            "complete_scenario_groups": complete,
            "complete_method_decisions": complete * len(N09MethodId),
            "complete_outcome_actions": actions,
            "support_applied": support_applied,
            "planned_scenario_groups": len(self.scenarios),
        }


def aggregate_n09_scores(root: Path, scenarios: tuple[StochasticScenario, ...]) -> dict[str, Any]:
    labels = ("KEEP", "BENEFICIAL", "HARMFUL", "NO_MATERIAL_DIFFERENCE")
    counts = {method.value: {label: 0 for label in labels} for method in N09MethodId}
    support = {"applied": 0, "beneficial": 0, "harmful": 0, "other": 0}
    for scenario in scenarios:
        payload = _object(root / "scores" / scenario.scenario_id / "score.json")
        by_method = {row["method_id"]: row for row in payload["rows"]}
        for row in payload["rows"]:
            counts[row["method_id"]][row["physical_label"]] += 1
        if payload["support_applied"]:
            support["applied"] += 1
            label = by_method[N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value]["physical_label"]
            key = label.lower() if label in {"BENEFICIAL", "HARMFUL"} else "other"
            support[key] += 1
    d1 = counts[N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value]
    b3 = counts[N09MethodId.B3_CAPACITY.value]
    continuation = (
        d1["HARMFUL"] == 0
        and b3["HARMFUL"] == 0
        and d1["BENEFICIAL"] >= b3["BENEFICIAL"] + 8
        and support["applied"] == 16
    )
    return {
        "schema_version": "n09-fresh-development-aggregate-v1",
        "scenario_groups": len(scenarios),
        "method_counts": counts,
        "support_opportunities": support,
        "continuation_gate_passes": continuation,
        "development_only": True,
    }
