"""Frozen unchanged-policy N13 transport validation on external backbone topologies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from ictc.ns3.n12 import (
    ACTIONS,
    EVENTS,
    MATRICES,
    METHODS,
    TRAFFIC_PROFILES,
    N12Scenario,
    read_object,
    sha256,
)
from ictc.ns3.n13 import N13Runner
from ictc.ns3.n13 import build_decision as build_n13_decision

TRANSPORT_SEEDS = tuple(range(90000, 90020))
TOPOLOGIES = ("geant-2012-03", "garr-2011-09")


@dataclass(frozen=True, slots=True)
class N13TransportScenario(N12Scenario):
    topology: str
    topology_config: str

    def command(self, action: str, output: Path, *, probe: bool) -> list[str]:
        return [
            f"--scenarioId={self.scenario_id}",
            f"--topologyConfig={self.topology_config}",
            f"--traffic={self.traffic}",
            f"--event={self.event}",
            f"--matrix={self.matrix}",
            f"--action={action}",
            f"--output={output}",
            f"--seed={self.seed}",
            f"--run={self.run}",
            f"--duration={self.duration_seconds}",
            f"--decisionAt={self.decision_at_seconds}",
            f"--eventAt={self.event_at_seconds}",
            f"--probeOnly={1 if probe else 0}",
        ]


def load_matrix(path: Path) -> tuple[dict[str, Any], tuple[N13TransportScenario, ...]]:
    raw = read_object(path)
    if raw.get("schema_version") != "n13-external-transport-matrix-v1":
        raise ValueError("unexpected N13 transport matrix schema")
    if raw.get("phase") != "external-confirmatory" or raw.get("seeds") != list(TRANSPORT_SEEDS):
        raise ValueError("N13 transport phase or seeds differ from the frozen declaration")
    if raw.get("traffic_profiles") != list(TRAFFIC_PROFILES):
        raise ValueError("N13 transport traffic profiles are incomplete")
    if raw.get("events") != list(EVENTS) or raw.get("od_matrices") != list(MATRICES):
        raise ValueError("N13 transport event or OD declaration is incomplete")
    if raw.get("actions") != list(ACTIONS) or raw.get("methods") != list(METHODS):
        raise ValueError("N13 transport action or method declaration is incomplete")
    topology_values = raw.get("topologies")
    if not isinstance(topology_values, list) or [
        item.get("topology_id") for item in topology_values
    ] != list(TOPOLOGIES):
        raise ValueError("N13 transport topology declaration is incomplete")
    topology_paths: dict[str, Path] = {}
    for value in topology_values:
        relative = Path(str(value["native_relative_path"]))
        repository_config = (path.parent / relative).resolve()
        execution_config = Path(str(value["native_execution_path"])).resolve()
        config = execution_config if execution_config.is_file() else repository_config
        derived = (path.parent / Path(str(value["derived_json_relative_path"]))).resolve()
        if (
            sha256(repository_config) != value["native_sha256"]
            or sha256(config) != value["native_sha256"]
            or sha256(derived) != value["derived_json_sha256"]
        ):
            raise ValueError("N13 transport topology input differs from its freeze")
        topology_paths[str(value["topology_id"])] = config
    controls = cast(dict[str, Any], raw.get("controls"))
    duration = float(controls["duration_seconds"])
    decision = float(controls["decision_at_seconds"])
    event_at = float(controls["event_at_seconds"])
    if not 0.5 < event_at + 0.5 < decision < duration - 1.0:
        raise ValueError("N13 transport timing boundary is invalid")
    scenarios = tuple(
        N13TransportScenario(
            scenario_id=f"n13-transport-{topology}-{traffic}-{event}-od{matrix}-seed-{seed}",
            phase="external-confirmatory",
            topology=topology,
            topology_config=str(topology_paths[topology]),
            traffic=traffic,
            event=event,
            matrix=matrix,
            seed=seed,
            run=1,
            duration_seconds=duration,
            decision_at_seconds=decision,
            event_at_seconds=event_at,
        )
        for topology in TOPOLOGIES
        for traffic in TRAFFIC_PROFILES
        for event in EVENTS
        for matrix in MATRICES
        for seed in TRANSPORT_SEEDS
    )
    if len(scenarios) != raw.get("expected_scenario_groups"):
        raise ValueError("N13 transport scenario count differs from its declaration")
    if len(scenarios) * 4 != raw.get("expected_native_runs"):
        raise ValueError("N13 transport native run count differs from its declaration")
    if len(scenarios) * len(METHODS) != raw.get("expected_controller_decisions"):
        raise ValueError("N13 transport decision count differs from its declaration")
    return raw, scenarios


def build_decision(
    scenario: N12Scenario, probe_raw: Path, thresholds: dict[str, Any]
) -> dict[str, Any]:
    payload = build_n13_decision(cast(Any, scenario), probe_raw, thresholds)
    payload["schema_version"] = "n13-external-transport-controller-decisions-v1"
    payload["scenario"] = asdict(scenario)
    payload["policy_transport_rule"] = (
        "M005 is applied without coefficient, threshold, feature, endpoint, or utility retuning."
    )
    return payload


class N13TransportRunner(N13Runner):
    """Append-only journal for the prospective external transport confirmation."""

    study_id = "N13_TRANSPORT"
    journal_prefix = "n13_transport"
    journal_schema = "n13-external-transport-immutable-journal-v1"
    native_run_schema = "n13-external-transport-native-run-v1"
    resource_schema = "n13-external-transport-resource-metrics-v1"
    paired_score_schema = "n13-external-transport-paired-score-v1"
    group_schema = "n13-external-transport-group-manifest-v1"
    matrix_loader = staticmethod(load_matrix)
    decision_builder = staticmethod(build_decision)
    native_source_relative = Path("ns3/scratch/ictc_n13_transport.cc")
    input_manifest_relative = Path("configs/experiments/n13_transport_derived/manifest.json")
    confirmatory_phases = ("external-confirmatory",)
