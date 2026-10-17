"""Strict expansion of the final M14 simulation robustness matrix."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ictc.simulation.models import (
    ArrivalMode,
    ArrivalProcess,
    PathConfig,
    PathTrafficConfig,
    SimulationConfig,
    TelemetryConfig,
)

EXPECTED_CONDITIONS = {
    "complete",
    "missing",
    "stale",
    "delayed",
    "clock_skew",
    "source_failure",
    "malicious_telemetry",
    "topology_change",
    "unseen_topology",
    "cross_traffic_shift",
    "contract_infeasibility",
}


@dataclass(frozen=True, slots=True)
class M14SimulationScenario:
    """One fully bound action-independent simulation scenario."""

    scenario_id: str
    condition_id: str
    contract_state: str
    topology_family_id: str
    topology_split: str
    traffic_id: str
    seed: int
    config: SimulationConfig


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain an object")
    return cast(dict[str, Any], value)


def _require_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _mapping(value: object, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def load_m14_simulation_matrix(path: Path) -> tuple[M14SimulationScenario, ...]:
    """Materialize all 4,320 frozen M14 paired simulation configurations."""
    matrix = _load_object(path)
    if matrix.get("schema_version") != "1.0.0":
        raise ValueError("unexpected M14 simulation matrix schema")
    seeds = matrix.get("seeds")
    if (
        not isinstance(seeds, list)
        or len(seeds) != 20
        or len(set(seeds)) != 20
        or any(not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("M14 simulation matrix requires exactly twenty distinct seeds")
    base = path.parents[2]
    topology_source = base / _require_string(
        matrix.get("topology_traffic_source"), "topology_traffic_source"
    )
    condition_source = base / _require_string(matrix.get("condition_map"), "condition_map")
    topology_raw = _load_object(topology_source)
    condition_raw = _load_object(condition_source)
    defaults = _mapping(topology_raw.get("defaults"), "topology defaults")
    families = topology_raw.get("topology_families")
    traffic_levels = topology_raw.get("traffic_levels")
    conditions = condition_raw.get("conditions")
    if not isinstance(families, list) or not isinstance(traffic_levels, list):
        raise ValueError("M14 topology source is incomplete")
    if not isinstance(conditions, list):
        raise ValueError("M14 condition map is incomplete")
    condition_by_id: dict[str, dict[str, Any]] = {}
    for value in conditions:
        if not isinstance(value, dict):
            raise ValueError("M14 condition must be an object")
        identifier = _require_string(value.get("condition_id"), "condition ID")
        if identifier in condition_by_id:
            raise ValueError("M14 condition ID is duplicated")
        condition_by_id[identifier] = cast(dict[str, Any], value)
    if set(condition_by_id) != EXPECTED_CONDITIONS:
        raise ValueError("M14 condition map does not exactly match the frozen protocol")
    scenarios: list[M14SimulationScenario] = []
    identities: set[tuple[str, str, str, int]] = set()
    for condition_id in sorted(condition_by_id):
        condition = condition_by_id[condition_id]
        split_scope = condition.get("topology_split")
        for family_value in families:
            family = _mapping(family_value, "topology family")
            split = _require_string(family.get("split"), "topology split")
            if split_scope is not None and split != split_scope:
                continue
            for traffic_value in traffic_levels:
                traffic = _mapping(traffic_value, "traffic level")
                for seed in seeds:
                    identifier = (
                        f"m14-{condition_id}-{family['family_id']}-{traffic['traffic_id']}-{seed}"
                    )
                    config = _simulation(identifier, seed, defaults, family, traffic, condition)
                    identity = (
                        condition_id,
                        str(family["family_id"]),
                        str(traffic["traffic_id"]),
                        seed,
                    )
                    if identity in identities:
                        raise ValueError("M14 simulation identity is duplicated")
                    identities.add(identity)
                    scenarios.append(
                        M14SimulationScenario(
                            scenario_id=identifier,
                            condition_id=condition_id,
                            contract_state=_require_string(
                                condition.get("contract_state"), "contract state"
                            ),
                            topology_family_id=_require_string(
                                family.get("family_id"), "family ID"
                            ),
                            topology_split=split,
                            traffic_id=_require_string(traffic.get("traffic_id"), "traffic ID"),
                            seed=seed,
                            config=config,
                        )
                    )
    expected_count = matrix.get("expected_paired_scenarios")
    if not isinstance(expected_count, int) or len(scenarios) != expected_count:
        raise ValueError("M14 simulation expansion count differs from its frozen declaration")
    return tuple(scenarios)


def _simulation(
    scenario_id: str,
    seed: int,
    defaults: dict[str, Any],
    family: dict[str, Any],
    traffic: dict[str, Any],
    condition: dict[str, Any],
) -> SimulationConfig:
    telemetry_values = _mapping(condition.get("telemetry"), "condition telemetry")
    simulation_values = _mapping(condition.get("simulation"), "condition simulation")
    current = "path-current"
    candidate = "path-candidate"
    mode = ArrivalMode(_require_string(traffic.get("arrival_mode"), "arrival mode"))
    telemetry = TelemetryConfig(
        sample_every_ticks=int(telemetry_values.get("sample_every_ticks", 1)),
        minimum_delay_ticks=int(telemetry_values.get("minimum_delay_ticks", 0)),
        maximum_delay_ticks=int(telemetry_values.get("maximum_delay_ticks", 0)),
        missing_probability=float(telemetry_values.get("missing_probability", 0.0)),
        source_failure_probability=float(telemetry_values.get("source_failure_probability", 0.0)),
        malicious_probability=float(telemetry_values.get("malicious_probability", 0.0)),
        malicious_queue_offset_packets=int(
            telemetry_values.get("malicious_queue_offset_packets", 0)
        ),
        maximum_clock_skew_ticks=int(telemetry_values.get("maximum_clock_skew_ticks", 0)),
        stale_after_ticks=int(telemetry_values.get("stale_after_ticks", 0)),
    )
    return SimulationConfig(
        scenario_id=scenario_id,
        seed=seed,
        duration_ticks=int(defaults["duration_ticks"]),
        tick_duration_ms=float(defaults["tick_duration_ms"]),
        deadline_ms=float(defaults["deadline_ms"]),
        current_path_id=current,
        candidate_path_id=candidate,
        paths=(
            PathConfig(
                path_id=current,
                service_packets_per_tick=int(family["current_service_packets_per_tick"]),
                queue_limit_packets=int(family["current_queue_limit_packets"]),
                propagation_delay_ms=float(family["current_propagation_delay_ms"]),
            ),
            PathConfig(
                path_id=candidate,
                service_packets_per_tick=int(family["candidate_service_packets_per_tick"]),
                queue_limit_packets=int(family["candidate_queue_limit_packets"]),
                propagation_delay_ms=float(family["candidate_propagation_delay_ms"]),
            ),
        ),
        foreground_process=ArrivalProcess(
            mode=mode, rate_packets_per_tick=float(traffic["foreground_rate_packets_per_tick"])
        ),
        cross_traffic=(
            PathTrafficConfig(
                path_id=current,
                process=ArrivalProcess(
                    mode=mode,
                    rate_packets_per_tick=float(traffic["current_cross_rate_packets_per_tick"]),
                ),
            ),
            PathTrafficConfig(
                path_id=candidate,
                process=ArrivalProcess(
                    mode=mode,
                    rate_packets_per_tick=float(traffic["candidate_cross_rate_packets_per_tick"]),
                ),
            ),
        ),
        telemetry=telemetry,
        capacity_change_tick=simulation_values.get("capacity_change_tick"),
        current_service_after_change=simulation_values.get("current_service_after_change"),
        candidate_service_after_change=simulation_values.get("candidate_service_after_change"),
        cross_traffic_shift_tick=simulation_values.get("cross_traffic_shift_tick"),
        cross_traffic_shift_multiplier=float(
            simulation_values.get("cross_traffic_shift_multiplier", 1.0)
        ),
    )


def matrix_digest(path: Path) -> str:
    """Return the immutable byte digest of the M14 simulation declaration."""
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()
