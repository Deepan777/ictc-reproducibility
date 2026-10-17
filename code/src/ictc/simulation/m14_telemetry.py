"""Controller-safe telemetry reconstruction for the frozen M14 simulations."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from ictc.simulation.environment import run_paired
from ictc.simulation.m14_matrix import M14SimulationScenario


@dataclass(frozen=True, slots=True)
class M14PreActionTelemetry:
    """The latest controller-visible samples available before the frozen decision deadline."""

    scenario_id: str
    condition_id: str
    topology_family_id: str
    topology_split: str
    traffic_id: str
    seed: int
    decision_tick: int
    config_sha256: str
    schedule_sha256: str
    contract_feasible: bool
    samples: tuple[dict[str, object], ...]


def reconstruct_pre_action_telemetry(
    scenario: M14SimulationScenario, *, decision_tick: int = 5, foreground_start_tick: int = 6
) -> M14PreActionTelemetry:
    """Derive only controller-visible samples available at the fixed pre-action deadline."""
    if decision_tick < 0:
        raise ValueError("M14 decision tick must be non-negative")
    if foreground_start_tick <= decision_tick:
        raise ValueError("controlled foreground action must follow the acquisition cutoff")
    # The schedule is action-independent.  We retain only its controller-facing projection and
    # deliberately do not expose the paired outcomes or oracle trace from this local execution.
    paired = run_paired(scenario.config, foreground_start_tick=foreground_start_tick)
    controller_trace = paired.baseline.controller_input()
    path_indices = {path.path_id: index for index, path in enumerate(scenario.config.paths)}
    latest: dict[str, dict[str, object]] = {}
    for record in controller_trace.available_at(decision_tick):
        latest[record.path_id] = {
            "sample_tick": record.sample_tick,
            "available_tick": record.available_tick,
            "observed_timestamp_tick": record.observed_timestamp_tick,
            "path_id": record.path_id,
            "queue_occupancy": record.queue_occupancy,
            "utilization": record.utilization,
            "sample_age_ticks": record.sample_age_ticks,
            "measurement_delay_ticks": record.measurement_delay_ticks,
            "clock_skew_ticks": record.clock_skew_ticks,
            "stale": record.stale,
            "missing": record.missing,
            "source_id": record.source_id,
            "provenance_digest": record.provenance_digest,
            # This is the result of the simulated pre-action attestation verifier,
            # not the hidden true queue value or a post-action outcome.
            "provenance_valid": not paired.schedule.ticks[record.sample_tick]
            .telemetry_impairments[path_indices[record.path_id]]
            .malicious,
        }
    return M14PreActionTelemetry(
        scenario.scenario_id,
        scenario.condition_id,
        scenario.topology_family_id,
        scenario.topology_split,
        scenario.traffic_id,
        scenario.seed,
        decision_tick,
        scenario.config.config_hash,
        paired.schedule.digest,
        scenario.contract_state != "INFEASIBLE",
        tuple(latest[path_id] for path_id in sorted(latest)),
    )


def write_telemetry_immutable(
    scenarios: tuple[M14SimulationScenario, ...], output: Path
) -> dict[str, object]:
    """Write the complete controller-safe M14 telemetry table exactly once."""
    if output.exists():
        raise FileExistsError(f"refusing to overwrite M14 controller telemetry: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(reconstruct_pre_action_telemetry(item)) for item in scenarios]
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=output.parent, delete=False
    ) as temporary:
        for row in rows:
            temporary.write(json.dumps(row, sort_keys=True) + "\n")
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output)
    return {
        "row_count": len(rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest().upper(),
        "outcome_or_oracle_fields_written": False,
    }
