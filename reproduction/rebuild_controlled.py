"""Rebuild the controlled diagnostic with a shared pre-action background-only prelude.

Only training and held-out rows used by the controlled comparison are rebuilt.
No ns-3 execution, model selection, threshold search or figure generation occurs.
Existing output directories are never overwritten. Pair schedules remain unchanged.
"""

# ruff: noqa: E402 -- explicit checkout-local imports after portable path bootstrap.
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/src")]
from ictc.simulation.environment import SimulationRun, run_paired
from ictc.simulation.m14_matrix import load_m14_simulation_matrix
from ictc.simulation.m14_telemetry import reconstruct_pre_action_telemetry


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(matrix: Path, split: str, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    scenarios = tuple(
        s for s in load_m14_simulation_matrix(matrix) if s.topology_split == split
    )
    estimate = len(scenarios) * 20000
    if shutil.disk_usage(output.parent).free < 5 * 1024**3 + 1.3 * estimate:
        raise RuntimeError("storage margin is insufficient")
    output.mkdir()
    records = []
    manifest: dict[str, Any] = {
        "matrix_sha256": sha(matrix),
        "split": split,
        "foreground_start_tick": 6,
        "acquisition_cutoff_tick": 5,
        "estimation_uncertainty": "seed-cluster CR1 approximate normal",
        "post_review_amendment": True,
        "records": {},
    }
    for index, item in enumerate(scenarios, 1):
        pair = run_paired(item.config, foreground_start_tick=6)

        def arm(run: SimulationRun) -> dict[str, float]:
            return {
                "latency_ms": run.outcome.latency_ms,
                "loss_ratio": run.outcome.packet_loss_ratio,
                "jitter_ms": run.outcome.jitter_ms,
                "deadline_violation_rate": run.outcome.deadline_violation_rate,
            }

        # Verify the complete visible prefix, excluding only run-specific identifiers.
        left = [asdict(r) for r in pair.baseline.controller_trace.available_at(5)]
        right = [asdict(r) for r in pair.intervention.controller_trace.available_at(5)]
        for row in left + right:
            row.pop("run_id")
            row.pop("provenance_digest")
        if left != right:
            raise ValueError("paired pre-action telemetry differs")
        target = output / "simulation" / item.scenario_id
        target.mkdir(parents=True)
        outcome = {
            "keep": arm(pair.baseline),
            "reroute": arm(pair.intervention),
            "scenario_id": item.scenario_id,
            "foreground_start_tick": 6,
            "outcome_arrival_stop_tick": 64,
            "queue_drain_complete": True,
            "schedule_sha256": pair.schedule.digest,
        }
        path = target / "outcomes.json"
        path.write_text(json.dumps(outcome, sort_keys=True, indent=2) + "\n")
        records.append(
            asdict(reconstruct_pre_action_telemetry(item, foreground_start_tick=6))
        )
        manifest["records"][item.scenario_id] = {
            "outcomes_sha256": sha(path),
            "schedule_sha256": pair.schedule.digest,
            "pre_action_prefix_identical": True,
        }
        if index % 200 == 0:
            print(
                json.dumps(
                    {
                        "split": split,
                        "completed_pairs": index,
                        "total_pairs": len(scenarios),
                    }
                ),
                flush=True,
            )
    telemetry = output / "controller-safe-telemetry.jsonl"
    telemetry.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    manifest["telemetry_sha256"] = sha(telemetry)
    (output / "controlled_rebuild_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "split": split,
                "pairs": len(scenarios),
                "telemetry_sha256": sha(telemetry),
            }
        ),
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if os.name == "nt":
        output = Path("\\\\?\\" + str(output))
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    build(
        ROOT / "code/configs/experiments/m14_final_simulation_matrix.json",
        "train",
        output / "training",
    )
    build(
        ROOT / "code/configs/experiments/m14_confirmatory_repair_matrix.json",
        "test",
        output / "heldout",
    )


if __name__ == "__main__":
    main()
