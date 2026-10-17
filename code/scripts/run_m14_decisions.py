"""Run the authorized M14 training and held-out decision persistence phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.pilot.m14_authorization import load_m14_controller_authorization
from ictc.pilot.m14_decisions import run_m14_decisions

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / "configs/experiments/m14_controller_authorization.json",
    )
    arguments = parser.parse_args()
    authorization = load_m14_controller_authorization(arguments.authorization, repository=ROOT)
    result = run_m14_decisions(
        repository=ROOT,
        telemetry_path=authorization.raw_matrix_root / "controller-safe-telemetry.jsonl",
        authorization=authorization,
        baseline_config=ROOT / "configs/baselines/m09_frozen.json",
        protocol_path=ROOT / "configs/experiments/m10_pilot_protocol_amendment_001.json",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
