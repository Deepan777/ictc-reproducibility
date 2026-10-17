"""Score all persisted M14 held-out decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from ictc.pilot.m14_scoring import score_m14_decisions

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        type=Path,
        default=ROOT / "configs/experiments/m14_controller_authorization.json",
    )
    arguments = parser.parse_args()
    authorization = cast(
        dict[str, object],
        json.loads(arguments.authorization.read_text(encoding="utf-8")),
    )
    raw_root = Path(str(authorization["raw_matrix_root"]))
    result = score_m14_decisions(
        telemetry_path=raw_root / "controller-safe-telemetry.jsonl",
        raw_root=raw_root,
        decision_root=Path(str(authorization["decision_output_root"])),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
