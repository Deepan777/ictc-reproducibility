"""Materialize controller-safe telemetry for every frozen M14 simulation scenario."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.simulation.m14_matrix import load_m14_simulation_matrix
from ictc.simulation.m14_telemetry import write_telemetry_immutable


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = write_telemetry_immutable(
        load_m14_simulation_matrix(arguments.matrix), arguments.output
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
