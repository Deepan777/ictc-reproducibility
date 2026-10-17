"""Run or validate immutable M15 ns-3/M12 raw evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.m15 import M15Orchestrator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--runtime-bin", required=True, type=Path)
    parser.add_argument("--ns3-library-bin", required=True, type=Path)
    parser.add_argument(
        "--simulation-matrix",
        type=Path,
        default=None,
    )
    parser.add_argument("--protocol", type=Path, default=None)
    parser.add_argument("--skip-m12", action="store_true")
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-ns3", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    orchestrator = M15Orchestrator(
        repository=root,
        output_root=arguments.output_root,
        matrix_path=root / "configs/experiments/m14_final_ns3_matrix.json",
        simulation_matrix_path=(
            arguments.simulation_matrix
            or root / "configs/experiments/m14_final_simulation_matrix.json"
        ),
        protocol_path=(arguments.protocol or root / "configs/experiments/m14_final_protocol.json"),
        executable=arguments.executable,
        runtime_bin=arguments.runtime_bin,
        ns3_library_bin=arguments.ns3_library_bin,
    )
    if not arguments.validate_only:
        orchestrator.run(
            include_m12=not arguments.skip_m12,
            include_simulation=not arguments.skip_simulation,
            include_ns3=not arguments.skip_ns3,
        )
    print(
        json.dumps(
            orchestrator.validate(
                include_m12=not arguments.skip_m12,
                include_simulation=not arguments.skip_simulation,
                include_ns3=not arguments.skip_ns3,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
