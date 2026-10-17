"""Run or validate a frozen N12 realistic Abilene journal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.ns3.n12 import N12Runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--runtime-bin", required=True, type=Path)
    parser.add_argument("--library-bin", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate", action="store_true")
    arguments = parser.parse_args()
    runner = N12Runner(
        repository=arguments.repository,
        output_root=arguments.output_root,
        matrix_path=arguments.matrix,
        executable=arguments.executable,
        runtime_directories=(arguments.library_bin, arguments.runtime_bin),
    )
    result = (
        runner.validate()
        if arguments.validate
        else runner.run(limit=arguments.limit, workers=arguments.workers)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
