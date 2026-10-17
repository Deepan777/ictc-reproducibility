"""Run the frozen N13 external transport analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.ns3.n13_transport_analysis import analyze


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            analyze(arguments.journal, arguments.matrix, arguments.output), indent=2, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
