"""Select the N13 policy from only its authorized development journal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ictc.ns3.n13_selection import select_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    result = select_policy(
        arguments.journal,
        arguments.matrix,
        arguments.protocol,
        arguments.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
