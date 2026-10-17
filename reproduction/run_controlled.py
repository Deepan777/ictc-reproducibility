"""Execute controlled held-out decisions using explicit portable record locations."""

# ruff: noqa: E402 -- explicit checkout-local imports after portable path bootstrap.
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT / "code/src")]
from ictc.pilot.m14_authorization import M14ControllerAuthorization
from ictc.pilot.m14_decisions import run_m14_decisions
from ictc.pilot.m14_oracle import persist_b0_oracle
from ictc.pilot.m14_scoring import score_m14_decisions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("decisions", "oracle", "score"))
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--decision-root", type=Path, required=True)
    args = parser.parse_args()
    raw = args.raw_root.resolve()
    output = args.decision_root.resolve()
    for path in (raw, args.training_root.resolve()):
        manifest = json.loads((path / "controlled_rebuild_manifest.json").read_text())
        if (
            manifest.get("foreground_start_tick"),
            manifest.get("acquisition_cutoff_tick"),
        ) != (6, 5):
            raise ValueError(
                "controlled records require the corrected pre-action prelude"
            )
    if args.phase == "decisions":
        if output.exists():
            raise FileExistsError(output)
        authorization = M14ControllerAuthorization(
            "controlled-reproduction", raw, 1320, output, args.training_root.resolve()
        )
        result = run_m14_decisions(
            repository=ROOT / "code",
            telemetry_path=raw / "controller-safe-telemetry.jsonl",
            authorization=authorization,
            baseline_config=ROOT / "code/configs/baselines/m09_frozen.json",
            protocol_path=ROOT
            / "code/configs/experiments/m10_pilot_protocol_amendment_001.json",
        )
    elif args.phase == "oracle":
        result = persist_b0_oracle(
            telemetry_path=raw / "controller-safe-telemetry.jsonl",
            raw_root=raw,
            decision_root=output,
        )
    else:
        result = score_m14_decisions(
            telemetry_path=raw / "controller-safe-telemetry.jsonl",
            raw_root=raw,
            decision_root=output,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
