"""Reconstruct decision parameters from a separately supplied supplementary CSV.

No fitting, simulation, outcome reading, or private model file is involved.
The output is an inference wrapper, not the historical selection/audit archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "code/src"))
from ictc.ns3.n13 import RISK_FEATURE_NAMES  # noqa: E402 - portable direct CLI execution

ORIGINAL_MODEL_SHA256 = (
    "ee5aad0b4183f3be195d7dff695bb8c11e68babb38177ff97c59c6a87b5d7c5c"
)


def model_digest(model: dict[str, Any]) -> str:
    content = json.dumps(model, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def model_from_csv(
    path: Path, *, expected_sha256: str | None = ORIGINAL_MODEL_SHA256
) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = [
            r
            for r in csv.DictReader(stream)
            if r["topology"] == "abilene" and r["risk_model"] == "original"
        ]
    names = list(RISK_FEATURE_NAMES)
    if len(rows) != len(names) or {r["feature"] for r in rows} != set(names):
        raise ValueError("CSV must contain each original Abilene feature exactly once")
    by_name = {r["feature"]: r for r in rows}
    ordered = [by_name[name] for name in names]
    for field in ("intercept", "regularization_C"):
        if len({float(r[field]) for r in ordered}) != 1:
            raise ValueError(f"Inconsistent original-model {field}")
    model: dict[str, Any] = {
        "feature_names": names,
        "means": [float(r["mean"]) for r in ordered],
        "scales": [float(r["scale"]) for r in ordered],
        "coefficients": [float(r["coefficient"]) for r in ordered],
        "intercept": float(ordered[0]["intercept"]),
        "regularization_C": float(ordered[0]["regularization_C"]),
        "maximum_predicted_harm_probability": 0.30,
    }
    numbers = [
        *model["means"],
        *model["scales"],
        *model["coefficients"],
        model["intercept"],
        model["regularization_C"],
    ]
    if not all(math.isfinite(x) for x in numbers) or min(model["scales"]) <= 0:
        raise ValueError("Model values must be finite, with strictly positive scales")
    if model["regularization_C"] <= 0:
        raise ValueError("Regularization must be positive")
    if expected_sha256 is not None and model_digest(model) != expected_sha256:
        raise ValueError("Parameters differ from the retained original decision model")
    return model


def reconstruct(csv_path: Path, output: Path) -> dict[str, Any]:
    model = model_from_csv(csv_path)
    payload = {
        "schema_version": "ictc-reconstructed-decision-policy-v1",
        "controller_thresholds": {"risk_model": model},
        "reconstruction": {
            "parameter_csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            "active_model_sha256": model_digest(model),
            "historical_selection_metadata_reconstructed": False,
            "fitting_performed": False,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-coefficients",
        type=Path,
        required=True,
        help="Full-precision model_coefficients.csv from the supplementary sources",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New inference-policy JSON; existing files are never overwritten",
    )
    args = parser.parse_args()
    payload = reconstruct(args.model_coefficients, args.output)
    print(json.dumps(payload["reconstruction"], indent=2))


if __name__ == "__main__":
    main()
