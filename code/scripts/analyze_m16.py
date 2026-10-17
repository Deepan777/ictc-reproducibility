"""Run immutable M16 inference and generate paper-ready tables and figures."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

from ictc.evaluation.m16 import calculate_m16, primary_table_rows

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M16 evidence: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _write_csv(path: Path, rows: tuple[dict[str, object], ...]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M16 table: {path}")
    if not rows:
        raise ValueError("M16 table cannot be empty")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _primary_figure(path: Path, results: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M16 figure: {path}")
    labels = ["A0", *results["baseline_comparisons"]]
    metrics = (
        ("false_admission_rate", "False admissions"),
        ("harmful_action_risk", "Harmful actions"),
        ("admission_fraction", "Admissions"),
    )
    summaries = [
        results["a0_summary"],
        *[results["baseline_comparisons"][baseline]["summary"] for baseline in labels[1:]],
    ]
    x = np.arange(len(labels))
    width = 0.25
    fig, axis = plt.subplots(figsize=(10.5, 4.8), constrained_layout=True)
    for index, (metric, title) in enumerate(metrics):
        axis.bar(
            x + (index - 1) * width,
            [summary[metric] for summary in summaries],
            width,
            label=title,
        )
    axis.set_ylabel("Fraction of 1,320 held-out pairs")
    axis.set_xticks(x, labels)
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, frameon=False, loc="upper center")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _condition_figure(path: Path, results: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite M16 figure: {path}")
    conditions = list(results["a0_condition_summaries"])
    summaries = [results["a0_condition_summaries"][key] for key in conditions]
    x = np.arange(len(conditions))
    width = 0.27
    fig, axis = plt.subplots(figsize=(12.5, 5.4), constrained_layout=True)
    for index, (metric, label) in enumerate(
        (
            ("false_admission_rate", "False admissions"),
            ("harmful_action_risk", "Harmful actions"),
            ("admission_fraction", "Admissions"),
        )
    ):
        axis.bar(
            x + (index - 1) * width,
            [summary[metric] for summary in summaries],
            width,
            label=label,
        )
    axis.set_ylabel("A0 fraction")
    axis.set_xticks(x, conditions, rotation=35, ha="right")
    axis.set_ylim(0, 1.0)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=3, frameon=False)
    fig.savefig(path, dpi=300)
    plt.close(fig)


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
    decision_root = Path(str(authorization["decision_output_root"]))
    output = decision_root / "m16"
    output.mkdir(exist_ok=False)
    amendment = ROOT / "configs/experiments/m16_analysis_amendment.json"
    results = calculate_m16(decision_root, amendment)
    _write_json(output / "m16_results.json", results)
    _write_csv(output / "table_primary_comparisons.csv", primary_table_rows(results))
    _primary_figure(output / "figure_primary_controller_rates.png", results)
    _condition_figure(output / "figure_a0_condition_profile.png", results)
    print(
        json.dumps(
            {
                "output": str(output),
                "pair_count": results["pair_count"],
                "excluded_pair_count": results["excluded_pair_count"],
                "primary_table_rows": len(primary_table_rows(results)),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
