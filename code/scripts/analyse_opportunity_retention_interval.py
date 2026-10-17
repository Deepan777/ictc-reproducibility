"""Compute the descriptive seed-block interval for opportunity retention.

The confirmatory decision rule used the prospectively specified point-estimate
threshold.  This script adds a descriptive uncertainty interval without changing
that decision rule.  It resamples complete confirmatory seeds so that the
scenario dependence within a seed is preserved.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "n13_confirmation_final" / "n13_seed_level_metrics.csv"
METHOD = "S1_ICTC_CONTRACT"
SCENARIOS_PER_SEED = 36
BOOTSTRAP_SEED = 20260721
RESAMPLES = 10_000


def main() -> None:
    with SOURCE.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["method"] == METHOD]

    opportunities = np.asarray(
        [float(row["opportunity_present"]) * SCENARIOS_PER_SEED for row in rows]
    )
    retained = np.asarray(
        [float(row["opportunity_retained"]) * SCENARIOS_PER_SEED for row in rows]
    )
    if not rows or opportunities.sum() == 0:
        raise ValueError("opportunity-retention source contains no eligible observations")

    random = np.random.default_rng(BOOTSTRAP_SEED)
    estimates = np.empty(RESAMPLES, dtype=float)
    for index in range(RESAMPLES):
        sampled = random.integers(0, len(rows), len(rows))
        estimates[index] = retained[sampled].sum() / opportunities[sampled].sum()

    lower, upper = np.quantile(estimates, [0.025, 0.975])
    print(f"seed_blocks={len(rows)}")
    print(f"opportunities={int(round(float(opportunities.sum())))}")
    print(f"retained={int(round(float(retained.sum())))}")
    print(f"retention={retained.sum() / opportunities.sum():.8f}")
    print(f"descriptive_95_percent_interval=[{lower:.8f}, {upper:.8f}]")


if __name__ == "__main__":
    main()
