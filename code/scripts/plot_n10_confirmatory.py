"""Render the N10 confirmatory utility and outcome-count figure."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "docs" / "milestones" / "evidence" / "n10_confirmatory_analysis.json"
OUTPUT = ROOT / "manuscript" / "figures" / "figure_n10_confirmatory.png"
METHODS = (
    "D1_CONTEXT_SUPPORTED_CAPACITY",
    "B0_KEEP",
    "B1_REROUTE",
    "B2_PROPAGATION",
    "B3_CAPACITY",
    "B4_QUEUE",
)
LABELS = ("D1", "B0", "B1", "B2", "B3", "B4")


def main() -> int:
    payload = cast(dict[str, Any], json.loads(ANALYSIS.read_text(encoding="utf-8")))
    results = payload["method_results"]
    means = np.array([results[method]["mean_seed_utility"] for method in METHODS])
    intervals = np.array([results[method]["bootstrap_95_ci"] for method in METHODS])
    errors = np.vstack((means - intervals[:, 0], intervals[:, 1] - means))
    beneficial = np.array([results[method]["counts"]["BENEFICIAL"] for method in METHODS])
    harmful = np.array([results[method]["counts"]["HARMFUL"] for method in METHODS])
    keep = np.array([results[method]["counts"]["KEEP"] for method in METHODS])
    neutral = np.array([results[method]["counts"]["NO_MATERIAL_DIFFERENCE"] for method in METHODS])

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 4.4), constrained_layout=True)
    colors = ["#0072B2", "#999999", "#D55E00", "#CC79A7", "#009E73", "#E69F00"]
    x = np.arange(len(METHODS))
    axes[0].bar(x, means, color=colors, width=0.72, zorder=2)
    axes[0].errorbar(x, means, yerr=errors, fmt="none", ecolor="#222222", capsize=3, zorder=3)
    axes[0].axhline(0, color="#333333", linewidth=0.8)
    axes[0].set_xticks(x, LABELS)
    axes[0].set_ylabel("Mean utility per seed block")
    axes[0].set_title("A. Frozen seed-block utility (95% bootstrap CI)", loc="left", weight="bold")
    axes[0].grid(axis="y", color="#DDDDDD", linewidth=0.6, zorder=1)

    stacks = (
        (beneficial, "Beneficial", "#009E73"),
        (harmful, "Harmful", "#D55E00"),
        (keep, "KEEP", "#999999"),
        (neutral, "No material difference", "#56B4E9"),
    )
    left = np.zeros(len(METHODS), dtype=float)
    for values, label, color in stacks:
        axes[1].barh(x, values, left=left, label=label, color=color, height=0.68)
        left += values
    axes[1].set_yticks(x, LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 720)
    axes[1].set_xlabel("Decisions across 720 factorial cells")
    axes[1].set_title("B. Retained physical outcome labels", loc="left", weight="bold")
    axes[1].grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    axes[1].legend(
        frameon=False,
        ncol=2,
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        borderaxespad=0,
    )
    figure.suptitle(
        "N10 prospective confirmation: D1 improves capacity-safe B3 without observed harm",
        fontsize=11,
        weight="bold",
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
