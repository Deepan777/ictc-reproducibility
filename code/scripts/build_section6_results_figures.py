"""Build publication figures for the experimental methodology and results sections.

Every plotted value is read from the pre-specified analysis results. The script writes a
vector PDF for the manuscript and a 300 dpi PNG for visual inspection and reuse.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EVIDENCE = ROOT / "manuscript" / "evidence"
OUTPUT = ROOT / "manuscript" / "elsevier-cas" / "els-cas-templates" / "figs"

# Okabe-Ito-derived, colour-blind-safe palette.
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
GRAY = "#777777"
LIGHT_GRAY = "#D4D4D4"
BLACK = "#202020"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "semibold",
        }
    )


def _save(figure: plt.Figure, stem: str) -> None:
    figure.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    figure.savefig(OUTPUT / f"{stem}.png", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _method_style(label: str) -> tuple[str, str, int]:
    if label == "ICTC":
        return BLUE, "D", 42
    if label == "Outcome-informed oracle":
        return ORANGE, "*", 68
    if label == "Greedy telemetry":
        return GREEN, "o", 42
    if label == "Incumbent retention":
        return BLACK, "s", 36
    return GRAY, "o", 34


def controller_performance_figure() -> None:
    """Compare deployable controllers on utility, harm, and beneficial actions."""
    data = pd.read_csv(RESULTS / "n13_confirmation_final" / "n13_method_summary.csv")
    names = {
        "B0_KEEP": "Incumbent retention",
        "B1_SHORTEST_DELAY": "Shortest delay",
        "B2_MAX_CAPACITY": "Maximum capacity",
        "B3_MIN_QUEUE": "Minimum queue",
        "B4_GREEDY_TELEMETRY": "Greedy telemetry",
        "B5_ECMP_HASH": "Hash-based multipath",
        "S1_ICTC_CONTRACT": "ICTC",
        "O0_OFFLINE_ORACLE_UPPER_BOUND": "Outcome-informed oracle",
    }
    order = list(names)
    data = data.set_index("method").loc[order].reset_index()
    data["label"] = data["method"].map(names)
    y = np.arange(len(data))[::-1]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(10.8, 4.15),
        gridspec_kw={"width_ratios": [1.45, 1.0, 1.0]},
        sharey=True,
    )

    for index, row in data.iterrows():
        color, marker, size = _method_style(row["label"])
        position = y[index]
        axes[0].errorbar(
            row["mean_utility"],
            position,
            xerr=[
                [row["mean_utility"] - row["utility_ci_low"]],
                [row["utility_ci_high"] - row["mean_utility"]],
            ],
            fmt=marker,
            color=color,
            markersize=np.sqrt(size),
            capsize=2.5,
            linewidth=1.1,
            zorder=3,
        )
        axes[1].barh(position, row["harmful_action_rate"], color=color, height=0.56)
        axes[2].barh(position, row["beneficial_action_rate"], color=color, height=0.56)

    axes[0].set_yticks(y, data["label"])
    axes[0].set_xlabel("Mean network utility (95% CI)")
    axes[0].set_xlim(0.915, 0.963)
    axes[0].set_title("(a) Composite performance")
    axes[0].axvline(
        float(data.loc[data["label"] == "Greedy telemetry", "mean_utility"].iloc[0]),
        color=GREEN,
        linestyle=":",
        linewidth=1.0,
    )

    axes[1].set_xlabel("Harmful-action rate")
    axes[1].set_xlim(0, 0.16)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[1].set_title("(b) Harmful interventions")

    axes[2].set_xlabel("Beneficial-action rate")
    axes[2].set_xlim(0, 0.40)
    axes[2].xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].set_title("(c) Useful interventions")

    for axis in axes:
        axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.55, alpha=0.7)
        axis.tick_params(axis="y", length=0)
    figure.tight_layout(w_pad=1.3)
    _save(figure, "section6_controller_performance")


def primary_figure() -> None:
    analysis = json.loads(
        (RESULTS / "n13_confirmation_final" / "n13_analysis.json").read_text(encoding="utf-8")
    )["primary_confirmatory_result"]

    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.85))

    axes[0].bar(
        ["Greedy\ntelemetry", "ICTC"],
        [0.0638888889, 0.0],
        color=[GREEN, BLUE],
        width=0.58,
    )
    axes[0].set_ylabel("Harmful-action rate")
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].set_ylim(0, 0.075)
    axes[0].text(0, 0.066, "6.39%", ha="center", va="bottom")
    axes[0].text(1, 0.002, "0%", ha="center", va="bottom")
    axes[0].set_title("(a) Observed harm")

    utility = float(analysis["utility_difference"])
    utility_low, utility_high = [float(value) for value in analysis["utility_difference_ci"]]
    axes[1].axvline(0, color=BLACK, linewidth=0.9)
    axes[1].axvline(-0.01, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[1].errorbar(
        utility,
        0,
        xerr=[[utility - utility_low], [utility_high - utility]],
        fmt="D",
        color=BLUE,
        capsize=4,
        markersize=5.5,
    )
    axes[1].set_xlim(-0.0115, 0.0015)
    axes[1].set_yticks([])
    axes[1].set_xlabel("ICTC - greedy utility")
    axes[1].set_title("(b) Utility non-inferiority")
    axes[1].text(-0.0098, 0.10, "margin -0.01", color=ORANGE, ha="left", va="bottom")
    axes[1].text(utility, -0.10, f"{utility:.4f}", color=BLUE, ha="center", va="top")
    axes[1].set_ylim(-0.22, 0.22)

    retention = float(analysis["opportunity_retention_rate"])
    axes[2].bar(["ICTC"], [retention], color=BLUE, width=0.50)
    axes[2].axhline(0.45, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[2].set_ylim(0, 0.55)
    axes[2].set_ylabel("Opportunity retention")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].set_title("(c) Retained opportunities")
    axes[2].text(0, retention + 0.012, f"{retention:.3f}", ha="center", va="bottom")
    axes[2].text(0.28, 0.452, "threshold 0.45", color=ORANGE, ha="left", va="bottom")

    figure.tight_layout(w_pad=2.0)
    _save(figure, "section6_primary_gates")


def seed_effects_figure() -> None:
    """Expose the paired 20-seed distribution behind the primary contrasts."""
    seed_data = pd.read_csv(
        RESULTS / "n13_confirmation_final" / "n13_seed_level_metrics.csv"
    )
    analysis = json.loads(
        (RESULTS / "n13_confirmation_final" / "n13_analysis.json").read_text(encoding="utf-8")
    )["primary_confirmatory_result"]
    selected = seed_data[
        seed_data["method"].isin(["B4_GREEDY_TELEMETRY", "S1_ICTC_CONTRACT"])
    ]
    utility = selected.pivot(index="seed", columns="method", values="utility")
    harmful = selected.pivot(index="seed", columns="method", values="harmful")
    utility_difference = utility["S1_ICTC_CONTRACT"] - utility["B4_GREEDY_TELEMETRY"]
    harmful_difference = harmful["S1_ICTC_CONTRACT"] - harmful["B4_GREEDY_TELEMETRY"]
    positions = np.arange(1, len(utility_difference) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.0), sharex=True)
    panels = (
        (
            axes[0],
            harmful_difference,
            float(analysis["harm_rate_difference"]),
            [float(value) for value in analysis["harm_rate_difference_ci"]],
            0.0,
            "(a) Harm-rate difference by seed",
            "ICTC - greedy harm rate",
        ),
        (
            axes[1],
            utility_difference,
            float(analysis["utility_difference"]),
            [float(value) for value in analysis["utility_difference_ci"]],
            -0.01,
            "(b) Utility difference by seed",
            "ICTC - greedy utility",
        ),
    )
    for axis, values, estimate, interval, reference, title, ylabel in panels:
        axis.axhline(0, color=BLACK, linewidth=0.8)
        axis.axhline(reference, color=ORANGE, linestyle="--", linewidth=1.0)
        axis.axhspan(interval[0], interval[1], color=BLUE, alpha=0.14)
        axis.axhline(estimate, color=BLUE, linewidth=1.3)
        axis.scatter(positions, values.values, color=BLUE, s=23, zorder=3)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Independent confirmation seed block")
        axis.set_xticks([1, 5, 10, 15, 20])
        axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[0].text(20, -0.004, "zero-harm boundary", ha="right", va="top", color=BLACK)
    axes[1].text(20, -0.0105, "non-inferiority margin", ha="right", va="top", color=ORANGE)
    figure.tight_layout(w_pad=1.6)
    _save(figure, "section6_seed_effects")


def ablation_figure() -> None:
    """Show clause-removal effects and pre-specified sensitivity analyses."""
    evidence = json.loads((EVIDENCE / "m16_results.json").read_text(encoding="utf-8"))
    comparisons = evidence["ablation_comparisons"]

    ablation_ids = ["A0"] + [f"A{index}" for index in range(1, 15)]
    labels = [
        "Reference ICTC criterion",
        "No identification",
        "No effect-acceptability criterion",
        "No freshness",
        "No synchronisation",
        "No provenance",
        "Trust threshold removed",
        "Source diversity reduced",
        "Zero-age freshness",
        "Association-selected covariates",
        "Fivefold telemetry acquisition",
        "One-source acquisition",
        "No breach monitoring",
        "No predeclared fallback",
        "Misspecified graph",
    ]
    summaries = [evidence["a0_summary"]] + [
        comparisons[f"A{index}:default"]["summary"] for index in range(1, 15)
    ]
    false_admission = np.array([float(item["false_admission_rate"]) for item in summaries])
    harmful = np.array([float(item["harmful_action_risk"]) for item in summaries])
    deferral = np.array([float(item["unnecessary_deferral_rate"]) for item in summaries])

    figure = plt.figure(figsize=(11.0, 5.6))
    grid = figure.add_gridspec(2, 2, width_ratios=[1.62, 1.0], hspace=0.55, wspace=0.35)
    main_axis = figure.add_subplot(grid[:, 0])
    width_axis = figure.add_subplot(grid[0, 1])
    budget_axis = figure.add_subplot(grid[1, 1])

    y = np.arange(len(ablation_ids))[::-1]
    main_axis.scatter(
        false_admission,
        y + 0.20,
        color=ORANGE,
        marker="o",
        s=27,
        label="Reference-criterion violation",
    )
    main_axis.scatter(
        harmful,
        y,
        color=BLACK,
        marker="^",
        s=25,
        label="Latency non-improvement",
    )
    main_axis.scatter(
        deferral,
        y - 0.20,
        color=BLUE,
        marker="s",
        s=24,
        label="Unnecessary deferral",
    )
    main_axis.hlines(
        y,
        0,
        np.maximum.reduce([false_admission, harmful, deferral]),
        color=LIGHT_GRAY,
        linewidth=0.65,
    )
    main_axis.set_yticks(y, labels)
    main_axis.set_xlim(-0.01, 0.55)
    main_axis.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    main_axis.set_xlabel("Rate over 1,320 held-out proposals")
    main_axis.set_title("(a) Mechanism ablation")
    main_axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    main_axis.legend(loc="lower right", frameon=False, ncol=1)

    widths = [0, 1, 5, 20, 1000]
    width_summary = [comparisons[f"A15:width_ms={value}"]["summary"] for value in widths]
    width_axis.plot(
        range(len(widths)),
        [float(item["admission_fraction"]) for item in width_summary],
        color=GREEN,
        marker="o",
        label="Admission",
    )
    width_axis.plot(
        range(len(widths)),
        [float(item["unnecessary_deferral_rate"]) for item in width_summary],
        color=BLUE,
        marker="s",
        label="Unnecessary deferral",
    )
    width_axis.set_xticks(range(len(widths)), [str(value) for value in widths])
    width_axis.set_xlabel("Maximum effect-interval width (ms)")
    width_axis.set_ylabel("Rate")
    width_axis.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    width_axis.set_ylim(-0.02, 0.58)
    width_axis.set_title("(b) Uncertainty-width sensitivity")
    width_axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, alpha=0.65)
    width_axis.legend(loc="lower right", frameon=False)

    deadlines = [0, 2, 5]
    budgets = [64, 128, 640]
    feasibility = np.array(
        [
            [
                float(
                    comparisons[
                        f"A16:deadline_ticks={deadline};budget_bytes={budget}"
                    ]["summary"]["admission_fraction"]
                )
                for budget in budgets
            ]
            for deadline in deadlines
        ]
    )
    heat = budget_axis.imshow(feasibility, vmin=0, vmax=0.55, cmap="Blues", aspect="auto")
    for row in range(feasibility.shape[0]):
        for column in range(feasibility.shape[1]):
            value = feasibility[row, column]
            budget_axis.text(
                column,
                row,
                f"{100 * value:.1f}%",
                ha="center",
                va="center",
                color="white" if value > 0.30 else BLACK,
                fontsize=8,
            )
    budget_axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
    budget_axis.set_yticks(range(len(deadlines)), [str(value) for value in deadlines])
    budget_axis.set_xlabel("Telemetry budget (bytes)")
    budget_axis.set_ylabel("Acquisition deadline (ms)")
    budget_axis.set_title("(c) Admission under resource constraints")
    colorbar = figure.colorbar(heat, ax=budget_axis, fraction=0.046, pad=0.04)
    colorbar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    colorbar.set_label("Admission rate", fontsize=8)

    _save(figure, "section6_ablation_study")


def heterogeneity_figure() -> None:
    """Display where harm reduction and utility cost occur."""
    data = pd.read_csv(RESULTS / "n13_confirmation_final" / "n13_condition_summary.csv")
    methods = data[data["method"].isin(["B4_GREEDY_TELEMETRY", "S1_ICTC_CONTRACT"])]
    index = ["bulk", "web", "realtime"]
    columns = ["stable", "failure", "capacity", "cross-shift"]
    pretty_rows = ["Bulk TCP", "Web TCP", "Real-time UDP"]
    pretty_columns = ["Stable", "Failure", "Capacity", "Cross-traffic"]

    greedy = methods[methods["method"] == "B4_GREEDY_TELEMETRY"].pivot(
        index="traffic", columns="event", values="harmful"
    ).loc[index, columns]
    ictc = methods[methods["method"] == "S1_ICTC_CONTRACT"].pivot(
        index="traffic", columns="event", values="utility"
    ).loc[index, columns]
    greedy_utility = methods[methods["method"] == "B4_GREEDY_TELEMETRY"].pivot(
        index="traffic", columns="event", values="utility"
    ).loc[index, columns]
    difference = ictc - greedy_utility

    figure, axes = plt.subplots(1, 2, figsize=(10.4, 3.05))
    harm_image = axes[0].imshow(greedy.values, cmap="OrRd", vmin=0, vmax=0.55, aspect="auto")
    utility_norm = TwoSlopeNorm(vmin=-0.045, vcenter=0.0, vmax=0.045)
    utility_image = axes[1].imshow(difference.values, cmap="RdBu", norm=utility_norm, aspect="auto")

    for axis in axes:
        axis.set_xticks(range(len(columns)), pretty_columns, rotation=22, ha="right")
        axis.set_yticks(range(len(index)), pretty_rows)
    for row in range(greedy.shape[0]):
        for column in range(greedy.shape[1]):
            harm = float(greedy.iloc[row, column])
            axes[0].text(
                column,
                row,
                f"{100 * harm:.1f}%",
                ha="center",
                va="center",
                color="white" if harm > 0.28 else BLACK,
            )
            value = float(difference.iloc[row, column])
            axes[1].text(
                column,
                row,
                f"{value:+.3f}",
                ha="center",
                va="center",
                color="white" if abs(value) > 0.028 else BLACK,
            )

    axes[0].set_title("(a) Greedy harmful-action rate")
    axes[1].set_title("(b) ICTC - greedy mean utility")
    harm_bar = figure.colorbar(harm_image, ax=axes[0], fraction=0.046, pad=0.04)
    harm_bar.ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    harm_bar.set_label("Harmful-action rate", fontsize=8)
    utility_bar = figure.colorbar(utility_image, ax=axes[1], fraction=0.046, pad=0.04)
    utility_bar.set_label("Utility difference", fontsize=8)
    figure.tight_layout(w_pad=1.4)
    _save(figure, "section6_condition_heterogeneity")


def transport_figure() -> None:
    analysis = json.loads(
        (RESULTS / "n13_transport_final" / "n13_transport_analysis.json").read_text(
            encoding="utf-8"
        )
    )["topology_results"]
    labels = ["GEANT", "GARR"]
    keys = ["geant-2012-03", "garr-2011-09"]
    figure, axes = plt.subplots(1, 3, figsize=(10.5, 2.85))

    for axis, field, ci_field, title, xlabel, margin in (
        (
            axes[0],
            "harm_rate_difference",
            "harm_rate_difference_ci",
            "(a) Harm contrast",
            "ICTC - greedy harm rate",
            0.0,
        ),
        (
            axes[1],
            "utility_difference",
            "utility_difference_ci",
            "(b) Utility contrast",
            "ICTC - greedy utility",
            -0.01,
        ),
    ):
        for index_value, key in enumerate(keys):
            value = float(analysis[key][field])
            low, high = [float(item) for item in analysis[key][ci_field]]
            axis.errorbar(
                value,
                index_value,
                xerr=[[value - low], [high - value]],
                fmt="D",
                color=BLUE,
                capsize=4,
                markersize=5,
            )
        axis.axvline(0, color=BLACK, linewidth=0.9)
        if margin != 0:
            axis.axvline(margin, color=ORANGE, linestyle="--", linewidth=1.2)
        axis.set_yticks(range(2), labels)
        axis.invert_yaxis()
        axis.set_xlabel(xlabel)
        axis.set_title(title)

    retention = [float(analysis[key]["opportunity_retention_rate"]) for key in keys]
    axes[2].bar(labels, retention, color=BLUE, width=0.55)
    axes[2].axhline(0.45, color=ORANGE, linestyle="--", linewidth=1.2)
    axes[2].set_ylim(0, 0.68)
    axes[2].set_ylabel("Opportunity retention")
    axes[2].yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    axes[2].set_title("(c) Retained opportunities")
    for index_value, value in enumerate(retention):
        axes[2].text(index_value, value + 0.018, f"{value:.3f}", ha="center", va="bottom")

    figure.tight_layout(w_pad=2.0)
    _save(figure, "section6_transport_gates")


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _style()
    controller_performance_figure()
    primary_figure()
    seed_effects_figure()
    ablation_figure()
    heterogeneity_figure()
    transport_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
