"""Select N12 thresholds from only the authorized development journal."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd  # type: ignore[import-untyped]

from ictc.ns3.n12 import build_decision, load_matrix, read_object, sha256, utility, write_new


def select_policy(
    journal: Path, matrix_path: Path, protocol_path: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite N12 policy selection: {output}")
    matrix, scenarios = load_matrix(matrix_path)
    protocol = read_object(protocol_path)
    if (
        matrix["phase"] != "development"
        or sorted({item.seed for item in scenarios}) != protocol["source_seeds"]
    ):
        raise ValueError("N12 policy selection may use only the declared development seeds")
    if set(matrix["seeds"]) & set(protocol["forbidden_seeds"]):
        raise ValueError("N12 policy selection touched a forbidden confirmatory seed")
    grid = cast(dict[str, list[float]], protocol["candidate_grid"])
    fixed = cast(dict[str, Any], protocol["fixed_quality_thresholds"])
    candidates = []
    for index, values in enumerate(
        itertools.product(
            grid["queue_penalty_per_packet"],
            grid["delay_penalty_per_ms"],
            grid["minimum_score_improvement"],
        )
    ):
        thresholds = {
            **fixed,
            "queue_penalty_per_packet": values[0],
            "delay_penalty_per_ms": values[1],
            "minimum_score_improvement": values[2],
        }
        results = []
        for scenario in scenarios:
            probe = journal / "runs" / scenario.scenario_id / "probe-route0" / "raw"
            action = build_decision(scenario, probe, thresholds)["decisions"]["S1_ICTC_CONTRACT"]
            action_rows = {
                route: pd.read_csv(
                    journal
                    / "runs"
                    / scenario.scenario_id
                    / f"action-{route}"
                    / "raw"
                    / "summary.csv",
                    dtype=str,
                )
                .iloc[0]
                .to_dict()
                for route in ("route0", "route1", "route2")
            }
            utilities = {route: utility(row) for route, row in action_rows.items()}
            keep = utilities["route0"]
            best = max(utilities, key=utilities.__getitem__)
            results.append(
                {
                    "utility": utilities[action],
                    "harmful": utilities[action] < keep - 0.02,
                    "opportunity": utilities[best] > keep + 0.02,
                    "retained": (
                        utilities[best] > keep + 0.02
                        and utilities[action] >= utilities[best] - 0.02
                    ),
                }
            )
        frame = pd.DataFrame(results)
        opportunities = int(frame["opportunity"].sum())
        candidates.append(
            {
                "candidate_id": f"C{index:03d}",
                **thresholds,
                "mean_utility": float(frame["utility"].mean()),
                "harmful_action_rate": float(frame["harmful"].mean()),
                "opportunity_retention_rate": (
                    float(frame["retained"].sum() / opportunities) if opportunities else 0.0
                ),
            }
        )
    candidate_frame = pd.DataFrame(candidates)
    minimum_retention = float(protocol["selection_rule"]["minimum_opportunity_retention_rate"])
    eligible = candidate_frame[
        candidate_frame["opportunity_retention_rate"] >= minimum_retention
    ].copy()
    if eligible.empty:
        raise ValueError("N12 policy grid has no candidate meeting the retention constraint")
    selected = (
        eligible.sort_values(
            [
                "harmful_action_rate",
                "mean_utility",
                "opportunity_retention_rate",
                "candidate_id",
            ],
            ascending=[True, False, False, True],
        )
        .iloc[0]
        .to_dict()
    )
    output.mkdir(parents=True)
    candidate_path = output / "n12_policy_candidates.csv"
    candidate_frame.to_csv(candidate_path, index=False, lineterminator="\n")
    payload = {
        "schema_version": "n12-selected-policy-v1",
        "selected": selected,
        "candidate_count": len(candidate_frame),
        "eligible_candidate_count": len(eligible),
        "protocol_sha256": sha256(protocol_path),
        "development_matrix_sha256": sha256(matrix_path),
        "development_manifest_sha256": sha256(journal / "n12_manifest.json"),
        "confirmatory_seeds_accessed": False,
        "initial_result_retained_at": str(journal / "analysis-v1"),
    }
    write_new(output / "n12_selected_policy.json", payload)
    write_new(
        output / "selection_manifest.json",
        {
            "schema_version": "n12-policy-selection-manifest-v1",
            "candidates_sha256": sha256(candidate_path),
            "selected_policy_sha256": sha256(output / "n12_selected_policy.json"),
        },
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(
        json.dumps(
            select_policy(
                arguments.journal,
                arguments.matrix,
                arguments.protocol,
                arguments.output,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
