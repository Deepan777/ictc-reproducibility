"""Persist the declared B0 full-state oracle after the pre-truth decision boundary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from ictc.pilot.m14_decisions import _json_rows, _sha256, _write_jsonl


def _objective(outcome: dict[str, Any]) -> float:
    return (
        float(outcome["latency_ms"])
        + 1000.0 * float(outcome["loss_ratio"])
        + float(outcome["jitter_ms"])
        + 1000.0 * float(outcome["deadline_violation_rate"])
    )


def persist_b0_oracle(
    *, telemetry_path: Path, raw_root: Path, decision_root: Path
) -> dict[str, object]:
    """Read held-out truth only for B0 and persist its choices before any scoring."""
    pretruth = decision_root / "pretruth_audit.json"
    if not pretruth.is_file():
        raise PermissionError("B0 requires the completed pre-truth persistence audit")
    audit = json.loads(pretruth.read_text(encoding="utf-8"))
    if audit.get("heldout_outcome_rows_read_before_decision_persistence") != 0:
        raise PermissionError("pre-truth audit reports held-out leakage")
    if audit.get("outcome_scoring_started") is not False:
        raise PermissionError("B0 must be persisted before outcome scoring starts")
    test_rows = tuple(row for row in _json_rows(telemetry_path) if row["topology_split"] == "test")
    decisions: list[dict[str, object]] = []
    for row in test_rows:
        scenario_id = str(row["scenario_id"])
        outcome_path = raw_root / "simulation" / scenario_id / "outcomes.json"
        outcome = cast(dict[str, Any], json.loads(outcome_path.read_text(encoding="utf-8")))
        keep_objective = _objective(cast(dict[str, Any], outcome["keep"]))
        reroute_objective = _objective(cast(dict[str, Any], outcome["reroute"]))
        decisions.append(
            {
                "baseline_id": "B0",
                "pair_id": scenario_id,
                "scenario_id": scenario_id,
                "disposition": "EXECUTE",
                "reason": "ORACLE_BEST_ACTION",
                "selected_action": (
                    "REROUTE_TO_PATH" if reroute_objective < keep_objective else "KEEP_CURRENT_PATH"
                ),
                "telemetry_items": 0,
                "telemetry_bytes": 0,
                "keep_objective": keep_objective,
                "reroute_objective": reroute_objective,
                "score": keep_objective - reroute_objective,
                "oracle_only": True,
            }
        )
    target = decision_root / "baseline_b0_oracle_decisions.jsonl"
    digest = _write_jsonl(target, tuple(decisions))
    result = {
        "schema_version": "1.0.0",
        "purpose": "B0 full-state oracle comparator only",
        "heldout_outcome_rows_read": len(test_rows),
        "outcome_scoring_started": False,
        "decision_count": len(decisions),
        "decision_sha256": digest,
        "pretruth_audit_sha256": _sha256(pretruth),
    }
    (decision_root / "oracle_access_audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
