"""Post-persistence truth join and scoring for the authorized M14 study."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind, AdmissionDecision
from ictc.pilot.ablation import (
    AblationId,
    FixtureKind,
    PilotDecision,
    score_persisted_decisions,
)
from ictc.pilot.benchmark import PilotOutcomeTruth, PotentialOutcomeTruth
from ictc.pilot.m14_decisions import _json_rows, _sha256, _write_jsonl


def _pilot_decision(row: dict[str, Any]) -> PilotDecision:
    executable = row["executable_action"]
    return PilotDecision(
        opportunity_id=str(row["opportunity_id"]),
        pair_id=str(row["pair_id"]),
        scenario_id=str(row["scenario_id"]),
        split=DatasetSplit(str(row["split"])),
        test_authorization_id=cast(str | None, row["test_authorization_id"]),
        ablation_id=AblationId(str(row["ablation_id"])),
        configuration_id=str(row["configuration_id"]),
        fixture=FixtureKind(str(row["fixture"])),
        decision=AdmissionDecision(str(row["decision"])),
        executable_action=ActionKind(str(executable)) if executable is not None else None,
        reason_codes=tuple(str(value) for value in row["reason_codes"]),
        reference_identified=bool(row["reference_identified"]),
        model_identified=bool(row["model_identified"]),
        reference_contract_satisfied=bool(row["reference_contract_satisfied"]),
        contract_satisfied=bool(row["contract_satisfied"]),
        estimable=bool(row["estimable"]),
        estimate_point_ms=cast(float | None, row["estimate_point_ms"]),
        estimate_lower_ms=cast(float | None, row["estimate_lower_ms"]),
        estimate_upper_ms=cast(float | None, row["estimate_upper_ms"]),
        telemetry_items=int(row["telemetry_items"]),
        telemetry_bytes=int(row["telemetry_bytes"]),
        contract_acquisition_latency_ms=float(row["contract_acquisition_latency_ms"]),
        protocol_messages=int(row["protocol_messages"]),
        admission_wall_time_ns=int(row["admission_wall_time_ns"]),
        admission_cpu_time_ns=int(row["admission_cpu_time_ns"]),
        breach_detection_time_ns=cast(int | None, row["breach_detection_time_ns"]),
        fallback_execution_time_ns=cast(int | None, row["fallback_execution_time_ns"]),
        breach_detected=bool(row["breach_detected"]),
        fallback_executed=bool(row["fallback_executed"]),
    )


def _truth(outcomes: dict[str, Any], pair_id: str, pressure: int) -> PotentialOutcomeTruth:
    def arm(name: str, route_changes: int) -> PilotOutcomeTruth:
        value = cast(dict[str, Any], outcomes[name])
        loss = float(value["loss_ratio"])
        return PilotOutcomeTruth(
            mean_latency_ms=float(value["latency_ms"]),
            packet_delivery_ratio=1.0 - loss,
            packet_loss_ratio=loss,
            jitter_ms=float(value["jitter_ms"]),
            # The immutable M15 simulation summary does not contain generated and
            # delivered packet counts.  Zero is a construction-only sentinel; the
            # serialized scores replace it with null and M16 prohibits throughput
            # inference from this study.
            throughput_packets_per_tick=0.0,
            deadline_violation_rate=float(value["deadline_violation_rate"]),
            route_changes=route_changes,
        )

    return PotentialOutcomeTruth(pair_id, pressure, arm("keep", 0), arm("reroute", 1))


def _pressure(row: dict[str, Any]) -> int:
    samples = {
        str(value["path_id"]): value
        for value in row["samples"]
        if not value["missing"] and value["queue_occupancy"] is not None
    }
    if set(samples) != {"path-current", "path-candidate"}:
        return 0
    difference = int(samples["path-current"]["queue_occupancy"]) - int(
        samples["path-candidate"]["queue_occupancy"]
    )
    return -1 if difference < 0 else 1 if difference > 0 else 0


def _write_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite scoring evidence: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def score_m14_decisions(
    *, telemetry_path: Path, raw_root: Path, decision_root: Path
) -> dict[str, object]:
    """Score persisted decisions without changing or excluding any held-out row."""
    pretruth_path = decision_root / "pretruth_audit.json"
    oracle_audit_path = decision_root / "oracle_access_audit.json"
    pretruth = json.loads(pretruth_path.read_text(encoding="utf-8"))
    oracle_audit = json.loads(oracle_audit_path.read_text(encoding="utf-8"))
    if pretruth.get("outcome_scoring_started") is not False:
        raise PermissionError("M14 scoring has already started")
    if oracle_audit.get("outcome_scoring_started") is not False:
        raise PermissionError("B0 audit does not precede scoring")
    telemetry_rows = tuple(
        row for row in _json_rows(telemetry_path) if row["topology_split"] == "test"
    )
    telemetry_by_id = {str(row["scenario_id"]): row for row in telemetry_rows}
    truth_by_pair: dict[str, PotentialOutcomeTruth] = {}
    conditions: dict[str, str] = {}
    for pair_id, telemetry in telemetry_by_id.items():
        outcome = cast(
            dict[str, Any],
            json.loads(
                (raw_root / "simulation" / pair_id / "outcomes.json").read_text(encoding="utf-8")
            ),
        )
        truth_by_pair[pair_id] = _truth(outcome, pair_id, _pressure(telemetry))
        conditions[pair_id] = str(telemetry["condition_id"])

    ablation_path = decision_root / "ablation_decisions.jsonl"
    decisions = tuple(_pilot_decision(row) for row in _json_rows(ablation_path))
    ablation_scores = score_persisted_decisions(decisions, truth_by_pair)
    ablation_score_rows_list: list[dict[str, object]] = []
    reference_admissible_by_pair: dict[str, bool] = {}
    for decision, score in zip(decisions, ablation_scores, strict=True):
        row = cast(dict[str, object], asdict(score))
        row["condition_id"] = conditions[score.pair_id]
        # M15 did not retain packet accounting needed to derive throughput.
        row["selected_throughput_packets_per_tick"] = None
        ablation_score_rows_list.append(row)
        if decision.ablation_id is AblationId.A0:
            reference_admissible_by_pair[decision.pair_id] = (
                decision.reference_identified and decision.reference_contract_satisfied
            )
    if set(reference_admissible_by_pair) != set(truth_by_pair):
        raise ValueError("one A0 reference-contract decision is required per held-out pair")
    ablation_score_rows = tuple(ablation_score_rows_list)
    ablation_score_hash = _write_jsonl(decision_root / "ablation_scores.jsonl", ablation_score_rows)

    baseline_rows = (
        *_json_rows(decision_root / "baseline_b0_oracle_decisions.jsonl"),
        *_json_rows(decision_root / "baseline_b1_b8_decisions.jsonl"),
    )
    baseline_scores: list[dict[str, object]] = []
    for baseline_decision in baseline_rows:
        pair_id = str(baseline_decision["pair_id"])
        truth = truth_by_pair[pair_id]
        action = baseline_decision["selected_action"]
        rerouted = action == ActionKind.REROUTE_TO_PATH.value
        # Apply the same frozen reference contract used by A0 instead of
        # reconstructing admissibility from condition names after seeing truth.
        reference_admissible = reference_admissible_by_pair[pair_id]
        selected = truth.reroute if rerouted else truth.keep if action else None
        baseline_scores.append(
            {
                "baseline_id": baseline_decision["baseline_id"],
                "pair_id": pair_id,
                "condition_id": conditions[pair_id],
                "executed_reroute": rerouted,
                "false_admission": rerouted and not reference_admissible,
                "unnecessary_deferral": (
                    not rerouted and reference_admissible and truth.reroute_beneficial
                ),
                "harmful_action": rerouted and not truth.reroute_beneficial,
                "selected_latency_ms": selected.mean_latency_ms if selected else None,
                "selected_packet_loss_ratio": selected.packet_loss_ratio if selected else None,
                "selected_jitter_ms": selected.jitter_ms if selected else None,
                "selected_deadline_violation_rate": (
                    selected.deadline_violation_rate if selected else None
                ),
                "selected_throughput_packets_per_tick": None,
                "selected_route_changes": selected.route_changes if selected else None,
            }
        )
    baseline_score_hash = _write_jsonl(
        decision_root / "baseline_scores.jsonl", tuple(baseline_scores)
    )
    audit = {
        "schema_version": "1.0.0",
        "heldout_truth_rows_read": len(truth_by_pair),
        "ablation_score_count": len(ablation_scores),
        "baseline_score_count": len(baseline_scores),
        "excluded_rows": 0,
        "negative_outcomes_retained": True,
        "throughput_measured": False,
        "throughput_policy": (
            "not estimable from immutable M15 summaries; serialized as null and "
            "excluded from M16 inference"
        ),
        "ablation_decision_sha256": _sha256(ablation_path),
        "ablation_score_sha256": ablation_score_hash,
        "baseline_score_sha256": baseline_score_hash,
        "pretruth_audit_sha256": _sha256(pretruth_path),
        "oracle_access_audit_sha256": _sha256(oracle_audit_path),
    }
    _write_json(decision_root / "scoring_audit.json", audit)
    return audit
