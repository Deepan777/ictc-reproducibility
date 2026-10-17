"""Decision-before-truth execution for the authorized M14 held-out study."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from ictc.baselines import OracleChoice, OracleController, load_frozen_baseline_suite
from ictc.baselines.types import BaselineInput, PathSnapshot
from ictc.datasets import DatasetSplit
from ictc.domain.enums import ActionKind
from ictc.domain.models import NetworkAction
from ictc.pilot.ablation import (
    FixtureKind,
    PilotOpportunity,
    build_opportunities,
    fit_pilot_model,
    run_ablation_suite,
)
from ictc.pilot.authorization import TestAccessAuthorization
from ictc.pilot.benchmark import (
    ASSIGNMENT_POLICY_ID,
    ObservationalDecision,
    PreActionEvidence,
    reference_identification,
)
from ictc.pilot.m14_authorization import M14ControllerAuthorization
from ictc.pilot.protocol import load_frozen_protocol


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _json_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path.name} contains a non-object row")
        rows.append(cast(dict[str, Any], value))
    return tuple(rows)


def _write_jsonl(path: Path, rows: tuple[dict[str, object], ...]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite decision evidence: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
    return _sha256(path)


def _pressure_bin(current: int, candidate: int) -> int:
    difference = current - candidate
    return -1 if difference < 0 else 1 if difference > 0 else 0


def _assignment(pair_id: str, pressure_bin: int) -> tuple[ActionKind, float, float]:
    probability = {-1: 0.25, 0: 0.5, 1: 0.75}[pressure_bin]
    integer = int.from_bytes(
        hashlib.sha256(f"{ASSIGNMENT_POLICY_ID}\0{pair_id}".encode()).digest()[:8], "big"
    )
    uniform = integer / 2**64
    action = ActionKind.REROUTE_TO_PATH if uniform < probability else ActionKind.KEEP_CURRENT_PATH
    return action, probability, uniform


def _evidence(row: dict[str, Any]) -> PreActionEvidence:
    samples = [sample for sample in row["samples"] if not sample["missing"]]
    by_path = {str(sample["path_id"]): sample for sample in samples}
    expected = {"path-current", "path-candidate"}
    sources = tuple(sorted(str(sample["source_id"]) for sample in samples))
    digests = tuple(sorted(str(sample["provenance_digest"]) for sample in samples))
    if set(by_path) != expected or any(
        by_path[path]["queue_occupancy"] is None or by_path[path]["utilization"] is None
        for path in expected
    ):
        return PreActionEvidence(
            None, None, None, None, None, sources, digests, len(samples), 64 * len(samples)
        )
    current = by_path["path-current"]
    candidate = by_path["path-candidate"]
    delivered = max(int(current["available_tick"]), int(candidate["available_tick"]))
    observed = (int(current["observed_timestamp_tick"]), int(candidate["observed_timestamp_tick"]))
    return PreActionEvidence(
        _pressure_bin(int(current["queue_occupancy"]), int(candidate["queue_occupancy"])),
        float(current["utilization"]) - float(candidate["utilization"]),
        delivered,
        max(int(current["sample_age_ticks"]), int(candidate["sample_age_ticks"])),
        max(observed) - min(observed),
        sources,
        digests,
        2,
        128,
    )


def _outcome_path(raw_root: Path, scenario_id: str) -> Path:
    return raw_root / "simulation" / scenario_id / "outcomes.json"


def _observational_row(
    row: dict[str, Any], *, raw_root: Path, allow_outcome: bool
) -> ObservationalDecision:
    evidence = _evidence(row)
    pressure = evidence.queue_pressure_bin or 0
    action, probability, uniform = _assignment(str(row["scenario_id"]), pressure)
    observed_latency: float | None = None
    if allow_outcome:
        outcomes = json.loads(
            _outcome_path(raw_root, str(row["scenario_id"])).read_text(encoding="utf-8")
        )
        arm = "reroute" if action is ActionKind.REROUTE_TO_PATH else "keep"
        observed_latency = float(outcomes[arm]["latency_ms"])
    split = DatasetSplit(str(row["topology_split"]))
    return ObservationalDecision(
        str(row["scenario_id"]),
        str(row["scenario_id"]),
        split,
        int(row["seed"]),
        str(row["topology_family_id"]),
        str(row["traffic_id"]),
        str(row["condition_id"]),
        action,
        observed_latency,
        evidence.queue_pressure_bin,
        evidence,
        reference_identification(evidence.queue_pressure_bin is not None),
        probability,
        uniform,
    )


def _baseline_input(row: dict[str, Any]) -> BaselineInput:
    snapshots: list[PathSnapshot] = []
    for sample in row["samples"]:
        if sample["missing"] or sample["queue_occupancy"] is None or sample["utilization"] is None:
            continue
        snapshots.append(
            PathSnapshot(
                path_id=str(sample["path_id"]),
                sample_tick=int(sample["sample_tick"]),
                available_tick=int(sample["available_tick"]),
                observed_timestamp_tick=int(sample["observed_timestamp_tick"]),
                queue_occupancy=int(sample["queue_occupancy"]),
                utilization=float(sample["utilization"]),
                sample_age_ticks=int(sample["sample_age_ticks"]),
                measurement_delay_ticks=int(sample["measurement_delay_ticks"]),
                clock_skew_ticks=int(sample["clock_skew_ticks"]),
                stale=bool(sample["stale"]),
                source_id=str(sample["source_id"]),
                provenance_digest=str(sample["provenance_digest"]),
            )
        )
    scenario_id = str(row["scenario_id"])
    return BaselineInput(
        scenario_id,
        scenario_id,
        DatasetSplit.TEST,
        int(row["seed"]),
        int(row["decision_tick"]),
        "path-current",
        "path-candidate",
        NetworkAction(
            action_id=f"m14:{scenario_id}:reroute",
            kind=ActionKind.REROUTE_TO_PATH,
            path_id="path-candidate",
        ),
        tuple(snapshots),
        str(row["config_sha256"]),
        str(row["schedule_sha256"]),
    )


def _baseline_payload(result: object) -> dict[str, object]:
    value = cast(Any, result)
    selected = value.selected_action
    return {
        "baseline_id": value.baseline_id.value,
        "pair_id": value.pair_id,
        "scenario_id": value.scenario_id,
        "disposition": value.disposition.value,
        "reason": value.reason.value,
        "selected_action": selected.kind.value if selected is not None else None,
        "telemetry_items": value.telemetry_items,
        "telemetry_bytes": value.telemetry_bytes,
        "inference_time_ns": value.inference_time_ns,
        "configuration_digest": value.configuration_digest,
        "score": value.score,
    }


def _controller_fixture(row: dict[str, Any]) -> FixtureKind:
    """Map only controller-visible verification results to a contract fixture."""
    if row.get("contract_feasible") is False:
        return FixtureKind.CONTRACT_INFEASIBLE
    samples = cast(list[dict[str, Any]], row["samples"])
    if any(sample.get("provenance_valid") is False for sample in samples):
        return FixtureKind.INVALID_PROVENANCE
    return FixtureKind.OBSERVED


def run_m14_decisions(
    *,
    repository: Path,
    telemetry_path: Path,
    authorization: M14ControllerAuthorization,
    baseline_config: Path,
    protocol_path: Path,
) -> dict[str, object]:
    """Fit permitted methods and persist held-out decisions before any scoring."""
    rows = _json_rows(telemetry_path)
    test_rows = tuple(row for row in rows if row["topology_split"] == "test")
    if len(test_rows) != authorization.expected_test_pair_count:
        raise ValueError("held-out telemetry count differs from the authorization")
    training_root = authorization.training_matrix_root or authorization.raw_matrix_root
    training_rows = (
        rows
        if training_root == authorization.raw_matrix_root
        else _json_rows(training_root / "controller-safe-telemetry.jsonl")
    )
    training = tuple(
        _observational_row(row, raw_root=training_root, allow_outcome=True)
        for row in training_rows
        if row["topology_split"] == "train"
    )
    model = fit_pilot_model(training)
    test_decisions = tuple(
        _observational_row(row, raw_root=authorization.raw_matrix_root, allow_outcome=False)
        for row in test_rows
    )
    observed_opportunities = build_opportunities(
        test_decisions,
        include_controlled_faults=False,
        test_authorization=cast(TestAccessAuthorization, authorization),
    )
    opportunities = tuple(
        PilotOpportunity(item.opportunity_id, item.decision, _controller_fixture(row))
        for item, row in zip(observed_opportunities, test_rows, strict=True)
    )
    protocol = load_frozen_protocol(protocol_path).operational
    ablations = run_ablation_suite(
        opportunities,
        model,
        protocol,
        cast(TestAccessAuthorization, authorization),
    )
    output = authorization.decision_output_root
    output.mkdir(parents=True)
    ablation_hash = _write_jsonl(
        output / "ablation_decisions.jsonl",
        tuple(cast(dict[str, object], asdict(item)) for item in ablations),
    )
    dummy = OracleController((OracleChoice("authorization-dummy", False, 0.0, 0.0, 1),))
    suite = load_frozen_baseline_suite(baseline_config, oracle=dummy)
    baseline_rows = tuple(
        _baseline_payload(controller.decide(_baseline_input(row)))
        for row in test_rows
        for controller in suite.controllers[1:9]
    )
    baseline_hash = _write_jsonl(output / "baseline_b1_b8_decisions.jsonl", baseline_rows)
    audit = {
        "schema_version": "1.0.0",
        "authorization_id": authorization.authorization_id,
        "training_outcome_rows_read": len(training),
        "training_matrix_root": str(training_root),
        "development_outcome_rows_read": 0,
        "heldout_outcome_rows_read_before_decision_persistence": 0,
        "heldout_pair_count": len(test_rows),
        "ablation_decision_count": len(ablations),
        "baseline_b1_b8_decision_count": len(baseline_rows),
        "ablation_sha256": ablation_hash,
        "baseline_sha256": baseline_hash,
        "telemetry_sha256": _sha256(telemetry_path),
        "outcome_scoring_started": False,
    }
    audit_path = output / "pretruth_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit
