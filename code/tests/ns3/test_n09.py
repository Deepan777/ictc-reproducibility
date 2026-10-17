"""Tests for the frozen N09 fresh-seed contextual-support evaluation."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n08 import ChangeProbeEvidence
from ictc.ns3.n09 import (
    N09DevelopmentRunner,
    N09MethodId,
    aggregate_n09_scores,
    build_n09_generations,
    load_n09_matrix,
)
from ictc.ns3.stochastic import EventRegime, PathAction, StochasticScenario, WorkloadRegime, sha256

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n09_fresh_support_development.json"
DESIGN = ROOT / "docs" / "milestones" / "evidence" / "n09_context_support_design.json"
SUMMARY_FIELDS = [
    "scenario_id",
    "topology_family",
    "workload",
    "event",
    "action",
    "seed",
    "run",
    "tx_packets",
    "rx_packets",
    "lost_packets",
    "mean_delay_ns",
    "jitter_ns",
    "throughput_bps",
    "first_tx_ns",
    "last_rx_ns",
]


def _probe(
    scenario: StochasticScenario,
    *,
    current_up: bool = True,
    current_rate: int = 5_000_000,
    candidate_rate: int = 5_000_000,
) -> ChangeProbeEvidence:
    return ChangeProbeEvidence(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        path_ids=("current", "candidate"),
        latest_timestamp_ns=3_700_000_000,
        latest_queues={"current": 0, "candidate": 0},
        latest_link_up={"current": current_up, "candidate": True},
        latest_rate_bps={"current": current_rate, "candidate": candidate_rate},
        pre_decision_digest="1" * 64,
        state_digest="2" * 64,
        observation_digest="3" * 64,
        event_detected=True,
        telemetry_sha256="4" * 64,
        state_sha256="5" * 64,
        background_sha256="6" * 64,
    )


def _build(scenario: StochasticScenario, probe: ChangeProbeEvidence) -> dict[str, Any]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    return build_n09_generations(
        scenario,
        probe,
        "a" * 64,
        support_design_digest=matrix["support_design_digest"],
        support_design_file_sha256=matrix["support_design_file_sha256"],
        source_n08_manifest_sha256=matrix["source_n08_manifest_sha256"],
        supported_contexts=tuple(matrix["supported_contexts"]),
    )


def _write_raw(raw: Path, scenario: StochasticScenario, action: PathAction, *, probe: bool) -> None:
    raw.mkdir(parents=True)
    with (raw / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        if not probe:
            writer.writerow(
                {
                    "scenario_id": scenario.scenario_id,
                    "topology_family": scenario.topology_family.value,
                    "workload": scenario.workload.value,
                    "event": scenario.event.value,
                    "action": action.value,
                    "seed": scenario.seed,
                    "run": scenario.run,
                    "tx_packets": 100,
                    "rx_packets": 100,
                    "lost_packets": 0,
                    "mean_delay_ns": 5_000_000 if action is PathAction.KEEP else 4_000_000,
                    "jitter_ns": 1,
                    "throughput_bps": 1,
                    "first_tx_ns": 3_800_000_000,
                    "last_rx_ns": 5_000_000_000,
                }
            )
    with (raw / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "queue_packets"]
        )
        writer.writeheader()
        for path in ("current", "candidate"):
            writer.writerow(
                {
                    "timestamp_ns": 3_700_000_000,
                    "phase": "PRE_DECISION",
                    "path_id": path,
                    "queue_packets": 0,
                }
            )
    with (raw / "state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "link_up", "rate_bps"]
        )
        writer.writeheader()
        for path in ("current", "candidate"):
            writer.writerow(
                {
                    "timestamp_ns": 3_700_000_000,
                    "phase": "PRE_DECISION",
                    "path_id": path,
                    "link_up": 1,
                    "rate_bps": 5_000_000,
                }
            )
    with (raw / "background.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path_id",
                "flow_id",
                "tx_packets",
                "rx_packets",
                "lost_packets",
                "mean_delay_ns",
            ],
        )
        writer.writeheader()
        for index, path in enumerate(("current", "candidate"), start=1):
            writer.writerow(
                {
                    "path_id": path,
                    "flow_id": index,
                    "tx_packets": 10,
                    "rx_packets": 10,
                    "lost_packets": 0,
                    "mean_delay_ns": 1,
                }
            )
    (raw / "parameters.json").write_text(
        json.dumps(
            {
                "scenario_id": scenario.scenario_id,
                "topology_family": scenario.topology_family.value,
                "workload": scenario.workload.value,
                "event": scenario.event.value,
                "action": action.value,
                "seed": scenario.seed,
                "run": scenario.run,
                "probe_only": probe,
            }
        ),
        encoding="utf-8",
    )


def _runner(tmp_path: Path) -> N09DevelopmentRunner:
    tmp_path.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "source-n08"
    source.mkdir()
    (source / "n08_manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
    design = tmp_path / "design.json"
    design_payload = json.loads(DESIGN.read_text(encoding="utf-8"))
    design.write_text(json.dumps(design_payload), encoding="utf-8")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["support_design_file_sha256"] = sha256(design)
    matrix["source_n08_manifest_sha256"] = sha256(source / "n08_manifest.json")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    return N09DevelopmentRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=matrix_path,
        executable=executable,
        runtime_directories=(tmp_path,),
        support_design_path=design,
        source_n08_root=source,
    )


def _mock_native(scenario: StochasticScenario) -> Any:
    def execute(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        values = dict(item[2:].split("=", 1) for item in command[1:])
        _write_raw(
            Path(values["output"]),
            scenario,
            PathAction(values["action"]),
            probe=values.get("probeOnly") == "true",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    return execute


def test_n09_matrix_is_complete_fresh_and_reserved_safe(tmp_path: Path) -> None:
    scenarios = load_n09_matrix(MATRIX)
    assert len(scenarios) == 288
    assert sum(len(item.actions) for item in scenarios) == 672
    assert {item.seed for item in scenarios} == set(range(81100, 81108))
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["seeds"] = list(range(82000, 82008))
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen set"):
        load_n09_matrix(invalid)


def test_n09_matrix_rejects_all_frozen_declaration_defects(tmp_path: Path) -> None:
    base = json.loads(MATRIX.read_text(encoding="utf-8"))
    variants: list[tuple[str, object, str]] = [
        ("schema_version", "wrong", "schema"),
        ("reserved_confirmatory_seeds", [82000], "reservation"),
        ("topology_families", [], "topology"),
        ("workloads", [], "workload"),
        ("events", [], "event"),
        ("controls", None, "controls are missing"),
        ("expected_scenario_groups", 1, "scenario count"),
        ("expected_action_executions", 1, "action count"),
        ("supported_contexts", [], "supported contexts"),
        ("support_design_digest", "short", "support_design_digest"),
    ]
    for index, (field, value, match) in enumerate(variants):
        payload = json.loads(json.dumps(base))
        payload[field] = value
        path = tmp_path / f"variant-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_n09_matrix(path)
    controls = json.loads(json.dumps(base))
    controls["controls"].pop("duration_seconds")
    path = tmp_path / "controls.json"
    path.write_text(json.dumps(controls), encoding="utf-8")
    with pytest.raises(ValueError, match="controls are invalid"):
        load_n09_matrix(path)
    timing = json.loads(json.dumps(base))
    timing["controls"]["decision_at_seconds"] = 2.0
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(timing), encoding="utf-8")
    with pytest.raises(ValueError, match="event-before-redecision"):
        load_n09_matrix(path)


def test_d1_applies_only_supported_healthy_context_and_hash_chains() -> None:
    scenario = load_n09_matrix(MATRIX)[0]
    payload = _build(scenario, _probe(scenario))
    generations = payload["generations"]
    assert generations[1]["support_applied"] is True
    assert generations[1]["previous_generation_digest"] == generations[0]["recorded_digest"]
    decisions = {row["method_id"]: row for row in generations[1]["decisions"]}
    assert (
        decisions[N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value]["selected_action"] == "reroute"
    )
    assert decisions[N09MethodId.B3_CAPACITY.value]["selected_action"] == "keep"
    assert not any(payload["forbidden_inputs"].values())

    unsupported = replace(scenario, workload=WorkloadRegime.BURSTY)
    decision = _build(unsupported, _probe(unsupported))["generations"][1]
    assert decision["support_applied"] is False
    assert decision["decisions"][0]["selected_action"] == "keep"


def test_d1_falls_back_to_capacity_on_failure_and_rejects_bad_binding() -> None:
    scenario = replace(load_n09_matrix(MATRIX)[0], event=EventRegime.CURRENT_FAILURE)
    probe = _probe(scenario, current_up=False)
    generation = _build(scenario, probe)["generations"][1]
    assert generation["support_applied"] is False
    assert generation["decisions"][0]["selected_action"] == "reroute"
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="binding"):
        build_n09_generations(
            scenario,
            replace(probe, scenario_id="wrong"),
            "a" * 64,
            support_design_digest=matrix["support_design_digest"],
            support_design_file_sha256=matrix["support_design_file_sha256"],
            source_n08_manifest_sha256=matrix["source_n08_manifest_sha256"],
            supported_contexts=tuple(matrix["supported_contexts"]),
        )


def test_runner_persists_fresh_redecision_before_outcomes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    scenario = runner.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    assert runner.run(limit=1) == {
        "scenario_groups": 1,
        "method_decisions": 6,
        "outcome_actions": 2,
        "reserved_confirmatory_seeds_executed": 0,
    }
    decision = runner.output_root / "decisions" / scenario.scenario_id / "redecision.json"
    for action in scenario.actions:
        manifest = json.loads(
            (
                runner.action_runner.output_root
                / "scenarios"
                / scenario.scenario_id
                / action.value
                / "action_manifest.json"
            ).read_text(encoding="utf-8")
        )
        assert manifest["decision_manifest_sha256"] == sha256(decision)
    aggregate = aggregate_n09_scores(runner.output_root, (scenario,))
    assert aggregate["support_opportunities"] == {
        "applied": 1,
        "beneficial": 1,
        "harmful": 0,
        "other": 0,
    }


def test_runner_rejects_freeze_and_redecision_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _runner(tmp_path / "frozen")
    frozen.initialise()
    manifest = frozen.output_root / "n09_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different N09 freeze"):
        frozen.initialise()

    decision_tamper = _runner(tmp_path / "decision")
    scenario = decision_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    decision_tamper.run(limit=1)
    decision = decision_tamper.output_root / "decisions" / scenario.scenario_id / "redecision.json"
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["current_generation"] = 0
    decision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="redecision differs"):
        decision_tamper.run(limit=1)


def test_validator_rejects_incomplete_fresh_matrix(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="matrix is incomplete"):
        _runner(tmp_path).validate()
