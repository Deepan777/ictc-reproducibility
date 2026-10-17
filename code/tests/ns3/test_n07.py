"""Tests for N07 probe-first decision persistence and scoring."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n07 import (
    MethodId,
    N07DevelopmentRunner,
    aggregate_development_scores,
    build_method_decisions,
    read_probe_evidence,
)
from ictc.ns3.stochastic import PathAction, StochasticScenario, load_stochastic_matrix, sha256

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n06_stochastic_ns3_development.json"
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


def _scenario() -> StochasticScenario:
    return load_stochastic_matrix(MATRIX)[0]


def _telemetry(raw: Path, queue: int, *, post: bool = False) -> None:
    with (raw / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "queue_packets"]
        )
        writer.writeheader()
        for path in ("current", "candidate"):
            writer.writerow(
                {
                    "timestamp_ns": 1_900_000_000,
                    "phase": "POST_DECISION" if post else "PRE_DECISION",
                    "path_id": path,
                    "queue_packets": queue,
                }
            )


def _background(raw: Path) -> None:
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


def _parameters(raw: Path, scenario: StochasticScenario, action: PathAction, probe: bool) -> None:
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


def _write_probe(
    raw: Path, scenario: StochasticScenario, *, queue: int = 0, post: bool = False
) -> None:
    raw.mkdir(parents=True)
    with (raw / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS).writeheader()
    _telemetry(raw, queue, post=post)
    _background(raw)
    _parameters(raw, scenario, PathAction.KEEP, True)


def _write_action(
    raw: Path, scenario: StochasticScenario, action: PathAction, *, queue: int = 0
) -> None:
    raw.mkdir(parents=True)
    with (raw / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
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
                "first_tx_ns": 2_000_000_000,
                "last_rx_ns": 3_000_000_000,
            }
        )
    _telemetry(raw, queue)
    _background(raw)
    _parameters(raw, scenario, action, False)


def _runner(tmp_path: Path) -> N07DevelopmentRunner:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    return N07DevelopmentRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=MATRIX,
        executable=executable,
        runtime_directories=(tmp_path,),
    )


def _mock_native(scenario: StochasticScenario, *, action_queue: int = 0) -> Any:
    def execute(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        values = dict(item[2:].split("=", 1) for item in command[1:])
        raw = Path(values["output"])
        if values.get("probeOnly") == "true":
            _write_probe(raw, scenario)
        else:
            _write_action(raw, scenario, PathAction(values["action"]), queue=action_queue)
        return subprocess.CompletedProcess(command, 0, "", "")

    return execute


def test_probe_reader_exposes_only_complete_pre_decision_state(tmp_path: Path) -> None:
    scenario = _scenario()
    _write_probe(tmp_path / "probe", scenario, queue=3)
    evidence = read_probe_evidence(tmp_path / "probe", scenario)
    assert evidence.latest_queues == {"current": 3, "candidate": 3}
    assert len(evidence.pre_decision_digest) == 64

    _write_probe(tmp_path / "post", scenario, post=True)
    with pytest.raises(ValueError, match="post-decision"):
        read_probe_evidence(tmp_path / "post", scenario)


def test_method_decisions_are_deterministic_and_outcome_free(tmp_path: Path) -> None:
    scenario = _scenario()
    _write_probe(tmp_path / "probe", scenario)
    evidence = read_probe_evidence(tmp_path / "probe", scenario)
    payload = build_method_decisions(scenario, evidence, "a" * 64)
    decisions = {item["method_id"]: item["selected_action"] for item in payload["decisions"]}
    assert decisions[MethodId.C0_CONTRACT_QUEUE.value] == "reroute"
    assert decisions[MethodId.B0_KEEP.value] == "keep"
    assert decisions[MethodId.B3_CAPACITY.value] == "keep"
    assert not any(payload["forbidden_inputs"].values())
    assert "physical_label" not in json.dumps(payload)


def test_runner_persists_decisions_before_byte_bound_outcomes(
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
    decisions = runner.output_root / "decisions" / scenario.scenario_id / "decisions.json"
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
        assert manifest["decision_manifest_sha256"] == sha256(decisions)
    aggregate = aggregate_development_scores(runner.output_root, (scenario,))
    assert aggregate["scenario_groups"] == 1
    assert aggregate["method_counts"][MethodId.C0_CONTRACT_QUEUE.value]["BENEFICIAL"] == 1


def test_runner_rejects_probe_action_pre_decision_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr(subprocess, "run", _mock_native(runner.scenarios[0], action_queue=1))
    with pytest.raises(ValueError, match="probe and action"):
        runner.run(limit=1)


def test_probe_and_method_input_failures_are_explicit(tmp_path: Path) -> None:
    scenario = _scenario()
    raw = tmp_path / "probe"
    _write_probe(raw, scenario)
    with (raw / "summary.csv").open("a", newline="", encoding="utf-8") as handle:
        handle.write("foreground,row\n")
    with pytest.raises(ValueError, match="no foreground"):
        read_probe_evidence(raw, scenario)

    valid = tmp_path / "valid"
    _write_probe(valid, scenario)
    evidence = read_probe_evidence(valid, scenario)
    with pytest.raises(ValueError, match="binding"):
        build_method_decisions(scenario, evidence, "short")
    evidence.latest_queues.pop("candidate")
    with pytest.raises(ValueError, match="contract"):
        build_method_decisions(scenario, evidence, "a" * 64)


def test_probe_reader_rejects_incomplete_identity_and_schema_variants(tmp_path: Path) -> None:
    scenario = _scenario()
    incomplete = tmp_path / "incomplete"
    _write_probe(incomplete, scenario)
    (incomplete / "background.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        read_probe_evidence(incomplete, scenario)

    declaration = tmp_path / "declaration"
    _write_probe(declaration, scenario)
    payload = json.loads((declaration / "parameters.json").read_text(encoding="utf-8"))
    payload["probe_only"] = False
    (declaration / "parameters.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="declaration"):
        read_probe_evidence(declaration, scenario)

    identity = tmp_path / "identity"
    _write_probe(identity, scenario)
    payload = json.loads((identity / "parameters.json").read_text(encoding="utf-8"))
    payload["seed"] = 81003
    (identity / "parameters.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_probe_evidence(identity, scenario)

    value = tmp_path / "value"
    _write_probe(value, scenario)
    text = (value / "telemetry.csv").read_text(encoding="utf-8").replace("1900000000", "invalid", 1)
    (value / "telemetry.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry value"):
        read_probe_evidence(value, scenario)

    layout = tmp_path / "layout"
    _write_probe(layout, scenario)
    rows = list(csv.DictReader((layout / "telemetry.csv").open(newline="", encoding="utf-8")))
    with (layout / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "queue_packets"]
        )
        writer.writeheader()
        writer.writerow(rows[0])
    with pytest.raises(ValueError, match="one ordered latest row"):
        read_probe_evidence(layout, scenario)

    background = tmp_path / "background"
    _write_probe(background, scenario)
    text = (
        (background / "background.csv")
        .read_text(encoding="utf-8")
        .replace("current,1", "unknown,1")
    )
    (background / "background.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="background"):
        read_probe_evidence(background, scenario)


def test_runner_retains_probe_process_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonzero = _runner(tmp_path / "nonzero")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 4, "", "bad"),
    )
    with pytest.raises(RuntimeError, match="failure retained"):
        nonzero.run(limit=1)
    assert next((nonzero.output_root / "failures").rglob("failure.json")).is_file()

    timeout = _runner(tmp_path / "timeout")

    def raise_timeout(command: list[str], **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 120)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    with pytest.raises(RuntimeError, match="timeout retained"):
        timeout.run(limit=1)

    invalid = _runner(tmp_path / "invalid")

    def invalid_probe(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        values = dict(item[2:].split("=", 1) for item in command[1:])
        _write_probe(Path(values["output"]), invalid.scenarios[0], post=True)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", invalid_probe)
    with pytest.raises(RuntimeError, match="invalid probe retained"):
        invalid.run(limit=1)


def test_runner_rejects_freeze_probe_decision_and_score_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _runner(tmp_path / "frozen")
    frozen.initialise()
    manifest = frozen.output_root / "n07_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different N07 freeze"):
        frozen.initialise()

    probe_tamper = _runner(tmp_path / "probe-tamper")
    scenario = probe_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    probe_tamper.run(limit=1)
    telemetry = probe_tamper.output_root / "probes" / scenario.scenario_id / "raw" / "telemetry.csv"
    telemetry.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="invalid N07 probe"):
        probe_tamper.run(limit=1)

    decision_tamper = _runner(tmp_path / "decision-tamper")
    scenario = decision_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    decision_tamper.run(limit=1)
    decision = decision_tamper.output_root / "decisions" / scenario.scenario_id / "decisions.json"
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["contract_ready"] = False
    decision.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="decision differs"):
        decision_tamper.run(limit=1)

    score_tamper = _runner(tmp_path / "score-tamper")
    scenario = score_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    score_tamper.run(limit=1)
    score = score_tamper.output_root / "scores" / scenario.scenario_id / "score.json"
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["development_only"] = False
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="score does not match"):
        score_tamper.run(limit=1)


def test_validator_rejects_incomplete_development_journal(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(ValueError, match="matrix is incomplete"):
        runner.validate(require_complete=True)


def test_n07_source_has_no_outcome_or_oracle_dependency_in_decision_builder() -> None:
    source = (ROOT / "src" / "ictc" / "ns3" / "n07.py").read_text(encoding="utf-8")
    decision_source = source.split("def build_method_decisions", 1)[1].split(
        "class N07DevelopmentRunner", 1
    )[0]
    for forbidden in ("ictc.oracle", "ictc.controller", "physical_benefit.json", "summary.csv"):
        assert forbidden not in decision_source
