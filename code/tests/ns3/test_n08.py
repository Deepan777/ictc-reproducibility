"""Regression tests for N08 post-change immutable redecision development."""

from __future__ import annotations

import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n07 import PATH_SPECS
from ictc.ns3.n08 import (
    N08DevelopmentRunner,
    N08MethodId,
    aggregate_n08_scores,
    build_redecision_generations,
    load_n08_matrix,
    read_change_probe,
)
from ictc.ns3.stochastic import EventRegime, PathAction, StochasticScenario, sha256

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n08_post_change_development.json"
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


def _write_raw(
    raw: Path,
    scenario: StochasticScenario,
    action: PathAction,
    *,
    probe: bool,
    queues: dict[str, int] | None = None,
    rates: dict[str, int] | None = None,
    link_up: dict[str, bool] | None = None,
) -> None:
    raw.mkdir(parents=True)
    paths = tuple(PATH_SPECS[scenario.topology_family.value])
    queues = queues or dict.fromkeys(paths, 0)
    rates = rates or {
        path: int(PATH_SPECS[scenario.topology_family.value][path][0] * 1_000_000) for path in paths
    }
    link_up = link_up or dict.fromkeys(paths, True)
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
        for path in paths:
            writer.writerow(
                {
                    "timestamp_ns": 3_700_000_000,
                    "phase": "PRE_DECISION",
                    "path_id": path,
                    "queue_packets": queues[path],
                }
            )
    with (raw / "state.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "link_up", "rate_bps"]
        )
        writer.writeheader()
        for path in paths:
            writer.writerow(
                {
                    "timestamp_ns": 3_700_000_000,
                    "phase": "PRE_DECISION",
                    "path_id": path,
                    "link_up": int(link_up[path]),
                    "rate_bps": rates[path],
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
        for index, path in enumerate(paths, start=1):
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


def _source_decision(scenario: StochasticScenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id.replace("n08-", "n06-", 1),
        "decisions": [
            {
                "method_id": "C0_CONTRACT_QUEUE",
                "selected_path": "candidate",
                "selected_action": "reroute",
            }
        ],
    }


def _runner(tmp_path: Path) -> N08DevelopmentRunner:
    scenario = load_n08_matrix(MATRIX)[0]
    source = tmp_path / "source-n07"
    source.mkdir(parents=True)
    (source / "n07_manifest.json").write_text('{"fixture":true}\n', encoding="utf-8")
    source_decision = source / "decisions" / scenario.scenario_id.replace("n08-", "n06-", 1)
    source_decision.mkdir(parents=True)
    (source_decision / "decisions.json").write_text(
        json.dumps(_source_decision(scenario)), encoding="utf-8"
    )
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["source_n07_manifest_sha256"] = sha256(source / "n07_manifest.json")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    return N08DevelopmentRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=matrix_path,
        executable=executable,
        runtime_directories=(tmp_path,),
        source_n07_root=source,
    )


def _mock_native(
    scenario: StochasticScenario, *, drift_state: bool = False, action_queue: int = 0
) -> Any:
    def execute(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        values = dict(item[2:].split("=", 1) for item in command[1:])
        probe = values.get("probeOnly") == "true"
        rates = None
        if drift_state and not probe:
            rates = {"current": 4_000_000, "candidate": 5_000_000}
        queues = None if probe else {"current": action_queue, "candidate": action_queue}
        _write_raw(
            Path(values["output"]),
            scenario,
            PathAction(values["action"]),
            probe=probe,
            queues=queues,
            rates=rates,
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    return execute


def test_n08_matrix_is_complete_and_reserved_seeds_are_blocked(tmp_path: Path) -> None:
    scenarios = load_n08_matrix(MATRIX)
    assert len(scenarios) == 144
    assert sum(len(item.actions) for item in scenarios) == 336
    assert {item.seed for item in scenarios} == {81000, 81001, 81002, 81003}
    assert all(item.event_at_seconds < item.decision_at_seconds for item in scenarios)

    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["seeds"] = [82000, 82001, 82002, 82003]
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized"):
        load_n08_matrix(invalid)


def test_n08_matrix_rejects_every_frozen_declaration_defect(tmp_path: Path) -> None:
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
        ("source_n07_manifest_sha256", "short", "binding"),
    ]
    for index, (field, value, match) in enumerate(variants):
        payload = json.loads(json.dumps(base))
        payload[field] = value
        path = tmp_path / f"variant-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_n08_matrix(path)
    missing_control = json.loads(json.dumps(base))
    missing_control["controls"].pop("duration_seconds")
    path = tmp_path / "missing-control.json"
    path.write_text(json.dumps(missing_control), encoding="utf-8")
    with pytest.raises(ValueError, match="controls are invalid"):
        load_n08_matrix(path)
    timing = json.loads(json.dumps(base))
    timing["controls"]["decision_at_seconds"] = 2.0
    path = tmp_path / "timing.json"
    path.write_text(json.dumps(timing), encoding="utf-8")
    with pytest.raises(ValueError, match="event-before-redecision"):
        load_n08_matrix(path)
    non_object = tmp_path / "list.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_n08_matrix(non_object)


def test_change_probe_detects_capacity_failure_and_cross_shift(tmp_path: Path) -> None:
    base = load_n08_matrix(MATRIX)[0]
    capacity = replace(base, event=EventRegime.CANDIDATE_CAPACITY_DROP)
    rates = {"current": 5_000_000, "candidate": 1_000_000}
    _write_raw(tmp_path / "capacity", capacity, PathAction.KEEP, probe=True, rates=rates)
    assert read_change_probe(tmp_path / "capacity", capacity).event_detected

    failure = replace(base, event=EventRegime.CURRENT_FAILURE)
    _write_raw(
        tmp_path / "failure",
        failure,
        PathAction.KEEP,
        probe=True,
        link_up={"current": False, "candidate": True},
    )
    assert read_change_probe(tmp_path / "failure", failure).event_detected

    shift = replace(base, event=EventRegime.CANDIDATE_CROSS_SHIFT)
    _write_raw(
        tmp_path / "shift",
        shift,
        PathAction.KEEP,
        probe=True,
        queues={"current": 0, "candidate": 100},
    )
    assert read_change_probe(tmp_path / "shift", shift).event_detected


def test_redecision_chain_uses_post_change_state_and_excludes_outcomes(tmp_path: Path) -> None:
    scenario = replace(load_n08_matrix(MATRIX)[0], event=EventRegime.CURRENT_FAILURE)
    raw = tmp_path / "probe"
    _write_raw(
        raw,
        scenario,
        PathAction.KEEP,
        probe=True,
        link_up={"current": False, "candidate": True},
    )
    probe = read_change_probe(raw, scenario)
    source = _source_decision(scenario)
    decision = build_redecision_generations(scenario, probe, "a" * 64, source, "b" * 64)
    generations = decision["generations"]
    assert generations[1]["previous_generation_digest"] == generations[0]["recorded_digest"]
    rows = {item["method_id"]: item for item in generations[1]["decisions"]}
    assert rows[N08MethodId.D0_CHANGE_RESYNTHESIS.value]["selected_action"] == "reroute"
    assert not any(decision["forbidden_inputs"].values())
    assert "physical_label" not in json.dumps(decision)


def test_runner_persists_redecision_before_bound_outcomes(
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
    aggregate = aggregate_n08_scores(runner.output_root, (scenario,))
    assert aggregate["scenario_groups"] == 1
    assert aggregate["event_detection"]["stable"] == {"detected": 1, "total": 1}


def test_runner_rejects_state_drift_and_invalid_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    drift = _runner(tmp_path / "drift")
    monkeypatch.setattr(subprocess, "run", _mock_native(drift.scenarios[0], drift_state=True))
    with pytest.raises(ValueError, match="state differs"):
        drift.run(limit=1)

    frozen = _runner(tmp_path / "frozen")
    frozen.initialise()
    manifest = frozen.output_root / "n08_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different N08 freeze"):
        frozen.initialise()


def test_probe_rejects_post_decision_state_and_source_binding(tmp_path: Path) -> None:
    scenario = load_n08_matrix(MATRIX)[0]
    raw = tmp_path / "probe"
    _write_raw(raw, scenario, PathAction.KEEP, probe=True)
    state = (
        (raw / "state.csv").read_text(encoding="utf-8").replace("PRE_DECISION", "POST_DECISION", 1)
    )
    (raw / "state.csv").write_text(state, encoding="utf-8")
    with pytest.raises(ValueError, match="post-decision state"):
        read_change_probe(raw, scenario)

    valid = tmp_path / "valid"
    _write_raw(valid, scenario, PathAction.KEEP, probe=True)
    probe = read_change_probe(valid, scenario)
    bad_source = _source_decision(scenario)
    bad_source["scenario_id"] = "wrong"
    with pytest.raises(ValueError, match="binding"):
        build_redecision_generations(scenario, probe, "a" * 64, bad_source, "b" * 64)


def test_probe_rejects_schema_value_layout_and_latest_sample_defects(tmp_path: Path) -> None:
    scenario = load_n08_matrix(MATRIX)[0]
    incomplete = tmp_path / "incomplete"
    _write_raw(incomplete, scenario, PathAction.KEEP, probe=True)
    (incomplete / "state.csv").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        read_change_probe(incomplete, scenario)

    outcome = tmp_path / "outcome"
    _write_raw(outcome, scenario, PathAction.KEEP, probe=True)
    with (outcome / "summary.csv").open("a", encoding="utf-8") as handle:
        handle.write("unexpected,row\n")
    with pytest.raises(ValueError, match="no foreground"):
        read_change_probe(outcome, scenario)

    identity = tmp_path / "identity"
    _write_raw(identity, scenario, PathAction.KEEP, probe=True)
    parameters = json.loads((identity / "parameters.json").read_text(encoding="utf-8"))
    parameters["seed"] = 999
    (identity / "parameters.json").write_text(json.dumps(parameters), encoding="utf-8")
    with pytest.raises(ValueError, match="identity"):
        read_change_probe(identity, scenario)

    telemetry_schema = tmp_path / "telemetry-schema"
    _write_raw(telemetry_schema, scenario, PathAction.KEEP, probe=True)
    (telemetry_schema / "telemetry.csv").write_text("wrong\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry schema"):
        read_change_probe(telemetry_schema, scenario)

    telemetry_value = tmp_path / "telemetry-value"
    _write_raw(telemetry_value, scenario, PathAction.KEEP, probe=True)
    text = (telemetry_value / "telemetry.csv").read_text(encoding="utf-8")
    (telemetry_value / "telemetry.csv").write_text(
        text.replace("3700000000", "invalid", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="telemetry value"):
        read_change_probe(telemetry_value, scenario)

    state_schema = tmp_path / "state-schema"
    _write_raw(state_schema, scenario, PathAction.KEEP, probe=True)
    (state_schema / "state.csv").write_text("wrong\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="state schema"):
        read_change_probe(state_schema, scenario)

    state_value = tmp_path / "state-value"
    _write_raw(state_value, scenario, PathAction.KEEP, probe=True)
    text = (state_value / "state.csv").read_text(encoding="utf-8")
    (state_value / "state.csv").write_text(
        text.replace("3700000000", "invalid", 1), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="state value"):
        read_change_probe(state_value, scenario)

    layout = tmp_path / "layout"
    _write_raw(layout, scenario, PathAction.KEEP, probe=True)
    state_rows = (layout / "state.csv").read_text(encoding="utf-8").splitlines()
    (layout / "state.csv").write_text("\n".join(state_rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="layouts differ"):
        read_change_probe(layout, scenario)

    latest = tmp_path / "latest"
    _write_raw(latest, scenario, PathAction.KEEP, probe=True)
    for name in ("telemetry.csv", "state.csv"):
        rows = (latest / name).read_text(encoding="utf-8").splitlines()
        (latest / name).write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="latest sample"):
        read_change_probe(latest, scenario)


def test_redecision_rejects_missing_source_decision_shapes(tmp_path: Path) -> None:
    scenario = load_n08_matrix(MATRIX)[0]
    raw = tmp_path / "probe"
    _write_raw(raw, scenario, PathAction.KEEP, probe=True)
    probe = read_change_probe(raw, scenario)
    missing_list = {"scenario_id": scenario.scenario_id.replace("n08-", "n06-", 1)}
    with pytest.raises(ValueError, match="decisions are invalid"):
        build_redecision_generations(scenario, probe, "a" * 64, missing_list, "b" * 64)
    missing_method = {**missing_list, "decisions": []}
    with pytest.raises(ValueError, match="controller decision is missing"):
        build_redecision_generations(scenario, probe, "a" * 64, missing_method, "b" * 64)


def test_runner_retains_probe_failures_and_rejects_queue_drift(
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
        _write_raw(Path(values["output"]), invalid.scenarios[0], PathAction.KEEP, probe=True)
        (Path(values["output"]) / "state.csv").write_text("bad\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", invalid_probe)
    with pytest.raises(RuntimeError, match="invalid probe retained"):
        invalid.run(limit=1)

    queue_drift = _runner(tmp_path / "queue-drift")
    monkeypatch.setattr(subprocess, "run", _mock_native(queue_drift.scenarios[0], action_queue=1))
    with pytest.raises(ValueError, match="queue traces differ"):
        queue_drift.run(limit=1)


def test_runner_rejects_probe_decision_score_and_source_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    probe_tamper = _runner(tmp_path / "probe")
    scenario = probe_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    probe_tamper.run(limit=1)
    telemetry = probe_tamper.output_root / "probes" / scenario.scenario_id / "raw" / "telemetry.csv"
    telemetry.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="invalid N08 probe"):
        probe_tamper.run(limit=1)

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

    score_tamper = _runner(tmp_path / "score")
    scenario = score_tamper.scenarios[0]
    monkeypatch.setattr(subprocess, "run", _mock_native(scenario))
    score_tamper.run(limit=1)
    score = score_tamper.output_root / "scores" / scenario.scenario_id / "score.json"
    payload = json.loads(score.read_text(encoding="utf-8"))
    payload["development_only"] = False
    score.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="score does not match"):
        score_tamper.run(limit=1)

    missing_source = _runner(tmp_path / "source")
    missing_source._source_decision(missing_source.scenarios[0]).unlink()
    monkeypatch.setattr(subprocess, "run", _mock_native(missing_source.scenarios[0]))
    with pytest.raises(ValueError, match="source N07 decision is missing"):
        missing_source.run(limit=1)


def test_validator_rejects_an_incomplete_n08_journal(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    with pytest.raises(ValueError, match="matrix is incomplete"):
        runner.validate()


def test_redecision_builder_has_no_outcome_or_oracle_dependency() -> None:
    source = (ROOT / "src" / "ictc" / "ns3" / "n08.py").read_text(encoding="utf-8")
    builder = source.split("def build_redecision_generations", 1)[1].split(
        "class N08DevelopmentRunner", 1
    )[0]
    for forbidden in ("ictc.oracle", "physical_benefit.json", "summary.csv"):
        assert forbidden not in builder


def test_native_failure_keeps_attempted_flow_observable_and_emits_state() -> None:
    source = (ROOT / "ns3" / "scratch" / "ictc_stochastic_paths.cc").read_text(encoding="utf-8")
    assert 'state << "timestamp_ns,phase,path_id,link_up,rate_bps' in source
    assert "SetPathFailure" in source
    assert "RateErrorModel" in source
    assert "Ipv4::SetDown" not in source
