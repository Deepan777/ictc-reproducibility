"""Tests for the N06 stochastic ns-3 boundary and development runner."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n06 import N06PilotRunner, load_registry_latest
from ictc.ns3.stochastic import (
    EventRegime,
    ForegroundOutcome,
    PathAction,
    PhysicalBenefit,
    StochasticScenario,
    TopologyFamily,
    classify_physical_benefit,
    load_stochastic_matrix,
    read_action_evidence,
    validate_paired_pre_decision,
)

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


def _runner(tmp_path: Path) -> N06PilotRunner:
    tmp_path.mkdir(parents=True, exist_ok=True)
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    return N06PilotRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=MATRIX,
        executable=executable,
        runtime_directories=(tmp_path,),
    )


def _mock_success(
    scenario: StochasticScenario,
    *,
    invalid: bool = False,
) -> Any:
    def execute(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        values = dict(item[2:].split("=", 1) for item in command[1:])
        action = PathAction(values["action"])
        _write_raw(
            Path(values["output"]),
            scenario,
            action,
            delay_ns=5_000_000 if action is PathAction.KEEP else 4_000_000,
        )
        if invalid:
            summary = Path(values["output"]) / "summary.csv"
            summary.write_text("invalid\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    return execute


def _write_raw(
    raw: Path,
    scenario: StochasticScenario,
    action: PathAction,
    *,
    rx: int = 99,
    delay_ns: int = 5_000_000,
    queue: int = 2,
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
                "rx_packets": rx,
                "lost_packets": 100 - rx,
                "mean_delay_ns": delay_ns,
                "jitter_ns": 100,
                "throughput_bps": 10_000,
                "first_tx_ns": 2_000_000_000,
                "last_rx_ns": 3_000_000_000 if rx else 0,
            }
        )
    with (raw / "telemetry.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["timestamp_ns", "phase", "path_id", "queue_packets"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "timestamp_ns": 1_000_000_000,
                "phase": "PRE_DECISION",
                "path_id": "current",
                "queue_packets": queue,
            }
        )
        writer.writerow(
            {
                "timestamp_ns": 2_000_000_000,
                "phase": "POST_DECISION",
                "path_id": "current",
                "queue_packets": queue + 1,
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
        writer.writerow(
            {
                "path_id": "current",
                "flow_id": 1,
                "tx_packets": 50,
                "rx_packets": 49,
                "lost_packets": 1,
                "mean_delay_ns": 2_000_000,
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
            }
        ),
        encoding="utf-8",
    )


def _outcome(
    action: str,
    *,
    rx: int = 100,
    delay: int = 5_000_000,
    scenario: str = "pair",
) -> ForegroundOutcome:
    return ForegroundOutcome(
        scenario_id=scenario,
        topology_family="two-balanced",
        workload="bursty",
        event="stable",
        action=action,
        seed=81000,
        run=1,
        tx_packets=100,
        rx_packets=rx,
        lost_packets=100 - rx,
        mean_delay_ns=delay,
        jitter_ns=10,
        throughput_bps=1.0,
        first_tx_ns=1,
        last_rx_ns=2 if rx else 0,
    )


def test_development_matrix_is_exact_and_confirmatory_seeds_are_reserved() -> None:
    scenarios = load_stochastic_matrix(MATRIX)
    assert len(scenarios) == 144
    assert sum(len(item.actions) for item in scenarios) == 336
    assert {item.seed for item in scenarios} == {81000, 81001, 81002, 81003}
    assert {item.event for item in scenarios} == set(EventRegime)
    assert (
        sum(item.topology_family is TopologyFamily.THREE_HETEROGENEOUS for item in scenarios) == 48
    )


def test_matrix_rejects_reserved_seed_execution(tmp_path: Path) -> None:
    raw = json.loads(MATRIX.read_text(encoding="utf-8"))
    raw["seeds"] = [82000, 81001, 81002, 81003]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="authorized"):
        load_stochastic_matrix(path)


def test_matrix_rejects_every_frozen_structure_violation(tmp_path: Path) -> None:
    source = json.loads(MATRIX.read_text(encoding="utf-8"))
    cases: list[tuple[str, Any, str]] = []
    raw = copy.deepcopy(source)
    raw["schema_version"] = "wrong"
    cases.append(("schema", raw, "schema"))
    raw = copy.deepcopy(source)
    raw["phase"] = "confirmatory"
    cases.append(("phase", raw, "development matrix"))
    raw = copy.deepcopy(source)
    raw["reserved_confirmatory_seeds"] = []
    cases.append(("reserved", raw, "reservation"))
    raw = copy.deepcopy(source)
    raw["topology_families"] = []
    cases.append(("topology", raw, "topology"))
    raw = copy.deepcopy(source)
    raw["events"] = []
    cases.append(("events", raw, "workload/event"))
    raw = copy.deepcopy(source)
    raw["controls"] = []
    cases.append(("controls-object", raw, "controls are missing"))
    raw = copy.deepcopy(source)
    del raw["controls"]["duration_seconds"]
    cases.append(("controls-value", raw, "controls are invalid"))
    raw = copy.deepcopy(source)
    raw["controls"]["decision_at_seconds"] = 4
    raw["controls"]["event_at_seconds"] = 3
    cases.append(("timing", raw, "timing"))
    raw = copy.deepcopy(source)
    raw["expected_scenario_groups"] = 1
    cases.append(("group-count", raw, "scenario count"))
    raw = copy.deepcopy(source)
    raw["expected_action_executions"] = 1
    cases.append(("action-count", raw, "action count"))
    for name, payload, message in cases:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            load_stochastic_matrix(path)

    non_object = tmp_path / "non-object.json"
    non_object.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain an object"):
        load_stochastic_matrix(non_object)


def test_action_command_and_topology_scope_are_explicit(tmp_path: Path) -> None:
    scenario = _scenario()
    command = scenario.command_arguments(PathAction.KEEP, tmp_path / "raw")
    assert "--seed=81000" in command
    assert "--event=stable" in command
    with pytest.raises(ValueError, match="does not exist"):
        scenario.command_arguments(PathAction.ALTERNATE, tmp_path / "raw")


def test_raw_bridge_validates_identity_accounting_and_phases(tmp_path: Path) -> None:
    scenario = _scenario()
    raw = tmp_path / "raw"
    _write_raw(raw, scenario, PathAction.KEEP)
    evidence = read_action_evidence(raw, scenario, PathAction.KEEP)
    assert evidence.outcome.delivery_ratio == 0.99
    assert len(evidence.pre_decision) == 1
    assert set(evidence.raw_sha256) == {
        "summary.csv",
        "telemetry.csv",
        "background.csv",
        "parameters.json",
    }
    with pytest.raises(ValueError, match="planned"):
        read_action_evidence(raw, replace(scenario, seed=81001), PathAction.KEEP)


def test_bridge_retains_flow_monitor_packets_still_in_flight(tmp_path: Path) -> None:
    scenario = _scenario()
    raw = tmp_path / "raw"
    _write_raw(raw, scenario, PathAction.REROUTE, rx=80)
    summary = list(csv.DictReader((raw / "summary.csv").open(newline="", encoding="utf-8")))
    summary[0]["lost_packets"] = "5"
    with (raw / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    evidence = read_action_evidence(raw, scenario, PathAction.REROUTE)
    assert evidence.outcome.tx_packets - evidence.outcome.rx_packets == 20
    assert evidence.outcome.lost_packets == 5


def test_raw_bridge_rejects_malformed_and_unsafe_inputs(tmp_path: Path) -> None:
    scenario = _scenario()

    missing = tmp_path / "missing"
    _write_raw(missing, scenario, PathAction.KEEP)
    (missing / "parameters.json").unlink()
    with pytest.raises(ValueError, match="incomplete"):
        read_action_evidence(missing, scenario, PathAction.KEEP)

    bad_summary = tmp_path / "bad-summary"
    _write_raw(bad_summary, scenario, PathAction.KEEP)
    (bad_summary / "summary.csv").write_text("wrong\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact-schema"):
        read_action_evidence(bad_summary, scenario, PathAction.KEEP)

    bad_number = tmp_path / "bad-number"
    _write_raw(bad_number, scenario, PathAction.KEEP)
    text = (
        (bad_number / "summary.csv")
        .read_text(encoding="utf-8")
        .replace(",81000,1,", ",not-a-seed,1,")
    )
    (bad_number / "summary.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="invalid values"):
        read_action_evidence(bad_number, scenario, PathAction.KEEP)

    bad_accounting = tmp_path / "bad-accounting"
    _write_raw(bad_accounting, scenario, PathAction.KEEP)
    text = (
        (bad_accounting / "summary.csv")
        .read_text(encoding="utf-8")
        .replace(",100,99,1,", ",0,0,0,")
    )
    (bad_accounting / "summary.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="packet accounting"):
        read_action_evidence(bad_accounting, scenario, PathAction.KEEP)

    telemetry_schema = tmp_path / "telemetry-schema"
    _write_raw(telemetry_schema, scenario, PathAction.KEEP)
    (telemetry_schema / "telemetry.csv").write_text("wrong\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry schema"):
        read_action_evidence(telemetry_schema, scenario, PathAction.KEEP)

    telemetry_value = tmp_path / "telemetry-value"
    _write_raw(telemetry_value, scenario, PathAction.KEEP)
    text = (
        (telemetry_value / "telemetry.csv")
        .read_text(encoding="utf-8")
        .replace("PRE_DECISION,current,2", "PRE_DECISION,current,invalid")
    )
    (telemetry_value / "telemetry.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="telemetry contains invalid"):
        read_action_evidence(telemetry_value, scenario, PathAction.KEEP)

    no_pre = tmp_path / "no-pre"
    _write_raw(no_pre, scenario, PathAction.KEEP)
    text = (
        (no_pre / "telemetry.csv")
        .read_text(encoding="utf-8")
        .replace("PRE_DECISION", "POST_DECISION")
    )
    (no_pre / "telemetry.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="no pre-decision"):
        read_action_evidence(no_pre, scenario, PathAction.KEEP)

    background_schema = tmp_path / "background-schema"
    _write_raw(background_schema, scenario, PathAction.KEEP)
    (background_schema / "background.csv").write_text("wrong\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="background schema"):
        read_action_evidence(background_schema, scenario, PathAction.KEEP)

    background_value = tmp_path / "background-value"
    _write_raw(background_value, scenario, PathAction.KEEP)
    text = (
        (background_value / "background.csv")
        .read_text(encoding="utf-8")
        .replace("current,1,50,49,1", "current,1,0,0,0")
    )
    (background_value / "background.csv").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="background packet accounting"):
        read_action_evidence(background_value, scenario, PathAction.KEEP)

    parameters = tmp_path / "parameters"
    _write_raw(parameters, scenario, PathAction.KEEP)
    payload = json.loads((parameters / "parameters.json").read_text(encoding="utf-8"))
    payload["seed"] = 81003
    (parameters / "parameters.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="parameters"):
        read_action_evidence(parameters, scenario, PathAction.KEEP)


def test_paired_pre_decision_requires_exact_action_independence(tmp_path: Path) -> None:
    scenario = _scenario()
    _write_raw(tmp_path / "keep", scenario, PathAction.KEEP)
    _write_raw(tmp_path / "reroute", scenario, PathAction.REROUTE)
    evidence = {
        PathAction.KEEP: read_action_evidence(tmp_path / "keep", scenario, PathAction.KEEP),
        PathAction.REROUTE: read_action_evidence(
            tmp_path / "reroute", scenario, PathAction.REROUTE
        ),
    }
    assert len(validate_paired_pre_decision(evidence)) == 64
    _write_raw(tmp_path / "different", scenario, PathAction.REROUTE, queue=3)
    evidence[PathAction.REROUTE] = read_action_evidence(
        tmp_path / "different", scenario, PathAction.REROUTE
    )
    with pytest.raises(ValueError, match="before the decision"):
        validate_paired_pre_decision(evidence)


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [
        (_outcome("reroute", delay=4_000_000), PhysicalBenefit.BENEFICIAL),
        (_outcome("reroute", rx=90, delay=4_000_000), PhysicalBenefit.HARMFUL),
        (_outcome("reroute", delay=5_100_000), PhysicalBenefit.NO_MATERIAL_DIFFERENCE),
        (_outcome("reroute", delay=6_000_000), PhysicalBenefit.HARMFUL),
    ],
)
def test_physical_benefit_rules_are_frozen(
    candidate: ForegroundOutcome, expected: PhysicalBenefit
) -> None:
    assert classify_physical_benefit(_outcome("keep"), candidate) is expected


def test_zero_receive_and_pairing_boundaries_are_explicit() -> None:
    assert (
        classify_physical_benefit(_outcome("keep", rx=0), _outcome("reroute", rx=50))
        is PhysicalBenefit.BENEFICIAL
    )
    assert (
        classify_physical_benefit(_outcome("keep", rx=50), _outcome("reroute", rx=0))
        is PhysicalBenefit.HARMFUL
    )
    with pytest.raises(ValueError, match="paired"):
        classify_physical_benefit(_outcome("keep"), _outcome("reroute", scenario="other"))
    with pytest.raises(ValueError, match="threshold"):
        classify_physical_benefit(_outcome("keep"), _outcome("reroute"), delay_margin_ns=0)
    with pytest.raises(ValueError, match="requires KEEP"):
        classify_physical_benefit(_outcome("reroute"), _outcome("keep"))


def test_pair_validator_requires_both_primary_actions(tmp_path: Path) -> None:
    scenario = _scenario()
    _write_raw(tmp_path / "keep", scenario, PathAction.KEEP)
    keep = read_action_evidence(tmp_path / "keep", scenario, PathAction.KEEP)
    with pytest.raises(ValueError, match="requires KEEP and REROUTE"):
        validate_paired_pre_decision({PathAction.KEEP: keep})


def test_group_label_is_created_only_after_action_manifests(tmp_path: Path) -> None:
    scenario = _scenario()
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    runner = N06PilotRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=MATRIX,
        executable=executable,
        runtime_directories=(tmp_path,),
    )
    group = runner.output_root / "scenarios" / scenario.scenario_id
    targets = {}
    for action, delay in ((PathAction.KEEP, 5_000_000), (PathAction.REROUTE, 4_000_000)):
        target = group / action.value
        _write_raw(target / "raw", scenario, action, delay_ns=delay)
        targets[action] = target
    with pytest.raises(FileNotFoundError):
        runner._seal_group(scenario, targets)
    for target in targets.values():
        (target / "action_manifest.json").write_text("{}\n", encoding="utf-8")
    runner._seal_group(scenario, targets)
    label = json.loads((group / "physical_benefit.json").read_text(encoding="utf-8"))
    assert label["labels"] == {"reroute": "BENEFICIAL"}
    assert "not graphical identification" in label["semantic_boundary"]


def test_runner_commits_resumes_and_validates_mocked_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr(subprocess, "run", _mock_success(runner.scenarios[0]))
    assert runner.run(limit=1)["action_executions"] == 2
    rows_after_first = len(runner.registry_path.read_text(encoding="utf-8").splitlines())
    assert runner.run(limit=1)["action_executions"] == 2
    assert len(runner.registry_path.read_text(encoding="utf-8").splitlines()) == rows_after_first
    validation = runner.validate(require_complete=False)
    assert validation["complete_scenario_groups"] == 1
    with pytest.raises(ValueError, match="incomplete"):
        runner.validate(require_complete=True)


def test_runner_refuses_invalid_existing_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = _runner(tmp_path)
    monkeypatch.setattr(subprocess, "run", _mock_success(runner.scenarios[0]))
    runner.run(limit=1)
    target = (
        runner.output_root
        / "scenarios"
        / runner.scenarios[0].scenario_id
        / "keep"
        / "raw"
        / "summary.csv"
    )
    target.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="invalid existing"):
        runner.run(limit=1)


def test_runner_retains_nonzero_timeout_and_validation_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    nonzero = _runner(tmp_path / "nonzero")
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 9, "out", "err"),
    )
    with pytest.raises(RuntimeError, match="failed; evidence retained"):
        nonzero.run(limit=1)
    assert next((nonzero.output_root / "failures").rglob("failure.json")).is_file()

    timeout = _runner(tmp_path / "timeout")

    def raise_timeout(command: list[str], **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(command, 120, output=b"out", stderr=b"err")

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    with pytest.raises(RuntimeError, match="timeout retained"):
        timeout.run(limit=1)
    assert next((timeout.output_root / "failures").rglob("stdout.txt")).read_text() == "out"

    invalid = _runner(tmp_path / "invalid")
    monkeypatch.setattr(subprocess, "run", _mock_success(invalid.scenarios[0], invalid=True))
    with pytest.raises(RuntimeError, match="invalid raw evidence retained"):
        invalid.run(limit=1)
    failure = next((invalid.output_root / "failures").rglob("failure.json"))
    assert "EVIDENCE_VALIDATION_ERROR" in failure.read_text(encoding="utf-8")


def test_runner_rejects_freeze_mismatch_and_invalid_registry(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.initialise()
    manifest = runner.output_root / "n06_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different N06 freeze"):
        runner.initialise()

    registry = tmp_path / "bad-registry.csv"
    registry.write_text("wrong\nvalue\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry schema"):
        load_registry_latest(registry)


def test_registry_latest_and_forbidden_import_boundary(tmp_path: Path) -> None:
    registry = tmp_path / "registry.csv"
    with registry.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "scenario_id",
                "action",
                "seed",
                "status",
                "started_at_utc",
                "completed_at_utc",
                "config_sha256",
                "executable_sha256",
                "manifest_path",
                "failure_classification",
            ],
        )
        writer.writeheader()
        writer.writerow({"scenario_id": "s", "action": "keep", "status": "STARTED"})
        writer.writerow({"scenario_id": "s", "action": "keep", "status": "COMPLETE"})
    assert load_registry_latest(registry)[("s", "keep")]["status"] == "COMPLETE"
    source = (ROOT / "src" / "ictc" / "ns3" / "stochastic.py").read_text(encoding="utf-8")
    for forbidden in ("ictc.oracle", "ictc.controller", "m14_reference", "post_action_outcome"):
        assert forbidden not in source
