"""Tests for the frozen N10 confirmatory runner."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ictc.ns3.n08 import ChangeProbeEvidence
from ictc.ns3.n09 import N09MethodId
from ictc.ns3.n10 import N10ConfirmatoryRunner, build_n10_generations, load_n10_matrix
from ictc.ns3.stochastic import StochasticScenario, sha256

ROOT = Path(__file__).resolve().parents[2]
MATRIX = ROOT / "configs" / "experiments" / "n10_confirmatory_support.json"
DESIGN = ROOT / "docs" / "milestones" / "evidence" / "n09_context_support_design.json"


def _probe(scenario: StochasticScenario) -> ChangeProbeEvidence:
    return ChangeProbeEvidence(
        scenario_id=scenario.scenario_id,
        seed=scenario.seed,
        path_ids=("current", "candidate"),
        latest_timestamp_ns=3_700_000_000,
        latest_queues={"current": 0, "candidate": 0},
        latest_link_up={"current": True, "candidate": True},
        latest_rate_bps={"current": 5_000_000, "candidate": 5_000_000},
        pre_decision_digest="1" * 64,
        state_digest="2" * 64,
        observation_digest="3" * 64,
        event_detected=True,
        telemetry_sha256="4" * 64,
        state_sha256="5" * 64,
        background_sha256="6" * 64,
    )


def _build(scenario: StochasticScenario) -> dict[str, Any]:
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    return build_n10_generations(
        scenario,
        _probe(scenario),
        "a" * 64,
        support_design_digest=matrix["support_design_digest"],
        support_design_file_sha256=matrix["support_design_file_sha256"],
        source_n09_manifest_sha256=matrix["source_n09_manifest_sha256"],
        source_n08_manifest_sha256=matrix["source_n08_manifest_sha256"],
        supported_contexts=tuple(matrix["supported_contexts"]),
    )


def _runner(tmp_path: Path) -> N10ConfirmatoryRunner:
    source_n09 = tmp_path / "source-n09"
    source_n08 = tmp_path / "source-n08"
    source_n09.mkdir(parents=True)
    source_n08.mkdir(parents=True)
    (source_n09 / "n09_manifest.json").write_text('{"fixture":"n09"}\n', encoding="utf-8")
    (source_n08 / "n08_manifest.json").write_text('{"fixture":"n08"}\n', encoding="utf-8")
    design = tmp_path / "design.json"
    design.write_text(DESIGN.read_text(encoding="utf-8"), encoding="utf-8")
    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    matrix["support_design_file_sha256"] = sha256(design)
    matrix["source_n09_manifest_sha256"] = sha256(source_n09 / "n09_manifest.json")
    matrix["source_n08_manifest_sha256"] = sha256(source_n08 / "n08_manifest.json")
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    executable = tmp_path / "simulator.exe"
    executable.write_bytes(b"fixture")
    return N10ConfirmatoryRunner(
        repository=ROOT,
        output_root=tmp_path / "journal",
        matrix_path=matrix_path,
        executable=executable,
        runtime_directories=(tmp_path,),
        support_design_path=design,
        source_n09_root=source_n09,
        source_n08_root=source_n08,
    )


def test_n10_matrix_is_exact_complete_and_reserved(tmp_path: Path) -> None:
    scenarios = load_n10_matrix(MATRIX)
    assert len(scenarios) == 720
    assert sum(len(item.actions) for item in scenarios) == 1680
    assert len(scenarios) * len(N09MethodId) == 4320
    assert {scenario.seed for scenario in scenarios} == set(range(82000, 82020))
    assert all(scenario.run == 1 for scenario in scenarios)

    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    payload["seeds"][0] = 81999
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved set"):
        load_n10_matrix(invalid)


def test_n10_matrix_rejects_freeze_defects(tmp_path: Path) -> None:
    base = json.loads(MATRIX.read_text(encoding="utf-8"))
    variants: list[tuple[str, object, str]] = [
        ("schema_version", "wrong", "schema"),
        ("topology_families", [], "topology"),
        ("workloads", [], "workload"),
        ("events", [], "event"),
        ("controls", None, "controls are missing"),
        ("expected_scenario_groups", 1, "scenario count"),
        ("expected_action_executions", 1, "action count"),
        ("expected_method_decisions", 1, "method-decision count"),
        ("supported_contexts", [], "supported contexts"),
        ("analysis", {}, "analysis freeze"),
        ("source_n09_manifest_sha256", "short", "source_n09_manifest_sha256"),
    ]
    for index, (field, value, match) in enumerate(variants):
        payload = json.loads(json.dumps(base))
        payload[field] = value
        path = tmp_path / f"variant-{index}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=match):
            load_n10_matrix(path)

    controls = json.loads(json.dumps(base))
    controls["controls"].pop("duration_seconds")
    path = tmp_path / "invalid-controls.json"
    path.write_text(json.dumps(controls), encoding="utf-8")
    with pytest.raises(ValueError, match="controls are invalid"):
        load_n10_matrix(path)

    timing = json.loads(json.dumps(base))
    timing["controls"]["decision_at_seconds"] = 2.0
    path = tmp_path / "invalid-timing.json"
    path.write_text(json.dumps(timing), encoding="utf-8")
    with pytest.raises(ValueError, match="event-before-redecision"):
        load_n10_matrix(path)


def test_n10_generation_binds_both_sources_and_support_decision() -> None:
    scenario = load_n10_matrix(MATRIX)[0]
    payload = _build(scenario)
    zero, one = payload["generations"]
    assert (
        zero["source_n09_manifest_sha256"]
        == json.loads(MATRIX.read_text(encoding="utf-8"))["source_n09_manifest_sha256"]
    )
    assert one["previous_generation_digest"] == zero["recorded_digest"]
    assert one["support_applied"] is True
    decisions = {row["method_id"]: row for row in one["decisions"]}
    assert (
        decisions[N09MethodId.D1_CONTEXT_SUPPORTED_CAPACITY.value]["selected_action"] == "reroute"
    )
    assert decisions[N09MethodId.B3_CAPACITY.value]["selected_action"] == "keep"
    assert not any(payload["forbidden_inputs"].values())

    matrix = json.loads(MATRIX.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="source N09 manifest binding"):
        build_n10_generations(
            scenario,
            _probe(scenario),
            "a" * 64,
            support_design_digest=matrix["support_design_digest"],
            support_design_file_sha256=matrix["support_design_file_sha256"],
            source_n09_manifest_sha256="short",
            source_n08_manifest_sha256=matrix["source_n08_manifest_sha256"],
            supported_contexts=tuple(matrix["supported_contexts"]),
        )


def test_n10_runner_seals_freeze_and_rejects_manifest_tampering(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    runner.initialise()
    manifest = runner.output_root / "n10_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["scenario_groups"] == 720
    assert payload["confirmatory_seeds"] == list(range(82000, 82020))
    payload["config_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="different N10 freeze"):
        runner.initialise()


def test_n10_runner_rejects_source_manifest_mismatch(tmp_path: Path) -> None:
    runner = _runner(tmp_path)
    source_manifest = runner.source_n09_root / "n09_manifest.json"
    source_manifest.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="source N09 manifest"):
        N10ConfirmatoryRunner(
            repository=ROOT,
            output_root=tmp_path / "second-journal",
            matrix_path=runner.matrix_path,
            executable=runner.executable,
            runtime_directories=runner.runtime_directories,
            support_design_path=runner.support_design_path,
            source_n09_root=runner.source_n09_root,
            source_n08_root=runner.source_n08_root,
        )


def test_n10_runner_rejects_support_and_n08_mismatches(tmp_path: Path) -> None:
    design_runner = _runner(tmp_path / "design")
    design_runner.support_design_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="support-design file"):
        N10ConfirmatoryRunner(
            repository=ROOT,
            output_root=tmp_path / "design-journal",
            matrix_path=design_runner.matrix_path,
            executable=design_runner.executable,
            runtime_directories=design_runner.runtime_directories,
            support_design_path=design_runner.support_design_path,
            source_n09_root=design_runner.source_n09_root,
            source_n08_root=design_runner.source_n08_root,
        )

    n08_runner = _runner(tmp_path / "n08")
    (n08_runner.source_n08_root / "n08_manifest.json").write_text(
        '{"tampered":true}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="source N08 manifest"):
        N10ConfirmatoryRunner(
            repository=ROOT,
            output_root=tmp_path / "n08-journal",
            matrix_path=n08_runner.matrix_path,
            executable=n08_runner.executable,
            runtime_directories=n08_runner.runtime_directories,
            support_design_path=n08_runner.support_design_path,
            source_n09_root=n08_runner.source_n09_root,
            source_n08_root=n08_runner.source_n08_root,
        )
