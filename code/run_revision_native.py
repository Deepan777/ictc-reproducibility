"""Immutable native experiment orchestration, restricted to Revision outputs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
from ictc.ns3.n12 import N12Scenario, _csv_rows, digest, pairing_digest, sha256, utility
from revision_controller import context, decisions

ROOT = Path(os.environ.get("ICTC_REPRO_ROOT", str(Path(__file__).resolve().parents[1]))).resolve()
LOCK = threading.Lock()
FIELDS = ["scenario_id", "kind", "action", "status", "elapsed_seconds", "manifest", "detail"]


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def new(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise ValueError(f"immutable artifact changed: {path}")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)


def append(registry: Path, **row: Any) -> None:
    with LOCK, registry.open("a", encoding="utf-8", newline="") as stream:
        csv.DictWriter(stream, fieldnames=FIELDS).writerow({key: row.get(key, "") for key in FIELDS})


def prepare() -> None:
    protocol = read(ROOT / "configs/revision_protocol.json")
    for name in ("geant-2012-03", "garr-2011-09"):
        shutil.copyfile(ROOT / f"code/configs/experiments/n13_transport_derived/{name}.topology", ROOT / f"configs/{name}.topology")
    lines = (ROOT / "configs/geant-2012-03.topology").read_text().splitlines()
    header = lines[0].split()
    header[1] = "geant-stress"
    header[-1] = "20"
    nodes = int(header[2])
    stress = [" ".join(header)] + [line for line in lines[1:] if not line.startswith("BACKGROUND")]
    # Deterministic, distinct source/destination pairs. The native application
    # handles collision with the foreground destination before traffic starts.
    stress += [f"BACKGROUND {i % nodes} {(i + nodes // 2 + 1) % nodes}" for i in range(20)]
    path = ROOT / "configs/geant-stress.topology"
    if path.exists() and path.read_text() != "\n".join(stress) + "\n":
        raise ValueError("stress topology definition already differs")
    if not path.exists():
        with path.open("x", encoding="ascii", newline="\n") as stream:
            stream.write("\n".join(stress) + "\n")
    for topology, base in protocol["seed_bases"].items():
        phases = ("test",) if topology == "geant-stress" else ("development", "validation", "test")
        for phase in phases:
            offsets = protocol[f"{phase}_offsets"]
            scenarios = []
            for index, offset in enumerate(offsets):
                for traffic in protocol["traffic_profiles"]:
                    for event in protocol["events"]:
                        seed = base + offset
                        scenarios.append(asdict(N12Scenario(
                            scenario_id=f"revision-{topology}-{phase}-{traffic}-{event}-od{index % 3}-seed{seed}",
                            phase=phase, traffic=traffic, event=event, matrix=index % 3,
                            seed=seed, run=1, duration_seconds=protocol["duration_seconds"],
                            decision_at_seconds=protocol["decision_at_seconds"], event_at_seconds=protocol["event_at_seconds"],
                        )))
            executable = ROOT / ("runtime/ns3.46.1-ictc_n12_abilene.exe" if topology == "abilene" else "runtime/ns3.46.1-ictc_n13_transport.exe")
            topo_config = None if topology == "abilene" else ROOT / f"configs/{topology}.topology"
            new(ROOT / f"configs/{topology}-{phase}-matrix.json", {
                "topology": topology, "phase": phase, "scenarios": scenarios,
                "protocol_sha256": sha256(ROOT / "configs/revision_protocol.json"),
                "executable_sha256": sha256(executable),
                "topology_config_sha256": sha256(topo_config) if topo_config else None,
                "planned_native_runs": len(scenarios) * 4,
            })
    new(ROOT / "audit/protocol_freeze.json", {
        "created_at_unix_seconds": time.time(), "test_outcomes_opened": False,
        "protocol_sha256": sha256(ROOT / "configs/revision_protocol.json"),
        "matrices": {p.name: sha256(p) for p in sorted((ROOT / "configs").glob("*-matrix.json"))},
        "topology_inputs": {p.name: sha256(p) for p in sorted((ROOT / "configs").glob("*.topology"))},
    })


def run(topology: str, phase: str, *, workers: int = 4, limit: int | None = None) -> None:
    matrix_path = ROOT / f"configs/{topology}-{phase}-matrix.json"
    matrix = read(matrix_path)
    if matrix["protocol_sha256"] != sha256(ROOT / "configs/revision_protocol.json"):
        raise ValueError("protocol changed after freeze")
    output = ROOT / f"experiments/{topology}/{phase}"
    output.mkdir(parents=True, exist_ok=True)
    if not output.resolve().is_relative_to(ROOT):
        raise ValueError("output escapes Revision")
    executable = ROOT / ("runtime/ns3.46.1-ictc_n12_abilene.exe" if topology == "abilene" else "runtime/ns3.46.1-ictc_n13_transport.exe")
    if sha256(executable) != matrix["executable_sha256"]:
        raise ValueError("native executable changed after freeze")
    model_topology = "geant-2012-03" if topology == "geant-stress" else topology
    bundle_path = ROOT / f"analysis/models/{model_topology}/bundle.joblib"
    freeze_path = ROOT / f"analysis/models/{model_topology}/selection.json"
    bundle = joblib.load(bundle_path) if phase != "development" else None
    selection = read(freeze_path) if phase == "test" else None
    if phase == "test" and sha256(bundle_path) != selection["bundle_sha256"]:
        raise ValueError("selected model bundle changed")
    original = read(ROOT / "code/configs/experiments/n13_selected_policy.json")["controller_thresholds"]["risk_model"]
    metadata = {"matrix_sha256": sha256(matrix_path), "runner_sha256": sha256(Path(__file__)),
                "controller_sha256": sha256(ROOT / "code/revision_controller.py"),
                "bundle_sha256": sha256(bundle_path) if bundle else None,
                "selection_sha256": sha256(freeze_path) if selection else None,
                "original_policy_sha256": sha256(ROOT / "code/configs/experiments/n13_selected_policy.json")}
    new(output / "freeze.json", metadata)
    registry = output / "EXPERIMENT_REGISTRY.csv"
    if not registry.exists():
        with registry.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            for config in matrix["scenarios"]:
                for kind, action in (("probe", "route0"), ("action", "route0"), ("action", "route1"), ("action", "route2")):
                    writer.writerow({"scenario_id": config["scenario_id"], "kind": kind, "action": action, "status": "PLANNED"})
    environment = os.environ.copy()
    runtime_paths = [environment[name] for name in ("ICTC_NS3_LIBRARY_BIN", "ICTC_RUNTIME_BIN") if environment.get(name)]
    environment["PATH"] = os.pathsep.join([*runtime_paths, environment["PATH"]])
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"

    def execute(scenario: N12Scenario, kind: str, action: str) -> Path:
        target = output / "groups" / scenario.scenario_id / f"{kind}-{action}"
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "manifest.json"
        if manifest.exists():
            seal = read(manifest)
            if seal["freeze_digest"] != digest(metadata) or any(sha256(target / name) != value for name, value in seal["files"].items()):
                raise ValueError(f"invalid retained run: {target}")
            return target / "raw"
        raw = target / "raw"
        if raw.exists():
            raise ValueError(f"partial native run retained; manual validity audit required: {raw}")
        # ns-3's Windows argument parser splits spaces inside option values.
        # Relative output paths stay under the explicit Revision process cwd.
        command = [str(executable), *scenario.command(action, raw.relative_to(ROOT), probe=kind == "probe")]
        if topology != "abilene":
            command.append(f"--topologyConfig=configs/{topology}.topology")
        append(registry, scenario_id=scenario.scenario_id, kind=kind, action=action, status="STARTED")
        started = time.perf_counter()
        try:
            completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=180, check=False)
            elapsed = time.perf_counter() - started
            (target / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
            (target / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
            new(target / "resource_metrics.json", {"wall_seconds": elapsed, "scope": "complete native simulation instance including process launch"})
            if completed.returncode != 0:
                raise RuntimeError(f"native exit {completed.returncode}: {completed.stderr[-500:]}")
            rows = _csv_rows(raw / "summary.csv")
            if len(rows) != (0 if kind == "probe" else 1):
                raise ValueError("unexpected foreground result count")
            files = {str(p.relative_to(target)).replace("\\", "/"): sha256(p) for p in sorted(target.rglob("*")) if p.is_file()}
            new(manifest, {"command": command, "freeze_digest": digest(metadata), "files": files,
                           "completed_at_unix_seconds": time.time(), "returncode": 0})
            append(registry, scenario_id=scenario.scenario_id, kind=kind, action=action, status="COMPLETE", elapsed_seconds=elapsed, manifest=str(manifest))
            return raw
        except Exception as error:
            new(target / "failure.json", {"type": type(error).__name__, "detail": str(error), "command": command})
            append(registry, scenario_id=scenario.scenario_id, kind=kind, action=action, status="FAILED", detail=str(error))
            raise

    def group(config: dict[str, Any]) -> str:
        scenario = N12Scenario(**config)
        root = output / "groups" / scenario.scenario_id
        if (root / "group_manifest.json").exists():
            for kind, action in (("probe", "route0"), ("action", "route0"), ("action", "route1"), ("action", "route2")):
                execute(scenario, kind, action)
            return scenario.scenario_id
        probe = execute(scenario, "probe", "route0")
        ctx = context(probe, scenario)
        selected = decisions(scenario, ctx, original, bundle, selection)
        decision_path = root / "decisions.json"
        if not decision_path.exists():
            new(decision_path, {"scenario": config, "topology": topology, "decisions": selected,
                "features": {route: list(__import__('revision_controller').risk_features(scenario, ctx, route)) for route in ("route0", "route1", "route2")},
                "snapshot": ctx["snapshot"], "created_at_unix_seconds": time.time(),
                "information_boundary": "received telemetry only; no delivery audit, plan, events, or outcomes",
                "probe_manifest_sha256": sha256(root / "probe-route0/manifest.json"), "freeze_digest": digest(metadata)})
        elif read(decision_path)["decisions"] != selected:
            raise ValueError("retained decision differs from frozen inputs")
        action_raw = {action: execute(scenario, "action", action) for action in ("route0", "route1", "route2")}
        pairings = {pairing_digest(probe, scenario)} | {pairing_digest(raw, scenario) for raw in action_raw.values()}
        if len(pairings) != 1:
            raise ValueError("paired pre-decision state differs across route actions")
        outcomes = {action: _csv_rows(raw / "summary.csv")[0] for action, raw in action_raw.items()}
        values = {action: utility(row) for action, row in outcomes.items()}
        best = max(sorted(values), key=values.__getitem__)
        all_selected = {**selected, "OfflineOracle": best}
        opportunity = values[best] > values["route0"] + 0.02
        rows = [{"method": method, "action": action, "utility": values[action],
                 "harmful": values[action] < values["route0"] - 0.02,
                 "beneficial": values[action] > values["route0"] + 0.02,
                 "opportunity": opportunity,
                 "retained": opportunity and values[action] >= values[best] - 0.02,
                 "metrics": outcomes[action]} for method, action in all_selected.items()]
        new(root / "score.json", {"scenario": config, "rows": rows, "utilities": values,
                                   "decision_sha256": sha256(decision_path)})
        new(root / "group_manifest.json", {"pairing_digest": next(iter(pairings)),
            "decision_sha256": sha256(decision_path), "score_sha256": sha256(root / "score.json"),
            "native_manifests": {f"{kind}-{action}": sha256(root / f"{kind}-{action}/manifest.json") for kind, action in (("probe", "route0"), ("action", "route0"), ("action", "route1"), ("action", "route2"))}})
        return scenario.scenario_id

    scenarios = matrix["scenarios"][:limit] if limit else matrix["scenarios"]
    free = shutil.disk_usage(ROOT).free
    estimated = len(scenarios) * 400_000
    if free < 1.30 * estimated + 5 * 1024**3:
        raise RuntimeError("resource preflight failed")
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(group, config) for config in scenarios]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 12 == 0 or completed == len(scenarios):
                print(json.dumps({"topology": topology, "phase": phase, "complete_groups": completed, "planned_groups": len(scenarios)}), flush=True)
    print(json.dumps({"topology": topology, "phase": phase, "complete_groups": completed, "native_runs": completed * 4, "failures": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--topology", default="abilene")
    parser.add_argument("--phase", default="development")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    else:
        run(args.topology, args.phase, workers=args.workers, limit=args.limit)
