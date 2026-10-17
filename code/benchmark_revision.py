"""Controller microbenchmarks and frozen evidence-budget sensitivities."""
from __future__ import annotations
import csv
import ctypes
import json
import os
import platform
import statistics
import struct
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any, Callable
import joblib
import numpy as np
import sklearn
from ictc.causal import BackdoorIdentifier, CausalDAG
from ictc.contract.planning import QueryContractPlanner
from ictc.contract.planning_benchmark import capability_registry, planning_policy, routing_graph, routing_query
from ictc.ns3.n12 import N12Scenario, _csv_rows, utility
from ictc.ns3.n13 import predict_harm_probability
from revision_controller import admit, context, context_from_rows, received_rows, risk_features, learned_proposal, fluid_proposal
from run_revision_native import ROOT, new, read
from analyze_revision import csv_new, row_record, summary


def baseline(raw: Path, scenario: N12Scenario) -> str:
    latest = {}
    for row in received_rows(raw, int(scenario.decision_at_seconds*1e9)):
        route = f"route{row['route_id']}"
        if route not in latest or int(row["available_timestamp_ns"]) > int(latest[route]["available_timestamp_ns"]):
            latest[route] = row
    eligible = [r for r in sorted(latest) if int(latest[r]["link_up"])] or ["route0"]
    return max(eligible, key=lambda r: (float(latest[r]["bottleneck_capacity_bps"])/1e6-0.5*int(latest[r]["queue_packets"])-0.08*float(latest[r]["propagation_delay_ms"]), -int(r[-1])))


def measurement(name: str, operation: Callable[[], Any], repetitions: int = 1000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    for _ in range(50):
        operation()
    values, cpu_values = [], []
    for index in range(repetitions):
        started, cpu_started = time.perf_counter_ns(), time.process_time_ns()
        operation()
        values.append((time.perf_counter_ns()-started)/1e6)
        cpu_values.append((time.process_time_ns()-cpu_started)/1e6)
    tracemalloc.start()
    before_current, _ = tracemalloc.get_traced_memory()
    operation()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"component": name, "repetitions": repetitions, "warmup": 50,
            "median_ms": float(np.median(values)), "p95_ms": float(np.quantile(values,0.95)),
            "p99_ms": float(np.quantile(values,0.99)), "median_cpu_ms": float(np.median(cpu_values)),
            "mean_cpu_ms_per_call": float(np.mean(cpu_values)),
            "python_peak_increment_bytes": peak-before_current, "python_retained_increment_bytes": current-before_current}, [
                {"component": name, "repetition": i, "elapsed_ms": value, "cpu_ms": cpu_values[i]} for i,value in enumerate(values)]


def process_memory() -> dict[str, int]:
    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    value = Counters()
    value.cb = ctypes.sizeof(value)
    get_handle = ctypes.windll.kernel32.GetCurrentProcess
    get_handle.restype = ctypes.c_void_p
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    get_memory.restype = ctypes.c_int
    handle = get_handle()
    if not get_memory(handle, ctypes.byref(value), value.cb):
        return {"measurement_unavailable": 1}
    return {"working_set_bytes": value.WorkingSetSize, "peak_working_set_bytes": value.PeakWorkingSetSize,
            "private_commit_bytes": value.PagefileUsage, "peak_private_commit_bytes": value.PeakPagefileUsage}


def main() -> None:
    output = ROOT / "analysis/controller_overhead"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    matrix = read(ROOT / "configs/abilene-test-matrix.json")
    scenario = N12Scenario(**matrix["scenarios"][0])
    raw = ROOT / f"experiments/abilene/test/groups/{scenario.scenario_id}/probe-route0/raw"
    original = read(ROOT / "code/configs/experiments/n13_selected_policy.json")["controller_thresholds"]["risk_model"]
    decision_ns = int(scenario.decision_at_seconds*1e9)
    rows = received_rows(raw, decision_ns)
    ctx = context_from_rows(rows, decision_ns)
    features = risk_features(scenario, ctx, ctx["greedy_action"])
    bundle = joblib.load(ROOT / "analysis/models/abilene/bundle.joblib")
    selection = read(ROOT / "analysis/models/abilene/selection.json")
    selected_depth = selection["depth"]
    learned_model = bundle["proposers"][selected_depth]
    assert baseline(raw, scenario) == ctx["greedy_action"]
    graph, query, policy = routing_graph(), routing_query(), planning_policy()
    planner = QueryContractPlanner(capability_registry())
    def plan() -> Any:
        return planner.plan(plan_id="benchmark-plan", proposal_id="benchmark-proposal", contract_id="benchmark-contract", query=query, graph=graph, policy=policy)
    cached = {"stable-contract": plan()}
    def cache_read() -> Any:
        value = cached["stable-contract"]
        # Access and validate the cached graph/query version; no synthesis.
        assert value.query.graph_id == graph.graph_id
        return value
    def online() -> str:
        value = context(raw, scenario)
        return admit(scenario, value, value["greedy_action"], original, 0.30)
    def learned_online(with_admission: bool) -> str:
        value = context(raw, scenario)
        proposal = learned_proposal(scenario, value, learned_model)
        return admit(scenario, value, proposal, bundle["risk"][f"learned{selected_depth}"], selection["thresholds"][f"learned{selected_depth}"]) if with_admission else proposal
    def fluid_online(with_admission: bool) -> str:
        value = context(raw, scenario)
        proposal = fluid_proposal(scenario, value)
        return admit(scenario, value, proposal, bundle["risk"]["fluid"], selection["thresholds"]["fluid"]) if with_admission else proposal
    operations = [
        ("Greedy including file parsing", lambda: baseline(raw, scenario)),
        ("Received telemetry parsing", lambda: received_rows(raw, decision_ns)),
        ("Evidence window quality and temporal statistics", lambda: context_from_rows(rows, decision_ns)),
        ("Candidate risk feature construction", lambda: risk_features(scenario, ctx, ctx["greedy_action"])),
        ("Logistic risk inference only", lambda: predict_harm_probability(features, original)),
        ("ICTC total including parsing", online),
        ("Learned proposer including parsing", lambda: learned_online(False)),
        ("Learned proposer with ICTC including parsing", lambda: learned_online(True)),
        ("Fluid optimizer including parsing", lambda: fluid_online(False)),
        ("Fluid optimizer with ICTC including parsing", lambda: fluid_online(True)),
        ("Cached contract lookup and graph version check", cache_read),
        ("Uncached identification and exact contract synthesis", plan),
        ("Identification only", lambda: BackdoorIdentifier(CausalDAG(("U","A","Y"),(("U","A"),("U","Y"),("A","Y")))).identify("A","Y")),
    ]
    results, raw_samples = [], []
    for name, operation in operations:
        result, samples = measurement(name, operation)
        results.append(result)
        raw_samples += samples
    # Exact same implementation, larger delivered histories and proposal batches.
    stress_rows = []
    for batches in (12,48,192):
        synthetic = []
        template = {r["route_id"]: r for r in rows[-3:]}
        for index in range(batches):
            sample = decision_ns - (batches-index)*100_000_000
            for route in ("0","1","2"):
                row = dict(template[route])
                row.update(sample_timestamp_ns=str(sample), available_timestamp_ns=str(sample+5_000_000))
                synthetic.append(row)
        result, samples = measurement(f"Window input {batches} batches", lambda data=synthetic: context_from_rows(data, decision_ns))
        stress_rows.append(result)
        raw_samples += samples
    for proposals in (3,12,48):
        result, samples = measurement(f"Sequential batch of {proposals} proposal decisions", lambda n=proposals: [online() for _ in range(n)], repetitions=1000)
        stress_rows.append(result)
        raw_samples += samples
    csv_new(output / "component_latency.csv", results)
    csv_new(output / "stress_latency.csv", stress_rows)
    csv_new(output / "raw_timing_samples.csv", raw_samples)
    volume_rows = []
    for config in matrix["scenarios"]:
        scenario_value = N12Scenario(**config)
        input_path = ROOT / f"experiments/abilene/test/groups/{scenario_value.scenario_id}/probe-route0/raw"
        delivered = received_rows(input_path, int(scenario_value.decision_at_seconds*1e9))
        volume_rows.append({"scenario_id":scenario_value.scenario_id,"seed":scenario_value.seed,
                           "delivered_records":len(delivered),"value_bytes":len(delivered)*25,
                           "incremental_metadata_bytes":len(delivered)*16,
                           "receiver_audit_bytes":len(delivered)*8,
                           "csv_file_bytes":(input_path/"telemetry.csv").stat().st_size})
    csv_new(output / "telemetry_volume_per_proposal.csv", volume_rows)
    # Explicit serialization sizes; these are local serialization measurements,
    # not packets observed on a telemetry transport.
    baseline_bytes, metadata_bytes, audit_bytes = 0,0,0
    for row in rows:
        payload = struct.pack("!IQdIB",int(row["route_id"]),int(row["bottleneck_capacity_bps"]),float(row["propagation_delay_ms"]),int(row["queue_packets"]),int(row["link_up"]))
        envelope = struct.pack("!IIQ",1,int(row["sample_timestamp_ns"])//100_000_000,int(row["sample_timestamp_ns"]))
        receipt = struct.pack("!Q",int(row["available_timestamp_ns"]))
        baseline_bytes += len(payload)
        metadata_bytes += len(envelope)
        audit_bytes += len(receipt)
    new(output / "environment_and_memory.json", {
        "python": sys.version, "python_executable": sys.executable, "numpy": np.__version__,
        "scikit_learn": sklearn.__version__, "os": platform.platform(), "processor": platform.processor(),
        "logical_processors": os.cpu_count(), "process_memory": process_memory(),
        "risk_model_json_storage_bytes": len(json.dumps(original).encode()),
        "selected_learning_model_tree_array_bytes": sum(estimator.tree_.__getstate__()["nodes"].nbytes + estimator.tree_.__getstate__()["values"].nbytes for estimator in learned_model.estimators_),
        "received_csv_file_bytes": (raw / "telemetry.csv").stat().st_size,
        "serialized_baseline_value_bytes": baseline_bytes, "serialized_incremental_transmitted_metadata_bytes": metadata_bytes,
        "serialized_controller_receipt_audit_bytes": audit_bytes, "delivered_record_count": len(rows),
        "mean_baseline_value_bytes_per_proposal":statistics.mean(r["value_bytes"] for r in volume_rows),
        "mean_incremental_metadata_bytes_per_proposal":statistics.mean(r["incremental_metadata_bytes"] for r in volume_rows),
        "mean_receiver_audit_bytes_per_proposal":statistics.mean(r["receiver_audit_bytes"] for r in volume_rows),
        "analytical_attempted_baseline_payload_bytes_per_second": 750,
        "analytical_attempted_incremental_metadata_bytes_per_second": 480,
        "excluded_overhead": "transport headers, TLS or report signatures, retransmissions, acknowledgements, SDN actuation and energy",
        "hardware_scope": "host-local Python microbenchmark; no live network deployment",
        "concurrent_native_simulations": False,
        "cached_contract_scope": "in-process prototype lookup plus graph-version assertion; production invalidation and distributed synchronization not measured",
    })
    print("CONTROLLER_OVERHEAD_BENCHMARKS_COMPLETE", flush=True)


def sensitivity(topology: str) -> None:
    output = ROOT / f"analysis/cost_sensitivity/{topology}"
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    matrix = read(ROOT / f"configs/{topology}-test-matrix.json")
    original = read(ROOT / "code/configs/experiments/n13_selected_policy.json")["controller_thresholds"]["risk_model"]
    records = []
    for config in matrix["scenarios"]:
        scenario = N12Scenario(**config)
        root = ROOT / f"experiments/{topology}/test/groups/{scenario.scenario_id}"
        outcomes = {a: _csv_rows(root / f"action-{a}/raw/summary.csv")[0] for a in ("route0","route1","route2")}
        values = {a: utility(row) for a,row in outcomes.items()}
        for window in (4,8,12):
            for budget in (8,20,32):
                for deadline in (100,200,300):
                    ctx = context(root / "probe-route0/raw", scenario, window=window,budget=budget,deadline_ms=deadline)
                    selected = admit(scenario,ctx,ctx["greedy_action"],original,0.30)
                    row = row_record(scenario.seed,f"W{window}B{budget}D{deadline}",selected,values,outcomes)
                    row.update({"window":window,"budget":budget,"deadline_ms":deadline,
                                "serialized_evidence_bytes":ctx["snapshot"]["delivered_batches"]*3*41})
                    records.append(row)
    csv_new(output / "records.csv", records)
    new(output / "analysis.json", summary(records,baseline="W12B32D300"))


if __name__ == "__main__":
    main()
    sensitivity("abilene")
