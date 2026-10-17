"""Deterministic N03 exhaustive-semantics evidence."""

from __future__ import annotations

import json

from ictc.domain.enums import AdmissionDecision
from ictc.formal.conformance import run_conformance
from ictc.formal.semantics import (
    ActionDisposition,
    CrossLayerTrace,
    TraceEvent,
    TraceRecord,
    maximal_lifecycle_paths,
    validate_trace,
)

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _record(event: TraceEvent, action: ActionDisposition | None = None) -> TraceRecord:
    return TraceRecord(event=event, action=action)


def _trace_payload(trace: CrossLayerTrace) -> dict[str, JsonValue]:
    return {
        "decision": trace.decision.value,
        "records": [
            {
                "event": record.event.value,
                "action": None if record.action is None else record.action.value,
            }
            for record in trace.records
        ],
        "violations": [item.value for item in validate_trace(trace)],
    }


def build_n03_evidence() -> dict[str, JsonValue]:
    conformance = run_conformance()
    valid_admit = CrossLayerTrace(
        decision=AdmissionDecision.ADMIT,
        records=(
            _record(TraceEvent.QUERY_DECLARED),
            _record(TraceEvent.CONTRACT_PLANNED),
            _record(TraceEvent.CONTRACT_SATISFIED),
            _record(TraceEvent.DECISION_PERSISTED),
            _record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
            _record(TraceEvent.ORACLE_LABEL_ATTACHED),
        ),
    )
    valid_breach_fallback = CrossLayerTrace(
        decision=AdmissionDecision.ADMIT,
        records=(
            _record(TraceEvent.QUERY_DECLARED),
            _record(TraceEvent.CONTRACT_SATISFIED),
            _record(TraceEvent.DECISION_PERSISTED),
            _record(TraceEvent.BREACH_RECORDED),
            _record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),
            _record(TraceEvent.ORACLE_LABEL_ATTACHED),
        ),
    )
    invalid_oracle_order = CrossLayerTrace(
        decision=AdmissionDecision.REJECT,
        records=(
            _record(TraceEvent.QUERY_DECLARED),
            _record(TraceEvent.ORACLE_LABEL_ATTACHED),
            _record(TraceEvent.DECISION_PERSISTED),
            _record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),
        ),
    )
    invalid_nonadmit_action = CrossLayerTrace(
        decision=AdmissionDecision.DEFER,
        records=(
            _record(TraceEvent.QUERY_DECLARED),
            _record(TraceEvent.DECISION_PERSISTED),
            _record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
        ),
    )
    lifecycle_paths = maximal_lifecycle_paths()
    return {
        "schema_version": "n03-exhaustive-semantics-evidence-v1",
        "milestone": "N03",
        "admission_conformance": {
            "cases": conformance.admission_cases,
            "mismatches": conformance.admission_mismatches,
            "reference_invariant_violations": conformance.invariant_violations,
            "deterministic_digest_mismatches": (conformance.deterministic_digest_mismatches),
            "decision_counts": dict(conformance.decision_counts),
            "decisive_gate_counts": dict(conformance.decisive_gate_counts),
        },
        "lifecycle_conformance": {
            "state_target_cases": conformance.lifecycle_cases,
            "mismatches": conformance.lifecycle_mismatches,
            "maximal_path_count": len(lifecycle_paths),
            "maximal_paths": [[state.value for state in path] for path in lifecycle_paths],
        },
        "cross_layer_traces": {
            "valid_admit": _trace_payload(valid_admit),
            "valid_breach_fallback": _trace_payload(valid_breach_fallback),
            "invalid_oracle_order": _trace_payload(invalid_oracle_order),
            "invalid_nonadmit_action": _trace_payload(invalid_nonadmit_action),
        },
        "enumeration": "complete finite Cartesian product; no random sampling",
        "claim_boundary": (
            "Exhaustive only for the declared finite abstraction; not a proof of physical, OS, "
            "network, causal-assumption, or arbitrary Python correctness."
        ),
    }


def render_n03_evidence() -> str:
    return json.dumps(build_n03_evidence(), indent=2, sort_keys=True) + "\n"
