"""Exhaustive conformance and trace-order tests for N03 finite semantics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ictc.domain.enums import AdmissionDecision, ContractState, SafetyDisposition
from ictc.formal.benchmark import build_n03_evidence, render_n03_evidence
from ictc.formal.conformance import run_conformance
from ictc.formal.semantics import (
    REFERENCE_TRANSITIONS,
    ActionDisposition,
    AdmissionFacts,
    CrossLayerTrace,
    TraceEvent,
    TraceRecord,
    TraceViolation,
    all_admission_facts,
    maximal_lifecycle_paths,
    reference_admission,
    validate_reference_invariants,
    validate_trace,
)

ROOT = Path(__file__).resolve().parents[2]


def record(event: TraceEvent, action: ActionDisposition | None = None) -> TraceRecord:
    return TraceRecord(event=event, action=action)


def test_complete_admission_abstraction_has_zero_production_mismatches() -> None:
    facts = all_admission_facts()
    summary = run_conformance()

    assert len(facts) == 6912
    assert summary.admission_cases == len(facts)
    assert summary.admission_mismatches == 0
    assert summary.invariant_violations == 0
    assert summary.deterministic_digest_mismatches == 0
    assert summary.lifecycle_cases == len(ContractState) ** 2
    assert summary.lifecycle_mismatches == 0
    assert sum(count for _, count in summary.decision_counts) == len(facts)


def test_reference_invariants_detect_impossible_admit_and_defer_results() -> None:
    nonidentified = AdmissionFacts(
        deadline_met=True,
        safety=SafetyDisposition.SAFE,
        policy_permitted=True,
        identified=False,
        contract_state=ContractState.SATISFIED,
        estimable=True,
        uncertainty_acceptable=True,
        effect_within_limit=True,
        remediation_possible=False,
    )
    actual = reference_admission(nonidentified)
    assert actual.decision is AdmissionDecision.REJECT
    assert validate_reference_invariants(nonidentified, actual) == ()

    impossible_admit = actual.__class__(
        decision=AdmissionDecision.ADMIT,
        action=ActionDisposition.FALLBACK,
        decisive_gate=actual.decisive_gate,
        required_reason=actual.required_reason,
    )
    assert set(validate_reference_invariants(nonidentified, impossible_admit)) == {
        "ADMIT_IFF_EXECUTABLE",
        "NON_ADMIT_REQUIRES_FALLBACK",
    }

    impossible_defer = actual.__class__(
        decision=AdmissionDecision.DEFER,
        action=ActionDisposition.FALLBACK,
        decisive_gate=actual.decisive_gate,
        required_reason=actual.required_reason,
    )
    assert validate_reference_invariants(nonidentified, impossible_defer) == (
        "DEFER_REQUIRES_REMEDIATION",
    )


def test_all_maximal_lifecycle_paths_are_ordered_and_terminal() -> None:
    paths = maximal_lifecycle_paths()
    assert paths
    for path in paths:
        assert path[0] is ContractState.PROPOSED
        assert not REFERENCE_TRANSITIONS[path[-1]]
        for source, target in zip(path, path[1:], strict=False):
            assert target in REFERENCE_TRANSITIONS[source]
        if ContractState.SATISFIED in path:
            positions = {state: path.index(state) for state in path}
            assert (
                positions[ContractState.REQUIREMENTS_GENERATED] < positions[ContractState.SATISFIED]
            )
            assert positions[ContractState.OFFERS_RECEIVED] < positions[ContractState.SATISFIED]
            assert positions[ContractState.COMMITTED] < positions[ContractState.SATISFIED]
            assert positions[ContractState.ACTIVE] < positions[ContractState.SATISFIED]


def test_valid_admit_and_preexecution_breach_fallback_traces_pass() -> None:
    admit = CrossLayerTrace(
        decision=AdmissionDecision.ADMIT,
        records=(
            record(TraceEvent.QUERY_DECLARED),
            record(TraceEvent.CONTRACT_SATISFIED),
            record(TraceEvent.DECISION_PERSISTED),
            record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
            record(TraceEvent.ORACLE_LABEL_ATTACHED),
        ),
    )
    breach = CrossLayerTrace(
        decision=AdmissionDecision.ADMIT,
        records=(
            record(TraceEvent.CONTRACT_SATISFIED),
            record(TraceEvent.DECISION_PERSISTED),
            record(TraceEvent.BREACH_RECORDED),
            record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),
            record(TraceEvent.ORACLE_LABEL_ATTACHED),
        ),
    )
    assert validate_trace(admit) == ()
    assert validate_trace(breach) == ()


@pytest.mark.parametrize(
    ("trace", "violations"),
    [
        (
            CrossLayerTrace(
                AdmissionDecision.REJECT,
                (
                    record(TraceEvent.ORACLE_LABEL_ATTACHED),
                    record(TraceEvent.DECISION_PERSISTED),
                    record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),
                ),
            ),
            {TraceViolation.ORACLE_BEFORE_DECISION},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.DEFER,
                (
                    record(TraceEvent.DECISION_PERSISTED),
                    record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
                ),
            ),
            {TraceViolation.PROPOSED_ACTION_WITHOUT_ADMIT},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.ADMIT,
                (
                    record(TraceEvent.DECISION_PERSISTED),
                    record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
                ),
            ),
            {TraceViolation.ADMIT_WITHOUT_PRIOR_CONTRACT},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.ADMIT,
                (
                    record(TraceEvent.CONTRACT_SATISFIED),
                    record(TraceEvent.DECISION_PERSISTED),
                    record(TraceEvent.BREACH_RECORDED),
                    record(TraceEvent.ACTION_EXECUTED, ActionDisposition.PROPOSED),
                ),
            ),
            {TraceViolation.PROPOSED_ACTION_AFTER_BREACH},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.REJECT,
                (record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),),
            ),
            {TraceViolation.MISSING_OR_DUPLICATE_DECISION},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.REJECT,
                (record(TraceEvent.DECISION_PERSISTED),),
            ),
            {TraceViolation.MISSING_OR_DUPLICATE_ACTION},
        ),
        (
            CrossLayerTrace(
                AdmissionDecision.REJECT,
                (
                    record(TraceEvent.ACTION_EXECUTED, ActionDisposition.FALLBACK),
                    record(TraceEvent.DECISION_PERSISTED),
                ),
            ),
            {TraceViolation.ACTION_BEFORE_DECISION},
        ),
    ],
)
def test_invalid_cross_layer_traces_are_rejected(
    trace: CrossLayerTrace, violations: set[TraceViolation]
) -> None:
    assert violations <= set(validate_trace(trace))


def test_trace_records_reject_missing_or_misplaced_action_fields() -> None:
    with pytest.raises(ValueError, match="requires an action"):
        TraceRecord(TraceEvent.ACTION_EXECUTED)
    with pytest.raises(ValueError, match="only action execution"):
        TraceRecord(TraceEvent.QUERY_DECLARED, ActionDisposition.FALLBACK)


def test_semantics_source_is_independent_of_implementations_and_truth() -> None:
    source = (ROOT / "src/ictc/formal/semantics.py").read_text(encoding="utf-8")
    for token in (
        "ictc.controller",
        "ictc.protocol",
        "ictc.contract",
        "ictc.pilot",
        "ictc.oracle",
        "post_action_outcome",
        "reference_admissible",
    ):
        assert token not in source


def test_n03_evidence_is_deterministic_and_complete() -> None:
    payload = json.loads(render_n03_evidence())
    assert payload == build_n03_evidence()
    assert payload["admission_conformance"]["cases"] == 6912
    assert payload["admission_conformance"]["mismatches"] == 0
    assert payload["lifecycle_conformance"]["mismatches"] == 0
    assert payload["cross_layer_traces"]["valid_admit"]["violations"] == []


def test_tracked_n03_evidence_matches_fresh_rendering() -> None:
    evidence = ROOT / "docs/milestones/evidence/n03_exhaustive_semantics.json"
    assert evidence.read_text(encoding="utf-8") == render_n03_evidence()
