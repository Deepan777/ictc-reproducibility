"""Executable finite reference semantics and conformance utilities."""

from ictc.formal.semantics import (
    ActionDisposition,
    AdmissionFacts,
    CrossLayerTrace,
    DecisiveGate,
    ReferenceAdmission,
    TraceEvent,
    TraceRecord,
    TraceViolation,
    all_admission_facts,
    maximal_lifecycle_paths,
    reference_admission,
    validate_reference_invariants,
    validate_trace,
)

__all__ = [
    "ActionDisposition",
    "AdmissionFacts",
    "CrossLayerTrace",
    "DecisiveGate",
    "ReferenceAdmission",
    "TraceEvent",
    "TraceRecord",
    "TraceViolation",
    "all_admission_facts",
    "maximal_lifecycle_paths",
    "reference_admission",
    "validate_reference_invariants",
    "validate_trace",
]
