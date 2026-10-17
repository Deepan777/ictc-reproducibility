"""Formal causal witnesses and finite-DAG identification algorithms."""

from ictc.causal.counterexamples import (
    CounterexamplePair,
    FiniteSCM,
    PotentialOutcomeRow,
    build_m02_evidence,
    counterexample_pairs,
    hidden_cross_traffic_pair,
    observational_equivalence_pair,
    render_m02_evidence,
)
from ictc.causal.estimation import (
    AdjustmentObservation,
    EstimationReason,
    EstimationResult,
    StratumDiagnostic,
    estimate_stratified_backdoor,
)
from ictc.causal.graph import (
    CausalDAG,
    GraphComplexityError,
    InvalidCausalGraphError,
)
from ictc.causal.identification import (
    BackdoorIdentifier,
    IdentifiabilityReason,
    IdentifiabilityResult,
)

__all__ = [
    "AdjustmentObservation",
    "BackdoorIdentifier",
    "CausalDAG",
    "CounterexamplePair",
    "EstimationReason",
    "EstimationResult",
    "FiniteSCM",
    "GraphComplexityError",
    "IdentifiabilityReason",
    "IdentifiabilityResult",
    "InvalidCausalGraphError",
    "PotentialOutcomeRow",
    "StratumDiagnostic",
    "build_m02_evidence",
    "counterexample_pairs",
    "estimate_stratified_backdoor",
    "hidden_cross_traffic_pair",
    "observational_equivalence_pair",
    "render_m02_evidence",
]
