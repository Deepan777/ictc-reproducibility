"""Enumerations shared by ICTC domain and protocol models."""

from __future__ import annotations

from enum import StrEnum


class ActionClass(StrEnum):
    """Supported intervention families in the initial prototype."""

    ROUTING = "ROUTING"


class ActionKind(StrEnum):
    """Routing actions available to the initial controller."""

    KEEP_CURRENT_PATH = "KEEP_CURRENT_PATH"
    REROUTE_TO_PATH = "REROUTE_TO_PATH"


class TargetOutcomeMetric(StrEnum):
    """Outcome metrics preserved independently in every experiment."""

    END_TO_END_LATENCY = "END_TO_END_LATENCY"
    PACKET_LOSS_RATIO = "PACKET_LOSS_RATIO"
    JITTER = "JITTER"
    DEADLINE_VIOLATION_RATE = "DEADLINE_VIOLATION_RATE"


class VariableRole(StrEnum):
    """Causal role declared for a variable within the temporal graph."""

    TREATMENT = "TREATMENT"
    OUTCOME = "OUTCOME"
    CONFOUNDER = "CONFOUNDER"
    MEDIATOR = "MEDIATOR"
    INSTRUMENT = "INSTRUMENT"
    SELECTION = "SELECTION"
    IRRELEVANT = "IRRELEVANT"
    LATENT = "LATENT"


class IdentificationMethod(StrEnum):
    """Supported identification strategies; estimation is implemented later."""

    BACKDOOR = "BACKDOOR"
    FRONTDOOR = "FRONTDOOR"
    RANDOMIZED = "RANDOMIZED"
    NONE = "NONE"


class TelemetrySourceClass(StrEnum):
    """Classes of telemetry source that may offer evidence."""

    ENDPOINT = "ENDPOINT"
    SWITCH = "SWITCH"
    ROUTER = "ROUTER"
    LINK_MONITOR = "LINK_MONITOR"
    CONTROLLER = "CONTROLLER"
    ACTIVE_PROBE = "ACTIVE_PROBE"


class ContractState(StrEnum):
    """Lifecycle states defined by the ICTC protocol."""

    PROPOSED = "PROPOSED"
    REQUIREMENTS_GENERATED = "REQUIREMENTS_GENERATED"
    OFFERS_RECEIVED = "OFFERS_RECEIVED"
    COMMITTED = "COMMITTED"
    ACTIVE = "ACTIVE"
    SATISFIED = "SATISFIED"
    RELEASED = "RELEASED"
    INFEASIBLE = "INFEASIBLE"
    EXPIRED = "EXPIRED"
    BREACHED = "BREACHED"
    BREACHED_BEFORE_EXECUTION = "BREACHED_BEFORE_EXECUTION"
    CANCELLED = "CANCELLED"


class ClauseKind(StrEnum):
    """Evidence obligation clauses that compose a telemetry contract."""

    FRESHNESS = "FRESHNESS"
    SYNCHRONIZATION = "SYNCHRONIZATION"
    PROVENANCE = "PROVENANCE"
    TRUST = "TRUST"
    REDUNDANCY = "REDUNDANCY"
    PRIVACY = "PRIVACY"
    BANDWIDTH = "BANDWIDTH"
    DEADLINE = "DEADLINE"


class ClauseSatisfaction(StrEnum):
    """Runtime truth value for an individual contract clause."""

    PENDING = "PENDING"
    SATISFIED = "SATISFIED"
    UNSATISFIED = "UNSATISFIED"
    BREACHED = "BREACHED"


class BreachSeverity(StrEnum):
    """Severity categories used by breach and fallback handling."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class SafetyDisposition(StrEnum):
    """Safety assessment independent of causal identifiability."""

    SAFE = "SAFE"
    UNSAFE = "UNSAFE"
    UNKNOWN = "UNKNOWN"


class AdmissionDecision(StrEnum):
    """The only legal causal-admission decisions."""

    ADMIT = "ADMIT"
    MODIFY = "MODIFY"
    DEFER = "DEFER"
    REJECT = "REJECT"


class PrivacyClass(StrEnum):
    """Coarse privacy handling level for telemetry clauses."""

    PUBLIC = "PUBLIC"
    INTERNAL = "INTERNAL"
    RESTRICTED = "RESTRICTED"


class ReasonCode(StrEnum):
    """Explicit reasons retained for every nontrivial evidence decision."""

    IDENTIFIABLE = "IDENTIFIABLE"
    NOT_IDENTIFIABLE = "NOT_IDENTIFIABLE"
    ESTIMABLE = "ESTIMABLE"
    ESTIMATION_UNAVAILABLE = "ESTIMATION_UNAVAILABLE"
    INSUFFICIENT_ACTION_SUPPORT = "INSUFFICIENT_ACTION_SUPPORT"
    INSUFFICIENT_SAMPLE_SIZE = "INSUFFICIENT_SAMPLE_SIZE"
    MISSING_REQUIRED_TELEMETRY = "MISSING_REQUIRED_TELEMETRY"
    CONTRACT_INFEASIBLE = "CONTRACT_INFEASIBLE"
    STALE_OBSERVATION = "STALE_OBSERVATION"
    SYNCHRONIZATION_BOUND_VIOLATED = "SYNCHRONIZATION_BOUND_VIOLATED"
    PROVENANCE_UNVERIFIED = "PROVENANCE_UNVERIFIED"
    TRUST_BELOW_THRESHOLD = "TRUST_BELOW_THRESHOLD"
    REDUNDANCY_UNSATISFIED = "REDUNDANCY_UNSATISFIED"
    PRIVACY_CONSTRAINT_VIOLATED = "PRIVACY_CONSTRAINT_VIOLATED"
    BANDWIDTH_BUDGET_EXCEEDED = "BANDWIDTH_BUDGET_EXCEEDED"
    DEADLINE_EXCEEDED = "DEADLINE_EXCEEDED"
    UNCERTAINTY_TOO_WIDE = "UNCERTAINTY_TOO_WIDE"
    SAFETY_INVARIANT_VIOLATED = "SAFETY_INVARIANT_VIOLATED"
    POLICY_VIOLATION = "POLICY_VIOLATION"
    PROTOCOL_INVALID_TRANSITION = "PROTOCOL_INVALID_TRANSITION"
    CONTRACT_EXPIRED = "CONTRACT_EXPIRED"
    CONTRACT_BREACHED = "CONTRACT_BREACHED"
    CONTRACT_SATISFIED = "CONTRACT_SATISFIED"
    UNCERTAINTY_ACCEPTABLE = "UNCERTAINTY_ACCEPTABLE"
    SAFETY_PASSED = "SAFETY_PASSED"
    POLICY_PASSED = "POLICY_PASSED"
    ACTION_ADMITTED = "ACTION_ADMITTED"
    ACTION_MODIFIED = "ACTION_MODIFIED"
    ACTION_DEFERRED = "ACTION_DEFERRED"
    ACTION_REJECTED = "ACTION_REJECTED"
    FALLBACK_REQUIRED = "FALLBACK_REQUIRED"
    FALLBACK_EXECUTED = "FALLBACK_EXECUTED"
    CANCELLED = "CANCELLED"
