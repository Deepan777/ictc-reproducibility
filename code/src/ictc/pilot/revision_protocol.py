"""Frozen protocol and bounded assignment for the post-M10 revision corpus."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ictc.domain.enums import ActionKind
from ictc.pilot.revision import (
    M10R_ASSIGNMENT_POLICY_ID,
    M10R_GRAPH_DIGEST,
    M10R_GRAPH_ID,
)

M10R_CONTEXT_FIELDS = (
    "service_relation",
    "propagation_delay_relation",
    "queue_limit_relation",
)
M10R_ESTIMATOR_SCOPES = ("global", "traffic-only", "context-local")
M10R_MATRIX_CONFIG_SHA256 = "D466E7ADCF0BC3186A2BBDD6EB7DEC30F43A6CF63DA0252D03DCC342ADE73558"
M10R_INHERITED_METRIC_PROTOCOL_FILE_SHA256 = (
    "F5E80511CB83DC3F7F7FB73A15819E586E92492CC0946BDF35D49AFCA3803438"
)
M10R_PRIMARY_ENDPOINTS = (
    ("false_causal_admission_rate", "reference non-identifiable opportunities"),
    ("harmful_event_burden", "all paired opportunities"),
    ("conditional_harmful_action_rate", "executed reroutes"),
)
M10R_DEVELOPMENT_THRESHOLD_GRID_MS = (0.0, -0.25, -0.5, -1.0, -2.0)
M10R_SAFETY_GATE_ORDER = (
    "identification",
    "contract",
    "estimability",
    "frozen_support",
    "uncertainty",
    "context_effect_safety",
    "policy",
)


@dataclass(frozen=True, slots=True)
class FrozenRevisionProtocol:
    """Validated pre-outcome declarations for revision train/development generation."""

    protocol_id: str
    configuration_digest: str
    assignment_policy_id: str
    assignment_probabilities: tuple[tuple[int, float], ...]
    minimum_per_arm: int
    analysis_seed: int
    payload: dict[str, object]

    def assignment_probability(self, queue_pressure_bin: int) -> float:
        try:
            return dict(self.assignment_probabilities)[queue_pressure_bin]
        except KeyError as error:
            raise ValueError("queue pressure bin must be -1, 0, or 1") from error


@dataclass(frozen=True, slots=True)
class FrozenRevisionDevelopmentProtocol:
    """Pre-truth safety comparison grid and development-only selection rule."""

    protocol_id: str
    configuration_digest: str
    base: FrozenRevisionProtocol
    upper_interval_threshold_grid_ms: tuple[float, ...]
    minimum_retained_base_reroute_fraction: float
    payload: dict[str, object]


@dataclass(frozen=True, slots=True)
class FrozenRevisionEvaluationProtocol:
    """Post-development operating rule frozen before any evaluation generation."""

    protocol_id: str
    configuration_digest: str
    development: FrozenRevisionDevelopmentProtocol
    selected_configuration_id: str
    selected_scope: str
    upper_interval_threshold_ms: float
    expected_pair_count: int
    payload: dict[str, object]


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _require_mapping(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"M10 revision protocol requires object field {key}")
    return cast(dict[str, object], value)


def load_revision_protocol(path: Path) -> FrozenRevisionProtocol:
    """Load the train/development-only revision declaration and reject evaluation access."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("M10 revision protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("M10 revision protocol schema version is unsupported")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("M10 revision protocol ID is required")
    if payload.get("authorized_splits") != ["train", "development"]:
        raise ValueError("M10 revision protocol must authorize only train and development")

    matrix = _require_mapping(payload, "corpus_matrix")
    if matrix != {
        "corpus_id": "ictc-m10r-contextual",
        "configuration_digest": M10R_MATRIX_CONFIG_SHA256,
        "train_pair_count": 2160,
        "development_pair_count": 720,
        "sealed_evaluation_pair_count": 720,
    }:
        raise ValueError("M10 revision protocol corpus matrix binding has changed")

    graph = _require_mapping(payload, "causal_graph")
    if graph != {"graph_id": M10R_GRAPH_ID, "graph_digest": M10R_GRAPH_DIGEST}:
        raise ValueError("M10 revision protocol graph does not match the committed graph")

    context = _require_mapping(payload, "context_schema")
    if context.get("fields") != list(M10R_CONTEXT_FIELDS):
        raise ValueError("M10 revision context schema fields are incomplete or reordered")
    if context.get("source") != "frozen pre-simulation path configuration":
        raise ValueError("M10 revision context must come from pre-simulation configuration")
    if context.get("outcome_or_post_action_fields_allowed") is not False:
        raise ValueError("M10 revision context cannot allow outcome or post-action fields")

    assignment = _require_mapping(payload, "assignment_policy")
    if assignment.get("policy_id") != M10R_ASSIGNMENT_POLICY_ID:
        raise ValueError("M10 revision assignment policy ID does not match the implementation")
    if assignment.get("conditioning_variable") != "QueuePressureBin":
        raise ValueError("M10 revision assignment may condition only on QueuePressureBin")
    raw_probabilities = assignment.get("reroute_probability_by_bin")
    if not isinstance(raw_probabilities, dict) or set(raw_probabilities) != {"-1", "0", "1"}:
        raise ValueError("M10 revision assignment requires all three queue-pressure bins")
    probabilities = tuple(
        (pressure, float(cast(dict[str, int | float], raw_probabilities)[str(pressure)]))
        for pressure in (-1, 0, 1)
    )
    lower = assignment.get("minimum_probability")
    upper = assignment.get("maximum_probability")
    if not isinstance(lower, int | float) or not isinstance(upper, int | float):
        raise ValueError("M10 revision assignment bounds must be numeric")
    if not 0 < lower <= upper < 1:
        raise ValueError("M10 revision assignment bounds must lie strictly within zero and one")
    if any(not lower <= probability <= upper for _pressure, probability in probabilities):
        raise ValueError("M10 revision assignment probability violates its frozen bounds")

    estimators = _require_mapping(payload, "estimators")
    if estimators.get("scopes") != list(M10R_ESTIMATOR_SCOPES):
        raise ValueError("M10 revision estimator comparison scopes are incomplete or reordered")
    if estimators.get("adjustment_set") != ["QueuePressureBin"]:
        raise ValueError("M10 revision estimator adjustment set must remain QueuePressureBin")
    if estimators.get("broader_fallback_for_unsupported_context") is not False:
        raise ValueError("M10 revision protocol cannot enable broader estimator fallback")
    minimum_per_arm = estimators.get("minimum_per_arm")
    if not isinstance(minimum_per_arm, int) or isinstance(minimum_per_arm, bool):
        raise ValueError("M10 revision minimum per-arm support must be an integer")
    if minimum_per_arm < 2:
        raise ValueError("M10 revision minimum per-arm support must be at least two")

    endpoints = payload.get("primary_endpoints")
    expected_endpoints = [
        {"metric_id": metric_id, "denominator": denominator}
        for metric_id, denominator in M10R_PRIMARY_ENDPOINTS
    ]
    if endpoints != expected_endpoints:
        raise ValueError("M10 revision primary endpoints or denominators have changed")
    inherited = _require_mapping(payload, "prior_metric_dispositions")
    if inherited != {
        "source": "m10_pilot_protocol_amendment_001.json",
        "source_file_sha256": M10R_INHERITED_METRIC_PROTOCOL_FILE_SHA256,
        "rule": "retain every prior M10 classification, denominator, and applicability",
    }:
        raise ValueError("M10 revision prior metric dispositions have changed")
    exclusions = _require_mapping(payload, "exclusion_policy")
    if exclusions.get("outcome_based_exclusions") != []:
        raise ValueError("M10 revision protocol cannot contain outcome-based exclusions")
    if exclusions.get("failed_runs") != "retain and classify":
        raise ValueError("M10 revision protocol must retain and classify failed runs")

    analysis = _require_mapping(payload, "analysis")
    analysis_seed = analysis.get("seed")
    if not isinstance(analysis_seed, int) or isinstance(analysis_seed, bool) or analysis_seed < 0:
        raise ValueError("M10 revision analysis seed must be a non-negative integer")
    if analysis.get("confidence_level") != 0.95:
        raise ValueError("M10 revision confidence level must remain frozen at 0.95")

    thresholds = _require_mapping(payload, "operating_thresholds")
    if thresholds != {
        "status": "development-only-not-yet-frozen",
        "values": None,
    }:
        raise ValueError("M10 revision operating thresholds must remain unset before development")
    evaluation = _require_mapping(payload, "sealed_evaluation")
    if (
        evaluation.get("status") != "NOT_AUTHORIZED"
        or evaluation.get("output_generated") is not False
    ):
        raise PermissionError("M10 revision sealed evaluation must remain unauthorized and absent")

    return FrozenRevisionProtocol(
        protocol_id,
        _digest(payload),
        M10R_ASSIGNMENT_POLICY_ID,
        probabilities,
        minimum_per_arm,
        analysis_seed,
        payload,
    )


def load_revision_development_protocol(path: Path) -> FrozenRevisionDevelopmentProtocol:
    """Load and validate the grid that must be frozen before development truth is read."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("M10 revision development protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("M10 revision development protocol schema version is unsupported")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("M10 revision development protocol ID is required")
    base_name = payload.get("base_protocol")
    if not isinstance(base_name, str) or not base_name:
        raise ValueError("M10 revision development protocol requires a base protocol")
    base = load_revision_protocol((path.parent / base_name).resolve())
    if payload.get("base_configuration_digest") != base.configuration_digest:
        raise ValueError("M10 revision development base protocol digest does not match")
    if payload.get("authorized_splits") != ["train", "development"]:
        raise ValueError("M10 revision development may authorize only train and development")

    safety = _require_mapping(payload, "safety_gate")
    if safety.get("execution_order") != list(M10R_SAFETY_GATE_ORDER):
        raise ValueError("M10 revision safety gate order has changed")
    if safety.get("comparison_scopes") != list(M10R_ESTIMATOR_SCOPES):
        raise ValueError("M10 revision safety comparison scopes have changed")
    raw_grid = safety.get("upper_interval_threshold_grid_ms")
    if raw_grid != list(M10R_DEVELOPMENT_THRESHOLD_GRID_MS):
        raise ValueError("M10 revision development threshold grid has changed")
    if safety.get("unsupported_context_action") != "DEFER_TO_KEEP":
        raise ValueError("M10 revision unsupported contexts must defer to KEEP")
    if safety.get("broader_fallback_for_unsupported_context") is not False:
        raise ValueError("M10 revision development cannot enable broader estimator fallback")
    if safety.get("mean_effect_not_individual_certificate") is not True:
        raise ValueError("M10 revision must disclaim individual-effect certification")

    selection = _require_mapping(payload, "selection_rule")
    if selection.get("causal_admissibility_required") is not True:
        raise ValueError("M10 revision safety selection must retain causal admissibility")
    if selection.get("maximum_false_causal_admissions") != 0:
        raise ValueError("M10 revision development cannot allow false causal admissions")
    retained = selection.get("minimum_retained_base_reroute_fraction")
    if not isinstance(retained, int | float) or isinstance(retained, bool):
        raise ValueError("M10 revision retained-reroute threshold must be numeric")
    if not 0 < retained <= 1:
        raise ValueError("M10 revision retained-reroute threshold must lie in (0, 1]")
    if selection.get("safety_subclaim_may_be_rejected_independently") is not True:
        raise ValueError("M10 revision safety subclaim must remain independently falsifiable")

    persistence = _require_mapping(payload, "decision_persistence")
    if persistence.get("required_before_development_truth_join") is not True:
        raise ValueError("M10 revision development decisions must precede truth joins")
    forbidden = persistence.get("forbidden_fields")
    expected_forbidden = [
        "keep_latency_ms",
        "reroute_latency_ms",
        "latency_effect_ms",
        "reroute_beneficial",
        "true_queue_pressure_bin",
    ]
    if forbidden != expected_forbidden:
        raise ValueError("M10 revision persisted-decision forbidden fields have changed")
    evaluation = _require_mapping(payload, "sealed_evaluation")
    if evaluation != {
        "status": "NOT_AUTHORIZED",
        "generation_allowed": False,
        "outcome_access_allowed": False,
    }:
        raise PermissionError("M10 revision development protocol opened sealed evaluation")

    return FrozenRevisionDevelopmentProtocol(
        protocol_id,
        _digest(payload),
        base,
        M10R_DEVELOPMENT_THRESHOLD_GRID_MS,
        float(retained),
        payload,
    )


def load_revision_evaluation_protocol(
    path: Path, *, repo_root: Path
) -> FrozenRevisionEvaluationProtocol:
    """Validate the post-development rule without granting evaluation access itself."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("M10 revision evaluation protocol must be a JSON object")
    payload = cast(dict[str, object], raw)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("M10 revision evaluation protocol schema version is unsupported")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("M10 revision evaluation protocol ID is required")
    development_name = payload.get("development_protocol")
    if not isinstance(development_name, str) or not development_name:
        raise ValueError("M10 revision evaluation requires its development protocol")
    development = load_revision_development_protocol((path.parent / development_name).resolve())
    if payload.get("development_protocol_configuration_digest") != development.configuration_digest:
        raise ValueError("M10 revision evaluation development digest differs")

    evidence = _require_mapping(payload, "development_evidence")
    evidence_path = evidence.get("path")
    evidence_digest = evidence.get("sha256")
    implementation_commit = evidence.get("implementation_commit")
    if not isinstance(evidence_path, str) or not isinstance(evidence_digest, str):
        raise ValueError("M10 revision evaluation evidence binding is invalid")
    if _file_digest((repo_root / evidence_path).resolve()) != evidence_digest:
        raise ValueError("M10 revision development evidence digest differs")
    if not isinstance(implementation_commit, str) or len(implementation_commit) != 40:
        raise ValueError("M10 revision implementation commit must be a full Git hash")

    selected = _require_mapping(payload, "selected_safety_gate")
    selected_configuration_id = selected.get("configuration_id")
    selected_scope = selected.get("scope")
    threshold = selected.get("upper_interval_threshold_ms")
    if selected_configuration_id != "context-local:upper<=-0.25ms":
        raise ValueError("M10 revision selected configuration has changed")
    if selected_scope != "context-local":
        raise ValueError("M10 revision selected scope must remain context-local")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError("M10 revision selected threshold must be numeric")
    if float(threshold) != -0.25:
        raise ValueError("M10 revision selected threshold has changed")
    if selected.get("minimum_per_arm") != development.base.minimum_per_arm:
        raise ValueError("M10 revision evaluation support threshold has changed")
    if selected.get("unsupported_context_action") != "DEFER_TO_KEEP":
        raise ValueError("M10 revision unsupported contexts must defer to KEEP")
    if selected.get("broader_fallback_for_unsupported_context") is not False:
        raise ValueError("M10 revision evaluation cannot use broader fallback")
    if selected.get("mean_effect_not_individual_certificate") is not True:
        raise ValueError("M10 revision evaluation must disclaim individual certification")
    if payload.get("execution_order") != list(M10R_SAFETY_GATE_ORDER):
        raise ValueError("M10 revision evaluation gate order has changed")

    evaluation = _require_mapping(payload, "evaluation")
    expected_pair_count = evaluation.get("expected_pair_count")
    if evaluation.get("split") != "test" or expected_pair_count != 720:
        raise ValueError("M10 revision evaluation split or pair count has changed")
    if evaluation.get("generation_and_execution_require_single_use_authorization") is not True:
        raise PermissionError("M10 revision evaluation requires single-use authorization")
    if evaluation.get("decisions_required_before_truth_join") is not True:
        raise ValueError("M10 revision evaluation decisions must precede truth joins")
    if evaluation.get("selective_exclusions") != []:
        raise ValueError("M10 revision evaluation cannot add selective exclusions")

    endpoints = payload.get("primary_endpoints")
    expected_endpoints = [
        {"metric_id": metric_id, "denominator": denominator}
        for metric_id, denominator in M10R_PRIMARY_ENDPOINTS
    ]
    if endpoints != expected_endpoints:
        raise ValueError("M10 revision evaluation endpoints have changed")
    decision_rule = _require_mapping(payload, "decision_rule")
    if decision_rule.get("allowed_values") != ["CONTINUE", "REVISE PROBLEM", "NO-GO"]:
        raise ValueError("M10 revision final decision values have changed")

    return FrozenRevisionEvaluationProtocol(
        protocol_id,
        _digest(payload),
        development,
        selected_configuration_id,
        selected_scope,
        float(threshold),
        expected_pair_count,
        payload,
    )


def revision_assignment(
    pair_id: str,
    queue_pressure_bin: int,
    protocol: FrozenRevisionProtocol,
) -> tuple[ActionKind, float, float]:
    """Assign observational treatment using only the frozen confounder probability."""
    if not pair_id:
        raise ValueError("revision assignment requires a pair ID")
    if protocol.assignment_policy_id != M10R_ASSIGNMENT_POLICY_ID:
        raise ValueError("revision assignment protocol ID mismatch")
    probability = protocol.assignment_probability(queue_pressure_bin)
    encoded = f"{protocol.assignment_policy_id}\0{pair_id}".encode()
    uniform = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") / 2**64
    action = ActionKind.REROUTE_TO_PATH if uniform < probability else ActionKind.KEEP_CURRENT_PATH
    return action, probability, uniform
