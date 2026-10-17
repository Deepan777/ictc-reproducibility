"""Frozen M10 protocol loader and completeness validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ictc.pilot.ablation import AblationId, PilotProtocol

MASTER_METRICS = (
    "non_identifiable_action_rate",
    "false_causal_admission_rate",
    "unnecessary_deferral_rate",
    "harmful_action_rate",
    "invariant_violation_rate",
    "causal_effect_estimation_error",
    "confidence_interval_coverage",
    "confidence_interval_width",
    "end_to_end_latency",
    "packet_delivery_ratio",
    "packet_loss_ratio",
    "jitter",
    "throughput",
    "deadline_violation_rate",
    "route_changes",
    "control_oscillation_count",
    "convergence_time",
    "recovery_time",
    "service_availability",
    "contract_acquisition_latency",
    "action_admission_latency",
    "telemetry_bytes_per_proposed_action",
    "protocol_messages_per_action",
    "telemetry_sampling_cost",
    "cpu_time",
    "wall_clock_time",
    "peak_ram",
    "gpu_memory_if_used",
    "storage_volume",
    "scalability",
    "breach_detection_time",
    "fallback_execution_time",
)
METRIC_CLASSIFICATIONS = frozenset({"PRIMARY", "SECONDARY", "DIAGNOSTIC", "NOT_MEASURABLE_IN_M10"})


@dataclass(frozen=True, slots=True)
class FrozenPilotProtocol:
    protocol_id: str
    configuration_digest: str
    operational: PilotProtocol
    payload: dict[str, object]


def _digest(payload: dict[str, object]) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest().upper()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_amendment(
    path: Path, source: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    base_name = source.get("base_protocol")
    expected_digest = source.get("base_protocol_sha256")
    if not isinstance(base_name, str) or not isinstance(expected_digest, str):
        return source, source
    base_path = (path.parent / base_name).resolve()
    if _file_digest(base_path) != expected_digest:
        raise ValueError("M10 protocol amendment base digest does not match")
    base = cast(dict[str, object], json.loads(base_path.read_text(encoding="utf-8")))
    merged = dict(base)
    protocol_id = source.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("M10 protocol amendment requires a new protocol ID")
    merged["protocol_id"] = protocol_id
    raw_overrides = source.get("metric_overrides")
    if not isinstance(raw_overrides, list) or not all(
        isinstance(item, dict) for item in raw_overrides
    ):
        raise ValueError("M10 protocol amendment metric overrides are invalid")
    metrics = cast(list[dict[str, object]], merged["metrics"])
    by_id = {cast(str, item["metric_id"]): dict(item) for item in metrics}
    for raw_override in cast(list[dict[str, object]], raw_overrides):
        metric_id = raw_override.get("metric_id")
        if not isinstance(metric_id, str) or metric_id not in by_id:
            raise ValueError("M10 protocol amendment names an unknown metric")
        by_id[metric_id].update(raw_override)
    merged["metrics"] = [by_id[metric_id] for metric_id in MASTER_METRICS]
    merged["amendment"] = source
    return merged, {"base_protocol": base, "amendment": source}


def load_frozen_protocol(path: Path) -> FrozenPilotProtocol:
    source = cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))
    payload, digest_payload = _resolve_amendment(path, source)
    if payload.get("schema_version") != "1.0.0":
        raise ValueError("M10 protocol schema version is unsupported")
    protocol_id = payload.get("protocol_id")
    if not isinstance(protocol_id, str) or not protocol_id:
        raise ValueError("M10 protocol ID is required")
    splits = payload.get("authorized_splits")
    if splits != ["train", "development"]:
        raise ValueError("M10 dry-run protocol must authorize only train and development")
    methods = payload.get("ablations")
    expected_methods = [item.value for item in AblationId]
    if methods != expected_methods:
        raise ValueError("M10 protocol must contain ordered A0-A16")
    raw_metrics = payload.get("metrics")
    if not isinstance(raw_metrics, list) or not all(isinstance(item, dict) for item in raw_metrics):
        raise ValueError("M10 protocol metrics must be a list of objects")
    metrics = cast(list[dict[str, object]], raw_metrics)
    identifiers = [item.get("metric_id") for item in metrics]
    if tuple(identifiers) != MASTER_METRICS:
        raise ValueError("M10 protocol must classify every master-prompt metric in order")
    for item in metrics:
        if item.get("classification") not in METRIC_CLASSIFICATIONS:
            raise ValueError("M10 protocol metric has an invalid classification")
        if not isinstance(item.get("denominator"), str) or not item["denominator"]:
            raise ValueError("every M10 protocol metric requires a denominator or applicability")
        if not isinstance(item.get("reason"), str) or not item["reason"]:
            raise ValueError("every M10 protocol metric requires a reason")
    raw = payload.get("operational_thresholds")
    if not isinstance(raw, dict):
        raise ValueError("M10 operational thresholds are required")
    thresholds = cast(dict[str, object], raw)
    uncertainty = thresholds.get("uncertainty_width_grid_ms")
    grid = thresholds.get("deadline_budget_grid")
    if not isinstance(uncertainty, list) or not isinstance(grid, list):
        raise ValueError("M10 sensitivity grids must be lists")
    operational = PilotProtocol(
        freshness_limit_ticks=int(cast(int, thresholds["freshness_limit_ticks"])),
        synchronization_limit_ticks=int(cast(int, thresholds["synchronization_limit_ticks"])),
        acquisition_deadline_ticks=int(cast(int, thresholds["acquisition_deadline_ticks"])),
        telemetry_budget_bytes=int(cast(int, thresholds["telemetry_budget_bytes"])),
        minimum_trust_millionths=int(cast(int, thresholds["minimum_trust_millionths"])),
        minimum_source_diversity=int(cast(int, thresholds["minimum_source_diversity"])),
        maximum_uncertainty_width_ms=float(
            cast(int | float, thresholds["maximum_uncertainty_width_ms"])
        ),
        maximum_acceptable_effect_ms=float(
            cast(int | float, thresholds["maximum_acceptable_effect_ms"])
        ),
        maximum_deadline_violation_rate=float(
            cast(int | float, thresholds["maximum_deadline_violation_rate"])
        ),
        uncertainty_width_grid_ms=tuple(float(cast(int | float, item)) for item in uncertainty),
        deadline_budget_grid=tuple(
            (
                int(cast(dict[str, int], item)["deadline_ticks"]),
                int(cast(dict[str, int], item)["budget_bytes"]),
            )
            for item in grid
        ),
    )
    return FrozenPilotProtocol(protocol_id, _digest(digest_payload), operational, payload)
