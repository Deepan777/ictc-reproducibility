"""Versioned, immutable serialization for the development-frozen M09 suite."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import cast

from ictc.baselines.methods import (
    CONFIDENCE_THRESHOLD_CANDIDATES,
    CORRELATION_TOP_K_CANDIDATES,
    DIGITAL_TWIN_ALPHA_CANDIDATES,
    MAXIMUM_SELECTIVE_ERROR,
    AdaptiveCoverageController,
    ConstraintOnlyController,
    CorrelationSelectorController,
    DigitalTwinController,
    FixedCausalController,
    FixedTelemetryController,
    LinearClassifier,
    LinearRegressor,
    NoGateController,
    OracleController,
    SelectiveConfidenceController,
    StaticContractController,
)
from ictc.baselines.runner import BaselineSuite, TrainingCost
from ictc.baselines.types import BaselineController, BaselineId

FROZEN_BASELINE_SCHEMA_VERSION = "1.0.0"

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _pair_set_digest(pair_ids: tuple[str, ...]) -> str:
    if not pair_ids or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("frozen corpus pair IDs must be nonempty and unique")
    payload = "\n".join(sorted(pair_ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _model_payload(model: LinearClassifier | LinearRegressor) -> dict[str, JsonValue]:
    return {
        "mean": list(model.mean),
        "scale": list(model.scale),
        "coefficients": list(model.coefficients),
        "intercept": model.intercept,
    }


def _parameter_payload(controller: BaselineController) -> dict[str, JsonValue]:
    if isinstance(controller, OracleController):
        return {
            "loss_weight": controller.loss_weight,
            "jitter_weight": controller.jitter_weight,
            "deadline_weight": controller.deadline_weight,
        }
    if isinstance(controller, NoGateController):
        return {}
    if isinstance(controller, FixedTelemetryController):
        return {
            "minimum_improvement": controller.minimum_improvement,
            "utilization_weight": controller.utilization_weight,
        }
    if isinstance(controller, AdaptiveCoverageController):
        return {
            "field_weights": [[name, value] for name, value in controller.field_weights],
            "maximum_selected_fields": controller.maximum_selected_fields,
            "freshness_penalty": controller.freshness_penalty,
            "switching_penalty": controller.switching_penalty,
            "minimum_score": controller.minimum_score,
        }
    if isinstance(controller, SelectiveConfidenceController):
        return {
            "model": _model_payload(controller.model),
            "fields": list(controller.fields),
            "confidence_threshold": controller.confidence_threshold,
            "regularization_c": controller.regularization_c,
        }
    if isinstance(controller, FixedCausalController):
        return {
            "identified": controller.identified,
            "point_effect_ms": controller.point_effect_ms,
            "upper_effect_ms": controller.upper_effect_ms,
            "sample_size": controller.sample_size,
            "graph_id": controller.graph_id,
            "graph_digest": controller.graph_digest,
            "adjustment_set": list(controller.adjustment_set),
        }
    if isinstance(controller, ConstraintOnlyController):
        return {
            "peak_utilization_limit": controller.peak_utilization_limit,
            "average_utilization_limit": controller.average_utilization_limit,
            "queue_limit": controller.queue_limit,
        }
    if isinstance(controller, StaticContractController):
        return {
            "required_fields": list(controller.required_fields),
            "maximum_age_ticks": controller.maximum_age_ticks,
        }
    if isinstance(controller, DigitalTwinController):
        return {
            "model": _model_payload(controller.model),
            "fields": list(controller.fields),
            "ridge_alpha": controller.ridge_alpha,
        }
    if isinstance(controller, CorrelationSelectorController):
        return {
            "model": _model_payload(controller.model),
            "selected_fields": list(controller.selected_fields),
            "mutual_information": [[name, value] for name, value in controller.mutual_information],
            "top_k": controller.top_k,
        }
    raise TypeError(f"unsupported baseline controller: {type(controller).__name__}")


def frozen_baseline_payload(
    suite: BaselineSuite,
    *,
    implementation_commit: str,
    corpus_root: str,
    manifest_tree_digest: str,
    training_pair_ids: tuple[str, ...],
    development_pair_ids: tuple[str, ...],
    smoke_pair_ids: tuple[str, ...],
) -> dict[str, JsonValue]:
    """Build the canonical reloadable configuration and its split provenance."""
    if len(implementation_commit) != 40:
        raise ValueError("implementation commit must be a full Git object ID")
    if not corpus_root or len(manifest_tree_digest) != 64:
        raise ValueError("corpus root and manifest-tree digest are required")
    train_digest = _pair_set_digest(training_pair_ids)
    development_digest = _pair_set_digest(development_pair_ids)
    smoke_digest = _pair_set_digest(smoke_pair_ids)
    if set(training_pair_ids) & set(development_pair_ids):
        raise ValueError("training and development pair IDs must be disjoint")
    if not set(smoke_pair_ids) <= set(development_pair_ids):
        raise ValueError("smoke pairs must be a development subset")

    controllers: list[JsonValue] = []
    for controller, cost in zip(suite.controllers, suite.training_costs, strict=True):
        policy = controller.observation_policy
        controllers.append(
            {
                "baseline_id": controller.baseline_id.value,
                "configuration_digest": controller.configuration_digest,
                "observation_policy": {
                    "fields": list(policy.fields),
                    "maximum_items": policy.maximum_items,
                    "maximum_bytes": policy.maximum_bytes,
                    "adaptive": policy.adaptive,
                },
                "parameters": _parameter_payload(controller),
                "training_cost": {
                    "training_time_ns": cost.training_time_ns,
                    "parameter_count": cost.parameter_count,
                    "model_storage_bytes": cost.model_storage_bytes,
                    "cpu_only": cost.cpu_only,
                },
            }
        )
    return {
        "schema_version": FROZEN_BASELINE_SCHEMA_VERSION,
        "milestone": "M09",
        "implementation_commit": implementation_commit,
        "deadline_ns": suite.deadline_ns,
        "source_corpus": {
            "root": corpus_root,
            "manifest_tree_digest": manifest_tree_digest,
            "training_pair_count": len(training_pair_ids),
            "training_pair_set_digest": train_digest,
            "development_pair_count": len(development_pair_ids),
            "development_pair_set_digest": development_digest,
            "smoke_pair_count": len(smoke_pair_ids),
            "smoke_pair_set_digest": smoke_digest,
        },
        "tuning_policy": {
            "fit_split": "train",
            "selection_split": "development",
            "forbidden_split": "test",
            "confidence_threshold_candidates": list(CONFIDENCE_THRESHOLD_CANDIDATES),
            "maximum_selective_error": MAXIMUM_SELECTIVE_ERROR,
            "digital_twin_alpha_candidates": list(DIGITAL_TWIN_ALPHA_CANDIDATES),
            "correlation_top_k_candidates": list(CORRELATION_TOP_K_CANDIDATES),
        },
        "controllers": controllers,
    }


def write_frozen_baseline_suite(payload: dict[str, JsonValue], target: Path) -> None:
    """Atomically create a frozen configuration without overwriting evidence."""
    if target.exists():
        raise FileExistsError(f"refusing to overwrite frozen baseline configuration: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tempfile.NamedTemporaryFile(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _dict(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a string-keyed object")
    return cast(dict[str, object], value)


def _list(value: object, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return cast(list[object], value)


def _str(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _float(value: object, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else _float(value, name)


def _float_tuple(value: object, name: str) -> tuple[float, ...]:
    return tuple(_float(item, name) for item in _list(value, name))


def _str_tuple(value: object, name: str) -> tuple[str, ...]:
    return tuple(_str(item, name) for item in _list(value, name))


def _linear_classifier(value: object) -> LinearClassifier:
    payload = _dict(value, "linear classifier")
    return LinearClassifier(
        _float_tuple(payload.get("mean"), "classifier mean"),
        _float_tuple(payload.get("scale"), "classifier scale"),
        _float_tuple(payload.get("coefficients"), "classifier coefficients"),
        _float(payload.get("intercept"), "classifier intercept"),
    )


def _linear_regressor(value: object) -> LinearRegressor:
    payload = _dict(value, "linear regressor")
    return LinearRegressor(
        _float_tuple(payload.get("mean"), "regressor mean"),
        _float_tuple(payload.get("scale"), "regressor scale"),
        _float_tuple(payload.get("coefficients"), "regressor coefficients"),
        _float(payload.get("intercept"), "regressor intercept"),
    )


def _controller_from_payload(
    payload: dict[str, object], oracle: OracleController
) -> BaselineController:
    baseline_id = BaselineId(_str(payload.get("baseline_id"), "baseline ID"))
    parameters = _dict(payload.get("parameters"), "baseline parameters")
    if baseline_id is BaselineId.B0:
        expected = (
            _float(parameters.get("loss_weight"), "oracle loss weight"),
            _float(parameters.get("jitter_weight"), "oracle jitter weight"),
            _float(parameters.get("deadline_weight"), "oracle deadline weight"),
        )
        actual = (oracle.loss_weight, oracle.jitter_weight, oracle.deadline_weight)
        if actual != expected:
            raise ValueError("evaluation oracle weights differ from the frozen configuration")
        controller: BaselineController = oracle
    elif baseline_id is BaselineId.B1:
        controller = NoGateController()
    elif baseline_id is BaselineId.B2:
        controller = FixedTelemetryController(
            _float(parameters.get("minimum_improvement"), "B2 minimum improvement"),
            _float(parameters.get("utilization_weight"), "B2 utilization weight"),
        )
    elif baseline_id is BaselineId.B3:
        raw_weights = _list(parameters.get("field_weights"), "B3 field weights")
        weights = tuple(
            (
                _str(_list(item, "B3 field-weight entry")[0], "B3 field name"),
                _float(_list(item, "B3 field-weight entry")[1], "B3 field weight"),
            )
            for item in raw_weights
        )
        controller = AdaptiveCoverageController(
            weights,
            _int(parameters.get("maximum_selected_fields"), "B3 maximum fields"),
            _float(parameters.get("freshness_penalty"), "B3 freshness penalty"),
            _float(parameters.get("switching_penalty"), "B3 switching penalty"),
            _float(parameters.get("minimum_score"), "B3 minimum score"),
        )
    elif baseline_id is BaselineId.B4:
        controller = SelectiveConfidenceController(
            _linear_classifier(parameters.get("model")),
            _str_tuple(parameters.get("fields"), "B4 fields"),
            _float(parameters.get("confidence_threshold"), "B4 confidence threshold"),
            _float(parameters.get("regularization_c"), "B4 regularization C"),
        )
    elif baseline_id is BaselineId.B5:
        identified = parameters.get("identified")
        if not isinstance(identified, bool):
            raise ValueError("B5 identified must be Boolean")
        controller = FixedCausalController(
            identified,
            _optional_float(parameters.get("point_effect_ms"), "B5 point effect"),
            _optional_float(parameters.get("upper_effect_ms"), "B5 upper effect"),
            _int(parameters.get("sample_size"), "B5 sample size"),
            _str(parameters.get("graph_id"), "B5 graph ID"),
            _str(parameters.get("graph_digest"), "B5 graph digest"),
            _str_tuple(parameters.get("adjustment_set"), "B5 adjustment set"),
        )
    elif baseline_id is BaselineId.B6:
        controller = ConstraintOnlyController(
            _float(parameters.get("peak_utilization_limit"), "B6 peak limit"),
            _float(parameters.get("average_utilization_limit"), "B6 average limit"),
            _int(parameters.get("queue_limit"), "B6 queue limit"),
        )
    elif baseline_id is BaselineId.B7:
        controller = StaticContractController(
            _str_tuple(parameters.get("required_fields"), "B7 required fields"),
            _int(parameters.get("maximum_age_ticks"), "B7 maximum age"),
        )
    elif baseline_id is BaselineId.B8:
        controller = DigitalTwinController(
            _linear_regressor(parameters.get("model")),
            _str_tuple(parameters.get("fields"), "B8 fields"),
            _float(parameters.get("ridge_alpha"), "B8 ridge alpha"),
        )
    else:
        raw_mi = _list(parameters.get("mutual_information"), "B9 mutual information")
        mutual_information = tuple(
            (
                _str(_list(item, "B9 MI entry")[0], "B9 MI field"),
                _float(_list(item, "B9 MI entry")[1], "B9 MI score"),
            )
            for item in raw_mi
        )
        controller = CorrelationSelectorController(
            _linear_classifier(parameters.get("model")),
            _str_tuple(parameters.get("selected_fields"), "B9 selected fields"),
            mutual_information,
            _int(parameters.get("top_k"), "B9 top-k"),
        )

    expected_digest = _str(payload.get("configuration_digest"), "configuration digest")
    if controller.configuration_digest != expected_digest:
        raise ValueError(f"{baseline_id.value} frozen configuration digest does not verify")
    policy = _dict(payload.get("observation_policy"), "observation policy")
    actual_policy = controller.observation_policy
    expected_policy = (
        _str_tuple(policy.get("fields"), "observation fields"),
        _int(policy.get("maximum_items"), "maximum telemetry items"),
        _int(policy.get("maximum_bytes"), "maximum telemetry bytes"),
        policy.get("adaptive"),
    )
    if not isinstance(expected_policy[3], bool) or expected_policy != (
        actual_policy.fields,
        actual_policy.maximum_items,
        actual_policy.maximum_bytes,
        actual_policy.adaptive,
    ):
        raise ValueError(f"{baseline_id.value} frozen observation policy does not verify")
    return controller


def load_frozen_baseline_suite(target: Path, *, oracle: OracleController) -> BaselineSuite:
    """Load and fully verify a frozen B0-B9 suite for later pilot evaluation."""
    raw = json.loads(target.read_text(encoding="utf-8"))
    root = _dict(raw, "frozen baseline configuration")
    if root.get("schema_version") != FROZEN_BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported frozen baseline schema version")
    if root.get("milestone") != "M09":
        raise ValueError("frozen baseline configuration has the wrong milestone")
    tuning = _dict(root.get("tuning_policy"), "tuning policy")
    if (
        tuning.get("fit_split"),
        tuning.get("selection_split"),
        tuning.get("forbidden_split"),
    ) != ("train", "development", "test"):
        raise ValueError("frozen tuning policy violates split isolation")

    raw_controllers = _list(root.get("controllers"), "controllers")
    payloads = tuple(_dict(value, "controller") for value in raw_controllers)
    if tuple(payload.get("baseline_id") for payload in payloads) != tuple(
        item.value for item in BaselineId
    ):
        raise ValueError("frozen suite must contain ordered B0-B9 configurations")
    controllers = tuple(_controller_from_payload(payload, oracle) for payload in payloads)
    costs: list[TrainingCost] = []
    for baseline_id, payload in zip(BaselineId, payloads, strict=True):
        raw_cost = _dict(payload.get("training_cost"), "training cost")
        cpu_only = raw_cost.get("cpu_only")
        if not isinstance(cpu_only, bool):
            raise ValueError("training-cost CPU flag must be Boolean")
        costs.append(
            TrainingCost(
                baseline_id,
                _int(raw_cost.get("training_time_ns"), "training time"),
                _int(raw_cost.get("parameter_count"), "parameter count"),
                _int(raw_cost.get("model_storage_bytes"), "model storage"),
                cpu_only,
            )
        )
    return BaselineSuite(
        controllers,
        tuple(costs),
        _int(root.get("deadline_ns"), "baseline deadline"),
    )
