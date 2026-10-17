"""Code-only reconstruction checks; no historical policy or outcomes are loaded."""

import csv
import importlib.util
import math
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "policy_reconstruction", SOURCE / "reproduction/reconstruct_policy.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def parameter_csv(path, *, mutation=None):
    rows = [
        dict(
            topology="abilene",
            risk_model="original",
            feature=feature,
            mean="0.0",
            scale="1.0",
            coefficient="0.0",
            intercept="0.0",
            regularization_C="0.05",
        )
        for feature in MODULE.RISK_FEATURE_NAMES
    ]
    if mutation is not None:
        mutation(rows)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_reconstruction_uses_declared_feature_order(tmp_path):
    source = parameter_csv(tmp_path / "params.csv", mutation=lambda rows: rows.reverse())
    model = MODULE.model_from_csv(source, expected_sha256=None)
    assert model["feature_names"] == list(MODULE.RISK_FEATURE_NAMES)
    assert model["maximum_predicted_harm_probability"] == 0.3
    assert model["intercept"] == 0.0
    assert len(MODULE.model_digest(model)) == 64


@pytest.mark.parametrize(
    "mutation",
    [
        lambda rows: rows.pop(),
        lambda rows: rows.append(rows[0].copy()),
        lambda rows: rows[0].update(scale="0"),
        lambda rows: rows[0].update(scale="nan"),
        lambda rows: rows[0].update(coefficient="inf"),
        lambda rows: rows[0].update(intercept="1"),
        lambda rows: rows[0].update(regularization_C="-1"),
    ],
)
def test_reconstruction_rejects_invalid_parameters(tmp_path, mutation):
    source = parameter_csv(tmp_path / "params.csv", mutation=mutation)
    with pytest.raises(ValueError):
        MODULE.model_from_csv(source, expected_sha256=None)


def test_reconstruction_rejects_nonmatching_original(tmp_path):
    source = parameter_csv(tmp_path / "params.csv")
    with pytest.raises(ValueError, match="retained original"):
        MODULE.model_from_csv(source)


def test_reconstruction_never_overwrites(tmp_path, monkeypatch):
    source = parameter_csv(tmp_path / "params.csv")
    model = MODULE.model_from_csv(source, expected_sha256=None)
    monkeypatch.setattr(MODULE, "model_from_csv", lambda path: model)
    output = tmp_path / "policy.json"
    result = MODULE.reconstruct(source, output)
    assert result["reconstruction"]["fitting_performed"] is False
    assert math.isfinite(result["controller_thresholds"]["risk_model"]["intercept"])
    with pytest.raises(FileExistsError):
        MODULE.reconstruct(source, output)
