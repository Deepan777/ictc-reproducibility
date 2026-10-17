from __future__ import annotations

from ictc.pilot.ablation import FixtureKind
from ictc.pilot.m14_decisions import _controller_fixture


def test_controller_fixture_uses_verification_results_not_condition_name() -> None:
    assert (
        _controller_fixture(
            {
                "condition_id": "complete",
                "contract_feasible": False,
                "samples": [{"provenance_valid": True}],
            }
        )
        is FixtureKind.CONTRACT_INFEASIBLE
    )
    assert (
        _controller_fixture(
            {
                "condition_id": "complete",
                "contract_feasible": True,
                "samples": [{"provenance_valid": False}],
            }
        )
        is FixtureKind.INVALID_PROVENANCE
    )
    assert (
        _controller_fixture(
            {
                "condition_id": "malicious_telemetry",
                "contract_feasible": True,
                "samples": [{"provenance_valid": True}],
            }
        )
        is FixtureKind.OBSERVED
    )
