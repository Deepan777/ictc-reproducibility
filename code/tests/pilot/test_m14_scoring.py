from __future__ import annotations

from ictc.pilot.m14_scoring import _pressure


def test_pressure_requires_two_complete_path_samples() -> None:
    assert _pressure({"samples": [{"path_id": "path-current", "missing": True}]}) == 0
    assert (
        _pressure(
            {
                "samples": [
                    {
                        "path_id": "path-current",
                        "missing": False,
                        "queue_occupancy": 8,
                    },
                    {
                        "path_id": "path-candidate",
                        "missing": False,
                        "queue_occupancy": 3,
                    },
                ]
            }
        )
        == 1
    )
