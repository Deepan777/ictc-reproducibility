from __future__ import annotations

import json
from pathlib import Path

from ictc.ns3.n12 import EVENTS, METHODS, TRAFFIC_PROFILES, sha256
from ictc.ns3.n13_transport import TOPOLOGIES, TRANSPORT_SEEDS, load_matrix

REPOSITORY = Path(__file__).resolve().parents[2]


def test_n13_transport_matrix_is_complete_and_policy_locked() -> None:
    matrix_path = (
        REPOSITORY / "configs" / "experiments" / "n13_external_transport_confirmation.json"
    )
    raw, scenarios = load_matrix(matrix_path)
    assert len(scenarios) == 2 * 3 * 4 * 3 * 20 == 1440
    assert {item.topology for item in scenarios} == set(TOPOLOGIES)
    assert {item.seed for item in scenarios} == set(TRANSPORT_SEEDS)
    assert {item.traffic for item in scenarios} == set(TRAFFIC_PROFILES)
    assert {item.event for item in scenarios} == set(EVENTS)
    assert raw["methods"] == list(METHODS)
    assert raw["selected_policy_sha256"] == sha256(
        REPOSITORY / "configs" / "experiments" / "n13_selected_policy.json"
    )
    assert "unchanged" in raw["policy_transport"]


def test_n13_transport_derivations_preserve_connected_backbone_and_routes() -> None:
    root = REPOSITORY / "configs" / "experiments" / "n13_transport_derived"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert [item["topology_id"] for item in manifest["topologies"]] == list(TOPOLOGIES)
    for topology in manifest["topologies"]:
        payload = json.loads((root / topology["json"]).read_text(encoding="utf-8"))
        assert len(payload["nodes"]) == topology["nodes"]
        assert len(payload["edges"]) == topology["edges"]
        assert len(payload["od_cells"]) == 3
        assert len(payload["background_pairs"]) == 6
        for cell in payload["od_cells"]:
            assert len(cell["routes"]) == 3
            assert all(route[0] == cell["source"] for route in cell["routes"])
            assert all(route[-1] == cell["target"] for route in cell["routes"])


def test_n13_transport_native_source_keeps_required_network_mechanisms() -> None:
    source = (REPOSITORY / "ns3" / "scratch" / "ictc_n13_transport.cc").read_text(encoding="utf-8")
    assert "LoadTopology(topologyConfig)" in source
    assert "TcpCubic::GetTypeId()" in source
    assert "FqCoDelQueueDisc" in source
    assert all(event in source for event in ("failure", "capacity", "cross-shift"))
    assert "AFFECTED_EDGES[matrix]" in source
