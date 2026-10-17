"""Verify and transform the frozen N12 Abilene and MAWI public inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path
from typing import Any

ABILENE_SHA256 = "8CD694280D98B336BB9B51FC3B2129A514F1B1B1AC80F2022ACA02A57EF1E371"
MAWI_SHA256 = "5823E9208FC97E878929494EEA201E15D27221CB590DCC1BFF6A0E51C5D5FE07"
GRAPHML_NS = {"g": "http://graphml.graphdrawing.org/xmlns"}
OD_PAIRS = ((3, 0), (5, 1), (8, 2))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"input checksum mismatch for {path}: {actual} != {expected}")


def _graphml(path: Path) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    root = ET.parse(path).getroot()
    key_names = {
        item.attrib["id"]: item.attrib.get("attr.name", "")
        for item in root.findall("g:key", GRAPHML_NS)
    }
    graph = root.find("g:graph", GRAPHML_NS)
    if graph is None or graph.attrib.get("edgedefault") != "undirected":
        raise ValueError("Abilene GraphML must contain one undirected graph")
    nodes: list[dict[str, Any]] = []
    for element in graph.findall("g:node", GRAPHML_NS):
        values = {
            key_names[data.attrib["key"]]: (data.text or "")
            for data in element.findall("g:data", GRAPHML_NS)
        }
        nodes.append(
            {
                "node_id": int(element.attrib["id"]),
                "label": values["label"],
                "latitude": float(values["Latitude"]),
                "longitude": float(values["Longitude"]),
            }
        )
    edges = sorted(
        (
            min(int(item.attrib["source"]), int(item.attrib["target"])),
            max(int(item.attrib["source"]), int(item.attrib["target"])),
        )
        for item in graph.findall("g:edge", GRAPHML_NS)
    )
    if [item["node_id"] for item in nodes] != list(range(11)) or len(edges) != 14:
        raise ValueError("unexpected Abilene graph dimensions")
    return nodes, edges


def _haversine_km(left: dict[str, Any], right: dict[str, Any]) -> float:
    radius = 6371.0088
    lat1, lat2 = math.radians(left["latitude"]), math.radians(right["latitude"])
    dlat = lat2 - lat1
    dlon = math.radians(right["longitude"] - left["longitude"])
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(value))


def _simple_paths(edges: list[tuple[int, int]], source: int, target: int) -> list[list[int]]:
    adjacency: dict[int, list[int]] = {index: [] for index in range(11)}
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)
    queue: deque[list[int]] = deque([[source]])
    paths: list[list[int]] = []
    while queue and len(paths) < 24:
        path = queue.popleft()
        if path[-1] == target:
            paths.append(path)
            continue
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor not in path:
                queue.append([*path, neighbor])
    # Greedy diversity after shortest path: minimize shared undirected edges, then hop count.
    chosen: list[list[int]] = []
    while paths and len(chosen) < 3:
        used = {
            tuple(sorted((left, right)))
            for path in chosen
            for left, right in zip(path, path[1:], strict=False)
        }
        paths.sort(
            key=lambda path: (
                sum(tuple(sorted(edge)) in used for edge in zip(path, path[1:], strict=False)),
                len(path),
                path,
            )
        )
        chosen.append(paths.pop(0))
    if len(chosen) != 3:
        raise ValueError(f"could not derive three paths for {source}->{target}")
    return chosen


def _mawi(path: Path) -> dict[str, float | int | str]:
    content = path.read_text(encoding="utf-8")
    patterns = {
        "trace_id": r"<b>Id:</b>\s*(\d+)",
        "duration_seconds": r"<b>TotalTime:</b>\s*([0-9.]+)",
        "mean_rate_mbps": r"<b>AvgRate:</b>\s*([0-9.]+)Mbps",
        "rate_stddev_mbps": r"<b>AvgRate:</b>.*stddev:([0-9.]+)M",
        "total_packets": r"\n total\s+(\d+)",
        "tcp_packets": r"\n  tcp\s+(\d+).*?\)\s+([0-9.]+)\n",
        "udp_packets": r"\n  udp\s+(\d+).*?\)\s+([0-9.]+)\n",
    }
    matches = {name: re.search(pattern, content) for name, pattern in patterns.items()}
    if any(match is None for match in matches.values()):
        missing = [name for name, match in matches.items() if match is None]
        raise ValueError(f"missing MAWI calibration fields: {missing}")
    tcp = matches["tcp_packets"]
    udp = matches["udp_packets"]
    assert tcp is not None and udp is not None
    tcp_packets, udp_packets = int(tcp.group(1)), int(udp.group(1))
    protocol_total = tcp_packets + udp_packets
    return {
        "trace_id": matches["trace_id"].group(1),  # type: ignore[union-attr]
        "duration_seconds": float(matches["duration_seconds"].group(1)),  # type: ignore[union-attr]
        "mean_rate_mbps": float(matches["mean_rate_mbps"].group(1)),  # type: ignore[union-attr]
        "rate_stddev_mbps": float(matches["rate_stddev_mbps"].group(1)),  # type: ignore[union-attr]
        "total_packets": int(matches["total_packets"].group(1)),  # type: ignore[union-attr]
        "tcp_packets": tcp_packets,
        "udp_packets": udp_packets,
        "tcp_mean_bytes": float(tcp.group(2)),
        "udp_mean_bytes": float(udp.group(2)),
        "tcp_share_of_tcp_udp_packets": tcp_packets / protocol_total,
        "udp_share_of_tcp_udp_packets": udp_packets / protocol_total,
    }


def prepare(abilene: Path, mawi: Path, output: Path) -> dict[str, Any]:
    _verify(abilene, ABILENE_SHA256)
    _verify(mawi, MAWI_SHA256)
    nodes, raw_edges = _graphml(abilene)
    by_id = {item["node_id"]: item for item in nodes}
    edges = []
    for index, (left, right) in enumerate(raw_edges):
        distance = _haversine_km(by_id[left], by_id[right])
        edges.append(
            {
                "edge_id": index,
                "left": left,
                "right": right,
                "distance_km": round(distance, 6),
                # Fibre route inflation 1.25 and approximately 200 km/ms propagation.
                "one_way_delay_ms": round(max(1.0, distance * 1.25 / 200.0), 6),
                # OC-192 is scaled to 25 Mbps for tractable packet-level experiments.
                "capacity_mbps": 25.0,
            }
        )
    routes: list[dict[str, Any]] = []
    for matrix_id, (source, target) in enumerate(OD_PAIRS):
        for route_id, path in enumerate(_simple_paths(raw_edges, source, target)):
            routes.append(
                {
                    "matrix_id": matrix_id,
                    "source": source,
                    "target": target,
                    "route_id": route_id,
                    "nodes": path,
                }
            )
    document: dict[str, Any] = {
        "schema_version": "n12-derived-inputs-v1",
        "sources": {
            "abilene_graphml_sha256": ABILENE_SHA256,
            "mawi_statistics_sha256": MAWI_SHA256,
        },
        "transform": {
            "geodesic": "WGS84-like haversine, radius 6371.0088 km",
            "fibre_route_inflation": 1.25,
            "propagation_km_per_ms": 200.0,
            "minimum_one_way_delay_ms": 1.0,
            "nominal_scaled_capacity_mbps": 25.0,
            "route_selection": (
                "greedy minimum shared-edge count, then hop count, then lexicographic"
            ),
        },
        "nodes": nodes,
        "edges": edges,
        "routes": routes,
        "mawi_calibration": _mawi(mawi),
    }
    output.mkdir(parents=True, exist_ok=True)
    derived_target = output / "n12_derived_inputs.json"
    derived_target.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output / "n12_topology.csv").open("w", newline="", encoding="utf-8") as handle:
        topology_writer = csv.DictWriter(handle, fieldnames=tuple(edges[0]))
        topology_writer.writeheader()
        topology_writer.writerows(edges)
    with (output / "n12_routes.csv").open("w", newline="", encoding="utf-8") as handle:
        route_writer = csv.writer(handle)
        route_writer.writerow(("matrix_id", "source", "target", "route_id", "nodes"))
        for row in routes:
            route_writer.writerow(
                (
                    *[row[key] for key in ("matrix_id", "source", "target", "route_id")],
                    ";".join(str(item) for item in row["nodes"]),
                )
            )
    manifest = {
        "schema_version": "n12-input-transform-manifest-v1",
        "derived_json_sha256": sha256(derived_target),
        "topology_csv_sha256": sha256(output / "n12_topology.csv"),
        "routes_csv_sha256": sha256(output / "n12_routes.csv"),
    }
    (output / "n12_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--abilene", required=True, type=Path)
    parser.add_argument("--mawi", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    print(json.dumps(prepare(arguments.abilene, arguments.mawi, arguments.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
