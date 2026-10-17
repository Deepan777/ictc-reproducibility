"""Freeze deterministic ns-3 transport inputs from official Topology Zoo GraphML files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx  # type: ignore[import-untyped]


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest().upper()


def haversine_ms(left: dict[str, Any], right: dict[str, Any]) -> float:
    if not all(key in left and key in right for key in ("Latitude", "Longitude")):
        return 2.0
    lat1, lon1 = math.radians(float(left["Latitude"])), math.radians(float(left["Longitude"]))
    lat2, lon2 = math.radians(float(right["Latitude"])), math.radians(float(right["Longitude"]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    distance_km = 6371.0088 * 2 * math.asin(min(1.0, math.sqrt(value)))
    return max(0.5, distance_km / 200.0)


def capacity_scale(raw: float | None, minimum: float, maximum: float) -> float:
    if raw is None or raw <= 0 or maximum <= minimum:
        return 1.0
    position = (math.log10(raw) - math.log10(minimum)) / (math.log10(maximum) - math.log10(minimum))
    return 0.7 + 0.6 * position


def edge_key(left: int, right: int) -> tuple[int, int]:
    return (left, right) if left < right else (right, left)


def choose_routes(graph: nx.Graph, nodes: list[int]) -> list[dict[str, Any]]:
    candidates: list[tuple[float, int, int, list[list[int]], tuple[int, int]]] = []
    for source_index, source in enumerate(nodes):
        for target in nodes[source_index + 1 :]:
            paths = []
            try:
                generator = nx.shortest_simple_paths(graph, source, target, weight="delay_ms")
                for _ in range(3):
                    paths.append(list(next(generator)))
            except (nx.NetworkXNoPath, StopIteration):
                continue
            route_edges = [
                {edge_key(path[index], path[index + 1]) for index in range(len(path) - 1)}
                for path in paths
            ]
            unique = sorted(route_edges[0] - route_edges[1] - route_edges[2])
            if not unique:
                continue
            geographic = (
                graph.nodes[source].get("Latitude") is not None
                and graph.nodes[target].get("Latitude") is not None
            )
            direct_score = nx.path_weight(graph, paths[0], "delay_ms")
            diversity = len(route_edges[0] | route_edges[1] | route_edges[2]) / max(
                1, len(route_edges[0])
            )
            score = direct_score * diversity * (1.0 if geographic else 0.5)
            candidates.append((score, source, target, paths, unique[len(unique) // 2]))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected: list[dict[str, Any]] = []
    used_endpoints: set[int] = set()
    for _score, source, target, paths, affected in candidates:
        if source in used_endpoints or target in used_endpoints:
            continue
        selected.append(
            {
                "matrix": len(selected),
                "source": source,
                "target": target,
                "routes": paths,
                "affected_edge": list(affected),
            }
        )
        used_endpoints.update((source, target))
        if len(selected) == 3:
            return selected
    raise ValueError("topology does not provide three endpoint-disjoint OD cells with three routes")


def transform(source: Path, output: Path, topology_id: str) -> dict[str, Any]:
    original = nx.read_graphml(source)
    internal = [
        node for node, data in original.nodes(data=True) if int(data.get("Internal", 1)) == 1
    ]
    graph0 = original.subgraph(internal).copy().to_undirected()
    component = sorted(
        nx.connected_components(graph0), key=lambda values: (-len(values), sorted(values))
    )[0]
    graph0 = graph0.subgraph(component).copy()
    ordered_original = sorted(
        graph0.nodes, key=lambda node: int(original.nodes[node].get("id", node))
    )
    remap = {node: index for index, node in enumerate(ordered_original)}
    graph = nx.Graph()
    for node in ordered_original:
        graph.add_node(remap[node], **dict(original.nodes[node]))
    raw_speeds = [
        float(data["LinkSpeedRaw"])
        for *_edge, data in graph0.edges(data=True)
        if data.get("LinkSpeedRaw") is not None and float(data["LinkSpeedRaw"]) > 0
    ]
    minimum, maximum = min(raw_speeds), max(raw_speeds)
    for left0, right0, data in graph0.edges(data=True):
        left, right = remap[left0], remap[right0]
        delay = haversine_ms(dict(original.nodes[left0]), dict(original.nodes[right0]))
        raw = float(data["LinkSpeedRaw"]) if data.get("LinkSpeedRaw") is not None else None
        scale = capacity_scale(raw, minimum, maximum)
        if graph.has_edge(left, right):
            current = graph[left][right]
            current["delay_ms"] = min(float(current["delay_ms"]), delay)
            current["capacity_scale"] = max(float(current["capacity_scale"]), scale)
            if raw is not None:
                current["published_capacity_bps"] = max(
                    float(current.get("published_capacity_bps", 0.0)), raw
                )
            current["parallel_links_collapsed"] = int(current["parallel_links_collapsed"]) + 1
        else:
            graph.add_edge(
                left,
                right,
                delay_ms=delay,
                capacity_scale=scale,
                published_capacity_bps=raw,
                parallel_links_collapsed=1,
            )
    routes = choose_routes(graph, sorted(graph.nodes))
    edges = []
    edge_ids: dict[tuple[int, int], int] = {}
    for edge_id, (left, right, data) in enumerate(sorted(graph.edges(data=True))):
        edge_ids[edge_key(left, right)] = edge_id
        edges.append(
            {
                "edge_id": edge_id,
                "left": left,
                "right": right,
                "delay_ms": round(float(data["delay_ms"]), 9),
                "capacity_scale": round(float(data["capacity_scale"]), 9),
                "published_capacity_bps": data.get("published_capacity_bps"),
                "parallel_links_collapsed": int(data["parallel_links_collapsed"]),
            }
        )
    background: list[list[int]] = []
    route_endpoints = {value for cell in routes for value in (cell["source"], cell["target"])}
    available = [node for node in sorted(graph.nodes) if node not in route_endpoints]
    for index in range(min(6, len(available) // 2)):
        background.append([available[index], available[-1 - index]])
    if len(background) < 6:
        raise ValueError("topology does not contain six independent background pairs")
    node_rows = [
        {
            "node_id": remap[node],
            "source_id": str(node),
            "label": str(original.nodes[node].get("label", node)),
            "latitude": original.nodes[node].get("Latitude"),
            "longitude": original.nodes[node].get("Longitude"),
            "country": original.nodes[node].get("Country"),
        }
        for node in ordered_original
    ]
    payload = {
        "schema_version": "n13-transport-topology-v1",
        "topology_id": topology_id,
        "official_graphml_file": source.name,
        "official_graphml_sha256": sha256(source),
        "transform": {
            "node_scope": "largest connected component of Internal=1 nodes",
            "parallel_edge_rule": (
                "collapse to minimum geodesic delay and maximum normalized capacity"
            ),
            "delay_rule": (
                "great-circle distance / 200 km/ms, minimum 0.5 ms; 2 ms if coordinates absent"
            ),
            "capacity_rule": (
                "published LinkSpeedRaw log-min-max mapped to [0.7,1.3]; missing speed maps to 1.0"
            ),
            "route_rule": (
                "three lowest-delay loopless paths; OD cells ranked by "
                "distance-diversity and endpoint-disjoint"
            ),
            "affected_edge_rule": "middle sorted route0 edge absent from both candidate routes",
        },
        "nodes": node_rows,
        "edges": edges,
        "od_cells": routes,
        "background_pairs": background,
    }
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    native = output.with_suffix(".topology")
    lines = [f"HEADER {topology_id} {len(node_rows)} {len(edges)} {len(routes)} {len(background)}"]
    lines.extend(
        f"EDGE {row['left']} {row['right']} {row['delay_ms']:.9f} {row['capacity_scale']:.9f}"
        for row in edges
    )
    for cell in routes:
        for route_id, path in enumerate(cell["routes"]):
            lines.append(
                f"ROUTE {cell['matrix']} {route_id} {cell['source']} {cell['target']} "
                + ",".join(map(str, path))
            )
        lines.append(f"AFFECTED {cell['matrix']} {edge_ids[tuple(cell['affected_edge'])]}")
    lines.extend(f"BACKGROUND {left} {right}" for left, right in background)
    native.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    return {
        "topology_id": topology_id,
        "json": output.name,
        "json_sha256": sha256(output),
        "native": native.name,
        "native_sha256": sha256(native),
        "nodes": len(node_rows),
        "edges": len(edges),
        "od_cells": len(routes),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    studies = []
    for filename, topology_id in (
        ("Geant2012.graphml", "geant-2012-03"),
        ("Garr201109.graphml", "garr-2011-09"),
    ):
        studies.append(
            transform(
                args.input_root / filename, args.output_root / f"{topology_id}.json", topology_id
            )
        )
    manifest = {
        "schema_version": "n13-transport-input-manifest-v1",
        "source": "Topology Zoo official dataset",
        "source_url": "https://topology-zoo.org/dataset.html",
        "topologies": studies,
    }
    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
