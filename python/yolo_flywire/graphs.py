from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import random

import numpy as np


@dataclass(frozen=True)
class DirectedGraph:
    num_nodes: int
    src: tuple[int, ...]
    dst: tuple[int, ...]
    weight: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive")
        if not (len(self.src) == len(self.dst) == len(self.weight)):
            raise ValueError("src, dst, and weight must have equal length")
        edges = list(zip(self.src, self.dst))
        if len(edges) != len(set(edges)):
            raise ValueError("duplicate directed edges are not allowed")
        for source, target in edges:
            if not (0 <= source < self.num_nodes and 0 <= target < self.num_nodes):
                raise ValueError("edge endpoint is outside node range")
        if any(not np.isfinite(value) for value in self.weight):
            raise ValueError("edge weights must be finite")

    @property
    def num_edges(self) -> int:
        return len(self.src)


def graph_fingerprint(graph: DirectedGraph) -> str:
    canonical_edges = sorted(
        (int(source), int(target), float(weight))
        for source, target, weight in zip(graph.src, graph.dst, graph.weight)
    )
    payload = {
        "num_nodes": graph.num_nodes,
        "edges": canonical_edges,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def random_sparse_graph(num_nodes: int, num_edges: int, seed: int) -> DirectedGraph:
    if num_nodes <= 1:
        raise ValueError("num_nodes must be greater than one")
    max_edges = num_nodes * (num_nodes - 1)
    if not (0 <= num_edges <= max_edges):
        raise ValueError("num_edges exceeds the simple directed graph capacity")

    candidates = [(source, target) for source in range(num_nodes) for target in range(num_nodes) if source != target]
    rng = np.random.default_rng(seed)
    selected = rng.choice(len(candidates), size=num_edges, replace=False) if num_edges else np.asarray([], dtype=int)
    edges = [candidates[int(index)] for index in selected]
    return DirectedGraph(
        num_nodes=num_nodes,
        src=tuple(source for source, _ in edges),
        dst=tuple(target for _, target in edges),
        weight=tuple(1.0 for _ in edges),
    )


def rewire_degree_preserving(
    graph: DirectedGraph,
    seed: int,
    swaps: int,
    *,
    max_attempt_factor: int = 100,
) -> DirectedGraph:
    """Directed double-edge swaps with exact diagonal population edges fixed.

    Only off-diagonal edges are eligible for swapping. This preserves the directed
    in/out degree sequence, the edge-weight multiset, and every diagonal edge with
    its weight exactly. The standard-library RNG makes the rewired topology stable
    across NumPy versions.
    """
    if swaps < 0:
        raise ValueError("swaps must be non-negative")
    if swaps == 0:
        return graph

    src = list(graph.src)
    dst = list(graph.dst)
    weight = list(graph.weight)
    eligible = [index for index, (source, target) in enumerate(zip(src, dst)) if source != target]
    if len(eligible) < 2:
        raise ValueError("at least two off-diagonal edges are required for rewiring")

    rng = random.Random(seed)
    edge_set = set(zip(src, dst))
    successful = 0
    attempts = 0
    max_attempts = max(max_attempt_factor * swaps, 1)

    while successful < swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.sample(eligible, 2)
        a, b = src[i], dst[i]
        c, d = src[j], dst[j]

        proposed_one = (a, d)
        proposed_two = (c, b)
        if proposed_one[0] == proposed_one[1] or proposed_two[0] == proposed_two[1]:
            continue
        if proposed_one == proposed_two:
            continue

        old_one = (a, b)
        old_two = (c, d)
        occupied = edge_set - {old_one, old_two}
        if proposed_one in occupied or proposed_two in occupied:
            continue
        if proposed_one == old_one and proposed_two == old_two:
            continue

        edge_set.remove(old_one)
        edge_set.remove(old_two)
        dst[i] = d
        dst[j] = b
        edge_set.add(proposed_one)
        edge_set.add(proposed_two)
        successful += 1

    if successful != swaps:
        raise ValueError(
            f"could not complete {swaps} degree-preserving swaps within {max_attempts} attempts"
        )

    return DirectedGraph(
        num_nodes=graph.num_nodes,
        src=tuple(src),
        dst=tuple(dst),
        weight=tuple(weight),
    )


def lesion_graph(
    graph: DirectedGraph,
    *,
    node_ids: tuple[int, ...] = (),
    edge_indices: tuple[int, ...] = (),
) -> DirectedGraph:
    """Return a new graph with declared nodes and/or original edge indices removed."""
    removed_nodes = set(node_ids)
    removed_edges = set(edge_indices)
    if any(node < 0 or node >= graph.num_nodes for node in removed_nodes):
        raise ValueError("lesion node id is outside node range")
    if any(index < 0 or index >= graph.num_edges for index in removed_edges):
        raise ValueError("lesion edge index is outside edge range")
    if len(removed_nodes) >= graph.num_nodes:
        raise ValueError("lesion cannot remove every node")

    surviving_nodes = [node for node in range(graph.num_nodes) if node not in removed_nodes]
    remap = {node: index for index, node in enumerate(surviving_nodes)}
    edges: list[tuple[int, int, float]] = []
    for index, (source, target, weight) in enumerate(zip(graph.src, graph.dst, graph.weight)):
        if index in removed_edges or source in removed_nodes or target in removed_nodes:
            continue
        edges.append((remap[source], remap[target], weight))

    return DirectedGraph(
        num_nodes=len(surviving_nodes),
        src=tuple(source for source, _, _ in edges),
        dst=tuple(target for _, target, _ in edges),
        weight=tuple(weight for _, _, weight in edges),
    )
