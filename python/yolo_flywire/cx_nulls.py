from __future__ import annotations

from collections import Counter
import hashlib
import json
import random
from typing import Any, Callable

from .cx_artifact import CxArtifact
from .graphs import DirectedGraph, graph_fingerprint


_DEGREE_ALGORITHM = "cx-directed-double-edge-swap-v1"
_BLOCK_ALGORITHM = "cx-block-preserving-double-edge-swap-v1"


def _hash_payload(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_blocks(artifact: CxArtifact) -> tuple[tuple[str, str], ...]:
    return tuple((node.role, node.family) for node in artifact.nodes)


def _rewire(
    artifact: CxArtifact,
    *,
    seed: int,
    swaps: int,
    algorithm: str,
    proposal_allowed: Callable[[int, int, int, int], bool],
    max_attempt_factor: int,
) -> CxArtifact:
    if type(seed) is not int:
        raise ValueError("rewiring seed must be an integer")
    if type(swaps) is not int or swaps < 0:
        raise ValueError("rewiring swaps must be a non-negative integer")
    if type(max_attempt_factor) is not int or max_attempt_factor <= 0:
        raise ValueError("max_attempt_factor must be a positive integer")

    src = list(artifact.graph.src)
    dst = list(artifact.graph.dst)
    weights = list(artifact.graph.weight)
    signs = list(artifact.edge_signs)
    eligible = [index for index, (source, target) in enumerate(zip(src, dst)) if source != target]
    if swaps and len(eligible) < 2:
        raise ValueError("at least two off-diagonal CX edges are required for rewiring")

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
        old_one = (a, b)
        old_two = (c, d)
        proposed_one = (a, d)
        proposed_two = (c, b)

        if proposed_one[0] == proposed_one[1] or proposed_two[0] == proposed_two[1]:
            continue
        if proposed_one == proposed_two:
            continue
        if proposed_one == old_one and proposed_two == old_two:
            continue
        if (
            (proposed_one in edge_set and proposed_one not in (old_one, old_two))
            or (proposed_two in edge_set and proposed_two not in (old_one, old_two))
        ):
            continue
        if not proposal_allowed(a, b, c, d):
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
            f"could not complete {swaps} CX rewiring swaps within {max_attempts} attempts; completed {successful}"
        )

    graph = DirectedGraph(
        num_nodes=artifact.graph.num_nodes,
        src=tuple(src),
        dst=tuple(dst),
        weight=tuple(weights),
    )
    # Edge slots retain their original source, weight, and sign. Only destinations
    # are swapped, so transmitter sign remains attached to the same presynaptic node.
    edge_signs = tuple(signs)
    fingerprint = _hash_payload(
        {
            "parent_fingerprint": artifact.fingerprint,
            "algorithm": algorithm,
            "seed": seed,
            "successful_swaps": successful,
            "graph_fingerprint": graph_fingerprint(graph),
            "edge_signs": list(edge_signs),
        }
    )
    return CxArtifact(
        nodes=artifact.nodes,
        graph=graph,
        input_indices=artifact.input_indices,
        core_indices=artifact.core_indices,
        output_indices=artifact.output_indices,
        edge_signs=edge_signs,
        source_hashes=artifact.source_hashes,
        matched_primary_types=artifact.matched_primary_types,
        fingerprint=fingerprint,
    )


def rewire_cx_degree_preserving(
    artifact: CxArtifact,
    seed: int,
    swaps: int,
    *,
    max_attempt_factor: int = 100,
) -> CxArtifact:
    return _rewire(
        artifact,
        seed=seed,
        swaps=swaps,
        algorithm=_DEGREE_ALGORITHM,
        proposal_allowed=lambda _a, _b, _c, _d: True,
        max_attempt_factor=max_attempt_factor,
    )


def rewire_cx_block_preserving(
    artifact: CxArtifact,
    seed: int,
    swaps: int,
    *,
    max_attempt_factor: int = 100,
) -> CxArtifact:
    blocks = _node_blocks(artifact)

    def preserves_blocks(_a: int, b: int, _c: int, d: int) -> bool:
        # Sources do not change under the directed destination swap. Requiring the
        # two target nodes to share one frozen (role, family) block preserves each
        # edge's source-block -> target-block category exactly.
        return blocks[b] == blocks[d]

    return _rewire(
        artifact,
        seed=seed,
        swaps=swaps,
        algorithm=_BLOCK_ALGORITHM,
        proposal_allowed=preserves_blocks,
        max_attempt_factor=max_attempt_factor,
    )


def _degrees(graph: DirectedGraph) -> tuple[tuple[int, ...], tuple[int, ...]]:
    incoming = [0] * graph.num_nodes
    outgoing = [0] * graph.num_nodes
    for source, target in zip(graph.src, graph.dst):
        outgoing[source] += 1
        incoming[target] += 1
    return tuple(incoming), tuple(outgoing)


def _block_matrix(artifact: CxArtifact) -> Counter[tuple[tuple[str, str], tuple[str, str]]]:
    blocks = _node_blocks(artifact)
    return Counter((blocks[source], blocks[target]) for source, target in zip(artifact.graph.src, artifact.graph.dst))


def _diagonal_edges(artifact: CxArtifact) -> Counter[tuple[int, int, float, int]]:
    return Counter(
        (source, target, float(weight), sign)
        for source, target, weight, sign in zip(
            artifact.graph.src, artifact.graph.dst, artifact.graph.weight, artifact.edge_signs
        )
        if source == target
    )


def cx_null_report(real: CxArtifact, null: CxArtifact) -> dict[str, bool]:
    real_in, real_out = _degrees(real.graph)
    null_in, null_out = _degrees(null.graph)
    return {
        "same_num_nodes": real.graph.num_nodes == null.graph.num_nodes,
        "same_num_edges": real.graph.num_edges == null.graph.num_edges,
        "same_in_degree": real_in == null_in,
        "same_out_degree": real_out == null_out,
        "same_weight_multiset": Counter(real.graph.weight) == Counter(null.graph.weight),
        "same_sign_multiset": Counter(real.edge_signs) == Counter(null.edge_signs),
        "same_nodes": real.nodes == null.nodes,
        "same_roles": (
            real.input_indices == null.input_indices
            and real.core_indices == null.core_indices
            and real.output_indices == null.output_indices
        ),
        "same_source_hashes": real.source_hashes == null.source_hashes,
        "same_diagonal_edges": _diagonal_edges(real) == _diagonal_edges(null),
        "same_block_matrix": _block_matrix(real) == _block_matrix(null),
        "graph_changed": graph_fingerprint(real.graph) != graph_fingerprint(null.graph),
    }
