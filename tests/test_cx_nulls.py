from __future__ import annotations

from collections import Counter

import pytest

from yolo_flywire.cx_artifact import CxArtifact, CxNode
from yolo_flywire.cx_nulls import (
    cx_null_report,
    rewire_cx_block_preserving,
    rewire_cx_degree_preserving,
)
from yolo_flywire.graphs import DirectedGraph, graph_fingerprint


def _artifact() -> CxArtifact:
    nodes = (
        CxNode(100, "ER1", "ER", "input"),
        CxNode(101, "ER2", "ER", "input"),
        CxNode(200, "EPG_a", "EPG", "core"),
        CxNode(201, "EPG_b", "EPG", "core"),
        CxNode(300, "PEN_a", "PEN", "core"),
        CxNode(301, "PEN_b", "PEN", "core"),
        CxNode(400, "PFL2", "PFL", "output"),
        CxNode(401, "PFL3", "PFL", "output"),
    )
    edges = (
        (0, 2, 5.0, +1),
        (1, 3, 6.0, +1),
        (0, 4, 7.0, +1),
        (1, 5, 8.0, +1),
        (2, 4, 9.0, +1),
        (3, 5, 10.0, +1),
        (2, 6, 11.0, +1),
        (3, 7, 12.0, +1),
        (4, 6, 13.0, -1),
        (5, 7, 14.0, -1),
        (6, 2, 15.0, -1),
        (7, 3, 16.0, -1),
    )
    graph = DirectedGraph(
        num_nodes=len(nodes),
        src=tuple(src for src, _, _, _ in edges),
        dst=tuple(dst for _, dst, _, _ in edges),
        weight=tuple(weight for _, _, weight, _ in edges),
    )
    return CxArtifact(
        nodes=nodes,
        graph=graph,
        input_indices=(0, 1),
        core_indices=(2, 3, 4, 5),
        output_indices=(6, 7),
        edge_signs=tuple(sign for _, _, _, sign in edges),
        source_hashes=(("consolidated_cell_types", "a" * 64), ("connections_princeton", "b" * 64)),
        matched_primary_types=tuple(node.primary_type for node in nodes),
        fingerprint="c" * 64,
    )


def _degree_sequence(graph: DirectedGraph):
    incoming = [0] * graph.num_nodes
    outgoing = [0] * graph.num_nodes
    for src, dst in zip(graph.src, graph.dst):
        outgoing[src] += 1
        incoming[dst] += 1
    return tuple(incoming), tuple(outgoing)


def _block_matrix(artifact: CxArtifact):
    blocks = tuple((node.role, node.family) for node in artifact.nodes)
    return Counter((blocks[src], blocks[dst]) for src, dst in zip(artifact.graph.src, artifact.graph.dst))


def test_degree_null_preserves_decisive_invariants():
    real = _artifact()
    null = rewire_cx_degree_preserving(real, seed=7, swaps=8)
    assert graph_fingerprint(null.graph) != graph_fingerprint(real.graph)
    assert _degree_sequence(null.graph) == _degree_sequence(real.graph)
    assert Counter(null.graph.weight) == Counter(real.graph.weight)
    assert Counter(null.edge_signs) == Counter(real.edge_signs)
    assert null.nodes == real.nodes
    assert null.input_indices == real.input_indices
    assert null.core_indices == real.core_indices
    assert null.output_indices == real.output_indices
    assert null.source_hashes == real.source_hashes
    assert null.fingerprint != real.fingerprint


def test_degree_null_is_deterministic_for_seed_and_budget():
    real = _artifact()
    first = rewire_cx_degree_preserving(real, seed=11, swaps=6)
    second = rewire_cx_degree_preserving(real, seed=11, swaps=6)
    assert first.graph == second.graph
    assert first.edge_signs == second.edge_signs
    assert first.fingerprint == second.fingerprint


def test_block_null_preserves_family_role_block_matrix():
    real = _artifact()
    null = rewire_cx_block_preserving(real, seed=19, swaps=6)
    assert graph_fingerprint(null.graph) != graph_fingerprint(real.graph)
    assert _degree_sequence(null.graph) == _degree_sequence(real.graph)
    assert _block_matrix(null) == _block_matrix(real)
    assert Counter(null.graph.weight) == Counter(real.graph.weight)
    assert Counter(null.edge_signs) == Counter(real.edge_signs)
    assert null.nodes == real.nodes


def test_null_report_distinguishes_degree_and_block_conformance():
    real = _artifact()
    degree = rewire_cx_degree_preserving(real, seed=23, swaps=6)
    block = rewire_cx_block_preserving(real, seed=23, swaps=6)
    degree_report = cx_null_report(real, degree)
    block_report = cx_null_report(real, block)
    for report in (degree_report, block_report):
        assert report["same_num_nodes"] is True
        assert report["same_num_edges"] is True
        assert report["same_in_degree"] is True
        assert report["same_out_degree"] is True
        assert report["same_weight_multiset"] is True
        assert report["same_sign_multiset"] is True
        assert report["same_nodes"] is True
        assert report["same_roles"] is True
        assert report["same_source_hashes"] is True
        assert report["same_diagonal_edges"] is True
        assert report["graph_changed"] is True
    assert degree_report["same_block_matrix"] is False
    assert block_report["same_block_matrix"] is True


def test_rewiring_does_not_create_self_edges():
    real = _artifact()
    for builder in (rewire_cx_degree_preserving, rewire_cx_block_preserving):
        null = builder(real, seed=31, swaps=6)
        assert all(src != dst for src, dst in zip(null.graph.src, null.graph.dst))


def test_impossible_rewire_fails_instead_of_returning_partial_null():
    nodes = (
        CxNode(1, "ER1", "ER", "input"),
        CxNode(2, "EPG", "EPG", "core"),
        CxNode(3, "PFL3", "PFL", "output"),
    )
    graph = DirectedGraph(num_nodes=3, src=(0, 1), dst=(1, 2), weight=(5.0, 6.0))
    artifact = CxArtifact(
        nodes=nodes,
        graph=graph,
        input_indices=(0,),
        core_indices=(1,),
        output_indices=(2,),
        edge_signs=(1, 1),
        source_hashes=(("consolidated_cell_types", "a" * 64), ("connections_princeton", "b" * 64)),
        matched_primary_types=("EPG", "ER1", "PFL3"),
        fingerprint="c" * 64,
    )
    with pytest.raises(ValueError, match="could not complete"):
        rewire_cx_degree_preserving(artifact, seed=7, swaps=1, max_attempt_factor=5)
