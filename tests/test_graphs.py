from collections import Counter

from yolo_flywire.graphs import (
    DirectedGraph,
    graph_fingerprint,
    rewire_degree_preserving,
)


def degrees(graph):
    return Counter(graph.src), Counter(graph.dst)


def self_loop_count(graph):
    return sum(source == target for source, target in zip(graph.src, graph.dst))


def test_rewiring_preserves_directed_degree_sequence_and_weight_multiset():
    graph = DirectedGraph(
        num_nodes=5,
        src=(0, 0, 1, 2, 3, 4),
        dst=(1, 2, 2, 3, 4, 0),
        weight=(1.0, 2.0, 1.0, 1.0, 3.0, 1.0),
    )
    rewired = rewire_degree_preserving(graph, seed=9, swaps=2)
    assert degrees(rewired) == degrees(graph)
    assert sorted(rewired.weight) == sorted(graph.weight)
    assert set(zip(rewired.src, rewired.dst)) != set(zip(graph.src, graph.dst))


def test_type_level_graph_allows_population_self_edges():
    graph = DirectedGraph(
        num_nodes=3,
        src=(0, 0, 1, 2),
        dst=(0, 1, 2, 2),
        weight=(11.0, 2.0, 3.0, 7.0),
    )
    assert self_loop_count(graph) == 2


def test_rewiring_preserves_self_loop_count_for_type_level_controls():
    graph = DirectedGraph(
        num_nodes=6,
        src=(0, 0, 1, 1, 2, 2, 3, 4, 5, 5),
        dst=(0, 1, 1, 2, 2, 3, 4, 5, 0, 5),
        weight=(9.0, 1.0, 8.0, 1.0, 7.0, 1.0, 1.0, 1.0, 1.0, 6.0),
    )
    rewired = rewire_degree_preserving(graph, seed=13, swaps=2)
    assert degrees(rewired) == degrees(graph)
    assert sorted(rewired.weight) == sorted(graph.weight)
    assert self_loop_count(rewired) == self_loop_count(graph)
    assert set(zip(rewired.src, rewired.dst)) != set(zip(graph.src, graph.dst))


def test_graph_fingerprint_is_independent_of_edge_input_order():
    a = DirectedGraph(
        num_nodes=4,
        src=(0, 1, 2, 3),
        dst=(1, 2, 3, 0),
        weight=(1.0, 2.0, 3.0, 4.0),
    )
    b = DirectedGraph(
        num_nodes=4,
        src=(3, 1, 0, 2),
        dst=(0, 2, 1, 3),
        weight=(4.0, 2.0, 1.0, 3.0),
    )
    assert graph_fingerprint(a) == graph_fingerprint(b)
