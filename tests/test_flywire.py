from collections import Counter

from yolo_flywire.flywire import load_flywire_csv, selection_to_graph
from yolo_flywire.graphs import graph_fingerprint, rewire_degree_preserving


def _degrees(graph):
    return Counter(graph.src), Counter(graph.dst)


def test_flywire_selection_requires_release_and_rule(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text("pre_id,post_id,synapse_count\n1,2,4\n", encoding="utf-8")
    try:
        load_flywire_csv(path, release_id="", selection_rule="")
    except ValueError as exc:
        assert "provenance" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError")


def test_flywire_graph_and_rewired_control_are_matched(tmp_path):
    path = tmp_path / "edges.csv"
    path.write_text(
        "pre_id,post_id,synapse_count,region,cell_type\n"
        "10,20,4,optic,A\n"
        "10,30,2,optic,A\n"
        "20,30,3,optic,B\n"
        "20,40,5,optic,B\n"
        "30,40,1,optic,C\n"
        "30,50,2,optic,C\n"
        "40,50,4,optic,D\n"
        "40,10,2,optic,D\n"
        "50,10,3,optic,E\n"
        "50,20,1,optic,E\n",
        encoding="utf-8",
    )
    selection = load_flywire_csv(
        path,
        release_id="fixture-release-v1",
        selection_rule="all fixture optic edges",
    )
    graph = selection_to_graph(selection)
    rewired = rewire_degree_preserving(graph, seed=7, swaps=5)

    assert graph.num_nodes == rewired.num_nodes
    assert len(graph.src) == len(rewired.src)
    assert _degrees(graph) == _degrees(rewired)
    assert sorted(graph.weight) == sorted(rewired.weight)
    assert graph_fingerprint(graph) != graph_fingerprint(rewired)
