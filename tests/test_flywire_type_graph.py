from __future__ import annotations

from pathlib import Path

from yolo_flywire.flywire import load_visual_type_graph
from yolo_flywire.graphs import graph_fingerprint


def test_visual_type_graph_applies_frozen_t4_t5_selection_rule(tmp_path: Path):
    source = tmp_path / "types.csv"
    source.write_text(
        "from type,to type,connections total,connections RHS,connections LHS,connections bilateral,synapses total,synapses RHS,synapses LHS,synapses bilateral\n"
        "T4a,T4a,1,1,0,0,11,11,0,0\n"
        "T4a,Zeta,1,1,0,0,6,6,0,0\n"
        "T5a,Alpha,1,1,0,0,5,5,0,0\n"
        "T4a,BelowThreshold,1,1,0,0,4,4,0,0\n"
        "Alpha,T4a,1,1,0,0,3,3,0,0\n"
        "Zeta,Alpha,1,1,0,0,2,2,0,0\n"
        "BelowThreshold,Alpha,1,1,0,0,100,100,0,0\n"
        "Outside,Zeta,1,1,0,0,100,100,0,0\n",
        encoding="utf-8",
    )

    selection = load_visual_type_graph(
        source,
        seed_types=("T4a", "T5a"),
        min_seed_synapses=5,
    )

    assert selection.node_types == ("Alpha", "T4a", "T5a", "Zeta")
    assert selection.graph.num_nodes == 4
    assert set(zip(selection.graph.src, selection.graph.dst, selection.graph.weight)) == {
        (1, 1, 11.0),  # T4a -> T4a means within-type population connectivity
        (1, 3, 6.0),  # T4a -> Zeta
        (2, 0, 5.0),  # T5a -> Alpha
        (0, 1, 3.0),  # Alpha -> T4a, retained by induced-subgraph rule
        (3, 0, 2.0),  # Zeta -> Alpha, retained by induced-subgraph rule
    }
    assert len(graph_fingerprint(selection.graph)) == 64


def test_visual_type_graph_requires_all_declared_seed_types(tmp_path: Path):
    source = tmp_path / "types.csv"
    source.write_text(
        "from type,to type,synapses total\n"
        "T4a,Zeta,6\n",
        encoding="utf-8",
    )

    try:
        load_visual_type_graph(source, seed_types=("T4a", "T5a"), min_seed_synapses=5)
    except ValueError as exc:
        assert "seed" in str(exc).lower()
        assert "T5a" in str(exc)
    else:
        raise AssertionError("missing declared seed type must be rejected")
