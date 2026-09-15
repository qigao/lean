from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_flywire.cx_artifact import build_cx_artifact, write_cx_artifact
from yolo_flywire.cx_protocol import load_cx_protocol
from yolo_flywire.graphs import graph_fingerprint


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "cx"
CELL_TYPES = FIXTURES / "consolidated_cell_types.csv"
CONNECTIONS = FIXTURES / "connections_princeton.csv"
PROTOCOL = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"


@pytest.fixture
def protocol():
    return load_cx_protocol(PROTOCOL)


def test_build_cx_artifact_aggregates_pair_before_threshold(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    assert artifact.graph.num_nodes == 8
    assert len(artifact.graph.src) == 8
    assert 7.0 in artifact.graph.weight
    assert 4.0 not in artifact.graph.weight
    assert set(artifact.graph.weight) == {5.0, 6.0, 7.0, 8.0, 9.0, 10.0}


def test_roles_are_family_driven_and_disjoint(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    assert artifact.input_indices == (0, 1, 2)
    assert artifact.core_indices == (3, 4, 5, 6)
    assert artifact.output_indices == (7,)
    assert not (set(artifact.input_indices) & set(artifact.core_indices))
    assert not (set(artifact.input_indices) & set(artifact.output_indices))
    assert not (set(artifact.core_indices) & set(artifact.output_indices))
    assert [node.family for node in artifact.nodes] == [
        "ER",
        "ExR",
        "PFN",
        "EPG",
        "PEN",
        "Delta7",
        "hDelta",
        "PFL",
    ]


def test_non_cx_types_and_their_edges_are_excluded(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    assert {node.root_id for node in artifact.nodes} == set(range(1, 9))
    root_ids = [node.root_id for node in artifact.nodes]
    pairs = {(root_ids[src], root_ids[dst]) for src, dst in zip(artifact.graph.src, artifact.graph.dst)}
    assert all(9 not in pair for pair in pairs)


def test_edge_signs_align_with_retained_presynaptic_transmitter(protocol):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    root_ids = [node.root_id for node in artifact.nodes]
    signs = {
        (root_ids[src], root_ids[dst]): sign
        for src, dst, sign in zip(artifact.graph.src, artifact.graph.dst, artifact.edge_signs)
    }
    assert signs[(1, 4)] == 1
    assert signs[(5, 6)] == -1
    assert signs[(6, 7)] == -1
    assert signs[(4, 8)] == 1


def test_unknown_retained_transmitter_fails(protocol, tmp_path):
    bad = tmp_path / "connections.csv"
    bad.write_text(
        "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
        "1,4,PB,6,UNKNOWN\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="transmitter"):
        build_cx_artifact(CELL_TYPES, bad, protocol)


def test_inconsistent_transmitter_across_pair_rows_fails(protocol, tmp_path):
    bad = tmp_path / "connections.csv"
    bad.write_text(
        "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
        "1,4,PB,3,ACH\n"
        "1,4,EB,4,GABA\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="inconsistent.*transmitter"):
        build_cx_artifact(CELL_TYPES, bad, protocol)


def test_duplicate_root_id_fails(protocol, tmp_path):
    bad = tmp_path / "cell_types.csv"
    bad.write_text(
        "root_id,primary_type,additional_type(s)\n"
        "1,ER4d,\n"
        "1,EPG,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate root_id"):
        build_cx_artifact(bad, CONNECTIONS, protocol)


def test_graph_is_canonical_even_if_connection_rows_are_reordered(protocol, tmp_path):
    lines = CONNECTIONS.read_text(encoding="utf-8").splitlines()
    reordered = tmp_path / "connections.csv"
    reordered.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
    a = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    b = build_cx_artifact(CELL_TYPES, reordered, protocol)
    assert graph_fingerprint(a.graph) == graph_fingerprint(b.graph)
    assert a.edge_signs == b.edge_signs
    assert a.fingerprint != b.fingerprint, "source-byte provenance must remain part of the artifact identity"


def test_artifact_serialization_is_explicit_and_hashes_outputs(protocol, tmp_path):
    artifact = build_cx_artifact(CELL_TYPES, CONNECTIONS, protocol)
    output = tmp_path / "artifact"
    hashes = write_cx_artifact(artifact, output)
    assert set(hashes) == {"nodes.csv", "edges.csv", "roles.json", "metadata.json"}
    assert all(len(value) == 64 for value in hashes.values())
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["artifact_fingerprint"] == artifact.fingerprint
    assert metadata["graph_fingerprint"] == graph_fingerprint(artifact.graph)
    assert metadata["input_root_ids"] == [1, 2, 3]
    assert metadata["output_root_ids"] == [8]
    with pytest.raises(FileExistsError):
        write_cx_artifact(artifact, output)
