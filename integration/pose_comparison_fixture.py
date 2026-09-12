"""Generated graph assets for engineering tests, NOT real FlyWire data."""
import hashlib
import json

from yolo_flywire.flywire import load_visual_type_graph
from yolo_flywire.graphs import graph_fingerprint, rewire_degree_preserving
from yolo_flywire.provenance import _context, generate_controls

SEED_TYPES = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")


def graph_inputs(directory, seeds=(7, 11)):
    directory.mkdir()
    source = directory / "connectivity.csv"
    source.write_text("from type,to type,synapses total\n" + "".join(
        f"{name},{SEED_TYPES[(i + 1) % 8]},5\n" for i, name in enumerate(SEED_TYPES)
    ) + "T4a,T4a,2\n", encoding="utf-8")
    graph = load_visual_type_graph(source, seed_types=SEED_TYPES, min_seed_synapses=5).graph
    raw = source.read_bytes()
    config = {
        "evidence_scope": "generated_graph_fixture_not_fafb",
        "seeds": list(seeds), "budget": {"epochs": 2, "max_updates": 6, "parameter_ceiling": 10000},
        "rewiring_algorithm": "directed-double-edge-swap-v2-diagonal-fixed",
        "rewiring_successful_swaps_per_offdiagonal_edge": 10,
        "flywire_connectivity_sha256": sha(source),
        "flywire_connectivity_git_blob_sha1": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest(),
        "selected_graph_fingerprint": graph_fingerprint(graph),
        "selected_graph_num_nodes": 8, "selected_graph_num_edges": 9,
        "selected_graph_num_diagonal_edges": 1,
        "rewired_graph_fingerprints": {str(seed): graph_fingerprint(
            rewire_degree_preserving(graph, seed, 80)) for seed in seeds},
    }
    _, identity = _context(config, source)
    protocol, controls = directory / "protocol.json", directory / "controls.json"
    write_json(protocol, config)
    write_json(controls, generate_controls(graph, identity))
    return dict(topology_protocol=protocol, connectivity=source, controls=controls,
                expected_topology_sha256=sha(protocol), expected_controls_sha256=sha(controls))
