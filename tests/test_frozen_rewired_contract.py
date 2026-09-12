"""Freeze measured controls without treating graph provenance as recognition evidence."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from yolo_flywire.cli import _validate_frozen_protocol, compare
from yolo_flywire import provenance
from yolo_flywire.flywire import load_visual_type_graph
from yolo_flywire.graphs import graph_fingerprint

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [7, 11, 19, 23, 31]
ALGORITHM = "directed-double-edge-swap-v2-diagonal-fixed"
# Independently read from #71 and #73 artifacts; all five values matched.
MEASURED = {
    "7": "85a0a73e58a3ae9b1394cb59f7970538cd301fee03e04d22c2cf4af1dc38deda",
    "11": "e88cdf1652e39d6a2ef75dcbff3d2b79c7ed9b703b5beae3cee92e13e7c10387",
    "19": "aaae8ec540b26c1228916a3a24e08b8fe61a909c802f031b73ce659447e82231",
    "23": "c6aad8f35dcf875730a08f6607e2e3c67bb4943ad2710f8677d329b6b5daaed4",
    "31": "e80b997850d2ff16e024a4009cd7e66c0a86b2648caa81ce282ed59061e26469",
}


def _load(name="v0-real-ntu120-preflight.json"):
    return json.loads((ROOT / "protocols" / name).read_text(encoding="utf-8"))


def _ready_fixture():
    config = _load()
    # Test-only stand-ins: these are never written to a repository protocol.
    for field in ("dataset_content_hash", "yolo_weights_sha256", "split_hash",
                  "observation_schema_hash"):
        config[field] = hashlib.sha256(("fixture:" + field).encode()).hexdigest()
    config["ultralytics_package_version"] = "fixture-version"
    config["rewiring_algorithm"] = ALGORITHM
    config["rewiring_successful_swaps_per_offdiagonal_edge"] = 10
    config["rewired_graph_fingerprints"] = dict(MEASURED)
    return config


@pytest.mark.parametrize("name", ["v0-real-ntu120-preflight.json", "v0-real-template.json"])
def test_real_protocols_freeze_all_measured_controls(name):
    config = _load(name)
    assert config.get("rewired_graph_fingerprints") == MEASURED
    assert config["seeds"] == SEEDS
    assert config["rewiring_algorithm"] == ALGORITHM
    assert config["rewiring_successful_swaps_per_offdiagonal_edge"] == 10
    assert config["final_test_used_for_selection"] is False
    for field in ("dataset_content_hash", "ultralytics_package_version", "yolo_weights_sha256",
                  "split_hash", "observation_schema_hash"):
        assert config[field] is None


def test_real_template_and_preflight_share_one_experiment_contract():
    preflight = _load()
    template = _load("v0-real-template.json")
    metadata = {"protocol_id", "frozen_preflight", "evidence_scope"}
    assert {k: v for k, v in template.items() if k not in metadata} == {
        k: v for k, v in preflight.items() if k not in metadata
    }


def test_complete_fingerprint_contract_is_structurally_admissible_only():
    assert _validate_frozen_protocol(_ready_fixture()) == ("gru", "random_graph", "rewired", "flywire")


@pytest.mark.parametrize("case", [
    "missing", "null", "empty", "missing-seed", "extra-seed", "short-hash",
    "uppercase-hash", "non-string-hash", "original-graph", "integer-key",
    "duplicate-seeds", "boolean-seed", "string-seed", "v1-algorithm",
    "count-only-algorithm", "missing-budget", "reduced-budget", "float-budget",
])
def test_real_comparison_rejects_incomplete_or_changed_control_contract(case):
    config = _ready_fixture()
    mapping = config["rewired_graph_fingerprints"]
    if case == "missing":
        del config["rewired_graph_fingerprints"]
    elif case == "null":
        config["rewired_graph_fingerprints"] = None
    elif case == "empty":
        config["rewired_graph_fingerprints"] = {}
    elif case == "missing-seed":
        del mapping["7"]
    elif case == "extra-seed":
        mapping["99"] = "a" * 64
    elif case == "short-hash":
        mapping["7"] = "not-a-sha256"
    elif case == "uppercase-hash":
        mapping["7"] = mapping["7"].upper()
    elif case == "non-string-hash":
        mapping["7"] = 7
    elif case == "original-graph":
        mapping["7"] = config["selected_graph_fingerprint"]
    elif case == "integer-key":
        mapping[7] = mapping.pop("7")
    elif case == "duplicate-seeds":
        config["seeds"] = [7, 7, 19, 23, 31]
    elif case == "boolean-seed":
        config["seeds"] = [True, 11, 19, 23, 31]
    elif case == "string-seed":
        config["seeds"] = ["7", 11, 19, 23, 31]
    elif case == "v1-algorithm":
        config["rewiring_algorithm"] = "directed-double-edge-swap-v1"
    elif case == "count-only-algorithm":
        config["rewiring_algorithm"] = "directed-double-edge-swap-v2-self-loop-matched"
    elif case == "missing-budget":
        del config["rewiring_successful_swaps_per_offdiagonal_edge"]
    elif case == "reduced-budget":
        config["rewiring_successful_swaps_per_offdiagonal_edge"] = 9
    elif case == "float-budget":
        config["rewiring_successful_swaps_per_offdiagonal_edge"] = 10.0
    with pytest.raises(ValueError, match="rewir|seed|fingerprint"):
        _validate_frozen_protocol(config)


def test_rejected_comparison_does_not_create_evidence(tmp_path):
    config = _ready_fixture()
    config["rewired_graph_fingerprints"] = None
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "evidence"
    with pytest.raises(ValueError, match="rewired_graph_fingerprints"):
        compare(config_path, output)
    assert not output.exists()


def _source_fixture(tmp_path):
    types = ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")
    text = "from type,to type,synapses total\n" + "".join(
        f"{name},{types[(i + 1) % len(types)]},6\n{name},{name},3\n"
        for i, name in enumerate(types)
    )
    source = tmp_path / "source.csv"
    raw = text.encode("utf-8")
    source.write_bytes(raw)
    config = _ready_fixture()
    config["flywire_connectivity_sha256"] = hashlib.sha256(raw).hexdigest()
    config["flywire_connectivity_git_blob_sha1"] = hashlib.sha1(
        f"blob {len(raw)}\0".encode() + raw
    ).hexdigest()
    graph = load_visual_type_graph(source, seed_types=types, min_seed_synapses=5).graph
    config.update(selected_graph_fingerprint=graph_fingerprint(graph),
                  selected_graph_num_nodes=graph.num_nodes,
                  selected_graph_num_edges=graph.num_edges,
                  selected_graph_num_diagonal_edges=len(types))
    return source, config


def test_provenance_context_rejects_unfrozen_fingerprints(tmp_path):
    source, config = _source_fixture(tmp_path)
    config["rewired_graph_fingerprints"] = None
    with pytest.raises(ValueError, match="rewired_graph_fingerprints"):
        provenance._context(config, source)


def test_provenance_rejects_well_formed_but_wrong_fingerprints_without_output(tmp_path, capsys):
    source, config = _source_fixture(tmp_path)
    # Valid digests, but they are measured on a different graph, not this fixture.
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "controls"
    with pytest.raises(SystemExit) as error:
        provenance.main(["generate", "--config", str(path), "--source", str(source),
                         "--output", str(output)])
    assert error.value.code == 1
    assert "rewired fingerprints do not match" in capsys.readouterr().err
    assert not output.exists()


def test_synthetic_contract_is_not_silently_upgraded_to_real_evidence():
    config = _load("v0-synthetic.json")
    assert _validate_frozen_protocol(config) == ("gru", "random_graph", "rewired", "flywire")
