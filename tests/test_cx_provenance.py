from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_flywire.cx_protocol import load_cx_protocol, validate_cx_protocol_dict
from yolo_flywire.cx_provenance import build_cx_provenance, verify_cx_provenance


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"
SEEDS = (7, 11, 19, 23, 31)

_REAL_SOURCE_HASHES = {
    "consolidated_cell_types": "8aba246d71dc40361677493629972ce3883048c3d02010adc42bda22962a1a2d",
    "connections_princeton": "445f996bf6c4b1803b9ba186189138a3061ff8623aa94c0abcf38af30a5bd48b",
}
_REAL_ARTIFACT = "ee1387398e0d6d7404d78bdbce99783209e18f4ae408001d9a9dd2596ed75cd7"
_REAL_INPUT = "f917eb9fa128c2090671ecba8fd1c75db5dd57d2e603b1b0a649bea64cfd17b8"
_REAL_OUTPUT = "f966a185ccb9d34369c55ba20e1f509c1bbc16265e1b4880e45618ad451aa597"
_REAL_DEGREE = {
    "7": "6c47a71b0ad9787608b84e89f2cce5615b2b29a5cf2a9cd0edc35ff6486fb6f0",
    "11": "6ad62b7cc2e95429c9bfc7a0fde2e44f0c4d6a4e6f027f3c1fccef6a3cb170f0",
    "19": "038553265a0d4835e8f2107eb6a499bf232b6612fed17c75c79459ce7e567944",
    "23": "465e5eb04527f30577119fab1eb00b9ea965f006774e91b8470a0706f472e82b",
    "31": "dbf6cb2b43f6dcced1f43575cfe566c43501a541815eee7db392dab854583bc8",
}
_REAL_BLOCK = {
    "7": "06afe32d006fca0be3d940bd2d38ef7cac99930f6963f52e555e6965c3738edf",
    "11": "fbbe1b51ff7a14db4b0b356cbef1196903f3684891c74e80546172b0b1ed3e53",
    "19": "85fa3c223228c1360e4116cb03bec41d71a632e8cc79a10180f5894ea963f6a9",
    "23": "a36f4549f059c7cea3155ae344a2d5651d0a4cfd2909a640d5910f17134f1bd6",
    "31": "d07aa024a6556d76c9223eb6a12376509f365075fa8fd2f3ce7fe53a6bc40486",
}


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    cells = tmp_path / "consolidated_cell_types.csv"
    cells.write_text(
        "root_id,primary_type,additional_type(s)\n"
        "100,ER1,\n"
        "101,ER2,\n"
        "200,EPG_a,\n"
        "201,EPG_b,\n"
        "300,PEN_a,\n"
        "301,PEN_b,\n"
        "400,PFL2,\n"
        "401,PFL3,\n",
        encoding="utf-8",
    )
    connections = tmp_path / "connections_princeton.csv"
    rows = (
        (100, 200, 5, "ACH"),
        (101, 201, 6, "ACH"),
        (100, 300, 7, "ACH"),
        (101, 301, 8, "ACH"),
        (200, 300, 9, "ACH"),
        (201, 301, 10, "ACH"),
        (200, 400, 11, "ACH"),
        (201, 401, 12, "ACH"),
        (300, 400, 13, "GABA"),
        (301, 401, 14, "GABA"),
        (400, 200, 15, "GABA"),
        (401, 201, 16, "GABA"),
    )
    connections.write_text(
        "pre_root_id,post_root_id,neuropil,syn_count,nt_type\n"
        + "".join(f"{pre},{post},FB,{count},{nt}\n" for pre, post, count, nt in rows),
        encoding="utf-8",
    )
    return cells, connections


def _raw_protocol() -> dict:
    return json.loads(PREFLIGHT.read_text(encoding="utf-8"))


def _fixture_protocol():
    raw = _raw_protocol()
    raw["source_hashes"] = {
        "consolidated_cell_types": None,
        "connections_princeton": None,
    }
    raw["input_node_fingerprint"] = None
    raw["output_node_fingerprint"] = None
    raw["cx_artifact_fingerprint"] = None
    raw["degree_null_fingerprints"] = None
    raw["block_null_fingerprints"] = None
    return validate_cx_protocol_dict(raw)


def test_committed_preflight_contains_exact_measured_fafb_cx_pins():
    protocol = load_cx_protocol(PREFLIGHT)
    assert dict(protocol.source_hashes) == _REAL_SOURCE_HASHES
    assert protocol.cx_artifact_fingerprint == _REAL_ARTIFACT
    assert protocol.input_node_fingerprint == _REAL_INPUT
    assert protocol.output_node_fingerprint == _REAL_OUTPUT
    assert dict(protocol.degree_null_fingerprints or ()) == _REAL_DEGREE
    assert dict(protocol.block_null_fingerprints or ()) == _REAL_BLOCK
    assert protocol.final_test_enabled is False


def test_build_and_verify_reconstructs_real_and_all_frozen_null_seeds(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_fixture_protocol(), cells, connections, bundle)
    assert report.seeds == SEEDS
    assert report.successful_swaps == 120
    assert len(report.real_artifact_fingerprint) == 64
    assert len(report.input_node_fingerprint) == 64
    assert len(report.output_node_fingerprint) == 64
    assert tuple(seed for seed, _ in report.degree_null_fingerprints) == SEEDS
    assert tuple(seed for seed, _ in report.block_null_fingerprints) == SEEDS
    assert verify_cx_provenance(_fixture_protocol(), cells, connections, bundle) == report
    with pytest.raises(FileExistsError):
        build_cx_provenance(_fixture_protocol(), cells, connections, bundle)


def test_wrong_frozen_source_hash_fails_before_bundle_creation(tmp_path):
    cells, connections = _sources(tmp_path)
    raw = _raw_protocol()
    raw["source_hashes"] = {
        "consolidated_cell_types": "a" * 64,
        "connections_princeton": None,
    }
    raw["input_node_fingerprint"] = None
    raw["output_node_fingerprint"] = None
    raw["cx_artifact_fingerprint"] = None
    raw["degree_null_fingerprints"] = None
    raw["block_null_fingerprints"] = None
    protocol = validate_cx_protocol_dict(raw)
    bundle = tmp_path / "bundle"
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_cx_provenance(protocol, cells, connections, bundle)
    assert not bundle.exists()


def test_wrong_frozen_real_artifact_fingerprint_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_fixture_protocol(), cells, connections, bundle)
    raw = _raw_protocol()
    raw["source_hashes"] = {"consolidated_cell_types": None, "connections_princeton": None}
    raw["input_node_fingerprint"] = None
    raw["output_node_fingerprint"] = None
    raw["cx_artifact_fingerprint"] = "a" * 64
    raw["degree_null_fingerprints"] = None
    raw["block_null_fingerprints"] = None
    protocol = validate_cx_protocol_dict(raw)
    assert report.real_artifact_fingerprint != "a" * 64
    with pytest.raises(ValueError, match="cx_artifact_fingerprint"):
        verify_cx_provenance(protocol, cells, connections, bundle)


def test_wrong_frozen_null_seed_map_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_fixture_protocol(), cells, connections, bundle)
    raw = _raw_protocol()
    raw["source_hashes"] = {"consolidated_cell_types": None, "connections_princeton": None}
    raw["input_node_fingerprint"] = None
    raw["output_node_fingerprint"] = None
    raw["cx_artifact_fingerprint"] = None
    raw["degree_null_fingerprints"] = {str(seed): digest for seed, digest in report.degree_null_fingerprints}
    raw["block_null_fingerprints"] = {str(seed): digest for seed, digest in report.block_null_fingerprints}
    raw["degree_null_fingerprints"]["19"] = "a" * 64
    protocol = validate_cx_protocol_dict(raw)
    with pytest.raises(ValueError, match="degree_null_fingerprints"):
        verify_cx_provenance(protocol, cells, connections, bundle)


def test_changed_swap_budget_or_sign_policy_in_bundle_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    build_cx_provenance(_fixture_protocol(), cells, connections, bundle)
    manifest_path = bundle / "provenance.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = dict(original)
    changed["successful_swaps"] = original["successful_swaps"] - 1
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="swap budget|bundle hash"):
        verify_cx_provenance(_fixture_protocol(), cells, connections, bundle)

    manifest_path.write_text(json.dumps(original, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    changed = dict(original)
    changed["nt_signs"] = dict(original["nt_signs"])
    changed["nt_signs"]["GABA"] = 1
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sign policy|bundle hash"):
        verify_cx_provenance(_fixture_protocol(), cells, connections, bundle)


def test_missing_role_fingerprint_or_corrupted_serialized_artifact_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    build_cx_provenance(_fixture_protocol(), cells, connections, bundle)
    manifest_path = bundle / "provenance.json"
    original_text = manifest_path.read_text(encoding="utf-8")
    original = json.loads(original_text)

    changed = dict(original)
    changed.pop("input_node_fingerprint")
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_node_fingerprint|bundle"):
        verify_cx_provenance(_fixture_protocol(), cells, connections, bundle)

    manifest_path.write_text(original_text, encoding="utf-8")
    edges = bundle / "real" / "edges.csv"
    edges.write_text(edges.read_text(encoding="utf-8") + "#corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact file hash"):
        verify_cx_provenance(_fixture_protocol(), cells, connections, bundle)
