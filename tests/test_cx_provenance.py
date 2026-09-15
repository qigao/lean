from __future__ import annotations

import json
from pathlib import Path

import pytest

from yolo_flywire.cx_protocol import load_cx_protocol, validate_cx_protocol_dict
from yolo_flywire.cx_provenance import build_cx_provenance, verify_cx_provenance


ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT / "protocols" / "v2-cx-temporal-gate1-preflight.json"
SEEDS = (7, 11, 19, 23, 31)


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


def _preflight():
    return load_cx_protocol(PREFLIGHT)


def _raw_protocol() -> dict:
    return json.loads(PREFLIGHT.read_text(encoding="utf-8"))


def test_build_and_verify_reconstructs_real_and_all_frozen_null_seeds(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_preflight(), cells, connections, bundle)
    assert report.seeds == SEEDS
    assert report.successful_swaps == 120
    assert len(report.real_artifact_fingerprint) == 64
    assert len(report.input_node_fingerprint) == 64
    assert len(report.output_node_fingerprint) == 64
    assert tuple(seed for seed, _ in report.degree_null_fingerprints) == SEEDS
    assert tuple(seed for seed, _ in report.block_null_fingerprints) == SEEDS
    assert verify_cx_provenance(_preflight(), cells, connections, bundle) == report
    with pytest.raises(FileExistsError):
        build_cx_provenance(_preflight(), cells, connections, bundle)


def test_wrong_frozen_source_hash_fails_before_bundle_creation(tmp_path):
    cells, connections = _sources(tmp_path)
    raw = _raw_protocol()
    raw["source_hashes"]["consolidated_cell_types"] = "a" * 64
    protocol = validate_cx_protocol_dict(raw)
    bundle = tmp_path / "bundle"
    with pytest.raises(ValueError, match="source hash mismatch"):
        build_cx_provenance(protocol, cells, connections, bundle)
    assert not bundle.exists()


def test_wrong_frozen_real_artifact_fingerprint_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_preflight(), cells, connections, bundle)
    raw = _raw_protocol()
    raw["cx_artifact_fingerprint"] = "a" * 64
    protocol = validate_cx_protocol_dict(raw)
    assert report.real_artifact_fingerprint != "a" * 64
    with pytest.raises(ValueError, match="cx_artifact_fingerprint"):
        verify_cx_provenance(protocol, cells, connections, bundle)


def test_wrong_frozen_null_seed_map_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    report = build_cx_provenance(_preflight(), cells, connections, bundle)
    raw = _raw_protocol()
    raw["degree_null_fingerprints"] = {str(seed): digest for seed, digest in report.degree_null_fingerprints}
    raw["block_null_fingerprints"] = {str(seed): digest for seed, digest in report.block_null_fingerprints}
    raw["degree_null_fingerprints"]["19"] = "a" * 64
    protocol = validate_cx_protocol_dict(raw)
    with pytest.raises(ValueError, match="degree_null_fingerprints"):
        verify_cx_provenance(protocol, cells, connections, bundle)


def test_changed_swap_budget_or_sign_policy_in_bundle_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    build_cx_provenance(_preflight(), cells, connections, bundle)
    manifest_path = bundle / "provenance.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    changed = dict(original)
    changed["successful_swaps"] = original["successful_swaps"] - 1
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="swap budget|bundle hash"):
        verify_cx_provenance(_preflight(), cells, connections, bundle)

    manifest_path.write_text(json.dumps(original, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    changed = dict(original)
    changed["nt_signs"] = dict(original["nt_signs"])
    changed["nt_signs"]["GABA"] = 1
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sign policy|bundle hash"):
        verify_cx_provenance(_preflight(), cells, connections, bundle)


def test_missing_role_fingerprint_or_corrupted_serialized_artifact_is_rejected(tmp_path):
    cells, connections = _sources(tmp_path)
    bundle = tmp_path / "bundle"
    build_cx_provenance(_preflight(), cells, connections, bundle)
    manifest_path = bundle / "provenance.json"
    original_text = manifest_path.read_text(encoding="utf-8")
    original = json.loads(original_text)

    changed = dict(original)
    changed.pop("input_node_fingerprint")
    manifest_path.write_text(json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="input_node_fingerprint|bundle"):
        verify_cx_provenance(_preflight(), cells, connections, bundle)

    manifest_path.write_text(original_text, encoding="utf-8")
    edges = bundle / "real" / "edges.csv"
    edges.write_text(edges.read_text(encoding="utf-8") + "#corrupt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact file hash"):
        verify_cx_provenance(_preflight(), cells, connections, bundle)
