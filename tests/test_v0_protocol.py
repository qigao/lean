from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / "protocols" / name).read_text(encoding="utf-8"))


def _run_compare(config: Path, output: Path):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "yolo_flywire.cli",
            "compare",
            "--config",
            str(config),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_frozen_synthetic_protocol_has_fixed_evaluation_boundary():
    protocol = _load("v0-synthetic.json")
    assert protocol["dataset_id"] == "synthetic-v0"
    assert protocol["split_hash"] == "6474f3228489f3bf46efaa46e84241b837be987c30993f4a4a50cc814044da84"
    assert protocol["observation_schema_hash"] == "1b27173701fe23b9dd6a66e05a96b6abf9598d26f0a22383a8fe11d10f74e706"
    assert len(protocol["seeds"]) >= 5
    assert protocol["primary_metric"] == "macro_f1"
    assert "balanced_accuracy" in protocol["secondary_metrics"]
    assert protocol["rewiring_algorithm"] == "directed-double-edge-swap-v1"
    assert protocol["final_test_used_for_selection"] is False
    assert protocol["success_threshold"] is not None


@pytest.mark.parametrize("protocol_name", ["v0-real-template.json", "v0-real-ntu120-preflight.json"])
def test_real_protocol_freezes_design_but_remains_non_executable(tmp_path: Path, protocol_name: str):
    config = ROOT / "protocols" / protocol_name
    protocol = _load(protocol_name)

    assert protocol["dataset_id"].startswith("NTU-RGB+D-120")
    assert protocol["yolo_version"] == "ultralytics-yolo26n-pose"
    assert protocol["ultralytics_package_version"] == "8.4.146"
    assert protocol["pose_feature_spec"] == {"confidence_threshold": 0.05, "scale_epsilon": 1e-6}
    assert protocol["flywire_release"] == "FAFB-v783"
    assert protocol["flywire_source_commit"] == "0d8574d46627ce7fadd968a3c5d602e837325373"
    assert protocol["flywire_connectivity_git_blob_sha1"] == "5183755ecbb41d5c8cee1a4a2d99b8eecba75c52"
    # These values were measured by the pinned-source provenance CI, not placeholders.
    assert protocol["flywire_connectivity_sha256"] == "215cf7a65895f9f84768db34052964542f7ebbe986ce9864b7e9ed2976c55e38"
    assert protocol["selected_graph_fingerprint"] == "a7088c8590aa10d6b204c2b11ad51d5f7889dc10ea25b68ffcf24794b06d692d"
    assert protocol["selected_graph_num_nodes"] == 187
    assert protocol["selected_graph_num_edges"] == 14542
    assert protocol["selected_graph_num_diagonal_edges"] == 126
    assert protocol["final_test_used_for_selection"] is False

    # Freezing graph/design provenance does not supply byte-backed real input identities.
    for field in ("split_hash", "observation_schema_hash", "pose_encoder_hash", "extraction_spec_hash"):
        assert protocol[field] is None

    output = tmp_path / "out"
    result = _run_compare(config, output)
    assert result.returncode != 0
    assert "freeze" in result.stderr.lower() or "required" in result.stderr.lower()
    for field in ("split_hash", "observation_schema_hash", "pose_encoder_hash", "extraction_spec_hash"):
        assert field in result.stderr
    assert not output.exists(), "rejected protocols must not emit evidence artifacts"


def test_real_topology_claim_rejects_missing_byte_level_flywire_provenance(tmp_path: Path):
    protocol = _load("v0-real-ntu120-preflight.json")
    protocol.update(
        {
            "dataset_content_hash": "dataset-sha256",
            "ultralytics_package_version": "8.4.146",
            "yolo_weights_sha256": "weights-sha256",
            "split_hash": "split-sha256",
            "observation_schema_hash": "schema-sha256",
            "pose_encoder_hash": "encoder-sha256",
            "extraction_spec_hash": "extraction-sha256",
            "selected_graph_fingerprint": "graph-sha256",
        }
    )
    protocol["flywire_connectivity_sha256"] = None

    config = tmp_path / "missing-flywire-byte-provenance.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")
    result = _run_compare(config, tmp_path / "out")

    assert result.returncode != 0
    assert "flywire_connectivity_sha256" in result.stderr
