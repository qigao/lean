from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


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


def test_real_protocol_template_freezes_design_but_remains_non_executable(tmp_path: Path):
    template = ROOT / "protocols" / "v0-real-template.json"
    protocol = json.loads(template.read_text(encoding="utf-8"))

    assert protocol["dataset_id"].startswith("NTU-RGB+D-120")
    assert protocol["yolo_version"] == "ultralytics-yolo26n-pose"
    assert protocol["flywire_release"] == "FAFB-v783"
    assert protocol["flywire_source_commit"] == "0d8574d46627ce7fadd968a3c5d602e837325373"
    assert protocol["flywire_connectivity_git_blob_sha1"] == "5183755ecbb41d5c8cee1a4a2d99b8eecba75c52"

    for field in (
        "split_hash",
        "observation_schema_hash",
        "flywire_connectivity_sha256",
    ):
        assert protocol[field] is None

    result = _run_compare(template, tmp_path)
    assert result.returncode != 0
    assert "freeze" in result.stderr.lower() or "required" in result.stderr.lower()


def test_real_topology_claim_rejects_missing_byte_level_flywire_provenance(tmp_path: Path):
    protocol = _load("v0-real-ntu120-preflight.json")
    protocol.update(
        {
            "dataset_content_hash": "dataset-sha256",
            "ultralytics_package_version": "frozen-version",
            "yolo_weights_sha256": "weights-sha256",
            "split_hash": "split-sha256",
            "observation_schema_hash": "schema-sha256",
            "selected_graph_fingerprint": "graph-sha256",
        }
    )
    protocol["flywire_connectivity_sha256"] = None

    config = tmp_path / "missing-flywire-byte-provenance.json"
    config.write_text(json.dumps(protocol), encoding="utf-8")
    result = _run_compare(config, tmp_path / "out")

    assert result.returncode != 0
    assert "flywire_connectivity_sha256" in result.stderr
