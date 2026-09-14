from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from yolo_flywire.manifests import aggregate_topology_evidence


def _protocol() -> dict:
    return {
        "protocol_id": "e2e-synthetic-fixture-v0",
        "claim": "topology_specific_advantage",
        "dataset_id": "synthetic-v0",
        "yolo_version": "synthetic-point-sequence-v0",
        "split_hash": "6474f3228489f3bf46efaa46e84241b837be987c30993f4a4a50cc814044da84",
        "observation_schema_hash": "1b27173701fe23b9dd6a66e05a96b6abf9598d26f0a22383a8fe11d10f74e706",
        "flywire_release": "fixture-release-v0",
        "selection_rule": "inline directed fixture graph",
        "seeds": [3, 5],
        "budget": {"epochs": 2, "max_updates": 4, "parameter_ceiling": 50000},
        "arms": [
            {"family": "gru", "topology": "dense"},
            {"family": "random_graph", "topology": "random_sparse"},
            {"family": "rewired", "topology": "rewired"},
            {"family": "flywire", "topology": "connectome"},
        ],
        "graph_fixture": {
            "num_nodes": 6,
            "src": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "dst": [1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1],
            "weight": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        },
        "rewiring_algorithm": "directed-double-edge-swap-v1",
        "primary_metric": "macro_f1",
        "secondary_metrics": ["balanced_accuracy"],
        "success_threshold": 0.01,
        "final_test_used_for_selection": False,
    }


def test_compare_executes_all_arms_and_writes_structurally_matched_report(tmp_path: Path):
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(_protocol()), encoding="utf-8")
    output = tmp_path / "out"
    result = subprocess.run(
        [sys.executable, "-m", "yolo_flywire.cli", "compare", "--config", str(config), "--output", str(output)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads((output / "aggregate_report.json").read_text(encoding="utf-8"))
    assert report["conclusion"] in {"topology_advantage_supported", "topology_advantage_not_established"}
    assert set(report["arms"]) == {"gru", "random_graph", "rewired", "flywire"}
    assert len(report["seed_results"]) == 8

    split_hashes = {row["split_hash"] for row in report["seed_results"]}
    schema_hashes = {row["observation_schema_hash"] for row in report["seed_results"]}
    budgets = {json.dumps(row["budget"], sort_keys=True) for row in report["seed_results"]}
    assert split_hashes == {_protocol()["split_hash"]}
    assert schema_hashes == {_protocol()["observation_schema_hash"]}
    assert len(budgets) == 1


def test_negative_topology_result_is_explicitly_accepted():
    report = aggregate_topology_evidence(
        seed_metrics={
            3: {"rewired": 0.70, "flywire": 0.60},
            5: {"rewired": 0.65, "flywire": 0.64},
        },
        success_threshold=0.01,
    )
    assert report.conclusion == "topology_advantage_not_established"
    assert report.mean_paired_difference < 0.0
