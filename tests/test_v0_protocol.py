from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    return json.loads((ROOT / "protocols" / name).read_text(encoding="utf-8"))


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


def test_real_protocol_template_is_deliberately_non_executable(tmp_path: Path):
    template = ROOT / "protocols" / "v0-real-template.json"
    protocol = json.loads(template.read_text(encoding="utf-8"))
    for field in (
        "dataset_id",
        "yolo_version",
        "split_hash",
        "observation_schema_hash",
        "flywire_release",
        "selection_rule",
        "success_threshold",
    ):
        assert protocol[field] is None

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yolo_flywire.cli",
            "compare",
            "--config",
            str(template),
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "freeze" in result.stderr.lower() or "required" in result.stderr.lower()
