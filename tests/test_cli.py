from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def test_synthetic_run_writes_manifest_and_metrics(tmp_path: Path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yolo_flywire.cli",
            "synthetic-run",
            "--seed",
            "7",
            "--output",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    manifest_path = tmp_path / "manifest.json"
    metrics_path = tmp_path / "metrics.json"
    assert manifest_path.is_file()
    assert metrics_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert manifest["seed"] == 7
    assert manifest["model_family"] == "gru"
    assert "macro_f1" in metrics
    assert "balanced_accuracy" in metrics
