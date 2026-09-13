import hashlib
import json
from pathlib import Path

from yolo_flywire import hosted_freeze
from yolo_flywire.hosted_freeze import freeze_hosted_inputs
from yolo_flywire.ntu_io import _ACTION_LABELS, _split_policy, split_for_subject


ROOT = Path(__file__).resolve().parents[1]


def _manifest():
    subjects = {
        split: next(subject for subject in range(1, 107) if split_for_subject(subject) == split)
        for split in ("train", "validation", "final_test")
    }
    rows = []
    for split, subject in subjects.items():
        for action, label in _ACTION_LABELS.items():
            filename = f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi"
            sample_id = filename.removesuffix("_rgb.avi")
            data = sample_id.encode()
            rows.append({
                "filename": filename, "sample_id": sample_id,
                "setup": 1, "camera": 1, "subject": subject, "repetition": 1,
                "action": action, "label": label, "split": split,
                "size_bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                "locator": f"https://example.invalid/{sample_id}",
            })
    return {
        "format_version": 1, "kind": "ntu120_rgb_remote_transport",
        "task_labels": [_ACTION_LABELS[action] for action in sorted(_ACTION_LABELS)],
        "split_policy": _split_policy(), "samples": rows,
    }


def test_hosted_freeze_binds_yolo11_and_keeps_transport_out_of_protocol(tmp_path, monkeypatch):
    protocol = json.loads((ROOT / "protocols/v1-hosted-yolo11n-preflight.json").read_text())
    weights = tmp_path / "yolo11n-pose.pt"
    weights.write_bytes(b"trusted-test-yolo11")
    versions = {
        "ultralytics": "8.4.146",
        "torch": "2.14.0",
        "numpy": "2.4.6",
        "av": "15.1.0",
        "opencv-python": "5.0.0.93",
    }
    monkeypatch.setattr(hosted_freeze, "runtime_versions", lambda: versions)
    frozen, transport = freeze_hosted_inputs(protocol, remote_manifest=_manifest(), weights=weights)
    assert frozen["protocol_id"] == "v1-hosted-yolo11n-preflight"
    assert frozen["yolo_weights"] == "yolo11n-pose.pt"
    assert frozen["yolo_weights_sha256"] == hashlib.sha256(weights.read_bytes()).hexdigest()
    assert frozen["extraction_spec"]["model"] == "yolo11n-pose.pt"
    assert frozen["source_manifest_hash"] == transport["source_manifest_hash"]
    assert "transport_manifest_hash" not in frozen
    assert transport["locator_contents_recorded"] is False
    assert frozen["final_test_decoded"] is False
    assert frozen["classifier_evaluated"] is False


def test_existing_yolo26_protocol_identity_is_unchanged():
    protocol = json.loads((ROOT / "protocols/v0-real-ntu120-preflight.json").read_text())
    assert protocol["yolo_weights"] == "yolo26n-pose.pt"
    assert protocol["yolo_version"] == "ultralytics-yolo26n-pose"
