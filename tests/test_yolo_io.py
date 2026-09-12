from __future__ import annotations

import json

from yolo_flywire.yolo_io import load_pose_jsonl


def _record(sample_id: str, frame_index: int, label: str, offset: float = 0.0):
    return {
        "sample_id": sample_id,
        "frame_index": frame_index,
        "label": label,
        "body_keypoints": [
            [0.0 + offset, 0.0, 0.9],
            [-0.2 + offset, -0.1, 0.8],
            [0.2 + offset, -0.1, 0.7],
        ],
        "bbox": [0.0, 0.0, 1.0, 2.0],
        "detector_confidence": 0.95,
    }


def test_frozen_pose_loader_orders_frames_and_reports_schema_hash(tmp_path):
    path = tmp_path / "pose.jsonl"
    records = [
        _record("b", 1, "walking", 0.2),
        _record("a", 1, "standing", 0.1),
        _record("a", 0, "standing", 0.0),
        _record("b", 0, "walking", 0.0),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")

    loaded = load_pose_jsonl(path)
    assert [sample.sample_id for sample in loaded.samples] == ["a", "b"]
    assert [frame.points[0][0] for frame in loaded.samples[0].sequence.frames] == [0.0, 0.1]
    assert loaded.samples[0].label == "standing"
    assert loaded.samples[1].label == "walking"
    assert len(loaded.observation_schema_hash) == 64


def test_frozen_pose_loader_rejects_duplicate_frame_identity(tmp_path):
    path = tmp_path / "duplicate.jsonl"
    records = [_record("a", 0, "standing"), _record("a", 0, "standing", 0.1)]
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    try:
        load_pose_jsonl(path)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("expected duplicate frame rejection")


def test_frozen_pose_loader_rejects_label_leakage_and_schema_drift(tmp_path):
    leakage = tmp_path / "leakage.jsonl"
    record = _record("a", 0, "standing")
    record["feature_payload"] = {"gesture": "standing"}
    leakage.write_text(json.dumps(record) + "\n", encoding="utf-8")
    try:
        load_pose_jsonl(leakage)
    except ValueError as exc:
        assert "label" in str(exc).lower() or "leak" in str(exc).lower()
    else:
        raise AssertionError("expected feature payload leakage rejection")

    drift = tmp_path / "drift.jsonl"
    first = _record("a", 0, "standing")
    second = _record("a", 1, "standing")
    second["body_keypoints"] = second["body_keypoints"][:-1]
    drift.write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n", encoding="utf-8")
    try:
        load_pose_jsonl(drift)
    except ValueError as exc:
        assert "keypoint" in str(exc).lower() or "schema" in str(exc).lower()
    else:
        raise AssertionError("expected keypoint schema drift rejection")
