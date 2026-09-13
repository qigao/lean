from fractions import Fraction
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

import yolo_flywire.kth_extract as kth_extract
from yolo_flywire.kth_extract import KthExtractionSpec, extract_action_shard
from yolo_flywire.kth_source import _ACTIONS
from yolo_flywire.ntu_io import _hash_json


def _sequence_text() -> str:
    lines = [
        "Training:   person11, 12, 13, 14, 15, 16, 17, 18",
        "Validation: person19, 20, 21, 23, 24, 25, 01, 04",
        "Test:       person22, 02, 03, 05, 06, 07, 08, 09, 10",
    ]
    reduced = 5
    for subject in range(1, 26):
        for action in _ACTIONS:
            for scenario in range(1, 5):
                key = f"person{subject:02d}_{action}_d{scenario}"
                if key == "person13_handclapping_d3":
                    lines.append(f"{key} *missing*")
                    continue
                ranges = ["1-2", "3-4", "5-6", "7-8"]
                if reduced:
                    ranges.pop()
                    reduced -= 1
                lines.append(key + " frames " + ", ".join(ranges))
    return "\n".join(lines) + "\n"


def _spec(weights: Path, versions: dict[str, str]) -> KthExtractionSpec:
    return KthExtractionSpec(
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        ultralytics_version=versions["ultralytics"], torch_version=versions["torch"],
        numpy_version=versions["numpy"], av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )


def test_kth_person_policy_is_frozen_as_unique_largest_detector_bbox():
    spec = KthExtractionSpec(
        weights_sha256="0" * 64,
        ultralytics_version="8.4.146", torch_version="2.14.0", numpy_version="2.4.6",
        av_version="15.1.0", opencv_version="5.0.0.93",
    )
    assert spec.descriptor()["person_policy"] == (
        "zero-mask-or-single-or-unique-largest-detector-bbox-else-error"
    )
    assert spec.descriptor()["decoder"] == "pyav-avi-single-thread-through-last-official-frame"


def test_kth_pose_selects_unique_largest_detector_bbox_not_highest_confidence():
    points = np.zeros((2, 17, 3), dtype=np.float32)
    points[0, :, 0] = 10.0
    points[1, :, 0] = 20.0
    points[..., 2] = 0.8
    scores = np.array([0.95, 0.75], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 30.0, 20.0]], dtype=np.float32)

    body, confidence = kth_extract._kth_pose(points, scores, boxes)

    assert np.array_equal(np.asarray(body, dtype=np.float32), points[1])
    assert confidence == pytest.approx(0.75)


def test_kth_pose_rejects_tied_largest_detector_bbox():
    points = np.zeros((2, 17, 3), dtype=np.float32)
    points[..., 2] = 0.8
    scores = np.array([0.9, 0.8], dtype=np.float32)
    boxes = np.array([[0.0, 0.0, 10.0, 20.0], [5.0, 5.0, 25.0, 15.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="largest detector bbox"):
        kth_extract._kth_pose(points, scores, boxes)


def test_kth_parent_never_requests_frame_after_last_official_interval(tmp_path, monkeypatch):
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    def fake_decode(_path):
        yield 0, Fraction(1, 25), image
        yield 1, Fraction(1, 25), image
        raise ValueError("decoder advanced beyond last official frame")

    monkeypatch.setattr(kth_extract, "decode_video", fake_decode)

    def fake_predict(_image):
        points = np.zeros((1, 17, 3), dtype=np.float32)
        points[..., 2] = 0.8
        scores = np.array([0.9], dtype=np.float32)
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        return points, scores, boxes

    rows = [{
        "sample_id": "person11_boxing_d1#01",
        "video_key": "person11_boxing_d1",
        "split": "train",
        "label": "boxing",
        "subject": 11,
        "scenario": 1,
        "range_index": 1,
        "start_frame": 1,
        "end_frame": 2,
    }]
    reports = kth_extract._extract_parent(
        tmp_path / "clip.avi", {}, rows, fake_predict, tmp_path / "samples"
    )
    assert reports[0]["frame_count"] == 2


def test_kth_action_shard_never_decodes_final_test_subjects(tmp_path, monkeypatch):
    sequence = tmp_path / "00sequences.txt"
    raw_sequence = _sequence_text().encode()
    sequence.write_bytes(raw_sequence)
    action = "boxing"
    archive = tmp_path / "boxing.zip"
    with ZipFile(archive, "w") as zipped:
        for subject in range(1, 26):
            for scenario in range(1, 5):
                zipped.writestr(
                    f"person{subject:02d}_{action}_d{scenario}_uncomp.avi",
                    f"video:{subject}:{scenario}".encode(),
                )
    weights = tmp_path / "yolo11n-pose.pt"
    weights.write_bytes(b"fixture-yolo11")
    versions = {
        "ultralytics": "8.4.146", "torch": "2.14.0", "numpy": "2.4.6",
        "av": "15.1.0", "opencv-python": "5.0.0.93",
    }
    spec = _spec(weights, versions)
    protocol = {
        "sequence_file_sha256": hashlib.sha256(raw_sequence).hexdigest(),
        "extraction_spec": spec.descriptor(),
        "extraction_spec_hash": _hash_json(spec.descriptor()),
    }
    monkeypatch.setattr(kth_extract, "runtime_versions", lambda: dict(versions))
    decode_subjects = []

    def fake_decode(path):
        decode_subjects.append(int(path.name[6:8]))
        for index in range(8):
            yield index, Fraction(1, 25), np.zeros((120, 160, 3), dtype=np.uint8)

    monkeypatch.setattr(kth_extract, "decode_video", fake_decode)

    def fake_predict(_image):
        points = np.zeros((1, 17, 3), dtype=np.float32)
        points[..., 2] = 0.8
        scores = np.array([0.9], dtype=np.float32)
        boxes = np.array([[0.0, 0.0, 100.0, 100.0]], dtype=np.float32)
        return points, scores, boxes

    monkeypatch.setattr(kth_extract, "_load_predictor", lambda _weights, _spec: fake_predict)
    output = tmp_path / "shard"
    report = extract_action_shard(
        action, archive, sequence_file=sequence, protocol=protocol, weights=weights, output=output,
    )
    final_subjects = {22, 2, 3, 5, 6, 7, 8, 9, 10}
    assert len(report["videos"]) == 100
    assert len(decode_subjects) == 64
    assert not (set(decode_subjects) & final_subjects)
    for video in report["videos"]:
        assert video["decoded"] is (video["subject"] not in final_subjects)
    assert all(sample["split"] != "final_test" for sample in report["samples"])
    assert report["final_test_decoded"] is False
    assert not (output / ".scratch").exists()
