"""Extraction orchestration tests; test doubles are not real detector evidence."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

from yolo_flywire.ntu_io import build_rgb_manifest
from yolo_flywire.yolo_io import load_pose_jsonl


ACTIONS = (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
VERSIONS = {"ultralytics": "8.4.146", "torch": "2.14.0", "numpy": "2.4.6",
            "av": "14.4.0", "opencv-python": "4.12.0.88"}


def _module():
    return importlib.import_module("yolo_flywire.pose_extract")


def _case(tmp_path, monkeypatch):
    module = _module()
    root = tmp_path / "rgb"
    root.mkdir()
    for subject in (1, 14, 3):  # train, validation, final_test
        for action in ACTIONS:
            name = f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi"
            (root / name).write_bytes(name.encode("ascii"))
    inventory = build_rgb_manifest(root)
    weights = tmp_path / "yolo26n-pose.pt"
    weights.write_bytes(b"explicit-test-only-checkpoint")
    spec = module.ExtractionSpec(
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        ultralytics_version=VERSIONS["ultralytics"], torch_version=VERSIONS["torch"],
        numpy_version=VERSIONS["numpy"], av_version=VERSIONS["av"],
        opencv_version=VERSIONS["opencv-python"],
    )
    monkeypatch.setattr(module, "runtime_versions", lambda: dict(VERSIONS))
    decoded = []

    def decode(path):
        decoded.append(path.name)
        assert "P003" not in path.name, "final-test video was decoded"
        for index in range(3):
            yield index * 2, Fraction(1, 60), np.full((8, 12, 3), index, dtype=np.uint8)

    def predict(image):
        if image[0, 0, 0] == 1:
            return np.empty((0, 17, 3)), np.empty(0)
        points = np.tile([4.0, 3.0, 0.8], (1, 17, 1))
        return points, np.array([0.9])

    monkeypatch.setattr(module, "decode_video", decode)
    monkeypatch.setattr(module, "load_predictor", lambda path, config: predict)
    return module, root, inventory, weights, spec, decoded


def _run(case, output):
    module, root, inventory, weights, spec, _ = case
    return module.extract_development(root, inventory, weights, spec, output)


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_stream_preserves_masks_timing_and_only_development_partitions(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    output = tmp_path / "poses"
    report = _run(case, output)
    assert len(case[-1]) == 20
    assert report["evidence_scope"] == "development_extraction_only"
    assert report["final_test_decoded"] is False
    assert report["classifier_evaluated"] is False
    assert report["dataset_content_hash"] == case[2]["dataset_content_hash"]
    assert report["split_hash"] == case[2]["split_hash"]
    assert report["versions"] == VERSIONS
    assert json.loads((output / "manifest.json").read_text()) == report
    poses, timing = _rows(output / "observations.jsonl"), _rows(output / "timing.jsonl")
    assert len(poses) == len(timing) == 60
    assert {(r["sample_id"], r["frame_index"]) for r in poses} == {
        (r["sample_id"], r["frame_index"]) for r in timing}
    assert timing[1]["pts"] == 2
    assert timing[1]["time_base"] == [1, 60]
    for row in poses:
        assert len(row["body_keypoints"]) == 17
        assert "P003" not in row["sample_id"]
        assert set(row) == {"sample_id", "label", "frame_index", "body_keypoints", "detector_confidence"}
        if row["frame_index"] == 1:
            assert row["body_keypoints"] == [[0.0, 0.0, 0.0]] * 17
            assert row["detector_confidence"] == 0.0
    for name, digest in report["output_sha256"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert len(load_pose_jsonl(output / "observations.jsonl").samples) == 20


def test_repeated_extraction_has_identical_observation_and_timing_bytes(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    _run(case, tmp_path / "first")
    _run(case, tmp_path / "second")
    for name in ("observations.jsonl", "timing.jsonl", "manifest.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()


@pytest.mark.parametrize("problem", ["missing-weights", "wrong-hash", "wrong-version", "changed-roster", "changed-label"])
def test_bad_provenance_fails_before_decoder_or_predictor(tmp_path, monkeypatch, problem):
    case = _case(tmp_path, monkeypatch)
    module, root, inventory, weights, _, decoded = case
    if problem == "missing-weights":
        weights.unlink()
    elif problem == "wrong-hash":
        weights.write_bytes(b"wrong")
    elif problem == "wrong-version":
        monkeypatch.setattr(module, "runtime_versions", lambda: {**VERSIONS, "torch": "9.9.9"})
    elif problem == "changed-roster":
        (root / "S001C002P001R001A008_rgb.avi").write_bytes(b"extra")
    else:
        inventory["samples"][0]["label"] = "forged"
    def forbidden(*args):
        raise AssertionError("model construction before provenance verification")
    monkeypatch.setattr(module, "load_predictor", forbidden)
    with pytest.raises((ValueError, OSError)):
        _run(case, tmp_path / "out")
    assert not decoded
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["multiple-people", "wrong-count", "nan", "bad-confidence", "score-shape"])
def test_invalid_predictions_abort_without_committed_output(tmp_path, monkeypatch, problem):
    case = _case(tmp_path, monkeypatch)
    points = np.tile([1., 2., .8], (1, 17, 1))
    scores = np.array([.9])
    if problem == "multiple-people":
        points, scores = np.repeat(points, 2, axis=0), np.array([.9, .8])
    elif problem == "wrong-count":
        points = points[:, :16]
    elif problem == "nan":
        points[0, 0, 0] = np.nan
    elif problem == "bad-confidence":
        points[0, 0, 2] = -0.1
    else:
        scores = np.array([[.9]])
    monkeypatch.setattr(case[0], "load_predictor", lambda *args: lambda image: (points, scores))
    with pytest.raises(ValueError):
        _run(case, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("problem", ["empty", "missing-pts", "duplicate-pts", "reversed-pts", "zero-timebase", "float-pts", "wrong-image", "dimension-drift", "decode-error"])
def test_invalid_video_timing_or_decode_is_not_silently_repaired(tmp_path, monkeypatch, problem):
    case = _case(tmp_path, monkeypatch)
    image = np.zeros((8, 12, 3), dtype=np.uint8)
    def broken(path):
        if problem == "decode-error":
            raise ValueError("corrupt video")
        if problem == "empty":
            return
        yield 0, Fraction(1, 30), image
        pts = {"missing-pts": None, "duplicate-pts": 0, "reversed-pts": -1, "float-pts": 1.0}.get(problem, 1)
        base = Fraction(0) if problem == "zero-timebase" else Fraction(1, 30)
        frame = image.astype(float) if problem == "wrong-image" else image
        if problem == "dimension-drift":
            frame = np.zeros((9, 12, 3), dtype=np.uint8)
        yield pts, base, frame
    monkeypatch.setattr(case[0], "decode_video", broken)
    with pytest.raises(ValueError):
        _run(case, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_input_mutation_during_extraction_is_rejected(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    original = case[0].decode_video
    changed = False
    def mutate(path):
        nonlocal changed
        yield from original(path)
        if not changed:
            path.write_bytes(b"changed after decode")
            changed = True
    monkeypatch.setattr(case[0], "decode_video", mutate)
    with pytest.raises(ValueError):
        _run(case, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_output_is_never_overwritten_or_removed(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep.txt").write_text("existing evidence")
    with pytest.raises((ValueError, FileExistsError)):
        _run(case, output)
    assert (output / "keep.txt").read_text() == "existing evidence"
    assert not case[-1]


def test_manifest_publication_failure_cleans_only_owned_output(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    def fail(*args):
        raise OSError("injected publication failure")
    monkeypatch.setattr(case[0], "_publish_exclusive", fail)
    with pytest.raises(OSError):
        _run(case, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("field,value", [("weights_sha256", "bad"), ("weights_sha256", "A" * 64),
    ("ultralytics_version", "latest"), ("torch_version", ""), ("av_version", None)])
def test_extraction_spec_rejects_unpinned_identity(tmp_path, monkeypatch, field, value):
    case = _case(tmp_path, monkeypatch)
    from dataclasses import asdict
    values = {**asdict(case[4]), field: value}
    with pytest.raises(ValueError):
        case[0].ExtractionSpec(**values)


def test_cli_exposes_no_final_test_override(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    with pytest.raises(SystemExit):
        case[0].main(["--split", "final_test"])
    assert not case[-1]
