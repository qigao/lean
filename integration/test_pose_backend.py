"""Real decoder/model API checks on generated inputs, never NTU or pretrained evidence."""
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import socket

import av
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("integration fixtures must not download models or data")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket.socket, "connect_ex", forbidden)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(tmp_path / "ultralytics-config"))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")


def _video(path, value=0):
    frames = []
    with av.open(str(path), "w", format="avi") as container:
        stream = container.add_stream("rawvideo", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr24"
        for index in range(2):
            image = np.full((24, 32, 3), value + index, dtype=np.uint8)
            frames.append(image)
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return frames


def test_actual_pyav_decoder_preserves_images_and_presentation_times(tmp_path):
    from yolo_flywire.pose_backend import decode_video
    path = tmp_path / "fixture.avi"
    expected = _video(path, 30)
    actual = list(decode_video(path))
    assert len(actual) == 2
    assert [pts * base for pts, base, _ in actual] == [Fraction(0), Fraction(1, 30)]
    for (_, _, image), reference in zip(actual, expected):
        np.testing.assert_array_equal(image, reference)


def test_actual_decoder_rejects_invalid_video(tmp_path):
    from yolo_flywire.pose_backend import decode_video
    path = tmp_path / "corrupt.avi"
    path.write_bytes(b"this is not a video")
    with pytest.raises((ValueError, OSError)):
        list(decode_video(path))


def test_real_untrained_model_and_extractor_are_repeatable_and_keep_test_sealed(tmp_path, monkeypatch):
    from ultralytics import YOLO
    from yolo_flywire.ntu_io import build_rgb_manifest
    from yolo_flywire.pose_extract import ExtractionSpec, extract_development
    from yolo_flywire.pose_backend import runtime_versions
    import yolo_flywire.pose_extract as extraction

    # The YAML resolves within the installed package. Network access is blocked.
    weights = tmp_path / "yolo26n-pose.pt"
    YOLO("yolo26n-pose.yaml", task="pose").save(str(weights))
    versions = runtime_versions()
    spec = ExtractionSpec(
        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        ultralytics_version=versions["ultralytics"], torch_version=versions["torch"],
        numpy_version=versions["numpy"], av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )
    root = tmp_path / "rgb"
    root.mkdir()
    value = 0
    for subject in (1, 14, 3):
        for action in (8, 9, 22, 23, 26, 27, 31, 34, 35, 36):
            _video(root / f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi", value)
            value += 3
    manifest = build_rgb_manifest(root)
    actual_decode = extraction.decode_video
    decoded = []
    def guarded_decode(path):
        assert "P003" not in path.name, "sealed final-test fixture was decoded"
        decoded.append(path.name)
        yield from actual_decode(path)
    monkeypatch.setattr(extraction, "decode_video", guarded_decode)
    for name in ("first", "second"):
        report = extract_development(root, manifest, weights, spec, tmp_path / name)
        assert report["final_test_decoded"] is False
        assert sum(row["frame_count"] for row in report["samples"]) == 40
    assert len(decoded) == 40  # 20 development clips, decoded twice
    for name in ("observations.jsonl", "timing.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    rows = [json.loads(line) for line in (tmp_path / "first" / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == 40
    assert all(len(row["body_keypoints"]) == 17 for row in rows)
    print("Real-library smoke: 40 generated video frames x 2; untrained checkpoint, NOT recognition evidence")
