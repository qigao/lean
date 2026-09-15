from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pytest

from pose_graph_ssm.protocol import load_protocol
from pose_graph_ssm.pyskl_source import _restricted_load, prepare_pyskl_data, read_pyskl_3d_annotations


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(ROOT / "protocols" / "v1-pyskl-development-preflight.json")


def _sequence(frames: int, *, offset: float = 0.0) -> np.ndarray:
    value = np.zeros((1, frames, 25, 3), dtype=np.float32)
    for t in range(frames):
        value[0, t, :, 0] = np.arange(25, dtype=np.float32) * 0.01 + offset
        value[0, t, :, 1] = np.arange(25, dtype=np.float32) * 0.02 + t * 0.01
        value[0, t, :, 2] = np.arange(25, dtype=np.float32) * 0.03
        value[0, t, 0] = (offset, 0.0, 0.0)
        value[0, t, 20] = (offset, 1.0 + 0.01 * t, 0.0)
    return value


def _annotation(sample_id: str, action: int, frames: int, *, offset: float = 0.0) -> dict[str, object]:
    return {
        "frame_dir": sample_id,
        "label": action - 1,
        "total_frames": frames,
        "keypoint": _sequence(frames, offset=offset),
    }


def _payload() -> dict[str, object]:
    train_subjects = [56, 57, 58, 59, 70, 78, 80, 81]
    train_lengths = [12, 16, 20, 24, 28, 32, 36, 40]
    annotations: list[dict[str, object]] = []
    for index, (subject, frames) in enumerate(zip(train_subjects, train_lengths)):
        annotations.append(_annotation(f"S001C001P{subject:03d}R001A008", 8, frames, offset=index * 0.01))
    annotations.append(_annotation("S001C001P014R001A008", 8, 30, offset=0.2))
    annotations.append(_annotation("S001C001P003R001A008", 8, 30, offset=0.3))
    return {
        "split": {"xsub_train": [row["frame_dir"] for row in annotations[:-1]], "xsub_val": [annotations[-1]["frame_dir"]]},
        "annotations": annotations,
    }


def _write(path: Path, payload: dict[str, object]) -> str:
    raw = pickle.dumps(payload, protocol=4)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_restricted_loader_supports_protocol2_bytes_encoding():
    assert _restricted_load(pickle.dumps(b"ntu", protocol=2)) == b"ntu"


def test_restricted_loader_rejects_arbitrary_globals():
    class Unsafe:
        def __reduce__(self):
            return os.system, ("echo forbidden",)

    with pytest.raises(ValueError, match="safely decode"):
        _restricted_load(pickle.dumps(Unsafe(), protocol=2))


def test_reads_strict_pyskl_3d_annotation_contract(tmp_path: Path):
    path = tmp_path / "ntu120_3danno.pkl"
    digest = _write(path, _payload())
    source = read_pyskl_3d_annotations(path, PROTOCOL)
    assert source.source_sha256 == digest
    assert len(source.samples) == 10
    assert source.samples[0].positions.shape[1:] == (25, 3)
    assert {sample.split for sample in source.samples} == {"train", "validation", "final_test"}


def test_uses_protocol_subject_split_not_pickle_split(tmp_path: Path):
    path = tmp_path / "ntu120_3danno.pkl"
    payload = _payload()
    payload["split"] = {"xsub_train": [], "xsub_val": [row["frame_dir"] for row in payload["annotations"]]}
    _write(path, payload)
    source = read_pyskl_3d_annotations(path, PROTOCOL)
    by_id = {sample.sample_id: sample.split for sample in source.samples}
    assert by_id["S001C001P056R001A008"] == "train"
    assert by_id["S001C001P014R001A008"] == "validation"


def test_rejects_label_identity_drift(tmp_path: Path):
    path = tmp_path / "ntu120_3danno.pkl"
    payload = _payload()
    payload["annotations"][0]["label"] = 99
    _write(path, payload)
    with pytest.raises(ValueError, match="label"):
        read_pyskl_3d_annotations(path, PROTOCOL)


def test_rejects_non_ntu25_3d_keypoints(tmp_path: Path):
    path = tmp_path / "ntu120_3danno.pkl"
    payload = _payload()
    payload["annotations"][0]["keypoint"] = np.zeros((1, 12, 17, 3), dtype=np.float32)
    _write(path, payload)
    with pytest.raises(ValueError, match="25.*3|shape"):
        read_pyskl_3d_annotations(path, PROTOCOL)


def test_interpolates_only_all_zero_padding_frames(tmp_path: Path):
    path = tmp_path / "ntu120_3danno.pkl"
    payload = _payload()
    keypoint = payload["annotations"][0]["keypoint"]
    expected = 0.5 * (keypoint[0, 4] + keypoint[0, 6])
    keypoint[0, 5] = 0.0
    _write(path, payload)
    source = read_pyskl_3d_annotations(path, PROTOCOL)
    first = next(sample for sample in source.samples if sample.sample_id == "S001C001P056R001A008")
    assert np.allclose(first.positions[5], expected)


def test_prepare_processed_source_keeps_final_test_feature_sealed(tmp_path: Path):
    pytest.importorskip("torch")
    path = tmp_path / "ntu120_3danno.pkl"
    _write(path, _payload())
    prepared = prepare_pyskl_data(path, PROTOCOL)
    assert len(prepared.train_samples) == 8
    assert len(prepared.validation_samples) == 1
    assert len(prepared.final_test_inventory) == 1
    assert "features" not in prepared.final_test_inventory[0]
    assert prepared.binding.dataset_content_hash == hashlib.sha256(path.read_bytes()).hexdigest()
