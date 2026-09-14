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
    config_dir = tmp_path / "ultralytics-config"
    (config_dir / "Ultralytics").mkdir(parents=True)
    monkeypatch.setenv("YOLO_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("YOLO_AUTOINSTALL", "false")
    monkeypatch.setenv("YOLO_OFFLINE", "true")


def _video(path, value=0, frame_count=2):
    frames = []
    with av.open(str(path), "w", format="avi") as container:
        stream = container.add_stream("rawvideo", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "bgr24"
        for index in range(frame_count):
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
    fixture_model = YOLO("yolo26n-pose.yaml", task="pose")
    # PoseTrainer.set_model_attributes supplies this metadata in trained checkpoints.
    # Build a complete untrained fixture before hashing; never repair production inputs.
    assert fixture_model.model.yaml["kpt_shape"] == [17, 3]
    fixture_model.model.kpt_shape = [17, 3]
    fixture_model.save(str(weights))
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
    expected_lengths = {}
    for subject in (1, 14, 3):
        for action in (8, 9, 22, 23, 26, 27, 31, 34, 35, 36):
            sample_id = f"S001C001P{subject:03d}R001A{action:03d}"
            count = 1 + action % 3
            _video(root / f"{sample_id}_rgb.avi", value, frame_count=count)
            if subject != 3:
                expected_lengths[sample_id] = count
            value += 3
    manifest = build_rgb_manifest(root)
    actual_decode = extraction.decode_video
    decoded = []
    def guarded_decode(path):
        assert "P003" not in path.name, "sealed final-test fixture was decoded"
        decoded.append(path.name)
        yield from actual_decode(path)
    monkeypatch.setattr(extraction, "decode_video", guarded_decode)
    bundles = []
    for name in ("first", "second"):
        report = extract_development(root, manifest, weights, spec, tmp_path / name)
        assert report["final_test_decoded"] is False
        assert sum(row["frame_count"] for row in report["samples"]) == sum(expected_lengths.values())
        from yolo_flywire.pose_bundle import load_development_bundle
        bundles.append(load_development_bundle(
            tmp_path / name, root=root, inventory=manifest, spec=spec,
            expected_manifest_sha256=hashlib.sha256((tmp_path / name / "manifest.json").read_bytes()).hexdigest(),
        ))
        assert len(bundles[-1].samples) == 20
        for sample in bundles[-1].samples:
            assert sample.timestamps == tuple(Fraction(i, 30) for i in range(expected_lengths[sample.sample_id]))
        assert {sample.subject for sample in bundles[-1].samples} == {1, 14}
    assert bundles[0] == bundles[1]
    assert len(decoded) == 40  # 20 development clips, decoded twice; reader never decodes.
    for name in ("observations.jsonl", "timing.jsonl"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()
    rows = [json.loads(line) for line in (tmp_path / "first" / "observations.jsonl").read_text().splitlines()]
    assert len(rows) == sum(expected_lengths.values())
    assert all(len(row["body_keypoints"]) == 17 for row in rows)
    from yolo_flywire.pose_features import PoseFeatureSpec, encode_timed_pose
    def forbidden(*args, **kwargs):
        raise AssertionError("timed encoding must not decode, infer or use the synthetic encoder")
    monkeypatch.setattr(extraction, "decode_video", forbidden)
    monkeypatch.setattr(extraction, "load_predictor", forbidden)
    monkeypatch.setattr("yolo_flywire.features.encode_sequence", forbidden)
    encoded_runs = [[], []]
    for first, second in zip(bundles[0].samples, bundles[1].samples, strict=True):
        encoded = [encode_timed_pose(
            sample.sequence, pts=sample.pts, time_bases=sample.time_bases,
            detector_confidences=sample.detector_confidences, spec=PoseFeatureSpec(),
        ) for sample in (first, second)]
        np.testing.assert_array_equal(encoded[0], encoded[1])
        count = expected_lengths[first.sample_id]
        assert encoded[0].shape == (count, 121) and encoded[0].dtype == np.float64
        assert np.isfinite(encoded[0]).all()
        np.testing.assert_array_equal(encoded[0][:, 120], [0.] + [1 / 30] * (count - 1))
        assert not encoded[0][0, 34:68].any()
        assert not encoded[0][0, 102:119].any()
        for index in (0, 1):
            encoded_runs[index].append(encoded[index])
    from yolo_flywire.pose_batches import PoseBatch, collate_pose_features
    from yolo_flywire.graphs import DirectedGraph
    from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
    import torch
    from torch.nn import functional as F
    # Fixture classifiers and a toy graph test ingestion, not the real four-arm study.
    torch.manual_seed(203)
    networks = (GRUClassifier(121, 4, 10), GraphRecurrentClassifier(
        121, DirectedGraph(3, (0, 0, 1, 2), (0, 1, 2, 0), (.2, .4, .3, .5)), 2, 10,
    ))
    for split in ("train", "validation"):
        indices = [i for i, sample in enumerate(bundles[0].samples) if sample.split == split]
        assert len(indices) == 10
        batches = [collate_pose_features(tuple(run[i] for i in indices)) for run in encoded_runs]
        assert batches[0].features.shape == (10, 3, 121)
        assert batches[0].features.dtype == torch.float32
        assert set(batches[0].lengths.tolist()) == {1, 2, 3}
        for name in ("features", "lengths", "time_mask"):
            assert torch.equal(getattr(batches[0], name), getattr(batches[1], name))
        batch = batches[0]
        for row, index in enumerate(indices):
            source = encoded_runs[0][index]
            count = len(source)
            assert batch.lengths[row].item() == count
            assert batch.time_mask[row].tolist() == [True] * count + [False] * (3 - count)
            np.testing.assert_array_equal(batch.features[row, :count].numpy(), source.astype(np.float32))
            assert not batch.features[row, count:].any().item()
        extended = PoseBatch(F.pad(batch.features, (0, 0, 0, 5)), batch.lengths.clone(),
                             F.pad(batch.time_mask, (0, 5), value=False))
        with torch.no_grad():
            for network in networks:
                network.eval()
                predictions = network.forward_padded(batch)
                assert predictions.shape == (10, 10) and torch.isfinite(predictions).all()
                torch.testing.assert_close(predictions, network.forward_padded(batches[1]), rtol=0, atol=0)
                torch.testing.assert_close(predictions, network.forward_padded(extended), rtol=0, atol=0)
                reference = torch.cat([network(batch.features[row:row+1, :count])
                                       for row, count in enumerate(batch.lengths.tolist())])
                torch.testing.assert_close(predictions, reference, rtol=1e-5, atol=1e-6)
    from copy import deepcopy
    from yolo_flywire.pose_training import PosePartition, evaluate_padded, train_padded_model
    from yolo_flywire.train import TrainConfig
    # Fixed fixture class map outside classifier observations; not a real-data freeze.
    classes = tuple(sorted({sample.label for sample in bundles[0].samples}))
    assert len(classes) == 10
    class_index = {label: index for index, label in enumerate(classes)}
    partitions = []
    for bundle, encoded in zip(bundles, encoded_runs, strict=True):
        pair = {}
        for split in ("train", "validation"):
            indices = [i for i, sample in enumerate(bundle.samples) if sample.split == split]
            samples = tuple(bundle.samples[i] for i in indices)
            pair[split] = PosePartition(
                observations=collate_pose_features(tuple(encoded[i] for i in indices)),
                targets=torch.tensor([class_index[s.label] for s in samples], dtype=torch.int64),
                classes=classes, sample_ids=tuple(s.sample_id for s in samples),
                subjects=tuple(s.subject for s in samples), split=split,
            )
        partitions.append(pair)
    for network in networks:
        runs = [train_padded_model(
            deepcopy(network), pair["train"], pair["validation"],
            TrainConfig(seed=7, epochs=2, lr=.01, batch_size=4, parameter_ceiling=10000),
        ) for pair in partitions]
        assert runs[0].state_hash == runs[1].state_hash
        assert runs[0].epoch_order_hashes == runs[1].epoch_order_hashes
        for run, pair in zip(runs, partitions, strict=True):
            assert run.optimizer_steps == 6 and 1 <= run.best_epoch <= 2
            assert evaluate_padded(run.model, pair["validation"], batch_size=4).macro_f1 == run.best_validation_macro_f1
    assert len(decoded) == 40
    print("Real-library smoke: variable-length generated clips x 2, verified timed reads, COCO17 features, split-separated padded tensors and repeatable padded GRU/graph training; untrained YOLO checkpoint, NOT recognition evidence")
