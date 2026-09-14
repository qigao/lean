"""Verified development binding: generated fixtures, not recognition evidence."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import importlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from test_pose_bundle import extracted, _load, _sha
from yolo_flywire.pose_features import PoseFeatureSpec, encode_timed_pose, pose_encoder_hash

CLASSES = (
    "A8:sitting_down", "A9:standing_up", "A22:cheer_up", "A23:hand_waving",
    "A26:hopping", "A27:jump_up", "A31:pointing", "A34:rub_two_hands_together",
    "A35:nod_head_or_bow", "A36:shake_head",
)


def _api():
    return importlib.import_module("yolo_flywire.pose_development")


def _options(extracted):
    case, output = extracted
    spec = PoseFeatureSpec()
    return dict(root=case[1], inventory=case[2], extraction_spec=case[4],
                feature_spec=spec, classes=CLASSES,
                expected_manifest_sha256=_sha(output / "manifest.json"),
                expected_encoder_hash=pose_encoder_hash(spec))


def _prepare(extracted, **changes):
    api = _api()
    options = _options(extracted)
    options.update(changes)
    return api.load_pose_development(extracted[1], **options)


@pytest.mark.parametrize("split", ("train", "validation"))
def test_complete_file_roster_targets_and_observations(extracted, split, monkeypatch):
    _api()
    source = _load(extracted)
    def forbidden(*args, **kwargs):
        raise AssertionError("preparation must not decode, infer or use synthetic encoding")
    monkeypatch.setattr(extracted[0][0], "decode_video", forbidden)
    monkeypatch.setattr(extracted[0][0], "load_predictor", forbidden)
    monkeypatch.setattr("yolo_flywire.features.encode_sequence", forbidden)
    prepared = _prepare(extracted)
    prepared.verify()
    part = getattr(prepared, split)
    samples = tuple(s for s in source.samples if s.split == split)
    assert part.classes == CLASSES and part.split == split
    assert part.sample_ids == tuple(s.sample_id for s in samples)
    assert part.subjects == tuple(s.subject for s in samples)
    assert part.targets.tolist() == [CLASSES.index(s.label) for s in samples]
    assert part.targets.dtype == torch.int64 and part.targets.device.type == "cpu"
    assert part.observations.features.shape == (10, 3, 121)
    assert part.observations.lengths.tolist() == [3] * 10
    assert part.observations.time_mask.all().item()
    for row, sample in enumerate(samples):
        expected = encode_timed_pose(sample.sequence, pts=sample.pts,
            time_bases=sample.time_bases, detector_confidences=sample.detector_confidences,
            spec=PoseFeatureSpec())
        np.testing.assert_array_equal(part.observations.features[row].numpy(), expected.astype(np.float32))
        assert not part.observations.features[row, 1, :120].any().item()
        assert part.observations.features[row, 1, 120] > 0  # Missing observation, not padding.
    assert set(prepared.train.sample_ids).isdisjoint(prepared.validation.sample_ids)
    assert set(prepared.train.subjects).isdisjoint(prepared.validation.subjects)
    assert all("P003" not in sid for sid in part.sample_ids)


def test_binding_record_and_repeatability(extracted):
    api = _api()
    case, output = extracted
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    inventory = deepcopy(case[2])
    first, second, source = _prepare(extracted), _prepare(extracted), _load(extracted)
    assert first.binding_sha256 == second.binding_sha256
    record = first.descriptor()
    assert record["format_version"] == 1
    assert record["evidence_scope"] == "development_preparation_only"
    assert record["classes"] == list(CLASSES)
    assert record["encoder"] == {"hash": pose_encoder_hash(PoseFeatureSpec()),
                                 "descriptor": PoseFeatureSpec().descriptor()}
    for name in ("manifest_sha256", "dataset_content_hash", "split_hash", "input_inventory_hash",
                 "extraction_spec_hash", "observation_schema_hash", "extractor_code_hash"):
        assert record["source"][name] == getattr(source, name)
    assert record["source"]["output_sha256"] == dict(source.output_sha256)
    assert record["runtime"]["torch"] == torch.__version__
    assert record["runtime"]["numpy"] == np.__version__
    for name in ("pose_development.py", "pose_batches.py", "pose_training.py", "pose_bundle.py"):
        assert record["preparation"]["source_sha256"][name] == _sha(Path(api.__file__).with_name(name))
    assert set(record["partitions"]["train"]["tensors"]) == {"features", "lengths", "time_mask", "targets"}
    assert first.binding_json == json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert first.binding_sha256 == hashlib.sha256(first.binding_json.encode()).hexdigest()
    record["classes"][0] = "changed"
    assert first.descriptor()["classes"] == list(CLASSES)
    first.train.observations.features[0, 0, 0] += 1
    with pytest.raises(ValueError):
        first.verify()
    second.verify()
    assert inventory == case[2] and before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_class_order_and_encoder_policy_are_explicit_binding_inputs(extracted):
    _api()
    first, second = _prepare(extracted), _prepare(extracted, classes=CLASSES[::-1])
    assert first.binding_sha256 != second.binding_sha256
    for split in ("train", "validation"):
        a, b = getattr(first, split), getattr(second, split)
        assert a.sample_ids == b.sample_ids
        assert torch.equal(a.observations.features, b.observations.features)
        assert torch.equal(b.targets, 9 - a.targets)
    changed = PoseFeatureSpec(confidence_threshold=.9)
    with pytest.raises(ValueError):
        _prepare(extracted, feature_spec=changed)
    third = _prepare(extracted, feature_spec=changed, expected_encoder_hash=pose_encoder_hash(changed))
    assert first.binding_sha256 != third.binding_sha256
    third.verify()


@pytest.mark.parametrize("classes", ((), list(CLASSES), (True,) + CLASSES[1:],
    (CLASSES[1],) + CLASSES[1:], ("",) + CLASSES[1:], CLASSES[:-1], CLASSES + ("unknown",)))
def test_vocabulary_is_not_inferred_or_repaired(extracted, classes):
    _api()
    with pytest.raises(ValueError):
        _prepare(extracted, classes=classes)


@pytest.mark.parametrize("pin", (None, "", "A" * 64, "z" * 64, "0" * 64))
def test_encoder_pin_checked_before_file_read(extracted, pin, monkeypatch):
    api = _api()
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid encoder pin reached file reader")
    monkeypatch.setattr(api, "load_development_bundle", forbidden)
    with pytest.raises(ValueError):
        _prepare(extracted, expected_encoder_hash=pin)


@pytest.mark.parametrize("filename", ("manifest.json", "observations.jsonl", "timing.jsonl", "rgb"))
def test_changed_source_bytes_rejected_before_encoding(extracted, filename, monkeypatch):
    api = _api()
    options = _options(extracted)
    path = (next(extracted[0][1].glob("*P001*_rgb.avi")) if filename == "rgb"
            else extracted[1] / filename)
    with path.open("ab") as handle:
        handle.write(b" ")
    def forbidden(*args, **kwargs):
        raise AssertionError("unverified bytes reached encoding")
    monkeypatch.setattr(api, "encode_timed_pose", forbidden)
    with pytest.raises(ValueError):
        api.load_pose_development(extracted[1], **options)


@pytest.mark.parametrize("field", ("features", "lengths", "time_mask", "targets",
                                  "sample_ids", "subjects", "classes", "split"))
def test_mutated_partition_rejected_by_binding(extracted, field):
    _api()
    prepared = _prepare(extracted)
    part = prepared.train
    if field == "features": part.observations.features[0, 0, 0] += 1
    elif field == "lengths": part.observations.lengths[0] -= 1
    elif field == "time_mask": part.observations.time_mask[0, 0] = False
    elif field == "targets": part.targets[0] = (part.targets[0] + 1) % 10
    elif field == "sample_ids": part = replace(part, sample_ids=part.sample_ids[::-1])
    elif field == "subjects": part = replace(part, subjects=(2,) * 10)
    elif field == "classes": part = replace(part, classes=CLASSES[::-1])
    else: part = replace(part, split="final_test")
    with pytest.raises(ValueError):
        replace(prepared, train=part).verify()


def test_record_mutation_and_unverified_object_shortcut_rejected(extracted):
    api = _api()
    prepared = _prepare(extracted)
    record = prepared.descriptor()
    record["source"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        replace(prepared, binding_json=json.dumps(record, sort_keys=True, separators=(",", ":"))).verify()
    assert tuple(inspect.signature(api.load_pose_development).parameters) == (
        "bundle", "root", "inventory", "extraction_spec", "feature_spec", "classes",
        "expected_manifest_sha256", "expected_encoder_hash",
    )
    with pytest.raises(ValueError):
        api.load_pose_development(_load(extracted), **_options(extracted))


@pytest.mark.parametrize("kind", ("gru", "graph"))
def test_bound_partitions_feed_actual_repeatable_training(extracted, kind):
    _api()
    from yolo_flywire.graphs import DirectedGraph
    from yolo_flywire.models import GRUClassifier, GraphRecurrentClassifier
    from yolo_flywire.pose_training import train_padded_model, evaluate_padded
    from yolo_flywire.train import TrainConfig
    prepared, runs = _prepare(extracted), []
    for _ in range(2):
        model = (GRUClassifier(121, 4, 10) if kind == "gru" else GraphRecurrentClassifier(
            121, DirectedGraph(3, (0, 1, 2), (1, 2, 0), (.2, .3, .4)), 2, 10))
        run = train_padded_model(model, prepared.train, prepared.validation,
            TrainConfig(seed=7, epochs=2, lr=.01, batch_size=4, parameter_ceiling=10000))
        assert run.optimizer_steps == 6
        assert evaluate_padded(run.model, prepared.validation, batch_size=4).macro_f1 == run.best_validation_macro_f1
        prepared.verify()
        runs.append(run)
    assert runs[0].state_hash == runs[1].state_hash
    assert runs[0].epoch_order_hashes == runs[1].epoch_order_hashes
