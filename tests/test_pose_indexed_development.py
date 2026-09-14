"""File-bound requested feature batches; generated inputs, not recognition evidence."""
from dataclasses import fields, is_dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch

from test_pose_bundle import extracted, _rows, _write_rows
from test_pose_development import CLASSES, _options, _prepare as _eager


def _api():
    return importlib.import_module("yolo_flywire.pose_indexed_development")


def _prepare(extracted, **changes):
    options = _options(extracted)
    options.update(changes)
    return _api().load_indexed_pose_development(extracted[1], **options)


def _read(prepared, split="train", indices=(9, 0, 4), **changes):
    options = dict(split=split, indices=indices, expected_binding_sha256=prepared.binding_sha256)
    options.update(changes)
    return prepared.read_partition(**options)


def _forbidden(*args, **kwargs):
    raise AssertionError("unexpected eager materialization, encoding or file access")


@pytest.mark.parametrize("split", ("train", "validation"))
@pytest.mark.parametrize("indices", ((9, 0, 4), (1,)))
def test_requested_partitions_exactly_match_eager_values_and_metadata(extracted, split, indices):
    prepared, eager = _prepare(extracted), _eager(extracted)
    pin = prepared.binding_sha256
    prepared.verify(expected_binding_sha256=pin)
    part, reference = _read(prepared, split, indices), getattr(eager, split)
    steps = max(reference.observations.lengths[list(indices)].tolist())
    assert part.sample_ids == tuple(reference.sample_ids[i] for i in indices)
    assert part.subjects == tuple(reference.subjects[i] for i in indices)
    assert part.classes == CLASSES and part.split == split
    assert torch.equal(part.targets, reference.targets[list(indices)])
    for name in ("features", "time_mask"):
        assert torch.equal(getattr(part.observations, name),
                           getattr(reference.observations, name)[list(indices), :steps])
    assert torch.equal(part.observations.lengths, reference.observations.lengths[list(indices)])
    assert part.observations.time_mask.all().item()  # Includes the actual missing middle frame.
    assert not part.observations.features[:, 1, :120].any().item()
    prepared.verify_partition(part, split=split, indices=indices, expected_binding_sha256=pin)


def test_binding_is_complete_repeatable_and_carries_no_observations(extracted):
    prepared, second = _prepare(extracted), _prepare(extracted)
    assert prepared.binding_sha256 == second.binding_sha256
    record = prepared.descriptor()
    assert record["kind"] == "indexed_pose_development"
    assert record["evidence_scope"] == "development_preparation_only"
    assert record["final_test_evaluated"] is False
    assert record["index_sha256"] == prepared.index.index_sha256
    assert record["classes"] == list(CLASSES)
    assert len(record["rows"]) == 20
    assert {row["split"] for row in record["rows"]} == {"train", "validation"}
    assert all(row["features"]["shape"] == [3, 121] for row in record["rows"])
    assert all(row["features"]["dtype"] == "<f4" for row in record["rows"])
    assert prepared.binding_json == json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert prepared.binding_sha256 == hashlib.sha256(prepared.binding_json.encode()).hexdigest()
    source = Path(_api().__file__)
    assert record["preparation"]["source_sha256"][source.name] == hashlib.sha256(source.read_bytes()).hexdigest()
    record["rows"][0]["target"] = -1
    assert prepared.descriptor()["rows"][0]["target"] == 0
    from yolo_flywire.pose_bundle import TimedPoseSample
    from yolo_flywire.schema import KeypointFrame, PointSequence
    def scalar_tree(value):
        assert not isinstance(value, (np.ndarray, torch.Tensor, TimedPoseSample, KeypointFrame, PointSequence))
        if is_dataclass(value):
            for field in fields(value): scalar_tree(getattr(value, field.name))
        elif isinstance(value, (tuple, list)):
            for item in value: scalar_tree(item)
        elif isinstance(value, dict):
            for item in value.values(): scalar_tree(item)
    scalar_tree(prepared)


def test_preparation_encodes_one_clip_at_a_time_without_eager_fallback(extracted, monkeypatch):
    api = _api()
    real_encode, real_collate = api.encode_timed_pose, api.collate_pose_features
    arrays, tensors, calls = [], [], []
    def encode(sequence, **kwargs):
        assert not any(ref() is not None for ref in arrays + tensors)
        calls.append(len(sequence.frames))
        result = real_encode(sequence, **kwargs)
        arrays.append(weakref.ref(result))
        return result
    def collate(sequences):
        assert len(sequences) == 1, "whole partition was collated during preparation"
        result = real_collate(sequences)
        tensors.extend(weakref.ref(getattr(result, name)) for name in ("features", "lengths", "time_mask"))
        return result
    monkeypatch.setattr(api, "encode_timed_pose", encode)
    monkeypatch.setattr(api, "collate_pose_features", collate)
    monkeypatch.setattr("yolo_flywire.pose_development.load_pose_development", _forbidden)
    monkeypatch.setattr("yolo_flywire.pose_bundle.load_development_bundle", _forbidden)
    prepared = _prepare(extracted)
    assert calls == [3] * 20
    assert not any(ref() is not None for ref in arrays + tensors)
    prepared.verify(expected_binding_sha256=prepared.binding_sha256)


def test_reads_parse_and_encode_only_requested_clips(extracted, monkeypatch):
    api = _api()
    prepared = _prepare(extracted)
    import yolo_flywire.pose_index as index_api
    read, encode = index_api._read_sample, api.encode_timed_pose
    reads, encodes = [], []
    def counted_read(geometry, timing, entry):
        reads.append(entry.sample_id)
        return read(geometry, timing, entry)
    def counted_encode(sequence, **kwargs):
        encodes.append(len(sequence.frames))
        return encode(sequence, **kwargs)
    monkeypatch.setattr(index_api, "_read_sample", counted_read)
    monkeypatch.setattr(api, "encode_timed_pose", counted_encode)
    monkeypatch.setattr(api, "index_development_bundle", _forbidden)
    part = _read(prepared, indices=(8, 2))
    assert reads == list(part.sample_ids) and encodes == [3, 3]
    prepared.verify_partition(part, split="train", indices=(8, 2),
                              expected_binding_sha256=prepared.binding_sha256)
    assert len(reads) == 2 and len(encodes) == 2


@pytest.mark.parametrize("pin", (None, "", "A" * 64, "0" * 64))
def test_external_binding_pin_is_required_before_encoding(extracted, monkeypatch, pin):
    api = _api()
    prepared = _prepare(extracted)
    monkeypatch.setattr(api, "encode_timed_pose", _forbidden)
    with pytest.raises(ValueError): _read(prepared, expected_binding_sha256=pin)


@pytest.mark.parametrize("split,indices", (("final_test", (0,)), ("all", (0,)),
    ("train", ()), ("train", [0]), ("train", (True,)), ("train", (-1,)),
    ("train", (10,)), ("train", (0, 0))))
def test_invalid_requests_are_rejected_before_file_io(extracted, monkeypatch, split, indices):
    prepared = _prepare(extracted)
    monkeypatch.setattr(Path, "open", _forbidden)
    with pytest.raises(ValueError): _read(prepared, split, indices)


@pytest.mark.parametrize("classes", ((), list(CLASSES), CLASSES[:-1], CLASSES + ("other",),
                                    (CLASSES[1],) + CLASSES[1:]))
def test_vocabulary_is_explicit_and_complete(extracted, classes):
    _api()
    with pytest.raises(ValueError): _prepare(extracted, classes=classes)


def test_class_order_changes_targets_and_binding_but_not_features(extracted):
    first, second = _prepare(extracted), _prepare(extracted, classes=CLASSES[::-1])
    a, b = _read(first), _read(second)
    assert first.binding_sha256 != second.binding_sha256
    assert a.sample_ids == b.sample_ids
    assert torch.equal(a.observations.features, b.observations.features)
    assert torch.equal(9 - a.targets, b.targets)


@pytest.mark.parametrize("filename", ("manifest.json", "observations.jsonl", "timing.jsonl", "rgb"))
def test_all_raw_source_verification_precedes_any_encoding(extracted, monkeypatch, filename):
    api, options = _api(), _options(extracted)
    path = (next(extracted[0][1].glob("*P001*_rgb.avi")) if filename == "rgb"
            else extracted[1] / filename)
    with path.open("ab") as handle: handle.write(b" ")
    monkeypatch.setattr(api, "encode_timed_pose", _forbidden)
    with pytest.raises(ValueError): api.load_indexed_pose_development(extracted[1], **options)


def test_last_clip_float32_failure_prevents_prepared_source(extracted):
    _api()
    output = extracted[1]
    rows = _rows(output, "observations.jsonl")
    points = [[3., 3., .9] for _ in range(17)]
    points[5], points[6] = [2., 0., .9], [4., 0., .9]
    points[11], points[12] = [2., 2., .9], [4., 2., .9]
    points[9] = [1e39, 3., .9]  # Finite float64, not finite float32 after normalization.
    rows[-1]["body_keypoints"] = points
    _write_rows(output, "observations.jsonl", rows)
    with pytest.raises(ValueError, match="float32"):
        _prepare(extracted)


@pytest.mark.parametrize("field", ("features", "lengths", "time_mask", "targets", "sample_ids",
                                  "subjects", "classes", "split", "padding", "indices"))
def test_returned_batch_mutations_are_rejected(extracted, field):
    prepared = _prepare(extracted)
    part = _read(prepared)
    indices = (9, 0, 4)
    if field == "features": part.observations.features[0, 0, 0] += 1
    elif field == "lengths": part.observations.lengths[0] -= 1
    elif field == "time_mask": part.observations.time_mask[0, 0] = False
    elif field == "targets": part.targets[0] = 1
    elif field == "sample_ids": part = replace(part, sample_ids=part.sample_ids[::-1])
    elif field == "subjects": part = replace(part, subjects=(2,) * 3)
    elif field == "classes": part = replace(part, classes=CLASSES[::-1])
    elif field == "split": part = replace(part, split="final_test")
    elif field == "indices": indices = (0, 9, 4)
    else:
        b = part.observations
        part = replace(part, observations=replace(b,
            features=torch.cat((b.features, torch.zeros(3, 1, 121)), dim=1),
            time_mask=torch.cat((b.time_mask, torch.zeros(3, 1, dtype=torch.bool)), dim=1)))
    with pytest.raises(ValueError):
        prepared.verify_partition(part, split="train", indices=indices,
                                  expected_binding_sha256=prepared.binding_sha256)


def test_mutated_output_does_not_poison_subsequent_reads(extracted):
    prepared = _prepare(extracted)
    first, second = _read(prepared), _read(prepared)
    first.observations.features.fill_(42)
    first.targets.fill_(3)
    third = _read(prepared)
    assert torch.equal(second.observations.features, third.observations.features)
    assert torch.equal(second.targets, third.targets)


def test_binding_record_cannot_be_rehashed_around_external_pin(extracted):
    prepared = _prepare(extracted)
    pin, record = prepared.binding_sha256, prepared.descriptor()
    record["rows"][0]["target"] = 9
    text = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
    changed = replace(prepared, binding_json=text, binding_sha256=hashlib.sha256(text.encode()).hexdigest())
    with pytest.raises(ValueError): changed.verify(expected_binding_sha256=pin)


@pytest.mark.parametrize("filename", ("observations.jsonl", "timing.jsonl", "manifest.json"))
def test_post_preparation_file_changes_do_not_reindex_or_encode(extracted, monkeypatch, filename):
    api, prepared = _api(), _prepare(extracted)
    with (extracted[1] / filename).open("ab") as handle: handle.write(b" ")
    monkeypatch.setattr(api, "index_development_bundle", _forbidden)
    monkeypatch.setattr(api, "encode_timed_pose", _forbidden)
    with pytest.raises(ValueError): _read(prepared)


@pytest.mark.parametrize("identity", ("encoder", "runtime"))
def test_execution_identity_drift_is_not_silently_accepted(extracted, monkeypatch, identity):
    api, prepared = _api(), _prepare(extracted)
    if identity == "encoder": monkeypatch.setattr(api, "pose_encoder_hash", lambda spec: "0" * 64)
    else: monkeypatch.setattr(api, "_runtime", lambda: {"changed": "runtime"})
    with pytest.raises(ValueError): _read(prepared)


def test_index_reader_row_reordering_is_detected(extracted, monkeypatch):
    prepared = _prepare(extracted)
    cls = type(prepared.index)
    original = cls.read_samples
    def reversed_read(self, **kwargs): return original(self, **kwargs)[::-1]
    monkeypatch.setattr(cls, "read_samples", reversed_read)
    with pytest.raises(ValueError): _read(prepared)


def test_no_preconstructed_object_shortcut_and_encoder_pin_checked_first(extracted, monkeypatch):
    api, prepared = _api(), _prepare(extracted)
    options = _options(extracted)
    with pytest.raises(ValueError): api.load_indexed_pose_development(prepared.index, **options)
    monkeypatch.setattr(api, "index_development_bundle", _forbidden)
    with pytest.raises(ValueError): _prepare(extracted, expected_encoder_hash="0" * 64)
