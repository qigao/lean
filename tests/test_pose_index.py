"""Indexed sidecars: complete verification, requested-only materialization, no recognition claims."""
from dataclasses import FrozenInstanceError, asdict, replace
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path
import weakref

import pytest

from test_pose_bundle import extracted, _load, _rows, _sha, _write_report, _write_rows


def _api():
    return importlib.import_module("yolo_flywire.pose_index")


def _index(extracted, **changes):
    case, output = extracted
    options = dict(root=case[1], inventory=case[2], spec=case[4],
                   expected_manifest_sha256=_sha(output / "manifest.json"))
    options.update(changes)
    return _api().index_development_bundle(output, **options)


def _forbidden(*args, **kwargs):
    raise AssertionError("unexpected whole-corpus materialization or inference")


def test_index_matches_eager_samples_in_explicit_split_local_order(extracted, monkeypatch):
    reference = _load(extracted)
    monkeypatch.setattr("yolo_flywire.pose_bundle.load_development_bundle", _forbidden)
    monkeypatch.setattr(extracted[0][0], "decode_video", _forbidden)
    monkeypatch.setattr(extracted[0][0], "load_predictor", _forbidden)
    indexed = _index(extracted)
    assert len(indexed.samples) == 20
    assert indexed.manifest_sha256 == reference.manifest_sha256
    assert indexed.output_sha256 == reference.output_sha256
    assert indexed.dataset_content_hash == reference.dataset_content_hash
    assert indexed.split_hash == reference.split_hash
    for split in ("train", "validation"):
        expected = tuple(sample for sample in reference.samples if sample.split == split)
        order = (9, 0, 4, 2)
        assert indexed.read_samples(split=split, indices=order) == tuple(expected[i] for i in order)
        assert indexed.read_samples(split=split, indices=tuple(range(10))) == expected
        assert all(sample.subject != 3 for sample in indexed.read_samples(split=split, indices=(0,)))
    assert indexed.descriptor()["evidence_scope"] == "development_index_only"
    assert indexed.descriptor()["final_test_decoded"] is False
    indexed.verify()
    with pytest.raises(FrozenInstanceError):
        indexed.samples[0].frame_count = 99


def test_byte_spans_cover_exact_sidecars_and_index_identity_is_path_independent(extracted, tmp_path):
    import shutil
    indexed = _index(extracted)
    for field, filename in (("geometry", "observations.jsonl"), ("timing", "timing.jsonl")):
        raw = (extracted[1] / filename).read_bytes()
        cursor = 0
        for sample in indexed.samples:
            span = getattr(sample, field)
            assert span.start == cursor and span.end > span.start
            assert hashlib.sha256(raw[span.start:span.end]).hexdigest() == span.sha256
            assert len(raw[span.start:span.end].splitlines()) == sample.frame_count
            cursor = span.end
        assert cursor == len(raw)
    other = tmp_path / "relocated"
    shutil.copytree(extracted[1], other)
    second = _index((extracted[0], other))
    assert second.descriptor() == indexed.descriptor()
    assert second.index_sha256 == indexed.index_sha256
    assert second.read_samples(split="validation", indices=(2, 0)) == indexed.read_samples(split="validation", indices=(2, 0))


@pytest.mark.parametrize("frames_per_sample", (3, 67))
def test_complete_indexing_retains_no_frames_or_materialized_sequences(extracted, monkeypatch, frames_per_sample):
    api = _api()
    _, output = extracted
    poses, times = _rows(output, "observations.jsonl"), _rows(output, "timing.jsonl")
    report = json.loads((output / "manifest.json").read_text())
    new_poses, new_times = [], []
    for position, sample in enumerate(report["samples"]):
        for frame in range(frames_per_sample):
            new_poses.append({**poses[position * 3], "frame_index": frame})
            new_times.append({**times[position * 3], "frame_index": frame, "pts": frame * 2})
        sample["frame_count"] = frames_per_sample
        sample["missing_person_frames"] = 0
    _write_report(output, report)
    _write_rows(output, "observations.jsonl", new_poses)
    _write_rows(output, "timing.jsonl", new_times)
    refs, live = [], []
    actual = api._frame
    def observe(*args, **kwargs):
        result = actual(*args, **kwargs)
        refs.append(weakref.ref(result[0]))
        live.append(sum(ref() is not None for ref in refs))
        return result
    monkeypatch.setattr(api, "_frame", observe)
    monkeypatch.setattr(api, "TimedPoseSample", _forbidden)
    monkeypatch.setattr(api, "PointSequence", _forbidden)
    indexed = _index(extracted)
    assert len(refs) == 20 * frames_per_sample  # Not merely hash-only validation.
    assert max(live) <= 2
    assert all(ref() is None for ref in refs)
    assert all(sample.frame_count == frames_per_sample for sample in indexed.samples)
    def scalar_tree(value):
        if type(value) is dict:
            return all(type(k) is str and scalar_tree(v) for k, v in value.items())
        if type(value) in (tuple, list):
            return all(scalar_tree(v) for v in value)
        return type(value) in (str, int, float, bool, type(None))
    assert scalar_tree(tuple(asdict(sample) for sample in indexed.samples))


def test_requested_read_touches_only_selected_spans(extracted, monkeypatch):
    api = _api()
    indexed = _index(extracted)
    expected = tuple(sample for sample in _load(extracted).samples if sample.split == "train")
    entries = tuple(sample for sample in indexed.samples if sample.split == "train")
    original = Path.open
    consumed, parsed = {"observations.jsonl": 0, "timing.jsonl": 0}, []
    class Counted:
        def __init__(self, handle, name): self.handle, self.name = handle, name
        def __enter__(self): return self
        def __exit__(self, *args): return self.handle.__exit__(*args)
        def __getattr__(self, name): return getattr(self.handle, name)
        def readline(self, *args):
            raw = self.handle.readline(*args)
            consumed[self.name] += len(raw)
            return raw
        def read(self, size=-1):
            assert size >= 0, "must not read a whole sidecar"
            raw = self.handle.read(size)
            consumed[self.name] += len(raw)
            return raw
    def tracked(path, *args, **kwargs):
        handle = original(path, *args, **kwargs)
        return Counted(handle, path.name) if path.name in consumed else handle
    actual = api._frame
    def frame(*args, **kwargs):
        parsed.append((args[2]["sample_id"], args[3]))
        return actual(*args, **kwargs)
    monkeypatch.setattr(Path, "open", tracked)
    monkeypatch.setattr(api, "_frame", frame)
    result = indexed.read_samples(split="train", indices=(9, 1))
    assert result == (expected[9], expected[1])
    assert parsed == [(expected[i].sample_id, frame) for i in (9, 1) for frame in range(3)]
    for field, filename in (("geometry", "observations.jsonl"), ("timing", "timing.jsonl")):
        assert consumed[filename] == sum(getattr(entries[i], field).end - getattr(entries[i], field).start for i in (9, 1))


@pytest.mark.parametrize("split,indices", (("final_test", (0,)), ("test", (0,)), ("all", (0,)),
    (None, (0,)), ("train", []), ("train", ()), ("train", (True,)), ("train", (1.0,)),
    ("train", (-1,)), ("train", (10,)), ("train", (0, 0)), ("train", None)))
def test_invalid_selectors_fail_before_file_read(extracted, monkeypatch, split, indices):
    indexed = _index(extracted)
    monkeypatch.setattr(Path, "open", _forbidden)
    with pytest.raises(ValueError):
        indexed.read_samples(split=split, indices=indices)


@pytest.mark.parametrize("pin", (None, "", "A" * 64, "0" * 64))
def test_independent_pin_is_required(extracted, pin):
    _api()
    with pytest.raises(ValueError):
        _index(extracted, expected_manifest_sha256=pin)


@pytest.mark.parametrize("problem", ("late-time", "late-label", "missing-count", "extra-row",
    "missing-row", "extra-json-key", "bad-manifest", "rgb-changed"))
def test_full_verification_rejects_late_corruption_before_returning_index(extracted, problem):
    _api()
    case, output = extracted
    if problem == "rgb-changed":
        media = next(case[1].glob("*.avi"))
        with media.open("ab") as handle: handle.write(b"changed")
    elif problem == "bad-manifest":
        record = json.loads((output / "manifest.json").read_text())
        record["final_test_decoded"] = True
        _write_report(output, record)
    elif problem == "late-time":
        rows = _rows(output, "timing.jsonl")
        rows[-1]["pts"] = rows[-2]["pts"]
        _write_rows(output, "timing.jsonl", rows)
    else:
        rows = _rows(output, "observations.jsonl")
        if problem == "late-label": rows[-1]["label"] = "wrong"
        elif problem == "missing-count": rows[-1].update(detector_confidence=0., body_keypoints=[[0., 0., 0.]] * 17)
        elif problem == "extra-row": rows.append(rows[-1])
        elif problem == "missing-row": rows.pop()
        else: rows[-1]["undeclared"] = 1
        _write_rows(output, "observations.jsonl", rows)
    with pytest.raises(ValueError):
        _index(extracted)


@pytest.mark.parametrize("filename", ("manifest.json", "observations.jsonl", "timing.jsonl"))
def test_file_changes_after_indexing_are_rejected(extracted, filename):
    indexed = _index(extracted)
    with (extracted[1] / filename).open("ab") as handle: handle.write(b" ")
    with pytest.raises(ValueError):
        indexed.read_samples(split="train", indices=(0,))


@pytest.mark.parametrize("filename", ("observations.jsonl", "timing.jsonl"))
def test_requested_span_hash_is_checked_independently_of_stat_guard(extracted, monkeypatch, filename):
    api = _api()
    indexed = _index(extracted)
    before = api._bundle_state(extracted[1])
    path = extracted[1] / filename
    raw = path.read_bytes()
    # Preserve exact row length/offsets and leave the manifest unchanged.
    if filename == "observations.jsonl": changed = raw.replace(b"4.0", b"5.0", 1)
    else: changed = raw.replace(b'"pts":0', b'"pts":1', 1)
    assert changed != raw and len(changed) == len(raw)
    path.write_bytes(changed)
    monkeypatch.setattr(api, "_bundle_state", lambda directory: before)
    with pytest.raises(ValueError, match="SHA|hash|span"):
        indexed.read_samples(split="train", indices=(0,))


def test_index_record_mutation_is_not_trusted(extracted):
    indexed = _index(extracted)
    altered = replace(indexed, samples=(replace(indexed.samples[0], label="wrong"), *indexed.samples[1:]))
    with pytest.raises(ValueError): altered.read_samples(split="train", indices=(0,))
    with pytest.raises(ValueError): replace(indexed, index_sha256="0" * 64).verify()


def test_nonuniform_rational_times_missing_frames_and_joint_confidences_survive(extracted):
    _api()
    rows = _rows(extracted[1], "timing.jsonl")
    rows[0].update(pts=-1, time_base=[1, 120])
    rows[1].update(pts=1, time_base=[1, 60])
    rows[2].update(pts=7, time_base=[1, 120])
    _write_rows(extracted[1], "timing.jsonl", rows)
    sample = _index(extracted).read_samples(split="train", indices=(0,))[0]
    assert sample.timestamps == (Fraction(-1, 120), Fraction(1, 60), Fraction(7, 120))
    assert sample.missing_person == (False, True, False)
    assert sample == _load(extracted).samples[0]
