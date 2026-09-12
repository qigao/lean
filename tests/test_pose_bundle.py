"""Verified reader contract; fixture extraction is not real recognition evidence."""
from dataclasses import FrozenInstanceError, asdict
from fractions import Fraction
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from test_pose_extract import _case, _run


def _module():
    return importlib.import_module("yolo_flywire.pose_bundle")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def extracted(tmp_path, monkeypatch):
    case = _case(tmp_path, monkeypatch)
    output = tmp_path / "bundle"
    _run(case, output)
    return case, output


def _load(extracted, **kwargs):
    module = _module()
    case, output = extracted
    options = dict(root=case[1], inventory=case[2], spec=case[4])
    # Explicit pins must reach the reader even when the manifest has been removed.
    if "expected_manifest_sha256" not in kwargs:
        options["expected_manifest_sha256"] = _sha(output / "manifest.json")
    options.update(kwargs)
    return module.load_development_bundle(output, **options)


def _rows(output, name):
    return [json.loads(line) for line in (output / name).read_text().splitlines()]


def _write_rows(output, name, rows):
    (output / name).write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report = json.loads((output / "manifest.json").read_text())
    report["output_sha256"][name] = _sha(output / name)
    _write_report(output, report)


def _write_report(output, report):
    (output / "manifest.json").write_text(json.dumps(report, sort_keys=True) + "\n")


def test_reader_preserves_immutable_timed_observations_without_inference(extracted, monkeypatch):
    case, output = extracted
    def forbidden(*args, **kwargs):
        raise AssertionError("bundle reading must not decode, infer or use geometry-only fallback")
    monkeypatch.setattr(case[0], "decode_video", forbidden)
    monkeypatch.setattr(case[0], "load_predictor", forbidden)
    monkeypatch.setattr("yolo_flywire.yolo_io.load_pose_jsonl", forbidden)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    bundle = _load(extracted)
    assert bundle.dataset_content_hash == case[2]["dataset_content_hash"]
    assert bundle.split_hash == case[2]["split_hash"]
    assert bundle.manifest_sha256 == _sha(output / "manifest.json")
    assert len(bundle.samples) == 20 and isinstance(bundle.samples, tuple)
    assert {sample.split for sample in bundle.samples} == {"train", "validation"}
    assert {sample.subject for sample in bundle.samples} == {1, 14}
    sample = bundle.samples[0]
    assert sample.sample_id == "S001C001P001R001A008"
    assert sample.label == "A8:sitting_down" and sample.split == "train"
    assert (sample.width, sample.height) == (12, 8)
    assert sample.pts == (0, 2, 4)
    assert sample.time_bases == (Fraction(1, 60),) * 3
    assert sample.timestamps == (Fraction(0), Fraction(1, 30), Fraction(1, 15))
    assert sample.detector_confidences == (0.9, 0.0, 0.9)
    assert sample.missing_person == (False, True, False)
    assert sample.sequence.frames[0].points == ((4.0, 3.0, 0.8),) * 17
    assert sample.sequence.frames[1].points == ((0.0, 0.0, 0.0),) * 17
    with pytest.raises(FrozenInstanceError):
        sample.label = "changed"
    with pytest.raises(TypeError):
        sample.sequence.frames[0].points[0] = (0.0, 0.0, 1.0)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_reader_preserves_nonuniform_rational_time_and_zero_confidence_joints(extracted):
    _, output = extracted
    rows = _rows(output, "timing.jsonl")
    rows[0].update(pts=-1, time_base=[1, 120])
    rows[1].update(pts=1, time_base=[1, 60])
    rows[2].update(pts=7, time_base=[1, 120])
    _write_rows(output, "timing.jsonl", rows)
    poses = _rows(output, "observations.jsonl")
    poses[0]["body_keypoints"][0] = [5.0, 6.0, 0.0]
    _write_rows(output, "observations.jsonl", poses)
    sample = _load(extracted).samples[0]
    assert sample.timestamps == (Fraction(-1, 120), Fraction(1, 60), Fraction(7, 120))
    assert sample.sequence.frames[0].points[0] == (5.0, 6.0, 0.0)


@pytest.mark.parametrize("digest", [None, "", "x" * 64, "A" * 64, 1])
def test_reader_requires_explicit_manifest_pin(extracted, digest):
    with pytest.raises(ValueError, match="manifest.*SHA|manifest.*hash"):
        _load(extracted, expected_manifest_sha256=digest)


def test_changed_manifest_fails_against_independently_supplied_pin(extracted):
    _, output = extracted
    expected = _sha(output / "manifest.json")
    with (output / "manifest.json").open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="manifest"):
        _load(extracted, expected_manifest_sha256=expected)


@pytest.mark.parametrize("filename", ["observations.jsonl", "timing.jsonl"])
def test_changed_sidecar_bytes_are_not_trusted(extracted, filename):
    _, output = extracted
    with (output / filename).open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError):
        _load(extracted)


@pytest.mark.parametrize("field,value", [
    ("format_version", True), ("format_version", 1.0),
    ("kind", "unknown"), ("evidence_scope", "confirmatory"),
    ("final_test_decoded", True), ("classifier_evaluated", True),
    ("dataset_content_hash", "0" * 64), ("split_hash", "0" * 64),
    ("input_inventory_hash", "0" * 64), ("extraction_spec_hash", "0" * 64),
    ("observation_schema_hash", "0" * 64), ("extractor_code_hash", "0" * 64),
    ("versions", {}), ("python_version", ""), ("platform", ""),
    ("extraction_spec", {}), ("observation_schema", {}),
    ("output_sha256", {"../../outside": "0" * 64}),
])
def test_self_consistent_manifest_changes_cannot_bypass_contract(extracted, field, value):
    _, output = extracted
    report = json.loads((output / "manifest.json").read_text())
    report[field] = value
    _write_report(output, report)
    # Repinning test mutations exercises semantic validation independently of hashing.
    with pytest.raises(ValueError):
        _load(extracted)


@pytest.mark.parametrize("problem", ["missing-sample", "duplicate-sample", "reordered-samples", "final-test",
    "wrong-source", "wrong-split", "float-count", "wrong-count", "wrong-missing-count", "bad-dimensions"])
def test_manifest_samples_match_complete_verified_development_roster(extracted, problem):
    _, output = extracted
    report = json.loads((output / "manifest.json").read_text())
    sample = report["samples"][0]
    if problem == "missing-sample":
        report["samples"].pop()
    elif problem == "duplicate-sample":
        report["samples"].append(dict(sample))
    elif problem == "reordered-samples":
        report["samples"].reverse()
    elif problem == "final-test":
        sample["sample_id"] = "S001C001P003R001A008"
    elif problem == "wrong-source":
        sample["source_sha256"] = "0" * 64
    elif problem == "wrong-split":
        sample["split"] = "validation"
    elif problem == "float-count":
        sample["frame_count"] = 3.0
    elif problem == "wrong-count":
        sample["frame_count"] = 4
    elif problem == "wrong-missing-count":
        sample["missing_person_frames"] = 0
    else:
        sample["width"] = 0
    _write_report(output, report)
    with pytest.raises(ValueError):
        _load(extracted)


@pytest.mark.parametrize("problem", ["label", "identity", "duplicate", "gap", "bool-index", "float-index",
    "short-points", "bool-coordinate", "string-coordinate", "nan", "bad-confidence", "bad-person-score",
    "missing-person-not-zero", "extra-feature", "missing-row", "extra-row", "reordered"])
def test_geometry_validation_rejects_tampering_even_after_rehash(extracted, problem):
    _, output = extracted
    rows = _rows(output, "observations.jsonl")
    row = rows[0]
    if problem == "label":
        row["label"] = "A9:standing_up"
    elif problem == "identity":
        row["sample_id"] = "S001C001P003R001A008"
    elif problem == "duplicate":
        rows[1] = dict(row)
    elif problem == "gap":
        row["frame_index"] = 8
    elif problem == "bool-index":
        row["frame_index"] = False
    elif problem == "float-index":
        row["frame_index"] = 0.0
    elif problem == "short-points":
        row["body_keypoints"].pop()
    elif problem in ("bool-coordinate", "string-coordinate", "nan"):
        row["body_keypoints"][0][0] = {"bool-coordinate": True, "string-coordinate": "4", "nan": float("nan")}[problem]
    elif problem == "bad-confidence":
        row["body_keypoints"][0][2] = 1.1
    elif problem == "bad-person-score":
        row["detector_confidence"] = True
    elif problem == "missing-person-not-zero":
        rows[1]["body_keypoints"][0][0] = 1.0
    elif problem == "extra-feature":
        row["feature_payload"] = [1.0]
    elif problem == "missing-row":
        rows.pop()
    elif problem == "extra-row":
        rows.append(dict(rows[-1]))
    else:
        rows[0], rows[1] = rows[1], rows[0]
    _write_rows(output, "observations.jsonl", rows)
    with pytest.raises(ValueError):
        _load(extracted)


@pytest.mark.parametrize("problem", ["identity", "index", "missing-row", "extra-row", "duplicate-pts",
    "reverse-pts", "float-pts", "bool-pts", "zero-base", "negative-base", "bool-base", "noncanonical-base"])
def test_timing_is_exact_complete_and_strictly_increasing(extracted, problem):
    _, output = extracted
    rows = _rows(output, "timing.jsonl")
    if problem == "identity":
        rows[0]["sample_id"] = rows[3]["sample_id"]
    elif problem == "index":
        rows[0]["frame_index"] = 1
    elif problem == "missing-row":
        rows.pop()
    elif problem == "extra-row":
        rows.append(dict(rows[-1]))
    elif problem in ("duplicate-pts", "reverse-pts", "float-pts", "bool-pts"):
        rows[1]["pts"] = {"duplicate-pts": 0, "reverse-pts": -1, "float-pts": 2.0, "bool-pts": True}[problem]
    else:
        rows[0]["time_base"] = {"zero-base": [1, 0], "negative-base": [-1, 60],
            "bool-base": [True, 60], "noncanonical-base": [2, 120]}[problem]
    _write_rows(output, "timing.jsonl", rows)
    with pytest.raises(ValueError):
        _load(extracted)


@pytest.mark.parametrize("name", ["manifest.json", "observations.jsonl", "timing.jsonl"])
def test_duplicate_json_keys_are_errors(extracted, name):
    _, output = extracted
    raw = (output / name).read_text()
    key = "format_version" if name == "manifest.json" else "frame_index"
    raw = raw.replace("{", '{"' + key + '":123,', 1)
    (output / name).write_text(raw)
    if name != "manifest.json":
        report = json.loads((output / "manifest.json").read_text())
        report["output_sha256"][name] = _sha(output / name)
        _write_report(output, report)
    with pytest.raises(ValueError, match="duplicate"):
        _load(extracted)


@pytest.mark.parametrize("problem", ["missing-manifest", "missing-timing", "extra-file", "symlink", "directory"])
def test_incomplete_or_ambiguous_bundle_is_not_loaded(extracted, problem):
    _, output = extracted
    expected = _sha(output / "manifest.json")
    if problem == "missing-manifest":
        (output / "manifest.json").unlink()
    elif problem == "missing-timing":
        (output / "timing.jsonl").unlink()
    elif problem == "extra-file":
        (output / "uncommitted.jsonl").write_text("{}")
    elif problem == "directory":
        (output / "timing.jsonl").unlink()
        (output / "timing.jsonl").mkdir()
    else:
        path = output / "timing.jsonl"
        moved = output.parent / "elsewhere.jsonl"
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises((ValueError, OSError)):
        _load(extracted, expected_manifest_sha256=expected)


def test_changed_original_rgb_bytes_are_reverified(extracted):
    case, _ = extracted
    row = case[2]["samples"][0]
    (case[1] / row["relative_path"]).write_bytes(b"changed original bytes")
    with pytest.raises(ValueError):
        _load(extracted)


def test_wrong_independent_spec_is_rejected(extracted):
    case, _ = extracted
    spec = type(case[4])(**{**asdict(case[4]), "weights_sha256": "0" * 64})
    with pytest.raises(ValueError):
        _load(extracted, spec=spec)


def test_read_only_cli_accepts_verified_bundle_and_rejects_wrong_pin(extracted, tmp_path, capsys):
    case, output = extracted
    inventory, config = tmp_path / "inventory.json", tmp_path / "spec.json"
    inventory.write_text(json.dumps(case[2]))
    config.write_text(json.dumps(asdict(case[4])))
    args = ["--bundle", str(output), "--root", str(case[1]), "--inventory", str(inventory),
            "--config", str(config), "--manifest-sha256", _sha(output / "manifest.json")]
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    assert _module().main(args) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["samples"] == 20 and summary["frames"] == 60
    assert summary["evidence_scope"] == "development_bundle_verified_only"
    with pytest.raises(SystemExit) as error:
        _module().main([*args[:-1], "0" * 64])
    assert error.value.code != 0
    assert capsys.readouterr().out == ""
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
