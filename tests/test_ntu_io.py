"""Byte fixtures test inventory plumbing, not video decoding or recognition."""
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import shutil

import pytest


ACTIONS = (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
# Liu et al., arXiv:1905.04757v2, section 3.2.1 (explicit reference).
PAPER_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31,
    34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95,
    97, 98, 100, 103,
}
VALIDATION = {14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103}


def _ntu():
    return importlib.import_module("yolo_flywire.ntu_io")


def _write(root, subject, action, *, camera=1, folder="rgb", payload=None):
    name = f"S001C{camera:03d}P{subject:03d}R001A{action:03d}_rgb.avi"
    path = root / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload if payload is not None else ("test bytes: " + name).encode())
    return path


def _dataset(root):
    for subject in (1, 14, 3):  # training / validation / official final-test side
        for action in ACTIONS:
            _write(root, subject, action)
    return root


def _hash(value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def test_parse_rgb_identity_and_ranges():
    ntu = _ntu()
    sample = ntu.parse_rgb_name("S032C003P106R002A120_rgb.avi")
    assert (sample.sample_id, sample.setup, sample.camera, sample.subject,
            sample.repetition, sample.action) == (
        "S032C003P106R002A120", 32, 3, 106, 2, 120)


@pytest.mark.parametrize("name", [
    "S001C001P001R001A008.skeleton", "S001C001P001R001A008_ir.avi",
    "S001C001P001R001A008.avi", "../S001C001P001R001A008_rgb.avi",
    "x/S001C001P001R001A008_rgb.avi", "S1C001P001R001A008_rgb.avi",
    "S000C001P001R001A008_rgb.avi", "S033C001P001R001A008_rgb.avi",
    "S001C004P001R001A008_rgb.avi", "S001C001P107R001A008_rgb.avi",
    "S001C001P001R003A008_rgb.avi", "S001C001P001R001A121_rgb.avi",
])
def test_reject_noncanonical_or_out_of_range_rgb_names(name):
    ntu = _ntu()
    with pytest.raises(ValueError):
        ntu.parse_rgb_name(name)


def test_subject_split_matches_explicit_reference_and_is_not_per_frame():
    ntu = _ntu()
    groups = {name: {p for p in range(1, 107) if ntu.split_for_subject(p) == name}
              for name in ("train", "validation", "final_test")}
    assert groups["train"] == PAPER_TRAIN - VALIDATION
    assert groups["validation"] == VALIDATION
    assert groups["final_test"] == set(range(1, 107)) - PAPER_TRAIN
    assert [len(groups[k]) for k in groups] == [42, 11, 53]
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["final_test"].isdisjoint(PAPER_TRAIN)
    # Pin the reference's 56/106 assignments rather than assuming any library's split.
    assert ntu.split_for_subject(56) == "train"
    assert ntu.split_for_subject(106) == "final_test"


@pytest.mark.parametrize("subject", [0, 107, True, "1", 1.0])
def test_reject_invalid_subjects_without_coercion(subject):
    ntu = _ntu()
    with pytest.raises(ValueError):
        ntu.split_for_subject(subject)


def test_manifest_hashes_actual_bytes_and_is_portable(tmp_path):
    ntu = _ntu()
    root = _dataset(tmp_path / "first")
    one = ntu.build_rgb_manifest(root)
    moved = tmp_path / "second"
    shutil.copytree(root, moved)
    two = ntu.build_rgb_manifest(moved)
    assert one == two
    assert one["evidence_scope"] == "input_inventory_only"
    assert one["media_decoding_verified"] is False
    assert one["official_release_completeness_verified"] is False
    assert len(one["samples"]) == 30
    assert one["samples"] == sorted(one["samples"], key=lambda r: r["sample_id"])
    assert str(root) not in json.dumps(one)
    content = []
    for row in one["samples"]:
        raw = (root / row["relative_path"]).read_bytes()
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
        assert row["size_bytes"] == len(raw)
        assert row["label"].startswith(f"A{row['action']}:")
        content.append({k: row[k] for k in ("sample_id", "size_bytes", "sha256")})
    assert one["dataset_content_hash"] == _hash(content)
    split = {"dataset_content_hash": one["dataset_content_hash"],
             "policy": one["split_policy"], "task_labels": one["task_labels"],
             "assignments": [{"sample_id": r["sample_id"], "split": r["split"]}
                             for r in one["samples"]]}
    assert one["split_hash"] == _hash(split)
    assert ntu.verify_rgb_manifest(moved, one) == one


def test_ignore_other_modalities_and_nonselected_actions_without_reading_them(tmp_path, monkeypatch):
    ntu = _ntu()
    root = _dataset(tmp_path / "data")
    expected = ntu.build_rgb_manifest(root)
    excluded = _write(root, 1, 1)
    skeleton = root / "rgb" / "S001C001P001R001A008.skeleton"
    skeleton.write_text("must not be ingested", encoding="utf-8")
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path in (excluded, skeleton):
            raise AssertionError("opened nonselected media or skeleton input")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert ntu.build_rgb_manifest(root) == expected


def test_all_cameras_and_repetitions_for_a_subject_stay_in_same_partition(tmp_path):
    ntu = _ntu()
    root = _dataset(tmp_path / "data")
    _write(root, 14, 23, camera=2)
    manifest = ntu.build_rgb_manifest(root)
    assert {r["split"] for r in manifest["samples"] if r["subject"] == 14} == {"validation"}


@pytest.mark.parametrize("change", ["bytes", "added", "removed", "label", "split", "hash", "policy"])
def test_verification_rejects_changed_bytes_roster_or_manifest(tmp_path, change):
    ntu = _ntu()
    root = _dataset(tmp_path / "data")
    manifest = deepcopy(ntu.build_rgb_manifest(root))
    path = root / manifest["samples"][0]["relative_path"]
    if change == "bytes":
        # Same size: a size-only check would miss the corruption.
        raw = path.read_bytes()
        path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    elif change == "added":
        _write(root, 2, 8)
    elif change == "removed":
        path.unlink()
    elif change == "label":
        manifest["samples"][0]["label"] = "A9:standing_up"
    elif change == "split":
        manifest["samples"][0]["split"] = "final_test"
    elif change == "hash":
        manifest["dataset_content_hash"] = "a" * 64
    else:
        manifest["split_policy"]["validation_subjects"] = [1]
    with pytest.raises(ValueError):
        ntu.verify_rgb_manifest(root, manifest)


@pytest.mark.parametrize("problem", ["empty", "incomplete", "duplicate-id", "duplicate-bytes", "bad-name"])
def test_inventory_fails_closed(tmp_path, problem):
    ntu = _ntu()
    root = tmp_path / "data"
    root.mkdir()
    if problem != "empty":
        _dataset(root)
    if problem == "incomplete":
        (root / "rgb" / "S001C001P003R001A008_rgb.avi").unlink()
    elif problem == "duplicate-id":
        _write(root, 1, 8, folder="duplicate")
    elif problem == "duplicate-bytes":
        raw = (root / "rgb" / "S001C001P001R001A008_rgb.avi").read_bytes()
        _write(root, 3, 9, payload=raw)
    elif problem == "bad-name":
        (root / "bad_rgb.avi").write_bytes(b"bad")
    with pytest.raises(ValueError):
        ntu.build_rgb_manifest(root)


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_inventory_rejects_symlink_aliases(tmp_path, kind):
    ntu = _ntu()
    root = _dataset(tmp_path / "data")
    if kind == "file":
        target = root / "rgb" / "S001C001P001R001A008_rgb.avi"
        (root / target.name).symlink_to(target)
    else:
        (root / "alias").symlink_to(root / "rgb", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        ntu.build_rgb_manifest(root)


def test_freeze_cli_publishes_verified_manifest_and_refuses_overwrite(tmp_path):
    ntu = _ntu()
    root = _dataset(tmp_path / "data")
    output = tmp_path / "output" / "inventory.json"
    assert ntu.main(["freeze", "--root", str(root), "--output", str(output)]) == 0
    frozen = output.read_bytes()
    manifest = json.loads(frozen)
    assert ntu.verify_rgb_manifest(root, manifest) == manifest
    assert ntu.main(["verify", "--root", str(root), "--manifest", str(output)]) == 0
    with pytest.raises(SystemExit) as exc:
        ntu.main(["freeze", "--root", str(root), "--output", str(output)])
    assert exc.value.code != 0
    assert output.read_bytes() == frozen
    assert sorted(p.name for p in output.parent.iterdir()) == ["inventory.json"]


def test_failed_freeze_does_not_create_output(tmp_path):
    ntu = _ntu()
    root = tmp_path / "empty"
    root.mkdir()
    output = tmp_path / "evidence" / "inventory.json"
    with pytest.raises(SystemExit) as exc:
        ntu.main(["freeze", "--root", str(root), "--output", str(output)])
    assert exc.value.code != 0
    assert not output.exists()
