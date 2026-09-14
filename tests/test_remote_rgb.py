import hashlib
from pathlib import Path

import pytest

from yolo_flywire.ntu_io import _ACTION_LABELS, _split_policy, build_rgb_manifest, split_for_subject
from yolo_flywire.remote_rgb import validate_remote_manifest


def _subjects():
    return {
        split: next(subject for subject in range(1, 107) if split_for_subject(subject) == split)
        for split in ("train", "validation", "final_test")
    }


def _manifest(tmp_path: Path | None = None):
    subjects = _subjects()
    rows = []
    for split, subject in subjects.items():
        for action, label in _ACTION_LABELS.items():
            filename = f"S001C001P{subject:03d}R001A{action:03d}_rgb.avi"
            sample_id = filename.removesuffix("_rgb.avi")
            data = (sample_id + "\n").encode()
            if tmp_path is not None:
                (tmp_path / filename).write_bytes(data)
            rows.append({
                "filename": filename,
                "sample_id": sample_id,
                "setup": 1,
                "camera": 1,
                "subject": subject,
                "repetition": 1,
                "action": action,
                "label": label,
                "split": split,
                "size_bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "locator": f"https://example.invalid/authorized/{sample_id}",
            })
    return {
        "format_version": 1,
        "kind": "ntu120_rgb_remote_transport",
        "task_labels": [_ACTION_LABELS[action] for action in sorted(_ACTION_LABELS)],
        "split_policy": _split_policy(),
        "samples": rows,
    }


def test_remote_manifest_matches_local_dataset_and_split_hashes(tmp_path):
    manifest = _manifest(tmp_path)
    remote = validate_remote_manifest(manifest)
    local = build_rgb_manifest(tmp_path)
    assert remote.dataset_content_hash == local["dataset_content_hash"]
    assert remote.split_hash == local["split_hash"]
    assert remote.public_record()["source_manifest_hash"] == remote.source_manifest_hash
    assert "locator" not in str(remote.public_record())


def test_transport_rotation_does_not_change_scientific_identity():
    first = _manifest()
    second = _manifest()
    for row in second["samples"]:
        row["locator"] += "?rotated=1"
    a = validate_remote_manifest(first)
    b = validate_remote_manifest(second)
    assert a.dataset_content_hash == b.dataset_content_hash
    assert a.split_hash == b.split_hash
    assert a.source_manifest_hash == b.source_manifest_hash
    assert a.transport_manifest_hash != b.transport_manifest_hash


def test_remote_manifest_rejects_split_drift():
    manifest = _manifest()
    manifest["samples"][0]["split"] = "final_test"
    with pytest.raises(ValueError, match="split"):
        validate_remote_manifest(manifest)
