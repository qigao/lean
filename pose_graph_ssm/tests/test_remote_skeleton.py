from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pose_graph_ssm.protocol import load_protocol
from pose_graph_ssm.remote_skeleton import (
    build_remote_skeleton_transport,
    require_development_coverage,
    validate_remote_skeleton_manifest,
    verify_downloaded_skeleton,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(ROOT / "protocols" / "v1-development-preflight.json")


def _joint_line(x: float, y: float, z: float, tracking: int = 2) -> str:
    floats = [x, y, z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    return " ".join(str(value) for value in floats) + f" {tracking}"


def _write_skeleton(root: Path, *, subject: int, action: int, frames: int = 4) -> Path:
    path = root / f"S001C001P{subject:03d}R001A{action:03d}.skeleton"
    lines = [str(frames)]
    for frame in range(frames):
        lines.extend(["1", "10 0 0 0 0 0 0 0 0 0", "25"])
        for joint in range(25):
            y = joint * 0.01 + (1.0 if joint == 20 else 0.0)
            lines.append(_joint_line(frame * 0.01 + joint * 0.001, y, 0.5))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _manifest(locator: str = "https://example.invalid/sample.skeleton"):
    payload = b"synthetic-skeleton-bytes\n"
    digest = hashlib.sha256(payload).hexdigest()
    return {
        "format_version": 1,
        "kind": "ntu120_skeleton_remote_transport",
        "samples": [
            {
                "filename": "S001C001P056R001A008.skeleton",
                "sample_id": "S001C001P056R001A008",
                "setup": 1,
                "camera": 1,
                "subject": 56,
                "repetition": 1,
                "action": 8,
                "split": "train",
                "size_bytes": len(payload),
                "sha256": digest,
                "locator": locator,
            }
        ],
    }, payload


def test_remote_manifest_binds_canonical_skeleton_identity():
    raw, _ = _manifest()
    verified = validate_remote_skeleton_manifest(raw, PROTOCOL)
    assert len(verified.samples) == 1
    row = verified.samples[0]
    assert row["sample_id"] == "S001C001P056R001A008"
    assert row["split"] == "train"
    assert len(verified.transport_manifest_hash) == 64
    assert len(verified.source_manifest_hash) == 64


def test_remote_manifest_rejects_non_https_locator():
    raw, _ = _manifest("http://example.invalid/sample.skeleton")
    with pytest.raises(ValueError, match="HTTPS"):
        validate_remote_skeleton_manifest(raw, PROTOCOL)


def test_remote_manifest_rejects_split_drift():
    raw, _ = _manifest()
    raw["samples"][0]["split"] = "validation"
    with pytest.raises(ValueError, match="split"):
        validate_remote_skeleton_manifest(raw, PROTOCOL)


def test_remote_manifest_rejects_action_outside_frozen_roster():
    raw, _ = _manifest()
    row = raw["samples"][0]
    row["filename"] = "S001C001P056R001A001.skeleton"
    row["sample_id"] = "S001C001P056R001A001"
    row["action"] = 1
    with pytest.raises(ValueError, match="action"):
        validate_remote_skeleton_manifest(raw, PROTOCOL)


def test_downloaded_skeleton_must_match_manifest_bytes(tmp_path):
    raw, payload = _manifest()
    verified = validate_remote_skeleton_manifest(raw, PROTOCOL)
    row = verified.samples[0]
    target = tmp_path / row["filename"]
    target.write_bytes(payload)
    verify_downloaded_skeleton(target, row)

    target.write_bytes(payload + b"tampered")
    with pytest.raises(ValueError, match="SHA-256|size"):
        verify_downloaded_skeleton(target, row)


def test_builder_uses_local_byte_identity_and_https_prefix(tmp_path):
    source = _write_skeleton(tmp_path, subject=56, action=8)
    raw = build_remote_skeleton_transport(
        tmp_path,
        PROTOCOL,
        locator_prefix="https://private.example.invalid/ntu120-skeleton/",
    )
    assert raw["format_version"] == 1
    assert raw["kind"] == "ntu120_skeleton_remote_transport"
    assert len(raw["samples"]) == 1
    row = raw["samples"][0]
    assert row["filename"] == source.name
    assert row["size_bytes"] == source.stat().st_size
    assert row["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert row["locator"] == f"https://private.example.invalid/ntu120-skeleton/{source.name}"
    validate_remote_skeleton_manifest(raw, PROTOCOL)


def test_builder_is_canonical_and_never_embeds_local_root(tmp_path):
    _write_skeleton(tmp_path, subject=56, action=8)
    first = build_remote_skeleton_transport(tmp_path, PROTOCOL, locator_prefix="https://private.example.invalid/data")
    second = build_remote_skeleton_transport(tmp_path, PROTOCOL, locator_prefix="https://private.example.invalid/data/")
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )
    assert str(tmp_path) not in json.dumps(first)


def test_development_coverage_requires_every_action_in_train_and_validation():
    raw, _ = _manifest()
    verified = validate_remote_skeleton_manifest(raw, PROTOCOL)
    with pytest.raises(ValueError, match="coverage"):
        require_development_coverage(verified, PROTOCOL)
