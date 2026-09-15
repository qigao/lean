from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from pose_graph_ssm.protocol import load_protocol
from pose_graph_ssm.remote_skeleton import (
    validate_remote_skeleton_manifest,
    verify_downloaded_skeleton,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_protocol(ROOT / "protocols" / "v1-development-preflight.json")


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
