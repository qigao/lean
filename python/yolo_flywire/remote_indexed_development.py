"""Bind verified hosted remote pose bundles into the existing indexed source type."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .ntu_io import _canonical_json
from .pose_bundle import _digest
from .pose_development import _runtime, _strings
from .pose_features import PoseFeatureSpec, pose_encoder_hash
from .pose_indexed_development import (
    IndexedPoseDevelopment, _prepare_row, _preparation_identity, _roster,
)
from .remote_pose_index import index_remote_development_bundle


def load_remote_indexed_pose_development(
    bundle: str | Path, *, remote_manifest: dict[str, Any], protocol: dict[str, Any],
    feature_spec: PoseFeatureSpec, classes: tuple[str, ...],
    expected_manifest_sha256: str, expected_encoder_hash: str,
) -> IndexedPoseDevelopment:
    """Verify remote source + pose bundle, then bind features one clip at a time."""
    if not isinstance(bundle, (str, Path)):
        raise ValueError("remote pose bundle must be a file path")
    if type(feature_spec) is not PoseFeatureSpec:
        raise ValueError("explicit PoseFeatureSpec required for remote indexed binding")
    _strings(classes, "classes")
    _digest(expected_manifest_sha256, "remote pose manifest SHA-256")
    encoder_pin = _digest(expected_encoder_hash, "encoder SHA-256")
    if pose_encoder_hash(feature_spec) != encoder_pin:
        raise ValueError("remote encoder hash does not match independent pin")
    identity, runtime = _preparation_identity(), _runtime()
    indexed = index_remote_development_bundle(
        bundle,
        remote_manifest=remote_manifest,
        protocol=protocol,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    _roster(indexed, classes)
    indexed.verify()
    with (indexed.directory / "observations.jsonl").open("rb") as geometry, \
            (indexed.directory / "timing.jsonl").open("rb") as timing:
        rows = [_prepare_row(geometry, timing, entry, feature_spec, classes)
                for entry in indexed.samples]
    indexed.verify()
    record = {
        "format_version": 1,
        "kind": "indexed_pose_development",
        "evidence_scope": "development_preparation_only",
        "final_test_evaluated": False,
        "index_sha256": indexed.index_sha256,
        "classes": list(classes),
        "encoder": {"hash": encoder_pin, "descriptor": feature_spec.descriptor()},
        "preparation": identity,
        "runtime": runtime,
        "rows": rows,
    }
    binding_json = _canonical_json(record)
    result = IndexedPoseDevelopment(
        indexed,
        feature_spec,
        classes,
        binding_json,
        hashlib.sha256(binding_json.encode("utf-8")).hexdigest(),
    )
    result.verify(expected_binding_sha256=result.binding_sha256)
    return result
