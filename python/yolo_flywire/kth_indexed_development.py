"""Bind verified KTH pose evidence into the existing IndexedPoseDevelopment type."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .kth_pose_index import index_kth_development_bundle
from .ntu_io import _canonical_json
from .pose_bundle import _digest
from .pose_development import _runtime, _strings
from .pose_features import PoseFeatureSpec, pose_encoder_hash
from .pose_indexed_development import (
    IndexedPoseDevelopment, _prepare_row, _preparation_identity, _roster,
)


def load_kth_indexed_pose_development(
    bundle: str | Path, *, source_manifest: dict[str, Any], protocol: dict[str, Any],
    feature_spec: PoseFeatureSpec, classes: tuple[str, ...],
    expected_manifest_sha256: str, expected_encoder_hash: str,
) -> IndexedPoseDevelopment:
    """Verify KTH source + aggregate pose bytes and return the existing strict indexed type."""
    if not isinstance(bundle, (str, Path)):
        raise ValueError("KTH pose bundle must be a file path")
    if type(feature_spec) is not PoseFeatureSpec:
        raise ValueError("KTH indexed binding requires explicit PoseFeatureSpec")
    _strings(classes, "classes")
    manifest_pin = _digest(expected_manifest_sha256, "KTH aggregate manifest SHA-256")
    encoder_pin = _digest(expected_encoder_hash, "KTH pose encoder SHA-256")
    if pose_encoder_hash(feature_spec) != encoder_pin:
        raise ValueError("KTH encoder hash differs from independent pin")
    identity, runtime = _preparation_identity(), _runtime()
    indexed = index_kth_development_bundle(
        bundle,
        source_manifest=source_manifest,
        protocol=protocol,
        expected_manifest_sha256=manifest_pin,
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
        indexed, feature_spec, classes, binding_json,
        hashlib.sha256(binding_json.encode("utf-8")).hexdigest(),
    )
    result.verify(expected_binding_sha256=result.binding_sha256)
    return result
