"""Index hosted remote pose bundles without requiring deleted NTU RGB files."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from .hosted_extract import hosted_extractor_code_hash, _spec_from_protocol, _verify_protocol_source
from .ntu_io import _canonical_json, _hash_json
from .pose_bundle import _bundle_state, _digest, _frame, _json, _row
from .pose_extract import _schema
from .pose_index import ByteSpan, IndexedPoseBundle, PoseSampleIndex, _DigestPair
from .remote_rgb import validate_remote_manifest


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"remote pose index {context} mismatch")


def _validate_hosted_report(report: dict[str, Any], remote, protocol, spec) -> list[dict[str, Any]]:
    expected = {
        "format_version": 1,
        "kind": "ntu120_yolo11_pose_remote_development",
        "evidence_scope": "development_extraction_only",
        "source_provenance": "remote_manifest_verified",
        "final_test_decoded": False,
        "classifier_evaluated": False,
        "dataset_content_hash": remote.dataset_content_hash,
        "split_hash": remote.split_hash,
        "input_inventory_hash": remote.input_inventory_hash,
        "source_manifest_hash": remote.source_manifest_hash,
        "transport_manifest_hash": remote.transport_manifest_hash,
        "versions": spec.expected_versions(),
        "extractor_code_hash": hosted_extractor_code_hash(),
        "extraction_spec": spec.descriptor(),
        "extraction_spec_hash": protocol["extraction_spec_hash"],
        "observation_schema": _schema(),
        "observation_schema_hash": protocol["observation_schema_hash"],
    }
    for name, value in expected.items():
        _same(report.get(name), value, f"manifest {name}")
    if type(report.get("shard_count")) is not int or report["shard_count"] <= 0:
        raise ValueError("remote pose manifest shard_count must be positive integer")
    for name in ("python_version", "platform"):
        if type(report.get(name)) is not str or not report[name].strip():
            raise ValueError(f"remote pose manifest {name} must be a nonempty string")
    outputs = report.get("output_sha256")
    if type(outputs) is not dict or set(outputs) != {"observations.jsonl", "timing.jsonl"}:
        raise ValueError("remote pose manifest output hashes malformed")
    for value in outputs.values():
        _digest(value, "remote pose output SHA-256")
    selected = [row for row in remote.inventory["samples"] if row["split"] in ("train", "validation")]
    samples = report.get("samples")
    if type(samples) is not list or len(samples) != len(selected) or not selected:
        raise ValueError("remote pose manifest must match complete development roster")
    for source, sample in zip(selected, samples, strict=True):
        if type(sample) is not dict or set(sample) != {
            "sample_id", "split", "source_sha256", "frame_count",
            "missing_person_frames", "height", "width",
        }:
            raise ValueError("remote pose sample report fields malformed")
        for name, value in (
            ("sample_id", source["sample_id"]),
            ("split", source["split"]),
            ("source_sha256", source["sha256"]),
        ):
            _same(sample[name], value, f"sample {source['sample_id']} {name}")
        for name in ("frame_count", "height", "width"):
            if type(sample[name]) is not int or sample[name] <= 0:
                raise ValueError(f"remote pose sample {name} must be positive integer")
        missing = sample["missing_person_frames"]
        if type(missing) is not int or not 0 <= missing <= sample["frame_count"]:
            raise ValueError("remote pose missing-person count invalid")
    return selected


def index_remote_development_bundle(
    bundle: str | Path, *, remote_manifest: dict[str, Any], protocol: dict[str, Any],
    expected_manifest_sha256: str,
) -> IndexedPoseBundle:
    """Verify remote source identity + all pose rows, never revisit original RGB bytes."""
    pin = _digest(expected_manifest_sha256, "remote pose manifest SHA-256")
    remote = validate_remote_manifest(remote_manifest)
    _verify_protocol_source(protocol, remote)
    spec = _spec_from_protocol(protocol)
    directory = Path(bundle).absolute()
    before = _bundle_state(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pin:
        raise ValueError("remote pose manifest SHA-256 does not match independent pin")
    report = _json(manifest_raw)
    if type(report) is not dict:
        raise ValueError("remote pose manifest must be a JSON object")
    selected = _validate_hosted_report(report, remote, protocol, spec)

    entries = []
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    with (directory / "observations.jsonl").open("rb") as geometry, \
            (directory / "timing.jsonl").open("rb") as timing:
        for source, sample in zip(selected, report["samples"], strict=True):
            geometry_start, timing_start = geometry.tell(), timing.tell()
            sample_geometry, sample_timing = hashlib.sha256(), hashlib.sha256()
            geometry_pair = _DigestPair(geometry_digest, sample_geometry)
            timing_pair = _DigestPair(timing_digest, sample_timing)
            previous, missing = None, 0
            for frame_index in range(sample["frame_count"]):
                frame, pts, base, score = _frame(
                    _row(geometry, geometry_pair, "remote geometry"),
                    _row(timing, timing_pair, "remote timing"),
                    source, frame_index, previous,
                )
                previous = pts * base
                missing += score == 0.0
                del frame
            if missing != sample["missing_person_frames"]:
                raise ValueError("remote pose missing-person count differs from manifest")
            entries.append(PoseSampleIndex(
                sample_id=source["sample_id"], label=source["label"], subject=source["subject"],
                split=source["split"], width=sample["width"], height=sample["height"],
                frame_count=sample["frame_count"], missing_person_frames=missing,
                source_sha256=source["sha256"],
                geometry=ByteSpan(geometry_start, geometry.tell(), sample_geometry.hexdigest()),
                timing=ByteSpan(timing_start, timing.tell(), sample_timing.hexdigest()),
            ))
        if geometry.read(1) or timing.read(1):
            raise ValueError("remote pose sidecar contains extra rows or trailing bytes")
    for name, digest in (("observations.jsonl", geometry_digest), ("timing.jsonl", timing_digest)):
        if digest.hexdigest() != report["output_sha256"][name]:
            raise ValueError(f"remote {name} bytes do not match manifest SHA-256")
    if _bundle_state(directory) != before:
        raise ValueError("remote pose bundle changed during verification")
    indexed = IndexedPoseBundle(
        directory=directory,
        samples=tuple(entries),
        manifest_sha256=pin,
        dataset_content_hash=remote.dataset_content_hash,
        split_hash=remote.split_hash,
        input_inventory_hash=remote.input_inventory_hash,
        extraction_spec_hash=report["extraction_spec_hash"],
        observation_schema_hash=report["observation_schema_hash"],
        extractor_code_hash=report["extractor_code_hash"],
        output_sha256=tuple(sorted(report["output_sha256"].items())),
        index_sha256="",
        _file_state=tuple(sorted(before.items())),
    )
    return replace(indexed, index_sha256=_hash_json(indexed.descriptor()))
