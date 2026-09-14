"""Index aggregated KTH pose evidence without revisiting source RGB bytes."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Any

from .kth_extract import _spec_from_protocol, kth_extractor_code_hash
from .kth_source import (
    _ACTIONS,
    _CORRUPT_BOXING_ARCHIVE_SHA256,
    _CORRUPT_BOXING_MEMBER_SHA256,
    _CORRUPT_BOXING_VIDEO_KEY,
    _MISSING_VIDEO_KEY,
    development_subsequences,
    expected_source_exclusions,
)
from .ntu_io import _canonical_json, _hash_json
from .pose_bundle import _bundle_state, _digest, _frame, _json, _row
from .pose_extract import _schema
from .pose_index import ByteSpan, IndexedPoseBundle, PoseSampleIndex, _DigestPair


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"KTH pose index {context} mismatch")


def verify_kth_source(source: dict[str, Any]) -> dict[str, Any]:
    if type(source) is not dict or source.get("kind") != "kth_rgb_source_manifest":
        raise ValueError("KTH source manifest kind invalid")
    if source.get("format_version") != 1 or source.get("classes") != list(_ACTIONS):
        raise ValueError("KTH source manifest class/version invalid")
    videos = source.get("videos")
    missing_videos = source.get("missing_videos")
    subsequences = source.get("subsequences")
    source_exclusions = source.get("source_exclusions")
    expected_missing = [{
        "video_key": _MISSING_VIDEO_KEY,
        "filename": _MISSING_VIDEO_KEY + "_uncomp.avi",
        "action": "handclapping", "subject": 13, "scenario": 3, "split": "train",
    }]
    if (type(videos) is not list or len(videos) != 599
            or missing_videos != expected_missing
            or type(subsequences) is not list or len(subsequences) != 2391
            or type(source_exclusions) is not list):
        raise ValueError("KTH source manifest present/missing/subsequence cardinality invalid")
    if _MISSING_VIDEO_KEY in {row.get("video_key") for row in videos}:
        raise ValueError("KTH missing parent video cannot appear in present byte roster")
    archives = source.get("archives")
    if type(archives) is not list or len(archives) != len(_ACTIONS):
        raise ValueError("KTH source manifest archive roster invalid")
    boxing_archive = next((row for row in archives if row.get("action") == "boxing"), None)
    boxing_video = next((row for row in videos if row.get("video_key") == _CORRUPT_BOXING_VIDEO_KEY), None)
    exact_corrupt_bytes = bool(
        boxing_archive is not None and boxing_video is not None
        and boxing_archive.get("sha256") == _CORRUPT_BOXING_ARCHIVE_SHA256
        and boxing_video.get("sha256") == _CORRUPT_BOXING_MEMBER_SHA256
    )
    expected_exclusions = expected_source_exclusions() if exact_corrupt_bytes else []
    if source_exclusions != expected_exclusions:
        raise ValueError("KTH source exclusion differs from exact official byte identity")
    dataset_hash = _hash_json({
        "sequence_file_sha256": source.get("sequence_file_sha256"),
        "videos": [{key: row[key] for key in ("video_key", "size_bytes", "sha256")} for row in videos],
        "missing_videos": missing_videos,
    })
    _same(source.get("dataset_content_hash"), dataset_hash, "dataset_content_hash")
    split_hash = _hash_json({
        "dataset_content_hash": dataset_hash, "classes": list(_ACTIONS),
        "policy": source.get("split_policy"),
        "assignments": [{"sample_id": row["sample_id"], "split": row["split"]} for row in subsequences],
        "source_exclusions": source_exclusions,
    })
    _same(source.get("split_hash"), split_hash, "split_hash")
    base = {key: value for key, value in source.items()
            if key not in {"source_manifest_hash", "input_inventory_hash"}}
    _same(source.get("source_manifest_hash"), _hash_json(base), "source_manifest_hash")
    inventory_hash = _hash_json({
        "sequence_file_sha256": source.get("sequence_file_sha256"),
        "videos": videos, "missing_videos": missing_videos, "subsequences": subsequences,
        "source_exclusions": source_exclusions,
    })
    _same(source.get("input_inventory_hash"), inventory_hash, "input_inventory_hash")
    return source


def _verify_protocol(protocol: dict[str, Any], source: dict[str, Any]) -> None:
    if type(protocol) is not dict or protocol.get("protocol_id") != "v1-kth-yolo11n-real-ci":
        raise ValueError("KTH frozen protocol identity invalid")
    expected = {
        "dataset_content_hash": source["dataset_content_hash"],
        "split_hash": source["split_hash"],
        "input_inventory_hash": source["input_inventory_hash"],
        "source_manifest_hash": source["source_manifest_hash"],
        "sequence_file_sha256": source["sequence_file_sha256"],
        "archive_sha256": {row["action"]: row["sha256"] for row in source["archives"]},
        "task_labels": list(_ACTIONS),
        "final_test_decoded": False,
        "classifier_evaluated": False,
    }
    for name, value in expected.items():
        _same(protocol.get(name), value, f"protocol {name}")


def _validate_manifest(report: dict[str, Any], source: dict[str, Any], protocol: dict[str, Any], spec) -> list[dict[str, Any]]:
    expected = {
        "format_version": 1, "kind": "kth_yolo11_pose_development",
        "evidence_scope": "development_extraction_only",
        "source_provenance": "official_kth_archives_verified",
        "final_test_decoded": False, "classifier_evaluated": False,
        "dataset_content_hash": source["dataset_content_hash"],
        "split_hash": source["split_hash"], "input_inventory_hash": source["input_inventory_hash"],
        "source_manifest_hash": source["source_manifest_hash"],
        "sequence_file_sha256": source["sequence_file_sha256"],
        "archive_sha256": protocol["archive_sha256"],
        "versions": spec.expected_versions(), "extractor_code_hash": kth_extractor_code_hash(),
        "extraction_spec": spec.descriptor(), "extraction_spec_hash": protocol["extraction_spec_hash"],
        "observation_schema": _schema(), "observation_schema_hash": protocol["observation_schema_hash"],
        "source_exclusions": source["source_exclusions"],
    }
    for name, value in expected.items():
        _same(report.get(name), value, f"manifest {name}")
    for name in ("python_version", "platform"):
        if type(report.get(name)) is not str or not report[name].strip():
            raise ValueError(f"KTH manifest {name} must be nonempty")
    outputs = report.get("output_sha256")
    if type(outputs) is not dict or set(outputs) != {"observations.jsonl", "timing.jsonl"}:
        raise ValueError("KTH manifest output hashes malformed")
    for value in outputs.values():
        _digest(value, "KTH pose output SHA-256")
    expected_samples = development_subsequences(source["subsequences"], source["source_exclusions"])
    samples = report.get("samples")
    if type(samples) is not list or len(samples) != len(expected_samples):
        raise ValueError("KTH manifest sample roster incomplete")
    videos = {row["video_key"]: row for row in source["videos"]}
    for expected_sample, actual in zip(expected_samples, samples, strict=True):
        for name in ("sample_id", "video_key", "label", "subject", "scenario", "split",
                     "start_frame", "end_frame"):
            _same(actual.get(name), expected_sample[name], f"sample {expected_sample['sample_id']} {name}")
        _same(actual.get("source_sha256"), videos[expected_sample["video_key"]]["sha256"],
              f"sample {expected_sample['sample_id']} source_sha256")
        for name in ("frame_count", "height", "width"):
            if type(actual.get(name)) is not int or actual[name] <= 0:
                raise ValueError("KTH sample dimension/frame count invalid")
        missing = actual.get("missing_person_frames")
        if type(missing) is not int or not 0 <= missing <= actual["frame_count"]:
            raise ValueError("KTH sample missing-person count invalid")
    return expected_samples


def index_kth_development_bundle(bundle: str | Path, *, source_manifest: dict[str, Any],
                                 protocol: dict[str, Any], expected_manifest_sha256: str) -> IndexedPoseBundle:
    pin = _digest(expected_manifest_sha256, "KTH aggregate manifest SHA-256")
    source = verify_kth_source(source_manifest)
    _verify_protocol(protocol, source)
    spec = _spec_from_protocol(protocol)
    directory = Path(bundle).absolute()
    before = _bundle_state(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pin:
        raise ValueError("KTH aggregate manifest bytes differ from independent pin")
    report = _json(manifest_raw)
    if type(report) is not dict:
        raise ValueError("KTH aggregate manifest must be JSON object")
    selected = _validate_manifest(report, source, protocol, spec)

    entries: list[PoseSampleIndex] = []
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    with (directory / "observations.jsonl").open("rb") as geometry, \
            (directory / "timing.jsonl").open("rb") as timing:
        for source_row, sample in zip(selected, report["samples"], strict=True):
            geometry_start, timing_start = geometry.tell(), timing.tell()
            sample_geometry, sample_timing = hashlib.sha256(), hashlib.sha256()
            geometry_pair = _DigestPair(geometry_digest, sample_geometry)
            timing_pair = _DigestPair(timing_digest, sample_timing)
            previous, missing = None, 0
            for frame_index in range(sample["frame_count"]):
                frame, pts, base, score = _frame(
                    _row(geometry, geometry_pair, "KTH geometry"),
                    _row(timing, timing_pair, "KTH timing"),
                    {"sample_id": source_row["sample_id"], "label": source_row["label"]},
                    frame_index, previous,
                )
                previous = pts * base
                missing += score == 0.0
                del frame
            if missing != sample["missing_person_frames"]:
                raise ValueError("KTH sample missing-person count differs from manifest")
            entries.append(PoseSampleIndex(
                sample_id=source_row["sample_id"], label=source_row["label"],
                subject=source_row["subject"], split=source_row["split"],
                width=sample["width"], height=sample["height"], frame_count=sample["frame_count"],
                missing_person_frames=missing, source_sha256=sample["source_sha256"],
                geometry=ByteSpan(geometry_start, geometry.tell(), sample_geometry.hexdigest()),
                timing=ByteSpan(timing_start, timing.tell(), sample_timing.hexdigest()),
            ))
        if geometry.read(1) or timing.read(1):
            raise ValueError("KTH aggregate sidecar contains extra rows/trailing bytes")
    for name, digest in (("observations.jsonl", geometry_digest), ("timing.jsonl", timing_digest)):
        if digest.hexdigest() != report["output_sha256"][name]:
            raise ValueError(f"KTH {name} hash differs from manifest")
    if _bundle_state(directory) != before:
        raise ValueError("KTH aggregate bundle changed during verification")
    indexed = IndexedPoseBundle(
        directory=directory, samples=tuple(entries), manifest_sha256=pin,
        dataset_content_hash=source["dataset_content_hash"], split_hash=source["split_hash"],
        input_inventory_hash=source["input_inventory_hash"],
        extraction_spec_hash=report["extraction_spec_hash"],
        observation_schema_hash=report["observation_schema_hash"],
        extractor_code_hash=report["extractor_code_hash"],
        output_sha256=tuple(sorted(report["output_sha256"].items())), index_sha256="",
        _file_state=tuple(sorted(before.items())),
    )
    return replace(indexed, index_sha256=_hash_json(indexed.descriptor()))
