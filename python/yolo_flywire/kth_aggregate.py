"""Aggregate six verified KTH action shards into one development pose bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .kth_extract import _spec_from_protocol, kth_extractor_code_hash
from .kth_source import (
    _ACTIONS, _MISSING_VIDEO_KEY, _canonical_json, _hash_json,
    development_subsequences, expected_source_exclusions, parse_sequence_file,
)
from .ntu_io import _hash_file, _publish_exclusive
from .pose_extract import _schema


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"KTH aggregation {context} mismatch")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name}") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_sidecar(sample_dir: Path, record: dict[str, Any]) -> None:
    if sample_dir.is_symlink() or not sample_dir.is_dir():
        raise ValueError("KTH sample sidecar directory missing")
    files = {path.name for path in sample_dir.iterdir()}
    if files != {"observations.jsonl", "timing.jsonl"}:
        raise ValueError("KTH sample sidecar directory has unexpected files")
    hashes = record.get("output_sha256")
    if type(hashes) is not dict or set(hashes) != files:
        raise ValueError("KTH sample output hash record malformed")
    for name in sorted(files):
        path = sample_dir / name
        if path.is_symlink() or not path.is_file() or _hash_file(path)[1] != hashes[name]:
            raise ValueError(f"KTH sidecar hash mismatch: {record.get('sample_id')} {name}")


def _source_record(plan, sequence_sha: str, archive_records: list[dict[str, Any]],
                   video_records: list[dict[str, Any]], missing_records: list[dict[str, Any]],
                   exclusion_records: list[dict[str, Any]]) -> dict[str, Any]:
    videos = sorted(video_records, key=lambda row: (_ACTIONS.index(row["action"]), row["subject"], row["scenario"]))
    if len(videos) != 599:
        raise ValueError("KTH source evidence must cover exactly 599 present parent videos")
    missing = sorted(missing_records, key=lambda row: row["video_key"])
    expected_missing = [{
        "video_key": _MISSING_VIDEO_KEY,
        "filename": _MISSING_VIDEO_KEY + "_uncomp.avi",
        "action": "handclapping", "subject": 13, "scenario": 3, "split": "train",
    }]
    if missing != expected_missing:
        raise ValueError("KTH source evidence missing-video record changed")
    exclusions = json.loads(_canonical_json(exclusion_records))
    if _canonical_json(exclusions) != _canonical_json(expected_source_exclusions()):
        raise ValueError("KTH source exclusion record changed")
    archives = sorted(archive_records, key=lambda row: _ACTIONS.index(row["action"]))
    exclusion = exclusions[0]
    boxing_archive = next(row for row in archives if row["action"] == "boxing")
    corrupt_parent = next(row for row in videos if row["video_key"] == exclusion["video_key"])
    if (boxing_archive["sha256"] != exclusion["archive_sha256"]
            or corrupt_parent["sha256"] != exclusion["member_sha256"]):
        raise ValueError("KTH source exclusion is not bound to exact official bytes")
    subsequences = [dict(row) for row in plan.subsequences]
    dataset_content_hash = _hash_json({
        "sequence_file_sha256": sequence_sha,
        "videos": [{key: row[key] for key in ("video_key", "size_bytes", "sha256")} for row in videos],
        "missing_videos": missing,
    })
    split_hash = _hash_json({
        "dataset_content_hash": dataset_content_hash, "classes": list(_ACTIONS),
        "policy": plan.split_policy,
        "assignments": [{"sample_id": row["sample_id"], "split": row["split"]} for row in subsequences],
        "source_exclusions": exclusions,
    })
    record = {
        "format_version": 1, "kind": "kth_rgb_source_manifest",
        "evidence_scope": "real_source_bytes_only_no_final_test_decode",
        "official_source": "https://www.csc.kth.se/cvap/actions/",
        "classes": list(_ACTIONS), "split_policy": plan.split_policy,
        "sequence_file_sha256": sequence_sha,
        "archives": archives,
        "videos": videos, "missing_videos": missing, "subsequences": subsequences,
        "source_exclusions": exclusions,
        "dataset_content_hash": dataset_content_hash, "split_hash": split_hash,
    }
    record["source_manifest_hash"] = _hash_json(record)
    record["input_inventory_hash"] = _hash_json({
        "sequence_file_sha256": sequence_sha, "videos": videos,
        "missing_videos": missing, "subsequences": subsequences,
        "source_exclusions": exclusions,
    })
    return json.loads(_canonical_json(record))


def aggregate_kth_shards(sequence_file: str | Path, *, protocol: dict[str, Any],
                         shard_paths: tuple[str | Path, ...], output: str | Path) -> dict[str, Any]:
    sequence_path = Path(sequence_file).absolute()
    raw_sequence = sequence_path.read_bytes()
    sequence_sha = hashlib.sha256(raw_sequence).hexdigest()
    if sequence_sha != protocol.get("sequence_file_sha256"):
        raise ValueError("KTH sequence file differs from frozen coordinator bytes")
    plan = parse_sequence_file(raw_sequence.decode("utf-8"))
    spec = _spec_from_protocol(protocol)
    if type(shard_paths) is not tuple or not shard_paths:
        raise ValueError("KTH aggregation requires explicit shard paths")

    reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    all_videos: dict[str, dict[str, Any]] = {}
    all_missing: dict[str, dict[str, Any]] = {}
    all_samples: dict[str, tuple[dict[str, Any], Path]] = {}
    all_exclusions: list[dict[str, Any]] = []
    archive_records: list[dict[str, Any]] = []
    python_version = platform_name = None
    usable_development = development_subsequences(plan.subsequences, expected_source_exclusions())
    for raw_path in shard_paths:
        path = Path(raw_path).absolute()
        report_path = path / "shard-report.json"
        if path.is_symlink() or not report_path.is_file() or report_path.is_symlink():
            raise ValueError("KTH shard/report path invalid")
        report = _read_json(report_path, "KTH shard report")
        action = report.get("action")
        if action not in _ACTIONS or action in reports:
            raise ValueError("KTH shards must contain each action exactly once")
        expected_common = {
            "format_version": 1,
            "kind": "kth_yolo11_pose_action_shard",
            "evidence_scope": "kth_public_real_action_extraction_only",
            "final_test_decoded": False, "classifier_evaluated": False,
            "sequence_file_sha256": sequence_sha,
            "versions": spec.expected_versions(),
            "extractor_code_hash": kth_extractor_code_hash(),
            "extraction_spec": spec.descriptor(),
            "extraction_spec_hash": protocol["extraction_spec_hash"],
            "observation_schema": _schema(),
            "observation_schema_hash": protocol["observation_schema_hash"],
        }
        for name, value in expected_common.items():
            _same(report.get(name), value, f"shard {action} {name}")
        expected_exclusions = expected_source_exclusions() if action == "boxing" else []
        _same(report.get("source_exclusions"), expected_exclusions, f"shard {action} source_exclusions")
        all_exclusions.extend(expected_exclusions)
        if python_version is None:
            python_version, platform_name = report.get("python_version"), report.get("platform")
        if report.get("python_version") != python_version or report.get("platform") != platform_name:
            raise ValueError("KTH shards differ in Python/platform evidence")
        archive = report.get("archive")
        if type(archive) is not dict or set(archive) != {"url", "size_bytes", "sha256"}:
            raise ValueError("KTH shard archive evidence malformed")
        if type(archive["size_bytes"]) is not int or archive["size_bytes"] <= 0:
            raise ValueError("KTH archive size evidence invalid")
        if type(archive["sha256"]) is not str or len(archive["sha256"]) != 64:
            raise ValueError("KTH archive SHA evidence invalid")
        archive_records.append({"action": action, **archive})

        logical_videos = [row for row in plan.videos if row["action"] == action]
        expected_videos = [row for row in logical_videos if not row["missing"]]
        expected_missing = [
            {key: row[key] for key in ("video_key", "filename", "action", "subject", "scenario", "split")}
            for row in logical_videos if row["missing"]
        ]
        videos = report.get("videos")
        if type(videos) is not list or len(videos) != len(expected_videos):
            raise ValueError(f"KTH {action} shard present-video count mismatch")
        _same(report.get("missing_videos"), expected_missing, f"shard {action} missing_videos")
        for missing in expected_missing:
            if missing["video_key"] in all_missing:
                raise ValueError("duplicate KTH missing-video evidence")
            all_missing[missing["video_key"]] = missing
        for expected, actual in zip(expected_videos, videos, strict=True):
            for name in ("video_key", "filename", "action", "subject", "scenario", "split"):
                _same(actual.get(name), expected[name], f"video {expected['video_key']} {name}")
            if actual.get("verified_bytes") is not True:
                raise ValueError("KTH parent video lacks byte verification")
            decoded = actual.get("decoded")
            if decoded is not (expected["split"] != "final_test"):
                raise ValueError("KTH final-test/development decode boundary changed")
            if type(actual.get("size_bytes")) is not int or actual["size_bytes"] <= 0:
                raise ValueError("KTH parent video size invalid")
            digest = actual.get("sha256")
            if type(digest) is not str or len(digest) != 64:
                raise ValueError("KTH parent video SHA invalid")
            if expected["video_key"] in all_videos:
                raise ValueError("duplicate KTH parent video evidence")
            all_videos[expected["video_key"]] = {
                key: actual[key] for key in (
                    "video_key", "filename", "action", "subject", "scenario", "split", "size_bytes", "sha256"
                )
            }

        samples = report.get("samples")
        expected_samples = [row for row in usable_development if row["label"] == action]
        if type(samples) is not list or len(samples) != len(expected_samples):
            raise ValueError("KTH action shard development sample count mismatch")
        for expected, actual in zip(expected_samples, samples, strict=True):
            for name in ("sample_id", "video_key", "split", "label", "subject", "scenario",
                         "range_index", "start_frame", "end_frame"):
                _same(actual.get(name), expected[name], f"sample {expected['sample_id']} {name}")
            for name in ("frame_count", "height", "width"):
                if type(actual.get(name)) is not int or actual[name] <= 0:
                    raise ValueError("KTH sample geometry metadata invalid")
            missing = actual.get("missing_person_frames")
            if type(missing) is not int or not 0 <= missing <= actual["frame_count"]:
                raise ValueError("KTH sample missing-person count invalid")
            sample_dir = path / "samples" / expected["sample_id"]
            _verify_sidecar(sample_dir, actual)
            if expected["sample_id"] in all_samples:
                raise ValueError("duplicate KTH subsequence pose evidence")
            all_samples[expected["sample_id"]] = (actual, sample_dir)
        expected_dirs = {row["sample_id"] for row in expected_samples}
        root = path / "samples"
        actual_dirs = {entry.name for entry in root.iterdir()} if root.exists() else set()
        if actual_dirs != expected_dirs:
            raise ValueError("KTH action shard contains unexpected/missing pose directories")
        reports[action] = (path, report)

    expected_present_keys = {row["video_key"] for row in plan.videos if not row["missing"]}
    expected_missing_keys = {row["video_key"] for row in plan.videos if row["missing"]}
    if (set(reports) != set(_ACTIONS) or set(all_videos) != expected_present_keys
            or set(all_missing) != expected_missing_keys or len(all_videos) != 599):
        raise ValueError("KTH aggregation requires six actions, 599 present videos and one missing slot")
    if _canonical_json(all_exclusions) != _canonical_json(expected_source_exclusions()):
        raise ValueError("KTH aggregation requires exact corrupt-source exclusion evidence")
    expected_development = development_subsequences(plan.subsequences, all_exclusions)
    if set(all_samples) != {row["sample_id"] for row in expected_development}:
        raise ValueError("KTH aggregation requires complete usable development subsequence evidence")

    source = _source_record(
        plan, sequence_sha, archive_records, list(all_videos.values()), list(all_missing.values()),
        all_exclusions,
    )
    frozen = json.loads(_canonical_json(protocol))
    for name in ("dataset_content_hash", "input_inventory_hash", "source_manifest_hash", "split_hash"):
        if frozen.get(name) is not None:
            raise ValueError(f"KTH runtime-frozen protocol field {name} must remain null before aggregation")
        frozen[name] = source[name]
    if frozen.get("archive_sha256") is not None:
        raise ValueError("KTH archive SHA field must be null before aggregation")
    frozen["archive_sha256"] = {row["action"]: row["sha256"] for row in source["archives"]}
    frozen["evidence_scope"] = "kth_public_real_ci_source_and_runtime_frozen"

    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH aggregate output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    geometry_path, timing_path = destination / "observations.jsonl", destination / "timing.jsonl"
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    sample_reports: list[dict[str, Any]] = []
    try:
        with geometry_path.open("xb") as geometry_out, timing_path.open("xb") as timing_out:
            for expected in expected_development:
                actual, sample_dir = all_samples[expected["sample_id"]]
                _verify_sidecar(sample_dir, actual)
                parent = all_videos[expected["video_key"]]
                sample_reports.append({
                    "sample_id": expected["sample_id"], "video_key": expected["video_key"],
                    "label": expected["label"], "subject": expected["subject"],
                    "scenario": expected["scenario"], "split": expected["split"],
                    "source_sha256": parent["sha256"], "start_frame": expected["start_frame"],
                    "end_frame": expected["end_frame"], "frame_count": actual["frame_count"],
                    "missing_person_frames": actual["missing_person_frames"],
                    "height": actual["height"], "width": actual["width"],
                })
                for name, target, digest in (
                    ("observations.jsonl", geometry_out, geometry_digest),
                    ("timing.jsonl", timing_out, timing_digest),
                ):
                    with (sample_dir / name).open("rb") as handle:
                        while chunk := handle.read(1024 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
            for handle in (geometry_out, timing_out):
                handle.flush()
                os.fsync(handle.fileno())
        manifest = {
            "format_version": 1, "kind": "kth_yolo11_pose_development",
            "evidence_scope": "development_extraction_only",
            "source_provenance": "official_kth_archives_verified",
            "final_test_decoded": False, "classifier_evaluated": False,
            "dataset_content_hash": source["dataset_content_hash"],
            "split_hash": source["split_hash"], "input_inventory_hash": source["input_inventory_hash"],
            "source_manifest_hash": source["source_manifest_hash"],
            "source_exclusions": source["source_exclusions"],
            "sequence_file_sha256": sequence_sha,
            "archive_sha256": frozen["archive_sha256"],
            "versions": spec.expected_versions(), "python_version": python_version,
            "platform": platform_name, "extractor_code_hash": kth_extractor_code_hash(),
            "extraction_spec": spec.descriptor(), "extraction_spec_hash": frozen["extraction_spec_hash"],
            "observation_schema": _schema(), "observation_schema_hash": frozen["observation_schema_hash"],
            "samples": sample_reports,
            "output_sha256": {
                "observations.jsonl": geometry_digest.hexdigest(),
                "timing.jsonl": timing_digest.hexdigest(),
            },
        }
        _publish_exclusive(destination / "manifest.json", manifest)
        _publish_exclusive(destination / "source-manifest.json", source)
        _publish_exclusive(destination / "frozen-protocol.json", frozen)
        return json.loads(_canonical_json(manifest))
    except BaseException:
        geometry_path.unlink(missing_ok=True)
        timing_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-file", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        shard_paths = tuple(sorted({path.parent for path in args.shards_root.rglob("shard-report.json")},
                                   key=lambda path: path.as_posix()))
        report = aggregate_kth_shards(
            args.sequence_file, protocol=_read_json(args.protocol, "KTH protocol"),
            shard_paths=shard_paths, output=args.output,
        )
        print(_canonical_json({"samples": len(report["samples"]), "final_test_decoded": False,
                               "dataset_content_hash": report["dataset_content_hash"]}))
        return 0
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        parser.exit(1, f"KTH aggregation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
