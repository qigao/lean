"""Deterministically aggregate verified hosted NTU pose shards into one development bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from .hosted_extract import hosted_extractor_code_hash, _spec_from_protocol, _verify_protocol_source
from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive
from .pose_extract import _schema
from .remote_rgb import shard_rows, validate_remote_manifest


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"hosted aggregation {context} mismatch")


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be a JSON object")
    return value


def _verify_shard_common(report: dict[str, Any], *, protocol: dict[str, Any], remote, spec) -> None:
    expected = {
        "format_version": 1,
        "kind": "ntu120_yolo11_pose_remote_shard",
        "evidence_scope": "hosted_shard_extraction_only",
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
    if type(report) is not dict:
        raise ValueError("hosted shard report must be a JSON object")
    for name, value in expected.items():
        _same(report.get(name), value, f"shard {name}")
    for name in ("python_version", "platform"):
        if type(report.get(name)) is not str or not report[name].strip():
            raise ValueError(f"hosted shard {name} must be a nonempty string")
    if type(report.get("shard_index")) is not int or type(report.get("shard_count")) is not int:
        raise ValueError("hosted shard index/count must be integers")
    if type(report.get("samples")) is not list:
        raise ValueError("hosted shard samples must be a JSON array")


def _verify_dev_sample(source: dict[str, Any], record: dict[str, Any], sample_dir: Path) -> dict[str, Any]:
    expected = {
        "sample_id": source["sample_id"],
        "split": source["split"],
        "source_size_bytes": source["size_bytes"],
        "source_sha256": source["sha256"],
        "verified_bytes": True,
        "decoded": True,
    }
    for name, value in expected.items():
        _same(record.get(name), value, f"sample {source['sample_id']} {name}")
    for name in ("frame_count", "height", "width"):
        if type(record.get(name)) is not int or record[name] <= 0:
            raise ValueError(f"hosted sample {source['sample_id']} {name} must be positive integer")
    missing = record.get("missing_person_frames")
    if type(missing) is not int or not 0 <= missing <= record["frame_count"]:
        raise ValueError("hosted sample missing-person count invalid")
    if sample_dir.is_symlink() or not sample_dir.is_dir():
        raise ValueError(f"hosted sample sidecar directory missing: {source['sample_id']}")
    files = {path.name for path in sample_dir.iterdir()}
    if files != {"observations.jsonl", "timing.jsonl"}:
        raise ValueError("hosted sample sidecar directory contains unexpected files")
    hashes = record.get("output_sha256")
    if type(hashes) is not dict or set(hashes) != files:
        raise ValueError("hosted sample output hash record malformed")
    for name in sorted(files):
        path = sample_dir / name
        if path.is_symlink() or not path.is_file() or _hash_file(path)[1] != hashes[name]:
            raise ValueError(f"hosted sample {source['sample_id']} {name} hash mismatch")
    return {
        "sample_id": source["sample_id"],
        "split": source["split"],
        "source_sha256": source["sha256"],
        "frame_count": record["frame_count"],
        "missing_person_frames": missing,
        "height": record["height"],
        "width": record["width"],
    }


def aggregate_remote_shards(
    remote_manifest: dict[str, Any], *, protocol: dict[str, Any],
    shard_paths: tuple[str | Path, ...], output: str | Path,
) -> dict[str, Any]:
    """Verify complete shard coverage and emit an ordered development pose bundle."""
    remote = validate_remote_manifest(remote_manifest)
    _verify_protocol_source(protocol, remote)
    spec = _spec_from_protocol(protocol)
    if type(shard_paths) is not tuple or not shard_paths:
        raise ValueError("explicit nonempty tuple of shard paths required")

    reports: dict[int, tuple[Path, dict[str, Any]]] = {}
    shard_count: int | None = None
    all_records: dict[str, tuple[dict[str, Any], Path | None]] = {}
    python_version = platform_name = None
    for raw_path in shard_paths:
        path = Path(raw_path).absolute()
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ValueError("hosted shard path must not traverse symlinks")
        report_path = path / "shard-report.json"
        if not report_path.is_file() or report_path.is_symlink():
            raise ValueError("hosted shard-report.json missing")
        report = _read_json(report_path, "hosted shard report")
        _verify_shard_common(report, protocol=protocol, remote=remote, spec=spec)
        index = report["shard_index"]
        count = report["shard_count"]
        if shard_count is None:
            shard_count = count
            python_version, platform_name = report["python_version"], report["platform"]
        if count != shard_count or report["python_version"] != python_version or report["platform"] != platform_name:
            raise ValueError("hosted shards differ in shard count/runtime platform record")
        if index in reports:
            raise ValueError(f"duplicate hosted shard index: {index}")
        expected_rows = shard_rows(remote, index, count)
        expected_ids = [row["sample_id"] for row in expected_rows]
        actual_ids = [row.get("sample_id") for row in report["samples"]]
        if actual_ids != expected_ids:
            raise ValueError("hosted shard sample order/roster differs from frozen assignment")
        reports[index] = (path, report)

        dev_ids: set[str] = set()
        for source, record in zip(expected_rows, report["samples"], strict=True):
            if source["sample_id"] in all_records:
                raise ValueError(f"duplicate hosted sample evidence: {source['sample_id']}")
            if source["split"] == "final_test":
                expected = {
                    "sample_id": source["sample_id"], "split": "final_test",
                    "source_size_bytes": source["size_bytes"], "source_sha256": source["sha256"],
                    "verified_bytes": True, "decoded": False, "output_sha256": None,
                }
                for name, value in expected.items():
                    _same(record.get(name), value, f"final-test {source['sample_id']} {name}")
                final_dir = path / "samples" / source["sample_id"]
                if final_dir.exists() or final_dir.is_symlink():
                    raise ValueError("final-test hosted sample must not have pose sidecars")
                all_records[source["sample_id"]] = (record, None)
            else:
                sample_dir = path / "samples" / source["sample_id"]
                _verify_dev_sample(source, record, sample_dir)
                dev_ids.add(source["sample_id"])
                all_records[source["sample_id"]] = (record, sample_dir)
        samples_root = path / "samples"
        if samples_root.exists():
            if samples_root.is_symlink() or not samples_root.is_dir():
                raise ValueError("hosted shard samples path must be a directory")
            actual_dirs = {entry.name for entry in samples_root.iterdir()}
            if actual_dirs != dev_ids:
                raise ValueError("hosted shard contains unexpected/missing sample sidecar directories")

    assert shard_count is not None
    if set(reports) != set(range(shard_count)):
        raise ValueError("hosted aggregation requires every deterministic shard exactly once")
    if set(all_records) != {row["sample_id"] for row in remote.samples}:
        raise ValueError("hosted aggregation does not cover the complete frozen remote roster")

    destination = Path(output).absolute()
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("hosted aggregate output must not traverse symlinks")
    if destination.exists():
        raise ValueError("hosted aggregate output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    geometry_path = destination / "observations.jsonl"
    timing_path = destination / "timing.jsonl"
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    sample_reports: list[dict[str, Any]] = []
    try:
        with geometry_path.open("xb") as geometry_out, timing_path.open("xb") as timing_out:
            for source in remote.samples:
                if source["split"] == "final_test":
                    continue
                record, sample_dir = all_records[source["sample_id"]]
                assert sample_dir is not None
                sample_reports.append(_verify_dev_sample(source, record, sample_dir))
                for name, target, digest in (
                    ("observations.jsonl", geometry_out, geometry_digest),
                    ("timing.jsonl", timing_out, timing_digest),
                ):
                    with (sample_dir / name).open("rb") as source_handle:
                        while chunk := source_handle.read(1024 * 1024):
                            target.write(chunk)
                            digest.update(chunk)
            for handle in (geometry_out, timing_out):
                handle.flush()
                os.fsync(handle.fileno())
        descriptor = spec.descriptor()
        report = {
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
            "shard_count": shard_count,
            "versions": spec.expected_versions(),
            "python_version": python_version,
            "platform": platform_name,
            "extractor_code_hash": hosted_extractor_code_hash(),
            "extraction_spec": descriptor,
            "extraction_spec_hash": _hash_json(descriptor),
            "observation_schema": _schema(),
            "observation_schema_hash": _hash_json(_schema()),
            "samples": sample_reports,
            "output_sha256": {
                "observations.jsonl": geometry_digest.hexdigest(),
                "timing.jsonl": timing_digest.hexdigest(),
            },
        }
        _publish_exclusive(destination / "manifest.json", report)
        return json.loads(_canonical_json(report))
    except BaseException:
        geometry_path.unlink(missing_ok=True)
        timing_path.unlink(missing_ok=True)
        if destination.exists() and not any(destination.iterdir()):
            destination.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--shards-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        shard_paths = tuple(sorted(
            {path.parent for path in args.shards_root.rglob("shard-report.json")},
            key=lambda path: path.as_posix(),
        ))
        report = aggregate_remote_shards(
            _read_json(args.remote_manifest, "remote manifest"),
            protocol=_read_json(args.protocol, "frozen hosted protocol"),
            shard_paths=shard_paths, output=args.output,
        )
        print(_canonical_json({
            "samples": len(report["samples"]),
            "shard_count": report["shard_count"],
            "final_test_decoded": False,
        }))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Hosted aggregation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
