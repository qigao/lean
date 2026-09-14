"""Read complete, pinned development pose bundles without decoding or inference."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import stat
from typing import Any, BinaryIO

from .ntu_io import _canonical_json, _hash_json, _signature, verify_rgb_manifest
from .pose_extract import ExtractionSpec, _schema
from .schema import KeypointFrame, PointSequence

_FILES = frozenset({"manifest.json", "observations.jsonl", "timing.jsonl"})
_POSE_FIELDS = frozenset({"sample_id", "frame_index", "label", "body_keypoints", "detector_confidence"})
_TIME_FIELDS = frozenset({"sample_id", "frame_index", "pts", "time_base"})
_SAMPLE_FIELDS = frozenset({"sample_id", "split", "source_sha256", "frame_count",
                            "missing_person_frames", "height", "width"})


@dataclass(frozen=True)
class TimedPoseSample:
    """Immutable observations and metadata returned after complete verification.

    Metadata is not a classifier feature. Constructing this dataclass directly
    does not verify provenance; use load_development_bundle at the file boundary.
    """

    sample_id: str
    label: str
    subject: int
    split: str
    width: int
    height: int
    sequence: PointSequence
    pts: tuple[int, ...]
    time_bases: tuple[Fraction, ...]
    detector_confidences: tuple[float, ...]

    @property
    def timestamps(self) -> tuple[Fraction, ...]:
        return tuple(pts * base for pts, base in zip(self.pts, self.time_bases, strict=True))

    @property
    def missing_person(self) -> tuple[bool, ...]:
        return tuple(confidence == 0.0 for confidence in self.detector_confidences)


@dataclass(frozen=True)
class VerifiedPoseBundle:
    """Materialized development observations, not authorization to open a final test."""

    samples: tuple[TimedPoseSample, ...]
    manifest_sha256: str
    dataset_content_hash: str
    split_hash: str
    input_inventory_hash: str
    extraction_spec_hash: str
    observation_schema_hash: str
    extractor_code_hash: str
    output_sha256: tuple[tuple[str, str], ...]


def _object(value: Any, keys: frozenset[str] | set[str], context: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{context}: missing or unknown fields")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number: {value}")


def _json(raw: bytes) -> Any:
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                      parse_constant=_nonfinite_constant)


def _same(value: Any, expected: Any, context: str) -> None:
    # Unlike Python equality, canonical JSON distinguishes bool/int/float.
    if _canonical_json(value) != _canonical_json(expected):
        raise ValueError(f"{context} does not match the frozen contract")


def _digest(value: Any, context: str) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{context} must be an explicit lowercase SHA-256")
    return value


def _integer(value: Any, context: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{context} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def _number(value: Any, context: str, confidence: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a finite number, not a coerced value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context} is outside finite float range") from exc
    if not math.isfinite(result) or (confidence and not 0.0 <= result <= 1.0):
        raise ValueError(f"{context} is non-finite or outside its allowed range")
    return result


def _bundle_state(directory: Path) -> dict[str, tuple[int, ...]]:
    if ".." in directory.parts:
        raise ValueError("bundle path must not contain parent traversal")
    for component in (directory, *directory.parents):
        if component.is_symlink():
            raise ValueError("bundle path must not traverse symlinks")
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != _FILES:
        raise ValueError("bundle must contain exactly manifest.json, observations.jsonl and timing.jsonl")
    signatures = {".": _signature(directory.lstat())}
    for name in sorted(_FILES):
        info = (directory / name).lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
            raise ValueError(f"bundle {name} must be a nonempty regular file, not a symlink")
        signatures[name] = _signature(info)
    return signatures


def _validate_report(report: Any, inventory: dict[str, Any], spec: ExtractionSpec) -> list[dict[str, Any]]:
    schema, descriptor = _schema(), spec.descriptor()
    code_dir = Path(__file__).parent
    code_hashes = {name: hashlib.sha256((code_dir / name).read_bytes()).hexdigest()
                   for name in ("pose_extract.py", "pose_backend.py", "ntu_io.py")}
    expected = {
        "format_version": 1, "kind": "ntu120_yolo_pose_development",
        "evidence_scope": "development_extraction_only", "final_test_decoded": False,
        "classifier_evaluated": False, "dataset_content_hash": inventory["dataset_content_hash"],
        "split_hash": inventory["split_hash"], "input_inventory_hash": _hash_json(inventory),
        "versions": spec.expected_versions(), "extractor_code_hash": _hash_json(code_hashes),
        "extraction_spec": descriptor, "extraction_spec_hash": _hash_json(descriptor),
        "observation_schema": schema, "observation_schema_hash": _hash_json(schema),
    }
    _object(report, set(expected) | {"python_version", "platform", "samples", "output_sha256"}, "manifest")
    for name, value in expected.items():
        _same(report[name], value, f"manifest {name}")
    for name in ("python_version", "platform"):
        if type(report[name]) is not str or not report[name].strip():
            raise ValueError(f"manifest {name} must be an explicit recorded string")
    outputs = _object(report["output_sha256"], {"observations.jsonl", "timing.jsonl"}, "output hashes")
    for name, value in outputs.items():
        _digest(value, name)
    selected = [row for row in inventory["samples"] if row["split"] in ("train", "validation")]
    reports = report["samples"]
    if type(reports) is not list or not selected or len(reports) != len(selected):
        raise ValueError("manifest samples must match the complete development roster")
    for source, sample in zip(selected, reports, strict=True):
        _object(sample, _SAMPLE_FIELDS, "sample report")
        for key, value in (("sample_id", source["sample_id"]), ("split", source["split"]),
                           ("source_sha256", source["sha256"])):
            _same(sample[key], value, f"sample {key}")
        count = _integer(sample["frame_count"], "frame_count", 1)
        missing = _integer(sample["missing_person_frames"], "missing_person_frames", 0)
        if missing > count:
            raise ValueError("missing-person count exceeds frame count")
        _integer(sample["height"], "height", 1)
        _integer(sample["width"], "width", 1)
    return selected


def _row(handle: BinaryIO, digest: Any, context: str) -> Any:
    raw = handle.readline()
    if not raw:
        raise ValueError(f"{context}: truncated sidecar or missing frame")
    digest.update(raw)
    if not raw.endswith(b"\n"):
        raise ValueError(f"{context}: unterminated JSONL row")
    return _json(raw)


def _frame(geometry: Any, timing: Any, source: dict[str, Any], index: int,
           previous: Fraction | None) -> tuple[KeypointFrame, int, Fraction, float]:
    _object(geometry, _POSE_FIELDS, "geometry row")
    _object(timing, _TIME_FIELDS, "timing row")
    for row in (geometry, timing):
        _same(row["sample_id"], source["sample_id"], "sample_id join/order")
        if _integer(row["frame_index"], "frame_index", 0) != index:
            raise ValueError("frame_index join/order must be contiguous and zero-based")
    _same(geometry["label"], source["label"], "sample label")
    pts = _integer(timing["pts"], "PTS")
    raw_base = timing["time_base"]
    if type(raw_base) is not list or len(raw_base) != 2:
        raise ValueError("time_base must contain numerator and denominator")
    numerator = _integer(raw_base[0], "time-base numerator", 1)
    denominator = _integer(raw_base[1], "time-base denominator", 1)
    base = Fraction(numerator, denominator)
    if [base.numerator, base.denominator] != raw_base:
        raise ValueError("time_base must be the producer's reduced positive fraction")
    timestamp = pts * base
    if previous is not None and timestamp <= previous:
        raise ValueError("presentation timestamps must strictly increase within a sample")
    score = _number(geometry["detector_confidence"], "detector confidence", confidence=True)
    points = geometry["body_keypoints"]
    if type(points) is not list or len(points) != 17:
        raise ValueError("expected exactly 17 COCO body keypoints")
    converted = []
    for point in points:
        if type(point) is not list or len(point) != 3:
            raise ValueError("keypoint must be [x, y, confidence]")
        converted.append((_number(point[0], "x"), _number(point[1], "y"),
                          _number(point[2], "keypoint confidence", confidence=True)))
    if score == 0.0 and any(value != 0.0 for point in converted for value in point):
        raise ValueError("missing person must have all-zero geometry and confidence")
    return KeypointFrame(tuple(converted)), pts, base, score


def load_development_bundle(
    bundle: str | Path, *, root: str | Path, inventory: dict[str, Any],
    spec: ExtractionSpec, expected_manifest_sha256: str,
) -> VerifiedPoseBundle:
    """Verify all inputs and rows before exposing immutable timed observations.

    Use private read-only trees. Observed stat/roster changes fail, but this is
    not an atomic filesystem snapshot. The digest pin is supplied independently;
    integrity does not establish pretrained origin, inference truth or accuracy.
    Sidecars stream during parsing; the returned dataset is materialized in RAM.
    """
    pin = _digest(expected_manifest_sha256, "manifest SHA-256")
    if type(spec) is not ExtractionSpec:
        raise ValueError("an independent ExtractionSpec is required")
    directory = Path(bundle).absolute()
    before = _bundle_state(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != pin:
        raise ValueError("manifest SHA-256 does not match the independent pin")
    report = _json(manifest_raw)
    verified = verify_rgb_manifest(root, inventory)
    selected = _validate_report(report, verified, spec)
    samples = []
    geometry_digest, timing_digest = hashlib.sha256(), hashlib.sha256()
    with (directory / "observations.jsonl").open("rb") as geometry, (directory / "timing.jsonl").open("rb") as timing:
        for source, sample in zip(selected, report["samples"], strict=True):
            frames, pts_values, bases, scores = [], [], [], []
            previous: Fraction | None = None
            for index in range(sample["frame_count"]):
                frame, pts, base, score = _frame(
                    _row(geometry, geometry_digest, "geometry"),
                    _row(timing, timing_digest, "timing"), source, index, previous,
                )
                previous = pts * base
                frames.append(frame)
                pts_values.append(pts)
                bases.append(base)
                scores.append(score)
            if sum(score == 0.0 for score in scores) != sample["missing_person_frames"]:
                raise ValueError("missing-person count differs from manifest")
            samples.append(TimedPoseSample(
                sample_id=source["sample_id"], label=source["label"], subject=source["subject"],
                split=source["split"], width=sample["width"], height=sample["height"],
                sequence=PointSequence(tuple(frames)), pts=tuple(pts_values),
                time_bases=tuple(bases), detector_confidences=tuple(scores),
            ))
        if geometry.read(1) or timing.read(1):
            raise ValueError("sidecar contains extra rows or trailing bytes")
    for name, digest in (("observations.jsonl", geometry_digest), ("timing.jsonl", timing_digest)):
        if digest.hexdigest() != report["output_sha256"][name]:
            raise ValueError(f"{name} bytes do not match manifest SHA-256")
    if _bundle_state(directory) != before:
        raise ValueError("bundle changed during verification")
    return VerifiedPoseBundle(
        samples=tuple(samples), manifest_sha256=pin,
        dataset_content_hash=verified["dataset_content_hash"], split_hash=verified["split_hash"],
        input_inventory_hash=report["input_inventory_hash"],
        extraction_spec_hash=report["extraction_spec_hash"],
        observation_schema_hash=report["observation_schema_hash"],
        extractor_code_hash=report["extractor_code_hash"],
        output_sha256=tuple(sorted(report["output_sha256"].items())),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "root", "inventory", "config"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    args = parser.parse_args(argv)
    try:
        config = _json(args.config.read_bytes())
        if type(config) is not dict:
            raise ValueError("extraction config must be a JSON object")
        spec = ExtractionSpec(**config)
        verified = load_development_bundle(
            args.bundle, root=args.root, inventory=_json(args.inventory.read_bytes()),
            spec=spec, expected_manifest_sha256=args.manifest_sha256,
        )
        print(_canonical_json({
            "samples": len(verified.samples),
            "frames": sum(len(sample.pts) for sample in verified.samples),
            "dataset_content_hash": verified.dataset_content_hash, "split_hash": verified.split_hash,
            "manifest_sha256": verified.manifest_sha256,
            "evidence_scope": "development_bundle_verified_only",
        }))
        return 0
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, f"Pose bundle error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
