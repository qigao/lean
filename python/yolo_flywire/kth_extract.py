"""Freeze and execute KTH YOLO11 real-CI extraction without touching final-test pixels."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
from typing import Any, Callable
from zipfile import BadZipFile, ZipFile

import numpy as np

from .kth_source import _ACTIONS, _ARCHIVE_URLS, _MEMBER, parse_sequence_file
from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive
from .pose_backend import decode_video, prediction_options, runtime_versions
from .pose_extract import _schema, _write_row
from .pose_features import PoseFeatureSpec, pose_encoder_hash
from .provenance import validate_rewiring_protocol

_MODEL = "yolo11n-pose.pt"
_ULTRALYTICS_VERSION = "8.4.146"
_DIGEST = re.compile(r"[0-9a-f]{64}")
_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class KthExtractionSpec:
    weights_sha256: str
    ultralytics_version: str
    torch_version: str
    numpy_version: str
    av_version: str
    opencv_version: str
    model: str = _MODEL

    def __post_init__(self) -> None:
        if type(self.weights_sha256) is not str or _DIGEST.fullmatch(self.weights_sha256) is None:
            raise ValueError("KTH weights_sha256 must be canonical lowercase SHA-256")
        if self.model != _MODEL:
            raise ValueError("KTH extractor requires exact yolo11n-pose.pt model")
        for name, version in self.expected_versions().items():
            if type(version) is not str or not re.fullmatch(
                r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*)", version
            ):
                raise ValueError(f"KTH {name} version must be explicit")

    def expected_versions(self) -> dict[str, str]:
        return {
            "ultralytics": self.ultralytics_version,
            "torch": self.torch_version,
            "numpy": self.numpy_version,
            "av": self.av_version,
            "opencv-python": self.opencv_version,
        }

    def descriptor(self) -> dict[str, Any]:
        identity = asdict(self)
        identity.pop("model")
        return {
            "identity": identity,
            "model": self.model,
            "predict": prediction_options(),
            "decoder": "pyav-avi-single-thread-all-frames",
            "sampling": "official-kth-list-order-identity; frame-membership-routing; overlaps-share-one-prediction",
            "partitions": ["train", "validation"],
            "person_policy": "zero-mask-or-single-or-unique-largest-detector-bbox-else-error",
        }


def _same(actual: Any, expected: Any, context: str) -> None:
    if _canonical_json(actual) != _canonical_json(expected):
        raise ValueError(f"KTH {context} differs from frozen contract")


def _validate_template(protocol: dict[str, Any]) -> PoseFeatureSpec:
    if type(protocol) is not dict:
        raise ValueError("KTH protocol must be a JSON object")
    expected = {
        "protocol_id": "v1-kth-yolo11n-real-ci",
        "claim": "development_topology_comparison",
        "dataset_id": "KTH-Human-Actions:official-cvap-release:RGB:6-class",
        "task_labels": list(_ACTIONS),
        "split_rule": "official-kth-icpr2004-subject-split; train=11-18; validation=19,20,21,23,24,25,01,04; final-test=22,02,03,05,06,07,08,09,10",
        "yolo_version": "ultralytics-yolo11n-pose",
        "ultralytics_package_version": _ULTRALYTICS_VERSION,
        "yolo_weights": _MODEL,
        "seeds": [7, 11, 19, 23, 31],
        "budget": {"epochs": 20, "max_updates": 40, "parameter_ceiling": 50000},
        "final_test_used_for_selection": False,
        "final_test_decoded": False,
        "classifier_evaluated": False,
    }
    for name, value in expected.items():
        _same(protocol.get(name), value, f"protocol {name}")
    for name in ("dataset_content_hash", "input_inventory_hash", "source_manifest_hash",
                 "split_hash", "sequence_file_sha256", "archive_sha256", "yolo_weights_sha256",
                 "observation_schema_hash", "pose_encoder_hash", "extraction_spec_hash"):
        if protocol.get(name) is not None:
            raise ValueError(f"KTH preflight field {name} must be null before freeze")
    if protocol.get("extraction_spec") is not None:
        raise ValueError("KTH preflight extraction_spec must be null before freeze")
    validate_rewiring_protocol(protocol)
    raw_feature = protocol.get("pose_feature_spec")
    if type(raw_feature) is not dict or set(raw_feature) != {"confidence_threshold", "scale_epsilon"}:
        raise ValueError("KTH pose_feature_spec malformed")
    return PoseFeatureSpec(**raw_feature)


def _weights_hash(path: Path) -> str:
    checkpoint = path.absolute()
    if checkpoint.name != _MODEL or checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("KTH requires an explicit regular yolo11n-pose.pt file")
    return _hash_file(checkpoint)[1]


def _spec_from_protocol(protocol: dict[str, Any]) -> KthExtractionSpec:
    try:
        descriptor = protocol["extraction_spec"]
        identity = descriptor["identity"]
        spec = KthExtractionSpec(**identity)
    except (KeyError, TypeError) as exc:
        raise ValueError("KTH frozen protocol missing extraction spec") from exc
    if descriptor.get("model") != _MODEL or _hash_json(spec.descriptor()) != protocol.get("extraction_spec_hash"):
        raise ValueError("KTH extraction descriptor/hash mismatch")
    return spec


def freeze_kth_runtime(protocol: dict[str, Any], *, sequence_file: str | Path,
                       weights: str | Path) -> dict[str, Any]:
    """Freeze sequence bytes + exact YOLO/runtime identity without downloading/decoding KTH RGB."""
    feature_spec = _validate_template(protocol)
    sequence_path = Path(sequence_file).absolute()
    if sequence_path.is_symlink() or not sequence_path.is_file():
        raise ValueError("KTH sequence file must be regular")
    sequence_raw = sequence_path.read_bytes()
    plan = parse_sequence_file(sequence_raw.decode("utf-8"))
    if (len(plan.videos) != 600 or sum(not row["missing"] for row in plan.videos) != 599
            or len(plan.subsequences) != 2391):
        raise ValueError("KTH sequence plan cardinality mismatch")
    versions = runtime_versions()
    if set(versions) != {"ultralytics", "torch", "numpy", "av", "opencv-python"}:
        raise ValueError("KTH runtime version record is incomplete")
    if versions["ultralytics"] != _ULTRALYTICS_VERSION:
        raise ValueError("KTH Ultralytics runtime differs from protocol")
    weight_sha = _weights_hash(Path(weights))
    spec = KthExtractionSpec(
        weights_sha256=weight_sha,
        ultralytics_version=versions["ultralytics"], torch_version=versions["torch"],
        numpy_version=versions["numpy"], av_version=versions["av"],
        opencv_version=versions["opencv-python"],
    )
    frozen = json.loads(_canonical_json(protocol))
    frozen["sequence_file_sha256"] = hashlib.sha256(sequence_raw).hexdigest()
    frozen["yolo_weights_sha256"] = weight_sha
    frozen["observation_schema_hash"] = _hash_json(_schema())
    frozen["pose_encoder_hash"] = pose_encoder_hash(feature_spec)
    frozen["extraction_spec"] = spec.descriptor()
    frozen["extraction_spec_hash"] = _hash_json(spec.descriptor())
    frozen["evidence_scope"] = "kth_public_real_ci_runtime_frozen_before_rgb_extraction"
    return json.loads(_canonical_json(frozen))


def _kth_pose(points: np.ndarray, scores: np.ndarray,
              boxes: np.ndarray) -> tuple[list[list[float]], float]:
    if (not isinstance(points, np.ndarray) or points.ndim != 3 or points.shape[1:] != (17, 3)
            or not isinstance(scores, np.ndarray) or scores.shape != (points.shape[0],)
            or not isinstance(boxes, np.ndarray) or boxes.shape != (points.shape[0], 4)):
        raise ValueError("KTH pose prediction must have shapes [N,17,3], [N], and [N,4]")
    if points.dtype.kind not in "fiu" or scores.dtype.kind not in "fiu" or boxes.dtype.kind not in "fiu":
        raise ValueError("KTH pose predictions must be numeric arrays")
    if not np.isfinite(points).all() or not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("non-finite KTH pose prediction")
    if np.any((points[..., 2] < 0) | (points[..., 2] > 1)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("KTH pose confidence must be in [0,1]")
    if not len(scores):
        return [[0.0, 0.0, 0.0] for _ in range(17)], 0.0
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    if np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("KTH detector bbox must have positive area")
    if len(scores) == 1:
        index = 0
    else:
        areas = widths * heights
        largest = areas.max()
        winners = np.flatnonzero(areas == largest)
        if len(winners) != 1:
            raise ValueError("KTH multiple people have tied largest detector bbox")
        index = int(winners[0])
    return points[index].astype(float).tolist(), float(scores[index])


def _load_predictor(
    weights: Path, spec: KthExtractionSpec,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if _weights_hash(weights) != spec.weights_sha256:
        raise ValueError("KTH YOLO11 bytes differ from frozen SHA-256")
    if runtime_versions() != spec.expected_versions():
        raise ValueError("KTH extraction runtime differs from frozen spec")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = YOLO(str(weights.absolute()), task="pose")
    if model.task != "pose" or tuple(model.model.yaml.get("kpt_shape", ())) != (17, 3):
        raise ValueError("KTH checkpoint is not an exact COCO17 pose model")
    model.model.float().eval()

    def predict(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with torch.inference_mode():
            results = model.predict(source=image, **prediction_options())
        if len(results) != 1:
            raise ValueError("KTH single-frame inference returned unexpected result count")
        result = results[0]
        if result.keypoints is None or result.boxes is None or result.boxes.is_track:
            raise ValueError("KTH pose result missing keypoints/boxes or contains tracking state")
        if not bool((result.boxes.cls == 0).all()):
            raise ValueError("KTH pose result contains non-person class")
        return (
            result.keypoints.data.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
            result.boxes.xyxy.detach().cpu().numpy(),
        )

    return predict


def _materialize_member(zipped: ZipFile, entry: Any, destination: Path) -> tuple[int, str]:
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH scratch AVI already exists")
    digest = hashlib.sha256()
    length = 0
    try:
        with zipped.open(entry, "r") as source, destination.open("xb") as target:
            while chunk := source.read(_CHUNK):
                target.write(chunk)
                digest.update(chunk)
                length += len(chunk)
            target.flush()
            os.fsync(target.fileno())
        if length != entry.file_size or length <= 0:
            raise ValueError("KTH ZIP member size changed during materialization")
        return length, digest.hexdigest()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _open_sample_handles(samples_root: Path, rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    states: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_dir = samples_root / row["sample_id"]
        sample_dir.mkdir(parents=True, exist_ok=False)
        geometry = sample_dir / "observations.jsonl"
        timing = sample_dir / "timing.jsonl"
        states[row["sample_id"]] = {
            "row": row, "geometry_path": geometry, "timing_path": timing,
            "geometry": geometry.open("xb"), "timing": timing.open("xb"),
            "count": 0, "missing": 0, "previous": None, "shape": None,
        }
    return states


def _close_sample_handles(states: dict[str, dict[str, Any]]) -> None:
    for state in states.values():
        for name in ("geometry", "timing"):
            handle = state[name]
            if not handle.closed:
                handle.flush()
                os.fsync(handle.fileno())
                handle.close()


def _extract_parent(
    clip: Path, video: dict[str, Any], sample_rows: list[dict[str, Any]],
    predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    samples_root: Path,
) -> list[dict[str, Any]]:
    """Route decoded frames by membership while preserving official listed range identity."""
    if not sample_rows:
        raise ValueError("present KTH development video must have at least one official subsequence")
    states = _open_sample_handles(samples_root, sample_rows)
    range_rows = sorted(sample_rows, key=lambda row: row["range_index"])
    max_end = max(row["end_frame"] for row in range_rows)
    try:
        with closing(decode_video(clip)) as frames:
            for frame_number, (pts, base, image) in enumerate(frames, 1):
                if frame_number > max_end:
                    break
                active = [row for row in range_rows
                          if row["start_frame"] <= frame_number <= row["end_frame"]]
                if not active:
                    continue
                if type(pts) is not int or not isinstance(base, Fraction) or base <= 0:
                    raise ValueError("KTH decoder returned invalid PTS/time base")
                if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                        or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                    raise ValueError("KTH decoded frame must be nonempty uint8 BGR")
                timestamp = pts * base
                shape = tuple(image.shape)
                points, confidence = _kth_pose(*predictor(image))
                for row in active:
                    state = states[row["sample_id"]]
                    if state["previous"] is not None and timestamp <= state["previous"]:
                        raise ValueError("KTH subsequence timestamps must strictly increase")
                    state["previous"] = timestamp
                    if state["shape"] is not None and state["shape"] != shape:
                        raise ValueError("KTH video dimensions changed within subsequence")
                    state["shape"] = shape
                    index = state["count"]
                    key = {"sample_id": row["sample_id"], "frame_index": index}
                    _write_row(state["geometry"], {
                        **key, "label": row["label"], "body_keypoints": points,
                        "detector_confidence": confidence,
                    })
                    _write_row(state["timing"], {
                        **key, "pts": pts, "time_base": [base.numerator, base.denominator],
                    })
                    state["count"] += 1
                    state["missing"] += int(confidence == 0.0)
        _close_sample_handles(states)
        reports = []
        for row in range_rows:
            state = states[row["sample_id"]]
            expected_frames = row["end_frame"] - row["start_frame"] + 1
            if state["count"] != expected_frames or state["shape"] is None:
                raise ValueError(f"KTH decoder did not produce full official interval: {row['sample_id']}")
            reports.append({
                "sample_id": row["sample_id"], "video_key": row["video_key"],
                "split": row["split"], "label": row["label"], "subject": row["subject"],
                "scenario": row["scenario"], "range_index": row["range_index"],
                "start_frame": row["start_frame"], "end_frame": row["end_frame"],
                "frame_count": state["count"], "missing_person_frames": state["missing"],
                "height": state["shape"][0], "width": state["shape"][1],
                "output_sha256": {
                    "observations.jsonl": _hash_file(state["geometry_path"])[1],
                    "timing.jsonl": _hash_file(state["timing_path"])[1],
                },
            })
        return reports
    finally:
        _close_sample_handles(states)


def kth_extractor_code_hash() -> str:
    directory = Path(__file__).parent
    names = ("kth_extract.py", "kth_source.py", "pose_backend.py", "pose_extract.py", "pose_features.py")
    return _hash_json({name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names})


def extract_action_shard(action: str, archive: str | Path, *, sequence_file: str | Path,
                         protocol: dict[str, Any], weights: str | Path,
                         output: str | Path) -> dict[str, Any]:
    """Extract one official KTH action archive, sealing final-test subjects from decode."""
    if type(action) is not str or action not in _ACTIONS:
        raise ValueError("KTH action must be one of the frozen six classes")
    sequence_path = Path(sequence_file).absolute()
    sequence_raw = sequence_path.read_bytes()
    if hashlib.sha256(sequence_raw).hexdigest() != protocol.get("sequence_file_sha256"):
        raise ValueError("KTH sequence bytes differ from coordinator freeze")
    plan = parse_sequence_file(sequence_raw.decode("utf-8"))
    spec = _spec_from_protocol(protocol)
    checkpoint = Path(weights).absolute()
    if _weights_hash(checkpoint) != spec.weights_sha256 or runtime_versions() != spec.expected_versions():
        raise ValueError("KTH model/runtime differs from coordinator freeze")

    archive_path = Path(archive).absolute()
    info = archive_path.lstat() if archive_path.exists() else None
    if (info is None or not stat.S_ISREG(info.st_mode) or archive_path.is_symlink() or info.st_size <= 0):
        raise ValueError("KTH action archive must be a nonempty regular ZIP")
    archive_size, archive_sha, _ = _hash_file(archive_path)
    destination = Path(output).absolute()
    if destination.exists() or destination.is_symlink():
        raise ValueError("KTH shard output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    scratch = destination / ".scratch"
    scratch.mkdir()
    samples_root = destination / "samples"
    logical_videos = [row for row in plan.videos if row["action"] == action]
    videos = [row for row in logical_videos if not row["missing"]]
    missing_videos = [
        {key: row[key] for key in ("video_key", "filename", "action", "subject", "scenario", "split")}
        for row in logical_videos if row["missing"]
    ]
    subsequences_by_video: dict[str, list[dict[str, Any]]] = {}
    for row in plan.subsequences:
        if row["label"] == action:
            subsequences_by_video.setdefault(row["video_key"], []).append(row)
    video_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    predictor = None
    try:
        with ZipFile(archive_path, "r") as zipped:
            entries = [entry for entry in zipped.infolist() if not entry.is_dir()]
            names = [entry.filename for entry in entries]
            expected_names = {row["filename"] for row in videos}
            if len(names) != len(set(names)) or set(names) != expected_names or len(names) != len(expected_names):
                raise ValueError(
                    f"KTH {action} ZIP roster differs from official present-video set; "
                    f"expected={len(expected_names)} actual={len(names)}"
                )
            by_name = {entry.filename: entry for entry in entries}
            for name in names:
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts or len(pure.parts) != 1 or _MEMBER.fullmatch(name) is None:
                    raise ValueError(f"unsafe/unexpected KTH ZIP member: {name}")
            for video in videos:
                if any(scratch.iterdir()):
                    raise ValueError("KTH scratch must be empty before parent video")
                entry = by_name[video["filename"]]
                clip = scratch / video["filename"]
                source_size, source_sha = _materialize_member(zipped, entry, clip)
                split = video["split"]
                if split == "final_test":
                    decoded = False
                else:
                    rows = subsequences_by_video.get(video["video_key"], [])
                    if not rows:
                        raise ValueError("present KTH development parent lacks official subsequences")
                    if predictor is None:
                        predictor = _load_predictor(checkpoint, spec)
                    samples_root.mkdir(exist_ok=True)
                    sample_records.extend(_extract_parent(clip, video, rows, predictor, samples_root))
                    decoded = True
                clip.unlink()
                if clip.exists() or any(scratch.iterdir()):
                    raise ValueError("KTH parent AVI was not deleted before next video")
                video_records.append({
                    "video_key": video["video_key"], "filename": video["filename"],
                    "action": action, "subject": video["subject"], "scenario": video["scenario"],
                    "split": split, "size_bytes": source_size, "sha256": source_sha,
                    "verified_bytes": True, "decoded": decoded,
                })
        scratch.rmdir()
        descriptor = spec.descriptor()
        report = {
            "format_version": 1, "kind": "kth_yolo11_pose_action_shard",
            "evidence_scope": "kth_public_real_action_extraction_only",
            "final_test_decoded": False, "classifier_evaluated": False,
            "action": action,
            "archive": {"url": _ARCHIVE_URLS[action], "size_bytes": archive_size, "sha256": archive_sha},
            "sequence_file_sha256": protocol["sequence_file_sha256"],
            "versions": spec.expected_versions(), "python_version": platform.python_version(),
            "platform": platform.platform(), "extractor_code_hash": kth_extractor_code_hash(),
            "extraction_spec": descriptor, "extraction_spec_hash": _hash_json(descriptor),
            "observation_schema": _schema(), "observation_schema_hash": _hash_json(_schema()),
            "videos": video_records, "missing_videos": missing_videos, "samples": sample_records,
        }
        _publish_exclusive(destination / "shard-report.json", report)
        return json.loads(_canonical_json(report))
    except BaseException:
        if scratch.exists():
            shutil.rmtree(scratch, ignore_errors=True)
        raise


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read {name} JSON") from exc
    if type(value) is not dict:
        raise ValueError(f"{name} must be JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--sequence-file", type=Path, required=True)
    freeze.add_argument("--weights", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--action", required=True)
    shard.add_argument("--archive", type=Path, required=True)
    shard.add_argument("--sequence-file", type=Path, required=True)
    shard.add_argument("--protocol", type=Path, required=True)
    shard.add_argument("--weights", type=Path, required=True)
    shard.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze":
            result = freeze_kth_runtime(
                _read_json(args.protocol, "KTH protocol"), sequence_file=args.sequence_file,
                weights=args.weights,
            )
            _publish_exclusive(args.output, result)
            print(_canonical_json({"sequence_file_sha256": result["sequence_file_sha256"],
                                   "yolo_weights_sha256": result["yolo_weights_sha256"],
                                   "final_test_decoded": False}))
        else:
            result = extract_action_shard(
                args.action, args.archive, sequence_file=args.sequence_file,
                protocol=_read_json(args.protocol, "frozen KTH protocol"),
                weights=args.weights, output=args.output,
            )
            print(_canonical_json({"action": result["action"], "videos": len(result["videos"]),
                                   "missing_videos": len(result["missing_videos"]),
                                   "samples": len(result["samples"]), "final_test_decoded": False}))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError, UnicodeError) as exc:
        parser.exit(1, f"KTH extraction error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())