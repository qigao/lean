"""Version-bound, development-only extraction of timed NTU YOLO/Pose observations."""
from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import re
from typing import Any

import numpy as np

from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive, verify_rgb_manifest
from .pose_backend import decode_video, load_predictor, prediction_options, runtime_versions

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionSpec:
    weights_sha256: str
    ultralytics_version: str
    torch_version: str
    numpy_version: str
    av_version: str
    opencv_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.weights_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", self.weights_sha256):
            raise ValueError("weights_sha256 must be a lowercase SHA-256")
        for name, version in self.expected_versions().items():
            if not isinstance(version, str) or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[a-zA-Z0-9.+-]*)", version):
                raise ValueError(f"{name} must have an explicit exact package version")

    def expected_versions(self) -> dict[str, str]:
        return {"ultralytics": self.ultralytics_version, "torch": self.torch_version,
                "numpy": self.numpy_version, "av": self.av_version,
                "opencv-python": self.opencv_version}

    def descriptor(self) -> dict[str, Any]:
        return {"identity": asdict(self), "model": "yolo26n-pose.pt",
                "predict": prediction_options(), "decoder": "pyav-avi-single-thread-all-frames",
                "partitions": ["train", "validation"], "person_policy": "zero-mask-or-single-else-error"}


def _check_weights(path: Path, spec: ExtractionSpec) -> None:
    if path.name != "yolo26n-pose.pt":
        raise ValueError("expected explicit local yolo26n-pose.pt; no alternate model")
    for component in (path.absolute(), *path.absolute().parents):
        if component.is_symlink():
            raise ValueError("weight path must not traverse symlinks")
    if _hash_file(path)[1] != spec.weights_sha256:
        raise ValueError("weight bytes do not match frozen SHA-256")


def _pose(points: np.ndarray, scores: np.ndarray) -> tuple[list[list[float]], float]:
    if (not isinstance(points, np.ndarray) or points.ndim != 3 or points.shape[1:] != (17, 3)
            or not isinstance(scores, np.ndarray) or scores.shape != (points.shape[0],)):
        raise ValueError("pose prediction must have shapes [N,17,3] and [N]")
    if points.dtype.kind not in "fiu" or scores.dtype.kind not in "fiu":
        raise ValueError("pose predictions must be numeric arrays")
    if not np.isfinite(points).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite pose prediction")
    if np.any((points[..., 2] < 0) | (points[..., 2] > 1)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("pose confidence must be in [0,1]")
    if len(scores) > 1:
        raise ValueError("multiple people detected; refusing identity selection")
    if not len(scores):
        return [[0.0, 0.0, 0.0] for _ in range(17)], 0.0
    return points[0].astype(float).tolist(), float(scores[0])


def _schema() -> dict[str, Any]:
    return {"format": "yolo-pose-jsonl-v0", "body_keypoints": "COCO17",
            "point_layout": ["x_pixels", "y_pixels", "confidence"],
            "detector_confidence": True, "hand_keypoints": False,
            "timing": "timing.jsonl joined by sample_id/frame_index; pts * rational time_base",
            "frame_sampling": "all-decoded-frames-no-interpolation",
            "missing_person": "17 zero-coordinate/zero-confidence points"}


def _write_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write((_canonical_json(row) + "\n").encode("utf-8"))


def extract_development(
    root: str | Path, inventory: dict[str, Any], weights: str | Path,
    spec: ExtractionSpec, output: str | Path,
) -> dict[str, Any]:
    """Extract train/validation only; manifest.json is the final commit marker.

    Inputs and output parent must be private, immutable during the operation.
    Reverification detects observed changes, not adversarial filesystem races.
    No output here authorizes final-test execution or proves recognition quality.
    """
    if not isinstance(spec, ExtractionSpec):
        raise ValueError("an explicit ExtractionSpec is required")
    destination = Path(output).absolute()
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("output must not traverse symlinks")
    if destination.exists():
        raise ValueError("output already exists; refusing to overwrite evidence")
    directory, checkpoint = Path(root).absolute(), Path(weights).absolute()
    if destination.is_relative_to(directory):
        raise ValueError("extraction output must be outside the immutable input tree")
    _check_weights(checkpoint, spec)
    versions = runtime_versions()
    if versions != spec.expected_versions():
        raise ValueError("installed package versions do not match the extraction spec")
    verified = verify_rgb_manifest(directory, inventory)
    selected = [row for row in verified["samples"] if row["split"] in ("train", "validation")]
    if not selected:
        raise ValueError("no development samples to extract")
    predictor = load_predictor(checkpoint, spec)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()  # Exclusive reservation; never remove a preexisting directory.
    owned: list[Path] = []
    try:
        sample_reports: list[dict[str, Any]] = []
        observation_path, timing_path = destination / "observations.jsonl", destination / "timing.jsonl"
        with observation_path.open("xb") as observations:
            owned.append(observation_path)
            with timing_path.open("xb") as timing:
                owned.append(timing_path)
                for row in selected:
                    _LOG.info("extract sample=%s split=%s", row["sample_id"], row["split"])
                    previous: Fraction | None = None
                    shape: tuple[int, ...] | None = None
                    count = missing = 0
                    # Paths come from a freshly rebuilt inventory, never the supplied JSON.
                    with closing(decode_video(directory / row["relative_path"])) as frames:
                        for index, (pts, base, image) in enumerate(frames):
                            if type(pts) is not int or not isinstance(base, Fraction) or base <= 0:
                                raise ValueError("missing or invalid integer PTS/rational time base")
                            timestamp = pts * base
                            if previous is not None and timestamp <= previous:
                                raise ValueError("presentation timestamps must strictly increase")
                            previous = timestamp
                            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                                    or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                                raise ValueError("decoded frame must be a nonempty uint8 BGR image")
                            if shape is not None and image.shape != shape:
                                raise ValueError("video dimensions changed within a sample")
                            shape = image.shape
                            points, confidence = _pose(*predictor(image))
                            missing += int(confidence == 0.0)
                            key = {"sample_id": row["sample_id"], "frame_index": index}
                            _write_row(observations, {**key, "label": row["label"],
                                "body_keypoints": points, "detector_confidence": confidence})
                            _write_row(timing, {**key, "pts": pts,
                                "time_base": [base.numerator, base.denominator]})
                            count += 1
                    if not count:
                        raise ValueError(f"video has no decoded frames: {row['sample_id']}")
                    sample_reports.append({"sample_id": row["sample_id"], "split": row["split"],
                        "source_sha256": row["sha256"], "frame_count": count,
                        "missing_person_frames": missing, "height": shape[0], "width": shape[1]})
                    _LOG.info("completed sample=%s frames=%d missing=%d", row["sample_id"], count, missing)
                for handle in (observations, timing):
                    handle.flush()
                    os.fsync(handle.fileno())
        verify_rgb_manifest(directory, verified)
        _check_weights(checkpoint, spec)
        schema, descriptor = _schema(), spec.descriptor()
        source_dir = Path(__file__).parent
        code_hashes = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
                       for name in ("pose_extract.py", "pose_backend.py", "ntu_io.py")}
        report = {"format_version": 1, "kind": "ntu120_yolo_pose_development",
            "evidence_scope": "development_extraction_only", "final_test_decoded": False,
            "classifier_evaluated": False, "dataset_content_hash": verified["dataset_content_hash"],
            "split_hash": verified["split_hash"], "input_inventory_hash": _hash_json(verified),
            "versions": versions, "python_version": platform.python_version(),
            "platform": platform.platform(), "extractor_code_hash": _hash_json(code_hashes),
            "extraction_spec": descriptor, "extraction_spec_hash": _hash_json(descriptor),
            "observation_schema": schema, "observation_schema_hash": _hash_json(schema),
            "samples": sample_reports,
            "output_sha256": {path.name: _hash_file(path)[1] for path in owned}}
        _publish_exclusive(destination / "manifest.json", report)
        return report
    except BaseException:
        # Only files opened by this operation are ours to remove. No recursive deletion.
        for path in reversed(owned):
            path.unlink(missing_ok=True)
        if not any(destination.iterdir()):
            destination.rmdir()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "inventory", "weights", "config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("extraction config must be an object")
        spec = ExtractionSpec(**config)
        inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
        report = extract_development(args.root, inventory, args.weights, spec, args.output)
        print(_canonical_json({"samples": len(report["samples"]),
            "evidence_scope": report["evidence_scope"], "final_test_decoded": False}))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Pose extraction error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
