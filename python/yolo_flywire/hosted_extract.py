"""Bounded one-clip-at-a-time YOLO11 extraction for authorized remote NTU sources."""
from __future__ import annotations

import argparse
from contextlib import closing
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any, Callable
from urllib.request import Request, urlopen

import numpy as np

from .hosted_freeze import HostedExtractionSpec, _check_weights
from .ntu_io import _canonical_json, _hash_file, _hash_json, _publish_exclusive
from .pose_backend import decode_video, prediction_options, runtime_versions
from .pose_extract import _pose, _schema, _write_row
from .remote_rgb import VerifiedRemoteManifest, shard_rows, validate_remote_manifest

_CHUNK = 1024 * 1024


def hosted_extractor_code_hash() -> str:
    directory = Path(__file__).parent
    names = (
        "hosted_extract.py", "hosted_freeze.py", "remote_rgb.py",
        "pose_backend.py", "pose_extract.py", "ntu_io.py",
    )
    return _hash_json({name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in names})


def _spec_from_protocol(protocol: dict[str, Any]) -> HostedExtractionSpec:
    if type(protocol) is not dict:
        raise ValueError("frozen hosted protocol must be a JSON object")
    try:
        descriptor = protocol["extraction_spec"]
        if type(descriptor) is not dict or descriptor.get("model") != "yolo11n-pose.pt":
            raise ValueError("frozen hosted extraction descriptor has wrong model")
        identity = descriptor["identity"]
        if type(identity) is not dict:
            raise ValueError("frozen hosted extraction identity must be an object")
        spec = HostedExtractionSpec(**identity)
    except (KeyError, TypeError) as exc:
        raise ValueError("frozen hosted protocol is missing extraction identity") from exc
    if _hash_json(spec.descriptor()) != protocol.get("extraction_spec_hash"):
        raise ValueError("frozen hosted extraction descriptor/hash mismatch")
    return spec


def _verify_protocol_source(protocol: dict[str, Any], source: VerifiedRemoteManifest) -> None:
    expected = {
        "dataset_content_hash": source.dataset_content_hash,
        "split_hash": source.split_hash,
        "input_inventory_hash": source.input_inventory_hash,
        "source_manifest_hash": source.source_manifest_hash,
    }
    for name, value in expected.items():
        if _canonical_json(protocol.get(name)) != _canonical_json(value):
            raise ValueError(f"frozen hosted protocol {name} differs from remote source")
    if protocol.get("final_test_decoded") is not False or protocol.get("classifier_evaluated") is not False:
        raise ValueError("frozen hosted protocol crossed final-test/classifier boundary")


def _load_predictor(weights: Path, spec: HostedExtractionSpec) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    _check_weights(weights, spec)
    if runtime_versions() != spec.expected_versions():
        raise ValueError("hosted extraction runtime differs from frozen specification")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = YOLO(str(weights.absolute()), task="pose")
    if model.task != "pose" or tuple(model.model.yaml.get("kpt_shape", ())) != (17, 3):
        raise ValueError("hosted checkpoint is not a 17-keypoint pose model")
    model.model.float().eval()

    def predict(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            results = model.predict(source=image, **prediction_options())
        if len(results) != 1:
            raise ValueError("single-frame hosted inference returned unexpected result count")
        result = results[0]
        if result.keypoints is None or result.boxes is None or result.boxes.is_track:
            raise ValueError("hosted pose result missing keypoints/boxes or contains tracking state")
        if not bool((result.boxes.cls == 0).all()):
            raise ValueError("unexpected detector class in hosted person-only extraction")
        return (
            result.keypoints.data.detach().cpu().numpy(),
            result.boxes.conf.detach().cpu().numpy(),
        )

    return predict


def _download_verified(locator: str, destination: Path, size: int, expected_sha256: str) -> None:
    if destination.exists() or destination.is_symlink():
        raise ValueError("temporary RGB destination already exists")
    request = Request(locator, headers={"User-Agent": "yolo-flywire-hosted/1"})
    digest = hashlib.sha256()
    length = 0
    try:
        with urlopen(request, timeout=120) as response, destination.open("xb") as handle:
            while chunk := response.read(_CHUNK):
                handle.write(chunk)
                digest.update(chunk)
                length += len(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if length != size or digest.hexdigest() != expected_sha256:
            raise ValueError("downloaded RGB bytes differ from frozen size/SHA-256")
        if _hash_file(destination)[:2] != (size, expected_sha256):
            raise ValueError("materialized RGB file differs after download verification")
    except BaseException:
        destination.unlink(missing_ok=True)
        raise


def _extract_sample(
    row: dict[str, Any], clip: Path, predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    sample_dir: Path,
) -> dict[str, Any]:
    sample_dir.mkdir(parents=True, exist_ok=False)
    geometry = sample_dir / "observations.jsonl"
    timing_path = sample_dir / "timing.jsonl"
    previous: Fraction | None = None
    shape: tuple[int, ...] | None = None
    count = missing = 0
    with geometry.open("xb") as observations, timing_path.open("xb") as timing:
        with closing(decode_video(clip)) as frames:
            for index, (pts, base, image) in enumerate(frames):
                if type(pts) is not int or not isinstance(base, Fraction) or base <= 0:
                    raise ValueError("hosted decoder returned invalid PTS/time base")
                timestamp = pts * base
                if previous is not None and timestamp <= previous:
                    raise ValueError("hosted presentation timestamps must strictly increase")
                previous = timestamp
                if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                        or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0):
                    raise ValueError("hosted decoded frame must be nonempty uint8 BGR")
                if shape is not None and image.shape != shape:
                    raise ValueError("hosted video dimensions changed within sample")
                shape = image.shape
                points, confidence = _pose(*predictor(image))
                missing += int(confidence == 0.0)
                key = {"sample_id": row["sample_id"], "frame_index": index}
                _write_row(observations, {
                    **key, "label": row["label"],
                    "body_keypoints": points, "detector_confidence": confidence,
                })
                _write_row(timing, {
                    **key, "pts": pts, "time_base": [base.numerator, base.denominator],
                })
                count += 1
        if not count or shape is None:
            raise ValueError(f"hosted video has no decoded frames: {row['sample_id']}")
        for handle in (observations, timing):
            handle.flush()
            os.fsync(handle.fileno())
    return {
        "sample_id": row["sample_id"], "split": row["split"],
        "source_size_bytes": row["size_bytes"], "source_sha256": row["sha256"],
        "verified_bytes": True, "decoded": True,
        "frame_count": count, "missing_person_frames": missing,
        "height": shape[0], "width": shape[1],
        "output_sha256": {
            "observations.jsonl": _hash_file(geometry)[1],
            "timing.jsonl": _hash_file(timing_path)[1],
        },
    }


def extract_remote_shard(
    remote_manifest: dict[str, Any], *, protocol: dict[str, Any],
    shard_index: int, shard_count: int, weights: str | Path, output: str | Path,
) -> dict[str, Any]:
    """Verify and process one deterministic shard with bounded RGB residency."""
    remote = validate_remote_manifest(remote_manifest)
    _verify_protocol_source(protocol, remote)
    spec = _spec_from_protocol(protocol)
    checkpoint = Path(weights).absolute()
    _check_weights(checkpoint, spec)
    versions = runtime_versions()
    if versions != spec.expected_versions():
        raise ValueError("hosted shard runtime differs from frozen protocol")
    rows = shard_rows(remote, shard_index, shard_count)
    if not rows:
        raise ValueError("hosted shard is empty; choose shard_count <= sample count")

    destination = Path(output).absolute()
    for component in (destination, *destination.parents):
        if component.is_symlink():
            raise ValueError("hosted shard output must not traverse symlinks")
    if destination.exists():
        raise ValueError("hosted shard output already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    scratch = destination / ".scratch"
    scratch.mkdir()
    samples_root = destination / "samples"
    records: list[dict[str, Any]] = []
    predictor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None
    try:
        for row in rows:
            if any(scratch.iterdir()):
                raise ValueError("hosted shard scratch must be empty before each sample")
            clip = scratch / row["filename"]
            _download_verified(row["locator"], clip, row["size_bytes"], row["sha256"])
            if row["split"] == "final_test":
                record = {
                    "sample_id": row["sample_id"], "split": row["split"],
                    "source_size_bytes": row["size_bytes"], "source_sha256": row["sha256"],
                    "verified_bytes": True, "decoded": False, "output_sha256": None,
                }
            else:
                if predictor is None:
                    predictor = _load_predictor(checkpoint, spec)
                samples_root.mkdir(exist_ok=True)
                record = _extract_sample(row, clip, predictor, samples_root / row["sample_id"])
            clip.unlink()
            if clip.exists() or any(scratch.iterdir()):
                raise ValueError("hosted RGB clip was not removed before next sample")
            records.append(record)
        scratch.rmdir()
        descriptor = spec.descriptor()
        report = {
            "format_version": 1,
            "kind": "ntu120_yolo11_pose_remote_shard",
            "evidence_scope": "hosted_shard_extraction_only",
            "final_test_decoded": False,
            "classifier_evaluated": False,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "dataset_content_hash": remote.dataset_content_hash,
            "split_hash": remote.split_hash,
            "input_inventory_hash": remote.input_inventory_hash,
            "source_manifest_hash": remote.source_manifest_hash,
            "transport_manifest_hash": remote.transport_manifest_hash,
            "versions": versions,
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "extractor_code_hash": hosted_extractor_code_hash(),
            "extraction_spec": descriptor,
            "extraction_spec_hash": _hash_json(descriptor),
            "observation_schema": _schema(),
            "observation_schema_hash": _hash_json(_schema()),
            "samples": records,
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
        raise ValueError(f"{name} must be a JSON object")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    shard = commands.add_parser("shard")
    shard.add_argument("--remote-manifest", type=Path, required=True)
    shard.add_argument("--protocol", type=Path, required=True)
    shard.add_argument("--weights", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--shard-count", type=int, required=True)
    shard.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = extract_remote_shard(
            _read_json(args.remote_manifest, "remote manifest"),
            protocol=_read_json(args.protocol, "frozen hosted protocol"),
            shard_index=args.shard_index, shard_count=args.shard_count,
            weights=args.weights, output=args.output,
        )
        print(_canonical_json({
            "shard_index": report["shard_index"],
            "samples": len(report["samples"]),
            "final_test_decoded": False,
        }))
        return 0
    except (OSError, ValueError, TypeError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"Hosted shard extraction error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
