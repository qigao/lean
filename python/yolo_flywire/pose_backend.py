"""Dependency-lazy PyAV and Ultralytics adapters; no downloads or model fallback."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from fractions import Fraction
from importlib.metadata import version
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .pose_extract import ExtractionSpec


def runtime_versions() -> dict[str, str]:
    return {name: version(name) for name in ("ultralytics", "torch", "numpy", "av", "opencv-python")}


def prediction_options() -> dict[str, Any]:
    return {"imgsz": 640, "conf": 0.25, "iou": 0.7, "device": "cpu", "quantize": 32,
            "batch": 1, "rect": False, "augment": False, "max_det": 300, "classes": [0],
            "save": False, "save_txt": False, "save_conf": False, "save_crop": False,
            "show": False, "visualize": False, "verbose": False, "stream": False}


def decode_video(path: Path) -> Iterator[tuple[int, Fraction, np.ndarray]]:
    """Decode every frame of one local AVI video stream, with original timing."""
    import av

    try:
        with path.open("rb") as handle, av.open(handle, mode="r", format="avi") as container:
            if len(container.streams.video) != 1:
                raise ValueError("expected exactly one video stream")
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            stream.codec_context.options = {"err_detect": "explode"}
            count = 0
            previous: Fraction | None = None
            for frame in container.decode(stream):
                if frame.is_corrupt or frame.pts is None or frame.time_base is None:
                    raise ValueError("corrupt frame or missing presentation timestamp")
                base = Fraction(frame.time_base)
                timestamp = frame.pts * base
                if base <= 0 or (previous is not None and timestamp <= previous):
                    raise ValueError("invalid/nonincreasing presentation timestamp")
                previous = timestamp
                count += 1
                yield frame.pts, base, frame.to_ndarray(format="bgr24")
            if not count:
                raise ValueError("video contains no decodable frames")
            if stream.frames and count != stream.frames:
                raise ValueError("decoded frame count differs from declared stream count")
    except av.error.FFmpegError as exc:
        raise ValueError(f"video decoding failed: {path.name}") from exc


def load_predictor(
    weights: Path, spec: ExtractionSpec,
) -> Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]:
    """Load verified local weights and run fixed CPU inference in a dedicated process.

    Hash binding establishes file identity, not pretrained origin or accuracy.
    Only load trusted checkpoints; PyTorch model files may contain executable pickle.
    """
    from .ntu_io import _hash_file

    if weights.name != "yolo26n-pose.pt" or weights.is_symlink():
        raise ValueError("expected explicit regular yolo26n-pose.pt")
    if _hash_file(weights)[1] != spec.weights_sha256:
        raise ValueError("weight SHA-256 mismatch before model construction")
    if runtime_versions() != spec.expected_versions():
        raise ValueError("package versions differ from the extraction spec")
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    import torch
    from ultralytics import YOLO

    # Dedicated CLI/process settings. Never silently switch device or precision.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    model = YOLO(str(weights.absolute()), task="pose")
    if model.task != "pose" or tuple(model.model.yaml.get("kpt_shape", ())) != (17, 3):
        raise ValueError("checkpoint is not a 17-keypoint pose model")
    model.model.float().eval()

    def predict(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            results = model.predict(source=image, **prediction_options())
        if len(results) != 1:
            raise ValueError("single-frame inference returned an unexpected result count")
        result = results[0]
        if result.keypoints is None or result.boxes is None or result.boxes.is_track:
            raise ValueError("pose result missing keypoints/boxes or contains tracking state")
        if not bool((result.boxes.cls == 0).all()):
            raise ValueError("unexpected detector class in person-only pose extraction")
        return (result.keypoints.data.detach().cpu().numpy(),
                result.boxes.conf.detach().cpu().numpy())

    return predict
