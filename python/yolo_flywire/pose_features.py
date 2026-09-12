"""COCO17 geometry and motion per second, with explicit validity masks."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .schema import KeypointFrame, PointSequence


@dataclass(frozen=True)
class PoseFeatureSpec:
    """One explicit, hashable encoding policy; no alternate-anchor fallback."""

    confidence_threshold: float = 0.05
    scale_epsilon: float = 1e-6

    def __post_init__(self) -> None:
        threshold = _number(self.confidence_threshold, "confidence_threshold", confidence=True)
        epsilon = _number(self.scale_epsilon, "scale_epsilon")
        if threshold <= 0.0 or epsilon <= 0.0:
            raise ValueError("confidence_threshold and scale_epsilon must be positive")
        object.__setattr__(self, "confidence_threshold", threshold)
        object.__setattr__(self, "scale_epsilon", epsilon)

    def descriptor(self) -> dict[str, Any]:
        """Return a fresh JSON-compatible description, never mutable shared state."""
        return {
            "id": "coco17-timed-pose-features-v1", "keypoint_schema": "COCO17",
            "parameters": asdict(self), "dtype": "float64", "feature_dim": 121,
            "anchor_keypoints": [11, 12], "anchor_rule": "hip-midpoint-per-frame",
            "scale_keypoints": [5, 6], "scale_rule": "shoulder-distance-per-frame",
            "scale_epsilon_units": "pixels", "rotation_normalization": False,
            "position_rule": "(xy-hip-midpoint)/shoulder-distance",
            "position_mask_rule": "person>0; joint-and-all-four-references>=threshold; span>epsilon",
            "invalid_geometry_rule": "zero-value-and-zero-mask; retain-raw-confidence",
            "velocity_rule": "adjacent-normalized-position-difference/dt-seconds",
            "velocity_mask_rule": "both-adjacent-position-masks; first-frame-invalid",
            "timing_rule": "subtract-pts-times-rational-base-before-float64; first-dt-zero",
            "frame_policy": "all-frames; no-interpolation-resampling-clipping-or-gap-bridging",
            "layout": [
                {"name": "position_xy", "start": 0, "stop": 34},
                {"name": "velocity_xy_per_second", "start": 34, "stop": 68},
                {"name": "joint_confidence", "start": 68, "stop": 85},
                {"name": "position_mask", "start": 85, "stop": 102},
                {"name": "velocity_mask", "start": 102, "stop": 119},
                {"name": "detector_confidence", "start": 119, "stop": 120},
                {"name": "dt_seconds", "start": 120, "stop": 121},
            ],
        }


def _number(value: Any, context: str, *, confidence: bool = False) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{context} must be a finite int/float, not a coerced value")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError(f"{context} exceeds finite float64 range") from exc
    if not math.isfinite(result) or (confidence and not 0.0 <= result <= 1.0):
        raise ValueError(f"{context} is non-finite or outside its allowed range")
    return result


def _tuple_length(value: Any, count: int, context: str) -> None:
    if type(value) is not tuple or len(value) != count:
        raise ValueError(f"{context} must be a tuple with exactly {count} entries")


def _observations(sequence: PointSequence) -> np.ndarray:
    if (type(sequence) is not PointSequence or type(sequence.frames) is not tuple
            or not sequence.frames):
        raise ValueError("expected a nonempty immutable PointSequence")
    values = np.empty((len(sequence.frames), 17, 3), dtype=np.float64)
    for index, frame in enumerate(sequence.frames):
        if type(frame) is not KeypointFrame:
            raise ValueError("expected KeypointFrame observations")
        _tuple_length(frame.points, 17, "COCO17 keypoints")
        for joint, point in enumerate(frame.points):
            _tuple_length(point, 3, "keypoint")
            values[index, joint] = (
                _number(point[0], "x"), _number(point[1], "y"),
                _number(point[2], "joint confidence", confidence=True),
            )
    return values


def _intervals(pts: tuple[int, ...], time_bases: tuple[Fraction, ...], count: int) -> np.ndarray:
    _tuple_length(pts, count, "PTS")
    _tuple_length(time_bases, count, "time_bases")
    intervals = np.zeros(count, dtype=np.float64)
    previous: Fraction | None = None
    for index, (value, base) in enumerate(zip(pts, time_bases, strict=True)):
        if type(value) is not int or type(base) is not Fraction or base <= 0:
            raise ValueError("timing requires integer PTS and positive Fraction time bases")
        timestamp = value * base
        if previous is not None:
            difference = timestamp - previous
            if difference <= 0:
                raise ValueError("presentation timestamps must strictly increase")
            try:
                seconds = float(difference)
            except OverflowError as exc:
                raise ValueError("frame interval exceeds float64 range") from exc
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError("frame interval is not representable as positive finite float64")
            intervals[index] = seconds
        previous = timestamp
    return intervals


def encode_timed_pose(
    sequence: PointSequence, *, pts: tuple[int, ...], time_bases: tuple[Fraction, ...],
    detector_confidences: tuple[float, ...], spec: PoseFeatureSpec,
) -> np.ndarray:
    """Encode only observations as a fresh [frames, 121] float64 array.

    File callers must first use load_development_bundle for provenance and split
    verification. This pure function has no labels, IDs, subjects, paths or split
    inputs and does not authenticate a directly constructed PointSequence.
    Velocity is change in framewise body-relative coordinates per second, not
    physical/world velocity. Invalid observations remain as explicitly masked
    rows; no time, reference geometry or missing joint is inferred or repaired.
    """
    if type(spec) is not PoseFeatureSpec:
        raise ValueError("an explicit PoseFeatureSpec is required")
    raw = _observations(sequence)
    count = len(raw)
    intervals = _intervals(pts, time_bases, count)
    _tuple_length(detector_confidences, count, "detector_confidences")
    person = np.array([_number(value, "detector confidence", confidence=True)
                       for value in detector_confidences], dtype=np.float64)
    if np.any(raw[person == 0.0] != 0.0):
        raise ValueError("missing person requires all-zero geometry and joint confidence")
    confidence = raw[:, :, 2]
    visible = (confidence >= spec.confidence_threshold) & (person[:, None] > 0.0)
    position = np.zeros((count, 17, 2), dtype=np.float64)
    velocity = np.zeros_like(position)
    position_mask = np.zeros((count, 17), dtype=bool)
    velocity_mask = np.zeros_like(position_mask)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            for index in range(count):
                if not visible[index, [5, 6, 11, 12]].all():
                    continue
                xy = raw[index, :, :2]
                shoulder = xy[5] - xy[6]
                span = np.hypot(shoulder[0], shoulder[1])
                if span <= spec.scale_epsilon:
                    continue
                # Halve before adding to avoid overflowing a representable midpoint.
                anchor = xy[11] * 0.5 + xy[12] * 0.5
                valid = visible[index]
                # Never perform geometry arithmetic on low-confidence coordinates.
                position[index, valid] = (xy[valid] - anchor) / span
                position_mask[index] = valid
            for index in range(1, count):
                valid = position_mask[index] & position_mask[index - 1]
                velocity[index, valid] = (position[index, valid] - position[index - 1, valid]) / intervals[index]
                velocity_mask[index] = valid
    except FloatingPointError as exc:
        raise ValueError("pose geometry or velocity exceeds finite float64 range") from exc
    features = np.concatenate((position.reshape(count, 34), velocity.reshape(count, 34),
                               confidence, position_mask, velocity_mask,
                               person[:, None], intervals[:, None]), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("encoded features must all be finite")
    return features


def pose_encoder_hash(spec: PoseFeatureSpec) -> str:
    """Bind policy and actual local source bytes, not an entire execution runtime."""
    if type(spec) is not PoseFeatureSpec:
        raise ValueError("an explicit PoseFeatureSpec is required")
    directory = Path(__file__).parent
    payload = {"descriptor": spec.descriptor(), "source_sha256": {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in ("pose_features.py", "schema.py")
    }}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
