from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import PointSequence


@dataclass(frozen=True)
class FeatureSpec:
    normalize_translation: bool = True
    normalize_scale: bool = True
    confidence_threshold: float = 0.05
    scale_epsilon: float = 1e-6


def _frame_geometry(sequence: PointSequence, spec: FeatureSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(sequence.frames, dtype=object)
    del raw  # PointSequence is converted explicitly below to avoid object-valued features.

    points = np.asarray(
        [[[x, y] for x, y, _ in frame.points] for frame in sequence.frames],
        dtype=np.float64,
    )
    confidence = np.asarray(
        [[c for _, _, c in frame.points] for frame in sequence.frames],
        dtype=np.float64,
    )
    mask = (confidence >= spec.confidence_threshold).astype(np.float64)

    if spec.normalize_translation:
        # Synthetic and V0 pose schema reserve point 0 as the torso anchor.
        points = points - points[:, :1, :]

    if spec.normalize_scale:
        if points.shape[1] < 3:
            raise ValueError("scale normalization requires torso and two shoulder keypoints")
        shoulder_span = np.linalg.norm(points[:, 1, :] - points[:, 2, :], axis=1)
        shoulder_span = np.maximum(shoulder_span, spec.scale_epsilon)
        points = points / shoulder_span[:, None, None]

    return points, confidence, mask


def encode_sequence(sequence: PointSequence, spec: FeatureSpec) -> np.ndarray:
    """Encode one frozen point sequence as [time, feature_dim].

    Each frame contains normalized xy coordinates, first-order xy velocity,
    detector confidence, and a low-confidence visibility mask. The encoder
    never consumes labels or sample identifiers.
    """

    points, confidence, mask = _frame_geometry(sequence, spec)
    velocity = np.zeros_like(points)
    if len(points) > 1:
        velocity[1:] = points[1:] - points[:-1]

    time_steps = points.shape[0]
    features = np.concatenate(
        (
            points.reshape(time_steps, -1),
            velocity.reshape(time_steps, -1),
            confidence,
            mask,
        ),
        axis=1,
    )
    return features.astype(np.float64, copy=False)
