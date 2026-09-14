from __future__ import annotations

import math

import numpy as np

from .schema import KeypointFrame, LabeledSequence, PointSequence


KEYPOINT_NAMES = (
    "torso",
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "pelvis",
    "right_wrist",
)
RIGHT_WRIST_INDEX = len(KEYPOINT_NAMES) - 1

_BASE_POINTS = np.asarray(
    [
        (0.00, 0.00),
        (-0.30, -0.20),
        (0.30, -0.20),
        (-0.55, 0.20),
        (0.00, 0.55),
        (0.55, 0.20),
    ],
    dtype=np.float64,
)

_LABELS = ("standing", "walking", "wave_left", "wave_right", "push", "pull")


def _motion(label: str, tau: float) -> np.ndarray:
    delta = np.zeros_like(_BASE_POINTS)
    if label == "walking":
        delta[:, 0] += 0.8 * tau
        swing = 0.08 * math.sin(2.0 * math.pi * tau)
        delta[3, 1] += swing
        delta[RIGHT_WRIST_INDEX, 1] -= swing
    elif label == "wave_left":
        delta[RIGHT_WRIST_INDEX, 0] -= 0.70 * tau
        delta[RIGHT_WRIST_INDEX, 1] += 0.10 * math.sin(4.0 * math.pi * tau)
    elif label == "wave_right":
        delta[RIGHT_WRIST_INDEX, 0] += 0.70 * tau
        delta[RIGHT_WRIST_INDEX, 1] += 0.10 * math.sin(4.0 * math.pi * tau)
    elif label == "push":
        delta[3, 1] -= 0.50 * tau
        delta[RIGHT_WRIST_INDEX, 1] -= 0.50 * tau
    elif label == "pull":
        delta[3, 1] += 0.50 * tau
        delta[RIGHT_WRIST_INDEX, 1] += 0.50 * tau
    elif label != "standing":
        raise ValueError(f"unknown synthetic behavior: {label}")
    return delta


def make_synthetic_dataset(
    seed: int,
    samples_per_class: int,
    frames: int,
) -> list[LabeledSequence]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    if frames < 2:
        raise ValueError("frames must be at least 2")

    rng = np.random.default_rng(seed)
    result: list[LabeledSequence] = []

    for label in _LABELS:
        for sample_index in range(samples_per_class):
            # Per-sample nuisance offset is label-independent. Class identity remains temporal.
            offset = rng.normal(0.0, 0.025, size=(1, 2))
            frame_values: list[KeypointFrame] = []
            for frame_index in range(frames):
                tau = frame_index / (frames - 1)
                jitter = np.clip(
                    rng.normal(0.0, 0.005, size=_BASE_POINTS.shape),
                    -0.012,
                    0.012,
                )
                xy = _BASE_POINTS + offset + _motion(label, tau) + jitter
                points = tuple((float(x), float(y), 1.0) for x, y in xy)
                frame_values.append(KeypointFrame(points=points))

            result.append(
                LabeledSequence(
                    sequence=PointSequence(frames=tuple(frame_values)),
                    label=label,
                    sample_id=f"{label}-{sample_index:04d}",
                )
            )

    return result
