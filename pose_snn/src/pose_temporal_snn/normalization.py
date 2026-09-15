from __future__ import annotations

import math

import numpy as np

from .ntu import NtuBody


_ROOT_JOINT = 0
_TORSO_JOINT = 20
_SCALE_EPSILON = 1e-6


def _interpolate_joint(positions: np.ndarray, usable: np.ndarray, joint: int) -> np.ndarray:
    valid = np.flatnonzero(usable[:, joint])
    if valid.size == 0:
        raise ValueError(f"NTU joint {joint} has no usable observation")
    frames = np.arange(positions.shape[0], dtype=np.float64)
    result = np.empty((positions.shape[0], 3), dtype=np.float64)
    for coordinate in range(3):
        result[:, coordinate] = np.interp(
            frames,
            valid.astype(np.float64, copy=False),
            positions[valid, joint, coordinate],
        )
    return result


def normalize_body(body: NtuBody, *, scale_epsilon: float = _SCALE_EPSILON) -> np.ndarray:
    if type(body) is not NtuBody:
        raise ValueError("body must be an NtuBody")
    if type(scale_epsilon) not in (int, float) or isinstance(scale_epsilon, bool):
        raise ValueError("scale_epsilon must be numeric")
    epsilon = float(scale_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("scale_epsilon must be positive and finite")

    positions = np.asarray(body.positions, dtype=np.float64)
    tracking = np.asarray(body.tracking_state, dtype=np.int64)
    present = np.asarray(body.present, dtype=bool)
    if positions.ndim != 3 or positions.shape[1:] != (25, 3):
        raise ValueError("NTU body positions must have shape [frames,25,3]")
    if tracking.shape != positions.shape[:2]:
        raise ValueError("NTU tracking state must have shape [frames,25]")
    if present.shape != (positions.shape[0],):
        raise ValueError("NTU body presence must have shape [frames]")
    if not np.isfinite(positions).all():
        raise ValueError("NTU body positions must be finite")

    usable = present[:, None] & (tracking >= 1)
    completed = np.empty_like(positions, dtype=np.float64)
    for joint in range(25):
        completed[:, joint] = _interpolate_joint(positions, usable, joint)

    torso = completed[:, _TORSO_JOINT] - completed[:, _ROOT_JOINT]
    torso_lengths = np.linalg.norm(torso, axis=1)
    if not np.isfinite(torso_lengths).all():
        raise ValueError("NTU torso scale must be finite")
    scale = float(np.median(torso_lengths))
    if not math.isfinite(scale) or scale <= epsilon:
        raise ValueError("NTU torso scale is degenerate")

    centered = completed - completed[:, _ROOT_JOINT : _ROOT_JOINT + 1]
    normalized = centered / scale
    if not np.isfinite(normalized).all():
        raise ValueError("normalized NTU body must be finite")
    return normalized
