from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


NTU25_PARENT = (
    -1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
)
_FEATURE_SPEC_ID = "ntu25-p-jm-b-bm-a-v1"
_EPSILON = 1e-6


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (25, 3) or value.shape[0] <= 0:
        raise ValueError("normalized skeleton must have shape [time,25,3]")
    if not np.isfinite(value).all():
        raise ValueError("normalized skeleton must be finite")
    return value


def kinematic_features(sequence: np.ndarray) -> np.ndarray:
    positions = _validate_sequence(sequence)
    joint_motion = np.zeros_like(positions)
    if positions.shape[0] > 1:
        joint_motion[1:] = positions[1:] - positions[:-1]

    bones = np.zeros_like(positions)
    for child, parent in enumerate(NTU25_PARENT):
        if parent >= 0:
            bones[:, child] = positions[:, child] - positions[:, parent]

    bone_motion = np.zeros_like(bones)
    if bones.shape[0] > 1:
        bone_motion[1:] = bones[1:] - bones[:-1]

    acceleration = np.zeros_like(joint_motion)
    if joint_motion.shape[0] > 1:
        acceleration[1:] = joint_motion[1:] - joint_motion[:-1]

    result = np.concatenate((positions, joint_motion, bones, bone_motion, acceleration), axis=2)
    if result.shape[2] != 15 or not np.isfinite(result).all():
        raise ValueError("kinematic feature construction failed")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class FeatureStandardizer:
    mean: np.ndarray
    std: np.ndarray
    epsilon: float = _EPSILON

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64).copy()
        std = np.asarray(self.std, dtype=np.float64).copy()
        if mean.shape != (25, 15) or std.shape != (25, 15):
            raise ValueError("feature statistics must have shape [25,15]")
        if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0.0):
            raise ValueError("feature statistics must be finite and non-negative")
        if not np.isfinite(float(self.epsilon)) or float(self.epsilon) <= 0.0:
            raise ValueError("standardization epsilon must be positive and finite")
        mean.setflags(write=False)
        std.setflags(write=False)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)
        object.__setattr__(self, "epsilon", float(self.epsilon))

    @classmethod
    def fit(cls, train_features: Iterable[np.ndarray], *, epsilon: float = _EPSILON) -> "FeatureStandardizer":
        sequences: list[np.ndarray] = []
        for features in train_features:
            value = np.asarray(features, dtype=np.float64)
            if value.ndim != 3 or value.shape[1:] != (25, 15) or value.shape[0] <= 0:
                raise ValueError("training features must have shape [time,25,15]")
            if not np.isfinite(value).all():
                raise ValueError("training features must be finite")
            sequences.append(value)
        if not sequences:
            raise ValueError("training features must be non-empty")
        stacked = np.concatenate(sequences, axis=0)
        return cls(mean=stacked.mean(axis=0), std=stacked.std(axis=0), epsilon=epsilon)

    def transform(self, features: np.ndarray) -> np.ndarray:
        value = np.asarray(features, dtype=np.float64)
        if value.ndim != 3 or value.shape[1:] != (25, 15) or value.shape[0] <= 0:
            raise ValueError("features must have shape [time,25,15]")
        if not np.isfinite(value).all():
            raise ValueError("features must be finite")
        result = (value - self.mean) / np.maximum(self.std, self.epsilon)
        if not np.isfinite(result).all():
            raise ValueError("standardized features must be finite")
        return result

    def _payload(self) -> dict[str, object]:
        return {
            "feature_spec_id": _FEATURE_SPEC_ID,
            "parent_map": list(NTU25_PARENT),
            "epsilon": self.epsilon,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    def to_json(self) -> str:
        return _canonical_json(self._payload())

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, raw: str) -> "FeatureStandardizer":
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid feature-statistics JSON") from exc
        if not isinstance(value, dict) or set(value) != {"feature_spec_id", "parent_map", "epsilon", "mean", "std"}:
            raise ValueError("feature-statistics schema mismatch")
        if value["feature_spec_id"] != _FEATURE_SPEC_ID or tuple(value["parent_map"]) != NTU25_PARENT:
            raise ValueError("feature-statistics specification mismatch")
        return cls(
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
            epsilon=float(value["epsilon"]),
        )
