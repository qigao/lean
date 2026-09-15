from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable

import numpy as np


# Canonical NTU RGB+D 25-joint kinematic tree, zero-based as (child, parent).
NTU25_BONES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 20),
    (2, 20),
    (3, 2),
    (4, 20),
    (5, 4),
    (6, 5),
    (7, 6),
    (8, 20),
    (9, 8),
    (10, 9),
    (11, 10),
    (12, 0),
    (13, 12),
    (14, 13),
    (15, 14),
    (16, 0),
    (17, 16),
    (18, 17),
    (19, 18),
    (21, 7),
    (22, 7),
    (23, 11),
    (24, 11),
)

_FORMAT_VERSION = 1
_THRESHOLD_EPSILON = 1e-6


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 3:
        raise ValueError("kinematic sequence must have shape [time,joints,coords]")
    if value.shape[0] <= 0 or value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError("kinematic sequence dimensions must be positive")
    if not np.isfinite(value).all():
        raise ValueError("kinematic sequence values must be finite")
    return value


def _validate_bones(bones: tuple[tuple[int, int], ...], joint_count: int) -> None:
    seen: set[tuple[int, int]] = set()
    for child, parent in bones:
        if type(child) is not int or type(parent) is not int:
            raise ValueError("bone indices must be integers")
        if child == parent or child < 0 or parent < 0 or child >= joint_count or parent >= joint_count:
            raise ValueError("bone index is outside sequence joint range")
        if (child, parent) in seen:
            raise ValueError("duplicate bone edge")
        seen.add((child, parent))


def _difference(value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value)
    if value.shape[0] > 1:
        result[1:] = value[1:] - value[:-1]
    return result


def _kinematic_features(sequence: np.ndarray, bones: tuple[tuple[int, int], ...]) -> np.ndarray:
    value = _validate_sequence(sequence)
    _validate_bones(bones, value.shape[1])

    joint_motion = _difference(value)
    joint_acceleration = _difference(joint_motion)

    if bones:
        child = np.asarray([edge[0] for edge in bones], dtype=np.int64)
        parent = np.asarray([edge[1] for edge in bones], dtype=np.int64)
        bone = value[:, child, :] - value[:, parent, :]
        bone_motion = _difference(bone)
    else:
        bone_motion = np.zeros((value.shape[0], 0, value.shape[2]), dtype=np.float64)

    return np.concatenate(
        (
            joint_motion.reshape(value.shape[0], -1),
            bone_motion.reshape(value.shape[0], -1),
            joint_acceleration.reshape(value.shape[0], -1),
        ),
        axis=1,
    )


@dataclass(frozen=True)
class KinematicEventEncoder:
    thresholds: np.ndarray
    bones: tuple[tuple[int, int], ...] = NTU25_BONES
    quantile: float | None = None
    broadcast_threshold: bool = False

    def __post_init__(self) -> None:
        thresholds = np.asarray(self.thresholds, dtype=np.float64).reshape(-1).copy()
        if thresholds.size == 0 or not np.isfinite(thresholds).all() or np.any(thresholds <= 0.0):
            raise ValueError("event thresholds must be positive finite values")
        if type(self.broadcast_threshold) is not bool:
            raise ValueError("broadcast_threshold must be a boolean")
        if self.broadcast_threshold and thresholds.size != 1:
            raise ValueError("broadcast threshold mode requires one scalar threshold")
        if self.quantile is not None:
            if not math.isfinite(self.quantile) or not 0.0 < self.quantile < 1.0:
                raise ValueError("threshold quantile must be finite and inside (0,1)")
        thresholds.setflags(write=False)
        object.__setattr__(self, "thresholds", thresholds)

    @property
    def feature_dim(self) -> int:
        return int(self.thresholds.size)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    @classmethod
    def fixed_for_test(cls, *, threshold: float) -> "KinematicEventEncoder":
        if type(threshold) not in (int, float) or isinstance(threshold, bool):
            raise ValueError("test threshold must be numeric")
        value = float(threshold)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("test threshold must be positive and finite")
        # Test fixtures may intentionally use one-joint sequences, so no NTU bone
        # topology is imposed in this explicit scalar-threshold helper.
        return cls(
            thresholds=np.asarray([value], dtype=np.float64),
            bones=(),
            quantile=None,
            broadcast_threshold=True,
        )

    @classmethod
    def fit(
        cls,
        train_sequences: Iterable[np.ndarray],
        *,
        quantile: float,
        bones: tuple[tuple[int, int], ...] = NTU25_BONES,
        threshold_epsilon: float = _THRESHOLD_EPSILON,
    ) -> "KinematicEventEncoder":
        if type(quantile) not in (int, float) or isinstance(quantile, bool):
            raise ValueError("threshold quantile must be numeric")
        q = float(quantile)
        if not math.isfinite(q) or not 0.0 < q < 1.0:
            raise ValueError("threshold quantile must be finite and inside (0,1)")
        if type(threshold_epsilon) not in (int, float) or isinstance(threshold_epsilon, bool):
            raise ValueError("threshold epsilon must be numeric")
        epsilon = float(threshold_epsilon)
        if not math.isfinite(epsilon) or epsilon <= 0.0:
            raise ValueError("threshold epsilon must be positive and finite")

        sequences = list(train_sequences)
        if not sequences:
            raise ValueError("event encoder training sequences must be non-empty")

        encoded: list[np.ndarray] = []
        feature_dim: int | None = None
        for sequence in sequences:
            features = _kinematic_features(sequence, bones)
            if feature_dim is None:
                feature_dim = int(features.shape[1])
            elif features.shape[1] != feature_dim:
                raise ValueError("event encoder training sequences have inconsistent feature dimensions")
            encoded.append(features)

        stacked = np.concatenate(encoded, axis=0)
        thresholds = np.quantile(np.abs(stacked), q, axis=0)
        thresholds = np.maximum(thresholds, epsilon).astype(np.float64, copy=False)
        if not np.isfinite(thresholds).all() or np.any(thresholds <= 0.0):
            raise ValueError("fitted event thresholds must be positive and finite")
        return cls(thresholds=thresholds, bones=bones, quantile=q, broadcast_threshold=False)

    def transform(self, sequence: np.ndarray) -> np.ndarray:
        features = _kinematic_features(sequence, self.bones)
        if self.broadcast_threshold:
            thresholds = np.full(features.shape[1], float(self.thresholds[0]), dtype=np.float64)
        else:
            if features.shape[1] != self.thresholds.size:
                raise ValueError(
                    f"event feature dimension mismatch: expected {self.thresholds.size}, got {features.shape[1]}"
                )
            thresholds = self.thresholds

        events = np.zeros(features.shape, dtype=np.float32)
        events[features > thresholds] = 1.0
        events[features < -thresholds] = -1.0
        return events

    def _payload(self) -> dict[str, object]:
        return {
            "format_version": _FORMAT_VERSION,
            "kind": "cx_kinematic_event_encoder",
            "feature_streams": ["joint_motion", "bone_motion", "joint_acceleration"],
            "bones": [list(edge) for edge in self.bones],
            "quantile": self.quantile,
            "broadcast_threshold": self.broadcast_threshold,
            "thresholds": [float(value) for value in self.thresholds],
        }

    def to_json(self) -> str:
        return _canonical_json(self._payload()) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "KinematicEventEncoder":
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("invalid event encoder JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("event encoder JSON must contain an object")
        expected_keys = {
            "format_version",
            "kind",
            "feature_streams",
            "bones",
            "quantile",
            "broadcast_threshold",
            "thresholds",
        }
        if set(value) != expected_keys:
            raise ValueError("event encoder JSON schema mismatch")
        if value["format_version"] != _FORMAT_VERSION or value["kind"] != "cx_kinematic_event_encoder":
            raise ValueError("unsupported event encoder identity")
        if value["feature_streams"] != ["joint_motion", "bone_motion", "joint_acceleration"]:
            raise ValueError("event encoder feature streams changed")
        raw_bones = value["bones"]
        if not isinstance(raw_bones, list):
            raise ValueError("event encoder bones must be a list")
        bones: list[tuple[int, int]] = []
        for edge in raw_bones:
            if (
                not isinstance(edge, list)
                or len(edge) != 2
                or type(edge[0]) is not int
                or type(edge[1]) is not int
            ):
                raise ValueError("event encoder bones contain a malformed edge")
            bones.append((edge[0], edge[1]))
        raw_thresholds = value["thresholds"]
        if not isinstance(raw_thresholds, list):
            raise ValueError("event encoder thresholds must be a list")
        try:
            thresholds = np.asarray(raw_thresholds, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("event encoder thresholds are malformed") from exc
        quantile = value["quantile"]
        if quantile is not None and type(quantile) not in (int, float):
            raise ValueError("event encoder quantile is malformed")
        return cls(
            thresholds=thresholds,
            bones=tuple(bones),
            quantile=None if quantile is None else float(quantile),
            broadcast_threshold=value["broadcast_threshold"],
        )
