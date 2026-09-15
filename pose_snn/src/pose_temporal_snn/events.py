from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Iterable

import numpy as np


NTU25_PARENT = (
    -1, 0, 20, 2, 20, 4, 5, 6, 20, 8, 9, 10,
    0, 12, 13, 14, 0, 16, 17, 18, 1, 7, 7, 11, 11,
)
_FEATURE_SPEC_ID = "ntu25-jm-bm-accel-signed-events-v1"
_THRESHOLD_FLOOR = 1e-6


@dataclass(frozen=True)
class EventSequence:
    flat: np.ndarray
    nodes: np.ndarray

    def __post_init__(self) -> None:
        flat = np.asarray(self.flat, dtype=np.float32).copy()
        nodes = np.asarray(self.nodes, dtype=np.float32).copy()
        if flat.ndim != 2 or flat.shape[1] != 225:
            raise ValueError("flat events must have shape [time,225]")
        if nodes.shape != (flat.shape[0], 25, 9):
            raise ValueError("node events must have shape [time,25,9]")
        if not np.isfinite(flat).all() or not np.isfinite(nodes).all():
            raise ValueError("events must be finite")
        allowed_flat = np.isin(flat, (-1.0, 0.0, 1.0))
        allowed_nodes = np.isin(nodes, (-1.0, 0.0, 1.0))
        if not allowed_flat.all() or not allowed_nodes.all():
            raise ValueError("events must contain only -1, 0, or +1")
        flat.setflags(write=False)
        nodes.setflags(write=False)
        object.__setattr__(self, "flat", flat)
        object.__setattr__(self, "nodes", nodes)


def _validate_sequence(sequence: np.ndarray) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float64)
    if value.ndim != 3 or value.shape[1:] != (25, 3) or value.shape[0] <= 0:
        raise ValueError("normalized skeleton sequence must have shape [time,25,3]")
    if not np.isfinite(value).all():
        raise ValueError("normalized skeleton sequence must be finite")
    return value


def _features(sequence: np.ndarray) -> np.ndarray:
    positions = _validate_sequence(sequence)
    time_steps = positions.shape[0]

    joint_motion = np.zeros_like(positions)
    if time_steps > 1:
        joint_motion[1:] = positions[1:] - positions[:-1]

    bones = np.zeros_like(positions)
    for child, parent in enumerate(NTU25_PARENT):
        if parent >= 0:
            bones[:, child] = positions[:, child] - positions[:, parent]

    bone_motion = np.zeros_like(bones)
    if time_steps > 1:
        bone_motion[1:] = bones[1:] - bones[:-1]

    acceleration = np.zeros_like(joint_motion)
    if time_steps > 1:
        acceleration[1:] = joint_motion[1:] - joint_motion[:-1]

    result = np.concatenate((joint_motion, bone_motion, acceleration), axis=2)
    if result.shape != (time_steps, 25, 9) or not np.isfinite(result).all():
        raise ValueError("kinematic event features are invalid")
    return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class EventEncoder:
    thresholds: np.ndarray
    quantile: float

    def __post_init__(self) -> None:
        thresholds = np.asarray(self.thresholds, dtype=np.float64).copy()
        if thresholds.shape != (225,):
            raise ValueError("event thresholds must have shape [225]")
        if not np.isfinite(thresholds).all() or np.any(thresholds < _THRESHOLD_FLOOR):
            raise ValueError("event thresholds must be finite and >= 1e-6")
        if type(self.quantile) not in (int, float) or isinstance(self.quantile, bool):
            raise ValueError("event quantile must be numeric")
        quantile = float(self.quantile)
        if not math.isfinite(quantile) or not 0.0 < quantile < 1.0:
            raise ValueError("event quantile must be inside (0,1)")
        thresholds.setflags(write=False)
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "quantile", quantile)

    @classmethod
    def fit(
        cls,
        train_sequences: Iterable[np.ndarray],
        *,
        quantile: float = 0.75,
    ) -> "EventEncoder":
        if type(quantile) not in (int, float) or isinstance(quantile, bool):
            raise ValueError("event quantile must be numeric")
        q = float(quantile)
        if not math.isfinite(q) or not 0.0 < q < 1.0:
            raise ValueError("event quantile must be inside (0,1)")
        sequences = tuple(train_sequences)
        if not sequences:
            raise ValueError("event encoder requires non-empty training sequences")
        flat_features = [_features(sequence).reshape(-1, 225) for sequence in sequences]
        stacked = np.concatenate(flat_features, axis=0)
        thresholds = np.quantile(np.abs(stacked), q, axis=0)
        thresholds = np.maximum(thresholds, _THRESHOLD_FLOOR)
        return cls(thresholds=thresholds, quantile=q)

    @classmethod
    def fixed_for_test(cls, threshold: float) -> "EventEncoder":
        if type(threshold) not in (int, float) or isinstance(threshold, bool):
            raise ValueError("test threshold must be numeric")
        value = float(threshold)
        if not math.isfinite(value) or value < _THRESHOLD_FLOOR:
            raise ValueError("test threshold must be finite and >= 1e-6")
        return cls(np.full((225,), value, dtype=np.float64), 0.75)

    def transform(self, sequence: np.ndarray) -> EventSequence:
        features = _features(sequence)
        flat_features = features.reshape(features.shape[0], 225)
        events = np.zeros_like(flat_features, dtype=np.float32)
        events[flat_features > self.thresholds[None, :]] = 1.0
        events[flat_features < -self.thresholds[None, :]] = -1.0
        nodes = events.reshape(events.shape[0], 25, 9)
        return EventSequence(flat=events, nodes=nodes)

    def _payload(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "kind": "pose_temporal_snn_event_encoder",
            "feature_spec_id": _FEATURE_SPEC_ID,
            "quantile": self.quantile,
            "threshold_floor": _THRESHOLD_FLOOR,
            "parent_map": list(NTU25_PARENT),
            "thresholds": [float(value) for value in self.thresholds],
        }

    def to_json(self) -> str:
        return _canonical_json(self._payload()) + "\n"

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "EventEncoder":
        try:
            raw = json.loads(text)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("could not parse event encoder JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError("event encoder JSON must be an object")
        expected_keys = {
            "format_version",
            "kind",
            "feature_spec_id",
            "quantile",
            "threshold_floor",
            "parent_map",
            "thresholds",
        }
        if set(raw) != expected_keys:
            raise ValueError("event encoder schema mismatch")
        if raw["format_version"] != 1 or raw["kind"] != "pose_temporal_snn_event_encoder":
            raise ValueError("event encoder identity mismatch")
        if raw["feature_spec_id"] != _FEATURE_SPEC_ID:
            raise ValueError("event encoder feature spec mismatch")
        if raw["threshold_floor"] != _THRESHOLD_FLOOR:
            raise ValueError("event encoder threshold floor mismatch")
        if tuple(raw["parent_map"]) != NTU25_PARENT:
            raise ValueError("event encoder parent map mismatch")
        return cls(np.asarray(raw["thresholds"], dtype=np.float64), float(raw["quantile"]))
