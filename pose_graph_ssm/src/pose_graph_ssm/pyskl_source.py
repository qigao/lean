from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np

from .ntu import split_for_subject
from .protocol import ExperimentProtocol


_ID = re.compile(r"^S([0-9]{3})C([0-9]{3})P([0-9]{3})R([0-9]{3})A([0-9]{3})$")
_SOURCE_KIND = "pyskl-ntu120-3danno-v1"
_PADDING_EPSILON = 1e-12
_SCALE_EPSILON = 1e-6


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    _ALLOWED: dict[tuple[str, str], object] = {
        ("numpy", "ndarray"): np.ndarray,
        ("numpy", "dtype"): np.dtype,
        ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
        ("numpy.core.multiarray", "scalar"): np.core.multiarray.scalar,
        ("numpy._core.multiarray", "scalar"): np.core.multiarray.scalar,
    }

    def find_class(self, module: str, name: str) -> object:
        value = self._ALLOWED.get((module, name))
        if value is None:
            raise pickle.UnpicklingError(f"forbidden PYSKL pickle global: {module}.{name}")
        return value


def _restricted_load(raw: bytes) -> object:
    try:
        return _RestrictedNumpyUnpickler(io.BytesIO(raw)).load()
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError, TypeError) as exc:
        raise ValueError("could not safely decode PYSKL annotation pickle") from exc


def _parse_id(sample_id: object) -> tuple[str, int, int]:
    match = _ID.fullmatch(sample_id) if isinstance(sample_id, str) else None
    if match is None:
        raise ValueError(f"noncanonical PYSKL NTU sample ID: {sample_id!r}")
    setup, camera, subject, repetition, action = (int(value) for value in match.groups())
    for field, value, maximum in (
        ("setup", setup, 32),
        ("camera", camera, 3),
        ("subject", subject, 106),
        ("repetition", repetition, 2),
        ("action", action, 120),
    ):
        if not 1 <= value <= maximum:
            raise ValueError(f"PYSKL NTU {field} must be in 1..{maximum}")
    return sample_id, subject, action


def _interpolate_zero_padding(positions: np.ndarray) -> np.ndarray:
    value = np.asarray(positions, dtype=np.float64).copy()
    if value.ndim != 3 or value.shape[1:] != (25, 3) or value.shape[0] <= 0:
        raise ValueError("PYSKL primary positions must have shape [time,25,3]")
    if not np.isfinite(value).all():
        raise ValueError("PYSKL primary positions must be finite")
    padding = np.sum(np.abs(value), axis=(1, 2)) <= _PADDING_EPSILON
    valid = np.flatnonzero(~padding)
    if valid.size == 0:
        raise ValueError("PYSKL primary trajectory contains no nonzero frame")
    if padding.any():
        frames = np.arange(value.shape[0], dtype=np.float64)
        valid_float = valid.astype(np.float64, copy=False)
        for joint in range(25):
            for axis in range(3):
                value[:, joint, axis] = np.interp(frames, valid_float, value[valid, joint, axis])
    return value


def _normalize_processed_positions(positions: np.ndarray) -> np.ndarray:
    completed = _interpolate_zero_padding(positions)
    torso = completed[:, 20] - completed[:, 0]
    lengths = np.linalg.norm(torso, axis=1)
    positive = lengths[lengths > _SCALE_EPSILON]
    if positive.size == 0:
        raise ValueError("PYSKL torso scale is degenerate")
    scale = float(np.median(positive))
    if not np.isfinite(scale) or scale <= _SCALE_EPSILON:
        raise ValueError("PYSKL torso scale is degenerate")
    normalized = (completed - completed[:, 0:1]) / scale
    if not np.isfinite(normalized).all():
        raise ValueError("normalized PYSKL skeleton must be finite")
    return normalized


@dataclass(frozen=True)
class PysklSample:
    sample_id: str
    subject: int
    action: int
    split: str
    positions: np.ndarray

    def __post_init__(self) -> None:
        value = np.asarray(self.positions, dtype=np.float64).copy()
        if value.ndim != 3 or value.shape[1:] != (25, 3) or value.shape[0] <= 0:
            raise ValueError("PYSKL sample positions must have shape [time,25,3]")
        if not np.isfinite(value).all():
            raise ValueError("PYSKL sample positions must be finite")
        value.setflags(write=False)
        object.__setattr__(self, "positions", value)


@dataclass(frozen=True)
class PysklSource:
    source_sha256: str
    samples: tuple[PysklSample, ...]


def read_pyskl_3d_annotations(path: str | Path, protocol: ExperimentProtocol) -> PysklSource:
    if type(protocol) is not ExperimentProtocol:
        raise ValueError("protocol must be an ExperimentProtocol")
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as exc:
        raise ValueError("could not read PYSKL annotation file") from exc
    if not raw:
        raise ValueError("PYSKL annotation file is empty")
    payload = _restricted_load(raw)
    if type(payload) is not dict or set(payload) != {"split", "annotations"}:
        raise ValueError("PYSKL annotation root must contain exactly split and annotations")
    if type(payload["split"]) is not dict:
        raise ValueError("PYSKL split field must be a dictionary")
    annotations = payload["annotations"]
    if type(annotations) is not list:
        raise ValueError("PYSKL annotations field must be a list")

    selected: list[PysklSample] = []
    seen: set[str] = set()
    for annotation in annotations:
        if type(annotation) is not dict or set(annotation) != {"frame_dir", "label", "keypoint", "total_frames"}:
            raise ValueError("PYSKL 3D annotation schema mismatch")
        sample_id, subject, action = _parse_id(annotation["frame_dir"])
        if action not in protocol.actions:
            continue
        if sample_id in seen:
            raise ValueError(f"duplicate PYSKL sample ID: {sample_id}")
        seen.add(sample_id)
        label = annotation["label"]
        if type(label) is not int or label != action - 1:
            raise ValueError(f"PYSKL label differs from NTU action identity: {sample_id}")
        total_frames = annotation["total_frames"]
        if type(total_frames) is not int or total_frames <= 0:
            raise ValueError("PYSKL total_frames must be a positive integer")
        keypoint = np.asarray(annotation["keypoint"])
        if keypoint.ndim != 4 or keypoint.shape[0] <= 0 or keypoint.shape[2:] != (25, 3):
            raise ValueError("PYSKL keypoint shape must be [persons,time,25,3]")
        if keypoint.shape[1] != total_frames:
            raise ValueError("PYSKL total_frames differs from keypoint time dimension")
        if not np.issubdtype(keypoint.dtype, np.number) or not np.isfinite(keypoint).all():
            raise ValueError("PYSKL keypoints must be finite numeric values")
        primary = _interpolate_zero_padding(keypoint[0])
        selected.append(
            PysklSample(
                sample_id=sample_id,
                subject=subject,
                action=action,
                split=split_for_subject(subject, protocol),
                positions=primary,
            )
        )
    if not selected:
        raise ValueError("PYSKL source contains no samples for frozen action roster")
    selected.sort(key=lambda sample: sample.sample_id)
    return PysklSource(source_sha256=hashlib.sha256(raw).hexdigest(), samples=tuple(selected))


def prepare_pyskl_data(path: str | Path, protocol: ExperimentProtocol):
    # Keep public-source validation NumPy-only. Training/evaluation dependencies are
    # imported only when this preparation path is actually requested.
    from .data import PreparedData, PreparedSample, PreparationBinding, _check_optional_pin
    from .evaluation import length_quartile_boundaries
    from .features import FeatureStandardizer, kinematic_features
    from .graph import adjacency_fingerprint
    from .models.selective_ssm import ssm_spec_fingerprint

    source = read_pyskl_3d_annotations(path, protocol)
    label_map = tuple((action, index) for index, action in enumerate(protocol.actions))
    label_lookup = dict(label_map)
    raw_train: list[tuple[PysklSample, np.ndarray]] = []
    raw_validation: list[tuple[PysklSample, np.ndarray]] = []
    final_inventory: list[dict[str, Any]] = []

    for sample in source.samples:
        if sample.split == "final_test":
            final_inventory.append(
                {
                    "sample_id": sample.sample_id,
                    "subject": sample.subject,
                    "action": sample.action,
                    "split": "final_test",
                    "frame_count": int(sample.positions.shape[0]),
                    "source_kind": _SOURCE_KIND,
                }
            )
            continue
        normalized = _normalize_processed_positions(sample.positions)
        features = kinematic_features(normalized)
        target = raw_train if sample.split == "train" else raw_validation
        target.append((sample, features))

    if not raw_train or not raw_validation:
        raise ValueError("processed NTU preparation requires non-empty train and validation splits")
    standardizer = FeatureStandardizer.fit(
        (features for _, features in raw_train), epsilon=protocol.standardization_epsilon
    )

    def materialize(rows: list[tuple[PysklSample, np.ndarray]]) -> tuple[PreparedSample, ...]:
        return tuple(
            PreparedSample(
                sample_id=sample.sample_id,
                label=label_lookup[sample.action],
                features=standardizer.transform(features),
            )
            for sample, features in rows
        )

    train_samples = materialize(raw_train)
    validation_samples = materialize(raw_validation)
    quartiles = length_quartile_boundaries([sample.features.shape[0] for sample in train_samples])
    assignments = [{"sample_id": sample.sample_id, "split": sample.split} for sample in source.samples]
    split_hash = _hash_json(
        {
            "source_sha256": source.source_sha256,
            "source_kind": _SOURCE_KIND,
            "actions": list(protocol.actions),
            "primary_track": "pyskl-person-0-motion-ranked",
            "zero_padding": "whole-frame-linear-interpolation",
            "normalization": "root0-median-torso0-20",
            "assignments": assignments,
        }
    )
    binding = PreparationBinding(
        dataset_content_hash=source.source_sha256,
        split_hash=split_hash,
        feature_stats_hash=standardizer.fingerprint,
        label_map=label_map,
        label_map_hash=PreparationBinding.hash_label_map(label_map),
        adjacency_hash=adjacency_fingerprint(),
        ssm_spec_hash=ssm_spec_fingerprint(),
        length_quartiles=quartiles,
    )

    _check_optional_pin("dataset_content_hash", protocol.dataset_content_hash, binding.dataset_content_hash)
    _check_optional_pin("split_hash", protocol.split_hash, binding.split_hash)
    _check_optional_pin("feature_stats_hash", protocol.feature_stats_hash, binding.feature_stats_hash)
    _check_optional_pin("label_map_hash", protocol.label_map_hash, binding.label_map_hash)
    _check_optional_pin("adjacency_hash", protocol.adjacency_hash, binding.adjacency_hash)
    _check_optional_pin("ssm_spec_hash", protocol.ssm_spec_hash, binding.ssm_spec_hash)
    _check_optional_pin("length_quartiles", protocol.length_quartiles, binding.length_quartiles)

    return PreparedData(
        binding=binding,
        standardizer=standardizer,
        train_samples=train_samples,
        validation_samples=validation_samples,
        final_test_inventory=tuple(final_inventory),
    )
