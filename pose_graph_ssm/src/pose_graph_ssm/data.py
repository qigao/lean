from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .evaluation import length_quartile_boundaries
from .features import FeatureStandardizer, kinematic_features
from .graph import adjacency_fingerprint
from .models.selective_ssm import ssm_spec_fingerprint
from .normalization import normalize_body
from .ntu import build_manifest, parse_skeleton_file, select_primary_body
from .protocol import ExperimentProtocol


_DIGEST = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_digest(name: str, value: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True)
class PreparedSample:
    sample_id: str
    label: int
    features: np.ndarray

    def __post_init__(self) -> None:
        if type(self.sample_id) is not str or not self.sample_id:
            raise ValueError("prepared sample_id must be non-empty")
        if type(self.label) is not int or self.label < 0:
            raise ValueError("prepared sample label must be a non-negative integer")
        features = np.asarray(self.features, dtype=np.float64).copy()
        if features.ndim != 3 or features.shape[1:] != (25, 15) or features.shape[0] <= 0:
            raise ValueError("prepared features must have shape [time,25,15]")
        if not np.isfinite(features).all():
            raise ValueError("prepared features must be finite")
        features.setflags(write=False)
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class PreparationBinding:
    dataset_content_hash: str
    split_hash: str
    feature_stats_hash: str
    label_map: tuple[tuple[int, int], ...]
    label_map_hash: str
    adjacency_hash: str
    ssm_spec_hash: str
    length_quartiles: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name in (
            "dataset_content_hash",
            "split_hash",
            "feature_stats_hash",
            "label_map_hash",
            "adjacency_hash",
            "ssm_spec_hash",
        ):
            _require_digest(name, getattr(self, name))
        if not self.label_map:
            raise ValueError("label map must be non-empty")
        actions = tuple(action for action, _ in self.label_map)
        labels = tuple(label for _, label in self.label_map)
        if any(type(action) is not int or action <= 0 for action in actions):
            raise ValueError("label-map action IDs must be positive integers")
        if labels != tuple(range(len(labels))):
            raise ValueError("label-map labels must be contiguous from zero")
        if len(set(actions)) != len(actions):
            raise ValueError("label-map action IDs must be unique")
        if self.label_map_hash != self.hash_label_map(self.label_map):
            raise ValueError("label-map hash does not match label map")
        quartiles = self.length_quartiles
        if (
            type(quartiles) is not tuple
            or len(quartiles) != 3
            or any(type(value) is not int or value <= 0 for value in quartiles)
            or not quartiles[0] < quartiles[1] < quartiles[2]
        ):
            raise ValueError("length quartiles must be three strictly increasing positive integers")

    @staticmethod
    def hash_label_map(label_map: tuple[tuple[int, int], ...]) -> str:
        payload = [{"action": action, "label": label} for action, label in label_map]
        return _sha256_json(payload)

    @property
    def fingerprint(self) -> str:
        return _sha256_json(
            {
                "dataset_content_hash": self.dataset_content_hash,
                "split_hash": self.split_hash,
                "feature_stats_hash": self.feature_stats_hash,
                "label_map": [list(pair) for pair in self.label_map],
                "label_map_hash": self.label_map_hash,
                "adjacency_hash": self.adjacency_hash,
                "ssm_spec_hash": self.ssm_spec_hash,
                "length_quartiles": list(self.length_quartiles),
            }
        )


@dataclass(frozen=True)
class PreparedData:
    binding: PreparationBinding
    standardizer: FeatureStandardizer
    train_samples: tuple[PreparedSample, ...]
    validation_samples: tuple[PreparedSample, ...]
    final_test_inventory: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        if type(self.binding) is not PreparationBinding:
            raise ValueError("prepared binding must be a PreparationBinding")
        if type(self.standardizer) is not FeatureStandardizer:
            raise ValueError("prepared standardizer must be a FeatureStandardizer")
        if self.binding.feature_stats_hash != self.standardizer.fingerprint:
            raise ValueError("prepared feature-statistics hash mismatch")
        if not self.train_samples:
            raise ValueError("prepared training samples must be non-empty")
        if not self.validation_samples:
            raise ValueError("prepared validation samples must be non-empty")
        train_ids = {sample.sample_id for sample in self.train_samples}
        validation_ids = {sample.sample_id for sample in self.validation_samples}
        final_ids = {row.get("sample_id") for row in self.final_test_inventory}
        if len(train_ids) != len(self.train_samples) or len(validation_ids) != len(self.validation_samples):
            raise ValueError("prepared sample IDs must be unique within splits")
        if train_ids & validation_ids or train_ids & final_ids or validation_ids & final_ids:
            raise ValueError("prepared sample IDs must not cross splits")
        num_classes = len(self.binding.label_map)
        if any(not 0 <= sample.label < num_classes for sample in (*self.train_samples, *self.validation_samples)):
            raise ValueError("prepared sample label is outside label map")
        for row in self.final_test_inventory:
            if row.get("split") != "final_test":
                raise ValueError("final-test inventory may contain final_test rows only")
            if "features" in row:
                raise ValueError("final-test inventory must not contain derived features")


def _check_optional_pin(name: str, expected: object, actual: object) -> None:
    if expected is not None and expected != actual:
        raise ValueError(f"prepared {name} does not match frozen protocol pin")


def prepare_data(root: str | Path, protocol: ExperimentProtocol) -> PreparedData:
    if type(protocol) is not ExperimentProtocol:
        raise ValueError("protocol must be an ExperimentProtocol")
    root_path = Path(root).absolute()
    manifest = build_manifest(root_path, protocol)

    label_map = tuple((action, index) for index, action in enumerate(protocol.actions))
    label_lookup = dict(label_map)
    raw_train: list[tuple[dict[str, Any], np.ndarray]] = []
    raw_validation: list[tuple[dict[str, Any], np.ndarray]] = []
    final_inventory: list[dict[str, Any]] = []

    for row in manifest["samples"]:
        split = row["split"]
        if split == "final_test":
            # Keep raw-byte provenance only. No normalization, kinematic transform,
            # or statistics may observe final-test coordinates while sealed.
            final_inventory.append(dict(row))
            continue
        if split not in ("train", "validation"):
            raise ValueError(f"unexpected development split: {split!r}")
        source = root_path / row["relative_path"]
        sample = parse_skeleton_file(source)
        body = select_primary_body(sample)
        if body.body_id != row["selected_body_id"]:
            raise ValueError("manifest selected-body identity drift")
        normalized = normalize_body(body)
        raw_features = kinematic_features(normalized)
        target = raw_train if split == "train" else raw_validation
        target.append((row, raw_features))

    if not raw_train or not raw_validation:
        raise ValueError("development preparation requires non-empty train and validation splits")

    standardizer = FeatureStandardizer.fit(
        (features for _, features in raw_train),
        epsilon=protocol.standardization_epsilon,
    )

    def materialize(rows: list[tuple[dict[str, Any], np.ndarray]]) -> tuple[PreparedSample, ...]:
        result: list[PreparedSample] = []
        for row, raw_features in rows:
            action = row["action"]
            if action not in label_lookup:
                raise ValueError("prepared action is outside frozen label map")
            result.append(
                PreparedSample(
                    sample_id=row["sample_id"],
                    label=label_lookup[action],
                    features=standardizer.transform(raw_features),
                )
            )
        return tuple(result)

    train_samples = materialize(raw_train)
    validation_samples = materialize(raw_validation)
    quartiles = length_quartile_boundaries([sample.features.shape[0] for sample in train_samples])
    binding = PreparationBinding(
        dataset_content_hash=manifest["dataset_content_hash"],
        split_hash=manifest["split_hash"],
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
