from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


_PROTOCOL_ID = "pose-graph-ssm-v1-development-preflight"
_ACTIONS = (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
_OUTER_TRAIN = (
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31,
    34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95,
    97, 98, 100, 103,
)
_VALIDATION = (14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)
_MODEL_KINDS = ("gru", "ssm_only", "graph_tcn", "graph_ssm")
_SEEDS = (7, 11, 19, 23, 31)
_RATIOS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
_PARAMETER_CEILING = 120_000
_PRIMARY_BODY_RULE = "max-fully-tracked-joints-then-lowest-body-id"
_NORMALIZATION_ID = "ntu25-tracking-interp-root0-torso20-v1"
_FEATURE_SPEC_ID = "ntu25-p-jm-b-bm-a-v1"
_STANDARDIZATION_EPSILON = 1e-6
_RETENTION_RATIO = 0.40
_RETENTION_HORIZON = 20
_DROPOUT_BURST = 8
_RECOVERY_HORIZON = 10
_PRIMARY_EARLY_EFFECT = 0.02
_TEMPORAL_EFFECT = 0.05
_ATTRIBUTION_EFFECT = 0.01
_POSITIVE_SEED_COUNT = 4
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = frozenset(
    {
        "protocol_id",
        "actions",
        "outer_train_subjects",
        "validation_subjects",
        "model_kinds",
        "seeds",
        "observation_ratios",
        "parameter_ceiling",
        "primary_body_rule",
        "normalization_id",
        "feature_spec_id",
        "standardization_epsilon",
        "retention_ratio",
        "retention_horizon",
        "dropout_burst",
        "recovery_horizon",
        "primary_early_effect",
        "temporal_effect",
        "attribution_effect",
        "positive_seed_count",
        "models",
        "training",
        "final_test_enabled",
        "dataset_content_hash",
        "split_hash",
        "feature_stats_hash",
        "label_map_hash",
        "adjacency_hash",
        "ssm_spec_hash",
        "model_fingerprints",
        "length_quartiles",
    }
)
_MODEL_FIELDS = frozenset({"kind", "hidden_size", "blocks"})
_TRAINING_FIELDS = frozenset(
    {
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "epochs",
        "max_updates",
        "checkpoint_rule",
    }
)


@dataclass(frozen=True)
class ModelConfig:
    kind: str
    hidden_size: int
    blocks: int


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    max_updates: int
    checkpoint_rule: str


@dataclass(frozen=True)
class ExperimentProtocol:
    protocol_id: str
    actions: tuple[int, ...]
    outer_train_subjects: tuple[int, ...]
    validation_subjects: tuple[int, ...]
    model_kinds: tuple[str, ...]
    seeds: tuple[int, ...]
    observation_ratios: tuple[float, ...]
    parameter_ceiling: int
    primary_body_rule: str
    normalization_id: str
    feature_spec_id: str
    standardization_epsilon: float
    retention_ratio: float
    retention_horizon: int
    dropout_burst: int
    recovery_horizon: int
    primary_early_effect: float
    temporal_effect: float
    attribution_effect: float
    positive_seed_count: int
    models: tuple[ModelConfig, ...]
    training: TrainingConfig
    final_test_enabled: bool
    dataset_content_hash: str | None
    split_hash: str | None
    feature_stats_hash: str | None
    label_map_hash: str | None
    adjacency_hash: str | None
    ssm_spec_hash: str | None
    model_fingerprints: tuple[tuple[str, str], ...] | None
    length_quartiles: tuple[int, int, int] | None

    def model(self, kind: str) -> ModelConfig:
        for config in self.models:
            if config.kind == kind:
                return config
        raise ValueError(f"unknown model kind: {kind!r}")


def _object(name: str, value: object, expected: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if extra:
            detail.append("extra=" + ",".join(extra))
        raise ValueError(f"{name} schema mismatch: " + " ".join(detail))
    return value


def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a non-empty integer list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _float_tuple(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty numeric list")
    return tuple(_finite_float(name, item) for item in value)


def _finite_float(name: str, value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _digest(name: str, value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be null or a lowercase SHA-256 digest")
    return value


def _models(value: object) -> tuple[ModelConfig, ...]:
    raw = _object("models", value, frozenset(_MODEL_KINDS))
    result: list[ModelConfig] = []
    expected_blocks = {"gru": 1, "ssm_only": 2, "graph_tcn": 2, "graph_ssm": 2}
    for kind in _MODEL_KINDS:
        item = _object(f"models.{kind}", raw[kind], _MODEL_FIELDS)
        if item["kind"] != kind:
            raise ValueError(f"models.{kind}.kind must remain frozen")
        if type(item["hidden_size"]) is not int or item["hidden_size"] != 64:
            raise ValueError(f"models.{kind}.hidden_size must remain frozen at 64")
        if type(item["blocks"]) is not int or item["blocks"] != expected_blocks[kind]:
            raise ValueError(f"models.{kind}.blocks must remain frozen")
        result.append(ModelConfig(kind=kind, hidden_size=64, blocks=expected_blocks[kind]))
    return tuple(result)


def _training(value: object) -> TrainingConfig:
    raw = _object("training", value, _TRAINING_FIELDS)
    expected = TrainingConfig(
        optimizer="adamw",
        learning_rate=3e-4,
        weight_decay=1e-4,
        batch_size=1,
        epochs=20,
        max_updates=4000,
        checkpoint_rule="best-validation-full-sequence-macro-f1",
    )
    candidate = TrainingConfig(
        optimizer=raw["optimizer"] if type(raw["optimizer"]) is str else "",
        learning_rate=_finite_float("training.learning_rate", raw["learning_rate"]),
        weight_decay=_finite_float("training.weight_decay", raw["weight_decay"]),
        batch_size=raw["batch_size"] if type(raw["batch_size"]) is int else -1,
        epochs=raw["epochs"] if type(raw["epochs"]) is int else -1,
        max_updates=raw["max_updates"] if type(raw["max_updates"]) is int else -1,
        checkpoint_rule=raw["checkpoint_rule"] if type(raw["checkpoint_rule"]) is str else "",
    )
    if candidate != expected:
        raise ValueError("training config must remain frozen")
    return candidate


def _model_fingerprints(value: object) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    raw = _object("model_fingerprints", value, frozenset(_MODEL_KINDS))
    result: list[tuple[str, str]] = []
    for kind in _MODEL_KINDS:
        digest = _digest(f"model_fingerprints.{kind}", raw[kind])
        assert digest is not None
        result.append((kind, digest))
    return tuple(result)


def _quartiles(value: object) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 3 or any(type(item) is not int or item <= 0 for item in value):
        raise ValueError("length_quartiles must be null or three positive integers")
    result = tuple(value)
    if not result[0] < result[1] < result[2]:
        raise ValueError("length_quartiles must be strictly increasing")
    return result  # type: ignore[return-value]


def validate_protocol_dict(value: dict[str, Any]) -> ExperimentProtocol:
    raw = _object("protocol", value, _REQUIRED_FIELDS)
    if raw["protocol_id"] != _PROTOCOL_ID:
        raise ValueError("protocol_id must remain frozen")

    actions = _int_tuple("actions", raw["actions"])
    outer = _int_tuple("outer_train_subjects", raw["outer_train_subjects"])
    validation = _int_tuple("validation_subjects", raw["validation_subjects"])
    model_kinds = tuple(raw["model_kinds"]) if isinstance(raw["model_kinds"], list) else ()
    seeds = _int_tuple("seeds", raw["seeds"])
    ratios = _float_tuple("observation_ratios", raw["observation_ratios"])

    if actions != _ACTIONS:
        raise ValueError("actions must remain frozen")
    if outer != _OUTER_TRAIN:
        raise ValueError("outer train subjects must remain frozen")
    if not set(validation).issubset(set(outer)):
        raise ValueError("validation subjects must stay inside outer train")
    if validation != _VALIDATION:
        raise ValueError("validation subjects must remain frozen")
    if model_kinds != _MODEL_KINDS:
        raise ValueError("model roster must remain frozen")
    if seeds != _SEEDS:
        raise ValueError("seeds must remain frozen")
    if ratios != _RATIOS:
        raise ValueError("observation ratios must remain frozen")
    if type(raw["parameter_ceiling"]) is not int or raw["parameter_ceiling"] != _PARAMETER_CEILING:
        raise ValueError("parameter_ceiling must remain frozen")

    frozen_strings = {
        "primary_body_rule": _PRIMARY_BODY_RULE,
        "normalization_id": _NORMALIZATION_ID,
        "feature_spec_id": _FEATURE_SPEC_ID,
    }
    for name, expected in frozen_strings.items():
        if raw[name] != expected:
            raise ValueError(f"{name} must remain frozen")

    exact_floats = {
        "standardization_epsilon": _STANDARDIZATION_EPSILON,
        "retention_ratio": _RETENTION_RATIO,
        "primary_early_effect": _PRIMARY_EARLY_EFFECT,
        "temporal_effect": _TEMPORAL_EFFECT,
        "attribution_effect": _ATTRIBUTION_EFFECT,
    }
    parsed_floats: dict[str, float] = {}
    for name, expected in exact_floats.items():
        parsed = _finite_float(name, raw[name])
        if parsed != expected:
            raise ValueError(f"{name} must remain frozen")
        parsed_floats[name] = parsed

    exact_ints = {
        "retention_horizon": _RETENTION_HORIZON,
        "dropout_burst": _DROPOUT_BURST,
        "recovery_horizon": _RECOVERY_HORIZON,
        "positive_seed_count": _POSITIVE_SEED_COUNT,
    }
    for name, expected in exact_ints.items():
        if type(raw[name]) is not int or raw[name] != expected:
            raise ValueError(f"{name} must remain frozen")

    models = _models(raw["models"])
    training = _training(raw["training"])
    if type(raw["final_test_enabled"]) is not bool:
        raise ValueError("final_test_enabled must be a boolean")

    dataset_content_hash = _digest("dataset_content_hash", raw["dataset_content_hash"])
    split_hash = _digest("split_hash", raw["split_hash"])
    feature_stats_hash = _digest("feature_stats_hash", raw["feature_stats_hash"])
    label_map_hash = _digest("label_map_hash", raw["label_map_hash"])
    adjacency_hash = _digest("adjacency_hash", raw["adjacency_hash"])
    ssm_spec_hash = _digest("ssm_spec_hash", raw["ssm_spec_hash"])
    model_fingerprints = _model_fingerprints(raw["model_fingerprints"])
    length_quartiles = _quartiles(raw["length_quartiles"])

    if raw["final_test_enabled"]:
        required = {
            "dataset_content_hash": dataset_content_hash,
            "split_hash": split_hash,
            "feature_stats_hash": feature_stats_hash,
            "label_map_hash": label_map_hash,
            "adjacency_hash": adjacency_hash,
            "ssm_spec_hash": ssm_spec_hash,
            "model_fingerprints": model_fingerprints,
            "length_quartiles": length_quartiles,
        }
        missing = [name for name, item in required.items() if item is None]
        if missing:
            raise ValueError("final test provenance is incomplete: " + ", ".join(missing))

    return ExperimentProtocol(
        protocol_id=_PROTOCOL_ID,
        actions=actions,
        outer_train_subjects=outer,
        validation_subjects=validation,
        model_kinds=_MODEL_KINDS,
        seeds=seeds,
        observation_ratios=ratios,
        parameter_ceiling=_PARAMETER_CEILING,
        primary_body_rule=_PRIMARY_BODY_RULE,
        normalization_id=_NORMALIZATION_ID,
        feature_spec_id=_FEATURE_SPEC_ID,
        standardization_epsilon=parsed_floats["standardization_epsilon"],
        retention_ratio=parsed_floats["retention_ratio"],
        retention_horizon=_RETENTION_HORIZON,
        dropout_burst=_DROPOUT_BURST,
        recovery_horizon=_RECOVERY_HORIZON,
        primary_early_effect=parsed_floats["primary_early_effect"],
        temporal_effect=parsed_floats["temporal_effect"],
        attribution_effect=parsed_floats["attribution_effect"],
        positive_seed_count=_POSITIVE_SEED_COUNT,
        models=models,
        training=training,
        final_test_enabled=raw["final_test_enabled"],
        dataset_content_hash=dataset_content_hash,
        split_hash=split_hash,
        feature_stats_hash=feature_stats_hash,
        label_map_hash=label_map_hash,
        adjacency_hash=adjacency_hash,
        ssm_spec_hash=ssm_spec_hash,
        model_fingerprints=model_fingerprints,
        length_quartiles=length_quartiles,
    )


def load_protocol(path: str | Path) -> ExperimentProtocol:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read Graph SSM protocol: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("protocol must be a JSON object")
    return validate_protocol_dict(raw)
