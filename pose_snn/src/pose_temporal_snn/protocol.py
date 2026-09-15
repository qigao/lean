from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any


_PROTOCOL_ID = "pose-temporal-snn-v1-development-preflight"
_ACTIONS = (8, 9, 22, 23, 26, 27, 31, 34, 35, 36)
_OUTER_TRAIN = (
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31,
    34, 35, 38, 45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59,
    70, 74, 78, 80, 81, 82, 83, 84, 85, 86, 89, 91, 92, 93, 94, 95,
    97, 98, 100, 103,
)
_VALIDATION = (14, 28, 35, 46, 50, 54, 74, 83, 84, 86, 103)
_EVENT_QUANTILE = 0.75
_OBSERVATION_RATIOS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
_SEEDS = (7, 11, 19, 23, 31)
_PARAMETER_CEILING = 50_000
_RETENTION_RATIO = 0.40
_RETENTION_HORIZON = 20
_PERTURBATION_FRACTION = 0.25
_RECOVERY_HORIZON = 10
_DIGEST = re.compile(r"^[0-9a-f]{64}$")

_REQUIRED_FIELDS = frozenset(
    {
        "protocol_id",
        "actions",
        "outer_train_subjects",
        "validation_subjects",
        "event_quantile",
        "observation_ratios",
        "seeds",
        "parameter_ceiling",
        "retention_ratio",
        "retention_horizon",
        "perturbation_fraction",
        "recovery_horizon",
        "rsnn",
        "spiking_graph",
        "gru",
        "training",
        "final_test_enabled",
        "dataset_content_hash",
        "split_hash",
        "encoder_hash",
        "label_map_hash",
        "model_fingerprints",
    }
)
_MODEL_FIELDS = frozenset(
    {
        "kind",
        "hidden_size",
        "membrane_decay",
        "synaptic_decay",
        "threshold",
        "reset",
        "surrogate_slope",
    }
)
_TRAINING_FIELDS = frozenset(
    {
        "optimizer",
        "learning_rate",
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
    membrane_decay: float | None = None
    synaptic_decay: float | None = None
    threshold: float | None = None
    reset: float | None = None
    surrogate_slope: float | None = None


@dataclass(frozen=True)
class TrainingConfig:
    optimizer: str
    learning_rate: float
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
    event_quantile: float
    observation_ratios: tuple[float, ...]
    seeds: tuple[int, ...]
    parameter_ceiling: int
    retention_ratio: float
    retention_horizon: int
    perturbation_fraction: float
    recovery_horizon: int
    rsnn: ModelConfig
    spiking_graph: ModelConfig
    gru: ModelConfig
    training: TrainingConfig
    final_test_enabled: bool
    dataset_content_hash: str | None
    split_hash: str | None
    encoder_hash: str | None
    label_map_hash: str | None
    model_fingerprints: tuple[tuple[str, str], ...] | None


def _exact_keys(name: str, value: object, expected: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    keys = set(value)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if extra:
            details.append("extra=" + ",".join(extra))
        raise ValueError(f"{name} schema mismatch: " + " ".join(details))
    return value


def _finite_float(name: str, value: object) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(f"{name} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite numeric")
    return result


def _optional_float(name: str, value: object) -> float | None:
    return None if value is None else _finite_float(name, value)


def _digest(name: str, value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be null or a lowercase SHA-256 digest")
    return value


def _int_tuple(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, list) or not value or any(type(item) is not int for item in value):
        raise ValueError(f"{name} must be a non-empty integer list")
    result = tuple(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique integers")
    return result


def _float_tuple(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty numeric list")
    return tuple(_finite_float(name, item) for item in value)


def _model(name: str, value: object) -> ModelConfig:
    raw = _exact_keys(name, value, _MODEL_FIELDS)
    if type(raw["kind"]) is not str or not raw["kind"]:
        raise ValueError(f"{name}.kind must be a non-empty string")
    if type(raw["hidden_size"]) is not int or raw["hidden_size"] <= 0:
        raise ValueError(f"{name}.hidden_size must be a positive integer")
    return ModelConfig(
        kind=raw["kind"],
        hidden_size=raw["hidden_size"],
        membrane_decay=_optional_float(f"{name}.membrane_decay", raw["membrane_decay"]),
        synaptic_decay=_optional_float(f"{name}.synaptic_decay", raw["synaptic_decay"]),
        threshold=_optional_float(f"{name}.threshold", raw["threshold"]),
        reset=_optional_float(f"{name}.reset", raw["reset"]),
        surrogate_slope=_optional_float(f"{name}.surrogate_slope", raw["surrogate_slope"]),
    )


def _training(value: object) -> TrainingConfig:
    raw = _exact_keys("training", value, _TRAINING_FIELDS)
    for field in ("optimizer", "checkpoint_rule"):
        if type(raw[field]) is not str or not raw[field]:
            raise ValueError(f"training.{field} must be a non-empty string")
    for field in ("batch_size", "epochs", "max_updates"):
        if type(raw[field]) is not int or raw[field] <= 0:
            raise ValueError(f"training.{field} must be a positive integer")
    return TrainingConfig(
        optimizer=raw["optimizer"],
        learning_rate=_finite_float("training.learning_rate", raw["learning_rate"]),
        batch_size=raw["batch_size"],
        epochs=raw["epochs"],
        max_updates=raw["max_updates"],
        checkpoint_rule=raw["checkpoint_rule"],
    )


def _validate_model_fingerprints(value: object) -> tuple[tuple[str, str], ...] | None:
    if value is None:
        return None
    raw = _exact_keys("model_fingerprints", value, frozenset({"rsnn", "spiking_graph", "gru"}))
    return tuple((name, _digest(f"model_fingerprints.{name}", raw[name])) for name in ("rsnn", "spiking_graph", "gru"))  # type: ignore[arg-type,return-value]


def validate_protocol_dict(value: dict[str, Any]) -> ExperimentProtocol:
    raw = _exact_keys("protocol", value, _REQUIRED_FIELDS)
    if raw["protocol_id"] != _PROTOCOL_ID:
        raise ValueError("protocol_id is not the frozen V1 protocol")

    actions = _int_tuple("actions", raw["actions"])
    outer_train = _int_tuple("outer_train_subjects", raw["outer_train_subjects"])
    validation = _int_tuple("validation_subjects", raw["validation_subjects"])
    ratios = _float_tuple("observation_ratios", raw["observation_ratios"])
    seeds = _int_tuple("seeds", raw["seeds"])

    if actions != _ACTIONS:
        raise ValueError("actions must remain frozen")
    if outer_train != _OUTER_TRAIN:
        raise ValueError("outer train subjects must remain frozen")
    if validation != _VALIDATION:
        if not set(validation).issubset(set(outer_train)):
            raise ValueError("validation subjects must stay inside outer train")
        raise ValueError("validation subjects must remain frozen")
    if not set(validation).issubset(set(outer_train)):
        raise ValueError("validation subjects must stay inside outer train")

    event_quantile = _finite_float("event_quantile", raw["event_quantile"])
    retention_ratio = _finite_float("retention_ratio", raw["retention_ratio"])
    perturbation_fraction = _finite_float("perturbation_fraction", raw["perturbation_fraction"])
    if event_quantile != _EVENT_QUANTILE:
        raise ValueError("event_quantile must remain frozen at 0.75")
    if ratios != _OBSERVATION_RATIOS:
        raise ValueError("observation ratios must remain frozen")
    if seeds != _SEEDS:
        raise ValueError("seeds must remain frozen")
    if type(raw["parameter_ceiling"]) is not int or raw["parameter_ceiling"] != _PARAMETER_CEILING:
        raise ValueError("parameter_ceiling must remain frozen at 50000")
    if retention_ratio != _RETENTION_RATIO:
        raise ValueError("retention_ratio must remain frozen at 0.40")
    if type(raw["retention_horizon"]) is not int or raw["retention_horizon"] != _RETENTION_HORIZON:
        raise ValueError("retention_horizon must remain frozen at 20")
    if perturbation_fraction != _PERTURBATION_FRACTION:
        raise ValueError("perturbation_fraction must remain frozen at 0.25")
    if type(raw["recovery_horizon"]) is not int or raw["recovery_horizon"] != _RECOVERY_HORIZON:
        raise ValueError("recovery_horizon must remain frozen at 10")

    rsnn = _model("rsnn", raw["rsnn"])
    spiking_graph = _model("spiking_graph", raw["spiking_graph"])
    gru = _model("gru", raw["gru"])
    expected_rsnn = ModelConfig("rsnn", 96, 0.95, 0.80, 1.0, 0.0, 5.0)
    expected_graph = ModelConfig("spiking_graph", 32, 0.95, 0.80, 1.0, 0.0, 5.0)
    expected_gru = ModelConfig("gru", 56, None, None, None, None, None)
    if rsnn != expected_rsnn:
        raise ValueError("rsnn config must remain frozen")
    if spiking_graph != expected_graph:
        raise ValueError("spiking_graph config must remain frozen")
    if gru != expected_gru:
        raise ValueError("gru config must remain frozen")

    training = _training(raw["training"])
    expected_training = TrainingConfig(
        optimizer="adam",
        learning_rate=1e-3,
        batch_size=1,
        epochs=10,
        max_updates=2000,
        checkpoint_rule="best-validation-macro-f1-over-all-epochs",
    )
    if training != expected_training:
        raise ValueError("training config must remain frozen")

    if type(raw["final_test_enabled"]) is not bool:
        raise ValueError("final_test_enabled must be a boolean")
    dataset_content_hash = _digest("dataset_content_hash", raw["dataset_content_hash"])
    split_hash = _digest("split_hash", raw["split_hash"])
    encoder_hash = _digest("encoder_hash", raw["encoder_hash"])
    label_map_hash = _digest("label_map_hash", raw["label_map_hash"])
    model_fingerprints = _validate_model_fingerprints(raw["model_fingerprints"])

    if raw["final_test_enabled"]:
        required = {
            "dataset_content_hash": dataset_content_hash,
            "split_hash": split_hash,
            "encoder_hash": encoder_hash,
            "label_map_hash": label_map_hash,
            "model_fingerprints": model_fingerprints,
        }
        missing = [name for name, item in required.items() if item is None]
        if missing:
            raise ValueError("final test provenance is incomplete: " + ", ".join(missing))

    return ExperimentProtocol(
        protocol_id=_PROTOCOL_ID,
        actions=actions,
        outer_train_subjects=outer_train,
        validation_subjects=validation,
        event_quantile=event_quantile,
        observation_ratios=ratios,
        seeds=seeds,
        parameter_ceiling=_PARAMETER_CEILING,
        retention_ratio=retention_ratio,
        retention_horizon=_RETENTION_HORIZON,
        perturbation_fraction=perturbation_fraction,
        recovery_horizon=_RECOVERY_HORIZON,
        rsnn=rsnn,
        spiking_graph=spiking_graph,
        gru=gru,
        training=training,
        final_test_enabled=raw["final_test_enabled"],
        dataset_content_hash=dataset_content_hash,
        split_hash=split_hash,
        encoder_hash=encoder_hash,
        label_map_hash=label_map_hash,
        model_fingerprints=model_fingerprints,
    )


def load_protocol(path: str | Path) -> ExperimentProtocol:
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read pose temporal SNN protocol: {source}") from exc
    if not isinstance(raw, dict):
        raise ValueError("protocol must be a JSON object")
    return validate_protocol_dict(raw)
