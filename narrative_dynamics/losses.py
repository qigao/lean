from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.metrics import weighted_squared_error


_PROBABILITY_TOLERANCE = 1e-12


@runtime_checkable
class MetricLoss(Protocol):
    """Callable metric loss with a stable manifest identity."""

    def __call__(
        self,
        observed: Mapping[str, float],
        target: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        ...

    def manifest_identity(self) -> Mapping[str, object]:
        ...


def _validated_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot contain surrounding whitespace")
    return value


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


@dataclass(frozen=True)
class CategoricalMetricGroup:
    """One complete categorical probability simplex within a metric schema."""

    name: str
    keys: tuple[str, ...]
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_text(self.name, label="categorical metric group name"),
        )
        keys = tuple(
            _validated_text(key, label="categorical metric key")
            for key in self.keys
        )
        if not keys:
            raise ValueError("categorical metric group requires at least one key")
        if len(set(keys)) != len(keys):
            raise ValueError("categorical metric group keys must be unique")
        object.__setattr__(self, "keys", keys)

        weight = _finite_number(
            self.weight,
            label="categorical metric group weight",
        )
        if weight < 0.0:
            raise ValueError("categorical metric group weight must be non-negative")
        object.__setattr__(self, "weight", weight)

    def manifest_identity(self) -> dict[str, object]:
        return {
            "name": self.name,
            "keys": self.keys,
            "weight": self.weight,
        }


def _validated_groups(
    groups: tuple[CategoricalMetricGroup, ...],
) -> tuple[CategoricalMetricGroup, ...]:
    canonical = tuple(groups)
    if not canonical:
        raise ValueError("categorical loss requires at least one metric group")
    if any(not isinstance(group, CategoricalMetricGroup) for group in canonical):
        raise TypeError(
            "categorical loss groups must be CategoricalMetricGroup values"
        )
    names = tuple(group.name for group in canonical)
    if len(set(names)) != len(names):
        raise ValueError("categorical loss group names must be unique")
    keys = tuple(key for group in canonical for key in group.keys)
    if len(set(keys)) != len(keys):
        raise ValueError("categorical loss metric keys cannot appear in multiple groups")
    return canonical


def _validate_complete_schema(
    groups: tuple[CategoricalMetricGroup, ...],
    observed: Mapping[str, float],
    target: Mapping[str, float],
) -> None:
    declared = {key for group in groups for key in group.keys}
    observed_keys = set(observed)
    target_keys = set(target)
    if observed_keys != target_keys:
        raise ValueError("observed and target metric schemas must match")
    if observed_keys != declared:
        missing = tuple(sorted(declared - observed_keys))
        extra = tuple(sorted(observed_keys - declared))
        detail = []
        if missing:
            detail.append(f"missing={missing}")
        if extra:
            detail.append(f"ungrouped={extra}")
        suffix = "" if not detail else f" ({', '.join(detail)})"
        raise ValueError(
            "categorical metric groups must cover the complete metric schema"
            f"{suffix}"
        )


def _group_vectors(
    group: CategoricalMetricGroup,
    observed: Mapping[str, float],
    target: Mapping[str, float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    predicted: list[float] = []
    masses: list[float] = []
    for key in group.keys:
        probability = _finite_number(
            observed[key],
            label=f"observed categorical probability {key!r}",
        )
        mass = _finite_number(
            target[key],
            label=f"target categorical mass {key!r}",
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError("observed categorical probabilities must be in [0, 1]")
        if mass < 0.0:
            raise ValueError("target categorical masses must be non-negative")
        predicted.append(probability)
        masses.append(mass)

    if not math.isclose(
        sum(predicted),
        1.0,
        rel_tol=_PROBABILITY_TOLERANCE,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError("observed categorical probabilities must sum to one")
    target_total = sum(masses)
    if not math.isfinite(target_total) or target_total <= 0.0:
        raise ValueError("target categorical masses must have positive total mass")
    return tuple(predicted), tuple(mass / target_total for mass in masses)


def _categorical_identity(
    *,
    name: str,
    version: str,
    groups: tuple[CategoricalMetricGroup, ...],
) -> dict[str, object]:
    payload = {
        "name": name,
        "version": version,
        "groups": tuple(group.manifest_identity() for group in groups),
        "probability_tolerance": _PROBABILITY_TOLERANCE,
    }
    return {**payload, "content_hash": stable_content_hash(payload)}


@dataclass(frozen=True)
class WeightedSquaredErrorLoss:
    name: str = "weighted_squared_error"
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validated_text(self.name, label="loss name"))
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, label="loss version"),
        )

    def __call__(
        self,
        observed: Mapping[str, float],
        target: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        return weighted_squared_error(observed, target, weights=weights)

    def manifest_identity(self) -> dict[str, object]:
        payload = {"name": self.name, "version": self.version}
        return {**payload, "content_hash": stable_content_hash(payload)}

    @property
    def content_hash(self) -> str:
        return str(self.manifest_identity()["content_hash"])


@dataclass(frozen=True)
class CategoricalBrierLoss:
    groups: tuple[CategoricalMetricGroup, ...]
    name: str = "categorical_brier"
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", _validated_groups(self.groups))
        object.__setattr__(self, "name", _validated_text(self.name, label="loss name"))
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, label="loss version"),
        )

    def __call__(
        self,
        observed: Mapping[str, float],
        target: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        if weights is not None:
            raise ValueError("categorical losses do not accept per-key weights")
        _validate_complete_schema(self.groups, observed, target)
        value = 0.0
        for group in self.groups:
            predicted, frequencies = _group_vectors(group, observed, target)
            value += group.weight * sum(
                (probability - frequency) ** 2
                for probability, frequency in zip(
                    predicted,
                    frequencies,
                    strict=True,
                )
            )
        return value

    def manifest_identity(self) -> dict[str, object]:
        return _categorical_identity(
            name=self.name,
            version=self.version,
            groups=self.groups,
        )

    @property
    def content_hash(self) -> str:
        return str(self.manifest_identity()["content_hash"])


@dataclass(frozen=True)
class CategoricalLogLoss:
    groups: tuple[CategoricalMetricGroup, ...]
    name: str = "categorical_log"
    version: str = "1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", _validated_groups(self.groups))
        object.__setattr__(self, "name", _validated_text(self.name, label="loss name"))
        object.__setattr__(
            self,
            "version",
            _validated_text(self.version, label="loss version"),
        )

    def __call__(
        self,
        observed: Mapping[str, float],
        target: Mapping[str, float],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        if weights is not None:
            raise ValueError("categorical losses do not accept per-key weights")
        _validate_complete_schema(self.groups, observed, target)
        value = 0.0
        for group in self.groups:
            predicted, frequencies = _group_vectors(group, observed, target)
            for probability, frequency in zip(
                predicted,
                frequencies,
                strict=True,
            ):
                if frequency == 0.0:
                    continue
                if probability <= 0.0:
                    raise ValueError(
                        "positive target mass cannot receive zero predicted probability"
                    )
                value -= group.weight * frequency * math.log(probability)
        return value

    def manifest_identity(self) -> dict[str, object]:
        return _categorical_identity(
            name=self.name,
            version=self.version,
            groups=self.groups,
        )

    @property
    def content_hash(self) -> str:
        return str(self.manifest_identity()["content_hash"])


DEFAULT_METRIC_LOSS = WeightedSquaredErrorLoss()


def evaluate_metric_loss(
    loss: MetricLoss,
    observed: Mapping[str, float],
    target: Mapping[str, float],
    *,
    weights: Mapping[str, float] | None = None,
) -> float:
    if not callable(loss):
        raise TypeError("metric loss must be callable")
    value = _finite_number(
        loss(observed, target, weights=weights),
        label="metric loss",
    )
    if value < 0.0:
        raise ValueError("metric loss must be non-negative")
    return value


def metric_loss_identity(loss: MetricLoss) -> dict[str, object]:
    identity_method = getattr(loss, "manifest_identity", None)
    if not callable(identity_method):
        raise TypeError("metric loss must expose manifest_identity()")
    identity = identity_method()
    if not isinstance(identity, Mapping):
        raise TypeError("metric loss identity must be a mapping")
    payload = dict(identity)
    expected_hash = payload.pop("content_hash", None)
    actual_hash = stable_content_hash(payload)
    if expected_hash is not None and expected_hash != actual_hash:
        raise ValueError("metric loss identity content hash is inconsistent")
    payload["content_hash"] = actual_hash
    return payload


__all__ = [
    "DEFAULT_METRIC_LOSS",
    "CategoricalBrierLoss",
    "CategoricalLogLoss",
    "CategoricalMetricGroup",
    "MetricLoss",
    "WeightedSquaredErrorLoss",
    "evaluate_metric_loss",
    "metric_loss_identity",
]
