from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random
from types import MappingProxyType
from typing import Protocol


def _freeze_canonical_value(value: object, *, label: str) -> object:
    """Detach and recursively freeze JSON-like runtime boundary data."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} numbers must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} keys must be non-empty strings")
            frozen[key] = _freeze_canonical_value(
                item,
                label=f"{label}.{key}",
            )
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_canonical_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _freeze_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = _freeze_canonical_value(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True)
class Scenario:
    """Opaque model input identified by a stable experiment id."""

    id: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must be non-empty")
        object.__setattr__(
            self,
            "payload",
            _freeze_mapping(self.payload, label="scenario payload"),
        )


@dataclass(frozen=True)
class TraceEvent:
    """One ordered event emitted by a simulation model."""

    tick: int
    kind: str
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.tick < 0:
            raise ValueError("event tick must be non-negative")
        if not self.kind:
            raise ValueError("event kind must be non-empty")
        object.__setattr__(
            self,
            "data",
            _freeze_mapping(self.data, label="event data"),
        )


@dataclass(frozen=True)
class ModelRun:
    """Model-owned output before trusted run metadata is attached."""

    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise TypeError("model run events must contain TraceEvent values")
        object.__setattr__(self, "events", events)
        object.__setattr__(
            self,
            "outcome",
            _freeze_mapping(self.outcome, label="model outcome"),
        )


@dataclass(frozen=True)
class SimulationTrace:
    """Canonical result used by metrics, calibration, and replay."""

    model_name: str
    scenario_id: str
    parameters: tuple[tuple[str, float], ...]
    seed: int
    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]

    def __post_init__(self) -> None:
        events = tuple(self.events)
        if any(not isinstance(event, TraceEvent) for event in events):
            raise TypeError("simulation trace events must contain TraceEvent values")
        object.__setattr__(self, "events", events)
        object.__setattr__(
            self,
            "outcome",
            _freeze_mapping(self.outcome, label="trace outcome"),
        )

    @property
    def parameter_map(self) -> dict[str, float]:
        return dict(self.parameters)


class SimulatorModel(Protocol):
    """Boundary implemented by existing, learned, or black-box models."""

    name: str

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        ...
