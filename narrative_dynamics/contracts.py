from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import random
from typing import Protocol


@dataclass(frozen=True)
class Scenario:
    """Opaque model input identified by a stable experiment id."""

    id: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("scenario id must be non-empty")


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


@dataclass(frozen=True)
class ModelRun:
    """Model-owned output before trusted run metadata is attached."""

    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]


@dataclass(frozen=True)
class SimulationTrace:
    """Canonical result used by metrics, calibration, and replay."""

    model_name: str
    scenario_id: str
    parameters: tuple[tuple[str, float], ...]
    seed: int
    events: tuple[TraceEvent, ...]
    outcome: Mapping[str, object]

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
