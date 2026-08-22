from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random

from narrative_dynamics.contracts import (
    ModelRun,
    Scenario,
    SimulationTrace,
    SimulatorModel,
)


def _canonical_parameters(
    parameters: Mapping[str, float],
) -> tuple[tuple[str, float], ...]:
    canonical: list[tuple[str, float]] = []
    for name, raw_value in parameters.items():
        if not isinstance(name, str) or not name:
            raise ValueError("parameter names must be non-empty strings")
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"parameter {name!r} must be numeric") from error
        if not math.isfinite(value):
            raise ValueError(f"parameter {name!r} must be finite")
        canonical.append((name, value))
    canonical.sort(key=lambda item: item[0])
    return tuple(canonical)


def _validated_model(model: SimulatorModel, *, expected_name: str | None = None) -> SimulatorModel:
    model_name = getattr(model, "name", None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model name must be a non-empty string")
    if expected_name is not None and model_name != expected_name:
        raise ValueError("factory-created model name must match its declared name")
    if not callable(getattr(model, "simulate", None)):
        raise TypeError("model must provide a callable simulate() method")
    return model


@dataclass(frozen=True)
class ModelFactory:
    """Create one fresh simulator instance for each runner batch."""

    name: str
    create: Callable[[], SimulatorModel]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("model factory name must be a non-empty string")
        if not callable(self.create):
            raise TypeError("model factory create must be callable")

    def instantiate(self) -> SimulatorModel:
        return _validated_model(self.create(), expected_name=self.name)


ModelSource = SimulatorModel | ModelFactory


def _materialize_model(model: ModelSource) -> SimulatorModel:
    if isinstance(model, ModelFactory):
        return model.instantiate()
    return _validated_model(model)


class SimulationRunner:
    """Owns seeds and canonical metadata around an untrusted model call."""

    def _run_once_with_model(
        self,
        model: SimulatorModel,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
    ) -> SimulationTrace:
        if not isinstance(seed, int):
            raise TypeError("simulation seed must be an integer")

        canonical = _canonical_parameters(parameters)
        rng = random.Random(seed)
        result = model.simulate(scenario, dict(canonical), rng)
        if not isinstance(result, ModelRun):
            raise TypeError("model simulate() must return ModelRun")

        return SimulationTrace(
            model_name=model.name,
            scenario_id=scenario.id,
            parameters=canonical,
            seed=seed,
            events=result.events,
            outcome=result.outcome,
        )

    def run_once(
        self,
        model: ModelSource,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
    ) -> SimulationTrace:
        return self._run_once_with_model(
            _materialize_model(model),
            scenario,
            parameters,
            seed=seed,
        )

    def run_batch(
        self,
        model: ModelSource,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seeds: Iterable[int],
    ) -> tuple[SimulationTrace, ...]:
        ordered_seeds = tuple(seeds)
        if not ordered_seeds:
            raise ValueError("simulation batch must contain at least one seed")

        batch_model = _materialize_model(model)
        return tuple(
            self._run_once_with_model(
                batch_model,
                scenario,
                parameters,
                seed=seed,
            )
            for seed in ordered_seeds
        )
