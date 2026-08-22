from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random
from typing import Protocol, cast

from narrative_dynamics.contracts import (
    ExperimentManifest,
    ExperimentStage,
    ModelRun,
    Scenario,
    SimulationTrace,
    SimulatorModel,
)
from narrative_dynamics.manifest import (
    RUNTIME_IDENTITY,
    component_identity,
    scenario_identity,
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


def _validated_model(
    model: SimulatorModel,
    *,
    expected_name: str | None = None,
) -> SimulatorModel:
    model_name = getattr(model, "name", None)
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model name must be a non-empty string")
    if expected_name is not None and model_name != expected_name:
        raise ValueError("factory-created model name must match its declared name")
    if not callable(getattr(model, "simulate", None)):
        raise TypeError("model must provide a callable simulate() method")
    return model


def _validated_identity_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


@dataclass(frozen=True)
class ModelFactory:
    """Create one fresh simulator instance for each runner batch."""

    name: str
    create: Callable[[], SimulatorModel]
    version: str = "unversioned"
    implementation_revision: str = "unversioned"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _validated_identity_text(self.name, label="model factory name"),
        )
        if not callable(self.create):
            raise TypeError("model factory create must be callable")
        object.__setattr__(
            self,
            "version",
            _validated_identity_text(
                self.version,
                label="model factory version",
            ),
        )
        object.__setattr__(
            self,
            "implementation_revision",
            _validated_identity_text(
                self.implementation_revision,
                label="model factory implementation revision",
            ),
        )

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def instantiate(self) -> SimulatorModel:
        return _validated_model(self.create(), expected_name=self.name)


class InstantiableModelSource(Protocol):
    """Structural source that can supply one model for a runner batch."""

    name: str

    def instantiate(self) -> SimulatorModel:
        ...


ModelSource = SimulatorModel | InstantiableModelSource


def _materialize_model(model: ModelSource) -> SimulatorModel:
    instantiate = getattr(model, "instantiate", None)
    if callable(instantiate):
        expected_name = getattr(model, "name", None)
        if not isinstance(expected_name, str) or not expected_name:
            raise ValueError("instantiable model source must have a name")
        return _validated_model(
            instantiate(),
            expected_name=expected_name,
        )
    return _validated_model(cast(SimulatorModel, model))


class SimulationRunner:
    """Owns seeds, canonical metadata, and run manifests around a model call."""

    def _run_once_with_model(
        self,
        model: SimulatorModel,
        identity_source: ModelSource,
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

        manifest = ExperimentManifest(
            stage=ExperimentStage.SIMULATION_RUN,
            inputs={
                "model": component_identity(identity_source),
                "scenario": scenario_identity(scenario),
                "parameters": canonical,
                "seed": seed,
                "runtime": RUNTIME_IDENTITY,
            },
        )
        return SimulationTrace(
            model_name=model.name,
            scenario_id=scenario.id,
            parameters=canonical,
            seed=seed,
            events=result.events,
            outcome=result.outcome,
            manifest=manifest,
        )

    def run_once(
        self,
        model: ModelSource,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
    ) -> SimulationTrace:
        materialized = _materialize_model(model)
        return self._run_once_with_model(
            materialized,
            model,
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
                model,
                scenario,
                parameters,
                seed=seed,
            )
            for seed in ordered_seeds
        )
