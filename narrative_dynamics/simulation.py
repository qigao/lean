from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random
from typing import Protocol, cast

from narrative_dynamics.contracts import (
    ExecutionCapture,
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
from narrative_dynamics.process_execution import (
    CancellationToken,
    ModelCancelled,
    ProcessExecutionResult,
    SubprocessModel,
)
from narrative_dynamics.schema_validation import (
    validate_contract_events,
    validate_contract_inputs,
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


def _validated_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("simulation seed must be an integer")
    return seed


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


class ExecutableModelSource(Protocol):
    """Structural source that owns execution outside the in-process model call."""

    name: str

    def execute(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
        cancellation: CancellationToken | None = None,
    ) -> ProcessExecutionResult:
        ...


ModelSource = SimulatorModel | InstantiableModelSource | ExecutableModelSource


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


def _execution_callable(model: ModelSource):
    if isinstance(model, SubprocessModel):
        return model.execute
    nested_source = getattr(model, "source", None)
    if isinstance(nested_source, SubprocessModel):
        return nested_source.execute
    return None


def _raise_if_cancelled(
    cancellation: CancellationToken | None,
) -> None:
    if cancellation is not None:
        if not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be a CancellationToken or None")
        if cancellation.cancelled:
            raise ModelCancelled(
                "model execution was cancelled before start"
            )


class SimulationRunner:
    """Owns seeds, schema checks, canonical metadata, and run manifests."""

    @staticmethod
    def _trace(
        *,
        identity_source: ModelSource,
        scenario: Scenario,
        canonical_parameters: tuple[tuple[str, float], ...],
        seed: int,
        result: ModelRun,
        execution: ExecutionCapture | None,
        input_schema_validation: Mapping[str, object] | None,
    ) -> SimulationTrace:
        if not isinstance(result, ModelRun):
            raise TypeError("model execution must return ModelRun")
        model_name = getattr(identity_source, "name", None)
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model source name must be a non-empty string")

        schema_validation = validate_contract_events(
            identity_source,
            result.events,
            input_schema_validation,
            execution=execution,
        )
        manifest_inputs: dict[str, object] = {
            "model": component_identity(identity_source),
            "scenario": scenario_identity(scenario),
            "parameters": canonical_parameters,
            "seed": seed,
            "runtime": RUNTIME_IDENTITY,
        }
        if schema_validation is not None:
            manifest_inputs["schema_validation"] = schema_validation

        manifest = ExperimentManifest(
            stage=ExperimentStage.SIMULATION_RUN,
            inputs=manifest_inputs,
        )
        return SimulationTrace(
            model_name=model_name,
            scenario_id=scenario.id,
            parameters=canonical_parameters,
            seed=seed,
            events=result.events,
            outcome=result.outcome,
            manifest=manifest,
            execution=execution,
        )

    def _run_once_with_model(
        self,
        model: SimulatorModel,
        identity_source: ModelSource,
        scenario: Scenario,
        canonical_parameters: tuple[tuple[str, float], ...],
        *,
        seed: int,
        cancellation: CancellationToken | None,
        input_schema_validation: Mapping[str, object] | None,
    ) -> SimulationTrace:
        _raise_if_cancelled(cancellation)
        result = model.simulate(
            scenario,
            dict(canonical_parameters),
            random.Random(seed),
        )
        return self._trace(
            identity_source=identity_source,
            scenario=scenario,
            canonical_parameters=canonical_parameters,
            seed=seed,
            result=result,
            execution=None,
            input_schema_validation=input_schema_validation,
        )

    def _run_once_with_executor(
        self,
        executor,
        identity_source: ModelSource,
        scenario: Scenario,
        canonical_parameters: tuple[tuple[str, float], ...],
        *,
        seed: int,
        cancellation: CancellationToken | None,
        input_schema_validation: Mapping[str, object] | None,
    ) -> SimulationTrace:
        _raise_if_cancelled(cancellation)
        executed = executor(
            scenario,
            dict(canonical_parameters),
            seed=seed,
            cancellation=cancellation,
        )
        if not isinstance(executed, ProcessExecutionResult):
            raise TypeError(
                "external model execute() must return ProcessExecutionResult"
            )
        return self._trace(
            identity_source=identity_source,
            scenario=scenario,
            canonical_parameters=canonical_parameters,
            seed=seed,
            result=executed.run,
            execution=executed.capture,
            input_schema_validation=input_schema_validation,
        )

    def run_once(
        self,
        model: ModelSource,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seed: int,
        cancellation: CancellationToken | None = None,
    ) -> SimulationTrace:
        validated_seed = _validated_seed(seed)
        canonical = _canonical_parameters(parameters)
        schema_validation = validate_contract_inputs(
            model,
            scenario,
            dict(canonical),
        )
        executor = _execution_callable(model)
        if executor is not None:
            return self._run_once_with_executor(
                executor,
                model,
                scenario,
                canonical,
                seed=validated_seed,
                cancellation=cancellation,
                input_schema_validation=schema_validation,
            )

        materialized = _materialize_model(model)
        return self._run_once_with_model(
            materialized,
            model,
            scenario,
            canonical,
            seed=validated_seed,
            cancellation=cancellation,
            input_schema_validation=schema_validation,
        )

    def run_batch(
        self,
        model: ModelSource,
        scenario: Scenario,
        parameters: Mapping[str, float],
        *,
        seeds: Iterable[int],
        cancellation: CancellationToken | None = None,
    ) -> tuple[SimulationTrace, ...]:
        ordered_seeds = tuple(_validated_seed(seed) for seed in seeds)
        if not ordered_seeds:
            raise ValueError("simulation batch must contain at least one seed")
        canonical = _canonical_parameters(parameters)
        schema_validation = validate_contract_inputs(
            model,
            scenario,
            dict(canonical),
        )

        executor = _execution_callable(model)
        if executor is not None:
            return tuple(
                self._run_once_with_executor(
                    executor,
                    model,
                    scenario,
                    canonical,
                    seed=seed,
                    cancellation=cancellation,
                    input_schema_validation=schema_validation,
                )
                for seed in ordered_seeds
            )

        batch_model = _materialize_model(model)
        return tuple(
            self._run_once_with_model(
                batch_model,
                model,
                scenario,
                canonical,
                seed=seed,
                cancellation=cancellation,
                input_schema_validation=schema_validation,
            )
            for seed in ordered_seeds
        )
