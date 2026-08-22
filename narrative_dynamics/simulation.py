from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import math
import random
from types import SimpleNamespace
from typing import Protocol, cast

from narrative_dynamics.attestation import (
    RepositoryIdentity,
    ResultArtifact,
    detect_repository_identity,
    implementation_attestation_identity,
)
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
    scenario_identity_from_payload,
)
from narrative_dynamics.process_execution import (
    CancellationToken,
    ModelCancelled,
    ProcessExecutionResult,
    SubprocessModel,
)
from narrative_dynamics.schema_validation import (
    ModelSchemaViolation,
    validate_contract_events,
    validate_contract_inputs,
)


_DEFAULT_REPOSITORY_IDENTITY = detect_repository_identity()


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


def _scenario_boundary_snapshot(
    model: ModelSource,
    scenario: object,
) -> tuple[object, Mapping[str, object]]:
    """Snapshot once for schema validation and the recorded scenario identity."""

    contract = getattr(model, "contract", None)
    scenario_schema = getattr(contract, "scenario_schema", None)
    if (
        not bool(getattr(scenario_schema, "specified", False))
        or isinstance(scenario, Scenario)
    ):
        return scenario, scenario_identity(scenario)

    manifest_payload = getattr(scenario, "manifest_payload", None)
    if not callable(manifest_payload):
        schema_name = getattr(scenario_schema, "name", "scenario")
        raise ModelSchemaViolation(
            "contracted custom scenarios must expose manifest_payload()",
            boundary="scenario",
            path="$",
            schema_name=schema_name,
        )
    payload = manifest_payload()
    if not isinstance(payload, Mapping):
        schema_name = getattr(scenario_schema, "name", "scenario")
        raise ModelSchemaViolation(
            "manifest_payload() must return a mapping",
            boundary="scenario",
            path="$",
            schema_name=schema_name,
        )

    return (
        SimpleNamespace(payload=payload),
        scenario_identity_from_payload(scenario, payload),
    )


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

    def __init__(
        self,
        *,
        repository_identity: RepositoryIdentity | None = None,
    ) -> None:
        if repository_identity is None:
            repository_identity = _DEFAULT_REPOSITORY_IDENTITY
        if not isinstance(repository_identity, RepositoryIdentity):
            raise TypeError(
                "repository_identity must be a RepositoryIdentity or None"
            )
        self._repository_identity = repository_identity

    @property
    def repository_identity(self) -> RepositoryIdentity:
        return self._repository_identity

    def _trace(
        self,
        *,
        identity_source: ModelSource,
        scenario: Scenario,
        scenario_manifest_identity: Mapping[str, object],
        canonical_parameters: tuple[tuple[str, float], ...],
        seed: int,
        result: ModelRun,
        execution: ExecutionCapture | None,
        input_schema_validation: Mapping[str, object] | None,
        implementation_attestation: Mapping[str, object],
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
        model_identity = component_identity(identity_source)
        model_identity["implementation_attestation"] = dict(
            implementation_attestation
        )
        result_artifact = ResultArtifact.from_result(
            result.events,
            result.outcome,
        )
        manifest_inputs: dict[str, object] = {
            "model": model_identity,
            "scenario": scenario_manifest_identity,
            "parameters": canonical_parameters,
            "seed": seed,
            "runtime": RUNTIME_IDENTITY,
            "repository": self._repository_identity.manifest_identity(),
            "result_artifact": result_artifact.manifest_identity(),
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
        scenario_manifest_identity: Mapping[str, object],
        canonical_parameters: tuple[tuple[str, float], ...],
        *,
        seed: int,
        cancellation: CancellationToken | None,
        input_schema_validation: Mapping[str, object] | None,
        implementation_attestation: Mapping[str, object],
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
            scenario_manifest_identity=scenario_manifest_identity,
            canonical_parameters=canonical_parameters,
            seed=seed,
            result=result,
            execution=None,
            input_schema_validation=input_schema_validation,
            implementation_attestation=implementation_attestation,
        )

    def _run_once_with_executor(
        self,
        executor,
        identity_source: ModelSource,
        scenario: Scenario,
        scenario_manifest_identity: Mapping[str, object],
        canonical_parameters: tuple[tuple[str, float], ...],
        *,
        seed: int,
        cancellation: CancellationToken | None,
        input_schema_validation: Mapping[str, object] | None,
        implementation_attestation: Mapping[str, object],
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
            scenario_manifest_identity=scenario_manifest_identity,
            canonical_parameters=canonical_parameters,
            seed=seed,
            result=executed.run,
            execution=executed.capture,
            input_schema_validation=input_schema_validation,
            implementation_attestation=implementation_attestation,
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
        scenario_validation_view, scenario_manifest_identity = (
            _scenario_boundary_snapshot(model, scenario)
        )
        schema_validation = validate_contract_inputs(
            model,
            scenario_validation_view,
            dict(canonical),
        )
        implementation_attestation = implementation_attestation_identity(model)
        executor = _execution_callable(model)
        if executor is not None:
            return self._run_once_with_executor(
                executor,
                model,
                scenario,
                scenario_manifest_identity,
                canonical,
                seed=validated_seed,
                cancellation=cancellation,
                input_schema_validation=schema_validation,
                implementation_attestation=implementation_attestation,
            )

        materialized = _materialize_model(model)
        return self._run_once_with_model(
            materialized,
            model,
            scenario,
            scenario_manifest_identity,
            canonical,
            seed=validated_seed,
            cancellation=cancellation,
            input_schema_validation=schema_validation,
            implementation_attestation=implementation_attestation,
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
        scenario_validation_view, scenario_manifest_identity = (
            _scenario_boundary_snapshot(model, scenario)
        )
        schema_validation = validate_contract_inputs(
            model,
            scenario_validation_view,
            dict(canonical),
        )
        implementation_attestation = implementation_attestation_identity(model)

        executor = _execution_callable(model)
        if executor is not None:
            return tuple(
                self._run_once_with_executor(
                    executor,
                    model,
                    scenario,
                    scenario_manifest_identity,
                    canonical,
                    seed=seed,
                    cancellation=cancellation,
                    input_schema_validation=schema_validation,
                    implementation_attestation=implementation_attestation,
                )
                for seed in ordered_seeds
            )

        batch_model = _materialize_model(model)
        return tuple(
            self._run_once_with_model(
                batch_model,
                model,
                scenario,
                scenario_manifest_identity,
                canonical,
                seed=seed,
                cancellation=cancellation,
                input_schema_validation=schema_validation,
                implementation_attestation=implementation_attestation,
            )
            for seed in ordered_seeds
        )
