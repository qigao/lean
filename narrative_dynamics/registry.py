from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import cast

from narrative_dynamics.contracts import SimulatorModel
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.model_contract import (
    ModelContract,
    ModelLifecycle,
    ModelSchema,
    validate_contract_matches_source,
)
from narrative_dynamics.process_execution import SubprocessModel
from narrative_dynamics.simulation import ModelSource


class ModelKind(str, Enum):
    """Stable adapter categories recognized by the research runtime."""

    GENERIC = "generic"
    PLANNER = "planner"
    POMDP = "pomdp"
    RULE_ENGINE = "rule_engine"
    POPULATION = "population"


_KIND_ALIASES = {
    "rule-engine": ModelKind.RULE_ENGINE,
    "rule_engine": ModelKind.RULE_ENGINE,
}


def _validated_registered_model(
    model: object,
    *,
    expected_name: str,
) -> SimulatorModel:
    name = getattr(model, "name", None)
    if name != expected_name:
        raise ValueError("instantiated model name must match registry descriptor")
    if not callable(getattr(model, "simulate", None)):
        raise TypeError("instantiated model must provide a callable simulate() method")
    return cast(SimulatorModel, model)


@dataclass(frozen=True)
class ModelDescriptor:
    """A registered execution source plus versioned boundary metadata."""

    name: str
    kind: ModelKind
    model: ModelSource
    lifecycle: ModelLifecycle
    contract: ModelContract

    @property
    def source(self) -> ModelSource:
        return self.model

    @property
    def version(self) -> str:
        return self.contract.version

    @property
    def implementation_revision(self) -> str:
        return self.contract.implementation_revision

    @property
    def production_ready(self) -> bool:
        """Whether registry metadata is complete.

        This is not an execution-isolation or empirical-validity certificate.
        """

        return self.contract.complete

    def instantiate(self) -> SimulatorModel:
        if isinstance(self.model, SubprocessModel):
            raise TypeError(
                "fresh-process model sources execute directly and cannot be instantiated"
            )
        instantiate = getattr(self.model, "instantiate", None)
        if callable(instantiate):
            return _validated_registered_model(
                instantiate(),
                expected_name=self.name,
            )
        return _validated_registered_model(
            self.model,
            expected_name=self.name,
        )

    def manifest_identity(self) -> dict[str, object]:
        identity = component_identity(self.model)
        identity.update(
            {
                "name": self.name,
                "version": self.contract.version,
                "kind": self.kind.value,
                "implementation_revision": self.contract.implementation_revision,
                "lifecycle": self.lifecycle.value,
                "contract_hash": self.contract.content_hash,
                "schemas": self.contract.schema_identities,
                "metadata_complete": self.contract.complete,
            }
        )
        return identity


class ModelRegistry:
    """Deterministic discovery for registered simulator sources.

    Raw ``resolve`` remains backward-compatible. ``execution_source`` returns
    the descriptor-backed source that carries lifecycle and schema identity
    into runner manifests. The registry deliberately exposes no run shortcut.
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}

    @staticmethod
    def _registration_kind(kind: ModelKind | str) -> ModelKind:
        if isinstance(kind, ModelKind):
            return kind
        if not isinstance(kind, str):
            raise ValueError("registered model kind must be a string or ModelKind")
        normalized = kind.strip().lower()
        if not normalized:
            raise ValueError("registered model kind must be non-empty")
        if normalized in _KIND_ALIASES:
            return _KIND_ALIASES[normalized]
        try:
            return ModelKind(normalized)
        except ValueError as error:
            raise ValueError(f"unsupported model kind: {kind!r}") from error

    @staticmethod
    def _query_kind(kind: ModelKind | str) -> ModelKind | None:
        if isinstance(kind, ModelKind):
            return kind
        if not isinstance(kind, str):
            raise ValueError("model kind filter must be a string or ModelKind")
        normalized = kind.strip().lower()
        if not normalized:
            raise ValueError("model kind filter must be non-empty")
        if normalized in _KIND_ALIASES:
            return _KIND_ALIASES[normalized]
        try:
            return ModelKind(normalized)
        except ValueError:
            return None

    def register(
        self,
        model: ModelSource,
        *,
        kind: ModelKind | str = ModelKind.GENERIC,
        contract: ModelContract | None = None,
    ) -> ModelDescriptor:
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registered model name must be a non-empty string")
        if name != name.strip():
            raise ValueError("registered model name cannot have surrounding whitespace")

        instantiate = getattr(model, "instantiate", None)
        if isinstance(model, SubprocessModel):
            lifecycle = ModelLifecycle.FRESH_PROCESS_PER_RUN
        elif callable(instantiate):
            lifecycle = ModelLifecycle.FRESH_PER_BATCH
        else:
            if not callable(getattr(model, "simulate", None)):
                raise TypeError(
                    "registered model must provide simulate(), instantiate(), or execute()"
                )
            lifecycle = ModelLifecycle.SHARED_INSTANCE

        if name in self._descriptors:
            raise ValueError(f"model {name!r} is already registered")

        registered_contract = (
            ModelContract.from_source(model) if contract is None else contract
        )
        if not isinstance(registered_contract, ModelContract):
            raise TypeError("registered model contract must be a ModelContract")
        validate_contract_matches_source(model, registered_contract)

        descriptor = ModelDescriptor(
            name=name,
            kind=self._registration_kind(kind),
            model=model,
            lifecycle=lifecycle,
            contract=registered_contract,
        )
        self._descriptors[name] = descriptor
        return descriptor

    def descriptor(self, name: str) -> ModelDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise KeyError(f"unknown simulator model {name!r}") from error

    def resolve(self, name: str) -> ModelSource:
        """Return the raw source for backward compatibility."""

        return self.descriptor(name).source

    def execution_source(self, name: str) -> ModelDescriptor:
        """Return the metadata-bound source intended for research execution."""

        return self.descriptor(name)

    def require_production_ready(self, name: str) -> ModelDescriptor:
        """Require complete version/revision/schema metadata.

        Process isolation and empirical validation remain separate gates.
        """

        descriptor = self.descriptor(name)
        if not descriptor.production_ready:
            raise ValueError(
                f"model {name!r} does not have a complete registry contract"
            )
        return descriptor

    def descriptors(
        self,
        *,
        kind: ModelKind | str | None = None,
    ) -> tuple[ModelDescriptor, ...]:
        normalized = None if kind is None else self._query_kind(kind)
        if kind is not None and normalized is None:
            return ()
        return tuple(
            descriptor
            for descriptor in sorted(
                self._descriptors.values(),
                key=lambda item: item.name,
            )
            if normalized is None or descriptor.kind is normalized
        )

    def entries(
        self,
        *,
        kind: ModelKind | str | None = None,
    ) -> tuple[ModelDescriptor, ...]:
        """Alias emphasizing registry entries rather than discovery metadata."""

        return self.descriptors(kind=kind)

    def names(self, *, kind: ModelKind | str | None = None) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.descriptors(kind=kind))


__all__ = [
    "ModelContract",
    "ModelDescriptor",
    "ModelKind",
    "ModelLifecycle",
    "ModelRegistry",
    "ModelSchema",
]
