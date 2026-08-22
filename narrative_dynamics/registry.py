from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, cast

from narrative_dynamics.contracts import SimulatorModel


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


@dataclass(frozen=True)
class ModelDescriptor:
    """Metadata for one simulator adapter without an execution shortcut."""

    name: str
    kind: ModelKind
    model: SimulatorModel


class ModelRegistry:
    """Deterministic discovery for already-constructed model adapters.

    The registry validates names, categories, and the minimal simulator
    protocol shape. It deliberately exposes no ``run`` method: seed ownership,
    parameter canonicalization, and trace construction remain the exclusive
    responsibility of :class:`narrative_dynamics.simulation.SimulationRunner`.
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
        model: SimulatorModel,
        *,
        kind: ModelKind | str = ModelKind.GENERIC,
    ) -> ModelDescriptor:
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registered model name must be a non-empty string")
        if name != name.strip():
            raise ValueError("registered model name cannot have surrounding whitespace")
        if not callable(getattr(model, "simulate", None)):
            raise TypeError("registered model must provide a callable simulate() method")
        if name in self._descriptors:
            raise ValueError(f"model {name!r} is already registered")

        descriptor = ModelDescriptor(
            name=name,
            kind=self._registration_kind(kind),
            model=cast(SimulatorModel, model),
        )
        self._descriptors[name] = descriptor
        return descriptor

    def descriptor(self, name: str) -> ModelDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise KeyError(f"unknown simulator model {name!r}") from error

    def resolve(self, name: str) -> SimulatorModel:
        return self.descriptor(name).model

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
