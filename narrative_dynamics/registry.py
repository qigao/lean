from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from narrative_dynamics.contracts import SimulatorModel


@dataclass(frozen=True)
class ModelDescriptor:
    """Metadata for one simulator adapter without an execution shortcut."""

    name: str
    kind: str
    model: SimulatorModel


class ModelRegistry:
    """Deterministic discovery for adapters that still run through a runner.

    The registry stores already-constructed simulator adapters. It deliberately
    exposes no ``run`` method: seed ownership, parameter canonicalization, and
    trace construction remain the exclusive responsibility of
    :class:`narrative_dynamics.simulation.SimulationRunner`.
    """

    def __init__(self) -> None:
        self._descriptors: dict[str, ModelDescriptor] = {}

    def register(self, model: SimulatorModel, *, kind: str) -> None:
        name = getattr(model, "name", None)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("registered model name must be a non-empty string")
        if name != name.strip():
            raise ValueError("registered model name cannot have surrounding whitespace")
        if not isinstance(kind, str) or not kind.strip():
            raise ValueError("registered model kind must be a non-empty string")
        if kind != kind.strip():
            raise ValueError("registered model kind cannot have surrounding whitespace")
        if not callable(getattr(model, "simulate", None)):
            raise TypeError("registered model must provide a callable simulate() method")
        if name in self._descriptors:
            raise ValueError(f"model {name!r} is already registered")

        self._descriptors[name] = ModelDescriptor(
            name=name,
            kind=kind,
            model=cast(SimulatorModel, model),
        )

    def descriptor(self, name: str) -> ModelDescriptor:
        try:
            return self._descriptors[name]
        except KeyError as error:
            raise KeyError(f"unknown simulator model {name!r}") from error

    def resolve(self, name: str) -> SimulatorModel:
        return self.descriptor(name).model

    def descriptors(self, *, kind: str | None = None) -> tuple[ModelDescriptor, ...]:
        if kind is not None and (not isinstance(kind, str) or not kind.strip()):
            raise ValueError("model kind filter must be a non-empty string")
        return tuple(
            descriptor
            for descriptor in sorted(
                self._descriptors.values(),
                key=lambda item: item.name,
            )
            if kind is None or descriptor.kind == kind
        )

    def names(self, *, kind: str | None = None) -> tuple[str, ...]:
        return tuple(descriptor.name for descriptor in self.descriptors(kind=kind))
