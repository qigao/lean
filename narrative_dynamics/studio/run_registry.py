"""Replaceable, process-local coordinator registry."""

from __future__ import annotations

from threading import RLock
from typing import Protocol

from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioRunView
from narrative_dynamics.studio.capabilities import StudioCapability


class ScenarioRunRegistry(Protocol):
    def register(self, run_id: str, coordinator: ScenarioCoordinator) -> None: ...

    def resolve(self, run_id: str) -> ScenarioCoordinator: ...

    def list_for(self, capability: StudioCapability) -> tuple[ScenarioRunView, ...]: ...


class InMemoryScenarioRunRegistry:
    """Thread-safe no-overwrite registry for local coordinators."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._coordinators: dict[str, ScenarioCoordinator] = {}

    def register(self, run_id: str, coordinator: ScenarioCoordinator) -> None:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("scenario run registry ID must be non-empty text")
        if not isinstance(coordinator, ScenarioCoordinator):
            raise TypeError("scenario run registry requires ScenarioCoordinator")
        if coordinator.run_view().run_id != run_id:
            raise ValueError("scenario run registry identity must match coordinator")
        with self._lock:
            if run_id in self._coordinators:
                raise ValueError("scenario run registry ID already exists")
            self._coordinators[run_id] = coordinator

    def resolve(self, run_id: str) -> ScenarioCoordinator:
        with self._lock:
            try:
                return self._coordinators[run_id]
            except (KeyError, TypeError):
                raise KeyError("unknown scenario run") from None

    def list_for(self, capability: StudioCapability) -> tuple[ScenarioRunView, ...]:
        if not isinstance(capability, StudioCapability):
            raise TypeError("scenario run listing requires StudioCapability")
        with self._lock:
            coordinators = tuple(
                self._coordinators[run_id]
                for run_id in sorted(capability.run_ids)
                if run_id in self._coordinators
            )
        return tuple(coordinator.run_view() for coordinator in coordinators)


__all__ = ("InMemoryScenarioRunRegistry", "ScenarioRunRegistry")
