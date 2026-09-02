"""Replaceable, process-local coordinator registry."""

from __future__ import annotations

from dataclasses import dataclass
import re
import secrets
from threading import Condition, RLock
from typing import Protocol

from narrative_dynamics.abm.scenario_coordinator import ScenarioCoordinator
from narrative_dynamics.abm.scenario_coordinator_contracts import ScenarioRunView
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.studio.capabilities import StudioCapability


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


class ScenarioRunReservationConflictError(ValueError):
    pass


@dataclass(frozen=True)
class ScenarioRunReservation:
    """One atomic run-ID ownership decision returned by a registry."""

    run_id: str
    request_hash: str
    owner_token: str | None
    coordinator: ScenarioCoordinator | None = None
    outcome: object | None = None

    @property
    def acquired(self) -> bool:
        return self.owner_token is not None


@dataclass(frozen=True)
class _PendingRun:
    request_hash: str
    owner_token: str


@dataclass(frozen=True)
class _RegisteredRun:
    request_hash: str
    coordinator: ScenarioCoordinator
    outcome: object | None


class ScenarioRunRegistry(Protocol):
    def reserve(self, run_id: str, request_hash: str) -> ScenarioRunReservation: ...

    def commit(
        self,
        reservation: ScenarioRunReservation,
        coordinator: ScenarioCoordinator,
        *,
        outcome: object | None = None,
    ) -> None: ...

    def abort(self, reservation: ScenarioRunReservation) -> None: ...

    def register(self, run_id: str, coordinator: ScenarioCoordinator) -> None: ...

    def resolve(self, run_id: str) -> ScenarioCoordinator: ...

    def list_for(self, capability: StudioCapability) -> tuple[ScenarioRunView, ...]: ...


class InMemoryScenarioRunRegistry:
    """Thread-safe no-overwrite registry for local coordinators."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._condition = Condition(self._lock)
        self._runs: dict[str, _PendingRun | _RegisteredRun] = {}

    @staticmethod
    def _run_id(run_id: object) -> str:
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("scenario run registry ID must be non-empty text")
        return run_id

    @staticmethod
    def _request_hash(request_hash: object) -> str:
        if not isinstance(request_hash, str) or _CONTENT_HASH.fullmatch(request_hash) is None:
            raise ValueError("scenario run reservation requires a content hash")
        return request_hash

    def reserve(self, run_id: str, request_hash: str) -> ScenarioRunReservation:
        exact_run_id = self._run_id(run_id)
        exact_request_hash = self._request_hash(request_hash)
        with self._condition:
            while True:
                entry = self._runs.get(exact_run_id)
                if entry is None:
                    owner_token = secrets.token_hex(24)
                    self._runs[exact_run_id] = _PendingRun(
                        exact_request_hash,
                        owner_token,
                    )
                    return ScenarioRunReservation(
                        exact_run_id,
                        exact_request_hash,
                        owner_token,
                    )
                if isinstance(entry, _PendingRun):
                    self._condition.wait()
                    continue
                if entry.request_hash != exact_request_hash:
                    raise ScenarioRunReservationConflictError(
                        "scenario run ID is owned by another request"
                    )
                return ScenarioRunReservation(
                    exact_run_id,
                    exact_request_hash,
                    None,
                    entry.coordinator,
                    entry.outcome,
                )

    def commit(
        self,
        reservation: ScenarioRunReservation,
        coordinator: ScenarioCoordinator,
        *,
        outcome: object | None = None,
    ) -> None:
        if not isinstance(reservation, ScenarioRunReservation) or not reservation.acquired:
            raise TypeError("scenario run commit requires an acquired reservation")
        if not isinstance(coordinator, ScenarioCoordinator):
            raise TypeError("scenario run registry requires ScenarioCoordinator")
        if coordinator.run_view().run_id != reservation.run_id:
            raise ValueError("scenario run registry identity must match coordinator")
        with self._condition:
            entry = self._runs.get(reservation.run_id)
            if not isinstance(entry, _PendingRun) or (
                entry.request_hash != reservation.request_hash
                or entry.owner_token != reservation.owner_token
            ):
                raise ScenarioRunReservationConflictError(
                    "scenario run reservation is no longer owned"
                )
            self._runs[reservation.run_id] = _RegisteredRun(
                reservation.request_hash,
                coordinator,
                outcome,
            )
            self._condition.notify_all()

    def abort(self, reservation: ScenarioRunReservation) -> None:
        if not isinstance(reservation, ScenarioRunReservation):
            raise TypeError("scenario run abort requires ScenarioRunReservation")
        if not reservation.acquired:
            return
        with self._condition:
            entry = self._runs.get(reservation.run_id)
            if isinstance(entry, _PendingRun) and (
                entry.request_hash == reservation.request_hash
                and entry.owner_token == reservation.owner_token
            ):
                del self._runs[reservation.run_id]
                self._condition.notify_all()

    def register(self, run_id: str, coordinator: ScenarioCoordinator) -> None:
        exact_run_id = self._run_id(run_id)
        if not isinstance(coordinator, ScenarioCoordinator):
            raise TypeError("scenario run registry requires ScenarioCoordinator")
        view = coordinator.run_view()
        if view.run_id != exact_run_id:
            raise ValueError("scenario run registry identity must match coordinator")
        request_hash = stable_content_hash(
            {
                "operation": "manual-register",
                "run_id": exact_run_id,
                "run_view_hash": view.content_hash,
            }
        )
        with self._condition:
            if exact_run_id in self._runs:
                raise ValueError("scenario run registry ID already exists")
            self._runs[exact_run_id] = _RegisteredRun(
                request_hash,
                coordinator,
                None,
            )
            self._condition.notify_all()

    def resolve(self, run_id: str) -> ScenarioCoordinator:
        with self._condition:
            while True:
                try:
                    entry = self._runs[run_id]
                except (KeyError, TypeError):
                    raise KeyError("unknown scenario run") from None
                if isinstance(entry, _PendingRun):
                    self._condition.wait()
                    continue
                return entry.coordinator

    def list_for(self, capability: StudioCapability) -> tuple[ScenarioRunView, ...]:
        if not isinstance(capability, StudioCapability):
            raise TypeError("scenario run listing requires StudioCapability")
        with self._condition:
            coordinators = tuple(
                entry.coordinator
                for run_id in sorted(capability.run_ids)
                if isinstance((entry := self._runs.get(run_id)), _RegisteredRun)
            )
        return tuple(coordinator.run_view() for coordinator in coordinators)


__all__ = (
    "InMemoryScenarioRunRegistry",
    "ScenarioRunRegistry",
    "ScenarioRunReservation",
    "ScenarioRunReservationConflictError",
)
