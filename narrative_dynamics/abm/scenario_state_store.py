"""Replaceable exact-state storage seam for scenario coordinators."""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from narrative_dynamics.abm.situated_network_contracts import (
    SituatedNetworkRuntimeState,
)


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _run_id(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scenario state store run id must be a non-empty string")
    return value


def _state(value: object) -> SituatedNetworkRuntimeState:
    if not isinstance(value, SituatedNetworkRuntimeState):
        raise TypeError("scenario state store requires SituatedNetworkRuntimeState")
    return value


def _content_hash(value: object) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError("scenario state store expected state hash must be a content hash")
    return value


@runtime_checkable
class ScenarioStateStore(Protocol):
    def initialize(
        self,
        run_id: str,
        state: SituatedNetworkRuntimeState,
    ) -> None: ...

    def load(self, run_id: str) -> SituatedNetworkRuntimeState: ...

    def compare_and_swap(
        self,
        run_id: str,
        expected_state_hash: str,
        next_state: SituatedNetworkRuntimeState,
    ) -> None: ...


class InMemoryScenarioStateStore:
    """Keep the exact supplied runtime state object for each owned run."""

    def __init__(self) -> None:
        self._states: dict[str, SituatedNetworkRuntimeState] = {}

    def initialize(
        self,
        run_id: str,
        state: SituatedNetworkRuntimeState,
    ) -> None:
        validated_run_id = _run_id(run_id)
        validated_state = _state(state)
        if validated_run_id in self._states:
            raise ValueError("scenario state store run is already initialized")
        self._states[validated_run_id] = validated_state

    def load(self, run_id: str) -> SituatedNetworkRuntimeState:
        validated_run_id = _run_id(run_id)
        try:
            return self._states[validated_run_id]
        except KeyError:
            raise KeyError("unknown scenario state store run") from None

    def compare_and_swap(
        self,
        run_id: str,
        expected_state_hash: str,
        next_state: SituatedNetworkRuntimeState,
    ) -> None:
        validated_run_id = _run_id(run_id)
        validated_hash = _content_hash(expected_state_hash)
        validated_next_state = _state(next_state)
        try:
            current = self._states[validated_run_id]
        except KeyError:
            raise KeyError("unknown scenario state store run") from None
        if current.content_hash != validated_hash:
            raise ValueError("stale scenario state store expected hash")
        self._states[validated_run_id] = validated_next_state


__all__ = (
    "ScenarioStateStore",
    "InMemoryScenarioStateStore",
)
