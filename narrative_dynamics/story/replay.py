from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from narrative_dynamics.story.schema import (
    DirectObservationV1,
    RelocationEventV1,
)


@dataclass(frozen=True)
class ObjectLocationState:
    object_id: str
    location: str
    supporting_event_id: str
    logical_time: int


def _at_time(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("story replay at_time must be an integer or None")
    if value < 0:
        raise ValueError("story replay at_time must be non-negative")
    return value


def objective_state(
    events: Iterable[RelocationEventV1],
    *,
    at_time: int | None = None,
) -> Mapping[str, ObjectLocationState]:
    """Replay every relocation visible to the objective story history."""

    limit = _at_time(at_time)
    state: dict[str, ObjectLocationState] = {}
    for event in events:
        if not isinstance(event, RelocationEventV1):
            raise TypeError("story replay events must be RelocationEventV1 values")
        if limit is not None and event.logical_time > limit:
            continue
        state[event.object] = ObjectLocationState(
            object_id=event.object,
            location=event.to_location,
            supporting_event_id=event.id,
            logical_time=event.logical_time,
        )
    return MappingProxyType(state)


def subjective_state(
    events: Iterable[RelocationEventV1],
    observations: Iterable[DirectObservationV1],
    agent: str,
    *,
    at_time: int | None = None,
) -> Mapping[str, ObjectLocationState]:
    """Replay only relocation events directly observed by one agent."""

    if not isinstance(agent, str) or not agent:
        raise ValueError("story replay agent must be a non-empty string")
    observed_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, DirectObservationV1):
            raise TypeError(
                "story replay observations must be DirectObservationV1 values"
            )
        if observation.agent == agent:
            observed_ids.add(observation.event)
    return objective_state(
        (event for event in events if event.id in observed_ids),
        at_time=at_time,
    )


def latest_object_location(
    state: Mapping[str, ObjectLocationState],
    object_id: str,
) -> ObjectLocationState:
    """Return the latest supported object location or fail closed."""

    if not isinstance(object_id, str) or not object_id:
        raise ValueError("story replay object id must be a non-empty string")
    try:
        value = state[object_id]
    except KeyError as error:
        raise ValueError(f"no supported location for object {object_id!r}") from error
    if not isinstance(value, ObjectLocationState):
        raise TypeError("story replay state must contain ObjectLocationState values")
    return value
