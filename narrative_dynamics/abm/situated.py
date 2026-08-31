"""Synchronous action resolution and local perception for situated worlds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_contracts import (
    AgentBodyState,
    EvidenceFact,
    PassageState,
    SituatedWorldModel,
    SituatedWorldState,
    WorldObjectState,
    validate_situated_state,
)


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


class SituatedActionKind(str, Enum):
    WAIT = "wait"
    MOVE = "move"
    LOOK = "look"
    INSPECT = "inspect"
    TAKE = "take"
    DROP = "drop"
    TELL = "tell"


class ObservationChannel(str, Enum):
    SELF = "self"
    VISUAL = "visual"
    AUDITORY = "auditory"
    INSPECTION = "inspection"


@dataclass(frozen=True)
class SituatedActionIntent:
    action_id: str
    agent_id: str
    kind: SituatedActionKind
    target_id: str | None = None
    message: str | None = None
    source_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, label="situated action id"))
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="situated action agent id"))
        if not isinstance(self.kind, SituatedActionKind):
            raise TypeError("situated action kind must be SituatedActionKind")
        object.__setattr__(self, "target_id", _optional_text(self.target_id, label="situated action target id"))
        object.__setattr__(self, "message", _optional_text(self.message, label="situated action message"))
        if not isinstance(self.source_event_ids, tuple):
            raise TypeError("situated action source event ids must be a tuple")
        refs = tuple(_text(item, label="situated action source event id") for item in self.source_event_ids)
        if len(set(refs)) != len(refs):
            raise ValueError("situated action source event ids must be unique")
        object.__setattr__(self, "source_event_ids", tuple(sorted(refs)))
        targeted = {SituatedActionKind.MOVE, SituatedActionKind.INSPECT, SituatedActionKind.TAKE, SituatedActionKind.DROP}
        if self.kind in targeted and self.target_id is None:
            raise ValueError(f"{self.kind.value} action requires a target id")
        if self.kind not in targeted and self.target_id is not None:
            raise ValueError(f"{self.kind.value} action cannot have a target id")
        if self.kind is SituatedActionKind.TELL:
            if self.message is None:
                raise ValueError("tell action requires a message")
        elif self.message is not None or refs:
            raise ValueError("only tell actions may carry a message or source events")

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "agent_id": self.agent_id,
            "kind": self.kind.value,
            "target_id": self.target_id,
            "message": self.message,
            "source_event_ids": list(self.source_event_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedWorldEvent:
    event_id: str
    round_index: int
    sequence: int
    action_id: str
    kind: SituatedActionKind
    actor_agent_id: str
    place_id: str
    target_id: str | None
    success: bool
    outcome: str
    details: tuple[EvidenceFact, ...] = ()
    cause_event_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("event_id", "action_id", "actor_agent_id", "place_id", "outcome"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"situated event {name}"))
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index <= 0:
            raise ValueError("situated event round index must be a positive integer")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence <= 0:
            raise ValueError("situated event sequence must be a positive integer")
        if not isinstance(self.kind, SituatedActionKind):
            raise TypeError("situated event kind must be SituatedActionKind")
        object.__setattr__(self, "target_id", _optional_text(self.target_id, label="situated event target id"))
        if not isinstance(self.success, bool):
            raise TypeError("situated event success must be boolean")
        if not isinstance(self.details, tuple) or any(not isinstance(item, EvidenceFact) for item in self.details):
            raise TypeError("situated event details must be a tuple of EvidenceFact values")
        if not isinstance(self.cause_event_ids, tuple):
            raise TypeError("situated event cause ids must be a tuple")
        causes = tuple(_text(item, label="situated event cause id") for item in self.cause_event_ids)
        if len(set(causes)) != len(causes):
            raise ValueError("situated event cause ids must be unique")
        object.__setattr__(self, "details", tuple(sorted(self.details, key=lambda item: (item.name, item.value))))
        object.__setattr__(self, "cause_event_ids", tuple(sorted(causes)))

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "round_index": self.round_index,
            "sequence": self.sequence,
            "action_id": self.action_id,
            "kind": self.kind.value,
            "actor_agent_id": self.actor_agent_id,
            "place_id": self.place_id,
            "target_id": self.target_id,
            "success": self.success,
            "outcome": self.outcome,
            "details": [item.to_dict() for item in self.details],
            "cause_event_ids": list(self.cause_event_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedObservation:
    observation_id: str
    round_index: int
    agent_id: str
    event_id: str
    event_hash: str
    channel: ObservationChannel

    def __post_init__(self) -> None:
        for name in ("observation_id", "agent_id", "event_id", "event_hash"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"situated observation {name}"))
        if _CONTENT_HASH.fullmatch(self.event_hash) is None:
            raise ValueError("situated observation event hash must be a sha256 content hash")
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index <= 0:
            raise ValueError("situated observation round index must be positive")
        if not isinstance(self.channel, ObservationChannel):
            raise TypeError("situated observation channel must be ObservationChannel")

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "channel": self.channel.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _expected_observation_channels(
    prior_state: SituatedWorldState,
    event: SituatedWorldEvent,
) -> dict[str, ObservationChannel]:
    bodies = {item.agent_id: item for item in prior_state.agents}
    if event.actor_agent_id not in bodies:
        raise ValueError("situated event actor must belong to the prior state")
    audience = {
        event.actor_agent_id: (
            ObservationChannel.INSPECTION
            if event.kind is SituatedActionKind.INSPECT
            else ObservationChannel.SELF
        )
    }
    if event.kind is SituatedActionKind.TELL:
        observable_places = {event.place_id}
        channel = ObservationChannel.AUDITORY
    elif event.kind in {SituatedActionKind.TAKE, SituatedActionKind.DROP}:
        observable_places = {event.place_id}
        channel = ObservationChannel.VISUAL
    elif event.kind is SituatedActionKind.MOVE:
        observable_places = {event.place_id}
        if event.success:
            destinations = tuple(
                item.value
                for item in event.details
                if item.name == "destination_place_id"
            )
            if len(destinations) != 1:
                raise ValueError(
                    "successful move event must identify exactly one destination"
                )
            observable_places.add(destinations[0])
        channel = ObservationChannel.VISUAL
    else:
        return audience
    for body in prior_state.agents:
        if (
            body.agent_id != event.actor_agent_id
            and body.place_id in observable_places
        ):
            audience[body.agent_id] = channel
    return audience


def _validate_round_observation_projection(
    prior_state: SituatedWorldState,
    events: tuple[SituatedWorldEvent, ...],
    observations: tuple[SituatedObservation, ...],
) -> None:
    observation_ids = tuple(item.observation_id for item in observations)
    if len(set(observation_ids)) != len(observation_ids):
        raise ValueError("situated round observation ids must be unique")
    pairs = tuple((item.agent_id, item.event_id) for item in observations)
    if len(set(pairs)) != len(pairs):
        raise ValueError("situated round observation agent/event pairs must be unique")
    event_by_id = {item.event_id: item for item in events}
    if len(event_by_id) != len(events):
        raise ValueError("situated round event ids must be unique")
    actual: dict[tuple[str, str], ObservationChannel] = {}
    for observation in observations:
        event = event_by_id.get(observation.event_id)
        if (
            event is None
            or observation.event_hash != event.content_hash
            or observation.round_index != event.round_index
        ):
            raise ValueError(
                "situated observation must reference an exact same-round event"
            )
        actual[(observation.agent_id, observation.event_id)] = observation.channel
    expected = {
        (agent_id, event.event_id): channel
        for event in events
        for agent_id, channel in _expected_observation_channels(
            prior_state,
            event,
        ).items()
    }
    if actual != expected:
        raise ValueError(
            "situated round observations must equal the authorized observation projection"
        )


@dataclass(frozen=True)
class SituatedRoundResult:
    prior_state: SituatedWorldState
    intents: tuple[SituatedActionIntent, ...]
    events: tuple[SituatedWorldEvent, ...]
    observations: tuple[SituatedObservation, ...]
    next_state: SituatedWorldState

    def __post_init__(self) -> None:
        if not isinstance(self.prior_state, SituatedWorldState) or not isinstance(self.next_state, SituatedWorldState):
            raise TypeError("situated round states must be SituatedWorldState")
        typed = (
            ("intents", self.intents, SituatedActionIntent),
            ("events", self.events, SituatedWorldEvent),
            ("observations", self.observations, SituatedObservation),
        )
        for name, values, expected in typed:
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"situated round {name} must be a tuple of {expected.__name__} values")
        if self.next_state.round_index != self.prior_state.round_index + 1:
            raise ValueError("situated round must advance exactly one round")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("situated round next state must reference its prior state")
        _validate_round_observation_projection(
            self.prior_state,
            self.events,
            self.observations,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_state_hash": self.prior_state.content_hash,
            "intents": [item.to_dict() for item in self.intents],
            "events": [item.to_dict() for item in self.events],
            "observations": [item.to_dict() for item in self.observations],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass
class _Resolution:
    success: bool
    outcome: str
    place_id: str
    details: tuple[EvidenceFact, ...] = ()
    causes: tuple[str, ...] = ()


def _canonical_intents(model: SituatedWorldModel, state: SituatedWorldState, intents: tuple[SituatedActionIntent, ...]) -> tuple[SituatedActionIntent, ...]:
    if not isinstance(intents, tuple) or any(not isinstance(item, SituatedActionIntent) for item in intents):
        raise TypeError("situated round intents must be a tuple of SituatedActionIntent values")
    model_agent_ids = {item.agent_id for item in model.agents}
    if any(item.agent_id not in model_agent_ids for item in intents):
        raise ValueError("situated action agent must belong to the model")
    agent_ids = tuple(item.agent_id for item in intents)
    if len(set(agent_ids)) != len(agent_ids):
        raise ValueError("situated round allows one action per agent")
    explicit_ids = tuple(item.action_id for item in intents)
    if len(set(explicit_ids)) != len(explicit_ids):
        raise ValueError("situated action ids must be unique")
    by_agent = {item.agent_id: item for item in intents}
    round_index = state.round_index + 1
    complete = tuple(
        by_agent.get(agent.agent_id, SituatedActionIntent(f"r{round_index:04d}:wait:{agent.agent_id}", agent.agent_id, SituatedActionKind.WAIT))
        for agent in model.agents
    )
    ids = tuple(item.action_id for item in complete)
    if len(set(ids)) != len(ids):
        raise ValueError("situated action ids must be unique")
    return tuple(sorted(complete, key=lambda item: item.action_id))


def resolve_situated_round(
    model: SituatedWorldModel,
    state: SituatedWorldState,
    intents: tuple[SituatedActionIntent, ...],
) -> SituatedRoundResult:
    """Resolve one deterministic synchronous round from a prior snapshot."""

    validate_situated_state(model, state)
    canonical = _canonical_intents(model, state, intents)
    round_index = state.round_index + 1
    event_ids = {intent.action_id: f"r{round_index:04d}:e{index:04d}" for index, intent in enumerate(canonical, 1)}
    bodies = {item.agent_id: item for item in state.agents}
    objects = {item.object_id: item for item in state.objects}
    passages = {item.passage_id: item for item in model.passages}
    passage_states = {item.passage_id: item for item in state.passages}
    agent_specs = {item.agent_id: item for item in model.agents}
    object_specs = {item.object_id: item for item in model.objects}
    held_counts = {agent_id: 0 for agent_id in bodies}
    for item in state.objects:
        if item.holder_agent_id is not None:
            held_counts[item.holder_agent_id] += 1

    resolutions: dict[str, _Resolution] = {}
    eligible_takes: dict[str, list[SituatedActionIntent]] = {}
    for intent in canonical:
        body = bodies[intent.agent_id]
        place_id = body.place_id
        if intent.kind is SituatedActionKind.WAIT:
            resolutions[intent.action_id] = _Resolution(True, "waited", place_id)
        elif intent.kind is SituatedActionKind.LOOK:
            details = []
            for other in state.agents:
                if other.place_id == place_id and other.agent_id != intent.agent_id:
                    details.append(EvidenceFact(f"agent:{other.agent_id}", agent_specs[other.agent_id].role))
            for obj in state.objects:
                if obj.place_id == place_id:
                    details.append(EvidenceFact(f"object:{obj.object_id}", object_specs[obj.object_id].kind))
            resolutions[intent.action_id] = _Resolution(True, "looked", place_id, tuple(details))
        elif intent.kind is SituatedActionKind.MOVE:
            passage = passages.get(intent.target_id)
            if passage is None:
                resolutions[intent.action_id] = _Resolution(False, "unknown_passage", place_id)
            elif passage.source_place_id != place_id:
                resolutions[intent.action_id] = _Resolution(False, "not_at_passage_source", place_id)
            elif not passage_states[passage.passage_id].open:
                resolutions[intent.action_id] = _Resolution(False, "passage_closed", place_id)
            else:
                resolutions[intent.action_id] = _Resolution(True, "moved", place_id, (EvidenceFact("destination_place_id", passage.target_place_id),))
        elif intent.kind is SituatedActionKind.INSPECT:
            obj = objects.get(intent.target_id)
            if obj is None:
                resolutions[intent.action_id] = _Resolution(False, "unknown_object", place_id)
            elif obj.holder_agent_id != intent.agent_id and obj.place_id != place_id:
                resolutions[intent.action_id] = _Resolution(False, "object_not_accessible", place_id)
            else:
                spec = object_specs[obj.object_id]
                details = (EvidenceFact("object_kind", spec.kind),) + spec.evidence
                resolutions[intent.action_id] = _Resolution(True, "inspected", place_id, details)
        elif intent.kind is SituatedActionKind.TAKE:
            obj = objects.get(intent.target_id)
            if obj is None:
                resolutions[intent.action_id] = _Resolution(False, "unknown_object", place_id)
            elif not object_specs[obj.object_id].portable:
                resolutions[intent.action_id] = _Resolution(False, "object_not_portable", place_id)
            elif obj.place_id != place_id:
                resolutions[intent.action_id] = _Resolution(False, "object_not_co_located", place_id)
            elif held_counts[intent.agent_id] >= agent_specs[intent.agent_id].inventory_capacity:
                resolutions[intent.action_id] = _Resolution(False, "inventory_full", place_id)
            else:
                eligible_takes.setdefault(obj.object_id, []).append(intent)
        elif intent.kind is SituatedActionKind.DROP:
            obj = objects.get(intent.target_id)
            if obj is None:
                resolutions[intent.action_id] = _Resolution(False, "unknown_object", place_id)
            elif obj.holder_agent_id != intent.agent_id:
                resolutions[intent.action_id] = _Resolution(False, "object_not_held", place_id)
            else:
                resolutions[intent.action_id] = _Resolution(True, "dropped", place_id)
        else:
            resolutions[intent.action_id] = _Resolution(
                True,
                "told",
                place_id,
                (EvidenceFact("message", intent.message or ""),),
                intent.source_event_ids,
            )

    for object_id, contenders in eligible_takes.items():
        ordered = sorted(contenders, key=lambda item: item.action_id)
        winner = ordered[0]
        resolutions[winner.action_id] = _Resolution(True, "taken", bodies[winner.agent_id].place_id)
        winner_event_id = event_ids[winner.action_id]
        for loser in ordered[1:]:
            resolutions[loser.action_id] = _Resolution(False, "take_conflict_lost", bodies[loser.agent_id].place_id, causes=(winner_event_id,))

    next_bodies = dict(bodies)
    next_objects = dict(objects)
    for intent in canonical:
        resolution = resolutions[intent.action_id]
        if not resolution.success:
            continue
        if intent.kind is SituatedActionKind.MOVE:
            next_bodies[intent.agent_id] = AgentBodyState(intent.agent_id, passages[intent.target_id].target_place_id)
        elif intent.kind is SituatedActionKind.TAKE:
            next_objects[intent.target_id] = WorldObjectState(intent.target_id, None, intent.agent_id)
        elif intent.kind is SituatedActionKind.DROP:
            next_objects[intent.target_id] = WorldObjectState(intent.target_id, bodies[intent.agent_id].place_id, None)

    events = tuple(
        SituatedWorldEvent(
            event_id=event_ids[intent.action_id],
            round_index=round_index,
            sequence=index,
            action_id=intent.action_id,
            kind=intent.kind,
            actor_agent_id=intent.agent_id,
            place_id=resolutions[intent.action_id].place_id,
            target_id=intent.target_id,
            success=resolutions[intent.action_id].success,
            outcome=resolutions[intent.action_id].outcome,
            details=resolutions[intent.action_id].details,
            cause_event_ids=resolutions[intent.action_id].causes,
        )
        for index, intent in enumerate(canonical, 1)
    )

    next_state = SituatedWorldState(
        model_id=model.model_id,
        model_hash=model.content_hash,
        round_index=round_index,
        parent_state_hash=state.content_hash,
        agents=tuple(next_bodies.values()),
        objects=tuple(next_objects.values()),
        passages=tuple(PassageState(item.passage_id, item.open) for item in state.passages),
    )
    validate_situated_state(model, next_state)

    observations: list[SituatedObservation] = []
    for event in events:
        audience = _expected_observation_channels(state, event)
        for agent_id, channel in sorted(audience.items()):
            observations.append(SituatedObservation(
                observation_id=f"{event.event_id}:o:{agent_id}",
                round_index=round_index,
                agent_id=agent_id,
                event_id=event.event_id,
                event_hash=event.content_hash,
                channel=channel,
            ))
    return SituatedRoundResult(
        prior_state=state,
        intents=canonical,
        events=events,
        observations=tuple(sorted(observations, key=lambda item: (item.event_id, item.agent_id))),
        next_state=next_state,
    )


__all__ = (
    "SituatedActionKind", "ObservationChannel", "SituatedActionIntent",
    "SituatedWorldEvent", "SituatedObservation", "SituatedRoundResult",
    "resolve_situated_round",
)
