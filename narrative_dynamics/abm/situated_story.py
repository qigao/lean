"""Objective and perspective story queries over situated ABM rounds."""

from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_contracts import (
    SituatedWorldModel,
    SituatedWorldState,
    validate_situated_state,
)
from narrative_dynamics.abm.situated import (
    SituatedActionIntent,
    SituatedActionKind,
    SituatedObservation,
    SituatedRoundResult,
    SituatedWorldEvent,
    resolve_situated_round,
)


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class SituatedPerspectiveEvent:
    """One observed event joined to the exact private access record."""

    event: SituatedWorldEvent
    observation: SituatedObservation

    def __post_init__(self) -> None:
        if not isinstance(self.event, SituatedWorldEvent) or not isinstance(self.observation, SituatedObservation):
            raise TypeError("situated perspective item requires an event and observation")
        if self.observation.event_id != self.event.event_id or self.observation.event_hash != self.event.content_hash:
            raise ValueError("situated perspective observation must reference the exact event")

    def to_dict(self) -> dict[str, object]:
        return {"event": self.event.to_dict(), "observation": self.observation.to_dict()}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedEventExplanation:
    event_id: str
    success: bool
    outcome: str
    rule: str
    direct_cause_event_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "event_id": self.event_id,
            "success": self.success,
            "outcome": self.outcome,
            "rule": self.rule,
            "direct_cause_event_ids": list(self.direct_cause_event_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedStory:
    model_id: str
    model_hash: str
    initial_state: SituatedWorldState
    rounds: tuple[SituatedRoundResult, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("situated story model id must be non-empty")
        if not isinstance(self.model_hash, str) or _CONTENT_HASH.fullmatch(self.model_hash) is None:
            raise ValueError("situated story model hash must be a content hash")
        if not isinstance(self.initial_state, SituatedWorldState):
            raise TypeError("situated story initial state must be SituatedWorldState")
        if self.initial_state.round_index != 0 or self.initial_state.parent_state_hash is not None:
            raise ValueError("situated story must begin at round zero")
        if self.initial_state.model_id != self.model_id or self.initial_state.model_hash != self.model_hash:
            raise ValueError("situated story initial state must reference its model")
        if not isinstance(self.rounds, tuple) or any(not isinstance(item, SituatedRoundResult) for item in self.rounds):
            raise TypeError("situated story rounds must be a tuple of SituatedRoundResult values")
        expected = self.initial_state
        known_events: dict[str, SituatedWorldEvent] = {}
        prior_observed: dict[str, set[str]] = {item.agent_id: set() for item in self.initial_state.agents}
        for result in self.rounds:
            if result.prior_state != expected:
                raise ValueError("situated story rounds must form one exact state chain")
            current_event_ids: set[str] = set()
            event_by_id = {item.event_id: item for item in result.events}
            if len(event_by_id) != len(result.events):
                raise ValueError("situated story event ids must be globally unique")
            intents = {item.action_id: item for item in result.intents}
            for event in result.events:
                if event.event_id in known_events:
                    raise ValueError("situated story event ids must be globally unique")
                for cause_id in event.cause_event_ids:
                    if cause_id not in known_events and cause_id not in current_event_ids:
                        raise ValueError("situated causal reference must identify an earlier event")
                intent = intents[event.action_id]
                if intent.kind is SituatedActionKind.TELL:
                    for cause_id in intent.source_event_ids:
                        if cause_id not in known_events:
                            raise ValueError("situated causal reference must identify an earlier event")
                        if cause_id not in prior_observed[event.actor_agent_id]:
                            raise ValueError("tell source must be an event the speaker previously observed")
                current_event_ids.add(event.event_id)
                known_events[event.event_id] = event
            seen_this_round: dict[str, set[str]] = {agent_id: set() for agent_id in prior_observed}
            for observation in result.observations:
                event = event_by_id.get(observation.event_id)
                if event is None or observation.event_hash != event.content_hash:
                    raise ValueError("situated observation must reference an exact same-round event")
                if observation.agent_id not in prior_observed:
                    raise ValueError("situated observation agent must belong to the story")
                seen_this_round[observation.agent_id].add(observation.event_id)
            for agent_id, event_ids in seen_this_round.items():
                prior_observed[agent_id].update(event_ids)
            expected = result.next_state

    @property
    def current_state(self) -> SituatedWorldState:
        return self.initial_state if not self.rounds else self.rounds[-1].next_state

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_state": self.initial_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_situated_story(model: SituatedWorldModel, initial_state: SituatedWorldState) -> SituatedStory:
    validate_situated_state(model, initial_state)
    if initial_state.round_index != 0:
        raise ValueError("situated story initialization requires round zero")
    return SituatedStory(model.model_id, model.content_hash, initial_state)


def advance_situated_story(
    model: SituatedWorldModel,
    story: SituatedStory,
    intents: tuple[SituatedActionIntent, ...],
) -> SituatedStory:
    if not isinstance(story, SituatedStory):
        raise TypeError("situated story must be SituatedStory")
    if story.model_id != model.model_id or story.model_hash != model.content_hash:
        raise ValueError("situated story must reference the exact model")
    result = resolve_situated_round(model, story.current_state, intents)
    return SituatedStory(story.model_id, story.model_hash, story.initial_state, story.rounds + (result,))


def replay_situated_story(
    model: SituatedWorldModel,
    initial_state: SituatedWorldState,
    action_schedule: tuple[tuple[SituatedActionIntent, ...], ...],
) -> SituatedStory:
    if not isinstance(action_schedule, tuple) or any(not isinstance(item, tuple) for item in action_schedule):
        raise TypeError("situated action schedule must be a tuple of intent tuples")
    story = initialize_situated_story(model, initial_state)
    for intents in action_schedule:
        story = advance_situated_story(model, story, intents)
    return story


def objective_timeline(story: SituatedStory) -> tuple[SituatedWorldEvent, ...]:
    if not isinstance(story, SituatedStory):
        raise TypeError("objective timeline requires a SituatedStory")
    return tuple(event for result in story.rounds for event in result.events)


def perspective_timeline(story: SituatedStory, agent_id: str) -> tuple[SituatedPerspectiveEvent, ...]:
    agent_ids = {item.agent_id for item in story.initial_state.agents}
    if agent_id not in agent_ids:
        raise ValueError("perspective agent must belong to the story")
    events = {item.event_id: item for item in objective_timeline(story)}
    return tuple(
        SituatedPerspectiveEvent(events[observation.event_id], observation)
        for result in story.rounds
        for observation in result.observations
        if observation.agent_id == agent_id
    )


def _event_index(story: SituatedStory) -> tuple[dict[str, SituatedWorldEvent], dict[str, int]]:
    timeline = objective_timeline(story)
    return ({item.event_id: item for item in timeline}, {item.event_id: index for index, item in enumerate(timeline)})


def _require_event(story: SituatedStory, event_id: str) -> SituatedWorldEvent:
    events, _ = _event_index(story)
    try:
        return events[event_id]
    except KeyError as error:
        raise ValueError("event id must belong to the situated story") from error


def direct_causes(story: SituatedStory, event_id: str) -> tuple[SituatedWorldEvent, ...]:
    event = _require_event(story, event_id)
    events, order = _event_index(story)
    return tuple(sorted((events[item] for item in event.cause_event_ids), key=lambda item: order[item.event_id]))


def causal_ancestors(story: SituatedStory, event_id: str) -> tuple[SituatedWorldEvent, ...]:
    target = _require_event(story, event_id)
    events, order = _event_index(story)
    found: set[str] = set()
    pending = list(target.cause_event_ids)
    while pending:
        current = pending.pop()
        if current in found:
            continue
        found.add(current)
        pending.extend(events[current].cause_event_ids)
    return tuple(events[item] for item in sorted(found, key=lambda item: order[item]))


def information_chain(story: SituatedStory, event_id: str) -> tuple[SituatedWorldEvent, ...]:
    target = _require_event(story, event_id)
    chain = causal_ancestors(story, event_id) + (target,)
    return tuple(item for item in chain if item.kind in {SituatedActionKind.INSPECT, SituatedActionKind.TELL})


_OUTCOME_RULES = {
    "waited": "the actor intentionally made no world change",
    "looked": "the actor inspected visible contents of its current place",
    "moved": "an open outgoing passage connected the origin to the destination",
    "unknown_passage": "the requested passage does not exist",
    "not_at_passage_source": "the actor was not at the requested passage source",
    "passage_closed": "the requested passage was closed",
    "inspected": "the object was co-located with or held by the actor",
    "unknown_object": "the requested object does not exist",
    "object_not_accessible": "the object was neither co-located with nor held by the actor",
    "taken": "the portable object was co-located and inventory capacity was available",
    "object_not_portable": "the requested object was not portable",
    "object_not_co_located": "the requested object was not at the actor's place",
    "inventory_full": "the actor had no remaining inventory capacity",
    "take_conflict_lost": "another eligible action won the canonical object conflict",
    "dropped": "the actor held the object and placed it at the current place",
    "object_not_held": "the actor did not hold the requested object",
    "told": "a local audible claim was emitted",
}


def explain_situated_event(story: SituatedStory, event_id: str) -> SituatedEventExplanation:
    event = _require_event(story, event_id)
    return SituatedEventExplanation(
        event_id=event.event_id,
        success=event.success,
        outcome=event.outcome,
        rule=_OUTCOME_RULES[event.outcome],
        direct_cause_event_ids=tuple(item.event_id for item in direct_causes(story, event_id)),
    )


__all__ = (
    "SituatedPerspectiveEvent", "SituatedEventExplanation", "SituatedStory",
    "initialize_situated_story", "advance_situated_story", "replay_situated_story",
    "objective_timeline", "perspective_timeline", "direct_causes", "causal_ancestors",
    "information_chain", "explain_situated_event",
)
