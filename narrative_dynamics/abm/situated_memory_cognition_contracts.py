"""Immutable contracts for V13 memory-augmented situated cognition."""

from __future__ import annotations

from dataclasses import dataclass

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentMindState,
    SituatedCognitiveModel,
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_memory_contracts import SituatedMemoryPolicy
from narrative_dynamics.abm.situated_story import SituatedStory


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strings(values: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(result))


def _bounded_limit(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 1000:
        raise ValueError(f"{label} must be between 1 and 1000")
    return value


def _unit_interval(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return number


@dataclass(frozen=True)
class SituatedMemoryRecallCue:
    cue_id: str
    text: str
    required_place_ids: tuple[str, ...] = ()
    event_kinds: tuple[SituatedActionKind, ...] = ()
    channels: tuple[ObservationChannel, ...] = ()
    min_confidence: float = 0.0
    limit: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "cue_id", _text(self.cue_id, label="situated memory recall cue id"))
        object.__setattr__(self, "text", _text(self.text, label="situated memory recall cue text"))
        object.__setattr__(self, "required_place_ids", _strings(self.required_place_ids, label="situated memory recall required place ids"))
        if not isinstance(self.event_kinds, tuple) or any(not isinstance(item, SituatedActionKind) for item in self.event_kinds):
            raise TypeError("situated memory recall event kinds must be a tuple of SituatedActionKind values")
        if len(set(self.event_kinds)) != len(self.event_kinds):
            raise ValueError("situated memory recall event kinds must be unique")
        object.__setattr__(self, "event_kinds", tuple(sorted(self.event_kinds, key=lambda item: item.value)))
        if not isinstance(self.channels, tuple) or any(not isinstance(item, ObservationChannel) for item in self.channels):
            raise TypeError("situated memory recall channels must be a tuple of ObservationChannel values")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("situated memory recall channels must be unique")
        object.__setattr__(self, "channels", tuple(sorted(self.channels, key=lambda item: item.value)))
        object.__setattr__(self, "min_confidence", _unit_interval(self.min_confidence, label="situated memory recall minimum confidence"))
        object.__setattr__(self, "limit", _bounded_limit(self.limit, label="situated memory recall cue limit"))

    def to_dict(self) -> dict[str, object]:
        return {
            "cue_id": self.cue_id,
            "text": self.text,
            "required_place_ids": list(self.required_place_ids),
            "event_kinds": [item.value for item in self.event_kinds],
            "channels": [item.value for item in self.channels],
            "min_confidence": self.min_confidence,
            "limit": self.limit,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedAgentRecallPolicy:
    agent_id: str
    cues: tuple[SituatedMemoryRecallCue, ...] = ()
    max_memories_per_round: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="situated recall policy agent id"))
        if not isinstance(self.cues, tuple) or any(not isinstance(item, SituatedMemoryRecallCue) for item in self.cues):
            raise TypeError("situated recall policy cues must be a tuple of SituatedMemoryRecallCue values")
        identities = tuple(item.cue_id for item in self.cues)
        if len(set(identities)) != len(identities):
            raise ValueError("situated recall cue ids must be unique per agent")
        object.__setattr__(self, "cues", tuple(sorted(self.cues, key=lambda item: item.cue_id)))
        object.__setattr__(
            self,
            "max_memories_per_round",
            _bounded_limit(self.max_memories_per_round, label="situated recall maximum memories per round"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "cues": [item.to_dict() for item in self.cues],
            "max_memories_per_round": self.max_memories_per_round,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedMemoryCognitiveModel:
    model_id: str
    version: str
    cognitive_model: SituatedCognitiveModel
    memory_policy: SituatedMemoryPolicy
    agents: tuple[SituatedAgentRecallPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated memory cognitive model id"))
        object.__setattr__(self, "version", _text(self.version, label="situated memory cognitive model version"))
        if not isinstance(self.cognitive_model, SituatedCognitiveModel):
            raise TypeError("situated memory cognitive model requires SituatedCognitiveModel")
        if not isinstance(self.memory_policy, SituatedMemoryPolicy):
            raise TypeError("situated memory cognitive model requires SituatedMemoryPolicy")
        if not isinstance(self.agents, tuple) or any(not isinstance(item, SituatedAgentRecallPolicy) for item in self.agents):
            raise TypeError("situated memory cognitive agents must be a tuple of SituatedAgentRecallPolicy values")
        identities = tuple(item.agent_id for item in self.agents)
        if len(set(identities)) != len(identities):
            raise ValueError("situated memory cognitive agent ids must be unique")
        expected = {item.agent_id for item in self.cognitive_model.agents}
        if set(identities) != expected:
            raise ValueError("situated memory cognitive policies must cover exact cognitive agent roster")
        place_ids = {item.place_id for item in self.cognitive_model.world_model.places}
        if any(not set(cue.required_place_ids).issubset(place_ids) for policy in self.agents for cue in policy.cues):
            raise ValueError("situated memory recall required place must belong to the world")
        object.__setattr__(self, "agents", tuple(sorted(self.agents, key=lambda item: item.agent_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "cognitive_model_hash": self.cognitive_model.content_hash,
            "memory_policy_hash": self.memory_policy.content_hash,
            "agents": [item.to_dict() for item in self.agents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_situated_memory_cognition(
    model: SituatedMemoryCognitiveModel,
    story: SituatedStory,
) -> SituatedCognitiveState:
    """Create a private cognitive checkpoint at the current situated story round."""

    if not isinstance(model, SituatedMemoryCognitiveModel) or not isinstance(story, SituatedStory):
        raise TypeError("situated memory cognition initialization requires model and story")
    cognition = model.cognitive_model
    if story.model_id != cognition.world_model.model_id or story.model_hash != cognition.world_model.content_hash:
        raise ValueError("situated memory cognitive model and story must bind the exact world")
    places = {item.agent_id: item.place_id for item in story.current_state.agents}
    round_index = story.current_state.round_index
    state = SituatedCognitiveState(
        cognition.model_id,
        cognition.content_hash,
        round_index,
        None,
        story.content_hash,
        tuple(
            SituatedAgentMindState(
                item.agent_id,
                item.prior_belief,
                places[item.agent_id],
                observation_floor_round=round_index,
            )
            for item in cognition.agents
        ),
        checkpoint=True,
    )
    validate_situated_cognitive_state(cognition, story, state)
    return state


__all__ = (
    "SituatedMemoryRecallCue",
    "SituatedAgentRecallPolicy",
    "SituatedMemoryCognitiveModel",
    "initialize_situated_memory_cognition",
)
