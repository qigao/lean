"""Immutable contracts for the V10 situated story world."""

from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _nonnegative_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


@dataclass(frozen=True)
class EvidenceFact:
    """One immutable fact physically encoded by a world object."""

    name: str
    value: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="evidence fact name"))
        object.__setattr__(self, "value", _text(self.value, label="evidence fact value"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PlaceSpec:
    place_id: str
    label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "place_id", _text(self.place_id, label="place id"))
        object.__setattr__(self, "label", _text(self.label, label="place label"))

    def to_dict(self) -> dict[str, object]:
        return {"place_id": self.place_id, "label": self.label}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PassageSpec:
    passage_id: str
    source_place_id: str
    target_place_id: str
    initially_open: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "passage_id", _text(self.passage_id, label="passage id"))
        object.__setattr__(self, "source_place_id", _text(self.source_place_id, label="passage source place id"))
        object.__setattr__(self, "target_place_id", _text(self.target_place_id, label="passage target place id"))
        if self.source_place_id == self.target_place_id:
            raise ValueError("passage endpoints must differ")
        if not isinstance(self.initially_open, bool):
            raise TypeError("passage initially open must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "passage_id": self.passage_id,
            "source_place_id": self.source_place_id,
            "target_place_id": self.target_place_id,
            "initially_open": self.initially_open,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class EmbodiedAgentSpec:
    agent_id: str
    role: str
    initial_place_id: str
    inventory_capacity: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="embodied agent id"))
        object.__setattr__(self, "role", _text(self.role, label="embodied agent role"))
        object.__setattr__(self, "initial_place_id", _text(self.initial_place_id, label="agent initial place id"))
        object.__setattr__(self, "inventory_capacity", _nonnegative_integer(self.inventory_capacity, label="agent inventory capacity"))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "initial_place_id": self.initial_place_id,
            "inventory_capacity": self.inventory_capacity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class WorldObjectSpec:
    object_id: str
    kind: str
    initial_place_id: str
    portable: bool = False
    evidence: tuple[EvidenceFact, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, label="world object id"))
        object.__setattr__(self, "kind", _text(self.kind, label="world object kind"))
        object.__setattr__(self, "initial_place_id", _text(self.initial_place_id, label="object initial place id"))
        if not isinstance(self.portable, bool):
            raise TypeError("world object portable must be boolean")
        if not isinstance(self.evidence, tuple) or any(not isinstance(item, EvidenceFact) for item in self.evidence):
            raise TypeError("world object evidence must be a tuple of EvidenceFact values")
        names = tuple(item.name for item in self.evidence)
        if len(set(names)) != len(names):
            raise ValueError("world object evidence names must be unique")
        object.__setattr__(self, "evidence", tuple(sorted(self.evidence, key=lambda item: item.name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "kind": self.kind,
            "initial_place_id": self.initial_place_id,
            "portable": self.portable,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedWorldModel:
    model_id: str
    version: str
    places: tuple[PlaceSpec, ...]
    passages: tuple[PassageSpec, ...]
    agents: tuple[EmbodiedAgentSpec, ...]
    objects: tuple[WorldObjectSpec, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated model id"))
        object.__setattr__(self, "version", _text(self.version, label="situated model version"))
        for name, values, expected in (
            ("places", self.places, PlaceSpec),
            ("passages", self.passages, PassageSpec),
            ("agents", self.agents, EmbodiedAgentSpec),
            ("objects", self.objects, WorldObjectSpec),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"situated model {name} must be a tuple of {expected.__name__} values")
        if not self.places:
            raise ValueError("situated model requires at least one place")
        if not self.agents:
            raise ValueError("situated model requires at least one agent")
        identities = (
            ("place ids", tuple(item.place_id for item in self.places)),
            ("passage ids", tuple(item.passage_id for item in self.passages)),
            ("agent ids", tuple(item.agent_id for item in self.agents)),
            ("object ids", tuple(item.object_id for item in self.objects)),
        )
        for label, ids in identities:
            if len(set(ids)) != len(ids):
                raise ValueError(f"situated model {label} must be unique")
        place_ids = set(identities[0][1])
        if any(item.source_place_id not in place_ids or item.target_place_id not in place_ids for item in self.passages):
            raise ValueError("passage endpoint must reference a model place")
        if any(item.initial_place_id not in place_ids for item in self.agents):
            raise ValueError("agent initial place must reference a model place")
        if any(item.initial_place_id not in place_ids for item in self.objects):
            raise ValueError("object initial place must reference a model place")
        object.__setattr__(self, "places", tuple(sorted(self.places, key=lambda item: item.place_id)))
        object.__setattr__(self, "passages", tuple(sorted(self.passages, key=lambda item: item.passage_id)))
        object.__setattr__(self, "agents", tuple(sorted(self.agents, key=lambda item: item.agent_id)))
        object.__setattr__(self, "objects", tuple(sorted(self.objects, key=lambda item: item.object_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "places": [item.to_dict() for item in self.places],
            "passages": [item.to_dict() for item in self.passages],
            "agents": [item.to_dict() for item in self.agents],
            "objects": [item.to_dict() for item in self.objects],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class AgentBodyState:
    agent_id: str
    place_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="body agent id"))
        object.__setattr__(self, "place_id", _text(self.place_id, label="body place id"))

    def to_dict(self) -> dict[str, object]:
        return {"agent_id": self.agent_id, "place_id": self.place_id}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class WorldObjectState:
    object_id: str
    place_id: str | None
    holder_agent_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _text(self.object_id, label="object state id"))
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="object state place id"))
        object.__setattr__(self, "holder_agent_id", _optional_text(self.holder_agent_id, label="object state holder agent id"))
        if (self.place_id is None) == (self.holder_agent_id is None):
            raise ValueError("object state requires exactly one place or holder")

    def to_dict(self) -> dict[str, object]:
        return {"object_id": self.object_id, "place_id": self.place_id, "holder_agent_id": self.holder_agent_id}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PassageState:
    passage_id: str
    open: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "passage_id", _text(self.passage_id, label="passage state id"))
        if not isinstance(self.open, bool):
            raise TypeError("passage state open must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {"passage_id": self.passage_id, "open": self.open}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedWorldState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    agents: tuple[AgentBodyState, ...]
    objects: tuple[WorldObjectState, ...]
    passages: tuple[PassageState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated state model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="situated state model hash"))
        object.__setattr__(self, "round_index", _nonnegative_integer(self.round_index, label="situated state round index"))
        if self.round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("situated state round zero cannot have a parent state hash")
        else:
            object.__setattr__(self, "parent_state_hash", _hash(self.parent_state_hash, label="situated state parent state hash"))
        for name, values, expected, key in (
            ("agents", self.agents, AgentBodyState, lambda item: item.agent_id),
            ("objects", self.objects, WorldObjectState, lambda item: item.object_id),
            ("passages", self.passages, PassageState, lambda item: item.passage_id),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"situated state {name} must be a tuple of {expected.__name__} values")
            ids = tuple(key(item) for item in values)
            if len(set(ids)) != len(ids):
                raise ValueError(f"situated state {name} identities must be unique")
            object.__setattr__(self, name, tuple(sorted(values, key=key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "agents": [item.to_dict() for item in self.agents],
            "objects": [item.to_dict() for item in self.objects],
            "passages": [item.to_dict() for item in self.passages],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_situated_world(model: SituatedWorldModel) -> SituatedWorldState:
    if not isinstance(model, SituatedWorldModel):
        raise TypeError("situated model must be SituatedWorldModel")
    state = SituatedWorldState(
        model_id=model.model_id,
        model_hash=model.content_hash,
        round_index=0,
        parent_state_hash=None,
        agents=tuple(AgentBodyState(item.agent_id, item.initial_place_id) for item in model.agents),
        objects=tuple(WorldObjectState(item.object_id, item.initial_place_id) for item in model.objects),
        passages=tuple(PassageState(item.passage_id, item.initially_open) for item in model.passages),
    )
    validate_situated_state(model, state)
    return state


def validate_situated_state(model: SituatedWorldModel, state: SituatedWorldState) -> None:
    if not isinstance(model, SituatedWorldModel) or not isinstance(state, SituatedWorldState):
        raise TypeError("situated state validation requires model and state contracts")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("situated state must reference the exact model")
    agent_ids = {item.agent_id for item in model.agents}
    object_ids = {item.object_id for item in model.objects}
    passage_ids = {item.passage_id for item in model.passages}
    place_ids = {item.place_id for item in model.places}
    if {item.agent_id for item in state.agents} != agent_ids:
        raise ValueError("situated state agent roster must exactly match the model")
    if {item.object_id for item in state.objects} != object_ids:
        raise ValueError("situated state object roster must exactly match the model")
    if {item.passage_id for item in state.passages} != passage_ids:
        raise ValueError("situated state passage roster must exactly match the model")
    if any(item.place_id not in place_ids for item in state.agents):
        raise ValueError("situated agent place must reference a model place")
    for item in state.objects:
        if item.place_id is not None and item.place_id not in place_ids:
            raise ValueError("situated object place must reference a model place")
        if item.holder_agent_id is not None and item.holder_agent_id not in agent_ids:
            raise ValueError("situated object holder must reference a model agent")
    capacities = {item.agent_id: item.inventory_capacity for item in model.agents}
    held = {agent_id: 0 for agent_id in agent_ids}
    for item in state.objects:
        if item.holder_agent_id is not None:
            held[item.holder_agent_id] += 1
    if any(held[agent_id] > capacities[agent_id] for agent_id in agent_ids):
        raise ValueError("situated agent inventory capacity exceeded")


__all__ = (
    "EvidenceFact", "PlaceSpec", "PassageSpec", "EmbodiedAgentSpec", "WorldObjectSpec",
    "SituatedWorldModel", "AgentBodyState", "WorldObjectState", "PassageState",
    "SituatedWorldState", "initialize_situated_world", "validate_situated_state",
)
