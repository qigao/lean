"""Immutable V15 contracts for situated perception reach and sanitized percepts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact, SituatedWorldModel


_CONTENT_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _finite_number(value: object, *, label: str, nonnegative: bool = True) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if nonnegative and number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


class SituatedPerceptionLayer(str, Enum):
    VISIBILITY = "visibility"
    AUDITORY = "auditory"
    INTERACTION = "interaction"


class SituatedEdgeActivation(str, Enum):
    ALWAYS = "always"
    PASSAGE_OPEN = "passage_open"
    PASSAGE_CLOSED = "passage_closed"


class SituatedPerceptFidelity(str, Enum):
    DETECTED = "detected"
    IDENTIFIED = "identified"
    EXACT = "exact"


@dataclass(frozen=True)
class SituatedPerceptionEdge:
    edge_id: str
    layer: SituatedPerceptionLayer
    source_place_id: str
    target_place_id: str
    cost: float
    activation: SituatedEdgeActivation = SituatedEdgeActivation.ALWAYS
    passage_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("edge_id", "source_place_id", "target_place_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"perception edge {name}"))
        if not isinstance(self.layer, SituatedPerceptionLayer):
            raise TypeError("perception edge layer must be SituatedPerceptionLayer")
        if not isinstance(self.activation, SituatedEdgeActivation):
            raise TypeError("perception edge activation must be SituatedEdgeActivation")
        object.__setattr__(self, "cost", _finite_number(self.cost, label="perception edge cost"))
        object.__setattr__(self, "passage_id", _optional_text(self.passage_id, label="perception edge passage id"))
        if self.activation is SituatedEdgeActivation.ALWAYS and self.passage_id is not None:
            raise ValueError("always perception edge cannot reference a passage")
        if self.activation is not SituatedEdgeActivation.ALWAYS and self.passage_id is None:
            raise ValueError("passage-activated perception edge requires a passage")
        if self.layer is SituatedPerceptionLayer.INTERACTION and self.cost != 0.0:
            raise ValueError("interaction perception edge cost must be exactly zero")

    def to_dict(self) -> dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "layer": self.layer.value,
            "source_place_id": self.source_place_id,
            "target_place_id": self.target_place_id,
            "cost": self.cost,
            "activation": self.activation.value,
            "passage_id": self.passage_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedAgentPerceptionProfile:
    agent_id: str
    max_visual_cost: float
    minimum_detectable_sound: float
    minimum_clear_sound: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="perception profile agent id"))
        for name in ("max_visual_cost", "minimum_detectable_sound", "minimum_clear_sound"):
            object.__setattr__(self, name, _finite_number(getattr(self, name), label=f"perception profile {name}"))
        if self.minimum_clear_sound < self.minimum_detectable_sound:
            raise ValueError("minimum clear sound must be at least minimum detectable sound")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "max_visual_cost": self.max_visual_cost,
            "minimum_detectable_sound": self.minimum_detectable_sound,
            "minimum_clear_sound": self.minimum_clear_sound,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedEventSignalProfile:
    kind: SituatedActionKind
    visually_observable: bool
    auditory_intensity: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SituatedActionKind):
            raise TypeError("event signal kind must be SituatedActionKind")
        if not isinstance(self.visually_observable, bool):
            raise TypeError("event signal visual observability must be boolean")
        if self.auditory_intensity is not None:
            object.__setattr__(
                self,
                "auditory_intensity",
                _finite_number(self.auditory_intensity, label="event signal auditory intensity"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "visually_observable": self.visually_observable,
            "auditory_intensity": self.auditory_intensity,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptionModel:
    model_id: str
    version: str
    world_model: SituatedWorldModel
    edges: tuple[SituatedPerceptionEdge, ...]
    agent_profiles: tuple[SituatedAgentPerceptionProfile, ...]
    signal_profiles: tuple[SituatedEventSignalProfile, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="perception model id"))
        object.__setattr__(self, "version", _text(self.version, label="perception model version"))
        if not isinstance(self.world_model, SituatedWorldModel):
            raise TypeError("perception model world_model must be SituatedWorldModel")
        for name, values, expected in (
            ("edges", self.edges, SituatedPerceptionEdge),
            ("agent_profiles", self.agent_profiles, SituatedAgentPerceptionProfile),
            ("signal_profiles", self.signal_profiles, SituatedEventSignalProfile),
        ):
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"perception model {name} must be a tuple of {expected.__name__} values")

        place_ids = {item.place_id for item in self.world_model.places}
        passage_ids = {item.passage_id for item in self.world_model.passages}
        edge_ids = tuple(item.edge_id for item in self.edges)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("perception model edge ids must be unique")
        for edge in self.edges:
            if edge.source_place_id not in place_ids or edge.target_place_id not in place_ids:
                raise ValueError("perception edge endpoint must reference a world place")
            if edge.passage_id is not None and edge.passage_id not in passage_ids:
                raise ValueError("perception edge passage must reference a world passage")

        world_agent_ids = {item.agent_id for item in self.world_model.agents}
        profile_ids = tuple(item.agent_id for item in self.agent_profiles)
        if len(set(profile_ids)) != len(profile_ids):
            raise ValueError("perception agent profile ids must be unique")
        if set(profile_ids) != world_agent_ids:
            raise ValueError("perception model requires exactly one profile for every world agent")
        signal_kinds = tuple(item.kind for item in self.signal_profiles)
        if len(set(signal_kinds)) != len(signal_kinds):
            raise ValueError("perception signal kinds must be unique")

        object.__setattr__(self, "edges", tuple(sorted(self.edges, key=lambda item: item.edge_id)))
        object.__setattr__(self, "agent_profiles", tuple(sorted(self.agent_profiles, key=lambda item: item.agent_id)))
        object.__setattr__(self, "signal_profiles", tuple(sorted(self.signal_profiles, key=lambda item: item.kind.value)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "world_model_hash": self.world_model.content_hash,
            "edges": [item.to_dict() for item in self.edges],
            "agent_profiles": [item.to_dict() for item in self.agent_profiles],
            "signal_profiles": [item.to_dict() for item in self.signal_profiles],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _finite_mapping(value: object, *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    values: dict[str, float] = {}
    for key, item in value.items():
        key_text = _text(key, label=f"{label} key")
        values[key_text] = _finite_number(item, label=f"{label}[{key_text}]")
    return MappingProxyType(dict(sorted(values.items())))


@dataclass(frozen=True)
class SituatedPerceptionReach:
    model_hash: str
    state_hash: str
    source_place_id: str
    visual_costs: Mapping[str, float]
    auditory_losses: Mapping[str, float]
    interaction_place_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="perception reach model hash"))
        object.__setattr__(self, "state_hash", _hash(self.state_hash, label="perception reach state hash"))
        object.__setattr__(self, "source_place_id", _text(self.source_place_id, label="perception reach source place id"))
        object.__setattr__(self, "visual_costs", _finite_mapping(self.visual_costs, label="perception reach visual costs"))
        object.__setattr__(self, "auditory_losses", _finite_mapping(self.auditory_losses, label="perception reach auditory losses"))
        if not isinstance(self.interaction_place_ids, tuple):
            raise TypeError("perception reach interaction place ids must be a tuple")
        interactions = tuple(_text(item, label="perception reach interaction place id") for item in self.interaction_place_ids)
        if len(set(interactions)) != len(interactions):
            raise ValueError("perception reach interaction place ids must be unique")
        object.__setattr__(self, "interaction_place_ids", tuple(sorted(interactions)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_hash": self.model_hash,
            "state_hash": self.state_hash,
            "source_place_id": self.source_place_id,
            "visual_costs": dict(sorted(self.visual_costs.items())),
            "auditory_losses": dict(sorted(self.auditory_losses.items())),
            "interaction_place_ids": list(self.interaction_place_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPercept:
    percept_id: str
    round_index: int
    agent_id: str
    source_event_id: str
    source_event_hash: str
    channels: tuple[ObservationChannel, ...]
    fidelity: SituatedPerceptFidelity
    actor_agent_id: str | None = None
    kind: SituatedActionKind | None = None
    place_id: str | None = None
    outcome: str | None = None
    details: tuple[EvidenceFact, ...] = ()

    def __post_init__(self) -> None:
        for name in ("percept_id", "agent_id", "source_event_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"situated percept {name}"))
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index < 0:
            raise ValueError("situated percept round index must be a non-negative integer")
        object.__setattr__(self, "source_event_hash", _hash(self.source_event_hash, label="situated percept source event hash"))
        if not isinstance(self.channels, tuple) or any(not isinstance(item, ObservationChannel) for item in self.channels):
            raise TypeError("situated percept channels must be a tuple of ObservationChannel values")
        if not self.channels:
            raise ValueError("situated percept requires at least one channel")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("situated percept channels must be unique")
        object.__setattr__(self, "channels", tuple(sorted(self.channels, key=lambda item: item.value)))
        if not isinstance(self.fidelity, SituatedPerceptFidelity):
            raise TypeError("situated percept fidelity must be SituatedPerceptFidelity")
        object.__setattr__(self, "actor_agent_id", _optional_text(self.actor_agent_id, label="situated percept actor agent id"))
        if self.kind is not None and not isinstance(self.kind, SituatedActionKind):
            raise TypeError("situated percept kind must be SituatedActionKind")
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="situated percept place id"))
        object.__setattr__(self, "outcome", _optional_text(self.outcome, label="situated percept outcome"))
        if not isinstance(self.details, tuple) or any(not isinstance(item, EvidenceFact) for item in self.details):
            raise TypeError("situated percept details must be a tuple of EvidenceFact values")
        object.__setattr__(self, "details", tuple(sorted(self.details, key=lambda item: (item.name, item.value))))

        identified = (self.actor_agent_id, self.kind, self.place_id)
        if self.fidelity is SituatedPerceptFidelity.DETECTED:
            if any(item is not None for item in identified) or self.outcome is not None or self.details:
                raise ValueError("detected percept cannot expose event identity or details")
        elif self.fidelity is SituatedPerceptFidelity.IDENTIFIED:
            if any(item is None for item in identified):
                raise ValueError("identified percept requires actor, kind, and place")
            if self.outcome is not None or self.details:
                raise ValueError("identified percept cannot expose outcome or details")
        else:
            if any(item is None for item in identified) or self.outcome is None:
                raise ValueError("exact percept requires actor, kind, place, and outcome")

        for private_channel in (ObservationChannel.SELF, ObservationChannel.INSPECTION):
            if private_channel in self.channels:
                if self.fidelity is not SituatedPerceptFidelity.EXACT:
                    raise ValueError(f"{private_channel.value} percepts require exact fidelity")
                if self.agent_id != self.actor_agent_id:
                    raise ValueError(f"{private_channel.value} percepts must be actor-private")

    def to_dict(self) -> dict[str, object]:
        return {
            "percept_id": self.percept_id,
            "round_index": self.round_index,
            "agent_id": self.agent_id,
            "source_event_id": self.source_event_id,
            "source_event_hash": self.source_event_hash,
            "channels": [item.value for item in self.channels],
            "fidelity": self.fidelity.value,
            "actor_agent_id": self.actor_agent_id,
            "kind": None if self.kind is None else self.kind.value,
            "place_id": self.place_id,
            "outcome": self.outcome,
            "details": [item.to_dict() for item in self.details],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptualProjection:
    model_id: str
    model_hash: str
    prior_state_hash: str
    round_result_hash: str
    percepts: tuple[SituatedPercept, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="perceptual projection model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="perceptual projection model hash"))
        object.__setattr__(self, "prior_state_hash", _hash(self.prior_state_hash, label="perceptual projection prior state hash"))
        object.__setattr__(self, "round_result_hash", _hash(self.round_result_hash, label="perceptual projection round result hash"))
        if not isinstance(self.percepts, tuple) or any(not isinstance(item, SituatedPercept) for item in self.percepts):
            raise TypeError("perceptual projection percepts must be a tuple of SituatedPercept values")
        percept_ids = tuple(item.percept_id for item in self.percepts)
        if len(set(percept_ids)) != len(percept_ids):
            raise ValueError("perceptual projection percept ids must be unique")
        pairs = tuple((item.agent_id, item.source_event_id) for item in self.percepts)
        if len(set(pairs)) != len(pairs):
            raise ValueError("perceptual projection agent/event pairs must be unique")
        object.__setattr__(self, "percepts", tuple(sorted(self.percepts, key=lambda item: (item.source_event_id, item.agent_id))))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "prior_state_hash": self.prior_state_hash,
            "round_result_hash": self.round_result_hash,
            "percepts": [item.to_dict() for item in self.percepts],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "SituatedPerceptionLayer",
    "SituatedEdgeActivation",
    "SituatedPerceptFidelity",
    "SituatedPerceptionEdge",
    "SituatedAgentPerceptionProfile",
    "SituatedEventSignalProfile",
    "SituatedPerceptionModel",
    "SituatedPerceptionReach",
    "SituatedPercept",
    "SituatedPerceptualProjection",
)
