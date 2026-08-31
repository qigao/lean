"""Immutable contracts for V12 agent-private situated long-term memory."""

from __future__ import annotations

from dataclasses import dataclass
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact


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


def _unit_interval(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return number


def _positive_integer(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


@dataclass(frozen=True)
class MemoryChannelPolicy:
    channel: ObservationChannel
    confidence: float
    salience: float

    def __post_init__(self) -> None:
        if not isinstance(self.channel, ObservationChannel):
            raise TypeError("memory channel policy channel must be ObservationChannel")
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, label="memory channel confidence"))
        object.__setattr__(self, "salience", _unit_interval(self.salience, label="memory channel salience"))

    def to_dict(self) -> dict[str, object]:
        return {
            "channel": self.channel.value,
            "confidence": self.confidence,
            "salience": self.salience,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedMemoryPolicy:
    policy_id: str
    version: str
    channels: tuple[MemoryChannelPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, label="situated memory policy id"))
        object.__setattr__(self, "version", _text(self.version, label="situated memory policy version"))
        if not isinstance(self.channels, tuple) or any(not isinstance(item, MemoryChannelPolicy) for item in self.channels):
            raise TypeError("situated memory channels must be a tuple of MemoryChannelPolicy values")
        actual = tuple(item.channel for item in self.channels)
        if len(actual) != len(ObservationChannel) or set(actual) != set(ObservationChannel):
            raise ValueError("situated memory policy requires exactly one policy for every observation channel")
        object.__setattr__(self, "channels", tuple(sorted(self.channels, key=lambda item: item.channel.value)))

    def for_channel(self, channel: ObservationChannel) -> MemoryChannelPolicy:
        if not isinstance(channel, ObservationChannel):
            raise TypeError("memory policy lookup channel must be ObservationChannel")
        return next(item for item in self.channels if item.channel is channel)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "channels": [item.to_dict() for item in self.channels],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def standard_situated_memory_policy() -> SituatedMemoryPolicy:
    """Return the explicit, deterministic V12 channel attribution policy."""

    return SituatedMemoryPolicy(
        "situated-memory-standard",
        "1",
        (
            MemoryChannelPolicy(ObservationChannel.SELF, 1.0, 0.5),
            MemoryChannelPolicy(ObservationChannel.VISUAL, 0.9, 0.65),
            MemoryChannelPolicy(ObservationChannel.AUDITORY, 0.7, 0.75),
            MemoryChannelPolicy(ObservationChannel.INSPECTION, 1.0, 1.0),
        ),
    )


@dataclass(frozen=True)
class SituatedMemoryRecord:
    memory_id: str
    agent_id: str
    observation_id: str
    story_model_id: str
    story_model_hash: str
    event_id: str
    event_hash: str
    round_index: int
    sequence: int
    action_id: str
    kind: SituatedActionKind
    actor_agent_id: str
    place_id: str
    target_id: str | None
    success: bool
    outcome: str
    details: tuple[EvidenceFact, ...]
    cause_event_ids: tuple[str, ...]
    channel: ObservationChannel
    confidence: float
    salience: float
    policy_hash: str
    summary: str
    active: bool = True

    def __post_init__(self) -> None:
        for name, label in (
            ("memory_id", "situated memory id"),
            ("agent_id", "situated memory agent id"),
            ("observation_id", "situated memory observation id"),
            ("story_model_id", "situated memory story model id"),
            ("event_id", "situated memory event id"),
            ("action_id", "situated memory action id"),
            ("actor_agent_id", "situated memory actor id"),
            ("place_id", "situated memory place id"),
            ("outcome", "situated memory outcome"),
            ("summary", "situated memory summary"),
        ):
            object.__setattr__(self, name, _text(getattr(self, name), label=label))
        if self.memory_id != self.observation_id:
            raise ValueError("situated memory id must equal observation id")
        object.__setattr__(self, "story_model_hash", _hash(self.story_model_hash, label="situated memory story model hash"))
        object.__setattr__(self, "event_hash", _hash(self.event_hash, label="situated memory event hash"))
        object.__setattr__(self, "policy_hash", _hash(self.policy_hash, label="situated memory policy hash"))
        object.__setattr__(self, "round_index", _positive_integer(self.round_index, label="situated memory round index"))
        object.__setattr__(self, "sequence", _positive_integer(self.sequence, label="situated memory sequence"))
        if not isinstance(self.kind, SituatedActionKind):
            raise TypeError("situated memory kind must be SituatedActionKind")
        if not isinstance(self.channel, ObservationChannel):
            raise TypeError("situated memory channel must be ObservationChannel")
        object.__setattr__(self, "target_id", _optional_text(self.target_id, label="situated memory target id"))
        if not isinstance(self.success, bool):
            raise TypeError("situated memory success must be boolean")
        if not isinstance(self.active, bool):
            raise TypeError("situated memory active must be boolean")
        if not isinstance(self.details, tuple) or any(not isinstance(item, EvidenceFact) for item in self.details):
            raise TypeError("situated memory details must be a tuple of EvidenceFact values")
        if not isinstance(self.cause_event_ids, tuple):
            raise TypeError("situated memory cause event ids must be a tuple")
        causes = tuple(_text(item, label="situated memory cause event id") for item in self.cause_event_ids)
        if len(set(causes)) != len(causes):
            raise ValueError("situated memory cause event ids must be unique")
        object.__setattr__(self, "details", tuple(sorted(self.details, key=lambda item: (item.name, item.value))))
        object.__setattr__(self, "cause_event_ids", tuple(sorted(causes)))
        object.__setattr__(self, "confidence", _unit_interval(self.confidence, label="situated memory confidence"))
        object.__setattr__(self, "salience", _unit_interval(self.salience, label="situated memory salience"))

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "observation_id": self.observation_id,
            "story_model_id": self.story_model_id,
            "story_model_hash": self.story_model_hash,
            "event_id": self.event_id,
            "event_hash": self.event_hash,
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
            "channel": self.channel.value,
            "confidence": self.confidence,
            "salience": self.salience,
            "policy_hash": self.policy_hash,
            "summary": self.summary,
            "active": self.active,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedMemoryQuery:
    agent_id: str
    text: str | None = None
    actor_agent_id: str | None = None
    place_id: str | None = None
    event_kinds: tuple[SituatedActionKind, ...] = ()
    channels: tuple[ObservationChannel, ...] = ()
    min_round: int | None = None
    max_round: int | None = None
    min_confidence: float = 0.0
    include_inactive: bool = False
    limit: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="situated memory query agent id"))
        if self.text is not None:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("situated memory query text must be non-empty")
            if "\x00" in self.text:
                raise ValueError("situated memory query text cannot contain NUL")
        object.__setattr__(self, "actor_agent_id", _optional_text(self.actor_agent_id, label="situated memory query actor id"))
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="situated memory query place id"))
        if not isinstance(self.event_kinds, tuple) or any(not isinstance(item, SituatedActionKind) for item in self.event_kinds):
            raise TypeError("situated memory query event kinds must be a tuple of SituatedActionKind values")
        if not isinstance(self.channels, tuple) or any(not isinstance(item, ObservationChannel) for item in self.channels):
            raise TypeError("situated memory query channels must be a tuple of ObservationChannel values")
        object.__setattr__(self, "event_kinds", tuple(sorted(set(self.event_kinds), key=lambda item: item.value)))
        object.__setattr__(self, "channels", tuple(sorted(set(self.channels), key=lambda item: item.value)))
        for name in ("min_round", "max_round"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive_integer(value, label=f"situated memory query {name.replace('_', ' ')}"))
        if self.min_round is not None and self.max_round is not None and self.min_round > self.max_round:
            raise ValueError("situated memory query minimum round cannot exceed maximum round")
        object.__setattr__(self, "min_confidence", _unit_interval(self.min_confidence, label="situated memory query minimum confidence"))
        if not isinstance(self.include_inactive, bool):
            raise TypeError("situated memory query include inactive must be boolean")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or not 1 <= self.limit <= 1000:
            raise ValueError("situated memory query limit must be between 1 and 1000")


@dataclass(frozen=True)
class SituatedMemorySearchHit:
    memory: SituatedMemoryRecord
    lexical_rank: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory, SituatedMemoryRecord):
            raise TypeError("situated memory search hit requires a SituatedMemoryRecord")
        if self.lexical_rank is not None and (not isinstance(self.lexical_rank, (int, float)) or isinstance(self.lexical_rank, bool)):
            raise TypeError("situated memory lexical rank must be numeric")


@dataclass(frozen=True)
class SituatedMemoryWriteReport:
    agent_id: str
    inserted_count: int
    existing_count: int
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="situated memory write agent id"))
        for name in ("inserted_count", "existing_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"situated memory write {name.replace('_', ' ')} must be non-negative")
        if not isinstance(self.memory_ids, tuple):
            raise TypeError("situated memory write ids must be a tuple")
        ids = tuple(_text(item, label="situated memory write id") for item in self.memory_ids)
        if len(set(ids)) != len(ids):
            raise ValueError("situated memory write ids must be unique")
        object.__setattr__(self, "memory_ids", tuple(sorted(ids)))
        if self.inserted_count + self.existing_count != len(self.memory_ids):
            raise ValueError("situated memory write counts must equal the number of memory ids")


@dataclass(frozen=True)
class SituatedMemoryIndexReport:
    indexed_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.indexed_count, int) or isinstance(self.indexed_count, bool) or self.indexed_count < 0:
            raise ValueError("situated memory indexed count must be non-negative")


__all__ = (
    "MemoryChannelPolicy",
    "SituatedMemoryPolicy",
    "SituatedMemoryRecord",
    "SituatedMemoryQuery",
    "SituatedMemorySearchHit",
    "SituatedMemoryWriteReport",
    "SituatedMemoryIndexReport",
    "standard_situated_memory_policy",
)
