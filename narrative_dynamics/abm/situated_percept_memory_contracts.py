"""Immutable contracts for agent-private sanitized percept memory."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPercept,
    SituatedPerceptFidelity,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _unit(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class SituatedPerceptMemoryFidelityPolicy:
    fidelity: SituatedPerceptFidelity
    confidence: float
    salience: float

    def __post_init__(self) -> None:
        if not isinstance(self.fidelity, SituatedPerceptFidelity):
            raise TypeError("percept memory policy fidelity must be SituatedPerceptFidelity")
        object.__setattr__(self, "confidence", _unit(self.confidence, label="percept memory confidence"))
        object.__setattr__(self, "salience", _unit(self.salience, label="percept memory salience"))

    def to_dict(self) -> dict[str, object]:
        return {
            "fidelity": self.fidelity.value,
            "confidence": self.confidence,
            "salience": self.salience,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptMemoryPolicy:
    policy_id: str
    version: str
    fidelities: tuple[SituatedPerceptMemoryFidelityPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "policy_id", _text(self.policy_id, label="percept memory policy id"))
        object.__setattr__(self, "version", _text(self.version, label="percept memory policy version"))
        if not isinstance(self.fidelities, tuple) or any(
            not isinstance(item, SituatedPerceptMemoryFidelityPolicy)
            for item in self.fidelities
        ):
            raise TypeError("percept memory fidelities must be a tuple of fidelity policies")
        actual = tuple(item.fidelity for item in self.fidelities)
        if len(actual) != len(SituatedPerceptFidelity) or set(actual) != set(SituatedPerceptFidelity):
            raise ValueError("percept memory policy requires every percept fidelity exactly once")
        object.__setattr__(
            self,
            "fidelities",
            tuple(sorted(self.fidelities, key=lambda item: item.fidelity.value)),
        )

    def for_fidelity(
        self,
        fidelity: SituatedPerceptFidelity,
    ) -> SituatedPerceptMemoryFidelityPolicy:
        if not isinstance(fidelity, SituatedPerceptFidelity):
            raise TypeError("percept memory policy lookup requires SituatedPerceptFidelity")
        return next(item for item in self.fidelities if item.fidelity is fidelity)

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_id": self.policy_id,
            "version": self.version,
            "fidelities": [item.to_dict() for item in self.fidelities],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def standard_situated_percept_memory_policy() -> SituatedPerceptMemoryPolicy:
    return SituatedPerceptMemoryPolicy(
        "situated-percept-memory-standard",
        "1",
        (
            SituatedPerceptMemoryFidelityPolicy(
                SituatedPerceptFidelity.DETECTED,
                0.25,
                0.35,
            ),
            SituatedPerceptMemoryFidelityPolicy(
                SituatedPerceptFidelity.IDENTIFIED,
                0.6,
                0.55,
            ),
            SituatedPerceptMemoryFidelityPolicy(
                SituatedPerceptFidelity.EXACT,
                1.0,
                0.8,
            ),
        ),
    )


@dataclass(frozen=True)
class SituatedPerceptMemoryRecord:
    memory_id: str
    agent_id: str
    percept_id: str
    perception_model_id: str
    perception_model_hash: str
    story_model_id: str
    story_model_hash: str
    projection_hash: str
    source_event_id: str
    source_event_hash: str
    round_index: int
    channels: tuple[ObservationChannel, ...]
    fidelity: SituatedPerceptFidelity
    actor_agent_id: str | None
    kind: SituatedActionKind | None
    place_id: str | None
    outcome: str | None
    details: tuple[EvidenceFact, ...]
    confidence: float
    salience: float
    policy_hash: str
    summary: str
    active: bool = True

    def __post_init__(self) -> None:
        for name, label in (
            ("memory_id", "percept memory id"),
            ("agent_id", "percept memory agent id"),
            ("percept_id", "percept memory percept id"),
            ("perception_model_id", "percept memory perception model id"),
            ("story_model_id", "percept memory story model id"),
            ("source_event_id", "percept memory source event id"),
            ("summary", "percept memory summary"),
        ):
            object.__setattr__(self, name, _text(getattr(self, name), label=label))
        if self.memory_id != self.percept_id:
            raise ValueError("percept memory id must equal percept id")
        for name, label in (
            ("perception_model_hash", "percept memory perception model hash"),
            ("story_model_hash", "percept memory story model hash"),
            ("projection_hash", "percept memory projection hash"),
            ("source_event_hash", "percept memory source event hash"),
            ("policy_hash", "percept memory policy hash"),
        ):
            object.__setattr__(self, name, _hash(getattr(self, name), label=label))
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index <= 0:
            raise ValueError("percept memory round index must be positive")
        if not isinstance(self.active, bool):
            raise TypeError("percept memory active flag must be boolean")
        object.__setattr__(self, "confidence", _unit(self.confidence, label="percept memory confidence"))
        object.__setattr__(self, "salience", _unit(self.salience, label="percept memory salience"))
        object.__setattr__(self, "actor_agent_id", _optional_text(self.actor_agent_id, label="percept memory actor id"))
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="percept memory place id"))
        object.__setattr__(self, "outcome", _optional_text(self.outcome, label="percept memory outcome"))

        percept = SituatedPercept(
            self.percept_id,
            self.round_index,
            self.agent_id,
            self.source_event_id,
            self.source_event_hash,
            self.channels,
            self.fidelity,
            self.actor_agent_id,
            self.kind,
            self.place_id,
            self.outcome,
            self.details,
        )
        object.__setattr__(self, "channels", percept.channels)
        object.__setattr__(self, "details", percept.details)

    def to_dict(self) -> dict[str, object]:
        return {
            "memory_id": self.memory_id,
            "agent_id": self.agent_id,
            "percept_id": self.percept_id,
            "perception_model_id": self.perception_model_id,
            "perception_model_hash": self.perception_model_hash,
            "story_model_id": self.story_model_id,
            "story_model_hash": self.story_model_hash,
            "projection_hash": self.projection_hash,
            "source_event_id": self.source_event_id,
            "source_event_hash": self.source_event_hash,
            "round_index": self.round_index,
            "channels": [item.value for item in self.channels],
            "fidelity": self.fidelity.value,
            "actor_agent_id": self.actor_agent_id,
            "kind": None if self.kind is None else self.kind.value,
            "place_id": self.place_id,
            "outcome": self.outcome,
            "details": [item.to_dict() for item in self.details],
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
class SituatedPerceptMemoryQuery:
    agent_id: str
    text: str | None = None
    actor_agent_id: str | None = None
    place_id: str | None = None
    fidelities: tuple[SituatedPerceptFidelity, ...] = ()
    event_kinds: tuple[SituatedActionKind, ...] = ()
    channels: tuple[ObservationChannel, ...] = ()
    min_round: int | None = None
    max_round: int | None = None
    min_confidence: float = 0.0
    include_inactive: bool = False
    limit: int = 20
    story_model_hash: str | None = None
    excluded_memory_ids: tuple[str, ...] = ()
    included_memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="percept memory query agent id"))
        if self.text is not None:
            if not isinstance(self.text, str) or not self.text.strip():
                raise ValueError("percept memory query text must be non-empty")
            if "\x00" in self.text:
                raise ValueError("percept memory query text cannot contain NUL")
        object.__setattr__(self, "actor_agent_id", _optional_text(self.actor_agent_id, label="percept memory query actor id"))
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="percept memory query place id"))
        for name, item_type in (
            ("fidelities", SituatedPerceptFidelity),
            ("event_kinds", SituatedActionKind),
            ("channels", ObservationChannel),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, item_type) for item in values):
                raise TypeError(f"percept memory query {name.replace('_', ' ')} has invalid values")
            object.__setattr__(self, name, tuple(sorted(set(values), key=lambda item: item.value)))
        for name in ("min_round", "max_round"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, int) or isinstance(value, bool) or value < 1
            ):
                raise ValueError(f"percept memory query {name.replace('_', ' ')} must be positive")
        if self.min_round is not None and self.max_round is not None and self.min_round > self.max_round:
            raise ValueError("percept memory query minimum round cannot exceed maximum round")
        object.__setattr__(self, "min_confidence", _unit(self.min_confidence, label="percept memory query minimum confidence"))
        if not isinstance(self.include_inactive, bool):
            raise TypeError("percept memory query include inactive must be boolean")
        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or not 1 <= self.limit <= 1000:
            raise ValueError("percept memory query limit must be between 1 and 1000")
        if self.story_model_hash is not None:
            object.__setattr__(self, "story_model_hash", _hash(self.story_model_hash, label="percept memory query story model hash"))
        for name in ("excluded_memory_ids", "included_memory_ids"):
            values = getattr(self, name)
            if not isinstance(values, tuple):
                raise TypeError(f"percept memory query {name.replace('_', ' ')} must be a tuple")
            canonical = tuple(sorted(_text(item, label="percept memory query memory id") for item in values))
            if len(set(canonical)) != len(canonical):
                raise ValueError(f"percept memory query {name.replace('_', ' ')} must be unique")
            object.__setattr__(self, name, canonical)
        if set(self.excluded_memory_ids) & set(self.included_memory_ids):
            raise ValueError("percept memory query included and excluded ids must be disjoint")


@dataclass(frozen=True)
class SituatedPerceptMemoryHit:
    memory: SituatedPerceptMemoryRecord
    lexical_rank: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.memory, SituatedPerceptMemoryRecord):
            raise TypeError("percept memory hit requires a SituatedPerceptMemoryRecord")
        if self.lexical_rank is not None and (
            not isinstance(self.lexical_rank, (int, float)) or isinstance(self.lexical_rank, bool)
        ):
            raise TypeError("percept memory lexical rank must be numeric")


@dataclass(frozen=True)
class SituatedPerceptMemoryWriteReport:
    agent_id: str
    inserted_count: int
    existing_count: int
    memory_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="percept memory report agent id"))
        for name in ("inserted_count", "existing_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"percept memory report {name.replace('_', ' ')} must be non-negative")
        if not isinstance(self.memory_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in self.memory_ids
        ):
            raise TypeError("percept memory report ids must be a tuple of strings")
        if len(set(self.memory_ids)) != len(self.memory_ids):
            raise ValueError("percept memory report ids must be unique")
        if self.inserted_count + self.existing_count != len(self.memory_ids):
            raise ValueError("percept memory report counts must cover exact ids")
        object.__setattr__(self, "memory_ids", tuple(sorted(self.memory_ids)))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "inserted_count": self.inserted_count,
            "existing_count": self.existing_count,
            "memory_ids": list(self.memory_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptMemoryIndexReport:
    record_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.record_count, int) or isinstance(self.record_count, bool) or self.record_count < 0:
            raise ValueError("percept memory index count must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return {"record_count": self.record_count}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "SituatedPerceptMemoryFidelityPolicy",
    "SituatedPerceptMemoryPolicy",
    "SituatedPerceptMemoryRecord",
    "SituatedPerceptMemoryQuery",
    "SituatedPerceptMemoryHit",
    "SituatedPerceptMemoryWriteReport",
    "SituatedPerceptMemoryIndexReport",
    "standard_situated_percept_memory_policy",
)
