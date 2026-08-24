from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash


GENERIC_NARRATIVE_SCHEMA_VERSION = 1
_ALLOWED_RELATIONS = frozenset({"equals", "not_equals"})
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _logical_time(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


@dataclass(frozen=True)
class EntityRef:
    entity_id: str
    entity_type: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "entity_id", _text(self.entity_id, label="entity ref id")
        )
        object.__setattr__(
            self, "entity_type", _text(self.entity_type, label="entity ref type")
        )

    def to_dict(self) -> dict[str, object]:
        return {"entity_id": self.entity_id, "entity_type": self.entity_type}


@dataclass(frozen=True)
class StateCellRef:
    subject: EntityRef
    state_variable: str

    def __post_init__(self) -> None:
        if not isinstance(self.subject, EntityRef):
            raise TypeError("state cell subject must be an EntityRef")
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="state variable"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject.to_dict(),
            "state_variable": self.state_variable,
        }


@dataclass(frozen=True)
class TypedValue:
    type_name: str
    value: bool | int | str | EntityRef

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "type_name", _text(self.type_name, label="typed value type")
        )
        if not isinstance(self.value, (bool, int, str, EntityRef)):
            raise TypeError(
                "typed value must contain a boolean, integer, string, or EntityRef"
            )

    def to_dict(self) -> dict[str, object]:
        value: object = (
            self.value.to_dict() if isinstance(self.value, EntityRef) else self.value
        )
        return {"type_name": self.type_name, "value": value}


def _typed_arguments(
    value: Mapping[str, TypedValue], *, label: str
) -> Mapping[str, TypedValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, TypedValue] = {}
    for key, item in value.items():
        name = _text(key, label=f"{label} key")
        if not isinstance(item, TypedValue):
            raise TypeError(f"{label} values must be TypedValue instances")
        frozen[name] = item
    return MappingProxyType(frozen)


def _arguments_dict(value: Mapping[str, TypedValue]) -> dict[str, object]:
    return {key: item.to_dict() for key, item in value.items()}


@dataclass(frozen=True)
class Entity:
    id: str
    type_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="entity id"))
        object.__setattr__(
            self, "type_name", _text(self.type_name, label="entity type")
        )

    def to_dict(self) -> dict[str, object]:
        return {"id": self.id, "type_name": self.type_name}


@dataclass(frozen=True)
class NarrativeEvent:
    id: str
    logical_time: int
    type_name: str
    actor_id: str | None
    arguments: Mapping[str, TypedValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="event id"))
        object.__setattr__(
            self,
            "logical_time",
            _logical_time(self.logical_time, label="event logical time"),
        )
        object.__setattr__(
            self, "type_name", _text(self.type_name, label="event type")
        )
        if self.actor_id is not None:
            object.__setattr__(
                self, "actor_id", _text(self.actor_id, label="event actor")
            )
        object.__setattr__(
            self,
            "arguments",
            _typed_arguments(self.arguments, label="event arguments"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_time": self.logical_time,
            "type_name": self.type_name,
            "actor_id": self.actor_id,
            "arguments": _arguments_dict(self.arguments),
        }


@dataclass(frozen=True)
class Observation:
    id: str
    agent_id: str
    event_id: str
    channel: str = "direct"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="observation id"))
        object.__setattr__(
            self, "agent_id", _text(self.agent_id, label="observation agent")
        )
        object.__setattr__(
            self, "event_id", _text(self.event_id, label="observation event")
        )
        object.__setattr__(
            self, "channel", _text(self.channel, label="observation channel")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "event_id": self.event_id,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class Proposition:
    subject: EntityRef
    state_variable: str
    relation: str
    value: TypedValue

    def __post_init__(self) -> None:
        if not isinstance(self.subject, EntityRef):
            raise TypeError("proposition subject must be an EntityRef")
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="proposition state variable"),
        )
        relation = _text(self.relation, label="proposition relation")
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("proposition relation must be equals or not_equals")
        object.__setattr__(self, "relation", relation)
        if not isinstance(self.value, TypedValue):
            raise TypeError("proposition value must be a TypedValue")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject.to_dict(),
            "state_variable": self.state_variable,
            "relation": self.relation,
            "value": self.value.to_dict(),
        }


@dataclass(frozen=True)
class Claim:
    id: str
    logical_time: int
    speaker_id: str
    proposition: Proposition
    support_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="claim id"))
        object.__setattr__(
            self,
            "logical_time",
            _logical_time(self.logical_time, label="claim logical time"),
        )
        object.__setattr__(
            self, "speaker_id", _text(self.speaker_id, label="claim speaker")
        )
        if not isinstance(self.proposition, Proposition):
            raise TypeError("claim proposition must be a Proposition")
        refs = tuple(_text(item, label="claim support ref") for item in self.support_refs)
        object.__setattr__(self, "support_refs", refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_time": self.logical_time,
            "speaker_id": self.speaker_id,
            "proposition": self.proposition.to_dict(),
            "support_refs": list(self.support_refs),
        }


@dataclass(frozen=True)
class Reception:
    id: str
    claim_id: str
    recipient_id: str
    channel: str = "direct_testimony"

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="reception id"))
        object.__setattr__(
            self, "claim_id", _text(self.claim_id, label="reception claim")
        )
        object.__setattr__(
            self,
            "recipient_id",
            _text(self.recipient_id, label="reception recipient"),
        )
        object.__setattr__(
            self, "channel", _text(self.channel, label="reception channel")
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "recipient_id": self.recipient_id,
            "channel": self.channel,
        }


@dataclass(frozen=True)
class ActionOption:
    id: str
    type_name: str
    arguments: Mapping[str, TypedValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="action id"))
        object.__setattr__(
            self, "type_name", _text(self.type_name, label="action type")
        )
        object.__setattr__(
            self,
            "arguments",
            _typed_arguments(self.arguments, label="action arguments"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type_name": self.type_name,
            "arguments": _arguments_dict(self.arguments),
        }


@dataclass(frozen=True)
class Decision:
    id: str
    logical_time: int
    actor_id: str
    type_name: str
    context_cells: tuple[StateCellRef, ...]
    actions: tuple[ActionOption, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="decision id"))
        object.__setattr__(
            self,
            "logical_time",
            _logical_time(self.logical_time, label="decision logical time"),
        )
        object.__setattr__(
            self, "actor_id", _text(self.actor_id, label="decision actor")
        )
        object.__setattr__(
            self, "type_name", _text(self.type_name, label="decision type")
        )
        context_cells = tuple(self.context_cells)
        if any(not isinstance(item, StateCellRef) for item in context_cells):
            raise TypeError("decision context cells must be StateCellRef values")
        actions = tuple(self.actions)
        if any(not isinstance(item, ActionOption) for item in actions):
            raise TypeError("decision actions must be ActionOption values")
        object.__setattr__(self, "context_cells", context_cells)
        object.__setattr__(self, "actions", actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "logical_time": self.logical_time,
            "actor_id": self.actor_id,
            "type_name": self.type_name,
            "context_cells": [item.to_dict() for item in self.context_cells],
            "actions": [item.to_dict() for item in self.actions],
        }


@dataclass(frozen=True)
class GenericNarrative:
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    entities: tuple[Entity, ...]
    events: tuple[NarrativeEvent, ...]
    observations: tuple[Observation, ...]
    claims: tuple[Claim, ...]
    receptions: tuple[Reception, ...]
    decisions: tuple[Decision, ...]
    schema_version: int = GENERIC_NARRATIVE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "domain_id", _text(self.domain_id, label="narrative domain id")
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="narrative domain version"),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _content_hash(self.domain_spec_hash, label="narrative domain spec hash"),
        )
        if (
            not isinstance(self.schema_version, int)
            or isinstance(self.schema_version, bool)
            or self.schema_version != GENERIC_NARRATIVE_SCHEMA_VERSION
        ):
            raise ValueError("generic narrative schema version must be 1")

        fields = (
            ("entities", Entity),
            ("events", NarrativeEvent),
            ("observations", Observation),
            ("claims", Claim),
            ("receptions", Reception),
            ("decisions", Decision),
        )
        for name, expected_type in fields:
            values = tuple(getattr(self, name))
            if any(not isinstance(item, expected_type) for item in values):
                raise TypeError(
                    f"generic narrative {name} must contain {expected_type.__name__} values"
                )
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "entities": [item.to_dict() for item in self.entities],
            "events": [item.to_dict() for item in self.events],
            "observations": [item.to_dict() for item in self.observations],
            "claims": [item.to_dict() for item in self.claims],
            "receptions": [item.to_dict() for item in self.receptions],
            "decisions": [item.to_dict() for item in self.decisions],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
