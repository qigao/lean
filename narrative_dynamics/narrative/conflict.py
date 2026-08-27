from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import re

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import StateDelta
from narrative_dynamics.narrative.ir import ActionOption, StateCellRef


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _canonical_delta(delta: StateDelta, *, label: str) -> StateDelta:
    if not isinstance(delta, StateDelta):
        raise TypeError(f"{label} must be StateDelta")
    operations = tuple(
        sorted(
            delta.operations,
            key=lambda item: (item.subject_id, item.state_variable),
        )
    )
    keys = tuple((item.subject_id, item.state_variable) for item in operations)
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} cannot write one cell twice")
    return StateDelta(operations)


def _delta_payload(delta: StateDelta) -> dict[str, object]:
    return {"operations": [item.to_dict() for item in delta.operations]}


@dataclass(frozen=True)
class ConflictParticipant:
    actor_id: str
    decision_id: str
    action: ActionOption
    transition_record_hash: str
    transition_spec_hash: str
    original_delta: StateDelta
    allowed_write_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "actor_id", _text(self.actor_id, label="conflict actor id"))
        object.__setattr__(self, "decision_id", _text(self.decision_id, label="conflict decision id"))
        if not isinstance(self.action, ActionOption):
            raise TypeError("conflict participant action must be ActionOption")
        object.__setattr__(self, "transition_record_hash", _hash(self.transition_record_hash, label="conflict transition record hash"))
        object.__setattr__(self, "transition_spec_hash", _hash(self.transition_spec_hash, label="conflict transition spec hash"))
        delta = _canonical_delta(self.original_delta, label="conflict participant original delta")
        cells = tuple(self.allowed_write_cells)
        if any(not isinstance(cell, StateCellRef) for cell in cells):
            raise TypeError("conflict participant allowed cells must be StateCellRef values")
        if len(set(cells)) != len(cells):
            raise ValueError("conflict participant allowed cells must be unique")
        cells = tuple(sorted(cells, key=_cell_key))
        allowed_keys = {(cell.subject.entity_id, cell.state_variable) for cell in cells}
        for operation in delta.operations:
            if (operation.subject_id, operation.state_variable) not in allowed_keys:
                raise ValueError("conflict participant original delta exceeds allowed cells")
        object.__setattr__(self, "original_delta", delta)
        object.__setattr__(self, "allowed_write_cells", cells)

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "decision_id": self.decision_id,
            "action": self.action.to_dict(),
            "transition_record_hash": self.transition_record_hash,
            "transition_spec_hash": self.transition_spec_hash,
            "original_delta": _delta_payload(self.original_delta),
            "allowed_write_cells": [cell.to_dict() for cell in self.allowed_write_cells],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _participant_write_cells(participant: ConflictParticipant) -> tuple[StateCellRef, ...]:
    by_key = {(cell.subject.entity_id, cell.state_variable): cell for cell in participant.allowed_write_cells}
    cells = {by_key[(operation.subject_id, operation.state_variable)] for operation in participant.original_delta.operations}
    return tuple(sorted(cells, key=_cell_key))


def _connected_participants(participants: tuple[ConflictParticipant, ...]) -> bool:
    writes = {participant.content_hash: frozenset(_participant_write_cells(participant)) for participant in participants}
    adjacency = {participant.content_hash: set() for participant in participants}
    for index, left in enumerate(participants):
        for right in participants[index + 1:]:
            if writes[left.content_hash] & writes[right.content_hash]:
                adjacency[left.content_hash].add(right.content_hash)
                adjacency[right.content_hash].add(left.content_hash)
    start = participants[0].content_hash
    seen: set[str] = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(sorted(adjacency[current], reverse=True))
    return len(seen) == len(participants)


@dataclass(frozen=True)
class ConflictResolutionContext:
    prior_state_hash: str
    participants: tuple[ConflictParticipant, ...]
    conflict_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "prior_state_hash", _hash(self.prior_state_hash, label="conflict prior state hash"))
        participants = tuple(self.participants)
        if len(participants) < 2:
            raise ValueError("conflict context requires at least two participants")
        if any(not isinstance(item, ConflictParticipant) for item in participants):
            raise TypeError("conflict context participants must be ConflictParticipant values")
        participants = tuple(sorted(participants, key=lambda item: (item.actor_id, item.decision_id, item.action.id)))
        keys = tuple((item.actor_id, item.decision_id, item.action.id) for item in participants)
        if len(set(keys)) != len(keys):
            raise ValueError("conflict context participant keys must be unique")
        if not _connected_participants(participants):
            raise ValueError("conflict context participants must form one component")
        counts = Counter(cell for participant in participants for cell in _participant_write_cells(participant))
        expected = tuple(sorted((cell for cell, count in counts.items() if count >= 2), key=_cell_key))
        supplied = tuple(self.conflict_cells)
        if any(not isinstance(cell, StateCellRef) for cell in supplied):
            raise TypeError("conflict context cells must be StateCellRef values")
        if len(set(supplied)) != len(supplied):
            raise ValueError("conflict context cells must be unique")
        supplied = tuple(sorted(supplied, key=_cell_key))
        if not expected or supplied != expected:
            raise ValueError("conflict context cells must equal exact overlapping writes")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "conflict_cells", supplied)

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_state_hash": self.prior_state_hash,
            "participants": [item.to_dict() for item in self.participants],
            "conflict_cells": [cell.to_dict() for cell in self.conflict_cells],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ConflictResolverSpec:
    resolver_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    supported_action_types: tuple[str, ...]
    resolver_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, label="conflict resolver id"))
        object.__setattr__(self, "version", _text(self.version, label="conflict resolver version"))
        object.__setattr__(self, "domain_id", _text(self.domain_id, label="conflict resolver domain id"))
        object.__setattr__(self, "domain_version", _text(self.domain_version, label="conflict resolver domain version"))
        object.__setattr__(self, "domain_spec_hash", _hash(self.domain_spec_hash, label="conflict resolver domain spec hash"))
        action_types = tuple(self.supported_action_types)
        if not action_types:
            raise ValueError("conflict resolver requires supported action types")
        normalized = tuple(_text(item, label="conflict resolver action type") for item in action_types)
        if len(set(normalized)) != len(normalized):
            raise ValueError("conflict resolver action types must be unique")
        if not callable(self.resolver_hook):
            raise TypeError("conflict resolver hook must be callable")
        object.__setattr__(self, "supported_action_types", tuple(sorted(normalized)))

    def to_dict(self) -> dict[str, object]:
        return {
            "resolver_id": self.resolver_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "supported_action_types": list(self.supported_action_types),
            "implementation_identity": measure_implementation(self.resolver_hook).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ConflictResolutionRecord:
    resolver_id: str
    resolver_hash: str
    prior_state_hash: str
    context: ConflictResolutionContext
    resolved_delta: StateDelta

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, label="conflict resolution resolver id"))
        object.__setattr__(self, "resolver_hash", _hash(self.resolver_hash, label="conflict resolution resolver hash"))
        object.__setattr__(self, "prior_state_hash", _hash(self.prior_state_hash, label="conflict resolution prior state hash"))
        if not isinstance(self.context, ConflictResolutionContext):
            raise TypeError("conflict resolution context must be ConflictResolutionContext")
        if self.context.prior_state_hash != self.prior_state_hash:
            raise ValueError("conflict resolution context must bind exact prior state")
        delta = _canonical_delta(self.resolved_delta, label="conflict resolution resolved delta")
        object.__setattr__(self, "resolved_delta", delta)

    def to_dict(self) -> dict[str, object]:
        return {
            "resolver_id": self.resolver_id,
            "resolver_hash": self.resolver_hash,
            "prior_state_hash": self.prior_state_hash,
            "context": self.context.to_dict(),
            "resolved_delta": _delta_payload(self.resolved_delta),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
