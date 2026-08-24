from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, StateDelta, validate_narrative
from narrative_dynamics.narrative.ir import (
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    StateCellRef,
    TypedValue,
)


_CELL_STATUSES = frozenset({"resolved", "unknown", "conflicted"})
_EVIDENCE_KINDS = frozenset({"direct_perception", "testimony"})
_EVIDENCE_RELATIONS = frozenset({"equals", "not_equals", "clear"})


def _cutoff(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("replay at_time must be a non-negative integer or None")
    return value


def _declared_agent(story: GenericNarrative, agent_id: str) -> None:
    if not isinstance(agent_id, str) or not agent_id.strip() or agent_id != agent_id.strip():
        raise ValueError("replay agent id must be a non-empty trimmed string")
    if not any(entity.id == agent_id for entity in story.entities):
        raise ValueError("replay agent must reference a declared entity")


def _cell_for(
    entities: Mapping[str, Entity],
    subject_id: str,
    state_variable: str,
) -> StateCellRef:
    subject = entities[subject_id]
    return StateCellRef(EntityRef(subject.id, subject.type_name), state_variable)


def _apply_delta(
    state: dict[StateCellRef, TypedValue],
    entities: Mapping[str, Entity],
    delta: StateDelta,
) -> None:
    for operation in delta.operations:
        cell = _cell_for(entities, operation.subject_id, operation.state_variable)
        if operation.kind == "clear":
            state.pop(cell, None)
        else:
            assert operation.value is not None
            state[cell] = operation.value


def _event_replay(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None,
) -> tuple[
    Mapping[StateCellRef, TypedValue],
    tuple[tuple[NarrativeEvent, StateDelta], ...],
]:
    cutoff = _cutoff(at_time)
    entities = {entity.id: entity for entity in story.entities}
    state: dict[StateCellRef, TypedValue] = {}
    rows: list[tuple[NarrativeEvent, StateDelta]] = []
    for event in sorted(story.events, key=lambda item: item.logical_time):
        if cutoff is not None and event.logical_time > cutoff:
            break
        delta = domain.apply_event(MappingProxyType(dict(state)), event, entities)
        rows.append((event, delta))
        _apply_delta(state, entities, delta)
    return MappingProxyType(dict(state)), tuple(rows)


@dataclass(frozen=True)
class EpistemicEvidence:
    cell: StateCellRef
    relation: str
    value: TypedValue | None
    evidence_kind: str
    supporting_id: str
    source_agent: str
    logical_time: int
    provenance_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("epistemic evidence cell must be a StateCellRef")
        if self.relation not in _EVIDENCE_RELATIONS:
            raise ValueError("epistemic evidence relation is not supported")
        if self.value is not None and not isinstance(self.value, TypedValue):
            raise TypeError("epistemic evidence value must be TypedValue or None")
        if self.relation == "clear" and self.value is not None:
            raise ValueError("clear epistemic evidence cannot contain a value")
        if self.relation != "clear" and self.value is None:
            raise ValueError("non-clear epistemic evidence requires a value")
        if self.evidence_kind not in _EVIDENCE_KINDS:
            raise ValueError("epistemic evidence kind is not supported")
        if not isinstance(self.supporting_id, str) or not self.supporting_id:
            raise ValueError("epistemic evidence supporting id must be non-empty")
        if not isinstance(self.source_agent, str) or not self.source_agent:
            raise ValueError("epistemic evidence source agent must be non-empty")
        if (
            not isinstance(self.logical_time, int)
            or isinstance(self.logical_time, bool)
            or self.logical_time < 0
        ):
            raise ValueError("epistemic evidence logical time must be non-negative")
        refs = tuple(self.provenance_refs)
        if any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("epistemic evidence provenance refs must be non-empty strings")
        object.__setattr__(self, "provenance_refs", refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "relation": self.relation,
            "value": None if self.value is None else self.value.to_dict(),
            "evidence_kind": self.evidence_kind,
            "supporting_id": self.supporting_id,
            "source_agent": self.source_agent,
            "logical_time": self.logical_time,
            "provenance_refs": list(self.provenance_refs),
        }


@dataclass(frozen=True)
class EpistemicCellView:
    cell: StateCellRef
    status: str
    resolved_value: TypedValue | None
    constraints: tuple[TypedValue, ...]
    evidence_kind: str | None
    supporting_id: str | None
    source_agent: str | None
    evidence_logical_time: int | None
    evidence_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("epistemic cell view requires a StateCellRef")
        if self.status not in _CELL_STATUSES:
            raise ValueError("epistemic cell status is not supported")
        if self.resolved_value is not None and not isinstance(
            self.resolved_value, TypedValue
        ):
            raise TypeError("epistemic resolved value must be TypedValue or None")
        if self.status == "resolved" and self.resolved_value is None:
            raise ValueError("resolved epistemic cell requires a value")
        if self.status != "resolved" and self.resolved_value is not None:
            raise ValueError("non-resolved epistemic cell cannot contain a resolved value")
        constraints = tuple(self.constraints)
        if any(not isinstance(item, TypedValue) for item in constraints):
            raise TypeError("epistemic constraints must be TypedValue values")
        refs = tuple(self.evidence_refs)
        if any(not isinstance(item, str) or not item for item in refs):
            raise ValueError("epistemic evidence refs must be non-empty strings")
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "evidence_refs", refs)

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "status": self.status,
            "resolved_value": (
                None if self.resolved_value is None else self.resolved_value.to_dict()
            ),
            "constraints": [item.to_dict() for item in self.constraints],
            "evidence_kind": self.evidence_kind,
            "supporting_id": self.supporting_id,
            "source_agent": self.source_agent,
            "evidence_logical_time": self.evidence_logical_time,
            "evidence_refs": list(self.evidence_refs),
        }


@dataclass(frozen=True)
class EpistemicState:
    evidence_history: tuple[EpistemicEvidence, ...]
    cells: Mapping[StateCellRef, EpistemicCellView]
    resolved_values: Mapping[StateCellRef, TypedValue]

    def __post_init__(self) -> None:
        history = tuple(self.evidence_history)
        if any(not isinstance(item, EpistemicEvidence) for item in history):
            raise TypeError("epistemic history must contain EpistemicEvidence values")
        cells = dict(self.cells)
        if any(
            not isinstance(key, StateCellRef)
            or not isinstance(value, EpistemicCellView)
            or key != value.cell
            for key, value in cells.items()
        ):
            raise TypeError("epistemic cells must map StateCellRef to matching views")
        resolved = dict(self.resolved_values)
        if any(
            not isinstance(key, StateCellRef) or not isinstance(value, TypedValue)
            for key, value in resolved.items()
        ):
            raise TypeError("epistemic resolved values must map StateCellRef to TypedValue")
        object.__setattr__(self, "evidence_history", history)
        object.__setattr__(self, "cells", MappingProxyType(cells))
        object.__setattr__(self, "resolved_values", MappingProxyType(resolved))

    def to_dict(self) -> dict[str, object]:
        ordered_cells = tuple(
            sorted(
                self.cells,
                key=lambda cell: (
                    cell.subject.entity_type,
                    cell.subject.entity_id,
                    cell.state_variable,
                ),
            )
        )
        return {
            "evidence_history": [item.to_dict() for item in self.evidence_history],
            "cells": [self.cells[cell].to_dict() for cell in ordered_cells],
            "resolved_values": [
                {
                    "cell": cell.to_dict(),
                    "value": self.resolved_values[cell].to_dict(),
                }
                for cell in ordered_cells
                if cell in self.resolved_values
            ],
        }


def _value_key(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def _unique_constraints(evidence: tuple[EpistemicEvidence, ...]) -> tuple[TypedValue, ...]:
    values: list[TypedValue] = []
    seen: set[str] = set()
    for item in evidence:
        if item.relation != "not_equals":
            continue
        assert item.value is not None
        key = _value_key(item.value)
        if key not in seen:
            seen.add(key)
            values.append(item.value)
    return tuple(values)


def _cell_view(
    cell: StateCellRef,
    evidence: tuple[EpistemicEvidence, ...],
) -> EpistemicCellView:
    last_direct_index: int | None = None
    for index, item in enumerate(evidence):
        if item.evidence_kind == "direct_perception":
            last_direct_index = index
    active = evidence if last_direct_index is None else evidence[last_direct_index:]
    constraints = _unique_constraints(active)
    constraint_keys = {_value_key(value) for value in constraints}

    direct_base = (
        None
        if last_direct_index is None
        else evidence[last_direct_index]
    )
    later_equalities = tuple(
        item
        for item in active
        if item.relation == "equals"
        and (direct_base is None or item is not direct_base)
    )

    main: EpistemicEvidence | None = None
    resolved: TypedValue | None = None
    status = "unknown"

    if later_equalities:
        distinct: dict[str, TypedValue] = {}
        for item in later_equalities:
            assert item.value is not None
            distinct[_value_key(item.value)] = item.value
        main = later_equalities[-1]
        if len(distinct) == 1:
            resolved = next(iter(distinct.values()))
            status = "resolved"
        else:
            status = "conflicted"
    elif direct_base is not None:
        main = direct_base
        if direct_base.relation == "equals":
            assert direct_base.value is not None
            resolved = direct_base.value
            status = "resolved"
        else:
            status = "unknown"
    elif active:
        equalities = tuple(item for item in active if item.relation == "equals")
        if equalities:
            distinct = {}
            for item in equalities:
                assert item.value is not None
                distinct[_value_key(item.value)] = item.value
            main = equalities[-1]
            if len(distinct) == 1:
                resolved = next(iter(distinct.values()))
                status = "resolved"
            else:
                status = "conflicted"
        else:
            main = active[-1]

    if resolved is not None and _value_key(resolved) in constraint_keys:
        resolved = None
        status = "conflicted"

    return EpistemicCellView(
        cell=cell,
        status=status,
        resolved_value=resolved,
        constraints=constraints,
        evidence_kind=None if main is None else main.evidence_kind,
        supporting_id=None if main is None else main.supporting_id,
        source_agent=None if main is None else main.source_agent,
        evidence_logical_time=None if main is None else main.logical_time,
        evidence_refs=() if main is None else main.provenance_refs,
    )


def _state_from_evidence(
    evidence: tuple[EpistemicEvidence, ...],
) -> EpistemicState:
    ordered = tuple(
        sorted(
            evidence,
            key=lambda item: (
                item.logical_time,
                item.supporting_id,
                item.evidence_kind,
            ),
        )
    )
    by_cell: dict[StateCellRef, list[EpistemicEvidence]] = {}
    for item in ordered:
        by_cell.setdefault(item.cell, []).append(item)
    cells = {
        cell: _cell_view(cell, tuple(items))
        for cell, items in by_cell.items()
    }
    resolved = {
        cell: view.resolved_value
        for cell, view in cells.items()
        if view.status == "resolved" and view.resolved_value is not None
    }
    return EpistemicState(ordered, cells, resolved)


def objective_state(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> Mapping[StateCellRef, TypedValue]:
    validate_narrative(story, domain)
    state, _ = _event_replay(story, domain, at_time=at_time)
    return state


def _direct_evidence(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    *,
    at_time: int | None,
) -> tuple[EpistemicEvidence, ...]:
    cutoff = _cutoff(at_time)
    entities = {entity.id: entity for entity in story.entities}
    _, rows = _event_replay(story, domain, at_time=cutoff)
    observed = {
        observation.event_id
        for observation in story.observations
        if observation.agent_id == agent_id
    }
    evidence: list[EpistemicEvidence] = []
    for event, delta in rows:
        if event.id not in observed:
            continue
        for operation in delta.operations:
            cell = _cell_for(entities, operation.subject_id, operation.state_variable)
            evidence.append(
                EpistemicEvidence(
                    cell=cell,
                    relation="clear" if operation.kind == "clear" else "equals",
                    value=operation.value,
                    evidence_kind="direct_perception",
                    supporting_id=event.id,
                    source_agent=agent_id,
                    logical_time=event.logical_time,
                    provenance_refs=(event.id,),
                )
            )
    return tuple(evidence)


def direct_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    *,
    at_time: int | None = None,
) -> EpistemicState:
    validate_narrative(story, domain)
    _declared_agent(story, agent_id)
    return _state_from_evidence(
        _direct_evidence(story, domain, agent_id, at_time=at_time)
    )


def epistemic_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    *,
    at_time: int | None = None,
) -> EpistemicState:
    validate_narrative(story, domain)
    _declared_agent(story, agent_id)
    cutoff = _cutoff(at_time)
    entities = {entity.id: entity for entity in story.entities}
    evidence = list(_direct_evidence(story, domain, agent_id, at_time=cutoff))
    received_claims = {
        reception.claim_id
        for reception in story.receptions
        if reception.recipient_id == agent_id
    }
    for claim in sorted(story.claims, key=lambda item: item.logical_time):
        if claim.id not in received_claims:
            continue
        if cutoff is not None and claim.logical_time > cutoff:
            continue
        proposition = claim.proposition
        subject = entities[proposition.subject.entity_id]
        cell = StateCellRef(
            EntityRef(subject.id, subject.type_name), proposition.state_variable
        )
        evidence.append(
            EpistemicEvidence(
                cell=cell,
                relation=proposition.relation,
                value=proposition.value,
                evidence_kind="testimony",
                supporting_id=claim.id,
                source_agent=claim.speaker_id,
                logical_time=claim.logical_time,
                provenance_refs=(claim.id,) + tuple(claim.support_refs),
            )
        )
    return _state_from_evidence(tuple(evidence))
