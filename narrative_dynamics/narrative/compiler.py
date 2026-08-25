from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from narrative_dynamics.contracts import stable_content_hash


_ALLOWED_RELATIONS = frozenset({"equals", "not_equals"})
_ALLOWED_RESOLUTION_DECISIONS = frozenset({"accepted", "rejected", "unresolved"})


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label=label)


def _logical_time(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer or None")
    return value


def _confidence(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("candidate confidence must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise ValueError("candidate confidence must be finite and in [0, 1]")
    return result


def _source_spans(values: object) -> tuple[SourceSpan, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("candidate source spans must be a sequence")
    spans = tuple(values)
    if not spans:
        raise ValueError("candidate requires at least one source span")
    if any(not isinstance(item, SourceSpan) for item in spans):
        raise TypeError("candidate source spans must contain SourceSpan values")
    return spans


def _freeze_json(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} floating-point values must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key in sorted(value):
            name = _text(key, label=f"{label} key")
            frozen[name] = _freeze_json(value[key], label=f"{label}.{name}")
        return MappingProxyType(frozen)
    if isinstance(value, (tuple, list)):
        return tuple(
            _freeze_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{label} must contain canonical JSON-like values")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class SourceDocument:
    document_id: str
    content: str
    media_type: str
    source_uri: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, label="source document id"))
        if not isinstance(self.content, str):
            raise TypeError("source document content must be text")
        object.__setattr__(self, "media_type", _text(self.media_type, label="source media type"))
        object.__setattr__(self, "source_uri", _text(self.source_uri, label="source uri"))

    @property
    def content_hash(self) -> str:
        return stable_content_hash({"content": self.content})

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "media_type": self.media_type,
            "source_uri": self.source_uri,
            "content_hash": self.content_hash,
        }


@dataclass(frozen=True)
class SourceSpan:
    document_id: str
    start: int
    end: int
    exact_text_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "document_id", _text(self.document_id, label="source span document id"))
        if (
            not isinstance(self.start, int)
            or isinstance(self.start, bool)
            or not isinstance(self.end, int)
            or isinstance(self.end, bool)
            or self.start < 0
            or self.end < self.start
        ):
            raise ValueError("source span offsets are invalid")
        object.__setattr__(self, "exact_text_hash", _text(self.exact_text_hash, label="source span text hash"))

    @classmethod
    def from_document(
        cls,
        document: SourceDocument,
        start: int,
        end: int,
    ) -> SourceSpan:
        if not isinstance(document, SourceDocument):
            raise TypeError("source span requires a SourceDocument")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end < start
            or end > len(document.content)
        ):
            raise ValueError("source span offsets are invalid")
        return cls(
            document.document_id,
            start,
            end,
            stable_content_hash({"text": document.content[start:end]}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "document_id": self.document_id,
            "start": self.start,
            "end": self.end,
            "exact_text_hash": self.exact_text_hash,
        }


@dataclass(frozen=True)
class SourceBundle:
    documents: tuple[SourceDocument, ...]

    def __post_init__(self) -> None:
        documents = tuple(self.documents)
        if not documents:
            raise ValueError("source bundle requires at least one document")
        if any(not isinstance(item, SourceDocument) for item in documents):
            raise TypeError("source bundle documents must be SourceDocument values")
        ids = tuple(item.document_id for item in documents)
        if len(set(ids)) != len(ids):
            raise ValueError("source document ids must be unique")
        object.__setattr__(self, "documents", documents)

    def validate_span(self, span: SourceSpan) -> None:
        if not isinstance(span, SourceSpan):
            raise TypeError("source bundle span validation requires SourceSpan")
        document = next(
            (item for item in self.documents if item.document_id == span.document_id),
            None,
        )
        if document is None:
            raise ValueError("source span must reference a source bundle document")
        if span.end > len(document.content):
            raise ValueError("source span offsets are invalid for source document")
        expected = stable_content_hash({"text": document.content[span.start : span.end]})
        if span.exact_text_hash != expected:
            raise ValueError("source span exact text hash does not match source document")

    def to_dict(self) -> dict[str, object]:
        return {"documents": [item.to_dict() for item in self.documents]}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ExtractorIdentity:
    name: str
    version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="extractor name"))
        object.__setattr__(self, "version", _text(self.version, label="extractor version"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "version": self.version}


@dataclass(frozen=True)
class CandidateEntityRef:
    candidate_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate entity ref"))

    def to_dict(self) -> dict[str, object]:
        return {"candidate_id": self.candidate_id}


@dataclass(frozen=True)
class CandidateStateCellRef:
    subject_candidate_id: str
    state_variable: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "subject_candidate_id",
            _text(self.subject_candidate_id, label="candidate state cell subject"),
        )
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="candidate state variable"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_candidate_id": self.subject_candidate_id,
            "state_variable": self.state_variable,
        }


@dataclass(frozen=True)
class CandidateValue:
    type_name: str
    value: bool | int | str | CandidateEntityRef

    def __post_init__(self) -> None:
        object.__setattr__(self, "type_name", _text(self.type_name, label="candidate value type"))
        if not isinstance(self.value, (bool, int, str, CandidateEntityRef)):
            raise TypeError(
                "candidate value must contain a boolean, integer, string, or CandidateEntityRef"
            )

    def to_dict(self) -> dict[str, object]:
        value: object = (
            self.value.to_dict()
            if isinstance(self.value, CandidateEntityRef)
            else self.value
        )
        return {"type_name": self.type_name, "value": value}


def _candidate_values(
    value: Mapping[str, CandidateValue], *, label: str
) -> Mapping[str, CandidateValue]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, CandidateValue] = {}
    for key, item in value.items():
        name = _text(key, label=f"{label} key")
        if not isinstance(item, CandidateValue):
            raise TypeError(f"{label} values must be CandidateValue instances")
        frozen[name] = item
    return MappingProxyType(frozen)


def _candidate_values_dict(value: Mapping[str, CandidateValue]) -> dict[str, object]:
    return {key: item.to_dict() for key, item in value.items()}


@dataclass(frozen=True)
class CandidateEntity:
    candidate_id: str
    entity_id: str | None
    type_name: str | None
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "entity_id", _optional_text(self.entity_id, label="candidate entity id"))
        object.__setattr__(self, "type_name", _optional_text(self.type_name, label="candidate entity type"))
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "entity",
            "candidate_id": self.candidate_id,
            "entity_id": self.entity_id,
            "type_name": self.type_name,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateEvent:
    candidate_id: str
    event_id: str
    type_name: str | None
    logical_time: int | None
    actor_candidate_id: str | None
    arguments: Mapping[str, CandidateValue]
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "event_id", _text(self.event_id, label="candidate event id"))
        object.__setattr__(self, "type_name", _optional_text(self.type_name, label="candidate event type"))
        object.__setattr__(
            self,
            "logical_time",
            _logical_time(self.logical_time, label="candidate event logical time"),
        )
        object.__setattr__(
            self,
            "actor_candidate_id",
            _optional_text(self.actor_candidate_id, label="candidate event actor"),
        )
        object.__setattr__(self, "arguments", _candidate_values(self.arguments, label="candidate event arguments"))
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "event",
            "candidate_id": self.candidate_id,
            "event_id": self.event_id,
            "type_name": self.type_name,
            "logical_time": self.logical_time,
            "actor_candidate_id": self.actor_candidate_id,
            "arguments": _candidate_values_dict(self.arguments),
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateObservation:
    candidate_id: str
    observation_id: str
    agent_candidate_id: str | None
    event_candidate_id: str | None
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "observation_id", _text(self.observation_id, label="candidate observation id"))
        object.__setattr__(self, "agent_candidate_id", _optional_text(self.agent_candidate_id, label="candidate observation agent"))
        object.__setattr__(self, "event_candidate_id", _optional_text(self.event_candidate_id, label="candidate observation event"))
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "observation",
            "candidate_id": self.candidate_id,
            "observation_id": self.observation_id,
            "agent_candidate_id": self.agent_candidate_id,
            "event_candidate_id": self.event_candidate_id,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateProposition:
    subject_candidate_id: str
    state_variable: str
    relation: str
    value: CandidateValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_candidate_id", _text(self.subject_candidate_id, label="candidate proposition subject"))
        object.__setattr__(self, "state_variable", _text(self.state_variable, label="candidate proposition state variable"))
        relation = _text(self.relation, label="candidate proposition relation")
        if relation not in _ALLOWED_RELATIONS:
            raise ValueError("candidate proposition relation must be equals or not_equals")
        object.__setattr__(self, "relation", relation)
        if not isinstance(self.value, CandidateValue):
            raise TypeError("candidate proposition value must be CandidateValue")

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_candidate_id": self.subject_candidate_id,
            "state_variable": self.state_variable,
            "relation": self.relation,
            "value": self.value.to_dict(),
        }


@dataclass(frozen=True)
class CandidateClaim:
    candidate_id: str
    claim_id: str
    logical_time: int | None
    speaker_candidate_id: str | None
    proposition: CandidateProposition
    support_candidate_ids: tuple[str, ...]
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "claim_id", _text(self.claim_id, label="candidate claim id"))
        object.__setattr__(self, "logical_time", _logical_time(self.logical_time, label="candidate claim logical time"))
        object.__setattr__(self, "speaker_candidate_id", _optional_text(self.speaker_candidate_id, label="candidate claim speaker"))
        if not isinstance(self.proposition, CandidateProposition):
            raise TypeError("candidate claim proposition must be CandidateProposition")
        support = tuple(
            _text(item, label="candidate claim support")
            for item in self.support_candidate_ids
        )
        if len(set(support)) != len(support):
            raise ValueError("candidate claim support ids must be unique")
        object.__setattr__(self, "support_candidate_ids", support)
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "claim",
            "candidate_id": self.candidate_id,
            "claim_id": self.claim_id,
            "logical_time": self.logical_time,
            "speaker_candidate_id": self.speaker_candidate_id,
            "proposition": self.proposition.to_dict(),
            "support_candidate_ids": list(self.support_candidate_ids),
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateReception:
    candidate_id: str
    reception_id: str
    claim_candidate_id: str | None
    recipient_candidate_id: str | None
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "reception_id", _text(self.reception_id, label="candidate reception id"))
        object.__setattr__(self, "claim_candidate_id", _optional_text(self.claim_candidate_id, label="candidate reception claim"))
        object.__setattr__(self, "recipient_candidate_id", _optional_text(self.recipient_candidate_id, label="candidate reception recipient"))
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "reception",
            "candidate_id": self.candidate_id,
            "reception_id": self.reception_id,
            "claim_candidate_id": self.claim_candidate_id,
            "recipient_candidate_id": self.recipient_candidate_id,
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CandidateActionOption:
    id: str
    type_name: str
    arguments: Mapping[str, CandidateValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _text(self.id, label="candidate action id"))
        object.__setattr__(self, "type_name", _text(self.type_name, label="candidate action type"))
        object.__setattr__(self, "arguments", _candidate_values(self.arguments, label="candidate action arguments"))

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "type_name": self.type_name,
            "arguments": _candidate_values_dict(self.arguments),
        }


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    decision_id: str
    logical_time: int | None
    actor_candidate_id: str | None
    type_name: str | None
    context_cells: tuple[CandidateStateCellRef, ...]
    actions: tuple[CandidateActionOption, ...]
    source_spans: tuple[SourceSpan, ...]
    confidence: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="candidate id"))
        object.__setattr__(self, "decision_id", _text(self.decision_id, label="candidate decision id"))
        object.__setattr__(self, "logical_time", _logical_time(self.logical_time, label="candidate decision logical time"))
        object.__setattr__(self, "actor_candidate_id", _optional_text(self.actor_candidate_id, label="candidate decision actor"))
        object.__setattr__(self, "type_name", _optional_text(self.type_name, label="candidate decision type"))
        cells = tuple(self.context_cells)
        if any(not isinstance(item, CandidateStateCellRef) for item in cells):
            raise TypeError("candidate decision context cells must be CandidateStateCellRef values")
        actions = tuple(self.actions)
        if any(not isinstance(item, CandidateActionOption) for item in actions):
            raise TypeError("candidate decision actions must be CandidateActionOption values")
        if len({item.id for item in actions}) != len(actions):
            raise ValueError("candidate decision action ids must be unique")
        object.__setattr__(self, "context_cells", cells)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "source_spans", _source_spans(self.source_spans))
        object.__setattr__(self, "confidence", _confidence(self.confidence))

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "decision",
            "candidate_id": self.candidate_id,
            "decision_id": self.decision_id,
            "logical_time": self.logical_time,
            "actor_candidate_id": self.actor_candidate_id,
            "type_name": self.type_name,
            "context_cells": [item.to_dict() for item in self.context_cells],
            "actions": [item.to_dict() for item in self.actions],
            "source_spans": [item.to_dict() for item in self.source_spans],
            "confidence": self.confidence,
        }


CandidateRecord = (
    CandidateEntity
    | CandidateEvent
    | CandidateObservation
    | CandidateClaim
    | CandidateReception
    | CandidateDecision
)


def _candidate_entity_refs(value: CandidateValue) -> tuple[str, ...]:
    if isinstance(value.value, CandidateEntityRef):
        return (value.value.candidate_id,)
    return ()


def _require_ref(
    candidate_id: str | None,
    allowed: set[str],
    *,
    label: str,
) -> None:
    if candidate_id is not None and candidate_id not in allowed:
        raise ValueError(f"{label} must reference a declared candidate id")


@dataclass(frozen=True)
class CandidateBundle:
    extractor: ExtractorIdentity
    candidates: tuple[CandidateRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.extractor, ExtractorIdentity):
            raise TypeError("candidate bundle extractor must be ExtractorIdentity")
        candidates = tuple(self.candidates)
        allowed_types = (
            CandidateEntity,
            CandidateEvent,
            CandidateObservation,
            CandidateClaim,
            CandidateReception,
            CandidateDecision,
        )
        if not candidates:
            raise ValueError("candidate bundle requires at least one candidate")
        if any(not isinstance(item, allowed_types) for item in candidates):
            raise TypeError("candidate bundle contains unsupported candidate record")
        ids = tuple(item.candidate_id for item in candidates)
        if len(set(ids)) != len(ids):
            raise ValueError("candidate ids must be unique")

        entity_ids = {
            item.candidate_id for item in candidates if isinstance(item, CandidateEntity)
        }
        event_ids = {
            item.candidate_id for item in candidates if isinstance(item, CandidateEvent)
        }
        claim_ids = {
            item.candidate_id for item in candidates if isinstance(item, CandidateClaim)
        }
        support_ids = event_ids | claim_ids

        for item in candidates:
            if isinstance(item, CandidateEvent):
                _require_ref(item.actor_candidate_id, entity_ids, label="candidate event actor")
                for value in item.arguments.values():
                    for ref in _candidate_entity_refs(value):
                        _require_ref(ref, entity_ids, label="candidate event argument")
            elif isinstance(item, CandidateObservation):
                _require_ref(item.agent_candidate_id, entity_ids, label="candidate observation agent")
                _require_ref(item.event_candidate_id, event_ids, label="candidate observation event")
            elif isinstance(item, CandidateClaim):
                _require_ref(item.speaker_candidate_id, entity_ids, label="candidate claim speaker")
                _require_ref(
                    item.proposition.subject_candidate_id,
                    entity_ids,
                    label="candidate proposition subject",
                )
                for ref in _candidate_entity_refs(item.proposition.value):
                    _require_ref(ref, entity_ids, label="candidate proposition value")
                for ref in item.support_candidate_ids:
                    _require_ref(ref, support_ids, label="candidate claim support")
            elif isinstance(item, CandidateReception):
                _require_ref(item.claim_candidate_id, claim_ids, label="candidate reception claim")
                _require_ref(item.recipient_candidate_id, entity_ids, label="candidate reception recipient")
            elif isinstance(item, CandidateDecision):
                _require_ref(item.actor_candidate_id, entity_ids, label="candidate decision actor")
                for cell in item.context_cells:
                    _require_ref(
                        cell.subject_candidate_id,
                        entity_ids,
                        label="candidate decision context cell",
                    )
                for action in item.actions:
                    for value in action.arguments.values():
                        for ref in _candidate_entity_refs(value):
                            _require_ref(ref, entity_ids, label="candidate action argument")

        object.__setattr__(self, "candidates", candidates)

    def to_dict(self) -> dict[str, object]:
        return {
            "extractor": self.extractor.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ResolutionRecord:
    candidate_id: str
    decision: str
    selected_fields: Mapping[str, object]
    resolver_id: str
    rationale: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_id", _text(self.candidate_id, label="resolution candidate id"))
        decision = _text(self.decision, label="resolution decision")
        if decision not in _ALLOWED_RESOLUTION_DECISIONS:
            raise ValueError("resolution decision must be accepted, rejected, or unresolved")
        object.__setattr__(self, "decision", decision)
        if not isinstance(self.selected_fields, Mapping):
            raise TypeError("resolution selected fields must be a mapping")
        frozen = _freeze_json(self.selected_fields, label="resolution selected fields")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "selected_fields", frozen)
        object.__setattr__(self, "resolver_id", _text(self.resolver_id, label="resolution resolver id"))
        object.__setattr__(self, "rationale", _text(self.rationale, label="resolution rationale"))

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "decision": self.decision,
            "selected_fields": _thaw_json(self.selected_fields),
            "resolver_id": self.resolver_id,
            "rationale": self.rationale,
        }


# Deterministic candidate-to-canonical compiler (V1).
from dataclasses import replace as _replace

from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Claim,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    Proposition,
    Reception,
    StateCellRef,
    TypedValue,
)


_COMPILATION_STATUSES = frozenset({"canonical", "incomplete", "rejected"})
_ALLOWED_RESOLUTION_FIELDS = {
    CandidateEntity: frozenset({"entity_id", "type_name"}),
    CandidateEvent: frozenset({"logical_time", "type_name", "actor_candidate_id"}),
    CandidateObservation: frozenset({"agent_candidate_id", "event_candidate_id"}),
    CandidateClaim: frozenset({"logical_time", "speaker_candidate_id"}),
    CandidateReception: frozenset({"claim_candidate_id", "recipient_candidate_id"}),
    CandidateDecision: frozenset({"logical_time", "actor_candidate_id", "type_name"}),
}


@dataclass(frozen=True)
class CompilationDiagnostic:
    code: str
    candidate_id: str | None
    field: str | None
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _text(self.code, label="compilation diagnostic code"))
        object.__setattr__(
            self,
            "candidate_id",
            _optional_text(self.candidate_id, label="compilation diagnostic candidate id"),
        )
        object.__setattr__(
            self,
            "field",
            _optional_text(self.field, label="compilation diagnostic field"),
        )
        object.__setattr__(
            self,
            "message",
            _text(self.message, label="compilation diagnostic message"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "candidate_id": self.candidate_id,
            "field": self.field,
            "message": self.message,
        }


@dataclass(frozen=True)
class CompilationResult:
    status: str
    domain_id: str
    domain_version: str
    source_bundle_hash: str
    candidate_bundle_hash: str
    resolution_bundle_hash: str
    canonical_scenario: GenericNarrative | None
    canonical_hash: str | None
    diagnostics: tuple[CompilationDiagnostic, ...]

    def __post_init__(self) -> None:
        status = _text(self.status, label="compilation status")
        if status not in _COMPILATION_STATUSES:
            raise ValueError("compilation status must be canonical, incomplete, or rejected")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "domain_id", _text(self.domain_id, label="compilation domain id"))
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="compilation domain version"),
        )
        for name in (
            "source_bundle_hash",
            "candidate_bundle_hash",
            "resolution_bundle_hash",
        ):
            object.__setattr__(
                self,
                name,
                _text(getattr(self, name), label=f"compilation {name.replace('_', ' ')}"),
            )
        diagnostics = tuple(self.diagnostics)
        if any(not isinstance(item, CompilationDiagnostic) for item in diagnostics):
            raise TypeError("compilation diagnostics must contain CompilationDiagnostic values")
        object.__setattr__(
            self,
            "diagnostics",
            tuple(
                sorted(
                    diagnostics,
                    key=lambda item: (
                        "" if item.candidate_id is None else item.candidate_id,
                        "" if item.field is None else item.field,
                        item.code,
                        item.message,
                    ),
                )
            ),
        )
        if status == "canonical":
            if not isinstance(self.canonical_scenario, GenericNarrative):
                raise TypeError("canonical compilation requires a GenericNarrative")
            if self.canonical_hash != self.canonical_scenario.content_hash:
                raise ValueError("canonical compilation hash must match canonical scenario")
            if self.canonical_scenario.domain_id != self.domain_id:
                raise ValueError("canonical compilation domain id must match scenario")
            if self.canonical_scenario.domain_version != self.domain_version:
                raise ValueError("canonical compilation domain version must match scenario")
        elif self.canonical_scenario is not None or self.canonical_hash is not None:
            raise ValueError("non-canonical compilation cannot contain canonical output")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "source_bundle_hash": self.source_bundle_hash,
            "candidate_bundle_hash": self.candidate_bundle_hash,
            "resolution_bundle_hash": self.resolution_bundle_hash,
            "canonical_scenario": (
                None if self.canonical_scenario is None else self.canonical_scenario.to_dict()
            ),
            "canonical_hash": self.canonical_hash,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _resolution_bundle_hash(resolutions: tuple[ResolutionRecord, ...]) -> str:
    ordered = tuple(
        sorted(
            resolutions,
            key=lambda item: (
                item.candidate_id,
                item.decision,
                stable_content_hash(item.to_dict()),
            ),
        )
    )
    return stable_content_hash(tuple(item.to_dict() for item in ordered))


def _compilation_result(
    *,
    status: str,
    domain: DomainSpec,
    source: SourceBundle,
    candidates: CandidateBundle,
    resolution_bundle_hash: str,
    canonical_scenario: GenericNarrative | None = None,
    diagnostics: tuple[CompilationDiagnostic, ...] = (),
) -> CompilationResult:
    return CompilationResult(
        status=status,
        domain_id=domain.domain_id,
        domain_version=domain.version,
        source_bundle_hash=source.content_hash,
        candidate_bundle_hash=candidates.content_hash,
        resolution_bundle_hash=resolution_bundle_hash,
        canonical_scenario=canonical_scenario,
        canonical_hash=(None if canonical_scenario is None else canonical_scenario.content_hash),
        diagnostics=diagnostics,
    )


def _rejected_compilation(
    *,
    domain: DomainSpec,
    source: SourceBundle,
    candidates: CandidateBundle,
    resolution_bundle_hash: str,
    code: str,
    error: ValueError,
    candidate_id: str | None = None,
    field: str | None = None,
) -> CompilationResult:
    return _compilation_result(
        status="rejected",
        domain=domain,
        source=source,
        candidates=candidates,
        resolution_bundle_hash=resolution_bundle_hash,
        diagnostics=(CompilationDiagnostic(code, candidate_id, field, str(error)),),
    )


def _candidate_lookup(candidates: CandidateBundle) -> dict[str, CandidateRecord]:
    return {item.candidate_id: item for item in candidates.candidates}


def _validate_source_spans(source: SourceBundle, candidates: CandidateBundle) -> None:
    for candidate in candidates.candidates:
        for span in candidate.source_spans:
            source.validate_span(span)


def _apply_resolutions(
    candidates: CandidateBundle,
    resolutions: tuple[ResolutionRecord, ...],
) -> tuple[tuple[CandidateRecord, ...], set[str]]:
    by_id = _candidate_lookup(candidates)
    resolution_by_id: dict[str, ResolutionRecord] = {}
    for resolution in resolutions:
        if resolution.candidate_id not in by_id:
            raise ValueError("resolution candidate must reference a declared candidate")
        if resolution.candidate_id in resolution_by_id:
            raise ValueError("candidate may have at most one resolution record")
        candidate = by_id[resolution.candidate_id]
        allowed = _ALLOWED_RESOLUTION_FIELDS[type(candidate)]
        if set(resolution.selected_fields) - set(allowed):
            raise ValueError("resolution selected field is not allowed for candidate type")
        resolution_by_id[resolution.candidate_id] = resolution

    resolved: list[CandidateRecord] = []
    explicitly_unresolved: set[str] = set()
    for candidate in candidates.candidates:
        resolution = resolution_by_id.get(candidate.candidate_id)
        if resolution is None:
            resolved.append(candidate)
        elif resolution.decision == "rejected":
            continue
        elif resolution.decision == "unresolved":
            explicitly_unresolved.add(candidate.candidate_id)
            resolved.append(candidate)
        else:
            selected = dict(resolution.selected_fields)
            resolved.append(_replace(candidate, **selected) if selected else candidate)
    return tuple(resolved), explicitly_unresolved


def _required_fields(candidate: CandidateRecord, domain: DomainSpec) -> tuple[str, ...]:
    if isinstance(candidate, CandidateEntity):
        return ("entity_id", "type_name")
    if isinstance(candidate, CandidateEvent):
        fields = ["logical_time", "type_name"]
        if candidate.type_name is not None and domain._event_type(candidate.type_name).actor_type is not None:
            fields.append("actor_candidate_id")
        return tuple(fields)
    if isinstance(candidate, CandidateObservation):
        return ("agent_candidate_id", "event_candidate_id")
    if isinstance(candidate, CandidateClaim):
        return ("logical_time", "speaker_candidate_id")
    if isinstance(candidate, CandidateReception):
        return ("claim_candidate_id", "recipient_candidate_id")
    if isinstance(candidate, CandidateDecision):
        return ("logical_time", "actor_candidate_id", "type_name")
    raise TypeError("unsupported candidate record")


def _incomplete_diagnostics(
    candidates: tuple[CandidateRecord, ...],
    explicitly_unresolved: set[str],
    domain: DomainSpec,
) -> tuple[CompilationDiagnostic, ...]:
    diagnostics: list[CompilationDiagnostic] = []
    for candidate in candidates:
        required = _required_fields(candidate, domain)
        missing = tuple(field for field in required if getattr(candidate, field) is None)
        if candidate.candidate_id in explicitly_unresolved:
            diagnostics.append(
                CompilationDiagnostic(
                    "unresolved_required_field",
                    candidate.candidate_id,
                    missing[0] if missing else "resolution",
                    "required semantic field is unresolved",
                )
            )
            continue
        for field in missing:
            diagnostics.append(
                CompilationDiagnostic(
                    "unresolved_required_field",
                    candidate.candidate_id,
                    field,
                    "required semantic field is unresolved",
                )
            )
    return tuple(diagnostics)


def _require_entity_ref(
    candidate_id: str | None,
    entity_refs: Mapping[str, EntityRef],
    *,
    label: str,
) -> EntityRef:
    if candidate_id is None or candidate_id not in entity_refs:
        raise ValueError(f"{label} must reference an accepted entity candidate")
    return entity_refs[candidate_id]


def _compile_candidate_value(
    value: CandidateValue,
    entity_refs: Mapping[str, EntityRef],
) -> TypedValue:
    raw: object = value.value
    if isinstance(raw, CandidateEntityRef):
        raw = _require_entity_ref(
            raw.candidate_id,
            entity_refs,
            label="candidate value entity reference",
        )
    return TypedValue(value.type_name, raw)


def _compile_story(
    candidates: tuple[CandidateRecord, ...],
    domain: DomainSpec,
) -> GenericNarrative:
    entity_refs: dict[str, EntityRef] = {}
    entities_by_id: dict[str, Entity] = {}
    for candidate in candidates:
        if not isinstance(candidate, CandidateEntity):
            continue
        assert candidate.entity_id is not None and candidate.type_name is not None
        existing = entities_by_id.get(candidate.entity_id)
        if existing is not None and existing.type_name != candidate.type_name:
            raise ValueError("duplicate canonical entity id must have one entity type")
        if existing is None:
            existing = Entity(candidate.entity_id, candidate.type_name)
            entities_by_id[candidate.entity_id] = existing
        entity_refs[candidate.candidate_id] = EntityRef(existing.id, existing.type_name)

    event_ids = {
        item.candidate_id: item.event_id
        for item in candidates
        if isinstance(item, CandidateEvent)
    }
    claim_ids = {
        item.candidate_id: item.claim_id
        for item in candidates
        if isinstance(item, CandidateClaim)
    }
    support_ids = {**event_ids, **claim_ids}

    events: list[NarrativeEvent] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateEvent):
            continue
        assert candidate.logical_time is not None and candidate.type_name is not None
        actor_id = None
        if candidate.actor_candidate_id is not None:
            actor_id = _require_entity_ref(
                candidate.actor_candidate_id,
                entity_refs,
                label="candidate event actor",
            ).entity_id
        events.append(
            NarrativeEvent(
                candidate.event_id,
                candidate.logical_time,
                candidate.type_name,
                actor_id,
                {
                    key: _compile_candidate_value(value, entity_refs)
                    for key, value in candidate.arguments.items()
                },
            )
        )

    observations: list[Observation] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateObservation):
            continue
        assert candidate.agent_candidate_id is not None
        assert candidate.event_candidate_id is not None
        if candidate.event_candidate_id not in event_ids:
            raise ValueError("candidate observation event must reference an accepted event")
        observations.append(
            Observation(
                candidate.observation_id,
                _require_entity_ref(
                    candidate.agent_candidate_id,
                    entity_refs,
                    label="candidate observation agent",
                ).entity_id,
                event_ids[candidate.event_candidate_id],
            )
        )

    claims: list[Claim] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateClaim):
            continue
        assert candidate.logical_time is not None
        assert candidate.speaker_candidate_id is not None
        support: list[str] = []
        for ref in candidate.support_candidate_ids:
            if ref not in support_ids:
                raise ValueError("candidate claim support must reference an accepted record")
            support.append(support_ids[ref])
        claims.append(
            Claim(
                candidate.claim_id,
                candidate.logical_time,
                _require_entity_ref(
                    candidate.speaker_candidate_id,
                    entity_refs,
                    label="candidate claim speaker",
                ).entity_id,
                Proposition(
                    _require_entity_ref(
                        candidate.proposition.subject_candidate_id,
                        entity_refs,
                        label="candidate proposition subject",
                    ),
                    candidate.proposition.state_variable,
                    candidate.proposition.relation,
                    _compile_candidate_value(candidate.proposition.value, entity_refs),
                ),
                tuple(support),
            )
        )

    receptions: list[Reception] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateReception):
            continue
        assert candidate.claim_candidate_id is not None
        assert candidate.recipient_candidate_id is not None
        if candidate.claim_candidate_id not in claim_ids:
            raise ValueError("candidate reception claim must reference an accepted claim")
        receptions.append(
            Reception(
                candidate.reception_id,
                claim_ids[candidate.claim_candidate_id],
                _require_entity_ref(
                    candidate.recipient_candidate_id,
                    entity_refs,
                    label="candidate reception recipient",
                ).entity_id,
            )
        )

    decisions: list[Decision] = []
    for candidate in candidates:
        if not isinstance(candidate, CandidateDecision):
            continue
        assert candidate.logical_time is not None
        assert candidate.actor_candidate_id is not None
        assert candidate.type_name is not None
        decisions.append(
            Decision(
                candidate.decision_id,
                candidate.logical_time,
                _require_entity_ref(
                    candidate.actor_candidate_id,
                    entity_refs,
                    label="candidate decision actor",
                ).entity_id,
                candidate.type_name,
                tuple(
                    StateCellRef(
                        _require_entity_ref(
                            cell.subject_candidate_id,
                            entity_refs,
                            label="candidate decision context cell",
                        ),
                        cell.state_variable,
                    )
                    for cell in candidate.context_cells
                ),
                tuple(
                    ActionOption(
                        action.id,
                        action.type_name,
                        {
                            key: _compile_candidate_value(value, entity_refs)
                            for key, value in action.arguments.items()
                        },
                    )
                    for action in candidate.actions
                ),
            )
        )

    story = GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=tuple(entities_by_id.values()),
        events=tuple(events),
        observations=tuple(observations),
        claims=tuple(claims),
        receptions=tuple(receptions),
        decisions=tuple(decisions),
    )
    validate_narrative(story, domain)
    return story


def compile_candidates(
    source: SourceBundle,
    candidates: CandidateBundle,
    resolutions: tuple[ResolutionRecord, ...],
    domain: DomainSpec,
) -> CompilationResult:
    if not isinstance(source, SourceBundle):
        raise TypeError("narrative compiler requires a SourceBundle")
    if not isinstance(candidates, CandidateBundle):
        raise TypeError("narrative compiler requires a CandidateBundle")
    if not isinstance(domain, DomainSpec):
        raise TypeError("narrative compiler requires a DomainSpec")
    resolution_values = tuple(resolutions)
    if any(not isinstance(item, ResolutionRecord) for item in resolution_values):
        raise TypeError("narrative compiler resolutions must be ResolutionRecord values")
    resolution_hash = _resolution_bundle_hash(resolution_values)

    try:
        _validate_source_spans(source, candidates)
    except ValueError as error:
        return _rejected_compilation(
            domain=domain,
            source=source,
            candidates=candidates,
            resolution_bundle_hash=resolution_hash,
            code="invalid_source_span",
            error=error,
        )

    try:
        resolved, explicitly_unresolved = _apply_resolutions(candidates, resolution_values)
    except ValueError as error:
        return _rejected_compilation(
            domain=domain,
            source=source,
            candidates=candidates,
            resolution_bundle_hash=resolution_hash,
            code="invalid_resolution",
            error=error,
        )

    try:
        diagnostics = _incomplete_diagnostics(resolved, explicitly_unresolved, domain)
    except ValueError as error:
        return _rejected_compilation(
            domain=domain,
            source=source,
            candidates=candidates,
            resolution_bundle_hash=resolution_hash,
            code="compiler_validation",
            error=error,
        )
    if diagnostics:
        return _compilation_result(
            status="incomplete",
            domain=domain,
            source=source,
            candidates=candidates,
            resolution_bundle_hash=resolution_hash,
            diagnostics=diagnostics,
        )

    try:
        story = _compile_story(resolved, domain)
    except ValueError as error:
        return _rejected_compilation(
            domain=domain,
            source=source,
            candidates=candidates,
            resolution_bundle_hash=resolution_hash,
            code="semantic_validation",
            error=error,
        )

    return _compilation_result(
        status="canonical",
        domain=domain,
        source=source,
        candidates=candidates,
        resolution_bundle_hash=resolution_hash,
        canonical_scenario=story,
    )
