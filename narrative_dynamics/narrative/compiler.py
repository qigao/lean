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
