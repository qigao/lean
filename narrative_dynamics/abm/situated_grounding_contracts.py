"""Immutable V16 contracts for private, provider-neutral semantic grounding."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import ObservationChannel, SituatedActionKind
from narrative_dynamics.abm.situated_contracts import EvidenceFact
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPercept,
    SituatedPerceptFidelity,
)
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedSocialMemoryModel,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{label} cannot contain NUL")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _strings(values: object, *, label: str, allow_empty: bool = True) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in values)
    if not allow_empty and not result:
        raise ValueError(f"{label} requires at least one value")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(result))


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _positive(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _confidence(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("grounded claim confidence must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("grounded claim confidence must be in [0, 1]")
    return 0.0 if result == 0.0 else result


class SituatedGroundingEvidenceKind(str, Enum):
    PERCEPT = "percept"
    MEMORY = "memory"


class SituatedGroundingPolarity(str, Enum):
    AFFIRMED = "affirmed"
    DENIED = "denied"


class SituatedGroundingModality(str, Enum):
    ASSERTED = "asserted"
    POSSIBLE = "possible"
    UNCERTAIN = "uncertain"


class SituatedGroundingTemporalScope(str, Enum):
    PAST = "past"
    PRESENT = "present"
    FUTURE = "future"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class SituatedGroundingPredicate:
    predicate_id: str
    value_ids: tuple[str, ...]
    social_topic_id: str | None = None
    subject_ids: tuple[str, ...] = ()
    minimum_evidence_fidelity: SituatedPerceptFidelity = SituatedPerceptFidelity.EXACT
    allowed_temporal_scopes: tuple[SituatedGroundingTemporalScope, ...] = (
        SituatedGroundingTemporalScope.PRESENT,
    )
    social_subject_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, label="grounding predicate id"))
        object.__setattr__(
            self,
            "value_ids",
            _strings(self.value_ids, label="grounding predicate value ids", allow_empty=False),
        )
        object.__setattr__(self, "social_topic_id", _optional_text(self.social_topic_id, label="grounding social topic id"))
        object.__setattr__(
            self,
            "subject_ids",
            _strings(self.subject_ids, label="grounding predicate subject ids", allow_empty=False),
        )
        if not isinstance(self.minimum_evidence_fidelity, SituatedPerceptFidelity):
            raise TypeError("grounding predicate minimum fidelity must be SituatedPerceptFidelity")
        if not isinstance(self.allowed_temporal_scopes, tuple) or not self.allowed_temporal_scopes or any(
            not isinstance(item, SituatedGroundingTemporalScope)
            for item in self.allowed_temporal_scopes
        ):
            raise TypeError("grounding predicate temporal scopes must be a non-empty tuple")
        object.__setattr__(
            self,
            "allowed_temporal_scopes",
            tuple(sorted(set(self.allowed_temporal_scopes), key=lambda item: item.value)),
        )
        object.__setattr__(self, "social_subject_id", _optional_text(self.social_subject_id, label="grounding social subject id"))
        if (self.social_topic_id is None) != (self.social_subject_id is None):
            raise ValueError("grounding predicate social topic and subject must be paired")
        if self.social_subject_id is not None and self.social_subject_id not in self.subject_ids:
            raise ValueError("grounding social subject must be allowed by its predicate")

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "value_ids": list(self.value_ids),
            "social_topic_id": self.social_topic_id,
            "subject_ids": list(self.subject_ids),
            "minimum_evidence_fidelity": self.minimum_evidence_fidelity.value,
            "allowed_temporal_scopes": [item.value for item in self.allowed_temporal_scopes],
            "social_subject_id": self.social_subject_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedGroundingProviderIdentity:
    provider_id: str
    version: str
    model_name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "provider_id", _text(self.provider_id, label="grounding provider id"))
        object.__setattr__(self, "version", _text(self.version, label="grounding provider version"))
        object.__setattr__(self, "model_name", _text(self.model_name, label="grounding provider model name"))

    def to_dict(self) -> dict[str, object]:
        return {
            "provider_id": self.provider_id,
            "version": self.version,
            "model_name": self.model_name,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSemanticGroundingModel:
    model_id: str
    version: str
    percept_memory_model: SituatedPerceptMemoryCognitiveModel
    social_memory_model: SituatedSocialMemoryModel
    subject_ids: tuple[str, ...]
    predicates: tuple[SituatedGroundingPredicate, ...]
    maximum_evidence_items: int = 8
    maximum_claims: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="semantic grounding model id"))
        object.__setattr__(self, "version", _text(self.version, label="semantic grounding model version"))
        if not isinstance(self.percept_memory_model, SituatedPerceptMemoryCognitiveModel):
            raise TypeError("semantic grounding requires a percept memory cognitive model")
        if not isinstance(self.social_memory_model, SituatedSocialMemoryModel):
            raise TypeError("semantic grounding requires a social memory model")
        cognition = self.percept_memory_model.cognitive_model
        if self.social_memory_model.memory_cognitive_model.cognitive_model != cognition:
            raise ValueError("semantic grounding memory and social models must bind exact cognition")
        subjects = _strings(self.subject_ids, label="semantic grounding subject ids", allow_empty=False)
        world = cognition.world_model
        world_entities = {
            *(item.agent_id for item in world.agents),
            *(item.object_id for item in world.objects),
            *(item.place_id for item in world.places),
            *(item.passage_id for item in world.passages),
        }
        if not set(subjects).issubset(world_entities):
            raise ValueError("semantic grounding subject must reference a world entity")
        object.__setattr__(self, "subject_ids", subjects)
        if not isinstance(self.predicates, tuple) or not self.predicates or any(
            not isinstance(item, SituatedGroundingPredicate) for item in self.predicates
        ):
            raise TypeError("semantic grounding predicates must be a non-empty tuple")
        predicate_ids = tuple(item.predicate_id for item in self.predicates)
        if len(set(predicate_ids)) != len(predicate_ids):
            raise ValueError("semantic grounding predicate ids must be unique")
        topics = {item.topic_id: item for item in self.social_memory_model.topics}
        for predicate in self.predicates:
            if not set(predicate.subject_ids).issubset(subjects):
                raise ValueError("grounding predicate subject must belong to the model vocabulary")
            if predicate.social_topic_id is None:
                continue
            topic = topics.get(predicate.social_topic_id)
            if topic is None:
                raise ValueError("grounding predicate social topic must be declared")
            if not set(predicate.value_ids).issubset(topic.symbol_ids):
                raise ValueError("grounding predicate value must be a declared topic symbol")
            if SituatedGroundingTemporalScope.PRESENT not in predicate.allowed_temporal_scopes:
                raise ValueError("social grounding predicate must allow present temporal scope")
        object.__setattr__(self, "predicates", tuple(sorted(self.predicates, key=lambda item: item.predicate_id)))
        object.__setattr__(self, "maximum_evidence_items", _positive(self.maximum_evidence_items, label="maximum grounding evidence items"))
        object.__setattr__(self, "maximum_claims", _positive(self.maximum_claims, label="maximum grounded claims"))

    def predicate(self, predicate_id: str) -> SituatedGroundingPredicate | None:
        return next((item for item in self.predicates if item.predicate_id == predicate_id), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "percept_memory_model_hash": self.percept_memory_model.content_hash,
            "social_memory_model_hash": self.social_memory_model.content_hash,
            "subject_ids": list(self.subject_ids),
            "predicates": [item.to_dict() for item in self.predicates],
            "maximum_evidence_items": self.maximum_evidence_items,
            "maximum_claims": self.maximum_claims,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedGroundingRequest:
    request_id: str
    agent_id: str
    primary_memory_id: str
    retrieval_query: str
    maximum_memories: int = 8

    def __post_init__(self) -> None:
        for name in ("request_id", "agent_id", "primary_memory_id", "retrieval_query"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"grounding request {name.replace('_', ' ')}"))
        if len(self.retrieval_query) > 4096:
            raise ValueError("grounding request retrieval query is too long")
        object.__setattr__(self, "maximum_memories", _positive(self.maximum_memories, label="grounding request maximum memories"))
        if self.maximum_memories > 100:
            raise ValueError("grounding request maximum memories cannot exceed 100")

    def to_dict(self) -> dict[str, object]:
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "primary_memory_id": self.primary_memory_id,
            "retrieval_query": self.retrieval_query,
            "maximum_memories": self.maximum_memories,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedGroundingRetrievalPlan:
    terms: tuple[str, ...]
    actor_agent_id: str | None = None
    place_id: str | None = None
    fidelities: tuple[SituatedPerceptFidelity, ...] = ()
    event_kinds: tuple[SituatedActionKind, ...] = ()
    channels: tuple[ObservationChannel, ...] = ()
    min_round: int | None = None
    max_round: int | None = None

    def __post_init__(self) -> None:
        terms = _strings(self.terms, label="grounding retrieval terms", allow_empty=False)
        if len(terms) > 4:
            raise ValueError("grounding retrieval plan allows at most four terms")
        if any(len(item) > 512 for item in terms):
            raise ValueError("grounding retrieval term is too long")
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "actor_agent_id", _optional_text(self.actor_agent_id, label="grounding retrieval actor agent id"))
        object.__setattr__(self, "place_id", _optional_text(self.place_id, label="grounding retrieval place id"))
        for name, expected in (
            ("fidelities", SituatedPerceptFidelity),
            ("event_kinds", SituatedActionKind),
            ("channels", ObservationChannel),
        ):
            values = getattr(self, name)
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"grounding retrieval {name.replace('_', ' ')} has invalid values")
            object.__setattr__(self, name, tuple(sorted(set(values), key=lambda item: item.value)))
        for name in ("min_round", "max_round"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _positive(value, label=f"grounding retrieval {name.replace('_', ' ')}"))
        if self.min_round is not None and self.max_round is not None and self.min_round > self.max_round:
            raise ValueError("grounding retrieval minimum round cannot exceed maximum round")

    def to_dict(self) -> dict[str, object]:
        return {
            "terms": list(self.terms),
            "actor_agent_id": self.actor_agent_id,
            "place_id": self.place_id,
            "fidelities": [item.value for item in self.fidelities],
            "event_kinds": [item.value for item in self.event_kinds],
            "channels": [item.value for item in self.channels],
            "min_round": self.min_round,
            "max_round": self.max_round,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedGroundingEvidence:
    evidence_id: str
    evidence_kind: SituatedGroundingEvidenceKind
    agent_id: str
    source_event_id: str
    source_event_hash: str
    round_index: int
    channels: tuple[ObservationChannel, ...]
    fidelity: SituatedPerceptFidelity
    summary: str
    actor_agent_id: str | None = None
    event_kind: SituatedActionKind | None = None
    place_id: str | None = None
    outcome: str | None = None
    details: tuple[EvidenceFact, ...] = ()
    memory_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_id", _text(self.evidence_id, label="grounding evidence id"))
        if not isinstance(self.evidence_kind, SituatedGroundingEvidenceKind):
            raise TypeError("grounding evidence kind must be SituatedGroundingEvidenceKind")
        object.__setattr__(self, "summary", _text(self.summary, label="grounding evidence summary"))
        object.__setattr__(self, "memory_id", _optional_text(self.memory_id, label="grounding evidence memory id"))
        if self.evidence_kind is SituatedGroundingEvidenceKind.PERCEPT and self.memory_id is not None:
            raise ValueError("percept evidence cannot name a memory")
        if self.evidence_kind is SituatedGroundingEvidenceKind.MEMORY:
            if self.memory_id is None or self.memory_id != self.evidence_id:
                raise ValueError("memory evidence requires its exact memory id")
        percept = SituatedPercept(
            self.evidence_id,
            self.round_index,
            self.agent_id,
            self.source_event_id,
            self.source_event_hash,
            self.channels,
            self.fidelity,
            self.actor_agent_id,
            self.event_kind,
            self.place_id,
            self.outcome,
            self.details,
        )
        object.__setattr__(self, "agent_id", percept.agent_id)
        object.__setattr__(self, "source_event_id", percept.source_event_id)
        object.__setattr__(self, "source_event_hash", percept.source_event_hash)
        object.__setattr__(self, "channels", percept.channels)
        object.__setattr__(self, "actor_agent_id", percept.actor_agent_id)
        object.__setattr__(self, "place_id", percept.place_id)
        object.__setattr__(self, "outcome", percept.outcome)
        object.__setattr__(self, "details", percept.details)

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_kind": self.evidence_kind.value,
            "agent_id": self.agent_id,
            "source_event_id": self.source_event_id,
            "source_event_hash": self.source_event_hash,
            "round_index": self.round_index,
            "channels": [item.value for item in self.channels],
            "fidelity": self.fidelity.value,
            "summary": self.summary,
            "actor_agent_id": self.actor_agent_id,
            "event_kind": None if self.event_kind is None else self.event_kind.value,
            "place_id": self.place_id,
            "outcome": self.outcome,
            "details": [item.to_dict() for item in self.details],
            "memory_id": self.memory_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _semantic_output_schema() -> dict[str, object]:
    return {
        "claims": [{
            "subject_id": "allowed predicate subject id",
            "predicate_id": "allowed predicate id",
            "value_id": "allowed value id for predicate",
            "polarity": [item.value for item in SituatedGroundingPolarity],
            "modality": [item.value for item in SituatedGroundingModality],
            "temporal_scope": [item.value for item in SituatedGroundingTemporalScope],
            "source_agent_id": "allowed agent id or null",
            "confidence": "number in [0, 1]",
            "evidence_ids": ["authorized evidence id including primary"],
        }],
    }


def _semantic_schema_hash() -> str:
    return stable_content_hash({
        "schema_version": "situated-semantic-grounding-v1",
        "output_schema": _semantic_output_schema(),
    })


def _semantic_prompt_template_hash() -> str:
    return stable_content_hash({
        "task": "situated_semantic_grounding_v1",
        "schema_version": "situated-semantic-grounding-v1",
        "payload_fields": [
            "allowed_predicates",
            "allowed_subject_ids",
            "claim_budget",
            "evidence",
            "evidence_budget",
            "interpretation_question",
            "observer_agent_id",
            "output_schema",
            "primary_evidence_id",
            "request_id",
            "schema_version",
        ],
        "output_schema": _semantic_output_schema(),
    })


def _private_context_hash(
    observer_agent_id: str,
    primary_evidence_id: str,
    evidence: tuple[SituatedGroundingEvidence, ...],
) -> str:
    return stable_content_hash({
        "observer_agent_id": observer_agent_id,
        "primary_evidence_id": primary_evidence_id,
        "evidence": [
            item.to_dict() for item in sorted(evidence, key=lambda item: item.evidence_id)
        ],
    })


@dataclass(frozen=True)
class SituatedGroundingPrompt:
    model: SituatedSemanticGroundingModel
    request: SituatedGroundingRequest
    retrieval_plan: SituatedGroundingRetrievalPlan
    primary_evidence_id: str
    evidence: tuple[SituatedGroundingEvidence, ...]
    retrieval_provider: SituatedGroundingProviderIdentity | None = None
    retrieval_response_hash: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, SituatedSemanticGroundingModel):
            raise TypeError("grounding prompt requires a semantic grounding model")
        if not isinstance(self.request, SituatedGroundingRequest):
            raise TypeError("grounding prompt requires a grounding request")
        if not isinstance(self.retrieval_plan, SituatedGroundingRetrievalPlan):
            raise TypeError("grounding prompt requires a retrieval plan")
        object.__setattr__(self, "primary_evidence_id", _text(self.primary_evidence_id, label="grounding prompt primary evidence id"))
        world_agents = {item.agent_id for item in self.model.percept_memory_model.cognitive_model.world_model.agents}
        if self.request.agent_id not in world_agents:
            raise ValueError("grounding prompt requester must be a world agent")
        if self.request.maximum_memories > self.model.maximum_evidence_items:
            raise ValueError("grounding request exceeds model evidence budget")
        if not isinstance(self.evidence, tuple) or not self.evidence or any(
            not isinstance(item, SituatedGroundingEvidence) for item in self.evidence
        ):
            raise TypeError("grounding prompt evidence must be a non-empty tuple")
        if len(self.evidence) > self.request.maximum_memories:
            raise ValueError("grounding prompt exceeds request evidence budget")
        ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(ids)) != len(ids):
            raise ValueError("grounding prompt evidence ids must be unique")
        if self.primary_evidence_id not in ids:
            raise ValueError("grounding prompt must contain its primary evidence")
        if self.request.primary_memory_id != self.primary_evidence_id:
            raise ValueError("grounding prompt primary evidence must bind the request memory")
        if any(item.agent_id != self.request.agent_id for item in self.evidence):
            raise ValueError("grounding prompt evidence must be private to the requester")
        if self.retrieval_provider is not None and not isinstance(
            self.retrieval_provider, SituatedGroundingProviderIdentity
        ):
            raise TypeError("grounding prompt retrieval provider must be a provider identity")
        if (self.retrieval_provider is None) != (self.retrieval_response_hash is None):
            raise ValueError("grounding prompt retrieval provider and response hash must be paired")
        if self.retrieval_response_hash is not None:
            object.__setattr__(
                self,
                "retrieval_response_hash",
                _hash(self.retrieval_response_hash, label="grounding retrieval response hash"),
            )

    @property
    def schema_hash(self) -> str:
        return _semantic_schema_hash()

    @property
    def prompt_template_hash(self) -> str:
        return _semantic_prompt_template_hash()

    @property
    def private_context_hash(self) -> str:
        return _private_context_hash(
            self.request.agent_id,
            self.primary_evidence_id,
            self.evidence,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model.model_id,
            "model_hash": self.model.content_hash,
            "request": self.request.to_dict(),
            "retrieval_plan": self.retrieval_plan.to_dict(),
            "primary_evidence_id": self.primary_evidence_id,
            "evidence": [item.to_dict() for item in self.evidence],
            "schema_hash": self.schema_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "private_context_hash": self.private_context_hash,
            "retrieval_provider": (
                None if self.retrieval_provider is None else self.retrieval_provider.to_dict()
            ),
            "retrieval_response_hash": self.retrieval_response_hash,
        }

    def to_provider_payload(self) -> dict[str, object]:
        return {
            "schema_version": "situated-semantic-grounding-v1",
            "request_id": self.request.request_id,
            "observer_agent_id": self.request.agent_id,
            "interpretation_question": self.request.retrieval_query,
            "primary_evidence_id": self.primary_evidence_id,
            "evidence_budget": self.model.maximum_evidence_items,
            "claim_budget": self.model.maximum_claims,
            "allowed_subject_ids": list(self.model.subject_ids),
            "allowed_predicates": [item.to_dict() for item in self.model.predicates],
            "evidence": [item.to_dict() for item in self.evidence],
            "output_schema": _semantic_output_schema(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedGroundedClaim:
    subject_id: str
    predicate_id: str
    value_id: str
    polarity: SituatedGroundingPolarity
    modality: SituatedGroundingModality
    temporal_scope: SituatedGroundingTemporalScope
    source_agent_id: str | None
    confidence: float
    evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("subject_id", "predicate_id", "value_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"grounded claim {name.replace('_', ' ')}"))
        for name, expected in (
            ("polarity", SituatedGroundingPolarity),
            ("modality", SituatedGroundingModality),
            ("temporal_scope", SituatedGroundingTemporalScope),
        ):
            if not isinstance(getattr(self, name), expected):
                raise TypeError(f"grounded claim {name.replace('_', ' ')} must be {expected.__name__}")
        object.__setattr__(self, "source_agent_id", _optional_text(self.source_agent_id, label="grounded claim source agent id"))
        object.__setattr__(self, "confidence", _confidence(self.confidence))
        object.__setattr__(
            self,
            "evidence_ids",
            _strings(self.evidence_ids, label="grounded claim evidence ids", allow_empty=False),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "subject_id": self.subject_id,
            "predicate_id": self.predicate_id,
            "value_id": self.value_id,
            "polarity": self.polarity.value,
            "modality": self.modality.value,
            "temporal_scope": self.temporal_scope.value,
            "source_agent_id": self.source_agent_id,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSemanticGroundingArtifact:
    model_id: str
    model_hash: str
    request_id: str
    observer_agent_id: str
    primary_evidence_id: str
    provider: SituatedGroundingProviderIdentity
    prompt_hash: str
    schema_hash: str
    prompt_template_hash: str
    private_context_hash: str
    provider_response_hash: str
    validation_result: str
    evidence: tuple[SituatedGroundingEvidence, ...]
    claims: tuple[SituatedGroundedClaim, ...]
    retrieval_provider: SituatedGroundingProviderIdentity | None = None
    retrieval_response_hash: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="grounding artifact model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="grounding artifact model hash"))
        object.__setattr__(self, "request_id", _text(self.request_id, label="grounding artifact request id"))
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="grounding artifact observer agent id"))
        object.__setattr__(self, "primary_evidence_id", _text(self.primary_evidence_id, label="grounding artifact primary evidence id"))
        if not isinstance(self.provider, SituatedGroundingProviderIdentity):
            raise TypeError("grounding artifact requires provider identity")
        object.__setattr__(self, "prompt_hash", _hash(self.prompt_hash, label="grounding artifact prompt hash"))
        object.__setattr__(self, "schema_hash", _hash(self.schema_hash, label="grounding artifact schema hash"))
        object.__setattr__(self, "prompt_template_hash", _hash(self.prompt_template_hash, label="grounding artifact prompt template hash"))
        object.__setattr__(self, "private_context_hash", _hash(self.private_context_hash, label="grounding artifact private context hash"))
        object.__setattr__(self, "provider_response_hash", _hash(self.provider_response_hash, label="grounding provider response hash"))
        if self.validation_result != "accepted":
            raise ValueError("grounding artifact validation result must be accepted")
        if not isinstance(self.evidence, tuple) or any(not isinstance(item, SituatedGroundingEvidence) for item in self.evidence):
            raise TypeError("grounding artifact evidence must be a tuple")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("grounding artifact evidence ids must be unique")
        if any(item.agent_id != self.observer_agent_id for item in self.evidence):
            raise ValueError("grounding artifact evidence must belong to its observer")
        object.__setattr__(self, "evidence", tuple(sorted(self.evidence, key=lambda item: item.evidence_id)))
        if self.primary_evidence_id not in evidence_ids:
            raise ValueError("grounding artifact must retain its primary evidence")
        if self.schema_hash != _semantic_schema_hash():
            raise ValueError("grounding artifact schema hash does not match V16")
        if self.prompt_template_hash != _semantic_prompt_template_hash():
            raise ValueError("grounding artifact prompt template hash does not match V16")
        if self.private_context_hash != _private_context_hash(
            self.observer_agent_id, self.primary_evidence_id, self.evidence
        ):
            raise ValueError("grounding artifact private context hash does not match evidence")
        if not isinstance(self.claims, tuple) or any(not isinstance(item, SituatedGroundedClaim) for item in self.claims):
            raise TypeError("grounding artifact claims must be a tuple")
        available = set(evidence_ids)
        if any(not set(item.evidence_ids).issubset(available) for item in self.claims):
            raise ValueError("grounded claim evidence must belong to the artifact")
        if any(self.primary_evidence_id not in item.evidence_ids for item in self.claims):
            raise ValueError("grounded claim must cite the primary evidence")
        claim_hashes = tuple(item.content_hash for item in self.claims)
        if len(set(claim_hashes)) != len(claim_hashes):
            raise ValueError("grounding artifact claims must be unique")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.content_hash)))
        if self.retrieval_provider is not None and not isinstance(
            self.retrieval_provider, SituatedGroundingProviderIdentity
        ):
            raise TypeError("grounding artifact retrieval provider must be a provider identity")
        if (self.retrieval_provider is None) != (self.retrieval_response_hash is None):
            raise ValueError("grounding artifact retrieval provider and response hash must be paired")
        if self.retrieval_response_hash is not None:
            object.__setattr__(
                self,
                "retrieval_response_hash",
                _hash(self.retrieval_response_hash, label="grounding artifact retrieval response hash"),
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "request_id": self.request_id,
            "observer_agent_id": self.observer_agent_id,
            "primary_evidence_id": self.primary_evidence_id,
            "provider": self.provider.to_dict(),
            "prompt_hash": self.prompt_hash,
            "schema_hash": self.schema_hash,
            "prompt_template_hash": self.prompt_template_hash,
            "private_context_hash": self.private_context_hash,
            "provider_response_hash": self.provider_response_hash,
            "validation_result": self.validation_result,
            "retrieval_provider": (
                None if self.retrieval_provider is None else self.retrieval_provider.to_dict()
            ),
            "retrieval_response_hash": self.retrieval_response_hash,
            "evidence": [item.to_dict() for item in self.evidence],
            "claims": [item.to_dict() for item in self.claims],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


__all__ = (
    "SituatedGroundingEvidenceKind",
    "SituatedGroundingPolarity",
    "SituatedGroundingModality",
    "SituatedGroundingTemporalScope",
    "SituatedGroundingPredicate",
    "SituatedGroundingProviderIdentity",
    "SituatedSemanticGroundingModel",
    "SituatedGroundingRequest",
    "SituatedGroundingRetrievalPlan",
    "SituatedGroundingEvidence",
    "SituatedGroundingPrompt",
    "SituatedGroundedClaim",
    "SituatedSemanticGroundingArtifact",
)
