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

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate_id", _text(self.predicate_id, label="grounding predicate id"))
        object.__setattr__(
            self,
            "value_ids",
            _strings(self.value_ids, label="grounding predicate value ids", allow_empty=False),
        )
        object.__setattr__(self, "social_topic_id", _optional_text(self.social_topic_id, label="grounding social topic id"))

    def to_dict(self) -> dict[str, object]:
        return {
            "predicate_id": self.predicate_id,
            "value_ids": list(self.value_ids),
            "social_topic_id": self.social_topic_id,
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
            if predicate.social_topic_id is None:
                continue
            topic = topics.get(predicate.social_topic_id)
            if topic is None:
                raise ValueError("grounding predicate social topic must be declared")
            if not set(predicate.value_ids).issubset(topic.symbol_ids):
                raise ValueError("grounding predicate value must be a declared topic symbol")
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
    provider: SituatedGroundingProviderIdentity
    prompt_hash: str
    provider_response_hash: str
    evidence: tuple[SituatedGroundingEvidence, ...]
    claims: tuple[SituatedGroundedClaim, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="grounding artifact model id"))
        object.__setattr__(self, "model_hash", _hash(self.model_hash, label="grounding artifact model hash"))
        object.__setattr__(self, "request_id", _text(self.request_id, label="grounding artifact request id"))
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="grounding artifact observer agent id"))
        if not isinstance(self.provider, SituatedGroundingProviderIdentity):
            raise TypeError("grounding artifact requires provider identity")
        object.__setattr__(self, "prompt_hash", _hash(self.prompt_hash, label="grounding artifact prompt hash"))
        object.__setattr__(self, "provider_response_hash", _hash(self.provider_response_hash, label="grounding provider response hash"))
        if not isinstance(self.evidence, tuple) or any(not isinstance(item, SituatedGroundingEvidence) for item in self.evidence):
            raise TypeError("grounding artifact evidence must be a tuple")
        evidence_ids = tuple(item.evidence_id for item in self.evidence)
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("grounding artifact evidence ids must be unique")
        if any(item.agent_id != self.observer_agent_id for item in self.evidence):
            raise ValueError("grounding artifact evidence must belong to its observer")
        object.__setattr__(self, "evidence", tuple(sorted(self.evidence, key=lambda item: item.evidence_id)))
        if not isinstance(self.claims, tuple) or any(not isinstance(item, SituatedGroundedClaim) for item in self.claims):
            raise TypeError("grounding artifact claims must be a tuple")
        available = set(evidence_ids)
        if any(not set(item.evidence_ids).issubset(available) for item in self.claims):
            raise ValueError("grounded claim evidence must belong to the artifact")
        claim_hashes = tuple(item.content_hash for item in self.claims)
        if len(set(claim_hashes)) != len(claim_hashes):
            raise ValueError("grounding artifact claims must be unique")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.content_hash)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "request_id": self.request_id,
            "observer_agent_id": self.observer_agent_id,
            "provider": self.provider.to_dict(),
            "prompt_hash": self.prompt_hash,
            "provider_response_hash": self.provider_response_hash,
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
    "SituatedGroundingEvidence",
    "SituatedGroundedClaim",
    "SituatedSemanticGroundingArtifact",
)
