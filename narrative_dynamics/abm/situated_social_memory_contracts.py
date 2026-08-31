"""Immutable contracts for V14 situated social-memory revision."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated_cognition_contracts import SituatedCognitiveState
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryCognitiveModel,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _strings(values: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(result))


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a content hash")
    return value


def _unit(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError(f"{label} must be finite numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between zero and one")
    return result


def _affinity(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TypeError("situated source affinity must be finite numeric")
    result = float(value)
    if not -1.0 <= result <= 1.0:
        raise ValueError("situated source affinity must be between minus one and one")
    return result


def _nonnegative(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _positive(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


class SituatedClaimStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    CONFIRMED = "confirmed"
    CONTRADICTED = "contradicted"
    FORGOTTEN = "forgotten"


class SituatedSocialEvidenceKind(str, Enum):
    TESTIMONY = "testimony"
    VERIFICATION = "verification"


@dataclass(frozen=True)
class SituatedClaimTopic:
    topic_id: str
    symbol_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "topic_id", _text(self.topic_id, label="situated claim topic id"))
        symbols = _strings(self.symbol_ids, label="situated claim topic symbol ids")
        if len(symbols) < 2:
            raise ValueError("situated claim topic requires at least two mutually exclusive symbols")
        object.__setattr__(self, "symbol_ids", symbols)

    def to_dict(self) -> dict[str, object]:
        return {"topic_id": self.topic_id, "symbol_ids": list(self.symbol_ids)}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSocialMemoryPolicy:
    initial_source_trust: float = 0.5
    confirmation_rate: float = 0.2
    contradiction_rate: float = 0.4
    confirmation_affinity_delta: float = 0.1
    contradiction_affinity_delta: float = 0.2
    max_unresolved_age_rounds: int = 20
    max_active_claims: int = 100

    def __post_init__(self) -> None:
        for name in (
            "initial_source_trust",
            "confirmation_rate",
            "contradiction_rate",
            "confirmation_affinity_delta",
            "contradiction_affinity_delta",
        ):
            object.__setattr__(self, name, _unit(getattr(self, name), label=f"situated social {name.replace('_', ' ')}"))
        object.__setattr__(
            self,
            "max_unresolved_age_rounds",
            _positive(self.max_unresolved_age_rounds, label="situated social maximum unresolved age"),
        )
        object.__setattr__(
            self,
            "max_active_claims",
            _positive(self.max_active_claims, label="situated social maximum active claims"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "initial_source_trust": self.initial_source_trust,
            "confirmation_rate": self.confirmation_rate,
            "contradiction_rate": self.contradiction_rate,
            "confirmation_affinity_delta": self.confirmation_affinity_delta,
            "contradiction_affinity_delta": self.contradiction_affinity_delta,
            "max_unresolved_age_rounds": self.max_unresolved_age_rounds,
            "max_active_claims": self.max_active_claims,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSocialMemoryModel:
    model_id: str
    version: str
    memory_cognitive_model: SituatedMemoryCognitiveModel
    topics: tuple[SituatedClaimTopic, ...]
    policy: SituatedSocialMemoryPolicy

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated social memory model id"))
        object.__setattr__(self, "version", _text(self.version, label="situated social memory model version"))
        if not isinstance(self.memory_cognitive_model, SituatedMemoryCognitiveModel):
            raise TypeError("situated social memory model requires SituatedMemoryCognitiveModel")
        if not isinstance(self.topics, tuple) or not self.topics or any(
            not isinstance(item, SituatedClaimTopic) for item in self.topics
        ):
            raise TypeError("situated social memory topics must be a non-empty tuple")
        topic_ids = tuple(item.topic_id for item in self.topics)
        if len(set(topic_ids)) != len(topic_ids):
            raise ValueError("situated social memory topic ids must be unique")
        symbols = tuple(symbol for topic in self.topics for symbol in topic.symbol_ids)
        if len(set(symbols)) != len(symbols):
            raise ValueError("each cognitive symbol may belong to only one social topic")
        agents = self.memory_cognitive_model.cognitive_model.agents
        available = set.intersection(*(
            {item.symbol_id for item in agent.observation_symbols}
            for agent in agents
        ))
        if not set(symbols).issubset(available):
            raise ValueError("situated social topic must use a cognitive observation symbol")
        if not isinstance(self.policy, SituatedSocialMemoryPolicy):
            raise TypeError("situated social memory model requires SituatedSocialMemoryPolicy")
        object.__setattr__(self, "topics", tuple(sorted(self.topics, key=lambda item: item.topic_id)))

    def topic_for_symbol(self, symbol_id: str) -> SituatedClaimTopic | None:
        return next((item for item in self.topics if symbol_id in item.symbol_ids), None)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "memory_cognitive_model_hash": self.memory_cognitive_model.content_hash,
            "topics": [item.to_dict() for item in self.topics],
            "policy": self.policy.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSourceRelationship:
    observer_agent_id: str
    source_agent_id: str
    trust: float
    affinity: float
    confirmation_count: int
    contradiction_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "observer_agent_id", _text(self.observer_agent_id, label="situated source observer id"))
        object.__setattr__(self, "source_agent_id", _text(self.source_agent_id, label="situated source agent id"))
        if self.observer_agent_id == self.source_agent_id:
            raise ValueError("situated source observer and source must differ")
        object.__setattr__(self, "trust", _unit(self.trust, label="situated source trust"))
        object.__setattr__(self, "affinity", _affinity(self.affinity))
        object.__setattr__(self, "confirmation_count", _nonnegative(self.confirmation_count, label="situated source confirmation count"))
        object.__setattr__(self, "contradiction_count", _nonnegative(self.contradiction_count, label="situated source contradiction count"))

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_agent_id": self.observer_agent_id,
            "source_agent_id": self.source_agent_id,
            "trust": self.trust,
            "affinity": self.affinity,
            "confirmation_count": self.confirmation_count,
            "contradiction_count": self.contradiction_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedConsolidatedClaim:
    claim_id: str
    observer_agent_id: str
    source_agent_id: str
    topic_id: str
    symbol_id: str
    event_ids: tuple[str, ...]
    memory_ids: tuple[str, ...]
    first_round: int
    last_round: int
    support_count: int
    status: SituatedClaimStatus

    def __post_init__(self) -> None:
        for name in ("claim_id", "observer_agent_id", "source_agent_id", "topic_id", "symbol_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"situated claim {name.replace('_', ' ')}"))
        if self.observer_agent_id == self.source_agent_id:
            raise ValueError("situated claim observer and source must differ")
        events = _strings(self.event_ids, label="situated claim event ids")
        memories = _strings(self.memory_ids, label="situated claim memory ids")
        if not events:
            raise ValueError("situated claim requires at least one event")
        object.__setattr__(self, "event_ids", events)
        object.__setattr__(self, "memory_ids", memories)
        object.__setattr__(self, "first_round", _positive(self.first_round, label="situated claim first round"))
        object.__setattr__(self, "last_round", _nonnegative(self.last_round, label="situated claim last round"))
        if self.last_round < self.first_round:
            raise ValueError("situated claim last round cannot precede first round")
        object.__setattr__(self, "support_count", _positive(self.support_count, label="situated claim support count"))
        if self.support_count != len(events):
            raise ValueError("situated claim support count must equal event count")
        if len(memories) > self.support_count:
            raise ValueError("situated claim memory count cannot exceed support count")
        if not isinstance(self.status, SituatedClaimStatus):
            raise TypeError("situated claim status must be SituatedClaimStatus")

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "observer_agent_id": self.observer_agent_id,
            "source_agent_id": self.source_agent_id,
            "topic_id": self.topic_id,
            "symbol_id": self.symbol_id,
            "event_ids": list(self.event_ids),
            "memory_ids": list(self.memory_ids),
            "first_round": self.first_round,
            "last_round": self.last_round,
            "support_count": self.support_count,
            "status": self.status.value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSocialEvidence:
    evidence_id: str
    kind: SituatedSocialEvidenceKind
    observer_agent_id: str
    topic_id: str
    symbol_id: str
    round_index: int
    event_id: str
    source_agent_id: str | None = None
    memory_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("evidence_id", "observer_agent_id", "topic_id", "symbol_id", "event_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"situated social evidence {name.replace('_', ' ')}"))
        if not isinstance(self.kind, SituatedSocialEvidenceKind):
            raise TypeError("situated social evidence kind must be SituatedSocialEvidenceKind")
        object.__setattr__(self, "round_index", _positive(self.round_index, label="situated social evidence round"))
        if self.source_agent_id is not None:
            object.__setattr__(self, "source_agent_id", _text(self.source_agent_id, label="situated social evidence source id"))
        if self.memory_id is not None:
            object.__setattr__(self, "memory_id", _text(self.memory_id, label="situated social evidence memory id"))
        if self.kind is SituatedSocialEvidenceKind.TESTIMONY:
            if self.source_agent_id is None or self.source_agent_id == self.observer_agent_id:
                raise ValueError("situated testimony requires a distinct source")
        elif self.source_agent_id is not None:
            raise ValueError("situated verification cannot name a source")

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind.value,
            "observer_agent_id": self.observer_agent_id,
            "topic_id": self.topic_id,
            "symbol_id": self.symbol_id,
            "round_index": self.round_index,
            "event_id": self.event_id,
            "source_agent_id": self.source_agent_id,
            "memory_id": self.memory_id,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSocialMemoryState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    cognitive_state_hash: str
    relationships: tuple[SituatedSourceRelationship, ...]
    claims: tuple[SituatedConsolidatedClaim, ...] = ()
    processed_evidence_ids: tuple[str, ...] = ()
    checkpoint: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated social state model id"))
        object.__setattr__(self, "model_hash", _content_hash(self.model_hash, label="situated social state model hash"))
        object.__setattr__(self, "cognitive_state_hash", _content_hash(self.cognitive_state_hash, label="situated social state cognitive hash"))
        object.__setattr__(self, "round_index", _nonnegative(self.round_index, label="situated social state round"))
        if not isinstance(self.checkpoint, bool):
            raise TypeError("situated social state checkpoint must be boolean")
        if self.checkpoint:
            if self.parent_state_hash is not None:
                raise ValueError("situated social checkpoint cannot have a parent")
        elif self.round_index > 0:
            object.__setattr__(self, "parent_state_hash", _content_hash(self.parent_state_hash, label="situated social parent state hash"))
        elif self.parent_state_hash is not None:
            raise ValueError("round-zero situated social state cannot have a parent")
        if not isinstance(self.relationships, tuple) or any(not isinstance(item, SituatedSourceRelationship) for item in self.relationships):
            raise TypeError("situated social relationships must be a tuple")
        pairs = tuple((item.observer_agent_id, item.source_agent_id) for item in self.relationships)
        if len(set(pairs)) != len(pairs):
            raise ValueError("situated social relationship pairs must be unique")
        object.__setattr__(self, "relationships", tuple(sorted(self.relationships, key=lambda item: (item.observer_agent_id, item.source_agent_id))))
        if not isinstance(self.claims, tuple) or any(not isinstance(item, SituatedConsolidatedClaim) for item in self.claims):
            raise TypeError("situated social claims must be a tuple")
        claim_ids = tuple(item.claim_id for item in self.claims)
        if len(set(claim_ids)) != len(claim_ids):
            raise ValueError("situated social claim ids must be unique")
        object.__setattr__(self, "claims", tuple(sorted(self.claims, key=lambda item: item.claim_id)))
        object.__setattr__(self, "processed_evidence_ids", _strings(self.processed_evidence_ids, label="situated social processed evidence ids"))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash,
            "cognitive_state_hash": self.cognitive_state_hash,
            "relationships": [item.to_dict() for item in self.relationships],
            "claims": [item.to_dict() for item in self.claims],
            "processed_evidence_ids": list(self.processed_evidence_ids),
            "checkpoint": self.checkpoint,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def validate_situated_social_memory_state(
    model: SituatedSocialMemoryModel,
    cognitive_state: SituatedCognitiveState,
    state: SituatedSocialMemoryState,
) -> None:
    if not isinstance(model, SituatedSocialMemoryModel) or not isinstance(cognitive_state, SituatedCognitiveState) or not isinstance(state, SituatedSocialMemoryState):
        raise TypeError("situated social validation requires model, cognitive state, and social state")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("situated social state must bind the exact model")
    if state.round_index != cognitive_state.round_index or state.cognitive_state_hash != cognitive_state.content_hash:
        raise ValueError("situated social state must bind the exact cognitive state")
    agent_ids = {item.agent_id for item in model.memory_cognitive_model.cognitive_model.agents}
    expected_pairs = {(observer, source) for observer in agent_ids for source in agent_ids if observer != source}
    actual_pairs = {(item.observer_agent_id, item.source_agent_id) for item in state.relationships}
    if actual_pairs != expected_pairs:
        raise ValueError("situated social relationships must cover every directed agent pair")
    topic_by_id = {item.topic_id: item for item in model.topics}
    for claim in state.claims:
        if claim.observer_agent_id not in agent_ids or claim.source_agent_id not in agent_ids:
            raise ValueError("situated social claim agents must belong to the model")
        topic = topic_by_id.get(claim.topic_id)
        if topic is None or claim.symbol_id not in topic.symbol_ids:
            raise ValueError("situated social claim must use a declared topic symbol")
        if claim.last_round > state.round_index:
            raise ValueError("situated social claim cannot come from a future round")


def initialize_situated_social_memory(
    model: SituatedSocialMemoryModel,
    cognitive_state: SituatedCognitiveState,
) -> SituatedSocialMemoryState:
    if not isinstance(model, SituatedSocialMemoryModel) or not isinstance(cognitive_state, SituatedCognitiveState):
        raise TypeError("situated social initialization requires model and cognitive state")
    cognition = model.memory_cognitive_model.cognitive_model
    if cognitive_state.model_id != cognition.model_id or cognitive_state.model_hash != cognition.content_hash:
        raise ValueError("situated social model and cognitive state must bind the exact cognition")
    agent_ids = tuple(item.agent_id for item in cognition.agents)
    state = SituatedSocialMemoryState(
        model.model_id,
        model.content_hash,
        cognitive_state.round_index,
        None,
        cognitive_state.content_hash,
        tuple(
            SituatedSourceRelationship(
                observer,
                source,
                model.policy.initial_source_trust,
                0.0,
                0,
                0,
            )
            for observer in agent_ids
            for source in agent_ids
            if observer != source
        ),
        checkpoint=True,
    )
    validate_situated_social_memory_state(model, cognitive_state, state)
    return state


__all__ = (
    "SituatedClaimStatus",
    "SituatedSocialEvidenceKind",
    "SituatedClaimTopic",
    "SituatedSocialMemoryPolicy",
    "SituatedSocialMemoryModel",
    "SituatedSourceRelationship",
    "SituatedConsolidatedClaim",
    "SituatedSocialEvidence",
    "SituatedSocialMemoryState",
    "validate_situated_social_memory_state",
    "initialize_situated_social_memory",
)
