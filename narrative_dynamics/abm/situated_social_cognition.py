"""V14 orchestration joining trusted recall to social-memory evolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_cognition import match_situated_observation_rule
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveState,
)
from narrative_dynamics.abm.situated_memory_cognition import (
    SituatedMemoryCognitiveRoundResult,
    SituatedMemoryRecallResult,
    _simulate_situated_memory_cognitive_round,
    recall_situated_memories,
)
from narrative_dynamics.abm.situated_social_memory import (
    SituatedSocialMemoryUpdate,
    advance_situated_social_memory,
)
from narrative_dynamics.abm.situated_social_memory_contracts import (
    SituatedClaimStatus,
    SituatedSocialEvidence,
    SituatedSocialEvidenceKind,
    SituatedSocialMemoryModel,
    SituatedSocialMemoryState,
    validate_situated_social_memory_state,
)
from narrative_dynamics.abm.situated_story import SituatedStory, perspective_timeline


@dataclass(frozen=True)
class SituatedSocialCognitiveRoundResult:
    model_id: str
    model_hash: str
    cognitive_round: SituatedMemoryCognitiveRoundResult
    social_update: SituatedSocialMemoryUpdate

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("situated social cognitive round model id must be non-empty")
        if not isinstance(self.model_hash, str) or not self.model_hash.startswith("sha256:"):
            raise ValueError("situated social cognitive round model hash must be a content hash")
        if not isinstance(self.cognitive_round, SituatedMemoryCognitiveRoundResult):
            raise TypeError("situated social cognitive round requires a memory cognitive round")
        if not isinstance(self.social_update, SituatedSocialMemoryUpdate):
            raise TypeError("situated social cognitive round requires a social update")
        if self.social_update.prior_state.cognitive_state_hash != self.cognitive_round.prior_state.content_hash:
            raise ValueError("situated social round prior chains must match")
        if self.social_update.next_state.cognitive_state_hash != self.cognitive_round.next_state.content_hash:
            raise ValueError("situated social round next chains must match")

    @property
    def decisions(self):
        return self.cognitive_round.decisions

    @property
    def recalls(self):
        return self.cognitive_round.recalls

    @property
    def next_story(self) -> SituatedStory:
        return self.cognitive_round.next_story

    @property
    def next_cognitive_state(self) -> SituatedCognitiveState:
        return self.cognitive_round.next_state

    @property
    def next_social_state(self) -> SituatedSocialMemoryState:
        return self.social_update.next_state

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "cognitive_round": self.cognitive_round.to_dict(),
            "social_update": self.social_update.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedSocialCognitiveTrajectory:
    model_id: str
    model_hash: str
    initial_story: SituatedStory
    initial_cognitive_state: SituatedCognitiveState
    initial_social_state: SituatedSocialMemoryState
    rounds: tuple[SituatedSocialCognitiveRoundResult, ...]
    final_story: SituatedStory
    final_cognitive_state: SituatedCognitiveState
    final_social_state: SituatedSocialMemoryState

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("situated social cognitive trajectory requires at least one round")
        story = self.initial_story
        cognition = self.initial_cognitive_state
        social = self.initial_social_state
        for item in self.rounds:
            if item.model_id != self.model_id or item.model_hash != self.model_hash:
                raise ValueError("situated social trajectory rounds must bind the exact model")
            if item.cognitive_round.cognitive_round.prior_story_hash != story.content_hash:
                raise ValueError("situated social trajectory story rounds must form one chain")
            if item.cognitive_round.prior_state != cognition or item.social_update.prior_state != social:
                raise ValueError("situated social trajectory state rounds must form exact chains")
            story = item.next_story
            cognition = item.next_cognitive_state
            social = item.next_social_state
        if story != self.final_story or cognition != self.final_cognitive_state or social != self.final_social_state:
            raise ValueError("situated social trajectory final values must equal the round chains")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_story_hash": self.initial_story.content_hash,
            "initial_cognitive_state": self.initial_cognitive_state.to_dict(),
            "initial_social_state": self.initial_social_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "final_story_hash": self.final_story.content_hash,
            "final_cognitive_state": self.final_cognitive_state.to_dict(),
            "final_social_state": self.final_social_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _trust_by_observer(
    state: SituatedSocialMemoryState,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for item in state.relationships:
        result.setdefault(item.observer_agent_id, {})[item.source_agent_id] = item.trust
    return result


def _topic_by_symbol(model: SituatedSocialMemoryModel) -> dict[str, str]:
    return {
        symbol_id: topic.topic_id
        for topic in model.topics
        for symbol_id in topic.symbol_ids
    }


def _active_claim_keys_by_observer(
    state: SituatedSocialMemoryState,
) -> dict[str, frozenset[tuple[str, str, str]]]:
    result: dict[str, set[tuple[str, str, str]]] = {}
    for claim in state.claims:
        if claim.status is SituatedClaimStatus.ACTIVE:
            result.setdefault(claim.observer_agent_id, set()).add((
                claim.source_agent_id,
                claim.topic_id,
                claim.symbol_id,
            ))
    return {key: frozenset(value) for key, value in result.items()}


def recall_situated_memories_with_social_trust(
    database_path: str | Path,
    model: SituatedSocialMemoryModel,
    agent_model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    *,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedMemoryRecallResult:
    """Recall testimony with the observer's current directed source trust."""

    validate_situated_social_memory_state(model, cognitive_state, social_state)
    trust = _trust_by_observer(social_state).get(mind.agent_id, {})
    claim_keys = _active_claim_keys_by_observer(social_state).get(mind.agent_id, frozenset())
    return recall_situated_memories(
        database_path,
        model.memory_cognitive_model,
        agent_model,
        mind,
        story=story,
        state=cognitive_state,
        _source_trust_by_source=trust,
        _claim_topic_by_symbol=_topic_by_symbol(model),
        _consolidated_claim_keys=claim_keys,
    )


def _social_evidence(
    model: SituatedSocialMemoryModel,
    story: SituatedStory,
    result: SituatedMemoryCognitiveRoundResult,
) -> tuple[SituatedSocialEvidence, ...]:
    agent_model = {
        item.agent_id: item
        for item in model.memory_cognitive_model.cognitive_model.agents
    }
    recall_by_agent = {
        item.prior_mind.agent_id: item
        for item in result.recalls
    }
    evidence = []
    for decision in result.decisions:
        observer = decision.agent_id
        perspective = perspective_timeline(story, observer)
        by_observation = {item.observation.observation_id: item for item in perspective}
        for observation_id, symbol_id in zip(
            decision.admitted_observation_ids,
            decision.admitted_symbol_ids,
            strict=True,
        ):
            item = by_observation[observation_id]
            topic = model.topic_for_symbol(symbol_id)
            if topic is None:
                continue
            testimony = (
                item.event.kind is SituatedActionKind.TELL
                and item.event.actor_agent_id != observer
            )
            evidence.append(SituatedSocialEvidence(
                f"direct:{observation_id}",
                SituatedSocialEvidenceKind.TESTIMONY if testimony else SituatedSocialEvidenceKind.VERIFICATION,
                observer,
                topic.topic_id,
                symbol_id,
                item.event.round_index,
                item.event.event_id,
                source_agent_id=item.event.actor_agent_id if testimony else None,
                memory_id=observation_id,
            ))
        for admission in recall_by_agent[observer].admissions:
            if admission.symbol_id is None:
                continue
            topic = model.topic_for_symbol(admission.symbol_id)
            if topic is None:
                continue
            item = by_observation[admission.memory_id]
            testimony = (
                item.event.kind is SituatedActionKind.TELL
                and item.event.actor_agent_id != observer
            )
            evidence.append(SituatedSocialEvidence(
                f"recall:{admission.memory_id}",
                SituatedSocialEvidenceKind.TESTIMONY if testimony else SituatedSocialEvidenceKind.VERIFICATION,
                observer,
                topic.topic_id,
                admission.symbol_id,
                item.event.round_index,
                item.event.event_id,
                source_agent_id=item.event.actor_agent_id if testimony else None,
                memory_id=admission.memory_id,
            ))
    return tuple(sorted(evidence, key=lambda item: (item.round_index, item.evidence_id)))


def simulate_situated_social_cognitive_round(
    database_path: str | Path,
    model: SituatedSocialMemoryModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedSocialCognitiveRoundResult:
    """Advance cognition with current trust, then evolve the social-memory state."""

    validate_situated_social_memory_state(model, cognitive_state, social_state)
    cognitive_round = _simulate_situated_memory_cognitive_round(
        database_path,
        model.memory_cognitive_model,
        story,
        cognitive_state,
        source_trust_by_observer=_trust_by_observer(social_state),
        claim_topic_by_symbol=_topic_by_symbol(model),
        consolidated_claim_keys_by_observer=_active_claim_keys_by_observer(social_state),
    )
    evidence = _social_evidence(model, story, cognitive_round)
    social_update = advance_situated_social_memory(
        model,
        cognitive_state,
        social_state,
        cognitive_round.next_state,
        evidence,
    )
    return SituatedSocialCognitiveRoundResult(
        model.model_id,
        model.content_hash,
        cognitive_round,
        social_update,
    )


def simulate_situated_social_cognition(
    database_path: str | Path,
    model: SituatedSocialMemoryModel,
    initial_story: SituatedStory,
    initial_cognitive_state: SituatedCognitiveState,
    initial_social_state: SituatedSocialMemoryState,
    *,
    round_count: int,
) -> SituatedSocialCognitiveTrajectory:
    """Run a deterministic V14 physical, cognitive, and social trajectory."""

    if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count <= 0:
        raise ValueError("situated social cognitive simulation requires a positive round count")
    validate_situated_social_memory_state(model, initial_cognitive_state, initial_social_state)
    story = initial_story
    cognition = initial_cognitive_state
    social = initial_social_state
    rounds = []
    for _ in range(round_count):
        result = simulate_situated_social_cognitive_round(
            database_path,
            model,
            story,
            cognition,
            social,
        )
        rounds.append(result)
        story = result.next_story
        cognition = result.next_cognitive_state
        social = result.next_social_state
    return SituatedSocialCognitiveTrajectory(
        model.model_id,
        model.content_hash,
        initial_story,
        initial_cognitive_state,
        initial_social_state,
        tuple(rounds),
        story,
        cognition,
        social,
    )


__all__ = (
    "SituatedSocialCognitiveRoundResult",
    "SituatedSocialCognitiveTrajectory",
    "recall_situated_memories_with_social_trust",
    "simulate_situated_social_cognitive_round",
    "simulate_situated_social_cognition",
)
