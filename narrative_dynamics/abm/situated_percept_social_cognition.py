"""Exact-percept social evidence over V15.1 private recall."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveState,
)
from narrative_dynamics.abm.situated_memory_cognition import SituatedMemoryRecallResult
from narrative_dynamics.abm.situated_perception_contracts import SituatedPerceptFidelity
from narrative_dynamics.abm.situated_percept_cognition import perceptual_timeline
from narrative_dynamics.abm.situated_percept_memory_cognition import (
    SituatedPerceptMemoryCognitiveModel,
    SituatedPerceptMemoryCognitiveRoundResult,
    _simulate_situated_percept_memory_cognitive_round,
    recall_situated_percept_memories,
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
from narrative_dynamics.abm.situated_story import SituatedStory


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _validate_models(
    memory_model: SituatedPerceptMemoryCognitiveModel,
    social_model: SituatedSocialMemoryModel,
) -> None:
    if not isinstance(memory_model, SituatedPerceptMemoryCognitiveModel):
        raise TypeError("percept social cognition requires a percept memory cognitive model")
    if not isinstance(social_model, SituatedSocialMemoryModel):
        raise TypeError("percept social cognition requires a social memory model")
    if social_model.memory_cognitive_model.cognitive_model != memory_model.cognitive_model:
        raise ValueError("percept and social models must bind the exact cognitive model")


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


def _active_claims(
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


@dataclass(frozen=True)
class SituatedPerceptSocialCognitiveRoundResult:
    percept_memory_model_id: str
    percept_memory_model_hash: str
    social_model_id: str
    social_model_hash: str
    cognitive_round: SituatedPerceptMemoryCognitiveRoundResult
    social_update: SituatedSocialMemoryUpdate

    def __post_init__(self) -> None:
        object.__setattr__(self, "percept_memory_model_id", _text(
            self.percept_memory_model_id, label="percept social memory model id"
        ))
        object.__setattr__(self, "social_model_id", _text(
            self.social_model_id, label="percept social model id"
        ))
        for name in ("percept_memory_model_hash", "social_model_hash"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).startswith("sha256:"):
                raise ValueError(f"{name.replace('_', ' ')} must be a content hash")
        if not isinstance(self.cognitive_round, SituatedPerceptMemoryCognitiveRoundResult):
            raise TypeError("percept social round requires a percept memory cognitive round")
        if not isinstance(self.social_update, SituatedSocialMemoryUpdate):
            raise TypeError("percept social round requires a social update")
        if self.social_update.prior_state.cognitive_state_hash != self.cognitive_round.cognitive_round.prior_state.content_hash:
            raise ValueError("percept social prior cognitive chains must match")
        if self.social_update.next_state.cognitive_state_hash != self.cognitive_round.next_state.content_hash:
            raise ValueError("percept social next cognitive chains must match")

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
            "percept_memory_model_id": self.percept_memory_model_id,
            "percept_memory_model_hash": self.percept_memory_model_hash,
            "social_model_id": self.social_model_id,
            "social_model_hash": self.social_model_hash,
            "cognitive_round": self.cognitive_round.to_dict(),
            "social_update": self.social_update.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptSocialCognitiveTrajectory:
    percept_memory_model_id: str
    percept_memory_model_hash: str
    social_model_id: str
    social_model_hash: str
    initial_story: SituatedStory
    initial_cognitive_state: SituatedCognitiveState
    initial_social_state: SituatedSocialMemoryState
    rounds: tuple[SituatedPerceptSocialCognitiveRoundResult, ...]
    final_story: SituatedStory
    final_cognitive_state: SituatedCognitiveState
    final_social_state: SituatedSocialMemoryState

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("percept social trajectory requires at least one round")
        story, cognition, social = (
            self.initial_story,
            self.initial_cognitive_state,
            self.initial_social_state,
        )
        for item in self.rounds:
            if (
                item.percept_memory_model_id != self.percept_memory_model_id
                or item.percept_memory_model_hash != self.percept_memory_model_hash
                or item.social_model_id != self.social_model_id
                or item.social_model_hash != self.social_model_hash
            ):
                raise ValueError("percept social trajectory must bind exact models")
            if item.cognitive_round.cognitive_round.prior_story_hash != story.content_hash:
                raise ValueError("percept social story rounds must form one chain")
            if item.cognitive_round.cognitive_round.prior_state != cognition or item.social_update.prior_state != social:
                raise ValueError("percept social state rounds must form exact chains")
            story, cognition, social = (
                item.next_story,
                item.next_cognitive_state,
                item.next_social_state,
            )
        if (
            story != self.final_story
            or cognition != self.final_cognitive_state
            or social != self.final_social_state
        ):
            raise ValueError("percept social final values must equal the round chains")

    def to_dict(self) -> dict[str, object]:
        return {
            "percept_memory_model_id": self.percept_memory_model_id,
            "percept_memory_model_hash": self.percept_memory_model_hash,
            "social_model_id": self.social_model_id,
            "social_model_hash": self.social_model_hash,
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


def recall_situated_percept_memories_with_social_trust(
    database_path: str | Path,
    memory_model: SituatedPerceptMemoryCognitiveModel,
    social_model: SituatedSocialMemoryModel,
    agent_model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    *,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedMemoryRecallResult:
    """Recall exact testimony using the observer's current directed trust."""

    _validate_models(memory_model, social_model)
    validate_situated_social_memory_state(social_model, cognitive_state, social_state)
    return recall_situated_percept_memories(
        database_path,
        memory_model,
        agent_model,
        mind,
        story=story,
        state=cognitive_state,
        _source_trust_by_source=_trust_by_observer(social_state).get(mind.agent_id, {}),
        _claim_topic_by_symbol=_topic_by_symbol(social_model),
        _consolidated_claim_keys=_active_claims(social_state).get(mind.agent_id, frozenset()),
    )


def _social_evidence(
    social_model: SituatedSocialMemoryModel,
    story: SituatedStory,
    result: SituatedPerceptMemoryCognitiveRoundResult,
) -> tuple[SituatedSocialEvidence, ...]:
    perception_model = story.perception_model
    if perception_model is None:
        raise ValueError("percept social evidence requires a perception-bound story")
    percepts = {
        agent_id: {
            item.percept_id: item
            for item in perceptual_timeline(
                perception_model,
                story,
                agent_id,
            )
        }
        for agent_id in (item.agent_id for item in social_model.memory_cognitive_model.cognitive_model.agents)
    }
    recall_by_agent = {
        item.prior_mind.agent_id: item for item in result.recalls
    }
    evidence = []

    def append_evidence(observer: str, percept_id: str, symbol_id: str, prefix: str) -> None:
        percept = percepts[observer].get(percept_id)
        if percept is None:
            raise ValueError("social admission must resolve to an authoritative private percept")
        if percept.fidelity is not SituatedPerceptFidelity.EXACT:
            return
        topic = social_model.topic_for_symbol(symbol_id)
        if topic is None:
            return
        if percept.kind is SituatedActionKind.TELL and percept.actor_agent_id != observer:
            kind = SituatedSocialEvidenceKind.TESTIMONY
            source = percept.actor_agent_id
        elif percept.kind is SituatedActionKind.INSPECT and percept.actor_agent_id == observer:
            kind = SituatedSocialEvidenceKind.VERIFICATION
            source = None
        else:
            return
        evidence.append(SituatedSocialEvidence(
            f"{prefix}:{percept.percept_id}",
            kind,
            observer,
            topic.topic_id,
            symbol_id,
            percept.round_index,
            percept.source_event_id,
            source_agent_id=source,
            memory_id=percept.percept_id,
        ))

    for decision in result.decisions:
        for percept_id, symbol_id in zip(
            decision.admitted_observation_ids,
            decision.admitted_symbol_ids,
            strict=True,
        ):
            append_evidence(decision.agent_id, percept_id, symbol_id, "direct-percept")
        for admission in recall_by_agent[decision.agent_id].admissions:
            if admission.symbol_id is not None:
                append_evidence(
                    decision.agent_id,
                    admission.memory_id,
                    admission.symbol_id,
                    "recalled-percept",
                )
    return tuple(sorted(evidence, key=lambda item: (item.round_index, item.evidence_id)))


def simulate_situated_percept_social_cognitive_round(
    database_path: str | Path,
    memory_model: SituatedPerceptMemoryCognitiveModel,
    social_model: SituatedSocialMemoryModel,
    story: SituatedStory,
    cognitive_state: SituatedCognitiveState,
    social_state: SituatedSocialMemoryState,
) -> SituatedPerceptSocialCognitiveRoundResult:
    """Advance private percept cognition, then exact-only social memory."""

    _validate_models(memory_model, social_model)
    validate_situated_social_memory_state(social_model, cognitive_state, social_state)
    if story.perception_model != memory_model.perception_model:
        raise ValueError("percept social story must bind the exact perception model")
    cognitive_round = _simulate_situated_percept_memory_cognitive_round(
        database_path,
        memory_model,
        story,
        cognitive_state,
        source_trust_by_observer=_trust_by_observer(social_state),
        claim_topic_by_symbol=_topic_by_symbol(social_model),
        consolidated_claim_keys_by_observer=_active_claims(social_state),
    )
    evidence = _social_evidence(social_model, story, cognitive_round)
    social_update = advance_situated_social_memory(
        social_model,
        cognitive_state,
        social_state,
        cognitive_round.next_state,
        evidence,
    )
    return SituatedPerceptSocialCognitiveRoundResult(
        memory_model.model_id,
        memory_model.content_hash,
        social_model.model_id,
        social_model.content_hash,
        cognitive_round,
        social_update,
    )


def simulate_situated_percept_social_cognition(
    database_path: str | Path,
    memory_model: SituatedPerceptMemoryCognitiveModel,
    social_model: SituatedSocialMemoryModel,
    initial_story: SituatedStory,
    initial_cognitive_state: SituatedCognitiveState,
    initial_social_state: SituatedSocialMemoryState,
    *,
    round_count: int,
) -> SituatedPerceptSocialCognitiveTrajectory:
    if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count <= 0:
        raise ValueError("percept social simulation requires a positive round count")
    _validate_models(memory_model, social_model)
    story, cognition, social = (
        initial_story,
        initial_cognitive_state,
        initial_social_state,
    )
    rounds = []
    for _ in range(round_count):
        item = simulate_situated_percept_social_cognitive_round(
            database_path,
            memory_model,
            social_model,
            story,
            cognition,
            social,
        )
        rounds.append(item)
        story, cognition, social = (
            item.next_story,
            item.next_cognitive_state,
            item.next_social_state,
        )
    return SituatedPerceptSocialCognitiveTrajectory(
        memory_model.model_id,
        memory_model.content_hash,
        social_model.model_id,
        social_model.content_hash,
        initial_story,
        initial_cognitive_state,
        initial_social_state,
        tuple(rounds),
        story,
        cognition,
        social,
    )


__all__ = (
    "SituatedPerceptSocialCognitiveRoundResult",
    "SituatedPerceptSocialCognitiveTrajectory",
    "recall_situated_percept_memories_with_social_trust",
    "simulate_situated_percept_social_cognitive_round",
    "simulate_situated_percept_social_cognition",
)
