"""Private SQLite recall for V13 memory-augmented situated cognition."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import SituatedObservation, SituatedWorldEvent
from narrative_dynamics.abm.situated_cognition import match_situated_observation_rule
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
)
from narrative_dynamics.abm.situated_memory import search_situated_memories
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_memory_contracts import (
    SituatedMemoryQuery,
    SituatedMemoryRecord,
)
from narrative_dynamics.abm.situated_story import SituatedPerspectiveEvent


@dataclass(frozen=True)
class SituatedMemoryRecallAdmission:
    agent_id: str
    cue_id: str
    memory_id: str
    event_id: str
    rule_id: str | None
    symbol_id: str | None
    likelihood_action_id: str | None
    confidence: float
    salience: float
    evidence_weight: float
    lexical_rank: float | None
    prior_belief: PlanningBeliefState
    posterior_belief: PlanningBeliefState

    def __post_init__(self) -> None:
        for name in ("agent_id", "cue_id", "memory_id", "event_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"situated recall {name.replace('_', ' ')} must be non-empty")
        matched = (self.rule_id, self.symbol_id, self.likelihood_action_id)
        if any(item is None for item in matched) and any(item is not None for item in matched):
            raise ValueError("situated recall cognitive match fields must be all present or all absent")
        for name in ("confidence", "salience", "evidence_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"situated recall {name.replace('_', ' ')} must be in [0, 1]")
        if not math.isclose(self.evidence_weight, self.confidence * self.salience, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("situated recall evidence weight must equal confidence times salience")
        if self.lexical_rank is not None and (
            isinstance(self.lexical_rank, bool) or not isinstance(self.lexical_rank, (int, float))
        ):
            raise TypeError("situated recall lexical rank must be numeric")
        if not isinstance(self.prior_belief, PlanningBeliefState) or not isinstance(self.posterior_belief, PlanningBeliefState):
            raise TypeError("situated recall admission requires planning beliefs")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "cue_id": self.cue_id,
            "memory_id": self.memory_id,
            "event_id": self.event_id,
            "rule_id": self.rule_id,
            "symbol_id": self.symbol_id,
            "likelihood_action_id": self.likelihood_action_id,
            "confidence": self.confidence,
            "salience": self.salience,
            "evidence_weight": self.evidence_weight,
            "lexical_rank": self.lexical_rank,
            "prior_belief": self.prior_belief.to_dict(),
            "posterior_belief": self.posterior_belief.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedMemoryRecallResult:
    prior_mind: SituatedAgentMindState
    admissions: tuple[SituatedMemoryRecallAdmission, ...]
    next_mind: SituatedAgentMindState

    def __post_init__(self) -> None:
        if not isinstance(self.prior_mind, SituatedAgentMindState) or not isinstance(self.next_mind, SituatedAgentMindState):
            raise TypeError("situated recall result requires mind states")
        if self.prior_mind.agent_id != self.next_mind.agent_id:
            raise ValueError("situated recall result mind agent must remain stable")
        if not isinstance(self.admissions, tuple) or any(
            not isinstance(item, SituatedMemoryRecallAdmission) for item in self.admissions
        ):
            raise TypeError("situated recall admissions must be a tuple")
        if any(item.agent_id != self.prior_mind.agent_id for item in self.admissions):
            raise ValueError("situated recall admissions must belong to the mind agent")

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_mind_hash": self.prior_mind.content_hash,
            "admissions": [item.to_dict() for item in self.admissions],
            "next_mind": self.next_mind.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _memory_perspective(memory: SituatedMemoryRecord) -> SituatedPerspectiveEvent:
    event = SituatedWorldEvent(
        memory.event_id,
        memory.round_index,
        memory.sequence,
        memory.action_id,
        memory.kind,
        memory.actor_agent_id,
        memory.place_id,
        memory.target_id,
        memory.success,
        memory.outcome,
        memory.details,
        memory.cause_event_ids,
    )
    observation = SituatedObservation(
        memory.observation_id,
        memory.round_index,
        memory.agent_id,
        memory.event_id,
        memory.event_hash,
        memory.channel,
    )
    return SituatedPerspectiveEvent(event, observation)


def _posterior_from_memory(
    model: SituatedAgentCognitiveModel,
    belief: PlanningBeliefState,
    likelihood_action_id: str,
    symbol_id: str,
    weight: float,
) -> PlanningBeliefState:
    likelihoods = {
        (item.action_id, item.hypothesis_id, item.symbol_id): item.probability
        for item in model.likelihoods
    }
    masses = {
        hypothesis_id: probability * (
            (1.0 - weight)
            + weight * likelihoods[(likelihood_action_id, hypothesis_id, symbol_id)]
        )
        for hypothesis_id, probability in belief.probabilities.items()
    }
    evidence = math.fsum(masses.values())
    if not math.isfinite(evidence) or evidence <= 0.0:
        raise ValueError("recalled observation has zero probability under current belief")
    return PlanningBeliefState({key: value / evidence for key, value in masses.items()})


def recall_situated_memories(
    database_path: str | Path,
    model: SituatedMemoryCognitiveModel,
    agent_model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    *,
    round_index: int,
) -> SituatedMemoryRecallResult:
    """Recall relevant private memories once and temper their Bayesian evidence."""

    if not isinstance(model, SituatedMemoryCognitiveModel):
        raise TypeError("situated recall requires a SituatedMemoryCognitiveModel")
    if not isinstance(agent_model, SituatedAgentCognitiveModel) or not isinstance(mind, SituatedAgentMindState):
        raise TypeError("situated recall requires an agent model and mind")
    if agent_model.agent_id != mind.agent_id:
        raise ValueError("situated recall agent model and mind must match")
    expected_agent = next(
        (item for item in model.cognitive_model.agents if item.agent_id == mind.agent_id),
        None,
    )
    if expected_agent != agent_model:
        raise ValueError("situated recall agent model must belong to the exact memory cognitive model")
    if not isinstance(round_index, int) or isinstance(round_index, bool) or round_index < 0:
        raise ValueError("situated recall round index must be non-negative")
    if round_index < mind.observation_floor_round:
        raise ValueError("situated recall cannot precede the mind observation floor")

    policy = next(item for item in model.agents if item.agent_id == mind.agent_id)
    if round_index == 0 or not policy.cues:
        return SituatedMemoryRecallResult(mind, (), mind)

    excluded_observations = set(mind.processed_observation_ids)
    recalled = set(mind.recalled_memory_ids)
    candidates = []
    seen_observations = set()
    for cue in policy.cues:
        if cue.required_place_ids and mind.own_place_id not in cue.required_place_ids:
            continue
        query = SituatedMemoryQuery(
            agent_id=mind.agent_id,
            text=cue.text,
            event_kinds=cue.event_kinds,
            channels=cue.channels,
            max_round=round_index,
            min_confidence=cue.min_confidence,
            limit=cue.limit,
            story_model_hash=model.cognitive_model.world_model.content_hash,
        )
        for hit in search_situated_memories(database_path, query):
            memory = hit.memory
            if (
                memory.observation_id in excluded_observations
                or memory.memory_id in recalled
                or memory.observation_id in seen_observations
            ):
                continue
            seen_observations.add(memory.observation_id)
            candidates.append((cue, hit))
            if len(candidates) >= policy.max_memories_per_round:
                break
        if len(candidates) >= policy.max_memories_per_round:
            break

    belief = mind.belief
    observed = set(mind.observed_event_ids)
    admissions = []
    for cue, hit in candidates:
        memory = hit.memory
        perspective = _memory_perspective(memory)
        rule = match_situated_observation_rule(agent_model, perspective)
        prior = belief
        weight = memory.confidence * memory.salience
        if rule is None:
            posterior = prior
            rule_id = symbol_id = likelihood_action_id = None
        else:
            posterior = _posterior_from_memory(
                agent_model,
                prior,
                rule.likelihood_action_id,
                rule.symbol_id,
                weight,
            )
            rule_id = rule.rule_id
            symbol_id = rule.symbol_id
            likelihood_action_id = rule.likelihood_action_id
        admissions.append(SituatedMemoryRecallAdmission(
            mind.agent_id,
            cue.cue_id,
            memory.memory_id,
            memory.event_id,
            rule_id,
            symbol_id,
            likelihood_action_id,
            memory.confidence,
            memory.salience,
            weight,
            hit.lexical_rank,
            prior,
            posterior,
        ))
        belief = posterior
        recalled.add(memory.memory_id)
        observed.add(memory.event_id)

    next_mind = replace(
        mind,
        belief=belief,
        recalled_memory_ids=tuple(recalled),
        observed_event_ids=tuple(observed),
    )
    return SituatedMemoryRecallResult(mind, tuple(admissions), next_mind)


__all__ = (
    "SituatedMemoryRecallAdmission",
    "SituatedMemoryRecallResult",
    "recall_situated_memories",
)
