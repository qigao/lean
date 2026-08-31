"""Private SQLite recall for V13 memory-augmented situated cognition."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from collections.abc import Mapping

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import (
    SituatedActionKind,
    SituatedObservation,
    SituatedWorldEvent,
)
from narrative_dynamics.abm.situated_cognition import (
    SituatedCognitiveRoundResult,
    _advance_situated_cognitive_round,
    admit_situated_observations,
    decide_situated_action,
    match_situated_observation_rule,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_memory import (
    ingest_situated_story,
    search_situated_memories,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedMemoryCognitiveModel,
)
from narrative_dynamics.abm.situated_memory_contracts import (
    SituatedMemoryQuery,
    SituatedMemoryRecord,
)
from narrative_dynamics.abm.situated_story import (
    SituatedPerspectiveEvent,
    SituatedStory,
    perspective_timeline,
)


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
    source_trust: float = 1.0
    consolidated: bool = False

    def __post_init__(self) -> None:
        for name in ("agent_id", "cue_id", "memory_id", "event_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"situated recall {name.replace('_', ' ')} must be non-empty")
        matched = (self.rule_id, self.symbol_id, self.likelihood_action_id)
        if any(item is None for item in matched) and any(item is not None for item in matched):
            raise ValueError("situated recall cognitive match fields must be all present or all absent")
        for name in ("confidence", "salience", "evidence_weight", "source_trust"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"situated recall {name.replace('_', ' ')} must be in [0, 1]")
        if not math.isclose(
            self.evidence_weight,
            self.confidence * self.salience * self.source_trust,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("situated recall evidence weight must include confidence, salience, and source trust")
        if self.lexical_rank is not None and (
            isinstance(self.lexical_rank, bool) or not isinstance(self.lexical_rank, (int, float))
        ):
            raise TypeError("situated recall lexical rank must be numeric")
        if not isinstance(self.prior_belief, PlanningBeliefState) or not isinstance(self.posterior_belief, PlanningBeliefState):
            raise TypeError("situated recall admission requires planning beliefs")
        if not isinstance(self.consolidated, bool):
            raise TypeError("situated recall consolidated flag must be boolean")
        if self.consolidated and self.prior_belief != self.posterior_belief:
            raise ValueError("consolidated recall cannot update belief again")

    def to_dict(self) -> dict[str, object]:
        payload = {
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
        if self.source_trust != 1.0:
            payload["source_trust"] = self.source_trust
        if self.consolidated:
            payload["consolidated"] = True
        return payload

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


@dataclass(frozen=True)
class SituatedMemoryCognitiveRoundResult:
    model_id: str
    model_hash: str
    cognitive_round: SituatedCognitiveRoundResult
    recalls: tuple[SituatedMemoryRecallResult, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("memory cognitive round model id must be non-empty")
        if not isinstance(self.model_hash, str) or not self.model_hash.startswith("sha256:"):
            raise ValueError("memory cognitive round model hash must be a content hash")
        if not isinstance(self.cognitive_round, SituatedCognitiveRoundResult):
            raise TypeError("memory cognitive round requires a cognitive round")
        if not isinstance(self.recalls, tuple) or any(
            not isinstance(item, SituatedMemoryRecallResult) for item in self.recalls
        ):
            raise TypeError("memory cognitive round recalls must be a tuple")
        agents = tuple(item.prior_mind.agent_id for item in self.recalls)
        expected = tuple(item.agent_id for item in self.cognitive_round.decisions)
        if set(agents) != set(expected) or len(agents) != len(set(agents)):
            raise ValueError("memory cognitive round recalls must cover exact decision roster")
        object.__setattr__(self, "recalls", tuple(sorted(self.recalls, key=lambda item: item.prior_mind.agent_id)))

    @property
    def prior_state(self) -> SituatedCognitiveState:
        return self.cognitive_round.prior_state

    @property
    def decisions(self):
        return self.cognitive_round.decisions

    @property
    def next_story(self) -> SituatedStory:
        return self.cognitive_round.next_story

    @property
    def next_state(self) -> SituatedCognitiveState:
        return self.cognitive_round.next_state

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "cognitive_round": self.cognitive_round.to_dict(),
            "recalls": [item.to_dict() for item in self.recalls],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedMemoryCognitiveTrajectory:
    model_id: str
    model_hash: str
    initial_story: SituatedStory
    initial_state: SituatedCognitiveState
    rounds: tuple[SituatedMemoryCognitiveRoundResult, ...]
    final_story: SituatedStory
    final_state: SituatedCognitiveState

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("memory cognitive trajectory requires at least one round")
        story = self.initial_story
        state = self.initial_state
        for item in self.rounds:
            if item.model_id != self.model_id or item.model_hash != self.model_hash:
                raise ValueError("memory cognitive trajectory rounds must bind the exact model")
            if item.cognitive_round.prior_story_hash != story.content_hash or item.prior_state != state:
                raise ValueError("memory cognitive trajectory rounds must form one exact chain")
            story = item.next_story
            state = item.next_state
        if story != self.final_story or state != self.final_state:
            raise ValueError("memory cognitive trajectory final values must equal the round chain")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "initial_story_hash": self.initial_story.content_hash,
            "initial_state": self.initial_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "final_story_hash": self.final_story.content_hash,
            "final_state": self.final_state.to_dict(),
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
    story: SituatedStory,
    state: SituatedCognitiveState,
    _source_trust_by_source: Mapping[str, float] | None = None,
    _claim_topic_by_symbol: Mapping[str, str] | None = None,
    _consolidated_claim_keys: frozenset[tuple[str, str, str]] | None = None,
    _allowed_observation_hashes: Mapping[str, str] | None = None,
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
    validate_situated_cognitive_state(model.cognitive_model, story, state)
    state_mind = next(item for item in state.minds if item.agent_id == mind.agent_id)
    if (
        mind.own_place_id != state_mind.own_place_id
        or mind.observation_floor_round != state_mind.observation_floor_round
        or mind.selected_action_ids != state_mind.selected_action_ids
        or mind.decision_count != state_mind.decision_count
    ):
        raise ValueError("situated recall mind must belong to the current cognitive state")
    round_index = state.round_index
    source_trust_by_source = {} if _source_trust_by_source is None else dict(_source_trust_by_source)
    claim_topic_by_symbol = {} if _claim_topic_by_symbol is None else dict(_claim_topic_by_symbol)
    consolidated_claim_keys = set(() if _consolidated_claim_keys is None else _consolidated_claim_keys)
    active_claim_symbols = {}
    for source_id, topic_id, symbol_id in consolidated_claim_keys:
        scope = (source_id, topic_id)
        existing = active_claim_symbols.get(scope)
        if existing is not None and existing != symbol_id:
            raise ValueError("situated recall claim scope has multiple active symbols")
        active_claim_symbols[scope] = symbol_id
    allowed_observation_hashes = (
        None
        if _allowed_observation_hashes is None
        else dict(_allowed_observation_hashes)
    )
    agent_ids = {item.agent_id for item in model.cognitive_model.agents}
    for source_id, trust in source_trust_by_source.items():
        if source_id not in agent_ids:
            raise ValueError("situated recall source trust agent must belong to the model")
        if isinstance(trust, bool) or not isinstance(trust, (int, float)) or not 0.0 <= trust <= 1.0:
            raise ValueError("situated recall source trust must be in [0, 1]")

    policy = next(item for item in model.agents if item.agent_id == mind.agent_id)
    if round_index == 0 or not policy.cues or allowed_observation_hashes == {}:
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
            excluded_memory_ids=tuple(
                excluded_observations | recalled | seen_observations
            ),
            included_memory_ids=(
                ()
                if allowed_observation_hashes is None
                else tuple(allowed_observation_hashes)
            ),
        )
        for hit in search_situated_memories(database_path, query):
            memory = hit.memory
            if (
                memory.observation_id in excluded_observations
                or memory.memory_id in recalled
                or memory.observation_id in seen_observations
                or (
                    allowed_observation_hashes is not None
                    and allowed_observation_hashes.get(memory.observation_id)
                    != memory.event_hash
                )
            ):
                continue
            seen_observations.add(memory.observation_id)
            candidates.append((cue, hit))
            if len(candidates) >= policy.max_memories_per_round:
                break
        if len(candidates) >= policy.max_memories_per_round:
            break

    if claim_topic_by_symbol:
        candidates.sort(key=lambda item: (
            item[1].memory.round_index,
            item[1].memory.sequence,
            item[1].memory.memory_id,
        ))

    belief = mind.belief
    observed = set(mind.observed_event_ids)
    admissions = []
    for cue, hit in candidates:
        memory = hit.memory
        perspective = _memory_perspective(memory)
        rule = match_situated_observation_rule(agent_model, perspective)
        prior = belief
        source_trust = (
            source_trust_by_source.get(memory.actor_agent_id, 1.0)
            if memory.kind is SituatedActionKind.TELL and memory.actor_agent_id != mind.agent_id
            else 1.0
        )
        weight = memory.confidence * memory.salience * source_trust
        claim_scope = None
        if (
            rule is not None
            and memory.kind is SituatedActionKind.TELL
            and memory.actor_agent_id != mind.agent_id
            and rule.symbol_id in claim_topic_by_symbol
        ):
            claim_scope = (
                memory.actor_agent_id,
                claim_topic_by_symbol[rule.symbol_id],
            )
        consolidated = (
            claim_scope is not None
            and active_claim_symbols.get(claim_scope) == rule.symbol_id
        )
        if rule is None:
            posterior = prior
            rule_id = symbol_id = likelihood_action_id = None
        elif consolidated:
            posterior = prior
            rule_id = rule.rule_id
            symbol_id = rule.symbol_id
            likelihood_action_id = rule.likelihood_action_id
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
            if claim_scope is not None:
                active_claim_symbols[claim_scope] = rule.symbol_id
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
            source_trust,
            consolidated,
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


def simulate_situated_memory_cognitive_round(
    database_path: str | Path,
    model: SituatedMemoryCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
) -> SituatedMemoryCognitiveRoundResult:
    """Advance one synchronous round after direct admission and private recall."""

    return _simulate_situated_memory_cognitive_round(
        database_path,
        model,
        story,
        state,
        source_trust_by_observer=None,
    )


def _simulate_situated_memory_cognitive_round(
    database_path: str | Path,
    model: SituatedMemoryCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
    *,
    source_trust_by_observer: Mapping[str, Mapping[str, float]] | None,
    claim_topic_by_symbol: Mapping[str, str] | None = None,
    consolidated_claim_keys_by_observer: Mapping[str, frozenset[tuple[str, str, str]]] | None = None,
    allowed_observation_hashes_by_observer: Mapping[str, Mapping[str, str]] | None = None,
) -> SituatedMemoryCognitiveRoundResult:
    """Internal V13 round hook used by V14 directed source trust."""

    if not isinstance(model, SituatedMemoryCognitiveModel):
        raise TypeError("memory cognitive simulation requires a SituatedMemoryCognitiveModel")
    cognition = model.cognitive_model
    validate_situated_cognitive_state(cognition, story, state)
    model_by_id = {item.agent_id: item for item in cognition.agents}
    mind_by_id = {item.agent_id: item for item in state.minds}
    next_round = state.round_index + 1

    for agent_id in sorted(model_by_id):
        ingest_situated_story(database_path, story, agent_id, model.memory_policy)

    admitted_minds = {}
    decisions = []
    recalls = []
    for agent_id in sorted(model_by_id):
        private = perspective_timeline(story, agent_id)
        direct_perspective = tuple(
            item
            for item in private
            if item.event.round_index > mind_by_id[agent_id].observation_floor_round
        )
        claim_keys = set(
            ()
            if consolidated_claim_keys_by_observer is None
            else consolidated_claim_keys_by_observer.get(agent_id, frozenset())
        )
        direct = admit_situated_observations(
            model_by_id[agent_id],
            mind_by_id[agent_id],
            direct_perspective,
            _claim_topic_by_symbol=claim_topic_by_symbol,
            _consolidated_claim_keys=frozenset(claim_keys),
        )
        direct_by_observation = {
            item.observation.observation_id: item
            for item in direct_perspective
        }
        for admission in direct.admissions:
            item = direct_by_observation[admission.observation_id]
            if (
                item.event.kind is SituatedActionKind.TELL
                and item.event.actor_agent_id != agent_id
                and claim_topic_by_symbol is not None
                and admission.symbol_id in claim_topic_by_symbol
            ):
                scope = (item.event.actor_agent_id, claim_topic_by_symbol[admission.symbol_id])
                claim_keys = {
                    key for key in claim_keys if key[:2] != scope
                }
                claim_keys.add((
                    item.event.actor_agent_id,
                    claim_topic_by_symbol[admission.symbol_id],
                    admission.symbol_id,
                ))
        recalled = recall_situated_memories(
            database_path,
            model,
            model_by_id[agent_id],
            direct.next_mind,
            story=story,
            state=state,
            _source_trust_by_source=(
                None
                if source_trust_by_observer is None
                else source_trust_by_observer.get(agent_id, {})
            ),
            _claim_topic_by_symbol=claim_topic_by_symbol,
            _consolidated_claim_keys=frozenset(claim_keys),
            _allowed_observation_hashes=(
                None
                if allowed_observation_hashes_by_observer is None
                else allowed_observation_hashes_by_observer.get(agent_id, {})
            ),
        )
        recalls.append(recalled)
        admitted_minds[agent_id] = recalled.next_mind
        decisions.append(decide_situated_action(
            model_by_id[agent_id],
            recalled.next_mind,
            private,
            round_index=next_round,
            prior_belief=direct.prior_mind.belief,
            admissions=direct.admissions,
            recalled_memory_ids=tuple(item.memory_id for item in recalled.admissions),
            recalled_symbol_ids=tuple(dict.fromkeys(
                item.symbol_id for item in recalled.admissions
                if item.symbol_id is not None and not item.consolidated
            )),
        ))

    cognitive_round = _advance_situated_cognitive_round(
        cognition,
        story,
        state,
        tuple(decisions),
        admitted_minds,
    )
    return SituatedMemoryCognitiveRoundResult(
        model.model_id,
        model.content_hash,
        cognitive_round,
        tuple(recalls),
    )


def simulate_situated_memory_cognition(
    database_path: str | Path,
    model: SituatedMemoryCognitiveModel,
    initial_story: SituatedStory,
    initial_state: SituatedCognitiveState,
    *,
    round_count: int,
) -> SituatedMemoryCognitiveTrajectory:
    """Run a deterministic memory-augmented trajectory over one private database."""

    if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count <= 0:
        raise ValueError("memory cognitive simulation requires a positive round count")
    validate_situated_cognitive_state(model.cognitive_model, initial_story, initial_state)
    story = initial_story
    state = initial_state
    rounds = []
    for _ in range(round_count):
        result = simulate_situated_memory_cognitive_round(
            database_path,
            model,
            story,
            state,
        )
        rounds.append(result)
        story = result.next_story
        state = result.next_state
    return SituatedMemoryCognitiveTrajectory(
        model.model_id,
        model.content_hash,
        initial_story,
        initial_state,
        tuple(rounds),
        story,
        state,
    )


__all__ = (
    "SituatedMemoryRecallAdmission",
    "SituatedMemoryRecallResult",
    "SituatedMemoryCognitiveRoundResult",
    "SituatedMemoryCognitiveTrajectory",
    "recall_situated_memories",
    "simulate_situated_memory_cognitive_round",
    "simulate_situated_memory_cognition",
)
