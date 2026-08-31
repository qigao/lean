"""Long-term recall of sanitized V15 percepts into private cognition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import SituatedActionKind, resolve_situated_round
from narrative_dynamics.abm.situated_cognition import (
    SituatedCognitiveRoundResult,
    _decide_situated_action,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveModel,
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_memory_cognition import (
    SituatedMemoryRecallAdmission,
    SituatedMemoryRecallResult,
)
from narrative_dynamics.abm.situated_memory_cognition_contracts import (
    SituatedAgentRecallPolicy,
)
from narrative_dynamics.abm.situated_perception import project_situated_percepts
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptionModel,
    SituatedPercept,
)
from narrative_dynamics.abm.situated_percept_cognition import (
    SituatedPerceptCognitiveRoundResult,
    admit_situated_percepts,
    match_situated_percept_rule,
    perceptual_timeline,
    project_situated_story_percepts,
)
from narrative_dynamics.abm.situated_percept_memory import (
    ingest_situated_percept_story,
    search_situated_percept_memories,
)
from narrative_dynamics.abm.situated_percept_memory_contracts import (
    SituatedPerceptMemoryPolicy,
    SituatedPerceptMemoryQuery,
    SituatedPerceptMemoryRecord,
)
from narrative_dynamics.abm.situated_story import SituatedStory


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SituatedPerceptMemoryCognitiveModel:
    model_id: str
    version: str
    perception_model: SituatedPerceptionModel
    cognitive_model: SituatedCognitiveModel
    memory_policy: SituatedPerceptMemoryPolicy
    agents: tuple[SituatedAgentRecallPolicy, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="percept memory cognitive model id"))
        object.__setattr__(self, "version", _text(self.version, label="percept memory cognitive model version"))
        if not isinstance(self.perception_model, SituatedPerceptionModel):
            raise TypeError("percept memory cognitive model requires SituatedPerceptionModel")
        if not isinstance(self.cognitive_model, SituatedCognitiveModel):
            raise TypeError("percept memory cognitive model requires SituatedCognitiveModel")
        if self.perception_model.world_model != self.cognitive_model.world_model:
            raise ValueError("percept memory cognitive models must bind the exact world")
        if not isinstance(self.memory_policy, SituatedPerceptMemoryPolicy):
            raise TypeError("percept memory cognitive model requires SituatedPerceptMemoryPolicy")
        if not isinstance(self.agents, tuple) or any(
            not isinstance(item, SituatedAgentRecallPolicy) for item in self.agents
        ):
            raise TypeError("percept memory cognitive agents must be recall policies")
        identities = tuple(item.agent_id for item in self.agents)
        if len(set(identities)) != len(identities):
            raise ValueError("percept memory cognitive agent ids must be unique")
        expected = {item.agent_id for item in self.cognitive_model.agents}
        if set(identities) != expected:
            raise ValueError("percept memory cognitive policies must cover exact agent roster")
        places = {item.place_id for item in self.cognitive_model.world_model.places}
        if any(
            not set(cue.required_place_ids).issubset(places)
            for policy in self.agents
            for cue in policy.cues
        ):
            raise ValueError("percept memory recall place must belong to the world")
        object.__setattr__(self, "agents", tuple(sorted(self.agents, key=lambda item: item.agent_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "perception_model_hash": self.perception_model.content_hash,
            "cognitive_model_hash": self.cognitive_model.content_hash,
            "memory_policy_hash": self.memory_policy.content_hash,
            "agents": [item.to_dict() for item in self.agents],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptMemoryCognitiveRoundResult:
    model_id: str
    model_hash: str
    percept_cognitive_round: SituatedPerceptCognitiveRoundResult
    recalls: tuple[SituatedMemoryRecallResult, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="percept memory round model id"))
        if not isinstance(self.model_hash, str) or not self.model_hash.startswith("sha256:"):
            raise ValueError("percept memory round model hash must be a content hash")
        if not isinstance(self.percept_cognitive_round, SituatedPerceptCognitiveRoundResult):
            raise TypeError("percept memory round requires a percept cognitive round")
        if not isinstance(self.recalls, tuple) or any(
            not isinstance(item, SituatedMemoryRecallResult) for item in self.recalls
        ):
            raise TypeError("percept memory round recalls must be a tuple")
        expected = {item.agent_id for item in self.decisions}
        actual = tuple(item.prior_mind.agent_id for item in self.recalls)
        if len(set(actual)) != len(actual) or set(actual) != expected:
            raise ValueError("percept memory round recalls must cover exact decision roster")
        object.__setattr__(self, "recalls", tuple(sorted(self.recalls, key=lambda item: item.prior_mind.agent_id)))

    @property
    def cognitive_round(self) -> SituatedCognitiveRoundResult:
        return self.percept_cognitive_round.cognitive_round

    @property
    def decisions(self):
        return self.percept_cognitive_round.decisions

    @property
    def next_story(self) -> SituatedStory:
        return self.percept_cognitive_round.next_story

    @property
    def next_state(self) -> SituatedCognitiveState:
        return self.percept_cognitive_round.next_state

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "percept_cognitive_round": self.percept_cognitive_round.to_dict(),
            "recalls": [item.to_dict() for item in self.recalls],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedPerceptMemoryCognitiveTrajectory:
    model_id: str
    model_hash: str
    initial_story: SituatedStory
    initial_state: SituatedCognitiveState
    rounds: tuple[SituatedPerceptMemoryCognitiveRoundResult, ...]
    final_story: SituatedStory
    final_state: SituatedCognitiveState

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("percept memory cognitive trajectory requires a round")
        story, state = self.initial_story, self.initial_state
        for item in self.rounds:
            if item.model_id != self.model_id or item.model_hash != self.model_hash:
                raise ValueError("percept memory trajectory must bind one exact model")
            if item.cognitive_round.prior_story_hash != story.content_hash or item.cognitive_round.prior_state != state:
                raise ValueError("percept memory trajectory rounds must form one chain")
            story, state = item.next_story, item.next_state
        if story != self.final_story or state != self.final_state:
            raise ValueError("percept memory trajectory final values must equal its chain")

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


def initialize_situated_percept_memory_cognition(
    model: SituatedPerceptMemoryCognitiveModel,
    story: SituatedStory,
) -> SituatedCognitiveState:
    """Create a recall checkpoint at the story's current round."""

    if not isinstance(model, SituatedPerceptMemoryCognitiveModel) or not isinstance(story, SituatedStory):
        raise TypeError("percept memory cognition initialization requires model and story")
    if story.perception_model != model.perception_model:
        raise ValueError("percept memory cognition story must bind the exact perception model")
    cognition = model.cognitive_model
    if story.model_hash != cognition.world_model.content_hash:
        raise ValueError("percept memory cognition story must bind the exact world")
    places = {item.agent_id: item.place_id for item in story.current_state.agents}
    round_index = story.current_state.round_index
    state = SituatedCognitiveState(
        cognition.model_id,
        cognition.content_hash,
        round_index,
        None,
        story.content_hash,
        tuple(
            SituatedAgentMindState(
                item.agent_id,
                item.prior_belief,
                places[item.agent_id],
                observation_floor_round=round_index,
            )
            for item in cognition.agents
        ),
        checkpoint=True,
    )
    validate_situated_cognitive_state(cognition, story, state)
    return state


def _percept_from_memory(memory: SituatedPerceptMemoryRecord) -> SituatedPercept:
    return SituatedPercept(
        memory.percept_id,
        memory.round_index,
        memory.agent_id,
        memory.source_event_id,
        memory.source_event_hash,
        memory.channels,
        memory.fidelity,
        memory.actor_agent_id,
        memory.kind,
        memory.place_id,
        memory.outcome,
        memory.details,
    )


def _posterior(
    model: SituatedAgentCognitiveModel,
    belief: PlanningBeliefState,
    action_id: str,
    symbol_id: str,
    weight: float,
) -> PlanningBeliefState:
    likelihoods = {
        (item.action_id, item.hypothesis_id, item.symbol_id): item.probability
        for item in model.likelihoods
    }
    masses = {
        hypothesis_id: probability * (
            (1.0 - weight) + weight * likelihoods[(action_id, hypothesis_id, symbol_id)]
        )
        for hypothesis_id, probability in belief.probabilities.items()
    }
    evidence = math.fsum(masses.values())
    if not math.isfinite(evidence) or evidence <= 0.0:
        raise ValueError("recalled percept has zero probability under current belief")
    return PlanningBeliefState({key: value / evidence for key, value in masses.items()})


def recall_situated_percept_memories(
    database_path: str | Path,
    model: SituatedPerceptMemoryCognitiveModel,
    agent_model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    *,
    story: SituatedStory,
    state: SituatedCognitiveState,
    _source_trust_by_source: Mapping[str, float] | None = None,
    _claim_topic_by_symbol: Mapping[str, str] | None = None,
    _consolidated_claim_keys: frozenset[tuple[str, str, str]] | None = None,
) -> SituatedMemoryRecallResult:
    """Recall validated private records without reconstructing objective events."""

    if not isinstance(model, SituatedPerceptMemoryCognitiveModel):
        raise TypeError("percept recall requires a SituatedPerceptMemoryCognitiveModel")
    if not isinstance(agent_model, SituatedAgentCognitiveModel) or not isinstance(mind, SituatedAgentMindState):
        raise TypeError("percept recall requires an agent model and mind")
    if agent_model.agent_id != mind.agent_id:
        raise ValueError("percept recall agent model and mind must match")
    expected_agent = next((item for item in model.cognitive_model.agents if item.agent_id == mind.agent_id), None)
    if expected_agent != agent_model:
        raise ValueError("percept recall agent model must belong to the exact model")
    if story.perception_model != model.perception_model:
        raise ValueError("percept recall story must bind the exact perception model")
    validate_situated_cognitive_state(model.cognitive_model, story, state)
    state_mind = next(item for item in state.minds if item.agent_id == mind.agent_id)
    if (
        mind.own_place_id != state_mind.own_place_id
        or mind.observation_floor_round != state_mind.observation_floor_round
        or mind.selected_action_ids != state_mind.selected_action_ids
        or mind.decision_count != state_mind.decision_count
    ):
        raise ValueError("percept recall mind must belong to the current cognitive state")
    source_trust = {} if _source_trust_by_source is None else dict(_source_trust_by_source)
    agent_ids = {item.agent_id for item in model.cognitive_model.agents}
    for source_id, value in source_trust.items():
        if source_id not in agent_ids:
            raise ValueError("percept recall source trust agent must belong to the model")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError("percept recall source trust must be in [0, 1]")
    claim_topics = {} if _claim_topic_by_symbol is None else dict(_claim_topic_by_symbol)
    active_claims: dict[tuple[str, str], str] = {}
    for source_id, topic_id, symbol_id in (() if _consolidated_claim_keys is None else _consolidated_claim_keys):
        scope = (source_id, topic_id)
        if scope in active_claims and active_claims[scope] != symbol_id:
            raise ValueError("percept recall claim scope has multiple active symbols")
        active_claims[scope] = symbol_id

    ingest_situated_percept_story(
        database_path,
        model.perception_model,
        story,
        mind.agent_id,
        model.memory_policy,
    )
    projections = project_situated_story_percepts(model.perception_model, story)
    authoritative: dict[str, tuple[SituatedPercept, str]] = {
        percept.percept_id: (percept, projection.content_hash)
        for projection in projections
        for percept in projection.percepts
        if percept.agent_id == mind.agent_id
    }
    policy = next(item for item in model.agents if item.agent_id == mind.agent_id)
    if state.round_index == 0 or not policy.cues:
        return SituatedMemoryRecallResult(mind, (), mind)
    excluded = set(mind.processed_observation_ids) | set(mind.recalled_memory_ids)
    candidates = []
    seen = set()
    for cue in policy.cues:
        if cue.required_place_ids and mind.own_place_id not in cue.required_place_ids:
            continue
        eligible_ids = set(authoritative) - excluded - seen
        if not eligible_ids:
            continue
        hits = search_situated_percept_memories(
            database_path,
            SituatedPerceptMemoryQuery(
                mind.agent_id,
                text=cue.text,
                event_kinds=cue.event_kinds,
                channels=cue.channels,
                max_round=state.round_index,
                min_confidence=cue.min_confidence,
                limit=cue.limit,
                story_model_hash=story.model_hash,
                excluded_memory_ids=tuple(excluded | seen),
                included_memory_ids=tuple(eligible_ids),
            ),
        )
        for hit in hits:
            memory = hit.memory
            expected_percept, projection_hash = authoritative[memory.percept_id]
            if (
                memory.memory_id in excluded
                or memory.memory_id in seen
                or memory.perception_model_id != model.perception_model.model_id
                or memory.perception_model_hash != model.perception_model.content_hash
                or memory.story_model_id != story.model_id
                or memory.story_model_hash != story.model_hash
                or memory.projection_hash != projection_hash
                or memory.policy_hash != model.memory_policy.content_hash
                or _percept_from_memory(memory) != expected_percept
            ):
                continue
            seen.add(memory.memory_id)
            candidates.append((cue, hit, expected_percept))
            if len(candidates) >= policy.max_memories_per_round:
                break
        if len(candidates) >= policy.max_memories_per_round:
            break

    belief = mind.belief
    recalled_ids = set(mind.recalled_memory_ids)
    observed = set(mind.observed_event_ids)
    admissions = []
    for cue, hit, percept in candidates:
        memory = hit.memory
        rule = match_situated_percept_rule(agent_model, percept)
        prior = belief
        trust = (
            source_trust.get(percept.actor_agent_id, 1.0)
            if percept.kind is SituatedActionKind.TELL
            and percept.actor_agent_id is not None
            and percept.actor_agent_id != mind.agent_id
            else 1.0
        )
        weight = memory.confidence * memory.salience * trust
        claim_scope = None
        if (
            rule is not None
            and percept.kind is SituatedActionKind.TELL
            and percept.actor_agent_id != mind.agent_id
            and percept.actor_agent_id is not None
            and rule.symbol_id in claim_topics
        ):
            claim_scope = (percept.actor_agent_id, claim_topics[rule.symbol_id])
        consolidated = claim_scope is not None and active_claims.get(claim_scope) == rule.symbol_id
        if rule is None:
            posterior = prior
            rule_id = symbol_id = likelihood_action_id = None
        elif consolidated:
            posterior = prior
            rule_id, symbol_id, likelihood_action_id = rule.rule_id, rule.symbol_id, rule.likelihood_action_id
        else:
            posterior = _posterior(
                agent_model, prior, rule.likelihood_action_id, rule.symbol_id, weight
            )
            rule_id, symbol_id, likelihood_action_id = rule.rule_id, rule.symbol_id, rule.likelihood_action_id
            if claim_scope is not None:
                active_claims[claim_scope] = rule.symbol_id
        admissions.append(SituatedMemoryRecallAdmission(
            mind.agent_id,
            cue.cue_id,
            memory.memory_id,
            memory.source_event_id,
            rule_id,
            symbol_id,
            likelihood_action_id,
            memory.confidence,
            memory.salience,
            weight,
            None,
            prior,
            posterior,
            trust,
            consolidated,
        ))
        belief = posterior
        recalled_ids.add(memory.memory_id)
        observed.add(memory.source_event_id)
    return SituatedMemoryRecallResult(
        mind,
        tuple(admissions),
        replace(
            mind,
            belief=belief,
            recalled_memory_ids=tuple(recalled_ids),
            observed_event_ids=tuple(observed),
        ),
    )


def simulate_situated_percept_memory_cognitive_round(
    database_path: str | Path,
    model: SituatedPerceptMemoryCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
) -> SituatedPerceptMemoryCognitiveRoundResult:
    return _simulate_situated_percept_memory_cognitive_round(
        database_path,
        model,
        story,
        state,
        source_trust_by_observer=None,
    )


def _simulate_situated_percept_memory_cognitive_round(
    database_path: str | Path,
    model: SituatedPerceptMemoryCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
    *,
    source_trust_by_observer: Mapping[str, Mapping[str, float]] | None,
    claim_topic_by_symbol: Mapping[str, str] | None = None,
    consolidated_claim_keys_by_observer: Mapping[str, frozenset[tuple[str, str, str]]] | None = None,
) -> SituatedPerceptMemoryCognitiveRoundResult:
    if not isinstance(model, SituatedPerceptMemoryCognitiveModel):
        raise TypeError("percept memory simulation requires its bound model")
    if story.perception_model != model.perception_model:
        raise ValueError("percept memory simulation story must bind the exact perception model")
    cognition = model.cognitive_model
    validate_situated_cognitive_state(cognition, story, state)
    projections = project_situated_story_percepts(model.perception_model, story)
    model_by_id = {item.agent_id: item for item in cognition.agents}
    mind_by_id = {item.agent_id: item for item in state.minds}
    decisions, recalls = [], []
    admitted_minds = {}
    next_round = state.round_index + 1
    for agent_id in sorted(model_by_id):
        private = perceptual_timeline(model.perception_model, story, agent_id)
        claim_keys = set(
            ()
            if consolidated_claim_keys_by_observer is None
            else consolidated_claim_keys_by_observer.get(agent_id, frozenset())
        )
        direct = admit_situated_percepts(
            model_by_id[agent_id],
            mind_by_id[agent_id],
            private,
            _claim_topic_by_symbol=claim_topic_by_symbol,
            _consolidated_claim_keys=frozenset(claim_keys),
        )
        private_by_id = {item.percept_id: item for item in private}
        if claim_topic_by_symbol is not None:
            for admission in direct.admissions:
                percept = private_by_id[admission.observation_id]
                if (
                    percept.kind is SituatedActionKind.TELL
                    and percept.actor_agent_id is not None
                    and percept.actor_agent_id != agent_id
                    and admission.symbol_id in claim_topic_by_symbol
                ):
                    scope = (
                        percept.actor_agent_id,
                        claim_topic_by_symbol[admission.symbol_id],
                    )
                    claim_keys = {key for key in claim_keys if key[:2] != scope}
                    claim_keys.add((scope[0], scope[1], admission.symbol_id))
        recalled = recall_situated_percept_memories(
            database_path,
            model,
            model_by_id[agent_id],
            direct.next_mind,
            story=story,
            state=state,
            _source_trust_by_source=(
                None if source_trust_by_observer is None else source_trust_by_observer.get(agent_id, {})
            ),
            _claim_topic_by_symbol=claim_topic_by_symbol,
            _consolidated_claim_keys=frozenset(claim_keys),
        )
        recalls.append(recalled)
        admitted_minds[agent_id] = recalled.next_mind
        decisions.append(_decide_situated_action(
            model_by_id[agent_id],
            recalled.next_mind,
            tuple(
                (item.source_event_id, item.kind)
                for item in private
                if item.kind is not None
            ),
            round_index=next_round,
            prior_belief=direct.prior_mind.belief,
            admissions=direct.admissions,
            recalled_memory_ids=tuple(item.memory_id for item in recalled.admissions),
            recalled_symbol_ids=tuple(dict.fromkeys(
                item.symbol_id
                for item in recalled.admissions
                if item.symbol_id is not None and not item.consolidated
            )),
        ))
    world_round = resolve_situated_round(
        cognition.world_model,
        story.current_state,
        tuple(item.intent for item in decisions),
    )
    next_story = SituatedStory(
        story.model_id,
        story.model_hash,
        story.initial_state,
        story.rounds + (world_round,),
        model.perception_model,
    )
    next_projection = project_situated_percepts(model.perception_model, world_round)
    newly_observed = {item.agent_id: set() for item in cognition.agents}
    for percept in next_projection.percepts:
        newly_observed[percept.agent_id].add(percept.source_event_id)
    events = {item.actor_agent_id: item for item in world_round.events}
    next_minds = []
    for decision in decisions:
        prior = admitted_minds[decision.agent_id]
        event = events[decision.agent_id]
        place_id = prior.own_place_id
        if event.kind is SituatedActionKind.MOVE and event.success:
            place_id = next(item.value for item in event.details if item.name == "destination_place_id")
        next_minds.append(replace(
            prior,
            own_place_id=place_id,
            observed_event_ids=tuple(set(prior.observed_event_ids) | newly_observed[decision.agent_id]),
            selected_action_ids=prior.selected_action_ids + (decision.selected_action_id,),
            decision_count=prior.decision_count + 1,
        ))
    next_state = SituatedCognitiveState(
        cognition.model_id,
        cognition.content_hash,
        next_round,
        state.content_hash,
        next_story.content_hash,
        tuple(next_minds),
    )
    validate_situated_cognitive_state(cognition, next_story, next_state)
    cognitive_round = SituatedCognitiveRoundResult(
        state,
        story.content_hash,
        tuple(decisions),
        next_story,
        next_state,
    )
    percept_round = SituatedPerceptCognitiveRoundResult(
        model.perception_model.model_id,
        model.perception_model.content_hash,
        tuple(item.content_hash for item in projections),
        cognitive_round,
        next_projection,
    )
    return SituatedPerceptMemoryCognitiveRoundResult(
        model.model_id,
        model.content_hash,
        percept_round,
        tuple(recalls),
    )


def simulate_situated_percept_memory_cognition(
    database_path: str | Path,
    model: SituatedPerceptMemoryCognitiveModel,
    initial_story: SituatedStory,
    initial_state: SituatedCognitiveState,
    *,
    round_count: int,
) -> SituatedPerceptMemoryCognitiveTrajectory:
    if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count <= 0:
        raise ValueError("percept memory simulation requires a positive round count")
    validate_situated_cognitive_state(model.cognitive_model, initial_story, initial_state)
    story, state = initial_story, initial_state
    rounds = []
    for _ in range(round_count):
        result = simulate_situated_percept_memory_cognitive_round(
            database_path, model, story, state
        )
        rounds.append(result)
        story, state = result.next_story, result.next_state
    return SituatedPerceptMemoryCognitiveTrajectory(
        model.model_id,
        model.content_hash,
        initial_story,
        initial_state,
        tuple(rounds),
        story,
        state,
    )


__all__ = (
    "SituatedPerceptMemoryCognitiveModel",
    "SituatedPerceptMemoryCognitiveRoundResult",
    "SituatedPerceptMemoryCognitiveTrajectory",
    "initialize_situated_percept_memory_cognition",
    "recall_situated_percept_memories",
    "simulate_situated_percept_memory_cognitive_round",
    "simulate_situated_percept_memory_cognition",
)
