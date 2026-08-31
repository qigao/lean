"""Percept-authorized cognition over an exact situated story."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
import math

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionKind,
    resolve_situated_round,
)
from narrative_dynamics.abm.situated_cognition import (
    SituatedBeliefAdmission,
    SituatedBeliefAdmissionResult,
    SituatedCognitiveRoundResult,
    _decide_situated_action,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveModel,
    SituatedCognitiveState,
    SituatedObservationRule,
    validate_situated_cognitive_state,
)
from narrative_dynamics.abm.situated_perception import project_situated_percepts
from narrative_dynamics.abm.situated_perception_contracts import (
    SituatedPerceptionModel,
    SituatedPercept,
    SituatedPerceptFidelity,
    SituatedPerceptualProjection,
)
from narrative_dynamics.abm.situated_story import SituatedStory


def _validate_story_binding(
    model: SituatedPerceptionModel,
    story: SituatedStory,
) -> None:
    if not isinstance(model, SituatedPerceptionModel):
        raise TypeError("perceptual cognition requires a SituatedPerceptionModel")
    if not isinstance(story, SituatedStory):
        raise TypeError("perceptual cognition requires a SituatedStory")
    if (
        story.model_id != model.world_model.model_id
        or story.model_hash != model.world_model.content_hash
    ):
        raise ValueError("perception model and story must bind the exact world")
    if story.perception_model is not None and story.perception_model != model:
        raise ValueError("perceptual story must retain the exact perception model")


def project_situated_story_percepts(
    model: SituatedPerceptionModel,
    story: SituatedStory,
) -> tuple[SituatedPerceptualProjection, ...]:
    """Re-derive the canonical private projection for every accepted story round."""

    _validate_story_binding(model, story)
    return tuple(project_situated_percepts(model, result) for result in story.rounds)


def perceptual_timeline(
    model: SituatedPerceptionModel,
    story: SituatedStory,
    agent_id: str,
) -> tuple[SituatedPercept, ...]:
    """Return only the sanitized percepts addressed to one embodied agent."""

    _validate_story_binding(model, story)
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("perceptual timeline agent id must be non-empty")
    if agent_id not in {item.agent_id for item in model.world_model.agents}:
        raise ValueError("perceptual timeline agent must belong to the world")
    return tuple(
        percept
        for projection in project_situated_story_percepts(model, story)
        for percept in projection.percepts
        if percept.agent_id == agent_id
    )


def match_situated_percept_rule(
    model: SituatedAgentCognitiveModel,
    percept: SituatedPercept,
) -> SituatedObservationRule | None:
    """Match one rule using only fields disclosed by a private percept."""

    if not isinstance(model, SituatedAgentCognitiveModel):
        raise TypeError("percept rule matching requires a SituatedAgentCognitiveModel")
    if not isinstance(percept, SituatedPercept):
        raise TypeError("percept rule matching requires a SituatedPercept")
    if percept.agent_id != model.agent_id:
        raise ValueError("percept rule matching requires the model agent's private percept")
    if percept.fidelity is SituatedPerceptFidelity.DETECTED:
        return None
    if (
        percept.kind is SituatedActionKind.TELL
        and ObservationChannel.SELF in percept.channels
    ):
        return None

    details = {(item.name, item.value) for item in percept.details}
    matches = []
    for rule in model.observation_rules:
        if rule.event_kind is not percept.kind:
            continue
        if rule.outcome is not None and rule.outcome != percept.outcome:
            continue
        if (
            rule.detail_name is not None
            and (rule.detail_name, rule.detail_value) not in details
        ):
            continue
        matches.append(rule)
    if len(matches) > 1:
        raise ValueError("private percept matches multiple cognitive rules")
    return None if not matches else matches[0]


def admit_situated_percepts(
    model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    percepts: tuple[SituatedPercept, ...],
    *,
    _claim_topic_by_symbol: Mapping[str, str] | None = None,
    _consolidated_claim_keys: frozenset[tuple[str, str, str]] | None = None,
) -> SituatedBeliefAdmissionResult:
    """Apply newly disclosed private percepts to one agent's Bayesian state."""

    if not isinstance(model, SituatedAgentCognitiveModel) or not isinstance(
        mind, SituatedAgentMindState
    ):
        raise TypeError("percept admission requires an agent model and mind")
    if model.agent_id != mind.agent_id:
        raise ValueError("percept admission model and mind agent must match")
    if not isinstance(percepts, tuple) or any(
        not isinstance(item, SituatedPercept) for item in percepts
    ):
        raise TypeError("percept admission percepts must be a tuple of SituatedPercept values")
    if any(item.agent_id != mind.agent_id for item in percepts):
        raise ValueError("percept admission may receive only the agent's private percepts")

    claim_topic_by_symbol = (
        {} if _claim_topic_by_symbol is None else dict(_claim_topic_by_symbol)
    )
    consolidated_claim_keys = set(
        () if _consolidated_claim_keys is None else _consolidated_claim_keys
    )
    active_claim_symbols: dict[tuple[str, str], str] = {}
    for source_id, topic_id, symbol_id in consolidated_claim_keys:
        scope = (source_id, topic_id)
        existing = active_claim_symbols.get(scope)
        if existing is not None and existing != symbol_id:
            raise ValueError("situated percept claim scope has multiple active symbols")
        active_claim_symbols[scope] = symbol_id

    likelihoods = {
        (item.action_id, item.hypothesis_id, item.symbol_id): item.probability
        for item in model.likelihoods
    }
    processed = set(mind.processed_observation_ids)
    observed = set(mind.observed_event_ids)
    belief = mind.belief
    admissions = []
    for percept in percepts:
        if (
            percept.round_index <= mind.observation_floor_round
            or percept.percept_id in processed
        ):
            continue
        processed.add(percept.percept_id)
        observed.add(percept.source_event_id)
        rule = match_situated_percept_rule(model, percept)
        if rule is None:
            continue
        claim_scope = None
        if (
            percept.kind is SituatedActionKind.TELL
            and percept.actor_agent_id != mind.agent_id
            and percept.actor_agent_id is not None
            and rule.symbol_id in claim_topic_by_symbol
        ):
            claim_scope = (
                percept.actor_agent_id,
                claim_topic_by_symbol[rule.symbol_id],
            )
        consolidated = (
            claim_scope is not None
            and active_claim_symbols.get(claim_scope) == rule.symbol_id
        )
        if consolidated:
            admissions.append(SituatedBeliefAdmission(
                percept.percept_id,
                percept.source_event_id,
                rule.symbol_id,
                rule.likelihood_action_id,
                belief,
                belief,
                True,
            ))
            continue
        masses = {
            hypothesis_id: probability
            * likelihoods[(rule.likelihood_action_id, hypothesis_id, rule.symbol_id)]
            for hypothesis_id, probability in belief.probabilities.items()
        }
        evidence = math.fsum(masses.values())
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise ValueError("private percept has zero probability under current belief")
        posterior = PlanningBeliefState(
            {key: value / evidence for key, value in masses.items()}
        )
        admissions.append(SituatedBeliefAdmission(
            percept.percept_id,
            percept.source_event_id,
            rule.symbol_id,
            rule.likelihood_action_id,
            belief,
            posterior,
        ))
        belief = posterior
        if claim_scope is not None:
            active_claim_symbols[claim_scope] = rule.symbol_id

    next_mind = replace(
        mind,
        belief=belief,
        processed_observation_ids=tuple(processed),
        observed_event_ids=tuple(observed),
    )
    return SituatedBeliefAdmissionResult(mind, tuple(admissions), next_mind)


@dataclass(frozen=True)
class SituatedPerceptCognitiveRoundResult:
    perception_model_id: str
    perception_model_hash: str
    prior_projection_hashes: tuple[str, ...]
    cognitive_round: SituatedCognitiveRoundResult
    next_projection: SituatedPerceptualProjection

    def __post_init__(self) -> None:
        if not isinstance(self.perception_model_id, str) or not self.perception_model_id.strip():
            raise ValueError("percept cognitive round model id must be non-empty")
        if not isinstance(self.perception_model_hash, str) or not self.perception_model_hash.startswith("sha256:"):
            raise ValueError("percept cognitive round model hash must be a content hash")
        if not isinstance(self.prior_projection_hashes, tuple) or any(
            not isinstance(item, str) or not item.startswith("sha256:")
            for item in self.prior_projection_hashes
        ):
            raise TypeError("percept cognitive prior projection hashes must be a tuple of content hashes")
        if not isinstance(self.cognitive_round, SituatedCognitiveRoundResult):
            raise TypeError("percept cognitive round requires a cognitive round")
        if not isinstance(self.next_projection, SituatedPerceptualProjection):
            raise TypeError("percept cognitive round requires a next projection")
        if (
            self.next_projection.model_id != self.perception_model_id
            or self.next_projection.model_hash != self.perception_model_hash
        ):
            raise ValueError("percept cognitive projection must bind the exact perception model")
        if (
            self.next_projection.round_result_hash
            != self.cognitive_round.next_story.rounds[-1].content_hash
        ):
            raise ValueError("percept cognitive projection must bind the resolved round")

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
            "perception_model_id": self.perception_model_id,
            "perception_model_hash": self.perception_model_hash,
            "prior_projection_hashes": list(self.prior_projection_hashes),
            "cognitive_round": self.cognitive_round.to_dict(),
            "next_projection": self.next_projection.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def simulate_situated_percept_cognitive_round(
    perception_model: SituatedPerceptionModel,
    cognitive_model: SituatedCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
) -> SituatedPerceptCognitiveRoundResult:
    """Advance one cognitive round using V15 percepts as private evidence."""

    _validate_story_binding(perception_model, story)
    if not isinstance(cognitive_model, SituatedCognitiveModel):
        raise TypeError("percept cognitive simulation requires a SituatedCognitiveModel")
    if cognitive_model.world_model != perception_model.world_model:
        raise ValueError("perception and cognitive models must bind the exact world")
    validate_situated_cognitive_state(cognitive_model, story, state)

    projections = project_situated_story_percepts(perception_model, story)
    private_by_agent = {
        item.agent_id: tuple(
            percept
            for projection in projections
            for percept in projection.percepts
            if percept.agent_id == item.agent_id
        )
        for item in cognitive_model.agents
    }
    model_by_id = {item.agent_id: item for item in cognitive_model.agents}
    mind_by_id = {item.agent_id: item for item in state.minds}
    admitted_minds = {}
    decisions = []
    next_round = state.round_index + 1
    for agent_id in sorted(model_by_id):
        private = private_by_agent[agent_id]
        admission = admit_situated_percepts(
            model_by_id[agent_id],
            mind_by_id[agent_id],
            private,
        )
        admitted_minds[agent_id] = admission.next_mind
        candidates = tuple(
            (item.source_event_id, item.kind)
            for item in private
            if item.kind is not None
        )
        decisions.append(_decide_situated_action(
            model_by_id[agent_id],
            admission.next_mind,
            candidates,
            round_index=next_round,
            prior_belief=admission.prior_mind.belief,
            admissions=admission.admissions,
        ))

    world_round = resolve_situated_round(
        cognitive_model.world_model,
        story.current_state,
        tuple(item.intent for item in decisions),
    )
    next_story = SituatedStory(
        story.model_id,
        story.model_hash,
        story.initial_state,
        story.rounds + (world_round,),
        perception_model,
    )
    next_projection = project_situated_percepts(
        perception_model,
        world_round,
    )
    newly_observed = {item.agent_id: set() for item in cognitive_model.agents}
    for percept in next_projection.percepts:
        newly_observed[percept.agent_id].add(percept.source_event_id)
    event_by_actor = {item.actor_agent_id: item for item in world_round.events}
    next_minds = []
    for decision in decisions:
        prior = admitted_minds[decision.agent_id]
        event = event_by_actor[decision.agent_id]
        place_id = prior.own_place_id
        if event.kind is SituatedActionKind.MOVE and event.success:
            place_id = next(
                item.value
                for item in event.details
                if item.name == "destination_place_id"
            )
        next_minds.append(replace(
            prior,
            own_place_id=place_id,
            observed_event_ids=tuple(
                set(prior.observed_event_ids)
                | newly_observed[decision.agent_id]
            ),
            selected_action_ids=prior.selected_action_ids
            + (decision.selected_action_id,),
            decision_count=prior.decision_count + 1,
        ))
    next_state = SituatedCognitiveState(
        cognitive_model.model_id,
        cognitive_model.content_hash,
        next_round,
        state.content_hash,
        next_story.content_hash,
        tuple(next_minds),
    )
    validate_situated_cognitive_state(
        cognitive_model,
        next_story,
        next_state,
    )
    cognitive_round = SituatedCognitiveRoundResult(
        state,
        story.content_hash,
        tuple(decisions),
        next_story,
        next_state,
    )
    return SituatedPerceptCognitiveRoundResult(
        perception_model.model_id,
        perception_model.content_hash,
        tuple(item.content_hash for item in projections),
        cognitive_round,
        next_projection,
    )


__all__ = (
    "project_situated_story_percepts",
    "perceptual_timeline",
    "match_situated_percept_rule",
    "admit_situated_percepts",
    "SituatedPerceptCognitiveRoundResult",
    "simulate_situated_percept_cognitive_round",
)
