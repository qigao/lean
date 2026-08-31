"""Private Bayesian cognition and POMDP action selection for V10 worlds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from collections.abc import Mapping

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import SituatedActionIntent, SituatedActionKind
from narrative_dynamics.abm.situated_story import (
    SituatedPerspectiveEvent,
    SituatedStory,
    advance_situated_story,
    perspective_timeline,
)
from narrative_dynamics.abm.situated_cognition_contracts import (
    SituatedActionSpec,
    SituatedAgentCognitiveModel,
    SituatedAgentMindState,
    SituatedCognitiveModel,
    SituatedCognitiveState,
    validate_situated_cognitive_state,
)


def _freeze_float_map(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    result = {}
    for key, raw in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{label} keys must be non-empty strings")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(raw):
            raise ValueError(f"{label} values must be finite numeric")
        result[key] = float(raw)
    return MappingProxyType({key: result[key] for key in sorted(result)})


@dataclass(frozen=True)
class SituatedBeliefAdmission:
    observation_id: str
    event_id: str
    symbol_id: str
    likelihood_action_id: str
    prior_belief: PlanningBeliefState
    posterior_belief: PlanningBeliefState

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id, "event_id": self.event_id,
            "symbol_id": self.symbol_id, "likelihood_action_id": self.likelihood_action_id,
            "prior_belief": self.prior_belief.to_dict(), "posterior_belief": self.posterior_belief.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedBeliefAdmissionResult:
    prior_mind: SituatedAgentMindState
    admissions: tuple[SituatedBeliefAdmission, ...]
    next_mind: SituatedAgentMindState

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
class SituatedCognitiveDecision:
    agent_id: str
    round_index: int
    prior_belief: PlanningBeliefState
    posterior_belief: PlanningBeliefState
    admitted_observation_ids: tuple[str, ...]
    admitted_symbol_ids: tuple[str, ...]
    feasible_action_ids: tuple[str, ...]
    action_values: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action_id: str
    selected_goal_contributions: Mapping[str, float]
    intent: SituatedActionIntent

    def __post_init__(self) -> None:
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index <= 0:
            raise ValueError("situated cognitive decision round must be positive")
        values = _freeze_float_map(self.action_values, label="situated decision action values")
        policy = _freeze_float_map(self.action_policy, label="situated decision action policy")
        if set(values) != set(policy) or set(values) != set(self.feasible_action_ids):
            raise ValueError("situated decision values and policy must cover feasible actions")
        if not math.isclose(math.fsum(policy.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("situated decision action policy must sum to 1")
        maximum = max(policy.values())
        expected = min(key for key, value in policy.items() if value == maximum)
        if self.selected_action_id != expected:
            raise ValueError("situated decision selected action must be lexical MAP")
        if self.intent.agent_id != self.agent_id:
            raise ValueError("situated decision intent must belong to its agent")
        object.__setattr__(self, "action_values", values)
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_goal_contributions", _freeze_float_map(self.selected_goal_contributions, label="situated decision goal contributions"))

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id, "round_index": self.round_index,
            "prior_belief": self.prior_belief.to_dict(), "posterior_belief": self.posterior_belief.to_dict(),
            "admitted_observation_ids": list(self.admitted_observation_ids), "admitted_symbol_ids": list(self.admitted_symbol_ids),
            "feasible_action_ids": list(self.feasible_action_ids), "action_values": dict(self.action_values),
            "action_policy": dict(self.action_policy), "selected_action_id": self.selected_action_id,
            "selected_goal_contributions": dict(self.selected_goal_contributions), "intent": self.intent.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedCognitiveRoundResult:
    prior_state: SituatedCognitiveState
    prior_story_hash: str
    decisions: tuple[SituatedCognitiveDecision, ...]
    next_story: SituatedStory
    next_state: SituatedCognitiveState

    def __post_init__(self) -> None:
        if self.prior_story_hash != self.prior_state.story_hash:
            raise ValueError("cognitive round prior story must match prior state")
        if self.next_state.story_hash != self.next_story.content_hash:
            raise ValueError("cognitive round next state must match next story")
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError("cognitive round next state must bind prior state")
        if self.next_state.round_index != self.prior_state.round_index + 1:
            raise ValueError("cognitive round must advance exactly once")
        if not isinstance(self.decisions, tuple) or any(not isinstance(item, SituatedCognitiveDecision) for item in self.decisions):
            raise TypeError("cognitive round decisions must be a tuple")
        ids = tuple(item.agent_id for item in self.decisions)
        if len(set(ids)) != len(ids):
            raise ValueError("cognitive round requires unique agent decisions")
        object.__setattr__(self, "decisions", tuple(sorted(self.decisions, key=lambda item: item.agent_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "prior_state_hash": self.prior_state.content_hash, "prior_story_hash": self.prior_story_hash,
            "decisions": [item.to_dict() for item in self.decisions],
            "next_story_hash": self.next_story.content_hash, "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _matching_rule(model: SituatedAgentCognitiveModel, item: SituatedPerspectiveEvent):
    matches = []
    details = {(detail.name, detail.value) for detail in item.event.details}
    for rule in model.observation_rules:
        if rule.event_kind is not item.event.kind:
            continue
        if rule.outcome is not None and rule.outcome != item.event.outcome:
            continue
        if rule.detail_name is not None and (rule.detail_name, rule.detail_value) not in details:
            continue
        matches.append(rule)
    if len(matches) > 1:
        raise ValueError("private observation matches multiple cognitive rules")
    return None if not matches else matches[0]


def admit_situated_observations(
    model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    perspective: tuple[SituatedPerspectiveEvent, ...],
) -> SituatedBeliefAdmissionResult:
    if model.agent_id != mind.agent_id:
        raise ValueError("belief admission model and mind agent must match")
    if not isinstance(perspective, tuple) or any(not isinstance(item, SituatedPerspectiveEvent) for item in perspective):
        raise TypeError("belief admission perspective must be a tuple of SituatedPerspectiveEvent values")
    if any(item.observation.agent_id != mind.agent_id for item in perspective):
        raise ValueError("belief admission may receive only the agent's private perspective")
    processed = set(mind.processed_observation_ids)
    observed = set(mind.observed_event_ids)
    belief = mind.belief
    likelihoods = {(item.action_id, item.hypothesis_id, item.symbol_id): item.probability for item in model.likelihoods}
    admissions = []
    new_items = [item for item in perspective if item.observation.observation_id not in processed]
    for item in new_items:
        processed.add(item.observation.observation_id)
        observed.add(item.event.event_id)
        rule = _matching_rule(model, item)
        if rule is None:
            continue
        masses = {
            hypothesis_id: probability * likelihoods[(rule.likelihood_action_id, hypothesis_id, rule.symbol_id)]
            for hypothesis_id, probability in belief.probabilities.items()
        }
        evidence = math.fsum(masses.values())
        if not math.isfinite(evidence) or evidence <= 0.0:
            raise ValueError("private observation has zero probability under current belief")
        posterior = PlanningBeliefState({key: value / evidence for key, value in masses.items()})
        admissions.append(SituatedBeliefAdmission(
            item.observation.observation_id, item.event.event_id, rule.symbol_id,
            rule.likelihood_action_id, belief, posterior,
        ))
        belief = posterior
    next_mind = replace(
        mind,
        belief=belief,
        processed_observation_ids=tuple(processed),
        observed_event_ids=tuple(observed),
    )
    return SituatedBeliefAdmissionResult(mind, tuple(admissions), next_mind)


def _transition(model: SituatedAgentCognitiveModel, action_id: str, prior: str, nxt: str) -> float:
    if not model.transitions:
        return float(prior == nxt)
    return next(item.probability for item in model.transitions if item.action_id == action_id and item.prior_hypothesis_id == prior and item.next_hypothesis_id == nxt)


def _reward_tables(model: SituatedAgentCognitiveModel):
    weights = {item.goal_id: item.weight for item in model.goals}
    rewards = {(item.goal_id, item.hypothesis_id, item.action_id): item.value for item in model.rewards}
    return weights, rewards


def _goal_contributions(model: SituatedAgentCognitiveModel, belief: PlanningBeliefState, action_id: str) -> dict[str, float]:
    weights, rewards = _reward_tables(model)
    return {
        goal_id: weights[goal_id] * math.fsum(
            probability * rewards[(goal_id, hypothesis_id, action_id)]
            for hypothesis_id, probability in belief.probabilities.items()
        )
        for goal_id in weights
    }


def _predict(model: SituatedAgentCognitiveModel, belief: PlanningBeliefState, action_id: str) -> PlanningBeliefState:
    hypotheses = tuple(item.hypothesis_id for item in model.hypotheses)
    return PlanningBeliefState({
        nxt: math.fsum(belief.probabilities[prior] * _transition(model, action_id, prior, nxt) for prior in hypotheses)
        for nxt in hypotheses
    })


def _solve_values(model: SituatedAgentCognitiveModel, root: PlanningBeliefState, root_actions: tuple[str, ...]) -> dict[str, float]:
    likelihoods = {(item.action_id, item.hypothesis_id, item.symbol_id): item.probability for item in model.likelihoods}
    symbols = tuple(item.symbol_id for item in model.observation_symbols)
    cache: dict[tuple[int, str], tuple[dict[str, float], float]] = {}

    def solve(depth: int, belief: PlanningBeliefState, actions: tuple[str, ...] | None = None):
        key = (depth, belief.content_hash)
        if actions is None and key in cache:
            return cache[key]
        action_ids = model.action_schedule[depth] if actions is None else actions
        values = {}
        for action_id in action_ids:
            immediate = math.fsum(_goal_contributions(model, belief, action_id).values())
            future = 0.0
            if depth + 1 < model.horizon:
                predicted = _predict(model, belief, action_id)
                for symbol_id in symbols:
                    probability = math.fsum(
                        predicted.probabilities[hypothesis_id] * likelihoods[(action_id, hypothesis_id, symbol_id)]
                        for hypothesis_id in predicted.probabilities
                    )
                    if probability <= 0.0:
                        continue
                    posterior = PlanningBeliefState({
                        hypothesis_id: predicted.probabilities[hypothesis_id] * likelihoods[(action_id, hypothesis_id, symbol_id)] / probability
                        for hypothesis_id in predicted.probabilities
                    })
                    _, continuation = solve(depth + 1, posterior)
                    future += probability * continuation
            values[action_id] = immediate + model.discount * future
        policy = finite_softmax(values, beta=model.beta)
        state_value = math.fsum(policy[action_id] * values[action_id] for action_id in values)
        result = (values, state_value)
        if actions is None:
            cache[key] = result
        return result

    return solve(0, root, root_actions)[0]


def _source_event_id(action: SituatedActionSpec, mind: SituatedAgentMindState, perspective: tuple[SituatedPerspectiveEvent, ...]) -> str | None:
    if not action.source_event_kinds:
        return None
    allowed = set(action.source_event_kinds)
    observed = set(mind.observed_event_ids)
    for item in reversed(perspective):
        if item.event.event_id in observed and item.event.kind in allowed:
            return item.event.event_id
    return None


def decide_situated_action(
    model: SituatedAgentCognitiveModel,
    mind: SituatedAgentMindState,
    perspective: tuple[SituatedPerspectiveEvent, ...],
    *,
    round_index: int,
    prior_belief: PlanningBeliefState | None = None,
    admissions: tuple[SituatedBeliefAdmission, ...] = (),
) -> SituatedCognitiveDecision:
    if model.agent_id != mind.agent_id:
        raise ValueError("situated decision model and mind agent must match")
    source_by_action = {item.action_id: _source_event_id(item, mind, perspective) for item in model.actions}
    prior_selected = set(mind.selected_action_ids)
    feasible_specs = tuple(
        item for item in model.actions
        if item.action_id in model.action_schedule[0]
        and (not item.required_place_ids or mind.own_place_id in item.required_place_ids)
        and (item.repeatable or item.action_id not in prior_selected)
        and (not item.source_event_kinds or source_by_action[item.action_id] is not None)
    )
    if not feasible_specs:
        raise ValueError("situated cognitive agent has no feasible root action")
    feasible_ids = tuple(sorted(item.action_id for item in feasible_specs))
    values = _solve_values(model, mind.belief, feasible_ids)
    policy = finite_softmax(values, beta=model.beta)
    maximum = max(policy.values())
    selected_id = min(key for key, value in policy.items() if value == maximum)
    selected = next(item for item in feasible_specs if item.action_id == selected_id)
    source_id = source_by_action[selected_id]
    intent = SituatedActionIntent(
        f"r{round_index:04d}:cognitive:{mind.agent_id}:{selected_id}",
        mind.agent_id,
        selected.kind,
        selected.target_id,
        selected.message,
        () if source_id is None else (source_id,),
    )
    return SituatedCognitiveDecision(
        mind.agent_id, round_index, prior_belief or mind.belief, mind.belief,
        tuple(item.observation_id for item in admissions), tuple(item.symbol_id for item in admissions),
        feasible_ids, values, policy, selected_id,
        _goal_contributions(model, mind.belief, selected_id), intent,
    )


def simulate_situated_cognitive_round(
    model: SituatedCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
) -> SituatedCognitiveRoundResult:
    validate_situated_cognitive_state(model, story, state)
    model_by_id = {item.agent_id: item for item in model.agents}
    mind_by_id = {item.agent_id: item for item in state.minds}
    admitted_minds = {}
    decisions = []
    next_round = state.round_index + 1
    for agent_id in sorted(model_by_id):
        private = perspective_timeline(story, agent_id)
        admission = admit_situated_observations(model_by_id[agent_id], mind_by_id[agent_id], private)
        admitted_minds[agent_id] = admission.next_mind
        decisions.append(decide_situated_action(
            model_by_id[agent_id], admission.next_mind, private,
            round_index=next_round, prior_belief=admission.prior_mind.belief,
            admissions=admission.admissions,
        ))
    next_story = advance_situated_story(model.world_model, story, tuple(item.intent for item in decisions))
    latest = next_story.rounds[-1]
    event_by_actor = {item.actor_agent_id: item for item in latest.events}
    new_observed = {agent_id: [] for agent_id in model_by_id}
    for item in latest.observations:
        new_observed[item.agent_id].append(item.event_id)
    next_minds = []
    for decision in decisions:
        prior = admitted_minds[decision.agent_id]
        event = event_by_actor[decision.agent_id]
        place_id = prior.own_place_id
        if event.kind is SituatedActionKind.MOVE and event.success:
            place_id = next(item.value for item in event.details if item.name == "destination_place_id")
        next_minds.append(replace(
            prior,
            own_place_id=place_id,
            observed_event_ids=tuple(set(prior.observed_event_ids) | set(new_observed[decision.agent_id])),
            selected_action_ids=prior.selected_action_ids + (decision.selected_action_id,),
            decision_count=prior.decision_count + 1,
        ))
    next_state = SituatedCognitiveState(
        model.model_id, model.content_hash, next_round, state.content_hash,
        next_story.content_hash, tuple(next_minds),
    )
    validate_situated_cognitive_state(model, next_story, next_state)
    return SituatedCognitiveRoundResult(state, story.content_hash, tuple(decisions), next_story, next_state)


__all__ = (
    "SituatedBeliefAdmission", "SituatedBeliefAdmissionResult", "SituatedCognitiveDecision",
    "SituatedCognitiveRoundResult", "admit_situated_observations", "decide_situated_action",
    "simulate_situated_cognitive_round",
)
