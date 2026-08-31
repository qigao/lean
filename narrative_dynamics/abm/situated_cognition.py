"""Private Bayesian cognition and POMDP action selection for V10 worlds."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from types import MappingProxyType
from collections.abc import Mapping

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState
from narrative_dynamics.abm.situated import (
    ObservationChannel,
    SituatedActionIntent,
    SituatedActionKind,
)
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
    consolidated: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.consolidated, bool):
            raise TypeError("situated belief admission consolidated must be boolean")
        if self.consolidated and self.prior_belief != self.posterior_belief:
            raise ValueError("consolidated situated belief admission cannot change belief")

    def to_dict(self) -> dict[str, object]:
        result = {
            "observation_id": self.observation_id, "event_id": self.event_id,
            "symbol_id": self.symbol_id, "likelihood_action_id": self.likelihood_action_id,
            "prior_belief": self.prior_belief.to_dict(), "posterior_belief": self.posterior_belief.to_dict(),
        }
        if self.consolidated:
            result["consolidated"] = True
        return result

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
    recalled_memory_ids: tuple[str, ...] = ()
    recalled_symbol_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index <= 0:
            raise ValueError("situated cognitive decision round must be positive")
        if not isinstance(self.feasible_action_ids, tuple) or any(
            not isinstance(item, str) or not item.strip()
            for item in self.feasible_action_ids
        ):
            raise TypeError("situated decision feasible action ids must be a tuple of strings")
        if len(set(self.feasible_action_ids)) != len(self.feasible_action_ids):
            raise ValueError("situated decision feasible action ids must be unique")
        object.__setattr__(self, "feasible_action_ids", tuple(sorted(self.feasible_action_ids)))
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
        for name in ("recalled_memory_ids", "recalled_symbol_ids"):
            items = getattr(self, name)
            if not isinstance(items, tuple) or any(
                not isinstance(item, str) or not item.strip() for item in items
            ):
                raise TypeError(f"situated decision {name.replace('_', ' ')} must be a tuple of strings")
            if len(set(items)) != len(items):
                raise ValueError(f"situated decision {name.replace('_', ' ')} must be unique")
        object.__setattr__(self, "action_values", values)
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_goal_contributions", _freeze_float_map(self.selected_goal_contributions, label="situated decision goal contributions"))

    def to_dict(self) -> dict[str, object]:
        payload = {
            "agent_id": self.agent_id, "round_index": self.round_index,
            "prior_belief": self.prior_belief.to_dict(), "posterior_belief": self.posterior_belief.to_dict(),
            "admitted_observation_ids": list(self.admitted_observation_ids), "admitted_symbol_ids": list(self.admitted_symbol_ids),
            "feasible_action_ids": list(self.feasible_action_ids), "action_values": dict(self.action_values),
            "action_policy": dict(self.action_policy), "selected_action_id": self.selected_action_id,
            "selected_goal_contributions": dict(self.selected_goal_contributions), "intent": self.intent.to_dict(),
        }
        if self.recalled_memory_ids or self.recalled_symbol_ids:
            payload["recalled_memory_ids"] = list(self.recalled_memory_ids)
            payload["recalled_symbol_ids"] = list(self.recalled_symbol_ids)
        return payload

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
        if set(ids) != {item.agent_id for item in self.prior_state.minds}:
            raise ValueError("cognitive round decisions must cover exact prior mind roster")
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


@dataclass(frozen=True)
class SituatedDecisionExplanation:
    agent_id: str
    round_index: int
    selected_action_id: str
    most_likely_hypothesis_id: str
    hypothesis_probability: float
    action_probability: float
    expected_value: float
    goal_contributions: Mapping[str, float]
    admitted_evidence_ids: tuple[str, ...]
    recalled_memory_ids: tuple[str, ...] = ()
    recalled_symbol_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("hypothesis_probability", "action_probability"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"situated explanation {name} must be in [0, 1]")
        object.__setattr__(self, "goal_contributions", _freeze_float_map(self.goal_contributions, label="situated explanation goal contributions"))

    def to_dict(self) -> dict[str, object]:
        payload = {
            "agent_id": self.agent_id, "round_index": self.round_index,
            "selected_action_id": self.selected_action_id,
            "most_likely_hypothesis_id": self.most_likely_hypothesis_id,
            "hypothesis_probability": self.hypothesis_probability,
            "action_probability": self.action_probability, "expected_value": self.expected_value,
            "goal_contributions": dict(self.goal_contributions),
            "admitted_evidence_ids": list(self.admitted_evidence_ids),
        }
        if self.recalled_memory_ids or self.recalled_symbol_ids:
            payload["recalled_memory_ids"] = list(self.recalled_memory_ids)
            payload["recalled_symbol_ids"] = list(self.recalled_symbol_ids)
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedCognitiveTrajectory:
    model_id: str
    model_hash: str
    initial_story: SituatedStory
    initial_state: SituatedCognitiveState
    rounds: tuple[SituatedCognitiveRoundResult, ...]
    final_story: SituatedStory
    final_state: SituatedCognitiveState

    def __post_init__(self) -> None:
        if not self.rounds:
            raise ValueError("situated cognitive trajectory requires at least one round")
        if (
            self.initial_state.model_id != self.model_id
            or self.initial_state.model_hash != self.model_hash
            or self.final_state.model_id != self.model_id
            or self.final_state.model_hash != self.model_hash
        ):
            raise ValueError("situated cognitive trajectory must bind exact model identity")
        story = self.initial_story
        state = self.initial_state
        for item in self.rounds:
            if item.prior_story_hash != story.content_hash or item.prior_state != state:
                raise ValueError("situated cognitive trajectory chain is discontinuous")
            story = item.next_story
            state = item.next_state
        if story != self.final_story or state != self.final_state:
            raise ValueError("situated cognitive trajectory final values must equal chain tail")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id, "model_hash": self.model_hash,
            "initial_story_hash": self.initial_story.content_hash,
            "initial_state": self.initial_state.to_dict(),
            "rounds": [item.to_dict() for item in self.rounds],
            "final_story_hash": self.final_story.content_hash,
            "final_state": self.final_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def match_situated_observation_rule(
    model: SituatedAgentCognitiveModel,
    item: SituatedPerspectiveEvent,
):
    """Return the unique cognitive rule matched by one private observation."""
    if (
        item.event.kind is SituatedActionKind.TELL
        and item.observation.channel is ObservationChannel.SELF
    ):
        return None
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
    *,
    _claim_topic_by_symbol: Mapping[str, str] | None = None,
    _consolidated_claim_keys: frozenset[tuple[str, str, str]] | None = None,
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
    claim_topic_by_symbol = {} if _claim_topic_by_symbol is None else dict(_claim_topic_by_symbol)
    consolidated_claim_keys = set(
        () if _consolidated_claim_keys is None else _consolidated_claim_keys
    )
    active_claim_symbols = {}
    for source_id, topic_id, symbol_id in consolidated_claim_keys:
        scope = (source_id, topic_id)
        existing = active_claim_symbols.get(scope)
        if existing is not None and existing != symbol_id:
            raise ValueError("situated direct admission claim scope has multiple active symbols")
        active_claim_symbols[scope] = symbol_id
    likelihoods = {(item.action_id, item.hypothesis_id, item.symbol_id): item.probability for item in model.likelihoods}
    admissions = []
    new_items = [
        item
        for item in perspective
        if item.event.round_index > mind.observation_floor_round
        and item.observation.observation_id not in processed
    ]
    for item in new_items:
        processed.add(item.observation.observation_id)
        observed.add(item.event.event_id)
        rule = match_situated_observation_rule(model, item)
        if rule is None:
            continue
        claim_scope = None
        if (
            item.event.kind is SituatedActionKind.TELL
            and item.event.actor_agent_id != mind.agent_id
            and rule.symbol_id in claim_topic_by_symbol
        ):
            claim_scope = (
                item.event.actor_agent_id,
                claim_topic_by_symbol[rule.symbol_id],
            )
        consolidated = (
            claim_scope is not None
            and active_claim_symbols.get(claim_scope) == rule.symbol_id
        )
        if consolidated:
            admissions.append(SituatedBeliefAdmission(
                item.observation.observation_id,
                item.event.event_id,
                rule.symbol_id,
                rule.likelihood_action_id,
                belief,
                belief,
                True,
            ))
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
        if claim_scope is not None:
            active_claim_symbols[claim_scope] = rule.symbol_id
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
    recalled_memory_ids: tuple[str, ...] = (),
    recalled_symbol_ids: tuple[str, ...] = (),
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
        recalled_memory_ids, recalled_symbol_ids,
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
    return _advance_situated_cognitive_round(
        model,
        story,
        state,
        tuple(decisions),
        admitted_minds,
    )


def _advance_situated_cognitive_round(
    model: SituatedCognitiveModel,
    story: SituatedStory,
    state: SituatedCognitiveState,
    decisions: tuple[SituatedCognitiveDecision, ...],
    admitted_minds: Mapping[str, SituatedAgentMindState],
) -> SituatedCognitiveRoundResult:
    """Synchronously commit already-computed private cognitive decisions."""

    model_by_id = {item.agent_id: item for item in model.agents}
    if set(admitted_minds) != set(model_by_id):
        raise ValueError("cognitive commit minds must cover exact agent roster")
    next_round = state.round_index + 1
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


def explain_situated_decision(
    result: SituatedCognitiveRoundResult,
    agent_id: str,
) -> SituatedDecisionExplanation:
    if not isinstance(result, SituatedCognitiveRoundResult):
        raise TypeError("situated decision explanation requires a cognitive round")
    try:
        decision = next(item for item in result.decisions if item.agent_id == agent_id)
    except StopIteration as error:
        raise ValueError("explanation agent must have a decision in the round") from error
    maximum = max(decision.posterior_belief.probabilities.values())
    hypothesis_id = min(
        key for key, value in decision.posterior_belief.probabilities.items()
        if value == maximum
    )
    return SituatedDecisionExplanation(
        agent_id=agent_id,
        round_index=decision.round_index,
        selected_action_id=decision.selected_action_id,
        most_likely_hypothesis_id=hypothesis_id,
        hypothesis_probability=maximum,
        action_probability=decision.action_policy[decision.selected_action_id],
        expected_value=decision.action_values[decision.selected_action_id],
        goal_contributions=decision.selected_goal_contributions,
        admitted_evidence_ids=decision.admitted_observation_ids,
        recalled_memory_ids=decision.recalled_memory_ids,
        recalled_symbol_ids=decision.recalled_symbol_ids,
    )


def simulate_situated_cognition(
    model: SituatedCognitiveModel,
    initial_story: SituatedStory,
    initial_state: SituatedCognitiveState,
    *,
    round_count: int,
) -> SituatedCognitiveTrajectory:
    if not isinstance(round_count, int) or isinstance(round_count, bool) or round_count <= 0:
        raise ValueError("situated cognitive simulation requires a positive round count")
    validate_situated_cognitive_state(model, initial_story, initial_state)
    story = initial_story
    state = initial_state
    rounds = []
    for _ in range(round_count):
        result = simulate_situated_cognitive_round(model, story, state)
        rounds.append(result)
        story = result.next_story
        state = result.next_state
    return SituatedCognitiveTrajectory(
        model.model_id, model.content_hash, initial_story, initial_state,
        tuple(rounds), story, state,
    )


__all__ = (
    "SituatedBeliefAdmission", "SituatedBeliefAdmissionResult", "SituatedCognitiveDecision",
    "SituatedCognitiveRoundResult", "SituatedDecisionExplanation", "SituatedCognitiveTrajectory",
    "match_situated_observation_rule", "admit_situated_observations", "decide_situated_action",
    "simulate_situated_cognitive_round", "explain_situated_decision",
    "simulate_situated_cognition",
)
