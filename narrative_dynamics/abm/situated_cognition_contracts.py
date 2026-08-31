"""Immutable V11 contracts for private situated cognition."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.abm.situated import SituatedActionKind
from narrative_dynamics.abm.situated_contracts import SituatedWorldModel
from narrative_dynamics.abm.situated_story import SituatedStory
from narrative_dynamics.narrative.runtime_planning import PlanningBeliefState


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_TOLERANCE = 1e-12


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _optional_text(value: object, *, label: str) -> str | None:
    return None if value is None else _text(value, label=label)


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: object, *, label: str) -> float:
    result = _finite(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _strings(values: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{label} must be a tuple")
    result = tuple(_text(item, label=label[:-1] if label.endswith("s") else label) for item in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must be unique")
    return tuple(sorted(result))


@dataclass(frozen=True)
class SituatedHypothesis:
    hypothesis_id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "hypothesis_id", _text(self.hypothesis_id, label="situated hypothesis id"))
        object.__setattr__(self, "description", _text(self.description, label="situated hypothesis description"))

    def to_dict(self) -> dict[str, object]:
        return {"hypothesis_id": self.hypothesis_id, "description": self.description}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedObservationSymbol:
    symbol_id: str
    description: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol_id", _text(self.symbol_id, label="situated observation symbol id"))
        object.__setattr__(self, "description", _text(self.description, label="situated observation symbol description"))

    def to_dict(self) -> dict[str, object]:
        return {"symbol_id": self.symbol_id, "description": self.description}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedObservationRule:
    rule_id: str
    symbol_id: str
    likelihood_action_id: str
    event_kind: SituatedActionKind
    outcome: str | None = None
    detail_name: str | None = None
    detail_value: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "rule_id", _text(self.rule_id, label="situated observation rule id"))
        object.__setattr__(self, "symbol_id", _text(self.symbol_id, label="situated observation rule symbol id"))
        object.__setattr__(self, "likelihood_action_id", _text(self.likelihood_action_id, label="situated observation rule likelihood action id"))
        if not isinstance(self.event_kind, SituatedActionKind):
            raise TypeError("situated observation rule event kind must be SituatedActionKind")
        object.__setattr__(self, "outcome", _optional_text(self.outcome, label="situated observation rule outcome"))
        object.__setattr__(self, "detail_name", _optional_text(self.detail_name, label="situated observation rule detail name"))
        object.__setattr__(self, "detail_value", _optional_text(self.detail_value, label="situated observation rule detail value"))
        if (self.detail_name is None) != (self.detail_value is None):
            raise ValueError("situated observation rule detail name and value must be paired")

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id, "symbol_id": self.symbol_id,
            "likelihood_action_id": self.likelihood_action_id,
            "event_kind": self.event_kind.value, "outcome": self.outcome,
            "detail_name": self.detail_name, "detail_value": self.detail_value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedObservationLikelihood:
    action_id: str
    hypothesis_id: str
    symbol_id: str
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, label="likelihood action id"))
        object.__setattr__(self, "hypothesis_id", _text(self.hypothesis_id, label="likelihood hypothesis id"))
        object.__setattr__(self, "symbol_id", _text(self.symbol_id, label="likelihood symbol id"))
        object.__setattr__(self, "probability", _probability(self.probability, label="likelihood probability"))

    def to_dict(self) -> dict[str, object]:
        return {"action_id": self.action_id, "hypothesis_id": self.hypothesis_id, "symbol_id": self.symbol_id, "probability": self.probability}


@dataclass(frozen=True)
class SituatedActionSpec:
    action_id: str
    kind: SituatedActionKind
    target_id: str | None = None
    message: str | None = None
    required_place_ids: tuple[str, ...] = ()
    repeatable: bool = True
    source_event_kinds: tuple[SituatedActionKind, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _text(self.action_id, label="cognitive action id"))
        if not isinstance(self.kind, SituatedActionKind):
            raise TypeError("cognitive action kind must be SituatedActionKind")
        object.__setattr__(self, "target_id", _optional_text(self.target_id, label="cognitive action target id"))
        object.__setattr__(self, "message", _optional_text(self.message, label="cognitive action message"))
        object.__setattr__(self, "required_place_ids", _strings(self.required_place_ids, label="cognitive action required place ids"))
        if not isinstance(self.repeatable, bool):
            raise TypeError("cognitive action repeatable must be boolean")
        if not isinstance(self.source_event_kinds, tuple) or any(not isinstance(item, SituatedActionKind) for item in self.source_event_kinds):
            raise TypeError("cognitive action source event kinds must be a tuple of SituatedActionKind values")
        if len(set(self.source_event_kinds)) != len(self.source_event_kinds):
            raise ValueError("cognitive action source event kinds must be unique")
        object.__setattr__(self, "source_event_kinds", tuple(sorted(self.source_event_kinds, key=lambda item: item.value)))
        targeted = {SituatedActionKind.MOVE, SituatedActionKind.INSPECT, SituatedActionKind.TAKE, SituatedActionKind.DROP}
        if (self.kind in targeted) != (self.target_id is not None):
            raise ValueError("cognitive physical action target does not match its kind")
        if self.kind is SituatedActionKind.TELL:
            if self.message is None:
                raise ValueError("cognitive tell action requires a message")
        elif self.message is not None or self.source_event_kinds:
            raise ValueError("only cognitive tell actions may carry message/source constraints")

    def to_dict(self) -> dict[str, object]:
        return {
            "action_id": self.action_id, "kind": self.kind.value,
            "target_id": self.target_id, "message": self.message,
            "required_place_ids": list(self.required_place_ids), "repeatable": self.repeatable,
            "source_event_kinds": [item.value for item in self.source_event_kinds],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedHypothesisTransition:
    action_id: str
    prior_hypothesis_id: str
    next_hypothesis_id: str
    probability: float

    def __post_init__(self) -> None:
        for name in ("action_id", "prior_hypothesis_id", "next_hypothesis_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"transition {name}"))
        object.__setattr__(self, "probability", _probability(self.probability, label="transition probability"))

    def to_dict(self) -> dict[str, object]:
        return {"action_id": self.action_id, "prior_hypothesis_id": self.prior_hypothesis_id, "next_hypothesis_id": self.next_hypothesis_id, "probability": self.probability}


@dataclass(frozen=True)
class SituatedGoalSpec:
    goal_id: str
    description: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, label="situated goal id"))
        object.__setattr__(self, "description", _text(self.description, label="situated goal description"))
        weight = _finite(self.weight, label="situated goal weight")
        if weight < 0.0:
            raise ValueError("situated goal weight must be non-negative")
        object.__setattr__(self, "weight", weight)

    def to_dict(self) -> dict[str, object]:
        return {"goal_id": self.goal_id, "description": self.description, "weight": self.weight}


@dataclass(frozen=True)
class SituatedGoalReward:
    goal_id: str
    hypothesis_id: str
    action_id: str
    value: float

    def __post_init__(self) -> None:
        for name in ("goal_id", "hypothesis_id", "action_id"):
            object.__setattr__(self, name, _text(getattr(self, name), label=f"reward {name}"))
        object.__setattr__(self, "value", _finite(self.value, label="reward value"))

    def to_dict(self) -> dict[str, object]:
        return {"goal_id": self.goal_id, "hypothesis_id": self.hypothesis_id, "action_id": self.action_id, "value": self.value}


@dataclass(frozen=True)
class SituatedAgentCognitiveModel:
    agent_id: str
    hypotheses: tuple[SituatedHypothesis, ...]
    prior_belief: PlanningBeliefState
    observation_symbols: tuple[SituatedObservationSymbol, ...]
    observation_rules: tuple[SituatedObservationRule, ...]
    likelihoods: tuple[SituatedObservationLikelihood, ...]
    actions: tuple[SituatedActionSpec, ...]
    action_schedule: tuple[tuple[str, ...], ...]
    transitions: tuple[SituatedHypothesisTransition, ...]
    goals: tuple[SituatedGoalSpec, ...]
    rewards: tuple[SituatedGoalReward, ...]
    discount: float
    beta: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="cognitive agent id"))
        specs = (
            ("hypotheses", self.hypotheses, SituatedHypothesis, lambda item: item.hypothesis_id),
            ("observation_symbols", self.observation_symbols, SituatedObservationSymbol, lambda item: item.symbol_id),
            ("observation_rules", self.observation_rules, SituatedObservationRule, lambda item: item.rule_id),
            ("likelihoods", self.likelihoods, SituatedObservationLikelihood, lambda item: (item.action_id, item.hypothesis_id, item.symbol_id)),
            ("actions", self.actions, SituatedActionSpec, lambda item: item.action_id),
            ("transitions", self.transitions, SituatedHypothesisTransition, lambda item: (item.action_id, item.prior_hypothesis_id, item.next_hypothesis_id)),
            ("goals", self.goals, SituatedGoalSpec, lambda item: item.goal_id),
            ("rewards", self.rewards, SituatedGoalReward, lambda item: (item.goal_id, item.hypothesis_id, item.action_id)),
        )
        for name, values, expected, key in specs:
            if not isinstance(values, tuple) or any(not isinstance(item, expected) for item in values):
                raise TypeError(f"cognitive agent {name} must be a tuple of {expected.__name__} values")
            identities = tuple(key(item) for item in values)
            if len(set(identities)) != len(identities):
                raise ValueError(f"cognitive agent {name} identities must be unique")
            object.__setattr__(self, name, tuple(sorted(values, key=key)))
        if not self.hypotheses or not self.observation_symbols or not self.actions or not self.goals:
            raise ValueError("cognitive agent requires hypotheses, symbols, actions, and goals")
        if not any(item.weight > 0.0 for item in self.goals):
            raise ValueError("cognitive agent requires a positive goal weight")
        hypothesis_ids = {item.hypothesis_id for item in self.hypotheses}
        symbol_ids = {item.symbol_id for item in self.observation_symbols}
        action_ids = {item.action_id for item in self.actions}
        goal_ids = {item.goal_id for item in self.goals}
        if not isinstance(self.prior_belief, PlanningBeliefState):
            raise TypeError("cognitive prior belief must be PlanningBeliefState")
        object.__setattr__(self, "prior_belief", PlanningBeliefState(dict(self.prior_belief.probabilities)))
        if set(self.prior_belief.probabilities) != hypothesis_ids:
            raise ValueError("cognitive prior belief must cover exact hypotheses")
        if any(item.symbol_id not in symbol_ids or item.likelihood_action_id not in action_ids for item in self.observation_rules):
            raise ValueError("observation rule must reference a declared symbol and action")
        likelihood_keys = {(item.action_id, item.hypothesis_id, item.symbol_id) for item in self.likelihoods}
        if likelihood_keys != {(a, h, s) for a in action_ids for h in hypothesis_ids for s in symbol_ids}:
            raise ValueError("cognitive likelihood matrix must cover exact actions, hypotheses, and symbols")
        for action_id in action_ids:
            for hypothesis_id in hypothesis_ids:
                total = math.fsum(item.probability for item in self.likelihoods if item.action_id == action_id and item.hypothesis_id == hypothesis_id)
                if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_TOLERANCE):
                    raise ValueError("cognitive likelihood row must sum to 1")
        if not isinstance(self.action_schedule, tuple) or not self.action_schedule:
            raise ValueError("cognitive action schedule must be a non-empty tuple")
        schedule = []
        for row in self.action_schedule:
            ids = _strings(row, label="cognitive action schedule ids")
            if not ids or not set(ids).issubset(action_ids):
                raise ValueError("cognitive action schedule must use non-empty declared action subsets")
            schedule.append(ids)
        if set(schedule[0]) != action_ids:
            raise ValueError("cognitive root action schedule must cover exact actions")
        object.__setattr__(self, "action_schedule", tuple(schedule))
        if self.transitions:
            transition_keys = {(item.action_id, item.prior_hypothesis_id, item.next_hypothesis_id) for item in self.transitions}
            expected = {(a, h, n) for a in action_ids for h in hypothesis_ids for n in hypothesis_ids}
            if transition_keys != expected:
                raise ValueError("cognitive transition matrix must cover exact actions and hypotheses")
            for action_id in action_ids:
                for prior_id in hypothesis_ids:
                    total = math.fsum(item.probability for item in self.transitions if item.action_id == action_id and item.prior_hypothesis_id == prior_id)
                    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=_TOLERANCE):
                        raise ValueError("cognitive transition row must sum to 1")
        reward_keys = {(item.goal_id, item.hypothesis_id, item.action_id) for item in self.rewards}
        if reward_keys != {(g, h, a) for g in goal_ids for h in hypothesis_ids for a in action_ids}:
            raise ValueError("cognitive reward matrix must cover exact goals, hypotheses, and actions")
        object.__setattr__(self, "discount", _probability(self.discount, label="cognitive discount"))
        beta = _finite(self.beta, label="cognitive beta")
        if beta <= 0.0:
            raise ValueError("cognitive beta must be positive")
        object.__setattr__(self, "beta", beta)

    @property
    def horizon(self) -> int:
        return len(self.action_schedule)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "prior_belief": self.prior_belief.to_dict(),
            "observation_symbols": [item.to_dict() for item in self.observation_symbols],
            "observation_rules": [item.to_dict() for item in self.observation_rules],
            "likelihoods": [item.to_dict() for item in self.likelihoods],
            "actions": [item.to_dict() for item in self.actions],
            "action_schedule": [list(row) for row in self.action_schedule],
            "transitions": [item.to_dict() for item in self.transitions],
            "goals": [item.to_dict() for item in self.goals],
            "rewards": [item.to_dict() for item in self.rewards],
            "discount": self.discount, "beta": self.beta,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedCognitiveModel:
    model_id: str
    version: str
    world_model: SituatedWorldModel
    agents: tuple[SituatedAgentCognitiveModel, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated cognitive model id"))
        object.__setattr__(self, "version", _text(self.version, label="situated cognitive model version"))
        if not isinstance(self.world_model, SituatedWorldModel):
            raise TypeError("situated cognitive model world must be SituatedWorldModel")
        if not isinstance(self.agents, tuple) or any(not isinstance(item, SituatedAgentCognitiveModel) for item in self.agents):
            raise TypeError("situated cognitive model agents must be a tuple of SituatedAgentCognitiveModel values")
        ids = tuple(item.agent_id for item in self.agents)
        if len(set(ids)) != len(ids):
            raise ValueError("situated cognitive model agent ids must be unique")
        if set(ids) != {item.agent_id for item in self.world_model.agents}:
            raise ValueError("situated cognitive model must cover exact world agent roster")
        place_ids = {item.place_id for item in self.world_model.places}
        if any(not set(action.required_place_ids).issubset(place_ids) for agent in self.agents for action in agent.actions):
            raise ValueError("cognitive action required place must belong to the world")
        object.__setattr__(self, "agents", tuple(sorted(self.agents, key=lambda item: item.agent_id)))

    def to_dict(self) -> dict[str, object]:
        return {"model_id": self.model_id, "version": self.version, "world_model_hash": self.world_model.content_hash, "agents": [item.to_dict() for item in self.agents]}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedAgentMindState:
    agent_id: str
    belief: PlanningBeliefState
    own_place_id: str
    processed_observation_ids: tuple[str, ...] = ()
    observed_event_ids: tuple[str, ...] = ()
    selected_action_ids: tuple[str, ...] = ()
    decision_count: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="situated mind agent id"))
        if not isinstance(self.belief, PlanningBeliefState):
            raise TypeError("situated mind belief must be PlanningBeliefState")
        object.__setattr__(self, "belief", PlanningBeliefState(dict(self.belief.probabilities)))
        object.__setattr__(self, "own_place_id", _text(self.own_place_id, label="situated mind own place id"))
        for name in ("processed_observation_ids", "observed_event_ids"):
            object.__setattr__(self, name, _strings(getattr(self, name), label=f"situated mind {name}"))
        if not isinstance(self.selected_action_ids, tuple) or any(not isinstance(item, str) or not item.strip() for item in self.selected_action_ids):
            raise TypeError("situated mind selected action ids must be a tuple of strings")
        if not isinstance(self.decision_count, int) or isinstance(self.decision_count, bool) or self.decision_count < 0:
            raise ValueError("situated mind decision count must be non-negative")
        if len(self.selected_action_ids) != self.decision_count:
            raise ValueError("situated mind action history must match decision count")

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id, "belief": self.belief.to_dict(), "own_place_id": self.own_place_id,
            "processed_observation_ids": list(self.processed_observation_ids), "observed_event_ids": list(self.observed_event_ids),
            "selected_action_ids": list(self.selected_action_ids), "decision_count": self.decision_count,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class SituatedCognitiveState:
    model_id: str
    model_hash: str
    round_index: int
    parent_state_hash: str | None
    story_hash: str
    minds: tuple[SituatedAgentMindState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="situated cognitive state model id"))
        for name in ("model_hash", "story_hash"):
            value = getattr(self, name)
            if not isinstance(value, str) or _HASH.fullmatch(value) is None:
                raise ValueError(f"situated cognitive state {name} must be a content hash")
        if not isinstance(self.round_index, int) or isinstance(self.round_index, bool) or self.round_index < 0:
            raise ValueError("situated cognitive state round index must be non-negative")
        if self.round_index == 0:
            if self.parent_state_hash is not None:
                raise ValueError("round-zero cognitive state cannot have a parent")
        elif not isinstance(self.parent_state_hash, str) or _HASH.fullmatch(self.parent_state_hash) is None:
            raise ValueError("non-initial cognitive state requires a parent state hash")
        if not isinstance(self.minds, tuple) or any(not isinstance(item, SituatedAgentMindState) for item in self.minds):
            raise TypeError("situated cognitive state minds must be a tuple of SituatedAgentMindState values")
        ids = tuple(item.agent_id for item in self.minds)
        if len(set(ids)) != len(ids):
            raise ValueError("situated cognitive state mind ids must be unique")
        object.__setattr__(self, "minds", tuple(sorted(self.minds, key=lambda item: item.agent_id)))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id, "model_hash": self.model_hash, "round_index": self.round_index,
            "parent_state_hash": self.parent_state_hash, "story_hash": self.story_hash,
            "minds": [item.to_dict() for item in self.minds],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def initialize_situated_cognition(model: SituatedCognitiveModel, story: SituatedStory) -> SituatedCognitiveState:
    if not isinstance(model, SituatedCognitiveModel) or not isinstance(story, SituatedStory):
        raise TypeError("situated cognition initialization requires model and story")
    if story.model_id != model.world_model.model_id or story.model_hash != model.world_model.content_hash:
        raise ValueError("situated cognitive model and story must bind the exact world")
    if story.current_state.round_index != 0:
        raise ValueError("situated cognition must initialize from a round-zero story")
    places = {item.agent_id: item.place_id for item in story.current_state.agents}
    state = SituatedCognitiveState(
        model.model_id, model.content_hash, 0, None, story.content_hash,
        tuple(SituatedAgentMindState(item.agent_id, item.prior_belief, places[item.agent_id]) for item in model.agents),
    )
    validate_situated_cognitive_state(model, story, state)
    return state


def validate_situated_cognitive_state(model: SituatedCognitiveModel, story: SituatedStory, state: SituatedCognitiveState) -> None:
    if not isinstance(model, SituatedCognitiveModel) or not isinstance(story, SituatedStory) or not isinstance(state, SituatedCognitiveState):
        raise TypeError("situated cognitive state validation requires model, story, and state")
    if state.model_id != model.model_id or state.model_hash != model.content_hash:
        raise ValueError("situated cognitive state must bind the exact model")
    if state.story_hash != story.content_hash or state.round_index != story.current_state.round_index:
        raise ValueError("situated cognitive state must bind the exact current story")
    models = {item.agent_id: item for item in model.agents}
    if {item.agent_id for item in state.minds} != set(models):
        raise ValueError("situated cognitive state mind roster must cover exact model agents")
    place_ids = {item.place_id for item in model.world_model.places}
    for mind in state.minds:
        if set(mind.belief.probabilities) != {item.hypothesis_id for item in models[mind.agent_id].hypotheses}:
            raise ValueError("situated mind belief must cover exact agent hypotheses")
        if mind.own_place_id not in place_ids:
            raise ValueError("situated mind own place must belong to the world")
        declared_actions = {item.action_id for item in models[mind.agent_id].actions}
        if not set(mind.selected_action_ids).issubset(declared_actions):
            raise ValueError("situated mind action history must use declared actions")


__all__ = (
    "SituatedHypothesis", "SituatedObservationSymbol", "SituatedObservationRule",
    "SituatedObservationLikelihood", "SituatedActionSpec", "SituatedHypothesisTransition",
    "SituatedGoalSpec", "SituatedGoalReward", "SituatedAgentCognitiveModel",
    "SituatedCognitiveModel", "SituatedAgentMindState", "SituatedCognitiveState",
    "initialize_situated_cognition", "validate_situated_cognitive_state",
)
