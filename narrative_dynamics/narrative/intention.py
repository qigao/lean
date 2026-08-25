from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.decision import DecisionResolutionError
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import GenericNarrative, StateCellRef
from narrative_dynamics.narrative.uncertain import (
    UncertainBeliefModelSpec,
    UncertainBeliefState,
    uncertain_epistemic_state,
)


_PROBABILITY_TOLERANCE = 1e-12
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class IntentionalDecisionResolutionError(DecisionResolutionError):
    """An intentional model could not resolve a declared narrative decision."""


class GoalResolutionError(IntentionalDecisionResolutionError):
    """Belief-conditioned goal scoring could not produce a valid goal state."""


class ChoiceResolutionError(IntentionalDecisionResolutionError):
    """Goal-conditioned action choice could not produce a valid action policy."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _non_negative(value: object, *, label: str) -> float:
    number = _finite(value, label=label)
    if number < 0.0:
        raise ValueError(f"{label} must be non-negative")
    return number


def _positive(value: object, *, label: str) -> float:
    number = _finite(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _content_hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _CONTENT_HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _typed_cell_mapping(
    value: object,
    *,
    label: str,
    value_validator,
) -> Mapping[StateCellRef, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    frozen: dict[StateCellRef, object] = {}
    for cell, raw in value.items():
        if not isinstance(cell, StateCellRef):
            raise TypeError(f"{label} keys must be StateCellRef values")
        frozen[cell] = value_validator(raw, cell)
    return MappingProxyType(frozen)


def _freeze_instrumentality(
    value: object,
) -> Mapping[StateCellRef, Mapping[str, float]]:
    def validate_inner(raw: object, cell: StateCellRef) -> Mapping[str, float]:
        if not isinstance(raw, Mapping):
            raise TypeError("goal instrumentality cell values must be mappings")
        if not raw:
            raise ValueError("goal instrumentality cell values must be non-empty")
        frozen: dict[str, float] = {}
        for hypothesis_hash, score in raw.items():
            key = _content_hash(
                hypothesis_hash,
                label="goal instrumentality hypothesis hash",
            )
            frozen[key] = _finite(score, label="goal instrumentality value")
        return MappingProxyType(frozen)

    result = _typed_cell_mapping(
        value,
        label="goal instrumentality",
        value_validator=validate_inner,
    )
    return result  # type: ignore[return-value]


def _freeze_cell_weights(value: object) -> Mapping[StateCellRef, float]:
    result = _typed_cell_mapping(
        value,
        label="goal cell weights",
        value_validator=lambda raw, _cell: _non_negative(
            raw,
            label="goal cell weight",
        ),
    )
    return result  # type: ignore[return-value]


def _freeze_choice_values(
    value: object,
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, Mapping):
        raise TypeError("choice values must be a mapping")
    if not value:
        raise ValueError("choice values must contain at least one goal")
    frozen: dict[str, Mapping[str, float]] = {}
    for goal_id, raw_actions in value.items():
        goal = _text(goal_id, label="choice goal id")
        if not isinstance(raw_actions, Mapping):
            raise TypeError("choice goal values must be mappings")
        if not raw_actions:
            raise ValueError("choice goal must contain at least one action")
        actions: dict[str, float] = {}
        for action_id, raw_value in raw_actions.items():
            action = _text(action_id, label="choice action id")
            actions[action] = _finite(raw_value, label="choice action value")
        frozen[goal] = MappingProxyType(actions)
    return MappingProxyType(frozen)


def _freeze_vector(
    value: object,
    *,
    label: str,
    min_size: int = 1,
) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, label=f"{label} key")
        frozen[key] = _finite(raw_value, label=f"{label} value")
    if len(frozen) < min_size:
        raise ValueError(f"{label} must contain at least {min_size} entries")
    return MappingProxyType(frozen)


def _freeze_policy(
    value: object,
    *,
    label: str,
    min_size: int = 1,
    error_type: type[IntentionalDecisionResolutionError] | None = None,
) -> Mapping[str, float]:
    try:
        policy = _freeze_vector(value, label=label, min_size=min_size)
        for probability in policy.values():
            if probability < 0.0:
                raise ValueError(f"{label} values must be non-negative")
        total = math.fsum(policy.values())
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError(f"{label} must sum to 1")
        return policy
    except (TypeError, ValueError) as error:
        if error_type is None:
            raise
        raise error_type(f"{label} is invalid") from error


def _freeze_nested_policies(
    value: object,
    *,
    label: str,
) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    frozen: dict[str, Mapping[str, float]] = {}
    for raw_goal, raw_policy in value.items():
        goal = _text(raw_goal, label=f"{label} goal id")
        frozen[goal] = _freeze_policy(
            raw_policy,
            label=f"{label} for {goal}",
        )
    return MappingProxyType(frozen)


def _map_choice(policy: Mapping[str, float]) -> str:
    if not policy:
        raise ValueError("MAP projection requires a non-empty policy")
    maximum = max(policy.values())
    return min(key for key, value in policy.items() if value == maximum)


def _mapping_payload(value: Mapping[str, float]) -> dict[str, float]:
    return {key: value[key] for key in sorted(value)}


def _nested_mapping_payload(
    value: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        outer: _mapping_payload(value[outer])
        for outer in sorted(value)
    }


@dataclass(frozen=True)
class GoalSpec:
    goal_id: str
    pressure: float
    cell_weights: Mapping[StateCellRef, float]
    instrumentality: Mapping[StateCellRef, Mapping[str, float]]
    cost: float = 0.0
    risk: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "goal_id", _text(self.goal_id, label="goal id"))
        object.__setattr__(
            self,
            "pressure",
            _non_negative(self.pressure, label="goal pressure"),
        )
        object.__setattr__(self, "cost", _non_negative(self.cost, label="goal cost"))
        object.__setattr__(self, "risk", _non_negative(self.risk, label="goal risk"))
        object.__setattr__(self, "cell_weights", _freeze_cell_weights(self.cell_weights))
        object.__setattr__(
            self,
            "instrumentality",
            _freeze_instrumentality(self.instrumentality),
        )

    def to_dict(self) -> dict[str, object]:
        cells = tuple(
            sorted(
                set(self.cell_weights) | set(self.instrumentality),
                key=_cell_key,
            )
        )
        return {
            "goal_id": self.goal_id,
            "pressure": self.pressure,
            "cost": self.cost,
            "risk": self.risk,
            "cell_weights": [
                {
                    "cell": cell.to_dict(),
                    "weight": self.cell_weights.get(cell),
                }
                for cell in cells
            ],
            "instrumentality": [
                {
                    "cell": cell.to_dict(),
                    "values": (
                        None
                        if cell not in self.instrumentality
                        else {
                            key: self.instrumentality[cell][key]
                            for key in sorted(self.instrumentality[cell])
                        }
                    ),
                }
                for cell in cells
            ],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class GoalModelSpec:
    model_id: str
    version: str
    beta_goal: float
    goals: tuple[GoalSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="goal model id"))
        object.__setattr__(self, "version", _text(self.version, label="goal model version"))
        object.__setattr__(
            self,
            "beta_goal",
            _positive(self.beta_goal, label="goal model beta_goal"),
        )
        goals = tuple(self.goals)
        if len(goals) < 2:
            raise ValueError("goal model requires at least two goals")
        if any(not isinstance(goal, GoalSpec) for goal in goals):
            raise TypeError("goal model goals must be GoalSpec values")
        if len({goal.goal_id for goal in goals}) != len(goals):
            raise ValueError("goal model goal ids must be unique")
        object.__setattr__(
            self,
            "goals",
            tuple(sorted(goals, key=lambda goal: goal.goal_id)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "beta_goal": self.beta_goal,
            "goals": [goal.to_dict() for goal in self.goals],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class GoalState:
    model_id: str
    model_hash: str
    belief_state_hash: str
    scores: Mapping[str, float]
    policy: Mapping[str, float]
    selected_goal: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="goal state model id"))
        object.__setattr__(
            self,
            "model_hash",
            _content_hash(self.model_hash, label="goal state model hash"),
        )
        object.__setattr__(
            self,
            "belief_state_hash",
            _content_hash(self.belief_state_hash, label="goal state belief hash"),
        )
        scores = _freeze_vector(self.scores, label="goal scores", min_size=2)
        policy = _freeze_policy(self.policy, label="goal policy", min_size=2)
        if set(scores) != set(policy):
            raise ValueError("goal scores and policy must cover the same goals")
        selected = _text(self.selected_goal, label="selected goal")
        if selected not in policy:
            raise ValueError("selected goal must belong to the goal policy")
        if selected != _map_choice(policy):
            raise ValueError("selected goal must be the deterministic MAP goal")
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "policy", policy)
        object.__setattr__(self, "selected_goal", selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "belief_state_hash": self.belief_state_hash,
            "scores": _mapping_payload(self.scores),
            "policy": _mapping_payload(self.policy),
            "selected_goal": self.selected_goal,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ChoiceModelSpec:
    model_id: str
    version: str
    beta_action: float
    values: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="choice model id"))
        object.__setattr__(self, "version", _text(self.version, label="choice model version"))
        object.__setattr__(
            self,
            "beta_action",
            _positive(self.beta_action, label="choice model beta_action"),
        )
        object.__setattr__(self, "values", _freeze_choice_values(self.values))

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "beta_action": self.beta_action,
            "values": _nested_mapping_payload(self.values),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class IntentionalDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    belief_model: UncertainBeliefModelSpec
    goal_model: GoalModelSpec
    choice_model: ChoiceModelSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="intentional decision model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="intentional decision model version"),
        )
        decision_types = tuple(
            _text(item, label="supported intentional decision type")
            for item in self.supported_decision_types
        )
        if not decision_types:
            raise ValueError("intentional model must support at least one decision type")
        if len(set(decision_types)) != len(decision_types):
            raise ValueError("supported intentional decision types must be unique")
        if not isinstance(self.belief_model, UncertainBeliefModelSpec):
            raise TypeError("intentional model belief_model must be UncertainBeliefModelSpec")
        if not isinstance(self.goal_model, GoalModelSpec):
            raise TypeError("intentional model goal_model must be GoalModelSpec")
        if not isinstance(self.choice_model, ChoiceModelSpec):
            raise TypeError("intentional model choice_model must be ChoiceModelSpec")
        goal_ids = {goal.goal_id for goal in self.goal_model.goals}
        if set(self.choice_model.values) != goal_ids:
            raise ChoiceResolutionError(
                "choice model goals must match goal model goals exactly"
            )
        object.__setattr__(
            self,
            "supported_decision_types",
            tuple(sorted(decision_types)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "supported_decision_types": list(self.supported_decision_types),
            "belief_model_hash": self.belief_model.content_hash,
            "goal_model_hash": self.goal_model.content_hash,
            "choice_model_hash": self.choice_model.content_hash,
            "implementation_identity": measure_implementation(
                IntentionalDecisionModelSpec
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class IntentionalDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    belief_state: UncertainBeliefState
    goal_state: GoalState
    conditional_action_policies: Mapping[str, Mapping[str, float]]
    action_scores: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="intentional result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _content_hash(self.model_hash, label="intentional result model hash"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="intentional result decision id"),
        )
        if not isinstance(self.belief_state, UncertainBeliefState):
            raise TypeError("intentional result belief_state must be UncertainBeliefState")
        if not isinstance(self.goal_state, GoalState):
            raise TypeError("intentional result goal_state must be GoalState")
        expected_belief_hash = stable_content_hash(self.belief_state.to_dict())
        if self.goal_state.belief_state_hash != expected_belief_hash:
            raise ValueError("goal state must bind the exact upstream belief state")

        conditional = _freeze_nested_policies(
            self.conditional_action_policies,
            label="conditional action policies",
        )
        if set(conditional) != set(self.goal_state.policy):
            raise ValueError(
                "conditional action policies must cover the goal policy exactly"
            )
        action_sets = {frozenset(policy) for policy in conditional.values()}
        if len(action_sets) != 1:
            raise ValueError(
                "conditional action policies must cover one common action set"
            )
        action_set = next(iter(action_sets))
        if not action_set:
            raise ValueError("intentional result must contain at least one action")

        scores = _freeze_vector(self.action_scores, label="intentional action scores")
        policy = _freeze_policy(self.action_policy, label="intentional action policy")
        if set(scores) != action_set or set(policy) != action_set:
            raise ValueError(
                "action scores and final policy must match conditional actions exactly"
            )
        selected = _text(self.selected_action, label="selected intentional action")
        if selected not in policy:
            raise ValueError("selected intentional action must belong to the final policy")
        if selected != _map_choice(policy):
            raise ValueError(
                "selected intentional action must be the deterministic MAP action"
            )

        object.__setattr__(self, "conditional_action_policies", conditional)
        object.__setattr__(self, "action_scores", scores)
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_action", selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "decision_id": self.decision_id,
            "belief_state": self.belief_state.to_dict(),
            "goal_state": self.goal_state.to_dict(),
            "conditional_action_policies": _nested_mapping_payload(
                self.conditional_action_policies
            ),
            "action_scores": _mapping_payload(self.action_scores),
            "action_policy": _mapping_payload(self.action_policy),
            "selected_action": self.selected_action,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validated_derived_policy(
    value: object,
    *,
    label: str,
    expected_keys: set[str],
    error_type: type[IntentionalDecisionResolutionError],
) -> Mapping[str, float]:
    try:
        policy = _freeze_policy(
            value,
            label=label,
            min_size=1,
            error_type=error_type,
        )
    except error_type:
        raise
    if set(policy) != expected_keys:
        raise error_type(f"{label} must cover the configured alternatives exactly")
    return policy


def _goal_scores(
    decision_context: tuple[StateCellRef, ...],
    belief_state: UncertainBeliefState,
    goal_model: GoalModelSpec,
) -> dict[str, float]:
    expected_cells = set(decision_context)
    scores: dict[str, float] = {}

    for goal in goal_model.goals:
        if set(goal.cell_weights) != expected_cells:
            raise GoalResolutionError(
                "goal weights must cover decision context cells exactly"
            )
        if set(goal.instrumentality) != expected_cells:
            raise GoalResolutionError(
                "goal instrumentality must cover decision context cells exactly"
            )
        weight_total = math.fsum(goal.cell_weights.values())
        if not math.isclose(
            weight_total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise GoalResolutionError("goal cell weights must sum to 1")

        weighted_cells: list[float] = []
        for cell in sorted(decision_context, key=_cell_key):
            view = belief_state.cells.get(cell)
            if view is None:
                raise GoalResolutionError(
                    "uncertain belief state does not contain a decision context cell"
                )
            posterior = view.posterior
            hypothesis_hashes = {
                stable_content_hash(mass.value.to_dict())
                for mass in posterior.masses
            }
            configured = goal.instrumentality[cell]
            if set(configured) != hypothesis_hashes:
                raise GoalResolutionError(
                    "goal instrumentality hypotheses must match posterior hypotheses exactly"
                )
            expected_for_cell = math.fsum(
                mass.probability
                * configured[stable_content_hash(mass.value.to_dict())]
                for mass in posterior.masses
            )
            weighted_cells.append(goal.cell_weights[cell] * expected_for_cell)

        expected_instrumentality = math.fsum(weighted_cells)
        score = (
            goal.pressure * expected_instrumentality
            - goal.cost
            - goal.risk
        )
        if not math.isfinite(score):
            raise GoalResolutionError("goal score must be finite")
        scores[goal.goal_id] = score

    if len(scores) < 2:
        raise GoalResolutionError("intentional decision requires at least two usable goals")
    return {goal_id: scores[goal_id] for goal_id in sorted(scores)}


def _goal_state(
    belief_state: UncertainBeliefState,
    goal_model: GoalModelSpec,
    scores: Mapping[str, float],
) -> GoalState:
    try:
        raw_policy = finite_softmax(scores, beta=goal_model.beta_goal)
    except (TypeError, ValueError, OverflowError) as error:
        raise GoalResolutionError("goal softmax could not produce a valid policy") from error
    policy = _validated_derived_policy(
        raw_policy,
        label="goal policy",
        expected_keys=set(scores),
        error_type=GoalResolutionError,
    )
    selected = _map_choice(policy)
    return GoalState(
        model_id=goal_model.model_id,
        model_hash=goal_model.content_hash,
        belief_state_hash=stable_content_hash(belief_state.to_dict()),
        scores=scores,
        policy=policy,
        selected_goal=selected,
    )


def _conditional_action_policies(
    choice_model: ChoiceModelSpec,
    goal_policy: Mapping[str, float],
    declared_actions: set[str],
) -> dict[str, Mapping[str, float]]:
    if set(choice_model.values) != set(goal_policy):
        raise ChoiceResolutionError("choice goals must match goal policy exactly")

    result: dict[str, Mapping[str, float]] = {}
    for goal_id in sorted(goal_policy):
        values = choice_model.values[goal_id]
        if set(values) != declared_actions:
            raise ChoiceResolutionError(
                "choice actions must match declared actions exactly"
            )
        try:
            raw_policy = finite_softmax(values, beta=choice_model.beta_action)
        except (TypeError, ValueError, OverflowError) as error:
            raise ChoiceResolutionError(
                "conditional action softmax could not produce a valid policy"
            ) from error
        result[goal_id] = _validated_derived_policy(
            raw_policy,
            label=f"conditional action policy for {goal_id}",
            expected_keys=declared_actions,
            error_type=ChoiceResolutionError,
        )
    return result


def _marginal_action_policy(
    goal_policy: Mapping[str, float],
    conditional: Mapping[str, Mapping[str, float]],
    declared_actions: set[str],
) -> dict[str, float]:
    policy = {
        action_id: math.fsum(
            goal_policy[goal_id] * conditional[goal_id][action_id]
            for goal_id in sorted(goal_policy)
        )
        for action_id in sorted(declared_actions)
    }
    if any(not math.isfinite(value) or value < 0.0 for value in policy.values()):
        raise ChoiceResolutionError(
            "marginal action policy must contain finite non-negative probabilities"
        )
    total = math.fsum(policy.values())
    if not math.isclose(
        total,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ChoiceResolutionError("marginal action policy must sum to 1")

    residual = 1.0 - total
    if residual != 0.0:
        maximum = max(policy.values())
        correction_action = min(
            action_id
            for action_id, probability in policy.items()
            if probability == maximum
        )
        policy[correction_action] += residual

    if any(not math.isfinite(value) or value < 0.0 for value in policy.values()):
        raise ChoiceResolutionError(
            "marginal action residual correction must preserve non-negative mass"
        )
    corrected_total = math.fsum(policy.values())
    if not math.isclose(
        corrected_total,
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ChoiceResolutionError(
            "marginal action residual correction must preserve normalization"
        )
    return policy


def run_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: IntentionalDecisionModelSpec,
) -> IntentionalDecisionResult:
    """Run one deterministic belief -> latent-goal -> action-policy model."""

    validate_narrative(story, domain)
    if not isinstance(model, IntentionalDecisionModelSpec):
        raise IntentionalDecisionResolutionError(
            "intentional execution requires IntentionalDecisionModelSpec"
        )
    try:
        decision_id = _text(decision_id, label="decision id")
    except ValueError as error:
        raise IntentionalDecisionResolutionError(
            "decision id must be valid"
        ) from error
    decision = next((item for item in story.decisions if item.id == decision_id), None)
    if decision is None:
        raise IntentionalDecisionResolutionError(
            "decision id is not declared by the narrative"
        )
    if decision.type_name not in model.supported_decision_types:
        raise IntentionalDecisionResolutionError(
            "decision type is not supported by the intentional model"
        )
    if not decision.context_cells:
        raise IntentionalDecisionResolutionError(
            "intentional decision requires at least one context cell"
        )
    if len(set(decision.context_cells)) != len(decision.context_cells):
        raise IntentionalDecisionResolutionError(
            "intentional decision context cells must be unique"
        )
    if not decision.actions:
        raise IntentionalDecisionResolutionError(
            "intentional decision requires at least one declared action"
        )

    belief_state = uncertain_epistemic_state(
        story,
        domain,
        decision.actor_id,
        model.belief_model,
        decision.context_cells,
        at_time=decision.logical_time,
    )

    scores = _goal_scores(
        decision.context_cells,
        belief_state,
        model.goal_model,
    )
    goal_state = _goal_state(belief_state, model.goal_model, scores)

    declared_actions = {action.id for action in decision.actions}
    if len(declared_actions) != len(decision.actions):
        raise ChoiceResolutionError("declared action ids must be unique")
    conditional = _conditional_action_policies(
        model.choice_model,
        goal_state.policy,
        declared_actions,
    )

    action_scores = {
        action_id: math.fsum(
            goal_state.policy[goal_id]
            * model.choice_model.values[goal_id][action_id]
            for goal_id in sorted(goal_state.policy)
        )
        for action_id in sorted(declared_actions)
    }
    if any(not math.isfinite(value) for value in action_scores.values()):
        raise ChoiceResolutionError("diagnostic action scores must be finite")

    action_policy = _marginal_action_policy(
        goal_state.policy,
        conditional,
        declared_actions,
    )
    selected_action = _map_choice(action_policy)

    return IntentionalDecisionResult(
        model_id=model.model_id,
        model_hash=model.content_hash,
        decision_id=decision.id,
        belief_state=belief_state,
        goal_state=goal_state,
        conditional_action_policies=conditional,
        action_scores=action_scores,
        action_policy=action_policy,
        selected_action=selected_action,
    )
