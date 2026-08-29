from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    DomainSpec,
    EntityTypeSpec,
    EventTypeSpec,
    ParameterSpec,
    SemanticHookBinding,
    StateDelta,
    StateDeltaOp,
    StateEffectSpec,
    StateVariableSpec,
    ValueTypeSpec,
    validate_narrative,
)
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    GoalModelSpec,
    GoalSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.runtime_cognition import RuntimeBeliefModelSpec
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
)
from narrative_dynamics.narrative.runtime_perception import (
    RuntimeEvidenceLedger,
    runtime_evidence_ledger_from_story,
)
from narrative_dynamics.narrative.runtime_planning import (
    PlanningHiddenState,
    PlanningObservation,
    RuntimePlanningDecisionModelSpec,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
)
from narrative_dynamics.narrative.uncertain import UncertainBeliefModelSpec


_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "feher-hare-two-stage-adapter-v1"
_FAMILIES = ("reactive", "intentional", "planning")
_NAMES = {
    "reactive": "narrative-two-stage-reactive",
    "intentional": "narrative-two-stage-intentional",
    "planning": "narrative-two-stage-planning",
}
_ACTIONS = ("action_0", "action_1")
_TASK_VARIANTS = frozenset({"magic_carpet", "spaceship"})
_HISTORY_FIELDS = frozenset(
    {
        "trial_id",
        "first_stage_action",
        "transition_common",
        "final_state",
        "second_stage_action",
        "reward",
    }
)
_CUES = ("neutral", "favor_action_0", "favor_action_1")


@dataclass(frozen=True)
class _HistoryTrial:
    trial_id: int
    first_stage_action: str
    transition_common: bool
    final_state: str
    second_stage_action: str
    reward: int


@dataclass(frozen=True)
class _ScenarioValues:
    task_variant: str
    first_stage_configuration: tuple[tuple[str, object], ...]
    history: tuple[_HistoryTrial, ...]


@dataclass(frozen=True)
class _TwoStageCase:
    domain: DomainSpec
    story: GenericNarrative
    decision_id: str
    ledger: RuntimeEvidenceLedger
    cue_cell: StateCellRef
    reward_cells: tuple[StateCellRef, StateCellRef]
    stage_cell: StateCellRef


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _beta(value: object) -> float:
    beta = _finite_number(value, label="two-stage inverse temperature beta")
    if beta <= 0.0:
        raise ValueError("two-stage inverse temperature beta must be positive")
    return beta


def _memory_decay(value: object) -> float:
    decay = _finite_number(value, label="two-stage memory decay")
    if decay <= 0.0 or decay > 1.0:
        raise ValueError("two-stage memory decay must be in (0, 1]")
    return decay


def _configuration(
    task_variant: str,
    value: object,
) -> tuple[tuple[str, object], ...]:
    if isinstance(value, Mapping):
        rows = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        rows = tuple(value)
    else:
        raise TypeError(
            "two-stage first-stage configuration must be a mapping or pair sequence"
        )

    parsed: dict[str, object] = {}
    for row in rows:
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or len(row) != 2
        ):
            raise ValueError("two-stage first-stage configuration rows must be pairs")
        raw_key, raw_value = row
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError("two-stage first-stage configuration keys must be text")
        if raw_key in parsed:
            raise ValueError("two-stage first-stage configuration keys must be unique")
        parsed[raw_key] = raw_value

    keys = set(parsed)
    # Retain the original canonical adapter fixture shape as an accepted alias.
    if keys == set(_ACTIONS):
        positions = tuple(parsed[action] for action in _ACTIONS)
        if any(not isinstance(position, str) or not position for position in positions):
            raise ValueError(
                "two-stage canonical first-stage positions must be non-empty text"
            )
        return tuple((key, parsed[key]) for key in sorted(parsed))

    if task_variant == "magic_carpet":
        expected = {"action_0_position", "action_1_position"}
        if keys != expected:
            raise ValueError(
                "magic-carpet first-stage configuration must bind both action positions"
            )
        positions = tuple(parsed[key] for key in sorted(expected))
        if any(position not in ("left", "right") for position in positions):
            raise ValueError("magic-carpet action positions must be left or right")
        if len(set(positions)) != 2:
            raise ValueError("magic-carpet action positions must be distinct")
        return tuple((key, parsed[key]) for key in sorted(parsed))

    if task_variant == "spaceship":
        expected = {"symbol0", "symbol1"}
        if keys != expected:
            raise ValueError(
                "spaceship first-stage configuration must bind both displayed symbols"
            )
        symbols = tuple(parsed[key] for key in sorted(expected))
        if any(
            isinstance(symbol, bool)
            or not isinstance(symbol, int)
            or symbol not in (0, 1)
            for symbol in symbols
        ):
            raise ValueError("spaceship displayed symbols must be binary integers")
        if len(set(symbols)) != 2:
            raise ValueError("spaceship displayed symbols must be distinct")
        # Upstream re-encodes the observed first-stage choice by its common
        # destination.  In that canonical action space action_0 commonly reaches
        # state_0 and action_1 commonly reaches state_1 regardless of symbol order.
        return tuple((key, parsed[key]) for key in sorted(parsed))

    raise ValueError("unsupported two-stage task variant")


def _history(value: object) -> tuple[_HistoryTrial, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("two-stage history must be a sequence")
    rows: list[_HistoryTrial] = []
    previous_trial_id: int | None = None
    for raw in value:
        if not isinstance(raw, Mapping) or set(raw) != _HISTORY_FIELDS:
            raise ValueError("two-stage history rows must use the canonical retained-trial schema")
        trial_id = raw["trial_id"]
        if not isinstance(trial_id, int) or isinstance(trial_id, bool) or trial_id < 0:
            raise ValueError("two-stage history trial id must be a non-negative integer")
        if previous_trial_id is not None and trial_id <= previous_trial_id:
            raise ValueError("two-stage history trial ids must be strictly increasing")
        previous_trial_id = trial_id
        first_stage_action = raw["first_stage_action"]
        second_stage_action = raw["second_stage_action"]
        final_state = raw["final_state"]
        if first_stage_action not in _ACTIONS or second_stage_action not in _ACTIONS:
            raise ValueError("two-stage history actions must be canonical")
        if final_state not in ("state_0", "state_1"):
            raise ValueError("two-stage history final state must be canonical")
        transition_common = raw["transition_common"]
        if not isinstance(transition_common, bool):
            raise TypeError("two-stage history transition_common must be boolean")
        reward = raw["reward"]
        if reward not in (0, 1) or isinstance(reward, bool):
            raise ValueError("two-stage history reward must be binary")
        rows.append(
            _HistoryTrial(
                trial_id=trial_id,
                first_stage_action=first_stage_action,
                transition_common=transition_common,
                final_state=final_state,
                second_stage_action=second_stage_action,
                reward=reward,
            )
        )
    return tuple(rows)


def _scenario_values(scenario: object) -> _ScenarioValues:
    if not isinstance(scenario, Scenario):
        raise TypeError("narrative two-stage adapter requires Scenario")
    payload = scenario.payload
    if set(payload) != {"task_variant", "first_stage_configuration", "history"}:
        raise ValueError("two-stage scenario must contain exactly the preregistered visible fields")
    task_variant = payload["task_variant"]
    if task_variant not in _TASK_VARIANTS:
        raise ValueError("unsupported two-stage task variant")
    return _ScenarioValues(
        task_variant=task_variant,
        first_stage_configuration=_configuration(
            task_variant,
            payload["first_stage_configuration"],
        ),
        history=_history(payload["history"]),
    )


def _parameters(family: str, parameters: object) -> tuple[float, float | None]:
    if not isinstance(parameters, Mapping):
        raise TypeError("two-stage model parameters must be a mapping")
    expected = {"beta"} if family == "reactive" else {"beta", "memory_decay"}
    if set(parameters) != expected:
        raise ValueError(f"two-stage {family} parameters must be exactly {sorted(expected)}")
    beta = _beta(parameters["beta"])
    decay = None if family == "reactive" else _memory_decay(parameters["memory_decay"])
    return beta, decay


def _latest_reactive_cue(history: tuple[_HistoryTrial, ...]) -> str:
    if not history:
        return "neutral"
    latest = history[-1]
    favored = latest.first_stage_action
    if latest.reward == 0:
        favored = "action_1" if favored == "action_0" else "action_0"
    return f"favor_{favored}"


def _first_stage_reward_beliefs(
    history: tuple[_HistoryTrial, ...],
    decay: float,
) -> dict[str, float]:
    beliefs: dict[str, float] = {}
    newest_first = tuple(reversed(history))
    for action in _ACTIONS:
        weighted_observations = 0.0
        weighted_rewards = 0.0
        for lag, trial in enumerate(newest_first):
            if trial.first_stage_action != action:
                continue
            weight = decay**lag
            weighted_observations += weight
            weighted_rewards += weight * trial.reward
        beliefs[action] = (1.0 + weighted_rewards) / (2.0 + weighted_observations)
    return beliefs


def _second_stage_reward_beliefs(
    history: tuple[_HistoryTrial, ...],
    decay: float,
) -> dict[str, float]:
    beliefs: dict[str, float] = {}
    newest_first = tuple(reversed(history))
    for state in ("state_0", "state_1"):
        for action in _ACTIONS:
            weighted_observations = 0.0
            weighted_rewards = 0.0
            for lag, trial in enumerate(newest_first):
                if trial.final_state != state or trial.second_stage_action != action:
                    continue
                weight = decay**lag
                weighted_observations += weight
                weighted_rewards += weight * trial.reward
            beliefs[f"{state}.{action}"] = (1.0 + weighted_rewards) / (
                2.0 + weighted_observations
            )
    return beliefs


class _SeedTwoStageStateHook:
    def __call__(self, prior_state, event):
        agent = event.arguments["agent"].value
        if not isinstance(agent, EntityRef):
            raise TypeError("two-stage seed agent must contain EntityRef")
        return StateDelta(
            (
                StateDeltaOp("set", agent.entity_id, "agent.reactive_cue", event.arguments["cue"]),
                StateDeltaOp("set", agent.entity_id, "agent.reward_action_0", event.arguments["reward_action_0"]),
                StateDeltaOp("set", agent.entity_id, "agent.reward_action_1", event.arguments["reward_action_1"]),
                StateDeltaOp("set", agent.entity_id, "agent.stage2_state", event.arguments["stage2_state"]),
            )
        )


def _build_case(family: str, values: _ScenarioValues) -> _TwoStageCase:
    domain = DomainSpec(
        domain_id="feher-hare-two-stage-v1",
        version=_VERSION,
        entity_types=(EntityTypeSpec("Agent"),),
        value_types=(
            ValueTypeSpec("AgentRef", "entity_ref", entity_type="Agent"),
            ValueTypeSpec("ReactiveCue", "enum", _CUES),
            ValueTypeSpec("RewardBelief", "bool"),
            ValueTypeSpec("Stage2State", "enum", ("state_0", "state_1")),
        ),
        state_variables=(
            StateVariableSpec("agent.reactive_cue", "Agent", "ReactiveCue"),
            StateVariableSpec("agent.reward_action_0", "Agent", "RewardBelief"),
            StateVariableSpec("agent.reward_action_1", "Agent", "RewardBelief"),
            StateVariableSpec("agent.stage2_state", "Agent", "Stage2State"),
        ),
        event_types=(
            EventTypeSpec(
                "seed-two-stage-state",
                None,
                (
                    ParameterSpec("agent", "AgentRef"),
                    ParameterSpec("cue", "ReactiveCue"),
                    ParameterSpec("reward_action_0", "RewardBelief"),
                    ParameterSpec("reward_action_1", "RewardBelief"),
                    ParameterSpec("stage2_state", "Stage2State"),
                ),
                (
                    StateEffectSpec("agent.reactive_cue", "agent"),
                    StateEffectSpec("agent.reward_action_0", "agent"),
                    StateEffectSpec("agent.reward_action_1", "agent"),
                    StateEffectSpec("agent.stage2_state", "agent"),
                ),
                "seed-two-stage-state",
            ),
        ),
        action_types=(ActionTypeSpec("two-stage-action", ()),),
        decision_types=(
            DecisionTypeSpec("two-stage-choice", "Agent", "two-stage-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "seed-two-stage-state",
                "seed preregistered two-stage model-visible state",
                _SeedTwoStageStateHook(),
            ),
        ),
    )
    agent = Entity("agent", "Agent")
    agent_ref = EntityRef(agent.id, agent.type_name)
    cue_cell = StateCellRef(agent_ref, "agent.reactive_cue")
    reward_cells = (
        StateCellRef(agent_ref, "agent.reward_action_0"),
        StateCellRef(agent_ref, "agent.reward_action_1"),
    )
    stage_cell = StateCellRef(agent_ref, "agent.stage2_state")
    seed = NarrativeEvent(
        "seed-two-stage-state",
        0,
        "seed-two-stage-state",
        None,
        {
            "agent": TypedValue("AgentRef", agent_ref),
            "cue": TypedValue("ReactiveCue", _latest_reactive_cue(values.history)),
            "reward_action_0": TypedValue("RewardBelief", False),
            "reward_action_1": TypedValue("RewardBelief", False),
            "stage2_state": TypedValue("Stage2State", "state_0"),
        },
    )
    context_cells = {
        "reactive": (cue_cell,),
        "intentional": reward_cells,
        "planning": (stage_cell,),
    }[family]
    decision = Decision(
        "two-stage-choice",
        1,
        agent.id,
        "two-stage-choice",
        context_cells,
        tuple(ActionOption(action, "two-stage-action", {}) for action in _ACTIONS),
    )
    story = GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(agent,),
        events=(seed,),
        observations=(Observation("observe-two-stage-seed", agent.id, seed.id),),
        claims=(),
        receptions=(),
        decisions=(decision,),
    )
    validate_narrative(story, domain)
    ledger = runtime_evidence_ledger_from_story(
        story,
        domain,
        at_time=decision.logical_time,
    )
    return _TwoStageCase(
        domain=domain,
        story=story,
        decision_id=decision.id,
        ledger=ledger,
        cue_cell=cue_cell,
        reward_cells=reward_cells,
        stage_cell=stage_cell,
    )


class _ReactiveScoreHook:
    def __call__(self, context):
        cue = next(iter(context.cues.values()))
        if cue.status != "resolved" or cue.value is None:
            raise ValueError("two-stage reactive cue must be resolved")
        value = cue.value.value
        if value == "neutral":
            return {"action_0": 0.0, "action_1": 0.0}
        if value == "favor_action_0":
            return {"action_0": 1.0, "action_1": 0.0}
        if value == "favor_action_1":
            return {"action_0": 0.0, "action_1": 1.0}
        raise ValueError("unsupported two-stage reactive cue")


def _build_reactive_model(
    case: _TwoStageCase,
    beta: float,
    first_stage_beliefs: Mapping[str, float],
    second_stage_beliefs: Mapping[str, float],
) -> RuntimeReactiveDecisionModelSpec:
    return RuntimeReactiveDecisionModelSpec(
        _NAMES["reactive"],
        _VERSION,
        ("two-stage-choice",),
        (case.cue_cell,),
        {"rule": "latest-retained-choice-reward-only"},
        beta,
        _ReactiveScoreHook(),
    )


class _SeedPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        if cell.state_variable == "agent.reward_action_0":
            probability_true = float(parameters["reward_action_0"])
            return {
                stable_content_hash(hypothesis.to_dict()): (
                    probability_true if hypothesis.value is True else 1.0 - probability_true
                )
                for hypothesis in hypotheses
            }
        if cell.state_variable == "agent.reward_action_1":
            probability_true = float(parameters["reward_action_1"])
            return {
                stable_content_hash(hypothesis.to_dict()): (
                    probability_true if hypothesis.value is True else 1.0 - probability_true
                )
                for hypothesis in hypotheses
            }
        if cell.state_variable == "agent.stage2_state":
            return {
                stable_content_hash(hypothesis.to_dict()): 0.5
                for hypothesis in hypotheses
            }
        raise ValueError("unsupported two-stage belief cell")


class _SeedLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        return {
            stable_content_hash(hypothesis.to_dict()): 1.0
            for hypothesis in hypotheses
        }


class _RuntimeLikelihoodHook:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        return {
            stable_content_hash(hypothesis.to_dict()): 1.0
            for hypothesis in hypotheses
        }


def _build_belief_model(
    first_stage_beliefs: Mapping[str, float],
) -> RuntimeBeliefModelSpec:
    seed = UncertainBeliefModelSpec(
        "two-stage-seed-belief",
        _VERSION,
        {
            "reward_action_0": float(first_stage_beliefs["action_0"]),
            "reward_action_1": float(first_stage_beliefs["action_1"]),
        },
        _SeedPriorHook(),
        _SeedLikelihoodHook(),
    )
    return RuntimeBeliefModelSpec(
        "two-stage-runtime-belief",
        _VERSION,
        seed,
        {},
        _RuntimeLikelihoodHook(),
    )


def _typed_value_hash(type_name: str, value: object) -> str:
    return stable_content_hash(TypedValue(type_name, value).to_dict())


def _build_intentional_model(
    case: _TwoStageCase,
    beta: float,
    first_stage_beliefs: Mapping[str, float],
    second_stage_beliefs: Mapping[str, float],
) -> RuntimeIntentionalDecisionModelSpec:
    false_hash = _typed_value_hash("RewardBelief", False)
    true_hash = _typed_value_hash("RewardBelief", True)
    reward_0, reward_1 = case.reward_cells
    weights = {reward_0: 0.5, reward_1: 0.5}
    goal_0 = GoalSpec(
        "prefer_action_0",
        1.0,
        weights,
        {
            reward_0: {false_hash: -1.0, true_hash: 1.0},
            reward_1: {false_hash: 1.0, true_hash: -1.0},
        },
    )
    goal_1 = GoalSpec(
        "prefer_action_1",
        1.0,
        weights,
        {
            reward_0: {false_hash: 1.0, true_hash: -1.0},
            reward_1: {false_hash: -1.0, true_hash: 1.0},
        },
    )
    goal_model = GoalModelSpec(
        "two-stage-reward-goals",
        _VERSION,
        beta,
        (goal_0, goal_1),
    )
    choice_model = ChoiceModelSpec(
        "two-stage-first-stage-choice",
        _VERSION,
        beta,
        {
            "prefer_action_0": {"action_0": 1.0, "action_1": 0.0},
            "prefer_action_1": {"action_0": 0.0, "action_1": 1.0},
        },
    )
    return RuntimeIntentionalDecisionModelSpec(
        _NAMES["intentional"],
        _VERSION,
        ("two-stage-choice",),
        _build_belief_model(first_stage_beliefs),
        goal_model,
        choice_model,
    )


class _PlanningJointBeliefHook:
    def __call__(self, context):
        result: dict[str, float] = {}
        for state in context.hidden_states:
            probability = 1.0
            for cell, typed_value in state.cells.items():
                probability *= context.posterior[cell].probability_of(typed_value)
            result[state.state_id] = probability
        return result


class _PlanningTransitionHook:
    def __call__(self, context):
        result = {state.state_id: 0.0 for state in context.candidate_next_states}
        if context.depth == 0:
            if context.action.id == "action_0":
                result["state_0"] = 0.7
                result["state_1"] = 0.3
                return result
            if context.action.id == "action_1":
                result["state_0"] = 0.3
                result["state_1"] = 0.7
                return result
        if context.depth == 1:
            result[context.state.state_id] = 1.0
            return result
        raise ValueError("unsupported two-stage planning transition")


class _PlanningObservationHook:
    def __call__(self, context):
        return {
            observation.observation_id: (1.0 if observation.observation_id == "none" else 0.0)
            for observation in context.observations
        }


class _PlanningRewardHook:
    def __call__(self, context):
        if context.depth == 0:
            return 0.0
        if context.depth == 1:
            return float(
                context.parameters[
                    f"reward_{context.state.state_id}_{context.action.id}"
                ]
            )
        raise ValueError("unsupported two-stage planning reward depth")


def _build_planning_model(
    case: _TwoStageCase,
    beta: float,
    first_stage_beliefs: Mapping[str, float],
    second_stage_beliefs: Mapping[str, float],
) -> RuntimePlanningDecisionModelSpec:
    states = tuple(
        PlanningHiddenState(
            state,
            {case.stage_cell: TypedValue("Stage2State", state)},
        )
        for state in ("state_0", "state_1")
    )
    parameters = {
        f"reward_{key.replace('.', '_')}": float(value)
        for key, value in second_stage_beliefs.items()
    }
    return RuntimePlanningDecisionModelSpec(
        _NAMES["planning"],
        _VERSION,
        ("two-stage-choice",),
        (case.stage_cell,),
        (),
        _build_belief_model(first_stage_beliefs),
        states,
        (PlanningObservation("none", {}),),
        (_ACTIONS, _ACTIONS),
        1.0,
        beta,
        parameters,
        _PlanningJointBeliefHook(),
        _PlanningTransitionHook(),
        _PlanningObservationHook(),
        _PlanningRewardHook(),
    )


def _build_runtime_model(
    family: str,
    case: _TwoStageCase,
    beta: float,
    first_stage_beliefs: Mapping[str, float],
    second_stage_beliefs: Mapping[str, float],
):
    builder = {
        "reactive": _build_reactive_model,
        "intentional": _build_intentional_model,
        "planning": _build_planning_model,
    }[family]
    return builder(case, beta, first_stage_beliefs, second_stage_beliefs)


@dataclass(frozen=True)
class _NarrativeTwoStageModel:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("unsupported narrative two-stage family")

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    def simulate(self, scenario, parameters, rng) -> ModelRun:
        values = _scenario_values(scenario)
        beta, decay = _parameters(self.family, parameters)
        memory_decay = 1.0 if decay is None else decay
        first_stage_beliefs = _first_stage_reward_beliefs(values.history, memory_decay)
        second_stage_beliefs = _second_stage_reward_beliefs(values.history, memory_decay)
        case = _build_case(self.family, values)
        nested = _build_runtime_model(
            self.family,
            case,
            beta,
            first_stage_beliefs,
            second_stage_beliefs,
        )
        dispatch = run_runtime_decision(
            case.story,
            case.domain,
            case.decision_id,
            case.ledger,
            RuntimeDecisionModelSpec(self.family, nested),
        )
        dispatch_payload = dispatch.to_dict()
        if self.family == "intentional":
            dispatch_payload["parameters"] = {
                "beta_goal": beta,
                "beta_action": beta,
                "memory_decay": memory_decay,
            }
        elif self.family == "planning":
            dispatch_payload["parameters"] = {
                "beta": beta,
                "memory_decay": memory_decay,
                "discount": 1.0,
            }
        else:
            dispatch_payload["parameters"] = {"beta": beta}

        outcome: dict[str, object] = {
            "family": self.family,
            "task_variant": values.task_variant,
            "first_stage_policy": dict(dispatch.action_policy),
            "selected_action": dispatch.selected_action,
            "runtime_dispatch_hash": dispatch.content_hash,
            "runtime_dispatch": dispatch_payload,
            "translated_story_hash": case.story.content_hash,
            "translated_domain_hash": case.domain.content_hash,
            "runtime_ledger_hash": case.ledger.content_hash,
        }
        if self.family == "intentional":
            outcome["intentional_reward_beliefs"] = first_stage_beliefs
        if self.family == "planning":
            outcome["planning_transition"] = {
                "common_probability": 0.7,
                "rare_probability": 0.3,
            }
            outcome["planning_reward_beliefs"] = second_stage_beliefs
            outcome["planning_discount"] = 1.0
        return ModelRun(events=(), outcome=outcome)


@dataclass(frozen=True)
class NarrativeTwoStageModelSource:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("two-stage family must be reactive, intentional, or planning")

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def instantiate(self) -> _NarrativeTwoStageModel:
        return _NarrativeTwoStageModel(self.family)

    def manifest_identity(self) -> dict[str, object]:
        family_builder = {
            "reactive": _build_reactive_model,
            "intentional": _build_intentional_model,
            "planning": _build_planning_model,
        }[self.family]
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "implementation_revision": self.implementation_revision,
            "lifecycle": self.lifecycle,
            "source_implementation_identity": measure_implementation(
                NarrativeTwoStageModelSource
            ).manifest_identity(),
            "model_implementation_identity": measure_implementation(
                _NarrativeTwoStageModel
            ).manifest_identity(),
            "case_builder_implementation_identity": measure_implementation(
                _build_case
            ).manifest_identity(),
            "family_builder_implementation_identity": measure_implementation(
                family_builder
            ).manifest_identity(),
            "dispatch_implementation_identity": measure_implementation(
                run_runtime_decision
            ).manifest_identity(),
        }


def create_narrative_two_stage_reactive_source() -> NarrativeTwoStageModelSource:
    return NarrativeTwoStageModelSource("reactive")


def create_narrative_two_stage_intentional_source() -> NarrativeTwoStageModelSource:
    return NarrativeTwoStageModelSource("intentional")


def create_narrative_two_stage_planning_source() -> NarrativeTwoStageModelSource:
    return NarrativeTwoStageModelSource("planning")


__all__ = [
    "NarrativeTwoStageModelSource",
    "create_narrative_two_stage_intentional_source",
    "create_narrative_two_stage_planning_source",
    "create_narrative_two_stage_reactive_source",
]
