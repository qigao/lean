from __future__ import annotations

from collections.abc import Mapping
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


_BENCHMARK_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "narrative-synthetic-identification-v1"
_FAMILIES = ("reactive", "intentional", "planning")
_NAMES = {
    family: f"generic-narrative-identification-{family}"
    for family in _FAMILIES
}
_ACTIONS = ("scout", "escape", "submit")
_TERMINAL_ACTIONS = ("escape", "submit")
_REQUIRED_SCENARIO_FIELDS = frozenset(
    {
        "prior_weak",
        "memory_evidence_available",
        "memory_signal",
        "memory_signal_accuracy",
        "future_information_available",
        "future_signal_accuracy",
        "guard_persistence",
        "escape_reward",
        "capture_cost",
        "submit_reward",
        "scout_cost",
        "discount",
        "goal_score_gap",
        "action_value_gap",
        "mode",
        "latent_structure",
    }
)
_ALLOWED_MODE_STRUCTURES = frozenset(
    {
        ("latent_temperature", "identical"),
        ("latent_temperature", "mirror"),
        ("memory_evidence", "belief_driven"),
        ("future_information", "value_of_information"),
    }
)
_INTENTIONAL_PARAMETERS = frozenset(
    {
        "beta_goal",
        "beta_action",
        "goal_pressure_scale",
        "instrumentality_scale",
    }
)
_SIMPLE_PARAMETERS = frozenset({"beta_action"})


@dataclass(frozen=True)
class _ScenarioValues:
    prior_weak: float
    memory_evidence_available: bool
    memory_signal: str
    memory_signal_accuracy: float
    future_information_available: bool
    future_signal_accuracy: float
    guard_persistence: float
    escape_reward: float
    capture_cost: float
    submit_reward: float
    scout_cost: float
    discount: float
    goal_score_gap: float
    action_value_gap: float
    mode: str
    latent_structure: str


@dataclass(frozen=True)
class _IdentificationParameters:
    beta_action: float
    beta_goal: float = 1.0
    goal_pressure_scale: float = 1.0
    instrumentality_scale: float = 1.0


@dataclass(frozen=True)
class _IdentificationCase:
    values: _ScenarioValues
    domain: DomainSpec
    story: GenericNarrative
    decision_id: str
    ledger: RuntimeEvidenceLedger
    guard_cell: StateCellRef
    phase_cell: StateCellRef
    signal_cell: StateCellRef


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _positive_number(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number <= 0.0:
        raise ValueError(f"{label} must be positive")
    return number


def _probability(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _boolean(value: object, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be boolean")
    return value


def _scenario_values(scenario: object) -> _ScenarioValues:
    if not isinstance(scenario, Scenario):
        raise TypeError("narrative identification benchmark requires Scenario")
    payload = scenario.payload
    if set(payload) != _REQUIRED_SCENARIO_FIELDS:
        raise ValueError(
            "narrative identification scenario must contain exactly its declared fields"
        )
    mode = payload["mode"]
    structure = payload["latent_structure"]
    if not isinstance(mode, str) or not isinstance(structure, str):
        raise TypeError("narrative identification mode and latent structure must be strings")
    if (mode, structure) not in _ALLOWED_MODE_STRUCTURES:
        raise ValueError("narrative identification mode/latent structure is unsupported")

    prior_weak = _probability(payload["prior_weak"], label="prior weak probability")
    memory_accuracy = _probability(
        payload["memory_signal_accuracy"],
        label="memory signal accuracy",
    )
    future_accuracy = _probability(
        payload["future_signal_accuracy"],
        label="future signal accuracy",
    )
    if memory_accuracy < 0.5 or future_accuracy < 0.5:
        raise ValueError("identification signal accuracies must be at least 0.5")
    persistence = _probability(
        payload["guard_persistence"],
        label="guard persistence",
    )
    discount = _probability(payload["discount"], label="discount")

    memory_signal = payload["memory_signal"]
    if memory_signal not in ("none", "weak", "strong"):
        raise ValueError("memory signal must be none, weak, or strong")
    if mode == "memory_evidence" and memory_signal == "none":
        raise ValueError("memory-evidence mode requires a diagnostic memory signal")

    escape_reward = _finite_number(payload["escape_reward"], label="escape reward")
    capture_cost = _finite_number(payload["capture_cost"], label="capture cost")
    submit_reward = _finite_number(payload["submit_reward"], label="submit reward")
    scout_cost = _finite_number(payload["scout_cost"], label="scout cost")
    goal_gap = _finite_number(payload["goal_score_gap"], label="goal score gap")
    action_gap = _positive_number(payload["action_value_gap"], label="action value gap")
    if (
        escape_reward < 0.0
        or capture_cost < 0.0
        or scout_cost < 0.0
        or goal_gap < 0.0
    ):
        raise ValueError(
            "escape reward, capture/scout costs, and goal score gap must be non-negative"
        )

    return _ScenarioValues(
        prior_weak=prior_weak,
        memory_evidence_available=_boolean(
            payload["memory_evidence_available"],
            label="memory evidence availability",
        ),
        memory_signal=memory_signal,
        memory_signal_accuracy=memory_accuracy,
        future_information_available=_boolean(
            payload["future_information_available"],
            label="future information availability",
        ),
        future_signal_accuracy=future_accuracy,
        guard_persistence=persistence,
        escape_reward=escape_reward,
        capture_cost=capture_cost,
        submit_reward=submit_reward,
        scout_cost=scout_cost,
        discount=discount,
        goal_score_gap=goal_gap,
        action_value_gap=action_gap,
        mode=mode,
        latent_structure=structure,
    )


def _parameters(
    family: str,
    values: Mapping[str, float],
) -> _IdentificationParameters:
    if not isinstance(values, Mapping):
        raise TypeError("narrative identification parameters must be a mapping")
    expected = _INTENTIONAL_PARAMETERS if family == "intentional" else _SIMPLE_PARAMETERS
    if set(values) != expected:
        raise ValueError(
            f"narrative identification {family} parameters must match the family schema exactly"
        )
    beta_action = _positive_number(values["beta_action"], label="beta_action")
    if family != "intentional":
        return _IdentificationParameters(beta_action=beta_action)
    return _IdentificationParameters(
        beta_action=beta_action,
        beta_goal=_positive_number(values["beta_goal"], label="beta_goal"),
        goal_pressure_scale=_positive_number(
            values["goal_pressure_scale"],
            label="goal pressure scale",
        ),
        instrumentality_scale=_positive_number(
            values["instrumentality_scale"],
            label="instrumentality scale",
        ),
    )


class _SeedGuardStateHook:
    def __call__(self, prior_state, event):
        guard = event.arguments["guard"].value
        if not isinstance(guard, EntityRef):
            raise TypeError("seed guard argument must contain EntityRef")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    guard.entity_id,
                    "guard.status",
                    event.arguments["status"],
                ),
            )
        )


class _SeedPrisonStateHook:
    def __call__(self, prior_state, event):
        prisoner = event.arguments["prisoner"].value
        if not isinstance(prisoner, EntityRef):
            raise TypeError("seed prisoner argument must contain EntityRef")
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    prisoner.entity_id,
                    "prisoner.episode_phase",
                    event.arguments["phase"],
                ),
                StateDeltaOp(
                    "set",
                    prisoner.entity_id,
                    "prisoner.signal",
                    event.arguments["signal"],
                ),
            )
        )


def _build_identification_case(
    scenario: Scenario,
    values: _ScenarioValues,
) -> _IdentificationCase:
    domain = DomainSpec(
        domain_id="narrative-prison-identification",
        version=_BENCHMARK_VERSION,
        entity_types=(EntityTypeSpec("Guard"), EntityTypeSpec("Prisoner")),
        value_types=(
            ValueTypeSpec("GuardRef", "entity_ref", entity_type="Guard"),
            ValueTypeSpec("PrisonerRef", "entity_ref", entity_type="Prisoner"),
            ValueTypeSpec("GuardStatus", "enum", ("weak", "strong")),
            ValueTypeSpec("EpisodePhase", "enum", ("active", "terminal")),
            ValueTypeSpec("Signal", "enum", ("none", "clear", "alarm")),
        ),
        state_variables=(
            StateVariableSpec("guard.status", "Guard", "GuardStatus"),
            StateVariableSpec(
                "prisoner.episode_phase",
                "Prisoner",
                "EpisodePhase",
            ),
            StateVariableSpec("prisoner.signal", "Prisoner", "Signal"),
        ),
        event_types=(
            EventTypeSpec(
                "seed-guard-state",
                None,
                (
                    ParameterSpec("guard", "GuardRef"),
                    ParameterSpec("status", "GuardStatus"),
                ),
                (StateEffectSpec("guard.status", "guard"),),
                "seed-guard-state",
            ),
            EventTypeSpec(
                "seed-prison-state",
                None,
                (
                    ParameterSpec("prisoner", "PrisonerRef"),
                    ParameterSpec("phase", "EpisodePhase"),
                    ParameterSpec("signal", "Signal"),
                ),
                (
                    StateEffectSpec("prisoner.episode_phase", "prisoner"),
                    StateEffectSpec("prisoner.signal", "prisoner"),
                ),
                "seed-prison-state",
            ),
        ),
        action_types=(ActionTypeSpec("prison-action", ()),),
        decision_types=(
            DecisionTypeSpec("prison-choice", "Prisoner", "prison-action"),
        ),
        semantic_hooks=(
            SemanticHookBinding(
                "seed-guard-state",
                "author fixed objective guard state",
                _SeedGuardStateHook(),
            ),
            SemanticHookBinding(
                "seed-prison-state",
                "author deterministic prison identification state",
                _SeedPrisonStateHook(),
            ),
        ),
    )
    prisoner = Entity("prisoner", "Prisoner")
    guard = Entity("guard", "Guard")
    guard_cell = StateCellRef(EntityRef(guard.id, guard.type_name), "guard.status")
    phase_cell = StateCellRef(
        EntityRef(prisoner.id, prisoner.type_name),
        "prisoner.episode_phase",
    )
    signal_cell = StateCellRef(
        EntityRef(prisoner.id, prisoner.type_name),
        "prisoner.signal",
    )
    objective_guard = (
        values.memory_signal
        if values.mode == "memory_evidence"
        else "weak"
    )
    guard_seed = NarrativeEvent(
        "seed-guard-state",
        0,
        "seed-guard-state",
        None,
        {
            "guard": TypedValue(
                "GuardRef",
                EntityRef(guard.id, guard.type_name),
            ),
            "status": TypedValue("GuardStatus", objective_guard),
        },
    )
    prison_seed = NarrativeEvent(
        "seed-prison-state",
        1,
        "seed-prison-state",
        None,
        {
            "prisoner": TypedValue(
                "PrisonerRef",
                EntityRef(prisoner.id, prisoner.type_name),
            ),
            "phase": TypedValue("EpisodePhase", "active"),
            "signal": TypedValue("Signal", "none"),
        },
    )
    action_ids = _ACTIONS if values.mode == "future_information" else _TERMINAL_ACTIONS
    decision = Decision(
        "prison-choice",
        2,
        prisoner.id,
        "prison-choice",
        (guard_cell, phase_cell, signal_cell),
        tuple(ActionOption(action, "prison-action", {}) for action in action_ids),
    )
    observations = [
        Observation("observe-prison-seed", prisoner.id, prison_seed.id),
    ]
    if values.mode == "memory_evidence" and values.memory_evidence_available:
        observations.append(
            Observation("observe-memory-signal", prisoner.id, guard_seed.id)
        )
    story = GenericNarrative(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        entities=(prisoner, guard),
        events=(guard_seed, prison_seed),
        observations=tuple(observations),
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
    return _IdentificationCase(
        values=values,
        domain=domain,
        story=story,
        decision_id=decision.id,
        ledger=ledger,
        guard_cell=guard_cell,
        phase_cell=phase_cell,
        signal_cell=signal_cell,
    )


class _SeedPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        if cell.state_variable == "guard.status":
            prior_weak = float(parameters["prior_weak"])
            return {
                stable_content_hash(hypothesis.to_dict()): (
                    prior_weak if hypothesis.value == "weak" else 1.0 - prior_weak
                )
                for hypothesis in hypotheses
            }
        mass = 1.0 / len(hypotheses)
        return {
            stable_content_hash(hypothesis.to_dict()): mass
            for hypothesis in hypotheses
        }


class _SeedLikelihoodHook:
    def __call__(self, agent_id, evidence, hypotheses, parameters):
        if evidence.relation == "clear":
            return {
                stable_content_hash(hypothesis.to_dict()): 1.0
                for hypothesis in hypotheses
            }
        if evidence.relation != "equals" or evidence.value is None:
            raise ValueError("unsupported identification authored evidence")
        if evidence.cell.state_variable == "guard.status":
            accuracy = float(parameters["memory_signal_accuracy"])
            return {
                stable_content_hash(hypothesis.to_dict()): (
                    accuracy
                    if hypothesis == evidence.value
                    else 1.0 - accuracy
                )
                for hypothesis in hypotheses
            }
        return {
            stable_content_hash(hypothesis.to_dict()): (
                1.0 if hypothesis == evidence.value else 0.0
            )
            for hypothesis in hypotheses
        }


class _RuntimeLikelihoodHook:
    def __call__(self, agent_id, percept_view, hypotheses, parameters):
        if percept_view.relation == "clear":
            return {
                stable_content_hash(hypothesis.to_dict()): 1.0
                for hypothesis in hypotheses
            }
        if percept_view.relation != "equals" or percept_view.value is None:
            raise ValueError("unsupported identification runtime percept")
        return {
            stable_content_hash(hypothesis.to_dict()): (
                1.0 if hypothesis == percept_view.value else 0.0
            )
            for hypothesis in hypotheses
        }


def _build_belief_model(case: _IdentificationCase) -> RuntimeBeliefModelSpec:
    seed_model = UncertainBeliefModelSpec(
        "generic-narrative-identification-seed-belief",
        _BENCHMARK_VERSION,
        {
            "prior_weak": case.values.prior_weak,
            "memory_signal_accuracy": case.values.memory_signal_accuracy,
        },
        _SeedPriorHook(),
        _SeedLikelihoodHook(),
    )
    return RuntimeBeliefModelSpec(
        "generic-narrative-identification-runtime-belief",
        _BENCHMARK_VERSION,
        seed_model,
        {},
        _RuntimeLikelihoodHook(),
    )


def _typed_value_hash(type_name: str, value: object) -> str:
    return stable_content_hash(TypedValue(type_name, value).to_dict())


class _ReactiveScoreHook:
    def __call__(self, context):
        parameters = context.parameters
        mode = str(parameters["mode"])
        if mode == "future_information":
            prior_weak = float(parameters["prior_weak"])
            direct_escape = (
                prior_weak * float(parameters["escape_reward"])
                - (1.0 - prior_weak) * float(parameters["capture_cost"])
            )
            return {
                "escape": direct_escape,
                "submit": float(parameters["submit_reward"]),
                "scout": -float(parameters["scout_cost"]),
            }
        gap = float(parameters["action_value_gap"])
        return {
            "escape": gap / 2.0,
            "submit": -gap / 2.0,
        }


def _build_reactive_model(
    case: _IdentificationCase,
    parameters: _IdentificationParameters,
) -> RuntimeReactiveDecisionModelSpec:
    values = case.values
    return RuntimeReactiveDecisionModelSpec(
        "generic-narrative-identification-reactive",
        _BENCHMARK_VERSION,
        ("prison-choice",),
        (case.guard_cell,),
        {
            "mode": values.mode,
            "prior_weak": values.prior_weak,
            "escape_reward": values.escape_reward,
            "capture_cost": values.capture_cost,
            "submit_reward": values.submit_reward,
            "scout_cost": values.scout_cost,
            "action_value_gap": values.action_value_gap,
        },
        parameters.beta_action,
        _ReactiveScoreHook(),
    )


def _zero_instrumentality(case: _IdentificationCase):
    return {
        case.phase_cell: {
            _typed_value_hash("EpisodePhase", "active"): 0.0,
            _typed_value_hash("EpisodePhase", "terminal"): 0.0,
        },
        case.signal_cell: {
            _typed_value_hash("Signal", "none"): 0.0,
            _typed_value_hash("Signal", "clear"): 0.0,
            _typed_value_hash("Signal", "alarm"): 0.0,
        },
    }


def _goal_guard_instrumentality(
    values: _ScenarioValues,
    parameters: _IdentificationParameters,
) -> tuple[dict[str, float], dict[str, float]]:
    weak_hash = _typed_value_hash("GuardStatus", "weak")
    strong_hash = _typed_value_hash("GuardStatus", "strong")
    if values.mode == "latent_temperature":
        half_gap = (
            values.goal_score_gap
            * parameters.instrumentality_scale
            / 2.0
        )
        return (
            {weak_hash: half_gap, strong_hash: half_gap},
            {weak_hash: -half_gap, strong_hash: -half_gap},
        )
    if values.mode == "memory_evidence":
        scale = parameters.instrumentality_scale
        return (
            {weak_hash: scale, strong_hash: -scale},
            {weak_hash: -scale, strong_hash: scale},
        )
    return (
        {weak_hash: 0.0, strong_hash: 0.0},
        {weak_hash: 0.0, strong_hash: 0.0},
    )


def _intentional_choice_values(
    values: _ScenarioValues,
) -> dict[str, dict[str, float]]:
    if values.mode == "future_information":
        direct_escape = (
            values.prior_weak * values.escape_reward
            - (1.0 - values.prior_weak) * values.capture_cost
        )
        immediate = {
            "escape": direct_escape,
            "submit": values.submit_reward,
            "scout": -values.scout_cost,
        }
        return {
            "freedom": dict(immediate),
            "safety": dict(immediate),
        }
    half_gap = values.action_value_gap / 2.0
    freedom = {"escape": half_gap, "submit": -half_gap}
    if values.latent_structure == "identical":
        return {
            "freedom": dict(freedom),
            "safety": dict(freedom),
        }
    return {
        "freedom": freedom,
        "safety": {"escape": -half_gap, "submit": half_gap},
    }


def _build_intentional_model(
    case: _IdentificationCase,
    parameters: _IdentificationParameters,
) -> RuntimeIntentionalDecisionModelSpec:
    cell_weights = {
        case.guard_cell: 1.0,
        case.phase_cell: 0.0,
        case.signal_cell: 0.0,
    }
    zeros = _zero_instrumentality(case)
    freedom_guard, safety_guard = _goal_guard_instrumentality(
        case.values,
        parameters,
    )
    freedom = GoalSpec(
        "freedom",
        parameters.goal_pressure_scale,
        cell_weights,
        {
            case.guard_cell: freedom_guard,
            case.phase_cell: zeros[case.phase_cell],
            case.signal_cell: zeros[case.signal_cell],
        },
    )
    safety = GoalSpec(
        "safety",
        parameters.goal_pressure_scale,
        cell_weights,
        {
            case.guard_cell: safety_guard,
            case.phase_cell: zeros[case.phase_cell],
            case.signal_cell: zeros[case.signal_cell],
        },
    )
    goal_model = GoalModelSpec(
        "generic-narrative-identification-goals",
        _BENCHMARK_VERSION,
        parameters.beta_goal,
        (freedom, safety),
    )
    choice_model = ChoiceModelSpec(
        "generic-narrative-identification-choice",
        _BENCHMARK_VERSION,
        parameters.beta_action,
        _intentional_choice_values(case.values),
    )
    return RuntimeIntentionalDecisionModelSpec(
        "generic-narrative-identification-intentional",
        _BENCHMARK_VERSION,
        ("prison-choice",),
        _build_belief_model(case),
        goal_model,
        choice_model,
    )


def _state_parts(state_id: str) -> tuple[str, str]:
    guard, phase = state_id.split("-", 1)
    return guard, phase


class _ProductJointBeliefHook:
    def __call__(self, context):
        result: dict[str, float] = {}
        for state in context.hidden_states:
            probability = 1.0
            for cell, distribution in context.posterior.items():
                probability *= distribution.probability_of(state.cells[cell])
            result[state.state_id] = probability
        return result


class _PlanningTransitionHook:
    def __call__(self, context):
        persistence = float(context.parameters["guard_persistence"])
        guard, phase = _state_parts(context.state.state_id)
        result = {
            state.state_id: 0.0
            for state in context.candidate_next_states
        }
        if phase == "terminal":
            result[context.state.state_id] = 1.0
            return result
        if context.depth == 0:
            if context.action.id == "scout":
                result[context.state.state_id] = 1.0
                return result
            if context.action.id in _TERMINAL_ACTIONS:
                result[f"{guard}-terminal"] = 1.0
                return result
        if context.depth == 1:
            if context.action.id == "submit":
                result[f"{guard}-terminal"] = 1.0
                return result
            if context.action.id == "escape":
                other = "strong" if guard == "weak" else "weak"
                result[f"{guard}-terminal"] = persistence
                result[f"{other}-terminal"] = 1.0 - persistence
                return result
        raise ValueError("unsupported narrative identification planning transition")


class _PlanningObservationHook:
    def __call__(self, context):
        accuracy = float(context.parameters["signal_accuracy"])
        guard, phase = _state_parts(context.next_state.state_id)
        result = {
            observation.observation_id: 0.0
            for observation in context.observations
        }
        if context.depth == 0 and context.action.id == "scout" and phase == "active":
            clear = accuracy if guard == "weak" else 1.0 - accuracy
            result["clear"] = clear
            result["alarm"] = 1.0 - clear
            return result
        result["none"] = 1.0
        return result


class _PlanningRewardHook:
    def __call__(self, context):
        parameters = context.parameters
        mode = str(parameters["mode"])
        guard, phase = _state_parts(context.state.state_id)
        next_guard, _ = _state_parts(context.next_state.state_id)
        if phase == "terminal":
            return 0.0

        if mode == "latent_temperature":
            gap = float(parameters["action_value_gap"])
            if context.action.id == "escape":
                return gap / 2.0
            if context.action.id == "submit":
                return -gap / 2.0
            raise ValueError("latent-temperature planning does not support scout")

        if context.depth == 0:
            if context.action.id == "scout":
                return -float(parameters["scout_cost"])
            if context.action.id == "submit":
                return float(parameters["submit_reward"])
            if context.action.id == "escape":
                return (
                    float(parameters["escape_reward"])
                    if guard == "weak"
                    else -float(parameters["capture_cost"])
                )
        if context.depth == 1:
            if context.action.id == "submit":
                return float(parameters["submit_reward"])
            if context.action.id == "escape":
                return (
                    float(parameters["escape_reward"])
                    if next_guard == "weak"
                    else -float(parameters["capture_cost"])
                )
        raise ValueError("unsupported narrative identification planning reward")


def _build_planning_model(
    case: _IdentificationCase,
    parameters: _IdentificationParameters,
) -> RuntimePlanningDecisionModelSpec:
    values = case.values
    states = tuple(
        PlanningHiddenState(
            f"{guard}-{phase}",
            {
                case.guard_cell: TypedValue("GuardStatus", guard),
                case.phase_cell: TypedValue("EpisodePhase", phase),
            },
        )
        for guard in ("weak", "strong")
        for phase in ("active", "terminal")
    )
    observations = tuple(
        PlanningObservation(
            signal,
            {case.signal_cell: TypedValue("Signal", signal)},
        )
        for signal in ("clear", "alarm", "none")
    )
    schedule = (
        (_ACTIONS, _TERMINAL_ACTIONS)
        if values.mode == "future_information"
        else (_TERMINAL_ACTIONS,)
    )
    signal_accuracy = (
        values.future_signal_accuracy
        if values.future_information_available
        else 0.5
    )
    return RuntimePlanningDecisionModelSpec(
        "generic-narrative-identification-planning",
        _BENCHMARK_VERSION,
        ("prison-choice",),
        (case.guard_cell, case.phase_cell),
        (case.signal_cell,),
        _build_belief_model(case),
        states,
        observations,
        schedule,
        values.discount,
        parameters.beta_action,
        {
            "mode": values.mode,
            "guard_persistence": values.guard_persistence,
            "signal_accuracy": signal_accuracy,
            "escape_reward": values.escape_reward,
            "capture_cost": values.capture_cost,
            "submit_reward": values.submit_reward,
            "scout_cost": values.scout_cost,
            "action_value_gap": values.action_value_gap,
        },
        _ProductJointBeliefHook(),
        _PlanningTransitionHook(),
        _PlanningObservationHook(),
        _PlanningRewardHook(),
    )


def _build_runtime_model(
    family: str,
    case: _IdentificationCase,
    parameters: _IdentificationParameters,
):
    builder = {
        "reactive": _build_reactive_model,
        "intentional": _build_intentional_model,
        "planning": _build_planning_model,
    }[family]
    return builder(case, parameters)


@dataclass(frozen=True)
class _NarrativePrisonIdentificationModel:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError("narrative identification family is unsupported")

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    def simulate(self, scenario, parameters, rng) -> ModelRun:
        values = _scenario_values(scenario)
        parsed = _parameters(self.family, parameters)
        case = _build_identification_case(scenario, values)
        nested = _build_runtime_model(self.family, case, parsed)
        dispatch = run_runtime_decision(
            case.story,
            case.domain,
            case.decision_id,
            case.ledger,
            RuntimeDecisionModelSpec(self.family, nested),
        )
        initial_policy = {action: 0.0 for action in _ACTIONS}
        initial_policy.update(
            {
                action: float(probability)
                for action, probability in dispatch.action_policy.items()
            }
        )
        return ModelRun(
            events=(),
            outcome={
                "model_kind": self.family,
                "initial_policy": initial_policy,
                "selected_action": dispatch.selected_action,
                "runtime_dispatch_hash": dispatch.content_hash,
                "runtime_dispatch": dispatch.to_dict(),
                "translated_story_hash": case.story.content_hash,
                "translated_domain_hash": case.domain.content_hash,
                "runtime_ledger_hash": case.ledger.content_hash,
            },
        )


@dataclass(frozen=True)
class NarrativePrisonIdentificationSource:
    family: str

    def __post_init__(self) -> None:
        if self.family not in _FAMILIES:
            raise ValueError(
                "narrative identification family must be reactive, intentional, or planning"
            )

    @property
    def name(self) -> str:
        return _NAMES[self.family]

    @property
    def version(self) -> str:
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self) -> str:
        return _IMPLEMENTATION_REVISION

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def instantiate(self) -> _NarrativePrisonIdentificationModel:
        return _NarrativePrisonIdentificationModel(self.family)

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
                NarrativePrisonIdentificationSource
            ).manifest_identity(),
            "model_implementation_identity": measure_implementation(
                _NarrativePrisonIdentificationModel
            ).manifest_identity(),
            "scenario_builder_implementation_identity": measure_implementation(
                _build_identification_case
            ).manifest_identity(),
            "family_builder_implementation_identity": measure_implementation(
                family_builder
            ).manifest_identity(),
            "dispatch_implementation_identity": measure_implementation(
                run_runtime_decision
            ).manifest_identity(),
        }


def create_narrative_identification_reactive_source() -> NarrativePrisonIdentificationSource:
    return NarrativePrisonIdentificationSource("reactive")


def create_narrative_identification_intentional_source() -> NarrativePrisonIdentificationSource:
    return NarrativePrisonIdentificationSource("intentional")


def create_narrative_identification_planning_source() -> NarrativePrisonIdentificationSource:
    return NarrativePrisonIdentificationSource("planning")


__all__ = [
    "NarrativePrisonIdentificationSource",
    "create_narrative_identification_intentional_source",
    "create_narrative_identification_planning_source",
    "create_narrative_identification_reactive_source",
]
