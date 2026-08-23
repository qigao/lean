from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random

from narrative_dynamics.contracts import ModelRun, TraceEvent
from narrative_dynamics.model_contract import ModelContract, ModelSchema
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel


_MODEL_NAME = "finite-prison-reactive"
_MODEL_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "prison-reactive-v1"
_INITIAL_ACTIONS = ("scout", "escape", "submit")
_TERMINAL_ACTIONS = ("escape", "submit")
_SIGNALS = ("clear", "alarm")


@dataclass(frozen=True)
class _ScenarioValues:
    prior_weak: float
    signal_accuracy: float
    guard_persistence: float
    escape_reward: float
    capture_cost: float
    submit_reward: float
    scout_cost: float
    discount: float
    horizon: int


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


def _probability(value: object, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _scenario_values(scenario: object) -> _ScenarioValues:
    payload = getattr(scenario, "payload", None)
    if not isinstance(payload, Mapping):
        raise TypeError("reactive prison scenario payload must be a mapping")
    required = {
        "prior_weak",
        "signal_accuracy",
        "guard_persistence",
        "escape_reward",
        "capture_cost",
        "submit_reward",
        "scout_cost",
        "discount",
        "horizon",
    }
    if set(payload) != required:
        raise ValueError(
            "reactive prison scenario must contain exactly its declared fields"
        )

    prior = _probability(payload["prior_weak"], label="prior weak probability")
    accuracy = _probability(payload["signal_accuracy"], label="signal accuracy")
    if accuracy < 0.5:
        raise ValueError("signal accuracy must be at least 0.5")
    persistence = _probability(
        payload["guard_persistence"], label="guard persistence"
    )
    escape_reward = _finite_number(payload["escape_reward"], label="escape reward")
    capture_cost = _finite_number(payload["capture_cost"], label="capture cost")
    submit_reward = _finite_number(payload["submit_reward"], label="submit reward")
    scout_cost = _finite_number(payload["scout_cost"], label="scout cost")
    if escape_reward < 0.0 or capture_cost < 0.0 or scout_cost < 0.0:
        raise ValueError("escape reward and prison costs must be non-negative")
    discount = _probability(payload["discount"], label="discount")
    horizon = payload["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("reactive prison horizon must be an integer")
    if horizon not in (1, 2):
        raise ValueError("reactive prison horizon must be 1 or 2")
    return _ScenarioValues(
        prior_weak=prior,
        signal_accuracy=accuracy,
        guard_persistence=persistence,
        escape_reward=escape_reward,
        capture_cost=capture_cost,
        submit_reward=submit_reward,
        scout_cost=scout_cost,
        discount=discount,
        horizon=horizon,
    )


def _beta(parameters: Mapping[str, float]) -> float:
    if set(parameters) != {"beta"}:
        raise ValueError("reactive prison model requires exactly one beta parameter")
    beta = _finite_number(parameters["beta"], label="inverse temperature beta")
    if beta <= 0.0:
        raise ValueError("inverse temperature beta must be positive")
    return beta


def _softmax(
    values: Mapping[str, float],
    *,
    beta: float,
    order: tuple[str, ...],
) -> dict[str, float]:
    if set(values) != set(order):
        raise ValueError("reactive softmax values must match the declared action order")
    numeric = {
        action: _finite_number(values[action], label=f"{action} action value")
        for action in order
    }
    maximum = max(numeric.values())
    masses = {
        action: math.exp(beta * (numeric[action] - maximum)) for action in order
    }
    total = sum(masses.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("reactive prison policy mass must be positive and finite")
    policy = {action: masses[action] / total for action in order}
    pivot = max(order, key=masses.__getitem__)
    policy[pivot] = 1.0 - sum(
        policy[action] for action in order if action != pivot
    )
    if any(
        not math.isfinite(probability) or probability < 0.0
        for probability in policy.values()
    ):
        raise ValueError("reactive prison policy must contain non-negative finite mass")
    return policy


def _sample(
    policy: Mapping[str, float],
    *,
    order: tuple[str, ...],
    rng: random.Random,
) -> tuple[str, float]:
    draw = rng.random()
    cumulative = 0.0
    selected = order[-1]
    for action in order:
        cumulative += float(policy[action])
        if draw < cumulative:
            selected = action
            break
    return selected, draw


def _direct_values(values: _ScenarioValues) -> dict[str, float]:
    escape = (
        values.prior_weak * values.escape_reward
        - (1.0 - values.prior_weak) * values.capture_cost
    )
    return {"escape": escape, "submit": values.submit_reward}


def _cue_values(values: _ScenarioValues) -> dict[str, dict[str, float]]:
    direct = _direct_values(values)
    cue_strength = 2.0 * values.signal_accuracy - 1.0
    cue_scale = (values.escape_reward + values.capture_cost) / 2.0
    cue_delta = cue_strength * cue_scale
    return {
        "clear": {
            "escape": direct["escape"] + cue_delta,
            "submit": values.submit_reward,
        },
        "alarm": {
            "escape": direct["escape"] - cue_delta,
            "submit": values.submit_reward,
        },
    }


def _signal_probability(values: _ScenarioValues, signal: str) -> float:
    clear = (
        values.prior_weak * values.signal_accuracy
        + (1.0 - values.prior_weak) * (1.0 - values.signal_accuracy)
    )
    return clear if signal == "clear" else 1.0 - clear


def _cue_policies(
    values: _ScenarioValues,
    beta: float,
    cues: Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        signal: _softmax(cues[signal], beta=beta, order=_TERMINAL_ACTIONS)
        for signal in _SIGNALS
    }


def _scout_value(
    values: _ScenarioValues,
    cues: Mapping[str, Mapping[str, float]],
    policies: Mapping[str, Mapping[str, float]],
) -> float:
    terminal_value = 0.0
    for signal in _SIGNALS:
        signal_mass = _signal_probability(values, signal)
        terminal_value += signal_mass * sum(
            policies[signal][action] * cues[signal][action]
            for action in _TERMINAL_ACTIONS
        )
    return -values.scout_cost + values.discount * terminal_value


def _initial_values(
    values: _ScenarioValues,
    beta: float,
    cues: Mapping[str, Mapping[str, float]],
    cue_policies: Mapping[str, Mapping[str, float]],
) -> dict[str, float]:
    direct = _direct_values(values)
    return {
        "scout": _scout_value(values, cues, cue_policies),
        "escape": direct["escape"],
        "submit": direct["submit"],
    }


def _initial_policy(
    values: _ScenarioValues,
    beta: float,
    initial_values: Mapping[str, float],
) -> dict[str, float]:
    if values.horizon == 1:
        terminal = _softmax(
            {action: initial_values[action] for action in _TERMINAL_ACTIONS},
            beta=beta,
            order=_TERMINAL_ACTIONS,
        )
        return {
            "scout": 0.0,
            "escape": terminal["escape"],
            "submit": terminal["submit"],
        }
    return _softmax(initial_values, beta=beta, order=_INITIAL_ACTIONS)


class FinitePrisonReactiveModel:
    """Finite cue-reactive prison decision model without posterior planning."""

    name = _MODEL_NAME
    version = _MODEL_VERSION
    implementation_revision = _IMPLEMENTATION_REVISION

    def simulate(
        self,
        scenario: object,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        values = _scenario_values(scenario)
        beta = _beta(parameters)
        cues = _cue_values(values)
        cue_policies = _cue_policies(values, beta, cues)
        initial_values = _initial_values(values, beta, cues, cue_policies)
        initial_policy = _initial_policy(values, beta, initial_values)

        initial_action, initial_draw = _sample(
            initial_policy, order=_INITIAL_ACTIONS, rng=rng
        )
        events: list[TraceEvent] = [
            TraceEvent(
                tick=0,
                kind="initial_decision",
                data={
                    "action": initial_action,
                    "policy": dict(initial_policy),
                    "values": dict(initial_values),
                    "draw": initial_draw,
                },
            )
        ]

        state_draw = rng.random()
        state_weak = state_draw < values.prior_weak
        terminal_action = initial_action
        signal = "none"
        scout_selected = initial_action == "scout"
        steps = 1

        if scout_selected:
            signal_draw = rng.random()
            clear_probability = (
                values.signal_accuracy
                if state_weak
                else 1.0 - values.signal_accuracy
            )
            signal = "clear" if signal_draw < clear_probability else "alarm"
            events.append(
                TraceEvent(
                    tick=1,
                    kind="observation_received",
                    data={"signal": signal, "draw": signal_draw},
                )
            )
            terminal_policy = cue_policies[signal]
            terminal_action, terminal_draw = _sample(
                terminal_policy, order=_TERMINAL_ACTIONS, rng=rng
            )
            events.append(
                TraceEvent(
                    tick=2,
                    kind="terminal_decision",
                    data={
                        "action": terminal_action,
                        "policy": dict(terminal_policy),
                        "values": dict(cues[signal]),
                        "draw": terminal_draw,
                    },
                )
            )
            steps = 2

        if terminal_action == "escape":
            if scout_selected:
                future_weak = (
                    values.guard_persistence
                    if state_weak
                    else 1.0 - values.guard_persistence
                )
                escaped = rng.random() < future_weak
            else:
                escaped = state_weak
            terminal_utility = (
                values.escape_reward if escaped else -values.capture_cost
            )
        else:
            escaped = False
            terminal_utility = values.submit_reward

        utility = (
            -values.scout_cost + values.discount * terminal_utility
            if scout_selected
            else terminal_utility
        )
        events.append(
            TraceEvent(
                tick=3 if scout_selected else 1,
                kind="episode_ended",
                data={
                    "terminal_action": terminal_action,
                    "escaped": escaped,
                    "utility": utility,
                    "steps": steps,
                    "state_draw": state_draw,
                },
            )
        )
        return ModelRun(
            events=tuple(events),
            outcome={
                "initial_policy": dict(initial_policy),
                "initial_values": dict(initial_values),
                "cue_values": {
                    signal_name: dict(cues[signal_name]) for signal_name in _SIGNALS
                },
                "initial_action": initial_action,
                "terminal_action": terminal_action,
                "signal": signal,
                "escaped": escaped,
                "utility": utility,
                "steps": steps,
            },
        )


def create_prison_reactive_model() -> FinitePrisonReactiveModel:
    return FinitePrisonReactiveModel()


def _object_schema(
    required: tuple[str, ...], properties: Mapping[str, object]
) -> dict[str, object]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additional_properties": False,
    }


def _policy_schema(actions: tuple[str, ...]) -> dict[str, object]:
    return _object_schema(
        actions,
        {
            action: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for action in actions
        },
    )


def _values_schema(actions: tuple[str, ...]) -> dict[str, object]:
    return _object_schema(
        actions,
        {action: {"type": "number"} for action in actions},
    )


def _decision_event_schema(actions: tuple[str, ...]) -> dict[str, object]:
    return _object_schema(
        ("action", "policy", "values", "draw"),
        {
            "action": {"type": "string", "enum": actions},
            "policy": _policy_schema(actions),
            "values": _values_schema(actions),
            "draw": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
    )


def prison_reactive_contract() -> ModelContract:
    scenario_fields = (
        "prior_weak",
        "signal_accuracy",
        "guard_persistence",
        "escape_reward",
        "capture_cost",
        "submit_reward",
        "scout_cost",
        "discount",
        "horizon",
    )
    parameter_schema = ModelSchema(
        name="finite-prison-reactive-parameters",
        version=_MODEL_VERSION,
        definition=_object_schema(
            ("beta",),
            {"beta": {"type": "number", "minimum": 1e-12}},
        ),
    )
    scenario_schema = ModelSchema(
        name="finite-prison-reactive-scenario",
        version=_MODEL_VERSION,
        definition=_object_schema(
            scenario_fields,
            {
                "prior_weak": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "signal_accuracy": {
                    "type": "number",
                    "minimum": 0.5,
                    "maximum": 1.0,
                },
                "guard_persistence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "escape_reward": {"type": "number", "minimum": 0.0},
                "capture_cost": {"type": "number", "minimum": 0.0},
                "submit_reward": {"type": "number"},
                "scout_cost": {"type": "number", "minimum": 0.0},
                "discount": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "horizon": {"type": "integer", "minimum": 1, "maximum": 2},
            },
        ),
    )
    event_schema = ModelSchema(
        name="finite-prison-reactive-events",
        version=_MODEL_VERSION,
        definition={
            "event_kinds": (
                "initial_decision",
                "observation_received",
                "terminal_decision",
                "episode_ended",
            ),
            "event_data": {
                "initial_decision": _decision_event_schema(_INITIAL_ACTIONS),
                "observation_received": _object_schema(
                    ("signal", "draw"),
                    {
                        "signal": {"type": "string", "enum": _SIGNALS},
                        "draw": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                ),
                "terminal_decision": _decision_event_schema(_TERMINAL_ACTIONS),
                "episode_ended": _object_schema(
                    ("terminal_action", "escaped", "utility", "steps", "state_draw"),
                    {
                        "terminal_action": {"type": "string", "enum": _TERMINAL_ACTIONS},
                        "escaped": {"type": "boolean"},
                        "utility": {"type": "number"},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 2},
                        "state_draw": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                ),
            },
            "allow_unlisted_events": False,
        },
    )
    outcome_schema = ModelSchema(
        name="finite-prison-reactive-outcome",
        version=_MODEL_VERSION,
        definition=_object_schema(
            (
                "initial_policy",
                "initial_values",
                "cue_values",
                "initial_action",
                "terminal_action",
                "signal",
                "escaped",
                "utility",
                "steps",
            ),
            {
                "initial_policy": _policy_schema(_INITIAL_ACTIONS),
                "initial_values": _values_schema(_INITIAL_ACTIONS),
                "cue_values": _object_schema(
                    _SIGNALS,
                    {
                        signal: _values_schema(_TERMINAL_ACTIONS)
                        for signal in _SIGNALS
                    },
                ),
                "initial_action": {"type": "string", "enum": _INITIAL_ACTIONS},
                "terminal_action": {"type": "string", "enum": _TERMINAL_ACTIONS},
                "signal": {"type": "string", "enum": ("none",) + _SIGNALS},
                "escaped": {"type": "boolean"},
                "utility": {"type": "number"},
                "steps": {"type": "integer", "minimum": 1, "maximum": 2},
            },
        ),
    )
    return ModelContract(
        version=_MODEL_VERSION,
        implementation_revision=_IMPLEMENTATION_REVISION,
        parameter_schema=parameter_schema,
        scenario_schema=scenario_schema,
        event_schema=event_schema,
        outcome_schema=outcome_schema,
    )


def prison_reactive_source(*, limits: ProcessLimits | None = None) -> SubprocessModel:
    return SubprocessModel(
        name=_MODEL_NAME,
        factory=(
            "narrative_dynamics.adapters.prison_reactive:"
            "create_prison_reactive_model"
        ),
        version=_MODEL_VERSION,
        implementation_revision=_IMPLEMENTATION_REVISION,
        limits=ProcessLimits() if limits is None else limits,
    )


__all__ = [
    "FinitePrisonReactiveModel",
    "create_prison_reactive_model",
    "prison_reactive_contract",
    "prison_reactive_source",
]
