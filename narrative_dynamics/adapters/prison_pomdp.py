from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random

from narrative_dynamics.contracts import ModelRun, SimulationTrace, TraceEvent
from narrative_dynamics.model_contract import ModelContract, ModelSchema
from narrative_dynamics.process_execution import ProcessLimits, SubprocessModel


_MODEL_NAME = "finite-prison-pomdp"
_MODEL_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "prison-pomdp-v1"
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
        result = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _probability(value: object, *, label: str) -> float:
    result = _finite_number(value, label=label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return result


def _scenario_values(scenario: object) -> _ScenarioValues:
    payload = getattr(scenario, "payload", None)
    if not isinstance(payload, Mapping):
        raise TypeError("prison POMDP scenario payload must be a mapping")
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
        raise ValueError("prison POMDP scenario must contain exactly its declared fields")

    prior = _probability(payload["prior_weak"], label="prior weak probability")
    accuracy = _probability(
        payload["signal_accuracy"],
        label="signal accuracy",
    )
    if accuracy < 0.5:
        raise ValueError("signal accuracy must be at least 0.5")
    persistence = _probability(
        payload["guard_persistence"],
        label="guard persistence",
    )
    escape_reward = _finite_number(
        payload["escape_reward"],
        label="escape reward",
    )
    capture_cost = _finite_number(
        payload["capture_cost"],
        label="capture cost",
    )
    submit_reward = _finite_number(
        payload["submit_reward"],
        label="submit reward",
    )
    scout_cost = _finite_number(payload["scout_cost"], label="scout cost")
    if escape_reward < 0.0 or capture_cost < 0.0 or scout_cost < 0.0:
        raise ValueError("escape reward and prison costs must be non-negative")
    discount = _probability(payload["discount"], label="discount")
    horizon = payload["horizon"]
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("prison POMDP horizon must be an integer")
    if horizon not in (1, 2):
        raise ValueError("prison POMDP horizon must be 1 or 2")
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
        raise ValueError("prison POMDP requires exactly one beta parameter")
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
        raise ValueError("softmax values must match their declared action order")
    maximum = max(float(values[action]) for action in order)
    masses = {
        action: math.exp(beta * (float(values[action]) - maximum))
        for action in order
    }
    total = sum(masses.values())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("prison POMDP policy mass must be positive and finite")
    policy = {action: masses[action] / total for action in order}
    policy[order[-1]] += 1.0 - sum(policy.values())
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


def _signal_probability(
    prior_weak: float,
    accuracy: float,
    signal: str,
) -> float:
    clear = prior_weak * accuracy + (1.0 - prior_weak) * (1.0 - accuracy)
    return clear if signal == "clear" else 1.0 - clear


def _posterior_weak(
    prior_weak: float,
    accuracy: float,
    signal: str,
) -> float:
    evidence = _signal_probability(prior_weak, accuracy, signal)
    if evidence <= 0.0:
        raise ValueError("prison POMDP observation has zero evidence mass")
    likelihood = accuracy if signal == "clear" else 1.0 - accuracy
    posterior = prior_weak * likelihood / evidence
    return min(1.0, max(0.0, posterior))


def _future_weak_probability(
    current_weak_probability: float,
    persistence: float,
) -> float:
    return (
        current_weak_probability * persistence
        + (1.0 - current_weak_probability) * (1.0 - persistence)
    )


def _route_values(
    weak_probability: float,
    scenario: _ScenarioValues,
) -> dict[str, float]:
    return {
        "escape": (
            weak_probability * scenario.escape_reward
            - (1.0 - weak_probability) * scenario.capture_cost
        ),
        "submit": scenario.submit_reward,
    }


def _route_policy(
    weak_probability: float,
    scenario: _ScenarioValues,
    beta: float,
) -> tuple[dict[str, float], dict[str, float]]:
    values = _route_values(weak_probability, scenario)
    return values, _softmax(values, beta=beta, order=_TERMINAL_ACTIONS)


def _scout_value(scenario: _ScenarioValues, beta: float) -> float:
    terminal_value = 0.0
    for signal in _SIGNALS:
        mass = _signal_probability(
            scenario.prior_weak,
            scenario.signal_accuracy,
            signal,
        )
        if mass <= 0.0:
            continue
        posterior = _posterior_weak(
            scenario.prior_weak,
            scenario.signal_accuracy,
            signal,
        )
        future_weak = _future_weak_probability(
            posterior,
            scenario.guard_persistence,
        )
        values, policy = _route_policy(future_weak, scenario, beta)
        terminal_value += mass * sum(
            policy[action] * values[action] for action in _TERMINAL_ACTIONS
        )
    return -scenario.scout_cost + scenario.discount * terminal_value


def _initial_values(
    scenario: _ScenarioValues,
    beta: float,
) -> dict[str, float]:
    direct = _route_values(scenario.prior_weak, scenario)
    return {
        "scout": _scout_value(scenario, beta),
        "escape": direct["escape"],
        "submit": direct["submit"],
    }


def _initial_policy(
    values: Mapping[str, float],
    scenario: _ScenarioValues,
    beta: float,
) -> dict[str, float]:
    if scenario.horizon == 1:
        terminal = _softmax(
            {action: values[action] for action in _TERMINAL_ACTIONS},
            beta=beta,
            order=_TERMINAL_ACTIONS,
        )
        return {
            "scout": 0.0,
            "escape": terminal["escape"],
            "submit": terminal["submit"],
        }
    return _softmax(values, beta=beta, order=_INITIAL_ACTIONS)


def _expected_coordinates(
    scenario: _ScenarioValues,
    beta: float,
    initial_policy: Mapping[str, float],
) -> dict[str, float]:
    escape_probability = float(initial_policy["escape"])
    submit_probability = float(initial_policy["submit"])
    success_probability = escape_probability * scenario.prior_weak
    direct_values = _route_values(scenario.prior_weak, scenario)
    expected_utility = (
        escape_probability * direct_values["escape"]
        + submit_probability * scenario.submit_reward
    )

    scout_probability = float(initial_policy["scout"])
    if scout_probability > 0.0:
        scout_terminal_utility = 0.0
        for state_weak, state_mass in (
            (True, scenario.prior_weak),
            (False, 1.0 - scenario.prior_weak),
        ):
            if state_mass <= 0.0:
                continue
            clear_probability = (
                scenario.signal_accuracy
                if state_weak
                else 1.0 - scenario.signal_accuracy
            )
            future_weak_probability = (
                scenario.guard_persistence
                if state_weak
                else 1.0 - scenario.guard_persistence
            )
            for signal, conditional_signal_mass in (
                ("clear", clear_probability),
                ("alarm", 1.0 - clear_probability),
            ):
                joint = state_mass * conditional_signal_mass
                if joint <= 0.0:
                    continue
                posterior = _posterior_weak(
                    scenario.prior_weak,
                    scenario.signal_accuracy,
                    signal,
                )
                predicted_weak = _future_weak_probability(
                    posterior,
                    scenario.guard_persistence,
                )
                _, route_policy = _route_policy(predicted_weak, scenario, beta)
                route_escape = route_policy["escape"]
                route_submit = route_policy["submit"]
                escape_probability += scout_probability * joint * route_escape
                submit_probability += scout_probability * joint * route_submit
                success_probability += (
                    scout_probability
                    * joint
                    * route_escape
                    * future_weak_probability
                )
                actual_escape_value = (
                    future_weak_probability * scenario.escape_reward
                    - (1.0 - future_weak_probability) * scenario.capture_cost
                )
                scout_terminal_utility += joint * (
                    route_escape * actual_escape_value
                    + route_submit * scenario.submit_reward
                )
        expected_utility += scout_probability * (
            -scenario.scout_cost
            + scenario.discount * scout_terminal_utility
        )

    return {
        "expected_terminal_escape": escape_probability,
        "expected_terminal_submit": submit_probability,
        "expected_escape_success": success_probability,
        "expected_utility": expected_utility,
    }


class FinitePrisonPOMDPModel:
    """Exact two-state, one-information-step prison decision model."""

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
        initial_values = _initial_values(values, beta)
        initial_policy = _initial_policy(initial_values, values, beta)
        expected = _expected_coordinates(values, beta, initial_policy)

        events: list[TraceEvent] = [
            TraceEvent(
                tick=0,
                kind="belief_state",
                data={
                    "prior_weak": values.prior_weak,
                    "horizon": values.horizon,
                },
            )
        ]
        initial_action, initial_draw = _sample(
            initial_policy,
            order=_INITIAL_ACTIONS,
            rng=rng,
        )
        events.append(
            TraceEvent(
                tick=1,
                kind="initial_decision",
                data={
                    "action": initial_action,
                    "policy": dict(initial_policy),
                    "values": dict(initial_values),
                    "draw": initial_draw,
                },
            )
        )

        state_draw = rng.random()
        state_weak = state_draw < values.prior_weak
        signal = "none"
        posterior_weak = values.prior_weak
        terminal_action = initial_action
        steps = 1
        scout_selected = initial_action == "scout"

        if scout_selected:
            signal_draw = rng.random()
            clear_probability = (
                values.signal_accuracy
                if state_weak
                else 1.0 - values.signal_accuracy
            )
            signal = "clear" if signal_draw < clear_probability else "alarm"
            posterior_weak = _posterior_weak(
                values.prior_weak,
                values.signal_accuracy,
                signal,
            )
            events.append(
                TraceEvent(
                    tick=2,
                    kind="observation_received",
                    data={
                        "signal": signal,
                        "posterior_weak": posterior_weak,
                        "draw": signal_draw,
                    },
                )
            )
            predicted_weak = _future_weak_probability(
                posterior_weak,
                values.guard_persistence,
            )
            terminal_values, terminal_policy = _route_policy(
                predicted_weak,
                values,
                beta,
            )
            terminal_action, terminal_draw = _sample(
                terminal_policy,
                order=_TERMINAL_ACTIONS,
                rng=rng,
            )
            events.append(
                TraceEvent(
                    tick=3,
                    kind="terminal_decision",
                    data={
                        "action": terminal_action,
                        "policy": dict(terminal_policy),
                        "values": dict(terminal_values),
                        "draw": terminal_draw,
                    },
                )
            )
            steps = 2

        if terminal_action == "escape":
            if scout_selected:
                transition_draw = rng.random()
                future_weak_probability = (
                    values.guard_persistence
                    if state_weak
                    else 1.0 - values.guard_persistence
                )
                escaped = transition_draw < future_weak_probability
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
                tick=4 if scout_selected else 2,
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
                "initial_action": initial_action,
                "terminal_action": terminal_action,
                "signal": signal,
                "posterior_weak": posterior_weak,
                "escaped": escaped,
                "utility": utility,
                "steps": steps,
                **expected,
            },
        )


def create_prison_pomdp_model() -> FinitePrisonPOMDPModel:
    return FinitePrisonPOMDPModel()


def _object_schema(
    *,
    required: tuple[str, ...],
    properties: Mapping[str, object],
) -> dict[str, object]:
    return {
        "type": "object",
        "required": required,
        "properties": properties,
        "additional_properties": False,
    }


def _policy_schema(actions: tuple[str, ...]) -> dict[str, object]:
    return _object_schema(
        required=actions,
        properties={
            action: {"type": "number", "minimum": 0.0, "maximum": 1.0}
            for action in actions
        },
    )


def _values_schema(actions: tuple[str, ...]) -> dict[str, object]:
    return _object_schema(
        required=actions,
        properties={action: {"type": "number"} for action in actions},
    )


def prison_pomdp_contract() -> ModelContract:
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
        name="finite-prison-pomdp-parameters",
        version=_MODEL_VERSION,
        definition=_object_schema(
            required=("beta",),
            properties={"beta": {"type": "number", "minimum": 1e-12}},
        ),
    )
    scenario_schema = ModelSchema(
        name="finite-prison-pomdp-scenario",
        version=_MODEL_VERSION,
        definition=_object_schema(
            required=scenario_fields,
            properties={
                "prior_weak": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
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
                "discount": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "horizon": {"type": "integer", "minimum": 1, "maximum": 2},
            },
        ),
    )
    event_schema = ModelSchema(
        name="finite-prison-pomdp-events",
        version=_MODEL_VERSION,
        definition={
            "event_kinds": (
                "belief_state",
                "initial_decision",
                "observation_received",
                "terminal_decision",
                "episode_ended",
            ),
            "event_data": {
                "belief_state": _object_schema(
                    required=("prior_weak", "horizon"),
                    properties={
                        "prior_weak": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "horizon": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 2,
                        },
                    },
                ),
                "initial_decision": _object_schema(
                    required=("action", "policy", "values", "draw"),
                    properties={
                        "action": {"type": "string", "enum": _INITIAL_ACTIONS},
                        "policy": _policy_schema(_INITIAL_ACTIONS),
                        "values": _values_schema(_INITIAL_ACTIONS),
                        "draw": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                ),
                "observation_received": _object_schema(
                    required=("signal", "posterior_weak", "draw"),
                    properties={
                        "signal": {"type": "string", "enum": _SIGNALS},
                        "posterior_weak": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                        "draw": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                ),
                "terminal_decision": _object_schema(
                    required=("action", "policy", "values", "draw"),
                    properties={
                        "action": {"type": "string", "enum": _TERMINAL_ACTIONS},
                        "policy": _policy_schema(_TERMINAL_ACTIONS),
                        "values": _values_schema(_TERMINAL_ACTIONS),
                        "draw": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                ),
                "episode_ended": _object_schema(
                    required=(
                        "terminal_action",
                        "escaped",
                        "utility",
                        "steps",
                        "state_draw",
                    ),
                    properties={
                        "terminal_action": {
                            "type": "string",
                            "enum": _TERMINAL_ACTIONS,
                        },
                        "escaped": {"type": "boolean"},
                        "utility": {"type": "number"},
                        "steps": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 2,
                        },
                        "state_draw": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                        },
                    },
                ),
            },
            "allow_unlisted_events": False,
        },
    )
    outcome_schema = ModelSchema(
        name="finite-prison-pomdp-outcome",
        version=_MODEL_VERSION,
        definition=_object_schema(
            required=(
                "initial_policy",
                "initial_values",
                "initial_action",
                "terminal_action",
                "signal",
                "posterior_weak",
                "escaped",
                "utility",
                "steps",
                "expected_terminal_escape",
                "expected_terminal_submit",
                "expected_escape_success",
                "expected_utility",
            ),
            properties={
                "initial_policy": _policy_schema(_INITIAL_ACTIONS),
                "initial_values": _values_schema(_INITIAL_ACTIONS),
                "initial_action": {"type": "string", "enum": _INITIAL_ACTIONS},
                "terminal_action": {
                    "type": "string",
                    "enum": _TERMINAL_ACTIONS,
                },
                "signal": {
                    "type": "string",
                    "enum": ("none",) + _SIGNALS,
                },
                "posterior_weak": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "escaped": {"type": "boolean"},
                "utility": {"type": "number"},
                "steps": {"type": "integer", "minimum": 1, "maximum": 2},
                "expected_terminal_escape": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "expected_terminal_submit": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "expected_escape_success": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "expected_utility": {"type": "number"},
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


def prison_pomdp_source(
    *,
    limits: ProcessLimits | None = None,
) -> SubprocessModel:
    return SubprocessModel(
        name=_MODEL_NAME,
        factory=(
            "narrative_dynamics.adapters.prison_pomdp:"
            "create_prison_pomdp_model"
        ),
        version=_MODEL_VERSION,
        implementation_revision=_IMPLEMENTATION_REVISION,
        limits=ProcessLimits() if limits is None else limits,
    )


def prison_policy_metrics(trace: SimulationTrace) -> dict[str, float]:
    outcome = trace.outcome
    initial_policy = outcome.get("initial_policy")
    if not isinstance(initial_policy, Mapping):
        raise ValueError("prison POMDP trace is missing its initial policy")

    metrics = {
        "initial.scout": initial_policy.get("scout"),
        "initial.escape": initial_policy.get("escape"),
        "initial.submit": initial_policy.get("submit"),
        "terminal.escape": outcome.get("expected_terminal_escape"),
        "terminal.submit": outcome.get("expected_terminal_submit"),
        "escape.success": outcome.get("expected_escape_success"),
        "utility.expected": outcome.get("expected_utility"),
    }
    canonical: dict[str, float] = {}
    for name, raw_value in metrics.items():
        value = _finite_number(raw_value, label=f"metric {name}")
        canonical[name] = value
    return canonical


prison_policy_metrics.version = _MODEL_VERSION


__all__ = [
    "FinitePrisonPOMDPModel",
    "create_prison_pomdp_model",
    "prison_pomdp_contract",
    "prison_pomdp_source",
    "prison_policy_metrics",
]
