from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.runtime_cognition import (
    RuntimeBeliefModelSpec,
    RuntimeUncertainBeliefState,
    runtime_uncertain_belief_state,
)
from narrative_dynamics.narrative.runtime_perception import RuntimeEvidenceLedger
from narrative_dynamics.narrative.uncertain import BeliefDistribution


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12


class RuntimePlanningDecisionResolutionError(ValueError):
    """Runtime posterior semantics could not produce a valid planning decision."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _depth(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _probability(value: object, *, label: str) -> float:
    number = _finite(value, label=label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _freeze_parameter(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} floats must be finite")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} mapping keys must be non-empty strings")
            frozen[key] = _freeze_parameter(item, label=f"{label}.{key}")
        return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_parameter(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _freeze_parameters(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime planning model parameters must be a mapping")
    frozen = _freeze_parameter(value, label="runtime planning model parameters")
    if not isinstance(frozen, Mapping):
        raise TypeError("runtime planning model parameters must be a mapping")
    return frozen


def _parameter_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _parameter_payload(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_parameter_payload(item) for item in value]
    return value


def _freeze_cell_values(
    value: object,
    *,
    label: str,
    allow_none: bool,
) -> Mapping[StateCellRef, TypedValue | None]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[StateCellRef, TypedValue | None] = {}
    for cell, item in value.items():
        if not isinstance(cell, StateCellRef):
            raise TypeError(f"{label} keys must be StateCellRef values")
        if item is None:
            if not allow_none:
                raise TypeError(f"{label} values must be TypedValue values")
        elif not isinstance(item, TypedValue):
            raise TypeError(
                f"{label} values must be TypedValue"
                + (" or None" if allow_none else "")
            )
        frozen[cell] = item
    return MappingProxyType(
        {cell: frozen[cell] for cell in sorted(frozen, key=_cell_key)}
    )


def _cell_value_payload(
    value: Mapping[StateCellRef, TypedValue | None],
) -> list[dict[str, object]]:
    return [
        {
            "cell": cell.to_dict(),
            "value": None if value[cell] is None else value[cell].to_dict(),
        }
        for cell in sorted(value, key=_cell_key)
    ]


def _semantic_cell_hash(
    value: Mapping[StateCellRef, TypedValue | None],
) -> str:
    return stable_content_hash(_cell_value_payload(value))


def _freeze_action_values(value: object, *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, label=f"{label} action id")
        frozen[key] = _finite(raw_value, label=f"{label} value")
    if not frozen:
        raise ValueError(f"{label} must contain at least one action")
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _freeze_policy(value: object, *, label: str) -> Mapping[str, float]:
    policy = _freeze_action_values(value, label=label)
    if any(probability < 0.0 for probability in policy.values()):
        raise ValueError(f"{label} values must be non-negative")
    if not math.isclose(
        math.fsum(policy.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{label} must sum to 1")
    return policy


def _map_choice(policy: Mapping[str, float]) -> str:
    if not policy:
        raise ValueError("runtime planning MAP requires a non-empty policy")
    maximum = max(policy.values())
    return min(key for key, value in policy.items() if value == maximum)


@dataclass(frozen=True)
class PlanningHiddenState:
    state_id: str
    cells: Mapping[StateCellRef, TypedValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_id",
            _text(self.state_id, label="planning hidden state id"),
        )
        object.__setattr__(
            self,
            "cells",
            _freeze_cell_values(
                self.cells,
                label="planning hidden state cells",
                allow_none=False,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "cells": _cell_value_payload(self.cells),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PlanningObservation:
    observation_id: str
    cues: Mapping[StateCellRef, TypedValue | None]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observation_id",
            _text(self.observation_id, label="planning observation id"),
        )
        object.__setattr__(
            self,
            "cues",
            _freeze_cell_values(
                self.cues,
                label="planning observation cues",
                allow_none=True,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observation_id": self.observation_id,
            "cues": _cell_value_payload(self.cues),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PlanningBeliefState:
    probabilities: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.probabilities, Mapping):
            raise TypeError("planning belief probabilities must be a mapping")
        frozen: dict[str, float] = {}
        for raw_key, raw_value in self.probabilities.items():
            key = _text(raw_key, label="planning belief state id")
            frozen[key] = _probability(
                raw_value,
                label=f"planning belief probability for {key}",
            )
        if not frozen:
            raise ValueError("planning belief requires at least one hidden state")
        if not math.isclose(
            math.fsum(frozen.values()),
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("planning belief probabilities must sum to 1")
        object.__setattr__(
            self,
            "probabilities",
            MappingProxyType({key: frozen[key] for key in sorted(frozen)}),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "probabilities": {
                key: self.probabilities[key] for key in sorted(self.probabilities)
            }
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _freeze_posterior(
    value: object,
    *,
    label: str,
) -> Mapping[StateCellRef, BeliefDistribution]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[StateCellRef, BeliefDistribution] = {}
    for cell, distribution in value.items():
        if not isinstance(cell, StateCellRef):
            raise TypeError(f"{label} keys must be StateCellRef values")
        if not isinstance(distribution, BeliefDistribution):
            raise TypeError(f"{label} values must be BeliefDistribution values")
        if distribution.cell != cell:
            raise ValueError(f"{label} distributions must match their cells")
        frozen[cell] = distribution
    return MappingProxyType(
        {cell: frozen[cell] for cell in sorted(frozen, key=_cell_key)}
    )


def _freeze_hidden_states(
    value: object,
    *,
    label: str,
) -> tuple[PlanningHiddenState, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    states = tuple(value)
    if not states:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, PlanningHiddenState) for item in states):
        raise TypeError(f"{label} must contain PlanningHiddenState values")
    ids = tuple(item.state_id for item in states)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} ids must be unique")
    return tuple(sorted(states, key=lambda item: item.state_id))


def _freeze_observations(
    value: object,
    *,
    label: str,
) -> tuple[PlanningObservation, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    observations = tuple(value)
    if not observations:
        raise ValueError(f"{label} must not be empty")
    if any(not isinstance(item, PlanningObservation) for item in observations):
        raise TypeError(f"{label} must contain PlanningObservation values")
    ids = tuple(item.observation_id for item in observations)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} ids must be unique")
    return tuple(sorted(observations, key=lambda item: item.observation_id))


@dataclass(frozen=True)
class RuntimePlanningBeliefContext:
    posterior: Mapping[StateCellRef, BeliefDistribution]
    hidden_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "posterior",
            _freeze_posterior(self.posterior, label="runtime planning posterior"),
        )
        object.__setattr__(
            self,
            "hidden_states",
            _freeze_hidden_states(
                self.hidden_states,
                label="runtime planning hidden states",
            ),
        )
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))


@dataclass(frozen=True)
class PlanningTransitionContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    candidate_next_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth",
            _depth(self.depth, label="planning transition depth"),
        )
        if not isinstance(self.state, PlanningHiddenState):
            raise TypeError("planning transition state must be PlanningHiddenState")
        if not isinstance(self.action, ActionOption):
            raise TypeError("planning transition action must be ActionOption")
        object.__setattr__(
            self,
            "candidate_next_states",
            _freeze_hidden_states(
                self.candidate_next_states,
                label="planning transition candidate next states",
            ),
        )
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))


@dataclass(frozen=True)
class PlanningObservationContext:
    depth: int
    next_state: PlanningHiddenState
    action: ActionOption
    observations: tuple[PlanningObservation, ...]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth",
            _depth(self.depth, label="planning observation depth"),
        )
        if not isinstance(self.next_state, PlanningHiddenState):
            raise TypeError("planning observation next state must be PlanningHiddenState")
        if not isinstance(self.action, ActionOption):
            raise TypeError("planning observation action must be ActionOption")
        object.__setattr__(
            self,
            "observations",
            _freeze_observations(
                self.observations,
                label="planning observations",
            ),
        )
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))


@dataclass(frozen=True)
class PlanningRewardContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    next_state: PlanningHiddenState
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth",
            _depth(self.depth, label="planning reward depth"),
        )
        if not isinstance(self.state, PlanningHiddenState):
            raise TypeError("planning reward state must be PlanningHiddenState")
        if not isinstance(self.action, ActionOption):
            raise TypeError("planning reward action must be ActionOption")
        if not isinstance(self.next_state, PlanningHiddenState):
            raise TypeError("planning reward next state must be PlanningHiddenState")
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))


@dataclass(frozen=True)
class PlanningValueRecord:
    depth: int
    belief_hash: str
    action_id: str
    expected_immediate_reward: float
    expected_future_value: float
    total_value: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth",
            _depth(self.depth, label="planning value depth"),
        )
        object.__setattr__(
            self,
            "belief_hash",
            _hash(self.belief_hash, label="planning value belief hash"),
        )
        object.__setattr__(
            self,
            "action_id",
            _text(self.action_id, label="planning value action id"),
        )
        object.__setattr__(
            self,
            "expected_immediate_reward",
            _finite(
                self.expected_immediate_reward,
                label="planning expected immediate reward",
            ),
        )
        object.__setattr__(
            self,
            "expected_future_value",
            _finite(
                self.expected_future_value,
                label="planning expected future value",
            ),
        )
        object.__setattr__(
            self,
            "total_value",
            _finite(self.total_value, label="planning total value"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "belief_hash": self.belief_hash,
            "action_id": self.action_id,
            "expected_immediate_reward": self.expected_immediate_reward,
            "expected_future_value": self.expected_future_value,
            "total_value": self.total_value,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class PlanningBeliefUpdate:
    depth: int
    prior_belief_hash: str
    action_id: str
    observation_id: str
    observation_probability: float
    posterior: PlanningBeliefState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "depth",
            _depth(self.depth, label="planning belief update depth"),
        )
        object.__setattr__(
            self,
            "prior_belief_hash",
            _hash(
                self.prior_belief_hash,
                label="planning belief update prior belief hash",
            ),
        )
        object.__setattr__(
            self,
            "action_id",
            _text(self.action_id, label="planning belief update action id"),
        )
        object.__setattr__(
            self,
            "observation_id",
            _text(
                self.observation_id,
                label="planning belief update observation id",
            ),
        )
        probability = _probability(
            self.observation_probability,
            label="planning belief update observation probability",
        )
        if probability <= 0.0:
            raise ValueError(
                "planning belief update observation probability must be positive"
            )
        object.__setattr__(self, "observation_probability", probability)
        if not isinstance(self.posterior, PlanningBeliefState):
            raise TypeError(
                "planning belief update posterior must be PlanningBeliefState"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "prior_belief_hash": self.prior_belief_hash,
            "action_id": self.action_id,
            "observation_id": self.observation_id,
            "observation_probability": self.observation_probability,
            "posterior": self.posterior.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimePlanningDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    planning_cells: tuple[StateCellRef, ...]
    observation_cells: tuple[StateCellRef, ...]
    belief_model: RuntimeBeliefModelSpec
    hidden_states: tuple[PlanningHiddenState, ...]
    observations: tuple[PlanningObservation, ...]
    action_schedule: tuple[tuple[str, ...], ...]
    discount: float
    beta: float
    parameters: Mapping[str, object]
    joint_belief_hook: object = field(compare=False, repr=False)
    transition_hook: object = field(compare=False, repr=False)
    observation_hook: object = field(compare=False, repr=False)
    reward_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime planning model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="runtime planning model version"),
        )
        if not isinstance(self.supported_decision_types, tuple):
            raise TypeError(
                "runtime planning supported decision types must be a tuple"
            )
        decision_types = tuple(
            _text(item, label="runtime planning decision type")
            for item in self.supported_decision_types
        )
        if not decision_types:
            raise ValueError(
                "runtime planning model requires at least one supported decision type"
            )
        if len(set(decision_types)) != len(decision_types):
            raise ValueError(
                "runtime planning supported decision types must be unique"
            )
        object.__setattr__(
            self,
            "supported_decision_types",
            tuple(sorted(decision_types)),
        )

        if not isinstance(self.planning_cells, tuple):
            raise TypeError("runtime planning cells must be a tuple")
        planning_cells = tuple(self.planning_cells)
        if not planning_cells:
            raise ValueError("runtime planning model requires planning cells")
        if any(not isinstance(cell, StateCellRef) for cell in planning_cells):
            raise TypeError(
                "runtime planning cells must contain StateCellRef values"
            )
        if len(set(planning_cells)) != len(planning_cells):
            raise ValueError("runtime planning cells must be unique")
        object.__setattr__(
            self,
            "planning_cells",
            tuple(sorted(planning_cells, key=_cell_key)),
        )

        if not isinstance(self.observation_cells, tuple):
            raise TypeError("runtime planning observation cells must be a tuple")
        observation_cells = tuple(self.observation_cells)
        if any(not isinstance(cell, StateCellRef) for cell in observation_cells):
            raise TypeError(
                "runtime planning observation cells must contain StateCellRef values"
            )
        if len(set(observation_cells)) != len(observation_cells):
            raise ValueError("runtime planning observation cells must be unique")
        object.__setattr__(
            self,
            "observation_cells",
            tuple(sorted(observation_cells, key=_cell_key)),
        )

        if not isinstance(self.belief_model, RuntimeBeliefModelSpec):
            raise TypeError(
                "runtime planning belief model must be RuntimeBeliefModelSpec"
            )

        object.__setattr__(
            self,
            "hidden_states",
            _freeze_hidden_states(
                self.hidden_states,
                label="runtime planning hidden states",
            ),
        )
        object.__setattr__(
            self,
            "observations",
            _freeze_observations(
                self.observations,
                label="runtime planning observations",
            ),
        )

        if not isinstance(self.action_schedule, tuple):
            raise TypeError("runtime planning action schedule must be a tuple")
        schedule_rows = tuple(self.action_schedule)
        if not schedule_rows:
            raise ValueError("runtime planning action schedule must not be empty")
        frozen_schedule: list[tuple[str, ...]] = []
        for depth_index, row in enumerate(schedule_rows):
            if not isinstance(row, tuple):
                raise TypeError(
                    "runtime planning action schedule rows must be tuples"
                )
            actions = tuple(
                _text(
                    action,
                    label=f"runtime planning action id at depth {depth_index}",
                )
                for action in row
            )
            if not actions:
                raise ValueError(
                    "runtime planning action schedule rows must not be empty"
                )
            if len(set(actions)) != len(actions):
                raise ValueError(
                    "runtime planning action schedule rows must have unique actions"
                )
            frozen_schedule.append(tuple(sorted(actions)))
        object.__setattr__(self, "action_schedule", tuple(frozen_schedule))

        object.__setattr__(
            self,
            "discount",
            _probability(self.discount, label="runtime planning discount"),
        )
        beta = _finite(self.beta, label="runtime planning beta")
        if beta <= 0.0:
            raise ValueError("runtime planning beta must be positive")
        object.__setattr__(self, "beta", beta)
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))

        for label, hook in (
            ("joint belief", self.joint_belief_hook),
            ("transition", self.transition_hook),
            ("observation", self.observation_hook),
            ("reward", self.reward_hook),
        ):
            if not callable(hook):
                raise TypeError(f"runtime planning {label} hook must be callable")

    @property
    def horizon(self) -> int:
        return len(self.action_schedule)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "supported_decision_types": list(self.supported_decision_types),
            "planning_cells": [cell.to_dict() for cell in self.planning_cells],
            "observation_cells": [
                cell.to_dict() for cell in self.observation_cells
            ],
            "belief_model_hash": self.belief_model.content_hash,
            "hidden_states": [state.to_dict() for state in self.hidden_states],
            "observations": [item.to_dict() for item in self.observations],
            "action_schedule": [list(row) for row in self.action_schedule],
            "discount": self.discount,
            "beta": self.beta,
            "parameters": _parameter_payload(self.parameters),
            "joint_belief_hook_identity": measure_implementation(
                self.joint_belief_hook
            ).manifest_identity(),
            "transition_hook_identity": measure_implementation(
                self.transition_hook
            ).manifest_identity(),
            "observation_hook_identity": measure_implementation(
                self.observation_hook
            ).manifest_identity(),
            "reward_hook_identity": measure_implementation(
                self.reward_hook
            ).manifest_identity(),
            "runtime_implementation_identity": measure_implementation(
                RuntimePlanningDecisionModelSpec
            ).manifest_identity(),
            "softmax_implementation_identity": measure_implementation(
                finite_softmax
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validated_value_record(record: object) -> PlanningValueRecord:
    if not isinstance(record, PlanningValueRecord):
        raise TypeError("planning result value records must contain PlanningValueRecord")
    return PlanningValueRecord(
        record.depth,
        record.belief_hash,
        record.action_id,
        record.expected_immediate_reward,
        record.expected_future_value,
        record.total_value,
    )


def _validated_belief_update(record: object) -> PlanningBeliefUpdate:
    if not isinstance(record, PlanningBeliefUpdate):
        raise TypeError(
            "planning result belief updates must contain PlanningBeliefUpdate"
        )
    if not isinstance(record.posterior, PlanningBeliefState):
        raise TypeError(
            "planning result belief update posterior must be PlanningBeliefState"
        )
    return PlanningBeliefUpdate(
        record.depth,
        record.prior_belief_hash,
        record.action_id,
        record.observation_id,
        record.observation_probability,
        PlanningBeliefState(dict(record.posterior.probabilities)),
    )


@dataclass(frozen=True)
class RuntimePlanningDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    actor_id: str
    step_index: int
    ledger_hash: str
    belief_state: RuntimeUncertainBeliefState
    planning_belief: PlanningBeliefState
    action_values: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str
    value_records: tuple[PlanningValueRecord, ...]
    belief_updates: tuple[PlanningBeliefUpdate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime planning result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="runtime planning result model hash"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="runtime planning result decision id"),
        )
        object.__setattr__(
            self,
            "actor_id",
            _text(self.actor_id, label="runtime planning result actor id"),
        )
        object.__setattr__(
            self,
            "step_index",
            _depth(self.step_index, label="runtime planning result step index"),
        )
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="runtime planning result ledger hash"),
        )
        if not isinstance(self.belief_state, RuntimeUncertainBeliefState):
            raise TypeError(
                "runtime planning result belief state must be RuntimeUncertainBeliefState"
            )
        if self.actor_id != self.belief_state.agent_id:
            raise ValueError(
                "runtime planning result actor must match runtime belief actor"
            )
        if self.step_index != self.belief_state.step_index:
            raise ValueError(
                "runtime planning result step must match runtime belief step"
            )
        if self.ledger_hash != self.belief_state.ledger_hash:
            raise ValueError(
                "runtime planning result ledger must match runtime belief ledger"
            )
        if not isinstance(self.planning_belief, PlanningBeliefState):
            raise TypeError(
                "runtime planning result planning belief must be PlanningBeliefState"
            )
        planning_belief = PlanningBeliefState(
            dict(self.planning_belief.probabilities)
        )
        object.__setattr__(self, "planning_belief", planning_belief)

        values = _freeze_action_values(
            self.action_values,
            label="runtime planning action values",
        )
        policy = _freeze_policy(
            self.action_policy,
            label="runtime planning action policy",
        )
        if set(values) != set(policy):
            raise ValueError(
                "runtime planning action values and policy must cover the same actions"
            )
        selected = _text(
            self.selected_action,
            label="runtime planning selected action",
        )
        if selected not in policy:
            raise ValueError(
                "runtime planning selected action must belong to the root policy"
            )
        if selected != _map_choice(policy):
            raise ValueError(
                "runtime planning selected action must be deterministic lexical MAP"
            )

        if not isinstance(self.value_records, tuple):
            raise TypeError("runtime planning value records must be a tuple")
        value_records = tuple(
            _validated_value_record(record) for record in self.value_records
        )
        if not value_records:
            raise ValueError(
                "runtime planning result requires planning value records"
            )
        value_keys = tuple(
            (record.depth, record.belief_hash, record.action_id)
            for record in value_records
        )
        if len(set(value_keys)) != len(value_keys):
            raise ValueError("runtime planning value record keys must be unique")
        value_records = tuple(
            sorted(
                value_records,
                key=lambda record: (
                    record.depth,
                    record.belief_hash,
                    record.action_id,
                ),
            )
        )
        root_records = {
            record.action_id: record
            for record in value_records
            if record.depth == 0
            and record.belief_hash == planning_belief.content_hash
        }
        if set(root_records) != set(values):
            raise ValueError(
                "runtime planning result must bind one root value record per action"
            )
        for action, record in root_records.items():
            if record.total_value != values[action]:
                raise ValueError(
                    "runtime planning root value record must bind action value"
                )
            if record.total_value != (
                record.expected_immediate_reward + record.expected_future_value
            ):
                raise ValueError(
                    "runtime planning value record total must equal immediate plus future"
                )

        if not isinstance(self.belief_updates, tuple):
            raise TypeError("runtime planning belief updates must be a tuple")
        belief_updates = tuple(
            _validated_belief_update(record) for record in self.belief_updates
        )
        update_keys = tuple(
            (
                record.depth,
                record.prior_belief_hash,
                record.action_id,
                record.observation_id,
            )
            for record in belief_updates
        )
        if len(set(update_keys)) != len(update_keys):
            raise ValueError("runtime planning belief update keys must be unique")
        belief_updates = tuple(
            sorted(
                belief_updates,
                key=lambda record: (
                    record.depth,
                    record.prior_belief_hash,
                    record.action_id,
                    record.observation_id,
                ),
            )
        )
        object.__setattr__(self, "action_values", values)
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_action", selected)
        object.__setattr__(self, "value_records", value_records)
        object.__setattr__(self, "belief_updates", belief_updates)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "decision_id": self.decision_id,
            "actor_id": self.actor_id,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "belief_state": self.belief_state.to_dict(),
            "planning_belief": self.planning_belief.to_dict(),
            "action_values": {
                key: self.action_values[key] for key in sorted(self.action_values)
            },
            "action_policy": {
                key: self.action_policy[key] for key in sorted(self.action_policy)
            },
            "selected_action": self.selected_action,
            "value_records": [record.to_dict() for record in self.value_records],
            "belief_updates": [record.to_dict() for record in self.belief_updates],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validated_model(
    model: object,
) -> RuntimePlanningDecisionModelSpec:
    if not isinstance(model, RuntimePlanningDecisionModelSpec):
        raise TypeError(
            "runtime planning execution requires RuntimePlanningDecisionModelSpec"
        )
    return RuntimePlanningDecisionModelSpec(
        model.model_id,
        model.version,
        tuple(model.supported_decision_types),
        tuple(model.planning_cells),
        tuple(model.observation_cells),
        model.belief_model,
        tuple(model.hidden_states),
        tuple(model.observations),
        tuple(tuple(row) for row in model.action_schedule),
        model.discount,
        model.beta,
        model.parameters,
        model.joint_belief_hook,
        model.transition_hook,
        model.observation_hook,
        model.reward_hook,
    )


def _validated_ledger(
    story: GenericNarrative,
    domain: DomainSpec,
    ledger: object,
) -> RuntimeEvidenceLedger:
    if not isinstance(ledger, RuntimeEvidenceLedger):
        raise TypeError("runtime planning execution requires RuntimeEvidenceLedger")
    validated = RuntimeEvidenceLedger(
        ledger.domain_id,
        ledger.domain_version,
        ledger.domain_spec_hash,
        ledger.source_story_hash,
        ledger.source_at_time,
        ledger.initial_world_state_hash,
        ledger.current_world_state_hash,
        tuple(ledger.batches),
    )
    if validated.domain_id != domain.domain_id:
        raise ValueError("runtime planning ledger domain id mismatch")
    if validated.domain_version != domain.version:
        raise ValueError("runtime planning ledger domain version mismatch")
    if validated.domain_spec_hash != domain.content_hash:
        raise ValueError("runtime planning ledger domain spec hash mismatch")
    if validated.source_story_hash != story.content_hash:
        raise ValueError("runtime planning ledger source story mismatch")
    return validated


def _validate_cell_value(
    cell: StateCellRef,
    value: TypedValue,
    *,
    story: GenericNarrative,
    domain: DomainSpec,
) -> None:
    entities = {entity.id: entity for entity in story.entities}
    subject = entities.get(cell.subject.entity_id)
    if subject is None:
        raise ValueError("planning cell subject must reference a declared entity")
    if subject.type_name != cell.subject.entity_type:
        raise ValueError("planning cell subject type must match declared entity")
    state_variable = domain._state_variable(cell.state_variable)
    if state_variable.subject_type != subject.type_name:
        raise ValueError("planning cell subject type must match state variable")
    domain._value_type(state_variable.value_type).validate(value, entities)


def _preflight_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimePlanningDecisionModelSpec,
) -> tuple[Decision, Mapping[str, ActionOption]]:
    decision_id = _text(decision_id, label="runtime planning decision id")
    decision = next(
        (item for item in story.decisions if item.id == decision_id),
        None,
    )
    if decision is None:
        raise ValueError("runtime planning decision template is not declared")
    if decision.type_name not in model.supported_decision_types:
        raise ValueError("runtime planning decision type is not supported")
    if (
        not decision.context_cells
        or len(set(decision.context_cells)) != len(decision.context_cells)
    ):
        raise ValueError(
            "runtime planning decision context cells must be non-empty and unique"
        )
    action_ids = tuple(action.id for action in decision.actions)
    if not action_ids or len(set(action_ids)) != len(action_ids):
        raise ValueError(
            "runtime planning decision actions must be non-empty and unique"
        )
    if ledger.source_at_time is not None and decision.logical_time > ledger.source_at_time:
        raise ValueError("runtime planning decision template occurs after source cutoff")

    context_cells = set(decision.context_cells)
    if not set(model.planning_cells).issubset(context_cells):
        raise ValueError("runtime planning cells must stay within decision context")
    if not set(model.observation_cells).issubset(context_cells):
        raise ValueError(
            "runtime planning observation cells must stay within decision context"
        )

    expected_planning = set(model.planning_cells)
    state_semantics: set[str] = set()
    for state in model.hidden_states:
        if set(state.cells) != expected_planning:
            raise ValueError(
                "planning hidden state must assign exactly the declared planning cells"
            )
        for cell, value in state.cells.items():
            assert isinstance(value, TypedValue)
            _validate_cell_value(cell, value, story=story, domain=domain)
        semantic_hash = _semantic_cell_hash(state.cells)
        if semantic_hash in state_semantics:
            raise ValueError(
                "planning hidden states must have unique semantic cell assignments"
            )
        state_semantics.add(semantic_hash)

    expected_observation = set(model.observation_cells)
    observation_semantics: set[str] = set()
    for observation in model.observations:
        if set(observation.cues) != expected_observation:
            raise ValueError(
                "planning observation must assign exactly the declared observation cells"
            )
        for cell, value in observation.cues.items():
            if value is not None:
                _validate_cell_value(cell, value, story=story, domain=domain)
        semantic_hash = _semantic_cell_hash(observation.cues)
        if semantic_hash in observation_semantics:
            raise ValueError(
                "planning observations must have unique semantic cue assignments"
            )
        observation_semantics.add(semantic_hash)

    if not model.observation_cells:
        if len(model.observations) != 1 or model.observations[0].cues:
            raise ValueError(
                "empty planning observation cells require one empty observation atom"
            )

    authored_actions = set(action_ids)
    if set(model.action_schedule[0]) != authored_actions:
        raise ValueError(
            "runtime planning root schedule must match authored actions exactly"
        )
    for row in model.action_schedule[1:]:
        if not set(row).issubset(authored_actions):
            raise ValueError(
                "runtime planning future schedule must use authored actions only"
            )

    return decision, MappingProxyType(
        {action.id: action for action in sorted(decision.actions, key=lambda item: item.id)}
    )


def _planning_belief_from_raw(
    raw: object,
    *,
    expected_ids: tuple[str, ...],
) -> PlanningBeliefState:
    if not isinstance(raw, Mapping):
        raise TypeError("planning joint belief hook must return a mapping")
    if set(raw) != set(expected_ids):
        raise ValueError("planning joint belief must cover exactly hidden-state ids")
    return PlanningBeliefState(
        {state_id: raw[state_id] for state_id in expected_ids}
    )


def _resolve_root_planning_belief(
    story: GenericNarrative,
    domain: DomainSpec,
    decision: Decision,
    ledger: RuntimeEvidenceLedger,
    model: RuntimePlanningDecisionModelSpec,
) -> tuple[RuntimeUncertainBeliefState, PlanningBeliefState]:
    belief_state = runtime_uncertain_belief_state(
        story,
        domain,
        decision.actor_id,
        ledger,
        model.belief_model,
        model.planning_cells,
    )
    if set(belief_state.cells) != set(model.planning_cells):
        raise ValueError(
            "runtime planning belief must cover exactly the planning cells"
        )
    posterior = MappingProxyType(
        {
            cell: belief_state.cells[cell].posterior
            for cell in sorted(model.planning_cells, key=_cell_key)
        }
    )
    raw_joint = model.joint_belief_hook(
        RuntimePlanningBeliefContext(
            posterior=posterior,
            hidden_states=model.hidden_states,
            parameters=model.parameters,
        )
    )
    state_ids = tuple(state.state_id for state in model.hidden_states)
    root_belief = _planning_belief_from_raw(raw_joint, expected_ids=state_ids)

    for cell, distribution in posterior.items():
        for mass in distribution.masses:
            matching = tuple(
                state
                for state in model.hidden_states
                if state.cells[cell] == mass.value
            )
            if mass.probability > 0.0 and not matching:
                raise ValueError(
                    "planning hidden states must represent positive runtime hypotheses"
                )
            joint_mass = math.fsum(
                root_belief.probabilities[state.state_id]
                for state in matching
            )
            if not math.isclose(
                joint_mass,
                mass.probability,
                rel_tol=0.0,
                abs_tol=_PROBABILITY_TOLERANCE,
            ):
                raise ValueError(
                    "planning joint belief must preserve runtime marginals"
                )
    return belief_state, root_belief


def run_runtime_planning_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimePlanningDecisionModelSpec,
) -> RuntimePlanningDecisionResult:
    try:
        validate_narrative(story, domain)
        validated_model = _validated_model(model)
        validated_ledger = _validated_ledger(story, domain, ledger)
        decision, _action_by_id = _preflight_decision(
            story,
            domain,
            decision_id,
            validated_ledger,
            validated_model,
        )
        _belief_state, _root_belief = _resolve_root_planning_belief(
            story,
            domain,
            decision,
            validated_ledger,
            validated_model,
        )
        raise RuntimePlanningDecisionResolutionError(
            "runtime planning decision solver is not implemented"
        )
    except RuntimePlanningDecisionResolutionError:
        raise
    except Exception as error:
        raise RuntimePlanningDecisionResolutionError(
            "runtime planning decision could not be resolved"
        ) from error


__all__ = (
    "PlanningHiddenState",
    "PlanningObservation",
    "PlanningBeliefState",
    "RuntimePlanningBeliefContext",
    "PlanningTransitionContext",
    "PlanningObservationContext",
    "PlanningRewardContext",
    "PlanningValueRecord",
    "PlanningBeliefUpdate",
    "RuntimePlanningDecisionModelSpec",
    "RuntimePlanningDecisionResult",
    "RuntimePlanningDecisionResolutionError",
    "run_runtime_planning_decision",
)
