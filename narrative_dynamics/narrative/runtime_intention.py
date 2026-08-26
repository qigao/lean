from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    GoalModelSpec,
    GoalState,
    IntentionalDecisionModelSpec,
)
from narrative_dynamics.narrative.ir import GenericNarrative, StateCellRef
from narrative_dynamics.narrative.runtime_cognition import (
    RuntimeBeliefModelSpec,
    RuntimeUncertainBeliefState,
)
from narrative_dynamics.narrative.runtime_perception import RuntimeEvidenceLedger
from narrative_dynamics.narrative.uncertain import BeliefDistribution


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12


class RuntimeIntentionalDecisionResolutionError(ValueError):
    """Runtime belief could not produce a valid intentional action selection."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _step(value: object, *, label: str) -> int:
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


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


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
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _freeze_policy(
    value: object,
    *,
    label: str,
    min_size: int = 1,
) -> Mapping[str, float]:
    policy = _freeze_vector(value, label=label, min_size=min_size)
    if any(probability < 0.0 for probability in policy.values()):
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
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _map_choice(policy: Mapping[str, float]) -> str:
    if not policy:
        raise ValueError("runtime MAP projection requires a non-empty policy")
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
class _PosteriorSemanticCell:
    cell: StateCellRef
    posterior: BeliefDistribution

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("posterior semantic cell requires StateCellRef")
        if not isinstance(self.posterior, BeliefDistribution):
            raise TypeError("posterior semantic cell requires BeliefDistribution")
        if self.posterior.cell != self.cell:
            raise ValueError("posterior semantic distribution must match its cell")


@dataclass(frozen=True)
class _PosteriorSemanticView:
    cells: Mapping[StateCellRef, _PosteriorSemanticCell]

    def __post_init__(self) -> None:
        if not isinstance(self.cells, Mapping):
            raise TypeError("posterior semantic cells must be a mapping")
        frozen = dict(self.cells)
        if not frozen:
            raise ValueError("posterior semantic view must contain at least one cell")
        if any(
            not isinstance(cell, StateCellRef)
            or not isinstance(view, _PosteriorSemanticCell)
            or view.cell != cell
            for cell, view in frozen.items()
        ):
            raise TypeError("posterior semantic cells must be canonical")
        object.__setattr__(self, "cells", MappingProxyType(frozen))

    def to_dict(self) -> dict[str, object]:
        ordered = tuple(sorted(self.cells, key=_cell_key))
        return {
            "cells": [
                {
                    "cell": cell.to_dict(),
                    "posterior": self.cells[cell].posterior.to_dict(),
                }
                for cell in ordered
            ]
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _posterior_semantic_view(
    belief_state: RuntimeUncertainBeliefState,
) -> _PosteriorSemanticView:
    if not isinstance(belief_state, RuntimeUncertainBeliefState):
        raise TypeError(
            "posterior semantic extraction requires RuntimeUncertainBeliefState"
        )
    return _PosteriorSemanticView(
        {
            cell: _PosteriorSemanticCell(cell, view.posterior)
            for cell, view in belief_state.cells.items()
        }
    )


@dataclass(frozen=True)
class RuntimeIntentionalDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    belief_model: RuntimeBeliefModelSpec
    goal_model: GoalModelSpec
    choice_model: ChoiceModelSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime intentional model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="runtime intentional model version"),
        )
        if not isinstance(self.supported_decision_types, tuple):
            raise TypeError("runtime intentional supported decision types must be a tuple")
        decision_types = tuple(
            _text(item, label="runtime intentional decision type")
            for item in self.supported_decision_types
        )
        if not decision_types:
            raise ValueError(
                "runtime intentional model must support at least one decision type"
            )
        if len(set(decision_types)) != len(decision_types):
            raise ValueError(
                "runtime intentional supported decision types must be unique"
            )
        if not isinstance(self.belief_model, RuntimeBeliefModelSpec):
            raise TypeError(
                "runtime intentional belief model must be RuntimeBeliefModelSpec"
            )
        if not isinstance(self.goal_model, GoalModelSpec):
            raise TypeError("runtime intentional goal model must be GoalModelSpec")
        if not isinstance(self.choice_model, ChoiceModelSpec):
            raise TypeError("runtime intentional choice model must be ChoiceModelSpec")
        goal_ids = {goal.goal_id for goal in self.goal_model.goals}
        if set(self.choice_model.values) != goal_ids:
            raise ValueError(
                "runtime intentional choice goals must match goal model exactly"
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
            "runtime_implementation_identity": measure_implementation(
                RuntimeIntentionalDecisionModelSpec
            ).manifest_identity(),
            "authored_intentional_implementation_identity": measure_implementation(
                IntentionalDecisionModelSpec
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeIntentionalDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    step_index: int
    belief_state: RuntimeUncertainBeliefState
    goal_state: GoalState
    conditional_action_policies: Mapping[str, Mapping[str, float]]
    action_scores: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime intentional result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="runtime intentional result model hash"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="runtime intentional result decision id"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime intentional result step index"),
        )
        if not isinstance(self.belief_state, RuntimeUncertainBeliefState):
            raise TypeError(
                "runtime intentional result belief must be RuntimeUncertainBeliefState"
            )
        if not isinstance(self.goal_state, GoalState):
            raise TypeError("runtime intentional result goal must be GoalState")
        if self.step_index != self.belief_state.step_index:
            raise ValueError(
                "runtime intentional result step must match runtime belief step"
            )
        semantic = _posterior_semantic_view(self.belief_state)
        if self.goal_state.belief_state_hash != semantic.content_hash:
            raise ValueError(
                "runtime goal state must bind posterior semantics exactly"
            )

        conditional = _freeze_nested_policies(
            self.conditional_action_policies,
            label="runtime conditional action policies",
        )
        if set(conditional) != set(self.goal_state.policy):
            raise ValueError(
                "runtime conditional action policies must cover the goal policy exactly"
            )
        action_sets = {frozenset(policy) for policy in conditional.values()}
        if len(action_sets) != 1:
            raise ValueError(
                "runtime conditional action policies must share one action set"
            )
        action_set = next(iter(action_sets))
        if not action_set:
            raise ValueError("runtime intentional result must contain actions")

        scores = _freeze_vector(
            self.action_scores,
            label="runtime intentional action scores",
        )
        policy = _freeze_policy(
            self.action_policy,
            label="runtime intentional action policy",
        )
        if set(scores) != action_set or set(policy) != action_set:
            raise ValueError(
                "runtime action scores and policy must match conditional actions exactly"
            )
        selected = _text(
            self.selected_action,
            label="runtime intentional selected action",
        )
        if selected not in policy:
            raise ValueError(
                "runtime intentional selected action must belong to final policy"
            )
        if selected != _map_choice(policy):
            raise ValueError(
                "runtime intentional selected action must be deterministic lexical MAP"
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
            "step_index": self.step_index,
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


def run_runtime_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeIntentionalDecisionModelSpec,
) -> RuntimeIntentionalDecisionResult:
    raise RuntimeIntentionalDecisionResolutionError(
        "runtime intentional execution is unavailable in this stage"
    )
