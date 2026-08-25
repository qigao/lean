from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, StateDelta
from narrative_dynamics.narrative.ir import (
    ActionOption,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import objective_state


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SUBJECT_SOURCES = frozenset({"actor", "argument"})


class WorldTransitionError(ValueError):
    """A canonical selected action could not be executed safely."""


class WorldTransitionConflictError(WorldTransitionError):
    """Two action deltas in one simultaneous step wrote the same cell."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _freeze_values(value: object) -> Mapping[StateCellRef, TypedValue]:
    if not isinstance(value, Mapping):
        raise TypeError("world values must be a mapping")
    frozen = dict(value)
    if any(
        not isinstance(cell, StateCellRef) or not isinstance(item, TypedValue)
        for cell, item in frozen.items()
    ):
        raise TypeError("world values must map StateCellRef to TypedValue")
    return MappingProxyType(frozen)


def _delta_payload(delta: StateDelta) -> dict[str, object]:
    return delta.to_dict()


@dataclass(frozen=True)
class ActionEffectSpec:
    state_variable: str
    subject_source: str
    subject_argument: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_variable",
            _text(
                self.state_variable,
                label="action effect state variable",
            ),
        )
        source = _text(
            self.subject_source,
            label="action effect subject source",
        )
        if source not in _SUBJECT_SOURCES:
            raise ValueError(
                "action effect subject source must be actor or argument"
            )
        if source == "actor":
            if self.subject_argument is not None:
                raise ValueError(
                    "actor action effect cannot declare a subject argument"
                )
        else:
            object.__setattr__(
                self,
                "subject_argument",
                _text(
                    self.subject_argument,
                    label="action effect subject argument",
                ),
            )
        object.__setattr__(self, "subject_source", source)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_variable": self.state_variable,
            "subject_source": self.subject_source,
            "subject_argument": self.subject_argument,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ActionTransitionSpec:
    action_type: str
    effects: tuple[ActionEffectSpec, ...]
    transition_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "action_type",
            _text(self.action_type, label="action transition type"),
        )
        effects = tuple(self.effects)
        if any(not isinstance(effect, ActionEffectSpec) for effect in effects):
            raise TypeError(
                "action transition effects must be ActionEffectSpec values"
            )
        keys = tuple(
            (
                effect.state_variable,
                effect.subject_source,
                effect.subject_argument,
            )
            for effect in effects
        )
        if len(set(keys)) != len(keys):
            raise ValueError("action transition effects must be unique")
        if not callable(self.transition_hook):
            raise TypeError("action transition hook must be callable")
        object.__setattr__(
            self,
            "effects",
            tuple(
                sorted(
                    effects,
                    key=lambda effect: (
                        effect.state_variable,
                        effect.subject_source,
                        effect.subject_argument or "",
                    ),
                )
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "action_type": self.action_type,
            "effects": [effect.to_dict() for effect in self.effects],
            "implementation_identity": measure_implementation(
                self.transition_hook
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class WorldTransitionModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    transitions: tuple[ActionTransitionSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="world transition model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="world transition model version"),
        )
        object.__setattr__(
            self,
            "domain_id",
            _text(self.domain_id, label="world transition domain id"),
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(
                self.domain_version,
                label="world transition domain version",
            ),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _hash(
                self.domain_spec_hash,
                label="world transition domain spec hash",
            ),
        )
        transitions = tuple(self.transitions)
        if any(
            not isinstance(item, ActionTransitionSpec)
            for item in transitions
        ):
            raise TypeError(
                "world transition model transitions must be "
                "ActionTransitionSpec values"
            )
        if len({item.action_type for item in transitions}) != len(transitions):
            raise ValueError(
                "world transition model action types must be unique"
            )
        object.__setattr__(
            self,
            "transitions",
            tuple(sorted(transitions, key=lambda item: item.action_type)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "transitions": [item.to_dict() for item in self.transitions],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ActionIntent:
    decision_id: str
    selected_action: str
    selection_model_id: str
    selection_result_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="action intent decision id"),
        )
        object.__setattr__(
            self,
            "selected_action",
            _text(
                self.selected_action,
                label="action intent selected action",
            ),
        )
        object.__setattr__(
            self,
            "selection_model_id",
            _text(
                self.selection_model_id,
                label="action intent selection model id",
            ),
        )
        object.__setattr__(
            self,
            "selection_result_hash",
            _hash(
                self.selection_result_hash,
                label="action intent selection result hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "selected_action": self.selected_action,
            "selection_model_id": self.selection_model_id,
            "selection_result_hash": self.selection_result_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class WorldState:
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    source_story_hash: str
    source_at_time: int | None
    step_index: int
    parent_state_hash: str | None
    transition_batch_hash: str | None
    values: Mapping[StateCellRef, TypedValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "domain_id",
            _text(self.domain_id, label="world state domain id"),
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(self.domain_version, label="world state domain version"),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _hash(
                self.domain_spec_hash,
                label="world state domain spec hash",
            ),
        )
        object.__setattr__(
            self,
            "source_story_hash",
            _hash(
                self.source_story_hash,
                label="world state source story hash",
            ),
        )
        if self.source_at_time is not None and (
            not isinstance(self.source_at_time, int)
            or isinstance(self.source_at_time, bool)
            or self.source_at_time < 0
        ):
            raise ValueError(
                "world source cutoff must be a non-negative integer or None"
            )
        if (
            not isinstance(self.step_index, int)
            or isinstance(self.step_index, bool)
            or self.step_index < 0
        ):
            raise ValueError(
                "world step index must be a non-negative integer"
            )
        if self.step_index == 0:
            if (
                self.parent_state_hash is not None
                or self.transition_batch_hash is not None
            ):
                raise ValueError(
                    "step-zero world state cannot have transition lineage"
                )
        else:
            object.__setattr__(
                self,
                "parent_state_hash",
                _hash(
                    self.parent_state_hash,
                    label="parent state hash",
                ),
            )
            object.__setattr__(
                self,
                "transition_batch_hash",
                _hash(
                    self.transition_batch_hash,
                    label="transition batch hash",
                ),
            )
        object.__setattr__(self, "values", _freeze_values(self.values))

    def to_dict(self) -> dict[str, object]:
        cells = tuple(sorted(self.values, key=_cell_key))
        return {
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "source_story_hash": self.source_story_hash,
            "source_at_time": self.source_at_time,
            "step_index": self.step_index,
            "parent_state_hash": self.parent_state_hash,
            "transition_batch_hash": self.transition_batch_hash,
            "values": [
                {
                    "cell": cell.to_dict(),
                    "value": self.values[cell].to_dict(),
                }
                for cell in cells
            ],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ActionTransitionRecord:
    intent: ActionIntent
    actor_id: str
    action: ActionOption
    transition_spec_hash: str
    prior_state_hash: str
    delta: StateDelta

    def __post_init__(self) -> None:
        if not isinstance(self.intent, ActionIntent):
            raise TypeError(
                "action transition record intent must be ActionIntent"
            )
        object.__setattr__(
            self,
            "actor_id",
            _text(
                self.actor_id,
                label="action transition actor id",
            ),
        )
        if not isinstance(self.action, ActionOption):
            raise TypeError(
                "action transition record action must be ActionOption"
            )
        object.__setattr__(
            self,
            "transition_spec_hash",
            _hash(
                self.transition_spec_hash,
                label="action transition spec hash",
            ),
        )
        object.__setattr__(
            self,
            "prior_state_hash",
            _hash(
                self.prior_state_hash,
                label="action transition prior state hash",
            ),
        )
        if not isinstance(self.delta, StateDelta):
            raise TypeError(
                "action transition record delta must be StateDelta"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "intent": self.intent.to_dict(),
            "actor_id": self.actor_id,
            "action": self.action.to_dict(),
            "transition_spec_hash": self.transition_spec_hash,
            "prior_state_hash": self.prior_state_hash,
            "delta": _delta_payload(self.delta),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _transition_record_key(
    record: ActionTransitionRecord,
) -> tuple[str, str, str]:
    return (
        record.actor_id,
        record.intent.decision_id,
        record.action.id,
    )


def _transition_batch_hash(
    transitions: tuple[ActionTransitionRecord, ...],
) -> str:
    return stable_content_hash(
        [item.to_dict() for item in transitions]
    )


@dataclass(frozen=True)
class WorldStepResult:
    model_id: str
    model_hash: str
    prior_state: WorldState
    transitions: tuple[ActionTransitionRecord, ...]
    next_state: WorldState

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="world step model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="world step model hash"),
        )
        if not isinstance(self.prior_state, WorldState):
            raise TypeError("world step prior_state must be WorldState")
        transitions = tuple(self.transitions)
        if not transitions:
            raise ValueError("world step requires at least one transition")
        if any(
            not isinstance(item, ActionTransitionRecord)
            for item in transitions
        ):
            raise TypeError(
                "world step transitions must be ActionTransitionRecord values"
            )
        transitions = tuple(sorted(transitions, key=_transition_record_key))
        if any(
            item.prior_state_hash != self.prior_state.content_hash
            for item in transitions
        ):
            raise ValueError(
                "world step transition records must bind the exact prior state"
            )
        if not isinstance(self.next_state, WorldState):
            raise TypeError("world step next_state must be WorldState")
        prior_identity = (
            self.prior_state.domain_id,
            self.prior_state.domain_version,
            self.prior_state.domain_spec_hash,
            self.prior_state.source_story_hash,
            self.prior_state.source_at_time,
        )
        next_identity = (
            self.next_state.domain_id,
            self.next_state.domain_version,
            self.next_state.domain_spec_hash,
            self.next_state.source_story_hash,
            self.next_state.source_at_time,
        )
        if next_identity != prior_identity:
            raise ValueError(
                "world step next state must preserve source/domain identity"
            )
        if self.next_state.step_index != self.prior_state.step_index + 1:
            raise ValueError(
                "world step next state index must increment the prior state"
            )
        if self.next_state.parent_state_hash != self.prior_state.content_hash:
            raise ValueError(
                "world step next state must bind the exact prior state"
            )
        expected_batch_hash = _transition_batch_hash(transitions)
        if self.next_state.transition_batch_hash != expected_batch_hash:
            raise ValueError(
                "world step next state must bind the exact transition batch"
            )
        object.__setattr__(self, "transitions", transitions)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "prior_state": self.prior_state.to_dict(),
            "transitions": [
                item.to_dict() for item in self.transitions
            ],
            "next_state": self.next_state.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> WorldState:
    values = objective_state(story, domain, at_time=at_time)
    return WorldState(
        domain_id=domain.domain_id,
        domain_version=domain.version,
        domain_spec_hash=domain.content_hash,
        source_story_hash=story.content_hash,
        source_at_time=at_time,
        step_index=0,
        parent_state_hash=None,
        transition_batch_hash=None,
        values=values,
    )


def advance_world_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: WorldState,
    model: WorldTransitionModelSpec,
    intents: tuple[ActionIntent, ...],
) -> WorldStepResult:
    """Task-2 importable placeholder; execution semantics arrive in Task 3."""

    raise NotImplementedError("world step execution is not implemented")
