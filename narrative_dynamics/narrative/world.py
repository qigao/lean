from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType

from narrative_dynamics.attestation import (
    ImplementationAttestationUnavailable,
    measure_implementation,
)
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.conflict import (
    ConflictParticipant,
    ConflictResolutionContext,
    ConflictResolutionRecord,
    ConflictResolverSpec,
)
from narrative_dynamics.narrative.domain import (
    DomainSpec,
    StateDelta,
    validate_narrative,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
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


class WorldTransitionConflictResolutionError(WorldTransitionConflictError):
    """A declared conflict could not be resolved safely."""


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
    operations = tuple(
        sorted(
            delta.operations,
            key=lambda operation: (
                operation.subject_id,
                operation.state_variable,
            ),
        )
    )
    return {
        "operations": [
            operation.to_dict() for operation in operations
        ]
    }


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
    conflict_resolver: ConflictResolverSpec | None = None

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
        resolver = self.conflict_resolver
        if resolver is not None:
            if not isinstance(resolver, ConflictResolverSpec):
                raise TypeError(
                    "world transition conflict resolver must be ConflictResolverSpec"
                )
            if (
                resolver.domain_id,
                resolver.domain_version,
                resolver.domain_spec_hash,
            ) != (
                self.domain_id,
                self.domain_version,
                self.domain_spec_hash,
            ):
                raise ValueError(
                    "world transition conflict resolver domain identity mismatch"
                )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "transitions": [item.to_dict() for item in self.transitions],
        }
        if self.conflict_resolver is not None:
            payload["conflict_resolver_hash"] = self.conflict_resolver.content_hash
        return payload

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
    canonical = tuple(
        sorted(transitions, key=_transition_record_key)
    )
    return stable_content_hash(
        [item.to_dict() for item in canonical]
    )


def _resolution_key(
    resolution: ConflictResolutionRecord,
) -> tuple[tuple[str, str, str], ...]:
    return tuple(
        (
            participant.actor_id,
            participant.decision_id,
            participant.action.id,
        )
        for participant in resolution.context.participants
    )


def _resolved_transition_batch_hash(
    transitions: tuple[ActionTransitionRecord, ...],
    resolutions: tuple[ConflictResolutionRecord, ...],
) -> str:
    canonical_transitions = tuple(
        sorted(transitions, key=_transition_record_key)
    )
    canonical_resolutions = tuple(
        sorted(resolutions, key=_resolution_key)
    )
    return stable_content_hash(
        {
            "transitions": [
                item.to_dict() for item in canonical_transitions
            ],
            "conflict_resolutions": [
                item.to_dict() for item in canonical_resolutions
            ],
        }
    )


@dataclass(frozen=True)
class WorldStepResult:
    model_id: str
    model_hash: str
    prior_state: WorldState
    transitions: tuple[ActionTransitionRecord, ...]
    next_state: WorldState
    conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()

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

        resolutions = tuple(self.conflict_resolutions)
        if any(
            not isinstance(item, ConflictResolutionRecord)
            for item in resolutions
        ):
            raise TypeError(
                "world step conflict resolutions must be "
                "ConflictResolutionRecord values"
            )
        resolutions = tuple(sorted(resolutions, key=_resolution_key))

        transition_hashes = {
            item.content_hash for item in transitions
        }
        resolved_transition_hashes: set[str] = set()
        for resolution in resolutions:
            if resolution.prior_state_hash != self.prior_state.content_hash:
                raise ValueError(
                    "world step conflict resolution must bind exact prior state"
                )
            participant_hashes = tuple(
                participant.transition_record_hash
                for participant in resolution.context.participants
            )
            if any(
                item not in transition_hashes
                for item in participant_hashes
            ):
                raise ValueError(
                    "world step conflict participant must bind an included transition"
                )
            if (
                resolved_transition_hashes
                & set(participant_hashes)
            ):
                raise ValueError(
                    "world step transition cannot belong to two conflict resolutions"
                )
            resolved_transition_hashes.update(participant_hashes)

        expected_batch_hash = (
            _resolved_transition_batch_hash(transitions, resolutions)
            if resolutions
            else _transition_batch_hash(transitions)
        )
        if self.next_state.transition_batch_hash != expected_batch_hash:
            raise ValueError(
                "world step next state must bind the exact transition batch"
            )
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "conflict_resolutions", resolutions)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "prior_state": self.prior_state.to_dict(),
            "transitions": [
                item.to_dict() for item in self.transitions
            ],
            "next_state": self.next_state.to_dict(),
        }
        if self.conflict_resolutions:
            payload["conflict_resolutions"] = [
                item.to_dict() for item in self.conflict_resolutions
            ]
        return payload

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


def _validate_prior_state(
    story: GenericNarrative,
    domain: DomainSpec,
    prior: WorldState,
    model: WorldTransitionModelSpec,
) -> dict[str, Entity]:
    try:
        validate_narrative(story, domain)
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "world execution narrative/domain validation failed"
        ) from error

    if not isinstance(prior, WorldState):
        raise WorldTransitionError("world execution requires WorldState")
    if not isinstance(model, WorldTransitionModelSpec):
        raise WorldTransitionError(
            "world execution requires WorldTransitionModelSpec"
        )

    domain_identity = (
        domain.domain_id,
        domain.version,
        domain.content_hash,
    )
    if (
        prior.domain_id,
        prior.domain_version,
        prior.domain_spec_hash,
    ) != domain_identity:
        raise WorldTransitionError(
            "world state domain identity does not match DomainSpec"
        )
    if (
        model.domain_id,
        model.domain_version,
        model.domain_spec_hash,
    ) != domain_identity:
        raise WorldTransitionError(
            "world transition model domain identity does not match DomainSpec"
        )
    if prior.source_story_hash != story.content_hash:
        raise WorldTransitionError(
            "world state source story does not match canonical narrative"
        )

    entities = {entity.id: entity for entity in story.entities}
    try:
        for cell, value in prior.values.items():
            if not isinstance(cell, StateCellRef):
                raise TypeError("world state cell must be StateCellRef")
            if not isinstance(value, TypedValue):
                raise TypeError("world state value must be TypedValue")
            subject = entities.get(cell.subject.entity_id)
            if (
                subject is None
                or subject.type_name != cell.subject.entity_type
            ):
                raise ValueError(
                    "world state cell subject is not canonical"
                )
            state_variable = domain._state_variable(
                cell.state_variable
            )
            if state_variable.subject_type != subject.type_name:
                raise ValueError(
                    "world state cell subject type does not match "
                    "state variable"
                )
            domain._value_type(
                state_variable.value_type
            ).validate(value, entities)
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "world state contains an invalid canonical value"
        ) from error

    return entities


def _validate_transition_declarations(
    domain: DomainSpec,
    model: WorldTransitionModelSpec,
) -> None:
    try:
        for transition in model.transitions:
            domain._action_type(transition.action_type)
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "world transition model names an undeclared action type"
        ) from error
    resolver = model.conflict_resolver
    if resolver is not None:
        try:
            for action_type in resolver.supported_action_types:
                domain._action_type(action_type)
        except (TypeError, ValueError) as error:
            raise WorldTransitionError(
                "conflict resolver names an undeclared action type"
            ) from error


def _resolve_intents(
    story: GenericNarrative,
    prior: WorldState,
    model: WorldTransitionModelSpec,
    intents: tuple[ActionIntent, ...],
) -> tuple[
    tuple[
        ActionIntent,
        Decision,
        ActionOption,
        ActionTransitionSpec,
    ],
    ...,
]:
    try:
        values = tuple(intents)
    except TypeError as error:
        raise WorldTransitionError(
            "world step intents must be an iterable of ActionIntent values"
        ) from error

    if not values:
        raise WorldTransitionError(
            "world step requires at least one action intent"
        )
    if any(not isinstance(item, ActionIntent) for item in values):
        raise WorldTransitionError(
            "world step intents must be ActionIntent values"
        )
    if len({item.decision_id for item in values}) != len(values):
        raise WorldTransitionError(
            "world step decision ids must be unique"
        )

    decisions = {decision.id: decision for decision in story.decisions}
    transitions = {
        transition.action_type: transition
        for transition in model.transitions
    }
    resolved: list[
        tuple[
            ActionIntent,
            Decision,
            ActionOption,
            ActionTransitionSpec,
        ]
    ] = []
    actors: set[str] = set()

    for item in values:
        decision = decisions.get(item.decision_id)
        if decision is None:
            raise WorldTransitionError(
                "action intent decision is not declared"
            )
        if (
            prior.source_at_time is not None
            and decision.logical_time > prior.source_at_time
        ):
            raise WorldTransitionError(
                "action intent decision occurs after world source cutoff"
            )
        if decision.actor_id in actors:
            raise WorldTransitionError(
                "one actor may contribute at most one action "
                "per world step"
            )
        actors.add(decision.actor_id)

        action = next(
            (
                candidate
                for candidate in decision.actions
                if candidate.id == item.selected_action
            ),
            None,
        )
        if action is None:
            raise WorldTransitionError(
                "action intent action is not declared by its decision"
            )
        transition = transitions.get(action.type_name)
        if transition is None:
            raise WorldTransitionError(
                "selected action type is not executable "
                "by this world model"
            )
        resolved.append((item, decision, action, transition))

    return tuple(resolved)


def _allowed_cells(
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    decision: Decision,
    action: ActionOption,
    transition: ActionTransitionSpec,
) -> frozenset[StateCellRef]:
    try:
        action_type = domain._action_type(action.type_name)
        parameters = {
            parameter.name: parameter
            for parameter in action_type.parameters
        }
        allowed: set[StateCellRef] = set()

        for effect in transition.effects:
            state_variable = domain._state_variable(
                effect.state_variable
            )
            if effect.subject_source == "actor":
                subject = entities.get(decision.actor_id)
                if subject is None:
                    raise ValueError(
                        "canonical action actor is not declared"
                    )
            else:
                argument_name = effect.subject_argument
                parameter = parameters.get(argument_name)
                if parameter is None:
                    raise ValueError(
                        "action effect subject argument "
                        "is not a declared parameter"
                    )
                argument = action.arguments.get(argument_name)
                if argument is None:
                    raise ValueError(
                        "action effect subject argument "
                        "is missing from canonical action"
                    )
                value = argument.value
                if not isinstance(value, EntityRef):
                    raise TypeError(
                        "action effect subject argument "
                        "must contain EntityRef"
                    )
                subject = entities.get(value.entity_id)
                if (
                    subject is None
                    or subject.type_name != value.entity_type
                ):
                    raise ValueError(
                        "action effect subject argument "
                        "is not canonical"
                    )

            if subject.type_name != state_variable.subject_type:
                raise ValueError(
                    "action effect target type does not match "
                    "state variable"
                )
            allowed.add(
                StateCellRef(
                    EntityRef(subject.id, subject.type_name),
                    state_variable.name,
                )
            )
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "action transition has an invalid effect capability"
        ) from error

    return frozenset(allowed)


def _validated_delta(
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    allowed: frozenset[StateCellRef],
    delta: object,
) -> StateDelta:
    if not isinstance(delta, StateDelta):
        raise WorldTransitionError(
            "action transition hook must return StateDelta"
        )

    seen: set[StateCellRef] = set()
    try:
        for operation in delta.operations:
            subject = entities.get(operation.subject_id)
            if subject is None:
                raise ValueError(
                    "state delta subject is not declared"
                )
            state_variable = domain._state_variable(
                operation.state_variable
            )
            if subject.type_name != state_variable.subject_type:
                raise ValueError(
                    "state delta subject type mismatch"
                )
            cell = StateCellRef(
                EntityRef(subject.id, subject.type_name),
                state_variable.name,
            )
            if cell not in allowed:
                raise ValueError(
                    "state delta wrote outside action effect capability"
                )
            if cell in seen:
                raise ValueError(
                    "one action delta cannot write one state cell twice"
                )
            seen.add(cell)

            if operation.kind == "clear":
                if operation.value is not None:
                    raise ValueError(
                        "clear state delta cannot contain a value"
                    )
            elif operation.kind == "set":
                if operation.value is None:
                    raise ValueError(
                        "set state delta requires a value"
                    )
                domain._value_type(
                    state_variable.value_type
                ).validate(operation.value, entities)
            else:
                raise ValueError(
                    "state delta operation kind is unsupported"
                )
    except (TypeError, ValueError) as error:
        raise WorldTransitionError(
            "action transition produced an invalid state delta"
        ) from error

    canonical_operations = tuple(
        sorted(
            delta.operations,
            key=lambda operation: _cell_key(
                StateCellRef(
                    EntityRef(
                        entities[operation.subject_id].id,
                        entities[operation.subject_id].type_name,
                    ),
                    operation.state_variable,
                )
            ),
        )
    )
    return StateDelta(canonical_operations)


def _attested_transition_hash(
    transition: ActionTransitionSpec,
) -> str:
    try:
        return transition.content_hash
    except ImplementationAttestationUnavailable as error:
        raise WorldTransitionError(
            "action transition implementation attestation is unavailable"
        ) from error


def _attested_resolver_hash(
    resolver: ConflictResolverSpec,
) -> str:
    try:
        return resolver.content_hash
    except ImplementationAttestationUnavailable as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver implementation attestation is unavailable"
        ) from error


def _attested_model_hash(
    model: WorldTransitionModelSpec,
) -> str:
    try:
        return model.content_hash
    except ImplementationAttestationUnavailable as error:
        raise WorldTransitionError(
            "world transition model implementation "
            "attestation is unavailable"
        ) from error


def _execute_one(
    snapshot: Mapping[StateCellRef, TypedValue],
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    prior: WorldState,
    resolved: tuple[
        ActionIntent,
        Decision,
        ActionOption,
        ActionTransitionSpec,
    ],
) -> ActionTransitionRecord:
    item, decision, action, transition = resolved
    allowed = _allowed_cells(
        domain,
        entities,
        decision,
        action,
        transition,
    )
    try:
        raw_delta = transition.transition_hook(
            snapshot,
            decision,
            action,
        )
    except Exception as error:
        raise WorldTransitionError(
            "action transition hook failed"
        ) from error

    delta = _validated_delta(
        domain,
        entities,
        allowed,
        raw_delta,
    )
    return ActionTransitionRecord(
        intent=item,
        actor_id=decision.actor_id,
        action=action,
        transition_spec_hash=_attested_transition_hash(
            transition
        ),
        prior_state_hash=prior.content_hash,
        delta=delta,
    )


def _record_write_cells(
    record: ActionTransitionRecord,
    entities: Mapping[str, Entity],
) -> tuple[StateCellRef, ...]:
    return tuple(
        StateCellRef(
            EntityRef(
                entities[operation.subject_id].id,
                entities[operation.subject_id].type_name,
            ),
            operation.state_variable,
        )
        for operation in record.delta.operations
    )


def _reject_write_conflicts(
    records: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> None:
    owner_by_cell: dict[
        StateCellRef,
        ActionTransitionRecord,
    ] = {}
    for record in records:
        for cell in _record_write_cells(record, entities):
            previous = owner_by_cell.get(cell)
            if previous is not None:
                raise WorldTransitionConflictError(
                    "world step writes "
                    f"{cell.state_variable!r} for "
                    f"{cell.subject.entity_id!r} more than once"
                )
            owner_by_cell[cell] = record


def _conflict_components(
    records: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> tuple[tuple[ActionTransitionRecord, ...], ...]:
    canonical = tuple(sorted(records, key=_transition_record_key))
    writes = tuple(
        frozenset(_record_write_cells(record, entities))
        for record in canonical
    )
    adjacency = [set() for _ in canonical]
    for left_index, left_writes in enumerate(writes):
        for right_index in range(left_index + 1, len(canonical)):
            if left_writes & writes[right_index]:
                adjacency[left_index].add(right_index)
                adjacency[right_index].add(left_index)

    seen: set[int] = set()
    components: list[tuple[ActionTransitionRecord, ...]] = []
    for start in range(len(canonical)):
        if start in seen:
            continue
        stack = [start]
        indices: list[int] = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            indices.append(current)
            stack.extend(
                sorted(adjacency[current], reverse=True)
            )
        if len(indices) >= 2:
            component = tuple(
                canonical[index] for index in sorted(indices)
            )
            components.append(component)

    return tuple(
        sorted(
            components,
            key=lambda component: tuple(
                _transition_record_key(record)
                for record in component
            ),
        )
    )


def _canonical_decision_action(
    story: GenericNarrative,
    record: ActionTransitionRecord,
) -> tuple[Decision, ActionOption]:
    decision = next(
        (
            item
            for item in story.decisions
            if item.id == record.intent.decision_id
        ),
        None,
    )
    if decision is None:
        raise WorldTransitionConflictResolutionError(
            "conflict transition decision is not canonical"
        )
    action = next(
        (
            item
            for item in decision.actions
            if item.id == record.intent.selected_action
        ),
        None,
    )
    if action is None:
        raise WorldTransitionConflictResolutionError(
            "conflict transition action is not canonical"
        )
    if record.actor_id != decision.actor_id:
        raise WorldTransitionConflictResolutionError(
            "conflict transition actor does not match decision"
        )
    if record.action != action:
        raise WorldTransitionConflictResolutionError(
            "conflict transition action payload is not canonical"
        )
    return decision, action


def _build_conflict_context(
    story: GenericNarrative,
    domain: DomainSpec,
    prior: WorldState,
    model: WorldTransitionModelSpec,
    component: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> ConflictResolutionContext:
    transitions = {
        transition.action_type: transition
        for transition in model.transitions
    }
    participants: list[ConflictParticipant] = []
    for record in component:
        decision, action = _canonical_decision_action(story, record)
        transition = transitions.get(action.type_name)
        if transition is None:
            raise WorldTransitionConflictResolutionError(
                "conflict action type is not executable by world model"
            )
        allowed = tuple(
            sorted(
                _allowed_cells(
                    domain,
                    entities,
                    decision,
                    action,
                    transition,
                ),
                key=_cell_key,
            )
        )
        participants.append(
            ConflictParticipant(
                actor_id=record.actor_id,
                decision_id=record.intent.decision_id,
                action=action,
                transition_record_hash=record.content_hash,
                transition_spec_hash=record.transition_spec_hash,
                original_delta=record.delta,
                allowed_write_cells=allowed,
            )
        )

    write_counts: dict[StateCellRef, int] = {}
    for record in component:
        for cell in _record_write_cells(record, entities):
            write_counts[cell] = write_counts.get(cell, 0) + 1
    conflict_cells = tuple(
        sorted(
            (
                cell
                for cell, count in write_counts.items()
                if count >= 2
            ),
            key=_cell_key,
        )
    )
    return ConflictResolutionContext(
        prior.content_hash,
        tuple(participants),
        conflict_cells,
    )


def _certify_conflict_context(
    story: GenericNarrative,
    domain: DomainSpec,
    prior: WorldState,
    model: WorldTransitionModelSpec,
    component: tuple[ActionTransitionRecord, ...],
    context: ConflictResolutionContext,
    entities: Mapping[str, Entity],
) -> None:
    if not isinstance(context, ConflictResolutionContext):
        raise WorldTransitionConflictResolutionError(
            "conflict context must be ConflictResolutionContext"
        )
    if context.prior_state_hash != prior.content_hash:
        raise WorldTransitionConflictResolutionError(
            "conflict context does not bind exact prior state"
        )

    component_by_hash = {
        record.content_hash: record for record in component
    }
    if {
        participant.transition_record_hash
        for participant in context.participants
    } != set(component_by_hash):
        raise WorldTransitionConflictResolutionError(
            "conflict context does not bind exact component transitions"
        )

    transitions = {
        transition.action_type: transition
        for transition in model.transitions
    }
    for participant in context.participants:
        record = component_by_hash[participant.transition_record_hash]
        decision, action = _canonical_decision_action(story, record)
        transition = transitions.get(action.type_name)
        if transition is None:
            raise WorldTransitionConflictResolutionError(
                "conflict participant action type is not executable"
            )
        expected_allowed = tuple(
            sorted(
                _allowed_cells(
                    domain,
                    entities,
                    decision,
                    action,
                    transition,
                ),
                key=_cell_key,
            )
        )
        if participant.actor_id != record.actor_id:
            raise WorldTransitionConflictResolutionError(
                "conflict participant actor identity mismatch"
            )
        if participant.decision_id != record.intent.decision_id:
            raise WorldTransitionConflictResolutionError(
                "conflict participant decision identity mismatch"
            )
        if participant.action != record.action:
            raise WorldTransitionConflictResolutionError(
                "conflict participant action payload mismatch"
            )
        if participant.transition_spec_hash != record.transition_spec_hash:
            raise WorldTransitionConflictResolutionError(
                "conflict participant transition spec mismatch"
            )
        if participant.original_delta != record.delta:
            raise WorldTransitionConflictResolutionError(
                "conflict participant original delta mismatch"
            )
        if participant.allowed_write_cells != expected_allowed:
            raise WorldTransitionConflictResolutionError(
                "conflict participant capability mismatch"
            )

    write_counts: dict[StateCellRef, int] = {}
    for record in component:
        for cell in _record_write_cells(record, entities):
            write_counts[cell] = write_counts.get(cell, 0) + 1
    expected_conflict_cells = tuple(
        sorted(
            (
                cell
                for cell, count in write_counts.items()
                if count >= 2
            ),
            key=_cell_key,
        )
    )
    if context.conflict_cells != expected_conflict_cells:
        raise WorldTransitionConflictResolutionError(
            "conflict context cells do not match exact overlapping writes"
        )


def _validated_resolution_delta(
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    context: ConflictResolutionContext,
    raw_delta: object,
) -> StateDelta:
    allowed = frozenset(
        cell
        for participant in context.participants
        for cell in participant.allowed_write_cells
    )
    try:
        return _validated_delta(
            domain,
            entities,
            allowed,
            raw_delta,
        )
    except WorldTransitionError as error:
        cause = error.__cause__
        if cause is None:
            raise WorldTransitionConflictResolutionError(
                "conflict resolver produced an invalid state delta"
            ) from error
        raise WorldTransitionConflictResolutionError(
            "conflict resolver produced an invalid state delta"
        ) from cause


def _resolve_component(
    snapshot: Mapping[StateCellRef, TypedValue],
    story: GenericNarrative,
    domain: DomainSpec,
    prior: WorldState,
    model: WorldTransitionModelSpec,
    component: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> ConflictResolutionRecord:
    resolver = model.conflict_resolver
    if resolver is None:
        raise WorldTransitionConflictError(
            "world step contains unresolved overlapping writes"
        )

    context = _build_conflict_context(
        story,
        domain,
        prior,
        model,
        component,
        entities,
    )
    _certify_conflict_context(
        story,
        domain,
        prior,
        model,
        component,
        context,
        entities,
    )

    unsupported = tuple(
        participant.action.type_name
        for participant in context.participants
        if participant.action.type_name
        not in resolver.supported_action_types
    )
    if unsupported:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver does not support every participant action type"
        )

    resolver_hash = _attested_resolver_hash(resolver)
    try:
        raw_delta = resolver.resolver_hook(snapshot, context)
    except Exception as error:
        raise WorldTransitionConflictResolutionError(
            "conflict resolver hook failed"
        ) from error

    resolved_delta = _validated_resolution_delta(
        domain,
        entities,
        context,
        raw_delta,
    )
    return ConflictResolutionRecord(
        resolver_id=resolver.resolver_id,
        resolver_hash=resolver_hash,
        prior_state_hash=prior.content_hash,
        context=context,
        resolved_delta=resolved_delta,
    )


def _mutation_write_cells(
    delta: StateDelta,
    entities: Mapping[str, Entity],
) -> tuple[StateCellRef, ...]:
    return tuple(
        StateCellRef(
            EntityRef(
                entities[operation.subject_id].id,
                entities[operation.subject_id].type_name,
            ),
            operation.state_variable,
        )
        for operation in delta.operations
    )


def _reject_final_resolution_collisions(
    deltas: tuple[StateDelta, ...],
    entities: Mapping[str, Entity],
) -> None:
    written: set[StateCellRef] = set()
    for delta in deltas:
        for cell in _mutation_write_cells(delta, entities):
            if cell in written:
                raise WorldTransitionConflictResolutionError(
                    "conflict resolution introduced a final write collision"
                )
            written.add(cell)


def _apply_delta(
    values: dict[StateCellRef, TypedValue],
    delta: StateDelta,
    entities: Mapping[str, Entity],
) -> None:
    for operation in delta.operations:
        subject = entities[operation.subject_id]
        cell = StateCellRef(
            EntityRef(subject.id, subject.type_name),
            operation.state_variable,
        )
        if operation.kind == "clear":
            values.pop(cell, None)
        else:
            assert operation.value is not None
            values[cell] = operation.value


def _atomic_result(
    story: GenericNarrative,
    domain: DomainSpec,
    snapshot: Mapping[StateCellRef, TypedValue],
    prior: WorldState,
    model: WorldTransitionModelSpec,
    records: tuple[ActionTransitionRecord, ...],
    entities: Mapping[str, Entity],
) -> WorldStepResult:
    canonical_records = tuple(
        sorted(records, key=_transition_record_key)
    )
    components = _conflict_components(
        canonical_records,
        entities,
    )

    if components and model.conflict_resolver is None:
        _reject_write_conflicts(canonical_records, entities)

    resolutions: tuple[ConflictResolutionRecord, ...] = ()
    conflicting_hashes: set[str] = set()
    if components:
        resolved_items: list[ConflictResolutionRecord] = []
        for component in components:
            resolved_items.append(
                _resolve_component(
                    snapshot,
                    story,
                    domain,
                    prior,
                    model,
                    component,
                    entities,
                )
            )
            conflicting_hashes.update(
                record.content_hash for record in component
            )
        resolutions = tuple(
            sorted(resolved_items, key=_resolution_key)
        )

    nonconflicting_records = tuple(
        record
        for record in canonical_records
        if record.content_hash not in conflicting_hashes
    )
    effective_deltas = tuple(
        record.delta for record in nonconflicting_records
    ) + tuple(
        resolution.resolved_delta
        for resolution in resolutions
    )
    if resolutions:
        _reject_final_resolution_collisions(
            effective_deltas,
            entities,
        )
    else:
        _reject_write_conflicts(canonical_records, entities)

    next_values = dict(prior.values)
    for delta in effective_deltas:
        _apply_delta(next_values, delta, entities)

    batch_hash = (
        _resolved_transition_batch_hash(
            canonical_records,
            resolutions,
        )
        if resolutions
        else _transition_batch_hash(canonical_records)
    )
    next_state = WorldState(
        domain_id=prior.domain_id,
        domain_version=prior.domain_version,
        domain_spec_hash=prior.domain_spec_hash,
        source_story_hash=prior.source_story_hash,
        source_at_time=prior.source_at_time,
        step_index=prior.step_index + 1,
        parent_state_hash=prior.content_hash,
        transition_batch_hash=batch_hash,
        values=next_values,
    )
    return WorldStepResult(
        model_id=model.model_id,
        model_hash=_attested_model_hash(model),
        prior_state=prior,
        transitions=canonical_records,
        next_state=next_state,
        conflict_resolutions=resolutions,
    )


def advance_world_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: WorldState,
    model: WorldTransitionModelSpec,
    intents: tuple[ActionIntent, ...],
) -> WorldStepResult:
    """Execute canonical actions simultaneously against one prior snapshot."""

    entities = _validate_prior_state(
        story,
        domain,
        prior_state,
        model,
    )
    _validate_transition_declarations(domain, model)
    resolved = _resolve_intents(
        story,
        prior_state,
        model,
        intents,
    )

    snapshot = MappingProxyType(dict(prior_state.values))
    records: list[ActionTransitionRecord] = []
    for item in resolved:
        records.append(
            _execute_one(
                snapshot,
                domain,
                entities,
                prior_state,
                item,
            )
        )

    return _atomic_result(
        story,
        domain,
        snapshot,
        prior_state,
        model,
        tuple(records),
        entities,
    )
