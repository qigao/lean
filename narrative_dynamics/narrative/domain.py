from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Claim,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    Observation,
    Proposition,
    Reception,
    StateCellRef,
    TypedValue,
)


_VALUE_KINDS = frozenset({"enum", "bool", "integer", "text", "entity_ref"})
_DELTA_KINDS = frozenset({"set", "clear"})


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _unique_named(values: tuple[object, ...], *, label: str) -> None:
    names = tuple(getattr(value, "name", None) for value in values)
    if any(not isinstance(name, str) or not name for name in names):
        raise TypeError(f"{label} values must expose non-empty names")
    if len(set(names)) != len(names):
        raise ValueError(f"{label} names must be unique")


def _unique_ids(values: tuple[object, ...], *, label: str) -> None:
    ids = tuple(getattr(value, "id", None) for value in values)
    if any(not isinstance(item, str) or not item for item in ids):
        raise TypeError(f"{label} values must expose non-empty ids")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{label} ids must be unique")


@dataclass(frozen=True)
class EntityTypeSpec:
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="entity type name"))

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name}


@dataclass(frozen=True)
class ValueTypeSpec:
    name: str
    kind: str
    allowed_values: tuple[str, ...] = ()
    entity_type: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="value type name"))
        kind = _text(self.kind, label="value type kind")
        if kind not in _VALUE_KINDS:
            raise ValueError("value type kind is not supported")
        object.__setattr__(self, "kind", kind)
        allowed_values = tuple(
            _text(item, label="allowed enum value") for item in self.allowed_values
        )
        if len(set(allowed_values)) != len(allowed_values):
            raise ValueError("allowed enum values must be unique")
        object.__setattr__(self, "allowed_values", allowed_values)

        if kind == "enum":
            if not allowed_values:
                raise ValueError("enum value type requires allowed values")
            if self.entity_type is not None:
                raise ValueError("enum value type cannot declare an entity type")
        elif kind == "entity_ref":
            if allowed_values:
                raise ValueError("entity_ref value type cannot declare enum values")
            object.__setattr__(
                self,
                "entity_type",
                _text(self.entity_type, label="referenced entity type"),
            )
        else:
            if allowed_values:
                raise ValueError(f"{kind} value type cannot declare enum values")
            if self.entity_type is not None:
                raise ValueError(f"{kind} value type cannot declare an entity type")

    def validate(
        self,
        value: TypedValue,
        entities: Mapping[str, Entity] | None = None,
    ) -> None:
        if not isinstance(value, TypedValue):
            raise TypeError("domain values must be TypedValue instances")
        if value.type_name != self.name:
            raise ValueError(
                f"typed value type must be {self.name!r}, got {value.type_name!r}"
            )
        raw = value.value
        if self.kind == "enum":
            if not isinstance(raw, str) or raw not in self.allowed_values:
                raise ValueError(f"{self.name} must be one of the declared enum values")
        elif self.kind == "bool":
            if not isinstance(raw, bool):
                raise TypeError(f"{self.name} must contain a boolean")
        elif self.kind == "integer":
            if not isinstance(raw, int) or isinstance(raw, bool):
                raise TypeError(f"{self.name} must contain an integer")
        elif self.kind == "text":
            if not isinstance(raw, str):
                raise TypeError(f"{self.name} must contain text")
        else:
            if not isinstance(raw, EntityRef):
                raise TypeError(f"{self.name} must contain an EntityRef")
            if raw.entity_type != self.entity_type:
                raise ValueError(
                    f"{self.name} must reference entity type {self.entity_type!r}"
                )
            if entities is not None:
                entity = entities.get(raw.entity_id)
                if entity is None:
                    raise ValueError("entity_ref value must reference a declared entity")
                if entity.type_name != raw.entity_type:
                    raise ValueError("entity_ref value type must match the declared entity")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "allowed_values": list(self.allowed_values),
            "entity_type": self.entity_type,
        }


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    value_type: str
    intervenable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="parameter name"))
        object.__setattr__(
            self,
            "value_type",
            _text(self.value_type, label="parameter value type"),
        )
        if not isinstance(self.intervenable, bool):
            raise TypeError("parameter intervenable flag must be boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "value_type": self.value_type,
            "intervenable": self.intervenable,
        }


@dataclass(frozen=True)
class StateVariableSpec:
    name: str
    subject_type: str
    value_type: str
    cardinality: str = "one"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _text(self.name, label="state variable name")
        )
        object.__setattr__(
            self,
            "subject_type",
            _text(self.subject_type, label="state variable subject type"),
        )
        object.__setattr__(
            self,
            "value_type",
            _text(self.value_type, label="state variable value type"),
        )
        if self.cardinality != "one":
            raise ValueError("state variable cardinality must be one in V1")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "subject_type": self.subject_type,
            "value_type": self.value_type,
            "cardinality": self.cardinality,
        }


@dataclass(frozen=True)
class StateEffectSpec:
    state_variable: str
    subject_parameter: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="state effect variable"),
        )
        object.__setattr__(
            self,
            "subject_parameter",
            _text(self.subject_parameter, label="state effect subject parameter"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "state_variable": self.state_variable,
            "subject_parameter": self.subject_parameter,
        }


@dataclass(frozen=True)
class EventTypeSpec:
    name: str
    actor_type: str | None
    parameters: tuple[ParameterSpec, ...]
    effects: tuple[StateEffectSpec, ...]
    transition_hook: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="event type name"))
        if self.actor_type is not None:
            object.__setattr__(
                self,
                "actor_type",
                _text(self.actor_type, label="event actor type"),
            )
        parameters = tuple(self.parameters)
        if any(not isinstance(item, ParameterSpec) for item in parameters):
            raise TypeError("event parameters must be ParameterSpec values")
        _unique_named(parameters, label="event parameter")
        effects = tuple(self.effects)
        if any(not isinstance(item, StateEffectSpec) for item in effects):
            raise TypeError("event effects must be StateEffectSpec values")
        if len({(item.state_variable, item.subject_parameter) for item in effects}) != len(
            effects
        ):
            raise ValueError("event effects must be unique")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "effects", effects)
        object.__setattr__(
            self,
            "transition_hook",
            _text(self.transition_hook, label="event transition hook"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "actor_type": self.actor_type,
            "parameters": [item.to_dict() for item in self.parameters],
            "effects": [item.to_dict() for item in self.effects],
            "transition_hook": self.transition_hook,
        }


@dataclass(frozen=True)
class ActionTypeSpec:
    name: str
    parameters: tuple[ParameterSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="action type name"))
        parameters = tuple(self.parameters)
        if any(not isinstance(item, ParameterSpec) for item in parameters):
            raise TypeError("action parameters must be ParameterSpec values")
        _unique_named(parameters, label="action parameter")
        object.__setattr__(self, "parameters", parameters)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "parameters": [item.to_dict() for item in self.parameters],
        }


@dataclass(frozen=True)
class DecisionTypeSpec:
    name: str
    actor_type: str
    action_type: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="decision type name"))
        object.__setattr__(
            self,
            "actor_type",
            _text(self.actor_type, label="decision actor type"),
        )
        object.__setattr__(
            self,
            "action_type",
            _text(self.action_type, label="decision action type"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "actor_type": self.actor_type,
            "action_type": self.action_type,
        }


@dataclass(frozen=True)
class StateDeltaOp:
    kind: str
    subject_id: str
    state_variable: str
    value: TypedValue | None = None

    def __post_init__(self) -> None:
        kind = _text(self.kind, label="state delta kind")
        if kind not in _DELTA_KINDS:
            raise ValueError("state delta kind must be set or clear")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(
            self,
            "subject_id",
            _text(self.subject_id, label="state delta subject"),
        )
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="state delta variable"),
        )
        if self.value is not None and not isinstance(self.value, TypedValue):
            raise TypeError("state delta value must be a TypedValue or None")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "subject_id": self.subject_id,
            "state_variable": self.state_variable,
            "value": None if self.value is None else self.value.to_dict(),
        }


@dataclass(frozen=True)
class StateDelta:
    operations: tuple[StateDeltaOp, ...]

    def __post_init__(self) -> None:
        operations = tuple(self.operations)
        if any(not isinstance(item, StateDeltaOp) for item in operations):
            raise TypeError("state delta operations must be StateDeltaOp values")
        object.__setattr__(self, "operations", operations)

    def to_dict(self) -> dict[str, object]:
        return {"operations": [item.to_dict() for item in self.operations]}


@dataclass(frozen=True)
class SemanticHookBinding:
    name: str
    declared_purpose: str
    hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, label="semantic hook name"))
        object.__setattr__(
            self,
            "declared_purpose",
            _text(self.declared_purpose, label="semantic hook purpose"),
        )
        if not callable(self.hook):
            raise TypeError("semantic hook must be callable")

    @property
    def implementation_hash(self) -> str:
        return measure_implementation(self.hook).content_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "hook_name": self.name,
            "declared_purpose": self.declared_purpose,
            "implementation_hash": self.implementation_hash,
        }


@dataclass(frozen=True)
class DomainSpec:
    domain_id: str
    version: str
    entity_types: tuple[EntityTypeSpec, ...]
    value_types: tuple[ValueTypeSpec, ...]
    state_variables: tuple[StateVariableSpec, ...]
    event_types: tuple[EventTypeSpec, ...]
    action_types: tuple[ActionTypeSpec, ...]
    decision_types: tuple[DecisionTypeSpec, ...]
    semantic_hooks: tuple[SemanticHookBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain_id", _text(self.domain_id, label="domain id"))
        object.__setattr__(self, "version", _text(self.version, label="domain version"))
        typed_fields = (
            ("entity_types", EntityTypeSpec),
            ("value_types", ValueTypeSpec),
            ("state_variables", StateVariableSpec),
            ("event_types", EventTypeSpec),
            ("action_types", ActionTypeSpec),
            ("decision_types", DecisionTypeSpec),
            ("semantic_hooks", SemanticHookBinding),
        )
        for name, expected in typed_fields:
            values = tuple(getattr(self, name))
            if any(not isinstance(item, expected) for item in values):
                raise TypeError(
                    f"domain {name} must contain {expected.__name__} values"
                )
            _unique_named(values, label=f"domain {name}")
            object.__setattr__(
                self, name, tuple(sorted(values, key=lambda item: item.name))
            )

        entity_types = {item.name: item for item in self.entity_types}
        value_types = {item.name: item for item in self.value_types}
        state_variables = {item.name: item for item in self.state_variables}
        action_types = {item.name: item for item in self.action_types}
        hooks = {item.name: item for item in self.semantic_hooks}

        for value_type in self.value_types:
            if value_type.kind == "entity_ref" and value_type.entity_type not in entity_types:
                raise ValueError("entity_ref value type must reference a declared entity type")
        for state_variable in self.state_variables:
            if state_variable.subject_type not in entity_types:
                raise ValueError("state variable subject type must be declared")
            if state_variable.value_type not in value_types:
                raise ValueError("state variable value type must be declared")
        for event_type in self.event_types:
            if event_type.actor_type is not None and event_type.actor_type not in entity_types:
                raise ValueError("event actor type must be declared")
            parameters = {item.name: item for item in event_type.parameters}
            for parameter in event_type.parameters:
                if parameter.value_type not in value_types:
                    raise ValueError("event parameter value type must be declared")
            for effect in event_type.effects:
                state_variable = state_variables.get(effect.state_variable)
                if state_variable is None:
                    raise ValueError("event effect state variable must be declared")
                subject_parameter = parameters.get(effect.subject_parameter)
                if subject_parameter is None:
                    raise ValueError("event effect subject parameter must be declared")
                subject_value_type = value_types[subject_parameter.value_type]
                if (
                    subject_value_type.kind != "entity_ref"
                    or subject_value_type.entity_type != state_variable.subject_type
                ):
                    raise ValueError(
                        "event effect subject parameter must reference the state subject type"
                    )
            if event_type.transition_hook not in hooks:
                raise ValueError("event transition hook must be declared")
        for action_type in self.action_types:
            for parameter in action_type.parameters:
                if parameter.value_type not in value_types:
                    raise ValueError("action parameter value type must be declared")
        for decision_type in self.decision_types:
            if decision_type.actor_type not in entity_types:
                raise ValueError("decision actor type must be declared")
            if decision_type.action_type not in action_types:
                raise ValueError("decision action type must be declared")

    def to_dict(self) -> dict[str, object]:
        return {
            "domain_id": self.domain_id,
            "version": self.version,
            "entity_types": [item.to_dict() for item in self.entity_types],
            "value_types": [item.to_dict() for item in self.value_types],
            "state_variables": [item.to_dict() for item in self.state_variables],
            "event_types": [item.to_dict() for item in self.event_types],
            "action_types": [item.to_dict() for item in self.action_types],
            "decision_types": [item.to_dict() for item in self.decision_types],
            "semantic_hooks": [item.to_dict() for item in self.semantic_hooks],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())

    def _entity_type(self, name: str) -> EntityTypeSpec:
        for item in self.entity_types:
            if item.name == name:
                return item
        raise ValueError(f"undeclared entity type {name!r}")

    def _value_type(self, name: str) -> ValueTypeSpec:
        for item in self.value_types:
            if item.name == name:
                return item
        raise ValueError(f"undeclared value type {name!r}")

    def _state_variable(self, name: str) -> StateVariableSpec:
        for item in self.state_variables:
            if item.name == name:
                return item
        raise ValueError(f"undeclared state variable {name!r}")

    def _event_type(self, name: str) -> EventTypeSpec:
        for item in self.event_types:
            if item.name == name:
                return item
        raise ValueError(f"undeclared event type {name!r}")

    def _action_type(self, name: str) -> ActionTypeSpec:
        for item in self.action_types:
            if item.name == name:
                return item
        raise ValueError(f"undeclared action type {name!r}")

    def _decision_type(self, name: str) -> DecisionTypeSpec:
        for item in self.decision_types:
            if item.name == name:
                return item
        raise ValueError(f"undeclared decision type {name!r}")

    def _hook(self, name: str) -> SemanticHookBinding:
        for item in self.semantic_hooks:
            if item.name == name:
                return item
        raise ValueError(f"undeclared semantic hook {name!r}")

    def apply_event(
        self,
        prior_state: Mapping[object, TypedValue],
        event: NarrativeEvent,
        entities: Mapping[str, Entity],
    ) -> StateDelta:
        if not isinstance(prior_state, Mapping):
            raise TypeError("prior state must be a mapping")
        if not isinstance(event, NarrativeEvent):
            raise TypeError("domain event must be a NarrativeEvent")
        if not isinstance(entities, Mapping):
            raise TypeError("domain entities must be a mapping")
        entity_map = dict(entities)
        if any(
            not isinstance(key, str)
            or not isinstance(value, Entity)
            or key != value.id
            for key, value in entity_map.items()
        ):
            raise TypeError("domain entity map must map exact entity IDs to Entity values")

        event_type = self._event_type(event.type_name)
        if event_type.actor_type is None:
            if event.actor_id is not None:
                raise ValueError("event type does not permit an actor")
        else:
            if event.actor_id is None:
                raise ValueError("event type requires an actor")
            actor = entity_map.get(event.actor_id)
            if actor is None or actor.type_name != event_type.actor_type:
                raise ValueError("event actor must match the declared actor type")

        parameter_map = {item.name: item for item in event_type.parameters}
        if set(event.arguments) != set(parameter_map):
            raise ValueError("event arguments must match declared parameters exactly")
        for name, parameter in parameter_map.items():
            self._value_type(parameter.value_type).validate(event.arguments[name], entity_map)

        result = self._hook(event_type.transition_hook).hook(
            MappingProxyType(dict(prior_state)), event
        )
        if not isinstance(result, StateDelta):
            raise TypeError("semantic hook must return StateDelta")

        expected_effects: set[tuple[str, str]] = set()
        for effect in event_type.effects:
            subject = event.arguments[effect.subject_parameter].value
            if not isinstance(subject, EntityRef):
                raise TypeError("state effect subject parameter must contain an EntityRef")
            expected_effects.add((effect.state_variable, subject.entity_id))

        seen_cells: set[tuple[str, str]] = set()
        for operation in result.operations:
            state_variable = self._state_variable(operation.state_variable)
            cell = (operation.state_variable, operation.subject_id)
            if cell not in expected_effects:
                raise ValueError("semantic hook wrote outside declared event effects")
            if cell in seen_cells:
                raise ValueError("semantic hook cannot write one state cell more than once")
            seen_cells.add(cell)
            subject = entity_map.get(operation.subject_id)
            if subject is None:
                raise ValueError("state delta subject must reference a declared entity")
            if subject.type_name != state_variable.subject_type:
                raise ValueError("state delta subject type does not match state variable")
            if operation.kind == "clear":
                if operation.value is not None:
                    raise ValueError("clear state delta cannot contain a value")
            else:
                if operation.value is None:
                    raise ValueError("set state delta requires a value")
                self._value_type(state_variable.value_type).validate(
                    operation.value, entity_map
                )
        return result


def _validate_proposition(
    proposition: Proposition,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> None:
    if not isinstance(proposition, Proposition):
        raise TypeError("claim proposition must be a Proposition")
    subject = entities.get(proposition.subject.entity_id)
    if subject is None:
        raise ValueError("proposition subject must reference a declared entity")
    if subject.type_name != proposition.subject.entity_type:
        raise ValueError("proposition subject entity type does not match declaration")
    state_variable = domain._state_variable(proposition.state_variable)
    if subject.type_name != state_variable.subject_type:
        raise ValueError("proposition subject type does not match state variable")
    domain._value_type(state_variable.value_type).validate(proposition.value, entities)


def _validate_action(
    action: ActionOption,
    action_type: ActionTypeSpec,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> None:
    if action.type_name != action_type.name:
        raise ValueError("decision action type must match declared decision type")
    parameter_map = {item.name: item for item in action_type.parameters}
    if set(action.arguments) != set(parameter_map):
        raise ValueError("decision action arguments must match declared parameters exactly")
    for name, parameter in parameter_map.items():
        domain._value_type(parameter.value_type).validate(action.arguments[name], entities)


def validate_narrative(story: GenericNarrative, domain: DomainSpec) -> None:
    if not isinstance(story, GenericNarrative):
        raise TypeError("generic narrative validation requires GenericNarrative")
    if not isinstance(domain, DomainSpec):
        raise TypeError("generic narrative validation requires DomainSpec")
    if (
        story.domain_id,
        story.domain_version,
        story.domain_spec_hash,
    ) != (domain.domain_id, domain.version, domain.content_hash):
        raise ValueError("generic narrative domain identity does not match DomainSpec")

    _unique_ids(story.entities, label="entity")
    _unique_ids(story.events, label="event")
    _unique_ids(story.observations, label="observation")
    _unique_ids(story.claims, label="claim")
    _unique_ids(story.receptions, label="reception")
    _unique_ids(story.decisions, label="decision")

    entity_by_id = {item.id: item for item in story.entities}
    for entity in story.entities:
        domain._entity_type(entity.type_name)

    event_by_id = {item.id: item for item in story.events}
    claim_by_id = {item.id: item for item in story.claims}
    if set(event_by_id) & set(claim_by_id):
        raise ValueError("event and claim support ids must be globally unique")

    timed = (
        tuple((item.logical_time, "event", item.id) for item in story.events)
        + tuple((item.logical_time, "claim", item.id) for item in story.claims)
        + tuple((item.logical_time, "decision", item.id) for item in story.decisions)
    )
    if len({item[0] for item in timed}) != len(timed):
        raise ValueError("event, claim, and decision logical times must be globally unique")

    world_state: dict[StateCellRef, TypedValue] = {}
    for event in sorted(story.events, key=lambda item: item.logical_time):
        delta = domain.apply_event(world_state, event, entity_by_id)
        for operation in delta.operations:
            subject = entity_by_id[operation.subject_id]
            cell = StateCellRef(
                EntityRef(subject.id, subject.type_name), operation.state_variable
            )
            if operation.kind == "clear":
                world_state.pop(cell, None)
            else:
                assert operation.value is not None
                world_state[cell] = operation.value

    observed_pairs: set[tuple[str, str]] = set()
    for observation in story.observations:
        if observation.channel != "direct":
            raise ValueError("observation channel must be direct")
        if observation.event_id not in event_by_id:
            raise ValueError("observation event must reference a declared event")
        if observation.agent_id not in entity_by_id:
            raise ValueError("observation agent must reference a declared entity")
        pair = (observation.event_id, observation.agent_id)
        if pair in observed_pairs:
            raise ValueError("observation event-agent pairs must be unique")
        observed_pairs.add(pair)

    reception_pairs: set[tuple[str, str]] = set()
    for reception in story.receptions:
        if reception.channel != "direct_testimony":
            raise ValueError("reception channel must be direct_testimony")
        if reception.claim_id not in claim_by_id:
            raise ValueError("reception claim must reference a declared claim")
        if reception.recipient_id not in entity_by_id:
            raise ValueError("reception recipient must reference a declared entity")
        pair = (reception.claim_id, reception.recipient_id)
        if pair in reception_pairs:
            raise ValueError("reception claim-recipient pairs must be unique")
        reception_pairs.add(pair)

    for claim in story.claims:
        if claim.speaker_id not in entity_by_id:
            raise ValueError("claim speaker must reference a declared entity")
        _validate_proposition(claim.proposition, domain, entity_by_id)
        if not claim.support_refs:
            raise ValueError("claim requires at least one support reference")
        if len(set(claim.support_refs)) != len(claim.support_refs):
            raise ValueError("claim support references must be unique")
        for support_ref in claim.support_refs:
            support_event = event_by_id.get(support_ref)
            if support_event is not None:
                if support_event.logical_time >= claim.logical_time:
                    raise ValueError("claim support must occur before the claim")
                if (support_event.id, claim.speaker_id) not in observed_pairs:
                    raise ValueError(
                        "claim speaker must have access to every support reference"
                    )
                continue
            support_claim = claim_by_id.get(support_ref)
            if support_claim is None:
                raise ValueError("claim support must reference a declared event or claim")
            if support_claim.logical_time >= claim.logical_time:
                raise ValueError("claim support must occur before the claim")
            if (support_claim.id, claim.speaker_id) not in reception_pairs:
                raise ValueError(
                    "claim speaker must have access to every support reference"
                )

    for decision in story.decisions:
        decision_type = domain._decision_type(decision.type_name)
        actor = entity_by_id.get(decision.actor_id)
        if actor is None:
            raise ValueError("decision actor must reference a declared entity")
        if actor.type_name != decision_type.actor_type:
            raise ValueError("decision actor type must match declared decision type")
        if not decision.context_cells:
            raise ValueError("decision requires at least one context state cell")
        for cell in decision.context_cells:
            subject = entity_by_id.get(cell.subject.entity_id)
            if subject is None:
                raise ValueError("decision context subject must reference a declared entity")
            if subject.type_name != cell.subject.entity_type:
                raise ValueError("decision context subject type does not match entity")
            state_variable = domain._state_variable(cell.state_variable)
            if subject.type_name != state_variable.subject_type:
                raise ValueError("decision context subject type does not match state variable")
        if not decision.actions:
            raise ValueError("decision requires at least one action")
        if len({item.id for item in decision.actions}) != len(decision.actions):
            raise ValueError("decision action ids must be unique")
        action_type = domain._action_type(decision_type.action_type)
        for action in decision.actions:
            _validate_action(action, action_type, domain, entity_by_id)
