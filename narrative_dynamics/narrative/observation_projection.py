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
from narrative_dynamics.narrative.domain import (
    DomainSpec,
    StateDelta,
    StateDeltaOp,
    validate_narrative,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Entity,
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.world import (
    ActionIntent,
    ActionTransitionRecord,
    WorldState,
    WorldStepResult,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCOPES = frozenset({"observer", "any"})
_RELATIONS = frozenset({"equals", "clear"})


class ObservationProjectionError(ValueError):
    """A runtime world step could not be projected safely."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _step_index(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


@dataclass(frozen=True)
class ObservationCapabilitySpec:
    state_variable: str
    subject_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_variable",
            _text(
                self.state_variable,
                label="observation capability state variable",
            ),
        )
        scope = _text(
            self.subject_scope,
            label="observation capability subject scope",
        )
        if scope not in _SCOPES:
            raise ValueError(
                "observation capability subject scope must be observer or any"
            )
        object.__setattr__(self, "subject_scope", scope)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_variable": self.state_variable,
            "subject_scope": self.subject_scope,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _capability_key(
    capability: ObservationCapabilitySpec,
) -> tuple[str, str]:
    return (capability.state_variable, capability.subject_scope)


def _contains(
    read: ObservationCapabilitySpec,
    emit: ObservationCapabilitySpec,
) -> bool:
    return (
        read.state_variable == emit.state_variable
        and (
            read.subject_scope == "any"
            or emit.subject_scope == "observer"
        )
    )


@dataclass(frozen=True)
class ObserverProjectionSpec:
    observer_type: str
    channel: str
    read_capabilities: tuple[ObservationCapabilitySpec, ...]
    emit_capabilities: tuple[ObservationCapabilitySpec, ...]
    projection_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_type",
            _text(self.observer_type, label="projection observer type"),
        )
        object.__setattr__(
            self,
            "channel",
            _text(self.channel, label="projection channel"),
        )
        if not isinstance(self.read_capabilities, tuple):
            raise TypeError("projection read capabilities must be a tuple")
        if not isinstance(self.emit_capabilities, tuple):
            raise TypeError("projection emit capabilities must be a tuple")
        reads = tuple(self.read_capabilities)
        emits = tuple(self.emit_capabilities)
        if not reads:
            raise ValueError("projection read capabilities must be non-empty")
        if not emits:
            raise ValueError("projection emit capabilities must be non-empty")
        if any(not isinstance(item, ObservationCapabilitySpec) for item in reads):
            raise TypeError(
                "projection read capabilities must contain ObservationCapabilitySpec values"
            )
        if any(not isinstance(item, ObservationCapabilitySpec) for item in emits):
            raise TypeError(
                "projection emit capabilities must contain ObservationCapabilitySpec values"
            )
        read_keys = tuple(_capability_key(item) for item in reads)
        emit_keys = tuple(_capability_key(item) for item in emits)
        if len(set(read_keys)) != len(read_keys):
            raise ValueError("projection read capabilities must be unique")
        if len(set(emit_keys)) != len(emit_keys):
            raise ValueError("projection emit capabilities must be unique")
        for emit in emits:
            if not any(_contains(read, emit) for read in reads):
                raise ValueError(
                    "projection emit capability must be contained by a read capability"
                )
        if not callable(self.projection_hook):
            raise TypeError("projection hook must be callable")
        object.__setattr__(
            self,
            "read_capabilities",
            tuple(sorted(reads, key=_capability_key)),
        )
        object.__setattr__(
            self,
            "emit_capabilities",
            tuple(sorted(emits, key=_capability_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_type": self.observer_type,
            "channel": self.channel,
            "read_capabilities": [
                item.to_dict() for item in self.read_capabilities
            ],
            "emit_capabilities": [
                item.to_dict() for item in self.emit_capabilities
            ],
            "implementation_identity": measure_implementation(
                self.projection_hook
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ObservationFact:
    cell: StateCellRef
    relation: str
    value: TypedValue | None

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("observation fact cell must be a StateCellRef")
        relation = _text(self.relation, label="observation fact relation")
        if relation not in _RELATIONS:
            raise ValueError("observation fact relation must be equals or clear")
        object.__setattr__(self, "relation", relation)
        if relation == "equals":
            if not isinstance(self.value, TypedValue):
                raise TypeError("equals observation fact requires a TypedValue")
        elif self.value is not None:
            raise ValueError("clear observation fact cannot contain a value")

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "relation": self.relation,
            "value": None if self.value is None else self.value.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ProjectedObservation:
    observer_id: str
    channel: str
    fact: ObservationFact
    step_index: int
    source_world_state_hash: str
    source_world_step_hash: str
    source_transition_hashes: tuple[str, ...]
    projection_spec_hash: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_id",
            _text(self.observer_id, label="projected observation observer id"),
        )
        object.__setattr__(
            self,
            "channel",
            _text(self.channel, label="projected observation channel"),
        )
        if not isinstance(self.fact, ObservationFact):
            raise TypeError(
                "projected observation fact must be an ObservationFact"
            )
        object.__setattr__(
            self,
            "step_index",
            _step_index(
                self.step_index,
                label="projected observation step index",
            ),
        )
        object.__setattr__(
            self,
            "source_world_state_hash",
            _hash(
                self.source_world_state_hash,
                label="projected observation source world state hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_step_hash",
            _hash(
                self.source_world_step_hash,
                label="projected observation source world step hash",
            ),
        )
        if not isinstance(self.source_transition_hashes, tuple):
            raise TypeError(
                "projected observation source transition hashes must be a tuple"
            )
        hashes = tuple(
            _hash(
                item,
                label="projected observation source transition hash",
            )
            for item in self.source_transition_hashes
        )
        if len(set(hashes)) != len(hashes):
            raise ValueError(
                "projected observation source transition hashes must be unique"
            )
        object.__setattr__(self, "source_transition_hashes", tuple(sorted(hashes)))
        object.__setattr__(
            self,
            "projection_spec_hash",
            _hash(
                self.projection_spec_hash,
                label="projected observation projection spec hash",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_id": self.observer_id,
            "channel": self.channel,
            "fact": self.fact.to_dict(),
            "step_index": self.step_index,
            "source_world_state_hash": self.source_world_state_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "source_transition_hashes": list(self.source_transition_hashes),
            "projection_spec_hash": self.projection_spec_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _projection_key(
    spec: ObserverProjectionSpec,
) -> tuple[str, str]:
    return (spec.observer_type, spec.channel)


@dataclass(frozen=True)
class ObservationProjectionModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    projections: tuple[ObserverProjectionSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="observation projection model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="observation projection model version"),
        )
        object.__setattr__(
            self,
            "domain_id",
            _text(self.domain_id, label="observation projection domain id"),
        )
        object.__setattr__(
            self,
            "domain_version",
            _text(
                self.domain_version,
                label="observation projection domain version",
            ),
        )
        object.__setattr__(
            self,
            "domain_spec_hash",
            _hash(
                self.domain_spec_hash,
                label="observation projection domain spec hash",
            ),
        )
        if not isinstance(self.projections, tuple):
            raise TypeError(
                "observation projection model projections must be a tuple"
            )
        projections = tuple(self.projections)
        if any(not isinstance(item, ObserverProjectionSpec) for item in projections):
            raise TypeError(
                "observation projection model projections must contain "
                "ObserverProjectionSpec values"
            )
        keys = tuple(_projection_key(item) for item in projections)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "observation projection model observer/channel pairs must be unique"
            )
        object.__setattr__(
            self,
            "projections",
            tuple(sorted(projections, key=_projection_key)),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "projections": [item.to_dict() for item in self.projections],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _observation_key(
    observation: ProjectedObservation,
) -> tuple[str, str, str, str, str]:
    cell = observation.fact.cell
    return (
        observation.observer_id,
        observation.channel,
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


@dataclass(frozen=True)
class ObservationProjectionResult:
    model_id: str
    model_hash: str
    source_world_step_hash: str
    source_world_state_hash: str
    step_index: int
    observations: tuple[ProjectedObservation, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="observation projection result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="observation projection result model hash"),
        )
        object.__setattr__(
            self,
            "source_world_step_hash",
            _hash(
                self.source_world_step_hash,
                label="observation projection result source world step hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_state_hash",
            _hash(
                self.source_world_state_hash,
                label="observation projection result source world state hash",
            ),
        )
        object.__setattr__(
            self,
            "step_index",
            _step_index(
                self.step_index,
                label="observation projection result step index",
            ),
        )
        if not isinstance(self.observations, tuple):
            raise TypeError(
                "observation projection result observations must be a tuple"
            )
        observations = tuple(self.observations)
        if any(not isinstance(item, ProjectedObservation) for item in observations):
            raise TypeError(
                "observation projection result observations must contain "
                "ProjectedObservation values"
            )
        observations = tuple(sorted(observations, key=_observation_key))
        keys = tuple(_observation_key(item) for item in observations)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "observation projection result observation keys must be unique"
            )
        for item in observations:
            if item.source_world_step_hash != self.source_world_step_hash:
                raise ValueError(
                    "projected observation must bind result source world step"
                )
            if item.source_world_state_hash != self.source_world_state_hash:
                raise ValueError(
                    "projected observation must bind result source world state"
                )
            if item.step_index != self.step_index:
                raise ValueError(
                    "projected observation must bind result step index"
                )
        object.__setattr__(self, "observations", observations)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "source_world_state_hash": self.source_world_state_hash,
            "step_index": self.step_index,
            "observations": [item.to_dict() for item in self.observations],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _projection_hash(spec: ObserverProjectionSpec) -> str:
    try:
        return spec.content_hash
    except ImplementationAttestationUnavailable as error:
        raise ObservationProjectionError(
            "observation projection implementation attestation is unavailable"
        ) from error


def _model_hash(model: ObservationProjectionModelSpec) -> str:
    try:
        return model.content_hash
    except ImplementationAttestationUnavailable as error:
        raise ObservationProjectionError(
            "observation projection model attestation is unavailable"
        ) from error


def _record_key(
    record: ActionTransitionRecord,
) -> tuple[str, str, str]:
    return (
        record.actor_id,
        record.intent.decision_id,
        record.action.id,
    )


def _batch_hash(
    records: tuple[ActionTransitionRecord, ...],
) -> str:
    return stable_content_hash(
        [record.to_dict() for record in sorted(records, key=_record_key)]
    )


def _validate_state_values(
    state: WorldState,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    *,
    label: str,
) -> None:
    if not isinstance(state, WorldState):
        raise TypeError(f"{label} world state must be WorldState")
    if not isinstance(state.values, Mapping):
        raise TypeError(f"{label} world values must be a mapping")
    for cell, value in state.values.items():
        if not isinstance(cell, StateCellRef):
            raise TypeError(f"{label} world cell must be StateCellRef")
        if not isinstance(value, TypedValue):
            raise TypeError(f"{label} world value must be TypedValue")
        subject = entities.get(cell.subject.entity_id)
        if subject is None or subject.type_name != cell.subject.entity_type:
            raise ValueError(f"{label} world cell subject is not canonical")
        variable = domain._state_variable(cell.state_variable)
        if variable.subject_type != subject.type_name:
            raise ValueError(f"{label} world cell subject type mismatch")
        domain._value_type(variable.value_type).validate(value, entities)


def _resolve_canonical_action(
    story: GenericNarrative,
    record: ActionTransitionRecord,
) -> tuple[object, ActionOption]:
    decision = next(
        (
            item
            for item in story.decisions
            if item.id == record.intent.decision_id
        ),
        None,
    )
    if decision is None:
        raise ValueError("transition intent decision is not canonical")
    action = next(
        (
            item
            for item in decision.actions
            if item.id == record.intent.selected_action
        ),
        None,
    )
    if action is None:
        raise ValueError("transition intent action is not canonical")
    if record.actor_id != decision.actor_id:
        raise ValueError("transition actor does not match canonical decision")
    if record.action is not action:
        raise ValueError("transition action is not the canonical selected action")
    return decision, action


def _validate_source_world_step(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    model: ObservationProjectionModelSpec,
) -> dict[str, Entity]:
    if not isinstance(story, GenericNarrative):
        raise TypeError("observation projection requires GenericNarrative")
    if not isinstance(domain, DomainSpec):
        raise TypeError("observation projection requires DomainSpec")
    if not isinstance(world_step, WorldStepResult):
        raise TypeError("observation projection requires WorldStepResult")
    if not isinstance(model, ObservationProjectionModelSpec):
        raise TypeError(
            "observation projection requires ObservationProjectionModelSpec"
        )

    validate_narrative(story, domain)
    domain_identity = (
        domain.domain_id,
        domain.version,
        domain.content_hash,
    )
    if (
        model.domain_id,
        model.domain_version,
        model.domain_spec_hash,
    ) != domain_identity:
        raise ValueError(
            "observation projection model domain identity does not match DomainSpec"
        )

    prior = world_step.prior_state
    next_state = world_step.next_state
    for label, state in (("prior", prior), ("next", next_state)):
        if (
            state.domain_id,
            state.domain_version,
            state.domain_spec_hash,
        ) != domain_identity:
            raise ValueError(
                f"observation projection {label} world state domain identity mismatch"
            )
        if state.source_story_hash != story.content_hash:
            raise ValueError(
                f"observation projection {label} world state story identity mismatch"
            )
    if next_state.source_at_time != prior.source_at_time:
        raise ValueError(
            "observation projection world states must preserve source cutoff"
        )
    if next_state.step_index != prior.step_index + 1:
        raise ValueError(
            "observation projection next world state must increment step index"
        )
    if next_state.parent_state_hash != prior.content_hash:
        raise ValueError(
            "observation projection next world state must bind exact prior state"
        )

    entities = {entity.id: entity for entity in story.entities}
    _validate_state_values(prior, domain, entities, label="prior")
    _validate_state_values(next_state, domain, entities, label="next")

    if not isinstance(world_step.transitions, tuple) or not world_step.transitions:
        raise ValueError("observation projection world step requires transitions")
    records = tuple(world_step.transitions)
    if any(not isinstance(record, ActionTransitionRecord) for record in records):
        raise TypeError(
            "observation projection transitions must be ActionTransitionRecord values"
        )

    actors: set[str] = set()
    written: set[StateCellRef] = set()
    replayed = dict(prior.values)
    prior_hash = prior.content_hash

    for record in records:
        if not isinstance(record.intent, ActionIntent):
            raise TypeError("transition record intent must be ActionIntent")
        if record.prior_state_hash != prior_hash:
            raise ValueError("transition record does not bind exact prior state")

        decision, _ = _resolve_canonical_action(story, record)
        if (
            prior.source_at_time is not None
            and decision.logical_time > prior.source_at_time
        ):
            raise ValueError("transition decision occurs after source cutoff")
        if record.actor_id in actors:
            raise ValueError("one actor may appear at most once per world step")
        actors.add(record.actor_id)

        if not isinstance(record.delta, StateDelta):
            raise TypeError("transition delta must be StateDelta")
        if not isinstance(record.delta.operations, tuple):
            raise TypeError("transition delta operations must be a tuple")
        local_written: set[StateCellRef] = set()
        for operation in record.delta.operations:
            if not isinstance(operation, StateDeltaOp):
                raise TypeError("transition delta operation must be StateDeltaOp")
            subject = entities.get(operation.subject_id)
            if subject is None:
                raise ValueError("transition delta subject is not canonical")
            variable = domain._state_variable(operation.state_variable)
            if variable.subject_type != subject.type_name:
                raise ValueError("transition delta subject type mismatch")
            cell = StateCellRef(
                EntityRef(subject.id, subject.type_name),
                variable.name,
            )
            if cell in local_written:
                raise ValueError("one transition cannot write one cell twice")
            if cell in written:
                raise ValueError("world transition records cannot overlap writes")
            local_written.add(cell)
            written.add(cell)

            if operation.kind == "clear":
                if operation.value is not None:
                    raise ValueError("clear transition delta cannot contain a value")
                replayed.pop(cell, None)
            elif operation.kind == "set":
                if operation.value is None:
                    raise ValueError("set transition delta requires a value")
                domain._value_type(variable.value_type).validate(
                    operation.value,
                    entities,
                )
                replayed[cell] = operation.value
            else:
                raise ValueError("transition delta operation kind is unsupported")

    if replayed != dict(next_state.values):
        raise ValueError(
            "next world state is not the extensional result of transition deltas"
        )
    if next_state.transition_batch_hash != _batch_hash(records):
        raise ValueError(
            "next world state does not bind exact transition batch identity"
        )
    return entities


def _validate_projection_declarations(
    domain: DomainSpec,
    model: ObservationProjectionModelSpec,
) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for spec in model.projections:
        domain._entity_type(spec.observer_type)
        for capability in (
            spec.read_capabilities + spec.emit_capabilities
        ):
            variable = domain._state_variable(capability.state_variable)
            if (
                capability.subject_scope == "observer"
                and variable.subject_type != spec.observer_type
            ):
                raise ValueError(
                    "observer-scoped capability state subject type "
                    "must match observer type"
                )
        hashes[_projection_key(spec)] = _projection_hash(spec)
    return hashes


def _can_read(
    capability: ObservationCapabilitySpec,
    cell: StateCellRef,
    observer: Entity,
) -> bool:
    if capability.state_variable != cell.state_variable:
        return False
    return (
        capability.subject_scope == "any"
        or cell.subject.entity_id == observer.id
    )


def _visible_values(
    values: Mapping[StateCellRef, TypedValue],
    capabilities: tuple[ObservationCapabilitySpec, ...],
    observer: Entity,
) -> Mapping[StateCellRef, TypedValue]:
    visible = {
        cell: value
        for cell, value in values.items()
        if any(
            _can_read(capability, cell, observer)
            for capability in capabilities
        )
    }
    return MappingProxyType(
        dict(sorted(visible.items(), key=lambda item: _cell_key(item[0])))
    )


def _emit_covers(
    capability: ObservationCapabilitySpec,
    cell: StateCellRef,
    observer: Entity,
) -> bool:
    if capability.state_variable != cell.state_variable:
        return False
    return (
        capability.subject_scope == "any"
        or cell.subject.entity_id == observer.id
    )


def _hook_facts(
    spec: ObserverProjectionSpec,
    prior_visible: Mapping[StateCellRef, TypedValue],
    next_visible: Mapping[StateCellRef, TypedValue],
    observer: Entity,
    step_index: int,
) -> tuple[ObservationFact, ...]:
    try:
        raw = spec.projection_hook(
            prior_visible,
            next_visible,
            observer,
            step_index,
        )
    except Exception as error:
        raise ObservationProjectionError(
            "observation projection hook execution failed"
        ) from error
    if type(raw) is not tuple:
        raise ObservationProjectionError(
            "observation projection hook must return an exact tuple"
        )
    if any(not isinstance(fact, ObservationFact) for fact in raw):
        raise ObservationProjectionError(
            "observation projection hook tuple must contain ObservationFact values"
        )
    cells = tuple(fact.cell for fact in raw)
    if len(set(cells)) != len(cells):
        raise ObservationProjectionError(
            "observation projection hook cannot emit one cell twice"
        )
    return raw


def _validate_fact_cell_shape(
    fact: ObservationFact,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> None:
    subject = entities.get(fact.cell.subject.entity_id)
    if (
        subject is None
        or subject.type_name != fact.cell.subject.entity_type
    ):
        raise ValueError("observation fact subject is not canonical")
    variable = domain._state_variable(fact.cell.state_variable)
    if variable.subject_type != subject.type_name:
        raise ValueError("observation fact subject type mismatch")


def _records_writing_cell(
    world_step: WorldStepResult,
    cell: StateCellRef,
) -> tuple[ActionTransitionRecord, ...]:
    matches = []
    for record in world_step.transitions:
        if any(
            operation.subject_id == cell.subject.entity_id
            and operation.state_variable == cell.state_variable
            for operation in record.delta.operations
        ):
            matches.append(record)
    return tuple(sorted(matches, key=_record_key))


def _accept_facts(
    facts: tuple[ObservationFact, ...],
    *,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    world_step: WorldStepResult,
    observer: Entity,
    spec: ObserverProjectionSpec,
    spec_hash: str,
) -> tuple[ProjectedObservation, ...]:
    accepted: list[ProjectedObservation] = []
    for fact in facts:
        cell = fact.cell
        _validate_fact_cell_shape(fact, domain, entities)
        variable = domain._state_variable(cell.state_variable)
        writing_records = _records_writing_cell(world_step, cell)

        if fact.relation == "equals":
            assert fact.value is not None
            domain._value_type(variable.value_type).validate(
                fact.value,
                entities,
            )
            expected = world_step.next_state.values.get(cell)
            if expected is None or fact.value != expected:
                raise ValueError(
                    "equals observation does not match post-step truth"
                )
        else:
            if cell in world_step.next_state.values:
                raise ValueError(
                    "clear observation requires post-step absence"
                )
            clear_records = tuple(
                record
                for record in writing_records
                if any(
                    operation.kind == "clear"
                    and operation.subject_id == cell.subject.entity_id
                    and operation.state_variable == cell.state_variable
                    for operation in record.delta.operations
                )
            )
            if len(clear_records) != 1:
                raise ValueError(
                    "clear observation requires one explicit current-step clear"
                )

        accepted.append(
            ProjectedObservation(
                observer_id=observer.id,
                channel=spec.channel,
                fact=fact,
                step_index=world_step.next_state.step_index,
                source_world_state_hash=world_step.next_state.content_hash,
                source_world_step_hash=world_step.content_hash,
                source_transition_hashes=tuple(
                    record.content_hash for record in writing_records
                ),
                projection_spec_hash=spec_hash,
            )
        )
    return tuple(accepted)


def project_world_observations(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    model: ObservationProjectionModelSpec,
) -> ObservationProjectionResult:
    """Project one validated world step into runtime observations."""

    try:
        entities = _validate_source_world_step(
            story,
            domain,
            world_step,
            model,
        )
        model_hash = _model_hash(model)
        spec_hashes = _validate_projection_declarations(domain, model)

        observations: list[ProjectedObservation] = []
        for spec in sorted(model.projections, key=_projection_key):
            spec_hash = spec_hashes[_projection_key(spec)]
            observers = tuple(
                sorted(
                    (
                        entity
                        for entity in story.entities
                        if entity.type_name == spec.observer_type
                    ),
                    key=lambda entity: entity.id,
                )
            )
            for observer in observers:
                prior_visible = _visible_values(
                    world_step.prior_state.values,
                    spec.read_capabilities,
                    observer,
                )
                next_visible = _visible_values(
                    world_step.next_state.values,
                    spec.read_capabilities,
                    observer,
                )
                facts = _hook_facts(
                    spec,
                    prior_visible,
                    next_visible,
                    observer,
                    world_step.next_state.step_index,
                )
                for fact in facts:
                    if not any(
                        _emit_covers(capability, fact.cell, observer)
                        for capability in spec.emit_capabilities
                    ):
                        raise ObservationProjectionError(
                            "observation fact is outside emit capability"
                        )
                    _validate_fact_cell_shape(fact, domain, entities)
                observations.extend(
                    _accept_facts(
                        facts,
                        domain=domain,
                        entities=entities,
                        world_step=world_step,
                        observer=observer,
                        spec=spec,
                        spec_hash=spec_hash,
                    )
                )

        return ObservationProjectionResult(
            model_id=model.model_id,
            model_hash=model_hash,
            source_world_step_hash=world_step.content_hash,
            source_world_state_hash=world_step.next_state.content_hash,
            step_index=world_step.next_state.step_index,
            observations=tuple(observations),
        )
    except ObservationProjectionError:
        raise
    except (
        TypeError,
        ValueError,
        ImplementationAttestationUnavailable,
    ) as error:
        raise ObservationProjectionError(
            "observation projection source validation failed"
        ) from error
