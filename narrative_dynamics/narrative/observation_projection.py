from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
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
)
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
from narrative_dynamics.narrative.randomness import (
    RANDOM_DERIVATION_VERSION,
    RandomSampleRecord,
    sample_categorical,
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


def _freeze_parameters(value: object) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("stochastic projection parameters must be a mapping")
    frozen: dict[str, float] = {}
    for raw_name, raw_value in value.items():
        name = _text(raw_name, label="stochastic projection parameter name")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise TypeError("stochastic projection parameter values must be numeric")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError("stochastic projection parameter values must be finite")
        frozen[name] = number
    return MappingProxyType(dict(sorted(frozen.items())))


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


def _canonical_capabilities(
    reads: object,
    emits: object,
    *,
    label: str,
) -> tuple[
    tuple[ObservationCapabilitySpec, ...],
    tuple[ObservationCapabilitySpec, ...],
]:
    if not isinstance(reads, tuple):
        raise TypeError(f"{label} read capabilities must be a tuple")
    if not isinstance(emits, tuple):
        raise TypeError(f"{label} emit capabilities must be a tuple")
    read_values = tuple(reads)
    emit_values = tuple(emits)
    if not read_values:
        raise ValueError(f"{label} read capabilities must be non-empty")
    if not emit_values:
        raise ValueError(f"{label} emit capabilities must be non-empty")
    if any(
        not isinstance(item, ObservationCapabilitySpec)
        for item in read_values
    ):
        raise TypeError(
            f"{label} read capabilities must contain ObservationCapabilitySpec values"
        )
    if any(
        not isinstance(item, ObservationCapabilitySpec)
        for item in emit_values
    ):
        raise TypeError(
            f"{label} emit capabilities must contain ObservationCapabilitySpec values"
        )
    read_keys = tuple(_capability_key(item) for item in read_values)
    emit_keys = tuple(_capability_key(item) for item in emit_values)
    if len(set(read_keys)) != len(read_keys):
        raise ValueError(f"{label} read capabilities must be unique")
    if len(set(emit_keys)) != len(emit_keys):
        raise ValueError(f"{label} emit capabilities must be unique")
    for emit in emit_values:
        if not any(_contains(read, emit) for read in read_values):
            raise ValueError(
                f"{label} emit capability must be contained by a read capability"
            )
    return (
        tuple(sorted(read_values, key=_capability_key)),
        tuple(sorted(emit_values, key=_capability_key)),
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
        reads, emits = _canonical_capabilities(
            self.read_capabilities,
            self.emit_capabilities,
            label="projection",
        )
        if not callable(self.projection_hook):
            raise TypeError("projection hook must be callable")
        object.__setattr__(self, "read_capabilities", reads)
        object.__setattr__(self, "emit_capabilities", emits)

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
class StochasticObserverProjectionSpec:
    observer_type: str
    channel: str
    read_capabilities: tuple[ObservationCapabilitySpec, ...]
    emit_capabilities: tuple[ObservationCapabilitySpec, ...]
    parameters: Mapping[str, float]
    distribution_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_type",
            _text(
                self.observer_type,
                label="stochastic projection observer type",
            ),
        )
        object.__setattr__(
            self,
            "channel",
            _text(self.channel, label="stochastic projection channel"),
        )
        reads, emits = _canonical_capabilities(
            self.read_capabilities,
            self.emit_capabilities,
            label="stochastic projection",
        )
        object.__setattr__(self, "read_capabilities", reads)
        object.__setattr__(self, "emit_capabilities", emits)
        object.__setattr__(
            self,
            "parameters",
            _freeze_parameters(self.parameters),
        )
        if not callable(self.distribution_hook):
            raise TypeError("stochastic projection hook must be callable")

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
            "parameters": [
                [name, value]
                for name, value in self.parameters.items()
            ],
            "random_derivation_version": RANDOM_DERIVATION_VERSION,
            "implementation_identity": measure_implementation(
                self.distribution_hook
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


ProjectionSpec = ObserverProjectionSpec | StochasticObserverProjectionSpec


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


def _fact_key(fact: ObservationFact) -> tuple[str, str, str, str, str]:
    return (
        fact.cell.subject.entity_type,
        fact.cell.subject.entity_id,
        fact.cell.state_variable,
        fact.relation,
        "" if fact.value is None else fact.value.content_hash,
    )


@dataclass(frozen=True)
class ObservationOutcome:
    outcome_id: str
    probability: float
    facts: tuple[ObservationFact, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "outcome_id",
            _text(self.outcome_id, label="observation outcome id"),
        )
        probability = self.probability
        if isinstance(probability, bool) or not isinstance(
            probability,
            (int, float),
        ):
            raise TypeError("observation outcome probability must be numeric")
        probability = float(probability)
        if (
            not math.isfinite(probability)
            or probability <= 0.0
            or probability > 1.0
        ):
            raise ValueError(
                "observation outcome probability must be finite and in (0, 1]"
            )
        if not isinstance(self.facts, tuple):
            raise TypeError("observation outcome facts must be a tuple")
        facts = tuple(self.facts)
        if any(not isinstance(item, ObservationFact) for item in facts):
            raise TypeError(
                "observation outcome facts must contain ObservationFact values"
            )
        cells = tuple(item.cell for item in facts)
        if len(set(cells)) != len(cells):
            raise ValueError("observation outcome cannot contain one cell twice")
        object.__setattr__(self, "probability", probability)
        object.__setattr__(self, "facts", tuple(sorted(facts, key=_fact_key)))

    def to_dict(self) -> dict[str, object]:
        return {
            "outcome_id": self.outcome_id,
            "probability": self.probability,
            "facts": [item.to_dict() for item in self.facts],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class ObservationOutcomeDistribution:
    outcomes: tuple[ObservationOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.outcomes, tuple):
            raise TypeError("observation outcome distribution outcomes must be a tuple")
        outcomes = tuple(self.outcomes)
        if len(outcomes) < 2:
            raise ValueError(
                "observation outcome distribution requires at least two outcomes"
            )
        if any(not isinstance(item, ObservationOutcome) for item in outcomes):
            raise TypeError(
                "observation outcome distribution must contain ObservationOutcome values"
            )
        outcomes = tuple(sorted(outcomes, key=lambda item: item.outcome_id))
        if len({item.outcome_id for item in outcomes}) != len(outcomes):
            raise ValueError("observation outcome ids must be unique")
        if math.fsum(item.probability for item in outcomes) != 1.0:
            raise ValueError(
                "observation outcome probabilities must sum exactly to one"
            )
        object.__setattr__(self, "outcomes", outcomes)

    def to_dict(self) -> dict[str, object]:
        return {"outcomes": [item.to_dict() for item in self.outcomes]}

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class StochasticObservationSample:
    observer_id: str
    channel: str
    step_index: int
    source_world_state_hash: str
    source_world_step_hash: str
    projection_spec_hash: str
    distribution: ObservationOutcomeDistribution
    sample_record: RandomSampleRecord

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "observer_id",
            _text(self.observer_id, label="stochastic observation observer id"),
        )
        object.__setattr__(
            self,
            "channel",
            _text(self.channel, label="stochastic observation channel"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step_index(
                self.step_index,
                label="stochastic observation step index",
            ),
        )
        object.__setattr__(
            self,
            "source_world_state_hash",
            _hash(
                self.source_world_state_hash,
                label="stochastic observation source world state hash",
            ),
        )
        object.__setattr__(
            self,
            "source_world_step_hash",
            _hash(
                self.source_world_step_hash,
                label="stochastic observation source world step hash",
            ),
        )
        object.__setattr__(
            self,
            "projection_spec_hash",
            _hash(
                self.projection_spec_hash,
                label="stochastic observation projection spec hash",
            ),
        )
        if not isinstance(self.distribution, ObservationOutcomeDistribution):
            raise TypeError(
                "stochastic observation distribution must be ObservationOutcomeDistribution"
            )
        if not isinstance(self.sample_record, RandomSampleRecord):
            raise TypeError(
                "stochastic observation sample record must be RandomSampleRecord"
            )
        record = self.sample_record
        if record.namespace != "observation.projection":
            raise ValueError("stochastic observation namespace mismatch")
        if record.step_index != self.step_index:
            raise ValueError("stochastic observation step mismatch")
        if record.source_hash != self.source_world_step_hash:
            raise ValueError("stochastic observation source world step mismatch")
        if record.component_hash != self.projection_spec_hash:
            raise ValueError("stochastic observation projection spec mismatch")
        if record.component_key != (self.observer_id, self.channel):
            raise ValueError("stochastic observation component key mismatch")
        if record.distribution_hash != self.distribution.content_hash:
            raise ValueError("stochastic observation distribution hash mismatch")
        matching = tuple(
            item
            for item in self.distribution.outcomes
            if item.outcome_id == record.selected_outcome_id
        )
        if len(matching) != 1:
            raise ValueError("stochastic observation selected outcome is missing")
        if matching[0].content_hash != record.selected_outcome_hash:
            raise ValueError("stochastic observation selected outcome hash mismatch")

    @property
    def selected_outcome(self) -> ObservationOutcome:
        return next(
            item
            for item in self.distribution.outcomes
            if item.outcome_id == self.sample_record.selected_outcome_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "observer_id": self.observer_id,
            "channel": self.channel,
            "step_index": self.step_index,
            "source_world_state_hash": self.source_world_state_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "projection_spec_hash": self.projection_spec_hash,
            "distribution": self.distribution.to_dict(),
            "sample_record": self.sample_record.to_dict(),
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
    stochastic_sample_hash: str | None = None

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
        if self.stochastic_sample_hash is not None:
            object.__setattr__(
                self,
                "stochastic_sample_hash",
                _hash(
                    self.stochastic_sample_hash,
                    label="projected observation stochastic sample hash",
                ),
            )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "observer_id": self.observer_id,
            "channel": self.channel,
            "fact": self.fact.to_dict(),
            "step_index": self.step_index,
            "source_world_state_hash": self.source_world_state_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "source_transition_hashes": list(self.source_transition_hashes),
            "projection_spec_hash": self.projection_spec_hash,
        }
        if self.stochastic_sample_hash is not None:
            payload["stochastic_sample_hash"] = self.stochastic_sample_hash
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _projection_key(spec: ProjectionSpec) -> tuple[str, str]:
    return (spec.observer_type, spec.channel)


@dataclass(frozen=True)
class ObservationProjectionModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    projections: tuple[ObserverProjectionSpec, ...]
    stochastic_projections: tuple[StochasticObserverProjectionSpec, ...] = ()

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
        if any(
            not isinstance(item, ObserverProjectionSpec)
            for item in projections
        ):
            raise TypeError(
                "observation projection model projections must contain ObserverProjectionSpec values"
            )
        if not isinstance(self.stochastic_projections, tuple):
            raise TypeError(
                "observation projection model stochastic projections must be a tuple"
            )
        stochastic = tuple(self.stochastic_projections)
        if any(
            not isinstance(item, StochasticObserverProjectionSpec)
            for item in stochastic
        ):
            raise TypeError(
                "observation projection model stochastic projections must contain StochasticObserverProjectionSpec values"
            )
        keys = tuple(_projection_key(item) for item in projections + stochastic)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "observation projection model observer/channel pairs must be unique across deterministic and stochastic lanes"
            )
        object.__setattr__(
            self,
            "projections",
            tuple(sorted(projections, key=_projection_key)),
        )
        object.__setattr__(
            self,
            "stochastic_projections",
            tuple(sorted(stochastic, key=_projection_key)),
        )

    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "version": self.version,
            "domain_id": self.domain_id,
            "domain_version": self.domain_version,
            "domain_spec_hash": self.domain_spec_hash,
            "projections": [item.to_dict() for item in self.projections],
        }
        if self.stochastic_projections:
            payload["stochastic_projections"] = [
                item.to_dict() for item in self.stochastic_projections
            ]
        return payload

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


def _sample_key(
    sample: StochasticObservationSample,
) -> tuple[str, str]:
    return (sample.observer_id, sample.channel)


@dataclass(frozen=True)
class ObservationProjectionResult:
    model_id: str
    model_hash: str
    source_world_step_hash: str
    source_world_state_hash: str
    step_index: int
    observations: tuple[ProjectedObservation, ...]
    stochastic_samples: tuple[StochasticObservationSample, ...] = ()

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
        if any(
            not isinstance(item, ProjectedObservation)
            for item in observations
        ):
            raise TypeError(
                "observation projection result observations must contain ProjectedObservation values"
            )
        observations = tuple(sorted(observations, key=_observation_key))
        keys = tuple(_observation_key(item) for item in observations)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "observation projection result observation keys must be unique"
            )
        if not isinstance(self.stochastic_samples, tuple):
            raise TypeError(
                "observation projection result stochastic samples must be a tuple"
            )
        samples = tuple(self.stochastic_samples)
        if any(
            not isinstance(item, StochasticObservationSample)
            for item in samples
        ):
            raise TypeError(
                "observation projection result stochastic samples must contain StochasticObservationSample values"
            )
        samples = tuple(sorted(samples, key=_sample_key))
        sample_keys = tuple(_sample_key(item) for item in samples)
        if len(set(sample_keys)) != len(sample_keys):
            raise ValueError(
                "observation projection result stochastic sample keys must be unique"
            )
        sample_hashes = {item.content_hash for item in samples}
        for item in samples:
            if item.source_world_step_hash != self.source_world_step_hash:
                raise ValueError(
                    "stochastic observation sample must bind result source world step"
                )
            if item.source_world_state_hash != self.source_world_state_hash:
                raise ValueError(
                    "stochastic observation sample must bind result source world state"
                )
            if item.step_index != self.step_index:
                raise ValueError(
                    "stochastic observation sample must bind result step index"
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
            if (
                item.stochastic_sample_hash is not None
                and item.stochastic_sample_hash not in sample_hashes
            ):
                raise ValueError(
                    "projected observation stochastic sample is not included in result"
                )
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "stochastic_samples", samples)

    def to_dict(self) -> dict[str, object]:
        payload = {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "source_world_step_hash": self.source_world_step_hash,
            "source_world_state_hash": self.source_world_state_hash,
            "step_index": self.step_index,
            "observations": [item.to_dict() for item in self.observations],
        }
        if self.stochastic_samples:
            payload["stochastic_samples"] = [
                item.to_dict() for item in self.stochastic_samples
            ]
        return payload

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _projection_hash(spec: ProjectionSpec) -> str:
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


def _resolved_batch_hash(
    records: tuple[ActionTransitionRecord, ...],
    resolutions: tuple[ConflictResolutionRecord, ...],
) -> str:
    return stable_content_hash(
        {
            "transitions": [
                record.to_dict()
                for record in sorted(records, key=_record_key)
            ],
            "conflict_resolutions": [
                resolution.to_dict()
                for resolution in sorted(resolutions, key=_resolution_key)
            ],
        }
    )


def _operation_cell(
    operation: StateDeltaOp,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> tuple[StateCellRef, object]:
    subject = entities.get(operation.subject_id)
    if subject is None:
        raise ValueError("state delta subject is not canonical")
    variable = domain._state_variable(operation.state_variable)
    if variable.subject_type != subject.type_name:
        raise ValueError("state delta subject type mismatch")
    return (
        StateCellRef(
            EntityRef(subject.id, subject.type_name),
            variable.name,
        ),
        variable,
    )


def _validated_delta_operations(
    delta: object,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    *,
    label: str,
) -> tuple[tuple[StateCellRef, StateDeltaOp], ...]:
    if not isinstance(delta, StateDelta):
        raise TypeError(f"{label} delta must be StateDelta")
    if not isinstance(delta.operations, tuple):
        raise TypeError(f"{label} delta operations must be a tuple")

    seen: set[StateCellRef] = set()
    validated: list[tuple[StateCellRef, StateDeltaOp]] = []
    for operation in delta.operations:
        if not isinstance(operation, StateDeltaOp):
            raise TypeError(f"{label} delta operation must be StateDeltaOp")
        cell, variable = _operation_cell(operation, domain, entities)
        if cell in seen:
            raise ValueError(f"{label} cannot write one cell twice")
        seen.add(cell)
        if operation.kind == "clear":
            if operation.value is not None:
                raise ValueError(f"clear {label} delta cannot contain a value")
        elif operation.kind == "set":
            if operation.value is None:
                raise ValueError(f"set {label} delta requires a value")
            domain._value_type(variable.value_type).validate(
                operation.value,
                entities,
            )
        else:
            raise ValueError(f"{label} delta operation kind is unsupported")
        validated.append((cell, operation))
    return tuple(validated)


def _derived_conflict_components(
    records: tuple[ActionTransitionRecord, ...],
    operations_by_hash: Mapping[
        str, tuple[tuple[StateCellRef, StateDeltaOp], ...]
    ],
) -> tuple[tuple[ActionTransitionRecord, ...], ...]:
    canonical = tuple(sorted(records, key=_record_key))
    writes = tuple(
        frozenset(
            cell
            for cell, _ in operations_by_hash[record.content_hash]
        )
        for record in canonical
    )
    adjacency = [set() for _ in canonical]
    for left, left_writes in enumerate(writes):
        for right in range(left + 1, len(canonical)):
            if left_writes & writes[right]:
                adjacency[left].add(right)
                adjacency[right].add(left)

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
            stack.extend(sorted(adjacency[current], reverse=True))
        if len(indices) >= 2:
            components.append(
                tuple(canonical[index] for index in sorted(indices))
            )
    return tuple(
        sorted(
            components,
            key=lambda component: tuple(
                _record_key(record) for record in component
            ),
        )
    )


def _apply_operation(
    values: dict[StateCellRef, TypedValue],
    cell: StateCellRef,
    operation: StateDeltaOp,
) -> None:
    if operation.kind == "clear":
        values.pop(cell, None)
    else:
        assert operation.value is not None
        values[cell] = operation.value


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
    if next_state.root_seed != prior.root_seed:
        raise ValueError(
            "observation projection world states must preserve root seed"
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
    prior_hash = prior.content_hash
    operations_by_hash: dict[
        str, tuple[tuple[StateCellRef, StateDeltaOp], ...]
    ] = {}
    record_by_hash: dict[str, ActionTransitionRecord] = {}
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
        record_hash = record.content_hash
        if record_hash in record_by_hash:
            raise ValueError("world transition records must have unique identities")
        record_by_hash[record_hash] = record
        operations_by_hash[record_hash] = _validated_delta_operations(
            record.delta,
            domain,
            entities,
            label="transition",
        )

    resolutions = tuple(world_step.conflict_resolutions)
    if not resolutions:
        written: set[StateCellRef] = set()
        replayed = dict(prior.values)
        for record in records:
            for cell, operation in operations_by_hash[record.content_hash]:
                if cell in written:
                    raise ValueError(
                        "world transition records cannot overlap writes"
                    )
                written.add(cell)
                _apply_operation(replayed, cell, operation)
        if replayed != dict(next_state.values):
            raise ValueError(
                "next world state is not the extensional result of transition deltas"
            )
        if next_state.transition_batch_hash != _batch_hash(records):
            raise ValueError(
                "next world state does not bind exact transition batch identity"
            )
        return entities

    if any(
        not isinstance(resolution, ConflictResolutionRecord)
        for resolution in resolutions
    ):
        raise TypeError(
            "observation projection conflict resolutions must be ConflictResolutionRecord values"
        )

    components = _derived_conflict_components(records, operations_by_hash)
    component_sets = {
        frozenset(record.content_hash for record in component)
        for component in components
    }
    resolved_component_sets: set[frozenset[str]] = set()
    resolved_record_hashes: set[str] = set()
    resolution_operations: dict[
        str, tuple[tuple[StateCellRef, StateDeltaOp], ...]
    ] = {}

    for resolution in resolutions:
        _text(resolution.resolver_id, label="conflict resolution resolver id")
        _hash(resolution.resolver_hash, label="conflict resolution resolver hash")
        if resolution.prior_state_hash != prior_hash:
            raise ValueError(
                "conflict resolution does not bind exact prior state"
            )
        context = resolution.context
        if not isinstance(context, ConflictResolutionContext):
            raise TypeError(
                "conflict resolution context must be ConflictResolutionContext"
            )
        if context.prior_state_hash != prior_hash:
            raise ValueError(
                "conflict resolution context does not bind exact prior state"
            )
        if not isinstance(context.participants, tuple):
            raise TypeError("conflict participants must be a tuple")
        if len(context.participants) < 2:
            raise ValueError(
                "conflict resolution requires at least two participants"
            )
        if any(
            not isinstance(participant, ConflictParticipant)
            for participant in context.participants
        ):
            raise TypeError(
                "conflict participants must be ConflictParticipant values"
            )

        participant_keys = tuple(
            (
                participant.actor_id,
                participant.decision_id,
                participant.action.id,
            )
            for participant in context.participants
        )
        if participant_keys != tuple(sorted(participant_keys)):
            raise ValueError("conflict participants must be canonical")
        participant_hashes = tuple(
            participant.transition_record_hash
            for participant in context.participants
        )
        if len(set(participant_hashes)) != len(participant_hashes):
            raise ValueError("conflict participant transitions must be unique")
        component_set = frozenset(participant_hashes)
        if component_set not in component_sets:
            raise ValueError(
                "conflict resolution does not bind an exact overlap component"
            )
        if component_set in resolved_component_sets:
            raise ValueError("conflict component cannot be resolved twice")
        resolved_component_sets.add(component_set)
        resolved_record_hashes.update(component_set)

        write_counts: dict[StateCellRef, int] = {}
        allowed_union: set[StateCellRef] = set()
        for participant in context.participants:
            record = record_by_hash.get(participant.transition_record_hash)
            if record is None:
                raise ValueError(
                    "conflict participant transition is not in world step"
                )
            if participant.actor_id != record.actor_id:
                raise ValueError("conflict participant actor mismatch")
            if participant.decision_id != record.intent.decision_id:
                raise ValueError("conflict participant decision mismatch")
            if participant.action != record.action:
                raise ValueError("conflict participant action payload mismatch")
            if participant.transition_spec_hash != record.transition_spec_hash:
                raise ValueError("conflict participant transition spec mismatch")
            if participant.original_delta != record.delta:
                raise ValueError("conflict participant original delta mismatch")
            if not isinstance(participant.allowed_write_cells, tuple):
                raise TypeError("conflict participant capabilities must be a tuple")
            if len(set(participant.allowed_write_cells)) != len(
                participant.allowed_write_cells
            ):
                raise ValueError("conflict participant capabilities must be unique")
            if tuple(
                sorted(participant.allowed_write_cells, key=_cell_key)
            ) != participant.allowed_write_cells:
                raise ValueError("conflict participant capabilities must be canonical")
            for allowed in participant.allowed_write_cells:
                if not isinstance(allowed, StateCellRef):
                    raise TypeError(
                        "conflict participant capability must be StateCellRef"
                    )
                subject = entities.get(allowed.subject.entity_id)
                if (
                    subject is None
                    or subject.type_name != allowed.subject.entity_type
                ):
                    raise ValueError(
                        "conflict participant capability subject is not canonical"
                    )
                variable = domain._state_variable(allowed.state_variable)
                if variable.subject_type != subject.type_name:
                    raise ValueError(
                        "conflict participant capability subject type mismatch"
                    )
                allowed_union.add(allowed)
            for cell, _ in operations_by_hash[record.content_hash]:
                if cell not in participant.allowed_write_cells:
                    raise ValueError(
                        "conflict participant original write exceeds capability"
                    )
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
            raise ValueError(
                "conflict context cells do not match exact overlapping writes"
            )

        resolved_ops = _validated_delta_operations(
            resolution.resolved_delta,
            domain,
            entities,
            label="conflict resolution",
        )
        if any(cell not in allowed_union for cell, _ in resolved_ops):
            raise ValueError(
                "conflict resolution wrote outside participant capability union"
            )
        resolution_operations[resolution.content_hash] = resolved_ops

    if resolved_component_sets != component_sets:
        raise ValueError(
            "conflict resolutions must cover every exact overlap component"
        )

    replayed = dict(prior.values)
    effective_written: set[StateCellRef] = set()
    for record in sorted(records, key=_record_key):
        if record.content_hash in resolved_record_hashes:
            continue
        for cell, operation in operations_by_hash[record.content_hash]:
            if cell in effective_written:
                raise ValueError(
                    "effective world mutations cannot overlap writes"
                )
            effective_written.add(cell)
            _apply_operation(replayed, cell, operation)
    for resolution in sorted(resolutions, key=_resolution_key):
        for cell, operation in resolution_operations[resolution.content_hash]:
            if cell in effective_written:
                raise ValueError(
                    "effective world mutations cannot overlap writes"
                )
            effective_written.add(cell)
            _apply_operation(replayed, cell, operation)

    if replayed != dict(next_state.values):
        raise ValueError(
            "next world state is not the extensional result of effective mutations"
        )
    if next_state.transition_batch_hash != _resolved_batch_hash(
        records,
        resolutions,
    ):
        raise ValueError(
            "next world state does not bind exact resolved transition batch identity"
        )
    return entities


def _all_projections(
    model: ObservationProjectionModelSpec,
) -> tuple[ProjectionSpec, ...]:
    return tuple(model.projections) + tuple(model.stochastic_projections)


def _validate_projection_declarations(
    domain: DomainSpec,
    model: ObservationProjectionModelSpec,
) -> dict[tuple[str, str], str]:
    hashes: dict[tuple[str, str], str] = {}
    for spec in _all_projections(model):
        domain._entity_type(spec.observer_type)
        for capability in spec.read_capabilities + spec.emit_capabilities:
            variable = domain._state_variable(capability.state_variable)
            if (
                capability.subject_scope == "observer"
                and variable.subject_type != spec.observer_type
            ):
                raise ValueError(
                    "observer-scoped capability state subject type must match observer type"
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


def _hook_distribution(
    spec: StochasticObserverProjectionSpec,
    prior_visible: Mapping[StateCellRef, TypedValue],
    next_visible: Mapping[StateCellRef, TypedValue],
    observer: Entity,
    step_index: int,
) -> ObservationOutcomeDistribution:
    try:
        raw = spec.distribution_hook(
            prior_visible,
            next_visible,
            observer,
            step_index,
            spec.parameters,
        )
    except Exception as error:
        raise ObservationProjectionError(
            "stochastic observation projection hook execution failed"
        ) from error
    if not isinstance(raw, ObservationOutcomeDistribution):
        raise ObservationProjectionError(
            "stochastic observation projection hook must return ObservationOutcomeDistribution"
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


def _effective_write_for_cell(
    world_step: WorldStepResult,
    cell: StateCellRef,
) -> tuple[str | None, tuple[str, ...]]:
    resolved_hashes: set[str] = set()
    for resolution in world_step.conflict_resolutions:
        participant_hashes = tuple(
            sorted(
                participant.transition_record_hash
                for participant in resolution.context.participants
            )
        )
        resolved_hashes.update(participant_hashes)
        matching = tuple(
            operation
            for operation in resolution.resolved_delta.operations
            if (
                operation.subject_id == cell.subject.entity_id
                and operation.state_variable == cell.state_variable
            )
        )
        if matching:
            if len(matching) != 1:
                raise ValueError(
                    "conflict resolution cannot effectively write one cell twice"
                )
            return matching[0].kind, participant_hashes

    matching_records = tuple(
        record
        for record in world_step.transitions
        if record.content_hash not in resolved_hashes
        and any(
            operation.subject_id == cell.subject.entity_id
            and operation.state_variable == cell.state_variable
            for operation in record.delta.operations
        )
    )
    if not matching_records:
        return None, ()
    if len(matching_records) != 1:
        raise ValueError("effective world writes cannot overlap")
    record = matching_records[0]
    operation = next(
        operation
        for operation in record.delta.operations
        if (
            operation.subject_id == cell.subject.entity_id
            and operation.state_variable == cell.state_variable
        )
    )
    return operation.kind, (record.content_hash,)


def _accept_facts(
    facts: tuple[ObservationFact, ...],
    *,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    world_step: WorldStepResult,
    observer: Entity,
    spec: ProjectionSpec,
    spec_hash: str,
    stochastic_sample_hash: str | None = None,
) -> tuple[ProjectedObservation, ...]:
    accepted: list[ProjectedObservation] = []
    for fact in facts:
        cell = fact.cell
        _validate_fact_cell_shape(fact, domain, entities)
        variable = domain._state_variable(cell.state_variable)
        effective_kind, source_transition_hashes = _effective_write_for_cell(
            world_step,
            cell,
        )

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
            if effective_kind != "clear":
                raise ValueError(
                    "clear observation requires one explicit effective current-step clear"
                )

        accepted.append(
            ProjectedObservation(
                observer_id=observer.id,
                channel=spec.channel,
                fact=fact,
                step_index=world_step.next_state.step_index,
                source_world_state_hash=world_step.next_state.content_hash,
                source_world_step_hash=world_step.content_hash,
                source_transition_hashes=source_transition_hashes,
                projection_spec_hash=spec_hash,
                stochastic_sample_hash=stochastic_sample_hash,
            )
        )
    return tuple(accepted)


def _validate_emit_and_truth(
    facts: tuple[ObservationFact, ...],
    *,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    world_step: WorldStepResult,
    observer: Entity,
    spec: ProjectionSpec,
    spec_hash: str,
) -> tuple[ProjectedObservation, ...]:
    for fact in facts:
        if not any(
            _emit_covers(capability, fact.cell, observer)
            for capability in spec.emit_capabilities
        ):
            raise ObservationProjectionError(
                "observation fact is outside emit capability"
            )
        _validate_fact_cell_shape(fact, domain, entities)
    return _accept_facts(
        facts,
        domain=domain,
        entities=entities,
        world_step=world_step,
        observer=observer,
        spec=spec,
        spec_hash=spec_hash,
    )


def _stochastic_projection(
    *,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
    world_step: WorldStepResult,
    observer: Entity,
    spec: StochasticObserverProjectionSpec,
    spec_hash: str,
) -> tuple[
    tuple[ProjectedObservation, ...],
    StochasticObservationSample,
]:
    root_seed = world_step.next_state.root_seed
    if root_seed is None:
        raise ObservationProjectionError(
            "stochastic observation projection requires a root seed"
        )
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
    distribution = _hook_distribution(
        spec,
        prior_visible,
        next_visible,
        observer,
        world_step.next_state.step_index,
    )

    for outcome in distribution.outcomes:
        _validate_emit_and_truth(
            outcome.facts,
            domain=domain,
            entities=entities,
            world_step=world_step,
            observer=observer,
            spec=spec,
            spec_hash=spec_hash,
        )

    sample_record = sample_categorical(
        root_seed=root_seed,
        namespace="observation.projection",
        step_index=world_step.next_state.step_index,
        source_hash=world_step.content_hash,
        component_hash=spec_hash,
        component_key=(observer.id, spec.channel),
        distribution_hash=distribution.content_hash,
        outcomes=tuple(
            (
                outcome.outcome_id,
                outcome.probability,
                outcome.content_hash,
            )
            for outcome in distribution.outcomes
        ),
    )
    sample = StochasticObservationSample(
        observer_id=observer.id,
        channel=spec.channel,
        step_index=world_step.next_state.step_index,
        source_world_state_hash=world_step.next_state.content_hash,
        source_world_step_hash=world_step.content_hash,
        projection_spec_hash=spec_hash,
        distribution=distribution,
        sample_record=sample_record,
    )
    selected = _accept_facts(
        sample.selected_outcome.facts,
        domain=domain,
        entities=entities,
        world_step=world_step,
        observer=observer,
        spec=spec,
        spec_hash=spec_hash,
        stochastic_sample_hash=sample.content_hash,
    )
    return selected, sample


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
        samples: list[StochasticObservationSample] = []
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
                observations.extend(
                    _validate_emit_and_truth(
                        facts,
                        domain=domain,
                        entities=entities,
                        world_step=world_step,
                        observer=observer,
                        spec=spec,
                        spec_hash=spec_hash,
                    )
                )

        for spec in sorted(model.stochastic_projections, key=_projection_key):
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
                projected, sample = _stochastic_projection(
                    domain=domain,
                    entities=entities,
                    world_step=world_step,
                    observer=observer,
                    spec=spec,
                    spec_hash=spec_hash,
                )
                observations.extend(projected)
                samples.append(sample)

        return ObservationProjectionResult(
            model_id=model.model_id,
            model_hash=model_hash,
            source_world_step_hash=world_step.content_hash,
            source_world_state_hash=world_step.next_state.content_hash,
            step_index=world_step.next_state.step_index,
            observations=tuple(observations),
            stochastic_samples=tuple(samples),
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
