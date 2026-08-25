from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import re
from types import MappingProxyType

from hypothesis_competition import posterior_distribution
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import EpistemicEvidence, epistemic_state


_PROBABILITY_TOLERANCE = 1e-12
_CONTENT_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


class UncertainBeliefResolutionError(ValueError):
    """An explicit uncertain-belief model could not produce a valid state."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _probability(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


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
        return MappingProxyType(frozen)
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
        raise TypeError("uncertain belief model parameters must be a mapping")
    frozen = _freeze_parameter(value, label="uncertain belief model parameters")
    assert isinstance(frozen, Mapping)
    return frozen


def _parameter_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            key: _parameter_payload(value[key])
            for key in sorted(value)
        }
    if isinstance(value, tuple):
        return [_parameter_payload(item) for item in value]
    return value


def _value_key(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


@dataclass(frozen=True)
class BeliefMass:
    value: TypedValue
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, TypedValue):
            raise TypeError("belief mass value must be a TypedValue")
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, label="belief mass probability"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value.to_dict(),
            "probability": self.probability,
        }


@dataclass(frozen=True)
class BeliefLikelihood:
    value: TypedValue
    likelihood: float

    def __post_init__(self) -> None:
        if not isinstance(self.value, TypedValue):
            raise TypeError("belief likelihood value must be a TypedValue")
        object.__setattr__(
            self,
            "likelihood",
            _probability(self.likelihood, label="belief likelihood"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "value": self.value.to_dict(),
            "likelihood": self.likelihood,
        }


@dataclass(frozen=True)
class BeliefDistribution:
    cell: StateCellRef
    masses: tuple[BeliefMass, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("belief distribution cell must be a StateCellRef")
        masses = tuple(self.masses)
        if len(masses) < 2:
            raise ValueError("belief distribution requires at least two hypotheses")
        if any(not isinstance(item, BeliefMass) for item in masses):
            raise TypeError("belief distribution masses must be BeliefMass values")
        keys = tuple(_value_key(item.value) for item in masses)
        if len(set(keys)) != len(keys):
            raise ValueError("belief distribution hypotheses must be unique")
        total = math.fsum(item.probability for item in masses)
        if not math.isclose(
            total,
            1.0,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("belief distribution probabilities must sum to 1")
        object.__setattr__(
            self,
            "masses",
            tuple(sorted(masses, key=lambda item: _value_key(item.value))),
        )

    def probability_of(self, value: TypedValue) -> float:
        if not isinstance(value, TypedValue):
            raise TypeError("belief lookup requires a TypedValue")
        for mass in self.masses:
            if mass.value == value:
                return mass.probability
        raise KeyError("belief distribution does not contain the requested value")

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "masses": [item.to_dict() for item in self.masses],
        }


@dataclass(frozen=True)
class BeliefUpdateStep:
    evidence: EpistemicEvidence
    prior: BeliefDistribution
    likelihoods: tuple[BeliefLikelihood, ...]
    posterior: BeliefDistribution

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, EpistemicEvidence):
            raise TypeError("belief update evidence must be EpistemicEvidence")
        if not isinstance(self.prior, BeliefDistribution) or not isinstance(
            self.posterior, BeliefDistribution
        ):
            raise TypeError("belief update prior/posterior must be BeliefDistribution")
        if not (
            self.evidence.cell == self.prior.cell == self.posterior.cell
        ):
            raise ValueError("belief update records must describe one state cell")
        likelihoods = tuple(self.likelihoods)
        if any(not isinstance(item, BeliefLikelihood) for item in likelihoods):
            raise TypeError("belief update likelihoods must be BeliefLikelihood values")
        likelihood_keys = tuple(_value_key(item.value) for item in likelihoods)
        if len(set(likelihood_keys)) != len(likelihood_keys):
            raise ValueError("belief update likelihood hypotheses must be unique")
        prior_keys = {_value_key(item.value) for item in self.prior.masses}
        posterior_keys = {_value_key(item.value) for item in self.posterior.masses}
        if prior_keys != posterior_keys or prior_keys != set(likelihood_keys):
            raise ValueError("belief update hypothesis sets must match exactly")
        object.__setattr__(
            self,
            "likelihoods",
            tuple(sorted(likelihoods, key=lambda item: _value_key(item.value))),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence": self.evidence.to_dict(),
            "prior": self.prior.to_dict(),
            "likelihoods": [item.to_dict() for item in self.likelihoods],
            "posterior": self.posterior.to_dict(),
        }


@dataclass(frozen=True)
class UncertainBeliefCellView:
    cell: StateCellRef
    prior: BeliefDistribution
    posterior: BeliefDistribution
    updates: tuple[BeliefUpdateStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("uncertain belief cell view requires StateCellRef")
        if not isinstance(self.prior, BeliefDistribution) or not isinstance(
            self.posterior, BeliefDistribution
        ):
            raise TypeError("uncertain belief cell prior/posterior must be distributions")
        if self.prior.cell != self.cell or self.posterior.cell != self.cell:
            raise ValueError("uncertain belief cell distributions must match the cell")
        updates = tuple(self.updates)
        if any(not isinstance(item, BeliefUpdateStep) for item in updates):
            raise TypeError("uncertain belief updates must be BeliefUpdateStep values")
        current = self.prior
        for step in updates:
            if (
                step.prior.cell != self.cell
                or step.posterior.cell != self.cell
                or step.prior != current
            ):
                raise ValueError("uncertain belief update chain is discontinuous")
            current = step.posterior
        if current != self.posterior:
            raise ValueError("uncertain belief posterior must equal the update-chain tail")
        object.__setattr__(self, "updates", updates)

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "prior": self.prior.to_dict(),
            "posterior": self.posterior.to_dict(),
            "updates": [item.to_dict() for item in self.updates],
        }


@dataclass(frozen=True)
class UncertainBeliefState:
    agent_id: str
    model_id: str
    model_hash: str
    logical_time: int | None
    evidence_history: tuple[EpistemicEvidence, ...]
    cells: Mapping[StateCellRef, UncertainBeliefCellView]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _text(self.agent_id, label="uncertain belief agent id"))
        object.__setattr__(self, "model_id", _text(self.model_id, label="uncertain belief model id"))
        if (
            not isinstance(self.model_hash, str)
            or _CONTENT_HASH_PATTERN.fullmatch(self.model_hash) is None
        ):
            raise ValueError("uncertain belief model hash must be a sha256 content hash")
        if self.logical_time is not None and (
            not isinstance(self.logical_time, int)
            or isinstance(self.logical_time, bool)
            or self.logical_time < 0
        ):
            raise ValueError("uncertain belief logical time must be non-negative or None")
        history = tuple(self.evidence_history)
        if any(not isinstance(item, EpistemicEvidence) for item in history):
            raise TypeError("uncertain belief history must contain EpistemicEvidence")
        cells = dict(self.cells)
        if not cells:
            raise ValueError("uncertain belief state must contain at least one tracked cell")
        if any(
            not isinstance(cell, StateCellRef)
            or not isinstance(view, UncertainBeliefCellView)
            or view.cell != cell
            for cell, view in cells.items()
        ):
            raise TypeError(
                "uncertain belief cells must map StateCellRef to matching views"
            )
        object.__setattr__(self, "evidence_history", history)
        object.__setattr__(self, "cells", MappingProxyType(cells))

    def to_dict(self) -> dict[str, object]:
        ordered = tuple(sorted(self.cells, key=_cell_key))
        return {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "logical_time": self.logical_time,
            "evidence_history": [item.to_dict() for item in self.evidence_history],
            "cells": [self.cells[cell].to_dict() for cell in ordered],
        }


@dataclass(frozen=True)
class UncertainBeliefModelSpec:
    model_id: str
    version: str
    parameters: Mapping[str, object]
    prior_hook: object = field(compare=False, repr=False)
    likelihood_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="uncertain belief model id"))
        object.__setattr__(self, "version", _text(self.version, label="uncertain belief model version"))
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        if not callable(self.prior_hook):
            raise TypeError("uncertain belief prior hook must be callable")
        if not callable(self.likelihood_hook):
            raise TypeError("uncertain belief likelihood hook must be callable")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "parameters": _parameter_payload(self.parameters),
            "prior_hook_identity": measure_implementation(
                self.prior_hook
            ).manifest_identity(),
            "likelihood_hook_identity": measure_implementation(
                self.likelihood_hook
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _hypotheses_for_cell(
    story: GenericNarrative,
    domain: DomainSpec,
    cell: StateCellRef,
) -> tuple[TypedValue, ...]:
    entities = {entity.id: entity for entity in story.entities}
    subject = entities.get(cell.subject.entity_id)
    if subject is None or subject.type_name != cell.subject.entity_type:
        raise UncertainBeliefResolutionError(
            "tracked cell subject is not declared exactly"
        )
    try:
        variable = domain._state_variable(cell.state_variable)
    except ValueError as error:
        raise UncertainBeliefResolutionError(
            "tracked state variable is undeclared"
        ) from error
    if subject.type_name != variable.subject_type:
        raise UncertainBeliefResolutionError(
            "tracked cell subject type does not match state variable"
        )
    try:
        value_type = domain._value_type(variable.value_type)
    except ValueError as error:
        raise UncertainBeliefResolutionError(
            "tracked state variable value type is undeclared"
        ) from error

    if value_type.kind == "enum":
        hypotheses = tuple(
            TypedValue(value_type.name, raw)
            for raw in value_type.allowed_values
        )
    elif value_type.kind == "bool":
        hypotheses = (
            TypedValue(value_type.name, False),
            TypedValue(value_type.name, True),
        )
    elif value_type.kind == "entity_ref":
        hypotheses = tuple(
            TypedValue(
                value_type.name,
                EntityRef(entity.id, entity.type_name),
            )
            for entity in sorted(
                story.entities,
                key=lambda item: (item.type_name, item.id),
            )
            if entity.type_name == value_type.entity_type
        )
    else:
        raise UncertainBeliefResolutionError(
            "tracked state variable is not finitely enumerable in V1"
        )

    if len(hypotheses) < 2:
        raise UncertainBeliefResolutionError(
            "uncertain belief requires at least two finite hypotheses"
        )
    return tuple(sorted(hypotheses, key=_value_key))


def _validated_vector(
    raw: object,
    hypotheses: tuple[TypedValue, ...],
    *,
    label: str,
    normalized: bool = False,
) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise UncertainBeliefResolutionError(f"{label} must be a mapping")
    expected = {_value_key(value) for value in hypotheses}
    if set(raw) != expected:
        raise UncertainBeliefResolutionError(
            f"{label} keys must match the finite hypotheses exactly"
        )
    vector: dict[str, float] = {}
    for key in expected:
        try:
            vector[key] = _probability(raw[key], label=f"{label} value")
        except (TypeError, ValueError) as error:
            raise UncertainBeliefResolutionError(
                f"{label} contains an invalid probability"
            ) from error
    if normalized and not math.isclose(
        math.fsum(vector.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise UncertainBeliefResolutionError(
            f"{label} must already sum to 1"
        )
    return vector


def _distribution_from_vector(
    cell: StateCellRef,
    hypotheses: tuple[TypedValue, ...],
    vector: Mapping[str, float],
) -> BeliefDistribution:
    return BeliefDistribution(
        cell,
        tuple(
            BeliefMass(value, vector[_value_key(value)])
            for value in hypotheses
        ),
    )


def _call_prior_hook(
    model: UncertainBeliefModelSpec,
    agent_id: str,
    cell: StateCellRef,
    hypotheses: tuple[TypedValue, ...],
) -> object:
    try:
        return model.prior_hook(
            agent_id,
            cell,
            hypotheses,
            model.parameters,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain prior hook could not produce a valid vector"
        ) from error


def _call_likelihood_hook(
    model: UncertainBeliefModelSpec,
    agent_id: str,
    evidence: EpistemicEvidence,
    hypotheses: tuple[TypedValue, ...],
) -> object:
    try:
        return model.likelihood_hook(
            agent_id,
            evidence,
            hypotheses,
            model.parameters,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise UncertainBeliefResolutionError(
            "uncertain likelihood hook could not produce a valid vector"
        ) from error


def uncertain_epistemic_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    model: UncertainBeliefModelSpec,
    tracked_cells: tuple[StateCellRef, ...],
    *,
    at_time: int | None = None,
) -> UncertainBeliefState:
    validate_narrative(story, domain)
    if not isinstance(model, UncertainBeliefModelSpec):
        raise TypeError("uncertain replay requires UncertainBeliefModelSpec")
    cells = tuple(tracked_cells)
    if not cells or any(not isinstance(cell, StateCellRef) for cell in cells):
        raise TypeError(
            "tracked cells must be a non-empty tuple of StateCellRef values"
        )
    if len(set(cells)) != len(cells):
        raise ValueError("tracked cells must be unique")

    admitted = epistemic_state(
        story,
        domain,
        agent_id,
        at_time=at_time,
    )
    views: dict[StateCellRef, UncertainBeliefCellView] = {}

    for cell in cells:
        hypotheses = _hypotheses_for_cell(story, domain, cell)
        raw_prior = _call_prior_hook(
            model,
            agent_id,
            cell,
            hypotheses,
        )
        prior_vector = _validated_vector(
            raw_prior,
            hypotheses,
            label="uncertain prior",
            normalized=True,
        )
        prior = _distribution_from_vector(
            cell,
            hypotheses,
            prior_vector,
        )
        current = prior
        updates: list[BeliefUpdateStep] = []

        for evidence in admitted.evidence_history:
            if evidence.cell != cell:
                continue
            current_hypotheses = tuple(
                mass.value for mass in current.masses
            )
            raw_likelihoods = _call_likelihood_hook(
                model,
                agent_id,
                evidence,
                current_hypotheses,
            )
            likelihood_vector = _validated_vector(
                raw_likelihoods,
                current_hypotheses,
                label="uncertain likelihood",
            )
            current_prior_vector = {
                _value_key(mass.value): mass.probability
                for mass in current.masses
            }
            try:
                posterior_vector = posterior_distribution(
                    current_prior_vector,
                    likelihood_vector,
                )
            except ValueError as error:
                raise UncertainBeliefResolutionError(
                    "uncertain evidence update has zero posterior mass"
                ) from error
            posterior = _distribution_from_vector(
                cell,
                current_hypotheses,
                posterior_vector,
            )
            likelihoods = tuple(
                BeliefLikelihood(
                    value,
                    likelihood_vector[_value_key(value)],
                )
                for value in current_hypotheses
            )
            updates.append(
                BeliefUpdateStep(
                    evidence,
                    current,
                    likelihoods,
                    posterior,
                )
            )
            current = posterior

        views[cell] = UncertainBeliefCellView(
            cell=cell,
            prior=prior,
            posterior=current,
            updates=tuple(updates),
        )

    return UncertainBeliefState(
        agent_id=agent_id,
        model_id=model.model_id,
        model_hash=model.content_hash,
        logical_time=at_time,
        evidence_history=admitted.evidence_history,
        cells=views,
    )
