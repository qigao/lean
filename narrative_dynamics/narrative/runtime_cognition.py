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
from narrative_dynamics.narrative.ir import GenericNarrative, StateCellRef, TypedValue
from narrative_dynamics.narrative.replay import EpistemicState, epistemic_state
from narrative_dynamics.narrative.runtime_perception import (
    RuntimeEvidenceLedger,
    RuntimeEpistemicEvidence,
)
from narrative_dynamics.narrative.uncertain import (
    BeliefDistribution,
    BeliefLikelihood,
    BeliefMass,
    UncertainBeliefModelSpec,
    UncertainBeliefState,
    uncertain_epistemic_state,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_STATUSES = frozenset({"resolved", "unknown", "conflicted"})
_BASES = frozenset({"authored_seed", "runtime_perception"})


class RuntimeEpistemicResolutionError(ValueError):
    """Runtime direct-perception evidence could not produce a valid epistemic state."""


class RuntimeBeliefResolutionError(ValueError):
    """Runtime percept evidence could not produce a valid uncertain belief state."""


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


def _cutoff(value: object, *, label: str) -> int | None:
    if value is None:
        return None
    return _step(value, label=label)


def _probability(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _value_key(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


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
        raise TypeError("runtime belief model parameters must be a mapping")
    frozen = _freeze_parameter(value, label="runtime belief model parameters")
    assert isinstance(frozen, Mapping)
    return frozen


def _parameter_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _parameter_payload(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_parameter_payload(item) for item in value]
    return value


def _runtime_history_key(
    item: RuntimeEpistemicEvidence,
) -> tuple[int, str, tuple[str, str, str]]:
    return (item.step_index, item.channel, _cell_key(item.cell))


@dataclass(frozen=True)
class RuntimeEpistemicCellView:
    cell: StateCellRef
    status: str
    resolved_value: TypedValue | None
    constraints: tuple[TypedValue, ...]
    basis: str
    supporting_runtime_evidence_hashes: tuple[str, ...]
    last_runtime_step: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("runtime epistemic cell view requires StateCellRef")
        if self.status not in _STATUSES:
            raise ValueError("runtime epistemic cell status is not supported")
        if self.resolved_value is not None and not isinstance(
            self.resolved_value, TypedValue
        ):
            raise TypeError("runtime epistemic resolved value must be TypedValue or None")
        if self.status == "resolved" and self.resolved_value is None:
            raise ValueError("resolved runtime epistemic cell requires a value")
        if self.status != "resolved" and self.resolved_value is not None:
            raise ValueError(
                "non-resolved runtime epistemic cell cannot contain a resolved value"
            )
        constraints = tuple(self.constraints)
        if any(not isinstance(item, TypedValue) for item in constraints):
            raise TypeError("runtime epistemic constraints must be TypedValue values")
        object.__setattr__(self, "constraints", constraints)
        if self.basis not in _BASES:
            raise ValueError("runtime epistemic basis is not supported")
        if not isinstance(self.supporting_runtime_evidence_hashes, tuple):
            raise TypeError("runtime epistemic support hashes must be a tuple")
        support = tuple(
            _hash(item, label="runtime epistemic support hash")
            for item in self.supporting_runtime_evidence_hashes
        )
        if len(set(support)) != len(support):
            raise ValueError("runtime epistemic support hashes must be unique")
        object.__setattr__(
            self,
            "supporting_runtime_evidence_hashes",
            tuple(sorted(support)),
        )
        if self.last_runtime_step is not None:
            object.__setattr__(
                self,
                "last_runtime_step",
                _step(self.last_runtime_step, label="runtime epistemic last step"),
            )
        if self.basis == "authored_seed":
            if support or self.last_runtime_step is not None:
                raise ValueError(
                    "authored-seed runtime epistemic view cannot contain runtime support"
                )
        else:
            if not support or self.last_runtime_step is None:
                raise ValueError(
                    "runtime-perception epistemic view requires runtime support and step"
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "status": self.status,
            "resolved_value": (
                None if self.resolved_value is None else self.resolved_value.to_dict()
            ),
            "constraints": [item.to_dict() for item in self.constraints],
            "basis": self.basis,
            "supporting_runtime_evidence_hashes": list(
                self.supporting_runtime_evidence_hashes
            ),
            "last_runtime_step": self.last_runtime_step,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeEpistemicState:
    agent_id: str
    source_at_time: int | None
    step_index: int
    ledger_hash: str
    seed_state: EpistemicState
    runtime_evidence_history: tuple[RuntimeEpistemicEvidence, ...]
    cells: Mapping[StateCellRef, RuntimeEpistemicCellView]
    resolved_values: Mapping[StateCellRef, TypedValue]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="runtime epistemic agent id"),
        )
        object.__setattr__(
            self,
            "source_at_time",
            _cutoff(self.source_at_time, label="runtime epistemic source cutoff"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime epistemic step index"),
        )
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="runtime epistemic ledger hash"),
        )
        if not isinstance(self.seed_state, EpistemicState):
            raise TypeError("runtime epistemic seed state must be EpistemicState")
        if not isinstance(self.runtime_evidence_history, tuple):
            raise TypeError("runtime epistemic evidence history must be a tuple")
        history = tuple(self.runtime_evidence_history)
        if any(not isinstance(item, RuntimeEpistemicEvidence) for item in history):
            raise TypeError(
                "runtime epistemic history must contain RuntimeEpistemicEvidence"
            )
        if any(item.observer_id != self.agent_id for item in history):
            raise ValueError("runtime epistemic history must contain only one agent")
        history = tuple(sorted(history, key=_runtime_history_key))
        object.__setattr__(self, "runtime_evidence_history", history)
        cells = dict(self.cells)
        if any(
            not isinstance(cell, StateCellRef)
            or not isinstance(view, RuntimeEpistemicCellView)
            or view.cell != cell
            for cell, view in cells.items()
        ):
            raise TypeError(
                "runtime epistemic cells must map StateCellRef to matching views"
            )
        resolved = dict(self.resolved_values)
        expected_resolved = {
            cell: view.resolved_value
            for cell, view in cells.items()
            if view.status == "resolved" and view.resolved_value is not None
        }
        if resolved != expected_resolved:
            raise ValueError(
                "runtime epistemic resolved values must match resolved cell views"
            )
        object.__setattr__(self, "cells", MappingProxyType(cells))
        object.__setattr__(self, "resolved_values", MappingProxyType(resolved))

    def to_dict(self) -> dict[str, object]:
        ordered = tuple(sorted(self.cells, key=_cell_key))
        return {
            "agent_id": self.agent_id,
            "source_at_time": self.source_at_time,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "seed_state": self.seed_state.to_dict(),
            "runtime_evidence_history": [
                item.to_dict() for item in self.runtime_evidence_history
            ],
            "cells": [self.cells[cell].to_dict() for cell in ordered],
            "resolved_values": [
                {
                    "cell": cell.to_dict(),
                    "value": self.resolved_values[cell].to_dict(),
                }
                for cell in ordered
                if cell in self.resolved_values
            ],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validate_ledger_identity(
    story: GenericNarrative,
    domain: DomainSpec,
    ledger: RuntimeEvidenceLedger,
) -> None:
    validate_narrative(story, domain)
    if not isinstance(ledger, RuntimeEvidenceLedger):
        raise TypeError("runtime cognition requires RuntimeEvidenceLedger")
    if ledger.domain_id != domain.domain_id:
        raise ValueError("runtime ledger domain id mismatch")
    if ledger.domain_version != domain.version:
        raise ValueError("runtime ledger domain version mismatch")
    if ledger.domain_spec_hash != domain.content_hash:
        raise ValueError("runtime ledger domain spec hash mismatch")
    if ledger.source_story_hash != story.content_hash:
        raise ValueError("runtime ledger source story mismatch")


def runtime_epistemic_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    ledger: RuntimeEvidenceLedger,
) -> RuntimeEpistemicState:
    try:
        _validate_ledger_identity(story, domain, ledger)
        seed = epistemic_state(
            story,
            domain,
            agent_id,
            at_time=ledger.source_at_time,
        )
        history = tuple(
            sorted(
                (
                    item
                    for batch in ledger.batches
                    for item in batch.evidence
                    if item.observer_id == agent_id
                ),
                key=_runtime_history_key,
            )
        )
        by_cell: dict[StateCellRef, list[RuntimeEpistemicEvidence]] = {}
        for item in history:
            by_cell.setdefault(item.cell, []).append(item)
        all_cells = set(seed.cells) | set(by_cell)
        cells: dict[StateCellRef, RuntimeEpistemicCellView] = {}
        for cell in sorted(all_cells, key=_cell_key):
            rows = tuple(by_cell.get(cell, ()))
            if not rows:
                seed_view = seed.cells[cell]
                cells[cell] = RuntimeEpistemicCellView(
                    cell=cell,
                    status=seed_view.status,
                    resolved_value=seed_view.resolved_value,
                    constraints=seed_view.constraints,
                    basis="authored_seed",
                    supporting_runtime_evidence_hashes=(),
                    last_runtime_step=None,
                )
                continue
            latest_step = max(item.step_index for item in rows)
            latest = tuple(item for item in rows if item.step_index == latest_step)
            semantics = {(item.relation, item.value) for item in latest}
            if len(semantics) != 1:
                raise RuntimeEpistemicResolutionError(
                    "same-step runtime percepts for one cell disagree semantically"
                )
            relation, value = next(iter(semantics))
            support = tuple(sorted(item.content_hash for item in latest))
            if relation == "equals":
                assert isinstance(value, TypedValue)
                status = "resolved"
                resolved_value = value
            else:
                status = "unknown"
                resolved_value = None
            cells[cell] = RuntimeEpistemicCellView(
                cell=cell,
                status=status,
                resolved_value=resolved_value,
                constraints=(),
                basis="runtime_perception",
                supporting_runtime_evidence_hashes=support,
                last_runtime_step=latest_step,
            )
        resolved = {
            cell: view.resolved_value
            for cell, view in cells.items()
            if view.status == "resolved" and view.resolved_value is not None
        }
        return RuntimeEpistemicState(
            agent_id=agent_id,
            source_at_time=ledger.source_at_time,
            step_index=ledger.current_step_index,
            ledger_hash=ledger.content_hash,
            seed_state=seed,
            runtime_evidence_history=history,
            cells=cells,
            resolved_values=resolved,
        )
    except RuntimeEpistemicResolutionError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeEpistemicResolutionError(
            "runtime epistemic state could not be resolved"
        ) from error


@dataclass(frozen=True)
class RuntimeBeliefModelSpec:
    model_id: str
    version: str
    seed_model: UncertainBeliefModelSpec
    runtime_parameters: Mapping[str, object]
    runtime_likelihood_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime belief model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="runtime belief model version"),
        )
        if not isinstance(self.seed_model, UncertainBeliefModelSpec):
            raise TypeError(
                "runtime belief seed model must be UncertainBeliefModelSpec"
            )
        object.__setattr__(
            self,
            "runtime_parameters",
            _freeze_parameters(self.runtime_parameters),
        )
        if not callable(self.runtime_likelihood_hook):
            raise TypeError("runtime belief likelihood hook must be callable")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "seed_model_hash": self.seed_model.content_hash,
            "runtime_parameters": _parameter_payload(self.runtime_parameters),
            "runtime_likelihood_hook_identity": measure_implementation(
                self.runtime_likelihood_hook
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeBeliefUpdateStep:
    evidence: RuntimeEpistemicEvidence
    prior: BeliefDistribution
    likelihoods: tuple[BeliefLikelihood, ...]
    posterior: BeliefDistribution

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, RuntimeEpistemicEvidence):
            raise TypeError(
                "runtime belief update evidence must be RuntimeEpistemicEvidence"
            )
        if not isinstance(self.prior, BeliefDistribution) or not isinstance(
            self.posterior, BeliefDistribution
        ):
            raise TypeError(
                "runtime belief update prior/posterior must be BeliefDistribution"
            )
        if not (
            self.evidence.cell == self.prior.cell == self.posterior.cell
        ):
            raise ValueError("runtime belief update records must describe one cell")
        if not isinstance(self.likelihoods, tuple):
            raise TypeError("runtime belief update likelihoods must be a tuple")
        likelihoods = tuple(self.likelihoods)
        if any(not isinstance(item, BeliefLikelihood) for item in likelihoods):
            raise TypeError(
                "runtime belief update likelihoods must be BeliefLikelihood values"
            )
        likelihood_keys = tuple(_value_key(item.value) for item in likelihoods)
        if len(set(likelihood_keys)) != len(likelihood_keys):
            raise ValueError("runtime belief update likelihood hypotheses must be unique")
        prior_keys = {_value_key(item.value) for item in self.prior.masses}
        posterior_keys = {_value_key(item.value) for item in self.posterior.masses}
        if prior_keys != posterior_keys or prior_keys != set(likelihood_keys):
            raise ValueError("runtime belief update hypothesis sets must match exactly")
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

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeUncertainBeliefCellView:
    cell: StateCellRef
    seed_posterior: BeliefDistribution
    posterior: BeliefDistribution
    updates: tuple[RuntimeBeliefUpdateStep, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("runtime uncertain belief cell view requires StateCellRef")
        if not isinstance(self.seed_posterior, BeliefDistribution) or not isinstance(
            self.posterior, BeliefDistribution
        ):
            raise TypeError(
                "runtime uncertain belief seed/posterior must be BeliefDistribution"
            )
        if self.seed_posterior.cell != self.cell or self.posterior.cell != self.cell:
            raise ValueError(
                "runtime uncertain belief distributions must match the cell"
            )
        if not isinstance(self.updates, tuple):
            raise TypeError("runtime uncertain belief updates must be a tuple")
        updates = tuple(self.updates)
        if any(not isinstance(item, RuntimeBeliefUpdateStep) for item in updates):
            raise TypeError(
                "runtime uncertain belief updates must be RuntimeBeliefUpdateStep values"
            )
        current = self.seed_posterior
        for update in updates:
            if update.prior.cell != self.cell or update.posterior.cell != self.cell:
                raise ValueError("runtime uncertain belief updates must match the cell")
            if update.prior != current:
                raise ValueError("runtime uncertain belief update chain is discontinuous")
            current = update.posterior
        if current != self.posterior:
            raise ValueError(
                "runtime uncertain belief posterior must equal the update-chain tail"
            )
        object.__setattr__(self, "updates", updates)

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "seed_posterior": self.seed_posterior.to_dict(),
            "posterior": self.posterior.to_dict(),
            "updates": [item.to_dict() for item in self.updates],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeUncertainBeliefState:
    agent_id: str
    model_id: str
    model_hash: str
    source_at_time: int | None
    step_index: int
    ledger_hash: str
    seed_belief_state: UncertainBeliefState
    runtime_evidence_history: tuple[RuntimeEpistemicEvidence, ...]
    cells: Mapping[StateCellRef, RuntimeUncertainBeliefCellView]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="runtime uncertain belief agent id"),
        )
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime uncertain belief model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="runtime uncertain belief model hash"),
        )
        object.__setattr__(
            self,
            "source_at_time",
            _cutoff(
                self.source_at_time,
                label="runtime uncertain belief source cutoff",
            ),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime uncertain belief step index"),
        )
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="runtime uncertain belief ledger hash"),
        )
        if not isinstance(self.seed_belief_state, UncertainBeliefState):
            raise TypeError(
                "runtime uncertain belief seed must be UncertainBeliefState"
            )
        if self.seed_belief_state.agent_id != self.agent_id:
            raise ValueError("runtime uncertain belief seed agent must match state agent")
        if not isinstance(self.runtime_evidence_history, tuple):
            raise TypeError("runtime uncertain belief evidence history must be a tuple")
        history = tuple(self.runtime_evidence_history)
        if any(not isinstance(item, RuntimeEpistemicEvidence) for item in history):
            raise TypeError(
                "runtime uncertain belief history must contain RuntimeEpistemicEvidence"
            )
        if any(item.observer_id != self.agent_id for item in history):
            raise ValueError("runtime uncertain belief history must contain only one agent")
        object.__setattr__(
            self,
            "runtime_evidence_history",
            tuple(sorted(history, key=_runtime_history_key)),
        )
        cells = dict(self.cells)
        if not cells:
            raise ValueError(
                "runtime uncertain belief state must contain at least one tracked cell"
            )
        if any(
            not isinstance(cell, StateCellRef)
            or not isinstance(view, RuntimeUncertainBeliefCellView)
            or view.cell != cell
            for cell, view in cells.items()
        ):
            raise TypeError(
                "runtime uncertain belief cells must map StateCellRef to matching views"
            )
        object.__setattr__(self, "cells", MappingProxyType(cells))

    def to_dict(self) -> dict[str, object]:
        ordered = tuple(sorted(self.cells, key=_cell_key))
        return {
            "agent_id": self.agent_id,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "source_at_time": self.source_at_time,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "seed_belief_state": self.seed_belief_state.to_dict(),
            "runtime_evidence_history": [
                item.to_dict() for item in self.runtime_evidence_history
            ],
            "cells": [self.cells[cell].to_dict() for cell in ordered],
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


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


def _validated_runtime_likelihoods(
    raw: object,
    hypotheses: tuple[TypedValue, ...],
) -> dict[str, float]:
    if not isinstance(raw, Mapping):
        raise RuntimeBeliefResolutionError(
            "runtime likelihood must be a mapping"
        )
    expected = {_value_key(value) for value in hypotheses}
    if set(raw) != expected:
        raise RuntimeBeliefResolutionError(
            "runtime likelihood keys must match the finite hypotheses exactly"
        )
    result: dict[str, float] = {}
    for key in expected:
        try:
            result[key] = _probability(
                raw[key],
                label="runtime likelihood value",
            )
        except (TypeError, ValueError) as error:
            raise RuntimeBeliefResolutionError(
                "runtime likelihood contains an invalid probability"
            ) from error
    return result


def _call_runtime_likelihood_hook(
    model: RuntimeBeliefModelSpec,
    agent_id: str,
    evidence: RuntimeEpistemicEvidence,
    hypotheses: tuple[TypedValue, ...],
) -> object:
    try:
        return model.runtime_likelihood_hook(
            agent_id,
            evidence.percept_view,
            hypotheses,
            model.runtime_parameters,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeBeliefResolutionError(
            "runtime likelihood hook could not produce a valid vector"
        ) from error


def runtime_uncertain_belief_state(
    story: GenericNarrative,
    domain: DomainSpec,
    agent_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeBeliefModelSpec,
    tracked_cells: tuple[StateCellRef, ...],
) -> RuntimeUncertainBeliefState:
    try:
        _validate_ledger_identity(story, domain, ledger)
        if not isinstance(model, RuntimeBeliefModelSpec):
            raise TypeError("runtime belief replay requires RuntimeBeliefModelSpec")
        if not isinstance(tracked_cells, tuple):
            raise TypeError("tracked cells must be a tuple")
        cells = tuple(tracked_cells)
        if not cells or any(not isinstance(cell, StateCellRef) for cell in cells):
            raise TypeError(
                "tracked cells must be a non-empty tuple of StateCellRef values"
            )
        if len(set(cells)) != len(cells):
            raise ValueError("tracked cells must be unique")

        seed = uncertain_epistemic_state(
            story,
            domain,
            agent_id,
            model.seed_model,
            cells,
            at_time=ledger.source_at_time,
        )
        tracked = set(cells)
        history = tuple(
            sorted(
                (
                    item
                    for batch in ledger.batches
                    for item in batch.evidence
                    if item.observer_id == agent_id and item.cell in tracked
                ),
                key=_runtime_history_key,
            )
        )
        by_cell: dict[StateCellRef, list[RuntimeEpistemicEvidence]] = {
            cell: [] for cell in cells
        }
        for item in history:
            by_cell[item.cell].append(item)

        views: dict[StateCellRef, RuntimeUncertainBeliefCellView] = {}
        for cell in cells:
            seed_posterior = seed.cells[cell].posterior
            current = seed_posterior
            updates: list[RuntimeBeliefUpdateStep] = []
            for evidence in by_cell[cell]:
                hypotheses = tuple(mass.value for mass in current.masses)
                raw_likelihoods = _call_runtime_likelihood_hook(
                    model,
                    agent_id,
                    evidence,
                    hypotheses,
                )
                likelihood_vector = _validated_runtime_likelihoods(
                    raw_likelihoods,
                    hypotheses,
                )
                prior_vector = {
                    _value_key(mass.value): mass.probability
                    for mass in current.masses
                }
                try:
                    posterior_vector = posterior_distribution(
                        prior_vector,
                        likelihood_vector,
                    )
                except ValueError as error:
                    raise RuntimeBeliefResolutionError(
                        "runtime evidence update has zero posterior mass"
                    ) from error
                posterior = _distribution_from_vector(
                    cell,
                    hypotheses,
                    posterior_vector,
                )
                likelihoods = tuple(
                    BeliefLikelihood(
                        value,
                        likelihood_vector[_value_key(value)],
                    )
                    for value in hypotheses
                )
                updates.append(
                    RuntimeBeliefUpdateStep(
                        evidence=evidence,
                        prior=current,
                        likelihoods=likelihoods,
                        posterior=posterior,
                    )
                )
                current = posterior
            views[cell] = RuntimeUncertainBeliefCellView(
                cell=cell,
                seed_posterior=seed_posterior,
                posterior=current,
                updates=tuple(updates),
            )

        return RuntimeUncertainBeliefState(
            agent_id=agent_id,
            model_id=model.model_id,
            model_hash=model.content_hash,
            source_at_time=ledger.source_at_time,
            step_index=ledger.current_step_index,
            ledger_hash=ledger.content_hash,
            seed_belief_state=seed,
            runtime_evidence_history=history,
            cells=views,
        )
    except RuntimeBeliefResolutionError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise RuntimeBeliefResolutionError(
            "runtime uncertain belief state could not be resolved"
        ) from error
