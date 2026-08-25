from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import (
    Decision,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.replay import (
    EpistemicCellView,
    EpistemicEvidence,
    direct_state,
    epistemic_state,
    objective_state,
)


class DecisionResolutionError(ValueError):
    """A decision model could not produce a valid declared action."""


class EpistemicResolutionError(DecisionResolutionError):
    """A model required a state cell that its evidence could not resolve."""


class EvidenceAccess(str, Enum):
    DIRECT_ONLY = "direct_only"
    EPISTEMIC = "epistemic"
    OMNISCIENT = "omniscient"


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


@dataclass(frozen=True)
class DecisionChoice:
    selected_action: str
    evidence_refs: tuple[str, ...]
    inspected_cells: tuple[StateCellRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_action",
            _text(self.selected_action, label="selected action"),
        )
        evidence_refs = tuple(
            _text(item, label="decision evidence ref") for item in self.evidence_refs
        )
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("decision evidence refs must be unique")
        inspected_cells = tuple(self.inspected_cells)
        if any(not isinstance(item, StateCellRef) for item in inspected_cells):
            raise TypeError("decision inspected cells must be StateCellRef values")
        if len(set(inspected_cells)) != len(inspected_cells):
            raise ValueError("decision inspected cells must be unique")
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "inspected_cells", inspected_cells)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected_action": self.selected_action,
            "evidence_refs": list(self.evidence_refs),
            "inspected_cells": [item.to_dict() for item in self.inspected_cells],
        }


@dataclass(frozen=True)
class DecisionCellView(EpistemicCellView):
    """Immutable state-cell view exposed across the decision capability boundary."""

    @classmethod
    def from_epistemic(cls, view: EpistemicCellView) -> DecisionCellView:
        if not isinstance(view, EpistemicCellView):
            raise TypeError("decision cell view requires EpistemicCellView")
        return cls(
            cell=view.cell,
            status=view.status,
            resolved_value=view.resolved_value,
            constraints=view.constraints,
            evidence_kind=view.evidence_kind,
            supporting_id=view.supporting_id,
            source_agent=view.source_agent,
            evidence_logical_time=view.evidence_logical_time,
            evidence_refs=view.evidence_refs,
        )



def _action_vector(value: Mapping[str, float], *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, float] = {}
    for action_id, raw in value.items():
        key = _text(action_id, label=f"{label} action id")
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise TypeError(f"{label} values must be numeric")
        score = float(raw)
        if not math.isfinite(score):
            raise ValueError(f"{label} values must be finite")
        frozen[key] = score
    if not frozen:
        raise ValueError(f"{label} must contain at least one action")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class DecisionResult(DecisionChoice):
    model_id: str
    action_scores: Mapping[str, float]
    policy: Mapping[str, float]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "model_id", _text(self.model_id, label="decision result model id"))
        scores = _action_vector(self.action_scores, label="decision action scores")
        policy = _action_vector(self.policy, label="decision policy")
        if set(scores) != set(policy):
            raise ValueError("decision result action scores and policy must cover the same actions")
        if self.selected_action not in scores:
            raise ValueError("decision result selected action must be covered by action vectors")
        for action_id in scores:
            expected = 1.0 if action_id == self.selected_action else 0.0
            if scores[action_id] != expected or policy[action_id] != expected:
                raise ValueError("deterministic decision result vectors must be one-hot")
        object.__setattr__(self, "action_scores", scores)
        object.__setattr__(self, "policy", policy)

    @property
    def basis(self) -> DecisionChoice:
        return DecisionChoice(
            self.selected_action,
            self.evidence_refs,
            self.inspected_cells,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "selected_action": self.selected_action,
            "action_scores": dict(self.action_scores),
            "policy": dict(self.policy),
            "basis": {
                "evidence_refs": list(self.evidence_refs),
                "inspected_cells": [item.to_dict() for item in self.inspected_cells],
            },
        }


@dataclass(frozen=True)
class DecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    evidence_access: EvidenceAccess
    parameter_schema: tuple[str, ...]
    decision_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_id", _text(self.model_id, label="model id"))
        object.__setattr__(self, "version", _text(self.version, label="model version"))
        decision_types = tuple(
            _text(item, label="supported decision type")
            for item in self.supported_decision_types
        )
        if not decision_types:
            raise ValueError("decision model must support at least one decision type")
        if len(set(decision_types)) != len(decision_types):
            raise ValueError("supported decision types must be unique")
        if not isinstance(self.evidence_access, EvidenceAccess):
            raise TypeError("decision model evidence access must be EvidenceAccess")
        parameter_schema = tuple(
            _text(item, label="decision model parameter")
            for item in self.parameter_schema
        )
        if len(set(parameter_schema)) != len(parameter_schema):
            raise ValueError("decision model parameters must be unique")
        if not callable(self.decision_hook):
            raise TypeError("decision model hook must be callable")
        object.__setattr__(self, "supported_decision_types", decision_types)
        object.__setattr__(self, "parameter_schema", parameter_schema)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "supported_decision_types": list(self.supported_decision_types),
            "evidence_access": self.evidence_access.value,
            "parameter_schema": list(self.parameter_schema),
            "decision_hook_hash": measure_implementation(
                self.decision_hook
            ).content_hash,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class DecisionContext:
    decision: Decision
    evidence_history: tuple[EpistemicEvidence, ...]
    cells: Mapping[StateCellRef, DecisionCellView]

    def __post_init__(self) -> None:
        if not isinstance(self.decision, Decision):
            raise TypeError("decision context requires a Decision")
        history = tuple(self.evidence_history)
        if any(not isinstance(item, EpistemicEvidence) for item in history):
            raise TypeError("decision evidence history must contain EpistemicEvidence")
        cells: dict[StateCellRef, DecisionCellView] = {}
        for cell, view in dict(self.cells).items():
            if not isinstance(cell, StateCellRef) or not isinstance(view, EpistemicCellView):
                raise TypeError(
                    "decision context cells must map StateCellRef to matching views"
                )
            normalized = (
                view
                if isinstance(view, DecisionCellView)
                else DecisionCellView.from_epistemic(view)
            )
            if normalized.cell != cell:
                raise TypeError(
                    "decision context cells must map StateCellRef to matching views"
                )
            cells[cell] = normalized
        object.__setattr__(self, "evidence_history", history)
        object.__setattr__(self, "cells", MappingProxyType(cells))


def require_resolved_cell(
    context: DecisionContext,
    cell: StateCellRef,
) -> DecisionCellView:
    if not isinstance(context, DecisionContext):
        raise TypeError("resolved-cell lookup requires DecisionContext")
    if not isinstance(cell, StateCellRef):
        raise TypeError("resolved-cell lookup requires StateCellRef")
    view = context.cells.get(cell)
    if view is None:
        raise EpistemicResolutionError("decision evidence does not expose the requested cell")
    if view.status != "resolved" or view.resolved_value is None:
        raise EpistemicResolutionError(
            f"decision evidence leaves requested cell {view.status}"
        )
    return view


def _objective_context(
    story: GenericNarrative,
    domain: DomainSpec,
    decision: Decision,
) -> DecisionContext:
    state = objective_state(story, domain, at_time=decision.logical_time)
    cells: dict[StateCellRef, DecisionCellView] = {}
    for cell in decision.context_cells:
        value: TypedValue | None = state.get(cell)
        cells[cell] = DecisionCellView(
            cell=cell,
            status="resolved" if value is not None else "unknown",
            resolved_value=value,
            constraints=(),
            evidence_kind=None,
            supporting_id=None,
            source_agent=None,
            evidence_logical_time=None,
            evidence_refs=(),
        )
    return DecisionContext(decision, (), cells)


def _decision_context(
    story: GenericNarrative,
    domain: DomainSpec,
    decision: Decision,
    access: EvidenceAccess,
) -> DecisionContext:
    if access is EvidenceAccess.DIRECT_ONLY:
        state = direct_state(
            story,
            domain,
            decision.actor_id,
            at_time=decision.logical_time,
        )
        return DecisionContext(decision, state.evidence_history, state.cells)
    if access is EvidenceAccess.EPISTEMIC:
        state = epistemic_state(
            story,
            domain,
            decision.actor_id,
            at_time=decision.logical_time,
        )
        return DecisionContext(decision, state.evidence_history, state.cells)
    if access is EvidenceAccess.OMNISCIENT:
        return _objective_context(story, domain, decision)
    raise TypeError("unsupported decision evidence access")


def _validate_choice(context: DecisionContext, choice: DecisionChoice) -> None:
    declared_actions = {action.id for action in context.decision.actions}
    if choice.selected_action not in declared_actions:
        raise DecisionResolutionError(
            "decision model selected an action not declared by the decision"
        )

    accessible_refs: set[str] = set()
    for item in context.evidence_history:
        accessible_refs.add(item.supporting_id)
        accessible_refs.update(item.provenance_refs)
    if any(item not in accessible_refs for item in choice.evidence_refs):
        raise DecisionResolutionError(
            "decision model cited evidence outside its capability boundary"
        )
    if any(cell not in context.cells for cell in choice.inspected_cells):
        raise DecisionResolutionError(
            "decision model inspected a cell outside its capability boundary"
        )


def _result_for(
    context: DecisionContext,
    model: DecisionModelSpec,
    choice: DecisionChoice,
) -> DecisionResult:
    vector = {
        action.id: 1.0 if action.id == choice.selected_action else 0.0
        for action in context.decision.actions
    }
    return DecisionResult(
        selected_action=choice.selected_action,
        evidence_refs=choice.evidence_refs,
        inspected_cells=choice.inspected_cells,
        model_id=model.model_id,
        action_scores=vector,
        policy=vector,
    )


def run_decision_model(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: DecisionModelSpec,
) -> DecisionResult:
    validate_narrative(story, domain)
    if not isinstance(model, DecisionModelSpec):
        raise TypeError("decision execution requires DecisionModelSpec")
    decision_id = _text(decision_id, label="decision id")
    decision = next((item for item in story.decisions if item.id == decision_id), None)
    if decision is None:
        raise DecisionResolutionError("decision id is not declared by the narrative")
    if decision.type_name not in model.supported_decision_types:
        raise DecisionResolutionError(
            "decision type is not supported by the decision model"
        )

    context = _decision_context(story, domain, decision, model.evidence_access)
    choice = model.decision_hook(context)
    if not isinstance(choice, DecisionChoice):
        raise DecisionResolutionError("decision model hook must return DecisionChoice")
    _validate_choice(context, choice)
    return _result_for(context, model, choice)
