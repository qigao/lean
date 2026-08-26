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
from narrative_dynamics.narrative.ir import GenericNarrative, StateCellRef, TypedValue
from narrative_dynamics.narrative.replay import direct_state
from narrative_dynamics.narrative.runtime_perception import RuntimeEvidenceLedger


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12
_CUE_STATUSES = frozenset({"resolved", "unknown"})


class RuntimeReactiveDecisionResolutionError(ValueError):
    """A runtime reactive model could not resolve a declared action policy."""


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


def _positive(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
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
        raise TypeError("runtime reactive parameters must be a mapping")
    frozen = _freeze_parameter(value, label="runtime reactive parameters")
    assert isinstance(frozen, Mapping)
    return frozen


def _parameter_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _parameter_payload(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_parameter_payload(item) for item in value]
    return value


def _freeze_cues(
    value: object,
    *,
    label: str,
    step_index: int,
) -> Mapping[StateCellRef, ReactiveCueView]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[StateCellRef, ReactiveCueView] = {}
    for cell, view in value.items():
        if not isinstance(cell, StateCellRef):
            raise TypeError(f"{label} keys must be StateCellRef values")
        if not isinstance(view, ReactiveCueView) or view.cell != cell:
            raise TypeError(f"{label} values must be matching ReactiveCueView values")
        if view.step_index != step_index:
            raise ValueError(f"{label} views must bind the containing step")
        frozen[cell] = view
    if not frozen:
        raise ValueError(f"{label} must contain at least one cue")
    return MappingProxyType(
        {cell: frozen[cell] for cell in sorted(frozen, key=_cell_key)}
    )


def _cue_payload(value: Mapping[StateCellRef, ReactiveCueView]) -> list[dict[str, object]]:
    return [value[cell].to_dict() for cell in sorted(value, key=_cell_key)]


def _freeze_scores(value: object, *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, label=f"{label} action id")
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise TypeError(f"{label} values must be numeric")
        number = float(raw_value)
        if not math.isfinite(number):
            raise ValueError(f"{label} values must be finite")
        frozen[key] = number
    if not frozen:
        raise ValueError(f"{label} must contain at least one action")
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _freeze_policy(value: object, *, label: str) -> Mapping[str, float]:
    policy = _freeze_scores(value, label=label)
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
        raise ValueError("runtime reactive MAP requires a non-empty policy")
    maximum = max(policy.values())
    return min(key for key, value in policy.items() if value == maximum)


@dataclass(frozen=True)
class ReactiveCueView:
    cell: StateCellRef
    status: str
    value: TypedValue | None
    step_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.cell, StateCellRef):
            raise TypeError("reactive cue cell must be StateCellRef")
        if self.status not in _CUE_STATUSES:
            raise ValueError("reactive cue status must be resolved or unknown")
        if self.status == "resolved":
            if not isinstance(self.value, TypedValue):
                raise TypeError("resolved reactive cue requires TypedValue")
        elif self.value is not None:
            raise ValueError("unknown reactive cue cannot contain a value")
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="reactive cue step index"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "cell": self.cell.to_dict(),
            "status": self.status,
            "value": None if self.value is None else self.value.to_dict(),
            "step_index": self.step_index,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeReactiveCueSnapshot:
    actor_id: str
    decision_id: str
    step_index: int
    ledger_hash: str
    cues: Mapping[StateCellRef, ReactiveCueView]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "actor_id",
            _text(self.actor_id, label="reactive snapshot actor id"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="reactive snapshot decision id"),
        )
        step = _step(self.step_index, label="reactive snapshot step index")
        object.__setattr__(self, "step_index", step)
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="reactive snapshot ledger hash"),
        )
        object.__setattr__(
            self,
            "cues",
            _freeze_cues(self.cues, label="reactive snapshot cues", step_index=step),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "actor_id": self.actor_id,
            "decision_id": self.decision_id,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "cues": _cue_payload(self.cues),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeReactiveDecisionContext:
    decision_id: str
    actor_id: str
    decision_type: str
    step_index: int
    actions: tuple[str, ...]
    cues: Mapping[StateCellRef, ReactiveCueView]
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="reactive context decision id"),
        )
        object.__setattr__(
            self,
            "actor_id",
            _text(self.actor_id, label="reactive context actor id"),
        )
        object.__setattr__(
            self,
            "decision_type",
            _text(self.decision_type, label="reactive context decision type"),
        )
        step = _step(self.step_index, label="reactive context step index")
        object.__setattr__(self, "step_index", step)
        if not isinstance(self.actions, tuple):
            raise TypeError("reactive context actions must be a tuple")
        actions = tuple(
            _text(action, label="reactive context action id") for action in self.actions
        )
        if not actions:
            raise ValueError("reactive context requires at least one action")
        if len(set(actions)) != len(actions):
            raise ValueError("reactive context action ids must be unique")
        object.__setattr__(self, "actions", tuple(sorted(actions)))
        object.__setattr__(
            self,
            "cues",
            _freeze_cues(self.cues, label="reactive context cues", step_index=step),
        )
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))

    def to_dict(self) -> dict[str, object]:
        return {
            "decision_id": self.decision_id,
            "actor_id": self.actor_id,
            "decision_type": self.decision_type,
            "step_index": self.step_index,
            "actions": list(self.actions),
            "cues": _cue_payload(self.cues),
            "parameters": _parameter_payload(self.parameters),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeReactiveDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    cue_cells: tuple[StateCellRef, ...]
    parameters: Mapping[str, object]
    beta: float
    score_hook: object = field(compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime reactive model id"),
        )
        object.__setattr__(
            self,
            "version",
            _text(self.version, label="runtime reactive model version"),
        )
        if not isinstance(self.supported_decision_types, tuple):
            raise TypeError("runtime reactive supported decision types must be a tuple")
        decision_types = tuple(
            _text(item, label="runtime reactive decision type")
            for item in self.supported_decision_types
        )
        if not decision_types:
            raise ValueError("runtime reactive model requires a supported decision type")
        if len(set(decision_types)) != len(decision_types):
            raise ValueError("runtime reactive supported decision types must be unique")
        object.__setattr__(
            self,
            "supported_decision_types",
            tuple(sorted(decision_types)),
        )
        if not isinstance(self.cue_cells, tuple):
            raise TypeError("runtime reactive cue cells must be a tuple")
        cue_cells = tuple(self.cue_cells)
        if not cue_cells:
            raise ValueError("runtime reactive model requires at least one cue cell")
        if any(not isinstance(cell, StateCellRef) for cell in cue_cells):
            raise TypeError("runtime reactive cue cells must contain StateCellRef values")
        if len(set(cue_cells)) != len(cue_cells):
            raise ValueError("runtime reactive cue cells must be unique")
        object.__setattr__(
            self,
            "cue_cells",
            tuple(sorted(cue_cells, key=_cell_key)),
        )
        object.__setattr__(self, "parameters", _freeze_parameters(self.parameters))
        object.__setattr__(
            self,
            "beta",
            _positive(self.beta, label="runtime reactive beta"),
        )
        if not callable(self.score_hook):
            raise TypeError("runtime reactive score hook must be callable")

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "version": self.version,
            "supported_decision_types": list(self.supported_decision_types),
            "cue_cells": [cell.to_dict() for cell in self.cue_cells],
            "parameters": _parameter_payload(self.parameters),
            "beta": self.beta,
            "score_hook_implementation_identity": measure_implementation(
                self.score_hook
            ).manifest_identity(),
            "runtime_implementation_identity": measure_implementation(
                RuntimeReactiveDecisionModelSpec
            ).manifest_identity(),
            "softmax_implementation_identity": measure_implementation(
                finite_softmax
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


@dataclass(frozen=True)
class RuntimeReactiveDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    actor_id: str
    step_index: int
    ledger_hash: str
    cue_snapshot_hash: str
    cue_snapshot: RuntimeReactiveCueSnapshot
    action_scores: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime reactive result model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="runtime reactive result model hash"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="runtime reactive result decision id"),
        )
        object.__setattr__(
            self,
            "actor_id",
            _text(self.actor_id, label="runtime reactive result actor id"),
        )
        step = _step(self.step_index, label="runtime reactive result step index")
        object.__setattr__(self, "step_index", step)
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="runtime reactive result ledger hash"),
        )
        object.__setattr__(
            self,
            "cue_snapshot_hash",
            _hash(
                self.cue_snapshot_hash,
                label="runtime reactive result cue snapshot hash",
            ),
        )
        if not isinstance(self.cue_snapshot, RuntimeReactiveCueSnapshot):
            raise TypeError(
                "runtime reactive result cue_snapshot must be RuntimeReactiveCueSnapshot"
            )
        if self.cue_snapshot.actor_id != self.actor_id:
            raise ValueError("runtime reactive result actor must match cue snapshot")
        if self.cue_snapshot.decision_id != self.decision_id:
            raise ValueError("runtime reactive result decision must match cue snapshot")
        if self.cue_snapshot.step_index != step:
            raise ValueError("runtime reactive result step must match cue snapshot")
        if self.cue_snapshot.ledger_hash != self.ledger_hash:
            raise ValueError("runtime reactive result ledger must match cue snapshot")
        if self.cue_snapshot.content_hash != self.cue_snapshot_hash:
            raise ValueError("runtime reactive result cue snapshot hash is inconsistent")
        scores = _freeze_scores(self.action_scores, label="runtime reactive action scores")
        policy = _freeze_policy(self.action_policy, label="runtime reactive action policy")
        if set(scores) != set(policy):
            raise ValueError("runtime reactive scores and policy must cover same actions")
        selected = _text(
            self.selected_action,
            label="runtime reactive selected action",
        )
        if selected not in policy:
            raise ValueError("runtime reactive selected action must belong to policy")
        if selected != _map_choice(policy):
            raise ValueError("runtime reactive selected action must be lexical MAP")
        object.__setattr__(self, "action_scores", scores)
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_action", selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "decision_id": self.decision_id,
            "actor_id": self.actor_id,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "cue_snapshot_hash": self.cue_snapshot_hash,
            "cue_snapshot": self.cue_snapshot.to_dict(),
            "action_scores": dict(self.action_scores),
            "action_policy": dict(self.action_policy),
            "selected_action": self.selected_action,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validated_model(model: object) -> RuntimeReactiveDecisionModelSpec:
    if not isinstance(model, RuntimeReactiveDecisionModelSpec):
        raise TypeError(
            "runtime reactive execution requires RuntimeReactiveDecisionModelSpec"
        )
    return RuntimeReactiveDecisionModelSpec(
        model.model_id,
        model.version,
        model.supported_decision_types,
        model.cue_cells,
        model.parameters,
        model.beta,
        model.score_hook,
    )


def _validated_ledger(
    story: GenericNarrative,
    domain: DomainSpec,
    ledger: object,
) -> RuntimeEvidenceLedger:
    if not isinstance(ledger, RuntimeEvidenceLedger):
        raise TypeError("runtime reactive execution requires RuntimeEvidenceLedger")
    canonical = RuntimeEvidenceLedger(
        domain_id=ledger.domain_id,
        domain_version=ledger.domain_version,
        domain_spec_hash=ledger.domain_spec_hash,
        source_story_hash=ledger.source_story_hash,
        source_at_time=ledger.source_at_time,
        initial_world_state_hash=ledger.initial_world_state_hash,
        current_world_state_hash=ledger.current_world_state_hash,
        batches=ledger.batches,
    )
    if canonical.domain_id != domain.domain_id:
        raise ValueError("runtime reactive ledger domain id mismatch")
    if canonical.domain_version != domain.version:
        raise ValueError("runtime reactive ledger domain version mismatch")
    if canonical.domain_spec_hash != domain.content_hash:
        raise ValueError("runtime reactive ledger domain spec mismatch")
    if canonical.source_story_hash != story.content_hash:
        raise ValueError("runtime reactive ledger source story mismatch")
    return canonical


def _current_cue_snapshot(
    story: GenericNarrative,
    domain: DomainSpec,
    decision,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeReactiveDecisionModelSpec,
) -> RuntimeReactiveCueSnapshot:
    step_index = ledger.current_step_index
    cues: dict[StateCellRef, ReactiveCueView] = {}

    if step_index == 0:
        state = direct_state(
            story,
            domain,
            decision.actor_id,
            at_time=ledger.source_at_time,
        )
        for cell in model.cue_cells:
            view = state.cells.get(cell)
            if (
                view is not None
                and view.status == "resolved"
                and view.resolved_value is not None
            ):
                cues[cell] = ReactiveCueView(
                    cell,
                    "resolved",
                    view.resolved_value,
                    step_index,
                )
            else:
                cues[cell] = ReactiveCueView(cell, "unknown", None, step_index)
    else:
        if not ledger.batches:
            raise ValueError("runtime reactive current step requires an evidence batch")
        batch = ledger.batches[-1]
        if batch.step_index != step_index:
            raise ValueError("runtime reactive latest batch must equal current step")
        for cell in model.cue_cells:
            rows = tuple(
                item
                for item in batch.evidence
                if item.observer_id == decision.actor_id and item.cell == cell
            )
            if not rows:
                cues[cell] = ReactiveCueView(cell, "unknown", None, step_index)
                continue
            semantics = {(item.relation, item.value) for item in rows}
            if len(semantics) != 1:
                raise RuntimeReactiveDecisionResolutionError(
                    "same-step reactive cues disagree semantically"
                )
            relation, value = next(iter(semantics))
            if relation == "equals":
                if not isinstance(value, TypedValue):
                    raise ValueError("reactive equals cue requires TypedValue")
                cues[cell] = ReactiveCueView(cell, "resolved", value, step_index)
            elif relation == "clear":
                cues[cell] = ReactiveCueView(cell, "unknown", None, step_index)
            else:
                raise ValueError("runtime reactive cue relation is unsupported")

    return RuntimeReactiveCueSnapshot(
        actor_id=decision.actor_id,
        decision_id=decision.id,
        step_index=step_index,
        ledger_hash=ledger.content_hash,
        cues=cues,
    )


def run_runtime_reactive_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeReactiveDecisionModelSpec,
) -> RuntimeReactiveDecisionResult:
    try:
        validate_narrative(story, domain)
        resolved_model = _validated_model(model)
        resolved_ledger = _validated_ledger(story, domain, ledger)
        resolved_decision_id = _text(decision_id, label="runtime reactive decision id")
        decision = next(
            (item for item in story.decisions if item.id == resolved_decision_id),
            None,
        )
        if decision is None:
            raise ValueError("runtime reactive decision template is not declared")
        if decision.type_name not in resolved_model.supported_decision_types:
            raise ValueError("runtime reactive decision type is not supported")
        if not decision.context_cells or len(set(decision.context_cells)) != len(
            decision.context_cells
        ):
            raise ValueError("runtime reactive decision context must be non-empty and unique")
        if not set(resolved_model.cue_cells).issubset(set(decision.context_cells)):
            raise ValueError("runtime reactive cue cells exceed decision context capability")
        action_ids = tuple(action.id for action in decision.actions)
        if not action_ids or len(set(action_ids)) != len(action_ids):
            raise ValueError("runtime reactive decision actions must be non-empty and unique")
        if (
            resolved_ledger.source_at_time is not None
            and decision.logical_time > resolved_ledger.source_at_time
        ):
            raise ValueError("runtime reactive decision template occurs after source cutoff")
        snapshot = _current_cue_snapshot(
            story,
            domain,
            decision,
            resolved_ledger,
            resolved_model,
        )
        context = RuntimeReactiveDecisionContext(
            decision_id=decision.id,
            actor_id=decision.actor_id,
            decision_type=decision.type_name,
            step_index=snapshot.step_index,
            actions=tuple(action_ids),
            cues=snapshot.cues,
            parameters=resolved_model.parameters,
        )
    except RuntimeReactiveDecisionResolutionError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise RuntimeReactiveDecisionResolutionError(
            "runtime reactive prerequisites could not be resolved"
        ) from error

    try:
        raw_scores = resolved_model.score_hook(context)
    except Exception as error:
        raise RuntimeReactiveDecisionResolutionError(
            "runtime reactive score hook failed"
        ) from error

    try:
        scores = _freeze_scores(raw_scores, label="runtime reactive action scores")
        if set(scores) != set(context.actions):
            raise ValueError(
                "runtime reactive scores must cover authored actions exactly"
            )
        try:
            raw_policy = finite_softmax(scores, beta=resolved_model.beta)
        except (TypeError, ValueError, OverflowError) as error:
            raise RuntimeReactiveDecisionResolutionError(
                "runtime reactive softmax could not produce a policy"
            ) from error
        policy = _freeze_policy(raw_policy, label="runtime reactive action policy")
        if set(policy) != set(context.actions):
            raise ValueError(
                "runtime reactive policy must cover authored actions exactly"
            )
        selected = _map_choice(policy)
        return RuntimeReactiveDecisionResult(
            model_id=resolved_model.model_id,
            model_hash=resolved_model.content_hash,
            decision_id=decision.id,
            actor_id=decision.actor_id,
            step_index=snapshot.step_index,
            ledger_hash=resolved_ledger.content_hash,
            cue_snapshot_hash=snapshot.content_hash,
            cue_snapshot=snapshot,
            action_scores=scores,
            action_policy=policy,
            selected_action=selected,
        )
    except RuntimeReactiveDecisionResolutionError:
        raise
    except (TypeError, ValueError, KeyError, OverflowError) as error:
        raise RuntimeReactiveDecisionResolutionError(
            "runtime reactive action selection could not be resolved"
        ) from error


__all__ = [
    "ReactiveCueView",
    "RuntimeReactiveCueSnapshot",
    "RuntimeReactiveDecisionContext",
    "RuntimeReactiveDecisionModelSpec",
    "RuntimeReactiveDecisionResolutionError",
    "RuntimeReactiveDecisionResult",
    "run_runtime_reactive_decision",
]
