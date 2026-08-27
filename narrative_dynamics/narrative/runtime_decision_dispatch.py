from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re
from types import MappingProxyType

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import DomainSpec, validate_narrative
from narrative_dynamics.narrative.ir import GenericNarrative
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
from narrative_dynamics.narrative.runtime_perception import RuntimeEvidenceLedger
from narrative_dynamics.narrative.runtime_planning import (
    RuntimePlanningDecisionModelSpec,
    RuntimePlanningDecisionResolutionError,
    RuntimePlanningDecisionResult,
    run_runtime_planning_decision,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
    RuntimeReactiveDecisionResolutionError,
    RuntimeReactiveDecisionResult,
    run_runtime_reactive_decision,
)


_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12
_MODEL_TYPES = {
    "reactive": RuntimeReactiveDecisionModelSpec,
    "intentional": RuntimeIntentionalDecisionModelSpec,
    "planning": RuntimePlanningDecisionModelSpec,
}
_RESULT_TYPES = {
    "reactive": RuntimeReactiveDecisionResult,
    "intentional": RuntimeIntentionalDecisionResult,
    "planning": RuntimePlanningDecisionResult,
}


class RuntimeDecisionDispatchError(ValueError):
    """A typed runtime decision family could not be dispatched safely."""


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


def _freeze_policy(value: object, *, label: str) -> Mapping[str, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a non-empty mapping")
    frozen: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _text(raw_key, label=f"{label} action id")
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            raise TypeError(f"{label} probabilities must be numeric")
        probability = float(raw_value)
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError(
                f"{label} probabilities must be finite and non-negative"
            )
        frozen[key] = probability
    if not math.isclose(
        math.fsum(frozen.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{label} must sum to 1")
    return MappingProxyType({key: frozen[key] for key in sorted(frozen)})


def _map_choice(policy: Mapping[str, float]) -> str:
    maximum = max(policy.values())
    return min(key for key, value in policy.items() if value == maximum)


@dataclass(frozen=True)
class RuntimeDecisionModelSpec:
    model_kind: str
    model: (
        RuntimeReactiveDecisionModelSpec
        | RuntimeIntentionalDecisionModelSpec
        | RuntimePlanningDecisionModelSpec
    )

    def __post_init__(self) -> None:
        kind = _text(self.model_kind, label="runtime decision model kind")
        expected = _MODEL_TYPES.get(kind)
        if expected is None:
            raise ValueError(
                "runtime decision model kind must be reactive, intentional, or planning"
            )
        if not isinstance(self.model, expected):
            raise TypeError(
                "runtime decision model kind does not match nested model type"
            )
        object.__setattr__(self, "model_kind", kind)

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def model_version(self) -> str:
        return self.model.version

    @property
    def supported_decision_types(self) -> tuple[str, ...]:
        return self.model.supported_decision_types

    @property
    def nested_model_hash(self) -> str:
        return self.model.content_hash

    def to_dict(self) -> dict[str, object]:
        return {
            "model_kind": self.model_kind,
            "nested_model_id": self.model_id,
            "nested_model_version": self.model_version,
            "nested_model_hash": self.nested_model_hash,
            "wrapper_implementation_identity": measure_implementation(
                RuntimeDecisionModelSpec
            ).manifest_identity(),
            "runner_implementation_identity": measure_implementation(
                run_runtime_decision
            ).manifest_identity(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _actor_id(kind: str, result: object) -> str:
    if kind == "intentional":
        assert isinstance(result, RuntimeIntentionalDecisionResult)
        return result.belief_state.agent_id
    if kind == "reactive":
        assert isinstance(result, RuntimeReactiveDecisionResult)
        return result.actor_id
    assert kind == "planning"
    assert isinstance(result, RuntimePlanningDecisionResult)
    return result.actor_id


def _ledger_hash(kind: str, result: object) -> str:
    if kind == "intentional":
        assert isinstance(result, RuntimeIntentionalDecisionResult)
        return result.belief_state.ledger_hash
    if kind == "reactive":
        assert isinstance(result, RuntimeReactiveDecisionResult)
        return result.ledger_hash
    assert kind == "planning"
    assert isinstance(result, RuntimePlanningDecisionResult)
    return result.ledger_hash


@dataclass(frozen=True)
class RuntimeDecisionDispatchResult:
    model_kind: str
    decision_model_hash: str
    model_id: str
    model_hash: str
    decision_id: str
    actor_id: str
    step_index: int
    ledger_hash: str
    action_policy: Mapping[str, float]
    selected_action: str
    model_result_hash: str
    model_result: (
        RuntimeReactiveDecisionResult
        | RuntimeIntentionalDecisionResult
        | RuntimePlanningDecisionResult
    )

    def __post_init__(self) -> None:
        kind = _text(self.model_kind, label="runtime dispatch result model kind")
        expected = _RESULT_TYPES.get(kind)
        if expected is None:
            raise ValueError("runtime dispatch result model kind is unsupported")
        if not isinstance(self.model_result, expected):
            raise TypeError(
                "runtime dispatch result kind does not match nested result type"
            )
        object.__setattr__(self, "model_kind", kind)
        object.__setattr__(
            self,
            "decision_model_hash",
            _hash(
                self.decision_model_hash,
                label="runtime dispatch decision model hash",
            ),
        )
        object.__setattr__(
            self,
            "model_id",
            _text(self.model_id, label="runtime dispatch model id"),
        )
        object.__setattr__(
            self,
            "model_hash",
            _hash(self.model_hash, label="runtime dispatch model hash"),
        )
        object.__setattr__(
            self,
            "decision_id",
            _text(self.decision_id, label="runtime dispatch decision id"),
        )
        object.__setattr__(
            self,
            "actor_id",
            _text(self.actor_id, label="runtime dispatch actor id"),
        )
        object.__setattr__(
            self,
            "step_index",
            _step(self.step_index, label="runtime dispatch step"),
        )
        object.__setattr__(
            self,
            "ledger_hash",
            _hash(self.ledger_hash, label="runtime dispatch ledger hash"),
        )
        object.__setattr__(
            self,
            "model_result_hash",
            _hash(
                self.model_result_hash,
                label="runtime dispatch nested result hash",
            ),
        )
        if self.model_id != self.model_result.model_id:
            raise ValueError("runtime dispatch model id must match nested result")
        if self.model_hash != self.model_result.model_hash:
            raise ValueError("runtime dispatch model hash must match nested result")
        if self.decision_id != self.model_result.decision_id:
            raise ValueError("runtime dispatch decision must match nested result")
        if self.actor_id != _actor_id(kind, self.model_result):
            raise ValueError("runtime dispatch actor must match nested result")
        if self.step_index != self.model_result.step_index:
            raise ValueError("runtime dispatch step must match nested result")
        if self.ledger_hash != _ledger_hash(kind, self.model_result):
            raise ValueError("runtime dispatch ledger must match nested result")
        if self.model_result_hash != self.model_result.content_hash:
            raise ValueError(
                "runtime dispatch nested result hash must match nested result"
            )
        policy = _freeze_policy(
            self.action_policy,
            label="runtime dispatch action policy",
        )
        if dict(policy) != dict(self.model_result.action_policy):
            raise ValueError("runtime dispatch policy must match nested result")
        selected = _text(
            self.selected_action,
            label="runtime dispatch selected action",
        )
        if selected != self.model_result.selected_action:
            raise ValueError(
                "runtime dispatch selected action must match nested result"
            )
        if selected != _map_choice(policy):
            raise ValueError(
                "runtime dispatch selected action must be lexical MAP"
            )
        object.__setattr__(self, "action_policy", policy)
        object.__setattr__(self, "selected_action", selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "model_kind": self.model_kind,
            "decision_model_hash": self.decision_model_hash,
            "model_id": self.model_id,
            "model_hash": self.model_hash,
            "decision_id": self.decision_id,
            "actor_id": self.actor_id,
            "step_index": self.step_index,
            "ledger_hash": self.ledger_hash,
            "action_policy": {
                key: self.action_policy[key]
                for key in sorted(self.action_policy)
            },
            "selected_action": self.selected_action,
            "model_result_hash": self.model_result_hash,
            "model_result": self.model_result.to_dict(),
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _validated_model(model: object) -> RuntimeDecisionModelSpec:
    if not isinstance(model, RuntimeDecisionModelSpec):
        raise TypeError(
            "runtime decision dispatch requires RuntimeDecisionModelSpec"
        )
    validated = RuntimeDecisionModelSpec(model.model_kind, model.model)
    if validated.to_dict() != model.to_dict():
        raise ValueError("runtime decision model wrapper must be canonical")
    return validated


def _preflight_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    model: RuntimeDecisionModelSpec,
):
    validate_narrative(story, domain)
    requested = _text(decision_id, label="runtime dispatch decision id")
    decision = next((item for item in story.decisions if item.id == requested), None)
    if decision is None:
        raise ValueError("runtime dispatch decision is not declared by story")
    if decision.type_name not in model.supported_decision_types:
        raise ValueError("runtime dispatch model does not support decision type")
    return decision


def _run_family(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeDecisionModelSpec,
):
    if model.model_kind == "reactive":
        assert isinstance(model.model, RuntimeReactiveDecisionModelSpec)
        return run_runtime_reactive_decision(
            story,
            domain,
            decision_id,
            ledger,
            model.model,
        )
    if model.model_kind == "intentional":
        assert isinstance(model.model, RuntimeIntentionalDecisionModelSpec)
        return run_runtime_intentional_decision(
            story,
            domain,
            decision_id,
            ledger,
            model.model,
        )
    assert model.model_kind == "planning"
    assert isinstance(model.model, RuntimePlanningDecisionModelSpec)
    return run_runtime_planning_decision(
        story,
        domain,
        decision_id,
        ledger,
        model.model,
    )


def _dispatch_result(
    model: RuntimeDecisionModelSpec,
    nested: object,
) -> RuntimeDecisionDispatchResult:
    return RuntimeDecisionDispatchResult(
        model_kind=model.model_kind,
        decision_model_hash=model.content_hash,
        model_id=nested.model_id,
        model_hash=nested.model_hash,
        decision_id=nested.decision_id,
        actor_id=_actor_id(model.model_kind, nested),
        step_index=nested.step_index,
        ledger_hash=_ledger_hash(model.model_kind, nested),
        action_policy=nested.action_policy,
        selected_action=nested.selected_action,
        model_result_hash=nested.content_hash,
        model_result=nested,
    )


def _validate_result_against_model(
    result: RuntimeDecisionDispatchResult,
    model: RuntimeDecisionModelSpec,
) -> None:
    if result.model_kind != model.model_kind:
        raise ValueError("runtime dispatch result kind must match wrapper")
    if result.decision_model_hash != model.content_hash:
        raise ValueError("runtime dispatch result must bind exact wrapper hash")
    if result.model_id != model.model_id:
        raise ValueError("runtime dispatch result model id must match wrapper")
    if result.model_hash != model.nested_model_hash:
        raise ValueError(
            "runtime dispatch result nested hash must match wrapper"
        )
    RuntimeDecisionDispatchResult(
        model_kind=result.model_kind,
        decision_model_hash=result.decision_model_hash,
        model_id=result.model_id,
        model_hash=result.model_hash,
        decision_id=result.decision_id,
        actor_id=result.actor_id,
        step_index=result.step_index,
        ledger_hash=result.ledger_hash,
        action_policy=result.action_policy,
        selected_action=result.selected_action,
        model_result_hash=result.model_result_hash,
        model_result=result.model_result,
    )


def run_runtime_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeDecisionModelSpec,
) -> RuntimeDecisionDispatchResult:
    try:
        validated_model = _validated_model(model)
        decision = _preflight_decision(
            story,
            domain,
            decision_id,
            validated_model,
        )
        nested = _run_family(
            story,
            domain,
            decision.id,
            ledger,
            validated_model,
        )
        result = _dispatch_result(validated_model, nested)
        _validate_result_against_model(result, validated_model)
        return result
    except RuntimeDecisionDispatchError:
        raise
    except (
        RuntimeReactiveDecisionResolutionError,
        RuntimeIntentionalDecisionResolutionError,
        RuntimePlanningDecisionResolutionError,
        TypeError,
        ValueError,
        KeyError,
    ) as error:
        raise RuntimeDecisionDispatchError(
            "runtime decision could not be dispatched"
        ) from error


__all__ = (
    "RuntimeDecisionModelSpec",
    "RuntimeDecisionDispatchResult",
    "RuntimeDecisionDispatchError",
    "run_runtime_decision",
)
