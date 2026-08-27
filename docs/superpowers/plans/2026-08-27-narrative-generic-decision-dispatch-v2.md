# Generic Runtime Decision Dispatch / Multi-Model Scheduler V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one deterministic multi-agent narrative simulation to schedule reactive, intentional, and planning agents in the same round through one closed typed runtime decision-dispatch boundary.

**Architecture:** Add an isolated `runtime_decision_dispatch.py` sidecar that owns the only three-family branch and returns a canonical common envelope containing the exact family-specific result. Migrate `simulation.py` from intentional-specific model/result fields to the generic wrapper/envelope while leaving world transition, observation projection, percept admission, and family-specific algorithms unchanged.

**Tech Stack:** Python 3, standard-library dataclasses/typing, existing narrative runtime contracts, existing implementation attestation and stable content hashing, `unittest`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-generic-decision-dispatch-v2-design.md`

## Global Constraints

- Integrated base is `proof/narrative-dynamics-v0` at `8317339cde03d46e0500798ebb30dfa27ca43e38` unless that branch moves before execution; if it moves, record the new exact base before the first test commit.
- Feature branch is `work/narrative-generic-decision-dispatch-v2`.
- Preserve strict RED -> GREEN -> atomic commit discipline.
- Pure changes under `docs/superpowers/specs/**` and `docs/superpowers/plans/**` must not trigger `proof`; test and production changes must trigger normal CI.
- Supported generic model kinds are exactly `reactive`, `intentional`, and `planning`.
- Do not accept arbitrary protocol, callback, duck-typed, or plugin decision models.
- Do not modify family algorithms in `runtime_reactive.py`, `runtime_intention.py`, or `runtime_planning.py`.
- `runtime_decision_dispatch.py` must not import or receive objective world state, scheduler state, observation projection, RNG, or an explicit step parameter.
- All scheduled agents must decide against the exact same immutable prior evidence ledger before `advance_world_step()` runs.
- Preserve atomic world-step behavior and current typed overlapping-write rejection; do not add actor priority or last-writer-wins behavior.
- Do not add stochastic world or observation semantics.
- Do not change `GenericNarrative`, `DomainSpec`, `runtime_cognition.py`, `world.py`, `observation_projection.py`, model-comparison infrastructure, prison adapters, or Lean sources.
- V2 preserves all-intentional behavioral semantics, not pre-V2 simulation/model/step/trajectory content hashes.
- `ActionIntent.selection_result_hash` must bind `RuntimeDecisionDispatchResult.content_hash`, which in turn must bind the exact nested family result hash and payload.
- Package-root `narrative_dynamics` must remain isolated from the new narrative runtime surface.
- Do not mark the feature complete until a PR-triggered exact-head `proof` run is `completed/success` and its Python log shows zero failures/errors.

---

## File Structure

### Create

- `narrative_dynamics/narrative/runtime_decision_dispatch.py`
  - Closed typed wrapper for the three family model specs.
  - Canonical common result envelope retaining the exact nested family result.
  - Typed dispatch error.
  - Generic `run_runtime_decision(...)` runner.
  - No world/simulation/projection capability.

- `tests/test_narrative_runtime_decision_dispatch.py`
  - Closed tag/type contract.
  - Identity and attestation binding.
  - Common-result binding and forgery rejection.
  - Exact direct dispatch equivalence to each existing family runner.
  - Error-cause preservation, replay, public signature, and import isolation.

### Modify

- `narrative_dynamics/narrative/simulation.py`
  - `RuntimeAgentSpec.intentional_model` -> `decision_model`.
  - `SimulationAgentStep.decision_result` -> `RuntimeDecisionDispatchResult`.
  - `simulate_step()` -> `run_runtime_decision()`.
  - World/projection/admission sequencing remains unchanged.

- `narrative_dynamics/narrative/__init__.py`
  - Export exactly four new dispatch public names from the narrative package.

- `tests/test_narrative_simulation.py`
  - Migrate existing intentional fixtures to explicit generic wrappers.
  - Preserve all-intentional behavioral regressions.
  - Add three-family heterogeneous single-round and two-round replay tests.

- `tests/test_narrative_trust_api.py`
  - Extend the exact narrative public surface by exactly four names.
  - Preserve root-package isolation.

### Explicitly unchanged

- `narrative_dynamics/narrative/runtime_reactive.py`
- `narrative_dynamics/narrative/runtime_intention.py`
- `narrative_dynamics/narrative/runtime_planning.py`
- `narrative_dynamics/narrative/runtime_cognition.py`
- `narrative_dynamics/narrative/world.py`
- `narrative_dynamics/narrative/observation_projection.py`
- `narrative_dynamics/model_comparison.py`
- `narrative_dynamics/adapters/prison_pomdp.py`
- all Lean sources

---

### Task 1: Establish the generic dispatch test-only RED boundary

**Files:**
- Create: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes existing family runners:
  - `run_runtime_reactive_decision(story, domain, decision_id, ledger, model)`
  - `run_runtime_intentional_decision(story, domain, decision_id, ledger, model)`
  - `run_runtime_planning_decision(story, domain, decision_id, ledger, model)`
- Defines the RED contract for:
  - `RuntimeDecisionModelSpec`
  - `RuntimeDecisionDispatchResult`
  - `RuntimeDecisionDispatchError`
  - `run_runtime_decision`

- [ ] **Step 1: Create guarded imports and reusable three-family fixtures**

Create `tests/test_narrative_runtime_decision_dispatch.py` with the exact existing fixture locations and a guarded import for the not-yet-created dispatch module:

```python
from __future__ import annotations

from dataclasses import fields, replace
import inspect
import unittest

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
from narrative_dynamics.narrative.runtime_planning import (
    PlanningHiddenState,
    PlanningObservation,
    RuntimePlanningDecisionModelSpec,
    RuntimePlanningDecisionResult,
    run_runtime_planning_decision,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
    RuntimeReactiveDecisionResolutionError,
    RuntimeReactiveDecisionResult,
    run_runtime_reactive_decision,
)
from tests.test_narrative_runtime_cognition import (
    empty_runtime_case,
    make_runtime_belief_model,
    phase_cell,
)
from tests.test_narrative_runtime_intention import make_runtime_intentional_model
from tests.test_narrative_runtime_planning import (
    IdentityTransitionHook,
    NoInformationObservationHook,
    ParameterRewardHook,
    ProductCouplingHook,
)
from tests.test_narrative_runtime_reactive import CueScoreHook, RaisingScoreHook

_DISPATCH_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.narrative.runtime_decision_dispatch as dispatch_module
    from narrative_dynamics.narrative.runtime_decision_dispatch import (
        RuntimeDecisionDispatchError,
        RuntimeDecisionDispatchResult,
        RuntimeDecisionModelSpec,
        run_runtime_decision,
    )
except ImportError as error:
    _DISPATCH_IMPORT_ERROR = error


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


def make_reactive_model(*, hook=None):
    if hook is None:
        hook = CueScoreHook()
    return RuntimeReactiveDecisionModelSpec(
        "dispatch-reactive",
        "1",
        ("phase-choice",),
        (phase_cell(),),
        {},
        2.0,
        hook,
    )


def make_planning_model():
    return RuntimePlanningDecisionModelSpec(
        "dispatch-planning",
        "1",
        ("phase-choice",),
        (phase_cell(),),
        (),
        make_runtime_belief_model(),
        (
            PlanningHiddenState(
                "active",
                {phase_cell(): TypedValue("PhaseState", "active")},
            ),
            PlanningHiddenState(
                "ready",
                {phase_cell(): TypedValue("PhaseState", "ready")},
            ),
        ),
        (PlanningObservation("none", {}),),
        (("a1-active", "a1-ready"),),
        1.0,
        2.0,
        {"action_rewards": {"a1-active": 2.0, "a1-ready": 0.0}},
        ProductCouplingHook(),
        IdentityTransitionHook(),
        NoInformationObservationHook(),
        ParameterRewardHook(),
    )


class NarrativeRuntimeDecisionDispatchTests(unittest.TestCase):
    def require_dispatch(self) -> None:
        if _DISPATCH_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime decision dispatch boundary is missing: "
                f"{_DISPATCH_IMPORT_ERROR}"
            )
```

- [ ] **Step 2: Add wrapper identity, closure, signature, and isolation tests**

```python
def test_model_wrapper_is_closed_typed_and_binds_nested_and_dispatch_identity(self):
    self.require_dispatch()
    reactive = make_reactive_model()
    intentional = make_runtime_intentional_model()
    planning = make_planning_model()
    wrappers = (
        RuntimeDecisionModelSpec("reactive", reactive),
        RuntimeDecisionModelSpec("intentional", intentional),
        RuntimeDecisionModelSpec("planning", planning),
    )
    self.assertEqual(
        tuple(item.model_kind for item in wrappers),
        ("reactive", "intentional", "planning"),
    )
    self.assertEqual(wrappers[0].model_id, reactive.model_id)
    self.assertEqual(wrappers[1].model_version, intentional.version)
    self.assertEqual(wrappers[2].nested_model_hash, planning.content_hash)
    self.assertEqual(
        wrappers[1].supported_decision_types,
        intentional.supported_decision_types,
    )
    self.assertEqual(
        RuntimeDecisionModelSpec("reactive", make_reactive_model()).content_hash,
        wrappers[0].content_hash,
    )
    self.assertNotEqual(wrappers[0].content_hash, wrappers[1].content_hash)
    self.assertNotEqual(wrappers[1].content_hash, wrappers[2].content_hash)
    payload = wrappers[0].to_dict()
    self.assertEqual(
        payload["wrapper_implementation_identity"],
        measure_implementation(RuntimeDecisionModelSpec).manifest_identity(),
    )
    self.assertEqual(
        payload["runner_implementation_identity"],
        measure_implementation(run_runtime_decision).manifest_identity(),
    )
    for kind, nested in (
        ("planning", reactive),
        ("reactive", intentional),
        ("intentional", planning),
        ("unknown", reactive),
    ):
        with self.subTest(kind=kind):
            with self.assertRaises((TypeError, ValueError)):
                RuntimeDecisionModelSpec(kind, nested)
    with self.assertRaises((TypeError, ValueError)):
        RuntimeDecisionModelSpec("reactive", object())


def test_public_signature_and_module_isolation_exclude_world_scheduler_projection_rng(self):
    self.require_dispatch()
    self.assertEqual(
        tuple(inspect.signature(run_runtime_decision).parameters),
        ("story", "domain", "decision_id", "ledger", "model"),
    )
    source = inspect.getsource(dispatch_module)
    for forbidden in (
        "narrative_dynamics.narrative.world",
        "narrative_dynamics.narrative.simulation",
        "narrative_dynamics.narrative.observation_projection",
        "import random",
        "from random",
    ):
        self.assertNotIn(forbidden, source)
    self.assertEqual(
        set(dispatch_module.__all__),
        {
            "RuntimeDecisionModelSpec",
            "RuntimeDecisionDispatchResult",
            "RuntimeDecisionDispatchError",
            "run_runtime_decision",
        },
    )
```

- [ ] **Step 3: Add exact direct-family projection and constructor-forgery tests**

```python
def test_direct_family_dispatch_preserves_exact_common_fields_and_nested_result(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    cases = (
        (
            "reactive",
            make_reactive_model(),
            run_runtime_reactive_decision,
            RuntimeReactiveDecisionResult,
        ),
        (
            "intentional",
            make_runtime_intentional_model(),
            run_runtime_intentional_decision,
            RuntimeIntentionalDecisionResult,
        ),
        (
            "planning",
            make_planning_model(),
            run_runtime_planning_decision,
            RuntimePlanningDecisionResult,
        ),
    )
    for kind, nested_model, family_runner, result_type in cases:
        with self.subTest(kind=kind):
            family = family_runner(
                story,
                domain,
                "d-a1-phase",
                ledger,
                nested_model,
            )
            wrapper = RuntimeDecisionModelSpec(kind, nested_model)
            result = run_runtime_decision(
                story,
                domain,
                "d-a1-phase",
                ledger,
                wrapper,
            )
            self.assertIsInstance(result.model_result, result_type)
            self.assertEqual(result.model_result, family)
            self.assertEqual(result.model_kind, kind)
            self.assertEqual(result.decision_model_hash, wrapper.content_hash)
            self.assertEqual(result.model_id, family.model_id)
            self.assertEqual(result.model_hash, family.model_hash)
            self.assertEqual(result.decision_id, family.decision_id)
            self.assertEqual(result.step_index, family.step_index)
            self.assertEqual(result.action_policy, family.action_policy)
            self.assertEqual(result.selected_action, family.selected_action)
            self.assertEqual(result.model_result_hash, family.content_hash)
            if kind == "intentional":
                self.assertEqual(result.actor_id, family.belief_state.agent_id)
                self.assertEqual(result.ledger_hash, family.belief_state.ledger_hash)
            else:
                self.assertEqual(result.actor_id, family.actor_id)
                self.assertEqual(result.ledger_hash, family.ledger_hash)


def test_dispatch_result_forgery_rejects_common_and_nested_binding_mismatch(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    model = make_reactive_model()
    nested = run_runtime_reactive_decision(
        story,
        domain,
        "d-a1-phase",
        ledger,
        model,
    )
    wrapper = RuntimeDecisionModelSpec("reactive", model)
    valid = dict(
        model_kind="reactive",
        decision_model_hash=wrapper.content_hash,
        model_id=nested.model_id,
        model_hash=nested.model_hash,
        decision_id=nested.decision_id,
        actor_id=nested.actor_id,
        step_index=nested.step_index,
        ledger_hash=nested.ledger_hash,
        action_policy=nested.action_policy,
        selected_action=nested.selected_action,
        model_result_hash=nested.content_hash,
        model_result=nested,
    )
    RuntimeDecisionDispatchResult(**valid)
    for changes in (
        {"model_kind": "planning"},
        {"model_id": "forged-model"},
        {"decision_id": "forged-decision"},
        {"actor_id": "forged-actor"},
        {"step_index": nested.step_index + 1},
        {"ledger_hash": "sha256:" + "1" * 64},
        {"action_policy": {"a1-active": 1.0, "a1-ready": 0.0}},
        {"selected_action": "a1-ready"},
        {"model_result_hash": "sha256:" + "2" * 64},
    ):
        with self.subTest(changes=changes):
            with self.assertRaises((TypeError, ValueError)):
                RuntimeDecisionDispatchResult(**{**valid, **changes})
```

- [ ] **Step 4: Add preflight, failure-cause, and replay tests**

```python
def test_wrapper_forgery_and_unsupported_decision_type_reject_before_family_hook(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    hook = CueScoreHook()
    model = make_reactive_model(hook=hook)
    wrapper = RuntimeDecisionModelSpec("reactive", model)
    forged = _forge(wrapper, model_kind="planning")
    with self.assertRaises(RuntimeDecisionDispatchError):
        run_runtime_decision(story, domain, "d-a1-phase", ledger, forged)
    self.assertEqual(hook.calls, [])

    unsupported_model = replace(
        model,
        supported_decision_types=("other-choice",),
    )
    unsupported = RuntimeDecisionModelSpec("reactive", unsupported_model)
    with self.assertRaises(RuntimeDecisionDispatchError):
        run_runtime_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            unsupported,
        )
    self.assertEqual(hook.calls, [])


def test_family_failures_are_typed_and_preserve_exact_cause(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    model = make_reactive_model(hook=RaisingScoreHook())
    with self.assertRaises(RuntimeDecisionDispatchError) as caught:
        run_runtime_decision(
            story,
            domain,
            "d-a1-phase",
            ledger,
            RuntimeDecisionModelSpec("reactive", model),
        )
    self.assertIsInstance(
        caught.exception.__cause__,
        RuntimeReactiveDecisionResolutionError,
    )
    self.assertIsInstance(caught.exception.__cause__.__cause__, RuntimeError)


def test_fixed_inputs_replay_to_exact_dispatch_result_and_hash(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    first_model = RuntimeDecisionModelSpec("planning", make_planning_model())
    second_model = RuntimeDecisionModelSpec("planning", make_planning_model())
    self.assertEqual(first_model.content_hash, second_model.content_hash)
    first = run_runtime_decision(
        story,
        domain,
        "d-a1-phase",
        ledger,
        first_model,
    )
    second = run_runtime_decision(
        story,
        domain,
        "d-a1-phase",
        ledger,
        second_model,
    )
    self.assertEqual(first, second)
    self.assertEqual(first.to_dict(), second.to_dict())
    self.assertEqual(first.content_hash, second.content_hash)
```

- [ ] **Step 5: Run the new file and verify the missing-module RED**

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: exactly seven test methods fail through `require_dispatch()` because `narrative_dynamics.narrative.runtime_decision_dispatch` does not yet exist. There must be no syntax error, test discovery error, missing existing fixture import, or family-runner failure.

- [ ] **Step 6: Commit the test-only RED**

```bash
git add tests/test_narrative_runtime_decision_dispatch.py
git commit -m "test: define runtime decision dispatch contract"
```

- [ ] **Step 7: Obtain authoritative exact-head RED CI**

Push the branch and open/update a Draft PR targeting `proof/narrative-dynamics-v0`.

Accept the RED only when GitHub Actions shows:

```text
workflow = proof
event = pull_request
head_sha = exact Task 1 test-only commit
status = completed
conclusion = failure
Lean dependency/conformance/build/theorem gates = success
Python failures = the seven new dispatch tests only
```

Record the exact SHA and proof run number in the PR body before production code is added.

---

### Task 2: Implement canonical dispatch wrapper and common result records

**Files:**
- Create: `narrative_dynamics/narrative/runtime_decision_dispatch.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Produces:
  - `RuntimeDecisionModelSpec(model_kind, model)`
  - `RuntimeDecisionDispatchResult(...)`
  - `RuntimeDecisionDispatchError`
  - final-signature `run_runtime_decision(...)` stub, intentionally unresolved until Task 3
- Consumes the three existing family model/result types without modifying them.

- [ ] **Step 1: Add imports, exact family tables, and generic validators**

```python
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
```

Add these validators with existing narrative conventions:

```python
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
            raise ValueError(f"{label} probabilities must be finite and non-negative")
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
```

- [ ] **Step 2: Implement the typed error and closed model wrapper**

```python
class RuntimeDecisionDispatchError(ValueError):
    """A typed runtime decision family could not be dispatched safely."""


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
```

There is no `intentional_model` compatibility alias and no model-supplied runner.

- [ ] **Step 3: Implement family-neutral actor and ledger extraction**

```python
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
```

These helpers may only read already-certified family results.

- [ ] **Step 4: Implement `RuntimeDecisionDispatchResult` self-validation**

```python
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
```

`__post_init__` must perform these exact checks:

```python
kind = _text(self.model_kind, label="runtime dispatch result model kind")
expected = _RESULT_TYPES.get(kind)
if expected is None:
    raise ValueError("runtime dispatch result model kind is unsupported")
if not isinstance(self.model_result, expected):
    raise TypeError("runtime dispatch result kind does not match nested result type")
object.__setattr__(self, "model_kind", kind)
object.__setattr__(
    self,
    "decision_model_hash",
    _hash(self.decision_model_hash, label="runtime dispatch decision model hash"),
)
object.__setattr__(self, "model_id", _text(self.model_id, label="runtime dispatch model id"))
object.__setattr__(self, "model_hash", _hash(self.model_hash, label="runtime dispatch model hash"))
object.__setattr__(self, "decision_id", _text(self.decision_id, label="runtime dispatch decision id"))
object.__setattr__(self, "actor_id", _text(self.actor_id, label="runtime dispatch actor id"))
object.__setattr__(self, "step_index", _step(self.step_index, label="runtime dispatch step"))
object.__setattr__(self, "ledger_hash", _hash(self.ledger_hash, label="runtime dispatch ledger hash"))
object.__setattr__(
    self,
    "model_result_hash",
    _hash(self.model_result_hash, label="runtime dispatch nested result hash"),
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
    raise ValueError("runtime dispatch nested result hash must match nested result")
policy = _freeze_policy(self.action_policy, label="runtime dispatch action policy")
if dict(policy) != dict(self.model_result.action_policy):
    raise ValueError("runtime dispatch policy must match nested result")
selected = _text(self.selected_action, label="runtime dispatch selected action")
if selected != self.model_result.selected_action:
    raise ValueError("runtime dispatch selected action must match nested result")
if selected != _map_choice(policy):
    raise ValueError("runtime dispatch selected action must be lexical MAP")
object.__setattr__(self, "action_policy", policy)
object.__setattr__(self, "selected_action", selected)
```

Use this exact serializable identity payload:

```python
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
```

- [ ] **Step 5: Add the final-signature runner stub and exact module surface**

```python
def run_runtime_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeDecisionModelSpec,
) -> RuntimeDecisionDispatchResult:
    raise RuntimeDecisionDispatchError(
        "runtime decision dispatch execution is not implemented"
    )


__all__ = (
    "RuntimeDecisionModelSpec",
    "RuntimeDecisionDispatchResult",
    "RuntimeDecisionDispatchError",
    "run_runtime_decision",
)
```

- [ ] **Step 6: Verify Task 2 GREEN is limited to records/model behavior**

```bash
python3 -m unittest \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_model_wrapper_is_closed_typed_and_binds_nested_and_dispatch_identity \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_public_signature_and_module_isolation_exclude_world_scheduler_projection_rng \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_dispatch_result_forgery_rejects_common_and_nested_binding_mismatch \
  -v
```

Expected: PASS.

Then run:

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: wrapper/result/signature/isolation tests PASS; tests that invoke `run_runtime_decision()` remain RED only because the typed stub is still present.

- [ ] **Step 7: Commit Task 2**

```bash
git add narrative_dynamics/narrative/runtime_decision_dispatch.py
git commit -m "feat: add runtime decision dispatch records"
```

Reviewer gate: this commit creates only the dispatch sidecar; no scheduler or family algorithm changes are allowed.

---

### Task 3: Implement closed three-family dispatch execution

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_decision_dispatch.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes `RuntimeDecisionModelSpec` and the three existing family runners.
- Produces a fully model-bound `RuntimeDecisionDispatchResult` through `run_runtime_decision(...)`.

- [ ] **Step 1: Add canonical wrapper reconstruction and authored-decision preflight**

```python
def _validated_model(model: object) -> RuntimeDecisionModelSpec:
    if not isinstance(model, RuntimeDecisionModelSpec):
        raise TypeError("runtime decision dispatch requires RuntimeDecisionModelSpec")
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
```

This preflight must execute before any family hook.

- [ ] **Step 2: Add the only permitted family branch**

```python
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
```

- [ ] **Step 3: Construct the generic envelope and bind it back to the wrapper**

```python
def _dispatch_result(
    model: RuntimeDecisionModelSpec,
    nested,
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
        raise ValueError("runtime dispatch result nested hash must match wrapper")
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
```

- [ ] **Step 4: Replace the stub with the final runner and exact typed error normalization**

```python
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
```

Do not catch `BaseException` or a blanket `Exception`.

- [ ] **Step 5: Verify dispatch GREEN and unchanged family suites**

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: all seven dispatch tests PASS.

```bash
python3 -m unittest \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add narrative_dynamics/narrative/runtime_decision_dispatch.py
git commit -m "feat: dispatch runtime decision families"
```

Reviewer gate: `runtime_decision_dispatch.py` still has no world/simulation/projection/RNG import and contains no fourth extensibility path.

---

### Task 4: Establish Scheduler V2 test-only RED, including three-family agents

**Files:**
- Modify: `tests/test_narrative_simulation.py`

**Interfaces:**
- Consumes Task 3 dispatch API.
- Defines the scheduler V2 contract before production migration:
  - `RuntimeAgentSpec.decision_model`
  - `SimulationAgentStep.decision_result: RuntimeDecisionDispatchResult`
  - same-prior-ledger dispatch
  - heterogeneous reactive/intentional/planning execution
  - two-round deterministic causal closure

- [ ] **Step 1: Add generic dispatch and nested-family test imports**

In the guarded runtime import section add:

```python
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionDispatchError,
    RuntimeDecisionDispatchResult,
    RuntimeDecisionModelSpec,
)
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
)
from narrative_dynamics.narrative.runtime_planning import (
    PlanningHiddenState,
    PlanningObservation,
    RuntimePlanningDecisionModelSpec,
    RuntimePlanningDecisionResult,
)
from narrative_dynamics.narrative.runtime_reactive import (
    RuntimeReactiveDecisionModelSpec,
    RuntimeReactiveDecisionResult,
)
```

At module scope also import the existing planning helper hooks:

```python
from tests.test_narrative_runtime_planning import (
    IdentityTransitionHook,
    NoInformationObservationHook,
    ProductCouplingHook,
)
```

- [ ] **Step 2: Migrate existing scheduler fixtures to explicit intentional wrappers**

Change `make_simulation_model()` agent construction to:

```python
agents = (
    RuntimeAgentSpec(
        "a1",
        "d-a1-scheduler",
        RuntimeDecisionModelSpec("intentional", a_model),
    ),
    RuntimeAgentSpec(
        "a2",
        "d-a2-scheduler",
        RuntimeDecisionModelSpec("intentional", b_model),
    ),
)
```

Change `make_conflict_case()` agent construction in the same way.

Where tests need the nested intentional model, access:

```python
nested = model.agents[1].decision_model.model
self.assertIsInstance(nested, RuntimeIntentionalDecisionModelSpec)
```

For the unsupported decision-type test construct:

```python
nested = model.agents[1].decision_model.model
assert isinstance(nested, RuntimeIntentionalDecisionModelSpec)
unsupported_nested = replace(
    nested,
    supported_decision_types=("scheduler-alert-choice",),
)
unsupported_model = replace(
    model,
    agents=(
        model.agents[0],
        replace(
            model.agents[1],
            decision_model=RuntimeDecisionModelSpec(
                "intentional",
                unsupported_nested,
            ),
        ),
    ),
)
```

- [ ] **Step 3: Migrate common scheduler assertions away from intentional internals**

In `test_static_round_runs_every_agent_once_against_same_prior_ledger_and_builds_exact_intents`, use:

```python
for agent_step in result.agent_steps:
    self.assertIsInstance(
        agent_step.decision_result,
        RuntimeDecisionDispatchResult,
    )
    self.assertEqual(
        agent_step.decision_result.ledger_hash,
        prior.evidence_ledger.content_hash,
    )
    self.assertEqual(
        agent_step.decision_result.step_index,
        prior.step_index,
    )
    self.assertEqual(
        agent_step.action_intent.decision_id,
        agent_step.decision_result.decision_id,
    )
    self.assertEqual(
        agent_step.action_intent.selected_action,
        agent_step.decision_result.selected_action,
    )
    self.assertEqual(
        agent_step.action_intent.selection_result_hash,
        agent_step.decision_result.content_hash,
    )
```

In `test_two_round_action_world_percept_belief_action_causal_closure_without_story_mutation`, unwrap the intentional result explicitly before reading belief state:

```python
nested = step1["a2"].decision_result.model_result
self.assertIsInstance(nested, RuntimeIntentionalDecisionResult)
round1_belief = nested.belief_state
```

In `test_cognition_failure_blocks_all_world_projection_and_next_state`, require the two-level cause chain:

```python
with self.assertRaises(SimulationStepError) as caught:
    simulate_step(story, domain, prior, model)
self.assertIsInstance(caught.exception.__cause__, RuntimeDecisionDispatchError)
self.assertIsInstance(
    caught.exception.__cause__.__cause__,
    RuntimeIntentionalDecisionResolutionError,
)
```

Keep the existing assertions that world hooks and projection hooks have zero calls after decision failure.

- [ ] **Step 4: Add test-only heterogeneous hooks and a third authored decision**

```python
class SchedulerReactiveScoreHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        cue = context.cues[alert_cell()]
        if (
            cue.status == "resolved"
            and cue.value == TypedValue("AlertState", True)
        ):
            return {"a2-respond": 3.0, "a2-wait": 0.0}
        return {"a2-respond": 0.0, "a2-wait": 3.0}


class SchedulerPlanningRewardHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        alert = context.state.cells[alert_cell()].value
        if context.action.id == "a3-respond":
            return 2.0 if alert is True else 0.0
        if context.action.id == "a3-wait":
            return 2.0 if alert is False else 0.0
        raise AssertionError("unexpected scheduler planning action")


def make_a3_decision():
    return Decision(
        "d-a3-scheduler",
        10,
        "a3",
        "scheduler-response-choice",
        (alert_cell(),),
        (
            ActionOption(
                "a3-respond",
                "response-action",
                {"respond": TypedValue("AlertState", True)},
            ),
            ActionOption(
                "a3-wait",
                "response-action",
                {"respond": TypedValue("AlertState", False)},
            ),
        ),
    )
```

- [ ] **Step 5: Add the three-family scheduler fixture**

```python
def make_heterogeneous_case():
    domain = make_scheduler_domain()
    story = make_scheduler_story(domain)
    story = replace(
        story,
        decisions=story.decisions + (make_a3_decision(),),
    )
    a_intentional, _ = make_intentional_models()
    b_reactive = RuntimeReactiveDecisionModelSpec(
        "b-runtime-reactive",
        "1",
        ("scheduler-response-choice",),
        (alert_cell(),),
        {},
        4.0,
        SchedulerReactiveScoreHook(),
    )
    c_planning = RuntimePlanningDecisionModelSpec(
        "c-runtime-planning",
        "1",
        ("scheduler-response-choice",),
        (alert_cell(),),
        (),
        make_runtime_belief_model(),
        (
            PlanningHiddenState(
                "alert-off",
                {alert_cell(): TypedValue("AlertState", False)},
            ),
            PlanningHiddenState(
                "alert-on",
                {alert_cell(): TypedValue("AlertState", True)},
            ),
        ),
        (PlanningObservation("none", {}),),
        (("a3-respond", "a3-wait"),),
        1.0,
        4.0,
        {},
        ProductCouplingHook(),
        IdentityTransitionHook(),
        NoInformationObservationHook(),
        SchedulerPlanningRewardHook(),
    )
    model = SimulationModelSpec(
        "heterogeneous-scheduler-model",
        "2",
        domain.domain_id,
        domain.version,
        domain.content_hash,
        (
            RuntimeAgentSpec(
                "a1",
                "d-a1-scheduler",
                RuntimeDecisionModelSpec("intentional", a_intentional),
            ),
            RuntimeAgentSpec(
                "a2",
                "d-a2-scheduler",
                RuntimeDecisionModelSpec("reactive", b_reactive),
            ),
            RuntimeAgentSpec(
                "a3",
                "d-a3-scheduler",
                RuntimeDecisionModelSpec("planning", c_planning),
            ),
        ),
        make_world_model(domain),
        make_projection_model(domain),
    )
    return domain, story, model
```

- [ ] **Step 6: Add the single-round three-family test**

```python
def test_heterogeneous_round_dispatches_three_families_against_one_prior_ledger(self):
    self.require_simulation()
    domain, story, model = make_heterogeneous_case()
    prior = simulation_state_from_story(story, domain, model, at_time=10)
    result = simulate_step(story, domain, prior, model)
    by_agent = {item.agent_id: item for item in result.agent_steps}
    self.assertEqual(tuple(sorted(by_agent)), ("a1", "a2", "a3"))
    self.assertEqual(
        {
            agent: step.decision_result.model_kind
            for agent, step in by_agent.items()
        },
        {"a1": "intentional", "a2": "reactive", "a3": "planning"},
    )
    self.assertIsInstance(
        by_agent["a1"].decision_result.model_result,
        RuntimeIntentionalDecisionResult,
    )
    self.assertIsInstance(
        by_agent["a2"].decision_result.model_result,
        RuntimeReactiveDecisionResult,
    )
    self.assertIsInstance(
        by_agent["a3"].decision_result.model_result,
        RuntimePlanningDecisionResult,
    )
    model_by_agent = {item.agent_id: item for item in model.agents}
    for agent_id, step in by_agent.items():
        self.assertEqual(
            step.decision_result.ledger_hash,
            prior.evidence_ledger.content_hash,
        )
        self.assertEqual(step.decision_result.step_index, prior.step_index)
        self.assertEqual(
            step.decision_result.decision_model_hash,
            model_by_agent[agent_id].decision_model.content_hash,
        )
        self.assertEqual(
            step.action_intent.selection_result_hash,
            step.decision_result.content_hash,
        )
    self.assertEqual(len(result.world_step.transitions), 3)
```

- [ ] **Step 7: Add the two-round heterogeneous causal-closure/replay test**

```python
def test_two_round_heterogeneous_replay_is_exact_and_uses_newly_admitted_information(self):
    self.require_simulation()
    domain1, story1, model1 = make_heterogeneous_case()
    domain2, story2, model2 = make_heterogeneous_case()
    self.assertEqual(model1.content_hash, model2.content_hash)
    initial1 = simulation_state_from_story(story1, domain1, model1, at_time=10)
    initial2 = simulation_state_from_story(story2, domain2, model2, at_time=10)
    first = simulate_trajectory(
        story1,
        domain1,
        initial1,
        model1,
        rounds=2,
    )
    second = simulate_trajectory(
        story2,
        domain2,
        initial2,
        model2,
        rounds=2,
    )
    self.assertEqual(first.to_dict(), second.to_dict())
    self.assertEqual(first.content_hash, second.content_hash)

    round0 = {item.agent_id: item for item in first.steps[0].agent_steps}
    round1 = {item.agent_id: item for item in first.steps[1].agent_steps}
    self.assertEqual(round0["a2"].decision_result.selected_action, "a2-wait")
    self.assertEqual(round1["a2"].decision_result.selected_action, "a2-respond")
    self.assertEqual(
        first.steps[0].next_state.world_state.values[alert_cell()],
        TypedValue("AlertState", True),
    )
    planning = round1["a3"].decision_result.model_result
    self.assertIsInstance(planning, RuntimePlanningDecisionResult)
    alert_view = planning.belief_state.cells[alert_cell()]
    self.assertGreater(
        alert_view.posterior.probability_of(TypedValue("AlertState", True)),
        alert_view.posterior.probability_of(TypedValue("AlertState", False)),
    )
    self.assertEqual(round1["a3"].decision_result.selected_action, "a3-respond")
```

- [ ] **Step 8: Verify Scheduler V2 RED before production migration**

```bash
python3 -m unittest tests.test_narrative_simulation -v
```

Expected: failures are restricted to the scheduler production contract still exposing `intentional_model`, requiring `RuntimeIntentionalDecisionResult`, and directly invoking the intentional runner. The new dispatch module from Task 3 must remain GREEN, and heterogeneous fixtures must construct successfully.

- [ ] **Step 9: Commit Scheduler V2 test-only RED and obtain exact-head RED CI**

```bash
git add tests/test_narrative_simulation.py
git commit -m "test: define multi-model scheduler v2 contract"
```

Accept the PR-triggered RED only when non-scheduler tests remain green and the failure set matches the V1/V2 scheduler mismatch.

---

### Task 5: Migrate the scheduler to generic decision dispatch

**Files:**
- Modify: `narrative_dynamics/narrative/simulation.py`
- Test: `tests/test_narrative_simulation.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes `RuntimeDecisionModelSpec`, `RuntimeDecisionDispatchResult`, `RuntimeDecisionDispatchError`, and `run_runtime_decision`.
- Preserves existing public simulation record names while replacing their intentional-specific internal boundary.

- [ ] **Step 1: Replace intentional-specific imports with the generic dispatch import**

Remove direct runtime-intention model/result/runner/error imports from `simulation.py` and add:

```python
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionDispatchError,
    RuntimeDecisionDispatchResult,
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)
```

`simulation.py` must not directly import reactive, intentional, or planning family modules after this change.

- [ ] **Step 2: Replace `RuntimeAgentSpec.intentional_model` with `decision_model`**

```python
@dataclass(frozen=True)
class RuntimeAgentSpec:
    agent_id: str
    decision_template_id: str
    decision_model: RuntimeDecisionModelSpec

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "agent_id",
            _text(self.agent_id, label="runtime agent id"),
        )
        object.__setattr__(
            self,
            "decision_template_id",
            _text(
                self.decision_template_id,
                label="runtime agent decision template id",
            ),
        )
        if not isinstance(self.decision_model, RuntimeDecisionModelSpec):
            raise TypeError(
                "runtime agent decision model must be RuntimeDecisionModelSpec"
            )
        validated = RuntimeDecisionModelSpec(
            self.decision_model.model_kind,
            self.decision_model.model,
        )
        if validated.to_dict() != self.decision_model.to_dict():
            raise ValueError("runtime agent decision model must be canonical")
        object.__setattr__(self, "decision_model", validated)

    def to_dict(self) -> dict[str, object]:
        return {
            "agent_id": self.agent_id,
            "decision_template_id": self.decision_template_id,
            "decision_model_hash": self.decision_model.content_hash,
        }
```

Do not add a compatibility alias.

- [ ] **Step 3: Make `_validate_execution_bindings()` family-neutral**

Replace the intentional field access with:

```python
if decision.type_name not in agent.decision_model.supported_decision_types:
    raise ValueError(
        "simulation runtime model does not support decision template type"
    )
```

The scheduler must not inspect cue cells, goals, hidden states, or planning horizon.

- [ ] **Step 4: Change `SimulationAgentStep` to the common dispatch result**

Use:

```python
@dataclass(frozen=True)
class SimulationAgentStep:
    agent_id: str
    decision_template_id: str
    decision_result: RuntimeDecisionDispatchResult
    action_intent: ActionIntent
```

Its constructor must require:

```python
if not isinstance(self.decision_result, RuntimeDecisionDispatchResult):
    raise TypeError(
        "simulation agent step decision_result must be RuntimeDecisionDispatchResult"
    )
if self.agent_id != self.decision_result.actor_id:
    raise ValueError("simulation agent step agent must match decision result actor")
if self.decision_template_id != self.decision_result.decision_id:
    raise ValueError("simulation agent step template must match decision result")
if self.action_intent.decision_id != self.decision_result.decision_id:
    raise ValueError("simulation action intent decision must match decision result")
if self.action_intent.selected_action != self.decision_result.selected_action:
    raise ValueError("simulation action intent action must match decision result")
if self.action_intent.selection_model_id != self.decision_result.model_id:
    raise ValueError("simulation action intent model must match decision result")
if self.action_intent.selection_result_hash != self.decision_result.content_hash:
    raise ValueError("simulation action intent result hash must bind decision result")
```

- [ ] **Step 5: Replace the direct intentional call in `simulate_step()`**

For each canonical agent:

```python
try:
    result = run_runtime_decision(
        story,
        domain,
        agent.decision_template_id,
        prior_state.evidence_ledger,
        agent.decision_model,
    )
except RuntimeDecisionDispatchError as error:
    raise SimulationStepError(
        f"simulation decision failed for agent {agent.agent_id}"
    ) from error
```

Before constructing `ActionIntent`, require:

```python
if result.actor_id != agent.agent_id:
    raise SimulationStepError(
        "runtime decision actor does not match scheduled agent"
    )
if result.step_index != prior_state.step_index:
    raise SimulationStepError(
        "runtime decision step does not match simulation prior"
    )
if result.ledger_hash != prior_state.evidence_ledger.content_hash:
    raise SimulationStepError(
        "runtime decision does not bind shared prior ledger"
    )
if result.decision_model_hash != agent.decision_model.content_hash:
    raise SimulationStepError(
        "runtime decision does not bind scheduled decision model"
    )
```

Then construct:

```python
intent = ActionIntent(
    decision_id=result.decision_id,
    selected_action=result.selected_action,
    selection_model_id=result.model_id,
    selection_result_hash=result.content_hash,
)
```

- [ ] **Step 6: Preserve the world/projection/admission call order exactly**

The code after all `agent_steps` are built must still call:

```python
world_step = advance_world_step(
    story,
    domain,
    prior_state.world_state,
    model.world_model,
    tuple(item.action_intent for item in agent_steps),
)

admission = admit_world_percepts(
    story,
    domain,
    world_step,
    model.observation_model,
    prior_state.evidence_ledger,
)
```

Do not expose same-round selected actions to another agent through the evidence ledger.

- [ ] **Step 7: Run dispatch, scheduler, and unchanged family suites**

```bash
python3 -m unittest \
  tests.test_narrative_runtime_decision_dispatch \
  tests.test_narrative_simulation \
  -v
```

Expected: PASS, including all-intentional behavioral expectations, three-family single-round execution, two-round heterogeneous replay, shared-prior-ledger assertions, and existing atomic failure tests.

```bash
python3 -m unittest \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit Scheduler V2 GREEN and obtain exact-head CI**

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: schedule typed runtime decision families"
```

Reviewer gate:

```text
simulation.py imports generic dispatch, not family decision modules
RuntimeAgentSpec has decision_model and no intentional_model alias
SimulationAgentStep binds RuntimeDecisionDispatchResult
all agent decisions finish before world transition
ActionIntent points to generic dispatch result hash
world/projection/admission semantics are unchanged
```

Do not proceed to public exports until the exact-head scheduler CI shows the scheduler/dispatch suites GREEN.

---

### Task 6: Lock the exact narrative public surface with a test-only RED

**Files:**
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Adds exactly four narrative-package public names.
- Root package remains unchanged.

- [ ] **Step 1: Extend `_EXPECTED_PUBLIC_API` by exactly four names**

Add between the planning and simulation sections:

```python
# Runtime generic decision dispatch.
"RuntimeDecisionModelSpec",
"RuntimeDecisionDispatchResult",
"RuntimeDecisionDispatchError",
"run_runtime_decision",
```

- [ ] **Step 2: Verify the exact four-name public-surface RED**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected: FAIL because exactly these four names are missing from `narrative_dynamics.narrative.__all__`. No root-isolation failure should occur because the root package has not changed.

- [ ] **Step 3: Commit the public-surface test-only RED and obtain exact-head CI**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: lock narrative decision dispatch surface"
```

Accept the RED only if the fresh Python log shows the trust-surface test as the only newly failing contract.

---

### Task 7: Export the dispatch surface and obtain final exact-head GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`
- Test: `tests/test_narrative_simulation.py`

**Interfaces:**
- Final narrative public additions:
  - `RuntimeDecisionModelSpec`
  - `RuntimeDecisionDispatchResult`
  - `RuntimeDecisionDispatchError`
  - `run_runtime_decision`

- [ ] **Step 1: Import exactly the four dispatch names into the narrative package**

```python
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionDispatchError,
    RuntimeDecisionDispatchResult,
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)
```

Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 2: Extend narrative `__all__` by exactly four names**

```python
"RuntimeDecisionModelSpec",
"RuntimeDecisionDispatchResult",
"RuntimeDecisionDispatchError",
"run_runtime_decision",
```

Do not export `_MODEL_TYPES`, `_RESULT_TYPES`, validators, or extraction helpers.

- [ ] **Step 3: Run targeted trust/dispatch/scheduler suites**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api \
  tests.test_narrative_runtime_decision_dispatch \
  tests.test_narrative_simulation \
  -v
```

Expected: PASS.

- [ ] **Step 4: Run the full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: zero failures and zero errors. Record the fresh `Ran N tests` count; do not reuse the previous 579-test count unless the fresh output still says 579 after the new tests are included.

- [ ] **Step 5: Verify final diff scope against the integrated base**

Only these eight paths may differ from the integrated base:

```text
docs/superpowers/specs/2026-08-27-narrative-generic-decision-dispatch-v2-design.md
docs/superpowers/plans/2026-08-27-narrative-generic-decision-dispatch-v2.md
narrative_dynamics/narrative/runtime_decision_dispatch.py
narrative_dynamics/narrative/simulation.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_decision_dispatch.py
tests/test_narrative_simulation.py
tests/test_narrative_trust_api.py
```

If any family algorithm, cognition/world/projection, root package, model-comparison/prison, or Lean path appears, stop before final CI and review the scope.

- [ ] **Step 6: Commit the public export GREEN**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export runtime decision dispatch surface"
```

- [ ] **Step 7: Obtain authoritative final exact-head PR CI**

The accepted proof must satisfy:

```text
workflow = proof
event = pull_request
head_sha = exact final feature head
status = completed
conclusion = success
Resolve Lean dependencies = success
Lean Python conformance vectors = success
Build Lean library = success
Lean theorem tests = success
Python numerical tests = success
Narrative story theorem tests = success
Narrative testimony theorem tests = success
```

Read the Python job log and record the fresh `Ran N tests` / `OK` line.

- [ ] **Step 8: Perform final review gates and update the PR body**

Verify:

```text
PR head SHA = exact final GREEN head
PR base = proof/narrative-dynamics-v0
review submissions = no unresolved request-changes blocker
inline threads = none unresolved
conversation comments = no unresolved blocker
changed paths = exactly approved scope
```

Update the PR body with:

```text
authoritative Task 1 RED run + exact SHA
dispatch records/runner GREEN commits
scheduler V2 RED/GREEN commits
three-family single-round evidence
two-round heterogeneous replay evidence
final exact-head proof run + fresh test count
explicit statement that pre-V2 simulation hashes are not compatibility targets
```

Mark the PR ready only after every gate is satisfied. Do not merge automatically; integration remains the user's finishing decision.

---

## Completion Checklist

Before any completion claim, verify each item with a test, source check, diff check, or exact-head CI artifact:

- [ ] Model kinds are exactly reactive / intentional / planning.
- [ ] Tag/type mismatch fails before family hooks.
- [ ] Wrapper identity binds nested model hash, wrapper implementation identity, and generic runner implementation identity.
- [ ] Common result contains only genuine shared fields and the exact typed nested family result.
- [ ] Intentional actor and ledger are derived from `RuntimeIntentionalDecisionResult.belief_state`; the intentional result type itself is unchanged.
- [ ] Generic runner signature is exactly `(story, domain, decision_id, ledger, model)`.
- [ ] Dispatcher has no world/simulation/projection/RNG capability.
- [ ] Family failures become `RuntimeDecisionDispatchError` with exact cause preserved.
- [ ] `RuntimeAgentSpec` has one `decision_model` field and no `intentional_model` alias.
- [ ] `SimulationAgentStep` binds `RuntimeDecisionDispatchResult`.
- [ ] `ActionIntent.selection_result_hash` points to the generic dispatch result.
- [ ] Every scheduled agent reads the exact same prior ledger in one round.
- [ ] Reactive + intentional + planning agents execute in one atomic round.
- [ ] Two-round heterogeneous execution consumes newly admitted evidence and replays exactly.
- [ ] Existing overlapping-write conflict remains typed/atomic and still blocks projection.
- [ ] All-intentional selected actions/world/ledger behavior remains unchanged even though V2 content hashes may change.
- [ ] Narrative package exports exactly four new dispatch names.
- [ ] Root package exports none of the new dispatch names.
- [ ] Final diff contains only the eight approved paths.
- [ ] Final exact-head PR `proof` is completed/success with fresh full-suite evidence.
