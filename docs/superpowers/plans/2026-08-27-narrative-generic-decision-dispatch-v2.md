# Generic Runtime Decision Dispatch / Multi-Model Scheduler V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow one deterministic multi-agent narrative simulation to schedule reactive, intentional, and planning agents in the same round through one closed typed runtime decision-dispatch boundary.

**Architecture:** Add an isolated `runtime_decision_dispatch.py` sidecar that owns the only three-family branch and returns a canonical common envelope containing the exact family-specific result. Migrate `simulation.py` from intentional-specific model/result fields to the generic wrapper/envelope while leaving world transition, observation projection, percept admission, and family-specific algorithms unchanged.

**Tech Stack:** Python 3 standard library dataclasses/typing, existing narrative runtime contracts, existing implementation attestation, existing stable content hashing, `unittest`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-generic-decision-dispatch-v2-design.md`

## Global Constraints

- Base branch is `proof/narrative-dynamics-v0` at integrated commit `8317339cde03d46e0500798ebb30dfa27ca43e38` unless the base moves before execution; if it moves, record the new exact base before the first test commit.
- Feature branch is `work/narrative-generic-decision-dispatch-v2`.
- Preserve strict RED -> GREEN -> atomic commit discipline.
- Pure files under `docs/superpowers/specs/**` and `docs/superpowers/plans/**` must not trigger `proof`; test or production changes must trigger normal CI.
- The generic dispatcher supports exactly three model kinds: `reactive`, `intentional`, `planning`.
- Do not accept arbitrary protocol, callback, duck-typed, or plugin decision models.
- Do not change family-specific decision algorithms in `runtime_reactive.py`, `runtime_intention.py`, or `runtime_planning.py`.
- Do not import or expose objective world state, scheduler state, observation projection, RNG, or explicit step inputs through `runtime_decision_dispatch.py`.
- Keep all agents in a simulation round bound to the same immutable prior evidence ledger before any world transition executes.
- Keep `advance_world_step()` atomic and preserve current overlapping-write conflict rejection; do not add actor priority or last-writer-wins behavior.
- Do not add stochastic world/observation semantics.
- Do not change `GenericNarrative`, `DomainSpec`, `world.py`, `observation_projection.py`, `runtime_cognition.py`, model-comparison infrastructure, prison adapters, or Lean sources.
- V2 preserves all-intentional behavioral semantics, not pre-V2 simulation/model/step/trajectory content hashes.
- `ActionIntent.selection_result_hash` must bind the generic dispatch result, and the generic result must bind the exact nested family result.
- Package-root `narrative_dynamics` remains isolated from the new narrative runtime surface.
- Do not mark the roadmap/integration complete until the final exact-head PR-triggered `proof` run is completed with `conclusion=success`.

---

## File Structure

### Create

- `narrative_dynamics/narrative/runtime_decision_dispatch.py`
  - Closed three-family model wrapper.
  - Canonical common dispatch result envelope.
  - Typed dispatch error.
  - Generic `run_runtime_decision(...)` runner.
  - No world/simulation/projection capability.

- `tests/test_narrative_runtime_decision_dispatch.py`
  - Wrapper identity and tag/type closure.
  - Common result binding and forgery rejection.
  - Direct dispatch equivalence to each family runner.
  - Failure-cause preservation, replay, public signature, and import isolation.

### Modify

- `narrative_dynamics/narrative/simulation.py`
  - `RuntimeAgentSpec.intentional_model` -> `decision_model`.
  - `SimulationAgentStep.decision_result` -> generic dispatch result.
  - `simulate_step()` -> generic runner.
  - Preserve world/projection/admission sequencing.

- `narrative_dynamics/narrative/__init__.py`
  - Export exactly the four new dispatch public names from the narrative package only.

- `tests/test_narrative_simulation.py`
  - Migrate intentional-only fixtures to explicit `RuntimeDecisionModelSpec("intentional", ...)` wrappers.
  - Add three-family heterogeneous single-round and multi-round fixtures.
  - Preserve existing atomicity, replay, and failure regression coverage.

- `tests/test_narrative_trust_api.py`
  - Extend exact narrative public surface by exactly four names.
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
- Lean sources

---

### Task 1: Establish the generic dispatch test-only RED boundary

**Files:**
- Create: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes existing family runners:
  - `run_runtime_reactive_decision(story, domain, decision_id, ledger, model)`
  - `run_runtime_intentional_decision(story, domain, decision_id, ledger, model)`
  - `run_runtime_planning_decision(story, domain, decision_id, ledger, model)`
- Produces the RED contract for future public names:
  - `RuntimeDecisionModelSpec`
  - `RuntimeDecisionDispatchResult`
  - `RuntimeDecisionDispatchError`
  - `run_runtime_decision`

- [ ] **Step 1: Create guarded imports and reusable three-family fixtures**

Create `tests/test_narrative_runtime_decision_dispatch.py` with these imports and helpers. Keep the new dispatch import guarded so the RED is a controlled missing-boundary failure rather than test discovery failure.

```python
from __future__ import annotations

from dataclasses import fields, replace
import inspect
import unittest

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.narrative.ir import TypedValue
from narrative_dynamics.narrative.runtime_cognition import make_runtime_belief_model as _missing_runtime_helper
```

Do **not** keep the last import above; it intentionally shows the wrong module ownership. Replace it immediately in the same edit with the existing test helper imports below, which are the canonical fixture locations:

```python
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionResolutionError,
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
```

The file must define:

```python
class NarrativeRuntimeDecisionDispatchTests(unittest.TestCase):
    def require_dispatch(self) -> None:
        if _DISPATCH_IMPORT_ERROR is not None:
            self.fail(
                "narrative runtime decision dispatch boundary is missing: "
                f"{_DISPATCH_IMPORT_ERROR}"
            )
```

- [ ] **Step 2: Add the closed wrapper/identity/public-signature tests**

Add exactly these tests to the class:

```python
def test_model_wrapper_is_closed_typed_and_binds_nested_and_dispatch_identity(self):
    self.require_dispatch()
    reactive = make_reactive_model()
    intentional = make_runtime_intentional_model()
    planning = make_planning_model()
    wrapped = (
        RuntimeDecisionModelSpec("reactive", reactive),
        RuntimeDecisionModelSpec("intentional", intentional),
        RuntimeDecisionModelSpec("planning", planning),
    )
    self.assertEqual(tuple(item.model_kind for item in wrapped), ("reactive", "intentional", "planning"))
    self.assertEqual(wrapped[0].model_id, reactive.model_id)
    self.assertEqual(wrapped[1].model_version, intentional.version)
    self.assertEqual(wrapped[2].nested_model_hash, planning.content_hash)
    self.assertEqual(wrapped[1].supported_decision_types, intentional.supported_decision_types)
    self.assertEqual(RuntimeDecisionModelSpec("reactive", make_reactive_model()).content_hash, wrapped[0].content_hash)
    self.assertNotEqual(wrapped[0].content_hash, wrapped[1].content_hash)
    self.assertNotEqual(wrapped[1].content_hash, wrapped[2].content_hash)
    payload = wrapped[0].to_dict()
    self.assertEqual(
        payload["wrapper_implementation_identity"],
        measure_implementation(RuntimeDecisionModelSpec).manifest_identity(),
    )
    self.assertEqual(
        payload["runner_implementation_identity"],
        measure_implementation(run_runtime_decision).manifest_identity(),
    )
    for kind, bad in (
        ("planning", reactive),
        ("reactive", intentional),
        ("intentional", planning),
        ("unknown", reactive),
    ):
        with self.subTest(kind=kind):
            with self.assertRaises((TypeError, ValueError)):
                RuntimeDecisionModelSpec(kind, bad)
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

- [ ] **Step 3: Add direct family dispatch and common-envelope binding tests**

Use one `empty_runtime_case()` for all three family runs so every generic result is bound to the same authored story/domain/ledger boundary.

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
            object,
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
            family = family_runner(story, domain, "d-a1-phase", ledger, nested_model)
            wrapped = RuntimeDecisionModelSpec(kind, nested_model)
            result = run_runtime_decision(story, domain, "d-a1-phase", ledger, wrapped)
            if result_type is not object:
                self.assertIsInstance(result.model_result, result_type)
            self.assertEqual(result.model_result, family)
            self.assertEqual(result.model_kind, kind)
            self.assertEqual(result.decision_model_hash, wrapped.content_hash)
            self.assertEqual(result.model_id, family.model_id)
            self.assertEqual(result.model_hash, family.model_hash)
            self.assertEqual(result.decision_id, family.decision_id)
            self.assertEqual(result.step_index, family.step_index)
            self.assertEqual(result.action_policy, family.action_policy)
            self.assertEqual(result.selected_action, family.selected_action)
            self.assertEqual(result.model_result_hash, family.content_hash)
            expected_actor = (
                family.belief_state.agent_id
                if kind == "intentional"
                else family.actor_id
            )
            expected_ledger = (
                family.belief_state.ledger_hash
                if kind == "intentional"
                else family.ledger_hash
            )
            self.assertEqual(result.actor_id, expected_actor)
            self.assertEqual(result.ledger_hash, expected_ledger)


def test_dispatch_result_forgery_rejects_common_and_nested_binding_mismatch(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    model = make_reactive_model()
    nested = run_runtime_reactive_decision(story, domain, "d-a1-phase", ledger, model)
    wrapped = RuntimeDecisionModelSpec("reactive", model)
    valid = dict(
        model_kind="reactive",
        decision_model_hash=wrapped.content_hash,
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

- [ ] **Step 4: Add runner preflight, error-cause, and deterministic replay tests**

```python
def test_wrapper_forgery_and_unsupported_decision_type_reject_before_family_hook(self):
    self.require_dispatch()
    domain, story, ledger = empty_runtime_case()
    hook = CueScoreHook()
    model = make_reactive_model(hook=hook)
    wrapped = RuntimeDecisionModelSpec("reactive", model)
    forged = _forge(wrapped, model_kind="planning")
    with self.assertRaises(RuntimeDecisionDispatchError):
        run_runtime_decision(story, domain, "d-a1-phase", ledger, forged)
    self.assertEqual(hook.calls, [])

    unsupported_model = replace(model, supported_decision_types=("other-choice",))
    unsupported = RuntimeDecisionModelSpec("reactive", unsupported_model)
    with self.assertRaises(RuntimeDecisionDispatchError):
        run_runtime_decision(story, domain, "d-a1-phase", ledger, unsupported)
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
    first = run_runtime_decision(story, domain, "d-a1-phase", ledger, first_model)
    second = run_runtime_decision(story, domain, "d-a1-phase", ledger, second_model)
    self.assertEqual(first, second)
    self.assertEqual(first.to_dict(), second.to_dict())
    self.assertEqual(first.content_hash, second.content_hash)
```

- [ ] **Step 5: Run the new file and verify the missing-module RED**

Run:

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: the seven new tests fail only through `require_dispatch()` because `narrative_dynamics.narrative.runtime_decision_dispatch` does not exist. There must be no syntax error, test discovery error, missing existing fixture import, or failure inside any existing family runner at this stage.

- [ ] **Step 6: Commit the test-only RED**

```bash
git add tests/test_narrative_runtime_decision_dispatch.py
git commit -m "test: define runtime decision dispatch contract"
```

- [ ] **Step 7: Obtain authoritative exact-head RED CI**

Push the branch and open/update a Draft PR targeting `proof/narrative-dynamics-v0` so the PR-triggered `proof` workflow runs on the exact test-only head.

Accept the RED only if:

```text
event = pull_request
head_sha = exact test-only commit
status = completed
conclusion = failure
Lean dependency/conformance/build/theorem gates = success
Python failure set = new dispatch tests only
```

Record the run number and exact SHA in the PR body before production code is added.

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
  - final-signature `run_runtime_decision(...)` stub that remains intentionally unresolved until Task 3
- Consumes the three existing family model/result types without changing them.

- [ ] **Step 1: Add imports, constants, validators, and exact family tables**

Start the module with only model-family/runtime contract imports, standard-library helpers, hashing, and attestation. Do not import world, simulation, observation projection, or RNG.

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

Add `_text`, `_hash`, `_step`, `_freeze_policy`, and `_map_choice` validators following existing narrative runtime conventions. `_freeze_policy` must reject bools, non-finite values, negative mass, empty mappings, and totals outside absolute tolerance `1e-12`, and must canonicalize keys lexically into `MappingProxyType`.

- [ ] **Step 2: Implement the typed dispatch error and wrapper**

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
            raise ValueError("runtime decision model kind must be reactive, intentional, or planning")
        if not isinstance(self.model, expected):
            raise TypeError("runtime decision model kind does not match nested model type")
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

Do not add `intentional_model`, callback, protocol, or arbitrary runner fields.

- [ ] **Step 3: Implement family-neutral common-field extraction**

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

These helpers must contain no objective-world lookup and no synthetic latent common state.

- [ ] **Step 4: Implement `RuntimeDecisionDispatchResult` constructor validation**

Use the exact public fields from the design:

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

Its `__post_init__` must reconstruct/validate every common binding that can be proved without an external wrapper:

```python
kind = _text(self.model_kind, label="runtime dispatch result model kind")
expected_type = _RESULT_TYPES.get(kind)
if expected_type is None:
    raise ValueError("runtime dispatch result model kind is unsupported")
if not isinstance(self.model_result, expected_type):
    raise TypeError("runtime dispatch result kind does not match nested result type")
if self.model_id != self.model_result.model_id:
    raise ValueError("runtime dispatch result model id must match nested result")
if self.model_hash != self.model_result.model_hash:
    raise ValueError("runtime dispatch result model hash must match nested result")
if self.decision_id != self.model_result.decision_id:
    raise ValueError("runtime dispatch result decision must match nested result")
if self.actor_id != _actor_id(kind, self.model_result):
    raise ValueError("runtime dispatch result actor must match nested result")
if self.step_index != self.model_result.step_index:
    raise ValueError("runtime dispatch result step must match nested result")
if self.ledger_hash != _ledger_hash(kind, self.model_result):
    raise ValueError("runtime dispatch result ledger must match nested result")
if self.model_result_hash != self.model_result.content_hash:
    raise ValueError("runtime dispatch result hash must match nested result")
```

Then independently freeze/validate `action_policy`, require exact equality with `model_result.action_policy`, normalize `selected_action`, require exact equality with the nested selected action, and require lexical MAP.

`decision_model_hash` is syntax-validated as a SHA-256 content hash here; Task 3 binds it to an external wrapper.

`to_dict()` must include the complete nested payload:

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
        "action_policy": {key: self.action_policy[key] for key in sorted(self.action_policy)},
        "selected_action": self.selected_action,
        "model_result_hash": self.model_result_hash,
        "model_result": self.model_result.to_dict(),
    }
```

- [ ] **Step 5: Add the final-signature typed runner stub and module-local surface**

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

The stub is temporary only inside this Task; it gives the wrapper identity a stable runner function to attest while keeping family execution RED for Task 3.

- [ ] **Step 6: Run the records/model subset**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_model_wrapper_is_closed_typed_and_binds_nested_and_dispatch_identity \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_public_signature_and_module_isolation_exclude_world_scheduler_projection_rng \
  tests.test_narrative_runtime_decision_dispatch.NarrativeRuntimeDecisionDispatchTests.test_dispatch_result_forgery_rejects_common_and_nested_binding_mismatch \
  -v
```

Expected: PASS.

Run the entire new file:

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: only tests that actually call `run_runtime_decision()` remain RED because of the intentional typed stub. Wrapper, result constructor, signature, and isolation tests must be GREEN.

- [ ] **Step 7: Commit the records/model slice**

```bash
git add narrative_dynamics/narrative/runtime_decision_dispatch.py
git commit -m "feat: add runtime decision dispatch records"
```

Review gate: confirm this commit creates only the dispatch sidecar and does not modify scheduler or family algorithms.

---

### Task 3: Implement closed three-family dispatch execution

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_decision_dispatch.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes `RuntimeDecisionModelSpec` and all three existing family runners.
- Produces a fully validated `RuntimeDecisionDispatchResult` from `run_runtime_decision(...)`.

- [ ] **Step 1: Add wrapper reconstruction and generic authored-decision preflight helpers**

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
    decision_id = _text(decision_id, label="runtime dispatch decision id")
    decision = next((item for item in story.decisions if item.id == decision_id), None)
    if decision is None:
        raise ValueError("runtime dispatch decision is not declared by story")
    if decision.type_name not in model.supported_decision_types:
        raise ValueError("runtime dispatch model does not support decision type")
    return decision
```

This generic preflight must run before any family hook.

- [ ] **Step 2: Add the exact closed family runner branch**

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
            story, domain, decision_id, ledger, model.model
        )
    if model.model_kind == "intentional":
        assert isinstance(model.model, RuntimeIntentionalDecisionModelSpec)
        return run_runtime_intentional_decision(
            story, domain, decision_id, ledger, model.model
        )
    assert model.model_kind == "planning"
    assert isinstance(model.model, RuntimePlanningDecisionModelSpec)
    return run_runtime_planning_decision(
        story, domain, decision_id, ledger, model.model
    )
```

No model-provided runner/callback is permitted.

- [ ] **Step 3: Construct and revalidate the common envelope**

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
```

Add a model-dependent validator used immediately before return:

```python
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

- [ ] **Step 4: Replace the stub with the final runner and typed error normalization**

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
        decision = _preflight_decision(story, domain, decision_id, validated_model)
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

Do not catch `BaseException` or arbitrary broad exceptions.

- [ ] **Step 5: Run the full dispatch test file**

```bash
python3 -m unittest tests.test_narrative_runtime_decision_dispatch -v
```

Expected: all seven dispatch tests PASS.

Also run the family suites to prove the sidecar did not change them:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  -v
```

Expected: PASS.

- [ ] **Step 6: Commit the closed dispatch runner**

```bash
git add narrative_dynamics/narrative/runtime_decision_dispatch.py
git commit -m "feat: dispatch runtime decision families"
```

Reviewer gate: verify `runtime_decision_dispatch.py` still has no world/simulation/projection/RNG imports and that only the three declared family runners are callable from the dispatch branch.

---

### Task 4: Establish Scheduler V2 test-only RED, including heterogeneous agents

**Files:**
- Modify: `tests/test_narrative_simulation.py`

**Interfaces:**
- Consumes Task 3 public dispatch types.
- Defines the required scheduler migration before `simulation.py` changes:
  - `RuntimeAgentSpec.decision_model`
  - `SimulationAgentStep.decision_result: RuntimeDecisionDispatchResult`
  - same-prior-ledger generic dispatch
  - heterogeneous reactive/intentional/planning execution

- [ ] **Step 1: Change scheduler-test imports to the generic boundary**

Inside the guarded simulation import block, add:

```python
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionDispatchError,
    RuntimeDecisionDispatchResult,
    RuntimeDecisionModelSpec,
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

Keep `RuntimeIntentionalDecisionModelSpec` and `RuntimeIntentionalDecisionResolutionError` imported because tests still construct nested intentional models and inspect the preserved cause chain.

Import existing planning helper hooks:

```python
from tests.test_narrative_runtime_planning import (
    IdentityTransitionHook,
    NoInformationObservationHook,
    ProductCouplingHook,
)
```

- [ ] **Step 2: Migrate existing simulation fixtures to explicit intentional wrappers**

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

Change `make_conflict_case()` in the same way.

Replace every fixture/test access to:

```text
agent.intentional_model
```

with:

```text
agent.decision_model.model
```

where the test needs the nested intentional model.

For unsupported decision-type construction use:

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

- [ ] **Step 3: Migrate assertions from intentional internals to common dispatch fields**

In the shared-prior-ledger test, replace intentional-specific access with:

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
        agent_step.action_intent.selection_result_hash,
        agent_step.decision_result.content_hash,
    )
```

Where an intentional belief is specifically under test, unwrap explicitly:

```python
nested = step1["a2"].decision_result.model_result
self.assertIsInstance(nested, RuntimeIntentionalDecisionResult)
round1_belief = nested.belief_state
```

Update the cognition-failure cause chain to:

```python
with self.assertRaises(SimulationStepError) as caught:
    simulate_step(story, domain, prior, model)
self.assertIsInstance(caught.exception.__cause__, RuntimeDecisionDispatchError)
self.assertIsInstance(
    caught.exception.__cause__.__cause__,
    RuntimeIntentionalDecisionResolutionError,
)
```

- [ ] **Step 4: Add heterogeneous runtime hooks and a third authored decision**

Add these test-only helpers:

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
```

Create the a3 authored decision:

```python
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

- [ ] **Step 5: Add a heterogeneous model fixture using all three families**

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
    a3_belief = make_runtime_belief_model()
    c_planning = RuntimePlanningDecisionModelSpec(
        "c-runtime-planning",
        "1",
        ("scheduler-response-choice",),
        (alert_cell(),),
        (),
        a3_belief,
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

- [ ] **Step 6: Add the single-round heterogeneous scheduler test**

```python
def test_heterogeneous_round_dispatches_three_families_against_one_prior_ledger(self):
    self.require_simulation()
    domain, story, model = make_heterogeneous_case()
    prior = simulation_state_from_story(story, domain, model, at_time=10)
    result = simulate_step(story, domain, prior, model)
    by_agent = {item.agent_id: item for item in result.agent_steps}
    self.assertEqual(tuple(sorted(by_agent)), ("a1", "a2", "a3"))
    self.assertEqual(
        {agent: step.decision_result.model_kind for agent, step in by_agent.items()},
        {"a1": "intentional", "a2": "reactive", "a3": "planning"},
    )
    self.assertIsInstance(
        by_agent["a2"].decision_result.model_result,
        RuntimeReactiveDecisionResult,
    )
    self.assertIsInstance(
        by_agent["a3"].decision_result.model_result,
        RuntimePlanningDecisionResult,
    )
    for step in by_agent.values():
        self.assertEqual(
            step.decision_result.ledger_hash,
            prior.evidence_ledger.content_hash,
        )
        self.assertEqual(step.decision_result.step_index, prior.step_index)
        self.assertEqual(
            step.decision_result.decision_model_hash,
            next(
                agent.decision_model.content_hash
                for agent in model.agents
                if agent.agent_id == step.agent_id
            ),
        )
        self.assertEqual(
            step.action_intent.selection_result_hash,
            step.decision_result.content_hash,
        )
    self.assertEqual(len(result.world_step.transitions), 3)
```

- [ ] **Step 7: Add two-round heterogeneous causal-closure and replay test**

```python
def test_two_round_heterogeneous_replay_is_exact_and_uses_newly_admitted_information(self):
    self.require_simulation()
    domain1, story1, model1 = make_heterogeneous_case()
    domain2, story2, model2 = make_heterogeneous_case()
    self.assertEqual(model1.content_hash, model2.content_hash)
    initial1 = simulation_state_from_story(story1, domain1, model1, at_time=10)
    initial2 = simulation_state_from_story(story2, domain2, model2, at_time=10)
    first = simulate_trajectory(story1, domain1, initial1, model1, rounds=2)
    second = simulate_trajectory(story2, domain2, initial2, model2, rounds=2)
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

This test proves the runtime loop, not a new planning algorithm: a1 changes the world, projection/admission creates runtime evidence, and the reactive/planning agents consume that evidence in the next round.

- [ ] **Step 8: Run scheduler tests and verify the intentional API migration RED**

```bash
python3 -m unittest tests.test_narrative_simulation -v
```

Expected: failures are caused by the existing production scheduler still exposing `RuntimeAgentSpec.intentional_model`, requiring `RuntimeIntentionalDecisionResult`, and directly calling the intentional runner. The new dispatch module itself is already GREEN from Task 3.

There must be no fixture-validation failure while constructing the new a3 decision, reactive model, or planning model.

- [ ] **Step 9: Commit the scheduler test-only RED**

```bash
git add tests/test_narrative_simulation.py
git commit -m "test: define multi-model scheduler v2 contract"
```

Obtain a PR-triggered exact-head RED CI. Accept it only if existing non-scheduler tests remain green and failures are restricted to the intended scheduler V1/V2 contract mismatch.

---

### Task 5: Migrate the scheduler to generic decision dispatch

**Files:**
- Modify: `narrative_dynamics/narrative/simulation.py`
- Test: `tests/test_narrative_simulation.py`
- Test: `tests/test_narrative_runtime_decision_dispatch.py`

**Interfaces:**
- Consumes `RuntimeDecisionModelSpec`, `RuntimeDecisionDispatchResult`, `RuntimeDecisionDispatchError`, `run_runtime_decision`.
- Produces family-neutral scheduler records while preserving `SimulationModelSpec`, `SimulationState`, `SimulationStepResult`, and `SimulationTrajectory` public names.

- [ ] **Step 1: Replace intentional-specific simulation imports with generic dispatch imports**

Remove the direct `runtime_intention` import block from `simulation.py` and add:

```python
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionDispatchError,
    RuntimeDecisionDispatchResult,
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)
```

Do not import reactive, intentional, or planning model/result types directly into `simulation.py`.

- [ ] **Step 2: Change `RuntimeAgentSpec` to one generic decision model field**

Replace the record with:

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

Do not add an `intentional_model` property or constructor alias.

- [ ] **Step 3: Make execution-binding validation family-neutral**

Change the decision-type check inside `_validate_execution_bindings()` to:

```python
if decision.type_name not in agent.decision_model.supported_decision_types:
    raise ValueError(
        "simulation runtime model does not support decision template type"
    )
```

Do not inspect cue cells, goals, hidden states, planning horizon, or other family-specific configuration.

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

Replace the intentional-specific constructor checks with:

```python
if not isinstance(self.decision_result, RuntimeDecisionDispatchResult):
    raise TypeError(
        "simulation agent step decision_result must be RuntimeDecisionDispatchResult"
    )
if self.agent_id != self.decision_result.actor_id:
    raise ValueError("simulation agent step agent must match decision result actor")
if self.decision_template_id != self.decision_result.decision_id:
    raise ValueError("simulation agent step template must match decision result")
```

Keep all existing `ActionIntent` binding checks, but require:

```python
self.action_intent.selection_result_hash == self.decision_result.content_hash
```

- [ ] **Step 5: Replace direct intentional execution inside `simulate_step()`**

For each canonical `agent` call:

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

Then validate the shared-snapshot bindings before constructing the intent:

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

Construct `ActionIntent` with the generic result hash:

```python
intent = ActionIntent(
    decision_id=result.decision_id,
    selected_action=result.selected_action,
    selection_model_id=result.model_id,
    selection_result_hash=result.content_hash,
)
```

Do not move `advance_world_step()` earlier. All generic decisions must finish before any world hook executes.

- [ ] **Step 6: Leave world/projection/admission code unchanged**

The production diff after the decision loop must retain the existing call order:

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

No decision family receives same-round selected actions through the ledger.

- [ ] **Step 7: Run dispatch + complete scheduler suites**

```bash
python3 -m unittest \
  tests.test_narrative_runtime_decision_dispatch \
  tests.test_narrative_simulation \
  -v
```

Expected: PASS, including:

```text
all-intentional expected actions and causal closure
three-family single-round dispatch
three-family two-round replay
shared prior ledger for every agent
existing world conflict failure before projection
existing projection/admission atomic failure behavior
existing deterministic input-order/replay properties
```

Also run the unchanged family suites:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_reactive \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_runtime_planning \
  -v
```

Expected: PASS.

- [ ] **Step 8: Commit Scheduler V2 GREEN**

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: schedule typed runtime decision families"
```

Reviewer gate:

```text
simulation.py imports generic dispatch, not family decision modules
RuntimeAgentSpec has exactly decision_model, no compatibility alias
SimulationAgentStep contains generic dispatch result
all agents still decide before world transition
ActionIntent points to dispatch result hash
world/projection/admission code is semantically unchanged
```

Obtain exact-head CI and verify scheduler tests are GREEN before the public-surface task.

---

### Task 6: Lock the exact narrative public surface with a test-only RED

**Files:**
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Adds exactly four narrative-package public names.
- Keeps the root `narrative_dynamics` package unchanged.

- [ ] **Step 1: Extend `_EXPECTED_PUBLIC_API` by exactly four dispatch names**

Add a section between planning selection and multi-step simulation:

```python
# Runtime generic decision dispatch.
"RuntimeDecisionModelSpec",
"RuntimeDecisionDispatchResult",
"RuntimeDecisionDispatchError",
"run_runtime_decision",
```

Do not add family-private helpers, unions, kind constants, or scheduler aliases.

- [ ] **Step 2: Run the trust surface test and verify RED**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected: FAIL with exactly these four names present in `_EXPECTED_PUBLIC_API` but absent from `narrative_dynamics.narrative.__all__`. Root isolation assertions must not be the cause yet.

- [ ] **Step 3: Commit the public-surface test-only RED**

```bash
git add tests/test_narrative_trust_api.py
git commit -m "test: lock narrative decision dispatch surface"
```

Obtain exact-head CI and confirm the only new failure is the expected four-name surface RED.

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

- [ ] **Step 1: Import the exact four names into the narrative package**

Add:

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

Place the four names adjacent to the runtime decision-family sections:

```python
"RuntimeDecisionModelSpec",
"RuntimeDecisionDispatchResult",
"RuntimeDecisionDispatchError",
"run_runtime_decision",
```

Do not export `_MODEL_TYPES`, `_RESULT_TYPES`, validators, or extraction helpers.

- [ ] **Step 3: Run targeted trust, dispatch, and scheduler suites**

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

Expected: all tests PASS with zero failures/errors. Record the exact test count from the output; do not copy the previous 579 count unless the fresh run still reports 579 after adding the new tests.

- [ ] **Step 5: Verify final diff scope against the integrated base**

The final feature diff may contain only:

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

If any family algorithm, world/projection/cognition, root package, prison/model-comparison, or Lean path appears, stop and review the scope before final CI.

- [ ] **Step 6: Commit the public export GREEN**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export runtime decision dispatch surface"
```

- [ ] **Step 7: Obtain authoritative final exact-head PR CI**

The final accepted proof must satisfy:

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

- [ ] **Step 8: Perform final review gates before integration**

Verify:

```text
PR review submissions: no unresolved request-changes blocker
inline review threads: none unresolved
conversation comments: no unresolved blocker
PR head SHA: exact final GREEN head
PR base: proof/narrative-dynamics-v0
changed paths: exactly the approved scope
```

Update the PR body with:

```text
authoritative RED run number + exact test-only SHA
dispatch GREEN slices
scheduler V2 RED/GREEN slices
heterogeneous three-family single-round evidence
heterogeneous two-round deterministic replay evidence
final exact-head proof run number and test count
explicit note that old V1 simulation hashes are not compatibility targets
```

Mark the PR ready only after all gates above are satisfied. Do not merge automatically; integration remains the user's finishing decision.

---

## Completion Checklist

Before calling the feature complete, verify every item from the design against a concrete test or diff:

- [ ] Closed model kinds are exactly reactive / intentional / planning.
- [ ] Wrapper tag/type mismatch fails before family hooks.
- [ ] Wrapper identity binds nested model, wrapper implementation, and generic runner implementation.
- [ ] Common result exposes only genuine shared fields.
- [ ] Common result retains exact nested family result payload and hash.
- [ ] Intentional actor/ledger extraction comes from its runtime belief state; no fake new latent field is added to the intentional result.
- [ ] Generic runner has exactly `(story, domain, decision_id, ledger, model)` parameters.
- [ ] Dispatcher has no world/simulation/projection/RNG capability.
- [ ] Family failures normalize to `RuntimeDecisionDispatchError` with exact cause preserved.
- [ ] `RuntimeAgentSpec` has one `decision_model` field and no `intentional_model` alias.
- [ ] `SimulationAgentStep` binds `RuntimeDecisionDispatchResult`.
- [ ] Action intent hash points to generic dispatch result.
- [ ] All agents use the exact same prior ledger in one round.
- [ ] Reactive + intentional + planning agents execute in one atomic round.
- [ ] A two-round heterogeneous trajectory consumes newly admitted evidence and replays exactly.
- [ ] Existing conflicting-write rejection remains atomic and projection does not run after world conflict.
- [ ] All-intentional behavioral expectations remain unchanged even though V2 content hashes are allowed to change.
- [ ] Narrative package exports exactly four new dispatch names.
- [ ] Root package exports none of the new narrative names.
- [ ] Final diff stays within the eight approved paths.
- [ ] Final exact-head PR `proof` is completed/success with fresh full-suite evidence.
