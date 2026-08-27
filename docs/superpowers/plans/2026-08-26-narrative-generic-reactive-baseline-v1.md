# Narrative Generic Reactive Baseline V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native, memoryless runtime reactive model that maps only current observable cues to a complete action probability policy, with deterministic replay, typed fail-closed boundaries, and exact comparison compatibility with the intentional family.

**Architecture:** Add one isolated sidecar module, `narrative_dynamics/narrative/runtime_reactive.py`. It extracts only the current cue snapshot from authored direct observation at step zero or the latest runtime evidence batch at later steps, passes a sanitized immutable context to an attested score hook, projects exact action scores through the existing `grounded_goal_softmax.finite_softmax`, and selects lexical MAP. Scheduler V1, Runtime Cognition, Runtime Intention, world/observation machinery, Lean sources, root exports, prison adapters, and evaluation infrastructure remain unchanged.

**Tech Stack:** Python 3 stdlib dataclasses/mappings/math/unittest; existing `GenericNarrative`, `DomainSpec`, `RuntimeEvidenceLedger`, authored direct replay, attestation/content hashing, and `grounded_goal_softmax.finite_softmax`; GitHub Actions `proof` workflow for exact-head verification.

**Spec:** `docs/superpowers/specs/2026-08-26-narrative-generic-reactive-baseline-v1-design.md`

## Global Constraints

- Exact base: `045f5b0602e30e8ec12e7c9fa73128e28cabebd8` on `proof/narrative-dynamics-v0`.
- Feature branch: `work/narrative-generic-reactive-baseline-v1`.
- Production changes are limited to `narrative_dynamics/narrative/runtime_reactive.py` and `narrative_dynamics/narrative/__init__.py`.
- Do not modify GenericNarrative IR, DomainSpec, `decision.py`, `intention.py`, `runtime_cognition.py`, `runtime_intention.py`, `runtime_perception.py`, `world.py`, `observation_projection.py`, `simulation.py`, Lean sources, root `narrative_dynamics` exports, prison adapters, model comparison, or loss implementations.
- Step zero uses authored **direct** observation only. Step > 0 uses only `ledger.batches[-1]`; missing current cues are `unknown` and never filled from history.
- The reactive score hook never receives story/domain/ledger/evidence provenance/world/belief/goal/planning objects.
- Actions are canonicalized lexically. Cue cells are canonicalized by `(entity_type, entity_id, state_variable)`.
- Parameters accept only recursively canonical scalar/mapping/list/tuple values from the approved spec and are frozen before hook invocation.
- Reactive and intentional choice share the existing measured `finite_softmax` numerical kernel; reactive model identity binds that implementation identity.
- `beta` is finite and strictly positive. Action scores are exact-schema finite numerics and not bool.
- Final action policy is a complete simplex over exactly the authored action set; sum tolerance is absolute `1e-12`, relative `0.0`.
- V1 consumes no RNG. Selected action is maximum policy probability with lexical action id tie-break.
- Public failures normalize to `RuntimeReactiveDecisionResolutionError`, preserving the original `Exception` as `__cause__`; do not catch `BaseException`.
- Exactly seven names are added to `narrative_dynamics.narrative`; root package isolation remains unchanged.
- Strict RED -> GREEN -> atomic commit discipline. The first executable feature commit is test-only RED.

---

### Task 1: Define the test-only RED contract

**Files:**
- Create: `tests/test_narrative_runtime_reactive.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: existing runtime fixtures `empty_runtime_case()`, `extend_ledger()`, `phase_cell()`, `make_runtime_intentional_model()`, `run_runtime_intentional_decision()`.
- Produces: executable contract for the exact seven public reactive names and all V1 semantic boundaries. No production symbol exists yet.

- [ ] **Step 1: Write import-safe RED tests for the wished-for public API**

Use the same staged-import pattern as `tests/test_narrative_runtime_intention.py`:

```python
_REACTIVE_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_reactive import (
        ReactiveCueView,
        RuntimeReactiveCueSnapshot,
        RuntimeReactiveDecisionContext,
        RuntimeReactiveDecisionModelSpec,
        RuntimeReactiveDecisionResult,
        RuntimeReactiveDecisionResolutionError,
        run_runtime_reactive_decision,
    )
except ImportError as error:
    _REACTIVE_IMPORT_ERROR = error
```

Every feature test begins with `require_reactive()` so the absent module produces an intentional assertion failure rather than test discovery errors.

- [ ] **Step 2: Lock the semantic matrix**

Create focused tests covering at minimum:

```text
model identity + parameter canonicalization
public signature excludes world/step inputs
step-zero direct cue extraction
step-positive latest-batch-only extraction
blackout -> unknown with no history fill
agreeing channels collapse semantically
same-step semantic conflict typed failure
sanitized immutable hook context
cue capability outside decision context rejected before hook
forged ledger/story/domain/current-step rejected before hook
exact action score schema + non-finite rejection
shared finite_softmax policy + exact simplex
lexical MAP tie break
hook Exception normalization and cause preservation
deterministic exact replay/content hash
observational equivalence with intentional exact policy
information-access intervention separates reactive/intentional policy
module import isolation
self-validating snapshot/result forgery rejection
```

Use recording hooks to prove invalid prerequisites do not invoke the hook.

- [ ] **Step 3: Extend the exact narrative public-surface expectation**

Add exactly these seven names to `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`:

```python
"ReactiveCueView",
"RuntimeReactiveCueSnapshot",
"RuntimeReactiveDecisionContext",
"RuntimeReactiveDecisionModelSpec",
"RuntimeReactiveDecisionResult",
"RuntimeReactiveDecisionResolutionError",
"run_runtime_reactive_decision",
```

Do not change root-package expectations.

- [ ] **Step 4: Verify RED on exact branch head**

Run locally when available:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Expected: prior tests remain green; new reactive semantic tests fail only because `narrative_dynamics.narrative.runtime_reactive` does not exist, and the exact public-surface test fails only for the seven missing names.

If local execution is unavailable, open a draft PR and use the exact-head `proof` workflow as authoritative RED evidence.

- [ ] **Step 5: Commit the test-only RED**

```bash
git add tests/test_narrative_runtime_reactive.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative generic reactive baseline v1"
```

Do not add `runtime_reactive.py` in this commit.

---

### Task 2: Add canonical reactive value records and model identity

**Files:**
- Create: `narrative_dynamics/narrative/runtime_reactive.py`
- Test: `tests/test_narrative_runtime_reactive.py`

**Interfaces:**
- Consumes: `StateCellRef`, `TypedValue`, `stable_content_hash`, `measure_implementation`, `finite_softmax`.
- Produces: `ReactiveCueView`, `RuntimeReactiveCueSnapshot`, `RuntimeReactiveDecisionContext`, `RuntimeReactiveDecisionModelSpec`, `RuntimeReactiveDecisionResolutionError` plus private canonicalization helpers.

- [ ] **Step 1: Select only the record/model tests from the RED suite**

Run:

```bash
python -m unittest \
  tests.test_narrative_runtime_reactive.NarrativeRuntimeReactiveTests.test_model_identity_binds_parameters_hook_and_shared_softmax \
  tests.test_narrative_runtime_reactive.NarrativeRuntimeReactiveTests.test_model_parameters_accept_only_canonical_values \
  tests.test_narrative_runtime_reactive.NarrativeRuntimeReactiveTests.test_value_records_are_immutable_canonical_and_self_validating
```

Expected: FAIL because the module/types are absent.

- [ ] **Step 2: Implement canonical scalar/container freezing**

Private helpers must accept exactly:

```python
None | bool | int | str | finite float | Mapping[str, value] | list[value] | tuple[value]
```

Freeze mappings with lexical string-key serialization and lists/tuples to tuples. Reject non-finite floats, bytes, sets, arbitrary objects, callables, and empty/invalid mapping keys.

- [ ] **Step 3: Implement the four value/model records**

Implement exact approved field sets and validations:

```python
@dataclass(frozen=True)
class ReactiveCueView: ...

@dataclass(frozen=True)
class RuntimeReactiveCueSnapshot: ...

@dataclass(frozen=True)
class RuntimeReactiveDecisionContext: ...

@dataclass(frozen=True)
class RuntimeReactiveDecisionModelSpec: ...
```

`RuntimeReactiveDecisionModelSpec.to_dict()` must bind:

```text
model id/version
supported decision types
cue cells
canonical parameters
beta
measure_implementation(score_hook).manifest_identity()
measure_implementation(RuntimeReactiveDecisionModelSpec).manifest_identity()
measure_implementation(finite_softmax).manifest_identity()
```

- [ ] **Step 4: Verify the record/model slice GREEN**

Run the three tests above plus existing runtime-intention identity tests. Expected: selected tests PASS; execution tests remain intentionally RED.

- [ ] **Step 5: Commit**

```bash
git add narrative_dynamics/narrative/runtime_reactive.py tests/test_narrative_runtime_reactive.py
git commit -m "feat: add runtime reactive records"
```

---

### Task 3: Implement current-cue extraction and fail-closed trust validation

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_reactive.py`
- Test: `tests/test_narrative_runtime_reactive.py`

**Interfaces:**
- Consumes: `direct_state(story, domain, actor_id, at_time=ledger.source_at_time)`, `RuntimeEvidenceLedger`, canonical authored `Decision`.
- Produces: private current-cue extraction plus prerequisite validation used by `run_runtime_reactive_decision()`.

- [ ] **Step 1: Run cue/trust RED tests**

Target tests for step-zero direct cues, current-batch-only extraction, blackout, agreeing channels, conflict, cue capability, cutoff, forged ledger/domain/story/current step, and pre-hook failure ordering.

Expected: FAIL because public execution is absent.

- [ ] **Step 2: Implement exact ledger and decision preflight**

Validate in order:

```text
story/domain
model type
ledger type + constructor-level invariants
ledger domain id/version/spec hash
ledger source story hash
decision id
decision type
cue_cells subset of decision.context_cells
action ids non-empty/unique
source cutoff >= decision.logical_time when numeric
```

Reconstruct `RuntimeEvidenceLedger(...)` from its public fields before hook invocation so constructor-bypassing forgeries fail.

- [ ] **Step 3: Implement step-zero extraction**

Use only:

```python
direct_state(story, domain, decision.actor_id, at_time=ledger.source_at_time)
```

For each declared cue cell, map resolved direct value to `resolved`; otherwise `unknown`. Do not call epistemic or runtime cognition APIs.

- [ ] **Step 4: Implement step-positive extraction**

Read only `ledger.batches[-1]`, require it to be the current step, filter to the actor, group eligible rows by cue cell, and resolve:

```text
no row -> unknown
all same equals(value) -> resolved(value)
all clear -> unknown
disagreement -> RuntimeReactiveDecisionResolutionError
```

Channel identity never enters `ReactiveCueView`.

- [ ] **Step 5: Verify cue/trust slice GREEN**

Run the targeted cue/trust tests. Assert recording hook call lists remain empty for prerequisite failures.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/narrative/runtime_reactive.py tests/test_narrative_runtime_reactive.py
git commit -m "feat: resolve runtime reactive cues"
```

---

### Task 4: Implement score hook, shared softmax policy, lexical MAP, and result record

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_reactive.py`
- Test: `tests/test_narrative_runtime_reactive.py`

**Interfaces:**
- Consumes: validated `RuntimeReactiveDecisionContext`, `finite_softmax`, exact authored action ids.
- Produces: `RuntimeReactiveDecisionResult` and fully working `run_runtime_reactive_decision(...)`.

- [ ] **Step 1: Run scoring/policy/result RED tests**

Target exact score coverage, bool/nan/inf rejection, hook exception wrapping, complete simplex, lexical tie, exact replay, result-forgery validation, and public signature tests.

- [ ] **Step 2: Invoke the score hook across the sanitized boundary**

Construct `RuntimeReactiveDecisionContext` from canonical values only, then call:

```python
raw_scores = model.score_hook(context)
```

Catch `Exception` and wrap as `RuntimeReactiveDecisionResolutionError` with `raise ... from error`; never catch `BaseException`.

- [ ] **Step 3: Validate exact score schema**

Require a mapping with keys exactly equal to authored action ids. Convert numeric int/float values to finite floats; reject bool, non-numeric, nan, inf, missing, and extra action ids. Store lexically ordered mapping.

- [ ] **Step 4: Project policy through the shared kernel**

Call exactly:

```python
raw_policy = finite_softmax(action_scores, beta=model.beta)
```

Validate exact action keys, finite non-negative masses, and `math.isclose(math.fsum(...), 1.0, rel_tol=0.0, abs_tol=1e-12)`. Canonicalize lexical key order.

- [ ] **Step 5: Select deterministic lexical MAP and construct result**

Implement private MAP projection:

```python
maximum = max(policy.values())
selected = min(action for action, probability in policy.items() if probability == maximum)
```

Implement self-validating `RuntimeReactiveDecisionResult` with exact model/ledger/snapshot/hash/policy consistency.

- [ ] **Step 6: Verify scoring/policy/result slice GREEN**

Run all non-scientific reactive tests. Expected: all pass except deliberate public export gate and any still-unimplemented scientific fixture tests.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/narrative/runtime_reactive.py tests/test_narrative_runtime_reactive.py
git commit -m "feat: select runtime reactive actions"
```

---

### Task 5: Lock scientific equivalence and separating-intervention behavior

**Files:**
- Modify: `tests/test_narrative_runtime_reactive.py`
- Modify only if a real contract gap is exposed: `narrative_dynamics/narrative/runtime_reactive.py`

**Interfaces:**
- Consumes: completed reactive runner, existing runtime intentional runner, existing runtime ledger fixtures.
- Produces: exact scientific tests showing when reactive and intentional are observationally equivalent and when observation-memory differences separate them.

- [ ] **Step 1: Run the observational-equivalence test RED/GREEN gate**

Construct one fixture where the intentional conditional action values are identical under all goals and equal the reactive hook scores, with identical `beta_action == reactive.beta`.

Require:

```python
self.assertEqual(reactive.action_policy, intentional.action_policy)
self.assertEqual(reactive.selected_action, intentional.selected_action)
```

Because both families call the same `finite_softmax` over the same ordered score mapping, equality must be exact, not approximate.

- [ ] **Step 2: Run the information-access separation test**

Build two ledgers with the same authored story/action space and no changed objective-state semantics used by the decision fixture:

```text
step 1: informative active cue admitted
step 2 visible: active cue admitted again
step 2 blackout: empty evidence batch
```

Require reactive visible vs blackout policies to differ because blackout yields `unknown`. Require intentional at blackout to retain/use prior evidence through history and therefore differ from reactive blackout under the configured fixture.

- [ ] **Step 3: Verify module isolation structurally**

Use `inspect.getsource(runtime_reactive)` and assert it does not import or mention production dependencies from:

```text
runtime_cognition
runtime_intention
narrative.intention
narrative.world
```

- [ ] **Step 4: Verify full reactive semantic suite GREEN except export gate**

Run:

```bash
python -m unittest tests.test_narrative_runtime_reactive -v
```

Expected: all reactive semantics PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_narrative_runtime_reactive.py narrative_dynamics/narrative/runtime_reactive.py
git commit -m "test: lock reactive model identification cases"
```

If no production adjustment was needed, commit only the strengthened tests.

---

### Task 6: Export exactly seven narrative-scoped names and close the feature GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: completed reactive module.
- Produces: exact narrative public API while preserving root isolation.

- [ ] **Step 1: Run the exact public-surface test RED**

```bash
python -m unittest tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: FAIL only because the seven reactive names are absent from `narrative_dynamics.narrative`.

- [ ] **Step 2: Add only the seven approved imports and `__all__` entries**

Export exactly:

```text
ReactiveCueView
RuntimeReactiveCueSnapshot
RuntimeReactiveDecisionContext
RuntimeReactiveDecisionModelSpec
RuntimeReactiveDecisionResult
RuntimeReactiveDecisionResolutionError
run_runtime_reactive_decision
```

Do not touch `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Verify exact surface GREEN**

Run the public-surface test and full reactive suite. Expected: PASS.

- [ ] **Step 4: Run complete Python regression**

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

Expected: all Python tests PASS.

- [ ] **Step 5: Audit final diff scope**

Compare against base `045f5b0602e30e8ec12e7c9fa73128e28cabebd8`. Expected final paths only:

```text
docs/superpowers/specs/2026-08-26-narrative-generic-reactive-baseline-v1-design.md
docs/superpowers/plans/2026-08-26-narrative-generic-reactive-baseline-v1.md
narrative_dynamics/narrative/runtime_reactive.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_reactive.py
tests/test_narrative_trust_api.py
```

- [ ] **Step 6: Commit export GREEN**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative reactive surface"
```

- [ ] **Step 7: Verify exact-head proof workflow**

Push/open or update the draft PR against `proof/narrative-dynamics-v0` and wait for the exact final head `proof` workflow. Require all gates to pass:

```text
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

Do not mark complete from an earlier head.

---

## Final Acceptance Checklist

- [ ] Generic runtime reactive model is narrative-native and independent of prison adapters.
- [ ] Step zero uses authored direct cues only; later steps use only the current batch.
- [ ] Blackout does not fill from prior history.
- [ ] Score hook sees only immutable declared cue semantics and canonical parameters.
- [ ] No world/belief/goal/planning/provenance object crosses the hook boundary.
- [ ] Exact authored action set -> finite scores -> shared `finite_softmax` -> complete simplex.
- [ ] Reactive model identity binds score hook, record implementation, and shared softmax implementation identities.
- [ ] Lexical MAP is deterministic and replay hash is exact.
- [ ] Forged/malformed prerequisites fail typed before hook execution.
- [ ] Observational-equivalence fixture matches intentional policy exactly.
- [ ] Information-access intervention separates memoryless reactive from history-driven intentional behavior.
- [ ] Exactly seven narrative names exported; root isolation unchanged.
- [ ] Final diff contains only the six approved paths.
- [ ] Exact final head passes the complete `proof` workflow.
