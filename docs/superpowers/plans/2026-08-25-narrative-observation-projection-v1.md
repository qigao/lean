# Narrative Observation Projection V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, typed, capability-limited, truth-validated, provenance-linked runtime projection from one validated `WorldStepResult` to per-agent `ProjectedObservation` records.

**Architecture:** Add one independent sidecar module, `narrative_dynamics/narrative/observation_projection.py`, consuming the existing canonical story/domain and `WorldStepResult`. Projection hooks receive only immutable capability-filtered prior/next world views and return untrusted `ObservationFact` proposals; the runtime validates domain declarations, source-world consistency, emit capability, exact post-step truth, and then derives all provenance. Authored `GenericNarrative.Observation`, replay, belief, decision, intention, and World Transition V1 remain unchanged.

**Tech Stack:** Python 3 standard library (`dataclasses`, `types.MappingProxyType`, `unittest`), existing `stable_content_hash`, `measure_implementation`, Generic Narrative IR/domain types, and World Transition V1 records/runtime.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md`

## Global Constraints

- Exact base commit: `44f46a13d228afb48ca6938b51a6a8606c52d984` on `proof/narrative-dynamics-v0`.
- Work only on `work/narrative-observation-projection-v1`; do not modify `proof/narrative-dynamics-v0` or `master` during implementation.
- Keep `NarrativeEvent.logical_time` / `Decision.logical_time` distinct from runtime `WorldState.step_index`.
- Runtime projected observations must never masquerade as authored `GenericNarrative.Observation` values or mutate `GenericNarrative.observations`.
- Projection hooks must never receive the complete objective world through the framework API; they receive only read-capability-filtered immutable mappings.
- A hook may choose visibility but may not choose truth: accepted `equals` facts must exactly equal `world_step.next_state`, and accepted `clear` facts require both post-step absence and an explicit current-step clear write.
- Transition provenance is based on actual write-set membership; a same-value `set` still contributes the exact transition-record hash.
- `ProjectedObservation` and `ObservationProjectionResult` are public data records; direct construction does not independently certify provenance.
- `ObservationProjectionModelSpec(projections=())` is valid and represents an explicit blackout condition.
- No RNG, noisy sensor semantics, testimony generation, event-occurrence percepts, scheduler, belief bridge, GenericNarrative schema changes, or DomainSpec identity changes in V1.
- Final feature diff relative to the base must contain exactly six paths: the approved spec, this plan, `observation_projection.py`, narrative-scoped `__init__.py`, dedicated projection tests, and the exact public-surface test.
- Follow strict test-only RED -> exact-head CI RED -> reviewable GREEN commits -> final scoped export -> full exact-head CI.
- Do not merge to `proof/narrative-dynamics-v0` without explicit user instruction.

## File Structure

- `docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md` — approved architecture and scientific/trust boundary; already committed.
- `docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md` — this implementation sequence.
- `narrative_dynamics/narrative/observation_projection.py` — all V1 records, model identity, source validation, filtered-view construction, hook execution, truth checks, provenance derivation, and public projection function. Keep this as one sidecar to match the existing `world.py` pattern and avoid changing existing semantic modules.
- `narrative_dynamics/narrative/__init__.py` — final narrative-scoped imports and eight `__all__` additions only.
- `tests/test_narrative_observation_projection.py` — isolated V1 fixtures and all semantic RED/GREEN tests; do not move projection-specific fixtures into shared test support.
- `tests/test_narrative_trust_api.py` — exact eight-name narrative API expectation and root-isolation lock only.

---

### Task 1: Commit the complete test-only RED contract

**Files:**
- Create: `tests/test_narrative_observation_projection.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: existing `DomainSpec`, `GenericNarrative`, `WorldStepResult`, `WorldState`, `ActionTransitionRecord`, `advance_world_step`, and `stable_content_hash`.
- Produces: a complete failing contract for exactly these future public names: `ObservationCapabilitySpec`, `ObserverProjectionSpec`, `ObservationProjectionModelSpec`, `ObservationFact`, `ProjectedObservation`, `ObservationProjectionResult`, `ObservationProjectionError`, `project_world_observations`.

- [ ] **Step 1: Add the guarded import and local projection fixture**

Create `tests/test_narrative_observation_projection.py` with `unittest` style matching the existing narrative tests. The import gate must make every new semantic test fail clearly when the module is absent:

```python
from __future__ import annotations

from dataclasses import replace
import unittest

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    DomainSpec,
    EntityTypeSpec,
    EventTypeSpec,
    ParameterSpec,
    SemanticHookBinding,
    StateDelta,
    StateDeltaOp,
    StateEffectSpec,
    StateVariableSpec,
    ValueTypeSpec,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    GenericNarrative,
    NarrativeEvent,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionIntent,
    ActionTransitionRecord,
    ActionTransitionSpec,
    WorldState,
    WorldStepResult,
    WorldTransitionModelSpec,
    advance_world_step,
    world_state_from_story,
)


_PROJECTION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.observation_projection import (
        ObservationCapabilitySpec,
        ObservationFact,
        ObservationProjectionError,
        ObservationProjectionModelSpec,
        ObservationProjectionResult,
        ObserverProjectionSpec,
        ProjectedObservation,
        project_world_observations,
    )
except ImportError as error:
    _PROJECTION_IMPORT_ERROR = error
```

Use a local domain containing `Agent`, `Service`, and `Room` entities; enum value types `PhaseState`, `LocationState`, `HealthState`, and `LightState`; boolean `AlertState`; and these state variables:

```python
StateVariableSpec("agent.phase", "Agent", "PhaseState")
StateVariableSpec("agent.location", "Agent", "LocationState")
StateVariableSpec("service.health", "Service", "HealthState")
StateVariableSpec("service.alert", "Service", "AlertState")
StateVariableSpec("room.lighting", "Room", "LightState")
```

The canonical story must declare two agents `a1` and `a2`, one service `svc`, one room `room`, authored seed events that establish both agent locations, agent phases, service health/alert, and room lighting, plus two canonical decisions owned by different agents. Use existing World Transition V1 to build valid runtime steps so projection tests do not duplicate the transition engine's positive-path semantics.

Define module-level callable classes used by model attestation rather than lambdas or nested functions. The projection hook witnesses must include:

```python
class RecordViewsHook:
    def __init__(self, facts=()):
        self.calls = []
        self.facts = tuple(facts)

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append(
            (dict(prior_visible), dict(next_visible), observer, step_index)
        )
        return self.facts


class OwnLocationHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(
            EntityRef(observer.id, observer.type_name),
            "agent.location",
        )
        value = next_visible.get(cell)
        if value is None:
            return ()
        return (ObservationFact(cell, "equals", value),)


class OwnLocationHookV2:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(
            EntityRef(observer.id, observer.type_name),
            "agent.location",
        )
        value = next_visible.get(cell)
        return () if value is None else (ObservationFact(cell, "equals", value),)


class RaisingProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        raise RuntimeError("projection boom")
```

Add a `require_projection()` helper on the test case:

```python
def require_projection(self) -> None:
    if _PROJECTION_IMPORT_ERROR is not None:
        self.fail(
            "narrative observation projection boundary is missing: "
            f"{_PROJECTION_IMPORT_ERROR}"
        )
```

- [ ] **Step 2: Lock records, identity, containment, and blackout behavior**

Add these exact test methods. Group multiple constructor cases in `subTest` blocks so each method represents one semantic boundary rather than one malformed scalar.

```python
def test_record_constructors_fail_closed_and_public_records_are_data_not_certification(self):
    self.require_projection()
    # Reject blank names, bad scopes, bad relations, bad hash formats,
    # non-TypedValue equals values, valued clear facts, duplicate transition hashes,
    # non-tuple record fields, and malformed step indices.
    # Construct a syntactically valid ProjectedObservation with arbitrary valid hashes
    # and assert only shape/content_hash behavior; do not treat construction as proof
    # that the referenced source world step exists.


def test_model_identity_binds_domain_capabilities_channel_and_hook_not_order(self):
    self.require_projection()
    # Build equivalent specs with reversed capability/spec input order and assert equal hash.
    # Change domain_spec_hash, channel, read capability, emit capability, and
    # OwnLocationHook -> OwnLocationHookV2 one at a time and assert different model hash.


def test_duplicate_declarations_and_semantic_containment_fail_closed(self):
    self.require_projection()
    # Reject exact duplicate capabilities and duplicate (observer_type, channel) specs.
    # Assert read(any) contains emit(any) and emit(observer),
    # read(observer) contains emit(observer), and
    # read(observer) cannot authorize emit(any).


def test_blackout_model_returns_empty_source_bound_result_without_hook_calls(self):
    self.require_projection()
    # Build ObservationProjectionModelSpec(..., projections=()).
    # Project a valid WorldStepResult and assert observations == (),
    # exact model hash, exact next-state hash, exact world-step hash, and exact step_index.
```

The constructor cases must assert ordinary `TypeError`/`ValueError`; runtime failures in subsequent methods must assert `ObservationProjectionError`.

- [ ] **Step 3: Lock domain declaration and hidden-state visibility boundaries**

Add:

```python
def test_undeclared_observer_variable_and_incompatible_observer_scope_reject_before_hook(self):
    self.require_projection()
    # Use RecordViewsHook and assert calls == [] after each runtime rejection:
    # observer_type="MissingAgent";
    # state_variable="missing.state";
    # observer_type="Agent" + ObservationCapabilitySpec("service.health", "observer").


def test_hook_sees_only_read_capability_cells_and_scope_is_exact(self):
    self.require_projection()
    # With read(agent.location, observer), each Agent invocation sees only its own location.
    # With read(agent.location, any), each Agent invocation sees both present agent locations.
    # service.health and room.lighting must be absent unless explicitly readable.
    # Assert supplied mappings reject item assignment with TypeError.


def test_readable_nonemittable_and_bad_hook_return_shapes_reject_typed(self):
    self.require_projection()
    # read includes room.lighting(any) + agent.location(any), emit only agent.location(any).
    # Hook proposal of room.lighting must raise ObservationProjectionError.
    # Return list, generator, mapping, scalar, tuple containing non-ObservationFact,
    # and duplicate same-cell facts; each must raise ObservationProjectionError.
    # RaisingProjectionHook must also be wrapped as ObservationProjectionError.
```

- [ ] **Step 4: Lock truth, clear semantics, and canonical cell validation**

Add:

```python
def test_equals_requires_exact_post_step_typed_truth(self):
    self.require_projection()
    # Exact next-state value succeeds.
    # Different enum value, transformed text/value, wrong TypedValue type, or absent cell rejects.


def test_clear_requires_explicit_current_step_clear(self):
    self.require_projection()
    # Build one valid world step whose transition explicitly clears service.alert;
    # clear fact succeeds and binds that transition.
    # Build a step where an already-absent cell remains absent without a clear write;
    # the same clear proposal must raise ObservationProjectionError.


def test_invalid_fact_cell_subject_type_or_variable_rejects(self):
    self.require_projection()
    # Reject unknown entity id, forged EntityRef type, undeclared variable,
    # and a declared variable applied to an incompatible subject type.
```

- [ ] **Step 5: Lock per-agent behavior, channels, provenance, and deterministic identity**

Add:

```python
def test_two_observers_can_receive_different_percepts_from_same_world_step(self):
    self.require_projection()
    # OwnLocationHook with observer-scoped read+emit must produce one observation per agent,
    # each carrying that observer's own location while source world hashes are identical.


def test_multiple_channels_and_empty_fact_tuple_are_supported(self):
    self.require_projection()
    # Same observer type has two specs with channels "vision" and "status".
    # One hook emits one fact and the other returns (). Verify one accepted observation,
    # both hooks invoked once per matching observer, and no channel substitution by hook output.


def test_transition_provenance_binds_written_cells_including_same_value_set(self):
    self.require_projection()
    # Build a step that sets a1.agent.phase to the exact value already in prior_state.
    # Observe a1.agent.phase and assert source_transition_hashes equals the exact
    # writing ActionTransitionRecord.content_hash despite no extensional value change.


def test_unwritten_persistent_value_has_empty_transition_provenance_and_exact_source_hashes(self):
    self.require_projection()
    # Observe room.lighting when no current transition writes it.
    # Assert source_transition_hashes == (), source_world_state_hash == next_state.content_hash,
    # source_world_step_hash == world_step.content_hash, and exact projection spec hash.


def test_runtime_derives_step_and_lineage_and_hook_cannot_supply_it(self):
    self.require_projection()
    # The hook returns ObservationFact only.
    # Assert accepted record fields are engine-derived and step_index is exactly
    # world_step.next_state.step_index, not authored Decision.logical_time.
```

- [ ] **Step 6: Lock source validation before hooks and mutation/order isolation**

Add:

```python
def test_source_identity_and_state_payload_reject_before_hook(self):
    self.require_projection()
    # For each forged world payload, RecordViewsHook.calls remains []:
    # mismatched story hash, mismatched domain hash, malformed prior cell/value,
    # malformed next cell/value, next step index mismatch, and parent hash mismatch.


def test_transition_decision_action_actor_and_cutoff_reject_before_hook(self):
    self.require_projection()
    # Forge transition record decision_id, selected_action, actor_id, canonical ActionOption,
    # and a source_at_time earlier than the selected decision. Each rejects pre-hook.


def test_extensional_forgery_duplicate_actor_and_write_collisions_reject_before_hook(self):
    self.require_projection()
    # Forge next_state.values so it does not equal applying supplied deltas;
    # add two transition records for one actor;
    # create one delta with duplicate cell writes;
    # create two records writing the same cell.
    # Every case raises ObservationProjectionError before RecordViewsHook is called.


def test_projection_order_hash_and_inputs_are_immutable(self):
    self.require_projection()
    # Reverse model spec order and fact order, use equivalent canonical inputs,
    # and assert identical accepted observation ordering and result.content_hash.
    # Snapshot story.to_dict(), domain.to_dict(), world_step.to_dict(), prior_state.to_dict(),
    # and next_state.to_dict() before projection and assert exact equality after projection.
```

These 19 methods cover all 36 required RED invariants from the design spec.

- [ ] **Step 7: Add the exact eight-name narrative API expectation without exporting production symbols**

Modify only `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py` by inserting:

```python
    # Runtime observation projection.
    "ObservationCapabilitySpec",
    "ObserverProjectionSpec",
    "ObservationProjectionModelSpec",
    "ObservationFact",
    "ProjectedObservation",
    "ObservationProjectionResult",
    "ObservationProjectionError",
    "project_world_observations",
```

Do not modify `narrative_dynamics/narrative/__init__.py` in this task. This makes public API exposure a deliberate final RED gate.

- [ ] **Step 8: Run syntax checks on the test-only files**

Run:

```bash
python3 -m py_compile \
  tests/test_narrative_observation_projection.py \
  tests/test_narrative_trust_api.py
```

Expected: exit 0. Syntax/discovery failures are not acceptable RED evidence.

- [ ] **Step 9: Run focused RED locally when a checkout is available**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_observation_projection \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected before production exists:

- all 19 projection methods fail only through `require_projection()` because `narrative_dynamics.narrative.observation_projection` is absent;
- the trust API method fails only because the exact eight approved projection names are missing;
- no fixture construction, syntax, test discovery, or existing world-transition failure occurs.

- [ ] **Step 10: Commit the test-only RED atomically**

```bash
git add \
  tests/test_narrative_observation_projection.py \
  tests/test_narrative_trust_api.py
git commit -m "test: define narrative observation projection v1"
```

The commit must contain exactly these two test paths.

- [ ] **Step 11: Open a Draft PR and obtain exact-head CI RED evidence**

Create a Draft PR titled:

```text
feat: add narrative observation projection v1
```

Base: `proof/narrative-dynamics-v0`.

The PR body must record the exact test-only RED head and state that production is intentionally absent. Wait for the `proof` workflow on that exact head and inspect the full job log.

Required RED evidence:

- Lean/Python conformance succeeds;
- full Lean build succeeds;
- Lean theorem suite succeeds;
- all existing Python tests, including all World Transition V1 tests, succeed;
- only the 19 new projection tests and the exact public-surface test fail for the intended missing module/API reasons;
- StoryState and Testimony theorem gates are not semantically regressed if the workflow reaches them.

Do not start Task 2 until this RED boundary is observed.

---

### Task 2: Implement immutable records, model identity, and explicit blackout projection

**Files:**
- Create: `narrative_dynamics/narrative/observation_projection.py`
- Test: `tests/test_narrative_observation_projection.py`

**Interfaces:**
- Consumes: `stable_content_hash`, `measure_implementation`, `ImplementationAttestationUnavailable`, `StateCellRef`, `TypedValue`, `Entity`, and `WorldStepResult`.
- Produces: all seven public record/error classes plus a temporary runtime implementation of `project_world_observations()` that fully supports source-bound empty-model blackout and raises a typed execution error for non-empty models until Task 3.

- [ ] **Step 1: Add structural helpers and public record constructors**

Start `observation_projection.py` with:

```python
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
from narrative_dynamics.narrative.domain import DomainSpec
from narrative_dynamics.narrative.ir import (
    Entity,
    EntityRef,
    GenericNarrative,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.world import (
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
```

Implement `ObservationCapabilitySpec`, `ObserverProjectionSpec`, and containment exactly as:

```python
@dataclass(frozen=True)
class ObservationCapabilitySpec:
    state_variable: str
    subject_scope: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "state_variable",
            _text(self.state_variable, label="observation capability state variable"),
        )
        scope = _text(self.subject_scope, label="observation capability subject scope")
        if scope not in _SCOPES:
            raise ValueError("observation capability subject scope must be observer or any")
        object.__setattr__(self, "subject_scope", scope)

    def to_dict(self) -> dict[str, object]:
        return {
            "state_variable": self.state_variable,
            "subject_scope": self.subject_scope,
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())


def _capability_key(item: ObservationCapabilitySpec) -> tuple[str, str]:
    return (item.state_variable, item.subject_scope)


def _contains(
    read: ObservationCapabilitySpec,
    emit: ObservationCapabilitySpec,
) -> bool:
    if read.state_variable != emit.state_variable:
        return False
    return read.subject_scope == "any" or emit.subject_scope == "observer"
```

`ObserverProjectionSpec.__post_init__()` must canonicalize read/emit tuples, reject non-record entries, empties, exact duplicates, non-callables, and any emit capability not semantically contained by at least one read capability. Its `to_dict()` must bind `measure_implementation(self.projection_hook).manifest_identity()` and its `content_hash` must hash that dictionary.

Implement `ObservationFact` exactly with `equals`/`clear` shape rules and stable hashing.

Implement `ProjectedObservation` with structural-only validation: trimmed observer/channel, `ObservationFact`, non-negative step, valid source/spec hashes, `source_transition_hashes` as an exact tuple of unique valid hashes canonicalized lexically, plus stable `to_dict()`/`content_hash`.

Implement `ObservationProjectionModelSpec` with exact domain identity strings/hash, a tuple of `ObserverProjectionSpec`, uniqueness by `(observer_type, channel)`, canonical sort, and empty tuple allowed. Its content hash must bind every projection spec dictionary, including hook implementation identity.

Implement `ObservationProjectionResult` so observations are canonically sorted by:

```python
(
    observation.observer_id,
    observation.channel,
    observation.fact.cell.subject.entity_type,
    observation.fact.cell.subject.entity_id,
    observation.fact.cell.state_variable,
)
```

The constructor must require every observation to match the result's source world-step hash, source world-state hash, and step index. The result itself stores `model_id`, `model_hash`, exact source hashes, non-negative step index, and observations; direct construction remains data construction rather than source certification.

- [ ] **Step 2: Add attestation wrappers used by runtime identity**

Use typed wrappers so attestation failures do not leak from the public runtime function:

```python
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
```

- [ ] **Step 3: Implement the empty-model blackout path and typed non-empty boundary**

At this stage, `project_world_observations()` must validate only the argument record types, exact story/domain/model domain identity, and exact world-step source story/domain identity needed for a trustworthy blackout result. It must return a complete empty result when `model.projections == ()` and raise a typed marker for non-empty models so semantic execution tests remain RED:

```python
def project_world_observations(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    model: ObservationProjectionModelSpec,
) -> ObservationProjectionResult:
    if not isinstance(story, GenericNarrative):
        raise ObservationProjectionError("projection requires GenericNarrative")
    if not isinstance(domain, DomainSpec):
        raise ObservationProjectionError("projection requires DomainSpec")
    if not isinstance(world_step, WorldStepResult):
        raise ObservationProjectionError("projection requires WorldStepResult")
    if not isinstance(model, ObservationProjectionModelSpec):
        raise ObservationProjectionError(
            "projection requires ObservationProjectionModelSpec"
        )

    domain_identity = (domain.domain_id, domain.version, domain.content_hash)
    if (model.domain_id, model.domain_version, model.domain_spec_hash) != domain_identity:
        raise ObservationProjectionError(
            "observation projection model domain identity does not match DomainSpec"
        )
    if (
        world_step.next_state.domain_id,
        world_step.next_state.domain_version,
        world_step.next_state.domain_spec_hash,
    ) != domain_identity:
        raise ObservationProjectionError(
            "observation source world domain identity does not match DomainSpec"
        )
    if world_step.next_state.source_story_hash != story.content_hash:
        raise ObservationProjectionError(
            "observation source world does not match canonical narrative"
        )

    model_hash = _model_hash(model)
    if model.projections:
        raise ObservationProjectionError(
            "non-empty observation projection execution is not available in this stage"
        )
    return ObservationProjectionResult(
        model_id=model.model_id,
        model_hash=model_hash,
        source_world_step_hash=world_step.content_hash,
        source_world_state_hash=world_step.next_state.content_hash,
        step_index=world_step.next_state.step_index,
        observations=(),
    )
```

The marker text is deliberately stage-local and must disappear in Task 3; no final test may depend on it.

- [ ] **Step 4: Run the staged record/identity/blackout subset**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_record_constructors_fail_closed_and_public_records_are_data_not_certification \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_model_identity_binds_domain_capabilities_channel_and_hook_not_order \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_duplicate_declarations_and_semantic_containment_fail_closed \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_blackout_model_returns_empty_source_bound_result_without_hook_calls \
  -v
```

Expected: all four pass. The remaining non-empty runtime tests must still fail through the typed stage boundary, and the exact public-surface test must remain RED.

- [ ] **Step 5: Run the full dedicated test file to preserve staged RED**

Run:

```bash
python3 -m unittest tests.test_narrative_observation_projection -v
```

Expected: the four record/identity/blackout methods pass; all non-empty projection methods still fail because execution is intentionally unavailable. Any different failure is investigated before Task 3.

- [ ] **Step 6: Commit Task 2**

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: add observation projection records"
```

The commit must touch only `observation_projection.py`.

- [ ] **Step 7: Obtain exact-head CI evidence for the staged boundary**

Wait for the Draft PR workflow on this exact head. Required evidence:

- existing Lean and Python suites remain green except new runtime/public API REDs;
- the four Task 2 methods are green;
- no import failure remains for `observation_projection.py`;
- no public exports have been added.

---

### Task 3: Implement source validation, domain declarations, and capability-filtered hook execution

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`
- Test: `tests/test_narrative_observation_projection.py`

**Interfaces:**
- Consumes: Task 2 records and exact `WorldStepResult`/`WorldState`/`ActionTransitionRecord` payloads.
- Produces: pre-hook source validation, domain-dependent capability validation, canonical observer iteration, immutable prior/next read views, exact hook-shape validation, and typed hook errors. Fact truth acceptance is completed in Task 4.

- [ ] **Step 1: Add canonical state and transition helpers without importing private `world.py` helpers**

Implement local helpers so `world.py` remains untouched:

```python
def _entity_map(story: GenericNarrative) -> dict[str, Entity]:
    return {entity.id: entity for entity in story.entities}


def _validate_state_values(
    state: WorldState,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> None:
    for cell, value in state.values.items():
        if not isinstance(cell, StateCellRef) or not isinstance(value, TypedValue):
            raise ValueError("world state must map canonical cells to typed values")
        subject = entities.get(cell.subject.entity_id)
        if subject is None or subject.type_name != cell.subject.entity_type:
            raise ValueError("world state cell subject is not canonical")
        variable = domain._state_variable(cell.state_variable)
        if variable.subject_type != subject.type_name:
            raise ValueError("world state cell subject type does not match state variable")
        domain._value_type(variable.value_type).validate(value, entities)


def _record_key(record: ActionTransitionRecord) -> tuple[str, str, str]:
    return (
        record.actor_id,
        record.intent.decision_id,
        record.action.id,
    )


def _batch_hash(records: tuple[ActionTransitionRecord, ...]) -> str:
    ordered = tuple(sorted(records, key=_record_key))
    return stable_content_hash([record.to_dict() for record in ordered])
```

- [ ] **Step 2: Validate the supplied world step extensionally before any projection hook**

Implement `_validate_source_world_step(story, domain, world_step)` and wrap its `TypeError`/`ValueError` failures as `ObservationProjectionError` from the public function. It must perform every spec Section 16 check.

Canonical decision/action resolution must use the story, never trust copies:

```python
decision_by_id = {decision.id: decision for decision in story.decisions}
seen_actors: set[str] = set()
seen_writes: set[StateCellRef] = set()
next_values = dict(world_step.prior_state.values)

for record in sorted(world_step.transitions, key=_record_key):
    if record.prior_state_hash != world_step.prior_state.content_hash:
        raise ValueError("transition record does not bind exact prior state")
    decision = decision_by_id.get(record.intent.decision_id)
    if decision is None:
        raise ValueError("transition intent decision is not canonical")
    if record.actor_id != decision.actor_id:
        raise ValueError("transition actor does not match canonical decision")
    if record.actor_id in seen_actors:
        raise ValueError("world step contains duplicate actor transitions")
    seen_actors.add(record.actor_id)
    action_by_id = {action.id: action for action in decision.actions}
    action = action_by_id.get(record.intent.selected_action)
    if action is None or record.action != action:
        raise ValueError("transition action does not match canonical selected action")
    if (
        world_step.prior_state.source_at_time is not None
        and decision.logical_time > world_step.prior_state.source_at_time
    ):
        raise ValueError("transition decision occurs after source cutoff")

    local_writes: set[StateCellRef] = set()
    for operation in record.delta.operations:
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
        if cell in local_writes:
            raise ValueError("one transition writes one cell twice")
        if cell in seen_writes:
            raise ValueError("world step contains overlapping transition writes")
        local_writes.add(cell)
        seen_writes.add(cell)
        if operation.kind == "clear":
            if operation.value is not None:
                raise ValueError("clear transition operation contains a value")
            next_values.pop(cell, None)
        elif operation.kind == "set":
            if operation.value is None:
                raise ValueError("set transition operation requires a value")
            domain._value_type(variable.value_type).validate(operation.value, entities)
            next_values[cell] = operation.value
        else:
            raise ValueError("transition operation kind is unsupported")
```

After iteration, require:

```python
if next_values != dict(world_step.next_state.values):
    raise ValueError("world step next state is not the extensional transition result")
if world_step.next_state.transition_batch_hash != _batch_hash(world_step.transitions):
    raise ValueError("world step transition batch hash is inconsistent")
```

Also validate same domain/source identity across prior and next, exact `step_index + 1`, exact parent-state hash, and both state maps through `_validate_state_values()`.

- [ ] **Step 3: Validate all projection declarations before any hook executes**

Implement:

```python
def _validate_projection_declarations(
    domain: DomainSpec,
    model: ObservationProjectionModelSpec,
) -> None:
    declared_types = {item.name for item in domain.entity_types}
    for spec in model.projections:
        if spec.observer_type not in declared_types:
            raise ValueError("projection observer type is not declared")
        for capability in spec.read_capabilities + spec.emit_capabilities:
            variable = domain._state_variable(capability.state_variable)
            if (
                capability.subject_scope == "observer"
                and variable.subject_type != spec.observer_type
            ):
                raise ValueError(
                    "observer-scoped projection capability has incompatible subject type"
                )
```

Call this for all specs before the first hook. Also precompute every `_projection_hash(spec)` before hook iteration so implementation-attestation failure is pre-hook and typed.

- [ ] **Step 4: Build immutable read-capability-filtered views**

Implement exact capability matching:

```python
def _can_read(
    capability: ObservationCapabilitySpec,
    cell: StateCellRef,
    observer: Entity,
) -> bool:
    if capability.state_variable != cell.state_variable:
        return False
    if capability.subject_scope == "any":
        return True
    return cell.subject.entity_id == observer.id


def _visible_values(
    values: Mapping[StateCellRef, TypedValue],
    capabilities: tuple[ObservationCapabilitySpec, ...],
    observer: Entity,
) -> Mapping[StateCellRef, TypedValue]:
    visible = {
        cell: value
        for cell, value in values.items()
        if any(_can_read(capability, cell, observer) for capability in capabilities)
    }
    ordered = dict(sorted(visible.items(), key=lambda item: _cell_key(item[0])))
    return MappingProxyType(ordered)
```

The hook must receive no `WorldStepResult`, transition records, intents, claims, beliefs, goals, or hidden cells.

- [ ] **Step 5: Invoke hooks canonically and validate exact output container shape**

Replace the Task 2 stage boundary for non-empty models. Sort specs by `(observer_type, channel)` and observers by `entity.id`. For each pair, call:

```python
raw = spec.projection_hook(
    _visible_values(
        world_step.prior_state.values,
        spec.read_capabilities,
        observer,
    ),
    _visible_values(
        world_step.next_state.values,
        spec.read_capabilities,
        observer,
    ),
    observer,
    world_step.next_state.step_index,
)
```

Wrap arbitrary hook exceptions:

```python
try:
    raw = spec.projection_hook(...)
except Exception as error:
    raise ObservationProjectionError("observation projection hook failed") from error
```

Require `type(raw) is tuple`, every element `ObservationFact`, and no duplicate `fact.cell` within one invocation. For this task, pass the validated tuple to a private `_accept_facts(...)` function whose body raises `ObservationProjectionError("observation fact acceptance is unavailable in this stage")`. This preserves a clean staged RED for truth/provenance methods while allowing source/declaration/view/shape tests to go green.

- [ ] **Step 6: Run the Task 3 focused GREEN set**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_undeclared_observer_variable_and_incompatible_observer_scope_reject_before_hook \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_hook_sees_only_read_capability_cells_and_scope_is_exact \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_readable_nonemittable_and_bad_hook_return_shapes_reject_typed \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_source_identity_and_state_payload_reject_before_hook \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_transition_decision_action_actor_and_cutoff_reject_before_hook \
  tests.test_narrative_observation_projection.NarrativeObservationProjectionTests.test_extensional_forgery_duplicate_actor_and_write_collisions_reject_before_hook \
  -v
```

Expected: all six pass. Truth/provenance/per-agent methods remain RED only at `_accept_facts`.

- [ ] **Step 7: Run all dedicated tests and confirm only Task 4 semantics remain RED**

Run:

```bash
python3 -m unittest tests.test_narrative_observation_projection -v
```

Expected: Task 2 + Task 3 methods are green; only methods requiring accepted non-empty facts remain RED. The trust public-surface method remains RED.

- [ ] **Step 8: Commit Task 3**

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: validate observation projection sources"
```

- [ ] **Step 9: Obtain exact-head staged CI evidence**

Wait for the Draft PR workflow on this exact head. Confirm existing suites remain green and the remaining failures are confined to Task 4 fact acceptance/provenance plus the deferred public API gate.

---

### Task 4: Implement truth validation, engine-derived provenance, per-agent observations, and deterministic result identity

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`
- Test: `tests/test_narrative_observation_projection.py`

**Interfaces:**
- Consumes: Task 3 validated source world, canonical observer/spec pairs, immutable visible mappings, and exact tuples of `ObservationFact` proposals.
- Produces: complete semantic `project_world_observations()` behavior returning zero or more canonical `ProjectedObservation` values with engine-derived lineage.

- [ ] **Step 1: Add emit-capability and canonical fact-cell validation**

Implement:

```python
def _capability_covers_cell(
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


def _canonical_fact_cell(
    fact: ObservationFact,
    domain: DomainSpec,
    entities: Mapping[str, Entity],
) -> StateCellRef:
    subject = entities.get(fact.cell.subject.entity_id)
    if subject is None or subject.type_name != fact.cell.subject.entity_type:
        raise ValueError("observation fact subject is not canonical")
    variable = domain._state_variable(fact.cell.state_variable)
    if variable.subject_type != subject.type_name:
        raise ValueError("observation fact subject type does not match state variable")
    return StateCellRef(EntityRef(subject.id, subject.type_name), variable.name)
```

Before truth checking, require at least one `spec.emit_capabilities` entry to cover the canonical cell for this exact observer.

- [ ] **Step 2: Implement exact `equals` and explicit-write `clear` truth rules**

For equals:

```python
expected = world_step.next_state.values.get(cell)
if expected is None or fact.value != expected:
    raise ValueError("observation equals fact does not match exact post-step truth")
variable = domain._state_variable(cell.state_variable)
domain._value_type(variable.value_type).validate(fact.value, entities)
```

For clear, derive write records first and require post-step absence plus exactly one explicit clear write:

```python
def _records_writing_cell(
    world_step: WorldStepResult,
    entities: Mapping[str, Entity],
    cell: StateCellRef,
) -> tuple[ActionTransitionRecord, ...]:
    matches = []
    for record in world_step.transitions:
        for operation in record.delta.operations:
            subject = entities[operation.subject_id]
            written = StateCellRef(
                EntityRef(subject.id, subject.type_name),
                operation.state_variable,
            )
            if written == cell:
                matches.append(record)
                break
    return tuple(sorted(matches, key=_record_key))
```

Then:

```python
if fact.relation == "clear":
    if cell in world_step.next_state.values:
        raise ValueError("clear observation requires post-step absence")
    records = _records_writing_cell(world_step, entities, cell)
    clear_records = tuple(
        record
        for record in records
        if any(
            operation.kind == "clear"
            and operation.subject_id == cell.subject.entity_id
            and operation.state_variable == cell.state_variable
            for operation in record.delta.operations
        )
    )
    if len(clear_records) != 1:
        raise ValueError("clear observation requires one explicit current-step clear write")
```

- [ ] **Step 3: Derive provenance inside the engine and construct accepted observations**

Replace `_accept_facts` with a function returning `ProjectedObservation` values. The hook supplies only `ObservationFact`; derive all other fields:

```python
records = _records_writing_cell(world_step, entities, cell)
transition_hashes = tuple(
    sorted(record.content_hash for record in records)
)
accepted = ProjectedObservation(
    observer_id=observer.id,
    channel=spec.channel,
    fact=ObservationFact(cell, fact.relation, fact.value),
    step_index=world_step.next_state.step_index,
    source_world_state_hash=world_step.next_state.content_hash,
    source_world_step_hash=world_step.content_hash,
    source_transition_hashes=transition_hashes,
    projection_spec_hash=spec_hash,
)
```

A same-value `set` is included because `_records_writing_cell` tests the write set, not value difference. An unwritten persistent cell yields `transition_hashes == ()`.

- [ ] **Step 4: Finish canonical result construction and empty-fact behavior**

Aggregate all accepted observations across every canonical `(spec, observer)` invocation and construct:

```python
return ObservationProjectionResult(
    model_id=model.model_id,
    model_hash=model_hash,
    source_world_step_hash=world_step.content_hash,
    source_world_state_hash=world_step.next_state.content_hash,
    step_index=world_step.next_state.step_index,
    observations=tuple(accepted_observations),
)
```

Do not mutate any source object. A hook returning `()` contributes no observations but is still a valid invocation. Multiple channels for one observer type remain independent specs.

- [ ] **Step 5: Make all public-runtime failures typed**

The public function must wrap all source/declaration/fact/domain errors that are not already `ObservationProjectionError`:

```python
try:
    ...
except ObservationProjectionError:
    raise
except (TypeError, ValueError) as error:
    raise ObservationProjectionError(
        "observation projection validation failed"
    ) from error
```

Keep public record constructor errors as ordinary `TypeError`/`ValueError`; only runtime projection failures use `ObservationProjectionError`.

- [ ] **Step 6: Run all 19 dedicated semantic tests**

Run:

```bash
python3 -m unittest tests.test_narrative_observation_projection -v
```

Expected: all 19 methods pass.

- [ ] **Step 7: Run narrative regression tests before exposing public API**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_replay \
  tests.test_narrative_uncertain_belief \
  tests.test_narrative_decision \
  tests.test_narrative_intention \
  tests.test_narrative_world_transition \
  tests.test_narrative_compiler \
  tests.test_narrative_interventions \
  tests.test_narrative_knives_out \
  tests.test_narrative_the_matrix \
  tests.test_narrative_memento \
  -v
```

Expected: all pass. The exact public-surface test is intentionally still RED because Task 5 owns exports.

- [ ] **Step 8: Commit Task 4**

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: project runtime world observations"
```

- [ ] **Step 9: Obtain the semantic-head CI gate**

Wait for the Draft PR workflow on this exact head. Required evidence before Task 5:

- all 19 dedicated projection tests are green;
- all existing Python tests are green except `test_exact_public_surface_and_root_isolation`;
- that one API test fails only because the eight approved names are not exported;
- all Lean gates remain green.

No `__init__.py` change is allowed until this exact semantic-head evidence is observed.

---

### Task 5: Export the exact narrative API and close the full regression loop

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: complete Task 4 semantic implementation.
- Produces: exact eight-name narrative-scoped public API with root isolation preserved.

- [ ] **Step 1: Add only the approved imports to narrative `__init__.py`**

Add:

```python
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionError,
    ObservationProjectionModelSpec,
    ObservationProjectionResult,
    ObserverProjectionSpec,
    ProjectedObservation,
    project_world_observations,
)
```

Add exactly these names to `__all__` in an observation-projection group:

```python
    "ObservationCapabilitySpec",
    "ObserverProjectionSpec",
    "ObservationProjectionModelSpec",
    "ObservationFact",
    "ProjectedObservation",
    "ObservationProjectionResult",
    "ObservationProjectionError",
    "project_world_observations",
```

Do not edit root `narrative_dynamics/__init__.py`.

- [ ] **Step 2: Run the exact public-surface/root-isolation gate**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected: pass. Each of the eight names exists on `narrative_dynamics.narrative` and remains absent from root `narrative_dynamics`.

- [ ] **Step 3: Run the dedicated projection tests again on the exported tree**

Run:

```bash
python3 -m unittest tests.test_narrative_observation_projection -v
```

Expected: all 19 pass.

- [ ] **Step 4: Run the full Python suite**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: exit 0, no failures or errors. Baseline before this feature is 468 tests; with the 19 new methods the expected total is 487 tests unless unrelated base work has legitimately changed the count, in which case inspect the exact discovered delta rather than accepting a silent count drift.

- [ ] **Step 5: Commit the scoped export atomically**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative observation projection api"
```

This commit must touch only `narrative_dynamics/narrative/__init__.py`.

- [ ] **Step 6: Run final exact-head CI and inspect every workflow gate**

Wait for `proof` on the exact final feature head. Required success evidence:

- Lean/Python conformance vectors: success;
- full Lean build: success;
- Lean theorem suite: success;
- Python suite: all discovered tests pass, expected 487 on the unchanged base;
- `NarrativeDynamics/Tests/StoryState.lean`: success;
- `NarrativeDynamics/Tests/Testimony.lean`: success.

Fetch the full job log and verify `test_exact_public_surface_and_root_isolation` and all `NarrativeObservationProjectionTests` are explicitly `ok`; do not infer from workflow status alone.

- [ ] **Step 7: Verify exact final diff boundary**

Compare the exact base `44f46a13d228afb48ca6938b51a6a8606c52d984` to the final feature head. The changed-file set must be exactly:

```text
docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_observation_projection.py
tests/test_narrative_trust_api.py
```

Any seventh path is a scope failure and must be investigated before integration.

- [ ] **Step 8: Update the PR body with exact RED/GREEN evidence and mark Ready for review**

Replace the Draft PR body with:

- exact base SHA;
- test-only RED SHA and workflow run;
- staged Task 2 and Task 3 heads/runs;
- semantic Task 4 head/run proving projection tests green while export alone is RED;
- final head and final successful workflow run;
- exact six-path scope statement;
- explicit statement that authored Observation/replay, World Transition V1, package root, DomainSpec, GenericNarrative IR, and Lean sources are unchanged.

Then mark PR Ready for review. Do not merge.

## Completion Evidence Checklist

Before saying Observation Projection V1 is complete, verify all of these on the exact final feature head:

- [ ] 19 dedicated semantic projection tests pass.
- [ ] Exact eight-name narrative public-surface/root-isolation test passes.
- [ ] Full Python suite passes with no silent discovery loss.
- [ ] Lean/Python conformance gate passes.
- [ ] Full Lean build passes.
- [ ] Lean theorem suite passes.
- [ ] StoryState theorem gate passes.
- [ ] Testimony theorem gate passes.
- [ ] Final diff is exactly six approved paths.
- [ ] PR head equals the exact CI-tested head.
- [ ] PR remains unmerged until explicit user instruction.
