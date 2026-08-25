# Narrative Observation Projection V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, typed, capability-limited, truth-validated, provenance-linked runtime projection from one validated `WorldStepResult` to per-agent `ProjectedObservation` records.

**Architecture:** Add one sidecar module, `narrative_dynamics/narrative/observation_projection.py`. Hooks receive only immutable read-capability-filtered prior/next world mappings and return `ObservationFact` proposals; the runtime validates source-world consistency, declaration/capability boundaries, exact post-step truth, and derives all lineage. Authored `GenericNarrative.Observation`, replay, belief, decision, intention, and World Transition V1 remain unchanged.

**Tech Stack:** Python standard library (`dataclasses`, `MappingProxyType`, `unittest`), existing `stable_content_hash`, `measure_implementation`, Generic Narrative IR/domain types, and World Transition V1.

**Spec:** `docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md`

## Global Constraints

- Exact base: `44f46a13d228afb48ca6938b51a6a8606c52d984` on `proof/narrative-dynamics-v0`.
- Work only on `work/narrative-observation-projection-v1`.
- Runtime `WorldState.step_index` must remain distinct from authored `logical_time`.
- Runtime percepts must not mutate or masquerade as authored `GenericNarrative.Observation` records.
- Hooks must never receive undeclared objective cells through the framework API.
- `equals` facts must exactly match `next_state`; `clear` facts require post-step absence plus an explicit current-step clear write.
- Transition provenance follows actual write-set membership; same-value `set` writes still bind provenance.
- `ProjectedObservation` and `ObservationProjectionResult` are public data records, not independent provenance certificates.
- Empty `ObservationProjectionModelSpec.projections` is valid and represents blackout.
- No RNG, noisy sensors, testimony generation, scheduler, belief bridge, event-occurrence observations, IR changes, or DomainSpec identity changes.
- Final diff must contain exactly six paths: approved spec, this plan, `observation_projection.py`, narrative `__init__.py`, dedicated projection tests, and trust API test.
- Follow test-only RED -> exact-head CI RED -> staged GREEN commits -> final export -> full exact-head CI.
- Never merge to `proof/narrative-dynamics-v0` without explicit user instruction.

## File Map

- `docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md`: approved architecture.
- `docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md`: this plan.
- `narrative_dynamics/narrative/observation_projection.py`: all V1 records/runtime.
- `narrative_dynamics/narrative/__init__.py`: final eight-name scoped export only.
- `tests/test_narrative_observation_projection.py`: all V1 semantic fixtures/tests.
- `tests/test_narrative_trust_api.py`: exact public API/root-isolation lock.

---

### Task 1: Commit the complete test-only RED contract

**Files:**
- Create: `tests/test_narrative_observation_projection.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: current Domain/IR/World Transition V1 APIs.
- Produces: 19 projection test methods covering all 36 required RED invariants plus the exact eight-name public-surface RED.

- [ ] **Step 1: Create the guarded import and exact local fixture helpers**

Use this import gate at the top of `tests/test_narrative_observation_projection.py`:

```python
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

Use only module-level callable classes for hooks so existing implementation attestation can measure them. Define these exact positive witnesses:

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
        return () if value is None else (ObservationFact(cell, "equals", value),)


class OwnLocationHookV2:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(
            EntityRef(observer.id, observer.type_name),
            "agent.location",
        )
        if cell not in next_visible:
            return ()
        return (ObservationFact(cell, "equals", next_visible[cell]),)


class RaisingProjectionHook:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        raise RuntimeError("projection boom")
```

Define one local domain/story fixture with canonical entities `a1:Agent`, `a2:Agent`, `svc:Service`, `room:Room`; value types `PhaseState`, `LocationState`, `HealthState`, `AlertState`, `LightState`; and state variables:

```python
StateVariableSpec("agent.phase", "Agent", "PhaseState")
StateVariableSpec("agent.location", "Agent", "LocationState")
StateVariableSpec("service.health", "Service", "HealthState")
StateVariableSpec("service.alert", "Service", "AlertState")
StateVariableSpec("room.lighting", "Room", "LightState")
```

Seed authored state through ordinary `EventTypeSpec` hooks, then declare two decisions owned by different agents so valid multi-actor `WorldStepResult` values can be produced through existing `advance_world_step()`. Include one action that sets `a1.agent.phase` and one action that clears `svc.service.alert`. Provide helper functions with these signatures:

```python
def make_projection_domain() -> DomainSpec: ...
def make_projection_story(domain: DomainSpec) -> GenericNarrative: ...
def make_transition_model(domain: DomainSpec) -> WorldTransitionModelSpec: ...
def make_world_step(
    *,
    phase: str = "active",
    clear_alert: bool = False,
    source_at_time: int | None = None,
) -> tuple[DomainSpec, GenericNarrative, WorldStepResult]: ...
def projection_model(
    domain: DomainSpec,
    *specs: ObserverProjectionSpec,
) -> ObservationProjectionModelSpec: ...
```

The bodies must construct real current production records; do not mock `WorldStepResult` on positive paths.

Add:

```python
class NarrativeObservationProjectionTests(unittest.TestCase):
    def require_projection(self) -> None:
        if _PROJECTION_IMPORT_ERROR is not None:
            self.fail(
                "narrative observation projection boundary is missing: "
                f"{_PROJECTION_IMPORT_ERROR}"
            )
```

- [ ] **Step 2: Add records/identity/blackout tests with concrete assertions**

Implement these four methods:

```python
def test_record_constructors_fail_closed_and_public_records_are_data_not_certification(self):
    self.require_projection()
    with self.assertRaises((TypeError, ValueError)):
        ObservationCapabilitySpec("", "any")
    with self.assertRaises((TypeError, ValueError)):
        ObservationCapabilitySpec("agent.location", "missing")
    cell = StateCellRef(EntityRef("a1", "Agent"), "agent.location")
    with self.assertRaises((TypeError, ValueError)):
        ObservationFact(cell, "not_equals", TypedValue("LocationState", "hall"))
    with self.assertRaises((TypeError, ValueError)):
        ObservationFact(cell, "equals", None)
    with self.assertRaises((TypeError, ValueError)):
        ObservationFact(cell, "clear", TypedValue("LocationState", "hall"))
    fake_hash = "sha256:" + "1" * 64
    fact = ObservationFact(cell, "equals", TypedValue("LocationState", "hall"))
    record = ProjectedObservation(
        observer_id="a1",
        channel="vision",
        fact=fact,
        step_index=1,
        source_world_state_hash=fake_hash,
        source_world_step_hash=fake_hash,
        source_transition_hashes=(),
        projection_spec_hash=fake_hash,
    )
    self.assertEqual(record.source_world_step_hash, fake_hash)
    self.assertEqual(record.content_hash, stable_content_hash(record.to_dict()))


def test_model_identity_binds_domain_capabilities_channel_and_hook_not_order(self):
    self.require_projection()
    domain = make_projection_domain()
    read = (
        ObservationCapabilitySpec("room.lighting", "any"),
        ObservationCapabilitySpec("agent.location", "any"),
    )
    emit = (ObservationCapabilitySpec("agent.location", "any"),)
    first = ObserverProjectionSpec("Agent", "vision", read, emit, OwnLocationHook())
    reordered = ObserverProjectionSpec(
        "Agent", "vision", tuple(reversed(read)), emit, OwnLocationHook()
    )
    m1 = projection_model(domain, first)
    m2 = projection_model(domain, reordered)
    self.assertEqual(m1.content_hash, m2.content_hash)
    changed_channel = projection_model(
        domain,
        ObserverProjectionSpec("Agent", "status", read, emit, OwnLocationHook()),
    )
    changed_hook = projection_model(
        domain,
        ObserverProjectionSpec("Agent", "vision", read, emit, OwnLocationHookV2()),
    )
    self.assertNotEqual(m1.content_hash, changed_channel.content_hash)
    self.assertNotEqual(m1.content_hash, changed_hook.content_hash)
    self.assertNotEqual(
        m1.content_hash,
        replace(m1, domain_spec_hash="sha256:" + "2" * 64).content_hash,
    )


def test_duplicate_declarations_and_semantic_containment_fail_closed(self):
    self.require_projection()
    cap_any = ObservationCapabilitySpec("agent.location", "any")
    cap_observer = ObservationCapabilitySpec("agent.location", "observer")
    with self.assertRaises((TypeError, ValueError)):
        ObserverProjectionSpec(
            "Agent", "vision", (cap_any, cap_any), (cap_observer,), OwnLocationHook()
        )
    ObserverProjectionSpec(
        "Agent", "vision", (cap_any,), (cap_any, cap_observer), OwnLocationHook()
    )
    ObserverProjectionSpec(
        "Agent", "vision", (cap_observer,), (cap_observer,), OwnLocationHook()
    )
    with self.assertRaises((TypeError, ValueError)):
        ObserverProjectionSpec(
            "Agent", "vision", (cap_observer,), (cap_any,), OwnLocationHook()
        )
    domain = make_projection_domain()
    spec = ObserverProjectionSpec(
        "Agent", "vision", (cap_any,), (cap_any,), OwnLocationHook()
    )
    with self.assertRaises((TypeError, ValueError)):
        projection_model(domain, spec, spec)


def test_blackout_model_returns_empty_source_bound_result_without_hook_calls(self):
    self.require_projection()
    domain, story, world_step = make_world_step()
    model = ObservationProjectionModelSpec(
        "blackout", "1", domain.domain_id, domain.version, domain.content_hash, ()
    )
    result = project_world_observations(story, domain, world_step, model)
    self.assertEqual(result.observations, ())
    self.assertEqual(result.model_hash, model.content_hash)
    self.assertEqual(result.source_world_step_hash, world_step.content_hash)
    self.assertEqual(result.source_world_state_hash, world_step.next_state.content_hash)
    self.assertEqual(result.step_index, world_step.next_state.step_index)
```

Expand the first method with subtests for malformed hashes, tuple fields, duplicate transition hashes, malformed observer/channel/model strings, and malformed step indices so all constructor-shape cases in spec invariant 1 are covered.

- [ ] **Step 3: Add domain/visibility tests**

Implement:

```python
def test_undeclared_observer_variable_and_incompatible_observer_scope_reject_before_hook(self):
    self.require_projection()
    domain, story, world_step = make_world_step()
    for spec in (
        ObserverProjectionSpec(
            "MissingAgent", "vision",
            (ObservationCapabilitySpec("agent.location", "any"),),
            (ObservationCapabilitySpec("agent.location", "any"),),
            RecordViewsHook(),
        ),
        ObserverProjectionSpec(
            "Agent", "vision",
            (ObservationCapabilitySpec("missing.state", "any"),),
            (ObservationCapabilitySpec("missing.state", "any"),),
            RecordViewsHook(),
        ),
        ObserverProjectionSpec(
            "Agent", "vision",
            (ObservationCapabilitySpec("service.health", "observer"),),
            (ObservationCapabilitySpec("service.health", "observer"),),
            RecordViewsHook(),
        ),
    ):
        hook = spec.projection_hook
        with self.assertRaises(ObservationProjectionError):
            project_world_observations(story, domain, world_step, projection_model(domain, spec))
        self.assertEqual(hook.calls, [])


def test_hook_sees_only_read_capability_cells_and_scope_is_exact(self):
    self.require_projection()
    domain, story, world_step = make_world_step()
    own_hook = RecordViewsHook(())
    own = ObserverProjectionSpec(
        "Agent", "vision",
        (ObservationCapabilitySpec("agent.location", "observer"),),
        (ObservationCapabilitySpec("agent.location", "observer"),),
        own_hook,
    )
    project_world_observations(story, domain, world_step, projection_model(domain, own))
    self.assertEqual(len(own_hook.calls), 2)
    for prior, next_values, observer, _ in own_hook.calls:
        expected = StateCellRef(EntityRef(observer.id, "Agent"), "agent.location")
        self.assertEqual(set(prior), {expected})
        self.assertEqual(set(next_values), {expected})
        self.assertNotIn(StateCellRef(EntityRef("svc", "Service"), "service.health"), next_values)
    any_hook = RecordViewsHook(())
    any_spec = ObserverProjectionSpec(
        "Agent", "vision",
        (ObservationCapabilitySpec("agent.location", "any"),),
        (ObservationCapabilitySpec("agent.location", "observer"),),
        any_hook,
    )
    project_world_observations(story, domain, world_step, projection_model(domain, any_spec))
    for _, next_values, _, _ in any_hook.calls:
        self.assertEqual(
            {cell.subject.entity_id for cell in next_values},
            {"a1", "a2"},
        )
```

The hook class used for immutability must attempt assignment to each supplied mapping and record that Python raises `TypeError`; assert both prior and next mappings are immutable.

- [ ] **Step 4: Add hook-shape and emit-capability tests**

Use one hook class per invalid return shape (`list`, generator, mapping, scalar, tuple containing a string) and one hook returning duplicate facts for the same cell. Assert every call raises `ObservationProjectionError`.

For readable-but-not-emittable:

```python
lighting = StateCellRef(EntityRef("room", "Room"), "room.lighting")
lighting_value = world_step.next_state.values[lighting]
hook = RecordViewsHook((ObservationFact(lighting, "equals", lighting_value),))
spec = ObserverProjectionSpec(
    "Agent",
    "vision",
    (
        ObservationCapabilitySpec("room.lighting", "any"),
        ObservationCapabilitySpec("agent.location", "any"),
    ),
    (ObservationCapabilitySpec("agent.location", "any"),),
    hook,
)
with self.assertRaises(ObservationProjectionError):
    project_world_observations(story, domain, world_step, projection_model(domain, spec))
self.assertGreater(len(hook.calls), 0)
```

Also assert `RaisingProjectionHook` is wrapped as `ObservationProjectionError`, not leaked as `RuntimeError`.

- [ ] **Step 5: Add truth/cell tests**

Implement exact-value success/failure, explicit-clear success/absence-only failure, and invalid canonical-cell cases. Use `ObservationFact` values that differ by one enum value or one `TypedValue.type_name` so failures are attributable to truth/type checks rather than constructor failure.

Required assertions:

```python
result = project_world_observations(story, domain, world_step, model)
self.assertEqual(result.observations[0].fact.value, expected_value)

with self.assertRaises(ObservationProjectionError):
    project_world_observations(story, domain, world_step, fabricated_model)
```

For clear, assert accepted provenance equals the exact transition record hash that contains `StateDeltaOp(kind="clear", subject_id="svc", state_variable="service.alert")`; construct a second valid world step where `service.alert` is absent without a current-step clear write and assert the same proposed clear fact rejects.

- [ ] **Step 6: Add per-agent/provenance/determinism tests**

Implement five methods:

```python
def test_two_observers_can_receive_different_percepts_from_same_world_step(self): ...
def test_multiple_channels_and_empty_fact_tuple_are_supported(self): ...
def test_transition_provenance_binds_written_cells_including_same_value_set(self): ...
def test_unwritten_persistent_value_has_empty_transition_provenance_and_exact_source_hashes(self): ...
def test_runtime_derives_step_and_lineage_and_hook_cannot_supply_it(self): ...
```

Use these exact assertions across them:

```python
self.assertEqual(
    {obs.observer_id: obs.fact.value.value for obs in result.observations},
    {"a1": "hall", "a2": "vault"},
)
self.assertTrue(all(obs.source_world_step_hash == world_step.content_hash for obs in result.observations))
self.assertTrue(all(obs.source_world_state_hash == world_step.next_state.content_hash for obs in result.observations))
self.assertTrue(all(obs.step_index == world_step.next_state.step_index for obs in result.observations))
```

For same-value `set`, create a world step whose action writes `a1.agent.phase="ready"` when prior already contains `"ready"`; assert `source_transition_hashes == (writing_record.content_hash,)`.

For unwritten persistence, observe `room.lighting` while no transition writes it and assert `source_transition_hashes == ()`.

For multiple channels, use two `ObserverProjectionSpec` values with the same `observer_type="Agent"` and channels `vision` and `status`; one hook returns a fact and the other returns `()`. Assert both hooks are invoked once per Agent, only the emitting channel appears in result observations, and channel is always the spec channel.

- [ ] **Step 7: Add pre-hook source validation and mutation/order tests**

Implement four methods:

```python
def test_source_identity_and_state_payload_reject_before_hook(self): ...
def test_transition_decision_action_actor_and_cutoff_reject_before_hook(self): ...
def test_extensional_forgery_duplicate_actor_and_write_collisions_reject_before_hook(self): ...
def test_projection_order_hash_and_inputs_are_immutable(self): ...
```

For every forged source case, use `RecordViewsHook(())`, call runtime inside `assertRaises(ObservationProjectionError)`, then assert `hook.calls == []`.

Forge only public-record-valid but semantically inconsistent payloads using `dataclasses.replace`: wrong story/domain hash; unknown canonical cell subject; wrong typed state value; next-state step/parent mismatch; transition intent decision/action mismatch; record actor mismatch; cutoff earlier than selected decision; duplicate actor records; duplicate writes within one `StateDelta`; cross-record write collision; and `next_state.values` that do not equal applying the supplied deltas. When a dependent hash must remain syntactically valid, recompute it with `stable_content_hash` rather than inserting malformed text.

For order/mutation, snapshot:

```python
before = (
    story.to_dict(),
    domain.to_dict(),
    world_step.to_dict(),
    world_step.prior_state.to_dict(),
    world_step.next_state.to_dict(),
)
```

Run equivalent models with reversed spec/fact order, then assert equal `result.content_hash`, equal canonical observation order, and exact equality of all five snapshots after projection.

These 19 test methods jointly cover design invariants 1–36.

- [ ] **Step 8: Extend only the expected narrative public API set**

Insert exactly:

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

into `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`. Do not edit production `__init__.py`.

- [ ] **Step 9: Verify syntax then exact RED**

Run:

```bash
python3 -m py_compile tests/test_narrative_observation_projection.py tests/test_narrative_trust_api.py
python3 -m unittest \
  tests.test_narrative_observation_projection \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected: syntax exit 0; all 19 new methods fail only through `require_projection()` because the module is absent; the API test fails only on the eight missing names.

- [ ] **Step 10: Commit test-only RED and obtain exact-head CI evidence**

```bash
git add tests/test_narrative_observation_projection.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative observation projection v1"
```

Open a Draft PR to `proof/narrative-dynamics-v0`. The RED commit must contain exactly these two test files. Exact-head `proof` CI must show Lean gates and all pre-existing Python tests green, with failures confined to the 19 missing projection tests plus the eight-name public-surface expectation.

Do not begin Task 2 until that RED is observed.

---

### Task 2: Implement records, identity, and explicit blackout

**Files:**
- Create: `narrative_dynamics/narrative/observation_projection.py`

**Interfaces:**
- Produces: `ObservationCapabilitySpec`, `ObserverProjectionSpec`, `ObservationProjectionModelSpec`, `ObservationFact`, `ProjectedObservation`, `ObservationProjectionResult`, `ObservationProjectionError`, plus a blackout-complete `project_world_observations()` and a typed non-empty stage boundary.

- [ ] **Step 1: Add structural helpers and record classes**

Start with:

```python
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_SCOPES = frozenset({"observer", "any"})
_RELATIONS = frozenset({"equals", "clear"})

class ObservationProjectionError(ValueError):
    """A runtime world step could not be projected safely."""

def _text(value: object, *, label: str) -> str: ...
def _hash(value: object, *, label: str) -> str: ...
def _step_index(value: object, *, label: str) -> int: ...
def _cell_key(cell: StateCellRef) -> tuple[str, str, str]: ...
```

Implement `_text`, `_hash`, and `_step_index` with the exact same trimmed-string/hash/non-negative-integer rules used in `world.py`.

Implement:

```python
@dataclass(frozen=True)
class ObservationCapabilitySpec:
    state_variable: str
    subject_scope: str
```

with exact scopes `observer|any`, canonical `to_dict()`, and `content_hash`.

Containment must be:

```python
def _contains(read, emit):
    return (
        read.state_variable == emit.state_variable
        and (read.subject_scope == "any" or emit.subject_scope == "observer")
    )
```

Implement `ObserverProjectionSpec` to require non-empty unique read/emit tuples, semantic containment for every emit entry, callable hook, canonical capability ordering, and hook identity through `measure_implementation(...).manifest_identity()`.

Implement `ObservationFact` with only `equals|clear`; equals requires `TypedValue`, clear requires `None`.

Implement `ProjectedObservation` with structural-only hash/type checks and lexically canonical unique `source_transition_hashes`.

Implement `ObservationProjectionModelSpec` with exact domain triple, uniqueness by `(observer_type, channel)`, canonical spec ordering, and empty projections allowed.

Implement `ObservationProjectionResult` with canonical observation ordering and checks that every contained observation matches the result's source step hash, source state hash, and step index.

- [ ] **Step 2: Add attestation wrappers**

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

- [ ] **Step 3: Implement the complete blackout path and typed non-empty stage boundary**

For all calls validate argument record types, `validate_narrative(story, domain)`, model/domain identity, world-step prior/next domain triple, source story hash, source step relation, parent hash, and source hashes needed to construct a source-bound result. For `model.projections == ()`, return:

```python
ObservationProjectionResult(
    model_id=model.model_id,
    model_hash=_model_hash(model),
    source_world_step_hash=world_step.content_hash,
    source_world_state_hash=world_step.next_state.content_hash,
    step_index=world_step.next_state.step_index,
    observations=(),
)
```

For non-empty projections raise `ObservationProjectionError("non-empty projection execution is unavailable in this stage")`.

- [ ] **Step 4: Verify staged GREEN/RED and commit**

Run the four Task 1 record/identity/containment/blackout methods explicitly; they must pass. Run the whole projection test module; all non-empty runtime methods must still fail at the typed stage boundary, not from imports or constructors.

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: add observation projection records"
```

Obtain exact-head CI evidence for this staged boundary before Task 3.

---

### Task 3: Implement source validation, declarations, filtered views, hook shape, and emit boundary

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`

**Interfaces:**
- Produces: full pre-hook source validation; domain-dependent declaration checks; immutable read views; canonical observer/spec iteration; exact hook-shape validation; syntactic emit-capability rejection. Exact canonical-cell truth and provenance remain Task 4.

- [ ] **Step 1: Validate public `WorldStepResult` extensionally without importing private `world.py` helpers**

Implement local `_record_key`, `_batch_hash`, `_validate_state_values`, and `_validate_source_world_step`.

`_validate_source_world_step` must:

```python
validate_narrative(story, domain)
entities = {entity.id: entity for entity in story.entities}
```

Then require prior/next exact domain and story identity; validate every state cell/value against canonical entities/domain; require `next.step_index == prior.step_index + 1` and exact parent hash; resolve every transition intent back to canonical `Decision` and selected `ActionOption`; require record actor equality; enforce source cutoff; reject duplicate actors; validate delta subjects/state variables/typed set values; reject duplicate per-record writes and cross-record collisions; apply every validated delta to a copy of prior values; require exact equality with `next_state.values`; and require:

```python
world_step.next_state.transition_batch_hash == stable_content_hash(
    [record.to_dict() for record in sorted(world_step.transitions, key=_record_key)]
)
```

All failures must occur before projection hooks run.

- [ ] **Step 2: Validate domain-dependent projection declarations before hooks**

For every spec before invocation require declared `observer_type`, declared state variables, and for every `subject_scope="observer"` capability require:

```python
domain._state_variable(capability.state_variable).subject_type == spec.observer_type
```

Precompute `_projection_hash(spec)` for every spec before any hook so attestation failure is pre-hook.

- [ ] **Step 3: Build immutable read views**

Implement:

```python
def _can_read(capability, cell, observer):
    if capability.state_variable != cell.state_variable:
        return False
    return (
        capability.subject_scope == "any"
        or cell.subject.entity_id == observer.id
    )


def _visible_values(values, capabilities, observer):
    visible = {
        cell: value
        for cell, value in values.items()
        if any(_can_read(cap, cell, observer) for cap in capabilities)
    }
    return MappingProxyType(
        dict(sorted(visible.items(), key=lambda item: _cell_key(item[0])))
    )
```

No other world-step/cognitive object is passed to hooks.

- [ ] **Step 4: Invoke specs/observers canonically and validate exact hook shape**

Sort specs by `(observer_type, channel)` and matching canonical entities by `id`. Invoke each hook once with filtered prior/next mappings, canonical `Entity`, and `world_step.next_state.step_index`.

Require `type(raw) is tuple`, all items are `ObservationFact`, and no duplicate `fact.cell` within the invocation. Wrap arbitrary hook exceptions as `ObservationProjectionError`.

- [ ] **Step 5: Enforce emit capability now, before the Task 4 fact-truth stage**

This stage owns the readable-but-nonemittable boundary so its test can become genuinely GREEN:

```python
def _emit_covers(capability, cell, observer):
    if capability.state_variable != cell.state_variable:
        return False
    return capability.subject_scope == "any" or cell.subject.entity_id == observer.id

for fact in facts:
    if not any(
        _emit_covers(capability, fact.cell, observer)
        for capability in spec.emit_capabilities
    ):
        raise ObservationProjectionError(
            "observation fact is outside emit capability"
        )
```

After this check, call a private `_accept_facts(...)` stage marker that raises `ObservationProjectionError("fact truth acceptance is unavailable in this stage")` for non-empty fact tuples. Empty tuples succeed and contribute no observations.

- [ ] **Step 6: Verify Task 3 staged GREEN and commit**

These methods must pass now: undeclared declaration pre-hook; exact read scopes/hidden cells; bad hook return shapes; readable-but-nonemittable; source identity/state forgery; transition decision/action/actor/cutoff forgery; extensional/duplicate-actor/write-collision forgery. Methods requiring accepted non-empty facts remain RED only at `_accept_facts`.

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: validate observation projection sources"
```

Obtain exact-head CI evidence with only Task 4 semantics plus public API still RED.

---

### Task 4: Implement canonical fact truth, engine-derived provenance, and deterministic results

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`

**Interfaces:**
- Produces: complete non-empty projection semantics.

- [ ] **Step 1: Canonicalize and validate fact cells**

Resolve `fact.cell.subject` against canonical story entities; require exact entity type; require declared state variable and compatible subject type. Any failure becomes `ObservationProjectionError` through the public runtime wrapper.

- [ ] **Step 2: Enforce exact truth**

For `equals`:

```python
expected = world_step.next_state.values.get(cell)
if expected is None or fact.value != expected:
    raise ValueError("equals observation does not match post-step truth")
variable = domain._state_variable(cell.state_variable)
domain._value_type(variable.value_type).validate(fact.value, entities)
```

For `clear`, require cell absent from `next_state` and exactly one current transition with matching `StateDeltaOp(kind="clear", subject_id=..., state_variable=...)`. Absence without explicit clear rejects.

- [ ] **Step 3: Derive write-set provenance inside the engine**

Implement `_records_writing_cell` by scanning every transition delta and matching canonical `StateCellRef`. Build:

```python
source_transition_hashes = tuple(
    sorted(record.content_hash for record in writing_records)
)
```

This includes same-value `set` writes. Unwritten persistent observations get `()`.

Construct every accepted record only inside runtime:

```python
ProjectedObservation(
    observer_id=observer.id,
    channel=spec.channel,
    fact=canonical_fact,
    step_index=world_step.next_state.step_index,
    source_world_state_hash=world_step.next_state.content_hash,
    source_world_step_hash=world_step.content_hash,
    source_transition_hashes=source_transition_hashes,
    projection_spec_hash=spec_hash,
)
```

- [ ] **Step 4: Construct one canonical atomic result**

Aggregate all accepted observations and return `ObservationProjectionResult` with exact model/source hashes and next-state step index. Fact/spec/observer input order must not affect output order or result hash. Do not mutate any source record or caller container.

- [ ] **Step 5: Make runtime failures typed and run semantic GREEN**

Public record constructors retain ordinary `TypeError`/`ValueError`. `project_world_observations()` must rethrow existing `ObservationProjectionError` and wrap internal `TypeError`/`ValueError` as `ObservationProjectionError` with exception chaining.

Run:

```bash
python3 -m unittest tests.test_narrative_observation_projection -v
```

Expected: all 19 projection methods pass.

Then run narrative regressions:

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

Expected: all pass. Exact public-surface test remains the sole deliberate projection-related RED.

- [ ] **Step 6: Commit semantic GREEN and obtain the export-only RED CI witness**

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: project runtime world observations"
```

On exact-head CI require all 19 projection tests green and the only feature-related failure to be `test_exact_public_surface_and_root_isolation` because the eight exports are still absent. Lean gates must remain green.

---

### Task 5: Export the exact API and close full regression

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

**Interfaces:**
- Produces: exactly eight new narrative-scoped names; root isolation preserved.

- [ ] **Step 1: Add only the approved scoped imports and `__all__` names**

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

Add exactly:

```python
"ObservationCapabilitySpec"
"ObserverProjectionSpec"
"ObservationProjectionModelSpec"
"ObservationFact"
"ProjectedObservation"
"ObservationProjectionResult"
"ObservationProjectionError"
"project_world_observations"
```

to narrative `__all__`; do not edit root `narrative_dynamics/__init__.py`.

- [ ] **Step 2: Run exact API, dedicated semantics, then full Python suite**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
python3 -m unittest tests.test_narrative_observation_projection -v
python3 -m unittest discover -s tests -v
```

Expected: API/root-isolation passes; all 19 projection tests pass; full suite exits 0. Baseline is 468 tests, so an unchanged base plus 19 methods should discover 487; inspect any count drift explicitly.

- [ ] **Step 3: Commit export-only change**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative observation projection api"
```

The commit must touch only narrative `__init__.py`.

- [ ] **Step 4: Run and inspect final exact-head `proof` CI**

Require success for Lean/Python conformance, full Lean 8687-style build gate, Lean theorem suite, full Python discovery, StoryState theorem gate, and Testimony theorem gate. Fetch the full job log and explicitly confirm all `NarrativeObservationProjectionTests` plus `test_exact_public_surface_and_root_isolation` are `ok`.

- [ ] **Step 5: Verify exact final six-path diff**

Compare base `44f46a13d228afb48ca6938b51a6a8606c52d984` to final feature head. Changed files must be exactly:

```text
docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_observation_projection.py
tests/test_narrative_trust_api.py
```

Any additional path stops integration.

- [ ] **Step 6: Update Draft PR evidence and mark Ready for review**

Record exact base, test-only RED head/run, Task 2 and Task 3 staged heads/runs, Task 4 semantic head/run proving export-only RED, final head/run, and exact six-path scope. State explicitly that GenericNarrative IR, DomainSpec, replay, belief/decision/intention, World Transition V1, root package, movie fixtures, runtime registry, observational protocol, prison models, and Lean sources are unchanged. Mark Ready for review; do not merge.

## Completion Evidence Checklist

- [ ] All 19 projection semantic methods pass on exact final head.
- [ ] Exact eight-name narrative API/root-isolation test passes.
- [ ] Full Python suite passes with no silent discovery loss.
- [ ] Lean/Python conformance passes.
- [ ] Full Lean build passes.
- [ ] Lean theorem suite passes.
- [ ] StoryState theorem gate passes.
- [ ] Testimony theorem gate passes.
- [ ] Final diff contains exactly six approved paths.
- [ ] PR head equals exact CI-tested head.
- [ ] PR remains unmerged until explicit user instruction.
