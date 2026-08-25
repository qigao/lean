# Narrative Observation Projection V1 Design

Status: approved in-chat architecture; implementation not started

Tracking issue: #27 — close the cognitive multi-agent narrative simulation loop

Base branch: `proof/narrative-dynamics-v0`

Exact base commit: `44f46a13d228afb48ca6938b51a6a8606c52d984`

Feature branch: `work/narrative-observation-projection-v1`

## 1. Purpose

World Transition V1 closed the action-to-world half-step:

```text
X_t + {A_t^i} -> {Delta_t^i} -> X_{t+1}
```

Observation Projection V1 closes the next explicit runtime boundary:

```text
X_{t+1} -> {O_{t+1}^i}
```

The increment defines a deterministic, typed, capability-limited, provenance-linked projection from one completed `WorldStepResult` to per-agent runtime percepts.

It stops at the percept boundary. It does not update beliefs, schedule another round, or mutate the authored canonical story.

## 2. Why this is a separate sidecar

The existing `GenericNarrative.Observation` is an authored-story record. It points to an authored `NarrativeEvent.event_id`, and `validate_narrative()` requires that reference to resolve inside the canonical story. Existing `direct_state()` then replays that authored event delta into `EpistemicEvidence`.

A runtime `ActionTransitionRecord` is not an authored `NarrativeEvent`. Turning it into a synthetic authored event would collapse two semantic layers:

- authored canonical facts; and
- generated runtime simulation results.

Therefore V1 adds an independent module:

```text
narrative_dynamics/narrative/observation_projection.py
```

The canonical IR, `DomainSpec`, authored `Observation`, replay, belief, decision, intention, and World Transition V1 remain unchanged.

## 3. Chosen architecture

The chosen runtime boundary is:

```text
WorldStepResult
  -> ObservationProjectionModelSpec
  -> ProjectedObservation[]
```

Rejected alternatives:

1. **Synthetic NarrativeEvent + Observation** — rejected because generated simulation state would masquerade as authored canonical history.
2. **Emit EpistemicEvidence directly** — rejected because it would couple the environment/sensor layer to cognition and prematurely equate runtime `step_index` with authored `logical_time`.

A later Multi-Step Scheduler V1 will define the explicit adapter from projected runtime observations into next-round evidence/belief.

## 4. Time semantics

V1 keeps two clocks distinct:

- `NarrativeEvent.logical_time` / `Decision.logical_time`: authored story time.
- `WorldState.step_index`: runtime simulation-step time.

A projected observation records only:

```text
step_index = world_step.next_state.step_index
```

It does not invent an authored `logical_time`.

## 5. Public API

Observation Projection V1 adds exactly these narrative-scoped public names:

```python
ObservationCapabilitySpec
ObserverProjectionSpec
ObservationProjectionModelSpec
ObservationFact
ProjectedObservation
ObservationProjectionResult
ObservationProjectionError
project_world_observations
```

They are exported from `narrative_dynamics.narrative` only, not from the root `narrative_dynamics` package.

## 6. ObservationCapabilitySpec

```python
@dataclass(frozen=True)
class ObservationCapabilitySpec:
    state_variable: str
    subject_scope: str  # "observer" | "any"
```

Purpose: declare objective-state cells that a projection hook may read or emit.

`subject_scope="observer"` means only the cell whose subject is the current observer entity.

`subject_scope="any"` means every canonical subject compatible with that state variable.

No other V1 scope exists.

The record is immutable, exposes `to_dict()`, and has a stable `content_hash`.

Canonical key:

```text
(state_variable, subject_scope)
```

Structural constructor checks validate non-empty names and the exact scope enum. Domain-dependent checks occur later during `project_world_observations()` because the record does not carry a `DomainSpec`.

## 7. ObserverProjectionSpec

```python
@dataclass(frozen=True)
class ObserverProjectionSpec:
    observer_type: str
    channel: str
    read_capabilities: tuple[ObservationCapabilitySpec, ...]
    emit_capabilities: tuple[ObservationCapabilitySpec, ...]
    projection_hook: object
```

One spec defines one runtime observation channel for one observer entity type.

### 7.1 Structural constructor checks

The constructor validates:

- `observer_type` and `channel` are non-empty trimmed strings;
- read capabilities are non-empty and unique;
- emit capabilities are non-empty and unique;
- every emit capability is semantically contained by a read capability;
- `projection_hook` is callable;
- capability ordering is canonicalized.

### 7.2 Domain-dependent execution checks

Before any hook executes, `project_world_observations()` validates against the exact bound `DomainSpec` that:

- `observer_type` is a declared entity type;
- every referenced state variable exists;
- every `observer`-scoped capability refers to a state variable whose subject type equals `observer_type`.

No constructor claims to have performed these domain-dependent checks.

### 7.3 Read vs emit capability

Read capability and emit capability are intentionally different.

Example:

```text
read: lighting(any), location(any)
emit: location(any)
```

The hook may use lighting to decide whether location is visible but cannot emit lighting.

For the same state variable, semantic containment is:

```text
read(any)      contains emit(any)
read(any)      contains emit(observer)
read(observer) contains emit(observer)
read(observer) does not contain emit(any)
```

### 7.4 Channel ownership

`ObservationFact` does not contain a channel. The engine assigns the exact `ObserverProjectionSpec.channel`, so a hook cannot silently emit through a different undeclared channel.

### 7.5 Hook identity

`ObserverProjectionSpec.content_hash` binds:

- observer type;
- channel;
- canonical read capabilities;
- canonical emit capabilities; and
- `measure_implementation(projection_hook).manifest_identity()`.

Changing the hook implementation changes the spec identity.

## 8. Projection hook contract

Conceptual signature:

```python
hook(
    prior_visible: Mapping[StateCellRef, TypedValue],
    next_visible: Mapping[StateCellRef, TypedValue],
    observer: Entity,
    step_index: int,
) -> tuple[ObservationFact, ...]
```

The engine passes immutable mappings and the exact canonical observer entity.

### 8.1 Hidden-state boundary

The hook never receives the complete objective world.

`prior_visible` and `next_visible` contain only cells admitted by the spec's read capabilities for the current observer.

The hook does not receive:

- `WorldStepResult`;
- transition records;
- action intents;
- hidden objective cells;
- canonical claims/receptions;
- belief state;
- goal state.

Therefore framework APIs do not give a hook an undeclared hidden fact that it can inspect merely to decide whether to leak another fact.

### 8.2 Capability-filtered mappings

For `observer` scope, the mapping includes only:

```text
StateCellRef(observer, state_variable)
```

when present.

For `any` scope, the mapping includes all present canonical cells of that state variable.

Absent state remains absent from the mapping.

### 8.3 Determinism boundary

The framework introduces no RNG, clock time, unordered iteration, or implicit scheduling into the hook call.

Specs, observers, capabilities, facts, and accepted observations are canonically ordered.

As with existing semantic/model hooks, deterministic behavior of the supplied Python callable is part of the attested model contract. V1 does not claim to prove callable purity.

## 9. ObservationFact

```python
@dataclass(frozen=True)
class ObservationFact:
    cell: StateCellRef
    relation: str       # "equals" | "clear"
    value: TypedValue | None
```

This is an untrusted hook proposal, not an accepted observation.

Rules:

- `equals` requires a `TypedValue`;
- `clear` requires `value=None`;
- no `not_equals` direct-perception fact exists in V1;
- the hook must return an exact tuple of `ObservationFact` values;
- lists, generators, mappings, scalars, or arbitrary records reject;
- one observer/spec invocation may propose at most one fact per cell.

The record is immutable, exposes `to_dict()`, and has a stable `content_hash`.

## 10. Truth and emit validation

The hook decides **what is visible**, not **what is true**.

Every proposed fact is validated after the hook returns.

### 10.1 Canonical cell validation

The cell must resolve to:

- a canonical story entity;
- the exact entity type named by its `EntityRef`;
- a declared state variable; and
- the state variable's declared subject type.

### 10.2 Emit capability

The cell must be covered by the current spec's emit capabilities for the exact observer.

A readable but non-emittable cell cannot be returned.

### 10.3 Equals truth

For:

```text
relation = equals
```

`next_state` must contain the exact cell and:

```text
fact.value == world_step.next_state.values[fact.cell]
```

The value is also validated against the domain value type.

No value transformation exists in V1.

### 10.4 Clear truth

For:

```text
relation = clear
```

both must hold:

1. the post-step state does not contain the cell; and
2. one current-step transition record contains an exact `StateDeltaOp(kind="clear", ...)` for that cell.

Absence alone is not observable clear evidence.

An explicit clear on an already-absent cell remains eligible because an explicit world operation occurred.

## 11. ProjectedObservation

```python
@dataclass(frozen=True)
class ProjectedObservation:
    observer_id: str
    channel: str
    fact: ObservationFact
    step_index: int
    source_world_state_hash: str
    source_world_step_hash: str
    source_transition_hashes: tuple[str, ...]
    projection_spec_hash: str
```

### 11.1 Engine-derived lineage

Inside `project_world_observations()`, the projection hook supplies only `ObservationFact` values. The engine derives and populates all lineage fields.

Exact source bindings:

```text
source_world_state_hash = world_step.next_state.content_hash
source_world_step_hash  = world_step.content_hash
step_index              = world_step.next_state.step_index
projection_spec_hash    = exact ObserverProjectionSpec.content_hash
```

### 11.2 Transition provenance uses actual write-set membership

For an observed cell, `source_transition_hashes` contains the content hash of every current-step `ActionTransitionRecord` whose delta **wrote** that cell.

This is based on the write set, not on whether the extensional value changed. A `set` to the same value still contributes transition provenance.

Under World Transition V1 conflict semantics the tuple has length zero or one, but a tuple is retained for future explicit conflict-resolution semantics.

If a persistent value is observed and no current-step transition wrote its cell:

```text
source_transition_hashes = ()
```

For `clear`, exactly one matching clear transition must exist under V1 semantics.

### 11.3 Public-record trust boundary

`ProjectedObservation` is a public data record. Directly constructing one with syntactically valid hashes does **not** certify provenance.

The trust claim is specifically about values returned by `project_world_observations()`: its hook cannot supply lineage fields, and the function derives those fields from the validated source world step.

The record constructor validates shape, hash formats, tuple uniqueness/canonicality, and internal field types only.

## 12. ObservationProjectionModelSpec

```python
@dataclass(frozen=True)
class ObservationProjectionModelSpec:
    model_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    projections: tuple[ObserverProjectionSpec, ...]
```

The model binds the exact domain triple:

```text
(domain_id, domain_version, domain_spec_hash)
```

Projection specs are unique by:

```text
(observer_type, channel)
```

One observer type may have multiple explicit channels.

Specs are canonically sorted by `(observer_type, channel)` before hashing.

A model with an empty `projections` tuple is valid and explicitly represents a no-observation / blackout projection condition. This is useful for information-ablation experiments without requiring a dummy hook.

The model hash changes when domain identity, channel, capability declaration, or hook implementation changes.

## 13. ObservationProjectionResult

```python
@dataclass(frozen=True)
class ObservationProjectionResult:
    model_id: str
    model_hash: str
    source_world_step_hash: str
    source_world_state_hash: str
    step_index: int
    observations: tuple[ProjectedObservation, ...]
```

The result is immutable and content-hashed.

An empty observation tuple is valid.

Canonical ordering:

```text
(observer_id, channel, subject_type, subject_id, state_variable)
```

The result constructor checks that every contained observation matches its declared source state/step, runtime step index, and model-produced projection identity relationships that can be checked from contained data.

Like other public records, direct construction is data construction, not independent certification that the source world step actually existed. Runtime trust comes from `project_world_observations()` validation.

## 14. ObservationProjectionError

Runtime projection failures use:

```python
class ObservationProjectionError(ValueError):
    ...
```

Public-record constructor shape failures retain ordinary `TypeError` / `ValueError` semantics.

`project_world_observations()` wraps source resolution, domain mismatch, attestation, hook execution, hook-output shape, capability, and truth failures as `ObservationProjectionError`, chaining the original exception where appropriate.

An arbitrary exception raised by a projection hook must not escape untyped.

## 15. project_world_observations()

Public function:

```python
def project_world_observations(
    story: GenericNarrative,
    domain: DomainSpec,
    world_step: WorldStepResult,
    model: ObservationProjectionModelSpec,
) -> ObservationProjectionResult:
    ...
```

Execution order:

1. validate canonical story/domain identity;
2. validate projection model/domain identity;
3. validate supplied world-step structural and extensional consistency;
4. resolve the canonical entity map;
5. validate all domain-dependent projection declarations;
6. sort projection specs canonically;
7. resolve all canonical observer entities matching each spec observer type;
8. build immutable capability-filtered prior/next mappings;
9. invoke each hook exactly once for each `(observer, spec)` pair;
10. validate exact tuple return shape;
11. validate fact cells, emit capability, typed values, and objective truth;
12. derive state/step/transition/spec lineage automatically;
13. canonicalize accepted observations; and
14. return one atomic `ObservationProjectionResult`.

If any phase fails, no partial result is returned.

For an empty-model blackout condition, phases 7–13 produce no hook calls and an empty observation tuple while still returning a model/source-bound result.

## 16. Upstream WorldStepResult trust boundary

`WorldStepResult` is public, so V1 does not assume every instance originated from `advance_world_step()`.

Before any projection hook runs, `project_world_observations()` validates enough upstream structure to trust the extensional post-step world used for truth checking.

Required checks:

- `world_step` is a `WorldStepResult`;
- prior and next states bind the exact canonical story/domain identity;
- both value maps contain only canonical typed domain cells/values;
- `next_state.step_index == prior_state.step_index + 1`;
- `next_state.parent_state_hash == prior_state.content_hash`;
- every transition record binds the exact prior-state hash;
- every transition intent resolves to a canonical story decision;
- record actor equals the canonical decision actor;
- record action equals the canonical selected `ActionOption` from that decision;
- if `source_at_time` is numeric, the decision is not later than that authored cutoff;
- one actor appears at most once in the step;
- every delta operation has canonical subject/state-variable/value shape;
- no transition writes one cell twice;
- no two transition records write the same cell under V1 conflict semantics;
- applying all validated deltas to prior values reproduces exact `next_state.values`; and
- the next state's transition-batch hash matches the canonical transition-record batch identity.

These checks reject structurally inconsistent or extensionally forged world-step payloads before observation hooks run.

### 16.1 Narrow trust non-claim

Observation Projection V1 does not receive `WorldTransitionModelSpec`, so it does not re-run or independently authorize the upstream transition implementation.

`world_step.model_hash`, transition-spec hashes, and selection-result hashes remain upstream provenance identities, not a second authorization token.

The projection-layer trust claim is narrower: the observed post-step state is canonical, typed, lineage-consistent, and extensionally equal to applying the supplied canonical action transition records.

## 17. Observer iteration

For each projection spec, the engine selects every canonical story entity whose `type_name` exactly equals `spec.observer_type`.

Observers are processed in lexical entity-ID order.

A model may omit a type entirely, meaning those entities receive no projection through that model.

A model may attach multiple channels to one observer type using multiple unique specs.

## 18. Information interventions

No separate information-intervention API is required in V1.

The exact same `WorldStepResult` may be projected with two different `ObservationProjectionModelSpec` values.

Therefore an experiment can hold:

```text
X_t, actions, deltas, X_{t+1}
```

fixed while changing only observation access.

Expected invariant:

```text
same source_world_step_hash
same source_world_state_hash
different projection model hash
potentially different projected-observation set
```

An empty projection model is the explicit blackout witness.

## 19. Atomicity and mutation boundary

Projection is read-only.

It must not mutate:

- `GenericNarrative`;
- `DomainSpec`;
- `WorldStepResult`;
- prior or next `WorldState`;
- any `ActionTransitionRecord`;
- caller-owned tuples or mappings.

If one hook or fact fails, the whole call raises. No partial result is returned.

## 20. Required RED invariants

The test-only RED must lock at least the following before production code exists.

### 20.1 Records and identity

1. public constructors fail closed on malformed strings, scopes, relations, hashes, types, tuple shapes, and duplicate transition-hash entries;
2. `ObservationFact`, capability, spec, model, projected observation, and result hashes are stable and canonical;
3. projection model identity binds exact domain, read/emit capabilities, channel, and hook implementation;
4. capability/spec declaration ordering does not change model hash;
5. duplicate capabilities and duplicate `(observer_type, channel)` specs reject;
6. empty projection model is valid and produces an explicit blackout result.

### 20.2 Domain declaration boundary

7. undeclared observer type rejects before hook execution;
8. undeclared state variable rejects before hook execution;
9. incompatible `observer`-scope state variable rejects before hook execution;
10. semantic read/emit containment follows the exact `any` / `observer` rules.

### 20.3 Visibility boundary

11. a hook receives only read-capability cells and never an undeclared hidden objective cell;
12. `observer` scope exposes only the current observer's compatible cell;
13. `any` scope exposes every present canonical compatible subject for the variable;
14. a readable but non-emittable cell cannot be emitted.

### 20.4 Truth boundary

15. `equals` succeeds only for the exact typed post-step value;
16. fabricated or transformed value rejects;
17. `clear` succeeds only when post-step state is absent and an explicit current-step clear write exists;
18. absence without an explicit clear rejects;
19. unknown subject, mismatched entity type, undeclared variable, or incompatible subject rejects.

### 20.5 Per-agent behavior

20. two observers can receive different observations from the same exact world step;
21. one observer type may have multiple explicit channels;
22. a hook may return an empty fact tuple.

### 20.6 Provenance

23. observation of a cell written by a current-step transition binds that exact transition-record hash automatically, including same-value `set` writes;
24. observation of an unwritten persistent cell has empty transition provenance but exact source-world-state and source-world-step hashes;
25. projected observation step index equals exact next-state step index;
26. hooks cannot provide provenance fields because their only output type is `ObservationFact`; runtime function-derived provenance is asserted exactly.

### 20.7 Source validation before hooks

27. story/domain/source-world identity mismatch rejects before hook execution;
28. malformed prior or next state cell/value rejects before hook execution;
29. canonical decision/action/actor mismatch in a transition record rejects before hook execution;
30. extensionally forged next-state values reject before hook execution;
31. duplicate actor, duplicate per-delta write, or cross-transition write collision rejects before hook execution.

### 20.8 Determinism and isolation

32. observer/spec/fact input order does not change accepted observation ordering or result hash;
33. projection does not mutate story, domain, world step, states, or transition records;
34. exact narrative-scoped public API adds only the eight approved names;
35. root package remains isolated;
36. existing replay, uncertain belief, deterministic decision, intentional decision, world transition, compiler, intervention, movie conformance, research-runtime, and Lean tests retain existing semantics.

## 21. Capability witness

Suppose the world contains:

```text
agent-1.location = hall
agent-2.location = vault
room-1.lighting = dark
service-1.health = degraded
```

and a projection spec declares:

```text
read:
  lighting(any)
  location(any)

emit:
  location(any)
```

The hook can use lighting and location to decide whether locations are visible. It cannot read `service-1.health`, and it cannot emit lighting.

A second projection model may change only capabilities while using the same world step. Source world hashes stay fixed while projection-model identity and percepts may change.

## 22. Clear witness

Prior world:

```text
service-1.alert = true
```

Current transition contains:

```text
clear service-1.alert
```

Post-step world omits the alert cell.

A hook with emit capability for `alert(any)` may propose:

```text
ObservationFact(service-1.alert, relation="clear", value=None)
```

The engine accepts it and derives the exact clear-transition record hash.

If the cell is merely absent before and after with no explicit clear write, the same proposal rejects.

## 23. V1 exclusions

Observation Projection V1 intentionally does not implement:

- initial-state projection before any `WorldStepResult` exists;
- automatic conversion to `EpistemicEvidence`;
- belief update from projected observations;
- multi-step scheduling;
- authored `logical_time` allocation for runtime observations;
- mutation or extension of `GenericNarrative.observations`;
- synthetic `NarrativeEvent` creation;
- testimony generation/reception;
- pure event/action occurrence observations independent of state cells;
- noisy/probabilistic sensors;
- RNG or seed management;
- latency, bandwidth, occlusion geometry, or distance as framework primitives;
- recursive theory of mind;
- World Transition conflict resolution;
- GenericNarrative schema changes;
- DomainSpec identity changes.

Domain-specific deterministic hooks may implement visibility rules using declared readable state, but the framework does not elevate a specific geometry or sensor ontology into V1 core semantics.

## 24. Scientific claim boundary

After V1 the software may claim:

> Given a canonical typed world step and an attested observation-projection model, the runtime produces canonically ordered, capability-limited, truth-validated, provenance-linked per-agent percepts without exposing undeclared objective cells through the projection-hook API.

It does not establish:

- that a real human or fictional character actually perceived those facts;
- perceptual realism;
- psychological validity;
- unique identification of latent cognition;
- external/population validity;
- stochastic sensor realism.

## 25. Expected implementation files

The complete feature is expected to touch exactly these six paths relative to the base commit:

```text
docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_observation_projection.py
tests/test_narrative_trust_api.py
```

Out of scope:

- Lean sources;
- Generic Narrative IR;
- DomainSpec;
- replay;
- uncertain belief;
- deterministic decision;
- intentional decision;
- World Transition V1;
- compiler;
- movie fixtures;
- root package exports;
- runtime/model registry;
- observational protocol;
- prison models.

If implementation reveals a genuine need to alter an out-of-scope boundary, stop and return to architectural review instead of silently widening the diff.

## 26. TDD and CI sequence

Implementation follows strict RED -> GREEN discipline:

1. commit design spec only;
2. after written-spec approval, write and commit implementation plan only;
3. add all V1 semantic tests plus exact public-surface expectation as one test-only RED commit;
4. obtain exact-head CI evidence that failures are only the intended missing implementation/API failures while existing gates remain green;
5. implement the minimum sidecar in reviewable GREEN increments;
6. defer public exports until semantic tests are green so API surface remains the final explicit gate;
7. run final exact-head CI with the full Lean/Python and narrative theorem gates;
8. verify final diff is exactly the approved paths before integration.

No merge to `proof/narrative-dynamics-v0` occurs without explicit user instruction.

## 27. Definition of done

Observation Projection V1 is complete when one validated `WorldStepResult` can produce zero or more runtime projected observations through:

```text
X_{t+1}
  -> capability-filtered hook input
  -> untrusted ObservationFact proposals
  -> truth/capability validation
  -> engine-derived provenance
  -> ProjectedObservation records
```

with all of the following true:

- hooks cannot inspect undeclared objective cells through the framework API;
- hooks cannot emit undeclared cells;
- accepted `equals` values match exact post-step objective truth;
- clear observations require an explicit current-step clear write;
- written-cell provenance binds the exact transition record automatically;
- unwritten persistent observations remain bound to exact source state/step;
- per-agent observation differences are explicit and deterministic at the framework ordering level;
- simulation `step_index` remains separate from authored `logical_time`;
- canonical authored observations remain untouched;
- public surface is scoped only to `narrative_dynamics.narrative`; and
- all existing regression/conformance gates remain green.

This closes:

```text
Action -> World Transition -> Observation
```

The next #27 increment, Multi-Step Scheduler V1, will define:

```text
ProjectedObservation -> next-round evidence/belief -> goal -> action
```
