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

The next architectural boundary is to make post-transition perception explicit:

```text
X_{t+1} -> {O_{t+1}^i}
```

Observation Projection V1 defines a deterministic, typed, capability-limited, provenance-linked runtime projection from one completed `WorldStepResult` to per-agent runtime percepts.

This increment deliberately stops at the percept boundary. It does not yet update beliefs, schedule the next decision round, or mutate the canonical authored story.

## 2. Problem statement

The existing `GenericNarrative.Observation` record is an authored-story record. It contains an `event_id`, and `validate_narrative()` requires that ID to resolve to an authored `NarrativeEvent`. Existing `direct_state()` then replays the referenced authored event delta and turns it into `EpistemicEvidence`.

A simulated `ActionTransitionRecord` is not a `NarrativeEvent`. Converting a runtime action delta into a synthetic authored event would collapse two different semantic layers:

- authored canonical narrative facts; and
- generated runtime simulation results.

Observation Projection V1 must therefore preserve that distinction rather than extending or mutating `GenericNarrative.observations`.

## 3. Architectural decision

Use an independent sidecar module:

```text
narrative_dynamics/narrative/observation_projection.py
```

The module consumes the existing canonical story/domain plus a completed `WorldStepResult` and an explicit projection model. It returns separate runtime `ProjectedObservation` records.

The canonical IR, `DomainSpec`, authored `Observation`, replay logic, uncertain belief, deterministic decision, intentional decision, and World Transition V1 remain unchanged.

### 3.1 Rejected alternative: synthesize NarrativeEvent + Observation

This would reuse current replay machinery, but it would make generated simulation output look authored and would require dynamically extending a canonical immutable `GenericNarrative`. Rejected.

### 3.2 Rejected alternative: emit EpistemicEvidence directly

This is smaller but couples the environment/sensor layer directly to cognition and forces a premature decision about mapping simulation `step_index` to authored `logical_time`. Rejected for V1.

### 3.3 Chosen alternative: runtime projected-observation sidecar

The chosen boundary is:

```text
WorldStepResult
  -> ObservationProjectionModelSpec
  -> ProjectedObservation[]
```

A later Multi-Step Scheduler increment will define the explicit bridge from runtime projected observations into the next-round belief/evidence state.

## 4. Time semantics

Observation Projection V1 has two distinct clocks and must not silently equate them.

- `Decision.logical_time` and `NarrativeEvent.logical_time` are authored canonical story time.
- `WorldState.step_index` is runtime simulation-step time.

A projected runtime observation records the post-step `WorldState.step_index`. It does not invent or claim an authored `logical_time`.

For a world step from `step_index = t` to `step_index = t + 1`, every projected observation produced by that step has:

```text
step_index = world_step.next_state.step_index
```

This is simulation ordering metadata only.

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

They are exported from `narrative_dynamics.narrative` only. They are not exported from the root `narrative_dynamics` package.

## 6. ObservationCapabilitySpec

```python
@dataclass(frozen=True)
class ObservationCapabilitySpec:
    state_variable: str
    subject_scope: str  # "observer" | "any"
```

Purpose: declare which objective state cells a projection hook may read or emit.

### 6.1 Subject scopes

`observer` means only the state cell whose subject is the current observer entity may be included.

`any` means any canonical entity whose type is compatible with the declared state variable may be included.

No other V1 scope exists.

For an `observer` capability, the declared state variable's `subject_type` must equal the containing `ObserverProjectionSpec.observer_type`.

### 6.2 Canonical identity

The record is immutable, has `to_dict()`, and has a stable `content_hash`. Duplicate capability keys are rejected.

Canonical capability key:

```text
(state_variable, subject_scope)
```

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

Each spec defines one observation channel for one canonical observer entity type.

### 7.1 Declaration requirements

- `observer_type` must be a declared domain entity type.
- `channel` is a non-empty trimmed runtime channel name.
- read capabilities are unique and non-empty.
- emit capabilities are unique and non-empty.
- every referenced state variable must exist in the bound `DomainSpec` at execution time.
- `projection_hook` must be callable.

The channel is declared by this exact spec. A hook cannot return a different channel because `ObservationFact` does not contain a channel; the engine assigns the spec channel.

### 7.2 Read vs emit capability

Read capability and emit capability are intentionally distinct.

A hook may need a state cell to decide visibility without being allowed to expose that cell. Example:

```text
read: lighting(any), location(any)
emit: location(any)
```

The hook may use lighting to decide whether a location is visible, but it may not emit lighting itself.

Every emit capability must be semantically contained by a read capability.

Containment is defined as follows for the same state variable:

```text
read(any)      contains emit(any)
read(any)      contains emit(observer)
read(observer) contains emit(observer)
read(observer) does not contain emit(any)
```

This is stronger and more useful than requiring literal tuple equality.

### 7.3 Hook identity

`ObserverProjectionSpec.content_hash` binds:

- observer type;
- channel;
- canonical read capabilities;
- canonical emit capabilities; and
- `measure_implementation(projection_hook).manifest_identity()`.

Changing hook implementation changes the spec identity.

## 8. Projection hook contract

The V1 hook signature is conceptually:

```python
hook(
    prior_visible: Mapping[StateCellRef, TypedValue],
    next_visible: Mapping[StateCellRef, TypedValue],
    observer: Entity,
    step_index: int,
) -> tuple[ObservationFact, ...]
```

The engine supplies immutable mappings.

### 8.1 Critical visibility rule

The hook never receives the complete objective world state.

`prior_visible` and `next_visible` contain only cells allowed by the spec's `read_capabilities` for the current observer.

The hook also does not receive:

- the complete `WorldStepResult`;
- action transition records;
- action intents;
- other hidden world cells;
- canonical claims/receptions;
- belief state;
- goal state.

This prevents a hook from inspecting a hidden objective fact merely to decide whether to leak another fact.

### 8.2 Observer value

The observer argument is the exact canonical `Entity` resolved from `story.entities`. The caller cannot supply a trusted observer copy.

### 8.3 Determinism boundary

The engine adds no RNG, timestamps, unordered iteration, or implicit scheduling to the hook contract. Specs, observers, capabilities, and outputs are canonically ordered.

As with existing semantic/model hooks, deterministic behavior of the supplied hook implementation is part of the attested model contract; V1 does not attempt to prove Python-callable purity.

## 9. ObservationFact

```python
@dataclass(frozen=True)
class ObservationFact:
    cell: StateCellRef
    relation: str       # "equals" | "clear"
    value: TypedValue | None
```

This is an untrusted hook proposal, not yet an accepted observation.

### 9.1 Shape rules

For `equals`:

- `value` must be a `TypedValue`.

For `clear`:

- `value` must be `None`.

No `not_equals` direct-perception fact exists in V1.

The hook returns an exact tuple of `ObservationFact` values. Lists, generators, mappings, scalars, or arbitrary records are rejected.

Within one observer/spec invocation, the hook may propose at most one fact per `StateCellRef`.

## 10. Truthfulness and emit validation

The hook decides what is visible. It does not decide what is true.

Every proposed fact is validated by the engine after the hook returns.

### 10.1 Canonical cell validation

The fact cell must resolve to:

- a canonical story entity;
- the exact entity type declared in the `EntityRef`;
- a declared domain state variable; and
- the state variable's declared subject type.

### 10.2 Emit capability validation

The fact cell must be covered by the current spec's `emit_capabilities` for the exact observer.

A readable but non-emittable cell cannot be returned.

### 10.3 Equals truth rule

For:

```text
relation = equals
```

the exact `next_state` must contain the cell and:

```text
fact.value == world_step.next_state.values[fact.cell]
```

The `TypedValue` is also validated against the domain value type.

A hook cannot fabricate or transform the value in V1.

### 10.4 Clear truth rule

For:

```text
relation = clear
```

both conditions are required:

1. the post-step `next_state` does not contain the cell; and
2. an exact `StateDeltaOp(kind="clear", ...)` for that cell exists in one transition record in this world step.

Absence alone is not observable `clear` evidence. A cell that never existed cannot be turned into an observation of clearing merely because it is absent.

This rule also permits an explicit clear operation on an already-absent cell because the transition itself is an explicit world operation.

## 11. ProjectedObservation

Accepted facts become engine-produced observations:

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

The caller does not supply these lineage fields.

### 11.1 Source state

`source_world_state_hash` is exactly:

```text
world_step.next_state.content_hash
```

### 11.2 Source world step

`source_world_step_hash` is exactly:

```text
world_step.content_hash
```

### 11.3 Source transition provenance

The engine derives transition provenance from the actual write set.

For an observed cell, `source_transition_hashes` contains the `ActionTransitionRecord.content_hash` of every transition record whose delta wrote that cell.

Under World Transition V1 conflict semantics this tuple has length zero or one, but it remains a tuple so a future explicit conflict resolver does not require changing the record shape.

If an unchanged persistent value is observed and no current-step transition touched that cell:

```text
source_transition_hashes = ()
```

The observation is still bound to the exact source world state and source world step.

For `clear`, exactly one matching clear transition must exist in V1.

### 11.4 Identity

`ProjectedObservation` is immutable, canonical, serializable, and content-hashed.

Its identity commits to the observer, channel, fact, runtime step, source state/step lineage, source transition lineage, and projection spec.

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

### 12.1 Domain identity

The model binds the exact domain triple:

```text
(domain_id, domain_version, domain_spec_hash)
```

Execution against any other `DomainSpec` fails before any projection hook runs.

### 12.2 Projection uniqueness

Projection specs are unique by:

```text
(observer_type, channel)
```

One observer type may have multiple explicitly named channels.

### 12.3 Canonical model identity

Specs are sorted by `(observer_type, channel)` before hashing. Capability declaration order does not affect model identity.

The model content hash changes if domain identity, channel, visibility capability, emit capability, or hook implementation changes.

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

The result is immutable and has a stable `content_hash`.

An empty observation tuple is valid when all invoked projection hooks return no facts.

Canonical observation ordering is:

```text
(observer_id, channel, subject_type, subject_id, state_variable)
```

No caller input ordering contributes to result identity.

## 14. project_world_observations()

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

### 14.1 Execution sequence

The function performs these phases in order:

1. validate canonical story/domain identity;
2. validate projection model/domain identity;
3. validate the supplied world-step structural and extensional consistency;
4. resolve canonical entity map;
5. validate projection declarations against the exact domain;
6. sort projection specs canonically;
7. resolve all canonical observer entities matching each spec's observer type;
8. build capability-filtered immutable prior/next views;
9. invoke each hook exactly once for each `(observer, spec)` pair;
10. validate the exact hook return shape;
11. validate fact cells, emit capabilities, typed values, and truthfulness;
12. derive world-state/world-step/transition provenance automatically;
13. canonicalize all accepted observations; and
14. return one atomic `ObservationProjectionResult`.

If any phase fails, no partial result is returned.

## 15. Upstream WorldStepResult validation

`WorldStepResult` is a public record, so Observation Projection V1 does not assume every instance originated from `advance_world_step()`.

Before any projection hook runs, the projection engine validates enough upstream structure to trust the extensional post-step world used for observation truth checks.

### 15.1 Required checks

The function must verify:

- `world_step` is a `WorldStepResult`;
- prior and next world states bind the same exact canonical story/domain identity;
- both state value maps contain only canonical, typed domain cells/values;
- `next_state.step_index == prior_state.step_index + 1`;
- `next_state.parent_state_hash == prior_state.content_hash`;
- every transition record binds the exact prior-state hash;
- every transition intent resolves to a canonical story decision;
- transition `actor_id` equals that decision actor;
- transition action equals the canonical selected `ActionOption` from that decision;
- if the source world has an authored cutoff, a selected decision may not occur after that cutoff;
- one actor appears at most once in the world step;
- every delta operation has a canonical subject/state-variable/value shape;
- no one transition writes one cell twice;
- no two transition records write the same cell under World Transition V1 semantics;
- applying all validated deltas to the prior values reproduces the exact `next_state.values`; and
- the next state's transition batch hash matches the canonical transition-record batch identity.

These checks reject structurally inconsistent or extensionally forged world-step payloads before observation hooks execute.

### 15.2 Trust non-claim

Observation Projection V1 does not receive `WorldTransitionModelSpec`, so it does not re-run or independently authorize the upstream transition model implementation. `world_step.model_hash`, transition-spec hashes, and selection-result hashes remain upstream provenance identities, not a second authorization token.

This layer's trust claim is narrower: the observed post-step world must be canonical, typed, lineage-consistent, and extensionally equal to applying the supplied canonical action transition records.

## 16. Capability-filtered views

For one observer/spec, the engine constructs two immutable mappings:

```text
prior_visible
next_visible
```

For each read capability:

### 16.1 observer scope

Include only:

```text
StateCellRef(observer, state_variable)
```

when present in the corresponding world state.

### 16.2 any scope

Include every present canonical state cell for that state variable.

Because state variables already declare their subject type in `DomainSpec`, unrelated entity types are never included.

### 16.3 Hidden-state guarantee

Cells outside the read capability set do not appear in either mapping. The hook receives no complete world-state reference from which to recover them.

## 17. Information interventions

V1 requires no separate intervention API.

The same exact `WorldStepResult` may be projected with two different attested `ObservationProjectionModelSpec` values.

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
potentially different ProjectedObservation set
```

This is the intended primitive for later information-manipulation and identifiability experiments.

## 18. Errors

Runtime projection failures use:

```python
class ObservationProjectionError(ValueError):
    ...
```

Constructor-level shape errors for the immutable public records retain normal `TypeError` / `ValueError` semantics, consistent with existing narrative records.

`project_world_observations()` wraps resolution, source validation, model-attestation, hook execution, hook-output, capability, and truthfulness failures as `ObservationProjectionError` with the original error chained where appropriate.

A projection-hook exception must never escape as an arbitrary untyped runtime exception.

## 19. Atomicity and mutation boundary

Projection is read-only.

It must not mutate:

- `GenericNarrative`;
- `DomainSpec`;
- `WorldStepResult`;
- prior `WorldState`;
- next `WorldState`;
- any `ActionTransitionRecord`;
- caller-owned mappings or tuples.

If one observer/spec emits an invalid fact, the whole projection call fails. No partial observation result is returned.

## 20. Required RED invariants

The test-only RED must lock at least the following behaviors before production implementation is added.

### 20.1 Records and identity

1. public record constructors fail closed on malformed strings, scopes, relations, hashes, types, and tuple shapes;
2. projection model identity binds exact domain, read/emit capabilities, channel, and hook implementation;
3. capability/spec declaration ordering does not change model hash;
4. duplicate capabilities and duplicate `(observer_type, channel)` specs reject.

### 20.2 Visibility boundary

5. a hook receives only read-capability cells, never a hidden objective cell;
6. `observer` scope exposes only the current observer's own compatible cell;
7. `any` scope exposes all canonical compatible subjects for that state variable;
8. a readable but non-emittable cell cannot be emitted;
9. semantic read/emit containment follows the explicit `any`/`observer` rules.

### 20.3 Truth boundary

10. `equals` succeeds only for the exact typed post-step objective value;
11. a fabricated or transformed value is rejected;
12. `clear` succeeds only when the cell is absent post-step and an explicit current-step clear operation exists;
13. absence without explicit clear is rejected;
14. unknown entity, mismatched entity type, undeclared variable, or incompatible subject rejects.

### 20.4 Per-agent behavior

15. two observers can receive different projected observations from the same exact world step;
16. one observer type may have multiple explicitly declared channels;
17. hooks may validly return an empty fact tuple.

### 20.5 Provenance

18. changed-cell observation binds the exact transition-record content hash automatically;
19. unchanged persistent-state observation has empty transition provenance but exact world-state/world-step provenance;
20. projected observation step index equals the exact next world-state step index;
21. caller cannot forge observation provenance because accepted records are engine-constructed.

### 20.6 Source trust boundary

22. story/domain/source-world identity mismatch rejects before hook execution;
23. malformed prior or next world-state cell/value rejects before hook execution;
24. canonical decision/action/actor mismatch in a transition record rejects before hook execution;
25. extensionally forged next-state values reject before hook execution;
26. duplicate actor, duplicate delta write, or cross-transition write collision rejects before hook execution.

### 20.7 Determinism and isolation

27. observer/spec/fact input order does not change observation ordering or result hash;
28. projection does not mutate story, domain, world step, state, or transition records;
29. exact narrative-scoped public API adds only the eight approved names;
30. root package remains isolated from these new names;
31. existing replay, uncertain belief, deterministic decision, intentional decision, world transition, compiler, intervention, movie conformance, research-runtime, and Lean tests retain their existing semantics.

## 21. Example capability witness

Suppose the world contains:

```text
agent-1.location = hall
agent-2.location = vault
room-1.lighting = dark
service-1.health = degraded
```

and the observer spec declares:

```text
read:
  lighting(any)
  location(any)

emit:
  location(any)
```

The hook can use `lighting` and `location` to decide whether locations are perceptible. It cannot read `service-1.health`, and it cannot emit `lighting` even though it can read it.

A second projection model with the same world step but different read/emit capabilities produces a different projection-model hash while preserving the exact same source-world hashes. This is the V1 information-intervention witness.

## 22. Example clear witness

Prior world:

```text
service-1.alert = true
```

Current world-step transition contains:

```text
clear service-1.alert
```

Post-step world omits `service-1.alert`.

A hook with emit capability for `alert(any)` may emit:

```text
ObservationFact(cell=service-1.alert, relation="clear", value=None)
```

The engine accepts it and automatically binds the transition-record hash that performed the clear.

If `service-1.alert` was simply absent before and after, with no explicit clear delta, the same proposed fact is rejected.

## 23. V1 exclusions

Observation Projection V1 intentionally does not implement:

- automatic conversion to `EpistemicEvidence`;
- belief update from runtime projected observations;
- multi-step scheduling;
- authored `logical_time` allocation for runtime observations;
- mutation or extension of `GenericNarrative.observations`;
- synthetic `NarrativeEvent` creation;
- testimony generation or reception;
- observation of pure action/event occurrence independent of state cells;
- probabilistic or noisy sensors;
- RNG or seed management;
- latency, bandwidth, occlusion geometry, or distance models as framework primitives;
- recursive theory of mind;
- World Transition conflict resolution;
- GenericNarrative schema changes;
- DomainSpec identity changes.

Domain-specific hooks may implement deterministic visibility rules using declared readable state, but the framework does not elevate any particular geometry or sensor ontology into V1 core semantics.

## 24. Scientific claims and non-claims

After V1 the engine may claim:

> Given a canonical typed world step and an attested observation-projection model, the runtime deterministically produces capability-limited, truth-validated, provenance-linked per-agent percepts without exposing undeclared objective state to the projection hook.

It does not establish:

- that a human or fictional character actually perceived those facts;
- perceptual realism;
- psychological validity;
- unique identification of latent cognition;
- population or external validity;
- stochastic sensor realism.

Those require empirical or model-comparison evidence beyond this software contract.

## 25. Expected implementation files

The complete V1 feature is expected to touch exactly these six paths relative to base:

```text
docs/superpowers/specs/2026-08-25-narrative-observation-projection-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-observation-projection-v1.md
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_observation_projection.py
tests/test_narrative_trust_api.py
```

No Lean source, Generic Narrative IR, DomainSpec, replay, uncertain belief, deterministic decision, intentional decision, World Transition V1, compiler, movie fixture, root package export, model registry, observational protocol, or prison-model file is in scope.

If implementation reveals a genuine need to change one of those boundaries, stop and return to architectural review instead of silently widening the diff.

## 26. TDD and CI sequence

Implementation follows strict RED -> GREEN discipline.

1. commit design spec only;
2. after spec approval, write and commit implementation plan only;
3. add all V1 semantic tests plus exact public-surface expectation as one test-only RED commit;
4. obtain exact-head CI evidence that new tests fail for the intended missing implementation/API reasons while existing gates remain green;
5. add the minimum sidecar records/model/projection implementation in reviewable GREEN increments;
6. keep public exports deferred until semantic tests are green, so API surface remains an explicit final gate;
7. run final exact-head CI with full Lean/Python regression plus narrative theorem gates;
8. verify final diff is exactly the approved paths before integration.

No merge to `proof/narrative-dynamics-v0` occurs without explicit user instruction.

## 27. Definition of done

Observation Projection V1 is complete when the engine can take one validated `WorldStepResult` and produce zero or more canonical runtime projected observations such that:

```text
X_{t+1}
  -> capability-filtered observation hook input
  -> untrusted ObservationFact proposals
  -> truth/capability validation
  -> provenance-linked ProjectedObservation records
```

with all of the following true:

- hooks cannot inspect undeclared objective cells through the framework API;
- hooks cannot emit undeclared cells;
- accepted values exactly match post-step objective truth;
- clear observations require an explicit current-step clear operation;
- changed-cell provenance binds the exact transition record automatically;
- unchanged-cell observations remain bound to the exact source world state/step;
- per-agent observation differences are explicit and deterministic;
- simulation `step_index` remains separate from authored `logical_time`;
- canonical authored story observations remain untouched;
- public surface is scoped only to `narrative_dynamics.narrative`; and
- all existing regression/conformance gates remain green.

This closes the explicit runtime boundary:

```text
Action -> World Transition -> Observation
```

The next issue #27 increment is Multi-Step Scheduler V1, which will define the explicit bridge:

```text
ProjectedObservation -> next-round evidence/belief -> goal -> action
```
