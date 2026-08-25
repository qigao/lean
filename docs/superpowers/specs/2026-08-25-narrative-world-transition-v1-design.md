# Narrative World Transition V1 Design

## Status

Approved architectural increment for the Generic Narrative Engine, immediately after Narrative Intentional Decision V1.

This design adds the first narrative-native execution layer in which selected actions can atomically change an explicit world state:

```text
canonical narrative + current world state
    -> selected action intents
    -> per-action state deltas from one shared snapshot
    -> conflict validation
    -> atomic next world state
```

Implementation is intentionally not part of this design commit. The first implementation step after review is a written implementation plan followed by test-only RED.

Base research branch at design start:

```text
proof/narrative-dynamics-v0
fa4df6c7607130f7b5364b0e02ef9ce1dcee16ec
```

## Problem

The Generic Narrative Engine now has a complete one-decision cognitive chain:

```text
admitted evidence
    -> uncertain belief
    -> latent goal policy
    -> goal-conditioned action policy
    -> observable action policy / selected action
```

It also already has a typed authored-world transition system:

```text
NarrativeEvent
    -> DomainSpec.apply_event(...)
    -> StateDelta
    -> objective_state(...)
```

However, these two capabilities are still disconnected. A selected `ActionOption` is currently an explanatory output only. It does not itself update objective world state.

As a result, the engine can explain why an agent chose an action but cannot yet execute multiple chosen actions against one shared state and produce the next state of the world. This blocks the next research layer:

```text
X_t
    -> agent decisions/actions
    -> X_t+1
    -> later observations/beliefs/decisions
```

Narrative World Transition V1 closes only the first missing edge:

```text
(A_1,t, ..., A_N,t, X_t) -> X_t+1
```

It does not yet generate observations or schedule another cognitive round.

## Scientific and simulation goal

V1 defines an explicit simultaneous-action transition semantics for a finite multi-agent world.

For one world step with current state `X_t` and resolved action intents `a_1, ..., a_N`, each action transition is evaluated against the same immutable prior snapshot:

```text
Delta_i = T_i(X_t, a_i)
```

All deltas are validated before any write becomes visible. If their actual write sets are pairwise disjoint, they are committed atomically:

```text
X_t+1 = commit(X_t, Delta_1, ..., Delta_N)
```

If two deltas write the same state cell, V1 rejects the whole step with a typed conflict instead of inventing an implicit priority or execution order.

This is deliberately a simulation mechanism, not a claim that simultaneous action is always the correct model of social interaction. Later model families may implement explicit ordering, bargaining, combat resolution, auctions, resource contention, or stochastic environment dynamics as separately testable mechanisms.

## Design goals

Narrative World Transition V1 must:

1. reuse the existing `StateDelta` / `StateDeltaOp` representation rather than introduce a second state-mutation language;
2. seed an explicit immutable `WorldState` from the existing `objective_state()` replay path;
3. keep `GenericNarrative`, `DomainSpec`, `ActionTypeSpec`, authored event semantics, and their content hashes unchanged;
4. bind the world-transition model to one exact `DomainSpec` identity by domain ID, version, and content hash;
5. resolve the real actor and `ActionOption` from the canonical `Decision` in the narrative rather than trust caller-supplied actor/action payloads;
6. allow action-transition hooks to read exactly one immutable prior world snapshot;
7. declare an explicit capability boundary describing which state cells an action transition may write;
8. validate every returned `StateDelta` against the canonical domain types and declared effect capability;
9. evaluate every action transition before applying any delta;
10. reject overlapping actual write sets with `WorldTransitionConflictError`, even when two writers propose the same value;
11. make successful non-conflicting steps invariant to input-intent ordering;
12. reject duplicate actors within one step so one actor contributes at most one action intent in V1;
13. preserve action-selection provenance through stable content hashes without coupling world execution to one particular decision-model family;
14. preserve exact parent-state and transition-batch lineage in the resulting world step artifact;
15. export the new API only from `narrative_dynamics.narrative`; the top-level `narrative_dynamics` package remains unchanged;
16. keep existing deterministic decision, uncertain belief, intentional decision, replay, intervention, movie conformance, and Lean behavior unchanged.

## Non-goals

V1 does **not** add:

- a new `GenericNarrative` schema version;
- action effects or transition hooks directly to `ActionTypeSpec`;
- mutation of authored `NarrativeEvent` history;
- conversion of selected actions into synthetic `NarrativeEvent` values;
- automatic observation generation from action outcomes;
- automatic testimony or reception generation;
- belief update after the world step;
- a multi-round scheduler;
- dynamically created decisions not already declared in the canonical narrative;
- stochastic environment transitions;
- stochastic sampling from an action policy;
- actor priority, initiative, last-writer-wins, or sequential execution semantics;
- conflict-resolution rules for combat, auctions, races, or resource contention;
- resource locks or transactions beyond whole-step atomic validation;
- continuous time;
- reversible execution / rollback API;
- learned action-transition hooks;
- POMDP planning integration;
- Theory-of-Mind recursion;
- calibration or empirical validation of transition mechanics;
- Lean formalization of the action-transition layer.

These are later increments after the atomic state-transition contract is stable.

## Existing foundations and compatibility constraints

### Existing authored event transition system

`DomainSpec.apply_event()` already provides the canonical pattern for typed world mutation:

```text
prior state + NarrativeEvent
    -> semantic transition hook
    -> StateDelta
```

It validates:

- actor type;
- exact event parameters;
- typed values;
- declared event effects;
- state-variable subject types;
- returned `StateDelta` shape;
- no duplicate write to one cell inside one event delta.

`objective_state()` replays authored events through this path and applies the resulting deltas.

World Transition V1 reuses the same `StateDelta` representation and domain typing assumptions, but it does not alter `DomainSpec.apply_event()` because an executed choice is not an authored event.

### Existing action and decision declarations

`Decision` already declares:

- `id`;
- `logical_time`;
- `actor_id`;
- `type_name`;
- `context_cells`;
- declared `ActionOption` values.

`ActionOption` already declares:

- `id`;
- `type_name`;
- typed action arguments.

`validate_narrative()` already verifies action type and parameter compatibility against `DecisionTypeSpec` / `ActionTypeSpec`.

World Transition V1 therefore treats the canonical narrative as the authority for actor identity and action payload. An `ActionIntent` names a decision and selected action, but execution re-resolves the exact canonical `Decision` and `ActionOption` from the story.

### Existing selected-action models

The world-transition layer must remain independent of how an action was selected. Existing selection families include at least:

- deterministic `DecisionResult`;
- probabilistic `IntentionalDecisionResult` with deterministic MAP projection;
- future reactive, planner, or POMDP adapters.

For that reason `ActionIntent` carries stable selection provenance but does not embed a concrete decision-result class.

## Alternatives considered

### A. Convert selected actions into synthetic `NarrativeEvent` values

Structure:

```text
selected action -> generated event -> existing event replay
```

Rejected for V1. It would reuse the event pipeline mechanically, but it would conflate two different sources of truth:

- authored canonical facts already present in the narrative;
- simulated consequences generated by a model after a decision.

That distinction matters for provenance, counterfactual analysis, empirical data roles, and later comparison of alternative transition models.

### B. Add effects and hooks directly to `ActionTypeSpec`

Structure:

```text
DomainSpec.ActionTypeSpec
    -> parameters + effects + transition hook
```

Rejected for V1. It is conceptually elegant, but it would change `DomainSpec.content_hash` for every domain fixture and would unnecessarily force all existing narratives, movie conformance fixtures, and domain identities through a schema-like migration.

The transition mechanism is a model assumption and can remain sidecar data until its semantics are stable.

### C. Independent sidecar `WorldTransitionModelSpec`

Structure:

```text
canonical DomainSpec identity
    + action transition declarations
    + attested hooks
```

Chosen. It keeps the factual narrative/domain schema stable, makes transition mechanics independently swappable and comparable, and permits a strict RED/GREEN boundary without disturbing existing replay semantics.

## Architecture

### New module

Add:

```text
narrative_dynamics/narrative/world.py
```

The module owns:

- immutable world-state snapshots;
- action-transition capability declarations;
- action-intent provenance records;
- action-transition hook attestation and model identity;
- canonical decision/action resolution;
- per-action delta validation;
- simultaneous snapshot evaluation;
- write-set conflict detection;
- atomic commit;
- step/result lineage.

The module does not own:

- authored event replay;
- evidence admission;
- belief updates;
- goal or choice models;
- observation projection;
- scheduling;
- stochastic sampling;
- model calibration/comparison infrastructure.

### Files unchanged by design

V1 does not change behavior in:

- `narrative_dynamics/narrative/ir.py`;
- `narrative_dynamics/narrative/domain.py`;
- `narrative_dynamics/narrative/replay.py`;
- `narrative_dynamics/narrative/decision.py`;
- `narrative_dynamics/narrative/uncertain.py`;
- `narrative_dynamics/narrative/intention.py`;
- existing domain/movie fixtures;
- Lean source.

The expected production changes are only:

- new `narrative_dynamics/narrative/world.py`;
- `narrative_dynamics/narrative/__init__.py` for scoped exports.

## Public records

### `ActionEffectSpec`

Immutable capability declaration for one potential state-cell target:

```python
ActionEffectSpec(
    state_variable: str,
    subject_source: str,
    subject_argument: str | None = None,
)
```

`subject_source` is exactly one of:

```text
actor
argument
```

Semantics:

- `actor`: the target subject is `Decision.actor_id`; `subject_argument` must be `None`;
- `argument`: the target subject is an `EntityRef` stored in the named action argument; `subject_argument` is required.

The spec identifies a possible write target. It does not require that the hook write the target on every execution. A returned `StateDelta` may write any subset of the resolved declared effects, including the empty set.

This permits explicit no-op / conditional actions without making undeclared writes possible.

### `ActionTransitionSpec`

Immutable action-type transition declaration:

```python
ActionTransitionSpec(
    action_type: str,
    effects: tuple[ActionEffectSpec, ...],
    transition_hook: object,
)
```

The transition hook contract is:

```python
hook(prior_state, decision, action) -> StateDelta
```

where:

- `prior_state` is an immutable mapping of `StateCellRef -> TypedValue` containing exactly the current `WorldState.values` snapshot;
- `decision` is the exact canonical `Decision` resolved from the story;
- `action` is the exact canonical selected `ActionOption` from that decision.

The hook must be callable and implementation-attestable under the repository's existing implementation-attestation rules. Model identity includes the measured implementation identity.

`effects` may be empty for an explicitly modeled no-op transition.

Duplicate effect declarations are rejected canonically.

### `WorldTransitionModelSpec`

Immutable sidecar model:

```python
WorldTransitionModelSpec(
    model_id: str,
    version: str,
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
    transitions: tuple[ActionTransitionSpec, ...],
)
```

Requirements:

- non-empty model ID/version/domain identity;
- valid SHA-256 domain content hash;
- unique transition entry per action type;
- deterministic canonical ordering by action type;
- model content hash binds model identity, exact domain identity, all effect declarations, and every measured transition-hook implementation identity.

The model is not required to cover every `ActionTypeSpec` in the domain. An action type without a transition declaration is simply not executable through this world model and fails closed if selected.

### `ActionIntent`

Immutable selection/provenance envelope:

```python
ActionIntent(
    decision_id: str,
    selected_action: str,
    selection_model_id: str,
    selection_result_hash: str,
)
```

`selection_result_hash` is the stable content hash of the exact upstream selection-result payload. For result types exposing `to_dict()`, the canonical convention is:

```python
stable_content_hash(result.to_dict())
```

The intent is deliberately **not** an authorization token. It does not allow a caller to inject an actor, action type, or action arguments. `advance_world_step()` always resolves those from the canonical story.

This keeps the world layer independent of deterministic, intentional, reactive, planner, or future POMDP result classes while preserving traceable upstream selection identity.

### `WorldState`

Immutable world-state artifact:

```python
WorldState(
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
    source_story_hash: str,
    source_at_time: int | None,
    step_index: int,
    parent_state_hash: str | None,
    transition_batch_hash: str | None,
    values: Mapping[StateCellRef, TypedValue],
)
```

Initial states produced by `world_state_from_story()` have:

```text
step_index = 0
parent_state_hash = None
transition_batch_hash = None
```

A successfully advanced state has:

```text
step_index = prior.step_index + 1
parent_state_hash = prior.content_hash
transition_batch_hash = stable hash of the canonical committed transition batch
```

`values` is immutable and canonically serialized by `(entity_type, entity_id, state_variable)`.

Including both parent-state and transition-batch identity means two identical value maps reached through different transition histories remain distinguishable as research artifacts. Callers interested only in extensional state values can compare the canonical values payload separately.

### `ActionTransitionRecord`

Immutable validated transition record for one intent:

```python
ActionTransitionRecord(
    intent: ActionIntent,
    actor_id: str,
    action: ActionOption,
    transition_spec_hash: str,
    prior_state_hash: str,
    delta: StateDelta,
)
```

The record stores the canonical resolved actor/action rather than caller-provided duplicates.

It therefore gives the lineage:

```text
selection_result_hash
    -> ActionIntent
    -> canonical Decision / ActionOption
    -> ActionTransitionSpec
    -> StateDelta
```

### `WorldStepResult`

Immutable atomic step result:

```python
WorldStepResult(
    model_id: str,
    model_hash: str,
    prior_state: WorldState,
    transitions: tuple[ActionTransitionRecord, ...],
    next_state: WorldState,
)
```

Requirements:

- at least one transition;
- every record binds the exact `prior_state.content_hash`;
- canonical transition ordering independent of input intent order;
- `next_state.parent_state_hash == prior_state.content_hash`;
- `next_state.step_index == prior_state.step_index + 1`;
- `next_state.transition_batch_hash` equals the stable hash of the canonical transition payload;
- model ID/hash match the executing `WorldTransitionModelSpec`.

### Error types

```python
class WorldTransitionError(ValueError): ...
class WorldTransitionConflictError(WorldTransitionError): ...
```

`WorldTransitionConflictError` is reserved for multi-intent write collisions. Other declaration, lineage, type, capability, unsupported-action, and transition-hook failures use `WorldTransitionError` with the original error chained when useful.

## Initial world-state construction

Public API:

```python
world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> WorldState
```

Semantics:

1. validate the canonical narrative/domain relationship through the existing replay path;
2. call existing `objective_state(story, domain, at_time=at_time)` rather than duplicate event replay;
3. snapshot the returned state into immutable `WorldState.values`;
4. bind the exact story content hash and requested cutoff;
5. initialize step/provenance fields to zero/`None`.

`at_time=None` means full authored objective replay, matching existing `objective_state()` semantics.

A numeric cutoff is retained exactly in `source_at_time` and must be a non-negative integer under the same replay rules.

## Model/domain validation

Before evaluating any intent, `advance_world_step()` validates that:

```text
(prior_state.domain_id,
 prior_state.domain_version,
 prior_state.domain_spec_hash)
==
(domain.domain_id,
 domain.version,
 domain.content_hash)
==
(model.domain_id,
 model.domain_version,
 model.domain_spec_hash)
```

It also requires:

```text
prior_state.source_story_hash == story.content_hash
```

This prevents a state/model generated for another domain or canonical story from being silently reused.

Every configured `ActionTransitionSpec.action_type` must name a declared `ActionTypeSpec` in the supplied domain when the model is executed.

## Canonical intent resolution

For each `ActionIntent`:

1. `decision_id` must resolve to exactly one canonical `Decision` in `story.decisions`;
2. `selected_action` must resolve to exactly one `ActionOption` in that decision;
3. the selected action type must have one matching `ActionTransitionSpec` in the model;
4. actor ID comes from the canonical decision;
5. action type and typed arguments come from the canonical action;
6. caller-supplied provenance fields are retained but never used to override canonical execution data.

Within one world step:

- decision IDs must be unique;
- resolved actor IDs must be unique;
- at least one intent is required.

If the initial `WorldState` was created with a numeric `source_at_time`, each selected authored decision must have `decision.logical_time <= source_at_time`. A full-story state (`source_at_time=None`) is considered to include every authored decision time.

This prevents a cutoff world snapshot from executing an authored decision that has not occurred yet.

## Effect-target capability resolution

For a resolved `ActionEffectSpec`, V1 computes one allowed `StateCellRef`.

### Actor target

For:

```text
subject_source = actor
```

subject ID/type come from the canonical decision actor.

The configured state variable must exist and its declared `subject_type` must match the actor entity type.

### Action-argument target

For:

```text
subject_source = argument
subject_argument = p
```

`p` must be a declared parameter of the selected `ActionTypeSpec` and the canonical `ActionOption.arguments[p]` must contain an `EntityRef`.

The referenced entity must exist in the canonical story, and its entity type must match the declared state-variable subject type.

### Capability set

The union of resolved effect targets is the action's allowed write set:

```text
Allowed_i
```

The hook may return a delta whose actual write set is any subset:

```text
Write_i subseteq Allowed_i
```

No implicit default write occurs for declared-but-unused effects.

## Transition-hook execution

Every transition hook receives an immutable snapshot derived from the exact same prior values:

```text
X_t = prior_state.values
```

Conceptually:

```text
Delta_i = hook_i(readonly_copy(X_t), decision_i, action_i)
```

No delta is applied before all hooks have returned and all deltas have passed validation.

A hook must return `StateDelta`. Any other return type fails with `WorldTransitionError`.

The hook must not receive `WorldStepResult`, other actions' deltas, a mutable shared world object, evidence state, belief state, or objective story replay beyond the supplied prior snapshot.

## Delta validation

For every returned `StateDelta`:

1. every operation must target a canonical declared entity;
2. every state variable must exist in the domain;
3. subject entity type must match the state variable's `subject_type`;
4. each operation target must belong to the action's resolved allowed write set;
5. one action delta may not write the same state cell more than once;
6. `clear` operations must contain no value;
7. `set` operations must contain a value;
8. set values must validate against the state variable's declared `ValueTypeSpec` and canonical entity map;
9. empty deltas are valid.

Validation reuses the existing `DomainSpec` state/value definitions; it does not create a second type system.

## Simultaneous multi-agent semantics

### Snapshot isolation

For one step containing intents `I = {i_1, ..., i_N}`, every hook is evaluated against the same extensional snapshot:

```text
forall k: InputState(hook_k) == X_t
```

No hook observes another hook's proposed writes.

### Actual write sets

After validation, each transition record has an actual write set:

```text
Write_k = {StateCellRef written by Delta_k}
```

Conflict detection uses actual writes, not the larger declared capability sets. Two actions may have overlapping possible effects but still coexist if their returned deltas do not write the same cell in this execution.

### Conflict rule

For any two distinct transition records:

```text
Write_i intersect Write_j != empty
```

causes the whole step to fail with `WorldTransitionConflictError`.

This applies even when both operations:

- are `set` with identical values;
- are both `clear`;
- would otherwise commute accidentally.

V1 intentionally refuses to infer that two independent writers are semantically equivalent.

### Atomic commit

Only after all hook and conflict validation succeeds does V1 construct a fresh mutable copy internally and apply all non-overlapping operations.

Because actual write sets are disjoint, commit order cannot change extensional state. Nevertheless, implementation uses canonical sorted transition/operation order for reproducibility.

If any intent, hook, delta, type check, capability check, or conflict check fails, no `WorldStepResult` and no `next_state` is returned. The immutable input `WorldState` is never mutated.

## Canonical ordering and order invariance

Input intent ordering is not semantic.

After canonical resolution, transition records are sorted by:

```text
(actor_id, decision_id, action.id)
```

Operations used for canonical transition-batch serialization are sorted by resolved state-cell key:

```text
(entity_type, entity_id, state_variable)
```

For any valid pairwise-disjoint intent set:

```text
advance_world_step(X, [a, b])
==
advance_world_step(X, [b, a])
```

for:

- transition record order;
- transition batch hash;
- next-state payload;
- next-state content hash;
- whole-result content hash.

## Lineage and identity

### Transition model identity

`WorldTransitionModelSpec.content_hash` binds:

- model ID/version;
- domain ID/version/spec hash;
- canonical action-transition declarations;
- all effect capabilities;
- measured transition-hook implementation identities.

Changing a transition hook's measured module bytes or changing any effect declaration changes model identity.

### Initial state lineage

An initial `WorldState` binds:

```text
canonical story hash
canonical domain identity
objective replay cutoff
extensional values
```

### Step lineage

Each `ActionTransitionRecord` binds:

```text
ActionIntent
canonical actor/action
transition spec hash
prior world-state hash
returned StateDelta
```

`WorldStepResult` binds:

```text
transition model hash
prior state
canonical transition batch
next state
```

The next state binds both parent-state hash and canonical transition-batch hash.

Therefore a later research artifact can reconstruct:

```text
selection result
    -> action intent
    -> action transition
    -> delta
    -> world step
    -> next world state
```

without claiming that the world-transition layer independently verifies the psychological correctness of the upstream action-selection model.

## Public functions

### `world_state_from_story()`

Creates step-zero `WorldState` from existing authored objective replay.

### `advance_world_step()`

Proposed signature:

```python
advance_world_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: WorldState,
    model: WorldTransitionModelSpec,
    intents: tuple[ActionIntent, ...],
) -> WorldStepResult
```

The function performs:

```text
validate story/domain/state/model identity
    -> resolve canonical intents
    -> verify one action per actor
    -> resolve effect capabilities
    -> execute all hooks against the same snapshot
    -> validate all StateDelta values
    -> reject actual write conflicts
    -> canonically commit
    -> produce lineage-bound next state/result
```

No RNG or execution-order parameter exists in V1.

## Public API surface

`narrative_dynamics.narrative` adds exactly these names:

```text
ActionEffectSpec
ActionTransitionSpec
WorldTransitionModelSpec
ActionIntent
WorldState
ActionTransitionRecord
WorldStepResult
WorldTransitionError
WorldTransitionConflictError
world_state_from_story
advance_world_step
```

The top-level `narrative_dynamics` package must not export them.

## Error handling

V1 is fail-closed.

`WorldTransitionError` covers, among other cases:

- invalid or mismatched story/domain/model/world-state identity;
- empty intent batch;
- duplicate decision IDs;
- duplicate resolved actor IDs;
- missing decision;
- missing selected action;
- action type without a configured transition;
- action transition referring to undeclared action type;
- malformed actor/argument effect target;
- state-variable / entity-type mismatch;
- hook attestation unavailable when model identity is requested;
- transition hook exception wrapped with cause;
- hook returning non-`StateDelta`;
- duplicate cell write inside one delta;
- write outside resolved capability;
- invalid `set`/`clear` shape;
- invalid typed set value;
- authored decision after a numeric source cutoff.

`WorldTransitionConflictError` covers only cross-intent actual write overlap.

The input `WorldState` remains immutable and reusable after any failure.

## Test strategy

Implementation must follow test-only RED before production code.

The new primary test module is expected to be:

```text
tests/test_narrative_world_transition.py
```

World-transition-specific fixtures remain local to that test module unless a second production consumer demonstrates a real shared-fixture need. Existing `tests/narrative_test_support.py` is not expanded solely for this feature.

### RED invariants

The first RED must lock at least the following behaviors.

#### 1. Initial state reuses authored replay

For a canonical story/cutoff:

```text
world_state_from_story(...).values == objective_state(...)
```

and source story/domain/cutoff lineage is exact.

#### 2. Single action changes declared world state

A canonical selected action writes a declared target and produces:

```text
step_index = 1
parent_state_hash = prior.content_hash
```

with exact action/delta lineage.

#### 3. Snapshot isolation

Two transition hooks in one step must both observe the same prior-state payload even when the first intent would change a cell read by the second hook.

The second hook must not observe first-intent writes.

#### 4. Non-conflicting order invariance

For two actions writing different cells:

```text
advance(X, (a, b)).content_hash
==
advance(X, (b, a)).content_hash
```

and next-state payloads are exactly equal.

#### 5. Cross-intent conflict rejection

Two actual deltas writing one common `StateCellRef` raise `WorldTransitionConflictError`.

Test both:

- different proposed values;
- identical proposed values.

#### 6. Atomic failure

If one of several actions is invalid, the call raises and the original prior-state payload/hash remains unchanged. No partial result is returned.

#### 7. One action per actor

Two different decisions by the same canonical actor in one step reject with `WorldTransitionError`.

#### 8. Canonical action resolution

Missing decision ID, undeclared selected action ID, and unsupported action type each fail typed resolution.

#### 9. Actor effect capability

An `actor` effect may write only a state variable whose subject type matches the canonical actor type.

#### 10. Argument effect capability

An `argument` effect requires the named canonical action argument to be an `EntityRef` of the correct subject type.

#### 11. Capability escape rejection

A hook that writes a valid domain cell not declared by its `ActionEffectSpec` still fails with `WorldTransitionError`.

#### 12. Delta shape/type rejection

Lock typed rejection for:

- duplicate writes inside one action delta;
- `set` with missing value;
- `clear` with a value;
- wrong value type;
- unknown state variable;
- unknown subject entity.

#### 13. Explicit no-op support

A configured action transition with empty effects and an empty `StateDelta` succeeds, records the action, and preserves extensional values while still incrementing the step and lineage.

#### 14. Cutoff discipline

A state created with numeric `source_at_time=t` rejects an intent for an authored decision whose logical time is greater than `t`.

#### 15. Model identity

Changing any of the following changes world-transition model content hash:

- domain hash;
- effect target declaration;
- transition hook implementation identity.

Input ordering of transition declarations does not change identity.

#### 16. Existing regression isolation

The existing full suite must remain green, including:

- GenericNarrative validation;
- objective/direct/epistemic replay;
- deterministic decision tests;
- uncertain-belief tests;
- intentional-decision tests;
- interventions/analysis;
- Knives Out, The Matrix, and Memento conformance;
- root API isolation;
- Lean conformance/build/theorem gates.

### Public-surface RED

`tests/test_narrative_trust_api.py` is expected to add the exact 11 new scoped exports above while retaining root-package isolation.

As with Intentional Decision V1, the implementation should first achieve semantic GREEN in the dedicated world-transition tests before opening the final scoped public exports if that separation makes the RED boundary clearer.

## Expected implementation files

The complete V1 implementation is expected to touch only:

```text
docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-world-transition-v1.md
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_world_transition.py
tests/test_narrative_trust_api.py
```

No `GenericNarrative` schema, `DomainSpec`, existing replay, existing decision model, movie fixture, root package, registry, observational protocol, prison model, or Lean-source changes are part of this increment.

If implementation discovers that one of those files must change to make V1 correct, the work must stop and the architecture must be re-reviewed rather than silently expanding scope.

## Relationship to later increments

World Transition V1 creates:

```text
X_t + selected actions -> X_t+1
```

The next intended increment can add observation projection:

```text
X_t+1 + visibility/channel model
    -> O_i,t+1
```

A later scheduler can then connect the loop:

```text
X_t
    -> O_i,t
    -> B_i,t
    -> G_i,t
    -> A_i,t
    -> atomic world transition
    -> X_t+1
    -> ...
```

At that point the Generic Narrative Engine becomes a multi-round cognitive multi-agent simulator while retaining separable model families for perception, belief, goal selection, action selection, transition mechanics, and planning.

## Acceptance boundary

Narrative World Transition V1 is complete only when all of the following are true on one exact feature head:

1. dedicated world-transition tests are green;
2. existing Python tests are green;
3. exact narrative public-surface/root-isolation tests are green;
4. Lean/Python conformance is green;
5. full Lean build is green;
6. Lean theorem tests are green;
7. Narrative StoryState theorem gate is green;
8. Narrative Testimony theorem gate is green;
9. diff remains inside the approved file boundary;
10. the verified feature head is integrated into `proof/narrative-dynamics-v0` only after explicit human merge approval.
