# Narrative World Transition V1 Design

## Status

Approved architectural increment for the Generic Narrative Engine, immediately after Narrative Intentional Decision V1.

This design adds the first narrative-native execution layer in which selected actions can atomically change an explicit world state:

```text
canonical narrative + current world state
    -> selected action intents
    -> per-action deltas from one shared snapshot
    -> capability/type/conflict validation
    -> atomic next world state
```

Implementation is intentionally not part of this design commit. After written-spec review, the next step is an implementation plan followed by test-only RED.

Design base:

```text
proof/narrative-dynamics-v0
fa4df6c7607130f7b5364b0e02ef9ce1dcee16ec
```

## Problem

The engine now has a complete one-decision cognitive chain:

```text
admitted evidence
    -> uncertain belief
    -> latent goal policy
    -> goal-conditioned action policy
    -> observable action policy / selected action
```

It also already has a typed authored-world transition path:

```text
NarrativeEvent
    -> DomainSpec.apply_event(...)
    -> StateDelta
    -> objective_state(...)
```

These capabilities are still disconnected. A selected `ActionOption` is an explanatory output; it does not itself update objective world state.

World Transition V1 closes only this edge:

```text
(A_1,t, ..., A_N,t, X_t) -> X_t+1
```

It does not yet generate observations, update beliefs, or schedule another cognitive round.

## Scientific semantics

For one world step with current state `X_t` and resolved actions `a_1, ..., a_N`, every transition is evaluated against the same immutable prior snapshot:

```text
Delta_i = T_i(X_t, a_i)
```

No delta becomes visible while another hook is running.

After all deltas pass validation, V1 computes their actual write sets. If those sets are pairwise disjoint, the deltas are committed atomically:

```text
X_t+1 = commit(X_t, Delta_1, ..., Delta_N)
```

If two deltas write the same `StateCellRef`, the entire step fails with a typed conflict, even if the writers propose the same value.

V1 therefore models simultaneous action explicitly. It does not silently introduce actor priority, last-writer-wins, or sequential order.

This is one falsifiable simulation mechanism, not a claim that simultaneous action is universally correct. Later models may add explicit ordering, bargaining, combat resolution, auctions, resource contention, or stochastic environment dynamics.

## Design goals

V1 must:

1. reuse existing `StateDelta` / `StateDeltaOp` instead of inventing another mutation language;
2. seed immutable `WorldState` from existing `objective_state()` replay;
3. leave `GenericNarrative`, `DomainSpec`, `ActionTypeSpec`, authored event semantics, and their identities unchanged;
4. bind each world-transition model to one exact domain ID/version/content hash;
5. re-resolve actor and action payload from canonical `Decision` / `ActionOption` rather than trust caller-supplied duplicates;
6. validate a supplied `WorldState` against the canonical story/domain before any transition hook runs;
7. give every transition hook exactly one immutable prior world snapshot;
8. declare an explicit action-effect capability boundary;
9. validate every returned delta against canonical entities, state variables, value types, and allowed effects;
10. evaluate all hooks before applying any write;
11. reject overlapping actual write sets with `WorldTransitionConflictError`;
12. make successful non-conflicting steps invariant to input-intent order;
13. allow at most one action per canonical actor in one step;
14. preserve upstream selection provenance without coupling world execution to one selection-model class;
15. bind parent-state and transition-batch lineage into the resulting state/result;
16. export the new API only from `narrative_dynamics.narrative`;
17. preserve all existing deterministic-decision, uncertain-belief, intentional-decision, replay, intervention, movie-conformance, root-isolation, and Lean semantics.

## Non-goals

V1 does not add:

- a new GenericNarrative schema version;
- action effects/hooks directly to `ActionTypeSpec`;
- synthetic `NarrativeEvent` generation from actions;
- mutation of authored event history;
- automatic observations, claims, receptions, or belief updates;
- a multi-round scheduler;
- dynamically generated decisions;
- stochastic transition mechanics;
- stochastic action sampling;
- initiative, actor priority, sequential execution, or last-writer-wins;
- conflict resolvers for combat, auctions, races, or contested resources;
- continuous time;
- rollback/reversible execution;
- learned transition hooks;
- POMDP planning integration;
- Theory-of-Mind recursion;
- calibration or empirical-validity claims;
- Lean formalization of the new action-transition layer.

## Existing foundations and compatibility constraints

### Authored event transition system

`DomainSpec.apply_event()` already establishes the canonical typed-mutation pattern:

```text
prior state + NarrativeEvent
    -> semantic hook
    -> StateDelta
```

It validates actor type, parameters, event-effect capability, state-variable subject types, typed values, result shape, and duplicate writes inside one event delta.

`objective_state()` replays authored events through that path.

World Transition V1 reuses the same state and type concepts but does not alter `DomainSpec.apply_event()`: simulated action consequences and authored events remain distinct provenance layers.

### Canonical decisions/actions

`Decision` already declares `id`, `logical_time`, `actor_id`, `type_name`, `context_cells`, and `ActionOption` alternatives. `ActionOption` already declares `id`, `type_name`, and typed arguments. `validate_narrative()` already checks each declared action against its domain action type.

Therefore the canonical narrative is authoritative for actor identity and action payload. An `ActionIntent` may name a decision/action, but execution re-resolves both from the story.

### Selection-model independence

Existing selected actions may come from deterministic `DecisionResult`, probabilistic `IntentionalDecisionResult`, or later reactive/planner/POMDP families. The world layer must not depend on one result class.

`ActionIntent` therefore carries selection provenance as stable identifiers/hashes but does not embed a concrete result object.

## Alternatives considered

### A. Action -> synthetic NarrativeEvent

Rejected. It would mechanically reuse replay but collapse authored facts and simulated consequences into one event class, weakening provenance and model comparison.

### B. Extend ActionTypeSpec directly

Rejected for V1. Adding action effects/hooks to `DomainSpec` would change every domain content hash and force unrelated narrative/movie fixtures through a migration before the transition semantics are proven.

### C. Sidecar WorldTransitionModelSpec

Chosen. It binds exact domain identity while keeping transition mechanics independently swappable, attestable, testable, and comparable.

## Architecture

Add one new production module:

```text
narrative_dynamics/narrative/world.py
```

It owns:

- immutable world-state artifacts;
- sidecar action-effect/transition declarations;
- action-intent provenance;
- canonical intent resolution;
- transition-hook attestation;
- prior-world validation;
- per-action capability/type validation;
- snapshot evaluation;
- write-conflict detection;
- atomic commit;
- transition/result lineage.

It does not own authored replay, evidence admission, belief/goal/choice models, observation projection, scheduling, stochastic sampling, or calibration/comparison infrastructure.

Production behavior outside new `world.py` must remain unchanged. The only other production change is scoped export wiring in `narrative_dynamics/narrative/__init__.py`.

## Public records

### ActionEffectSpec

```python
ActionEffectSpec(
    state_variable: str,
    subject_source: str,
    subject_argument: str | None = None,
)
```

`subject_source` is exactly:

```text
actor | argument
```

Semantics:

- `actor`: target subject is the canonical `Decision.actor_id`; `subject_argument` must be `None`.
- `argument`: target subject is the `EntityRef` in the named canonical action argument; `subject_argument` is required.

The declaration is a capability upper bound. A hook may return a delta writing any subset of its resolved declared effects, including the empty set. There is no implicit write for an unused effect.

### ActionTransitionSpec

```python
ActionTransitionSpec(
    action_type: str,
    effects: tuple[ActionEffectSpec, ...],
    transition_hook: object,
)
```

Hook contract:

```python
hook(prior_state, decision, action) -> StateDelta
```

Inputs are:

- immutable `Mapping[StateCellRef, TypedValue]` equal to the current world snapshot;
- exact canonical `Decision`;
- exact canonical selected `ActionOption`.

The hook must be callable and measurable under existing implementation-attestation rules. Its `content_hash` binds action type, canonical effect declarations, and measured implementation identity.

Effect declarations may be empty for an explicitly modeled no-op action. Duplicate declarations reject.

Transition hooks are model code; V1 does not add hidden instance-parameter identity. Any model parameter that must affect reproducibility must therefore be represented by canonical action/world data or by measured implementation code in this increment, rather than mutable unrecorded hook state.

### WorldTransitionModelSpec

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

- non-empty canonical IDs/version;
- valid SHA-256 domain hash;
- unique transition entry per action type;
- canonical ordering by action type;
- content hash binds model identity, exact domain identity, effect declarations, and measured transition implementations.

The model need not cover every domain action type. Selecting an uncovered type fails closed.

### ActionIntent

```python
ActionIntent(
    decision_id: str,
    selected_action: str,
    selection_model_id: str,
    selection_result_hash: str,
)
```

Canonical convention for upstream result records with `to_dict()`:

```python
selection_result_hash = stable_content_hash(result.to_dict())
```

The intent is provenance, not authorization. It cannot inject actor ID, action type, or action arguments. `advance_world_step()` always re-resolves those from the story.

### WorldState

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

Step-zero states produced by `world_state_from_story()` have:

```text
step_index = 0
parent_state_hash = None
transition_batch_hash = None
```

Advanced states have:

```text
step_index = prior.step_index + 1
parent_state_hash = prior.content_hash
transition_batch_hash = hash(canonical transition batch)
```

`values` is immutable and serialized by `(entity_type, entity_id, state_variable)`.

The state content hash includes provenance. Two equal value maps reached by different transition histories may therefore have different artifact hashes; extensional equality can be tested by comparing canonical values payloads.

Structural construction checks record shape. Domain/story compatibility of all state cells/values is revalidated at execution time because callers can construct public `WorldState` instances directly.

### ActionTransitionRecord

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

The record stores canonical resolved actor/action plus the exact transition spec and prior-state identity.

Lineage:

```text
selection_result_hash
    -> ActionIntent
    -> canonical Decision/ActionOption
    -> ActionTransitionSpec
    -> StateDelta
```

### WorldStepResult

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

- non-empty transitions;
- canonical transition ordering;
- every record binds `prior_state.content_hash`;
- next step index is prior + 1;
- next parent hash equals prior hash;
- next transition-batch hash equals the canonical transition payload hash;
- model ID/hash match the executing model.

The result exposes a stable `content_hash` over its complete canonical payload.

### Errors

```python
class WorldTransitionError(ValueError): ...
class WorldTransitionConflictError(WorldTransitionError): ...
```

`WorldTransitionConflictError` is reserved for cross-intent actual-write collisions. Other model/state/declaration/capability/type/hook failures use `WorldTransitionError`, chaining the original cause when useful.

## world_state_from_story()

```python
world_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    *,
    at_time: int | None = None,
) -> WorldState
```

Semantics:

1. use existing narrative/domain validation through `objective_state()`;
2. call `objective_state(story, domain, at_time=at_time)` rather than duplicate replay;
3. snapshot its values immutably;
4. bind exact story/domain identity and requested cutoff;
5. set step/provenance fields to zero/`None`.

`at_time=None` retains existing full authored replay semantics. Numeric cutoffs use existing non-negative replay rules and are stored exactly.

## Prior-state validation boundary

`advance_world_step()` must not assume a supplied `WorldState` came from `world_state_from_story()`.

Before any hook executes, it validates:

```text
prior domain identity == supplied DomainSpec identity == model domain identity
prior source_story_hash == story.content_hash
```

It then validates **every** `prior_state.values` entry against the canonical story/domain:

1. key is a `StateCellRef`;
2. referenced subject entity exists in `story.entities`;
3. `StateCellRef.subject.entity_type` matches that canonical entity;
4. state variable exists in `DomainSpec`;
5. state variable subject type matches the entity type;
6. value is a `TypedValue`;
7. value validates against the state variable's declared `ValueTypeSpec`, including entity-ref targets.

Any forged/invalid prior cell or value fails with `WorldTransitionError` before a transition hook runs.

This closes the public-record trust boundary without changing `DomainSpec` or replay APIs.

## Canonical intent resolution

For every intent:

1. resolve `decision_id` to one canonical story decision;
2. resolve `selected_action` to one canonical action in that decision;
3. take actor ID from the canonical decision;
4. take action type/arguments from the canonical action;
5. require a matching `ActionTransitionSpec` for the selected action type.

Within one step:

- at least one intent is required;
- decision IDs must be unique;
- canonical actor IDs must be unique.

If `prior_state.source_at_time` is numeric, every selected authored decision must satisfy:

```text
decision.logical_time <= source_at_time
```

A full-story source (`None`) is considered to include all authored decision times.

The world layer does not claim that decisions grouped into one transition step were selected simultaneously; scheduling/decision-time synchronization is a later subsystem. V1 only guarantees simultaneous **effect evaluation and commit** for the supplied intents.

## Effect-target capability resolution

Each `ActionEffectSpec` resolves to one allowed `StateCellRef`.

### actor source

The canonical decision actor is the subject. The configured state variable must exist and its subject type must match the actor's canonical entity type.

### argument source

The named parameter must exist on the selected domain `ActionTypeSpec`. The canonical selected action argument must contain an `EntityRef`. The referenced entity must exist and its type must match the target state variable subject type.

The resolved set is the capability upper bound:

```text
Allowed_i
```

The returned delta must satisfy:

```text
Write_i subseteq Allowed_i
```

Overlapping effect declarations that resolve to one allowed cell collapse only as a capability set; a returned delta still may not contain duplicate writes to that cell.

## Hook execution and delta validation

Every hook receives an immutable snapshot with extensional content exactly equal to `prior_state.values`.

Conceptually:

```text
Delta_i = hook_i(readonly_copy(X_t), canonical_decision_i, canonical_action_i)
```

No hook receives another action's delta, next-state view, belief state, goal state, or a mutable shared world object.

A hook must return `StateDelta`.

Every returned operation is validated:

1. subject entity exists;
2. state variable exists;
3. entity type matches state-variable subject type;
4. target belongs to the resolved action capability;
5. one delta cannot write the same cell twice;
6. `clear` carries no value;
7. `set` carries a value;
8. set value validates against the state variable's canonical value type.

Empty deltas are valid for explicit no-op/conditional transitions.

## Simultaneous multi-agent semantics

### Snapshot isolation

For all intents in one step:

```text
InputState(hook_i) == X_t
```

No earlier evaluated action can influence the state read by a later hook.

### Conflict detection

After delta validation, compute actual write sets:

```text
Write_i = {cells actually written by Delta_i}
```

Conflict detection uses actual writes, not declared capability sets.

For distinct `i, j`:

```text
Write_i intersect Write_j != empty
```

raises `WorldTransitionConflictError`.

This includes same-value set/set and clear/clear collisions.

### Atomicity

No mutation occurs until every hook/delta/capability/type/conflict check passes.

Successful commit applies all pairwise-disjoint operations to a fresh copy. Failed execution returns no `WorldStepResult` or next state and never mutates the immutable prior state.

## Canonical order and invariance

Input intent order is non-semantic.

Resolved transition records are canonicalized by:

```text
(actor_id, decision_id, action.id)
```

Operations for transition-batch serialization are canonicalized by state-cell key:

```text
(entity_type, entity_id, state_variable)
```

For a valid disjoint batch:

```text
advance(X, (a, b)).content_hash
==
advance(X, (b, a)).content_hash
```

and transition order, batch hash, next-state payload/hash, and result hash are identical.

## Identity and lineage

### Model identity

`WorldTransitionModelSpec.content_hash` binds:

- model ID/version;
- exact domain ID/version/hash;
- canonical action transition declarations;
- effect capabilities;
- measured transition-hook implementations.

### Initial state lineage

Step-zero state binds canonical story hash, domain identity, objective replay cutoff, and values.

### Step lineage

Each transition record binds intent, canonical actor/action, transition spec, prior state, and returned delta.

The canonical transition batch is content-hashed and bound into `next_state.transition_batch_hash`.

The result therefore supports:

```text
selection result
    -> action intent
    -> canonical action
    -> transition model
    -> delta
    -> world step
    -> next world state
```

`selection_result_hash` is provenance only. The world layer does not independently prove that a psychological model really selected the named action; it proves that the named action is canonical and that its execution is tied to the supplied upstream result identity.

## advance_world_step()

```python
advance_world_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: WorldState,
    model: WorldTransitionModelSpec,
    intents: tuple[ActionIntent, ...],
) -> WorldStepResult
```

Execution order:

```text
validate canonical story/domain
    -> validate prior world state completely
    -> validate model/domain identity
    -> resolve canonical intents/actions/actors
    -> enforce one intent per actor
    -> resolve effect capabilities
    -> execute every hook against one shared snapshot
    -> validate every delta
    -> reject actual write conflicts
    -> canonical atomic commit
    -> build lineage-bound next state/result
```

No RNG or execution-order parameter exists in V1.

## Public API surface

`narrative_dynamics.narrative` adds exactly 11 names:

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

## Fail-closed error boundary

`WorldTransitionError` covers at least:

- story/domain mismatch;
- world-state/domain/model mismatch;
- forged/invalid prior state cell or value;
- empty intent batch;
- duplicate decision IDs;
- duplicate canonical actors;
- missing decision/action;
- unsupported action type;
- transition model naming undeclared action type;
- invalid actor/argument effect target;
- undeclared or mistyped state variable;
- hook attestation unavailable when identity is requested;
- transition-hook exception (wrapped/chained);
- hook returning non-`StateDelta`;
- duplicate cell writes inside one delta;
- writes outside effect capability;
- invalid set/clear shape;
- invalid typed set value;
- decision after a numeric source cutoff.

`WorldTransitionConflictError` covers only cross-intent actual-write overlap.

## Test strategy

Implementation follows test-only RED before production code.

Primary new test file:

```text
tests/test_narrative_world_transition.py
```

World-specific test fixtures stay local to that module unless a later independent consumer proves a shared-fixture need. `tests/narrative_test_support.py` is not expanded solely for V1.

The initial RED must lock at least:

1. `world_state_from_story(...).values == objective_state(...)` for the same cutoff and exact source lineage;
2. single canonical action produces the declared state change and parent/step lineage;
3. a manually forged prior state with wrong subject/state/value typing rejects before hook execution;
4. two hooks in one step observe exactly the same prior snapshot;
5. disjoint action batches are order-invariant in payload and content hash;
6. different-value writes to one cell reject with `WorldTransitionConflictError`;
7. identical-value writes to one cell also reject;
8. any invalid intent/delta causes atomic whole-step failure and leaves prior state unchanged;
9. two decisions by the same canonical actor in one step reject;
10. missing decision, missing action, and uncovered action type reject typed;
11. actor-target effect capability enforces canonical actor type;
12. argument-target effect capability requires the named canonical `EntityRef` argument and correct subject type;
13. a hook writing a valid domain cell outside declared capability rejects;
14. duplicate writes inside one action delta reject;
15. invalid set/clear shape, unknown state variable/entity, and wrong value type reject;
16. explicit no-op transition (empty effects + empty delta) succeeds, increments lineage, and preserves extensional values;
17. numeric source cutoff rejects a later authored decision;
18. model identity changes with domain hash, effect declaration, or measured hook implementation, but not transition tuple insertion order;
19. transition/result lineage hashes bind exact prior state, intent, action, delta, and model;
20. exact scoped public API adds the 11 names while root isolation remains unchanged;
21. full existing Python/movie/Lean regressions remain green.

As with Intentional Decision V1, semantic world-transition GREEN should be established before final scoped export wiring if that produces a cleaner RED boundary.

## Expected implementation files

The complete V1 increment is limited to:

```text
docs/superpowers/specs/2026-08-25-narrative-world-transition-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-world-transition-v1.md
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_world_transition.py
tests/test_narrative_trust_api.py
```

No GenericNarrative schema, DomainSpec, replay, deterministic decision, uncertain belief, intentional decision, existing domain/movie fixture, top-level package, registry, observational protocol, prison model, or Lean-source change belongs to this increment.

If correct implementation appears to require one of those files, stop and re-review the architecture rather than silently broadening scope.

## Relationship to later increments

V1 establishes:

```text
X_t + selected actions -> X_t+1
```

The next independent increment can add observation projection:

```text
X_t+1 + visibility/channel model -> O_i,t+1
```

A later scheduler can close the loop:

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

This preserves separately testable model families for perception, belief, goal selection, action selection, world mechanics, and planning.

## Acceptance boundary

World Transition V1 is complete only when one exact feature head satisfies all of:

1. dedicated world-transition tests green;
2. full existing Python suite green;
3. exact narrative public-surface/root-isolation green;
4. Lean/Python conformance green;
5. full Lean build green;
6. Lean theorem tests green;
7. Narrative StoryState theorem gate green;
8. Narrative Testimony theorem gate green;
9. diff remains inside the approved six-file boundary;
10. integration into `proof/narrative-dynamics-v0` occurs only after explicit human merge approval.
