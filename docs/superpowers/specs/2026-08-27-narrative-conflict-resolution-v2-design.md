# Narrative Conflict Resolution V2 Design

Date: 2026-08-27

Status: design for review

Roadmap: #27 — P1 Conflict Resolution V2

Integrated base: `proof/narrative-dynamics-v0` at `1ad91cd3782e1da0154e3354737bbebf3f369197`

## 1. Purpose

World Transition V1 deliberately rejects every overlapping write set. That gives deterministic simultaneous semantics and prevents implicit last-writer-wins, but it cannot represent modeled interactions such as attack/defense, competing resource claims, auctions, or contests.

Conflict Resolution V2 adds one explicit, attested world-level resolver between validated per-action transition records and the final atomic world commit.

```text
Reactive | Intentional | Planning
              |
              v
       RuntimeDecisionDispatchResult
              |
              v
          ActionIntent
              |
              v
      per-action transition hooks
        all read one prior snapshot
              |
              v
      ActionTransitionRecord[]
              |
              v
      conflict graph/components
              |
       +------+------+
       |             |
   no overlap     overlap
       |             |
       |       ConflictResolver
       |             |
       +------+------+
              |
              v
       validate final deltas
              |
              v
       one atomic WorldState commit
```

Conflict resolution is world semantics. It is not cognition, planning, scheduling, observation projection, or model comparison.

## 2. Non-goals

This feature does not:

- change Reactive, Intentional, or Planning algorithms;
- let any cognitive model read objective world truth;
- add stochastic world or observation behavior;
- add RNG or seeds;
- add actor priority;
- add implicit lexical winner selection;
- add last-writer-wins semantics;
- modify GenericNarrative IR or DomainSpec identities;
- modify observation projection or runtime perception;
- integrate the three cognitive families into held-out model comparison;
- add a general game-theoretic mechanism language;
- change scheduler eligibility or decision timing.

## 3. Existing invariants to preserve

World Transition V1 already guarantees:

1. Every ActionIntent resolves to one authored Decision and ActionOption.
2. One actor contributes at most one action per world step.
3. Every action transition hook reads the same immutable prior world snapshot.
4. Every action hook returns a typed StateDelta.
5. Every delta write is capability-limited by the selected action's declared ActionEffectSpec values.
6. Every StateDelta is canonicalized before becoming an ActionTransitionRecord.
7. All records bind the exact prior WorldState hash.
8. Input order does not determine execution semantics.
9. No next WorldState is constructed until all per-action deltas have passed validation.
10. Any overlapping write set currently fails with WorldTransitionConflictError.

Conflict Resolution V2 starts only after the per-action transition records are valid and bound to one prior snapshot.

## 4. Chosen architecture

Use connected conflict components over validated ActionTransitionRecords.

A conflict graph has one vertex per transition record. Two records have an undirected edge when their actual StateDelta write sets intersect.

The connected components containing at least two records are conflict components.

Example:

```text
Alice writes X
Bob   writes X,Y
Carol writes Y
```

The engine forms one component `{Alice, Bob, Carol}`. It must not resolve `(Alice,Bob)` and then `(Bob,Carol)` sequentially because pairwise resolution order could change the result.

Each connected component is passed exactly once to the configured ConflictResolver.

## 5. Module boundary

Add:

`narrative_dynamics/narrative/conflict.py`

This module owns pure conflict records and resolver specification. It imports domain/IR/value primitives, but **does not import `world.py`**.

`world.py` imports those conflict records/specs and owns the runtime error subclass plus execution integration. This prevents a `conflict.py <-> world.py` import cycle.

## 6. ConflictParticipant

```python
@dataclass(frozen=True)
class ConflictParticipant:
    actor_id: str
    decision_id: str
    action: ActionOption
    transition_record_hash: str
    transition_spec_hash: str
    original_delta: StateDelta
    allowed_write_cells: tuple[StateCellRef, ...]
```

Requirements:

- all textual ids are non-empty trimmed strings;
- hashes are canonical sha256 content hashes;
- `action` is the exact canonical authored ActionOption from the referenced ActionTransitionRecord;
- full typed `action.arguments` are preserved because domain resolvers need semantic inputs such as bid amount, target, resource id, attack mode, or defense mode;
- `original_delta` is exactly the canonical delta in the referenced ActionTransitionRecord;
- `allowed_write_cells` is the exact canonical action-effect capability computed by world execution;
- cells are unique and canonical lexical order;
- the record is frozen and content-hashed.

The participant carries no belief, goal, cue, planning trace, RuntimeEvidenceLedger, SimulationState, or observation object.

The participant action type is always read from `participant.action.type_name`; no redundant independently forgeable `action_type` field is added.

## 7. ConflictResolutionContext

```python
@dataclass(frozen=True)
class ConflictResolutionContext:
    prior_state_hash: str
    participants: tuple[ConflictParticipant, ...]
    conflict_cells: tuple[StateCellRef, ...]
```

Requirements:

- at least two participants;
- participants sorted by `(actor_id, decision_id, action.id)`;
- participant keys unique;
- `conflict_cells` is the exact set of cells written by at least two participants;
- conflict cells are unique and canonical lexical order;
- every conflict cell belongs to the union of participant allowed_write_cells;
- context content hash is stable under original intent/record input ordering.

The context contains the attempted actions and original deltas because a resolver must reason about what actors attempted, not merely which cell collided.

## 8. ConflictResolverSpec

```python
@dataclass(frozen=True)
class ConflictResolverSpec:
    resolver_id: str
    version: str
    domain_id: str
    domain_version: str
    domain_spec_hash: str
    supported_action_types: tuple[str, ...]
    resolver_hook: object
```

The resolver is world-model configuration, not part of DomainSpec.

Requirements:

- resolver identity binds id, version, domain identity, supported action types, and measured implementation identity of `resolver_hook`;
- supported action types are non-empty, unique, canonical lexical order;
- all supported action types must exist in DomainSpec before any action transition or resolver hook executes;
- every participant `action.type_name` in a conflict component must be explicitly supported;
- `resolver_hook` must be callable;
- implementation attestation failure is a typed world-transition failure before resolver execution.

Hook signature:

```python
resolver_hook(
    prior_snapshot: Mapping[StateCellRef, TypedValue],
    context: ConflictResolutionContext,
) -> StateDelta
```

The prior snapshot is the same objective snapshot already passed to action transition hooks. Giving objective state to a world resolver does not grant objective-state capability to cognitive models.

## 9. ConflictResolutionRecord

```python
@dataclass(frozen=True)
class ConflictResolutionRecord:
    resolver_id: str
    resolver_hash: str
    prior_state_hash: str
    context: ConflictResolutionContext
    resolved_delta: StateDelta
```

Requirements:

- resolver id/hash are well-formed;
- prior state hash is well-formed and equals the context prior state hash;
- context and resolved delta are typed canonical records;
- frozen and content-hashed.

Important trust distinction:

`ConflictResolutionRecord` is public canonical data, not independent certification that its resolver hash belongs to a particular WorldTransitionModelSpec or that its participant hashes correspond to a particular WorldStepResult.

`advance_world_step()` performs those cross-object checks at runtime **before constructing a successful WorldStepResult**. `WorldStepResult.__post_init__` then checks all consistency that can be established from the records it actually contains.

## 10. Runtime error ownership

Keep existing:

```python
class WorldTransitionConflictError(WorldTransitionError): ...
```

Add in `world.py`:

```python
class WorldTransitionConflictResolutionError(WorldTransitionConflictError):
    """A declared conflict could not be resolved safely."""
```

Resolver hook exceptions, unsupported conflict action types, invalid outputs, capability violations, and post-resolution collisions are wrapped in this typed error and preserve an underlying exception as `__cause__` when one exists.

## 11. WorldTransitionModelSpec integration

Extend WorldTransitionModelSpec with an optional final field:

```python
conflict_resolver: ConflictResolverSpec | None = None
```

### 11.1 Backward identity compatibility

This is a hard requirement.

When `conflict_resolver is None`, `WorldTransitionModelSpec.to_dict()` must be semantically identical to V1 output. It must not emit a new `conflict_resolver: null` key.

Therefore resolver-free world-model hashes remain exact.

When a resolver is configured, `to_dict()` adds:

```text
conflict_resolver_hash
```

and the world-model identity changes.

Changing resolver id, version, supported action types, or implementation bytes changes the resolver hash and world-model hash.

## 12. Conflict detection

After every ActionTransitionRecord has been produced:

1. derive the actual write set of each canonical record;
2. construct an undirected overlap graph;
3. compute connected components deterministically;
4. discard singleton components;
5. sort members by the existing transition-record key `(actor_id, decision_id, action.id)`;
6. sort components by the lexical tuple of their member keys.

Conflict detection uses actual validated deltas, not declared capabilities alone. Two actions whose declared capabilities overlap but whose actual deltas do not overlap are not a runtime conflict.

An empty/no-op delta cannot create an edge.

## 13. Resolver input construction

For each conflict component, the world layer constructs ConflictParticipant values from the already validated ActionTransitionRecords and their resolved authored Decision/ActionOption data.

The world layer recomputes each participant's `allowed_write_cells` using the same `_allowed_cells(...)` logic used before the action transition hook. It never trusts hook-supplied capabilities.

`conflict_cells` is recomputed extensionally from actual participant deltas.

Before the resolver hook runs, runtime preflight verifies:

- exact prior state hash;
- exact transition-record hashes;
- exact actor/decision/action identities;
- exact authored ActionOption payloads including typed arguments;
- exact original deltas;
- exact effect capabilities;
- exact conflict-cell set;
- resolver supports every participant action type;
- resolver attestation is available.

Any mismatch fails before resolver execution.

## 14. Resolver output semantics

A resolver returns one replacement StateDelta for the entire connected conflict component.

This delta replaces the original deltas of every participant in that component for purposes of the final world mutation.

The original ActionTransitionRecords are retained unchanged for audit lineage.

Whole-component replacement is required for coherent interactions.

Example:

```text
Alice attack Bob:
  Bob.health = injured
  Alice.status = attack-success

Bob defend:
  Bob.health = healthy
  Bob.status = defense-success
```

If defense succeeds, resolving only `Bob.health` while retaining `Alice.status = attack-success` would create contradictory state. The resolver may instead return:

```text
Bob.health = healthy
Alice.status = attack-failed
Bob.status = defense-success
```

provided every write is within the component capability union.

## 15. Resolver capability boundary

The maximum legal write capability of a conflict component is:

```text
union(participant.allowed_write_cells)
```

The resolver cannot write any other cell.

The returned StateDelta is validated with the same typed rules as action deltas:

- StateDelta only;
- declared subject;
- declared state variable;
- subject type matches state variable;
- `set` carries a correctly typed TypedValue;
- `clear` carries no value;
- no duplicate writes inside one resolved delta;
- every write lies inside the component capability union;
- canonical operation ordering.

The resolver may omit cells. Omission means the conflict component makes no write to that cell; it does not implicitly preserve one participant's original write.

A resolver may return an empty delta, meaning the conflicting attempts collectively produce no state change.

## 16. Effective mutation set

Partition canonical ActionTransitionRecords into:

- non-conflicting records: records belonging to no conflict component;
- conflicting records: records belonging to exactly one connected component.

The final mutation set is:

```text
original delta from every non-conflicting record
+
one resolved delta per conflict component
```

Original deltas from conflicting records are never applied directly after resolution.

## 17. Final global collision validation

After all components resolve, perform one global collision check over the final mutation set.

This catches:

- a resolver writing a cell also written by a record that was originally non-conflicting;
- two independent conflict-component resolutions writing the same cell through broader declared capabilities;
- any new collision introduced by resolver output.

Any final collision fails the whole world step with WorldTransitionConflictResolutionError.

There is never a fallback ordering rule.

## 18. Behavior with no resolver

If no conflict exists:

- execute exact V1 atomic semantics;
- preserve exact resolver-free WorldStepResult payload/hash.

If a conflict exists and `conflict_resolver is None`:

- raise the existing WorldTransitionConflictError;
- preserve existing same-value and different-value conflict rejection;
- produce no ConflictResolutionRecord.

Upgrading the library does not change resolver-free simulations.

## 19. Resolver configured but unused in one step

A configured resolver changes WorldTransitionModelSpec identity because it changes declared world semantics.

If a particular step has no conflict:

- the resolver hook is not called;
- no ConflictResolutionRecord is produced;
- action deltas are applied exactly as in V1.

The WorldStepResult hash may differ from the equivalent resolver-free model because its `model_hash` legitimately differs.

## 20. WorldStepResult lineage

Extend WorldStepResult with an optional final field:

```python
conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()
```

### 20.1 Backward payload compatibility

When empty, `WorldStepResult.to_dict()` omits `conflict_resolutions` entirely. Existing resolver-free, conflict-free result payloads remain exact.

When non-empty, `to_dict()` adds canonical resolution records.

### 20.2 Runtime certification vs constructor consistency

Before constructing WorldStepResult, `advance_world_step()` must verify:

- every resolution record uses the configured resolver id/hash;
- every resolution prior hash equals the exact prior WorldState hash;
- every participant transition-record hash belongs to the exact conflict component derived from this step;
- the context embeds the exact authored ActionOption payload and original delta for each participant;
- component membership is complete and non-overlapping;
- resolved deltas are exactly the validated runtime outputs.

`WorldStepResult.__post_init__`, which does not receive a WorldTransitionModelSpec object, validates only what is derivable from its own payload:

- resolution prior hashes equal `prior_state.content_hash`;
- participant transition hashes exist in `transitions`;
- a transition appears in at most one conflict resolution;
- resolution component ordering is canonical;
- next-state transition-batch hash matches the result's transitions + resolutions under the rules below.

It must not pretend to independently prove that an arbitrary resolver hash belongs to an external model spec.

## 21. Transition batch lineage

V1 computes `next_state.transition_batch_hash` from canonical ActionTransitionRecords only.

For backward compatibility:

- when no resolution occurs, use the exact existing V1 batch hash function/payload;
- when one or more resolutions occur, use a V2 batch payload containing both original transitions and explicit resolution records.

Conceptually:

```text
{
  "transitions": [...],
  "conflict_resolutions": [...]
}
```

The V2 batch hash therefore records both what every actor attempted and how the world rule resolved the conflict.

There is no separate mode flag; non-empty resolution lineage selects the V2 payload.

## 22. Atomicity

The engine must not construct a successful next WorldState until all of the following succeed:

1. prior-state validation;
2. transition declaration validation;
3. resolver declaration/domain validation;
4. intent resolution;
5. all action transition hooks;
6. all action delta validation;
7. conflict graph construction;
8. resolver preflight for every component;
9. every resolver hook;
10. every resolved delta validation;
11. final global collision validation;
12. final canonical mutation application.

Any failure leaves the immutable prior WorldState unchanged and returns no successful WorldStepResult.

## 23. Snapshot semantics

All action transition hooks receive the same immutable prior snapshot.

All conflict resolver hooks receive that same prior snapshot.

Resolvers do not observe:

- another resolver invocation's output;
- partially applied action deltas;
- a temporary intermediate WorldState.

Independent components therefore cannot communicate through resolver execution order.

## 24. Determinism

Without a stochastic model, identical canonical inputs must produce identical:

- conflict graph;
- connected components;
- participant ordering;
- conflict cell ordering;
- resolver contexts;
- resolution records;
- next WorldState;
- content hashes.

Reordering ActionIntent input must not change any of the above.

The engine introduces no implicit actor priority. If a domain resolver uses lexical actor id as an auction tie-break, that rule lives explicitly inside the attested resolver implementation.

## 25. One resolver per world model

V2 intentionally supports exactly one optional ConflictResolverSpec on a WorldTransitionModelSpec.

This avoids:

- resolver routing ambiguity;
- resolver precedence;
- sequential resolver effects;
- unclear ownership of mixed-action connected components.

The resolver may support several action types and dispatch internally on the canonical participant action-type set. That dispatch logic is part of its attested implementation.

Multiple resolver routing is deferred to a separate architecture change if real domains require it.

## 26. Scheduler integration

`simulate_step()` already gathers every scheduled agent's ActionIntent and passes the complete tuple to `advance_world_step()`.

No scheduler production change is required for the core Conflict Resolution V2 path.

```text
heterogeneous cognitive agents
  -> one shared prior ledger
  -> typed dispatch results
  -> ActionIntents
  -> World Transition V2
       -> original action records
       -> conflict resolution if needed
       -> one atomic next world
  -> projection
  -> percept admission
  -> next simulation state
```

SimulationStepResult continues to bind every scheduled ActionIntent to an original ActionTransitionRecord. ConflictResolutionRecord explains why the effective world delta may differ from the raw attempted deltas.

## 27. Scheduler error propagation

WorldTransitionConflictResolutionError is a subclass of WorldTransitionError.

The scheduler continues to wrap it as:

```text
SimulationStepError("simulation world transition failed")
```

with the exact WorldTransitionConflictResolutionError retained as `__cause__`.

Trajectory wrapping remains unchanged.

## 28. Reference scenario A — competing resource claims

Two agents attempt to set the same resource ownership cell.

```text
Alice -> claim(resource)
Bob   -> claim(resource)
```

Both action hooks produce valid transition records against one prior snapshot. The conflict component contains both attempts and the ownership cell.

A deterministic reference resolver returns one typed ownership delta.

Tests prove:

- reversed input order produces the same result/hash;
- losing claims remain visible in original transition lineage;
- no last-writer-wins path exists;
- resolver identity is explicit in resolution lineage.

## 29. Reference scenario B — auction / contest

At least three participants compete for one resource.

Example test rule:

```text
highest bid argument wins
exact tied highest bid -> lexical actor id wins
```

The resolver reads bid amounts from `ConflictParticipant.action.arguments`.

The lexical tie-break is explicitly encoded in the resolver implementation and therefore attested. It is not an engine default.

The fixture includes a three-participant connected component and verifies complete input-order invariance.

## 30. Reference scenario C — attack / defense

One actor attacks a target while the target defends in the same world step.

The resolver reads semantic target/mode arguments from the canonical ActionOption payloads and sees both original deltas.

It returns one whole-component delta representing the joint outcome.

Tests demonstrate that conflicting participant raw deltas are not partially applied after resolution.

## 31. Heterogeneous cognitive integration fixture

A scheduler integration test uses different cognitive model families, preferably all three:

```text
Reactive Alice
Intentional Bob
Planning Carol
```

Their dispatch results produce ActionIntents in one round that create a modeled world conflict.

The test verifies:

- all agents decided against one shared prior ledger;
- cognitive result contracts are unchanged;
- resolver receives world semantics/action attempts, not cognitive internals;
- final WorldStepResult contains original transitions plus explicit resolution lineage;
- observation/percept admission succeeds from the resolved world step;
- deterministic two-round replay yields the same trajectory hash.

## 32. Validation ordering

Fail-closed ordering is observable and tested.

Before any action hook:

1. narrative/domain validation;
2. prior-state canonical validation;
3. world-model domain identity;
4. transition action-type declarations;
5. resolver domain identity and supported action-type declarations, when configured;
6. intent decision/action resolution.

Before any resolver hook:

1. all action hooks completed;
2. all action deltas typed/capability-valid;
3. conflict components canonical;
4. participants/contexts match exact runtime records/actions;
5. resolver supports every participant action type;
6. resolver implementation attestation available.

Before successful next state:

1. every resolver output typed/capability-valid;
2. final global collision validation passes;
3. runtime lineage bindings are exact.

## 33. Resolver identity / attestation

ConflictResolverSpec identity includes measured resolver-hook implementation identity.

A resolver implementation byte change changes:

- ConflictResolverSpec hash;
- configured WorldTransitionModelSpec hash;
- SimulationModelSpec hash when the world model is embedded;
- downstream simulation state and trajectory identities.

Resolver attestation is obtained before the resolver hook executes.

## 34. Public API

Narrative package exports exactly these new names:

1. `ConflictParticipant`
2. `ConflictResolutionContext`
3. `ConflictResolverSpec`
4. `ConflictResolutionRecord`
5. `WorldTransitionConflictResolutionError`

The first four are defined in `conflict.py`. The error is defined in `world.py`.

`WorldTransitionModelSpec`, `WorldStepResult`, and `advance_world_step` remain existing public names with extended semantics.

The package root `narrative_dynamics/__init__.py` remains unchanged.

## 35. Source layout

Expected production changes:

```text
ADD    narrative_dynamics/narrative/conflict.py
MODIFY narrative_dynamics/narrative/world.py
MODIFY narrative_dynamics/narrative/__init__.py
```

Expected test changes:

```text
ADD    tests/test_narrative_conflict_resolution.py
MODIFY tests/test_narrative_world_transition.py
MODIFY tests/test_narrative_simulation.py
MODIFY tests/test_narrative_trust_api.py
```

No changes are expected in:

```text
runtime_reactive.py
runtime_intention.py
runtime_planning.py
runtime_decision_dispatch.py
runtime_cognition.py
runtime_perception.py
observation_projection.py
ir.py
domain.py
model comparison / calibration modules
Lean sources
```

## 36. TDD locks

The implementation plan must establish RED tests before production for at least these properties.

### 36.1 V1 compatibility

- no resolver + no conflict preserves exact V1 world-model hash;
- no resolver + no conflict preserves exact WorldStepResult payload/hash;
- no resolver + different-value overlap still raises WorldTransitionConflictError;
- no resolver + same-value overlap still raises WorldTransitionConflictError.

### 36.2 Resolver records

- records are frozen/canonical/content-hashed;
- malformed record construction rejects;
- resolver implementation identity changes resolver/model hashes;
- unsupported action type rejects before resolver hook.

### 36.3 Component construction

- two-way conflict;
- three-way transitive connected conflict;
- two independent components;
- no-op actions do not create components;
- input reorder does not change contexts/hashes.

### 36.4 Semantic action payload

- resolver receives exact ActionOption arguments;
- changing bid/target arguments changes participant/context identity;
- action payload forgery rejects before resolver hook.

### 36.5 Capability/schema safety

- resolver cannot write outside union capability;
- invalid TypedValue rejects;
- invalid clear/set shape rejects;
- duplicate output cell rejects;
- hook exception is typed and preserves cause.

### 36.6 Atomicity

- one bad resolver component aborts the entire step;
- prior WorldState remains unchanged;
- no successful next world or downstream projection is produced.

### 36.7 Final collision validation

- resolver vs originally non-conflicting record collision rejects;
- resolution vs resolution collision rejects;
- no hidden execution-order fallback.

### 36.8 Reference semantics

- competing resource claims;
- three-party auction/contest;
- attack/defense whole-component replacement.

### 36.9 Scheduler integration

- heterogeneous cognitive agents can create and resolve a conflict in one round;
- all decisions still bind one shared prior ledger;
- projection binds the resolved WorldStepResult;
- two-round deterministic replay is exact.

## 37. Migration behavior

Existing positional construction remains valid because new fields are optional and appended:

```text
WorldTransitionModelSpec.conflict_resolver = None
WorldStepResult.conflict_resolutions = ()
```

Resolver-free tests should require no expected hash updates for world models and conflict-free results.

Unexpected resolver-free hash drift is a release blocker, not an accepted migration update.

## 38. Security / trust properties

The resolver is a privileged world hook, so the boundary is strict:

- implementation is attested;
- hook cannot supply its own lineage fields;
- hook receives immutable prior snapshot;
- hook receives canonical action attempts, not scheduler/cognition objects;
- hook cannot expand write capability;
- hook cannot choose execution order;
- hook cannot mutate participant records;
- hook cannot return a WorldState directly;
- hook cannot bypass final collision validation.

## 39. Definition of done

P1 Conflict Resolution V2 is complete when:

- a typed ConflictResolver contract exists;
- resolver identity/attestation is provenance-bound;
- conflicts are deterministic connected components;
- resolver receives complete canonical ActionOption semantics;
- output is one typed whole-component StateDelta;
- output is limited to union participant capabilities;
- attack/defense, competing claims, and auction/contest fixtures pass;
- no last-writer-wins path exists;
- the world step remains atomic;
- original attempted transitions and explicit resolution records are both preserved in lineage;
- resolver-free V1 behavior and hashes remain exact where required;
- heterogeneous scheduler integration replays deterministically;
- exact-head CI is GREEN.

## 40. Deferred work

Explicitly deferred:

- multiple resolver routing / precedence;
- stochastic conflict resolution;
- seeded contests;
- continuous-time or initiative ordering;
- strategic anticipation of resolver rules inside Planning models;
- mechanism-design DSL;
- held-out empirical comparison of conflict policies.

Those require separate architecture decisions and must not be smuggled into Conflict Resolution V2.
