# Narrative Conflict Resolution V2 Design

Date: 2026-08-27

Status: design for review

Roadmap: #27 — P1 Conflict Resolution V2

Integrated base: `proof/narrative-dynamics-v0` at `1ad91cd3782e1da0154e3354737bbebf3f369197`

## 1. Purpose

World Transition V1 deliberately rejects every overlapping write set. That gives deterministic simultaneous semantics and prevents implicit last-writer-wins, but it cannot represent modeled interactions such as attack/defense, competing resource claims, auctions, or contests.

Conflict Resolution V2 adds one explicit, attested world-level resolver between validated per-action transition records and the final atomic world commit.

The change must preserve the existing cognitive architecture:

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

Conflict Resolution V2 starts only after invariant 1-7 have already been established for every participant.

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

## 5. Public records and contracts

Add a focused module:

`narrative_dynamics/narrative/conflict.py`

### 5.1 ConflictParticipant

```python
@dataclass(frozen=True)
class ConflictParticipant:
    actor_id: str
    decision_id: str
    action_id: str
    action_type: str
    transition_record_hash: str
    transition_spec_hash: str
    original_delta: StateDelta
    allowed_write_cells: tuple[StateCellRef, ...]
```

Requirements:

- all textual ids are non-empty trimmed strings;
- hashes are canonical sha256 content hashes;
- `original_delta` is exactly the canonical delta in the referenced ActionTransitionRecord;
- `allowed_write_cells` is the exact canonical action-effect capability computed by world execution;
- cells are unique and canonical lexical order;
- the record is frozen and content-hashed.

The participant carries no belief, goal, cue, planning trace, RuntimeEvidenceLedger, SimulationState, or observation object.

### 5.2 ConflictResolutionContext

```python
@dataclass(frozen=True)
class ConflictResolutionContext:
    prior_state_hash: str
    participants: tuple[ConflictParticipant, ...]
    conflict_cells: tuple[StateCellRef, ...]
```

Requirements:

- at least two participants;
- participants sorted by `(actor_id, decision_id, action_id)`;
- participant ids unique under the same key;
- `conflict_cells` is the exact set of cells written by at least two participants;
- conflict cells are unique and canonical lexical order;
- every conflict cell belongs to the union of participant allowed_write_cells;
- context content hash is stable under original intent/record input ordering.

The context deliberately contains semantic action identities and original typed deltas, because attack/defense and auction rules need to know what agents attempted, not only which cell collided.

### 5.3 ConflictResolverSpec

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
- every participant action type in a conflict component must be explicitly supported by this resolver;
- `resolver_hook` must be callable;
- implementation attestation failure is a typed world-transition failure before resolver execution.

The hook signature is conceptually:

```python
resolver_hook(
    prior_snapshot: Mapping[StateCellRef, TypedValue],
    context: ConflictResolutionContext,
) -> StateDelta
```

The prior snapshot is the same objective snapshot already passed to action transition hooks. Giving objective world state to a world resolver does not grant objective-state capability to any cognitive model.

### 5.4 ConflictResolutionRecord

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

- binds the exact resolver identity;
- binds the exact prior state;
- embeds the exact canonical conflict context;
- embeds the validated canonical replacement delta;
- frozen and content-hashed;
- constructor reconstruction detects forged payloads before a successful WorldStepResult can be certified.

### 5.5 Error types

Keep existing `WorldTransitionConflictError` for unresolved conflicts.

Add:

```python
class WorldTransitionConflictResolutionError(WorldTransitionConflictError):
    """A declared conflict could not be resolved safely."""
```

Resolver hook exceptions, unsupported conflict action types, invalid resolver outputs, resolver capability violations, and post-resolution collisions are wrapped in this typed error and preserve the original exception as `__cause__` when one exists.

## 6. WorldTransitionModelSpec integration

Extend WorldTransitionModelSpec with an optional resolver:

```python
conflict_resolver: ConflictResolverSpec | None = None
```

### 6.1 Backward identity compatibility

This is a hard requirement.

When `conflict_resolver is None`, `WorldTransitionModelSpec.to_dict()` must be byte-for-byte semantically identical to V1 output. It must not emit a new `conflict_resolver: null` key.

Therefore existing resolver-free world model hashes remain exact.

When a resolver is configured, `to_dict()` adds:

```text
conflict_resolver_hash
```

and the world model identity changes.

Changing resolver id, version, supported action types, or implementation bytes changes the world model hash.

## 7. Conflict detection

After every ActionTransitionRecord has been produced:

1. derive the actual write set of each canonical record;
2. construct an undirected overlap graph;
3. compute connected components deterministically;
4. discard singleton components;
5. sort members of every component by `_transition_record_key`;
6. sort components by the lexical tuple of their member keys.

Conflict detection uses actual validated deltas, not declared capabilities alone. Two actions whose declared capabilities overlap but whose actual deltas do not overlap are not a runtime conflict.

An empty/no-op delta cannot create an edge.

## 8. Resolver input construction

For each conflict component, construct ConflictParticipant values from the already validated transition records.

The world layer recomputes each participant's `allowed_write_cells` using the same `_allowed_cells(...)` logic used before the original transition hook. It never trusts a hook-supplied capability declaration.

`conflict_cells` is recomputed extensionally from actual participant deltas.

Before the resolver hook runs, reconstruct all public input records and verify:

- exact prior state hash;
- exact record hashes;
- exact actor/decision/action identities;
- exact action types;
- exact original deltas;
- exact effect capabilities;
- exact conflict-cell set;
- resolver supports all participant action types.

Any mismatch fails before resolver execution.

## 9. Resolver output semantics

A resolver returns one replacement StateDelta for the entire connected conflict component.

This delta replaces the original deltas of every participant in that component for purposes of the final world-state mutation.

The original ActionTransitionRecords are retained unchanged for audit lineage.

This whole-component replacement is necessary for coherent interactions.

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

provided every write is within the union capability described below.

## 10. Resolver capability boundary

The maximum legal write capability of a conflict component is:

```text
union(participant.allowed_write_cells)
```

The resolver cannot write any other cell.

The returned StateDelta is validated using the same canonical typed rules as action deltas:

- StateDelta only;
- declared subject;
- declared state variable;
- subject type matches state variable;
- `set` carries a correctly typed TypedValue;
- `clear` carries no value;
- no duplicate writes inside one resolved delta;
- every write lies inside the component capability union;
- canonical operation ordering.

The resolver may omit cells. Omission means the conflict component makes no write to that cell; it does not implicitly preserve an original participant write.

A resolver may return an empty delta, meaning the conflicting attempts collectively produce no state change.

## 11. Non-conflicting records and resolved components

Partition canonical ActionTransitionRecords into:

- non-conflicting records: records belonging to no conflict component;
- conflicting records: records belonging to exactly one connected component.

The final mutation set consists of:

```text
all original deltas from non-conflicting records
+
one resolved delta per conflict component
```

Original deltas from conflicting records are never applied directly after a resolver runs.

## 12. Final collision validation

After all components have resolved, perform one global collision check over the final mutation set.

This catches cases such as:

- a resolver writes a cell also written by a non-conflicting record through a larger declared capability;
- two independent conflict-component resolutions write the same cell;
- a resolver output creates a new collision that was absent from original actual deltas.

Any final collision fails the entire world step with WorldTransitionConflictResolutionError.

There is never a fallback ordering rule.

## 13. Behavior when no resolver is configured

If no conflict exists:

- execute the exact V1 `_atomic_result` semantics;
- preserve exact WorldStepResult payload and hashes.

If a conflict exists and `conflict_resolver is None`:

- raise the same existing WorldTransitionConflictError class;
- preserve existing conflict-rejection behavior;
- do not synthesize ConflictResolutionRecord values.

Therefore merely upgrading the library does not change resolver-free simulations.

## 14. Behavior when a resolver is configured but unused

A configured resolver changes WorldTransitionModelSpec identity because it changes declared world semantics.

However, if one particular step has no conflict:

- the resolver hook is not called;
- no ConflictResolutionRecord is produced;
- action deltas are applied exactly as in V1;
- WorldStepResult records no resolution event for that step.

The WorldStepResult hash may still differ from a resolver-free model because `model_hash` legitimately differs.

## 15. WorldStepResult lineage

Extend WorldStepResult with:

```python
conflict_resolutions: tuple[ConflictResolutionRecord, ...] = ()
```

### 15.1 Backward payload compatibility

When `conflict_resolutions` is empty, `WorldStepResult.to_dict()` must omit the field entirely. Existing resolver-free, conflict-free result payloads remain exact.

When non-empty, `to_dict()` adds:

```text
conflict_resolutions: [...]
```

in canonical component order.

WorldStepResult validation reconstructs and binds every resolution record to:

- exact prior WorldState;
- exact resolver hash from WorldTransitionModelSpec execution;
- exact participant transition hashes present in `transitions`;
- exact resolved component set;
- exact next-state lineage.

A resolution record may not refer to a transition absent from the WorldStepResult.

A transition may belong to at most one resolution component.

## 16. Transition batch lineage

V1 computes `next_state.transition_batch_hash` from canonical ActionTransitionRecords only.

For backward compatibility:

- when no resolution occurs, use the exact existing V1 batch hash;
- when one or more resolutions occur, compute a V2 batch payload containing both original transition records and conflict resolution records.

Conceptually:

```text
{
  "transitions": [...],
  "conflict_resolutions": [...]
}
```

The V2 batch hash therefore records both what every actor attempted and how conflicts changed the effective world delta.

There is no hash mode flag; the presence of resolution records is sufficient to distinguish V2 lineage.

## 17. Atomicity

The engine must not mutate or construct the successful next WorldState until all of the following succeed:

1. prior-state validation;
2. transition declaration validation;
3. intent resolution;
4. all action transition hooks;
5. all action delta validation;
6. conflict graph construction;
7. resolver preflight for every component;
8. every resolver hook;
9. every resolved delta validation;
10. final global collision validation;
11. final canonical mutation application.

Any failure leaves the immutable prior WorldState unchanged and returns no successful WorldStepResult.

## 18. Snapshot semantics

All action transition hooks continue to receive the same immutable prior snapshot.

All conflict resolver hooks also receive that same prior snapshot.

Resolvers do not observe:

- another resolver's output;
- partially applied action deltas;
- a temporary intermediate WorldState.

Independent conflict components therefore cannot communicate through execution order.

## 19. Determinism

Without a stochastic model, identical canonical inputs must produce identical:

- conflict graph;
- connected components;
- participant ordering;
- conflict cell ordering;
- resolver contexts;
- resolver outputs, assuming deterministic hooks;
- resolution records;
- next WorldState;
- content hashes.

Reordering ActionIntent input must not change any of the above.

The engine introduces no implicit actor priority. If a domain resolver uses lexical actor id as an auction tie-break, that rule lives explicitly inside the attested resolver implementation.

## 20. Scheduler integration

`simulate_step()` already collects every scheduled agent's ActionIntent and passes them together to `advance_world_step()`.

No scheduler production change is required for the basic Conflict Resolution V2 architecture.

Simulation behavior becomes:

```text
heterogeneous cognitive agents
  -> one shared prior ledger
  -> typed dispatch results
  -> ActionIntents
  -> World Transition V2
       -> per-action records
       -> conflict resolution if needed
       -> one atomic next world
  -> projection
  -> percept admission
  -> next simulation state
```

SimulationStepResult continues to bind every scheduled ActionIntent to an original ActionTransitionRecord. ConflictResolutionRecord provides the additional world-level explanation for why the effective state change differs from the raw attempted deltas.

## 21. Failure propagation through scheduler

WorldTransitionConflictResolutionError remains a subclass of WorldTransitionError.

The scheduler therefore continues to wrap it as:

```text
SimulationStepError("simulation world transition failed")
```

with the exact WorldTransitionConflictResolutionError retained as `__cause__`.

Trajectory failure wrapping remains unchanged.

## 22. Reference scenario A — competing resource claims

Two agents attempt to set the same resource ownership cell.

```text
Alice -> claim(resource)
Bob   -> claim(resource)
```

Both transition hooks produce valid ActionTransitionRecords against one prior snapshot.

The conflict component contains both participants and the ownership cell.

A reference resolver applies an explicit deterministic contest rule and returns one typed ownership delta.

Tests must prove:

- reversed input order produces the same result/hash;
- the losing claim remains visible in original transition lineage;
- no last-writer-wins behavior exists;
- resolver identity is present in resolution lineage.

## 23. Reference scenario B — auction / contest

At least three participants compete for one resource.

Example semantic rule in the test resolver:

```text
highest bid wins
exact tied highest bid -> lexical actor id wins
```

The lexical tie-break is explicitly part of the resolver implementation and therefore attested in resolver identity. It is not an engine default.

The test must include a connected component with three participants and verify complete input-order invariance.

## 24. Reference scenario C — attack / defense

One actor attacks a target while the target defends in the same world step.

The two raw deltas overlap and may also contain non-overlapping status writes.

The resolver returns one whole-component delta representing the joint outcome.

Tests must demonstrate that conflicting participant raw deltas are not partially applied after resolution.

## 25. Heterogeneous cognitive integration fixture

A scheduler integration test uses at least two different cognitive model families and preferably all three:

```text
Reactive Alice
Intentional Bob
Planning Carol
```

Their typed dispatch results produce ActionIntents in one round that create a modeled world conflict.

The conflict is resolved in World Transition V2 and the next observation/percept step succeeds.

The test verifies:

- all agents decided against the same prior ledger;
- cognitive result contracts are unchanged;
- resolver sees world semantics only, not cognitive internals;
- the final world step includes original transitions plus explicit conflict-resolution lineage;
- deterministic replay yields the same trajectory hash.

## 26. Resolver identity and attestation

ConflictResolverSpec content identity must include measured implementation identity of the hook.

Preflight obtains the resolver attestation before any conflict resolver hook executes.

A resolver implementation byte change changes:

- ConflictResolverSpec hash;
- configured WorldTransitionModelSpec hash;
- SimulationModelSpec hash when embedded in a scheduler world model;
- downstream simulation state/trajectory identity.

Resolver ordering in authored input is irrelevant because V2 has exactly one optional resolver per WorldTransitionModelSpec.

Multiple resolver routing is intentionally deferred. It would introduce ambiguity about which resolver owns a mixed-action connected component.

## 27. Why one resolver per world model

V2 intentionally chooses one optional resolver rather than a list of resolver rules.

Advantages:

- no ambiguous routing;
- no resolver priority;
- mixed action-type components are handled jointly;
- one world-model identity clearly names the conflict semantics;
- connected components are resolved once, not sequentially.

A resolver may support several action types and dispatch internally on the canonical participant action-type set. That internal rule is part of its attested implementation.

A future architecture may add declarative resolver routing only if real domains require it.

## 28. Validation ordering

Fail-closed ordering is observable and must be tested.

Before any action hook:

1. narrative/domain validation;
2. prior state canonical validation;
3. WorldTransitionModelSpec domain identity;
4. transition action-type declarations;
5. resolver domain identity and supported action-type declarations;
6. intent decision/action resolution.

Before any resolver hook:

1. all action hooks have completed;
2. all action deltas are typed/capability-valid;
3. conflict components are canonical;
4. participant/context records reconstruct exactly;
5. resolver supports all participant action types;
6. resolver attestation is available.

Before successful next state:

1. every resolver output is typed/capability-valid;
2. final global collision validation passes;
3. lineage payloads reconstruct exactly.

## 29. Public API

Narrative package exports exactly these new names:

1. `ConflictParticipant`
2. `ConflictResolutionContext`
3. `ConflictResolverSpec`
4. `ConflictResolutionRecord`
5. `WorldTransitionConflictResolutionError`

`WorldTransitionModelSpec`, `WorldStepResult`, and `advance_world_step` remain existing public names with extended semantics.

The package root `narrative_dynamics/__init__.py` remains unchanged.

## 30. Source layout

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

## 31. Test plan / scientific and engineering locks

The implementation plan must create RED tests before production for at least the following properties.

### 31.1 V1 compatibility

- no resolver + no conflict preserves exact V1 world model hash;
- no resolver + no conflict preserves exact WorldStepResult payload/hash;
- no resolver + different-value overlap still raises WorldTransitionConflictError;
- no resolver + same-value overlap still raises WorldTransitionConflictError.

### 31.2 Resolver records

- records are frozen, canonical, self-validating, and content-hashed;
- constructor forgery is rejected;
- resolver implementation identity changes resolver/model hashes;
- unsupported action type is rejected before resolver hook.

### 31.3 Component construction

- two-way conflict;
- three-way transitive connected conflict;
- two independent components;
- no-op actions do not create components;
- input reordering does not change contexts or hashes.

### 31.4 Capability and schema safety

- resolver cannot write outside union capability;
- invalid TypedValue rejected;
- invalid clear/set shape rejected;
- duplicate output cell rejected;
- hook exception is typed and preserves cause.

### 31.5 Atomicity

- one bad resolver component aborts the entire step;
- prior WorldState remains unchanged;
- no successful WorldStepResult or projected percept exists downstream.

### 31.6 Final collision validation

- resolver vs non-conflicting record collision rejects;
- resolution vs resolution collision rejects;
- no hidden execution-order fallback.

### 31.7 Reference semantics

- competing resource claims;
- three-party auction/contest;
- attack/defense whole-component replacement.

### 31.8 Scheduler integration

- heterogeneous cognitive agents can create and resolve a conflict in one round;
- all decisions still bind one shared prior ledger;
- observation projection binds the resolved WorldStepResult;
- two-round deterministic replay is exact.

## 32. Migration behavior

Existing callers constructing `WorldTransitionModelSpec` positionally remain valid because the new resolver field is optional and appended with a default.

Existing callers constructing `WorldStepResult` positionally remain valid because `conflict_resolutions` is optional and appended with a default empty tuple.

Existing resolver-free tests should require no expected hash updates for world models and conflict-free results.

Any unexpected resolver-free hash drift is a release blocker, not a migration update to accept.

## 33. Security / trust properties

Conflict resolution introduces a privileged world hook, so the trust boundary is strict:

- resolver implementation is attested;
- resolver cannot alter its own lineage fields;
- resolver receives an immutable prior snapshot;
- resolver receives sanitized conflict records, not scheduler/cognition objects;
- resolver cannot expand write capability;
- resolver cannot choose execution order;
- resolver cannot mutate participant records;
- resolver cannot return a WorldState directly;
- resolver cannot bypass final collision validation.

## 34. Definition of done for Conflict Resolution V2

P1 Conflict Resolution V2 is complete when:

- a typed ConflictResolver contract exists;
- resolver identity/attestation is provenance-bound;
- conflict components are deterministic connected components;
- resolver output is one typed whole-component StateDelta;
- output is limited to union participant capabilities;
- attack/defense, competing claims, and auction/contest fixtures pass;
- no last-writer-wins path exists;
- the entire world step remains atomic;
- original attempted ActionTransitionRecords and explicit resolution records are both preserved in lineage;
- resolver-free V1 behavior and hashes remain exact where required;
- heterogeneous scheduler integration replays deterministically;
- exact-head CI is GREEN.

## 35. Deferred work

Explicitly deferred:

- multiple resolver routing / precedence;
- stochastic conflict resolution;
- seeded contests;
- continuous-time or initiative ordering;
- strategic anticipation of the resolver inside Planning models;
- mechanism design DSL;
- held-out empirical comparison of conflict policies.

Those require separate architecture decisions and must not be smuggled into Conflict Resolution V2.
