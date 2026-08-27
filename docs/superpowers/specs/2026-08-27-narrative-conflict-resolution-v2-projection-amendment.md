# Narrative Conflict Resolution V2 — Observation Projection Amendment

Date: 2026-08-27

Status: approved architectural amendment

Parent design: `docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-design.md`

Authority: this document amends the parent design. Where the parent design says observation projection is unchanged or out of scope, this amendment governs.

## 1. Why this amendment is required

The existing observation-projection trust boundary independently validates a `WorldStepResult` by replaying every raw `ActionTransitionRecord.delta`. It also rejects overlapping raw transition writes and requires the next world state to equal that raw replay.

Conflict Resolution V2 intentionally preserves overlapping raw action attempts in `WorldStepResult.transitions`, replaces each conflicting connected component with one `ConflictResolutionRecord.resolved_delta` for effective mutation, and binds the resulting next state to both attempted transitions and explicit resolution lineage.

Therefore a valid resolved Conflict V2 world step cannot pass the existing V1-only projection validator. This is an architectural compatibility issue, not a scheduler or cognition issue.

## 2. Chosen boundary

Modify only:

`narrative_dynamics/narrative/observation_projection.py`

for production projection compatibility.

Do not modify:

- `runtime_perception.py`;
- `runtime_cognition.py`;
- `runtime_reactive.py`;
- `runtime_intention.py`;
- `runtime_planning.py`;
- `runtime_decision_dispatch.py`;
- `simulation.py`;
- IR/domain identities;
- Lean sources.

No new public observation-projection API is introduced.

## 3. Exact V1 compatibility

When `world_step.conflict_resolutions == ()`, observation projection must preserve the exact existing V1 validation semantics:

- overlapping raw transition writes remain invalid;
- next-state replay uses raw transition deltas exactly;
- transition-batch hash validation uses the exact V1 list-of-transitions payload;
- current observation provenance behavior is unchanged.

Existing V1 projection tests and hashes require no expected-value updates.

## 4. Resolved-step validation

When `world_step.conflict_resolutions` is non-empty, projection validates the source step using the same effective-mutation semantics certified by World Transition V2.

Projection must still validate every original transition as canonical authored data:

- exact intent/decision/action/actor binding;
- exact prior-state hash;
- typed canonical delta operations;
- one action per actor;
- source-cutoff validity.

Raw overlap is allowed only as part of the explicit resolution lineage already contained in the `WorldStepResult`. Projection must not invent, route, or re-run a resolver.

Define:

```text
resolved_transition_hashes
  = union of participant.transition_record_hash
    over all conflict resolutions

effective mutations
  = raw delta of every transition not in resolved_transition_hashes
    + one resolved_delta per ConflictResolutionRecord
```

Projection independently revalidates that:

1. every resolution participant hash refers to an exact raw transition in the step;
2. a transition belongs to at most one resolution;
3. effective mutations have no duplicate final write cell;
4. applying effective mutations to the prior state yields exactly `world_step.next_state.values`;
5. the next-state transition-batch hash equals the canonical V2 payload:

```python
stable_content_hash({
    "transitions": [canonical raw transition payloads],
    "conflict_resolutions": [canonical resolution payloads],
})
```

The projection layer trusts no hidden world-model object and does not certify resolver ownership beyond the internally self-consistent `WorldStepResult` it receives.

## 5. Projection provenance for resolved cells

Observation facts continue to bind the final post-step truth.

For a cell effectively written or cleared by a non-conflicting raw transition, `source_transition_hashes` keeps the existing V1 behavior.

For a cell effectively written or cleared by a `ConflictResolutionRecord.resolved_delta`, `source_transition_hashes` is the canonical tuple of that resolution component's participant transition-record hashes. This preserves attribution to the original actor attempts while the enclosing `source_world_step_hash` binds the explicit resolver lineage and resolved delta.

If a resolver omits a cell and no other effective mutation writes that cell, the current step contributes no transition provenance for that persistent value.

A resolved explicit `clear` counts as the current-step clear for observation validation and uses the participant transition hashes as its provenance.

No new `ProjectedObservation` fields are required.

## 6. Determinism and fail-closed behavior

Resolved projection validation is canonical and input-order invariant.

Any of the following fails with the existing typed `ObservationProjectionError` wrapper before projection hooks are accepted:

- unresolved raw overlap not covered by valid resolution lineage;
- missing/duplicate participant transition references;
- resolution/raw effective collision;
- resolution/resolution effective collision;
- invalid resolved delta shape/type;
- effective replay mismatch with next state;
- V2 transition-batch hash mismatch.

There is no fallback to raw last-writer-wins or projection-time arbitration.

## 7. TDD locks

Before production modification of `observation_projection.py`, add focused test-only RED coverage in:

`tests/test_narrative_conflict_projection.py`

At minimum lock:

- resolver-free V1 projection behavior remains exact;
- a valid resolved claim world step projects successfully;
- overlapping raw attempts remain visible but no longer invalidate a correctly resolved step;
- resolved-cell observation provenance contains all component participant transition hashes in canonical order;
- a resolver-created write within component capability projects from final truth;
- a resolved explicit clear is accepted as current-step clear;
- forged resolution participant lineage rejects before projection hook acceptance;
- forged V2 batch hash rejects;
- effective replay mismatch rejects;
- unresolved raw overlap still rejects;
- heterogeneous Reactive + Intentional + Planning scheduler round can resolve, project, admit, and replay two rounds exactly.

## 8. Source-layout amendment

Final allowed production scope now includes:

```text
narrative_dynamics/narrative/conflict.py
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/__init__.py
```

The addition of `observation_projection.py` is required only to consume the already-certified resolved world-step lineage. It must not gain resolver hooks, cognition objects, scheduler state, or new world-model authority.

## 9. Definition-of-done amendment

Conflict Resolution V2 is not complete merely when `advance_world_step()` can produce a resolved `WorldStepResult`. It is complete only when that resolved result survives the existing world -> observation projection -> percept admission -> next-round cognition pipeline with deterministic replay and exact V1 behavior preserved for resolver-free steps.
