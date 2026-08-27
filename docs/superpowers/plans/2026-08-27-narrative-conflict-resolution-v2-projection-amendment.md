# Narrative Conflict Resolution V2 — Observation Projection Plan Amendment

Date: 2026-08-27

Status: approved implementation-plan amendment

Parent plan: `docs/superpowers/plans/2026-08-27-narrative-conflict-resolution-v2.md`

Parent design amendment: `docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-projection-amendment.md`

Authority: this document amends the parent implementation plan. Where the parent plan excludes `observation_projection.py`, expects scheduler acceptance to pass with world changes alone, or lists a final scope without projection compatibility, this amendment governs.

## Global-constraint changes

Replace the parent constraint that forbids changes to `observation_projection.py` with:

- `observation_projection.py` may change only to validate and consume a resolved `WorldStepResult` using effective-mutation semantics already certified by World Transition V2.
- resolver-free projection behavior remains exact V1.
- projection must not execute or route conflict resolvers.
- `runtime_perception.py`, cognition families, dispatch, scheduler production, IR/domain, model-comparison code, package root, and Lean remain unchanged.

## File-map changes

Add:

- Modify: `narrative_dynamics/narrative/observation_projection.py` — resolution-aware source-step validation and provenance only.
- Create: `tests/test_narrative_conflict_projection.py` — focused RED/GREEN contract for resolved-step projection.

## Execution-gate changes

Tasks 1-4 remain unchanged. Existing authoritative RED evidence remains valid:

- proof #1116, exact head `0b98621c97bdf3363fb96cc307aad8a77b720eae`;
- corrected safety RED proof #1118, exact head `b913825a149d0f991a53452dcb475b58bbad889f`.

Task 2 records/spec commit remains `22e38a370b41ad241d5d62ac269e3211a2002477`.

### Amended Task 5A: World Conflict Core GREEN

Complete the parent Task 5 world work first, but judge core GREEN using world/conflict tests only:

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_conflict_resolution_safety \
  tests.test_narrative_world_transition -v
```

Required behavior:

- connected components;
- typed resolver execution;
- union-capability validation;
- final global collision rejection;
- atomic resolved `WorldStepResult`;
- explicit V2 batch lineage;
- exact resolver-free V1 behavior/hash preservation.

Do not require heterogeneous scheduler acceptance yet, because projection remains intentionally V1-only at this gate.

Commit world core atomically before projection changes.

### New Task 5B: Observation Projection Test-Only RED

**Files:**
- Create: `tests/test_narrative_conflict_projection.py`
- Production: no changes

Using a valid resolved world step produced by the now-GREEN world core, add tests that lock:

1. resolver-free V1 projection still passes exact existing semantics;
2. valid resolved claim step reaches `project_world_observations()` successfully;
3. raw overlapping attempts do not invalidate a correctly resolved step;
4. observation of a resolver-written cell carries canonical participant transition hashes;
5. resolver-created legal write projects final post-step truth;
6. resolved explicit clear is accepted as one current-step clear;
7. forged participant transition reference rejects;
8. forged V2 batch hash rejects;
9. effective replay mismatch rejects;
10. unresolved raw overlap without valid resolution still rejects.

Commit only this new test file:

```bash
git add tests/test_narrative_conflict_projection.py
git commit -m "test: define resolved world projection contract"
```

Require PR-triggered exact-head RED. Accept RED only if existing projection tests remain GREEN and new failures are confined to V1-only source validation rejecting resolved-step lineage.

### New Task 5C: Resolution-Aware Projection GREEN

**Files:**
- Modify: `narrative_dynamics/narrative/observation_projection.py`

Implement one internal branch in source-world-step validation:

```text
world_step.conflict_resolutions empty
    -> exact existing V1 validation path

world_step.conflict_resolutions non-empty
    -> validate all raw transitions as canonical attempts
    -> derive resolved transition hashes from resolution participants
    -> build effective mutation set:
         non-conflicting raw deltas + resolved deltas
    -> globally reject effective collisions
    -> replay effective mutations to next-state values
    -> verify V2 batch hash
```

Add an internal provenance helper that maps one observed cell to effective source transition hashes:

- non-conflicting raw write/clear -> existing one-transition provenance;
- resolution delta write/clear -> all participant transition hashes of that one resolution, canonical order;
- no effective current-step write -> empty tuple.

Do not change `ProjectedObservation` public fields.

Run:

```bash
python3 -m unittest \
  tests.test_narrative_conflict_projection \
  tests.test_narrative_observation_projection -v
```

Expected: PASS with existing projection regression suite unchanged.

Commit only projection production:

```bash
git add narrative_dynamics/narrative/observation_projection.py
git commit -m "feat: project resolved world transitions"
```

### Amended Task 5D: Heterogeneous Scheduler GREEN

After Task 5C, run:

```bash
python3 -m unittest \
  tests.test_narrative_conflict_resolution \
  tests.test_narrative_conflict_resolution_safety \
  tests.test_narrative_conflict_projection \
  tests.test_narrative_world_transition \
  tests.test_narrative_observation_projection \
  tests.test_narrative_simulation -v
```

The heterogeneous Conflict V2 fixture must now prove the complete pipeline:

```text
Reactive / Intentional / Planning
  -> shared prior ledger
  -> ActionIntents
  -> raw overlapping ActionTransitionRecords
  -> ConflictResolutionRecord
  -> atomic next WorldState
  -> resolution-aware projection
  -> percept admission
  -> next-round cognition
```

Require exact two-round deterministic replay.

Production diff of `narrative_dynamics/narrative/simulation.py` must remain empty.

Require exact-head PR core GREEN before public-surface Task 6.

## Public-surface gate

Tasks 6 and 7 retain the same five new public names. This amendment adds no public API name.

## Final-scope amendment

Allowed base-to-feature paths are now exactly the parent allowed paths plus:

```text
docs/superpowers/specs/2026-08-27-narrative-conflict-resolution-v2-projection-amendment.md
docs/superpowers/plans/2026-08-27-narrative-conflict-resolution-v2-projection-amendment.md
narrative_dynamics/narrative/observation_projection.py
tests/test_narrative_conflict_projection.py
tests/test_narrative_conflict_resolution_safety.py
```

`tests/test_narrative_conflict_resolution_safety.py` records the pre-production safety RED that was added after the parent plan was written.

Any production diff in `simulation.py`, runtime cognition/model families, dispatch, runtime perception, IR/domain, model comparison, package root, or Lean remains a scope failure.

## Final verification amendment

Before final exact-head PR GREEN, require:

```bash
python3 -m unittest discover -s tests -v
```

plus PR-triggered proof success including dependency, conformance, Lean build, theorem tests, Python, Story, and Testimony.

Final reviewer must explicitly confirm that resolver-free world and projection paths retain their exact V1 payload/hash semantics while resolved steps use explicit resolution lineage end-to-end.
