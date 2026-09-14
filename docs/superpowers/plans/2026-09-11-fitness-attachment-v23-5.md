# Fitness Attachment V23.5 Implementation Record — BB-only maintained summary

Status: completed historical implementation plan, condensed for the maintained BB-only repository vocabulary. The original task-by-task execution record remains available in Git history.

## Goal and architecture

V23.5 built the exact finite BB attachment kernel that is now the repository's only attachment-model identity. The implementation separates:

1. exact rational normalization;
2. ordered selection without replacement;
3. atomic graph growth;
4. raw validation;
5. deterministic replay over supplied targets;
6. whole-trace exact probability and normalization;
7. the positive-probability seven-hop scope counterexample.

The authoritative graph state is `SimpleGraph (Fin n)` plus strictly positive rational fitness. Degree is always derived from actual adjacency. No second degree counter, second probability kernel, or alternate attachment-model implementation is maintained.

## Global constraints preserved by the implementation

- Valid seeds are connected, have `n >= 2`, and positive fitness everywhere.
- A growth trace fixes `m` with `1 <= m <= n0`.
- Each birth selects ordered distinct old vertices only; the newborn is not eligible during its own birth.
- The current graph/degree/fitness snapshot is frozen during one birth, while the target row is renormalized after each selected target.
- Zero total mass and invalid raw inputs are explicit errors; there is no uniform or epsilon fallback.
- The entire raw birth request is validated before a successor is produced.
- An unordered target-set event sums all its orderings.
- Replay is deterministic and consumes a supplied target trace; it is not a sampler.
- No `sorry`, `admit`, new user axioms, `native_decide` proof oracle, unsafe bypass, unchecked array default, or unlimited resource setting is allowed.
- Existing `MeshWalk`, `ReachWithin`, `SmallWorldMetrics`, Story, Testimony, conformance, and unrelated narrative semantics remain unchanged.
- No universal six-hop, power-law exponent, condensation, clustering, or asymptotic-distance claim follows from this finite layer.

## Implemented surfaces

### Task 1 — Checked exact rational normalization

Implemented finite rational `Row n`, checked normalization, explicit error cases, exact output mass formula, support, nonnegativity, total-one, and list adapters with bounded indexing.

Key outputs include:

```text
normalize
normalizeValues
normalize_mass_eq
normalize_sum_one
normalize_support
normalize_nonneg
normalize_conditions
normalize_exists_iff
```

### Task 2 — Graph-derived BB weights and fitness laws

Implemented:

```text
Snapshot
State
degree
weights
attachmentRow
scaleFitness
```

and generic properties for positive degree in valid connected states, positive remaining mass, exact weight ratios, common positive scaling, constant-fitness normalization, monotonicity, and strict monotonicity with positive competing mass.

The maintained BB-native names for constant-fitness comparison are:

```text
unitFitnessState
unitFitnessRow
attachment_constant_fitness
```

### Task 3 — Complete ordered target law

Implemented computable finite `Targets n m`, ordered and selected views, `traceMass`, complete ordered target mass `orderedMass`, unordered event mass `setMass`, positivity, total-one, wrong-cardinality zero mass, common scaling, and BB-native constant-fitness normalization:

```text
orderedMass_constant_fitness
orderedMass_scale
```

The implementation uses sequential conditional selection without replacement; it does not multiply independent initial-row probabilities.

### Task 4 — Atomic graph birth

Implemented `applyBirth` and proved old adjacency preservation, newborn adjacency exactly to selected targets, no self-loop, immutable old fitness, exact vertex/edge growth, degree updates, connectedness, order-independence of the graph result for the same selected set, and walk/reachability lifting.

### Task 5 — Raw validation

Implemented checked raw seed/birth parsing with explicit errors for invalid node counts, fitness size, nonpositive fitness, malformed or duplicate edges, disconnected seeds, invalid `m`, wrong target counts, out-of-range targets, and duplicate targets.

No partial successor is returned on validation failure.

### Task 6 — Deterministic finite replay

Implemented `replay`, `runBirths`, exact final node/edge/fitness properties, strict positivity of accepted trace probability, and `continuationMass` over every legal future target choice under a fixed positive fitness schedule.

`continuationMass_one` proves the complete finite conditional law normalizes to one on the real evolving graph.

Whole-schedule common scaling is preserved by:

```text
replay_scale
replay_scale_topology
replay_scale_probability
replay_constant_fitness_topology
replay_constant_fitness_probability
```

The regression boundary is explicit: scaling only seed fitness while leaving future newborn fitness unchanged changes a two-birth probability from `24/805` to `24/1505`.

### Task 7 — Positive-probability seven-hop counterexample and trust audit

The verified fixture begins from a two-node edge, uses `m=1`, unit positive fitness, and targets the current endpoint successively. It proves the actual replayed eight-vertex path:

```text
0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
```

with:

```text
nodes = 8
edges = 7
degrees = [1,2,2,2,2,2,2,1]
trace probability = 1/46080 > 0
shortestHopCount 0 7 = 7
meshDiameter = 7
```

The lower bound is proved, not inferred: any walk can advance the vertex label by at most one per edge, so no route of at most six hops can connect the endpoints.

## Verification boundary

The V23.5 line was verified with Lean 4.32.0 and pinned mathlib revision `81a5d257c8e410db227a6665ed08f64fea08e997`. The proof workflow covers conformance, the full root build, fitness attachment/birth/validation/replay/scope tests, general Lean theorems, Story, and Testimony.

Axiom reports for the audited theorems are limited to ordinary dependencies already present in the repository, principally `propext`, `Classical.choice`, and `Quot.sound`; no `sorryAx` or new user axiom is accepted.

The current fitness-branch workflow intentionally excludes peripheral Python discovery and World Studio jobs. Those exclusions are scope decisions, not verification successes.

## Handoff to V23.6

V23.5 provides the single-trace and recursive finite-law foundation:

```text
orderedMass
applyBirth
replay
continuationMass
continuationMass_one
```

V23.6 may make the complete finite trace space explicit and build exact event probabilities and expectations over it. It must reuse these semantics and keep BB as the only maintained attachment-model identity.
