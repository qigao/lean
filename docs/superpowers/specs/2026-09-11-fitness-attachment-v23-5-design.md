# Fitness Attachment V23.5 — BB foundation (archival current-tree summary)

Status: implemented and verified in the V23.5 line. This maintained copy records the resulting BB contract after the repository vocabulary was simplified to one attachment-model identity. The original review wording remains available in Git history.

## 1. Model boundary

V23.5 defines one finite BB attachment model in `NarrativeDynamics.FitnessAttachment`. It is an undirected simple-graph growth model over `Fin n` with immutable, strictly positive rational fitness. Degree is always computed from actual graph adjacency; there is no independent authoritative degree counter.

A valid seed is connected, has at least two vertices, and assigns positive fitness to every vertex. A growth trace fixes `m` with `1 <= m <= n0`. Each birth supplies one positive newborn fitness and an ordered list of `m` distinct existing targets.

The model is finite and exact. It does not define a PRNG, entropy source, continuous fitness sampler, infinite process, asymptotic degree law, condensation classification, or universal small-world guarantee.

## 2. Exact attachment kernel

For a frozen current state and a selected-prefix set `S`, the eligible weight of vertex `i` is

```text
0                       when i is already in S
fitness(i) * degree(i)  otherwise
```

The next target probability is that weight divided by the sum of all remaining weights. The row is normalized after each selected target, so an ordered target tuple is a product of conditional rows, not independent draws from the initial row.

Low-level normalization accepts nonnegative finite weights, assigns zero probability to zero-weight entries, and returns an explicit error for an all-zero or negative-weight request. Valid connected BB states with positive fitness guarantee positive remaining mass while a legal target remains.

`Targets n m` is an ordered injective embedding. `orderedMass` gives exact rational mass to one ordered target tuple. `setMass` sums all orderings whose selected set is the requested unordered event. Complete ordered mass and complete unordered-event mass both normalize to one when `m <= n`.

## 3. Atomic graph growth

`applyBirth` validates the complete typed target object before constructing the successor. One successful birth:

- adds exactly one new vertex with stable ID `n`;
- adds exactly `m` undirected edges from the newborn to the selected old vertices;
- preserves every old adjacency;
- preserves every old fitness value exactly;
- stores only the supplied positive newborn fitness at the new vertex;
- keeps the graph connected;
- gives the newborn degree `m`;
- increases each selected old degree by one and leaves every unselected old degree unchanged.

Actual node and edge counts therefore evolve as `N_t = N_0 + t` and `E_t = E_0 + m*t`. Existing bounded walks between old vertices lift through growth without increasing their hop budget.

## 4. Raw validation and deterministic replay

The raw boundary validates seed data, fixed `m`, fitness positivity, target count, bounds, and distinctness before each transition. Failure returns the first zero-based birth index and cause and never returns a partially updated graph or partial probability.

`replay` is deterministic over a supplied raw birth trace. It returns the authoritative final state plus the exact product of the real conditional `orderedMass` values encountered on the evolving graph. Every accepted finite trace has strictly positive probability.

`continuationMass` sums the exact law over every legal future target choice for a fixed positive fitness schedule, using the updated successor state at every level. `continuationMass_one` proves this complete finite conditional mass is exactly one.

## 5. BB invariance laws

V23.5 proves BB-native invariances rather than maintaining a second attachment-model identity.

- Replacing one common positive constant fitness value by unit fitness leaves the conditional target law unchanged.
- Multiplying every current fitness value by one positive constant leaves each conditional row and complete ordered target mass unchanged.
- Whole-replay scaling requires scaling both the seed fitness values and every future newborn fitness value; under that complete scaling, graph topology and trace probability are unchanged while stored fitness is scaled.
- Scaling only the seed while leaving future newborn fitness unchanged is deliberately not an invariance. The two-birth regression fixture changes from `24/805` to `24/1505`.
- For eligible vertices, probability odds follow the exact ratio of `fitness * degree` weights.
- Raising one eligible vertex fitness cannot reduce its probability; strict increase requires positive competing mass.

These are finite exact laws. They do not assert an asymptotic exponent, condensation threshold, or empirical fit.

## 6. Acceptance fixtures

For a triangle with degrees `(2,2,2)` and fitness `(1,2,4)`, the first weights are `(2,4,8)`. With `m=2`:

```text
P[(2,1)] = 8/21
P[(1,2)] = 8/35
P[{1,2}] = 64/105
```

The two orderings can produce the same final graph while carrying different ordered-trace probabilities. Under one common constant fitness value, the six ordered target pairs are uniform; this is treated only as an internal BB normalization property.

Raw replay fixtures cover empty replay, one and two births, full `m=n` selection, common scaling, partial-scaling failure, duplicate targets, out-of-range targets, wrong target counts, nonpositive newborn fitness, invalid seed dimensions, invalid seed fitness, and invalid `m`.

## 7. Positive-probability seven-hop scope counterexample

A two-node edge with unit positive fitness and `m=1`, followed by births targeting the current endpoint `[1]`, `[2]`, `[3]`, `[4]`, `[5]`, `[6]`, yields the actual eight-vertex path

```text
0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
```

with seven edges, degree list `[1,2,2,2,2,2,2,1]`, and exact trace probability

```text
(1/2)*(1/4)*(1/6)*(1/8)*(1/10)*(1/12) = 1/46080 > 0.
```

The verified proof establishes adjacency exactly between consecutive labels, constructs a seven-edge endpoint walk, proves every walk can change the label by at most one per edge, excludes every route of at most six hops, and concludes endpoint shortest-hop count `7` and mesh diameter `7`.

Therefore the finite BB model does not by itself imply a universal six-hop property. Typical, expected, high-probability, or asymptotic distance statements require an additional probability-analysis layer.

## 8. Trust and resource boundary

The implementation is checked with Lean 4.32.0 and pinned mathlib revision `81a5d257c8e410db227a6665ed08f64fea08e997`.

Generic proofs may use ordinary logical dependencies such as `propext`, `Classical.choice`, and `Quot.sound`, and their actual axiom reports are audited. The model does not use `sorry`, `admit`, new user axioms, `native_decide` as a proof oracle, unsafe bypasses, silent normalization fallback, or unlimited resource settings.

Concrete finite arithmetic may use equation-based `decide_cbv`. Replay and scope acceptance remain under bounded wall-clock gates. The model does not weaken `MeshWalk`, `ReachWithin`, `SmallWorldMetrics`, Story, Testimony, or unrelated narrative semantics.

## 9. Current-tree architectural role

V23.5 is the authoritative BB kernel/replay layer consumed by V23.6. The maintained public story is:

```text
actual graph + positive fitness
        ↓
exact BB conditional target law
        ↓
atomic applyBirth
        ↓
deterministic replay + exact trace mass
        ↓
continuationMass = 1
```

V23.6 may expose the complete finite trace distribution, event probabilities, and exact expectations, but it must reuse `orderedMass`, `applyBirth`, and `continuationMass` rather than introducing another stochastic kernel.
