# BB finite-path convergence design

Date: 2026-09-15  
Issue: #83  
Base: `proof/narrative-dynamics-v0@5ee133072b86b084087bc036b753c77f832741d6`

## 1. Purpose

Generalize the exact Path4/Path5 BB belief dynamics to a theorem for every finite path `Fin n`, with `2 ≤ n`, while preserving the existing modeling boundary:

- the executable dynamics remain `NetworkPropagation.propagate`;
- any matrix/kernel is proof-only evidence derived from actual propagation;
- the proof remains exact over `Rat` until the final convergence statement is cast to `Real`;
- the result is only for finite paths under the fixed profile `(receptivity = 1/2, threshold = 1/2)` and an all-broadcast initial region;
- the change must not imply convergence for arbitrary connected graphs or arbitrary profile parameters.

This design intentionally does **not** extend Path4's explicit eigenmode formula. The generic proof uses finite-step contraction instead of spectral decomposition.

## 2. Existing evidence and architectural boundary

Path4 already proves:

1. actual propagation is independent of exposure counters at the belief projection;
2. in the all-broadcast region, actual propagation agrees with a rational averaging operator;
3. the all-broadcast region is invariant;
4. the degree-weighted mean is invariant;
5. an explicit rational closed form exists for four nodes;
6. the closed form converges coordinatewise to the invariant mean after casting to `Real`.

Path5 already proves items 1--4 using actual propagation and has generated replay conformance against the runtime implementation. Its proof deliberately does not introduce a Path4-style closed form.

The Path4 closed form must not become the architecture for `Path n`. Its nontrivial eigenvalues happen to be rational for four nodes. For larger paths, a direct eigenbasis generally introduces algebraic/trigonometric values, which would enlarge the proof surface without strengthening the model claim we need.

The reusable boundary for `Path n` is therefore:

```text
actual NetworkPropagation.propagate
              |
              | theorem, only in all-broadcast region
              v
      finite rational averaging kernel
              |
              +--> invariant degree-weighted mean
              |
              +--> finite-block common-mass contraction
                          |
                          v
                 coordinate convergence
```

## 3. Considered approaches

### A. Explicit spectral decomposition

Construct the lazy-path transition matrix, diagonalize it, write a closed form for all modes, then prove every nonconstant mode tends to zero.

Rejected as the primary route. It is mathematically standard but creates unnecessary Lean obligations around algebraic/trigonometric eigenvalues, eigenvectors, diagonalization, and casts. It also makes the generic theorem substantially more complicated than the behavioral claim.

Path4's existing closed form remains valid evidence and a regression oracle, but is not generalized.

### B. Finite-block common-mass contraction — selected

Derive a row-stochastic rational kernel from actual propagation. Show that in `n - 1` steps every output coordinate contains at least

```text
δ_n = (1/4)^(n - 1)
```

of one common original coordinate, chosen as node `0`. The remaining mass is nonnegative and totals at most `1 - δ_n`, giving

```text
range (step^[n-1] x) ≤ (1 - δ_n) * range x.
```

Repeated block contraction forces the range to zero. The invariant degree-weighted mean always lies between the current minimum and maximum, so each coordinate converges to that mean.

Selected because it:

- stays rational;
- needs no eigenvalues;
- gives an explicit proof resource / convergence-rate witness;
- preserves the distinction between path-specific reachability evidence and a generic averaging lemma;
- scales structurally with `n` instead of enumerating vertices.

### C. Direct endpoint-to-endpoint min/max induction

Prove that extrema propagate inward and that after a bounded number of rounds the global range strictly shrinks, without introducing a matrix/kernel abstraction.

Not selected. It would couple convergence directly to the Path implementation and duplicate convex-combination arithmetic throughout the proof. The common-mass kernel criterion isolates the reusable mathematics with a much smaller interface.

## 4. Module boundaries

### 4.1 `NarrativeDynamics/Core/FiniteConsensus.lean`

A small proof-only library for exact finite averaging. It must not mention BB, paths, births, thresholds, or `NetworkPropagation`.

The canonical object is a finite rational kernel over a finite nonempty index type. Implementation may use `Matrix ι ι Rat`, but the public theorem surface should expose only the properties needed by consensus:

- nonnegative entries;
- every row sums to `1`;
- application to a vector;
- a common-column lower bound for a block kernel;
- preservation of a supplied mean functional;
- finite range / min / max.

A local thin predicate such as `AveragingKernel K` is preferred over coupling public project APIs to a large external stochastic-matrix API. Internally, existing Mathlib matrix lemmas may be used when convenient.

Required conceptual lemmas:

```text
range_nonneg
apply_between_min_max
range_apply_le
range_apply_le_of_commonColumn
range_block_iterate_le
mean_between_min_max
coordinate_dist_mean_le_range
block_contraction_tendsto
```

The last theorem should be stated narrowly: if a finite averaging step preserves a mean and some positive-length block has a common-column mass `δ > 0`, then each coordinate tends to the preserved mean.

It must **not** contain a theorem saying every connected finite graph satisfies the common-column hypothesis.

### 4.2 `NarrativeDynamics/Core/FitnessABMPathN.lean`

Owns the path-specific model and discharges the hypotheses of `FiniteConsensus`.

Proposed public objects:

```lean
abbrev Beliefs (n : Nat) := Fin n → Rat

def pathAdj (n : Nat) (i j : Fin n) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

def neighbors (n : Nat) (i : Fin n) : Finset (Fin n) := ...
def degree (n : Nat) (i : Fin n) : Nat := (neighbors n i).card

def population (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n := ...
def project (n : Nat) (p : Population n) : Beliefs n := ...
def allBroadcast (n : Nat) (x : Beliefs n) : Prop := ...

def beliefStep (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n (propagate (pathAdj n) (population n x (fun _ => 0)))

def trajectory (n : Nat) (x : Beliefs n) (k : Nat) : Beliefs n :=
  beliefStep n^[k] x
```

The exact binder order can be adjusted for Lean ergonomics, but `n` and the lower bound `2 ≤ n` must remain explicit in theorem statements where needed.

### 4.3 Existing Path4 / Path5 modules

Do not delete, rewrite, or make them aliases in the first generic-convergence change.

They are independent verified evidence and provide compatibility checks. The generic implementation must prove instance-level agreement for `n = 4` and `n = 5` before any future cleanup is considered.

No migration or public-surface reduction belongs in the first Path-n PR.

## 5. Proof-only path kernel

The generic proof kernel should be defined from the actual path neighbor set, not from an endpoint/interior case table.

For `2 ≤ n`, every path vertex has positive degree. Define conceptually:

```text
K_n(i,j) =
  1/2                         if i = j
  1 / (2 * degree(i))         if j is a path neighbor of i
  0                           otherwise
```

Because a finite path has degree `1` at the endpoints and `2` at interior vertices, this specializes to:

```text
endpoint:  1/2 self + 1/2 neighbor
interior:  1/2 self + 1/4 left + 1/4 right
```

The implementation of `beliefStep` remains `propagate`; `K_n` exists only to prove properties of that step.

Required path-kernel facts:

1. `degree_pos` for `2 ≤ n`;
2. path degrees are at most `2`;
3. kernel entries are nonnegative;
4. every row sums to `1`;
5. every self coefficient is at least `1/2`;
6. every adjacent influence coefficient is at least `1/4`;
7. applying the kernel agrees with the actual belief projection of propagation in the all-broadcast region.

The bridge theorem is the generic replacement for Path4/Path5 coordinate enumeration:

```lean
propagate_eq_kernel
  (hn : 2 ≤ n)
  (x : Beliefs n)
  (e : Fin n → Nat)
  (hx : allBroadcast n x) :
  project n (propagate (pathAdj n) (population n x e)) =
    applyKernel (pathKernel n) x
```

The proof must reason from `incoming`, `neighbors`, and cardinality. It must not use `fin_cases` over all vertices of generic `Fin n`.

Exposure independence remains a separate theorem and should hold outside the all-broadcast region, as it already does for Path4/Path5.

## 6. All-broadcast invariance

The generic invariant is:

```lean
allBroadcast_step
  (hn : 2 ≤ n)
  (x : Beliefs n)
  (hx : allBroadcast n x) :
  allBroadcast n (beliefStep n x)
```

Do not prove it by endpoint/interior arithmetic or coordinate enumeration.

Once `beliefStep = applyKernel K x` is established and every row of `K` is a nonnegative unit-sum combination, both bounds follow from convexity:

```text
1/2 ≤ every input coordinate ≤ 1
          =>
1/2 ≤ every output coordinate ≤ 1.
```

Then obtain `allBroadcast_iterate` by ordinary induction over `trajectory`.

This theorem is critical: it justifies using the linear/kernel bridge at every later time while the executable definition continues to be thresholded propagation.

## 7. Stationary degree-weighted mean

For a path with `n ≥ 2`, define the invariant mean from actual path degrees rather than a hard-coded vector:

```text
weight(i) = degree(n,i)
weightSum = Σ_i weight(i) = 2 * (n - 1)

mean_n(x) = (Σ_i weight(i) * x(i)) / weightSum
```

Required facts:

1. `weightSum_pos` from `2 ≤ n`;
2. `path_degree_sum : Σ degree = 2 * (n - 1)`;
3. detailed-balance-style identity for the proof kernel:

```text
degree(i) * K(i,j) = degree(j) * K(j,i)
```

for all coordinates, including the diagonal and zero/nonadjacent cases;
4. `mean_step` derived by finite sum rearrangement, not by expanding every coordinate;
5. `mean_iterate`;
6. `mean_between_min_max` because the normalized degree weights are nonnegative and sum to one.

For `n = 4` and `n = 5`, the generic mean must reduce extensionally to the existing weights:

```text
Path4: 1,2,2,1 / 6
Path5: 1,2,2,2,1 / 8
```

## 8. Common-column mass on a finite path

Let

```text
block(n) = n - 1
δ(n) = (1/4)^(n - 1).
```

For every target vertex `i : Fin n`, construct one explicit influence walk of exactly `n - 1` kernel steps from original coordinate `0` to output coordinate `i`:

1. move along the unique path `0,1,...,i` for `i.val` steps;
2. pad the remaining steps with self transitions at `i`.

Every move along the path contributes at least `1/4`; every self transition contributes at least `1/2`, hence at least `1/4`. Therefore the product contribution of this single walk is at least `δ(n)`.

Because all matrix/kernel path-sum terms are nonnegative, that one walk gives the lower bound

```text
(K_n ^ block(n))(i, 0) ≥ δ(n)
```

for every `i`.

This is the only graph-structure fact needed by the contraction theorem.

The proof must stay path-specific. We do not introduce a generic connectivity-to-common-mass theorem in this change.

## 9. Block contraction

For an averaging kernel `B` whose every row has at least `δ` mass in one common column `c`, write each output coordinate conceptually as

```text
(Bx)(i) = δ * x(c) + (1 - δ) * residual_i(x),
```

where `residual_i` is itself a convex combination when `δ < 1`.

The common term cancels when comparing two output coordinates, giving

```text
range(Bx) ≤ (1 - δ) * range(x).
```

For the path block kernel:

```text
range (trajectory n x (k + block n))
  ≤ (1 - δ n) * range (trajectory n x k).
```

Required scalar facts for `2 ≤ n`:

```text
0 < δ(n)
δ(n) ≤ 1
0 ≤ 1 - δ(n)
1 - δ(n) < 1.
```

The bound is deliberately conservative. No attempt is made to obtain the sharp spectral mixing rate.

## 10. Convergence without a spectral closed form

The final proof should avoid a theorem about `(1 - δ)^(k / block)` unless Mathlib already makes that route trivial.

Preferred Lean strategy:

1. prove range is nonincreasing after every single averaging step;
2. prove by induction that after `q` complete blocks,

```text
range (trajectory n x (q * block n))
  ≤ (1 - δ n)^q * range x;
```

3. use geometric decay of `(1 - δ)^q` over `Real`;
4. for an arbitrary later clock `k ≥ q * block`, use single-step range monotonicity to bound its range by the completed-block range;
5. because the invariant mean lies between the current min and max,

```text
|trajectory n x k i - mean_n x| ≤ range (trajectory n x k);
```

6. conclude coordinatewise `Tendsto`.

This avoids explicit `Nat.div` in the main analytic argument and keeps arithmetic obligations local.

Target theorem shape:

```lean
theorem trajectory_tendsto
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n)
    (hx : allBroadcast n x)
    (i : Fin n) :
  Tendsto
    (fun k : Nat => (trajectory n x k i : Real))
    atTop
    (nhds (mean n x : Real))
```

Exact names and implicit arguments may change, but the statement must retain:

- explicit finite path size;
- explicit `2 ≤ n`;
- explicit all-broadcast hypothesis;
- coordinatewise convergence;
- the degree-weighted mean as the limit.

## 11. Compatibility evidence

The first generic implementation must add theorem-level compatibility checks rather than replacing prior modules.

Required tests:

### Path4 compatibility

For all beliefs/exposures in the relevant region:

```text
PathN.pathAdj 4 = Path4.pathAdj
PathN belief projection / step agrees with Path4
PathN mean agrees with Path4.mean
PathN generic convergence has the same limit as Path4.trajectory_tendsto
```

The old Path4 explicit closed form remains independently compiled and tested.

### Path5 compatibility

Likewise:

```text
PathN.pathAdj 5 = Path5.path5Adj
PathN belief step agrees with Path5
PathN mean agrees with Path5.mean
```

The existing generated Path5 runtime conformance gate remains untouched and must continue to pass.

### Generic smoke instance

Include at least one `n = 6` theorem/example that instantiates the generic bridge/invariant/mean/convergence surface. This is not a new hand-written Path6 model and must not introduce Path6-specific production code.

Its role is only to catch accidental dependence on `Fin 4`/`Fin 5` enumeration.

## 12. Testing and CI boundary

Add a dedicated bounded gate rather than continually growing `check_fitness_abm_path4.sh`.

Proposed script:

```text
tools/check_fitness_abm_pathn.sh
```

It should:

1. source-audit `FiniteConsensus.lean`, `FitnessABMPathN.lean`, and the generic tests;
2. explicitly build both new core modules because the umbrella import may not include experimental/specialized proof modules;
3. run the generic Path-n theorem tests under the existing wall-clock envelope;
4. run Path4 and Path5 compatibility tests;
5. keep the existing Path4/Path5 exact gate and generated replay gate unchanged as regression evidence;
6. fail on proof errors or resource timeouts; never make timeouts advisory.

Initial per-command wall-clock limit remains `240s` unless evidence demonstrates that a narrower proof needs a justified change.

The implementation must not use:

- `set_option maxHeartbeats 0`;
- unbounded recursion/resource settings;
- native proof oracles;
- `sorry` / `admit`;
- generated axioms;
- a numeric finite enumeration pretending to prove generic `n`.

Any new public theorem added to the formal acceptance surface should be included in the existing trust audit pattern when practical.

## 13. Resource strategy

The generic proof should reduce rather than increase coordinate explosion.

Rules:

- no `fin_cases i` where `i : Fin n` and `n` is generic;
- no expansion of `(K^m)` into all matrix entries;
- prove one explicit walk contribution for the common-column lower bound and use nonnegativity for the rest;
- keep rational algebra symbolic;
- isolate finite-sum rewrites in small lemmas;
- only cast to `Real` in the final analytic convergence layer;
- if matrix powers cause elaboration blow-up, introduce a recursive kernel-application/block lemma rather than increasing global resource limits.

A resource failure is treated as a design signal, not as permission to disable the bound.

## 14. Non-goals

This design does not prove or introduce:

- convergence for arbitrary connected graphs;
- convergence for all stochastic matrices;
- a sharp convergence rate or spectral gap;
- arbitrary receptivity `α`;
- arbitrary broadcast threshold `τ`;
- activation from arbitrary non-broadcast initial states;
- preservation of graph distances under BB growth;
- a generic `TailModel` replacement;
- removal of Path4 or Path5 modules;
- a Path6 production module;
- changes to runtime BB growth semantics.

A later issue may generalize the common-mass criterion to graph classes, but this PR must stop at the finite-path theorem.

## 15. Error and boundary handling

`n = 0` and `n = 1` are excluded from the convergence theorem by `2 ≤ n`; they are not represented through runtime errors because this is a theorem precondition, not an executable parser API.

The lower-bound hypothesis is necessary because:

- the degree denominator must be positive;
- `n - 1` must be a positive block length;
- the path has the intended endpoint/interior structure.

No hidden default graph size is allowed.

The all-broadcast hypothesis remains explicit. The theorem must not silently linearize a state containing a non-broadcast coordinate.

## 16. Implementation phases after spec approval

The later implementation plan should preserve the following dependency order:

1. **FiniteConsensus RED and foundation** — range/min/max and generic averaging contraction lemmas.
2. **Generic path structure** — adjacency, neighbors, degree, population/project, actual propagation definitions.
3. **Actual-propagation bridge** — exposure independence and `propagate_eq_kernel`.
4. **Invariant region + stationary mean** — convexity, degree sum, detailed balance, mean preservation.
5. **Path block common mass** — explicit padded path walk and `δ(n)` lower bound.
6. **Generic convergence** — block contraction, geometric decay, final `Tendsto`.
7. **Path4/Path5 compatibility + n=6 smoke**.
8. **Bounded exact CI + trust audit + final review**.

Each phase must have a genuine RED boundary before production proof work where a testable consumer can be written.

## 17. Acceptance for the Path-n stage

The Path-n stage is review-ready only when all of the following are true on one exact head:

- `beliefStep` is still defined through actual `NetworkPropagation.propagate`;
- the generic kernel bridge is proved for every `n ≥ 2` in the all-broadcast region;
- exposure independence is proved generically;
- all-broadcast invariance is proved without finite-size enumeration;
- degree-weighted mean preservation is proved generically;
- `(n - 1)`-step common-column mass has an explicit positive rational lower bound;
- block range contraction is proved;
- coordinatewise convergence to the degree-weighted mean is proved;
- Path4 and Path5 compatibility checks pass;
- existing Path4 closed-form/convergence tests still pass;
- existing Path5 generated runtime conformance still passes;
- a generic `n = 6` smoke instantiation passes without Path6-specific production code;
- source/trust audits pass;
- bounded exact CI passes without raising resource limits merely to hide proof structure problems.

Passing these criteria completes the finite-path theorem requested by #83, while leaving arbitrary graph and parameterized-profile generalizations explicitly out of scope.
