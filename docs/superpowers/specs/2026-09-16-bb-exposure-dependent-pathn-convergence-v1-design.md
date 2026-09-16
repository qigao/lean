# BB exposure-dependent PathN convergence v1 design

Issue: #94  
Base: `proof/narrative-dynamics-v0@fb1c6aa160975b87aa31164abdb64dd961971d7f`

## Context

The merged baseline now has two separate exact-rational finite-path models that must remain distinct:

1. `NarrativeDynamics.FitnessABMPathNParameters` is the exposure-independent model. PR #93 proves that for every finite `Path n`, `n >= 2`, valid constant response parameters, strict interior constant receptivity `0 < alpha < 1`, and an initial all-broadcast belief vector, the real executable belief trajectory converges to the existing degree-weighted mean. Its stationary degree weights and final consensus value do not depend on the constant `alpha`.
2. `NarrativeDynamics.FitnessABMPathNExposure` is the separately named exposure-dependent model merged by PR #91. It stores an exposure count per agent and declares `receptivityAt : Nat -> Rat`. Its executable ordering is fixed: determine current broadcasters, let `received` be the current incoming count, set `exposure' = exposure + received`, and use `receptivityAt exposure'` for that same belief update.

The second point is essential. This phase must reason about the already merged post-incoming lookup semantics. It must not replace them with a pre-update lookup and must not mutate `NetworkPropagation` or the constant-alpha model.

The present gap is convergence and non-convergence for the exposure-dependent model. In particular, the pointwise condition

```text
forall e, 0 < alpha(e) < 1
```

is not expected to be sufficient, and row-dependent effective receptivities generally destroy the fixed degree-weighted invariant from PR #93.

## Goal

Establish a first exact theorem family for exposure-dependent finite-path consensus with four deliverables:

1. an executable all-broadcast bridge from `FitnessABMPathNExposure.step` to an explicit time-varying path-kernel sequence;
2. an exact Path2 disagreement/product characterization;
3. exact rational positive and negative schedules showing the true boundary is stronger than pointwise strict interiority;
4. a PathN sufficient theorem: a uniform interior bound on the effective receptivities used by the trajectory implies existence of a common consensus limit.

The PathN acceptance theorem has the semantic shape

```text
finite Path n, n >= 2
ExposureParameters.Valid
initial allBroadcast
exists eps > 0,
  every effective alpha used by the trajectory lies in [eps, 1-eps]
-------------------------------------------------------------------
exists c : Real,
  every executable belief coordinate tends to c
```

No closed-form value for `c` is claimed in v1.

## Semantic boundary

### Existing same-step exposure ordering is normative

For state `s` and receiver `i`, the merged executable step uses

```text
received_i   = current incoming broadcaster count
exposure'_i  = s[i].exposure + received_i
alpha_i      = p.receptivityAt exposure'_i
belief'_i    = (1-alpha_i) * belief_i + alpha_i * broadcasterMean_i
```

when `received_i != 0`.

Inside an all-broadcast finite path, `received_i = degree(i)`. Therefore, if `e0(i)` is the initial exposure, after `k` completed steps

```text
exposure_k(i) = e0(i) + k * degree(i),
```

and the transition from time `k` to `k+1` uses

```text
alpha_i(k) = p.receptivityAt (e0(i) + (k+1) * degree(i)).
```

All index formulas, Path2 products, fixtures, and theorem statements must follow this convention.

### Consensus existence and consensus value are separate questions

For constant `alpha`, PR #93 has a common stationary distribution proportional to path degree. For exposure-dependent response, the effective alpha can differ by row at the same time because exposure increments depend on degree and exposure history.

For a row-dependent path kernel,

```text
K_t(i,i) = 1 - alpha_i(t)
K_t(i,j) = alpha_i(t) / degree(i)   when j is adjacent to i.
```

The old degree weight `degree(i)/weightSum` is generally not stationary unless the row receptivities align appropriately. Therefore v1 must not state or imply that PathN converges to the old degree-weighted mean.

A concrete Path3 theorem-level witness must show this failure. One preferred fixture is threshold `0`, zero initial exposures, beliefs `[1,0,0]`, and a valid schedule with

```text
alpha(1) = 1/4
alpha(2) = 1/2
```

(with any valid values elsewhere). The first all-broadcast step uses `alpha(1)` at the endpoints and `alpha(2)` at the middle vertex, changing the old degree-weighted mean from `1/4` to `5/16`.

This counterexample is part of the acceptance surface because it prevents accidental reuse of PR #93's stationary-mean conclusion.

## Non-goals

This phase does not claim or implement:

- convergence for arbitrary connected graphs;
- a necessary-and-sufficient convergence characterization for every `PathN` schedule;
- a closed-form PathN consensus value for nonconstant schedules;
- restoration of the old degree-weighted invariant under arbitrary exposure schedules;
- convergence from arbitrary states outside the all-broadcast region;
- stochastic exposure dynamics;
- floating-point convergence or Python/Lean floating-point equivalence;
- psychological or behavioral interpretation of the exposure counter;
- any change to the merged `NetworkPropagation`, fixed PathN, parameterized constant-alpha, or exposure-step semantics.

## Architecture

### 1. Keep the executable exposure model unchanged

`NarrativeDynamics/Core/FitnessABMPathNExposure.lean` remains the semantic source of truth. The new work should consume its `ExposureParameters`, `AgentState`, `State`, `beliefs`, and `step` definitions.

Do not introduce a second exposure-dependent executable transition merely to simplify proofs.

A small supporting theorem may be added to the existing module only if it states a direct fact about the existing step and avoids duplication. The preferred design is additive modules with no production mutation to the merged exposure semantics.

### 2. Add a generic finite time-varying consensus helper

The existing `FiniteConsensus.block_contraction_tendsto` is intentionally a fixed-kernel theorem: it reasons about powers `K^b`, preserves one stationary weighted mean, and concludes convergence to that mean. It cannot be applied directly to a non-homogeneous kernel sequence.

Add a separate module, tentatively:

```text
NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean
```

importing `FiniteConsensus` and reusing its `Kernel`, `AveragingKernel`, `applyKernel`, `coordMin`, `coordMax`, `coordRange`, and common-column contraction lemma where convenient.

Define a kernel schedule and executable mathematical trajectory, for example:

```lean
abbrev KernelSchedule (ι) := Nat -> Kernel ι

def varyingTrajectory (K : KernelSchedule ι) (x : ι -> Rat) : Nat -> ι -> Rat
| 0     => x
| k + 1 => applyKernel (K k) (varyingTrajectory K x k)
```

Define a finite ordered window product with orientation chosen so that

```text
applyKernel (windowKernel K start len) x
```

is exactly the result of applying `K start`, then `K (start+1)`, through `K (start+len-1)`.

The preferred recursion is semantically

```text
windowKernel K start 0       = 1
windowKernel K start (t + 1) = K(start+t) * windowKernel K start t
```

because `applyKernel (A * B) x = applyKernel A (applyKernel B x)`.

Prove only the generic machinery needed by this phase:

- every step of an averaging-kernel schedule keeps each coordinate inside the previous coordinate interval;
- `coordMin` is nondecreasing along the trajectory;
- `coordMax` is nonincreasing;
- if every aligned block of some positive length `b` has a common-column mass at least fixed `delta` with `0 < delta < 1`, then coordinate range contracts geometrically at block boundaries and tends to zero;
- nested interval bounds plus vanishing range imply existence of a common real limit `c` for every coordinate.

The main generic conclusion should be existential, not stationary-mean based:

```lean
theorem block_contraction_consensus_exists ... :
  exists c : Real, forall i,
    Tendsto (fun k => (varyingTrajectory K x k i : Real)) atTop (nhds c)
```

This helper is graph-agnostic analytic infrastructure. It does **not** itself claim that arbitrary connected graphs satisfy the required common-column hypothesis.

Do not modify the existing fixed-kernel theorem merely to force the time-varying case into it.

### 3. Add an exposure-dependent PathN convergence module

Add:

```text
NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean
NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
```

The core module imports the merged exposure model and the new time-varying consensus helper.

Define the inclusive all-broadcast predicate for exposure states:

```text
allBroadcast p s := forall i, p.threshold <= (s i).belief and (s i).belief <= 1
```

and prove that valid schedules preserve it for finite paths. This uses the existing threshold semantics and convex-combination update; it does not assert that arbitrary initial states ever enter the region.

### 4. Exact exposure trajectory under all-broadcast

For `n >= 2`, prove that all-broadcast makes the incoming count exactly the path degree:

```text
incoming p s i = FitnessABMPathN.degree n i.
```

Then prove the finite-iterate exposure law for the actual executable step:

```text
(((step p n)^[k] s) i).exposure
  = (s i).exposure + k * FitnessABMPathN.degree n i.
```

The corresponding effective lookup used by step `k -> k+1` is therefore

```text
(s i).exposure + (k+1) * degree(i).
```

This theorem is the semantic bridge that prevents off-by-one ambiguity in every later result.

### 5. Time-varying path kernel bridge

For a pre-step exposure vector `e : Fin n -> Nat`, define a proof kernel whose row receptivity already reflects the merged post-incoming lookup:

```text
alpha_i(e) = p.receptivityAt (e(i) + degree(i))

K_e(i,i) = 1 - alpha_i(e)
K_e(i,j) = alpha_i(e) / degree(i)   when j is a path neighbor of i
K_e(i,j) = 0                        otherwise.
```

Under `p.Valid` and `n >= 2`, prove `K_e` is an averaging kernel.

Under all-broadcast, prove one real executable exposure step has belief projection exactly

```text
beliefs (step p n s) = applyKernel (K_exposure(s)) (beliefs s).
```

Using the exact exposure law, define the schedule from the initial state:

```text
K_s0(k) = K_(fun i => e0(i) + k * degree(i))
```

so that `K_s0(k)` uses `receptivityAt (e0(i) + (k+1)*degree(i))` internally. Prove the finite-iterate bridge

```text
beliefs ((step p n)^[k] s0)
  = varyingTrajectory K_s0 (beliefs s0) k.
```

This theorem must refer to the actual merged executable `step`.

### 6. Path2 exact disagreement theorem

For Path2, both vertices have degree `1`. If the initial exposures are equal to `e0`, they remain equal, so both rows use the same effective receptivity at each round:

```text
a_k = p.receptivityAt (e0 + k + 1).
```

Let `d_k` be the first belief minus the second belief after `k` executable steps. Prove the exact one-step recurrence

```text
d_(k+1) = (1 - 2*a_k) * d_k
```

and the exact finite product

```text
d_k = d_0 * product_{r < k} (1 - 2 * p.receptivityAt (e0 + r + 1)).
```

Because the two rows use the same receptivity, their arithmetic mean is preserved at every step. For non-consensus initial beliefs, derive a characterization in terms of the finite-product sequence:

```text
both executable coordinates tend to the initial arithmetic mean
iff
abs(product_{r < k} (1 - 2 * alpha(e0+r+1))) tends to 0.
```

The implementation may state the criterion through `Tendsto` of the finite product sequence rather than introducing an infinite-product object. This keeps the theorem exact while avoiding unnecessary infinite-product API surface.

Initial consensus states are trivial and need not be forced through the nonzero-disagreement iff theorem.

### 7. Exact Path2 schedules proving the boundary

Retain theorem-level, exact-rational evidence for all three schedule behaviors below. Use threshold `0`, equal initial exposures `0`, and a non-consensus Path2 belief split such as `[1,0]`.

#### A. Strict-interior schedule that approaches zero too quickly and fails consensus

Use

```text
alpha(e) = 1 / (2 * (e+1)^2).
```

The first update queries `e = 1`. For `k` completed steps the disagreement multiplier is

```text
product_{r < k} (1 - 1/(r+2)^2)
  = (k+2) / (2*(k+1)),
```

which tends to `1/2`, not zero.

Therefore every effective alpha is strictly inside `(0,1)`, yet the executable Path2 trajectory does not reach consensus. This is the primary counterexample to pointwise strict interiority.

#### B. Strict-interior schedule approaching one with persistent oscillation

Use

```text
alpha(e) = 1 - 1 / (2 * (e+1)^2).
```

The disagreement factor is the negative of schedule A's positive factor. Its magnitude tends to `1/2` while sign alternates. Prove an exact even/odd or subsequence result sufficient to show the executable beliefs do not converge.

This generalizes the merged constant `alpha = 1` Path2 two-cycle boundary: remaining strictly inside `(0,1)` at every queried exposure is still insufficient.

#### C. Schedule approaching zero that still reaches consensus

Use

```text
alpha(e) = 1 / (2 * (e+1)).
```

The queried factors telescope:

```text
product_{r < k} (1 - 1/(r+2)) = 1/(k+1) -> 0.
```

Hence `alpha(e) -> 0` does not by itself imply failure. The executable Path2 beliefs converge to their initial arithmetic mean.

Together these fixtures establish that endpoint tendency alone is not the criterion; accumulated contraction is what matters.

### 8. Uniform-interior PathN sufficient condition

Define or state a sufficient interior hypothesis with an explicit rational `eps`:

```text
0 < eps
forall effective lookup e used by the trajectory,
  eps <= p.receptivityAt e
  p.receptivityAt e <= 1 - eps.
```

A globally uniform condition `forall e` may be provided as a simple public corollary. The lower-level theorem should permit the exact reachable lookup points

```text
e0(i) + (k+1) * degree(i)
```

so the mathematical assumption matches the executable trajectory and does not unnecessarily constrain unused schedule entries.

From the uniform bound, every time-varying path kernel has

```text
self mass >= eps
adjacent edge mass >= eps / 2.
```

Use the conservative common local mass

```text
beta = eps / 2.
```

With the existing path block length

```text
b = n - 1,
```

every aligned `b`-step window can carry at least

```text
delta = beta^b
```

from path origin `0` to every row: move along the path one hop per step, then pad remaining steps with self mass.

Prove

```text
0 < delta < 1
```

and a common origin-column lower bound for every aligned time window. Then invoke the new generic time-varying block-contraction theorem.

The final executable theorem must conclude only

```text
exists c : Real, forall i,
  Tendsto (fun k => ((((step p n)^[k] s0) i).belief : Rat : Real))
    atTop (nhds c)
```

under the stated assumptions.

Do not identify `c` with `FitnessABMPathN.mean` unless an additional invariant has actually been proved for the schedule.

### 9. Optional value bounds, not a value formula

It is useful, but not required for the core acceptance theorem, to prove the existential consensus value lies inside the initial coordinate interval:

```text
coordMin (beliefs s0) <= c <= coordMax (beliefs s0).
```

This follows naturally from nested averaging intervals. Do not spend scope on a closed form for `c` in v1.

## TDD sequence

### RED/GREEN 1 — time-varying consensus infrastructure

Add consumers for:

- ordered window-product/trajectory equivalence;
- coordinate-range monotonicity for an averaging-kernel schedule;
- geometric block-range contraction from a uniform common-column mass;
- existential common-limit theorem.

The initial RED must fail on missing declarations. GREEN introduces only the generic non-homogeneous analytic helper; no BB/exposure semantics are changed.

### RED/GREEN 2 — executable exposure/all-broadcast bridge

Add consumers for:

- all-broadcast preservation;
- incoming count equals path degree;
- exact exposure iterate `e0 + k*degree`;
- row-dependent proof kernel is averaging;
- one-step executable belief/kernel equality;
- finite-iterate executable/time-varying trajectory equality.

The effective-alpha indexing must be tested explicitly at the first step so that a pre-step/post-incoming off-by-one implementation cannot pass.

### RED/GREEN 3 — Path2 exact product and mean

Add consumers for:

- equal exposures remain equal;
- one-step disagreement multiplier;
- finite disagreement product;
- arithmetic-mean preservation;
- non-consensus product-to-zero convergence characterization.

### RED/GREEN 4 — exact positive/negative schedules

Add exact theorem consumers for:

- slow-to-zero strict-interior non-consensus schedule;
- near-one strict-interior oscillatory non-convergence schedule;
- harmonic slow-to-zero consensus schedule;
- Path3 degree-weighted-mean non-invariance witness.

These must be exact proofs/closed forms, not floating-point simulations or tolerance checks.

### RED/GREEN 5 — PathN uniform-interior theorem

Add consumers for:

- self/edge lower bounds from `eps`;
- positive block common-column mass for every aligned time window;
- time-varying trajectory consensus existence;
- final executable exposure-step consensus theorem;
- global-uniform-interior convenience corollary if included.

### Final gate

Extend the existing bounded PathN/exposure gate additively to build and test the new modules and require their theorem reports. Preserve all existing fixed PathN, parameterized constant-alpha, exposure-learning, and PR #93 convergence checks.

No workflow edit is expected if the current PathN gate is already invoked by the proof workflow.

## Trust and resource boundary

- The theorem-bearing model remains exact `Rat`; convergence statements cast to `Real` only for analytic limits.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, or `native_decide` proof oracle.
- Keep existing bounded command timeouts and trust audit policy.
- Do not disable heartbeats or linters globally to obtain a proof.
- Counterexamples must be executable-model or exact proof-kernel consequences, not external numerical evidence.
- Reuse the already proved coordinate/common-column lemmas from `FiniteConsensus` where their hypotheses match; do not duplicate fixed-kernel stationary-mean machinery.

## Expected files

Expected implementation scope:

- add `NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean`;
- add `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`;
- add `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`;
- optionally add a focused generic test module if the existing test organization makes that clearer;
- minimally extend `tools/check_fitness_abm_pathn.sh`;
- add this design and its implementation plan.

Avoid changes to:

- `NarrativeDynamics/Core/NetworkPropagation.lean`;
- existing fixed `FitnessABMPathN` semantics;
- existing `FitnessABMPathNParameters` semantics;
- existing `FitnessABMPathNExposure.step` ordering.

A root import or workflow change is not part of the design unless the actual build demonstrates it is necessary.

## Acceptance

The implementation is merge-ready only when all of the following are true:

- the theorem subject is the actual merged `FitnessABMPathNExposure.step` trajectory;
- post-incoming same-step lookup indexing is preserved exactly;
- all-broadcast exposure counts are proved to follow `e0 + k*degree`;
- the executable belief trajectory is bridged to a genuinely time-varying kernel sequence;
- the generic contraction result does not assume a common stationary distribution;
- Path2 has an exact finite disagreement-product theorem and mean-preservation result under equal initial exposures;
- exact strict-interior non-consensus and oscillatory schedules are retained;
- an exact schedule with `alpha(e) -> 0` but consensus is retained;
- a Path3 exact witness proves the old degree-weighted mean is not generally invariant;
- every finite `Path n`, `n >= 2`, satisfies executable consensus existence under the stated uniform-interior effective-alpha condition;
- the final PathN theorem does not claim a closed-form consensus value;
- no arbitrary-connected-graph convergence claim is added;
- all existing baseline and PR #93 theorem gates remain green;
- the new theorem reports pass the existing trust audit before integration.
