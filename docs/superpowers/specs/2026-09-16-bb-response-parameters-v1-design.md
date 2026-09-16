# BB response-parameter dynamics v1 design

Issue: #82  
Parent roadmap: #80  
Base: `proof/narrative-dynamics-v0@b8766d3cc9767bbd8636f9adc0d8114f8efabb1e`

## Context

#81 is merged and supplies the checked replay → idle-tail boundary without introducing a second propagation implementation. #83 is also merged and supplies the canonical finite-path propagation/convergence surface in `NarrativeDynamics.Core.FitnessABMPathN` for the fixed profile `(receptivity = 1/2, threshold = 1/2)`.

An older #82 comment refers to `1e12766` based on the historical #81 `fd2517c`; those commits do not resolve in the current repository and are not implementation evidence. The useful semantic decisions in that comment are retained here: rational parameters, inclusive threshold semantics, explicit parameter validity, and exact specialization back to the fixed `1/2` model.

This design intentionally parameterizes the current finite-path architecture rather than recreating a Path4-only parallel model.

## Goal

Introduce a separate rational response-parameter layer for finite paths with parameters:

- `α : Rat` — receptivity;
- `τ : Rat` — inclusive broadcast threshold;
- validity domain `0 ≤ α ≤ 1` and `0 ≤ τ ≤ 1`.

The first implementation slice must establish the actual parameterized operator, exact specialization to the existing fixed model, exposure independence, and all-broadcast invariance. It must not change the baseline model or claim arbitrary-parameter convergence before the additional positivity inequalities are proved.

## Non-goals for v1

This first slice does not prove:

- convergence for every valid `α`;
- a contraction rate for arbitrary `α`;
- an activation-time formula as a function of `τ`;
- convergence from states outside the all-broadcast region;
- a floating-point equivalence theorem;
- an arbitrary connected-graph theorem;
- a psychological interpretation of `α`, `τ`, or exposure counts.

Those remain follow-on theorem slices under #82 after the v1 operator is merged.

## Architecture

### 1. Keep baseline production semantics unchanged

Do not edit the definitions of:

- `NetworkPropagation.AgentProfile`;
- `NetworkPropagation.broadcasting`;
- `NetworkPropagation.nextAgent`;
- `NetworkPropagation.propagate`;
- `FitnessABMPathN.population`;
- `FitnessABMPathN.beliefStep`;
- `FitnessABMPathN.pathKernel`.

The existing fixed model remains the regression authority.

### 2. Add a separate finite-path parameter module

Add:

`NarrativeDynamics/Core/FitnessABMPathNParameters.lean`

Namespace:

`NarrativeDynamics.FitnessABMPathNParameters`

Reuse the canonical path topology from `FitnessABMPathN`:

```lean
abbrev Beliefs := FitnessABMPathN.Beliefs

def ResponseParameters where
  receptivity : Rat
  threshold : Rat

def ResponseParameters.Valid (p : ResponseParameters) : Prop :=
  0 ≤ p.receptivity ∧ p.receptivity ≤ 1 ∧
  0 ≤ p.threshold ∧ p.threshold ≤ 1
```

The exact structure syntax may derive `DecidableEq` / `Repr`, but no proof fields are stored in the structure; validity remains a separate predicate, matching `AgentProfile`.

### 3. Parameterized population and real executable step

The parameterized population uses one constant profile for all path vertices and preserves supplied exposure counters:

```lean
def population (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨params.receptivity, params.threshold⟩,
    fun i => ⟨x i, e i⟩⟩
```

Observation reuses the existing projection shape:

```lean
def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief
```

The executable operator must remain the real propagation function:

```lean
def beliefStep (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n
    (NetworkPropagation.propagate
      (FitnessABMPathN.pathAdj n)
      (population params n x (fun _ => 0)))
```

No hand-written matrix is allowed as the executable semantics.

### 4. Inclusive all-broadcast region

Define:

```lean
def allBroadcast (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, params.threshold ≤ x i ∧ x i ≤ 1
```

Equality at `τ` must broadcast because baseline `broadcasting` is `decide (threshold ≤ belief)`.

The v1 theorem must prove that for valid parameters and `n ≥ 2`, one real `beliefStep` preserves this region. The proof should derive the actual step from a parameterized averaging kernel only inside the all-broadcast region.

### 5. Parameterized proof-only path kernel

Define the proof-only kernel:

```lean
def pathKernel (params : ResponseParameters) (n : Nat) : Kernel (Fin n) :=
  fun i j =>
    (if i = j then 1 - params.receptivity else 0) +
    (if j ∈ FitnessABMPathN.neighbors n i then
       params.receptivity / (FitnessABMPathN.degree n i : Rat)
     else 0)
```

For `params.Valid` and `n ≥ 2`, prove:

- kernel entries are nonnegative;
- each row sums to one;
- therefore it is an `AveragingKernel`;
- in the all-broadcast region, the real `propagate` projection equals `applyKernel` of this kernel.

This proof is the parameterized analogue of the existing fixed `FitnessABMPathN.propagate_eq_kernel`; it must not change that theorem.

### 6. Exposure independence

Exposure counters remain irrelevant to current broadcast decisions and belief arithmetic because `NetworkPropagation.broadcasting` reads threshold/belief only, while `nextAgent` uses exposures only to write the next cumulative count.

Prove for arbitrary rational parameters, without requiring parameter validity:

```lean
propagate_independent_exposures
  (params : ResponseParameters)
  (n : Nat) (x : Beliefs n)
  (e : Fin n → Nat) :
  project n (propagate ... (population params n x e)) =
    beliefStep params n x
```

This theorem is intentionally about belief projection, not equality of complete populations.

### 7. Exact specialization to the merged fixed model

Define the canonical fixed parameters:

```lean
def half : ResponseParameters :=
  ⟨1/2, 1/2⟩
```

Prove exact compatibility with the current baseline for every finite path:

```lean
population_half
beliefStep_half
allBroadcast_half
pathKernel_half
```

The fixed theorem surface remains untouched; the parameterized module points back to it.

### 8. Stationary mean boundary for later slices

The degree-weighted stationary distribution from `FitnessABMPathN` is expected to remain stationary for the parameterized kernel because every undirected path edge receives symmetric stationary flow proportional to `α`, while self-mass is local.

The v1 implementation may include the stationary-weight / mean-preservation theorem if it stays small and bounded, but it is not required for the first merge. No convergence theorem may be inferred merely from stationarity.

A later #82 slice may prove convergence under the strict positivity domain:

```text
0 < α < 1
```

with a common-column lower bound based on a positive per-step floor such as `min (1 - α) (α / 2)`. The exact statement must be proved before any rate claim is recorded.

### 9. Activation timing boundary

`τ` affects whether/when a concrete history enters the all-broadcast region. V1 only preserves inclusive threshold semantics and proves invariance after entry. It does not derive a closed-form activation time.

Later activation theorems must use actual checked replay / idle-tail states from #81 rather than a synthetic history runner.

## Required invariants

V1 must establish:

1. `ResponseParameters.Valid` exactly means both parameters lie in `[0,1]`.
2. Parameterized execution calls the existing `NetworkPropagation.propagate`.
3. Equality `belief = τ` broadcasts.
4. Belief projection is independent of exposure counters.
5. Under valid parameters and `n ≥ 2`, all-broadcast is invariant under one parameterized step and therefore under finite iteration.
6. `half` specializes exactly to existing `FitnessABMPathN` population, step, region, and kernel.
7. Existing Path4, Path5, PathN and #81 replay/idle-tail gates remain unchanged and green.

## File boundary

Expected production addition:

- `NarrativeDynamics/Core/FitnessABMPathNParameters.lean`

Expected test addition:

- `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`

Expected CI change:

- extend the existing bounded PathN gate in `.github/workflows/proof.yml` or its invoked gate script with source/build/test/trust coverage for the new module;
- do not create a second permanent workflow when the current bounded gate can own the checks.

Do not modify `NetworkPropagation.lean`, `FiniteConsensus.lean`, or the existing fixed `FitnessABMPathN` definitions unless a proven compatibility theorem requires a tiny exported lemma. Any such change must preserve behavior definitionally.

## TDD sequence

### RED 1 — parameter structure and executable step

Create the test consumer first, importing the not-yet-existing parameter module and requiring:

- `ResponseParameters`;
- `ResponseParameters.Valid`;
- `half`;
- parameterized `population`, `beliefStep`, and `allBroadcast`.

Expected RED: missing module/declarations.

### GREEN 1 — minimal parameter layer

Implement only the structure, validity predicate, population, projection, region, real executable step, and `half`.

### RED/GREEN 2 — exact fixed-model specialization

Add consumers for `population_half`, `beliefStep_half`, `allBroadcast_half`, and later `pathKernel_half`. Prove exact compatibility without changing baseline definitions.

### RED/GREEN 3 — exposure independence

Add a consumer using two different exposure functions over the same beliefs and parameters. Prove equality of projected beliefs through the real `propagate` path.

### RED/GREEN 4 — parameterized kernel and real-step bridge

Add the proof-only kernel, row/nonnegative laws under `Valid`, and the all-broadcast bridge from actual `propagate` to `applyKernel`.

### RED/GREEN 5 — all-broadcast invariance

Prove one-step and finite-iterate invariance under:

```text
params.Valid
2 ≤ n
allBroadcast params n x
```

The theorem must include equality-at-threshold cases.

### Final gate

Run the complete existing PathN gate plus the new bounded parameter tests and trust reports. Re-run existing Path4/Path5 generated replay coverage through the normal proof workflow before merge readiness.

## Trust and resource boundary

- no `sorry`, `admit`, new user `axiom`, `unsafe`, or `native_decide`;
- no unbounded heartbeat/recursion settings;
- direct focused Lean commands use the repository's existing 240-second timeout convention;
- mandatory theorem reports remain within the repository's current allowlist (`propext`, `Classical.choice`, `Quot.sound`) unless fresh evidence requires a narrower set;
- preserve exact rational arithmetic throughout.

## Acceptance for v1 merge

The first #82 PR is merge-ready only when:

- the parameterized finite-path operator exists in a separate module;
- exact `α = τ = 1/2` specialization to the merged baseline is proved;
- inclusive threshold semantics has an explicit equality-at-threshold consumer;
- exposure independence of projected beliefs is proved;
- all-broadcast invariance is proved under explicit valid-parameter assumptions;
- existing fixed Path4/Path5/PathN theorem and generated replay gates remain green;
- bounded source/build/test/trust checks are integrated;
- exact-head CI and independent review have no blocker;
- the PR states clearly that arbitrary-parameter convergence rate, limit convergence, and activation-time formulas remain unproved unless separately added.

#82 itself remains open after this v1 merge if those later theorem questions are still outstanding.
