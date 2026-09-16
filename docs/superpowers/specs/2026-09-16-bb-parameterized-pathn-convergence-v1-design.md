# BB parameterized PathN convergence v1 design

Issue: #92  
Base: `proof/narrative-dynamics-v0@8115c3862700114fb91e495f32cecf9e765d7455`

## Context

The current merged BB belief-dynamics baseline has two relevant layers:

1. `NarrativeDynamics.FitnessABMPathN` proves exact convergence for every finite path with fixed rational response profile `receptivity = threshold = 1/2`, under the all-broadcast hypothesis.
2. `NarrativeDynamics.FitnessABMPathNParameters` exposes exact rational `ResponseParameters { receptivity, threshold }`, proves its executable step uses the existing `NetworkPropagation.propagate` path, proves valid parameter kernels are averaging kernels, and proves the parameterized all-broadcast region is invariant under every finite iterate.

The remaining gap is convergence for nontrivial parameterized receptivity. This work must close that gap without changing runtime propagation semantics and without duplicating the generic contraction machinery already proved in `NarrativeDynamics.FiniteConsensus`.

## Goal

For every finite path of size `n >= 2`, valid exact-rational response parameters, strict interior receptivity

```text
0 < α < 1
```

and an initial belief vector inside the parameterized all-broadcast region, prove that every coordinate of the real executable trajectory converges to the existing degree-weighted PathN mean.

The principal theorem should have the semantic shape

```lean
theorem trajectory_tendsto
    (params : ResponseParameters)
    (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 <= n)
    (x : Beliefs n)
    (hx : allBroadcast params n x)
    (i : Fin n) :
  Tendsto
    (fun k : Nat => (((beliefStep params n)^[k] x) i : Real))
    atTop
    (nhds (FitnessABMPathN.mean n x : Real))
```

Exact final binder order and namespace may follow local Lean ergonomics, but the mathematical claim above is fixed.

## Interpretation boundary

Inside the all-broadcast regime, this theorem establishes three separate roles:

- `α` controls transition weights and therefore contraction speed;
- path degree determines the stationary influence weights and consensus value;
- `τ` determines entry into the all-broadcast regime, but once that regime is assumed and proved invariant, it does not alter the degree-weighted consensus value.

The theorem does **not** claim that all initial states enter the all-broadcast regime.

## Non-goals

This change does not claim or implement:

- convergence for `α = 0` or `α = 1`;
- convergence for arbitrary connected graphs;
- convergence outside the all-broadcast region;
- convergence of the exposure-dependent model from #84;
- a floating-point theorem or universal Python↔Lean equivalence;
- an optimal contraction-rate constant;
- a spectral/eigenvalue characterization;
- a replacement for `NetworkPropagation.propagate`.

The existing fixed `α = 1/2` PathN theorem remains valid, unchanged, and independently gated.

## Architecture

### 1. Separate convergence module

Add:

- `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`
- `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`

The new core imports `FitnessABMPathNParameters` and reuses:

- `FitnessABMPathNParameters.pathKernel`;
- `FitnessABMPathNParameters.propagate_eq_kernel`;
- `FitnessABMPathNParameters.allBroadcast_iterate`;
- `FitnessABMPathN.stationaryWeight`;
- `FitnessABMPathN.mean`;
- `FiniteConsensus.block_contraction_tendsto`.

No definition in `NetworkPropagation`, `FiniteConsensus`, `FitnessABMPathN`, or `FitnessABMPathNParameters` is to be reimplemented merely to obtain the theorem.

### 2. Parameter hypotheses

The convergence theorem requires:

```text
params.Valid
0 < params.receptivity
params.receptivity < 1
n >= 2
allBroadcast params n x
```

`params.Valid` retains the existing domain constraints for both receptivity and threshold. The strict interior assumptions are additional convergence hypotheses, not changes to `ResponseParameters.Valid`.

Do not strengthen `Valid` globally to exclude `0` or `1`; those values remain legitimate executable parameters and are needed for the boundary counterexamples.

### 3. Shared stationary distribution

Reuse the fixed-path degree-normalized weights:

```text
π(i) = degree(i) / weightSum(n)
```

Do not define a second parameter-specific stationary-weight function unless Lean elaboration makes an alias useful.

For the parameterized kernel

```text
Kα(i,i) = 1 - α
Kα(i,j) = α / degree(i)    when j is a path neighbor of i
Kα(i,j) = 0                otherwise
```

prove detailed balance for adjacent vertices:

```text
π(i) * Kα(i,j)
  = degree(i)/weightSum * α/degree(i)
  = α/weightSum
  = degree(j)/weightSum * α/degree(j)
  = π(j) * Kα(j,i)
```

and zero equality for nonadjacent vertices. Derive

```lean
StationaryWeights (pathKernel params n) (FitnessABMPathN.stationaryWeight n)
```

under `params.Valid` and `n >= 2`.

The stationary distribution itself must not depend on `α` or `τ`.

### 4. Mean preservation

Use the shared stationary distribution to prove that the existing

```lean
FitnessABMPathN.mean n x
```

is preserved by one parameterized kernel step and by all finite kernel iterates.

Where useful, expose a theorem for the real executable `beliefStep` under the all-broadcast hypothesis. This theorem is supporting structure; the acceptance target is coordinate convergence.

### 5. Conservative per-step positive mass

Define the proof constant

```text
β(params) = α * (1 - α) / 2
```

where `α = params.receptivity`.

For `0 < α < 1`, prove:

```text
0 < β
β < 1
```

and the two local lower bounds needed by the existing path common-column construction:

```text
β <= Kα(i,i)
```

for every vertex, and

```text
pathAdj n j i -> β <= Kα(i,j)
```

for every path edge.

The edge proof may use `degree(i) in {1,2}` and therefore

```text
α / degree(i) >= α / 2 >= α(1-α)/2.
```

The self proof uses

```text
α(1-α)/2 <= 1-α
```

from `0 < α < 1`.

This bound is intentionally conservative. Do not introduce `min (1-α) (α/2)` in v1; the simpler polynomial lower bound is chosen to keep proof branching small and stable.

### 6. Block and common-column mass

Reuse the existing path block length:

```text
block(n) = n - 1
```

or define a convergence-module alias that is definitionally equal to `FitnessABMPathN.block n`.

Define

```text
δ(params,n) = β(params) ^ block(n).
```

Under `0 < α < 1` and `n >= 2`, prove:

```text
0 < δ
δ < 1
```

Then generalize the existing origin-mass construction to the parameterized kernel:

1. start from path vertex `0`;
2. move right along the unique path, using the edge lower bound `β` once per hop;
3. after reaching a target vertex, pad the remaining block time using the self lower bound `β`;
4. conclude every row of `Kα^(n-1)` has at least `δ` mass in the origin column.

The required theorem is semantically:

```lean
CommonColumnMass ((pathKernel params n) ^ block n) (delta params n)
```

This proof should mirror the already verified fixed PathN structure rather than introducing an unrelated consensus argument.

### 7. Executable trajectory bridge

The parameter module already proves that inside the all-broadcast region one executable step equals application of the parameterized proof kernel, and that all-broadcast is invariant under every finite executable iterate.

The convergence module should therefore prove a finite-iterate bridge of the form

```text
(beliefStep params n)^[k] x
  = kernelTrajectory (pathKernel params n) x k
```

under `params.Valid`, `n >= 2`, and the initial `allBroadcast` hypothesis.

This bridge must use the actual executable `beliefStep`; no second executable transition function may be introduced.

### 8. Final convergence theorem

Apply `FiniteConsensus.block_contraction_tendsto` with:

- kernel `pathKernel params n`;
- stationary weights `FitnessABMPathN.stationaryWeight n`;
- block `n - 1`;
- common-column mass `δ = β^(n-1)`.

Then rewrite the kernel trajectory back to the executable parameterized trajectory.

The limit must be exactly

```text
FitnessABMPathN.mean n x
```

not a newly defined equivalent mean.

This yields the intended separation:

```text
α changes convergence dynamics
π and the final consensus value do not change with α
```

inside the theorem's all-broadcast assumptions.

## Required boundary counterexamples

The strict interior hypothesis `0 < α < 1` must be justified by executable or kernel-checked fixtures, not described only in prose.

### Boundary A — `α = 0`

Use a two-node path and a threshold that keeps both nodes broadcasting, for example threshold `0` with beliefs in `[0,1]`.

At `α = 0` the update is identity:

```text
[1, 0] -> [1, 0] -> ...
```

Prove at least a concrete one-step identity fixture and a theorem or finite-iterate fixture sufficient to establish that this non-consensus state does not approach the degree-weighted consensus value `1/2`.

The preferred minimal acceptance evidence is an exact theorem showing every finite iterate equals the initial vector for this fixture, because then the non-convergence claim is immediate and does not rely on numerical tolerance.

### Boundary B — `α = 1`

Use Path2 with threshold `0` and initial beliefs

```text
[1, 0].
```

Each vertex has one neighbor and zero self mass, so the states alternate:

```text
[1,0] -> [0,1] -> [1,0] -> ...
```

Prove exact even/odd behavior, or an equivalent exact two-step cycle theorem, sufficient to show convergence to `1/2` cannot hold.

Do not weaken the counterexample by choosing an initial consensus state.

## TDD sequence

### RED/GREEN 1 — stationary weights

Add consumers requiring parameterized detailed balance, stationary weights, and mean preservation. The initial RED must fail because these convergence declarations do not yet exist, while the existing parameter module still builds.

Minimal GREEN proves only these declarations.

### RED/GREEN 2 — positive lower bounds and common column

Add consumers for:

- `β > 0`;
- self-mass lower bound;
- adjacent-edge lower bound;
- `δ > 0` and `δ < 1`;
- parameterized block common-column mass.

The RED must fail on missing declarations, not on a broken test harness. GREEN should mirror the fixed PathN origin/reach/padding proof.

### RED/GREEN 3 — executable convergence

Add a consumer for the finite-iterate executable/kernel bridge and the final `trajectory_tendsto` theorem. Then implement the minimal bridge and compose `block_contraction_tendsto`.

### RED/GREEN 4 — endpoint counterexamples

Add exact `α=0` identity and `α=1` Path2 cycle consumers. These tests document that strict interior receptivity is a semantic requirement, not merely a proof convenience.

### Final gate

Extend the existing bounded PathN gate additively to build and test the new convergence module and require its theorem reports. Preserve every existing FiniteConsensus, fixed PathN, parameterized-response, exposure-dependent, Path4 and Path5 check.

## Trust and resource boundary

- Exact `Rat` arithmetic for theorem-bearing dynamics.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, or `native_decide` proof oracle.
- Keep existing bounded command timeouts; do not disable heartbeats or linters globally.
- The new theorem reports must pass `tools/audit_fitness_trust.py` under the existing allowlist.
- Reuse `FiniteConsensus.block_contraction_tendsto`; do not duplicate its analytic proof.

## Expected files

Implementation is expected to touch only:

- add `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`;
- add `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`;
- minimally extend `tools/check_fitness_abm_pathn.sh`;
- add this design and its implementation plan.

A root import is not required unless repository conventions or the actual build show that the module otherwise cannot participate in the intended build surface. Workflow files should not change if the existing PathN gate is already invoked by `proof.yml`.

## Acceptance

The implementation is merge-ready only when all of the following are true:

- the final theorem covers every finite `Path n` with `n >= 2` and exact-rational `0 < α < 1`;
- the executable trajectory, not merely a proof-only matrix iteration, is the theorem subject;
- the consensus limit is exactly the existing degree-weighted `FitnessABMPathN.mean`;
- the stationary distribution is proved independent of `α`;
- all-broadcast invariance is reused rather than assumed at every time step;
- a positive common-column mass is proved with an explicit parameter-dependent `β` and `δ`;
- exact `α=0` and `α=1` boundary counterexamples are retained;
- the existing fixed `α=1/2` convergence theorem remains unchanged and green;
- no arbitrary connected-graph or exposure-dependent convergence claim is added;
- permanent PathN proof/trust gates and exact-head CI are green before integration.
