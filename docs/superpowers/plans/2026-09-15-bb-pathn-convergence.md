# BB finite-path convergence implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Execute one task at a time, preserve each RED/GREEN boundary, and review before advancing.

**Goal:** Generalize the verified Path4/Path5 BB belief dynamics to every finite path `Fin n` with `2 ≤ n`, proving coordinatewise convergence of the actual `NetworkPropagation.propagate` trajectory to the degree-weighted stationary mean without using a spectral closed form.

**Architecture:** Keep executable BB dynamics unchanged. Add a proof-only `FiniteConsensus` layer over finite rational kernels and a path-specific `FitnessABMPathN` layer that derives its kernel from actual propagation in the all-broadcast region. Prove a path-specific `(n - 1)`-step common-column mass lower bound `δ(n) = (1/4)^(n-1)`, use it for geometric range contraction, then cast only the final limit statement to `Real`.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, exact `Rat` arithmetic, `Matrix`, existing `NetworkPropagation`, existing fitness trust audit, GitHub Actions proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-15-bb-pathn-convergence-design.md`, approved at `c3ba593215f551fb9473ce854c508870755a0077`.

## Global constraints

- `beliefStep` must remain defined through `NetworkPropagation.propagate`; a matrix/kernel is proof-only evidence.
- Keep Path4 and Path5 source and theorem surfaces intact in this first generic PR.
- Do not add a Path6 production module; `n = 6` appears only as a fixed smoke instance.
- No theorem may claim convergence for arbitrary connected graphs or arbitrary stochastic matrices.
- Keep profiles fixed at receptivity `1/2`, inclusive broadcast threshold `1/2`.
- The generic convergence theorem requires explicit `2 ≤ n` and explicit initial `allBroadcast`.
- Stay over `Rat` until the analytic limit layer; cast to `Real` only for `Tendsto`/absolute-value convergence.
- No `sorry`, `admit`, `native_decide`, new `axiom`, `unsafe`, `unlock_limits`, or unlimited resource settings.
- Allowed transitive Lean axioms remain exactly `propext`, `Classical.choice`, `Quot.sound`.
- Every focused build/test in the final gate is bounded by `timeout --kill-after=10s 240s` and reports GNU `time` peak RSS.
- Do not add `FiniteConsensus` or `FitnessABMPathN` to `NarrativeDynamics.lean` in this change. Like Path4/Path5, they are explicitly built by their proof gate so the umbrella library cannot accidentally mask missing specialized compilation.
- A proof-resource failure is a design signal: factor the proof or introduce a recursive helper before considering any resource increase.

## File map

| File | Responsibility | Tasks |
| --- | --- | --- |
| `NarrativeDynamics/Core/FiniteConsensus.lean` | Generic finite rational averaging, stationary weights, range contraction, generic analytic convergence | 1, 6 |
| `NarrativeDynamics/Tests/FiniteConsensus.lean` | RED/GREEN consumers and required axiom reports for generic consensus | 1, 6, 8 |
| `NarrativeDynamics/Core/FitnessABMPathN.lean` | Generic path structure, actual propagation bridge, stationary degree weights, path common mass, final Path-n theorem | 2–6 |
| `NarrativeDynamics/Tests/FitnessABMPathN.lean` | Generic contracts, Path4/Path5 compatibility, n=6 smoke, required axiom reports | 2–7 |
| `tools/check_fitness_abm_pathn.sh` | Dedicated bounded builds/tests/source+log trust audit | 8 |
| `.github/workflows/proof.yml` | Dedicated `BB finite-path convergence` step after the existing Path4/Path5 gate | 8 |
| Existing Path4/Path5/conformance files | Regression evidence only; no production edits planned | 7–8 |

## Preparation before Task 1

- [ ] Create execution branch `feature/bb-pathn-convergence-v1` from the approved plan commit, not from `master`:

```bash
git checkout design/bb-pathn-convergence-v1
git pull --ff-only
git checkout -b feature/bb-pathn-convergence-v1
git rev-parse HEAD
```

Expected starting head: the committed implementation-plan head produced by this document.

- [ ] Record immutable implementation base and make sure no production code differs from merged Path5 except approved docs:

```bash
git merge-base HEAD proof/narrative-dynamics-v0
git diff --stat proof/narrative-dynamics-v0...HEAD
```

Expected before implementation: only the Path-n spec and plan.

- [ ] Read `NetworkPropagation.lean`, `FitnessABMPath4.lean`, `FitnessABMPath4Convergence.lean`, `FitnessABMPath5.lean`, `FitnessABMPath5Vectors.lean`, `tools/check_fitness_abm_path4.sh`, and `tools/audit_fitness_trust.py` before editing.

---

## Task 1: FiniteConsensus RED and exact averaging foundation

**Files:** create `NarrativeDynamics/Tests/FiniteConsensus.lean`; create `NarrativeDynamics/Core/FiniteConsensus.lean`.

**Purpose:** Establish the generic finite rational kernel vocabulary and min/max/range/stationary-weight lemmas without any BB/path dependency. Do not prove asymptotic convergence yet.

### Public interfaces

Use a finite nonempty index type and a rational matrix kernel:

```lean
namespace NarrativeDynamics.FiniteConsensus

open scoped BigOperators

abbrev Kernel (ι : Type*) := Matrix ι ι Rat

def applyKernel [Fintype ι] (K : Kernel ι) (x : ι → Rat) : ι → Rat :=
  fun i => ∑ j, K i j * x j

structure AveragingKernel [Fintype ι] (K : Kernel ι) : Prop where
  nonneg : ∀ i j, 0 ≤ K i j
  row_sum : ∀ i, ∑ j, K i j = 1

structure StationaryWeights [Fintype ι] (K : Kernel ι) (π : ι → Rat) : Prop where
  nonneg : ∀ i, 0 ≤ π i
  sum_one : ∑ i, π i = 1
  stationary : ∀ j, ∑ i, π i * K i j = π j

def weightedMean [Fintype ι] (π x : ι → Rat) : Rat :=
  ∑ i, π i * x i

def kernelTrajectory [Fintype ι] (K : Kernel ι) (x : ι → Rat) (k : Nat) : ι → Rat :=
  (applyKernel K)^[k] x

def CommonColumnMass [Fintype ι] (K : Kernel ι) (δ : Rat) : Prop :=
  ∃ c, ∀ i, δ ≤ K i c
```

Define finite coordinate extrema and range under `[Fintype ι] [Nonempty ι] [DecidableEq ι]` as `coordMin`, `coordMax`, `coordRange := coordMax - coordMin`. The implementation may use the image of `Finset.univ` and finite min/max; consumers must not depend on the helper representation.

Required Task-1 theorems:

```text
coordMin_le
le_coordMax
coordRange_nonneg
applyKernel_between
coordRange_apply_le
weightedMean_apply
weightedMean_between
coordinate_dist_weightedMean_le_range
kernelTrajectory_zero
kernelTrajectory_succ
applyKernel_mul
kernelPow_apply
```

`weightedMean_apply` must consume `StationaryWeights K π` and prove
`weightedMean π (applyKernel K x) = weightedMean π x` by finite-sum rearrangement.

`kernelPow_apply` must connect matrix powers to iteration:

```lean
applyKernel (K ^ k) x = kernelTrajectory K x k
```

up to orientation required by the chosen `Matrix.mul_apply`; if the initial orientation is reversed, correct the definition once here rather than compensating later in PathN.

### TDD steps

- [ ] **RED 1.1:** create the consumer first, importing the absent module and requiring the public vocabulary:

```lean
import NarrativeDynamics.Core.FiniteConsensus

open NarrativeDynamics.FiniteConsensus

private def pairKernel : Kernel (Fin 2) := fun _ _ => 1/2
private def pairWeights : Fin 2 → Rat := fun _ => 1/2

example : applyKernel pairKernel ![0, 1] = ![1/2, 1/2] := by
  decide_cbv

example : weightedMean pairWeights ![0, 1] = 1/2 := by
  decide_cbv

example (x : Fin 2 → Rat) : 0 ≤ coordRange x := coordRange_nonneg x
```

- [ ] Run the RED consumer:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
```

Expected: failure because `NarrativeDynamics.Core.FiniteConsensus` is absent. A missing toolchain is not acceptable RED evidence.

- [ ] **GREEN 1.2:** implement only the interfaces/basic lemmas above. Prove `applyKernel_between` from entry nonnegativity + row sum; derive one-step range nonincrease. Prove stationary mean preservation with `Finset.sum_comm`/sum distribution, not coordinate enumeration.

- [ ] Run focused build and consumer:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
```

Expected: PASS under default Lean resource settings.

- [ ] Commit Task 1:

```bash
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean
git commit -m "feat(lean): add finite consensus kernel foundation"
```

**Review gate:** verify `FiniteConsensus.lean` imports no BB/path module and contains no graph/connectivity theorem.

---

## Task 2: Generic finite-path structure and proof kernel

**Files:** create `NarrativeDynamics/Core/FitnessABMPathN.lean`; create `NarrativeDynamics/Tests/FitnessABMPathN.lean`.

**Consumes:** `NetworkPropagation`, Task-1 `FiniteConsensus`.

**Produces:** path structure, actual model definitions, proof-only kernel, and structural row-stochastic facts. No propagation/kernel equality yet.

### Required interfaces

```lean
namespace NarrativeDynamics.FitnessABMPathN

open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FiniteConsensus
open scoped BigOperators

abbrev Beliefs (n : Nat) := Fin n → Rat

def pathAdj (n : Nat) (i j : Fin n) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val

instance (n : Nat) : DecidableRel (pathAdj n) := fun i j => inferInstance

def neighbors (n : Nat) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun j => pathAdj n j i

def degree (n : Nat) (i : Fin n) : Nat := (neighbors n i).card

def population (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : Population n :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩

def project (n : Nat) (p : Population n) : Beliefs n :=
  fun i => (p.agents i).belief

def allBroadcast (n : Nat) (x : Beliefs n) : Prop :=
  ∀ i, 1/2 ≤ x i ∧ x i ≤ 1

def beliefStep (n : Nat) (x : Beliefs n) : Beliefs n :=
  project n (propagate (pathAdj n) (population n x (fun _ => 0)))

def trajectory (n : Nat) (x : Beliefs n) (k : Nat) : Beliefs n :=
  (beliefStep n)^[k] x

def pathKernel (n : Nat) : Kernel (Fin n) := fun i j =>
  if i = j then 1/2
  else if pathAdj n j i then 1 / (2 * (degree n i : Rat))
  else 0
```

Required structural facts for `hn : 2 ≤ n`:

```text
neighbors_nonempty
degree_pos
degree_le_two
pathKernel_nonneg
pathKernel_row_sum
pathKernel_averaging : AveragingKernel (pathKernel n)
pathKernel_self : pathKernel n i i = 1/2
pathKernel_self_lower : 1/4 ≤ pathKernel n i i
pathKernel_adj_lower : pathAdj n j i → 1/4 ≤ pathKernel n i j
```

Do not use generic `fin_cases i` in any theorem quantified over arbitrary `n`.

### TDD steps

- [ ] **RED 2.1:** write fixed consumers that force the generic declarations while allowing finite evaluation only in the test:

```lean
import NarrativeDynamics.Core.FitnessABMPathN

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathN

example : pathAdj 6 (0 : Fin 6) 1 := by decide
example : degree 6 0 = 1 := by decide_cbv
example : degree 6 3 = 2 := by decide_cbv
example : pathKernel 6 0 0 = 1/2 := by decide_cbv
example : pathKernel 6 0 1 = 1/2 := by decide_cbv
example : pathKernel 6 3 2 = 1/4 := by decide_cbv
example : pathKernel 6 3 3 = 1/2 := by decide_cbv
```

- [ ] Run RED before creating the core module; expected missing-import failure.

- [ ] **GREEN 2.2:** implement generic path definitions. Prove neighbor/degree facts structurally from `Fin` value inequalities and the path adjacency predicate. The proof may split on whether `i.val = 0`, `i.val + 1 = n`, or interior, but must not enumerate `Fin n`.

- [ ] Prove row sum by rewriting the neighbor contribution as `degree * (1/(2*degree))` plus diagonal `1/2`; use `degree_pos` for the nonzero denominator.

- [ ] Run:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
```

- [ ] Commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): define generic finite path kernel"
```

**Review gate:** confirm `beliefStep` is visibly defined via `propagate`, never via `pathKernel`.

---

## Task 3: Actual-propagation bridge and all-broadcast invariance

**Files:** modify `NarrativeDynamics/Core/FitnessABMPathN.lean`; modify `NarrativeDynamics/Tests/FitnessABMPathN.lean`.

**Consumes:** Task-2 generic path and Task-1 averaging lemmas.

**Produces:** generic exposure independence, actual incoming-set characterization, propagation/kernel bridge, region invariance, and actual/kernel trajectory equality.

### Required theorems

```lean
propagate_independent_exposures
  (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
  project n (propagate (pathAdj n) (population n x e)) = beliefStep n x

incoming_eq_neighbors
  (n : Nat) (x : Beliefs n) (e : Fin n → Nat)
  (hx : allBroadcast n x) (i : Fin n) :
  incoming (pathAdj n) (population n x e) i = neighbors n i

propagate_eq_kernel
  (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (e : Fin n → Nat)
  (hx : allBroadcast n x) :
  project n (propagate (pathAdj n) (population n x e)) =
    applyKernel (pathKernel n) x

allBroadcast_step
  (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
  (hx : allBroadcast n x) :
  allBroadcast n (beliefStep n x)

allBroadcast_iterate
  (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
  (hx : allBroadcast n x) (k : Nat) :
  allBroadcast n (trajectory n x k)

trajectory_eq_kernelTrajectory
  (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
  (hx : allBroadcast n x) (k : Nat) :
  trajectory n x k = kernelTrajectory (pathKernel n) x k
```

### TDD steps

- [ ] **RED 3.1:** add consumers for the missing theorems before implementation:

```lean
example (n : Nat) (x : Beliefs n) (e : Fin n → Nat) :
    project n (propagate (pathAdj n) (population n x e)) = beliefStep n x :=
  propagate_independent_exposures n x e

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) (k : Nat) :
    trajectory n x k = kernelTrajectory (pathKernel n) x k :=
  trajectory_eq_kernelTrajectory n hn x hx k
```

Run the test and record unknown-declaration RED.

- [ ] **GREEN 3.2:** prove exposure independence exactly as Path4/Path5 do: incoming/broadcast decisions depend on profiles and beliefs, not exposure counts. Do not require `allBroadcast` for this theorem.

- [ ] Prove `incoming_eq_neighbors` from the inclusive threshold. Under `hx`, every source broadcasts, so the filter reduces to adjacency.

- [ ] Prove `propagate_eq_kernel` from the actual `nextAgent` formula and the neighbor cardinality. The self `1/2` and neighbor `1/(2*degree)` coefficients must be derived from receptivity `1/2`, not assumed as a replacement definition.

- [ ] Prove `allBroadcast_step` using `pathKernel_averaging` + `applyKernel_between`; do not split endpoint/interior coordinates. Lift to iteration by induction.

- [ ] Prove `trajectory_eq_kernelTrajectory` by induction, applying the actual bridge at each step using `allBroadcast_iterate`.

- [ ] Run focused build/test and commit:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): bridge generic paths to actual propagation"
```

**Review gate:** search the generic proof for `fin_cases`; none may be used to prove a theorem for arbitrary `Fin n`.

---

## Task 4: Stationary degree weights and invariant mean

**Files:** modify both PathN core/test files; Task-1 `FiniteConsensus` only if a genuinely generic stationary-weight helper is missing.

**Produces:** path degree sum, normalized weights, detailed balance, stationarity, mean preservation, and mean-in-range facts.

### Required interfaces

```lean
def weightSum (n : Nat) : Rat :=
  ∑ i : Fin n, (degree n i : Rat)

def stationaryWeight (n : Nat) (i : Fin n) : Rat :=
  (degree n i : Rat) / weightSum n

def mean (n : Nat) (x : Beliefs n) : Rat :=
  weightedMean (stationaryWeight n) x
```

Required theorems for `hn : 2 ≤ n`:

```text
path_degree_sum : weightSum n = 2 * (n - 1)
weightSum_pos
stationaryWeight_nonneg
stationaryWeight_sum_one
pathKernel_detailed_balance
path_stationary_weights : StationaryWeights (pathKernel n) (stationaryWeight n)
mean_kernel_step
mean_step
mean_iterate
mean_between
```

Use exact rational casts in `path_degree_sum`; the right side of the theorem should be typed as `Rat`, e.g. `(2 * (n - 1) : Nat)` cast or an equivalent explicit rational expression chosen once and used consistently.

### TDD steps

- [ ] **RED 4.1:** add generic consumers:

```lean
example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n)
    (hx : allBroadcast n x) :
    mean n (beliefStep n x) = mean n x :=
  mean_step n hn x hx

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    mean n (kernelTrajectory (pathKernel n) x k) = mean n x := by
  exact mean_kernel_iterate n hn x k
```

Run and capture unknown-declaration RED.

- [ ] **GREEN 4.2:** prove the path degree sum structurally. Prefer an endpoint/interior Finset decomposition or the finite undirected-handshake identity if it reduces obligations cleanly; do not introduce a theorem for arbitrary connected graphs.

- [ ] Prove detailed balance by cases `i = j`, adjacent distinct, nonadjacent. For adjacent distinct coordinates use path symmetry and cancellation of the receiver degree in `degree(i) * 1/(2*degree(i))`.

- [ ] Derive stationarity from detailed balance + row sum; normalize with `weightSum_pos`.

- [ ] Derive `mean_kernel_step` using generic `weightedMean_apply`, then obtain actual `mean_step` through `propagate_eq_kernel`; obtain `mean_iterate` from trajectory induction.

- [ ] Run focused checks and commit:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): prove generic path stationary mean"
```

Only include Task-1 files in this commit if they actually changed.

---

## Task 5: Prove the `(n - 1)`-step common-column mass

**Files:** modify PathN core/test.

**Purpose:** Supply the one genuinely path-specific mixing witness required by generic consensus; do not generalize connectivity.

### Required definitions and theorem

```lean
def block (n : Nat) : Nat := n - 1

def delta (n : Nat) : Rat := (1/4 : Rat) ^ block n
```

Required scalar theorems for `hn : 2 ≤ n`:

```text
block_pos
delta_pos
delta_lt_one
one_sub_delta_pos
one_sub_delta_lt_one
```

Main theorem:

```lean
theorem path_block_common_mass
    (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass ((pathKernel n) ^ block n) (delta n)
```

The witness column must be node `0`.

### Proof construction

Do not expand the whole matrix power. Establish one explicit nonnegative contribution per target:

1. Prove `left_reach_mass`: after `i.val` steps, influence from source `0` to target `i` is at least `(1/4)^i.val`. Induct along the unique predecessor chain `0 → 1 → ... → i` and keep only the predecessor term from each nonnegative matrix-multiplication sum.
2. Prove `self_pad_mass`: if `(K^t) i 0 ≥ a`, then after one more self step the same coefficient is at least `(1/4) * a`, using `pathKernel_self_lower`; iterate this for `block n - i.val` padding steps.
3. Combine `i.val ≤ n-1` with exponent arithmetic to get exactly `(1/4)^(n-1)`.

If direct `Matrix.pow_succ` elaboration becomes unstable, introduce a small recursive coefficient helper local to PathN and prove equivalence to matrix power. Do not raise global heartbeat/memory limits.

### TDD steps

- [ ] **RED 5.1:** consumer:

```lean
example (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass ((pathKernel n) ^ block n) (delta n) :=
  path_block_common_mass n hn
```

- [ ] Add fixed sanity checks only after the generic theorem is RED:

```lean
example : block 6 = 5 := by decide
example : delta 6 = 1/1024 := by norm_num [delta, block]
```

- [ ] **GREEN 5.2:** implement the scalar and path-mass lemmas, then run PathN build/test under 240 seconds.

- [ ] Commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): prove finite path common mass"
```

**Review gate:** main theorem must be quantified over arbitrary `n ≥ 2`; no `n = 4/5/6` case analysis is acceptable in its proof.

---

## Task 6: Generic block contraction and coordinate convergence

**Files:** modify `FiniteConsensus.lean`, `FiniteConsensus` tests, PathN core/test.

**Consumes:** Task-1 averaging/stationarity + Task-5 common mass.

**Produces:** reusable common-column contraction, block geometric decay, generic consensus `Tendsto`, then PathN `trajectory_tendsto`.

### Generic FiniteConsensus theorems

Add:

```text
coordRange_apply_le_of_commonColumn
coordRange_block_iterate_le
block_geometric_bound
block_contraction_tendsto
```

Target generic theorem shape:

```lean
theorem block_contraction_tendsto
    [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : Kernel ι) (π : ι → Rat)
    (hK : AveragingKernel K)
    (hπ : StationaryWeights K π)
    (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hcommon : CommonColumnMass (K ^ b) δ)
    (x : ι → Rat) (i : ι) :
  Tendsto
    (fun k : Nat => (kernelTrajectory K x k i : Real))
    atTop
    (nhds (weightedMean π x : Real))
```

The implementation may add explicit helper hypotheses/lemmas showing `K^b` is averaging, but it must derive them from `hK`, not require the Path layer to assert them independently.

### Required proof route

1. From common column `c`, subtract exactly `δ` mass from that column in every row and normalize the residual by `1-δ`; prove the residual is an averaging kernel using `0 < 1-δ`.
2. The shared `δ*x(c)` cancels between output coordinates, giving
   `coordRange (applyKernel (K^b) x) ≤ (1-δ) * coordRange x`.
3. Use `kernelPow_apply` to turn that into a bound after `b` iterations.
4. Induct over complete blocks:
   `coordRange (kernelTrajectory K x (q*b)) ≤ (1-δ)^q * coordRange x`.
5. Cast the inequality to `Real` and use the existing geometric-power limit theorem (`tendsto_pow_atTop_nhds_zero_of_lt_one` or the closest v4.32 theorem) with `0 ≤ 1-δ < 1`.
6. For arbitrary later clocks, use one-step range nonincrease rather than introducing a `Nat.div` closed form.
7. Use stationary mean preservation + `weightedMean_between` to get
   `|x_k i - weightedMean π x| ≤ coordRange x_k`, then squeeze to zero.

### PathN final theorem

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

Derive it by converting `trajectory` to `kernelTrajectory` using Task 3 and applying the generic theorem with:

```text
K = pathKernel n
π = stationaryWeight n
b = block n
delta = delta n
```

### TDD steps

- [ ] **RED 6.1:** first add a finite consensus consumer that references `block_contraction_tendsto` for `pairKernel`; then add a PathN consumer referencing `trajectory_tendsto`.

- [ ] Run both tests and capture missing-theorem RED before implementing convergence.

- [ ] **GREEN 6.2:** implement generic contraction/limit first. Keep all analytic imports confined to `FiniteConsensus.lean` unless a separate convergence file proves measurably clearer; if split, use `FiniteConsensusConvergence.lean` and update this plan's gate names consistently before advancing.

- [ ] Implement the thin PathN final theorem only after generic consensus passes.

- [ ] Add required axiom-report lines to test files:

```lean
#print axioms NarrativeDynamics.FiniteConsensus.coordRange_apply_le_of_commonColumn
#print axioms NarrativeDynamics.FiniteConsensus.block_contraction_tendsto
#print axioms NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures
#print axioms NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel
#print axioms NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate
#print axioms NarrativeDynamics.FitnessABMPathN.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathN.mean_step
#print axioms NarrativeDynamics.FitnessABMPathN.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
```

- [ ] Run focused build/tests and commit:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathN
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteConsensus.lean
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
git add NarrativeDynamics/Core/FiniteConsensus.lean \
        NarrativeDynamics/Core/FitnessABMPathN.lean \
        NarrativeDynamics/Tests/FiniteConsensus.lean \
        NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "feat(lean): prove finite path consensus convergence"
```

**Stop-and-fix rule:** if this task exceeds the 240-second focused envelope, do not continue to compatibility/CI. Identify whether min/max, matrix power, or analytic conversion is responsible and factor that proof first.

---

## Task 7: Path4/Path5 compatibility and generic n=6 smoke

**Files:** modify only `NarrativeDynamics/Tests/FitnessABMPathN.lean` unless a missing public compatibility helper in PathN is demonstrably needed. Do not modify existing Path4/Path5 production modules.

**Imports:** PathN, Path4, Path4Convergence, Path5.

### Required compatibility evidence

Use namespace aliases in the test to avoid ambiguous `Beliefs`, `mean`, `trajectory` names.

Path4:

```text
path4_adj_compat
path4_step_compat
path4_trajectory_compat
path4_mean_compat
```

Path5:

```text
path5_adj_compat
path5_step_compat
path5_trajectory_compat
path5_mean_compat
```

For fixed sizes, `fin_cases` is allowed in compatibility proofs. It is forbidden only as a substitute for the generic theorem.

Add a consumer showing the generic convergence theorem reproduces the old Path4 limit statement after compatibility rewrites:

```lean
example (x : NarrativeDynamics.FitnessABMPath4.Beliefs)
    (hx : NarrativeDynamics.FitnessABMPath4.allBroadcast x)
    (i : Fin 4) :
    Tendsto
      (fun k : Nat => (NarrativeDynamics.FitnessABMPath4.trajectory x k i : Real))
      atTop
      (nhds (NarrativeDynamics.FitnessABMPath4.mean x : Real)) := by
  -- rewrite old trajectory/mean to PathN and apply PathN.trajectory_tendsto
  ...
```

The proof must not call the existing Path4 `trajectory_tendsto`; this consumer demonstrates that the generic theorem independently reaches the same statement.

### n=6 smoke

Define only in the test:

```lean
def smoke6 : NarrativeDynamics.FitnessABMPathN.Beliefs 6 :=
  ![1, 7/8, 3/4, 5/8, 3/4, 1]
```

Prove fixed `allBroadcast 6 smoke6`, then instantiate:

```lean
example (i : Fin 6) :
    Tendsto
      (fun k : Nat => (trajectory 6 smoke6 k i : Real))
      atTop
      (nhds (mean 6 smoke6 : Real)) :=
  trajectory_tendsto 6 (by decide) smoke6 (by
    intro j
    fin_cases j <;> norm_num [smoke6, allBroadcast]) i
```

This may use `fin_cases` because `Fin 6` is a fixed smoke test; no `FitnessABMPath6.lean` is created.

### Regression execution

- [ ] **RED 7.1:** add compatibility consumers before compatibility theorem helpers; unknown declarations or unclosed definitional equalities are the intended RED.

- [ ] **GREEN 7.2:** prove compatibility, then run:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
timeout --kill-after=10s 240s bash tools/check_fitness_abm_path4.sh
```

Expected: generic tests pass; old Path4 explicit closed-form/convergence and existing Path5 generated replay golden remain unchanged and pass.

- [ ] Commit:

```bash
git add NarrativeDynamics/Tests/FitnessABMPathN.lean
git commit -m "test(lean): certify PathN compatibility and smoke"
```

---

## Task 8: Dedicated bounded gate, exact-head CI, trust audit, and review readiness

**Files:** create `tools/check_fitness_abm_pathn.sh`; modify `.github/workflows/proof.yml`. Do not fold this gate into `check_fitness_abm_path4.sh`.

### Gate script contract

Create `tools/check_fitness_abm_pathn.sh` with `set -euo pipefail`, repository-root normalization, GNU `time`, two temporary logs, and cleanup trap.

Source audit exactly the maintained PathN proof surface:

```bash
python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/FiniteConsensus.lean \
  NarrativeDynamics/Tests/FiniteConsensus.lean \
  NarrativeDynamics/Core/FitnessABMPathN.lean \
  NarrativeDynamics/Tests/FitnessABMPathN.lean
```

Explicitly build specialized modules because `NarrativeDynamics.lean` does not import them:

```bash
for module in \
    NarrativeDynamics.Core.FiniteConsensus \
    NarrativeDynamics.Core.FitnessABMPathN; do
  /usr/bin/time -f "$module elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s lake build "$module"
done
```

Run both test files separately and capture logs:

```bash
/usr/bin/time -f 'FiniteConsensus tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FiniteConsensus.lean \
  2>&1 | tee "$consensus_log"

/usr/bin/time -f 'FitnessABMPathN tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean \
  2>&1 | tee "$pathn_log"
```

Audit mandatory reports:

```bash
python3 tools/audit_fitness_trust.py log "$consensus_log" \
  --require NarrativeDynamics.FiniteConsensus.coordRange_apply_le_of_commonColumn \
  --require NarrativeDynamics.FiniteConsensus.block_contraction_tendsto

python3 tools/audit_fitness_trust.py log "$pathn_log" \
  --require NarrativeDynamics.FitnessABMPathN.propagate_independent_exposures \
  --require NarrativeDynamics.FitnessABMPathN.propagate_eq_kernel \
  --require NarrativeDynamics.FitnessABMPathN.allBroadcast_iterate \
  --require NarrativeDynamics.FitnessABMPathN.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathN.mean_step \
  --require NarrativeDynamics.FitnessABMPathN.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathN.trajectory_tendsto
```

If implementation chooses a separate `FiniteConsensusConvergence.lean` during Task 6, add that exact file/module to source audit and timed builds; do not leave stale names in the gate.

### Workflow integration

Insert one step immediately after existing `BB path-four convergence`:

```yaml
      - name: BB finite-path convergence
        timeout-minutes: 15
        run: bash tools/check_fitness_abm_pathn.sh
```

Do not alter event selection, exact-head checkout, Path4 gate, Path5 runtime golden, naming audit, or later regression steps.

### Final verification sequence

- [ ] Run shell/hygiene checks:

```bash
bash -n tools/check_fitness_abm_pathn.sh
git diff --check
git status --short
```

- [ ] Run dedicated gate locally if the toolchain is available:

```bash
timeout --kill-after=10s 900s bash tools/check_fitness_abm_pathn.sh
```

- [ ] Run the existing Path4/Path5 regression gate once on the same head:

```bash
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
```

- [ ] Commit CI integration:

```bash
git add tools/check_fitness_abm_pathn.sh .github/workflows/proof.yml
git commit -m "ci: gate generic finite path convergence"
```

- [ ] Push `feature/bb-pathn-convergence-v1` and open a draft PR against `proof/narrative-dynamics-v0`. Record the exact head SHA before CI.

- [ ] Require one exact-head PR proof run where all of the following are success on that same SHA:
  - Python numerical tests;
  - full Lean library build;
  - existing BB ABM joint gate;
  - existing runtime replay conformance;
  - existing Path4/Path5 gate including Path5 generated replay golden;
  - new `BB finite-path convergence` gate;
  - naming, attachment, replay, scope, distribution, general theorem, story, testimony tails;
  - World Studio exact-head workflow.

- [ ] Capture from the PathN gate logs:
  - exact head SHA;
  - `FiniteConsensus` build time/RSS;
  - `FitnessABMPathN` build time/RSS;
  - both test time/RSS values;
  - source-audit result;
  - mandatory axiom-report audit result.

- [ ] Request code review after the exact-head gate passes. Fix Critical/Important findings before marking ready.

- [ ] Update issue #83 with factual evidence only:
  - concrete Path5 evidence remains landed at merge `5ee1330`;
  - generic theorem now covers every `Fin n` with explicit `2 ≤ n` and all-broadcast hypothesis;
  - executable step is still actual `NetworkPropagation.propagate`;
  - stationary limit is degree-weighted;
  - contraction witness is the explicit conservative `δ(n) = (1/4)^(n-1)`;
  - arbitrary connected graphs and arbitrary `(α, τ)` remain out of scope.

- [ ] Stop at review/merge readiness. Do not close #83 or merge automatically unless the user explicitly asks to merge after reviewing the exact-head evidence.

## Final acceptance checklist

The implementation is review-ready only if all entries below are verified on one exact PR head:

- [ ] `beliefStep` is implemented through actual `NetworkPropagation.propagate`.
- [ ] Generic exposure independence holds without `allBroadcast`.
- [ ] `propagate_eq_kernel` holds for every `n ≥ 2` in the all-broadcast region.
- [ ] Generic all-broadcast invariance uses convexity, not finite enumeration.
- [ ] Degree sum and normalized stationary degree weights are proved.
- [ ] Degree-weighted mean preservation is proved for actual trajectories.
- [ ] `(n-1)`-step path common-column mass has lower bound `(1/4)^(n-1)`.
- [ ] Generic block range contraction is proved.
- [ ] Coordinatewise `Tendsto` to the stationary mean is proved for every `n ≥ 2`.
- [ ] Path4 adjacency/step/trajectory/mean compatibility passes.
- [ ] Generic theorem independently reproduces the Path4 convergence statement.
- [ ] Path5 adjacency/step/trajectory/mean compatibility passes.
- [ ] Existing Path5 runtime/model generated golden comparison remains green.
- [ ] Fixed `n = 6` smoke instantiates the generic theorem with no Path6 production module.
- [ ] No arbitrary connected-graph or arbitrary stochastic-matrix convergence theorem was added.
- [ ] No arbitrary receptivity/threshold generalization was added.
- [ ] Source audit and required theorem axiom reports pass.
- [ ] No global resource limit was disabled or raised merely to hide proof structure.
- [ ] Exact-head proof and World Studio workflows pass.
