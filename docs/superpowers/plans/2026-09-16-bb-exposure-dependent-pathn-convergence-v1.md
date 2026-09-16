# BB Exposure-Dependent PathN Convergence v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove exact convergence/non-convergence conditions for the merged exposure-dependent finite-Path executable model, including an exact Path2 product characterization and a finite-PathN consensus-existence theorem under a uniform interior bound on the effective receptivities actually used by the trajectory.

**Architecture:** Keep `FitnessABMPathNExposure.step` unchanged and bridge its all-broadcast executable trajectory to a genuinely time-varying sequence of exact-rational averaging kernels. Add a small graph-agnostic non-homogeneous consensus helper based on coordinate-range contraction rather than stationary weights; then build exposure-specific semantics, Path2 exact products/counterexamples, and the PathN uniform-interior common-column theorem above it. The final nonconstant PathN theorem is existential in the consensus value and must not reuse PR #93's degree-weighted stationary-mean conclusion.

**Tech Stack:** Lean 4.32.0, Mathlib, exact `Rat` theorem-bearing dynamics with `Real` casts only for limits, existing `NarrativeDynamics.FiniteConsensus`, existing `FitnessABMPathNExposure`, GitHub Actions proof/World Studio workflows, `tools/audit_fitness_trust.py`.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-exposure-dependent-pathn-convergence-v1-design.md`

## Global Constraints

- Base is `proof/narrative-dynamics-v0@fb1c6aa160975b87aa31164abdb64dd961971d7f` after merged PR #93.
- Work on `feature/bb-exposure-dependent-pathn-convergence-v1`; issue tracking is #94.
- `FitnessABMPathNExposure.step` is normative and unchanged: compute current broadcasters, set `e' = e + incoming`, then use `receptivityAt e'` for that same belief update.
- Under all-broadcast, transition `k -> k+1` must query `receptivityAt (e0(i) + (k+1) * degree(i))`; a pre-step lookup is a correctness failure.
- Do not change `NetworkPropagation`, fixed `FitnessABMPathN`, `FitnessABMPathNParameters`, `ResponseParameters.Valid`, or the existing exposure-step ordering.
- Do not apply `FiniteConsensus.block_contraction_tendsto` to a non-homogeneous kernel sequence; it is a fixed-kernel/stationary-mean theorem.
- The generic time-varying helper may reuse public coordinate/common-column lemmas from `FiniteConsensus`, but must not assume a common stationary distribution.
- The final nonconstant PathN result concludes `exists c : Real` with all executable belief coordinates tending to `c`; it does not identify `c` with `FitnessABMPathN.mean`.
- Keep finite paths, `n >= 2`, exact rational all-broadcast dynamics, and deterministic exposure schedules as the theorem scope.
- Do not claim arbitrary-connected-graph convergence, convergence from arbitrary states into all-broadcast, stochastic exposure dynamics, a necessary-and-sufficient PathN condition, or floating-point convergence.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, globally disabled heartbeats, or globally disabled linters.
- Focused Lean commands and permanent gate commands retain the repository's 240-second timeout convention.
- Every GREEN task must leave all previously completed consumers green before moving on.

---

## File Structure

- Create `NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean`: graph-agnostic ordered kernel windows, varying trajectory, averaging preservation, range contraction, and existential common-limit theorem.
- Create `NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean`: focused consumers for ordered composition and non-homogeneous block contraction.
- Create `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`: all-broadcast exposure semantics, time-varying PathN kernel bridge, Path2 exact theorems/examples, and PathN uniform-interior theorem.
- Create `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`: RED/GREEN theorem consumers, exact schedules, Path3 mean-noninvariance witness, and axiom reports.
- Modify `tools/check_fitness_abm_pathn.sh` only after theorem work is green, extending source/build/test/trust coverage additively.
- Keep `NarrativeDynamics/Core/FitnessABMPathNExposure.lean` unchanged unless a direct theorem about the existing `step` is proven impossible to state in the new convergence module. Any such exception requires an explicit review gate before editing that file.
- Do not edit `.github/workflows/proof.yml` unless the permanent gate is demonstrably not invoked for the new non-doc files.

---

### Task 1: Generic finite time-varying consensus infrastructure

**Files:**
- Create: `NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean`
- Create: `NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean`

**Interfaces:**
- Consumes: `FiniteConsensus.Kernel`, `applyKernel`, `AveragingKernel`, `coordMin`, `coordMax`, `coordRange`, `applyKernel_between`, `coordRange_apply_le`, `coordRange_apply_le_of_commonColumn`, `CommonColumnMass`, `applyKernel_mul`.
- Produces: `KernelSchedule`, `varyingTrajectory`, `windowKernel`, `varyingTrajectory_succ`, `apply_windowKernel`, `windowKernel_averaging`, `coordMin_varying_mono`, `coordMax_varying_anti`, `coordRange_varying_le`, `UniformBlockCommonColumn`, `block_geometric_bound`, `coordRange_tendsto_zero`, `block_contraction_consensus_exists`.

- [ ] **Step 1: Write the Task 1 RED consumer file**

Create `NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean` with a two-coordinate alternating schedule whose kernels are already concrete averaging matrices:

```lean
import NarrativeDynamics.Core.FiniteTimeVaryingConsensus

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open Filter Topology

private def K0 : Kernel (Fin 2) := !![3/4, 1/4; 1/4, 3/4]
private def K1 : Kernel (Fin 2) := !![2/3, 1/3; 1/3, 2/3]
private def sched : KernelSchedule (Fin 2) := fun k => if k % 2 = 0 then K0 else K1
private def x0 : Fin 2 → Rat := ![1, 0]

example :
    applyKernel (windowKernel sched 0 2) x0 =
      varyingTrajectory sched x0 2 := by
  exact apply_windowKernel sched x0 0 2

example (k : Nat) :
    coordRange (varyingTrajectory sched x0 (k + 1)) ≤
      coordRange (varyingTrajectory sched x0 k) := by
  apply coordRange_varying_le
  intro t
  by_cases h : t % 2 = 0 <;> simp [sched, h, K0, K1, AveragingKernel]
```

Add a second consumer using an abstract schedule and uniform block hypothesis:

```lean
example {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    ∃ c : Real, ∀ i,
      Tendsto (fun k => (varyingTrajectory K x k i : Real)) atTop (nhds c) := by
  exact block_contraction_consensus_exists K hK x b hb δ hδ0 hδ1 hc
```

- [ ] **Step 2: Run the focused RED and accept only missing-module/declaration failure**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FiniteConsensus
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
```

Expected: `FiniteConsensus` builds; the new test fails because `FiniteTimeVaryingConsensus` or its declarations do not yet exist. Do not accept matrix syntax, import, or test-harness failures as RED evidence.

- [ ] **Step 3: Commit the valid RED**

```bash
git add NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
git commit -m "test(lean): add time-varying consensus RED"
git push
```

Record the exact RED SHA and failing declaration on #94.

- [ ] **Step 4: Add the ordered schedule/trajectory definitions and composition lemmas**

Create `NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean` beginning with:

```lean
import NarrativeDynamics.Core.FiniteConsensus

namespace NarrativeDynamics.FiniteTimeVaryingConsensus

open NarrativeDynamics.FiniteConsensus
open Filter Topology
open scoped BigOperators

abbrev KernelSchedule (ι : Type*) := Nat → Kernel ι

def varyingTrajectory {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) : Nat → ι → Rat
  | 0 => x
  | k + 1 => applyKernel (K k) (varyingTrajectory K x k)

@[simp] theorem varyingTrajectory_zero {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) :
    varyingTrajectory K x 0 = x := rfl

@[simp] theorem varyingTrajectory_succ {ι : Type*} [Fintype ι]
    (K : KernelSchedule ι) (x : ι → Rat) (k : Nat) :
    varyingTrajectory K x (k + 1) =
      applyKernel (K k) (varyingTrajectory K x k) := rfl


def windowKernel {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (start : Nat) : Nat → Kernel ι
  | 0 => 1
  | len + 1 => K (start + len) * windowKernel K start len
```

Prove the public composition theorem with this exact orientation:

```lean
theorem apply_windowKernel {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (x : ι → Rat) (start len : Nat) :
    applyKernel (windowKernel K start len) (varyingTrajectory K x start) =
      varyingTrajectory K x (start + len) := by
  induction len with
  | zero => simp [windowKernel, applyKernel]
  | succ len ih =>
      rw [windowKernel, applyKernel_mul, ih]
      simpa [Nat.add_assoc] using
        (varyingTrajectory_succ K x (start + len)).symm
```

If `simp [applyKernel]` is needed for the identity-matrix base case, keep the definition and product orientation unchanged.

- [ ] **Step 5: Prove window averaging and coordinate monotonicity**

Use one private matrix-product closure theorem local to this module:

```lean
private theorem averaging_mul {ι : Type*} [Fintype ι]
    {A B : Kernel ι} (hA : AveragingKernel A) (hB : AveragingKernel B) :
    AveragingKernel (A * B) := by
  classical
  constructor
  · intro i j
    simp only [Matrix.mul_apply]
    exact Finset.sum_nonneg fun k _ => mul_nonneg (hA.nonneg i k) (hB.nonneg k j)
  · intro i
    simp only [Matrix.mul_apply]
    calc
      (∑ j, ∑ k, A i k * B k j) = ∑ k, ∑ j, A i k * B k j := by rw [Finset.sum_comm]
      _ = ∑ k, A i k * (∑ j, B k j) := by
        apply Finset.sum_congr rfl
        intro k _
        rw [Finset.mul_sum]
      _ = ∑ k, A i k := by simp [hB.row_sum]
      _ = 1 := hA.row_sum i
```

For the identity base case, prove `AveragingKernel (1 : Kernel ι)` locally using `Matrix.one_apply`. Then expose:

```lean
theorem windowKernel_averaging {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (start len : Nat) : AveragingKernel (windowKernel K start len)
```

and:

```lean
theorem coordRange_varying_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (k : Nat) :
    coordRange (varyingTrajectory K x (k + 1)) ≤
      coordRange (varyingTrajectory K x k) := by
  rw [varyingTrajectory_succ]
  exact coordRange_apply_le (hK k) _
```

For monotone extrema, unfold only `coordMin`/`coordMax` and use `applyKernel_between`:

```lean
theorem coordMin_varying_mono {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) :
    Monotone (fun k => coordMin (varyingTrajectory K x k))

theorem coordMax_varying_anti {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) :
    Antitone (fun k => coordMax (varyingTrajectory K x k))
```

Prove successor inequalities first using `Finset.le_inf'` and `Finset.sup'_le`, then lift them to `Monotone`/`Antitone` by induction on `Nat.le`.

- [ ] **Step 6: Add uniform block contraction and the geometric range bound**

Define:

```lean
def UniformBlockCommonColumn {ι : Type*} [Fintype ι] [DecidableEq ι]
    (K : KernelSchedule ι) (b : Nat) (δ : Rat) : Prop :=
  ∀ start, CommonColumnMass (windowKernel K start b) δ
```

Add the aligned one-block theorem:

```lean
theorem coordRange_block_le {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (start b : Nat)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    coordRange (varyingTrajectory K x (start + b)) ≤
      (1 - δ) * coordRange (varyingTrajectory K x start) := by
  rw [← apply_windowKernel K x start b]
  exact coordRange_apply_le_of_commonColumn
    (windowKernel_averaging K hK start b) δ hδ0 hδ1 (hc start) _
```

Then prove by induction:

```lean
theorem block_geometric_bound {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) (q : Nat) :
    coordRange (varyingTrajectory K x (q * b)) ≤
      (1 - δ) ^ q * coordRange x
```

Use `coordRange_block_le` at `start = q * b`, `Nat.succ_mul`, and nonnegativity of `1-δ`.

Also prove a tail lemma:

```lean
theorem coordRange_varying_le_of_le {ι : Type*} [Fintype ι] [Nonempty ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) {a k : Nat} (hak : a ≤ k) :
    coordRange (varyingTrajectory K x k) ≤
      coordRange (varyingTrajectory K x a)
```

by induction over the gap `k-a` using `coordRange_varying_le`.

- [ ] **Step 7: Prove range tends to zero and then the existential common limit**

First expose:

```lean
theorem coordRange_tendsto_zero {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    Tendsto (fun k => (coordRange (varyingTrajectory K x k) : Real))
      atTop (nhds 0)
```

Follow the already verified geometric-tail structure in `FiniteConsensus.block_contraction_tendsto`: obtain the geometric limit of `((1-δ : Rat) : Real)^q`, choose a block boundary `q*b`, then use `coordRange_varying_le_of_le` for arbitrary later `k`.

Before writing the common-limit proof, verify the exact complete-space API available in this pinned Mathlib with a temporary file:

```bash
cat >/tmp/check_complete.lean <<'EOF'
import Mathlib
#check Metric.cauchySeq_iff
#check cauchySeq_tendsto_of_complete
#check tendsto_nhds_unique
EOF
lake env lean /tmp/check_complete.lean
```

Use the available Cauchy-sequence theorem to prove the trajectory of one fixed coordinate converges. The Cauchy estimate must be based on a shared earlier interval: for `m,n >= N`, both coordinate values lie between `coordMin (varyingTrajectory K x N)` and `coordMax (varyingTrajectory K x N)`, hence their real distance is at most the range at `N`. Then show every other coordinate has the same limit because at each time its distance from the reference coordinate is bounded by the current range, which tends to zero.

Expose exactly:

```lean
theorem block_contraction_consensus_exists
    {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    ∃ c : Real, ∀ i,
      Tendsto (fun k => (varyingTrajectory K x k i : Real)) atTop (nhds c)
```

Do not add stationary weights or a weighted-mean parameter to this theorem.

- [ ] **Step 8: Run Task 1 GREEN and add axiom reports**

Append to the generic test file:

```lean
#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.apply_windowKernel
#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.coordRange_tendsto_zero
#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.block_contraction_consensus_exists
```

Run:

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FiniteTimeVaryingConsensus
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
```

Expected: PASS, with theorem reports using only the repository's existing trust allowlist.

- [ ] **Step 9: Commit Task 1 GREEN**

```bash
git add NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean \
  NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
git commit -m "feat(lean): add time-varying consensus contraction"
git push
```

Do not start Task 2 until the exact-head proof run is green.

---

### Task 2: Executable all-broadcast exposure trajectory and time-varying kernel bridge

**Files:**
- Create: `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`
- Create: `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`

**Interfaces:**
- Consumes: merged `FitnessABMPathNExposure.ExposureParameters`, `State`, `beliefs`, `incoming`, `step`, `BeliefsBounded`; `FitnessABMPathN.neighbors`, `degree`, `degree_pos`, `degree_le_two`; Task 1 `KernelSchedule`, `varyingTrajectory`; `FiniteConsensus.AveragingKernel`, `applyKernel`.
- Produces: `allBroadcast`, `allBroadcast_step`, `allBroadcast_iterate`, `incoming_eq_degree`, `exposure_iterate`, `exposureKernel`, `exposureKernel_averaging`, `step_beliefs_eq_kernel`, `kernelSchedule`, `beliefs_iterate_eq_varyingTrajectory`.

- [ ] **Step 1: Write Task 2 RED consumers including an explicit first-step indexing fixture**

Create `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`:

```lean
import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open NarrativeDynamics.FitnessABMPathNExposureConvergence
open Filter Topology

private def indexSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

private def path3zero : State 3 :=
  ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]

example : allBroadcast indexSchedule path3zero := by
  intro i
  fin_cases i <;> norm_num [allBroadcast, indexSchedule, path3zero]

example : incoming indexSchedule path3zero 0 = 1 := by decide_cbv
example : incoming indexSchedule path3zero 1 = 2 := by decide_cbv

-- First transition must query alpha(1) at endpoints and alpha(2) at the middle.
example :
    exposureKernel indexSchedule 3 (fun _ => 0) 0 0 = 3/4 := by
  norm_num [exposureKernel, indexSchedule, FitnessABMPathN.degree,
    FitnessABMPathN.neighbors]

example :
    exposureKernel indexSchedule 3 (fun _ => 0) 1 1 = 1/2 := by
  norm_num [exposureKernel, indexSchedule, FitnessABMPathN.degree,
    FitnessABMPathN.neighbors]

example (k : Nat) (i : Fin 3) :
    (((step indexSchedule 3)^[k] path3zero) i).exposure =
      (path3zero i).exposure + k * FitnessABMPathN.degree 3 i := by
  exact exposure_iterate indexSchedule
    (by norm_num [indexSchedule, ExposureParameters.Valid]) 3 (by omega)
    path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k i

example (k : Nat) :
    beliefs ((step indexSchedule 3)^[k] path3zero) =
      varyingTrajectory (kernelSchedule indexSchedule 3 path3zero)
        (beliefs path3zero) k := by
  exact beliefs_iterate_eq_varyingTrajectory indexSchedule
    (by norm_num [indexSchedule, ExposureParameters.Valid]) 3 (by omega)
    path3zero (by
      intro j
      fin_cases j <;> norm_num [allBroadcast, indexSchedule, path3zero]) k
```

- [ ] **Step 2: Run and commit the intended Task 2 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
```

Expected: missing convergence module/declarations. The existing `FitnessABMPathNExposure` test remains green:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNExposure.lean
```

Commit only the RED consumer:

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "test(lean): add exposure PathN convergence bridge RED"
git push
```

- [ ] **Step 3: Define all-broadcast and prove preservation**

Start the new core module:

```lean
import NarrativeDynamics.Core.FitnessABMPathNExposure
import NarrativeDynamics.Core.FiniteTimeVaryingConsensus

namespace NarrativeDynamics.FitnessABMPathNExposureConvergence

open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open NarrativeDynamics.FitnessABMPathNExposure
open Filter Topology
open scoped BigOperators

def allBroadcast (p : ExposureParameters) (s : State n) : Prop :=
  ∀ i, p.threshold ≤ (s i).belief ∧ (s i).belief ≤ 1
```

Prove `incoming_eq_degree` by showing the broadcaster filter is exactly `neighbors n i` when every neighbor broadcasts:

```lean
theorem incoming_eq_degree
    (p : ExposureParameters) (n : Nat) (s : State n)
    (h : allBroadcast p s) (i : Fin n) :
    incoming p s i = FitnessABMPathN.degree n i
```

Then prove:

```lean
theorem allBroadcast_step
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) : allBroadcast p (step p n s)

theorem allBroadcast_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) (k : Nat) :
    allBroadcast p ((step p n)^[k] s)
```

For `allBroadcast_step`, use the existing convex-combination reasoning and `hvalid`; do not mutate the existing exposure module just to export its private broadcaster-mean bound.

- [ ] **Step 4: Prove the exact exposure iterate law**

Add:

```lean
theorem exposure_step_eq_add_degree
    (p : ExposureParameters) (n : Nat) (s : State n)
    (h : allBroadcast p s) (i : Fin n) :
    (step p n s i).exposure =
      (s i).exposure + FitnessABMPathN.degree n i
```

by unfolding `step`, rewriting `incoming_eq_degree`, and using `degree_pos` under `n >= 2` when the `received = 0` branch must be excluded.

Then prove by induction:

```lean
theorem exposure_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) (k : Nat) (i : Fin n) :
    (((step p n)^[k] s) i).exposure =
      (s i).exposure + k * FitnessABMPathN.degree n i
```

The successor case must use `allBroadcast_iterate` for the current executable state. This is the proof that fixes the later effective-alpha index.

- [ ] **Step 5: Define the row-dependent proof kernel with post-incoming lookup semantics**

Add exactly one proof kernel:

```lean
def exposureKernel (p : ExposureParameters) (n : Nat)
    (e : Fin n → Nat) : Kernel (Fin n) :=
  fun i j =>
    let alpha := p.receptivityAt (e i + FitnessABMPathN.degree n i)
    (if i = j then 1 - alpha else 0) +
      (if j ∈ FitnessABMPathN.neighbors n i then
        alpha / (FitnessABMPathN.degree n i : Rat)
      else 0)
```

Prove:

```lean
theorem exposureKernel_averaging
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (e : Fin n → Nat) :
    AveragingKernel (exposureKernel p n e)
```

Use the same degree `1 ∨ 2` split already established in the constant-parameter modules. Row sums remain one even though `alpha` differs by row.

- [ ] **Step 6: Bridge one executable step and then all finite iterates**

Prove one-step belief equality:

```lean
theorem step_beliefs_eq_kernel
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s : State n)
    (h : allBroadcast p s) :
    beliefs (step p n s) =
      applyKernel (exposureKernel p n (fun i => (s i).exposure)) (beliefs s)
```

The proof must rewrite the executable `received` to `degree`, and the effective alpha to `p.receptivityAt ((s i).exposure + degree n i)`. Do not use the exposure-independent constant-schedule bridge here.

Define the initial-state-derived schedule:

```lean
def kernelSchedule (p : ExposureParameters) (n : Nat) (s0 : State n) :
    KernelSchedule (Fin n) :=
  fun k => exposureKernel p n
    (fun i => (s0 i).exposure + k * FitnessABMPathN.degree n i)
```

Then prove:

```lean
theorem kernelSchedule_averaging
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n) :
    ∀ k, AveragingKernel (kernelSchedule p n s0 k)

theorem beliefs_iterate_eq_varyingTrajectory
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (h : allBroadcast p s0) (k : Nat) :
    beliefs ((step p n)^[k] s0) =
      varyingTrajectory (kernelSchedule p n s0) (beliefs s0) k
```

The induction successor case must use both `exposure_iterate` and `allBroadcast_iterate`; this prevents an accidental `k` versus `k+1` lookup shift.

- [ ] **Step 7: Run Task 2 GREEN and commit**

Append axiom reports for `exposure_iterate` and `beliefs_iterate_eq_varyingTrajectory`, then run:

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
```

Expected: PASS, including the first-step `alpha(1)`/`alpha(2)` fixture.

Commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "feat(lean): bridge exposure PathN to varying kernels"
git push
```

Do not start Task 3 until exact-head proof CI is green.

---

### Task 3: Path2 exact disagreement product and arithmetic-mean characterization

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`

**Interfaces:**
- Consumes: Task 2 exact exposure law and executable/kernel bridge.
- Produces: `path2_equal_exposure_iterate`, `path2_disagreement_step`, `path2_disagreement_product`, `path2_mean_step`, `path2_mean_iterate`, `path2_consensus_iff_product_tendsto_zero`.

- [ ] **Step 1: Add the Path2 RED consumers**

Append a threshold-zero valid abstract schedule consumer. Use equal initial exposures `e0` and define the exact multiplier product:

```lean
private def path2Product (p : ExposureParameters) (e0 k : Nat) : Rat :=
  ∏ r ∈ Finset.range k,
    (1 - 2 * p.receptivityAt (e0 + r + 1))

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2Product p (s 0).exposure k := by
  exact path2_disagreement_product p hvalid s he hb k

example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 + beliefs ((step p 2)^[k] s) 1) / 2 =
      ((s 0).belief + (s 1).belief) / 2 := by
  exact path2_mean_iterate p hvalid s he hb k
```

Add the convergence-characterization consumer with nonzero initial disagreement:

```lean
example (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief) :
    (Tendsto
      (fun k => |(path2Product p (s 0).exposure k : Rat) : Real|)
      atTop (nhds 0)) ↔
    (∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real))) := by
  exact path2_consensus_iff_product_tendsto_zero p hvalid s he hb hne
```

- [ ] **Step 2: Run and commit the intended Task 3 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
```

Expected: Task 2 remains green; failure is on missing Path2 declarations.

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "test(lean): add exposure Path2 product RED"
git push
```

- [ ] **Step 3: Prove equal exposures and exact one-step disagreement**

For Path2, prove degree is one at each coordinate and therefore equal exposure histories remain equal:

```lean
theorem path2_equal_exposure_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (((step p 2)^[k] s) 0).exposure =
      (((step p 2)^[k] s) 1).exposure
```

Use `exposure_iterate` on both vertices and the exact Path2 degree computation.

Then prove the one-step recurrence for any all-broadcast Path2 state with equal exposures:

```lean
theorem path2_disagreement_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    (step p 2 s 0).belief - (step p 2 s 1).belief =
      (1 - 2 * p.receptivityAt ((s 0).exposure + 1)) *
        ((s 0).belief - (s 1).belief)
```

Unfold the actual executable step only after rewriting both incoming counts to one; use `he` to identify the two queried alphas.

- [ ] **Step 4: Prove the finite product formula and arithmetic mean preservation**

Define a public product helper in the convergence module so test and theorem names do not duplicate a private test definition:

```lean
def path2MultiplierProduct (p : ExposureParameters) (e0 k : Nat) : Rat :=
  ∏ r ∈ Finset.range k,
    (1 - 2 * p.receptivityAt (e0 + r + 1))
```

Update the RED consumer to use `path2MultiplierProduct` and remove its private duplicate.

Prove by induction:

```lean
theorem path2_disagreement_product
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 - beliefs ((step p 2)^[k] s) 1) =
      ((s 0).belief - (s 1).belief) *
        path2MultiplierProduct p (s 0).exposure k
```

Also prove:

```lean
theorem path2_mean_step
    (p : ExposureParameters) (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) :
    ((step p 2 s 0).belief + (step p 2 s 1).belief) / 2 =
      ((s 0).belief + (s 1).belief) / 2

theorem path2_mean_iterate
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s) (k : Nat) :
    (beliefs ((step p 2)^[k] s) 0 + beliefs ((step p 2)^[k] s) 1) / 2 =
      ((s 0).belief + (s 1).belief) / 2
```

The finite-iterate mean proof uses `path2_equal_exposure_iterate` and `allBroadcast_iterate` at the induction step.

- [ ] **Step 5: Derive the product-to-zero iff consensus theorem**

Use the exact algebraic identities

```text
x0 = mean + disagreement/2
x1 = mean - disagreement/2
```

for every iterate, together with `path2_mean_iterate` and `path2_disagreement_product`. Under `hne`, multiplication by the nonzero initial disagreement is reversible for the necessary direction.

Expose:

```lean
theorem path2_consensus_iff_product_tendsto_zero
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2) (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief) :
    (Tendsto
      (fun k => |(path2MultiplierProduct p (s 0).exposure k : Rat) : Real|)
      atTop (nhds 0)) ↔
    (∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop
        (nhds ((((s 0).belief + (s 1).belief) / 2 : Rat) : Real)))
```

If Mathlib's `abs` coercion normal form differs, normalize only the theorem syntax; preserve the exact mathematical iff.

- [ ] **Step 6: Run Task 3 GREEN and commit**

Add axiom reports for `path2_disagreement_product` and `path2_consensus_iff_product_tendsto_zero`, then run the focused module/test commands. Commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "feat(lean): characterize exposure Path2 disagreement"
git push
```

Require exact-head proof CI success before Task 4.

---

### Task 4: Exact Path2 positive/negative schedules and Path3 mean non-invariance

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`

**Interfaces:**
- Consumes: Task 3 product/mean theorems.
- Produces theorem-level exact witnesses for: strict-interior non-consensus with `alpha(e) -> 0` too fast, strict-interior oscillatory non-convergence with `alpha(e) -> 1`, a harmonic schedule with `alpha(e) -> 0` that does reach consensus, and Path3 failure of the old degree-weighted invariant.

- [ ] **Step 1: Add exact schedule definitions and RED consumers**

Define the schedules in the core module, not only in tests, so their theorem reports are stable:

```lean
def slowZeroSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def nearOneSchedule : ExposureParameters :=
  ⟨fun e => 1 - 1 / (2 * (((e + 1 : Nat) : Rat) ^ 2)), 0⟩

def harmonicSchedule : ExposureParameters :=
  ⟨fun e => 1 / (2 * ((e + 1 : Nat) : Rat)), 0⟩

private def split2 : State 2 := ![⟨1, 0⟩, ⟨0, 0⟩]
```

The test file must consume public validity and closed-form product theorems:

```lean
example : slowZeroSchedule.Valid := slowZeroSchedule_valid
example : nearOneSchedule.Valid := nearOneSchedule_valid
example : harmonicSchedule.Valid := harmonicSchedule_valid

example (k : Nat) :
    path2MultiplierProduct slowZeroSchedule 0 k =
      (k + 2 : Rat) / (2 * (k + 1 : Rat)) := by
  exact slowZero_product k

example (k : Nat) :
    path2MultiplierProduct harmonicSchedule 0 k =
      1 / (k + 1 : Rat) := by
  exact harmonic_product k
```

Add consumers for the non-convergence/convergence theorem reports and for the Path3 witness named below.

- [ ] **Step 2: Run and commit Task 4 RED**

Run the exposure convergence test. Expected: Tasks 2-3 stay green; failure is on the new schedule declarations/theorems. Commit the RED test change and push.

- [ ] **Step 3: Prove schedule validity and exact telescoping products**

For each schedule prove `ExposureParameters.Valid`. Use positivity of `(e+1 : Rat)` and denominator bounds; no numerical approximation.

Prove by induction on `k`, using `Finset.prod_range_succ` and field normalization:

```lean
theorem slowZero_product (k : Nat) :
    path2MultiplierProduct slowZeroSchedule 0 k =
      (k + 2 : Rat) / (2 * (k + 1 : Rat))

theorem nearOne_product (k : Nat) :
    path2MultiplierProduct nearOneSchedule 0 k =
      (-1 : Rat) ^ k * ((k + 2 : Rat) / (2 * (k + 1 : Rat)))

theorem harmonic_product (k : Nat) :
    path2MultiplierProduct harmonicSchedule 0 k =
      1 / (k + 1 : Rat)
```

The formulas are indexed for the merged post-incoming lookup: first update queries exposure `1`, so the first slow-zero factor is `1 - 1/4 = 3/4` and the closed form at `k=1` is `3/4`. Add a concrete `k=1` test so an off-by-one proof cannot pass.

- [ ] **Step 4: Prove the two negative and one positive Path2 behaviors**

Use exact real limits of the closed forms. Verify the pinned Mathlib sequence API before coding with:

```bash
cat >/tmp/check_seq.lean <<'EOF'
import Mathlib
#check tendsto_natCast_atTop_atTop
#check tendsto_one_div_add_atTop_nhds_zero_nat
#check tendsto_nhds_unique
EOF
lake env lean /tmp/check_seq.lean
```

Expose theorem-level behavior:

```lean
theorem slowZero_product_tendsto_half :
    Tendsto
      (fun k => (path2MultiplierProduct slowZeroSchedule 0 k : Real))
      atTop (nhds (1/2 : Real))

theorem slowZero_not_consensus :
    ¬ ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step slowZeroSchedule 2)^[k] split2) i : Real))
        atTop (nhds (1/2 : Real))

theorem nearOne_not_convergent :
    ¬ ∃ c : Real, ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step nearOneSchedule 2)^[k] split2) i : Real))
        atTop (nhds c)

theorem harmonic_consensus :
    ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step harmonicSchedule 2)^[k] split2) i : Real))
        atTop (nhds (1/2 : Real))
```

For `nearOne_not_convergent`, use the exact `(-1)^k` product and prove even/odd disagreement subsequences have distinct limits `1/2` and `-1/2`; then apply uniqueness of limits. Do not replace this with a finite-cycle test or floating-point evidence.

- [ ] **Step 5: Add the Path3 exact degree-weighted-mean non-invariance witness**

Define:

```lean
def degreeSplitSchedule : ExposureParameters :=
  ⟨fun e => if e = 1 then 1/4 else if e = 2 then 1/2 else 1/3, 0⟩

def degreeSplitState : State 3 :=
  ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]
```

Prove exact state/mean facts:

```lean
theorem degreeSplit_step_beliefs :
    beliefs (step degreeSplitSchedule 3 degreeSplitState) =
      ![3/4, 1/4, 0]

theorem degreeSplit_mean_before :
    FitnessABMPathN.mean 3 (beliefs degreeSplitState) = 1/4

theorem degreeSplit_mean_after :
    FitnessABMPathN.mean 3
      (beliefs (step degreeSplitSchedule 3 degreeSplitState)) = 5/16

theorem exposure_degree_weighted_mean_not_invariant :
    FitnessABMPathN.mean 3
        (beliefs (step degreeSplitSchedule 3 degreeSplitState)) ≠
      FitnessABMPathN.mean 3 (beliefs degreeSplitState)
```

This witness is mandatory regression evidence that PR #93's consensus-value theorem cannot be reused under row-dependent effective receptivity.

- [ ] **Step 6: Run Task 4 GREEN and commit**

Add axiom reports for `slowZero_not_consensus`, `nearOne_not_convergent`, `harmonic_consensus`, and `exposure_degree_weighted_mean_not_invariant`. Run the focused core/test commands and commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "feat(lean): prove exposure schedule convergence boundaries"
git push
```

Require exact-head proof CI success before Task 5.

---

### Task 5: Uniform-interior finite-PathN common-column theorem and executable consensus

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean`

**Interfaces:**
- Consumes: Task 1 `windowKernel`, `UniformBlockCommonColumn`, `block_contraction_consensus_exists`; Task 2 `kernelSchedule`, executable bridge; existing PathN degree/path/block helpers.
- Produces: `ReachableInterior`, `beta`, `delta`, `exposureKernel_self_lower`, `exposureKernel_adj_lower`, `path_window_common_mass`, `trajectory_consensus_exists`, `trajectory_consensus_exists_of_global_interior`.

- [ ] **Step 1: Add Task 5 RED consumers**

Append an abstract consumer using the exact reachable lookup points:

```lean
example (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Rat : Real))
        atTop (nhds c) := by
  exact trajectory_consensus_exists p hvalid n hn s0 hb eps heps hi
```

Add a convenience consumer for a global schedule bound:

```lean
example (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ∀ e, eps ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1 - eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Rat : Real))
        atTop (nhds c) := by
  exact trajectory_consensus_exists_of_global_interior
    p hvalid n hn s0 hb eps heps hi
```

- [ ] **Step 2: Run and commit Task 5 RED**

Run the focused exposure convergence test. Expected: all Task 2-4 consumers stay green; failure is on the uniform-interior declarations. Commit and push the RED test change.

- [ ] **Step 3: State the reachable interior condition and scalar contraction constants**

Define:

```lean
def ReachableInterior (p : ExposureParameters) (n : Nat)
    (s0 : State n) (eps : Rat) : Prop :=
  ∀ k i,
    eps ≤ p.receptivityAt
      ((s0 i).exposure + (k + 1) * FitnessABMPathN.degree n i) ∧
    p.receptivityAt
      ((s0 i).exposure + (k + 1) * FitnessABMPathN.degree n i) ≤ 1 - eps

def beta (eps : Rat) : Rat := eps / 2

def delta (eps : Rat) (n : Nat) : Rat :=
  beta eps ^ FitnessABMPathN.block n
```

Prove from `heps : 0 < eps`, `ReachableInterior`, and `n >= 2` that `0 < beta eps`, `beta eps < 1`, `0 < delta eps n`, and `delta eps n < 1`. To derive the upper bound on `eps`, choose vertex `0` and time `0` from the reachable condition; `eps ≤ alpha ≤ 1-eps` gives `eps ≤ 1/2`, hence `beta ≤ 1/4 < 1`.

- [ ] **Step 4: Prove uniform self/edge lower bounds for every scheduled kernel**

Expose:

```lean
theorem exposureKernel_self_lower
    (p : ExposureParameters) (n : Nat) (s0 : State n)
    (eps : Rat) (hi : ReachableInterior p n s0 eps)
    (k : Nat) (i : Fin n) :
    beta eps ≤ kernelSchedule p n s0 k i i

theorem exposureKernel_adj_lower
    (p : ExposureParameters) (n : Nat) (hn : 2 ≤ n)
    (s0 : State n) (eps : Rat) (hi : ReachableInterior p n s0 eps)
    (k : Nat) (i j : Fin n)
    (hadj : FitnessABMPathN.pathAdj n j i) :
    beta eps ≤ kernelSchedule p n s0 k i j
```

The self proof uses `alpha ≤ 1-eps`, so `1-alpha ≥ eps ≥ eps/2`. The edge proof uses `alpha ≥ eps` and `degree(i) ∈ {1,2}`, so `alpha/degree(i) ≥ eps/2`.

- [ ] **Step 5: Prove common origin-column mass for every aligned time window**

Use origin basis `z = 0` locally in the exposure convergence module:

```lean
private def originBasis {n : Nat} (z : Fin n) : Fin n → Rat :=
  fun j => if j = z then 1 else 0
```

Prove a time-varying reach lemma: after `r` steps starting at arbitrary schedule time `start`, the origin basis has at least `(beta eps)^r` mass at vertex `r` for `r < n`. Each induction step uses `exposureKernel_adj_lower` at schedule index `start + r` and nonnegativity from `kernelSchedule_averaging`.

Prove a self-padding lemma from any reached vertex for the remaining block steps using `exposureKernel_self_lower` at the correct later schedule indices.

Combine them for `b = FitnessABMPathN.block n = n-1`, then rewrite basis propagation through `apply_windowKernel` to matrix-column entries. Expose:

```lean
theorem path_window_common_mass
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    UniformBlockCommonColumn
      (kernelSchedule p n s0)
      (FitnessABMPathN.block n)
      (delta eps n)
```

The theorem must quantify all block starts through `UniformBlockCommonColumn`; proving only the first block is insufficient for a non-homogeneous trajectory.

- [ ] **Step 6: Compose generic contraction with the executable bridge**

First prove the kernel-trajectory theorem:

```lean
theorem varyingTrajectory_consensus_exists
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => (varyingTrajectory (kernelSchedule p n s0)
          (beliefs s0) k i : Real))
        atTop (nhds c)
```

Apply `block_contraction_consensus_exists` with:

```text
K     = kernelSchedule p n s0
b     = FitnessABMPathN.block n
delta = delta eps n
```

using `kernelSchedule_averaging`, `FitnessABMPathN.block_pos`, scalar delta bounds, and `path_window_common_mass`.

Then expose the acceptance theorem on the actual executable state trajectory:

```lean
theorem trajectory_consensus_exists
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ReachableInterior p n s0 eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Rat : Real))
        atTop (nhds c)
```

Rewrite each term using `beliefs_iterate_eq_varyingTrajectory`. Do not mention `FitnessABMPathN.mean` in the theorem or proof conclusion.

Finally add:

```lean
theorem trajectory_consensus_exists_of_global_interior
    (p : ExposureParameters) (hvalid : p.Valid)
    (n : Nat) (hn : 2 ≤ n) (s0 : State n)
    (hb : allBroadcast p s0)
    (eps : Rat) (heps : 0 < eps)
    (hi : ∀ e, eps ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1 - eps) :
    ∃ c : Real, ∀ i,
      Tendsto
        (fun k => ((((step p n)^[k] s0) i).belief : Rat : Real))
        atTop (nhds c)
```

by instantiating `ReachableInterior` at the exact reachable lookup index.

- [ ] **Step 7: Run Task 5 GREEN and commit**

Add axiom reports for `path_window_common_mass`, `trajectory_consensus_exists`, and the global corollary. Run focused build/test commands and commit:

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
git commit -m "feat(lean): prove exposure-dependent PathN consensus"
git push
```

Require exact-head proof CI success before the permanent-gate task.

---

### Task 6: Permanent PathN gate, trust audit, scope audit, and review-ready state

**Files:**
- Modify: `tools/check_fitness_abm_pathn.sh`
- No workflow edit expected.

**Interfaces:**
- Consumes: all Task 1-5 production/test modules and theorem reports.
- Produces: permanent bounded source/build/test/trust coverage for the new generic and exposure-convergence theorems while retaining every existing fixed/parameterized/exposure check.

- [ ] **Step 1: Extend temporary logs and cleanup without removing existing logs**

Add:

```bash
time_varying_log="$(mktemp)"
exposure_convergence_log="$(mktemp)"
```

and include both in the existing trap. Preserve `consensus_log`, `pathn_log`, `params_log`, `parameter_convergence_log`, and `exposure_log`.

- [ ] **Step 2: Extend source audit additively**

Add these four paths to the existing `audit_fitness_trust.py source` invocation:

```text
NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean
NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean
NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
```

Do not remove any existing audited source.

- [ ] **Step 3: Extend bounded module builds**

Add to the build loop:

```text
NarrativeDynamics.Core.FiniteTimeVaryingConsensus
NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
```

Keep the same `timeout --kill-after=10s 240s` resource boundary.

- [ ] **Step 4: Add focused test executions and theorem-report audits**

Add:

```bash
"$pathn_time" -f 'FiniteTimeVaryingConsensus tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean \
  2>&1 | tee "$time_varying_log"

"$pathn_time" -f 'FitnessABMPathNExposureConvergence tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean \
  2>&1 | tee "$exposure_convergence_log"
```

Require generic reports:

```bash
python3 tools/audit_fitness_trust.py log "$time_varying_log" \
  --require NarrativeDynamics.FiniteTimeVaryingConsensus.apply_windowKernel \
  --require NarrativeDynamics.FiniteTimeVaryingConsensus.coordRange_tendsto_zero \
  --require NarrativeDynamics.FiniteTimeVaryingConsensus.block_contraction_consensus_exists
```

Require exposure-convergence reports:

```bash
python3 tools/audit_fitness_trust.py log "$exposure_convergence_log" \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.exposure_iterate \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.beliefs_iterate_eq_varyingTrajectory \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.path2_disagreement_product \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.path2_consensus_iff_product_tendsto_zero \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.slowZero_not_consensus \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.nearOne_not_convergent \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.harmonic_consensus \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.exposure_degree_weighted_mean_not_invariant \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.path_window_common_mass \
  --require NarrativeDynamics.FitnessABMPathNExposureConvergence.trajectory_consensus_exists
```

Keep all existing `--require` entries for fixed PathN, PR #93 parameter convergence, and merged exposure-learning v1.

- [ ] **Step 5: Run the full permanent PathN gate locally**

```bash
timeout --kill-after=10s 240s bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS. If it exceeds the existing bound, profile the new focused modules; do not increase the timeout as the first response.

- [ ] **Step 6: Run root build and scope diff**

```bash
timeout --kill-after=10s 240s lake build
git diff --check
git diff --stat fb1c6aa160975b87aa31164abdb64dd961971d7f...HEAD
git diff --name-only fb1c6aa160975b87aa31164abdb64dd961971d7f...HEAD
```

Expected implementation scope after the already committed spec/plan:

```text
NarrativeDynamics/Core/FiniteTimeVaryingConsensus.lean
NarrativeDynamics/Tests/FiniteTimeVaryingConsensus.lean
NarrativeDynamics/Core/FitnessABMPathNExposureConvergence.lean
NarrativeDynamics/Tests/FitnessABMPathNExposureConvergence.lean
tools/check_fitness_abm_pathn.sh
docs/superpowers/specs/2026-09-16-bb-exposure-dependent-pathn-convergence-v1-design.md
docs/superpowers/plans/2026-09-16-bb-exposure-dependent-pathn-convergence-v1.md
```

Any production change to `NetworkPropagation`, fixed PathN, `FitnessABMPathNParameters`, or `FitnessABMPathNExposure.step` is a scope blocker and must be removed or separately re-approved.

- [ ] **Step 7: Commit the permanent gate and push exact head**

```bash
git add tools/check_fitness_abm_pathn.sh
git commit -m "test(lean): gate exposure-dependent PathN consensus"
git push
```

- [ ] **Step 8: Verify exact-head CI before review-ready status**

Require the proof workflow at the exact final head to show:

```text
Build Lean library                         success
BB finite-path convergence                success
Lean theorem tests                        success
```

Within `BB finite-path convergence`, the permanent script must show both new focused modules/tests and trust reports succeeding. Also require the World Studio workflow on the same exact head to succeed if it is triggered by the repository's normal push path.

Do not call the branch merge-ready if the CI evidence is from an earlier SHA.

- [ ] **Step 9: Final semantic audit before opening/marking a PR ready**

Read the final diff and verify each acceptance claim directly:

```text
actual theorem subject = FitnessABMPathNExposure.step
lookup order = incoming first, alpha(e') second
all-broadcast exposure law = e0 + k*degree
time-varying bridge uses alpha at e0 + (k+1)*degree
generic contraction has no StationaryWeights hypothesis
Path2 product is exact and post-incoming indexed
strict-interior non-consensus retained
near-one oscillatory non-convergence retained
alpha(e)->0 consensus example retained
Path3 1/4 -> 5/16 degree-weighted-mean witness retained
PathN uniform-interior conclusion is existential common c
no arbitrary-graph/floating-point/general-entry claim
```

Then use the normal review flow. The PR body must explicitly preserve the non-goals and cite exact-head proof/World Studio run IDs; do not summarize the new theorem as “degree-weighted consensus.”

---

## Plan Self-Review Checklist

Before implementation begins, confirm the plan covers every design requirement:

- Generic non-homogeneous consensus helper: Task 1.
- Ordered window multiplication orientation and trajectory equivalence: Task 1.
- Coordinate interval monotonicity, geometric range contraction, existential common limit: Task 1.
- Existing post-incoming exposure semantics and first-step indexing fixture: Task 2.
- Exact exposure iterate `e0 + k*degree`: Task 2.
- Actual executable-step to varying-kernel bridge: Task 2.
- Path2 exact disagreement product, equal-exposure mean preservation, iff criterion: Task 3.
- Two exact strict-interior failure schedules and one `alpha(e)->0` success schedule: Task 4.
- Path3 old degree-weighted mean changes from `1/4` to `5/16`: Task 4.
- Reachable uniform-interior lower bounds and every-window common origin mass: Task 5.
- Final executable PathN consensus-existence theorem without a closed-form value: Task 5.
- Existing fixed/parameterized/exposure theorems remain gated: Task 6.
- Trust/source/resource/scope audit and exact-head CI evidence: Task 6.

Implementation must stop and re-design rather than weaken a theorem if any task reveals that the spec's stated mathematical claim is false.