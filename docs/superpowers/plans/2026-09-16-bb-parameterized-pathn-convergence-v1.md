# BB Parameterized PathN Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that for every finite path `n >= 2`, every valid exact-rational response parameter with `0 < receptivity < 1`, and every initial state in the parameterized all-broadcast region, the actual executable belief trajectory converges coordinatewise to the existing degree-weighted PathN mean.

**Architecture:** Add one convergence module above `FitnessABMPathNParameters`; do not change runtime propagation semantics or `FiniteConsensus`. Reuse the public parameterized kernel, construct a local averaging witness from public nonnegativity/row-sum theorems, prove degree-normalized stationary weights, derive a conservative positive step mass `beta = alpha * (1 - alpha) / 2`, lift it to a block common-column mass over `n - 1` steps, apply `FiniteConsensus.block_contraction_tendsto`, and rewrite the proof-kernel trajectory back to the real executable `beliefStep` trajectory.

**Tech Stack:** Lean 4.32.0, Mathlib exact `Rat` arithmetic, existing `NarrativeDynamics.FiniteConsensus`, GitHub Actions proof/World Studio workflows, `tools/audit_fitness_trust.py`.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-parameterized-pathn-convergence-v1-design.md`

## Global Constraints

- Base production behavior is `proof/narrative-dynamics-v0@8115c3862700114fb91e495f32cecf9e765d7455`.
- Exact theorem-bearing state uses `Rat`; no floating-point theorem.
- Do not change `NetworkPropagation.propagate`, `FiniteConsensus`, fixed `FitnessABMPathN` semantics, or `ResponseParameters.Valid`.
- Strict convergence hypotheses are additional assumptions `0 < params.receptivity` and `params.receptivity < 1`; executable boundary values `0` and `1` remain valid parameters.
- Reuse `FiniteConsensus.block_contraction_tendsto`; do not duplicate its analytic proof.
- Do not claim arbitrary connected-graph convergence, convergence outside all-broadcast, or exposure-dependent convergence.
- Keep the existing fixed `alpha = 1/2` theorem and all existing Path4/Path5/PathN/exposure gates unchanged as regression evidence.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, globally disabled heartbeats, or disabled linters.
- Focused Lean commands and permanent gate commands retain the repository's 240-second timeout convention.
- Every production/test commit is pushed and verified on its exact SHA by the existing `proof.yml`; final integration additionally requires World Studio success.

---

## File Structure

- Create `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean` for generic parameterized convergence lemmas and the final theorem.
- Create `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean` for RED/GREEN consumers, exact endpoint fixtures, and axiom reports.
- Modify `tools/check_fitness_abm_pathn.sh` additively for bounded source/build/test/trust coverage.
- Do not modify `.github/workflows/proof.yml`; it already invokes the PathN gate for non-doc changes.
- Do not add a root import unless an actual build failure proves one is required.

---

### Task 1: Parameterized stationary weights and mean preservation

**Files:**
- Create: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Create: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: `FitnessABMPathNParameters.pathKernel`, `pathKernel_nonneg`, `pathKernel_rowsum`; `FitnessABMPathN.stationaryWeight`, `stationaryWeight_nonneg`, `stationaryWeight_sum_one`, `weightSum_pos`, `mean`; `FiniteConsensus.StationaryWeights`, `weightedMean_apply`, `kernelTrajectory`.
- Produces: `pathKernel_detailed_balance`, `path_stationary_weights`, `mean_kernel_step`, `mean_kernel_iterate`. A private `pathKernel_averaging` witness stays internal to the new module.

- [ ] **Step 1: Write the Task 1 RED consumers**

Create the test file with:

```lean
import NarrativeDynamics.Core.FitnessABMPathNParameterConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FitnessABMPathNParameters
open NarrativeDynamics.FitnessABMPathNParameterConvergence
open Filter Topology

private def p34 : ResponseParameters := ⟨3/4, 1/3⟩

example : ResponseParameters.Valid p34 := by
  norm_num [p34, ResponseParameters.Valid]

example (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights (pathKernel p34 n) (FitnessABMPathN.stationaryWeight n) := by
  exact path_stationary_weights p34
    (by norm_num [p34, ResponseParameters.Valid]) n hn

example (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) :
    FitnessABMPathN.mean n (applyKernel (pathKernel p34 n) x) =
      FitnessABMPathN.mean n x := by
  exact mean_kernel_step p34
    (by norm_num [p34, ResponseParameters.Valid]) n hn x
```

- [ ] **Step 2: Run the focused RED and accept only the intended missing-module/declaration failure**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathNParameters
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: `FitnessABMPathNParameters` builds; the new consumer fails because the convergence module or its declarations do not exist. A dependency/build-harness failure is invalid RED evidence.

- [ ] **Step 3: Commit and push the valid Task 1 RED**

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN convergence stationary RED"
git push
```

Record exact RED run/job IDs on #92.

- [ ] **Step 4: Create the convergence module and local averaging witness**

Start with:

```lean
import NarrativeDynamics.Core.FitnessABMPathNParameters

namespace NarrativeDynamics.FitnessABMPathNParameterConvergence

open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FitnessABMPathNParameters
open Filter Topology
open scoped BigOperators

private theorem pathKernel_averaging
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    AveragingKernel (pathKernel params n) := by
  exact ⟨
    FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn,
    FitnessABMPathNParameters.pathKernel_rowsum params hvalid n hn
  ⟩

private theorem pathAdj_symm {n : Nat} (i j : Fin n) :
    FitnessABMPathN.pathAdj n i j ↔ FitnessABMPathN.pathAdj n j i := by
  unfold FitnessABMPathN.pathAdj
  omega
```

Do not change visibility of the private averaging theorem already present in `FitnessABMPathNParameters`.

- [ ] **Step 5: Implement detailed balance with the exact public signature**

Add:

```lean
theorem pathKernel_detailed_balance
    (params : ResponseParameters) (_hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    FitnessABMPathN.stationaryWeight n i * pathKernel params n i j =
      FitnessABMPathN.stationaryWeight n j * pathKernel params n j i := by
  classical
  by_cases hij : i = j
  · subst j
    rfl
  · by_cases hadj : FitnessABMPathN.pathAdj n j i
    · have hji : FitnessABMPathN.pathAdj n i j := (pathAdj_symm i j).2 hadj
      have hwi : FitnessABMPathN.weightSum n ≠ 0 :=
        ne_of_gt (FitnessABMPathN.weightSum_pos n hn)
      have hdiNat : FitnessABMPathN.degree n i ≠ 0 :=
        Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn i)
      have hdjNat : FitnessABMPathN.degree n j ≠ 0 :=
        Nat.ne_of_gt (FitnessABMPathN.degree_pos n hn j)
      have hdi : (FitnessABMPathN.degree n i : Rat) ≠ 0 := by
        exact_mod_cast hdiNat
      have hdj : (FitnessABMPathN.degree n j : Rat) ≠ 0 := by
        exact_mod_cast hdjNat
      rw [show pathKernel params n i j =
          params.receptivity / (FitnessABMPathN.degree n i : Rat) by
        simp [pathKernel, hij, FitnessABMPathN.mem_neighbors_iff, hadj]]
      rw [show pathKernel params n j i =
          params.receptivity / (FitnessABMPathN.degree n j : Rat) by
        simp [pathKernel, Ne.symm hij, FitnessABMPathN.mem_neighbors_iff, hji]]
      unfold FitnessABMPathN.stationaryWeight
      field_simp [hwi, hdi, hdj]
    · have hji : ¬ FitnessABMPathN.pathAdj n i j := by
        intro h
        exact hadj ((pathAdj_symm i j).1 h)
      simp [pathKernel, hij, Ne.symm hij,
        FitnessABMPathN.mem_neighbors_iff, hadj, hji]
```

If exact simplifier normal forms differ, preserve this case structure and statement; only make local elaboration corrections.

- [ ] **Step 6: Implement stationary weights and mean preservation**

Add exactly these public interfaces:

```lean
theorem path_stationary_weights
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights (pathKernel params n)
      (FitnessABMPathN.stationaryWeight n) := by
  refine ⟨FitnessABMPathN.stationaryWeight_nonneg n hn,
    FitnessABMPathN.stationaryWeight_sum_one n hn, ?_⟩
  intro j
  calc
    (∑ i, FitnessABMPathN.stationaryWeight n i * pathKernel params n i j) =
        ∑ i, FitnessABMPathN.stationaryWeight n j * pathKernel params n j i := by
      apply Finset.sum_congr rfl
      intro i _
      exact pathKernel_detailed_balance params hvalid n hn i j
    _ = FitnessABMPathN.stationaryWeight n j *
        (∑ i, pathKernel params n j i) := by
      rw [Finset.mul_sum]
    _ = FitnessABMPathN.stationaryWeight n j := by
      rw [FitnessABMPathNParameters.pathKernel_rowsum params hvalid n hn j]
      ring

theorem mean_kernel_step
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) :
    FitnessABMPathN.mean n (applyKernel (pathKernel params n) x) =
      FitnessABMPathN.mean n x := by
  simpa [FitnessABMPathN.mean] using
    (weightedMean_apply (path_stationary_weights params hvalid n hn) x)

theorem mean_kernel_iterate
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (x : Beliefs n) (k : Nat) :
    FitnessABMPathN.mean n (kernelTrajectory (pathKernel params n) x k) =
      FitnessABMPathN.mean n x := by
  induction k with
  | zero => simp [FitnessABMPathN.mean]
  | succ k ih =>
      rw [show kernelTrajectory (pathKernel params n) x (Nat.succ k) =
          applyKernel (pathKernel params n)
            (kernelTrajectory (pathKernel params n) x k) by
        simpa [Nat.succ_eq_add_one] using
          kernelTrajectory_succ (pathKernel params n) x k]
      rw [mean_kernel_step params hvalid n hn, ih]
```

- [ ] **Step 7: Run Task 1 GREEN**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: PASS.

- [ ] **Step 8: Commit, push, and require exact-head proof success**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN stationary weights"
git push
```

Do not start Task 2 until the exact-head proof run succeeds.

---

### Task 2: Positive mass bounds and parameterized common column

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: Task 1 averaging witness, `FitnessABMPathN.pathAdj`, `degree_pos`, `degree_le_two`, `block`, `block_pos`; `FiniteConsensus.CommonColumnMass`, `kernelTrajectory`, `kernelPow_apply`.
- Produces: `beta`, `beta_pos`, `beta_lt_one`, `pathKernel_self_lower`, `pathKernel_adj_lower`, `delta`, `delta_pos`, `delta_lt_one`, `path_block_common_mass`.

- [ ] **Step 1: Add Task 2 RED consumers**

Append:

```lean
example : 0 < beta p34 := by
  exact beta_pos p34 (by norm_num [p34]) (by norm_num [p34])

example (n : Nat) (i : Fin n) :
    beta p34 ≤ pathKernel p34 n i i := by
  exact pathKernel_self_lower p34
    (by norm_num [p34]) (by norm_num [p34]) n i

example (n : Nat) (hn : 2 ≤ n) (i j : Fin n)
    (h : FitnessABMPathN.pathAdj n j i) :
    beta p34 ≤ pathKernel p34 n i j := by
  exact pathKernel_adj_lower p34
    (by norm_num [p34]) (by norm_num [p34]) n hn i j h

example (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass
      ((pathKernel p34 n) ^ FitnessABMPathN.block n)
      (delta p34 n) := by
  exact path_block_common_mass p34
    (by norm_num [p34, ResponseParameters.Valid])
    (by norm_num [p34]) (by norm_num [p34]) n hn
```

- [ ] **Step 2: Run and commit the intended Task 2 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: missing `beta`, lower-bound, `delta`, or common-column declarations while Task 1 consumers remain green.

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN common-mass RED"
git push
```

- [ ] **Step 3: Implement `beta` and its scalar bounds**

Add:

```lean
def beta (params : ResponseParameters) : Rat :=
  params.receptivity * (1 - params.receptivity) / 2

theorem beta_pos
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1) :
    0 < beta params := by
  unfold beta
  have h1 : 0 < 1 - params.receptivity := sub_pos.mpr ha1
  positivity

theorem beta_lt_one
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1) :
    beta params < 1 := by
  unfold beta
  have hαle : params.receptivity ≤ 1 := le_of_lt ha1
  have hsuble : 1 - params.receptivity ≤ 1 := by linarith
  have hprod : params.receptivity * (1 - params.receptivity) ≤ 1 :=
    mul_le_one₀ (le_of_lt ha0) hαle (sub_nonneg.mpr hαle) hsuble
  linarith
```

If `mul_le_one₀` has a different local argument order, prove `hprod` with `nlinarith [mul_nonneg (le_of_lt ha0) (sub_nonneg.mpr (le_of_lt ha1))]`; keep the theorem statements unchanged.

- [ ] **Step 4: Implement the self and edge lower bounds**

Add:

```lean
theorem pathKernel_self_lower
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (i : Fin n) :
    beta params ≤ pathKernel params n i i := by
  have hself : i ∉ FitnessABMPathN.neighbors n i := by
    intro h
    exact FitnessABMPathN.pathAdj_self i
      ((FitnessABMPathN.mem_neighbors_iff i i).mp h)
  simp [pathKernel, hself, beta]
  nlinarith

theorem pathKernel_adj_lower
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n)
    (h : FitnessABMPathN.pathAdj n j i) :
    beta params ≤ pathKernel params n i j := by
  classical
  have hmem : j ∈ FitnessABMPathN.neighbors n i :=
    (FitnessABMPathN.mem_neighbors_iff i j).mpr h
  have hne : i ≠ j := by
    intro hij
    subst j
    exact FitnessABMPathN.pathAdj_self i h
  have hpos := FitnessABMPathN.degree_pos n hn i
  have hle := FitnessABMPathN.degree_le_two n i
  have hd : FitnessABMPathN.degree n i = 1 ∨
      FitnessABMPathN.degree n i = 2 := by
    omega
  rcases hd with hd | hd <;>
    simp [pathKernel, beta, hne, hmem, hd] <;>
    nlinarith
```

- [ ] **Step 5: Implement `delta` and exact positive/interior bounds**

Add:

```lean
def delta (params : ResponseParameters) (n : Nat) : Rat :=
  beta params ^ FitnessABMPathN.block n

theorem delta_pos
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    0 < delta params n := by
  unfold delta
  exact pow_pos (beta_pos params ha0 ha1) _

theorem delta_lt_one
    (params : ResponseParameters)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    delta params n < 1 := by
  unfold delta
  exact pow_lt_one₀
    (le_of_lt (beta_pos params ha0 ha1))
    (beta_lt_one params ha0 ha1)
    (Nat.ne_of_gt (FitnessABMPathN.block_pos n hn))
```

- [ ] **Step 6: Implement the parameterized origin/reach/padding helpers with fixed signatures**

Add these private declarations, mirroring the already verified fixed PathN proof and replacing the fixed `1/4` mass by `beta params`:

```lean
private def originBasis {n : Nat} (z : Fin n) : Beliefs n :=
  fun j => if j = z then 1 else 0

private theorem kernelTrajectory_origin_nonneg
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (k : Nat) (i : Fin n) :
    0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) k i := by
  induction k generalizing i with
  | zero =>
      by_cases h : i = z <;> simp [kernelTrajectory, originBasis, h]
  | succ k ih =>
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      exact Finset.sum_nonneg fun j _ =>
        mul_nonneg
          (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
          (ih j)

private theorem left_reach_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (z : Fin n) (hz : z.val = 0)
    (k : Nat) (hk : k < n) :
    beta params ^ k ≤
      kernelTrajectory (pathKernel params n) (originBasis z) k ⟨k, hk⟩ := by
  induction k with
  | zero =>
      have hzero : (⟨0, hk⟩ : Fin n) = z := by
        apply Fin.ext
        simpa using hz.symm
      simp [kernelTrajectory, originBasis, hzero]
  | succ k ih =>
      let p : Fin n := ⟨k, by omega⟩
      let i : Fin n := ⟨k + 1, by omega⟩
      have hreach : beta params ^ k ≤
          kernelTrajectory (pathKernel params n) (originBasis z) k p :=
        ih (by omega)
      have hmass_nonneg :
          0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p :=
        kernelTrajectory_origin_nonneg params hvalid n hn z k p
      have hadj : FitnessABMPathN.pathAdj n p i := by
        left
        rfl
      have hkernel : beta params ≤ pathKernel params n i p :=
        pathKernel_adj_lower params ha0 ha1 n hn i p hadj
      change beta params ^ (k + 1) ≤
        kernelTrajectory (pathKernel params n) (originBasis z) (k + 1) i
      rw [kernelTrajectory_succ]
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      calc
        beta params ^ (k + 1) = beta params ^ k * beta params := by
          rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p *
            beta params :=
          mul_le_mul_of_nonneg_right hreach (le_of_lt (beta_pos params ha0 ha1))
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) k p *
            pathKernel params n i p :=
          mul_le_mul_of_nonneg_left hkernel hmass_nonneg
        _ = pathKernel params n i p *
            kernelTrajectory (pathKernel params n) (originBasis z) k p := by ring
        _ ≤ ∑ j : Fin n,
            pathKernel params n i j *
              kernelTrajectory (pathKernel params n) (originBasis z) k j :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
              (kernelTrajectory_origin_nonneg params hvalid n hn z k j))
            (Finset.mem_univ p)

private theorem self_pad_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) (z i : Fin n)
    (s t : Nat)
    (hstart : beta params ^ s ≤
      kernelTrajectory (pathKernel params n) (originBasis z) s i) :
    beta params ^ (s + t) ≤
      kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i := by
  induction t with
  | zero => simpa using hstart
  | succ t ih =>
      rw [Nat.add_succ, kernelTrajectory_succ]
      have hx : ∀ j,
          0 ≤ kernelTrajectory (pathKernel params n) (originBasis z) (s + t) j :=
        fun j => kernelTrajectory_origin_nonneg params hvalid n hn z (s + t) j
      simp only [applyKernel, Matrix.mulVec_apply, dotProduct]
      calc
        beta params ^ (s + t + 1) =
            beta params ^ (s + t) * beta params := by rw [pow_succ]
        _ ≤ kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i *
            beta params :=
          mul_le_mul_of_nonneg_right ih (le_of_lt (beta_pos params ha0 ha1))
        _ = beta params *
            kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i := by ring
        _ ≤ pathKernel params n i i *
            kernelTrajectory (pathKernel params n) (originBasis z) (s + t) i :=
          mul_le_mul_of_nonneg_right
            (pathKernel_self_lower params ha0 ha1 n i) (hx i)
        _ ≤ ∑ j : Fin n,
            pathKernel params n i j *
              kernelTrajectory (pathKernel params n) (originBasis z) (s + t) j :=
          Finset.single_le_sum
            (fun j _ => mul_nonneg
              (FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn i j)
              (hx j))
            (Finset.mem_univ i)

private theorem applyKernel_originBasis
    {n : Nat} (M : Kernel (Fin n)) (z i : Fin n) :
    applyKernel M (originBasis z) i = M i z := by
  classical
  simp [applyKernel, Matrix.mulVec_apply, dotProduct, originBasis]
```

- [ ] **Step 7: Implement the block common-column theorem**

Add:

```lean
theorem path_block_common_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass
      ((pathKernel params n) ^ FitnessABMPathN.block n)
      (delta params n) := by
  let z : Fin n := ⟨0, by omega⟩
  refine ⟨z, ?_⟩
  intro i
  have hi : i.val ≤ FitnessABMPathN.block n := by
    simp [FitnessABMPathN.block]
    omega
  have hreach : beta params ^ i.val ≤
      kernelTrajectory (pathKernel params n) (originBasis z) i.val i := by
    have hz : z.val = 0 := by rfl
    simpa using left_reach_mass params hvalid ha0 ha1 n hn z hz i.val i.isLt
  have hpad := self_pad_mass params hvalid ha0 ha1 n hn z i i.val
    (FitnessABMPathN.block n - i.val) hreach
  have htime : i.val + (FitnessABMPathN.block n - i.val) =
      FitnessABMPathN.block n := Nat.add_sub_of_le hi
  have hmass : delta params n ≤
      kernelTrajectory (pathKernel params n) (originBasis z)
        (FitnessABMPathN.block n) i := by
    simpa [delta, htime] using hpad
  have hp := congrFun
    (kernelPow_apply (pathKernel params n) (originBasis z)
      (FitnessABMPathN.block n)) i
  calc
    delta params n ≤
        kernelTrajectory (pathKernel params n) (originBasis z)
          (FitnessABMPathN.block n) i := hmass
    _ = applyKernel ((pathKernel params n) ^ FitnessABMPathN.block n)
        (originBasis z) i := hp.symm
    _ = ((pathKernel params n) ^ FitnessABMPathN.block n) i z :=
      applyKernel_originBasis
        ((pathKernel params n) ^ FitnessABMPathN.block n) z i
```

- [ ] **Step 8: Run Task 2 GREEN, commit, push, and require exact-head proof success**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean

git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN common mass"
git push
```

Do not start Task 3 until the exact-head proof run succeeds.

---

### Task 3: Bridge the executable trajectory and prove generic convergence

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: Task 1 stationary weights; Task 2 common-column mass; `FitnessABMPathNParameters.propagate_eq_kernel`, `allBroadcast_iterate`; `FiniteConsensus.block_contraction_tendsto`.
- Produces: `trajectory_eq_kernelTrajectory`, `trajectory_tendsto`.

- [ ] **Step 1: Add Task 3 RED consumers**

Append:

```lean
private def p23 : ResponseParameters := ⟨2/3, 1/4⟩
private def allBroadcast3 : Beliefs 3 := ![1/4, 3/4, 1]

example : ResponseParameters.Valid p23 := by
  norm_num [p23, ResponseParameters.Valid]

example : allBroadcast p23 3 allBroadcast3 := by
  intro i
  fin_cases i <;> norm_num [p23, allBroadcast3]

example (k : Nat) :
    ((beliefStep p23 3)^[k] allBroadcast3) =
      kernelTrajectory (pathKernel p23 3) allBroadcast3 k := by
  exact trajectory_eq_kernelTrajectory p23
    (by norm_num [p23, ResponseParameters.Valid]) 3 (by decide)
    allBroadcast3 (by
      intro i
      fin_cases i <;> norm_num [p23, allBroadcast3]) k

example (i : Fin 3) :
    Tendsto
      (fun k : Nat => ((((beliefStep p23 3)^[k] allBroadcast3) i : Rat) : Real))
      atTop
      (nhds (FitnessABMPathN.mean 3 allBroadcast3 : Real)) := by
  exact trajectory_tendsto p23
    (by norm_num [p23, ResponseParameters.Valid])
    (by norm_num [p23]) (by norm_num [p23])
    3 (by decide) allBroadcast3
    (by
      intro j
      fin_cases j <;> norm_num [p23, allBroadcast3]) i
```

- [ ] **Step 2: Run and commit the intended Task 3 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: missing trajectory bridge/final convergence declarations while Task 1–2 remain green.

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN convergence RED"
git push
```

- [ ] **Step 3: Implement the exact executable/kernel bridge**

Add:

```lean
theorem trajectory_eq_kernelTrajectory
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast params n x)
    (k : Nat) :
    (beliefStep params n)^[k] x =
      kernelTrajectory (pathKernel params n) x k := by
  induction k with
  | zero => simp [kernelTrajectory]
  | succ k ih =>
      have hk := FitnessABMPathNParameters.allBroadcast_iterate
        params hvalid n hn x hx k
      have hbridge :
          beliefStep params n ((beliefStep params n)^[k] x) =
            applyKernel (pathKernel params n) ((beliefStep params n)^[k] x) :=
        FitnessABMPathNParameters.propagate_eq_kernel
          params hvalid n hn ((beliefStep params n)^[k] x) hk
      rw [Function.iterate_succ_apply', hbridge, ih]
      simpa [Nat.succ_eq_add_one] using
        (kernelTrajectory_succ (pathKernel params n) x k).symm
```

- [ ] **Step 4: Implement the final convergence theorem**

Add:

```lean
theorem trajectory_tendsto
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast params n x)
    (i : Fin n) :
    Tendsto
      (fun k : Nat => ((((beliefStep params n)^[k] x) i : Rat) : Real))
      atTop
      (nhds (FitnessABMPathN.mean n x : Real)) := by
  let i0 : Fin n := ⟨0, by omega⟩
  letI : Nonempty (Fin n) := ⟨i0⟩
  have hkernel := block_contraction_tendsto
    (pathKernel params n) (FitnessABMPathN.stationaryWeight n)
    (pathKernel_averaging params hvalid n hn)
    (path_stationary_weights params hvalid n hn)
    (FitnessABMPathN.block n) (FitnessABMPathN.block_pos n hn)
    (delta params n) (delta_pos params ha0 ha1 n hn)
    (delta_lt_one params ha0 ha1 n hn)
    (path_block_common_mass params hvalid ha0 ha1 n hn) x i
  have htraj :
      (fun k : Nat => ((((beliefStep params n)^[k] x) i : Rat) : Real)) =
        (fun k : Nat => (kernelTrajectory (pathKernel params n) x k i : Real)) := by
    funext k
    rw [trajectory_eq_kernelTrajectory params hvalid n hn x hx k]
  rw [htraj]
  simpa [FitnessABMPathN.mean] using hkernel
```

- [ ] **Step 5: Run Task 3 GREEN, commit, push, and require exact-head proof success**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean

git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN convergence"
git push
```

Do not start Task 4 until the exact-head proof run succeeds.

---

### Task 4: Exact `alpha = 0` and `alpha = 1` boundary counterexamples

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: existing executable `beliefStep`, exact `Rat` reduction, Path2 topology.
- Produces: `zero_response_path2_iterate`, `one_response_path2_even` plus named one-step/two-cycle fixtures used by their proofs.

- [ ] **Step 1: Add concrete profiles and prove validity/all-broadcast**

```lean
private def zeroResponse : ResponseParameters := ⟨0, 0⟩
private def oneResponse : ResponseParameters := ⟨1, 0⟩
private def path2Split : Beliefs 2 := ![1, 0]
private def path2Swap : Beliefs 2 := ![0, 1]

example : ResponseParameters.Valid zeroResponse := by
  norm_num [zeroResponse, ResponseParameters.Valid]

example : ResponseParameters.Valid oneResponse := by
  norm_num [oneResponse, ResponseParameters.Valid]

example : allBroadcast zeroResponse 2 path2Split := by
  intro i
  fin_cases i <;> norm_num [zeroResponse, path2Split]

example : allBroadcast oneResponse 2 path2Split := by
  intro i
  fin_cases i <;> norm_num [oneResponse, path2Split]
```

- [ ] **Step 2: Prove `alpha = 0` identity at every finite iterate**

```lean
theorem zero_response_path2_step :
    beliefStep zeroResponse 2 path2Split = path2Split := by
  decide_cbv

theorem zero_response_path2_iterate (k : Nat) :
    (beliefStep zeroResponse 2)^[k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Function.iterate_succ_apply', ih]
      exact zero_response_path2_step
```

If `decide_cbv` cannot close the named one-step equality within the normal budget, unfold only the existing executable definitions for this concrete fixture. Do not use `native_decide` and do not raise global resource limits.

- [ ] **Step 3: Prove the exact `alpha = 1` two-cycle and all even iterates**

```lean
theorem one_response_path2_step :
    beliefStep oneResponse 2 path2Split = path2Swap := by
  decide_cbv

theorem one_response_path2_back :
    beliefStep oneResponse 2 path2Swap = path2Split := by
  decide_cbv

theorem one_response_path2_two :
    (beliefStep oneResponse 2)^[2] path2Split = path2Split := by
  simp [Function.iterate_succ_apply', one_response_path2_step,
    one_response_path2_back]

theorem one_response_path2_even (k : Nat) :
    (beliefStep oneResponse 2)^[2 * k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Nat.mul_succ]
      rw [Function.iterate_add_apply
        (beliefStep oneResponse 2) (2 * k) 2 path2Split]
      rw [one_response_path2_two, ih]
```

If the local orientation of `Function.iterate_add_apply` elaborates to the equivalent reverse composition, swap its two Nat arguments; preserve the theorem statement and use only this standard iterate lemma plus the named two-step cycle.

- [ ] **Step 4: Add axiom reports**

```lean
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto
#print axioms zero_response_path2_iterate
#print axioms one_response_path2_even
```

- [ ] **Step 5: Run Task 4, inspect axiom output, commit, push, and require exact-head proof success**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean

git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): verify parameterized convergence boundaries"
git push
```

Accept only exact equality/cycle evidence and theorem reports within the existing trust allowlist.

---

### Task 5: Additive permanent gate, final audit, PR, and integration

**Files:**
- Modify: `tools/check_fitness_abm_pathn.sh`
- No workflow-file changes expected.

**Interfaces:**
- Consumes: completed core/test module and existing PathN permanent gate.
- Produces: bounded build/test/trust enforcement for parameterized convergence while preserving every previous gate.

- [ ] **Step 1: Extend temp-log cleanup and source audit additively**

Add:

```bash
parameter_convergence_log="$(mktemp)"
```

and include it in the existing `trap`. Add these paths to the existing source audit:

```text
NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean
NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Do not remove any current FiniteConsensus, fixed PathN, parameter, exposure, Path4, or Path5 source audit.

- [ ] **Step 2: Extend bounded module build coverage**

Add this module to the existing `pathn_module` loop:

```text
NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
```

Keep the existing command shape:

```bash
"$pathn_time" -f "$pathn_module elapsed=%e s peak_rss=%M KiB" \
  timeout --kill-after=10s 240s lake build "$pathn_module"
```

- [ ] **Step 3: Add bounded test execution**

```bash
"$pathn_time" -f 'FitnessABMPathNParameterConvergence tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean \
  2>&1 | tee "$parameter_convergence_log"
```

- [ ] **Step 4: Add trust requirements**

```bash
python3 tools/audit_fitness_trust.py log "$parameter_convergence_log" \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto \
  --require zero_response_path2_iterate \
  --require one_response_path2_even
```

Preserve every existing `--require` block.

- [ ] **Step 5: Run the permanent PathN gate**

```bash
timeout --kill-after=10s 240s bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS with all old and new sections present.

- [ ] **Step 6: Commit and push permanent gate**

```bash
git add tools/check_fitness_abm_pathn.sh
git commit -m "test(ci): gate parameterized PathN convergence"
git push
```

- [ ] **Step 7: Require final exact-head proof and World Studio success**

On the exact current head verify:

```text
proof workflow
  Select proof event                 success
  Python tests                       success
  Lean proof                         success
    Build Lean library               success
    BB path-four convergence         success
    BB finite-path convergence       success
    Fitness attachment/replay/scope  success
    Fitness distribution/trust       success
    Lean theorem tests               success
    Narrative story theorem tests    success
    Narrative testimony tests        success

World Studio
  verify                             success
```

Do not substitute a previous-head run or the focused gate for this result.

- [ ] **Step 8: Perform the final scope audit**

Compare against `proof/narrative-dynamics-v0@8115c3862700114fb91e495f32cecf9e765d7455`. Require only:

```text
+ NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean
+ NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
~ tools/check_fitness_abm_pathn.sh
+ docs/superpowers/specs/2026-09-16-bb-parameterized-pathn-convergence-v1-design.md
+ docs/superpowers/plans/2026-09-16-bb-parameterized-pathn-convergence-v1.md
```

No `NetworkPropagation`, `FiniteConsensus`, fixed PathN, parameterized-response, exposure model, or workflow mutation is expected.

- [ ] **Step 9: Create the PR**

Target: `proof/narrative-dynamics-v0`.

Title:

```text
feat(lean): prove parameterized finite-path convergence
```

PR body must record:

```text
Closes #92.
Hypotheses: n >= 2, params.Valid, 0 < alpha < 1, initial all-broadcast.
Limit: existing FitnessABMPathN.mean.
Stationary degree weights are independent of alpha.
beta = alpha(1-alpha)/2 and delta = beta^(n-1) are conservative proof bounds, not optimal-rate claims.
Exact alpha=0 identity and alpha=1 Path2 cycle are retained as boundary evidence.
No arbitrary-graph, exposure-dependent, or floating-point convergence claim.
Include exact-head proof and World Studio run/job IDs.
```

- [ ] **Step 10: Review and merge only the verified exact head**

Review every changed file and every review thread. Fix Critical/Important findings before merge. After final PR-event proof and World Studio are complete, fetch the PR again, verify the head SHA did not move, and merge with `expected_head_sha=<verified head>` using merge method `merge`.

- [ ] **Step 11: Post-merge verification and issue closure**

Confirm `proof/narrative-dynamics-v0` points to the merge commit. Require post-merge push proof and World Studio success on that exact merge commit before closing #92 as `completed`. Final issue evidence must include merge SHA, proof run/job IDs, World Studio run/job ID, theorem hypotheses, exact limit, endpoint counterexamples, and remaining unproved scope.
