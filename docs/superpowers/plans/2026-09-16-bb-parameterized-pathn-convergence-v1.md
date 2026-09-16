# BB Parameterized PathN Convergence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove that for every finite path `n >= 2`, every valid exact-rational response parameter with `0 < receptivity < 1`, and every initial state in the parameterized all-broadcast region, the actual executable belief trajectory converges coordinatewise to the existing degree-weighted PathN mean.

**Architecture:** Add a separate convergence module on top of `FitnessABMPathNParameters`; do not change runtime propagation semantics or `FiniteConsensus`. Reuse the public parameterized kernel, assemble an averaging-kernel witness from public nonnegativity/row-sum theorems, prove degree-normalized stationary weights, obtain a conservative positive step mass `beta = alpha * (1 - alpha) / 2`, lift it to a block common-column mass over `n - 1` steps, then apply `FiniteConsensus.block_contraction_tendsto` and rewrite the proof-kernel trajectory back to the real executable `beliefStep` trajectory.

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

- Create `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean` — all new generic parameterized convergence lemmas and final theorem.
- Create `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean` — RED/GREEN consumers, exact boundary fixtures, and axiom reports.
- Modify `tools/check_fitness_abm_pathn.sh` — add bounded source/build/test/trust coverage for the new module without replacing any old gate.
- Do not modify `.github/workflows/proof.yml`; it already invokes the PathN gate on non-doc changes.
- Do not add a root import unless an actual build failure proves one is required.

---

### Task 1: Parameterized stationary weights and mean preservation

**Files:**
- Create: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Create: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: `FitnessABMPathNParameters.pathKernel`, `pathKernel_nonneg`, `pathKernel_rowsum`; `FitnessABMPathN.stationaryWeight`, `weightSum_pos`, `mean`; `FiniteConsensus.StationaryWeights`, `weightedMean_apply`.
- Produces:
  - `pathKernel_averaging`
  - `pathKernel_detailed_balance`
  - `path_stationary_weights`
  - `mean_kernel_step`
  - `mean_kernel_iterate`

- [ ] **Step 1: Write the Task 1 RED consumers**

Create the test file with the new module import and consumers equivalent to:

```lean
import NarrativeDynamics.Core.FitnessABMPathNParameterConvergence

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FitnessABMPathNParameters
open NarrativeDynamics.FitnessABMPathNParameterConvergence

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

- [ ] **Step 2: Run the focused RED and verify the failure is the missing convergence module/declarations**

Run:

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMPathNParameters
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: existing parameter module builds; the consumer fails because `NarrativeDynamics.Core.FitnessABMPathNParameterConvergence` or its stationary declarations do not exist. A dependency/build-harness failure is not an accepted RED.

- [ ] **Step 3: Commit the valid RED**

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN convergence stationary RED"
git push
```

Record the exact RED run/job in issue #92.

- [ ] **Step 4: Implement the minimal Task 1 core**

Create the convergence module in namespace:

```lean
namespace NarrativeDynamics.FitnessABMPathNParameterConvergence
```

Add an internal averaging witness assembled only from public parameter theorems:

```lean
private theorem pathKernel_averaging
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    AveragingKernel (FitnessABMPathNParameters.pathKernel params n) := by
  exact ⟨
    FitnessABMPathNParameters.pathKernel_nonneg params hvalid n hn,
    FitnessABMPathNParameters.pathKernel_rowsum params hvalid n hn
  ⟩
```

Do **not** expose or modify the existing private theorem in `FitnessABMPathNParameters`.

Prove a local symmetry fact for `FitnessABMPathN.pathAdj`, then prove:

```lean
theorem pathKernel_detailed_balance
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) (i j : Fin n) :
    FitnessABMPathN.stationaryWeight n i *
        FitnessABMPathNParameters.pathKernel params n i j =
      FitnessABMPathN.stationaryWeight n j *
        FitnessABMPathNParameters.pathKernel params n j i
```

Use cases `i = j`, adjacent, nonadjacent. In the adjacent case cancel degree factors with `FitnessABMPathN.degree_pos` and `FitnessABMPathN.weightSum_pos`; the equality should reduce to the same `alpha / weightSum` on both sides.

Then derive:

```lean
theorem path_stationary_weights
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n) :
    StationaryWeights
      (FitnessABMPathNParameters.pathKernel params n)
      (FitnessABMPathN.stationaryWeight n)
```

and:

```lean
theorem mean_kernel_step ...
theorem mean_kernel_iterate ...
```

where both preserve exactly `FitnessABMPathN.mean n x`.

- [ ] **Step 5: Run Task 1 GREEN**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: PASS.

- [ ] **Step 6: Commit and exact-head verify Task 1 GREEN**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN stationary weights"
git push
```

Require exact-head proof success before Task 2.

---

### Task 2: Positive mass bounds and parameterized common column

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: Task 1 averaging kernel, `FitnessABMPathN.pathAdj`, `degree_pos`, `degree_le_two`; `FiniteConsensus.CommonColumnMass`, `kernelTrajectory`, `kernelPow_apply`.
- Produces:
  - `beta`
  - `beta_pos`
  - `beta_lt_one`
  - `pathKernel_self_lower`
  - `pathKernel_adj_lower`
  - `delta`
  - `delta_pos`
  - `delta_lt_one`
  - `path_block_common_mass`

- [ ] **Step 1: Add Task 2 RED consumers**

Append exact consumers with a concrete strict-interior parameter:

```lean
example : 0 < beta p34 := by
  exact beta_pos p34 (by norm_num [p34]) (by norm_num [p34])

example (n : Nat) (hn : 2 ≤ n) (i : Fin n) :
    beta p34 ≤ pathKernel p34 n i i := by
  exact pathKernel_self_lower p34
    (by norm_num [p34]) (by norm_num [p34]) n hn i

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

- [ ] **Step 2: Run Task 2 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: failure on missing `beta`/lower-bound/common-column declarations while Task 1 consumers continue elaborating.

- [ ] **Step 3: Commit the Task 2 RED**

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN common-mass RED"
git push
```

- [ ] **Step 4: Implement `beta` and local lower bounds**

Add:

```lean
def beta (params : ResponseParameters) : Rat :=
  params.receptivity * (1 - params.receptivity) / 2
```

Prove under `ha0 : 0 < params.receptivity` and `ha1 : params.receptivity < 1`:

```lean
theorem beta_pos ... : 0 < beta params
theorem beta_lt_one ... : beta params < 1
```

Then prove:

```lean
theorem pathKernel_self_lower ... :
  beta params ≤ pathKernel params n i i
```

using `pathAdj_self` so the neighbor contribution at `(i,i)` is zero and the self mass is `1-alpha`.

Prove:

```lean
theorem pathKernel_adj_lower ...
    (h : FitnessABMPathN.pathAdj n j i) :
    beta params ≤ pathKernel params n i j
```

by deriving `degree n i = 1 ∨ degree n i = 2` from the public positive/upper degree bounds; for an edge `i != j`, the mass is `alpha / degree(i)`, and `1-alpha < 1` supplies the conservative polynomial inequality.

- [ ] **Step 5: Implement `delta` and the common-column construction**

Define:

```lean
def delta (params : ResponseParameters) (n : Nat) : Rat :=
  beta params ^ FitnessABMPathN.block n
```

Prove `delta_pos` and `delta_lt_one` using `FitnessABMPathN.block_pos`, `beta_pos`, `beta_lt_one`.

Add private helpers adapted from the fixed PathN proof structure:

```lean
private def originBasis ...
private theorem kernelTrajectory_origin_nonneg ...
private theorem left_reach_mass ...
private theorem self_pad_mass ...
private theorem applyKernel_originBasis ...
```

The only substantive change from the fixed proof is replacing every `1/4` lower bound by `beta params` and using the parameterized averaging/nonnegativity facts.

Finish with:

```lean
theorem path_block_common_mass
    (params : ResponseParameters) (hvalid : params.Valid)
    (ha0 : 0 < params.receptivity)
    (ha1 : params.receptivity < 1)
    (n : Nat) (hn : 2 ≤ n) :
    CommonColumnMass
      ((pathKernel params n) ^ FitnessABMPathN.block n)
      (delta params n)
```

- [ ] **Step 6: Run Task 2 GREEN**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: PASS.

- [ ] **Step 7: Commit and exact-head verify Task 2 GREEN**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN common mass"
git push
```

Require exact-head proof success before Task 3.

---

### Task 3: Bridge the real executable trajectory and prove generic convergence

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: Task 1 stationary weights; Task 2 common-column mass; `FitnessABMPathNParameters.propagate_eq_kernel`, `allBroadcast_iterate`; `FiniteConsensus.block_contraction_tendsto`.
- Produces:
  - `trajectory_eq_kernelTrajectory`
  - `trajectory_tendsto`

- [ ] **Step 1: Add the final theorem RED consumers**

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
      Filter.atTop
      (nhds (FitnessABMPathN.mean 3 allBroadcast3 : Real)) := by
  exact trajectory_tendsto p23
    (by norm_num [p23, ResponseParameters.Valid])
    (by norm_num [p23]) (by norm_num [p23])
    3 (by decide) allBroadcast3
    (by
      intro j
      fin_cases j <;> norm_num [p23, allBroadcast3]) i
```

- [ ] **Step 2: Run Task 3 RED**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: missing trajectory bridge/final convergence theorem; Task 1–2 consumers remain green.

- [ ] **Step 3: Commit Task 3 RED**

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): add parameterized PathN convergence RED"
git push
```

- [ ] **Step 4: Implement the executable/kernel finite-iterate bridge**

Prove by induction:

```lean
theorem trajectory_eq_kernelTrajectory
    (params : ResponseParameters) (hvalid : params.Valid)
    (n : Nat) (hn : 2 ≤ n)
    (x : Beliefs n) (hx : allBroadcast params n x)
    (k : Nat) :
    (beliefStep params n)^[k] x =
      kernelTrajectory (pathKernel params n) x k
```

At successor step use `FitnessABMPathNParameters.allBroadcast_iterate` to recover the all-broadcast premise for the current executable iterate, then use `propagate_eq_kernel` for the actual next step.

- [ ] **Step 5: Implement the final convergence theorem**

Apply:

```lean
FiniteConsensus.block_contraction_tendsto
```

with:

```text
K     = pathKernel params n
π     = FitnessABMPathN.stationaryWeight n
b     = FitnessABMPathN.block n
δ     = delta params n
```

and the Task 1/2 witnesses. Rewrite the kernel trajectory to the executable trajectory using `trajectory_eq_kernelTrajectory`.

Public theorem:

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
      Filter.atTop
      (nhds (FitnessABMPathN.mean n x : Real))
```

Do not define a new mean.

- [ ] **Step 6: Run Task 3 GREEN**

```bash
timeout --kill-after=10s 240s lake build \
  NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: PASS.

- [ ] **Step 7: Commit and exact-head verify Task 3 GREEN**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "feat(lean): prove parameterized PathN convergence"
git push
```

Require exact-head proof success before boundary work.

---

### Task 4: Kernel-checked `alpha = 0` and `alpha = 1` boundary counterexamples

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean`

**Interfaces:**
- Consumes: existing executable `beliefStep`, exact `Rat` reduction, Path2 topology.
- Produces: retained exact fixtures documenting why the final theorem requires strict `0 < alpha < 1`.

- [ ] **Step 1: Add the boundary fixtures**

Add:

```lean
private def zeroResponse : ResponseParameters := ⟨0, 0⟩
private def oneResponse : ResponseParameters := ⟨1, 0⟩
private def path2Split : Beliefs 2 := ![1, 0]
```

Prove validity and all-broadcast for both concrete profiles.

For `alpha = 0`, prove a generic concrete-state iterate identity by induction:

```lean
theorem zero_response_path2_iterate (k : Nat) :
    (beliefStep zeroResponse 2)^[k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      rw [Function.iterate_succ_apply', ih]
      decide_cbv
```

If `decide_cbv` cannot close the one-step equality within the normal resource budget, replace only that final line with explicit unfolding/rewrite of the existing executable definitions; do not use `native_decide` or increase global limits.

For `alpha = 1`, prove the exact two-cycle:

```lean
example : beliefStep oneResponse 2 path2Split = ![0, 1] := by
  decide_cbv

example :
    beliefStep oneResponse 2 (beliefStep oneResponse 2 path2Split) = path2Split := by
  decide_cbv
```

and retain an induction theorem for even iterations:

```lean
theorem one_response_path2_even (k : Nat) :
    (beliefStep oneResponse 2)^[2 * k] path2Split = path2Split := by
  induction k with
  | zero => simp
  | succ k ih =>
      -- rewrite two successor applications and use the exact two-cycle
      ...
```

Implement the omitted proof with explicit `Function.iterate_succ_apply'` rewrites and the named two-cycle equality; do not add a new production transition.

- [ ] **Step 2: Run the boundary fixtures**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Expected: PASS, with exact identities/cycle; no floating comparison or tolerance.

- [ ] **Step 3: Add theorem axiom reports**

Append:

```lean
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass
#print axioms NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto
#print axioms zero_response_path2_iterate
#print axioms one_response_path2_even
```

Run the same focused Lean command and inspect that no `sorryAx`, native-evaluation axiom, or new user axiom appears.

- [ ] **Step 4: Commit and exact-head verify Task 4**

```bash
git add NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
git commit -m "test(lean): verify parameterized convergence boundaries"
git push
```

Require exact-head proof success before permanent gate integration.

---

### Task 5: Additive permanent gate, final audit, PR, and integration

**Files:**
- Modify: `tools/check_fitness_abm_pathn.sh`
- No workflow-file changes expected.

**Interfaces:**
- Consumes: completed core/test module and existing PathN permanent gate.
- Produces: bounded build/test/trust enforcement for the new convergence theorem while preserving all previous gates.

- [ ] **Step 1: Extend the source audit and temp-log cleanup additively**

Add a `parameter_convergence_log="$(mktemp)"` and include it in the existing trap. Add both new files to the `audit_fitness_trust.py source` invocation:

```text
NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean
NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
```

Do not remove any current FiniteConsensus, fixed PathN, parameter, exposure, Path4, or Path5 source audit.

- [ ] **Step 2: Extend bounded module build coverage**

Add:

```text
NarrativeDynamics.Core.FitnessABMPathNParameterConvergence
```

to the existing `pathn_module` loop. Retain the same `timeout --kill-after=10s 240s lake build` convention.

- [ ] **Step 3: Add the new bounded consumer execution**

Add:

```bash
"$pathn_time" -f 'FitnessABMPathNParameterConvergence tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean \
  2>&1 | tee "$parameter_convergence_log"
```

- [ ] **Step 4: Add trust requirements without replacing existing requirements**

Require at minimum:

```bash
python3 tools/audit_fitness_trust.py log "$parameter_convergence_log" \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_stationary_weights \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.path_block_common_mass \
  --require NarrativeDynamics.FitnessABMPathNParameterConvergence.trajectory_tendsto \
  --require zero_response_path2_iterate \
  --require one_response_path2_even
```

- [ ] **Step 5: Run the permanent PathN gate locally/focused**

```bash
timeout --kill-after=10s 240s bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS with every pre-existing and new PathN section retained.

- [ ] **Step 6: Commit the permanent gate**

```bash
git add tools/check_fitness_abm_pathn.sh
git commit -m "test(ci): gate parameterized PathN convergence"
git push
```

- [ ] **Step 7: Verify exact-head proof and World Studio**

On the exact current head require:

```text
proof workflow:
  Select proof event       success
  Python tests             success
  Lean proof               success
    Build Lean library             success
    BB path-four convergence       success
    BB finite-path convergence     success
    all replay/scope/distribution  success
    Lean/Story/Testimony tests     success

World Studio:
  verify                   success
```

Do not infer success from a previous head or from only the focused gate.

- [ ] **Step 8: Perform final scope audit**

Compare against `proof/narrative-dynamics-v0@8115c3862700114fb91e495f32cecf9e765d7455` and require the intended scope only:

```text
+ NarrativeDynamics/Core/FitnessABMPathNParameterConvergence.lean
+ NarrativeDynamics/Tests/FitnessABMPathNParameterConvergence.lean
~ tools/check_fitness_abm_pathn.sh
+ docs/superpowers/specs/2026-09-16-bb-parameterized-pathn-convergence-v1-design.md
+ docs/superpowers/plans/2026-09-16-bb-parameterized-pathn-convergence-v1.md
```

No `NetworkPropagation`, `FiniteConsensus`, fixed PathN, parameterized-response, exposure model, or workflow mutation is expected.

- [ ] **Step 9: Create the PR against `proof/narrative-dynamics-v0`**

PR title:

```text
feat(lean): prove parameterized finite-path convergence
```

PR body must state:

- closes #92;
- exact theorem hypotheses `n >= 2`, `params.Valid`, `0 < alpha < 1`, initial all-broadcast;
- exact limit `FitnessABMPathN.mean`;
- degree stationary distribution is independent of `alpha`;
- `beta = alpha(1-alpha)/2`, `delta = beta^(n-1)` are conservative proof bounds, not claimed optimal rates;
- exact `alpha=0/1` counterexamples;
- no arbitrary-graph/exposure-dependent/float convergence claim;
- exact-head proof and World Studio run IDs.

- [ ] **Step 10: Review and merge only the verified exact head**

Review changed files and threads. Fix any Critical/Important finding before merge. Once final PR-event proof and World Studio are complete and the PR head SHA is unchanged, merge using an expected-head SHA guard and preserve commits with merge method `merge`.

- [ ] **Step 11: Post-merge verification and issue closure**

Confirm `proof/narrative-dynamics-v0` points at the merge commit. Require post-merge push proof and World Studio success on that exact merge commit before closing #92 as `completed`. Record merge SHA, proof run/job IDs, World Studio run/job ID, theorem boundary, and remaining unproved scope in the final issue comment.
