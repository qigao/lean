# BB Response Parameters V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate exact-rational finite-path response-parameter module with receptivity `α`, inclusive threshold `τ`, exact specialization to the current `(1/2,1/2)` model, exposure-independent projected beliefs, and all-broadcast invariance.

**Architecture:** Reuse `FitnessABMPathN.pathAdj`, neighbor/degree helpers, `NetworkPropagation.propagate`, and the existing proof-only `FiniteConsensus.Kernel` vocabulary. The new module is an adapter/proof layer only: existing `NetworkPropagation` and fixed `FitnessABMPathN` definitions remain unchanged and are regression authorities.

**Tech Stack:** Lean 4.32.0, mathlib, exact `Rat`, existing `NetworkPropagation`, `FitnessABMPathN`, `FiniteConsensus`, bounded trust audit and GitHub proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-response-parameters-v1-design.md`

## Global Constraints

- `ResponseParameters` has rational `receptivity` and `threshold`; `Valid` means each lies in `[0,1]`.
- Parameterized executable belief step must call existing `NetworkPropagation.propagate`.
- Threshold semantics remain inclusive: `threshold ≤ belief` broadcasts.
- Existing `NetworkPropagation`, `FitnessABMPathN`, `FiniteConsensus`, Path4 and Path5 semantics are unchanged.
- Exact half specialization `(α,τ)=(1/2,1/2)` is theorem-level compatibility, not duplicated implementation.
- Projected beliefs remain independent of exposure counters in this baseline parameterized model.
- V1 does not claim arbitrary-parameter convergence/rate or activation-time formulas.
- Direct Lean commands use `timeout --kill-after=10s 240s`.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, or unbounded proof-resource settings.

## File Map

| File | Responsibility |
| --- | --- |
| `NarrativeDynamics/Core/FitnessABMPathNParameters.lean` | parameters, real propagate-backed finite-path operator, parameter kernel, compatibility/invariance theorems |
| `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean` | RED/GREEN API consumers, inclusive-threshold fixture, theorem axiom reports |
| `tools/check_fitness_abm_pathn.sh` | bounded source/build/test/trust coverage for new module while retaining existing PathN checks |
| `.github/workflows/proof.yml` | existing `BB finite-path convergence` step continues to own the gate; no new workflow |
| `docs/superpowers/plans/2026-09-16-bb-response-parameters-v1.md` | execution record |

---

### Task 1: Parameter surface and real executable step

**Files:**
- Create RED: `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`
- Create GREEN: `NarrativeDynamics/Core/FitnessABMPathNParameters.lean`

**Interfaces:**

```lean
namespace NarrativeDynamics.FitnessABMPathNParameters

structure ResponseParameters where
  receptivity : Rat
  threshold : Rat

namespace ResponseParameters

def Valid (p : ResponseParameters) : Prop :=
  0 ≤ p.receptivity ∧ p.receptivity ≤ 1 ∧
  0 ≤ p.threshold ∧ p.threshold ≤ 1
end ResponseParameters

abbrev Beliefs := FitnessABMPathN.Beliefs

def population (params : ResponseParameters)
    (n : Nat) (x : Beliefs n) (e : Fin n → Nat) : NetworkPropagation.Population n

def project (n : Nat) (p : NetworkPropagation.Population n) : Beliefs n

def beliefStep (params : ResponseParameters) (n : Nat) (x : Beliefs n) : Beliefs n

def allBroadcast (params : ResponseParameters) (n : Nat) (x : Beliefs n) : Prop

def half : ResponseParameters := ⟨1/2, 1/2⟩
```

- [ ] **Step 1: Write RED consumer before core module exists.**

The test imports `NarrativeDynamics.Core.FitnessABMPathNParameters` and requires:

```lean
open NarrativeDynamics NetworkPropagation
open NarrativeDynamics.FitnessABMPathNParameters

example : ResponseParameters.Valid half := by norm_num [half, ResponseParameters.Valid]

example (n : Nat) (x : Beliefs n) :
    beliefStep half n x =
      project n (propagate (FitnessABMPathN.pathAdj n)
        (population half n x (fun _ => 0))) := rfl
```

Include one concrete equality-threshold consumer on `Fin 2`: a vertex with belief `1/2` and threshold `1/2` must be selected as broadcasting by the underlying population/profile semantics.

- [ ] **Step 2: Run bounded RED.**

Run:
```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
```
Expected: missing module/declarations only.

- [ ] **Step 3: Implement minimal structures/definitions.**

`population` must construct constant profiles `⟨params.receptivity, params.threshold⟩` and states `⟨x i, e i⟩`. `beliefStep` must call the existing `NetworkPropagation.propagate`; do not copy next-agent arithmetic.

- [ ] **Step 4: Run GREEN.**

Run the same bounded test and `lake build NarrativeDynamics.Core.FitnessABMPathNParameters`.
Expected: exit 0.

- [ ] **Step 5: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameters.lean NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
git commit -m "feat(lean): add finite-path response parameters"
```

---

### Task 2: Exact specialization to the fixed half model

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameters.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`

**Interfaces:**

```lean
theorem population_half ... :
  population half n x e = FitnessABMPathN.population n x e

theorem beliefStep_half ... :
  beliefStep half n x = FitnessABMPathN.beliefStep n x

theorem allBroadcast_half ... :
  allBroadcast half n x ↔ FitnessABMPathN.allBroadcast n x
```

- [ ] **Step 1: Add RED exact-compatibility consumers.**

Require all three statements above for arbitrary `n`, beliefs, and exposure function. Do not test only one concrete vector.

- [ ] **Step 2: Run RED.**

Expected: missing theorem declarations; Task 1 consumers still pass.

- [ ] **Step 3: Prove by unfolding/extensionality/simp without modifying fixed definitions.**

`population_half` should be definitional/extensional. `beliefStep_half` should rewrite the population equality through the same `propagate`. `allBroadcast_half` should reduce to the fixed threshold definition.

- [ ] **Step 4: Run GREEN and fixed PathN compatibility consumer.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathN.lean
```

- [ ] **Step 5: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameters.lean NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
git commit -m "proof(lean): specialize response parameters to half"
```

---

### Task 3: Exposure independence and parameter kernel

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameters.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`

**Interfaces:**

```lean
theorem propagate_independent_exposures
    (params : ResponseParameters) (n : Nat) (x : Beliefs n)
    (e : Fin n → Nat) :
  project n (NetworkPropagation.propagate (FitnessABMPathN.pathAdj n)
    (population params n x e)) = beliefStep params n x

def pathKernel (params : ResponseParameters) (n : Nat) : FiniteConsensus.Kernel (Fin n)

theorem pathKernel_half ... :
  pathKernel half n = FitnessABMPathN.pathKernel n
```

Kernel definition:

```lean
fun i j =>
  (if i = j then 1 - params.receptivity else 0) +
  (if j ∈ FitnessABMPathN.neighbors n i then
     params.receptivity / (FitnessABMPathN.degree n i : Rat)
   else 0)
```

- [ ] **Step 1: Add RED exposure consumer with two explicit different exposure functions and the generic theorem declaration.**

Also require `pathKernel_half` for arbitrary `n`.

- [ ] **Step 2: Run RED.**

Expected: missing theorem/kernel declarations.

- [ ] **Step 3: Generalize the fixed-model exposure-independence proof structurally.**

Do not assume `params.Valid`; belief projection independence follows because current `broadcasting` and belief arithmetic ignore prior exposure counts. Keep theorem scoped to projected beliefs, not whole populations.

- [ ] **Step 4: Implement `pathKernel` and exact half equality.**

Reuse `FitnessABMPathN.neighbors`, `degree`, and existing kernel vocabulary. No convergence statement in this task.

- [ ] **Step 5: Run GREEN and trust source audit.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
python3 tools/audit_fitness_trust.py source NarrativeDynamics/Core/FitnessABMPathNParameters.lean NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
```

- [ ] **Step 6: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameters.lean NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
git commit -m "proof(lean): parameterize path response kernel"
```

---

### Task 4: Valid kernel, real-step bridge, and all-broadcast invariance

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNParameters.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`

**Interfaces:**

For `params.Valid`, `2 ≤ n`:

```lean
theorem pathKernel_nonneg ... : 0 ≤ pathKernel params n i j

theorem pathKernel_rowsum ... :
  ∑ j, pathKernel params n i j = 1

theorem propagate_eq_kernel
    (hvalid : params.Valid) (hn : 2 ≤ n)
    (hall : allBroadcast params n x) :
  beliefStep params n x = FiniteConsensus.applyKernel (pathKernel params n) x

theorem allBroadcast_step ... :
  allBroadcast params n x → allBroadcast params n (beliefStep params n x)

theorem allBroadcast_iterate ... :
  allBroadcast params n x →
  ∀ k, allBroadcast params n ((beliefStep params n)^[k] x)
```

- [ ] **Step 1: Add RED theorem consumers including threshold equality.**

Use one concrete valid parameter tuple with `α = 3/4`, `τ = 1/3` and an all-broadcast vector containing exactly `τ` at one coordinate. Require one-step region preservation.

- [ ] **Step 2: Run RED.**

Expected: missing kernel laws/bridge/invariance only.

- [ ] **Step 3: Prove nonnegativity and row sum from `Valid`, path degree positivity for `n≥2`, and existing finite-set neighbor facts.**

Do not introduce a new matrix/graph abstraction.

- [ ] **Step 4: Prove real `propagate` equals `applyKernel` inside the all-broadcast region.**

Follow the fixed `FitnessABMPathN.propagate_eq_kernel` proof structure but keep `params.receptivity` symbolic and derive broadcasting from `params.threshold ≤ x i`.

- [ ] **Step 5: Prove one-step and finite-iterate all-broadcast invariance.**

Use convex row weights: if every coordinate is between `τ` and `1`, a row-stochastic nonnegative combination remains between those bounds. Do not state convergence.

- [ ] **Step 6: Run GREEN.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
```

- [ ] **Step 7: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNParameters.lean NarrativeDynamics/Tests/FitnessABMPathNParameters.lean
git commit -m "proof(lean): preserve parameterized all-broadcast region"
```

---

### Task 5: Bounded gate, exact-head CI, review

**Files:**
- Modify: `tools/check_fitness_abm_pathn.sh`
- No new workflow file should be necessary unless current path filters unexpectedly exclude the new files.

**Interfaces:**
- Existing `bash tools/check_fitness_abm_pathn.sh` becomes the permanent owner of parameter module source/build/test/trust reports while retaining all existing fixed PathN/FiniteConsensus requirements.

- [ ] **Step 1: Extend source audit/build/test logs.**

Add new core/test files to `audit_fitness_trust.py source`; add bounded `lake build NarrativeDynamics.Core.FitnessABMPathNParameters`; add a separate temp log for `NarrativeDynamics/Tests/FitnessABMPathNParameters.lean`.

- [ ] **Step 2: Require named trust reports.**

At minimum print/audit:

```text
NarrativeDynamics.FitnessABMPathNParameters.beliefStep_half
NarrativeDynamics.FitnessABMPathNParameters.propagate_independent_exposures
NarrativeDynamics.FitnessABMPathNParameters.pathKernel_half
NarrativeDynamics.FitnessABMPathNParameters.propagate_eq_kernel
NarrativeDynamics.FitnessABMPathNParameters.allBroadcast_iterate
```

Keep all existing PathN requirements unchanged.

- [ ] **Step 3: Run full bounded regression.**

```bash
bash -n tools/check_fitness_abm_pathn.sh
bash tools/check_fitness_abm_pathn.sh
bash tools/check_fitness_abm_path4.sh
git diff --check
```
Expected: all exit 0.

- [ ] **Step 4: Commit gate.**

```bash
git add tools/check_fitness_abm_pathn.sh
git commit -m "ci: gate parameterized BB path response"
```

- [ ] **Step 5: Open PR against `proof/narrative-dynamics-v0`.**

PR body must state exact v1 claims and nonclaims: no arbitrary-parameter convergence/rate and no activation-time formula.

- [ ] **Step 6: Require exact-head proof + World Studio CI and independent review.**

Inspect exact checkout SHA, PathN parameter gate, existing Path4/Path5/PathN results, review threads, and diff. Fix any Critical/Important finding before merge. Merge only after required exact-head checks are green.
