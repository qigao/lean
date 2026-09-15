# BB Path5 finite-path vertical slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a concrete, exact-rational five-node BB path model driven by the existing propagation operator, with invariant/mean evidence and bounded replay fixtures.

**Architecture:** Keep Path4 untouched and add a sibling `FitnessABMPath5` module. The Path5 step and trajectory are defined through `NetworkPropagation.propagate`; derived mean facts follow an actual incoming-set bridge. Timing fixtures remain concrete and finite rather than becoming a generic Path n theorem.

**Tech Stack:** Lean 4.32, mathlib, exact `Rat` arithmetic, existing `NetworkPropagation`, pinned `lake` toolchain, shell trust-audit gate.

**Spec:** `docs/superpowers/specs/2026-09-15-bb-path5-finite-path-design.md`

## Global Constraints

- Use actual `NetworkPropagation.propagate`; no hand-written matrix as implementation.
- Preserve the existing Path4 module and theorem surface unchanged.
- Use rational arithmetic only; no floating-point theorem.
- Preserve inclusive threshold semantics.
- Do not claim arbitrary finite `Path n` or arbitrary connected-graph convergence.
- Do not claim that adding vertices preserves old shortest distances or activation times.
- No `sorry`, `admit`, `native_decide`, unsafe escape, or new axiom.
- Keep explicit Lean build/test timeouts and peak-resource output in the gate.

---

### Task 1: Add the Path5 RED consumer

**Files:** Create `NarrativeDynamics/Tests/FitnessABMPath5.lean`.

**Interfaces:** It imports the not-yet-present `NarrativeDynamics.Core.FitnessABMPath5` and expects `path5Adj`, `population`, `project`, `beliefStep`, `trajectory`, `allBroadcast`, and `mean` in the `NarrativeDynamics.FitnessABMPath5` namespace.

- [ ] **Step 1: Write the failing consumer**

```lean
import NarrativeDynamics.Core.FitnessABMPath5
open NarrativeDynamics
open NarrativeDynamics.NetworkPropagation
open NarrativeDynamics.FitnessABMPath5

example (x : Beliefs) (e : Fin 5 → Nat) :
    project (propagate path5Adj (population x e)) =
      project (propagate path5Adj (population x (fun _ => 0))) := by
  exact propagate_independent_exposures x e

example (x : Beliefs) (hx : allBroadcast x) :
    allBroadcast (beliefStep x) := by
  exact allBroadcast_step x hx

example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  exact mean_step x hx
```

- [ ] **Step 2: Run RED**

```bash
export PATH=/workspace/scratch/d37189c1329e/bb-runtime-toolchain/lean-4.32.0-linux/bin:$PATH
export LD_PRELOAD=/workspace/scratch/32cbf1a29dda/lean-proc-self-shim.so
timeout --kill-after=10s 120s lake env lean NarrativeDynamics/Tests/FitnessABMPath5.lean
```

Expected: failure because the Path5 core module is absent.

- [ ] **Step 3: Commit**

```bash
git add NarrativeDynamics/Tests/FitnessABMPath5.lean
git commit -m "test(lean): define Path5 propagation RED"
```

### Task 2: Implement concrete Path5 propagation

**Files:** Create `NarrativeDynamics/Core/FitnessABMPath5.lean`; modify the Task 1 test.

**Interfaces:** Produce `Beliefs := Fin 5 → Rat`, `path5Adj`, `population`, `project`, `allBroadcast`, `beliefStep`, `trajectory`, `propagate_independent_exposures`, and `incoming_eq`.

- [ ] **Step 1: Define the actual model**

```lean
abbrev Beliefs := Fin 5 → Rat
def path5Adj (i j : Fin 5) : Prop :=
  i.val + 1 = j.val ∨ j.val + 1 = i.val
def population (x : Beliefs) (e : Fin 5 → Nat) : Population 5 :=
  ⟨fun _ => ⟨1/2, 1/2⟩, fun i => ⟨x i, e i⟩⟩
def project (p : Population 5) : Beliefs := fun i => (p.agents i).belief
def allBroadcast (x : Beliefs) : Prop := ∀ i, 1/2 ≤ x i ∧ x i ≤ 1
def beliefStep (x : Beliefs) : Beliefs :=
  project (propagate path5Adj (population x (fun _ => 0)))
def trajectory (x : Beliefs) (n : Nat) : Beliefs := beliefStep^[n] x
```

Enumerate actual incoming sets as `![({1} : Finset (Fin 5)), {0, 2}, {1, 3}, {2, 4}, {3}] i`; use `fin_cases`, `simp`, `norm_num`, and `ring`. Do not define the step through the enumeration.

- [ ] **Step 2: Run focused build**

```bash
lake build NarrativeDynamics.Core.FitnessABMPath5
lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath5.lean
```

Expected: actual-propagation definitions and exposure bridge pass; invariant/mean examples remain RED until Task 3.

- [ ] **Step 3: Commit**

```bash
git add NarrativeDynamics/Core/FitnessABMPath5.lean NarrativeDynamics/Tests/FitnessABMPath5.lean
git commit -m "feat(lean): add actual Path5 propagation operator"
```

### Task 3: Prove Path5 invariant and stationary mean

**Files:** Modify `NarrativeDynamics/Core/FitnessABMPath5.lean` and the Path5 test.

**Interfaces:** Consume Task 2 `incoming_eq`, `beliefStep`, and `allBroadcast`; produce `allBroadcast_step`, `allBroadcast_iterate`, `mean`, and `mean_step`.

- [ ] **Step 1: Add RED assertions**

```lean
example (x : Beliefs) (hx : allBroadcast x) (n : Nat) :
    allBroadcast (trajectory x n) := by
  exact allBroadcast_iterate x hx n
example (x : Beliefs) (hx : allBroadcast x) :
    mean (beliefStep x) = mean x := by
  exact mean_step x hx
```

- [ ] **Step 2: Prove invariant**

For every `Fin 5` coordinate, rewrite with `incoming_eq`, unfold `nextAgent`, and show the update is a convex combination of values in `[1/2, 1]`; lift one step to `allBroadcast_iterate` by induction.

- [ ] **Step 3: Prove weighted mean**

```lean
def mean (x : Beliefs) : Rat :=
  (x 0 + 2*x 1 + 2*x 2 + 2*x 3 + x 4) / 8
```

Rewrite the actual step to incoming-set-derived expressions and close the rational identity with `ring`.

- [ ] **Step 4: Run focused proof tests**

```bash
lake build NarrativeDynamics.Core.FitnessABMPath5
lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath5.lean
```

Expected: PASS with no new axioms or unsafe declarations.

- [ ] **Step 5: Commit**

```bash
git add NarrativeDynamics/Core/FitnessABMPath5.lean NarrativeDynamics/Tests/FitnessABMPath5.lean
git commit -m "feat(lean): prove Path5 invariant and weighted mean"
```

### Task 4: Add bounded timing/replay evidence

**Files:** Modify `NarrativeDynamics/Tests/FitnessABMPath5.lean`.

**Interfaces:** Consume the Path5 operator and existing `TailModel`; produce finite rational fixture checks for selected activation histories and common-clock tail comparisons.

- [ ] **Step 1: Add deterministic fixtures**

Use explicit `Fin 5 → Rat` vectors and supplied replay states. Assert only values calculated from actual `beliefStep`/trajectory and finite activation offsets; do not assert a generic timing formula.

- [ ] **Step 2: Run the fixture consumer**

```bash
lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMPath5.lean
```

Expected: PASS with exact rational equalities.

- [ ] **Step 3: Commit**

```bash
git add NarrativeDynamics/Tests/FitnessABMPath5.lean
git commit -m "test(lean): add bounded Path5 timing fixtures"
```

### Task 5: Integrate Path5 into the exact gate

**Files:** Modify `tools/check_fitness_abm_path4.sh`.

**Interfaces:** Consume committed Path5 modules; produce source audit, bounded build/test execution, and resource reporting without changing the existing Path4 theorem audit.

- [ ] **Step 1: Extend source audit and module build list**

Add both Path5 files to the source audit and add `NarrativeDynamics.Core.FitnessABMPath5` to the timed build loop.

- [ ] **Step 2: Add timed Path5 test invocation**

Use the same `timeout --kill-after=10s 240s` envelope and GNU `time` output format used by the existing Path4 test.

- [ ] **Step 3: Run the full gate**

```bash
export PATH=/workspace/scratch/32cbf1a29dda/gnu-time-runtime/install/bin:/workspace/scratch/d37189c1329e/bb-runtime-toolchain/lean-4.32.0-linux/bin:$PATH
export LD_PRELOAD=/workspace/scratch/32cbf1a29dda/lean-proc-self-shim.so
timeout --kill-after=10s 240s bash tools/check_fitness_abm_path4.sh
```

Expected: source trust audit passed, Path4 tests passed, Path5 tests passed, and the existing 16-report trust-log audit passed.

- [ ] **Step 4: Run hygiene checks and commit**

```bash
bash -n tools/check_fitness_abm_path4.sh
git diff --check
git status --short
```

Expected: no shell syntax errors, no whitespace errors, and a clean worktree after commit.

```bash
git add tools/check_fitness_abm_path4.sh
git commit -m "ci: gate exact Path5 evidence"
```

### Task 6: Review, push, and update issue #83

**Files:** No additional source files beyond the committed implementation and gate files.

- [ ] **Step 1: Verify final commit and remote ref**

```bash
git log --oneline -6
git push -u origin feature/bb-path4-convergence-v1
git rev-parse HEAD
git rev-parse origin/feature/bb-path4-convergence-v1
```

- [ ] **Step 2: Post evidence to issue #83**

Record the exact head SHA, Path5 test runtime/peak RSS, Path4 regression result, source/trust audit result, and the explicit boundary that no generic Path n or arbitrary-graph theorem was claimed.

- [ ] **Step 3: Stop at review readiness**

Do not close #83 or introduce Path n until a separate review decision accepts the concrete Path5 evidence.
