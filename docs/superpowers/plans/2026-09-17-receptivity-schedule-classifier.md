# Receptivity Schedule Classifier Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the proof-backed PathN analyzer with exact theorem-backed receptivity schedule-family classification for harmonic, polynomial, exponential, periodic, alternating, finite-piecewise, and named schedules while preserving the Phase 1 `PROVED / DISPROVED / UNKNOWN` trust boundary.

**Architecture:** Build the mathematics first in one focused Lean module: mixing-mass identities, finite-prefix product lemmas, parameterized schedule families, exact Path2 product classifications, and stable-vs-oscillatory consequences. Only after those theorems are independently green, extend Python as orchestration: exact closed AST parsing, family recognition, hard-coded theorem-route selection, closed certificate generation, result assembly, CLI rendering, and additive CI.

**Tech Stack:** Lean 4.32.0, Mathlib exact `Rat`/`Real` analysis, existing `NarrativeDynamics.FitnessABMPathNExposureConvergence`, Python >= 3.11 standard library, generated Lean certificates, repository `timeout --kill-after=10s 240s` proof convention, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-17-receptivity-schedule-classifier-design.md`

**Issue:** #99, parent #96. Exact Phase 1 base: `proof/narrative-dynamics-v0@b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b`.

## Global Constraints

- Start implementation only after this committed plan is reviewed and approved.
- `NarrativeDynamics.FitnessABMPathNExposure.step` remains the only executable theorem-bearing semantics. Do not duplicate or alter it.
- Preserve the post-incoming Path2 multiplier index exactly: `alpha(e0 + r + 1)`.
- Python never proves infinite-product or asymptotic facts. It may only parse exact data, recognize closed families, select fixed theorem routes, render trusted certificates, and assemble results after Lean compilation.
- Every theorem-bearing `PROVED` or `DISPROVED` claim requires a successfully compiled exact Lean certificate with provenance.
- Missing theorem coverage is `UNKNOWN`. Certificate generation/compile/timeout/digest errors remain analyzer failures.
- All rational inputs remain exact integer/fraction strings. No floats, decimal syntax, scientific notation, user Lean source, predicates, theorem names, tactics, or import paths are accepted from input.
- Keep Phase 1 `constant`, `piecewise`, and trusted `named` forms backward compatible.
- Add only `harmonic`, `polynomial`, `exponential`, `periodic`, and `alternating` closed forms.
- Polynomial exponent is `Nat` with `p >= 1`.
- Decay target is the closed enum `{zero, one}`.
- `criterion_strength` is metadata only: `SUFFICIENT`, `NECESSARY_AND_SUFFICIENT`, `COUNTEREXAMPLE`, or absent.
- Path2 product iff is used only after Lean certifies parameter validity, equal initial exposures, all-broadcast, and unequal initial beliefs.
- Equal initial beliefs use a direct theorem route.
- PathN remains sufficient-only in Phase 2; do not invent a PathN iff criterion.
- `alpha -> 0` and `alpha -> 1` are descriptive schedule facts, not consensus verdicts.
- Stable non-consensus/nodewise convergence and oscillatory nodewise non-convergence are distinct claims.
- A reached `alpha = 1/2` zeroes the disagreement product from that point onward and must be handled before no-zero-factor asymptotic routes.
- Generic family theorem code belongs in `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`.
- Existing fixed `slowZeroSchedule`, `nearOneSchedule`, and `harmonicSchedule` theorems remain regression oracles.
- Existing proof/trust/resource gates stay unchanged or are extended additively.
- TDD is mandatory: RED focused test, minimum GREEN implementation, rerun earlier focused gates, commit.
- Do not touch `.github/workflows/proof.yml` until focused Lean and Python Phase 2 gates are green.

---

## Planned File Structure

Create:

```text
NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean
NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
tools/check_fitness_abm_schedule_classifier.sh

narrative_analyzer/families.py
narrative_analyzer/family_certificate.py

tests/test_narrative_analyzer_families.py
tests/test_narrative_analyzer_family_certificate.py
tests/test_narrative_analyzer_phase2_golden.py
```

Modify:

```text
narrative_analyzer/model.py
narrative_analyzer/result.py
narrative_analyzer/certificate.py
narrative_analyzer/analyze.py
narrative_analyzer/cli.py
narrative_analyzer/__init__.py

tests/test_narrative_analyzer_model.py
tests/test_narrative_analyzer_result.py
tests/test_narrative_analyzer_analysis.py
tests/test_narrative_analyzer_cli.py
.github/workflows/proof.yml
```

Create fixtures:

```text
tests/fixtures/narrative_analyzer/harmonic_family_path2.toml
tests/fixtures/narrative_analyzer/harmonic_near_one_path2.toml
tests/fixtures/narrative_analyzer/polynomial_p2_zero_path2.toml
tests/fixtures/narrative_analyzer/polynomial_p2_one_path2.toml
tests/fixtures/narrative_analyzer/exponential_zero_path2.toml
tests/fixtures/narrative_analyzer/exponential_one_path2.toml
tests/fixtures/narrative_analyzer/periodic_contracting_path2.toml
tests/fixtures/narrative_analyzer/periodic_identity_path2.toml
tests/fixtures/narrative_analyzer/alternating_contracting_path2.toml
tests/fixtures/narrative_analyzer/piecewise_constant_tail_path2.toml
tests/fixtures/narrative_analyzer/family_unequal_exposure_path2.toml
tests/fixtures/narrative_analyzer/family_equal_beliefs_path2.toml
```

---

### Task 1: Lean classifier foundation — closed target, mixing mass, equal-belief route, finite-prefix helpers

**Files:**
- Create: `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`
- Create: `NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean`
- Create: `tools/check_fitness_abm_schedule_classifier.sh`

**Interfaces:**
- Consumes existing `path2MultiplierProduct`, `path2_disagreement_product`, `path2_mean_iterate`, `path2_consensus_iff_product_tendsto_zero`.
- Produces `DecayTarget`, `applyDecayTarget`, `mixingMass`, `abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass`, `path2_equal_belief_consensus`, and finite-prefix/tail product adapters.

- [ ] **Step 1: Add failing theorem tests**

```lean
import NarrativeDynamics.Core.FitnessABMPathNExposureScheduleClassifier

namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests

open NarrativeDynamics
open NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

example : applyDecayTarget DecayTarget.zero (1/4 : Rat) = 1/4 := by
  norm_num [applyDecayTarget]

example : applyDecayTarget DecayTarget.one (1/4 : Rat) = 3/4 := by
  norm_num [applyDecayTarget]

example : mixingMass (1/4 : Rat) = 1/4 := by
  norm_num [mixingMass]

example : mixingMass (3/4 : Rat) = 1/4 := by
  norm_num [mixingMass]

example : |(1 : Rat) - 2 * (3/4 : Rat)| = 1 - 2 * mixingMass (3/4 : Rat) := by
  exact abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
    (a := 3/4) (by norm_num) (by norm_num)

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests
```

- [ ] **Step 2: Verify RED**

```bash
timeout --kill-after=10s 240s lake env lean \
  -DmaxErrors=1 -DstderrAsMessages=false \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
```

Expected: missing module/public definitions.

- [ ] **Step 3: Add exact closed primitives**

```lean
inductive DecayTarget
  | zero
  | one
  deriving DecidableEq, Repr

def applyDecayTarget (target : DecayTarget) (d : Rat) : Rat :=
  match target with
  | .zero => d
  | .one => 1 - d

noncomputable def mixingMass (a : Rat) : Rat := min a (1 - a)
```

Add theorem with exact public signature:

```lean
theorem abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
    {a : Rat} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    |1 - 2 * a| = 1 - 2 * mixingMass a
```

Proof strategy: split on `a ≤ 1/2`, simplify `min` and absolute value in each branch, discharge by `linarith`/`nlinarith`. No approximation.

- [ ] **Step 4: Add the direct equal-belief Path2 theorem**

Public signature:

```lean
theorem path2_equal_belief_consensus
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hbelief : (s 0).belief = (s 1).belief) :
    ∀ i : Fin 2,
      Tendsto
        (fun k => (beliefs ((step p 2)^[k] s) i : Real))
        atTop (nhds ((s 0).belief : Real))
```

Proof strategy: use existing Path2 one-step formulas to prove both beliefs remain equal and unchanged; induct over iterate count; conclude each belief sequence is constant. Keep all stated assumptions explicit.

- [ ] **Step 5: Add exact finite-prefix helpers**

Public signatures:

```lean
theorem tendsto_zero_const_mul_iff
    {f : Nat → Real} {c : Real} (hc : c ≠ 0) :
    Tendsto (fun k => c * f k) atTop (nhds 0) ↔
      Tendsto f atTop (nhds 0)

theorem tendsto_const_mul
    {f : Nat → Real} {x c : Real}
    (hf : Tendsto f atTop (nhds x)) :
    Tendsto (fun k => c * f k) atTop (nhds (c * x))
```

Also add two finite-product adapters with explicit natural prefix length `N`:

- nonzero prefix: rewrite the full product after `N` as a fixed nonzero prefix times a shifted tail product;
- zero prefix: prove the full product is zero for every `k >= N`.

These adapters must carry concrete equality hypotheses; do not hide product decomposition behind automation.

- [ ] **Step 6: Add the focused gate**

`tools/check_fitness_abm_schedule_classifier.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

timeout --kill-after=10s 240s lake env lean \
  -DmaxErrors=1 -DstderrAsMessages=false \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
```

- [ ] **Step 7: Run GREEN gates**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean \
  tools/check_fitness_abm_schedule_classifier.sh
git commit -m "feat(lean): add schedule classifier proof foundation"
```

---

### Task 2: Periodic, alternating, and finite-piecewise constant-tail exact classification

**Files:**
- Modify the Task 1 Lean core/test files.

**Interfaces:**
- Produces `periodicReceptivity`, `periodic_abs_product_tendsto_zero_of_contracting_entry`, `periodic_abs_product_not_tendsto_zero_of_boundary_values`, `alternatingReceptivity`, `piecewiseConstantTailReceptivity`, `piecewise_constant_tail_abs_product_tendsto_zero_of_interior_tail`, `piecewise_constant_tail_abs_product_not_tendsto_zero_of_boundary_tail`.

- [ ] **Step 1: Add RED period-two tests**

```lean
private def periodContracting : Fin 2 → Rat := ![1/4, 3/4]
private def periodBoundary : Fin 2 → Rat := ![0, 1]

private def periodContractingParams : ExposureParameters :=
  ⟨periodicReceptivity 2 (by decide) periodContracting, 0⟩

private def periodBoundaryParams : ExposureParameters :=
  ⟨periodicReceptivity 2 (by decide) periodBoundary, 0⟩
```

Add examples proving `|path2MultiplierProduct periodContractingParams 0 k| -> 0` and disproving zero convergence for `periodBoundaryParams`.

- [ ] **Step 2: Verify RED**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
```

Expected: missing periodic definitions/theorems.

- [ ] **Step 3: Implement periodic receptivity**

```lean
def periodicReceptivity
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e : Nat) : Rat :=
  values ⟨e % period, Nat.mod_lt _ hperiod⟩
```

For valid values `0 ≤ values i ≤ 1`, prove:

```lean
theorem periodic_abs_product_tendsto_zero_of_contracting_entry
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat)
    (hvalid : ∀ i, 0 ≤ values i ∧ values i ≤ 1)
    (j : Fin period) (hj : 0 < values j ∧ values j < 1)
    (e0 : Nat) :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 k : Real)|)
      atTop (nhds 0)
```

Proof strategy: product over every complete period has absolute value `q` with `0 ≤ q < 1`; decompose into `q^m` times a bounded remainder; use geometric decay.

For boundary-only values, expose:

```lean
theorem periodic_abs_product_not_tendsto_zero_of_boundary_values
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat)
    (hboundary : ∀ i, values i = 0 ∨ values i = 1)
    (e0 : Nat) :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity period hperiod values, 0⟩ e0 k : Real)|)
      atTop (nhds 0)
```

Proof strategy: every absolute multiplier is exactly one, therefore every absolute product is one.

- [ ] **Step 4: Add alternating as period-two only**

```lean
def alternatingReceptivity (a b : Rat) : Nat → Rat :=
  periodicReceptivity 2 (by decide) ![a, b]
```

Any alternating theorem must be a corollary of periodic infrastructure, not a duplicate asymptotic proof.

- [ ] **Step 5: Add RED finite-piecewise-tail tests**

Construct a theorem-test schedule with two finite overrides and a constant tail. Cover:

- tail `1/4`: absolute product tends to zero;
- tail `0`, no reached `1/2` override: absolute product does not tend to zero;
- one reached `1/2` override: product is eventually zero regardless of the boundary tail.

- [ ] **Step 6: Implement concrete-start finite-prefix/tail theorems**

Define a theorem-friendly `piecewiseConstantTailReceptivity` matching Phase 1 semantics: exact finite exposure overrides, exact default tail. Public classification theorems must include concrete `e0` and a finite bound `N` after which no reachable override remains.

Positive theorem signature:

```lean
theorem piecewise_constant_tail_abs_product_tendsto_zero_of_interior_tail
    (p : ExposureParameters) (e0 N : Nat) (tail : Rat)
    (htail : 0 < tail ∧ tail < 1)
    (heventual : ∀ r, N ≤ r → p.receptivityAt (e0 + r + 1) = tail) :
    Tendsto
      (fun k => |(path2MultiplierProduct p e0 k : Real)|)
      atTop (nhds 0)
```

Boundary theorem signature:

```lean
theorem piecewise_constant_tail_abs_product_not_tendsto_zero_of_boundary_tail
    (p : ExposureParameters) (e0 N : Nat) (tail : Rat)
    (htail : tail = 0 ∨ tail = 1)
    (heventual : ∀ r, N ≤ r → p.receptivityAt (e0 + r + 1) = tail)
    (hprefix : path2MultiplierProduct p e0 N ≠ 0) :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct p e0 k : Real)|)
      atTop (nhds 0)
```

Use Task 1 prefix lemmas; add a separate zero-prefix theorem for reached `1/2`.

- [ ] **Step 7: Run GREEN gates and commit**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): classify periodic and piecewise schedules"
```

---

### Task 3: Reusable infinite-product criterion plus polynomial/harmonic regimes

**Files:**
- Modify the Lean core/test files.

**Interfaces:**
- Produces `polynomialDecay`, `polynomialReceptivity`, `harmonicReceptivity`, `abs_product_tendsto_zero_of_mixing_sum_tendsto_atTop`, `abs_product_has_nonzero_limit_of_summable_mixing`, `polynomial_abs_product_tendsto_zero_of_p_eq_one`, `polynomial_abs_product_has_nonzero_limit_of_two_le_p`.

- [ ] **Step 1: Add RED generic-parameter tests**

Use parameters different from old fixed fixtures:

```lean
private def polyP1Params : ExposureParameters :=
  ⟨polynomialReceptivity (1/3) 1 2 DecayTarget.zero, 0⟩

private def polyP2Params : ExposureParameters :=
  ⟨polynomialReceptivity (1/4) 2 2 DecayTarget.zero, 0⟩
```

Test `|product| -> 0` for `polyP1Params` and existence of a nonzero absolute-product limit for `polyP2Params`.

- [ ] **Step 2: Define exact polynomial forms**

```lean
def polynomialDecay (c : Rat) (p offset e : Nat) : Rat :=
  c / (((e + offset : Nat) : Rat) ^ p)

def polynomialReceptivity
    (c : Rat) (p offset : Nat) (target : DecayTarget) (e : Nat) : Rat :=
  applyDecayTarget target (polynomialDecay c p offset e)

def harmonicReceptivity
    (c : Rat) (offset : Nat) (target : DecayTarget) : Nat → Rat :=
  polynomialReceptivity c 1 offset target
```

- [ ] **Step 3: Prove reusable validity bounds**

Under `1 ≤ p`, `1 ≤ offset`, `0 < c`, `c ≤ offset^p`, prove `0 < polynomialDecay c p offset e ≤ 1` for every `e`, and then prove both targets are valid receptivities in `[0,1]`.

Use public theorem names:

```text
polynomialDecay_pos
polynomialDecay_le_one
polynomialReceptivity_bounds
```

- [ ] **Step 4: Prove the reusable infinite-product criterion**

Public divergent-mixing theorem:

```lean
theorem abs_product_tendsto_zero_of_mixing_sum_tendsto_atTop
    (m : Nat → Real)
    (hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1/2)
    (hdiv : Tendsto (fun n => ∑ k ∈ Finset.range n, m k) atTop atTop) :
    Tendsto
      (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
      atTop (nhds 0)
```

Public summable-mixing theorem:

```lean
theorem abs_product_has_nonzero_limit_of_summable_mixing
    (m : Nat → Real)
    (hm0 : ∀ k, 0 ≤ m k)
    (hmhalf : ∀ k, m k < 1/2)
    (hsum : Summable m) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun n => ∏ k ∈ Finset.range n, (1 - 2 * m k))
        atTop (nhds L)
```

Implementation order:

1. search Mathlib locally for an infinite-product theorem with these hypotheses;
2. if it matches, wrap it behind these stable project theorem names;
3. otherwise prove the divergent direction from `Real.log (1-x) ≤ -x` and the summable direction from a two-sided small-`x` logarithm bound plus Cauchy/convergence of the log series;
4. keep all bounds exact and theorem-local.

No finite truncation or float estimate is acceptable.

- [ ] **Step 5: Instantiate p-series regimes**

Prove:

```lean
theorem polynomial_abs_product_tendsto_zero_of_p_eq_one
    (c : Rat) (offset e0 : Nat) (target : DecayTarget)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ offset) :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨polynomialReceptivity c 1 offset target, 0⟩ e0 k : Real)|)
      atTop (nhds 0)
```

For `2 ≤ p`, prove a theorem returning a nonzero absolute-product limit under validity and a no-zero-factor hypothesis:

```lean
theorem polynomial_abs_product_has_nonzero_limit_of_two_le_p
    (c : Rat) (p offset e0 : Nat) (target : DecayTarget)
    (hp : 2 ≤ p)
    (hoffset : 1 ≤ offset)
    (hc0 : 0 < c)
    (hcvalid : c ≤ ((offset : Rat) ^ p))
    (hnozero : ∀ r, polynomialReceptivity c p offset target (e0 + r + 1) ≠ 1/2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct
          ⟨polynomialReceptivity c p offset target, 0⟩ e0 k : Real)|)
        atTop (nhds L)
```

Use harmonic divergence for `p=1` and p-series summability for `p>=2`.

- [ ] **Step 6: Add harmonic corollaries and old-fixture regression**

Expose harmonic theorem names that delegate to polynomial `p=1`. Add a theorem test showing the existing fixed `harmonicSchedule` still satisfies the new generic classification. Keep the old exact telescoping theorem intact.

- [ ] **Step 7: Run GREEN gates and commit**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): classify polynomial receptivity schedules"
```

---

### Task 4: Exponential regimes and stable-vs-oscillatory Path2 consequences

**Files:**
- Modify the Lean core/test files.

**Interfaces:**
- Produces `exponentialDecay`, `exponentialReceptivity`, `exponential_abs_product_has_nonzero_limit`, `path2_nodewise_converges_of_signed_product_tendsto`, `path2_not_consensus_of_signed_product_tendsto_nonzero`, `path2_oscillatory_nonconvergence_of_even_odd_product_limits`.

- [ ] **Step 1: Add RED exponential tests**

```lean
private def expZeroParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.zero, 0⟩

private def expOneParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.one, 0⟩
```

Require nonzero absolute-product limits for both. Add separate tests that zero-target signed products converge to one nonzero limit while one-target signed products have incompatible even/odd subsequential limits.

- [ ] **Step 2: Define exact exponential forms**

```lean
def exponentialDecay (c base : Rat) (offset e : Nat) : Rat :=
  c * base ^ (e + offset)

def exponentialReceptivity
    (c base : Rat) (offset : Nat) (target : DecayTarget) (e : Nat) : Rat :=
  applyDecayTarget target (exponentialDecay c base offset e)
```

- [ ] **Step 3: Prove exact geometric summability and validity**

Under `0 < base < 1`, `0 < c`, and `c * base^offset ≤ 1`, prove valid receptivity bounds and summability of the mixing-mass sequence. Apply Task 3's reusable summable-product theorem.

Public absolute-product theorem:

```lean
theorem exponential_abs_product_has_nonzero_limit
    (c base : Rat) (offset e0 : Nat) (target : DecayTarget)
    (hc0 : 0 < c)
    (hbase0 : 0 < base)
    (hbase1 : base < 1)
    (hcvalid : c * base ^ offset ≤ 1)
    (hnozero : ∀ r, exponentialReceptivity c base offset target (e0 + r + 1) ≠ 1/2) :
    ∃ L : Real, 0 < L ∧
      Tendsto
        (fun k => |(path2MultiplierProduct
          ⟨exponentialReceptivity c base offset target, 0⟩ e0 k : Real)|)
        atTop (nhds L)
```

- [ ] **Step 4: Prove stable signed-product consequences**

Public theorem:

```lean
theorem path2_nodewise_converges_of_signed_product_tendsto
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    {L : Real}
    (hprod : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure k : Real))
      atTop (nhds L)) :
    ∃ c0 c1 : Real,
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 0 : Real)) atTop (nhds c0) ∧
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 1 : Real)) atTop (nhds c1)
```

Derive limits from the invariant mean plus the disagreement product formula. Add `path2_not_consensus_of_signed_product_tendsto_nonzero` requiring unequal initial beliefs and `L ≠ 0`.

- [ ] **Step 5: Prove oscillatory consequence**

Public theorem:

```lean
theorem path2_oscillatory_nonconvergence_of_even_odd_product_limits
    (p : ExposureParameters) (hvalid : p.Valid)
    (s : State 2)
    (he : (s 0).exposure = (s 1).exposure)
    (hb : allBroadcast p s)
    (hne : (s 0).belief ≠ (s 1).belief)
    {L : Real} (hL : L ≠ 0)
    (heven : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure (2*k) : Real))
      atTop (nhds L))
    (hodd : Tendsto
      (fun k => (path2MultiplierProduct p (s 0).exposure (2*k+1) : Real))
      atTop (nhds (-L))) :
    ¬ ∃ c0 c1 : Real,
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 0 : Real)) atTop (nhds c0) ∧
      Tendsto (fun k => (beliefs ((step p 2)^[k] s) 1 : Real)) atTop (nhds c1)
```

Use disagreement subsequences and uniqueness of limits.

- [ ] **Step 6: Instantiate target-zero stable and target-one oscillatory corollaries**

For polynomial `p>=2` and exponential target zero, prove signed-product nonzero-limit corollaries. For target one, prove even/odd signed-product corollaries. Add regression showing the existing `nearOneSchedule` qualitative non-convergence is reproduced by the generic oscillatory route.

- [ ] **Step 7: Run GREEN gates and commit**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): prove exponential and oscillatory schedule regimes"
```

---

### Task 5: Exact Python schedule AST, family recognition, and result metadata

**Files:**
- Modify: `narrative_analyzer/model.py`, `narrative_analyzer/result.py`, `narrative_analyzer/__init__.py`
- Create: `narrative_analyzer/families.py`, `tests/test_narrative_analyzer_families.py`
- Modify: `tests/test_narrative_analyzer_model.py`, `tests/test_narrative_analyzer_result.py`

**Interfaces:**

Add exact public types:

```python
class DecayTarget(Enum):
    ZERO = "zero"
    ONE = "one"

@dataclass(frozen=True)
class HarmonicSchedule:
    c: ExactRat
    offset: int
    target: DecayTarget

@dataclass(frozen=True)
class PolynomialSchedule:
    c: ExactRat
    p: int
    offset: int
    target: DecayTarget

@dataclass(frozen=True)
class ExponentialSchedule:
    c: ExactRat
    base: ExactRat
    offset: int
    target: DecayTarget

@dataclass(frozen=True)
class PeriodicSchedule:
    values: tuple[ExactRat, ...]

@dataclass(frozen=True)
class AlternatingSchedule:
    a: ExactRat
    b: ExactRat

class FamilyRecognitionStatus(Enum):
    RECOGNIZED = "RECOGNIZED"
    UNRECOGNIZED = "UNRECOGNIZED"

class CriterionStrength(Enum):
    SUFFICIENT = "SUFFICIENT"
    NECESSARY_AND_SUFFICIENT = "NECESSARY_AND_SUFFICIENT"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"

@dataclass(frozen=True)
class ScheduleFamilyInfo:
    status: FamilyRecognitionStatus
    family_id: str
    canonical_family_id: str
    exact_parameters: Mapping[str, str]
```

Append to `ClaimResult`:

```python
criterion_strength: CriterionStrength | None = None
```

- [ ] **Step 1: RED parser tests**

Test exact parsing of:

```python
{'kind': 'harmonic', 'c': '1/2', 'offset': 1, 'target': 'zero'}
{'kind': 'polynomial', 'c': '1/3', 'p': 2, 'offset': 2, 'target': 'one'}
{'kind': 'exponential', 'c': '1/4', 'base': '1/2', 'offset': 0, 'target': 'zero'}
{'kind': 'periodic', 'values': ['1/4', '3/4']}
{'kind': 'alternating', 'a': '1/4', 'b': '3/4'}
```

- [ ] **Step 2: RED validation tests**

Reject polynomial `p=0`, harmonic/polynomial `offset=0`, nonpositive decay `c`, exponential base `<=0` or `>=1`, empty periodic values, unknown target, rational TOML floats, and source-like kind/target strings.

- [ ] **Step 3: Implement the closed parser additions**

Extend the existing `Schedule` union with only the five approved dataclasses. Reuse `_exact_rat` for every rational field. Add one closed target parser. Do not evaluate expressions.

- [ ] **Step 4: Add result backward-compatibility tests**

Verify old `ClaimResult` construction without `criterion_strength` still works and new metadata is immutable.

- [ ] **Step 5: Implement syntactic family recognition only**

Public API:

```python
def recognize_family(schedule: Schedule) -> ScheduleFamilyInfo
```

Canonical mapping:

```text
harmonic -> polynomial, add p=1
polynomial -> polynomial
exponential -> exponential
periodic -> periodic
alternating -> periodic
piecewise -> piecewise_constant_tail
constant -> constant
trusted named -> named or trusted registry canonical family
```

No truth verdict is returned by this function.

- [ ] **Step 6: Run GREEN and commit**

```bash
python -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result \
  tests.test_narrative_analyzer_families
bash tools/check_narrative_analyzer.sh
git add narrative_analyzer/model.py narrative_analyzer/result.py \
  narrative_analyzer/families.py narrative_analyzer/__init__.py \
  tests/test_narrative_analyzer_model.py tests/test_narrative_analyzer_result.py \
  tests/test_narrative_analyzer_families.py
git commit -m "feat(analyzer): add exact schedule family model"
```

---

### Task 6: Hard-coded theorem registry and closed family certificate templates

**Files:**
- Create: `narrative_analyzer/family_certificate.py`, `tests/test_narrative_analyzer_family_certificate.py`
- Modify: `narrative_analyzer/families.py`, `narrative_analyzer/certificate.py`, `tests/test_narrative_analyzer_certificate.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class FamilyRoute:
    family_id: str
    theorem_kind: str
    theorem: str
    criterion_strength: CriterionStrength | None

class FamilyCertificateBuilder:
    def build_schedule_facts(
        self, model: PathModel, claims: tuple[str, ...]
    ) -> Certificate: ...

    def build_path2_classification(
        self, model: PathModel, route: FamilyRoute
    ) -> Certificate: ...
```

The method bodies are implemented in this task; the signatures above are the stable cross-task contract.

- [ ] **Step 1: RED closed-rendering/source-injection tests**

Verify each schedule renders only normalized integer/rational literals plus fixed Lean identifiers. Malicious family strings must fail in `model.py` before rendering.

- [ ] **Step 2: Implement fixed Lean family renderers**

Examples of required output shapes:

```text
polynomialReceptivity (1/3 : Rat) 2 2 DecayTarget.zero
exponentialReceptivity (1/4 : Rat) (1/2 : Rat) 0 DecayTarget.one
alternatingReceptivity (1/4 : Rat) (3/4 : Rat)
```

Periodic rendering builds a fixed `Fin n -> Rat` vector from normalized values and a fixed positivity proof for literal `n > 0`.

- [ ] **Step 3: Implement fixed theorem registry metadata**

Route only to public theorem names created in Tasks 1-4. Parameter-based routing may inspect exact AST fields (`p == 1`, `p >= 2`, target, periodic values, reached `1/2`, piecewise tail), but missing coverage returns `None` rather than generating a theorem name.

- [ ] **Step 4: Implement schedule-fact certificates**

`alpha_tends_to_zero` and `alpha_tends_to_one` compile independently of Path2 model applicability. Periodic schedules do not receive a limit claim unless a specific exact theorem applies.

- [ ] **Step 5: Implement Path2 classification certificates**

Every nontrivial product-iff certificate proves/instantiates:

```text
ExposureParameters.Valid
equal initial exposures
initial allBroadcast
unequal initial beliefs
exact family/product theorem
path2_consensus_iff_product_tendsto_zero or stronger stable/oscillatory theorem
```

Equal beliefs select `path2_equal_belief_consensus`. Unequal exposures produce no product-iff route.

- [ ] **Step 6: Verify theorem provenance**

Every `CertificateClaim` records the generated helper theorem and the exact production theorem chain. `criterion_strength` is attached from `FamilyRoute`, not inferred from status.

- [ ] **Step 7: Run GREEN and commit**

```bash
python -m unittest \
  tests.test_narrative_analyzer_family_certificate \
  tests.test_narrative_analyzer_certificate
bash tools/check_narrative_analyzer.sh
git add narrative_analyzer/family_certificate.py narrative_analyzer/families.py \
  narrative_analyzer/certificate.py tests/test_narrative_analyzer_family_certificate.py \
  tests/test_narrative_analyzer_certificate.py
git commit -m "feat(analyzer): generate schedule family certificates"
```

---

### Task 7: Analyzer orchestration — independent schedule facts and strongest applicable Path2 claim

**Files:**
- Modify: `narrative_analyzer/analyze.py`, `narrative_analyzer/result.py`, `tests/test_narrative_analyzer_analysis.py`

**Produces additive claims:**

```text
alpha_tends_to_zero
alpha_tends_to_one
multiplier_abs_product_tends_to_zero
multiplier_nonzero_limit
path2_nodewise_convergence
path2_oscillatory_nonconvergence
```

Existing `path2_consensus` remains the primary common-consensus Path2 claim.

- [ ] **Step 1: RED routing/result tests**

Cover exact expected routing for:

1. harmonic p=1 -> product-to-zero PROVED, Path2 consensus PROVED, N&S metadata;
2. polynomial p=2 target zero -> nonzero signed limit, consensus DISPROVED, nodewise convergence PROVED;
3. polynomial p=2 target one -> consensus DISPROVED, oscillatory nonconvergence PROVED;
4. exponential zero target -> stable non-consensus;
5. exponential one target -> oscillatory non-convergence;
6. periodic interior -> consensus PROVED;
7. periodic `{0,1}` boundary -> consensus DISPROVED for unequal initial beliefs;
8. equal beliefs -> direct consensus PROVED;
9. unequal initial exposures -> Path2 product claims UNKNOWN while independent generic PathN route may still apply;
10. recognized family with no theorem route -> theorem claim UNKNOWN.

`FakeRunner` is allowed here only to test orchestration; theorem correctness remains covered by Lean/golden tests.

- [ ] **Step 2: Add optional immutable family metadata to `AnalysisResult`**

Use:

```python
schedule_family: ScheduleFamilyInfo | None = None
```

as a defaulted additive field.

- [ ] **Step 3: Compile schedule facts independently**

A certified `alpha_tends_to_zero` or `alpha_tends_to_one` claim must remain available even when equal-exposure/all-broadcast Path2 prerequisites fail.

- [ ] **Step 4: Implement Path2 route precedence**

Exact order:

```text
1 equal-belief direct theorem
2 reached zero-factor exact consensus
3 applicable parameterized family product/stable/oscillatory route
4 existing trusted named fixed-fixture route
5 UNKNOWN
```

Do not let an UNKNOWN route overwrite compiled evidence. Conflicting compiled truth evidence raises `ProvenanceMismatchError`.

- [ ] **Step 5: Preserve generic PathN independence**

Do not change existing `pathn_consensus_exists` global-interior logic. A contradictory compiled Path2 negative and generic PathN positive for the same concrete model is an internal invariant/provenance failure, not a precedence choice.

- [ ] **Step 6: Run GREEN and commit**

```bash
python -m unittest tests.test_narrative_analyzer_analysis
bash tools/check_narrative_analyzer.sh
git add narrative_analyzer/analyze.py narrative_analyzer/result.py \
  tests/test_narrative_analyzer_analysis.py
git commit -m "feat(analyzer): classify schedule consensus regimes"
```

---

### Task 8: CLI rendering and exact Phase 2 TOML fixtures

**Files:**
- Modify: `narrative_analyzer/cli.py`, `tests/test_narrative_analyzer_cli.py`
- Create all Phase 2 fixtures listed above.

- [ ] **Step 1: RED rendering tests**

Require deterministic additions such as:

```text
Schedule family: polynomial
Canonical family: polynomial
Family parameters: c=1/4, offset=2, p=2, target=zero
path2_consensus              DISPROVED
  criterion: NECESSARY_AND_SUFFICIENT
```

Alternating preserves user family `alternating` and canonical family `periodic`.

- [ ] **Step 2: Preserve exit taxonomy tests**

```text
valid theorem verdicts including DISPROVED/UNKNOWN -> 0
malformed family input -> 2
certificate generation/compile/timeout -> 3
provenance/internal invariant failure -> 4
```

- [ ] **Step 3: Implement deterministic rendering**

Sort exact parameter keys. Print `criterion:` only when present. Never print a truncated numerical product as proof evidence.

- [ ] **Step 4: Add exact fixtures**

Every theorem-bearing rational is a quoted integer/fraction string. Reuse the existing model schema for topology, initial state, and threshold.

- [ ] **Step 5: Run GREEN and commit**

```bash
python -m unittest tests.test_narrative_analyzer_cli tests.test_narrative_analyzer_model
bash tools/check_narrative_analyzer.sh
git add narrative_analyzer/cli.py tests/test_narrative_analyzer_cli.py \
  tests/fixtures/narrative_analyzer
git commit -m "feat(analyzer): render schedule family classifications"
```

---

### Task 9: Real Lean Phase 2 golden certificates and backward compatibility

**Files:**
- Create: `tests/test_narrative_analyzer_phase2_golden.py`
- Modify `tools/check_narrative_analyzer.sh` only if its existing unittest discovery does not pick up the new file.

- [ ] **Step 1: Add real-runner golden tests**

Required concrete cases:

```text
harmonic target zero p=1 consensus
harmonic target one p=1 consensus despite alpha -> 1
polynomial p=2 zero target stable non-consensus + nodewise convergence
polynomial p=2 one target oscillatory non-convergence
exponential zero target stable non-consensus
exponential one target oscillatory non-convergence
periodic interior consensus
periodic boundary non-consensus
alternating contracting consensus
piecewise interior constant-tail consensus
unequal Path2 exposures -> product iff UNKNOWN without certificate failure
equal initial beliefs -> direct consensus theorem
```

Each PROVED/DISPROVED claim asserts expected theorem provenance text.

- [ ] **Step 2: Add Phase 1 compatibility goldens**

Run all existing Phase 1 fixtures through the same public API. Prior truth statuses and exact values remain unchanged except additive family metadata.

- [ ] **Step 3: Add broken-certificate fail-closed regression**

Intentionally corrupt one generated Phase 2 certificate in test-only code and assert `CertificateCompileError`; never accept `UNKNOWN`.

- [ ] **Step 4: Run real gates**

```bash
bash tools/check_narrative_analyzer.sh
bash tools/check_fitness_abm_schedule_classifier.sh
```

Expected: all tests PASS, no skipped theorem-bearing golden cases. Record the actual analyzer test count for the PR body; do not predeclare a count.

- [ ] **Step 5: Commit**

```bash
git add tests/test_narrative_analyzer_phase2_golden.py tools/check_narrative_analyzer.sh
git commit -m "test(analyzer): verify Phase 2 Lean family certificates"
```

---

### Task 10: Additive CI, exact-head verification, formal review, review-ready PR

**Files:**
- Modify: `.github/workflows/proof.yml`
- Update #99 only after real evidence exists.

- [ ] **Step 1: Add the focused theorem gate immediately before the existing analyzer gate**

```yaml
      - name: Receptivity schedule classifier proofs
        timeout-minutes: 20
        run: bash tools/check_fitness_abm_schedule_classifier.sh

      - name: PathN exposure consensus analyzer
        timeout-minutes: 20
        run: bash tools/check_narrative_analyzer.sh
```

Do not remove or weaken `BB finite-path convergence`.

- [ ] **Step 2: Run focused/full pre-push verification**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
bash tools/check_narrative_analyzer.sh
lake build
```

Then run the exact Python test command currently present in `.github/workflows/proof.yml`.

Expected: all PASS.

- [ ] **Step 3: Audit scope against exact base**

Confirm base `b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b` -> head changes only:

```text
Phase 2 Lean theorem/test files
Phase 2 analyzer code/tests/fixtures
additive proof workflow step
approved spec + plan
```

Confirm no modification to `FitnessABMPathNExposure.step` or unrelated BB/World Studio modules.

- [ ] **Step 4: Commit CI integration**

```bash
git add .github/workflows/proof.yml
git commit -m "ci: verify receptivity schedule classifier"
```

- [ ] **Step 5: Push implementation branch and create a draft PR against `proof/narrative-dynamics-v0`**

PR body must contain issue #99, spec/plan paths, exact base SHA, trust-boundary statement, theorem families added, N&S Path2 vs sufficient PathN distinction, and focused verification evidence.

- [ ] **Step 6: Require exact-head CI**

For the exact final PR head SHA require:

```text
World Studio = success
proof / Select proof event = success
proof / Python tests = success
proof / Lean proof = success
Build Lean library = success
BB finite-path convergence = success
Receptivity schedule classifier proofs = success
PathN exposure consensus analyzer = success
all downstream proof/trust/story/testimony steps = success
```

Any new code/test/workflow commit invalidates old exact-head evidence.

- [ ] **Step 7: Perform formal review**

Review exact base -> head for:

```text
theorem assumptions and overclaiming
mixing-mass identity and zero-factor handling
p=1 vs p>=2 polynomial regimes
zero-target stable vs one-target oscillatory semantics
periodic/piecewise concrete e0 handling
equal-exposure and unequal-belief iff assumptions
no Python proof authority
source-injection safety
fail-closed certificate/provenance behavior
Phase 1 backward compatibility
additive CI only
```

Fix Critical/Important findings and regenerate exact-head CI.

- [ ] **Step 8: Mark review-ready only after all gates**

Update #99 with exact head, workflow run IDs, actual focused test counts, theorem coverage, scope compare, and review findings. Mark PR Ready for review. Do not merge automatically; merge remains a separate explicit user gate.

---

## Final Acceptance Matrix

| Requirement | Required evidence |
|---|---|
| Phase 1 inputs compatible | existing analyzer tests/goldens green |
| Closed exact family DSL | parser rejection + source-injection tests |
| Parameterized harmonic | public theorem + real Lean certificate |
| Polynomial p=1 consensus | public theorem + real Lean certificate |
| Polynomial p>=2 stable/oscillatory split | public theorems + zero/one target goldens |
| Exponential stable/oscillatory split | public theorems + real goldens |
| Periodic exact classification | cycle theorem + interior/boundary goldens |
| Alternating delegates to periodic | canonical metadata + periodic provenance |
| Piecewise finite-prefix/tail | concrete-e0 theorem + golden |
| Equal beliefs handled directly | direct theorem + golden |
| Unequal exposures do not misuse iff | UNKNOWN regression |
| Truth states unchanged | result + CLI tests |
| Criterion strength is metadata | immutable result tests |
| Python does not prove asymptotics | source review + real certificate requirement |
| Compile/timeout fails closed | failure regression |
| PathN not upgraded to iff | review + metadata tests |
| Existing gates intact | exact-head proof workflow success |

## Implementation Stop Conditions

Stop and return to the design/spec gate instead of broadening scope if:

- the polynomial/exponential infinite-product theorem cannot be proved with the frozen family semantics without materially changing assumptions;
- a theorem would require floating-point or numerical truncation as proof authority;
- the current Path2 iff assumptions are insufficient and would require changing executable semantics;
- validity requires a generic symbolic expression language;
- implementation would weaken existing trust gates;
- satisfying Phase 2 would require a PathN necessary-and-sufficient theorem.

A stop condition is a design finding, not permission to improvise a broader implementation.
