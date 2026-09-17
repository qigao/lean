# Receptivity Schedule Classifier Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the proof-backed PathN analyzer with exact, theorem-backed receptivity schedule-family classification for harmonic, polynomial, exponential, periodic, alternating, finite-piecewise, and named schedules while preserving the Phase 1 `PROVED / DISPROVED / UNKNOWN` trust boundary.

**Architecture:** Add one focused Lean theorem layer that proves reusable mixing-mass/product facts and parameterized family classifications, then extend the existing Python analyzer only as orchestration: exact closed AST parsing, family recognition, theorem-route selection, closed certificate generation, and result rendering. Path2 product classification remains necessary-and-sufficient only under the existing equal-exposure/all-broadcast/nontrivial-belief assumptions; PathN remains on sufficient theorem routes only.

**Tech Stack:** Lean 4.32.0, Mathlib exact `Rat`/`Real` analysis and finite products, existing `NarrativeDynamics.FitnessABMPathNExposureConvergence`, Python >= 3.11 standard library (`dataclasses`, `enum`, `fractions`, `tomllib`, `unittest`), generated Lean certificates, repository `timeout --kill-after=10s 240s` focused proof convention, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-17-receptivity-schedule-classifier-design.md`

**Issue:** #99, parent roadmap #96. Phase 1 baseline: PR #98 merged as `b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b`.

## Global Constraints

- Start implementation only after this committed plan is reviewed and approved.
- Implement from exact base `proof/narrative-dynamics-v0@b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b` or a descendant that contains no conflicting Phase 2 changes.
- `NarrativeDynamics.FitnessABMPathNExposure.step` remains the single executable theorem-bearing semantics; do not duplicate or alter it for classifier convenience.
- Preserve post-incoming lookup exactly: Path2 multiplier term at round `r` uses `alpha(e0 + r + 1)`, never `alpha(e0 + r)`.
- Python may parse, normalize, recognize families, select theorem routes, render trusted templates, and compute finite exact witnesses only. Python must not decide infinite-product convergence numerically.
- Every theorem-bearing `PROVED` or `DISPROVED` result requires a successfully compiled generated Lean certificate and theorem provenance.
- A recognized family without a certified applicable theorem route yields theorem claim `UNKNOWN`, not a guessed verdict.
- Certificate generation failure, Lean compile failure, timeout/resource failure, or digest/provenance mismatch is an analyzer error, never `UNKNOWN`.
- Keep theorem-bearing values exact. Rational values are string-encoded in TOML; reject floats, decimals, scientific notation, source expressions, and user Lean fragments.
- Phase 1 schedule kinds `constant`, `piecewise`, and trusted `named` remain backward compatible.
- Phase 2 adds only closed `harmonic`, `polynomial`, `exponential`, `periodic`, and `alternating` forms.
- Polynomial exponent is `Nat` with `p >= 1`; no rational/real exponents in Phase 2.
- Decay target is a closed enum `{zero, one}`; do not add a generic symbolic complement expression.
- `criterion_strength` is theorem-route metadata, not a new truth status. Allowed values are `SUFFICIENT`, `NECESSARY_AND_SUFFICIENT`, `COUNTEREXAMPLE`, or absent.
- Path2 product iff classification is used only after Lean certifies parameter validity, equal initial exposures, initial all-broadcast, and unequal initial beliefs.
- Equal initial beliefs use a separate direct theorem route; do not misuse the nontrivial iff theorem.
- PathN remains conservative. Do not expose a PathN necessary-and-sufficient schedule criterion in this phase.
- `alpha -> 0` and `alpha -> 1` are schedule facts, not consensus verdicts.
- Distinguish stable non-consensus/nodewise convergence from oscillatory nodewise non-convergence.
- A reached factor `alpha = 1/2` forces the Path2 disagreement product to zero from that factor onward and must be handled before asymptotic no-zero-factor routes.
- New generic schedule theorem code belongs in `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`; do not grow Python into mathematical proof logic.
- Existing fixed fixtures (`slowZeroSchedule`, `nearOneSchedule`, `harmonicSchedule`) remain regression oracles; do not delete or weaken them.
- Existing proof/trust/resource gates remain unchanged or are extended additively.
- Use TDD: each task starts with a failing focused test/proof fixture, demonstrates RED, implements the minimum GREEN change, reruns all earlier focused checks, then commits.
- Do not modify `.github/workflows/proof.yml` until all focused Lean and Python classifier checks are green.

---

## Planned File Structure

### Lean theorem foundation

Create:

```text
NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean
NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
tools/check_fitness_abm_schedule_classifier.sh
```

Responsibilities:

- `...ScheduleClassifier.lean`: exact family definitions/normal forms, mixing-mass lemmas, finite-prefix/product lemmas, polynomial/exponential/periodic/piecewise classification theorems, equal-belief Path2 route, stable-vs-oscillatory consequences.
- `...Tests/...ScheduleClassifier.lean`: theorem-level regression examples only; no analyzer/Python concerns.
- `tools/check_fitness_abm_schedule_classifier.sh`: bounded focused Lean gate for the new theorem module and theorem tests.

Do not move or rewrite the existing convergence foundation during this phase.

### Python analyzer extension

Create:

```text
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
```

Responsibilities:

- `model.py`: closed exact schedule AST only.
- `families.py`: syntactic family recognition/canonicalization and trusted theorem-route registry metadata; no proof verdict authority.
- `result.py`: additive `ScheduleFamilyInfo` and optional `CriterionStrength` metadata.
- `family_certificate.py`: Phase 2 closed Lean certificate templates.
- `certificate.py`: retain Phase 1 structural/global-interior certificates and delegate family-specific certificate generation.
- `analyze.py`: merge independent compiled evidence without converting missing theorem coverage into errors.
- `cli.py`: render family information and criterion strength without changing exit taxonomy.

### Fixtures

Create exact fixtures under:

```text
tests/fixtures/narrative_analyzer/
  harmonic_family_path2.toml
  harmonic_near_one_path2.toml
  polynomial_p2_zero_path2.toml
  polynomial_p2_one_path2.toml
  exponential_zero_path2.toml
  exponential_one_path2.toml
  periodic_contracting_path2.toml
  periodic_identity_path2.toml
  alternating_contracting_path2.toml
  piecewise_constant_tail_path2.toml
  family_unequal_exposure_path2.toml
  family_equal_beliefs_path2.toml
```

Phase 1 fixtures remain untouched and continue to run.

---

### Task 1: Establish the Lean schedule-classifier foundation and mixing-mass bridge

**Files:**
- Create: `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`
- Create: `NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean`
- Create: `tools/check_fitness_abm_schedule_classifier.sh`

**Interfaces:**
- Consumes: `NarrativeDynamics.FitnessABMPathNExposureConvergence.path2MultiplierProduct`, `path2_disagreement_product`, `path2_mean_iterate`, `path2_consensus_iff_product_tendsto_zero`.
- Produces public Lean definitions/lemmas used by later tasks:
  - `DecayTarget`
  - `applyDecayTarget`
  - `mixingMass`
  - `abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass`
  - `path2_equal_belief_consensus`
  - finite-prefix helper lemmas for product limits.

- [ ] **Step 1: Write the failing theorem tests for the closed target and mixing-mass identity**

Add to the new test file:

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
  exact abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass (by norm_num) (by norm_num)

end NarrativeDynamics.FitnessABMPathNExposureScheduleClassifierTests
```

- [ ] **Step 2: Run the new test file and verify RED**

Run:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 -DstderrAsMessages=false NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
```

Expected: FAIL because the new module/public definitions do not exist.

- [ ] **Step 3: Add the minimal closed family primitives**

In the new core module, define exactly:

```lean
namespace NarrativeDynamics.FitnessABMPathNExposureScheduleClassifier

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

Prove:

```lean
theorem abs_one_sub_two_mul_eq_one_sub_two_mul_mixingMass
    {a : Rat} (ha0 : 0 ≤ a) (ha1 : a ≤ 1) :
    |1 - 2 * a| = 1 - 2 * mixingMass a := by
  -- split on a ≤ 1/2 and normalize min/abs in each branch
```

The implementation must use exact ordered-field reasoning (`by_cases`, `simp`, `linarith`/`nlinarith`) and not decimal approximation.

- [ ] **Step 4: Add direct equal-belief Path2 consensus theorem**

Expose:

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
        atTop (nhds ((s 0).belief : Real)) := by
  -- prove one-step preservation from the existing Path2 formulas,
  -- then iterate and conclude the sequence is constant.
```

Do not weaken `p.Valid`, equal exposure, or `allBroadcast` assumptions merely to simplify the proof.

- [ ] **Step 5: Add finite-prefix product limit helpers**

Expose helper theorems with these logical contracts:

```lean
theorem tendsto_zero_mul_const_iff
    {f : Nat → Real} {c : Real} (hc : c ≠ 0) :
    Tendsto (fun k => c * f k) atTop (nhds 0) ↔
      Tendsto f atTop (nhds 0)
```

and a finite-prefix adapter that permits replacing a product sequence by a nonzero constant prefix times a tail product. If the prefix is zero, expose a separate theorem concluding eventual product zero. Keep zero-prefix and nonzero-prefix routes separate so later piecewise classification cannot divide by zero.

- [ ] **Step 6: Implement the focused gate script**

Create executable `tools/check_fitness_abm_schedule_classifier.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

timeout --kill-after=10s 240s lake env lean \
  -DmaxErrors=1 -DstderrAsMessages=false \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
```

- [ ] **Step 7: Run focused Lean GREEN plus existing PathN foundation gate**

Run:

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
```

Expected: both PASS.

- [ ] **Step 8: Commit Task 1**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean \
  tools/check_fitness_abm_schedule_classifier.sh
git commit -m "feat(lean): add schedule classifier proof foundation"
```

---

### Task 2: Add exact periodic, alternating, and finite-piecewise-tail classification theorems

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean`

**Interfaces:**
- Consumes: Task 1 mixing-mass and finite-prefix helpers.
- Produces:
  - `periodicReceptivity`
  - `periodic_abs_product_tendsto_zero_of_contracting_entry`
  - `periodic_abs_product_not_tendsto_zero_of_boundary_values`
  - `alternatingReceptivity`
  - `piecewiseConstantTailReceptivity`
  - `piecewise_constant_tail_product_tendsto_zero_iff`.

- [ ] **Step 1: Add RED examples for periodic contraction and identity boundary**

Add theorem tests using period-two vectors:

```lean
private def periodContracting : Fin 2 → Rat := ![1/4, 3/4]
private def periodBoundary : Fin 2 → Rat := ![0, 1]

example :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity 2 (by decide) periodContracting, 0⟩ 0 k : Real)|)
      atTop (nhds 0) := by
  exact periodic_abs_product_tendsto_zero_of_contracting_entry
    (period := 2) (by decide) periodContracting (by
      intro i; fin_cases i <;> norm_num) ⟨0, by decide⟩ (by norm_num)

example :
    ¬ Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨periodicReceptivity 2 (by decide) periodBoundary, 0⟩ 0 k : Real)|)
      atTop (nhds 0) := by
  exact periodic_abs_product_not_tendsto_zero_of_boundary_values
    (period := 2) (by decide) periodBoundary (by
      intro i; fin_cases i <;> norm_num)
```

Run the focused gate and verify RED because theorem names are absent.

- [ ] **Step 2: Implement periodic receptivity and exact cycle reasoning**

Define:

```lean
def periodicReceptivity
    (period : Nat) (hperiod : 0 < period)
    (values : Fin period → Rat) (e : Nat) : Rat :=
  values ⟨e % period, Nat.mod_lt _ hperiod⟩
```

Prove the positive route by grouping factors into full periods plus a bounded remainder. The proof must show that one strict interior entry produces a cycle absolute multiplier `q` with `0 ≤ q < 1`, hence powers `q^m -> 0`. Do not use numerical sampling.

Prove the boundary route under `∀ i, values i = 0 ∨ values i = 1`: every absolute multiplier is exactly `1`, so the absolute product is exactly `1` for all `k` and cannot tend to zero.

- [ ] **Step 3: Add alternating as a thin period-two specialization**

Define:

```lean
def alternatingReceptivity (a b : Rat) : Nat → Rat :=
  periodicReceptivity 2 (by decide) ![a, b]
```

Expose corollaries that delegate to periodic theorems; do not reprove product asymptotics.

- [ ] **Step 4: Add RED tests for finite piecewise + constant tail**

Use a concrete override map encoded in the theorem test (for example exposure 1 -> `1/4`, exposure 2 -> `3/4`, tail -> `1/4`) and verify:

```lean
example :
    Tendsto
      (fun k => |(path2MultiplierProduct piecewiseExampleParams 0 k : Real)|)
      atTop (nhds 0) := by
  exact piecewise_constant_tail_product_tendsto_zero_iff.mp ...
```

Also add a boundary tail fixture with tail `0` and no reached `1/2` override, and prove the absolute product does not tend to zero.

- [ ] **Step 5: Implement finite-prefix + constant-tail classification**

Define a theorem-friendly receptivity function that accepts finite exact overrides and a tail value, but keep the existing Python Phase 1 AST semantics unchanged. The key public theorem must distinguish:

1. some reached finite factor equals `1/2` -> product eventually exactly zero;
2. no zero factor and tail strictly interior -> absolute product tends to zero;
3. no zero factor and tail in `{0,1}` -> absolute product does not tend to zero.

The concrete starting exposure `e0` is part of the theorem statement so overrides below `e0 + 1` do not affect the result.

- [ ] **Step 6: Run GREEN gates**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): classify periodic and piecewise schedules"
```

---

### Task 3: Add the reusable infinite-product criterion and polynomial/harmonic family regimes

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean`

**Interfaces:**
- Consumes: Task 1 mixing-mass bridge.
- Produces:
  - `polynomialReceptivity`
  - `harmonicReceptivity`
  - a reusable absolute-product theorem reducing product-to-zero/nonzero behavior to divergent/summable nonnegative mixing mass under `m_k < 1/2` after explicit zero-factor handling
  - `polynomial_abs_product_tendsto_zero_of_p_eq_one`
  - `polynomial_abs_product_has_nonzero_limit_of_two_le_p`
  - harmonic corollaries.

- [ ] **Step 1: Add RED theorem tests for p=1 and p=2 using generic parameters**

Use parameters that are not equal to the old fixed fixture constants, so the tests prove the family is genuinely parameterized:

```lean
example :
    Tendsto
      (fun k => |(path2MultiplierProduct
        ⟨polynomialReceptivity (1/3) 1 2 DecayTarget.zero, 0⟩ 0 k : Real)|)
      atTop (nhds 0) := by
  exact polynomial_abs_product_tendsto_zero_of_p_eq_one
    (c := 1/3) (offset := 2) (target := .zero) (e0 := 0)
    (by norm_num) (by norm_num)

example :
    ∃ L : Real, L ≠ 0 ∧
      Tendsto
        (fun k => |(path2MultiplierProduct
          ⟨polynomialReceptivity (1/4) 2 2 DecayTarget.zero, 0⟩ 0 k : Real)|)
        atTop (nhds L) := by
  exact polynomial_abs_product_has_nonzero_limit_of_two_le_p
    (c := 1/4) (p := 2) (offset := 2) (target := .zero) (e0 := 0)
    (by norm_num) (by norm_num) (by norm_num)
```

Run focused Lean and verify RED.

- [ ] **Step 2: Define exact polynomial/harmonic receptivity**

Define:

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

Do not define harmonic with a second formula.

- [ ] **Step 3: Prove validity bounds as reusable assumptions/corollaries**

For `offset >= 1`, `p >= 1`, `0 < c`, and `c ≤ offset^p`, prove `0 < polynomialDecay ... ≤ 1` for all exposures and therefore both decay targets remain in `[0,1]`.

Expose theorem names used by certificates rather than forcing generated certificates to rebuild monotonic denominator arithmetic.

- [ ] **Step 4: Prove the reusable infinite-product criterion**

Implement one exact theorem over a nonnegative Real sequence `m : Nat → Real` satisfying eventually `0 ≤ 2*m k < 1`:

- if partial sums of `m` diverge to `+∞`, then `∏_{r<k} (1 - 2*m r) -> 0`;
- if `∑ m` converges and no factor is zero, the partial products converge to a positive/nonzero limit.

Use Mathlib infinite product/series results if their hypotheses match exactly. If they do not, prove the local bridge using `Real.log` inequalities (`log (1-x) ≤ -x` for the divergent direction and a two-sided small-`x` bound for the summable direction). The final public API must be independent of numerical truncation and expose all hypotheses explicitly.

- [ ] **Step 5: Instantiate p-series regimes**

Use exact comparison with the harmonic series / p-series infrastructure:

- `p = 1`: mixing mass series diverges -> absolute product tends to zero;
- `p >= 2`: mixing mass series converges -> absent a reached zero factor, absolute product has a nonzero limit.

Handle `target=zero` and `target=one` through the mixing-mass identity instead of duplicating absolute-product proofs.

- [ ] **Step 6: Expose harmonic corollaries and regress old fixture**

Prove the parameterized harmonic route with `p=1`, then add a theorem test showing the old `harmonicSchedule` product result is compatible with the generic harmonic classification. Do not delete the existing exact `harmonic_product` theorem.

- [ ] **Step 7: Run GREEN gates**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): classify polynomial receptivity schedules"
```

---

### Task 4: Add exponential regimes and stable-vs-oscillatory Path2 consequences

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean`

**Interfaces:**
- Consumes: Task 3 reusable summable-mixing-mass theorem and existing Path2 disagreement/mean formulas.
- Produces:
  - `exponentialReceptivity`
  - `exponential_abs_product_has_nonzero_limit`
  - `path2_nodewise_converges_of_signed_product_tendsto`
  - `path2_not_consensus_of_signed_product_tendsto_nonzero`
  - `path2_oscillatory_nonconvergence_of_even_odd_product_limits`.

- [ ] **Step 1: Add RED tests for exponential zero-target and one-target**

Use:

```lean
private def expZeroParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.zero, 0⟩

private def expOneParams : ExposureParameters :=
  ⟨exponentialReceptivity (1/4) (1/2) 0 DecayTarget.one, 0⟩
```

Test that both absolute products have nonzero limits, then test distinct signed behavior:

- zero-target signed product converges to a nonzero `L`;
- one-target even and odd subsequences have limits `L` and `-L` for some `L ≠ 0`.

- [ ] **Step 2: Implement exponential receptivity and exact geometric summability**

Define:

```lean
def exponentialDecay (c base : Rat) (offset e : Nat) : Rat :=
  c * base ^ (e + offset)

def exponentialReceptivity
    (c base : Rat) (offset : Nat) (target : DecayTarget) (e : Nat) : Rat :=
  applyDecayTarget target (exponentialDecay c base offset e)
```

Under `0 < base < 1`, prove the mixing-mass series is summable by comparison/equality to an exact geometric series, then apply Task 3's reusable product criterion.

- [ ] **Step 3: Prove stable signed-product consequence for zero-target decay**

For valid zero-target polynomial `p>=2` and exponential families, prove the signed factors are eventually positive and, after handling any finite prefix, the signed product converges to a nonzero `L`.

Expose:

```lean
theorem path2_nodewise_converges_of_signed_product_tendsto ...
```

Using `path2_mean_iterate` + disagreement formula, conclude each node converges to an exact expression involving the initial mean and `L * initialDisagreement / 2`.

Then expose a corollary proving common consensus is false when initial beliefs differ and `L ≠ 0`.

- [ ] **Step 4: Prove oscillatory consequence for target-one decay**

When the target-one family is eventually above `1/2`, signed multipliers are eventually negative. Combine nonzero absolute-product limit with parity to prove incompatible even/odd disagreement subsequential limits.

Expose:

```lean
theorem path2_oscillatory_nonconvergence_of_even_odd_product_limits ...
```

whose conclusion is nodewise non-convergence, stronger than merely non-consensus.

- [ ] **Step 5: Regress existing near-one fixture through the generic consequence**

Add a theorem test showing the old `nearOneSchedule` remains non-convergent and that the new generic oscillation lemma can reproduce the qualitative result without deleting `nearOne_not_convergent`.

- [ ] **Step 6: Run GREEN gates**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
```

Expected: PASS.

- [ ] **Step 7: Commit Task 4**

```bash
git add NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean \
  NarrativeDynamics/Tests/FitnessABMPathNExposureScheduleClassifier.lean
git commit -m "feat(lean): prove exponential and oscillatory schedule regimes"
```

---

### Task 5: Extend the exact Python schedule AST and result metadata

**Files:**
- Modify: `narrative_analyzer/model.py`
- Modify: `narrative_analyzer/result.py`
- Create: `narrative_analyzer/families.py`
- Modify: `narrative_analyzer/__init__.py`
- Modify: `tests/test_narrative_analyzer_model.py`
- Modify: `tests/test_narrative_analyzer_result.py`
- Create: `tests/test_narrative_analyzer_families.py`

**Interfaces:**
- Consumes: Phase 1 `ExactRat`, `PathModel`, existing schedule variants and immutable result classes.
- Produces Python types:

```python
class DecayTarget(Enum): ZERO = "zero"; ONE = "one"
@dataclass(frozen=True) class HarmonicSchedule: c: ExactRat; offset: int; target: DecayTarget
@dataclass(frozen=True) class PolynomialSchedule: c: ExactRat; p: int; offset: int; target: DecayTarget
@dataclass(frozen=True) class ExponentialSchedule: c: ExactRat; base: ExactRat; offset: int; target: DecayTarget
@dataclass(frozen=True) class PeriodicSchedule: values: tuple[ExactRat, ...]
@dataclass(frozen=True) class AlternatingSchedule: a: ExactRat; b: ExactRat

class CriterionStrength(Enum):
    SUFFICIENT = "SUFFICIENT"
    NECESSARY_AND_SUFFICIENT = "NECESSARY_AND_SUFFICIENT"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"

@dataclass(frozen=True) class ScheduleFamilyInfo:
    status: FamilyRecognitionStatus
    family_id: str
    canonical_family_id: str
    exact_parameters: Mapping[str, str]
```

`ClaimResult` gains `criterion_strength: CriterionStrength | None = None` as a backward-compatible final/default field.

- [ ] **Step 1: Write RED parser tests for every new family**

Add exact input cases including:

```python
{'kind': 'harmonic', 'c': '1/2', 'offset': 1, 'target': 'zero'}
{'kind': 'polynomial', 'c': '1/3', 'p': 2, 'offset': 2, 'target': 'one'}
{'kind': 'exponential', 'c': '1/4', 'base': '1/2', 'offset': 0, 'target': 'zero'}
{'kind': 'periodic', 'values': ['1/4', '3/4']}
{'kind': 'alternating', 'a': '1/4', 'b': '3/4'}
```

Assert exact dataclass equality.

- [ ] **Step 2: Write RED validation tests**

Reject:

- polynomial `p=0`;
- harmonic `offset=0`;
- polynomial `offset=0`;
- nonpositive `c` for decay families;
- exponential base `0`, `1`, negative, or `>1`;
- empty periodic list;
- unknown target;
- TOML float in any rational field;
- source-like strings as target/kind.

These are input errors, not theorem `UNKNOWN`.

- [ ] **Step 3: Implement minimal immutable AST/parser additions**

Extend the closed `Schedule` union only with the five approved dataclasses. Keep `_exact_rat` unchanged as the only rational parser. Add a closed target parser; never evaluate expressions.

- [ ] **Step 4: Write and implement result metadata tests**

Verify:

```python
ClaimResult(
    claim_id='path2_consensus',
    status=ClaimStatus.PROVED,
    theorem='T',
    assumptions=(),
    exact_values={},
    note=None,
    criterion_strength=CriterionStrength.NECESSARY_AND_SUFFICIENT,
)
```

is immutable and serially/renderably distinct from the truth status.

Verify old Phase 1 constructors that omit `criterion_strength` still work unchanged.

- [ ] **Step 5: Implement `families.py` recognition only**

Expose:

```python
def recognize_family(schedule: Schedule) -> ScheduleFamilyInfo
```

Rules:

- harmonic -> `family_id='harmonic'`, `canonical_family_id='polynomial'`, include `p='1'`;
- polynomial -> canonical `polynomial`;
- exponential -> canonical `exponential`;
- periodic -> canonical `periodic`;
- alternating -> `family_id='alternating'`, canonical `periodic`;
- Phase 1 piecewise -> recognized `piecewise_constant_tail`;
- trusted named -> `family_id='named'`, canonical family only from trusted registry metadata;
- constant -> recognized `constant`.

This function must not return any theorem verdict.

- [ ] **Step 6: Run Python GREEN plus Phase 1 regression**

```bash
python -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result \
  tests.test_narrative_analyzer_families
bash tools/check_narrative_analyzer.sh
```

Expected: PASS.

- [ ] **Step 7: Commit Task 5**

```bash
git add narrative_analyzer/model.py narrative_analyzer/result.py \
  narrative_analyzer/families.py narrative_analyzer/__init__.py \
  tests/test_narrative_analyzer_model.py tests/test_narrative_analyzer_result.py \
  tests/test_narrative_analyzer_families.py
git commit -m "feat(analyzer): add exact schedule family model"
```

---

### Task 6: Add trusted Phase 2 theorem registry and closed family certificate templates

**Files:**
- Create: `narrative_analyzer/family_certificate.py`
- Modify: `narrative_analyzer/certificate.py`
- Modify: `narrative_analyzer/families.py`
- Create: `tests/test_narrative_analyzer_family_certificate.py`
- Modify: `tests/test_narrative_analyzer_certificate.py`

**Interfaces:**
- Consumes: Task 5 family AST and Task 1-4 public Lean theorem names.
- Produces:

```python
@dataclass(frozen=True)
class FamilyRoute:
    family_id: str
    theorem_kind: str
    theorem: str
    criterion_strength: CriterionStrength | None

class FamilyCertificateBuilder:
    def build_schedule_facts(self, model: PathModel, claims: tuple[str, ...]) -> Certificate
    def build_path2_classification(self, model: PathModel, route: FamilyRoute) -> Certificate
```

- [ ] **Step 1: Write RED source-injection and rendering tests**

For each family, assert generated source contains only canonical exact literals and fixed Lean identifiers. Explicitly test that malicious strings cannot become theorem names/imports/tactics because the parser rejects them before certificate generation.

- [ ] **Step 2: Implement fixed Lean renderer for each closed AST**

Render only trusted forms, for example:

```python
PolynomialSchedule(...) ->
  "polynomialReceptivity <rat c> <p> <offset> DecayTarget.zero"
```

and:

```python
AlternatingSchedule(a, b) ->
  "alternatingReceptivity <rat a> <rat b>"
```

Periodic rendering must produce a closed `Fin n -> Rat` vector from exact values and a fixed positive-period proof; user text never becomes Lean syntax except validated integer/rational digits already normalized by `ExactRat`.

- [ ] **Step 3: Implement theorem registry metadata**

Map only approved theorem kinds to hard-coded fully qualified names from `FitnessABMPathNExposureScheduleClassifier`. Route selection may use exact parameter relations such as `p == 1`, `p >= 2`, target enum, periodic boundary/interior values, piecewise tail, and reached `1/2` witnesses.

Missing coverage returns no route; it must not synthesize theorem names.

- [ ] **Step 4: Build schedule-fact certificates**

Certificates for `alpha_tends_to_zero` / `alpha_tends_to_one` must instantiate the exact family theorem and compile independently. Periodic/constant schedules normally have neither limit claim unless constant/trivial; do not infer a limit from visual pattern recognition.

- [ ] **Step 5: Build Path2 classification certificates**

Each certificate must re-prove/instantiate:

- parameter validity;
- equal initial exposures for the product iff route;
- initial all-broadcast;
- unequal initial beliefs, or select the direct equal-belief theorem;
- exact family/product theorem;
- final consensus/stable-non-consensus/oscillatory theorem.

For unequal initial exposures, do not generate the product-iff certificate.

- [ ] **Step 6: Verify provenance metadata**

Every produced `CertificateClaim` records the exact generated helper theorem plus the production theorem chain. Examples:

```text
...polynomial_abs_product_tendsto_zero_of_p_eq_one;
...path2_consensus_iff_product_tendsto_zero
```

and for target-one p>=2/exponential:

```text
...path2_oscillatory_nonconvergence_of_even_odd_product_limits
```

- [ ] **Step 7: Run GREEN tests**

```bash
python -m unittest \
  tests.test_narrative_analyzer_family_certificate \
  tests.test_narrative_analyzer_certificate
bash tools/check_narrative_analyzer.sh
```

Expected: PASS.

- [ ] **Step 8: Commit Task 6**

```bash
git add narrative_analyzer/family_certificate.py narrative_analyzer/certificate.py \
  narrative_analyzer/families.py tests/test_narrative_analyzer_family_certificate.py \
  tests/test_narrative_analyzer_certificate.py
git commit -m "feat(analyzer): generate schedule family certificates"
```

---

### Task 7: Integrate family claims into analyzer orchestration without weakening Phase 1 semantics

**Files:**
- Modify: `narrative_analyzer/analyze.py`
- Modify: `narrative_analyzer/result.py`
- Modify: `tests/test_narrative_analyzer_analysis.py`

**Interfaces:**
- Consumes: Task 5 recognition and Task 6 compiled family certificates.
- Produces additive claims:
  - `alpha_tends_to_zero`
  - `alpha_tends_to_one`
  - `multiplier_abs_product_tends_to_zero`
  - `multiplier_nonzero_limit`
  - `path2_nodewise_convergence`
  - `path2_oscillatory_nonconvergence`
  - existing `path2_consensus` updated by strongest applicable certified route.

- [ ] **Step 1: Write RED orchestration tests for representative regimes**

Use `FakeRunner` only to test routing/result assembly, not theorem correctness. Cover:

- harmonic p=1 -> product zero PROVED, Path2 consensus PROVED, criterion N&S;
- polynomial p=2 target zero -> product-zero DISPROVED or nonzero-limit PROVED, Path2 consensus DISPROVED, nodewise convergence PROVED;
- polynomial p=2 target one -> Path2 consensus DISPROVED, oscillatory nonconvergence PROVED;
- exponential target zero -> stable non-consensus;
- periodic interior -> consensus PROVED;
- periodic all boundary `{0,1}` -> consensus DISPROVED when initial beliefs unequal;
- equal beliefs -> direct consensus PROVED without using nontrivial product iff;
- unequal initial exposures -> product claims/path2 iff UNKNOWN while generic PathN sufficient route may still independently prove consensus;
- recognized family with intentionally absent theorem route -> claim UNKNOWN, not analyzer error.

- [ ] **Step 2: Add `schedule_family` to `AnalysisResult`**

Make it an optional/additive immutable field so Phase 1 callers/tests stay valid. Populate it for all recognized closed schedule types.

- [ ] **Step 3: Add family schedule facts independently**

Compile schedule-fact certificates separately from Path2 classification so a valid `alpha_tends_to_zero` proof can survive when Path2 assumptions are unavailable.

- [ ] **Step 4: Add Path2 route precedence**

Use this precedence:

1. equal-belief direct consensus theorem;
2. reached zero-factor exact consensus route;
3. applicable family product iff route;
4. existing trusted named fixed-fixture route;
5. otherwise `UNKNOWN` for `path2_consensus`.

Do not allow a weaker UNKNOWN route to overwrite an already compiled PROVED/DISPROVED result. Conflicting compiled evidence remains `ProvenanceMismatchError`.

- [ ] **Step 5: Keep generic PathN consensus independent**

Existing `pathn_consensus_exists` global-interior logic stays unchanged. A Path2 family negative result is authoritative for the concrete Path2 common-consensus claim; a positive generic PathN route and negative Path2 route must never both compile for the same valid concrete model. Treat such a conflict as an internal provenance/invariant error rather than picking one silently.

- [ ] **Step 6: Run GREEN analysis regression**

```bash
python -m unittest tests.test_narrative_analyzer_analysis
bash tools/check_narrative_analyzer.sh
```

Expected: PASS.

- [ ] **Step 7: Commit Task 7**

```bash
git add narrative_analyzer/analyze.py narrative_analyzer/result.py \
  tests/test_narrative_analyzer_analysis.py
git commit -m "feat(analyzer): classify schedule consensus regimes"
```

---

### Task 8: Extend CLI rendering and add exact TOML fixtures

**Files:**
- Modify: `narrative_analyzer/cli.py`
- Modify: `tests/test_narrative_analyzer_cli.py`
- Create: Phase 2 fixture files listed in Planned File Structure.

**Interfaces:**
- Consumes: additive `ScheduleFamilyInfo`, `CriterionStrength`, and new claims.
- Produces deterministic text output only; exit codes remain Phase 1 compatible.

- [ ] **Step 1: Write RED CLI rendering tests**

Expected output additions should include lines such as:

```text
Schedule family: polynomial
Canonical family: polynomial
Family parameters: c=1/4, offset=2, p=2, target=zero
...
path2_consensus              DISPROVED
  criterion: NECESSARY_AND_SUFFICIENT
path2_nodewise_convergence   PROVED
```

For an alternating schedule, preserve user family `alternating` while showing canonical family `periodic`.

- [ ] **Step 2: Preserve exit taxonomy**

Verify:

- theorem verdicts including DISPROVED/UNKNOWN -> exit `0`;
- malformed family input -> `2`;
- certificate compile/timeout/generation failure -> `3`;
- provenance/internal invariant error -> `4`.

- [ ] **Step 3: Implement deterministic rendering**

Sort exact parameter keys. Print `criterion:` only when `criterion_strength` is present. Do not print a numerical product estimate.

- [ ] **Step 4: Add exact fixtures**

Create all Phase 2 TOML fixtures with rational strings only. Each fixture must document one regime through its file name and contain no redundant data outside the existing model schema.

- [ ] **Step 5: Run GREEN CLI/model regression**

```bash
python -m unittest \
  tests.test_narrative_analyzer_cli \
  tests.test_narrative_analyzer_model
bash tools/check_narrative_analyzer.sh
```

Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add narrative_analyzer/cli.py tests/test_narrative_analyzer_cli.py \
  tests/fixtures/narrative_analyzer
git commit -m "feat(analyzer): render schedule family classifications"
```

---

### Task 9: Add real Lean golden certificates for every theorem-bearing family regime

**Files:**
- Create: `tests/test_narrative_analyzer_phase2_golden.py`
- Modify: `tools/check_narrative_analyzer.sh` only if its discovery pattern does not already include the new file.

**Interfaces:**
- Consumes: real `LeanCertificateRunner`, all Phase 2 fixtures, family certificate builder.
- Produces end-to-end evidence that Python routing plus generated source compiles against the real Lean theorem layer.

- [ ] **Step 1: Write real golden tests**

For each fixture, call the real analyzer with the real runner and assert the compiled verdict/provenance. Required cases:

1. harmonic target zero p=1 consensus;
2. harmonic target one p=1 consensus by absolute product despite `alpha -> 1`;
3. polynomial p=2 target zero stable non-consensus + nodewise convergence;
4. polynomial p=2 target one oscillatory non-convergence;
5. exponential target zero stable non-consensus;
6. exponential target one oscillatory non-convergence;
7. periodic interior consensus;
8. periodic boundary non-consensus;
9. alternating contracting consensus;
10. finite piecewise + interior constant tail consensus;
11. unequal Path2 exposures -> product iff UNKNOWN without compile failure;
12. equal initial beliefs -> direct consensus theorem.

Every PROVED/DISPROVED claim must assert a nonempty theorem provenance string containing the expected public theorem name.

- [ ] **Step 2: Add backwards-compatibility goldens**

Re-run all existing Phase 1 fixtures through the unchanged public CLI/analyzer API. Assert prior statuses and exact values remain unchanged except for additive family metadata.

- [ ] **Step 3: Add deliberate broken-certificate regression**

Corrupt one Phase 2 generated certificate in a controlled test and verify `CertificateCompileError`, not `UNKNOWN`.

- [ ] **Step 4: Run the analyzer gate and record exact count**

```bash
bash tools/check_narrative_analyzer.sh
```

Expected: all tests PASS with zero skipped theorem-bearing goldens. Record the actual final test count in the PR body later; do not hard-code an expected total before implementation.

- [ ] **Step 5: Run focused Lean gate**

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
```

Expected: PASS.

- [ ] **Step 6: Commit Task 9**

```bash
git add tests/test_narrative_analyzer_phase2_golden.py tools/check_narrative_analyzer.sh
git commit -m "test(analyzer): verify Phase 2 Lean family certificates"
```

---

### Task 10: Integrate additive CI and complete exact-head acceptance verification

**Files:**
- Modify: `.github/workflows/proof.yml`
- Modify: issue #99 status comment through GitHub after evidence exists.

**Interfaces:**
- Consumes: all focused gates from Tasks 1-9.
- Produces permanent CI evidence and review-ready PR state; no automatic merge.

- [ ] **Step 1: Add the focused theorem gate before the analyzer gate**

In `.github/workflows/proof.yml`, keep existing order and add:

```yaml
      - name: Receptivity schedule classifier proofs
        timeout-minutes: 20
        run: bash tools/check_fitness_abm_schedule_classifier.sh

      - name: PathN exposure consensus analyzer
        timeout-minutes: 20
        run: bash tools/check_narrative_analyzer.sh
```

Do not remove, weaken, or merge the existing `BB finite-path convergence` gate.

- [ ] **Step 2: Run local/focused acceptance suite**

Run in this order:

```bash
bash tools/check_fitness_abm_schedule_classifier.sh
bash tools/check_fitness_abm_pathn.sh
bash tools/check_narrative_analyzer.sh
lake build
```

Expected: all PASS.

- [ ] **Step 3: Run all Python tests used by the proof workflow**

Use the same Python command currently configured in `.github/workflows/proof.yml`; do not invent a narrower replacement. Expected: PASS.

- [ ] **Step 4: Audit scope before push**

Compare against exact base `b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b` and confirm changes are limited to:

- new schedule-classifier Lean theorem/test files;
- Phase 2 analyzer files/tests/fixtures;
- additive proof workflow step;
- committed spec/plan.

Confirm no changes to `FitnessABMPathNExposure.step` or unrelated BB/world-studio modules.

- [ ] **Step 5: Commit CI integration**

```bash
git add .github/workflows/proof.yml
git commit -m "ci: verify receptivity schedule classifier"
```

- [ ] **Step 6: Push implementation branch and open a draft PR against `proof/narrative-dynamics-v0`**

PR body must include:

- issue #99;
- spec and plan paths;
- exact base SHA;
- trust-boundary statement;
- theorem families added;
- distinction between N&S Path2 routes and sufficient PathN routes;
- focused local verification results.

Keep the PR draft until exact-head CI and code review pass.

- [ ] **Step 7: Require exact-head GitHub CI evidence**

For the exact final PR head SHA, require:

- `World Studio`: success;
- `proof / Select proof event`: success;
- `proof / Python tests`: success;
- `proof / Lean proof`: success;
- `Build Lean library`: success;
- `BB finite-path convergence`: success;
- `Receptivity schedule classifier proofs`: success;
- `PathN exposure consensus analyzer`: success;
- every downstream proof/trust/story/testimony step: success.

If any code/test/workflow commit is added, old exact-head evidence is invalid and must be regenerated.

- [ ] **Step 8: Perform formal review before ready-for-review**

Review base -> exact head for:

- theorem assumptions and overclaiming;
- mixing-mass identity and zero-factor handling;
- p=1 vs p>=2 polynomial regimes;
- target-zero stable vs target-one oscillatory semantics;
- periodic/piecewise concrete `e0` handling;
- equal-exposure/nontrivial-belief iff assumptions;
- no Python proof authority;
- source-injection safety;
- fail-closed runner/provenance behavior;
- Phase 1 backward compatibility;
- additive CI only.

Critical/Important findings must be fixed before proceeding, followed by new exact-head CI.

- [ ] **Step 9: Mark PR ready only after all gates pass**

Update #99 with exact head, workflow run IDs, focused test counts, theorem coverage, scope compare, and review findings. Mark PR Ready for review. Do not merge automatically; merge remains a separate explicit user gate.

---

## Final Acceptance Matrix

Before declaring Phase 2 implementation complete, verify all rows:

| Requirement | Required evidence |
|---|---|
| Phase 1 inputs remain compatible | existing analyzer test/golden suite green |
| Closed exact family DSL only | parser rejection tests + source-injection tests |
| Harmonic parameterized route | real Lean golden, not only old named fixture |
| Polynomial p=1 consensus | public Lean theorem + real certificate |
| Polynomial p>=2 stable/oscillatory split | public Lean theorems + zero/one target goldens |
| Exponential stable/oscillatory split | public Lean theorems + real goldens |
| Periodic exact classification | cycle theorem + interior/boundary goldens |
| Alternating reuses periodic theorem | canonical family metadata + theorem provenance |
| Piecewise finite-prefix/tail | concrete `e0` theorem + golden |
| Equal beliefs handled directly | direct Lean theorem + golden |
| Unequal exposures do not misuse iff | UNKNOWN regression |
| `PROVED/DISPROVED/UNKNOWN` unchanged | result tests + CLI tests |
| Criterion strength is metadata only | immutable result tests |
| Python never proves asymptotics | architecture/source review + real certificate requirement |
| Compile/timeouts fail closed | runner/golden failure tests |
| PathN not upgraded to iff | review + claim metadata tests |
| Existing trust/resource gates intact | exact-head proof workflow success |

## Implementation Stop Conditions

Stop implementation and return to the design/spec gate instead of improvising if any of these occur:

- the required polynomial/exponential infinite-product theorem cannot be stated/proved with the frozen family semantics without materially changing assumptions;
- a proposed theorem would need floating-point or numerical truncation as proof authority;
- the current `path2_consensus_iff_product_tendsto_zero` assumptions are insufficient for the intended conclusion and require semantic changes to the executable model;
- family validity requires a broader symbolic expression language;
- implementation would require weakening existing trust gates;
- a PathN necessary-and-sufficient claim appears necessary to satisfy Phase 2 acceptance.

A stop condition is a design finding, not permission to silently broaden scope.
