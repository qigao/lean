# Temporal Credit Formal Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a small Lean theory proving the Phase 3C reward-causality, learner-view non-interference, terminal-TD(0) direct-credit, and eligibility-trace temporal-credit properties before any Python Phase 3C production implementation.

**Architecture:** Add one focused mathematical module under `NarrativeDynamics/Core` and one focused theorem/example module under `NarrativeDynamics/Tests`. The core model separates environment-internal causal metadata from the learner-visible projection, models terminal TD(0) only at the level of direct temporal-position credit, and models accumulating eligibility through a scalar recurrence whose closed form is proved. Existing Lean 4.32.0 / Mathlib v4.32.0 infrastructure remains unchanged.

**Tech Stack:** Lean 4.32.0, Mathlib v4.32.0, Lake, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-15-temporal-credit-proof-design.md`

## Global Constraints

- Work only in `qigao/lean` branch `formal/temporal-credit-v1`.
- Do not modify `lean-toolchain` or `lakefile.toml`.
- Do not modify Python Phase 3C production code in `qigao/yolo-motion-perception`.
- Add exactly one core module: `NarrativeDynamics/Core/TemporalCredit.lean`.
- Add exactly one theorem/example module: `NarrativeDynamics/Tests/TemporalCredit.lean`.
- No `sorry`, `admit`, or custom axioms.
- Theorems prove properties of the formal model only; do not claim NumPy/Python conformance or empirical convergence.
- The TD(0) theorem is about **direct temporal-position credit**, not indirect effects through shared representations.
- The eligibility theorem proves retained coefficient `(gamma * lambda)^d`, not useful magnitude or empirical success.
- Registered boundary distances are exactly `d = 0, 1, 3, 5`.
- Exact-head GitHub CI must pass before the proof task is considered complete.

---

## File map

- `NarrativeDynamics/Core/TemporalCredit.lean` — formal trial, reward projection, learner-view projection, terminal TD(0) direct-credit function, eligibility-trace recurrence, and all public theorems.
- `NarrativeDynamics/Tests/TemporalCredit.lean` — theorem instantiations for `d = 0,1,3,5`, distractor/label examples, and `#print axioms` audits.
- `NarrativeDynamics.lean` — export the new core module after it compiles independently.
- `.github/workflows/proof.yml` — run the new theorem test file in the existing `Lean theorem tests` step.

---

### Task 1: Formal trial, reward contract, and learner-view non-interference

**Files:**
- Create: `NarrativeDynamics/Core/TemporalCredit.lean`
- Create: `NarrativeDynamics/Tests/TemporalCredit.lean`

**Interfaces:**
- Produces:
  - `NarrativeDynamics.TemporalCredit.Trial`
  - `NarrativeDynamics.TemporalCredit.LearnerView`
  - `NarrativeDynamics.TemporalCredit.trialReward`
  - `NarrativeDynamics.TemporalCredit.learnerView`
  - `NarrativeDynamics.TemporalCredit.withDistractors`
  - `NarrativeDynamics.TemporalCredit.withCausalLabel`
  - `reward_invariant_under_distractor_substitution`
  - `learner_view_independent_of_causal_label`
- Consumes: only Mathlib and standard Lean `List`/structure support.

- [ ] **Step 1: Write the failing theorem-test surface first**

Create `NarrativeDynamics/Tests/TemporalCredit.lean` with imports and examples that reference the not-yet-existing module:

```lean
import NarrativeDynamics.Core.TemporalCredit

open NarrativeDynamics.TemporalCredit

namespace NarrativeDynamics.TemporalCredit.Tests

inductive Cue where
  | left
  | right
  deriving DecidableEq

inductive Action where
  | stay
  | move
  deriving DecidableEq

inductive Label where
  | causal
  | hidden
  deriving DecidableEq

abbrev Obs := Nat
abbrev Reward := Int

private def rewardFn : Cue → Action → Reward
  | .left, .stay => 1
  | .right, .move => 1
  | _, _ => -1

private def baseTrial : Trial Cue Action Obs Label :=
  {
    cue := .left
    causalAction := .stay
    distractors := [.move]
    observations := [10, 20]
    causalLabel := .causal
  }

example :
    trialReward rewardFn (withDistractors baseTrial [.stay, .move]) =
      trialReward rewardFn (withDistractors baseTrial [.move, .move, .stay]) := by
  exact reward_invariant_under_distractor_substitution
    rewardFn baseTrial [.stay, .move] [.move, .move, .stay]

example :
    learnerView rewardFn (withCausalLabel baseTrial .causal) =
      learnerView rewardFn (withCausalLabel baseTrial .hidden) := by
  exact learner_view_independent_of_causal_label rewardFn baseTrial .causal .hidden

end NarrativeDynamics.TemporalCredit.Tests
```

- [ ] **Step 2: Run the new test file and verify RED**

Run:

```bash
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: FAIL because `NarrativeDynamics.Core.TemporalCredit` does not exist.

- [ ] **Step 3: Implement the minimal formal data/projection model**

Create `NarrativeDynamics/Core/TemporalCredit.lean`:

```lean
import Mathlib

namespace NarrativeDynamics.TemporalCredit

structure Trial (Cue Action Obs Label : Type) where
  cue : Cue
  causalAction : Action
  distractors : List Action
  observations : List Obs
  causalLabel : Label

structure LearnerView (Obs Reward : Type) where
  observations : List Obs
  terminalReward : Reward
  deriving DecidableEq

def trialReward
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label) : Reward :=
  rewardFn trial.cue trial.causalAction

def learnerView
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label) : LearnerView Obs Reward :=
  {
    observations := trial.observations
    terminalReward := trialReward rewardFn trial
  }

def withDistractors
    (trial : Trial Cue Action Obs Label)
    (distractors : List Action) : Trial Cue Action Obs Label :=
  { trial with distractors := distractors }

def withCausalLabel
    (trial : Trial Cue Action Obs Label)
    (label : Label) : Trial Cue Action Obs Label :=
  { trial with causalLabel := label }

theorem reward_invariant_under_distractor_substitution
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label)
    (xs ys : List Action) :
    trialReward rewardFn (withDistractors trial xs) =
      trialReward rewardFn (withDistractors trial ys) := by
  rfl

theorem learner_view_independent_of_causal_label
    (rewardFn : Cue → Action → Reward)
    (trial : Trial Cue Action Obs Label)
    (label₁ label₂ : Label) :
    learnerView rewardFn (withCausalLabel trial label₁) =
      learnerView rewardFn (withCausalLabel trial label₂) := by
  rfl

end NarrativeDynamics.TemporalCredit
```

This definition deliberately excludes `causalLabel` and `distractors` from the reward function, and excludes `causalLabel` from `LearnerView`.

- [ ] **Step 4: Run Task 1 tests and verify GREEN**

Run:

```bash
lake env lean NarrativeDynamics/Core/TemporalCredit.lean
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: both commands exit 0.

- [ ] **Step 5: Confirm no forbidden proof holes**

Run:

```bash
if grep -R -n -E '\b(sorry|admit)\b' \
  NarrativeDynamics/Core/TemporalCredit.lean \
  NarrativeDynamics/Tests/TemporalCredit.lean; then
  exit 1
fi
```

Expected: no matches and exit 0.

- [ ] **Step 6: Commit Task 1**

```bash
git add NarrativeDynamics/Core/TemporalCredit.lean \
        NarrativeDynamics/Tests/TemporalCredit.lean
git commit -m "feat(lean): model temporal credit information boundary"
```

---

### Task 2: Prove terminal TD(0) direct-credit reachability

**Files:**
- Modify: `NarrativeDynamics/Core/TemporalCredit.lean`
- Modify: `NarrativeDynamics/Tests/TemporalCredit.lean`

**Interfaces:**
- Consumes: `NarrativeDynamics.TemporalCredit` namespace from Task 1.
- Produces:
  - `td0DirectCredit : Nat → Nat → ℝ`
  - `terminal_td0_zero_direct_causal_credit`
  - `terminal_td0_terminal_credit`

- [ ] **Step 1: Add failing TD(0) examples to the test module**

Append inside `namespace NarrativeDynamics.TemporalCredit.Tests`:

```lean
example : td0DirectCredit 0 0 = 1 := by
  exact terminal_td0_terminal_credit 0

example : td0DirectCredit 1 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 1 (by decide)

example : td0DirectCredit 3 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 3 (by decide)

example : td0DirectCredit 5 0 = 0 := by
  exact terminal_td0_zero_direct_causal_credit 5 (by decide)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: FAIL because `td0DirectCredit` and the TD(0) theorems are undefined.

- [ ] **Step 3: Add the minimal direct-credit definition and proofs**

Append before the namespace `end` in the core module:

```lean
def td0DirectCredit (terminalPos pos : Nat) : ℝ :=
  if pos = terminalPos then 1 else 0

theorem terminal_td0_zero_direct_causal_credit
    (d : Nat)
    (hd : 0 < d) :
    td0DirectCredit d 0 = 0 := by
  have hne : (0 : Nat) ≠ d := Nat.ne_of_lt hd
  simp [td0DirectCredit, hne]

theorem terminal_td0_terminal_credit
    (d : Nat) :
    td0DirectCredit d d = 1 := by
  simp [td0DirectCredit]
```

- [ ] **Step 4: Run core and TD(0) examples**

Run:

```bash
lake env lean NarrativeDynamics/Core/TemporalCredit.lean
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: both exit 0; registered TD(0) examples pass for `d = 0,1,3,5`.

- [ ] **Step 5: Commit Task 2**

```bash
git add NarrativeDynamics/Core/TemporalCredit.lean \
        NarrativeDynamics/Tests/TemporalCredit.lean
git commit -m "feat(lean): prove terminal td0 direct credit boundary"
```

---

### Task 3: Prove eligibility-trace temporal-credit closed form

**Files:**
- Modify: `NarrativeDynamics/Core/TemporalCredit.lean`
- Modify: `NarrativeDynamics/Tests/TemporalCredit.lean`

**Interfaces:**
- Produces:
  - `causalTraceCoeff : ℝ → ℝ → Nat → ℝ`
  - `causal_trace_coeff_closed_form`
  - `causal_trace_coeff_ne_zero`

- [ ] **Step 1: Add failing eligibility examples**

Append to the test namespace:

```lean
example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 0 = 1 := by
  simpa using causal_trace_coeff_closed_form gamma lambda 0

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 1 = gamma * lambda := by
  simpa using causal_trace_coeff_closed_form gamma lambda 1

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 3 = (gamma * lambda) ^ 3 := by
  exact causal_trace_coeff_closed_form gamma lambda 3

example (gamma lambda : ℝ) :
    causalTraceCoeff gamma lambda 5 = (gamma * lambda) ^ 5 := by
  exact causal_trace_coeff_closed_form gamma lambda 5

example
    (gamma lambda : ℝ)
    (hgamma : gamma ≠ 0)
    (hlambda : lambda ≠ 0) :
    causalTraceCoeff gamma lambda 5 ≠ 0 := by
  exact causal_trace_coeff_ne_zero gamma lambda 5 hgamma hlambda
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: FAIL because the eligibility definitions/theorems are undefined.

- [ ] **Step 3: Implement the recurrence and closed-form proof**

Append to the core module:

```lean
def causalTraceCoeff (gamma lambda : ℝ) : Nat → ℝ
  | 0 => 1
  | d + 1 => (gamma * lambda) * causalTraceCoeff gamma lambda d

theorem causal_trace_coeff_closed_form
    (gamma lambda : ℝ)
    (d : Nat) :
    causalTraceCoeff gamma lambda d = (gamma * lambda) ^ d := by
  induction d with
  | zero =>
      simp [causalTraceCoeff]
  | succ d ih =>
      simp [causalTraceCoeff, ih, pow_succ, mul_comm]

theorem causal_trace_coeff_ne_zero
    (gamma lambda : ℝ)
    (d : Nat)
    (hgamma : gamma ≠ 0)
    (hlambda : lambda ≠ 0) :
    causalTraceCoeff gamma lambda d ≠ 0 := by
  rw [causal_trace_coeff_closed_form]
  exact pow_ne_zero d (mul_ne_zero hgamma hlambda)
```

- [ ] **Step 4: Compile the recurrence proof and all boundary examples**

Run:

```bash
lake env lean NarrativeDynamics/Core/TemporalCredit.lean
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: exit 0; `d=0,1,3,5` eligibility examples all pass.

If Lean reports a multiplication-normalization mismatch in the successor case, preserve the theorem statement and replace only the successor proof body with:

```lean
  | succ d ih =>
      rw [causalTraceCoeff, ih, pow_succ]
      ring
```

Then rerun the same two commands. Do not weaken the theorem or change the recurrence.

- [ ] **Step 5: Commit Task 3**

```bash
git add NarrativeDynamics/Core/TemporalCredit.lean \
        NarrativeDynamics/Tests/TemporalCredit.lean
git commit -m "feat(lean): prove eligibility temporal credit decay"
```

---

### Task 4: Add theorem instantiations and axiom audits

**Files:**
- Modify: `NarrativeDynamics/Tests/TemporalCredit.lean`

**Interfaces:**
- Consumes all public theorems from Tasks 1-3.
- Produces exact axiom-audit output for the primary theorem surface.

- [ ] **Step 1: Add explicit theorem instantiations for arbitrary distractors and labels**

Ensure the test module contains at least these two independently constructed examples:

```lean
private def longTrial : Trial Cue Action Obs Label :=
  {
    cue := .right
    causalAction := .move
    distractors := [.stay, .stay, .move, .stay, .move]
    observations := [1, 2, 3, 4, 5, 6]
    causalLabel := .causal
  }

example :
    trialReward rewardFn (withDistractors longTrial []) =
      trialReward rewardFn (withDistractors longTrial [.move, .stay, .move]) := by
  exact reward_invariant_under_distractor_substitution
    rewardFn longTrial [] [.move, .stay, .move]

example :
    learnerView rewardFn (withCausalLabel longTrial .causal) =
      learnerView rewardFn (withCausalLabel longTrial .hidden) := by
  exact learner_view_independent_of_causal_label rewardFn longTrial .causal .hidden
```

- [ ] **Step 2: Add axiom audits at the bottom of the test module**

Outside the test namespace or after reopening the core namespace, add:

```lean
#print axioms NarrativeDynamics.TemporalCredit.reward_invariant_under_distractor_substitution
#print axioms NarrativeDynamics.TemporalCredit.learner_view_independent_of_causal_label
#print axioms NarrativeDynamics.TemporalCredit.terminal_td0_zero_direct_causal_credit
#print axioms NarrativeDynamics.TemporalCredit.terminal_td0_terminal_credit
#print axioms NarrativeDynamics.TemporalCredit.causal_trace_coeff_closed_form
#print axioms NarrativeDynamics.TemporalCredit.causal_trace_coeff_ne_zero
```

- [ ] **Step 3: Run the proof test and capture exact axiom output**

Run:

```bash
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean \
  2>&1 | tee /tmp/temporal-credit-proof.log
```

Expected: exit 0 and six `#print axioms` reports. Review the output and confirm there are no project-defined custom axioms and no `sorryAx` dependency.

Run:

```bash
if grep -E 'sorryAx|axiom .*TemporalCredit' /tmp/temporal-credit-proof.log; then
  exit 1
fi
```

Expected: no matches.

- [ ] **Step 4: Re-run forbidden-hole scan**

```bash
if grep -R -n -E '\b(sorry|admit)\b' \
  NarrativeDynamics/Core/TemporalCredit.lean \
  NarrativeDynamics/Tests/TemporalCredit.lean; then
  exit 1
fi
```

Expected: no matches.

- [ ] **Step 5: Commit Task 4**

```bash
git add NarrativeDynamics/Tests/TemporalCredit.lean
git commit -m "test(lean): audit temporal credit theorem surface"
```

---

### Task 5: Export the core module and wire exact-head CI

**Files:**
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`

**Interfaces:**
- Consumes: compiling core/test modules from Tasks 1-4.
- Produces: root-library visibility and permanent exact-head CI coverage.

- [ ] **Step 1: Verify the new core module compiles independently before export**

Run:

```bash
lake env lean NarrativeDynamics/Core/TemporalCredit.lean
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Expected: both exit 0.

- [ ] **Step 2: Add the root import**

Append to `NarrativeDynamics.lean`:

```lean
import NarrativeDynamics.Core.TemporalCredit
```

- [ ] **Step 3: Add the permanent theorem-test CI command**

In `.github/workflows/proof.yml`, under the existing `Lean theorem tests` shell block, add:

```yaml
          lake env lean NarrativeDynamics/Tests/TemporalCredit.lean
```

Do not add a one-off workflow or separate measurement job.

- [ ] **Step 4: Run the full Lean library build**

Run:

```bash
lake build
```

Expected: exit 0.

- [ ] **Step 5: Run the complete new theorem test and axiom audit again**

Run:

```bash
lake env lean NarrativeDynamics/Tests/TemporalCredit.lean \
  2>&1 | tee /tmp/temporal-credit-proof-final.log
```

Expected: exit 0; all registered examples and six axiom reports succeed.

- [ ] **Step 6: Run the repository's existing theorem-test surface most affected by the root import**

Run:

```bash
lake env lean NarrativeDynamics/Tests/Learning.lean
lake env lean NarrativeDynamics/Tests/CognitivePipeline.lean
lake env lean NarrativeDynamics/Tests/EpistemicGoal.lean
```

Expected: all exit 0.

- [ ] **Step 7: Commit Task 5**

```bash
git add NarrativeDynamics.lean .github/workflows/proof.yml
git commit -m "ci(lean): verify temporal credit proofs"
```

---

### Task 6: Exact-head proof review and completion gate

**Files:**
- No planned source modifications.
- Modify proof files only if exact-head CI exposes a real proof/toolchain issue; do not weaken theorem statements to make CI pass.

**Interfaces:**
- Consumes: exact branch head after Task 5.
- Produces: final verified proof status.

- [ ] **Step 1: Push `formal/temporal-credit-v1` and record the exact head SHA**

```bash
git push origin formal/temporal-credit-v1
git rev-parse HEAD
```

Expected: the printed SHA is the head used by GitHub Actions.

- [ ] **Step 2: Require the existing `proof` workflow to pass on that exact SHA**

Acceptance requires:

```text
Lean proof     PASS
Python tests   PASS
```

The `Lean proof` job must include:

```text
Build Lean library      PASS
Lean theorem tests      PASS
TemporalCredit.lean     PASS as part of Lean theorem tests
```

Do not use a CI result from an ancestor commit as completion evidence.

- [ ] **Step 3: Review the exact-head `#print axioms` output**

From the `Lean theorem tests` job log, verify the six TemporalCredit theorem audits are present and contain no `sorryAx` or project-defined custom axiom.

Record separately if Lean reports standard logical dependencies such as `propext`, `Classical.choice`, or `Quot.sound`; these are not custom TemporalCredit axioms and must not be misreported as theorem-specific assumptions.

- [ ] **Step 4: Run a final scope diff against the approved base**

Run:

```bash
git diff --name-only 4266a3e41ecbca43563d9033c05b329c2527f787..HEAD
```

Expected implementation/proof changes are limited to:

```text
NarrativeDynamics/Core/TemporalCredit.lean
NarrativeDynamics/Tests/TemporalCredit.lean
NarrativeDynamics.lean
.github/workflows/proof.yml
docs/superpowers/specs/2026-09-15-temporal-credit-proof-design.md
docs/superpowers/plans/2026-09-15-temporal-credit-proof.md
```

No `yolo-motion-perception` Python production file may be part of this proof task.

- [ ] **Step 5: State the final claim narrowly**

The completion statement must say, in substance:

```text
The formal Phase 3C model proves distractor-invariant terminal reward,
learner-view independence from internal causal labels, zero direct causal
credit for terminal TD(0) at positive distance, and eligibility causal
coefficient (gamma * lambda)^d with a nonzero corollary.
```

It must also state:

```text
This does not yet prove Python/NumPy conformance, empirical convergence,
or guaranteed behavioral success.
```

No further commit is required unless the final review finds a concrete defect.
