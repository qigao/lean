# BB Exposure-Dependent Learning V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate finite-path exact-rational model whose learning rate is a declared function of cumulative exposure, with constant-schedule compatibility, boundedness/exposure invariants, and an explicit theorem-level counterexample to baseline exposure independence.

**Architecture:** Reuse only the canonical `FitnessABMPathN` topology. The new module computes its own explicitly named exposure-dependent step: determine current broadcasters, accumulate current incoming broadcasts into `e'`, then use `α(e')` for the same belief update. Baseline `NetworkPropagation` and `FitnessABMPathN` remain untouched.

**Tech Stack:** Lean 4.32.0, mathlib, exact `Rat`, existing `FitnessABMPathN` topology, #82 response-parameter bridge, bounded trust audit/GitHub proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-exposure-learning-v1-design.md`

## Global Constraints

- Production implementation starts only after #82's parameterized response surface is stable enough to state the constant-schedule bridge.
- `ExposureParameters` has `receptivityAt : Nat → Rat` and rational `threshold`.
- `Valid` requires every `receptivityAt e` and `threshold` lie in `[0,1]`; monotonicity is not part of validity.
- One step uses pre-step broadcasters, computes incoming count, sets `e' = e + incoming`, then uses `α(e')` for that same update.
- With no broadcasting neighbors, belief remains unchanged.
- Baseline `NetworkPropagation`/`FitnessABMPathN` definitions and the theorem `propagate_independent_exposures` remain unchanged.
- No psychological interpretation, arbitrary-graph convergence, or float-runtime equivalence claim.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, or unbounded Lean resources.
- Direct Lean commands retain `timeout --kill-after=10s 240s`.

---

### Task 1: Separate exposure-model state and exact step

**Files:**
- Create RED: `NarrativeDynamics/Tests/FitnessABMPathNExposure.lean`
- Create GREEN: `NarrativeDynamics/Core/FitnessABMPathNExposure.lean`

**Interfaces:**

```lean
structure ExposureParameters where
  receptivityAt : Nat → Rat
  threshold : Rat

def ExposureParameters.Valid (p : ExposureParameters) : Prop :=
  (∀ e, 0 ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1) ∧
  0 ≤ p.threshold ∧ p.threshold ≤ 1

structure AgentState where
  belief : Rat
  exposure : Nat

abbrev State (n : Nat) := Fin n → AgentState

def broadcasting (p : ExposureParameters) (s : State n) (i : Fin n) : Bool

def incoming (p : ExposureParameters) (s : State n) (i : Fin n) : Nat

def step (p : ExposureParameters) (n : Nat) (s : State n) : State n

def beliefs (s : State n) : FitnessABMPathN.Beliefs n := fun i => (s i).belief
```

- [ ] Write RED consumers requiring the declarations and an equality-at-threshold broadcaster on `Fin 2`.
- [ ] Verify bounded RED fails on missing module/declarations only.
- [ ] Implement broadcaster collection from `FitnessABMPathN.neighbors n i`; exact mean over broadcasting neighbors; `e' = old exposure + incoming`; if incoming zero preserve belief, otherwise update using `p.receptivityAt e'`.
- [ ] Add a concrete consumer proving current incoming exposure affects the same step's selected `α(e')`, not the next step.
- [ ] Run bounded GREEN and source trust audit.
- [ ] Commit `feat(lean): add exposure-dependent path learning model`.

### Task 2: Constant-schedule bridge to #82/baseline

**Files:**
- Modify core/test exposure files.

**Interfaces:**

```lean
def constant (alpha tau : Rat) : ExposureParameters := ⟨fun _ => alpha, tau⟩

theorem constant_beliefStep ... :
  beliefs (step (constant alpha tau) n s) =
    FitnessABMPathNParameters.beliefStep ⟨alpha, tau⟩ n (beliefs s)

theorem half_beliefStep ... :
  beliefs (step (constant (1/2) (1/2)) n s) =
    FitnessABMPathN.beliefStep n (beliefs s)
```

- [ ] Add RED generic bridge consumers; whole-state equality is intentionally not required because exposure is explicit state.
- [ ] Verify missing bridge RED.
- [ ] Prove constant schedule makes belief arithmetic independent of the stored prior exposure value except through unchanged constant `α`; compose `half_beliefStep` through #82's `beliefStep_half`.
- [ ] Run new model tests + #82 parameter tests + fixed PathN tests.
- [ ] Commit `proof(lean): bridge constant exposure schedule to baseline`.

### Task 3: Positive invariants

**Files:**
- Modify core/test exposure files.

**Interfaces:**

```lean
def BeliefsBounded (s : State n) : Prop := ∀ i, 0 ≤ (s i).belief ∧ (s i).belief ≤ 1

theorem exposure_mono ... : (s i).exposure ≤ (step p n s i).exposure

theorem beliefs_bounded_step
    (hvalid : p.Valid) (hbounded : BeliefsBounded s) :
  BeliefsBounded (step p n s)
```

- [ ] Add RED consumers for both theorems including a case with two incoming broadcasters.
- [ ] Prove `exposure_mono` by `Nat.le_add_right`/incoming nonnegativity.
- [ ] Prove boundedness: unchanged branch is immediate; update branch uses exact mean of bounded broadcaster beliefs in `[0,1]` and convex coefficient `α(e')∈[0,1]`.
- [ ] Do not claim belief monotonicity.
- [ ] Run bounded GREEN/trust reports.
- [ ] Commit `proof(lean): bound exposure-dependent path updates`.

### Task 4: Explicit non-equivalence counterexample

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPathNExposure.lean`

**Fixture contract:** path of two or three vertices; identical graph, beliefs and threshold; receiver differs only in starting cumulative exposure. Use schedule `α(0)=1/4`, `α(e)=3/4` for positive `e`; provide a broadcasting neighbor with different belief.

- [ ] Add concrete definitions `lowExposureState` and `highExposureState` whose `beliefs` projections are equal before the step.
- [ ] Prove the targeted receiver's next beliefs are unequal after one exposure-dependent step.
- [ ] Also state the baseline control: applying `FitnessABMPathNParameters.beliefStep` to the shared belief vector gives one common result independent of either exposure history.
- [ ] Print axioms for the counterexample and positive invariants.
- [ ] Run bounded test and trust audit.
- [ ] Commit `test(lean): prove exposure-history counterexample`.

### Task 5: Permanent gate and review

**Files:**
- Modify: `tools/check_fitness_abm_pathn.sh`

- [ ] Add exposure core/test source audit, bounded build, bounded test log, and mandatory trust reports for constant bridge, boundedness, exposure monotonicity, and counterexample.
- [ ] Retain every existing FiniteConsensus/PathN/#82 requirement; do not replace them.
- [ ] Run `bash -n`, the full PathN gate, Path4 gate, and `git diff --check`.
- [ ] Open PR against the then-current merged `proof/narrative-dynamics-v0` only after #82 is integrated or retarget/rebase without rewriting reviewed #84 commits.
- [ ] Require exact-head proof + World Studio CI and independent review; no merge with Critical/Important finding.
- [ ] PR explicitly says this is a separate model and baseline exposure independence remains true for the baseline model.
