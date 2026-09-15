# Temporal Credit Formal Proof Design

> Status: design committed for review before implementation.
>
> Repository: `qigao/lean`
>
> Branch: `formal/temporal-credit-v1`
>
> Base: `feature/percept-cognition-memory-v15-1` at `4266a3e41ecbca43563d9033c05b329c2527f787`
>
> Date: 2026-09-15
>
> Consumer experiment: `qigao/yolo-motion-perception`, branch `experiment/neural-state-machine`, Phase 3C true temporal credit assignment.

## 1. Purpose

Phase 3B in `yolo-motion-perception` established a corrected delayed-feedback protocol with real overlap between later decisions and earlier pending rewards. Its learner nevertheless stores unresolved action credits in FIFO order, so a delivered scalar reward is still matched mechanically to the oldest unresolved decision.

Phase 3C removes that credit-identity assumption. One trial contains one causal action, zero or more later distractor actions, and exactly one terminal scalar reward. The learner receives the observable decision sequence and the terminal scalar reward but is not told which earlier action caused the outcome.

This Lean work proves four narrow mathematical properties needed before a Python Phase 3C implementation is accepted as a true temporal-credit experiment:

1. terminal reward is invariant under substitution of distractor actions when the causal action is unchanged;
2. learner-visible input is invariant under changes to internal causal-label metadata that do not change the observable trace;
3. terminal TD(0) has zero direct credit to the earlier causal action when at least one later distractor action exists;
4. an accumulating eligibility trace retains causal credit with coefficient `(gamma * lambda)^d` after `d` later decisions.

These theorems establish protocol and mechanism properties only. They do not prove that the Python/NumPy implementation conforms to the model, that an empirical learner converges, that TD(0) cannot learn indirectly through shared representations, or that eligibility traces must pass a behavioral benchmark.

## 2. Repository and toolchain

The proof lives in `qigao/lean`, not in the Python experiment repository.

The base branch already provides:

- Lean `4.32.0` via `lean-toolchain`;
- Mathlib `v4.32.0` via `lakefile.toml`;
- the `NarrativeDynamics` Lean library;
- the established split between `NarrativeDynamics/Core` and `NarrativeDynamics/Tests`.

The new proof must not change the Lean or Mathlib version.

## 3. Module boundaries

Create exactly two Lean modules:

```text
NarrativeDynamics/Core/TemporalCredit.lean
NarrativeDynamics/Tests/TemporalCredit.lean
```

`Core/TemporalCredit.lean` contains the mathematical model and public theorems.

`Tests/TemporalCredit.lean` contains executable examples, boundary cases, theorem instantiations, and axiom audits. It must not redefine the core model.

Add the core module to the root `NarrativeDynamics.lean` import surface only after the core file compiles independently.

## 4. Formal model

### 4.1 Trial structure

Use abstract finite types for cues and actions where possible. The theorem statements must not depend on concrete game actions.

A trial separates learner-visible data from environment-internal causal metadata.

Conceptually:

```lean
structure Trial (Cue Action Obs Reward : Type) where
  cue : Cue
  causalAction : Action
  distractors : List Action
  observations : List Obs
  terminalReward : Reward
  causalSlot : Nat
```

The implementation may use a smaller equivalent structure if that gives stronger type invariants. In particular, `causalSlot` is internal metadata and must not be needed to compute the learner view.

The Phase 3C registered protocol uses the causal action as the first decision and `d = distractors.length` later distractor decisions before the unique terminal reward. The formal model may encode this by construction instead of storing a redundant `causalSlot = 0` proof.

### 4.2 Reward contract

Define the terminal reward from the cue and causal action only:

```lean
terminalReward : Cue -> Action -> Reward
```

Distractor actions are intentionally excluded from this function.

A trial reward accessor is therefore equivalent to:

```lean
def trialReward (rewardFn : Cue -> Action -> Reward) (t : Trial ...) : Reward :=
  rewardFn t.cue t.causalAction
```

This makes the intended causal contract explicit and allows a theorem about distractor substitution rather than relying on an empirical correlation.

### 4.3 Learner view

Define a projection that contains only learner-visible values. The minimal proof model should expose the observable decision trace and the terminal scalar reward while excluding environment-internal causal metadata.

Conceptually:

```lean
structure LearnerView (Obs Reward : Type) where
  observations : List Obs
  terminalReward : Reward
```

and:

```lean
def learnerView (t : Trial ...) : LearnerView Obs Reward := ...
```

The projection must not contain `causalSlot`, `causalActionId`, reward delay, due step, environment timestamp, or any equivalent identity channel.

This theorem is a type/projection non-interference property. It does not by itself prove that the Python implementation has no leakage; later conformance work must map Python inputs to this formal projection.

### 4.4 Direct-credit model

The formal proof distinguishes *direct credit* from possible indirect effects through shared parameters.

For a trajectory with one causal decision followed by `d` distractor decisions, define positions `0 .. d`, with the terminal action at position `d`.

Terminal TD(0) direct credit assigns weight `1` to the terminal action and `0` to every earlier action.

Conceptually:

```lean
def td0DirectCredit (d pos : Nat) : Real :=
  if pos = d then 1 else 0
```

The causal action is at position `0`.

This is intentionally not a full neural-network update model. It proves only which temporal position is directly selected by the terminal TD(0) update rule.

### 4.5 Eligibility-trace model

For accumulating eligibility with scalar decay factor `gamma * lambda`, isolate the coefficient carried by the causal action through later decisions.

Use a recurrence equivalent to:

```lean
def causalTraceCoeff (gamma lambda : Real) : Nat -> Real
  | 0 => 1
  | d + 1 => (gamma * lambda) * causalTraceCoeff gamma lambda d
```

The formal theorem must derive the closed form rather than merely restating it:

```text
causalTraceCoeff gamma lambda d = (gamma * lambda)^d
```

A nonzero-credit corollary should require explicit nonzero hypotheses on `gamma` and `lambda`.

## 5. Required theorems

The implementation may adjust argument order or namespace qualification, but the mathematical statements must remain equivalent to the following.

### Theorem 1: distractor substitution preserves reward

For any cue, causal action, and two distractor sequences, changing only the distractors does not change terminal reward.

Target statement shape:

```lean
theorem reward_invariant_under_distractor_substitution
    (rewardFn : Cue -> Action -> Reward)
    (cue : Cue)
    (causal : Action)
    (xs ys : List Action) :
    trialReward rewardFn (mkTrial cue causal xs ...) =
    trialReward rewardFn (mkTrial cue causal ys ...) := by
  ...
```

The concrete constructor can differ, but the proof must quantify over arbitrary distractor sequences.

What this proves: the formal task contract identifies one causal action for the terminal reward.

What this does not prove: a future Python reward function actually satisfies the same contract unless conformance is checked.

### Theorem 2: learner view contains no causal-label channel

Changing only internal causal metadata while holding the observable trace and terminal scalar reward fixed must leave the learner view unchanged.

Target statement shape:

```lean
theorem learner_view_independent_of_causal_label
    (t : Trial ...)
    (label1 label2 : CausalLabel) :
    learnerView (t.withCausalLabel label1) =
    learnerView (t.withCausalLabel label2) := by
  ...
```

If the final model encodes causal position by construction and therefore has no mutable label field, prove the stronger projection theorem that any two internal records with equal observable fields have equal `learnerView` values.

What this proves: the formal learner interface cannot observe the internal label through the projection.

What this does not prove: semantic impossibility of inferring causality from observations, or absence of leakage in external Python code.

### Theorem 3: terminal TD(0) gives zero direct credit to an earlier causal action

For any positive temporal distance `d > 0`, the causal action at position `0` is not the terminal position and receives zero direct terminal-TD(0) credit.

Required statement:

```lean
theorem terminal_td0_zero_direct_causal_credit
    (d : Nat)
    (hd : 0 < d) :
    td0DirectCredit d 0 = 0 := by
  ...
```

A companion theorem for the terminal position should prove:

```lean
td0DirectCredit d d = 1
```

What this proves: under the formal direct-credit rule, terminal TD(0) does not directly update the earlier causal temporal position when a later action exists.

What this does not prove: TD(0) has zero *indirect* effect on representations or predictions when parameters/features are shared, nor that TD(0) must fail empirically.

### Theorem 4: eligibility trace retains causal weight `(gamma * lambda)^d`

Required closed-form theorem:

```lean
theorem causal_trace_coeff_closed_form
    (gamma lambda : Real)
    (d : Nat) :
    causalTraceCoeff gamma lambda d = (gamma * lambda) ^ d := by
  ...
```

Required nonzero corollary:

```lean
theorem causal_trace_coeff_ne_zero
    (gamma lambda : Real)
    (d : Nat)
    (hgamma : gamma != 0)
    (hlambda : lambda != 0) :
    causalTraceCoeff gamma lambda d != 0 := by
  ...
```

If Lean syntax uses `≠`, use the idiomatic form.

What this proves: the eligibility mechanism has a nonzero direct temporal credit path back to the causal decision for every finite `d` when both decay factors are nonzero.

What this does not prove: the credit has useful magnitude, has the correct sign after multiplying by a TD error, or guarantees convergence.

## 6. Boundary examples

`NarrativeDynamics/Tests/TemporalCredit.lean` must instantiate at least:

```text
d = 0
d = 1
d = 3
d = 5
```

Required checks:

- TD(0) causal direct credit is `1` at `d=0`;
- TD(0) causal direct credit is `0` at `d=1,3,5`;
- eligibility coefficient is `1` at `d=0`;
- eligibility coefficient is `gamma * lambda` at `d=1`;
- eligibility coefficient is `(gamma * lambda)^3` at `d=3`;
- eligibility coefficient is `(gamma * lambda)^5` at `d=5`;
- two different distractor lists preserve reward for the same cue/causal action;
- two different internal causal labels with the same observable projection produce identical learner views.

## 7. Proof-quality requirements

The proof implementation must satisfy all of the following:

- no `sorry`;
- no `admit`;
- no custom axioms introduced for these results;
- compile under the repository's existing Lean 4.32.0 / Mathlib v4.32.0 toolchain;
- each public theorem is checked in `NarrativeDynamics/Tests/TemporalCredit.lean`;
- run `#print axioms` on all four primary theorems and the nonzero trace corollary;
- record the exact axiom output in CI/review evidence rather than assuming axiom-freedom from source inspection.

The proof should prefer constructive/simple definitions over tactic-heavy automation. The target properties are small enough that the model should make illegal information flow difficult to express.

## 8. Relationship to Phase 3C Python work

This proof is a mathematical specification, not an implementation verifier.

The eventual Phase 3C Python design must define a conformance boundary mapping:

```text
Python trial / learner inputs
        -> formal observable learner view
Python reward function
        -> formal reward contract
Python terminal-TD0 arm
        -> formal td0DirectCredit semantics
Python eligibility-trace arm
        -> formal causalTraceCoeff recurrence
```

Until those mappings are tested, the valid claim is:

> The formal Phase 3C model has the intended causal/reward/interface properties and the stated TD(0)/eligibility direct-credit properties.

The invalid claim is:

> Lean has proved the NumPy implementation correct.

## 9. Acceptance

The formal proof task is complete only when:

1. `NarrativeDynamics/Core/TemporalCredit.lean` compiles;
2. `NarrativeDynamics/Tests/TemporalCredit.lean` compiles;
3. all four required primary theorems are present with the stated semantics;
4. the TD(0) terminal-position companion theorem is present;
5. the eligibility nonzero-credit corollary is present;
6. registered examples `d = 0,1,3,5` pass;
7. there are no `sorry`, `admit`, or new axioms in the module;
8. `#print axioms` output is reviewed;
9. exact-head CI passes on the proof branch;
10. no Python Phase 3C production code is modified as part of this proof task.

## 10. Out of scope

This first formal layer does not prove:

- empirical learning success or convergence;
- sample complexity;
- stability across random seeds;
- gradient correctness in NumPy;
- equivalence between Python floating-point arithmetic and real-number Lean semantics;
- actor-critic, replay, BPTT, STDP, or recurrent-weight learning;
- variable/out-of-order reward schedules;
- multi-causal-action decomposition;
- causal discovery from arbitrary trajectories.

Those require separate specs if Phase 3C advances beyond the single-causal-action terminal-reward experiment.
