# Fitness Distribution V23.6 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BB the only maintained attachment-model surface, then lift the existing exact finite BB replay kernel into an executable normalized distribution over all legal target traces with exact event probabilities and rational expectations.

**Architecture:** First remove the legacy alternate-model vocabulary without deleting the useful constant-fitness/common-scaling mathematics. Then add a schedule-length-indexed finite `TargetTrace` carrier in a new `FitnessDistribution` module, define trace probability and final-state evaluation from the existing `orderedMass` and `applyBirth`, and derive events/expectations by finite sums over that carrier. Reuse `continuationMass` as the normalization bridge so there is one stochastic kernel, not a second implementation.

**Tech Stack:** Lean `leanprover/lean4:v4.32.0`, pinned mathlib revision `81a5d257c8e410db227a6665ed08f64fea08e997`, exact `Rat`, `SimpleGraph (Fin n)`, existing `Targets`/`orderedMass`/`applyBirth`/`RunState`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-13-fitness-distribution-v23-6-design.md`

## Global Constraints

- BB is the only attachment-model identity in the maintained repository tree. Do not retain compatibility aliases for the removed legacy model vocabulary.
- Preserve the useful mathematics as BB-native constant-fitness and common-scaling properties.
- The probability experiment conditions on one valid typed initial state, fixed `m`, and a fixed positive newborn-fitness schedule. Only ordered target choices are random.
- All probabilities and expectations remain exact `Rat`; do not introduce floating point, `ENNReal`, PMF/Measure infrastructure, or a PRNG in V23.6.
- Every successor used by the distribution must be the real `applyBirth` state and every conditional factor must be the existing `orderedMass`.
- The trace carrier must be finite and executable without `Fintype.ofFinite` or another choice-based executable enumeration.
- Invalid raw requests remain replay validation errors and receive no probability mass in the typed distribution.
- Do not add `sorry`, `admit`, new user axioms, `native_decide`, unsafe escape hatches, unlimited resource settings, skipped proof gates, or `continue-on-error` around a failed contract.
- Preserve Lean 4.32.0, the pinned mathlib revision, existing conformance checks, old Lean theorem tests, Story, Testimony, and the current exact-head checkout/provenance behavior.
- Python discovery and World Studio remain intentionally excluded on `feature/fitness-*` branches under the existing scope policy. A skip is not a success and must be reported as an exclusion.
- Keep PR #70 Draft throughout plan execution until a later explicit review/merge authorization.
- No RNG, Monte Carlo, random fitness generation, asymptotics, power-law fitting, condensation classification, concentration inequality, or six-hop probability claim belongs to this plan.

## Task sequence

1. Enforce repository-wide BB-only naming and remove compatibility aliases.
2. Add finite executable `TargetTrace` carrier and the named distribution CI gate.
3. Define exact `traceProbability` and prove the trace sum equals existing `continuationMass = 1`.
4. Define `traceFinal` from real `applyBirth` transitions and prove node/edge/fitness/validity invariants.
5. Define exact finite `eventProbability` and prove true/false/nonnegativity/complement/monotonicity laws.
6. Define exact rational `expectation` and prove constant/addition/scalar/indicator laws.
7. Lift BB common scaling and constant-fitness normalization to the full trace distribution.
8. Verify small exact network events/expectations and complete trust, naming, README, and exact-head CI audits.

## Task 1 acceptance boundary

Task 1 is a semantic-preserving naming cleanup. The current tree must expose only BB vocabulary. Replace the legacy state/row/replay public names with:

```text
unitFitnessState
unitFitnessRow
attachment_constant_fitness
orderedMass_constant_fitness
replay_constant_fitness_topology
replay_constant_fitness_probability
```

Keep `scaleFitness`, `orderedMass_scale`, `replay_scale*`, `unitSeed`, `unitBirth`, `constantSeed`, and `constantBirth`. Do not leave aliases under removed names.

Add a workflow static audit that constructs forbidden legacy strings at runtime and searches the complete checked-out tree, including source, tests, README, current specs, current plans, and workflow files. Historical commits and already-closed PR discussions are not rewritten.

Task 1 RED is specifically the new naming audit failing on the existing surface. Task 1 GREEN requires the renamed Lean tests, root build, and complete naming audit to pass on the same exact head.

## Task 2 acceptance boundary

Create:

```lean
def TargetTrace (n m : Nat) : Nat → Type
  | 0 => PUnit
  | steps + 1 => Targets n m × TargetTrace (n + 1) m steps
```

and a recursive executable `Fintype` instance built from the existing computable `Fintype (Targets n m)`. Do not use `Fintype.ofFinite`.

Minimum exact cardinality fixtures:

```text
card(TargetTrace 3 2 0) = 1
card(TargetTrace 3 2 1) = 6
card(TargetTrace 2 1 2) = 6
```

Add `NarrativeDynamics/Core/FitnessDistribution.lean`, `NarrativeDynamics/Tests/FitnessDistribution.lean`, a root import only after the module exists, and a `Fitness distribution contract tests` workflow step.

## Task 3 acceptance boundary

Define:

```lean
def traceProbability {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) :
    (schedule : List PosFitness) → TargetTrace n m schedule.length → Rat
  | [], _ => 1
  | eta :: rest, (T, tail) =>
      orderedMass s T *
        traceProbability (applyBirth s T hm eta) m hm
          (Nat.le_trans hb (Nat.le_succ n)) rest tail
```

Prove:

```text
traceProbability_pos
traceProbability_nonneg
traceProbability_sum_continuationMass
traceProbability_sum_one
traceProbability_le_one
```

The normalization theorem must identify the finite trace sum with the existing `continuationMass`, then use `continuationMass_one`. Do not create a second normalization algorithm.

Concrete fixture: the two-node edge, `m=1`, two unit-fitness births, target 0 then target 0 has exact mass `1/4`.

## Task 4 acceptance boundary

Promote one shared birth fitness-list theorem into `FitnessBirth.lean` and delete the replay-private duplicate. Define:

```lean
def traceFinal {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) :
    (schedule : List PosFitness) → TargetTrace n m schedule.length → RunState
```

by recursively applying the real `applyBirth` transition. Prove:

```text
traceFinal_nodes
traceFinal_edges
traceFinal_fitness
traceFinal_valid
```

with final node count `n + schedule.length`, edge count increase `m * schedule.length`, and fitness extended by schedule values in order.

## Task 5 acceptance boundary

Define:

```lean
def eventProbability ...
    (event : RunState → Prop) [DecidablePred event] : Rat :=
  ∑ trace,
    if event (traceFinal ... trace)
    then traceProbability ... trace
    else 0
```

Prove:

```text
eventProbability_true
eventProbability_false
eventProbability_nonneg
eventProbability_le_one
eventProbability_compl
eventProbability_mono
```

Do not quotient or deduplicate final states: different ordered traces that reach the same graph contribute separately.

Concrete two-birth edge fixture: event “stable vertex 0 has final degree at least 2” has exact probability `5/8`.

## Task 6 acceptance boundary

Define exact rational expectation:

```lean
def expectation ... (observable : RunState → Rat) : Rat :=
  ∑ trace,
    traceProbability ... trace * observable (traceFinal ... trace)
```

Prove:

```text
expectation_const
expectation_add
expectation_smul
expectation_indicator
```

Concrete fixture: expected final degree of stable vertex 0 in the two-birth edge experiment is exactly `15/8`.

## Task 7 acceptance boundary

Expose one shared positive-fitness scaling helper and schedule map:

```lean
def scalePosFitness (c eta : PosFitness) : PosFitness :=
  ⟨c.val * eta.val, mul_pos c.property eta.property⟩

def scaleSchedule (c : PosFitness) (schedule : List PosFitness) : List PosFitness :=
  schedule.map (scalePosFitness c)
```

Prove full-distribution invariance:

```text
traceProbability_scale
traceFinal_scale
eventProbability_scale
```

`traceFinal_scale` must show the scaled experiment equals `scaleRunState` of the original final state. `eventProbability_scale` requires an explicit premise that the event is invariant under `scaleRunState`. Constant-fitness normalization remains a BB scaling corollary only.

Retain the existing negative replay fixture proving that scaling only part of the fitness schedule changes a multi-birth probability.

## Task 8 acceptance boundary

Use existing graph metrics; do not define alternate distance/diameter semantics.

For the two-birth unit-fitness edge experiment, the six traces have exact masses/outcomes:

```text
(0,0): 1/4, diameter 2
(0,1): 1/8, diameter 3
(0,2): 1/8, diameter 3
(1,0): 1/8, diameter 3
(1,1): 1/4, diameter 2
(1,2): 1/8, diameter 3
```

Therefore exact `P(diameter ≤ 2) = 1/2`.

For one birth from a unit-fitness triangle with `m=2`, the six ordered traces each have mass `1/6`. The final-graph event “newborn adjacent to stable IDs 1 and 2” has mass `1/3` because both target orders contribute.

Keep the exact expectation `15/8`, add `#print axioms` for the new generic laws, update README only after verified GREEN, and run the full exact-head proof workflow.

Final required Lean gates:

```text
conformance vectors
full root build
BB-only naming audit
Fitness attachment/birth/validation
Fitness replay
Fitness scope
Fitness distribution
general theorem tests
Story
Testimony
```

Python discovery and World Studio remain explicit scope exclusions under the existing fitness-branch policy.

## RED/GREEN and review protocol

Every task is test/contract first:

```text
write the new contract
→ observe the task-specific expected RED
→ implement the minimum BB-native change
→ run focused GREEN
→ run required prior regressions
→ commit/push
→ inspect exact-head CI
→ stop for review
```

A dependency outage, unrelated workflow failure, or skipped peripheral job is not a valid RED/GREEN result.

Generic proofs must be audited with `#print axioms`. Reject `sorryAx`, new user axioms, or native-oracle dependencies.

## Plan Review Gate

This plan is review-only. Production implementation has not started. The approved execution order is Task 1 through Task 8, with an exact-head review checkpoint after every task. PR #70 remains Draft and unmerged until an explicit later merge authorization.
