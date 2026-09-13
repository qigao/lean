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

## File map

| Path | Responsibility |
| --- | --- |
| `NarrativeDynamics/Core/FitnessAttachment.lean` | BB kernel; Task 1 removes legacy naming and exposes BB-native constant-fitness helpers. |
| `NarrativeDynamics/Core/FitnessBirth.lean` | Existing graph transition; Task 4 may promote the birth fitness-list update theorem here so replay and distribution share it. |
| `NarrativeDynamics/Core/FitnessReplay.lean` | Existing raw replay and `continuationMass`; Task 1 renames legacy public theorems, later tasks consume but do not duplicate the replay kernel. |
| `NarrativeDynamics/Core/FitnessDistribution.lean` | New finite trace carrier, exact trace law, final-state evaluation, event probability, expectation, and BB distribution invariances. |
| `NarrativeDynamics/Tests/FitnessAttachment.lean` | Task 1 regression updates for BB-only names. |
| `NarrativeDynamics/Tests/FitnessReplay.lean` | Task 1 replay regression updates for BB-only names. |
| `NarrativeDynamics/Tests/FitnessDistribution.lean` | New exhaustive finite-distribution contracts and network-event fixtures. |
| `NarrativeDynamics.lean` | Add the distribution core import only after the new module exists. |
| `.github/workflows/proof.yml` | BB-only static naming audit plus a named distribution contract step; preserve exact-head dedup behavior. |
| `README.md` | Remove the legacy model identity in Task 1; document the verified finite distribution only after Task 8 GREEN. |
| `docs/superpowers/specs/2026-09-11-fitness-attachment-v23-5-design.md` | Rewrite maintained terminology to BB-only in Task 1 without changing historical Git commits. |
| `docs/superpowers/plans/2026-09-11-fitness-attachment-v23-5.md` | Rewrite maintained terminology to BB-only in Task 1. |
| `docs/superpowers/specs/2026-09-13-fitness-distribution-v23-6-design.md` | Remove the literal legacy model token while retaining the BB-only invariant in Task 1. |

---

### Task 1: Enforce the repository-wide BB-only model surface

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessAttachment.lean`
- Modify: `NarrativeDynamics/Core/FitnessReplay.lean`
- Modify: `NarrativeDynamics/Tests/FitnessAttachment.lean`
- Modify: `NarrativeDynamics/Tests/FitnessReplay.lean`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-11-fitness-attachment-v23-5-design.md`
- Modify: `docs/superpowers/plans/2026-09-11-fitness-attachment-v23-5.md`
- Modify: `docs/superpowers/specs/2026-09-13-fitness-distribution-v23-6-design.md`
- Modify: `.github/workflows/proof.yml`

**Interfaces:**
- Consumes: current BB state/kernel/replay APIs at branch base `21fed4ca07cc69fce7d2cdc3115a571cd1e663f4`.
- Produces: `unitFitnessState`, `unitFitnessRow`, `attachment_constant_fitness`, `orderedMass_constant_fitness`, `replay_constant_fitness_topology`, `replay_constant_fitness_probability`, plus an exact current-tree naming audit. Existing `scaleFitness`, `orderedMass_scale`, `replay_scale*`, `unitSeed`, `unitBirth`, `constantSeed`, and `constantBirth` remain available.

- [ ] **Step 1: Add a failing BB-only static naming audit before changing source names**

Add a named step after checkout/provenance and before the fitness contract tests. Construct forbidden strings at runtime so the audit implementation does not itself contain them literally:

```yaml
      - name: BB-only naming audit
        run: |
          set -euo pipefail
          short="$(printf '%s%s' B A)"
          lower="$(printf '%s%s' b a)"
          full_ascii="$(printf '%s%s' 'Barabasi-' 'Albert')"
          full_unicode="$(printf '%s%s' 'Barabási–' 'Albert')"
          found=0
          for term in \
            "$short" "$full_ascii" "$full_unicode" \
            "as${short}" "${lower}Row" \
            "attachment_${lower}" "orderedMass_${lower}" \
            "replay_${lower}_"; do
            if git grep -n -F "$term" -- .; then
              found=1
            fi
          done
          if [ "$found" -ne 0 ]; then
            echo 'legacy attachment-model surface remains'
            exit 1
          fi
```

Do not exclude docs, plans, tests, README, or workflow files from this search. Closed PR discussions and historical Git commits are outside the checked-out tree and therefore outside this audit.

- [ ] **Step 2: Push the audit-only RED and verify the expected failure**

Run locally if a worktree is available:

```bash
short="$(printf '%s%s' B A)"
git grep -n -F "$short" -- .
```

Then commit/push the audit-only change:

```bash
git add .github/workflows/proof.yml
git commit -m "test(lean): require BB-only attachment naming"
git push
```

Expected RED: `BB-only naming audit` fails by reporting the known legacy identifiers/docs. The root build and pre-existing Lean checks must not be treated as the RED; this task's intended failure is specifically the new static naming gate.

- [ ] **Step 3: Replace the legacy state/row compatibility surface with BB-native constant-fitness names**

In `FitnessAttachment.lean`, use one unit-fitness projection and one row helper:

```lean
def unitFitnessState {n : Nat} (s : State n) : State n where
  snapshot := { s.snapshot with fitness := fun _ => 1 }
  valid := ⟨s.valid.1, s.valid.2.1, fun _ => by norm_num⟩

/-- Unit fitness is only a normalization of the same BB state. -/
def unitFitnessRow {n : Nat} (s : State n)
    (S : Finset (Fin n)) (hS : S.card < n) : Internal.Row n :=
  attachmentRow (unitFitnessState s) S hS

/-- A common constant fitness value cancels from one BB attachment row. -/
theorem attachment_constant_fitness {n : Nat} (s : State n)
    (S : Finset (Fin n)) (hS : S.card < n)
    (c : PosFitness) (constant : ∀ j, s.snapshot.fitness j = c.val)
    (i : Fin n) :
    (attachmentRow s S hS).mass i = (unitFitnessRow s S hS).mass i := by
  apply attachment_proportional (unitFitnessState s) s S hS c
  intro j
  change s.snapshot.fitness j * (degree s.snapshot j : Rat) =
    c.val * (1 * (degree s.snapshot j : Rat))
  rw [constant j, one_mul]
```

Rename the ordered-target law to the BB-native form and keep the same statement shape:

```lean
theorem orderedMass_constant_fitness {n m : Nat} (s : State n)
    (T : Targets n m) (c : PosFitness)
    (constant : ∀ i, s.snapshot.fitness i = c.val) :
    orderedMass s T = orderedMass (unitFitnessState s) T := by
  -- reuse the existing proof, replacing only the renamed helpers
  ...
```

The executor must port the existing proof body; do not add a second target-mass algorithm.

- [ ] **Step 4: Rename replay-level constant-fitness theorems without aliases**

Keep the existing private common-scaling derivation and expose only BB-native public names:

```lean
theorem replay_constant_fitness_topology
    (seed : RawSeed) (m : Nat) (bs : List RawBirth)
    (out : ReplayResult) (c : PosFitness)
    (h : replay (unitSeed seed) m (bs.map unitBirth) = .ok out) :
    ∃ next,
      replay (constantSeed seed c) m (bs.map (fun raw => constantBirth raw c)) = .ok next ∧
      next.final = scaleRunState out.final c := by
  exact ⟨scaleResult out c, replay_constant seed m bs out c h, rfl⟩

theorem replay_constant_fitness_probability
    (seed : RawSeed) (m : Nat) (bs : List RawBirth)
    (out : ReplayResult) (c : PosFitness)
    (h : replay (unitSeed seed) m (bs.map unitBirth) = .ok out) :
    ∃ next,
      replay (constantSeed seed c) m (bs.map (fun raw => constantBirth raw c)) = .ok next ∧
      next.probability = out.probability := by
  exact ⟨scaleResult out c, replay_constant seed m bs out c h, rfl⟩
```

Do not keep deprecated aliases under the old names.

- [ ] **Step 5: Update the existing Lean fixtures and axiom audits to the new BB-native surface**

Replace row fixtures with `unitFitnessRow`; replace generic theorem applications with `attachment_constant_fitness`, `orderedMass_constant_fitness`, `replay_constant_fitness_topology`, and `replay_constant_fitness_probability`. Preserve every existing numerical expectation, including the common-scaling positive fixture and the seed-only-scaling negative fixture.

Required examples include:

```lean
example : rowValues (.ok (unitFitnessRow Fixtures.star ∅ (by decide))) =
    .ok [1/2, 1/6, 1/6, 1/6] := by decide_cbv

example {n m : Nat} (s : State n) (T : Targets n m) (c : PosFitness)
    (constant : ∀ i, s.snapshot.fitness i = c.val) :
    orderedMass s T = orderedMass (unitFitnessState s) T :=
  orderedMass_constant_fitness s T c constant
```

Print the renamed theorem axioms explicitly.

- [ ] **Step 6: Rewrite maintained prose and the revised V23.6 spec to contain only BB vocabulary**

In README/spec/plan prose, describe constant positive fitness as a BB normalization/scaling property. Do not describe a recovered, compatible, baseline, or special-case second attachment model. The revised V23.6 spec itself must pass the static naming audit; refer to the removed surface only as “legacy alternate-model vocabulary.”

- [ ] **Step 7: Run Task 1 GREEN verification**

Run:

```bash
export PATH="$HOME/.elan/bin:$PATH"
lake env lean NarrativeDynamics/Tests/FitnessAttachment.lean
lake env lean NarrativeDynamics/Tests/FitnessReplay.lean
lake build
short="$(printf '%s%s' B A)"
! git grep -n -F "$short" -- .
git diff --check
```

Also run the workflow's full BB-only audit loop, not just the short token check. Expected: no matches; all renamed Lean fixtures pass; no compatibility alias remains.

- [ ] **Step 8: Commit the BB-only surface**

```bash
git add NarrativeDynamics/Core/FitnessAttachment.lean \
        NarrativeDynamics/Core/FitnessReplay.lean \
        NarrativeDynamics/Tests/FitnessAttachment.lean \
        NarrativeDynamics/Tests/FitnessReplay.lean \
        README.md docs/superpowers .github/workflows/proof.yml
git commit -m "refactor(lean): make BB the only attachment model surface"
git push
```

Stop for review. Do not start the distribution carrier until exact-head GREEN and the BB-only audit are confirmed.

---

### Task 2: Add the finite target-trace carrier and named distribution gate

**Files:**
- Create: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Create: `NarrativeDynamics/Tests/FitnessDistribution.lean`
- Modify: `NarrativeDynamics.lean`
- Modify: `.github/workflows/proof.yml`

**Interfaces:**
- Consumes: `Targets n m` and its computable `Fintype` instance from `FitnessAttachment`.
- Produces: `TargetTrace (n m steps : Nat)`, recursive executable `Fintype`, and test-local finite cardinality fixtures. Later tasks use `TargetTrace n m schedule.length`.

The plan deliberately indexes `TargetTrace` by the number of future births, not by the schedule values. Fitness values do not affect carrier shape, and the `Nat` index avoids unnecessary transports when common scaling changes schedule values.

- [ ] **Step 1: Add the distribution test file and workflow step before the core module exists**

Create `NarrativeDynamics/Tests/FitnessDistribution.lean`:

```lean
import NarrativeDynamics.Core.FitnessDistribution

open NarrativeDynamics NarrativeDynamics.FitnessAttachment
open NarrativeDynamics.FitnessAttachment.Internal

private def one : PosFitness := ⟨1, by norm_num⟩

example : Fintype.card (TargetTrace 3 2 0) = 1 := by decide
example : Fintype.card (TargetTrace 3 2 1) = 6 := by decide
example : Fintype.card (TargetTrace 2 1 2) = 6 := by decide
```

Add after the replay/scope gates:

```yaml
      - name: Fitness distribution contract tests
        timeout-minutes: 5
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          timeout --kill-after=10s 240s \
            lake env lean -DmaxErrors=8 NarrativeDynamics/Tests/FitnessDistribution.lean
```

- [ ] **Step 2: Push and record the intended missing-module RED**

```bash
git add NarrativeDynamics/Tests/FitnessDistribution.lean .github/workflows/proof.yml
git commit -m "test(lean): specify finite BB target trace carrier"
git push
```

Expected RED: the distribution contract step fails because `NarrativeDynamics.Core.FitnessDistribution` does not exist. The prior root build and fitness checks must remain green.

- [ ] **Step 3: Implement the schedule-length-indexed carrier and computable finite instance**

Create `NarrativeDynamics/Core/FitnessDistribution.lean`:

```lean
import NarrativeDynamics.Core.FitnessReplay

namespace NarrativeDynamics.FitnessAttachment

open Internal
open scoped BigOperators

/-- All legal ordered BB target choices for exactly `steps` future births. -/
def TargetTrace (n m : Nat) : Nat → Type
  | 0 => PUnit
  | steps + 1 => Targets n m × TargetTrace (n + 1) m steps

instance instFintypeTargetTrace (n m : Nat) :
    (steps : Nat) → Fintype (TargetTrace n m steps)
  | 0 => inferInstanceAs (Fintype PUnit)
  | steps + 1 =>
      letI := instFintypeTargetTrace (n + 1) m steps
      inferInstanceAs (Fintype (Targets n m × TargetTrace (n + 1) m steps))

end NarrativeDynamics.FitnessAttachment
```

Do not use `Fintype.ofFinite`.

- [ ] **Step 4: Add the root import and verify carrier GREEN**

Append:

```lean
import NarrativeDynamics.Core.FitnessDistribution
```

to `NarrativeDynamics.lean`, then run:

```bash
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake build
git diff --check
```

Expected cardinalities: `1`, `6`, and `6` as above.

- [ ] **Step 5: Commit**

```bash
git add NarrativeDynamics/Core/FitnessDistribution.lean \
        NarrativeDynamics/Tests/FitnessDistribution.lean NarrativeDynamics.lean
git commit -m "feat(lean): add finite BB target trace carrier"
git push
```

Stop for review.

---

### Task 3: Define exact trace probability and prove normalization through `continuationMass`

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`

**Interfaces:**
- Consumes: `TargetTrace`, `orderedMass`, `orderedMass_pos`, `applyBirth`, `continuationMass`, `continuationMass_one`.
- Produces: `traceProbability`, `traceProbability_pos`, `traceProbability_nonneg`, `traceProbability_sum_continuationMass`, `traceProbability_sum_one`, `traceProbability_le_one`.

- [ ] **Step 1: Add RED tests for concrete trace masses and total mass**

Add test-local connected states and singleton targets:

```lean
namespace DistributionFixtures

private theorem edgeConnected : (seedGraph (⟨2, #[1, 1], #[(0, 1)]⟩ : RawSeed)).Connected := by
  -- use the same two-vertex connectedness proof pattern as FitnessScope
  constructor
  · intro u v
    by_cases h : u = v
    · subst v; exact ⟨.nil⟩
    · exact ⟨.cons (by fin_cases u <;> fin_cases v <;> simp_all [seedGraph, canonicalEdge]) .nil⟩
  · exact ⟨0⟩

def edge : State 2 :=
  ⟨seedSnapshot (⟨2, #[1, 1], #[(0, 1)]⟩ : RawSeed) rfl,
    ⟨by decide, edgeConnected, by decide⟩⟩

def one : PosFitness := ⟨1, by norm_num⟩

def singletonTarget {n : Nat} (i : Fin n) : Targets n 1 :=
  ⟨fun _ => i, fun _ _ _ => Subsingleton.elim _ _⟩

def trace00 : TargetTrace 2 1 2 :=
  (singletonTarget 0, (singletonTarget 0, PUnit.unit))

end DistributionFixtures
```

Add intended contracts:

```lean
example : traceProbability DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    DistributionFixtures.trace00 = 1/4 := by decide_cbv

example :
    (∑ t : TargetTrace 2 1 2,
      traceProbability DistributionFixtures.edge 1 (by decide) (by decide)
        [DistributionFixtures.one, DistributionFixtures.one] t) = 1 := by
  exact traceProbability_sum_one _ _ _ _ _
```

- [ ] **Step 2: Run RED**

```bash
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
```

Expected: missing `traceProbability` / normalization theorem declarations.

- [ ] **Step 3: Implement recursive trace probability from the actual evolving state**

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

Prove positivity by induction using `orderedMass_pos` and `mul_pos`.

- [ ] **Step 4: Prove the finite trace sum is exactly the existing continuation mass**

Required statement:

```lean
theorem traceProbability_sum_continuationMass {n : Nat}
    (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
    (schedule : List PosFitness) :
    (∑ t : TargetTrace n m schedule.length,
      traceProbability s m hm hb schedule t) =
      continuationMass s m hm hb schedule := by
  induction schedule generalizing n with
  | nil => simp [TargetTrace, traceProbability, continuationMass]
  | cons eta rest ih =>
      -- Rewrite the product carrier sum as Σ T, Σ tail.
      simp only [List.length_cons, TargetTrace, traceProbability, continuationMass]
      rw [Fintype.sum_prod_type]
      apply Finset.sum_congr rfl
      intro T _
      rw [Finset.mul_sum]
      simp [ih]
```

If the pinned mathlib names the product-sum lemma differently, inspect the available finite-sum API and use the equivalent theorem; do not replace this with list enumeration or a second normalization algorithm.

Then derive:

```lean
theorem traceProbability_sum_one ... :
    (∑ t : TargetTrace n m schedule.length,
      traceProbability s m hm hb schedule t) = 1 := by
  rw [traceProbability_sum_continuationMass]
  exact continuationMass_one s m hm hb schedule
```

Derive `traceProbability_nonneg` from positivity and `traceProbability_le_one` by bounding one nonnegative summand by the sum-one total.

- [ ] **Step 5: Audit and run GREEN**

Append:

```lean
#print axioms traceProbability_pos
#print axioms traceProbability_sum_continuationMass
#print axioms traceProbability_sum_one
#print axioms traceProbability_le_one
```

Run:

```bash
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake env lean NarrativeDynamics/Tests/FitnessReplay.lean
lake build
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add NarrativeDynamics/Core/FitnessDistribution.lean NarrativeDynamics/Tests/FitnessDistribution.lean
git commit -m "feat(lean): prove normalized finite BB trace law"
git push
```

Stop for review.

---

### Task 4: Evaluate traces to authoritative final BB states and prove structural invariants

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessBirth.lean`
- Modify: `NarrativeDynamics/Core/FitnessReplay.lean`
- Modify: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`

**Interfaces:**
- Consumes: `TargetTrace`, `applyBirth`, `birth_edges`, existing birth fitness semantics.
- Produces: public `birth_fitness_list`, `traceFinal`, `traceFinal_nodes`, `traceFinal_edges`, `traceFinal_fitness`, `traceFinal_valid`.

- [ ] **Step 1: Add RED final-state contracts**

Add:

```lean
example : (traceFinal DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    DistributionFixtures.trace00).nodeCount = 4 := by decide_cbv

example : actualEdgeCount
    (traceFinal DistributionFixtures.edge 1 (by decide) (by decide)
      [DistributionFixtures.one, DistributionFixtures.one]
      DistributionFixtures.trace00).state.snapshot = 3 := by decide_cbv
```

Add generic theorem applications for node count, edge count, and fitness list.

- [ ] **Step 2: Run RED**

Expected missing `traceFinal` and structural theorem declarations.

- [ ] **Step 3: Promote the existing birth fitness-list helper to the birth module**

Move the replay-private theorem into `FitnessBirth.lean` as:

```lean
theorem birth_fitness_list {n m : Nat} (s : State n) (T : Targets n m)
    (hm : 0 < m) (eta : PosFitness) :
    List.ofFn (applyBirth s T hm eta).snapshot.fitness =
      List.ofFn s.snapshot.fitness ++ [eta.val] := by
  rw [List.ofFn_succ']
  simp [applyBirth, birthSnapshot, List.concat_eq_append]
```

Delete the duplicate private theorem from `FitnessReplay.lean` and let `runBirths_properties` use this public birth theorem.

- [ ] **Step 4: Implement final-state evaluation**

```lean
def traceFinal {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) :
    (schedule : List PosFitness) → TargetTrace n m schedule.length → RunState
  | [], _ => ⟨n, s⟩
  | eta :: rest, (T, tail) =>
      traceFinal (applyBirth s T hm eta) m hm
        (Nat.le_trans hb (Nat.le_succ n)) rest tail
```

Prove by induction:

```lean
traceFinal_nodes:
  (traceFinal s m hm hb schedule trace).nodeCount = n + schedule.length

traceFinal_edges:
  actualEdgeCount (traceFinal ...).state.snapshot =
    actualEdgeCount s.snapshot + m * schedule.length

traceFinal_fitness:
  List.ofFn (traceFinal ...).state.snapshot.fitness =
    List.ofFn s.snapshot.fitness ++ schedule.map Subtype.val

traceFinal_valid:
  (traceFinal ...).state.snapshot.Valid
```

Use `birth_edges` and `birth_fitness_list`; do not store counts or fitness redundantly in `TargetTrace`.

- [ ] **Step 5: GREEN verification and audit**

```bash
lake env lean NarrativeDynamics/Tests/FitnessBirth.lean
lake env lean NarrativeDynamics/Tests/FitnessReplay.lean
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake build
git diff --check
```

Print axioms for the new generic trace-final theorems.

- [ ] **Step 6: Commit**

```bash
git add NarrativeDynamics/Core/FitnessBirth.lean NarrativeDynamics/Core/FitnessReplay.lean \
        NarrativeDynamics/Core/FitnessDistribution.lean NarrativeDynamics/Tests/FitnessDistribution.lean
git commit -m "feat(lean): evaluate BB traces to authoritative final states"
git push
```

Stop for review.

---

### Task 5: Add exact finite event probabilities and probability laws

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`

**Interfaces:**
- Consumes: normalized `traceProbability`, `traceFinal`.
- Produces: `eventProbability`, `eventProbability_true`, `eventProbability_false`, `eventProbability_nonneg`, `eventProbability_le_one`, `eventProbability_compl`, `eventProbability_mono`.

- [ ] **Step 1: Add RED event-law tests**

Define one stable-ID degree predicate for the two-birth edge experiment:

```lean
def oldZeroDegree (r : RunState) : Nat :=
  degree r.state.snapshot ⟨0, by have := r.state.valid.1; omega⟩

def zeroDegreeAtLeastTwo (r : RunState) : Prop := 2 ≤ oldZeroDegree r

instance : DecidablePred zeroDegreeAtLeastTwo := fun _ => inferInstance
```

Add:

```lean
example : eventProbability DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    (fun _ => True) = 1 := by
  exact eventProbability_true _ _ _ _ _

example : eventProbability DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    zeroDegreeAtLeastTwo = 5/8 := by decide_cbv
```

The `5/8` fixture is independent arithmetic over the six typed traces: masses are `1/4, 1/8, 1/8, 1/8, 1/4, 1/8`, and the matching traces contribute `1/4 + 1/8 + 1/8 + 1/8`.

- [ ] **Step 2: Run RED**

Expected missing `eventProbability` declarations.

- [ ] **Step 3: Implement exact event mass**

```lean
def eventProbability {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (event : RunState → Prop) [DecidablePred event] : Rat :=
  ∑ trace : TargetTrace n m schedule.length,
    if event (traceFinal s m hm hb schedule trace)
    then traceProbability s m hm hb schedule trace
    else 0
```

- [ ] **Step 4: Prove the finite event laws from sum-one/nonnegativity**

Prove true/false by simplification. Prove nonnegativity termwise. Prove the complement law by partitioning each trace's mass between `event` and `¬ event` and rewriting the combined sum to `traceProbability_sum_one`. Prove upper bound by nonnegative complement. Prove monotonicity termwise under:

```lean
hEF : ∀ r, event r → larger r
```

Do not quotient/deduplicate final states.

- [ ] **Step 5: GREEN verification and audit**

Print axioms for all generic event laws and run:

```bash
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake build
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add NarrativeDynamics/Core/FitnessDistribution.lean NarrativeDynamics/Tests/FitnessDistribution.lean
git commit -m "feat(lean): prove exact BB event probabilities"
git push
```

Stop for review.

---

### Task 6: Add exact rational expectations and linearity

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`

**Interfaces:**
- Consumes: `traceProbability`, `traceFinal`, `eventProbability`.
- Produces: `expectation`, `expectation_const`, `expectation_add`, `expectation_smul`, `expectation_indicator`.

- [ ] **Step 1: Add RED expectation fixtures**

Add:

```lean
example : expectation DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    (fun r => (oldZeroDegree r : Rat)) = 15/8 := by decide_cbv

example : expectation DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    (fun _ => (7/3 : Rat)) = 7/3 := by
  exact expectation_const _ _ _ _ _ _
```

The explicit degree expectation is:

```text
3*(1/4) + 2*(1/8+1/8+1/8) + 1*(1/4+1/8) = 15/8.
```

- [ ] **Step 2: Run RED**

Expected missing expectation declarations.

- [ ] **Step 3: Implement expectation**

```lean
def expectation {n : Nat} (s : State n) (m : Nat)
    (hm : 0 < m) (hb : m ≤ n) (schedule : List PosFitness)
    (observable : RunState → Rat) : Rat :=
  ∑ trace : TargetTrace n m schedule.length,
    traceProbability s m hm hb schedule trace *
      observable (traceFinal s m hm hb schedule trace)
```

- [ ] **Step 4: Prove the algebraic base**

Required theorem shapes:

```lean
theorem expectation_const ... (q : Rat) :
    expectation s m hm hb schedule (fun _ => q) = q

theorem expectation_add ... (f g : RunState → Rat) :
    expectation s m hm hb schedule (fun r => f r + g r) =
      expectation s m hm hb schedule f + expectation s m hm hb schedule g

theorem expectation_smul ... (q : Rat) (f : RunState → Rat) :
    expectation s m hm hb schedule (fun r => q * f r) =
      q * expectation s m hm hb schedule f
```

For indicator equivalence use:

```lean
def indicator (event : RunState → Prop) [DecidablePred event] (r : RunState) : Rat :=
  if event r then 1 else 0

theorem expectation_indicator ... :
  expectation s m hm hb schedule (indicator event) =
    eventProbability s m hm hb schedule event
```

Use finite-sum distributivity and `traceProbability_sum_one`; do not import real-analysis expectation APIs.

- [ ] **Step 5: GREEN verification and audit**

```bash
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake build
git diff --check
```

Print axioms for all expectation laws.

- [ ] **Step 6: Commit**

```bash
git add NarrativeDynamics/Core/FitnessDistribution.lean NarrativeDynamics/Tests/FitnessDistribution.lean
git commit -m "feat(lean): add exact rational BB expectations"
git push
```

Stop for review.

---

### Task 7: Lift common fitness scaling and constant-fitness normalization to the full distribution

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessAttachment.lean`
- Modify: `NarrativeDynamics/Core/FitnessReplay.lean`
- Modify: `NarrativeDynamics/Core/FitnessDistribution.lean`
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`

**Interfaces:**
- Consumes: Task 1 BB-native constant-fitness names, `scaleFitness`, `orderedMass_scale`, replay scaling, trace law/event law.
- Produces: `scalePosFitness`, `scaleSchedule`, `traceProbability_scale`, `traceFinal_scale`, `eventProbability_scale` under a scale-invariant event premise, and constant-fitness normalization corollaries stated only as BB properties.

- [ ] **Step 1: Add RED distribution-invariance tests**

Define a positive scale:

```lean
private def two : PosFitness := ⟨2, by norm_num⟩
```

Require the same target trace to retain its probability when the entire typed experiment is scaled:

```lean
example :
    traceProbability (scaleFitness DistributionFixtures.edge two) 1 (by decide) (by decide)
      (scaleSchedule two [DistributionFixtures.one, DistributionFixtures.one])
      (by simpa [scaleSchedule] using DistributionFixtures.trace00) =
    traceProbability DistributionFixtures.edge 1 (by decide) (by decide)
      [DistributionFixtures.one, DistributionFixtures.one]
      DistributionFixtures.trace00 := by
  exact traceProbability_scale _ _ _ _ _ _ _
```

Also retain the existing replay negative fixture showing that scaling only part of the fitness schedule changes a multi-birth probability; do not weaken it.

- [ ] **Step 2: Run RED**

Expected missing typed schedule scaling / distribution invariance declarations.

- [ ] **Step 3: Expose one shared positive-fitness scaling helper**

Move the replay-private constructor to a public BB helper:

```lean
def scalePosFitness (c eta : PosFitness) : PosFitness :=
  ⟨c.val * eta.val, mul_pos c.property eta.property⟩

def scaleSchedule (c : PosFitness) (schedule : List PosFitness) : List PosFitness :=
  schedule.map (scalePosFitness c)
```

Update replay scaling internals to reuse `scalePosFitness`; do not keep a duplicate private constructor.

- [ ] **Step 4: Prove trace probability and final-state scaling by induction**

Required forms:

```lean
theorem traceProbability_scale ... :
  traceProbability (scaleFitness s c) m hm hb (scaleSchedule c schedule)
      (by simpa [scaleSchedule] using trace) =
    traceProbability s m hm hb schedule trace
```

and a final-state theorem showing the scaled run has the same graph and all fitness values multiplied by `c`. Reuse the existing `applyBirth_scale`/ordered-mass scaling proof; if `applyBirth_scale` is private in replay, move the minimal shared typed helper to the appropriate core module rather than reproving a parallel theorem in Distribution.

- [ ] **Step 5: Lift scaling to topology-only event probability**

Use an explicit event invariance premise:

```lean
scaleInvariantEvent : ∀ r, event (scaleRunState r c) ↔ event r
```

Then prove equality of event probabilities by finite-sum congruence and `traceProbability_scale`/`traceFinal_scale`.

State constant-fitness normalization only as a corollary of common scaling between two positive common values. Do not create another model namespace or compatibility family.

- [ ] **Step 6: GREEN verification and audit**

Run all four fitness layers plus the naming audit:

```bash
lake env lean NarrativeDynamics/Tests/FitnessAttachment.lean
lake env lean NarrativeDynamics/Tests/FitnessReplay.lean
lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
lake build
git diff --check
```

Print axioms for distribution scaling and constant-fitness corollaries.

- [ ] **Step 7: Commit**

```bash
git add NarrativeDynamics/Core/FitnessAttachment.lean NarrativeDynamics/Core/FitnessReplay.lean \
        NarrativeDynamics/Core/FitnessDistribution.lean NarrativeDynamics/Tests/FitnessDistribution.lean
git commit -m "feat(lean): lift BB fitness invariance to finite distributions"
git push
```

Stop for review.

---

### Task 8: Prove small exact network-event fixtures and complete the trust/CI audit

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessDistribution.lean`
- Modify: `README.md`
- Modify: `.github/workflows/proof.yml` only if the final gate needs ordering/comment clarification; do not weaken checks.

**Interfaces:**
- Consumes: complete V23.6 finite law, existing `state_bounded`, `meshDiameter`, `ReachWithin`, degree/edge metrics.
- Produces: exact small-network event/expectation demonstrations, final axiom audit, README scope statement, exact-head GREEN evidence.

- [ ] **Step 1: Add a test-local final-diameter event using existing metrics**

For a `RunState r`, obtain boundedness from `state_bounded r.state`; do not define a second distance metric. Define a decidable event equivalent to final diameter at most `2` for the two-birth edge experiment.

The six trace masses/outcomes are:

```text
first target 0, second 0 : mass 1/4, diameter 2
first target 0, second 1 : mass 1/8, diameter 3
first target 0, second 2 : mass 1/8, diameter 3
first target 1, second 0 : mass 1/8, diameter 3
first target 1, second 1 : mass 1/4, diameter 2
first target 1, second 2 : mass 1/8, diameter 3
```

Required exact event:

```lean
example : eventProbability DistributionFixtures.edge 1 (by decide) (by decide)
    [DistributionFixtures.one, DistributionFixtures.one]
    diameterAtMostTwo = 1/2 := by decide_cbv
```

If direct reduction of `meshDiameter` is expensive, prove the six final-state diameter facts as small opaque lemmas and let `eventProbability` sum those results; do not raise global resource limits.

- [ ] **Step 2: Add one one-birth triangle event showing ordered traces are not deduplicated**

Construct the unit-fitness triangle fixture and the event “the newborn is adjacent to both stable IDs 1 and 2.” Both target orders satisfy the same final graph event, so the exact mass is:

```lean
example : eventProbability unitTriangle 2 (by decide) (by decide)
    [DistributionFixtures.one] newbornAdjacentToOneAndTwo = 1/3 := by decide_cbv
```

This is `1/6 + 1/6`, not one representative trace.

- [ ] **Step 3: Re-run the exact expectation fixture**

Keep and verify:

```lean
E[final degree of stable vertex 0] = 15/8
```

for the two-birth edge experiment. This demonstrates that the event and expectation layers observe the authoritative final graph.

- [ ] **Step 4: Complete the generic axiom audit**

Ensure `#print axioms` covers at least:

```text
traceProbability_pos
traceProbability_sum_continuationMass
traceProbability_sum_one
traceFinal_edges
traceFinal_fitness
eventProbability_compl
eventProbability_mono
expectation_const
expectation_add
expectation_indicator
traceProbability_scale
eventProbability_scale
```

Review actual output. Accept only the repository's ordinary logical dependencies already seen in this line; reject `sorryAx`, new user axioms, or native-oracle dependencies.

- [ ] **Step 5: Update README with only verified V23.6 scope**

Document:

```text
fixed valid BB seed + fixed positive fitness schedule
        ↓
all legal ordered target traces
        ↓
exact positive Rat mass per trace
        ↓
total mass = 1
        ↓
exact final-state event probabilities
        ↓
exact rational expectations
```

State explicitly that this remains finite and conditional on the supplied fitness schedule, has no sampler, and makes no asymptotic/high-probability six-hop claim. Keep the repository model vocabulary BB-only.

- [ ] **Step 6: Run full exact-head verification**

Before claiming completion, run/inspect the full proof workflow on the exact branch head. Required successful Lean steps:

```text
Lean/Python conformance vectors
full root build
BB-only naming audit
Fitness attachment/birth/validation contracts
Fitness replay acceptance
Fitness scope counterexample
Fitness distribution contract tests
general Lean theorem tests
Story
Testimony
```

Python discovery and World Studio remain intentionally skipped by the existing fitness-branch policy and must be reported as exclusions.

Also run source checks:

```bash
git diff --check
short="$(printf '%s%s' B A)"
! git grep -n -F "$short" -- .
```

Then run the complete workflow audit loop from Task 1 to catch full-name and identifier variants.

- [ ] **Step 7: Commit the final acceptance layer**

```bash
git add NarrativeDynamics/Tests/FitnessDistribution.lean README.md .github/workflows/proof.yml
git commit -m "test(lean): verify finite BB distribution events and trust boundary"
git push
```

Stop for final review with PR #70 still Draft. Do not merge or enable auto-merge.

---

## Coverage checklist

| Revised spec requirement | Plan task |
| --- | --- |
| BB-only maintained model surface; no compatibility aliases | Task 1, re-audited Task 8 |
| Finite executable trace carrier | Task 2 |
| Exact positive trace probabilities | Task 3 |
| Sum over traces equals existing `continuationMass` and one | Task 3 |
| Authoritative final BB state and structural invariants | Task 4 |
| Exact decidable event probabilities and finite probability laws | Task 5 |
| Exact rational expectations and linearity | Task 6 |
| Whole-experiment common fitness scaling / constant-fitness normalization | Task 7 |
| Existing graph metrics lifted as events/observables | Task 8 |
| Ordered traces with same final graph are summed, not deduplicated | Task 8 triangle fixture |
| No RNG/PMF/Measure/asymptotics | Global constraints, every task |
| Trust/resource/axiom audit and exact-head CI | Tasks 1–8, final gate Task 8 |

## Execution handoff

Plan execution begins only after this committed plan is reviewed. Recommended execution is one fresh worker/reviewer context per task because Tasks 1, 3, 4, and 7 each change proof interfaces that later tasks depend on. Inline execution is acceptable if every task still stops at its explicit review gate.
