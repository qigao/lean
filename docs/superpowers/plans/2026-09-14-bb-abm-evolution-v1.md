# BB-driven ABM Evolution V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a concrete scalar-belief ABM can create agents through the existing BB kernel, propagate synchronously over the resulting graph, and preserve the existing finite BB topology law and exact probabilities.

**Architecture:** Add an exact rational propagation operator, compose it with authoritative BB births, then expose checked finite replay and probabilities of joint final-state events. Compare finite propagation examples with the existing Python V1 runtime through one Lean-generated corpus. Keep the situated runtime and production Python roster contracts unchanged.

**Tech Stack:** Lean `leanprover/lean4:v4.32.0`, mathlib at the existing `v4.32.0` dependency in `lakefile.toml`, `Rat`, `Fin`, `Finset`, `SimpleGraph`, Python standard-library `unittest` and `fractions`, and the existing GitHub Actions proof/trust infrastructure.

**Spec:** `docs/superpowers/specs/2026-09-14-bb-abm-evolution-v1-design.md`, approved from commit `49655229bc3c613a6b71f9c671a3e38a458551dc`.

## Global Constraints

- "Fitness is stored only in the existing BB state."
- "Profiles also remain fixed after birth."
- "Every next state is calculated from the same snapshot."
- "Newborn exposure is always zero before propagation."
- "A no-birth tick has probability factor one."
- "Agent updates are deterministic and contribute no additional probability factor."
- "The whole raw replay returns either one complete successful result or one error."
- "Every trace is evaluated exactly once."
- "The probability statement is conditional on the supplied birth calendar and agent parameters."
- "The maintained proof boundary remains the Lean kernel."
- "Start with the existing 240-second fixture limit; a failure requires diagnosis before changing the limit."
- "The test comparisons with Python do not constitute a production birth adapter or a proof of Python floating-point arithmetic."
- Reuse `applyBirth`, `oldId`, `newId`, `step`, `orderedMass`, `TargetTrace`, and `traceProbability_sum_one`; do not implement another attachment normalizer, degree cache, or trace sampler.
- The attachment count satisfies `0 < m` and `m <= seed.nodeCount`, including empty and no-birth-only schedules. A tick creates at most one agent and then runs exactly one propagation round.
- Use unit-influence bidirectional channels derived from each undirected BB edge. Fitness is not influence, receptivity, trust, or belief.
- Preserve the source/log audit allowlist `{propext, Classical.choice, Quot.sound}`. No unproved declarations, proof-oracle bypasses, unlimited resource settings, swallowed timeout failures, or skipped required gates.
- Do not change `lean-toolchain`, `lakefile.toml`, existing public Python contracts, V19 projections, or the earlier BB proof semantics.
- Use an implementation branch such as `feature/bb-abm-evolution-v1` so the existing `feature/fitness-*` exclusions do not suppress the required Python and World Studio jobs. Preserve workflow checkout and event-selection behavior.
- A successful foundation means the specific scalar-belief model is proved. It does not establish full semantic cognition, adaptive fitness, asymptotic small-world behavior, or a universal fixed-hop bound.

---

## File ownership and dependency order

| Task | Files | Responsibility |
| --- | --- | --- |
| 1 | Create `NarrativeDynamics/Core/NetworkPropagation.lean` and `NarrativeDynamics/Tests/NetworkPropagation.lean` | Concrete rational propagation and local invariants |
| 2 | Create `NarrativeDynamics/Core/FitnessABM.lean` and `NarrativeDynamics/Tests/FitnessABM.lean` | Typed joint state, identity transport, growth, and finite scheduling |
| 3 | Create `NarrativeDynamics/Core/FitnessABMReplay.lean` and `NarrativeDynamics/Tests/FitnessABMReplay.lean` | Agent validation, atomic raw replay, and projection agreement |
| 4 | Create `NarrativeDynamics/Core/FitnessABMDistribution.lean` and `NarrativeDynamics/Tests/FitnessABMDistribution.lean` | Reuse the BB trace law for agent-outcome probabilities |
| 5 | Create `NarrativeDynamics/Conformance/FitnessABMVectors.lean`, `conformance/bb_abm_v1.json`, and `tests/test_network_abm_bb_conformance.py` | Shared finite examples and Python single-round comparison |
| 6 | Create `tools/check_fitness_abm.sh`; modify `.github/workflows/proof.yml`, `NarrativeDynamics.lean`, `README.md`, and the four new Lean test modules | Public imports, bounded CI execution, required axiom reports, and verified scope documentation |

Tasks execute in this order. Their state and proof APIs are dependencies, so simultaneous edits to dependent tasks are not appropriate. No extra design, alternate stochastic model, or production Python adapter is part of these tasks.

The snippets fix names, data shapes, algorithms, and executable test targets. Lean elaboration and proof construction are checked during each task's RED/GREEN cycle; this planning commit does not claim those snippets have been compiled.

## Execution preparation

The planning worktree is clean at the approved design before the plan commit. No `AGENTS.md` was found in the checked repository paths. Recheck instructions and `git status` at execution time, reuse existing isolation, and create the implementation branch from the plan commit.

Lean is not installed in the current planning runtime, and there is no local mathlib cache. Provision the pinned toolchain through the normal Lean setup or run the task checks in GitHub CI; never report an unavailable local Lean command as a pass. Once the toolchain is available, record:

```bash
git rev-parse HEAD
cat lean-toolchain
lake update
lake env lean --version
git -C .lake/packages/mathlib rev-parse HEAD
lake build
python -m unittest tests.test_network_abm_simulation tests.test_network_abm_contracts -q
```

The current dependency specifies a tag rather than a committed local lockfile. Record the resolved mathlib SHA and keep that resolution for the implementation checks. Install Python dependencies from `requirements-world-studio.txt` when needed; do not introduce new dependency versions for these tests.

After a new core module is written, build that module before invoking its importing test file. For every task, capture the actual RED diagnostic and subsequent GREEN result. Check off a task only after its tests and stated proof targets pass, then commit its focused change. Missing imports are a legitimate first RED for a new module; subsequent behavioral examples must evaluate the real implementation.

## Task 1: Concrete synchronous rational propagation

**Files:** Create the two `NetworkPropagation.lean` files in the ownership table.

**Consumes:** `NarrativeDynamics.MeshGraph` from `NarrativeDynamics/Core/SocialMesh.lean`; finite sums and exact `Rat` from its existing Mathlib dependency.

**Produces:** Namespace `NarrativeDynamics.NetworkPropagation`, with the following contracts and functions. The value records deliberately contain ordinary data; the concrete transition must prove preservation of their separate validity predicates.

```lean
structure AgentProfile where
  receptivity : Rat
  threshold : Rat
  deriving DecidableEq, Repr

structure AgentState where
  belief : Rat
  exposures : Nat
  deriving DecidableEq, Repr

def AgentProfile.Valid (p : AgentProfile) : Prop :=
  0 <= p.receptivity ∧ p.receptivity <= 1 ∧
    0 <= p.threshold ∧ p.threshold <= 1

def AgentState.Valid (a : AgentState) : Prop :=
  0 <= a.belief ∧ a.belief <= 1

structure Population (n : Nat) where
  profiles : Fin n → AgentProfile
  agents : Fin n → AgentState

def Population.Valid {n : Nat} (p : Population n) : Prop :=
  (∀ i, (p.profiles i).Valid) ∧ (∀ i, (p.agents i).Valid)

def broadcasting (p : AgentProfile) (a : AgentState) : Bool :=
  decide (p.threshold <= a.belief)

def incoming {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) : Finset (Fin n) :=
  Finset.univ.filter fun j => g j i ∧ broadcasting (p.profiles j) (p.agents j) = true

def transmissions {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) : Finset (Fin n × Fin n) :=
  (Finset.univ.product Finset.univ).filter fun pair =>
    g pair.1 pair.2 ∧ broadcasting (p.profiles pair.1) (p.agents pair.1) = true

def nextAgent {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (i : Fin n) : AgentState :=
  let received := incoming g p i
  let count := received.card
  if count = 0 then p.agents i
  else
    let signal := (∑ j ∈ received, (p.agents j).belief) / (count : Rat)
    { belief := (1 - (p.profiles i).receptivity) * (p.agents i).belief +
        (p.profiles i).receptivity * signal
      exposures := (p.agents i).exposures + count }

def propagate {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) : Population n :=
  { p with agents := nextAgent g p }
```

- [ ] **Write the delayed-relay RED.** Import the future core module, open its namespace and `BigOperators`, and define this exact directed test graph and population:

```lean
def line3 : MeshGraph (Fin 3) := fun i j => i.val + 1 = j.val
instance : DecidableRel line3 := fun i j =>
  show Decidable (i.val + 1 = j.val) from inferInstance
def initial3 : Population 3 :=
  ⟨fun _ => ⟨1, 1/2⟩, ![⟨1, 0⟩, ⟨0, 0⟩, ⟨0, 0⟩]⟩

example : List.ofFn (fun i => ((propagate line3 initial3).agents i).belief) =
    [1, 1, 0] := by decide_cbv
example : List.ofFn (fun i =>
    ((propagate line3 (propagate line3 initial3)).agents i).belief) =
    [1, 1, 1] := by decide_cbv
example : transmissions line3 initial3 = {(0, 1)} := by decide_cbv
```

- [ ] **Run RED:** `lake env lean NarrativeDynamics/Tests/NetworkPropagation.lean`. Record the missing-module or missing-declaration failure, not a toolchain installation failure.
- [ ] **Implement the records and pure functions above.** Import `NarrativeDynamics.Core.SocialMesh`, keep executable decidability, and collect all incoming signals before returning any updated state.
- [ ] **Prove validity from finite sums.** For a nonempty incoming set, derive `0 < (received.card : Rat)`, a nonnegative signal sum, and `sum <= received.card` using `Finset.sum_nonneg` and `Finset.sum_le_sum`. Divide by the positive cardinality, then prove the convex update bounds with receptivity in `[0,1]`. For the empty set, use the original agent validity. Export this exact theorem interface:

```lean
example {n : Nat} (g : MeshGraph (Fin n)) [DecidableRel g]
    (p : Population n) (hp : p.Valid) : (propagate g p).Valid :=
  propagate_valid g p hp
```

- [ ] **Add and prove local contracts.** Export `transmission_iff` for `(j,i) ∈ transmissions g p ↔ g j i ∧ broadcasting (p.profiles j) (p.agents j) = true`; `nextAgent_no_incoming` for an empty incoming set; and `exposures_mono` for old exposure at most the computed exposure. Prove `nextAgent_locality`: with the graph fixed, equal receiver state/profile and equal adjacent source beliefs/broadcast booleans imply equal next receiver state. Source exposure counts and nonadjacent agent state are not read. Derive the transmission result by `simp [transmissions]`; use the identical incoming set and identical finite signal sum for locality.
- [ ] **Add numeric and direction boundaries.** Use the following exact expectations, changing only the indicated input fields in `initial3`:

```text
receiver 1 receptivity = 1/2: next belief = 1/2, next exposure = 1, broadcasting = true
receiver 1 receptivity = 0:   next belief = 0,   next exposure = 1
all beliefs = 0, thresholds = 1/2: no transmissions; every agent state preserved
source 0 belief = 0, threshold = 0: transmission (0,1) carries zero; receiver exposure increases
only source 1 broadcasts on line3: (1,2) transmits; (1,0) does not
```

- [ ] **Run GREEN and audit the new source:**

```bash
lake build NarrativeDynamics.Core.NetworkPropagation
timeout --kill-after=10s 240s lake env lean NarrativeDynamics/Tests/NetworkPropagation.lean
python3 tools/audit_fitness_trust.py source NarrativeDynamics/Core/NetworkPropagation.lean NarrativeDynamics/Tests/NetworkPropagation.lean
git diff --check
git add NarrativeDynamics/Core/NetworkPropagation.lean NarrativeDynamics/Tests/NetworkPropagation.lean
git commit -m "feat(lean): prove synchronous rational network propagation"
```

## Task 2: Typed BB births and joint finite scheduling

**Files:** Create the two `FitnessABM.lean` files.

**Consumes:** Task 1 `Population`, `Population.Valid`, `propagate`, `propagate_valid`; existing BB `State`, `Targets`, `applyBirth`, `oldId`, `newId`, `orderedMass`, and `orderedMass_pos`. The positive-fitness type is `NarrativeDynamics.FitnessAttachment.Internal.PosFitness`, not a new type.

**Produces:** Namespace `NarrativeDynamics.FitnessABM`. Within this namespace, open `NarrativeDynamics.NetworkPropagation`, `NarrativeDynamics.FitnessAttachment`, and its `Internal` namespace for the following declarations:

```lean
structure JointState (n : Nat) where
  network : FitnessAttachment.State n
  population : Population n
  populationValid : population.Valid

structure NewAgent where
  profile : AgentProfile
  initialBelief : Rat
  profileValid : profile.Valid
  beliefValid : 0 <= initialBelief ∧ initialBelief <= 1

structure BirthData where
  fitness : PosFitness
  agent : NewAgent

structure RunState where
  nodeCount : Nat
  roundIndex : Nat
  state : JointState nodeCount

structure Result where
  final : RunState
  probability : Rat

inductive Schedule : Nat → Nat → Type where
  | nil {n m : Nat} : Schedule n m
  | idle {n m : Nat} (rest : Schedule n m) : Schedule n m
  | birth {n m : Nat} (targets : Targets n m) (data : BirthData)
      (rest : Schedule (n + 1) m) : Schedule n m

inductive TickInput (n m : Nat) : Type where
  | idle
  | birth (targets : Targets n m) (data : BirthData)
```

Implement these exact interfaces: `extendPopulation {n} (p : Population n) (a : NewAgent) : Population (n+1)`; `grow {n m} (s : JointState n) (T : Targets n m) (hm : 0 < m) (b : BirthData) : JointState (n+1)`; `advance {n} (s : JointState n) : JointState n`; `runTyped {n m} (s : JointState n) (hm : 0 < m) (schedule : Schedule n m) (roundIndex : Nat := 0) : Result`; and `tick` with the same state, `hm`, and round argument but one `TickInput n m`.

- [ ] **Write a typed birth RED.** Construct the seed network as the complete graph on `Fin 2`, with fitness `[1,1]`; prove its connectedness by the zero-length walk for equal vertices and one edge otherwise. Its population is profiles `fun _ => ⟨1,1/2⟩`, beliefs `[1,0]`, exposures `[0,0]`, with bounds discharged by finite cases and `norm_num`. Name it `seed2` in the test namespace. Define:

```lean
def newbornZero : NewAgent :=
  { profile := ⟨1, 1/2⟩, initialBelief := 0
    profileValid := by norm_num [AgentProfile.Valid]
    beliefValid := by norm_num }
def birthOne : BirthData := ⟨⟨1, by norm_num⟩, newbornZero⟩
def target1 : Targets 2 1 := ⟨![1], by decide⟩
def joined := grow seed2 target1 (by decide) birthOne

example : joined.population.agents (oldId 2 0) = seed2.population.agents 0 := by
  simpa [joined] using grow_old_state seed2 target1 (by decide) birthOne 0
example : joined.population.agents (newId 2) = ⟨0,0⟩ := by decide_cbv
example : List.ofFn (fun i => ((advance joined).population.agents i).belief) =
    [1,1,0] := by decide_cbv
```

- [ ] **Run RED:** `lake env lean NarrativeDynamics/Tests/FitnessABM.lean` and record the absent typed composition surface.
- [ ] **Implement identity transport and growth.** Set profiles to `Fin.lastCases a.profile p.profiles`; states to `Fin.lastCases ⟨a.initialBelief,0⟩ p.agents`; prove validity by `Fin.lastCases`. Define the new network only with `applyBirth s.network T hm b.fitness`. Export `extendPopulation_valid`, `grow_projection`, `grow_old_state`, and `grow_new_state`. Export `grow_old_profile` as the corresponding profile identity theorem.
- [ ] **Implement one synchronous advance.** Install `s.network.snapshot.adjDec`, use `propagate s.network.snapshot.graph.Adj s.population`, and derive its validity with `propagate_valid`. Export `advance_projection` for unchanged network and `advance_profiles` for unchanged profiles.
- [ ] **Implement structural schedule recursion.** Use these equations, with `tail.final` retained and probability multiplied only at births:

```text
runTyped s hm nil r = Result(RunState(n,r,s), 1)
runTyped s hm (idle rest) r = runTyped (advance s) hm rest (r+1)
runTyped s hm (birth T b rest) r =
  let tail = runTyped (advance (grow s T hm b)) hm rest (r+1)
  Result(tail.final, orderedMass s.network T * tail.probability)
tick s hm idle r = runTyped s hm (idle nil) r
tick s hm (birth T b) r = runTyped s hm (birth T b nil) r
```

Define `Schedule.tickCount` and `Schedule.birthCount` by recursion: both are zero at `nil`; `idle` adds only one tick; `birth` adds one to both. Prove `runTyped_counts`: final node count is `n + birthCount`, final round index is `r + tickCount`, and final edge count is the original edge count plus `m * birthCount`. Prove `runTyped_probability_pos` from `orderedMass_pos` and multiplication of positive masses. Totality comes from structural recursion, not an assumption about arbitrary callbacks.

- [ ] **Prove target-order topology and behavior agreement.** Export `grow_order_irrelevant`: if `T.selected = U.selected`, then the grown graphs and their one-round propagated population observations agree. Reuse `birth_order_irrelevant` and identical population extension; do not assert equal ordered probabilities. The comparison observes profiles, beliefs, and exposures, rather than proof fields.
- [ ] **Add successive-birth and idle-tick tests.** In the seed fixture, use `.birth target1 birthOne (.birth ⟨![2], by decide⟩ birthOne .nil)`. Require beliefs `[1,1,1,0]`, exposures `[1,2,1,0]`, node count four, edge count three, round index two, and probability `1/8`. Appending `.idle .nil` instead of the final `.nil` must make agent `3` receive at round three. Also test an immediately broadcasting newborn supplied with belief one, old profile retention, and unit probability for an idle-only schedule.
- [ ] **Run GREEN, then commit:**

```bash
lake build NarrativeDynamics.Core.FitnessABM
timeout --kill-after=10s 240s lake env lean NarrativeDynamics/Tests/FitnessABM.lean
python3 tools/audit_fitness_trust.py source NarrativeDynamics/Core/FitnessABM.lean NarrativeDynamics/Tests/FitnessABM.lean
git diff --check
git add NarrativeDynamics/Core/FitnessABM.lean NarrativeDynamics/Tests/FitnessABM.lean
git commit -m "feat(lean): compose BB births with synchronous agent updates"
```

## Task 3: Checked agent input and atomic joint replay

**Files:** Create the two `FitnessABMReplay.lean` files.

**Consumes:** Task 2 joint state, extension, advance, typed schedule, and result; existing `RawSeed`, `RawBirth`, `parseSeed`, `step`, `step_spec`, BB raw validity predicates, and BB `replay`.

**Produces:** Continue namespace `NarrativeDynamics.FitnessABM` with these data contracts:

```lean
structure RawAgent where
  receptivity : Rat
  threshold : Rat
  belief : Rat
  exposures : Nat
  deriving DecidableEq, Repr

structure RawBirthInput where
  birth : FitnessAttachment.RawBirth
  receptivity : Rat
  threshold : Rat
  belief : Rat

abbrev RawTick := Option RawBirthInput

inductive AgentField where
  | receptivity | threshold | belief
  deriving DecidableEq, Repr

inductive BirthError where
  | network (cause : FitnessAttachment.Internal.Error)
  | agent (field : AgentField)
  deriving DecidableEq, Repr

inductive JointError where
  | seedNetwork (cause : FitnessAttachment.Internal.Error)
  | initialM
  | seedAgentCount (expected actual : Nat)
  | seedAgent (index : Nat) (field : AgentField)
  | tickNetwork (tickIndex birthIndex : Nat) (cause : FitnessAttachment.Internal.Error)
  | tickAgent (tickIndex birthIndex : Nat) (field : AgentField)
  deriving DecidableEq, Repr
```

Implement `parseAgent (raw : RawAgent) : Except AgentField {pair : AgentProfile × AgentState // pair.1.Valid ∧ pair.2.Valid}`; `parseAgents (n : Nat) (raw : Array RawAgent) : Except JointError {p : Population n // p.Valid}`; `checkedBirth {n} (s : JointState n) (m : Nat) (raw : RawBirthInput) : Except BirthError (JointState (n+1) × Rat)`; `runInputs (m tickIndex birthIndex : Nat) (s : RunState) (ticks : List RawTick) : Except JointError Result`; and `replay (seed : FitnessAttachment.RawSeed) (m : Nat) (agents : Array RawAgent) (ticks : List RawTick) : Except JointError Result`.

- [ ] **Write checked replay RED with an observable summary.** In the test namespace define:

```lean
def seedRaw : FitnessAttachment.RawSeed := ⟨2, #[1,1], #[(0,1)]⟩
def agentsRaw : Array RawAgent := #[⟨1,1/2,1,0⟩, ⟨1,1/2,0,0⟩]
def attach0 : RawBirthInput := ⟨⟨1, #[0]⟩, 1, 1/2, 0⟩
def attach1 : RawBirthInput := ⟨⟨1, #[1]⟩, 1, 1/2, 0⟩
def summary (r : Except JointError Result) :
    Except JointError (Nat × Nat × Nat × List Rat × List Nat × Rat) :=
  r.map fun out =>
    (out.final.nodeCount, out.final.roundIndex,
     FitnessAttachment.actualEdgeCount out.final.state.network.snapshot,
     List.ofFn (fun i => (out.final.state.population.agents i).belief),
     List.ofFn (fun i => (out.final.state.population.agents i).exposures),
     out.probability)

example : summary (replay seedRaw 1 agentsRaw [some attach0]) =
    .ok (3,1,2,[1,1,1],[0,1,1],1/2) := by decide_cbv
example : summary (replay seedRaw 1 agentsRaw [some attach1]) =
    .ok (3,1,2,[1,1,0],[0,1,0],1/2) := by decide_cbv
example : summary (replay seedRaw 1 agentsRaw [some attach1,none]) =
    .ok (3,2,2,[1,1,1],[1,2,1],1/2) := by decide_cbv
```

- [ ] **Run RED:** `lake env lean NarrativeDynamics/Tests/FitnessABMReplay.lean`. Record the absent checked joint replay rather than a dependency-install failure.
- [ ] **Implement scalar and roster validation in the approved order.** `parseAgent` checks receptivity bounds, threshold bounds, then belief bounds, returning the original values plus proofs; it does not clamp. `parseAgents` checks array size first, traverses records in ascending index order, and returns a population only after all records validate. Use a structurally recursive traversal that retains the first failure. Prove `parseAgent_sound`, `parseAgent_complete`, `parseAgents_sound`, and `parseAgents_complete` from the actual branch conditions.
- [ ] **Implement `checkedBirth` by calling existing `step` first.** On BB failure, return `.network cause`. On success `(nextNetwork,mass)`, validate the newborn as a `RawAgent` with exposure zero. Build its `NewAgent` from those checked values, extend the population onto `nextNetwork`, and return the same `mass`. Use `step_spec` to prove `checkedBirth_spec`: every success equals the typed `grow` with the validated targets, fitness and agent data, paired with the existing `orderedMass`. Do not call a second attachment kernel or modify the raw target order.
- [ ] **Implement the raw run recursion with separate indices.** For `none`, advance once, increment round and tick index, keep birth index and probability factor one. For `some raw`, call `checkedBirth`, wrap an error with the current two indices, advance the successful grown state, and increment both indices. Only after the suffix succeeds return its final state with the current birth mass multiplied in. Empty input returns the current state with mass one. The outer `replay` applies `parseSeed`, the fixed `m` check, then `parseAgents`, and starts all indices at zero.
- [ ] **Define exact validity and projection predicates.** Add `RawAgent.Valid` as the three interval requirements and `RawBirthInput.AgentValid` for its corresponding fields. Define `WellFormedTicks n m`: true at the empty list; recurse with unchanged `n` for `none`; for a birth require `raw.birth.Valid n m`, `raw.AgentValid`, and `WellFormedTicks (n+1) m rest`. Define `ReplayInputValid` as BB seed validity, the initial `m` requirement, exact agent-array length, every seed-agent validity, and this tick validity. Define `inputBirths (ticks : List RawTick) : List FitnessAttachment.RawBirth` by `filterMap` of each supplied birth.
- [ ] **Prove execution and BB agreement.** Export `replay_success_iff_valid` with `(∃ out, replay seed m agents ticks = .ok out) ↔ ReplayInputValid seed m agents ticks`. Export `replay_projection`: a successful joint result projects to a successful existing BB replay on `inputBirths ticks`, with the same node count, adjacency, stored fitness, and probability. Compare the existential graph carriers with a witnessed node-count equality and the resulting `Fin.cast`; never replace this with a comparison of counters alone. Derive `replay_counts` and `replay_probability_pos` through this projection and the tick-count induction.
- [ ] **Prove failure stability.** Export `runInputs_append_error`: if a prefix already returns `.error e`, appending a suffix returns that same error. Prove it by induction on the prefix, preserving the exact current state and indices. The output type exposes no prefix state or accumulated probability on error. Add conflicting-error examples in addition to the generic theorem.
- [ ] **Add the complete failure matrix.** Use exact error constructors, including both positions:

| Input change | Expected first error |
| --- | --- |
| Bad BB seed plus invalid agent fields | `seedNetwork` with original BB cause |
| Valid seed, `m=0` or `m>2`, and empty schedule | `initialM` |
| Valid seed and `m`, one seed agent instead of two | `seedAgentCount 2 1` |
| Seed agent 0 receptivity below zero and threshold above one | `seedAgent 0 receptivity` |
| Seed agent 1 invalid belief | `seedAgent 1 belief` |
| Invalid newborn fitness and invalid newborn belief together | `tickNetwork 0 0 nonpositiveFitness` |
| Valid BB birth, invalid newborn threshold and belief | `tickAgent 0 0 threshold` |
| `m=1`, a two-target tuple with repeated targets | `tickNetwork 0 0 targetCountMismatch` |
| `m=2`, a two-target tuple with repeated targets | `tickNetwork 0 0 duplicateTarget` |
| At node count two, target ID two | `tickNetwork 0 0 targetOutOfRange` |
| `[none, some attach1, none, some bad]`, where `bad` targets ID three | `tickNetwork 3 1 targetOutOfRange` |

Also assert the empty replay summary `.ok (2,0,1,[1,0],[0,0],1)` and idle-only projection to empty BB replay. For an `m=2` seed edge, target orders `[0,1]` and `[1,0]` under seed fitness `[1,3]` produce masses `1/4` and `3/4`, identical grown graphs, and identical agent results. Preserve both ordered requests.

- [ ] **Run GREEN and commit.** Use symbolic step equations and small observational summaries if full raw reduction becomes expensive; do not expand validity proofs merely to compute three-node data.

```bash
lake build NarrativeDynamics.Core.FitnessABMReplay
/usr/bin/time -f 'FitnessABMReplay elapsed=%e s peak_rss=%M KiB' timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMReplay.lean
python3 tools/audit_fitness_trust.py source NarrativeDynamics/Core/FitnessABMReplay.lean NarrativeDynamics/Tests/FitnessABMReplay.lean
git diff --check
git add NarrativeDynamics/Core/FitnessABMReplay.lean NarrativeDynamics/Tests/FitnessABMReplay.lean
git commit -m "feat(lean): validate and replay finite BB agent evolution"
```

## Task 4: Exact probabilities of agent outcomes

**Files:** Create the two `FitnessABMDistribution.lean` files.

**Consumes:** Task 2 `BirthData`, `Schedule`, `JointState`, `runTyped`; existing `TargetTrace`, `traceProbability`, `traceFinal`, and `traceProbability_sum_one`. Task 3 supplies independently checked raw acceptance examples.

**Produces:** In `NarrativeDynamics.FitnessABM`, `Calendar := List (Option BirthData)`; `fitnessSchedule (calendar : Calendar) : List PosFitness`; `scheduleOfTrace {n m} (calendar : Calendar) : TargetTrace n m (fitnessSchedule calendar).length → Schedule n m`; and these outcome/probability interfaces:

```text
jointFinal {n} (s : JointState n) (m : Nat) (hm : 0 < m)
  (calendar : Calendar) (trace : TargetTrace n m (fitnessSchedule calendar).length)
  : RunState

jointProbability {n} (s : JointState n) (m : Nat) (hm : 0 < m) (hb : m <= n)
  (calendar : Calendar) (trace : TargetTrace n m (fitnessSchedule calendar).length)
  : Rat

eventProbability {n} (s : JointState n) (m : Nat) (hm : 0 < m) (hb : m <= n)
  (calendar : Calendar) (event : RunState → Prop) [DecidablePred event]
  : Rat
```

- [ ] **Write the agent-event RED.** In test namespace `NarrativeDynamics.FitnessABM.DistributionFixtures`, define the two-node joint seed directly: complete graph on `Fin 2`, connected by at most one edge, unit fitness, profiles `⟨1,1/2⟩`, and agent states `[⟨1,0⟩,⟨0,0⟩]`. Name it `seed2`. Define `birthOne` with unit positive fitness and `NewAgent` profile `⟨1,1/2⟩`, initial belief zero, and bounds proved by `norm_num`. Define this total observable over the existential final carrier:

```lean
def newbornBroadcasts (out : RunState) : Bool :=
  if h : 2 < out.nodeCount then
    NetworkPropagation.broadcasting
      (out.state.population.profiles ⟨2,h⟩)
      (out.state.population.agents ⟨2,h⟩)
  else false

theorem newborn_mass_unit : eventProbability seed2 1 (by decide) (by decide)
    [some birthOne] (fun out => newbornBroadcasts out = true) = 1/2 := by
  decide_cbv
```

- [ ] **Run RED:** `lake env lean NarrativeDynamics/Tests/FitnessABMDistribution.lean` and record the missing joint-event surface.
- [ ] **Attach the existing trace to the fixed calendar.** Define `fitnessSchedule` by filtering births and mapping `BirthData.fitness`. Recurse over the calendar in `scheduleOfTrace`: the empty calendar gives `.nil`, `none` gives `.idle` without consuming a target, and `some data` consumes the head target and constructs `.birth`. Use explicit length equalities and transport at dependent boundaries where Lean requires them. Do not add a second target-trace type or Fintype enumeration.
- [ ] **Define outcomes and probabilities from existing functions.** Set `jointFinal` to `(runTyped s hm (scheduleOfTrace calendar trace)).final`. Set `jointProbability` to `traceProbability s.network m hm hb (fitnessSchedule calendar) trace`. Use the following event sum, with the existing executable Fintype instance; never discard traces based on their resulting agent beliefs:

```lean
def eventProbability {n : Nat} (s : JointState n) (m : Nat)
    (hm : 0 < m) (hb : m <= n) (calendar : Calendar)
    (event : RunState → Prop) [DecidablePred event] : Rat :=
  ∑ trace : TargetTrace n m (fitnessSchedule calendar).length,
    if event (jointFinal s m hm calendar trace)
    then jointProbability s m hm hb calendar trace else 0
```
- [ ] **Prove the probability bridge.** Export `jointProbability_eq_runTyped`: the existing trace probability equals the mass actually returned by the composed run. Induct over the calendar; idle contributes no mass or network change, while a birth uses `advance_projection` before the next factor. Export `jointFinal_projection` comparing its graph observations with `traceFinal`, `jointProbability_sum_one` by the existing normalized law, and `eventProbability_true`, `eventProbability_false`, `eventProbability_nonneg`, and `eventProbability_le_one` by finite sums. The upper bound compares each summand with its nonnegative trace mass, then uses normalization.
- [ ] **Prove the complete exact acceptance set:**

```text
unit seed fitness, one birth: newbornBroadcasts probability = 1/2
seed fitness [1,3], same one-birth agent data: probability = 1/4
unit seed fitness, one birth then idle: probability = 1
unit seed fitness, target sequence [1], [2]: composed run mass = 1/8
calendar with no births: one target trace; event mass is 0 or 1
empty calendar: final round zero and mass 1
m=2, seed fitness [1,3], same full target set in two orders:
  masses 1/4 and 3/4 both counted; always-true event mass = 1
```

Keep the changed-fitness comparison conditional: it changes target probabilities, not the propagation rule on a chosen graph. For the one-birth unit-fitness case, verify both final graphs have diameter two with existing graph witnesses while the newborn outcome differs. This documents why graph diameter alone does not determine an ABM outcome.

Name the weighted-seed equality `newborn_mass_weighted` and the birth-then-idle equality `newborn_mass_after_idle` in the same test namespace. Print their actual axiom reports with that of `newborn_mass_unit` and require all three in the final audit.

- [ ] **Run GREEN and commit.** Keep the event carrier tiny and compute only observational data at fixture boundaries.

```bash
lake build NarrativeDynamics.Core.FitnessABMDistribution
/usr/bin/time -f 'FitnessABMDistribution elapsed=%e s peak_rss=%M KiB' timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMDistribution.lean
python3 tools/audit_fitness_trust.py source NarrativeDynamics/Core/FitnessABMDistribution.lean NarrativeDynamics/Tests/FitnessABMDistribution.lean
git diff --check
git add NarrativeDynamics/Core/FitnessABMDistribution.lean NarrativeDynamics/Tests/FitnessABMDistribution.lean
git commit -m "feat(lean): prove exact probabilities of BB agent outcomes"
```

## Task 5: Shared Lean/Python propagation examples

**Files:** Create the three conformance files in the ownership table. Do not modify `narrative_dynamics/abm/simulation.py` or its contracts.

**Consumes:** `FitnessABM.replay` for validating a supplied snapshot with an empty tick list, `advance` for its one actual next state, `NetworkPropagation.transmissions` for its actual directed deliveries, and Python's existing public `simulate_round`.

**Produces:** A deterministic Lean executable that writes one JSON object, a checked-in corpus generated by that executable, and a Python test that compares each vector with one independently valid V1 round. This layer does not sample BB traces or construct a changing-roster Python trajectory.

- [ ] **Write the corpus-consumer RED.** Use the following test body and imports in `tests/test_network_abm_bb_conformance.py`. All chosen numerical examples have exactly representable binary values, so require exact equality and use no tolerance:

```python
import json
from fractions import Fraction
from pathlib import Path
import unittest

from narrative_dynamics.abm.contracts import (
    NetworkABMModel, NetworkAgentSpec, NetworkAgentState,
    PopulationState, SocialEdge, SocialNetwork,
)
from narrative_dynamics.abm.simulation import simulate_round

CORPUS = Path(__file__).resolve().parents[1] / "conformance" / "bb_abm_v1.json"

def number(text):
    return float(Fraction(text))

class BBPropagationConformanceTests(unittest.TestCase):
    def test_shared_vectors_match_existing_v1_round(self):
        corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(corpus["schema"], "bb-abm-v1")
        self.assertEqual(len(corpus["vectors"]), 8)
        self.assertEqual({v["id"] for v in corpus["vectors"]}, {
            "attach-source", "attach-relay", "relay-next-round", "second-birth",
            "half-receptive", "zero-receptive", "silent", "zero-threshold",
        })
        for vector in corpus["vectors"]:
            for reverse in (False, True):
                with self.subTest(case=vector["id"], reverse=reverse):
                    raw, expected = vector["input"], vector["expected"]
                    ids = tuple(str(i) for i in range(len(raw["beliefs"])))
                    profiles = tuple(
                        NetworkAgentSpec(
                            agent_id, "peer", number(raw["receptivity"][i]),
                            0.5, number(raw["thresholds"][i]),
                        ) for i, agent_id in enumerate(ids)
                    )
                    edges = tuple(
                        SocialEdge(str(source), str(target), "peer", 1.0)
                        for u, v in raw["edges"]
                        for source, target in ((u, v), (v, u))
                    )
                    model = NetworkABMModel(
                        "bb-conformance:" + vector["id"], "1",
                        profiles[::-1] if reverse else profiles,
                        SocialNetwork(
                            ids[::-1] if reverse else ids,
                            edges[::-1] if reverse else edges,
                        ),
                    )
                    states = tuple(
                        NetworkAgentState(
                            agent_id, number(raw["beliefs"][i]), raw["exposures"][i],
                            number(raw["beliefs"][i]) >= number(raw["thresholds"][i]),
                        ) for i, agent_id in enumerate(ids)
                    )
                    prior = PopulationState(
                        model.model_id, model.content_hash, 0, None,
                        states[::-1] if reverse else states,
                    )
                    result = simulate_round(model, prior)
                    actual = sorted(result.next_state.agents, key=lambda a: int(a.agent_id))
                    self.assertEqual([a.belief for a in actual], list(map(number, expected["beliefs"])))
                    self.assertEqual([a.exposure_count for a in actual], expected["exposures"])
                    self.assertEqual([a.broadcasting for a in actual], expected["broadcasting"])
                    delivered = sorted(
                        [int(t.source_agent_id), int(t.target_agent_id)]
                        for t in result.transmissions
                    )
                    self.assertEqual(delivered, expected["transmissions"])
```

The explicit `PopulationState` construction is essential: `initialize_population` counts supplied beliefs as an exposure and would change the approved vectors. Each vector is an independent round-zero snapshot; no artificial cross-model parent hash is introduced.

- [ ] **Run RED:** `python -m unittest tests.test_network_abm_bb_conformance -v`. The expected initial failure is the absent shared corpus, not an unrelated import error.
- [ ] **Define the eight exact inputs in the Lean exporter.** Give every listed graph unit fitness, `m=1`, default receptivity one, and default threshold `1/2`. The graphs are post-growth snapshots for a single propagation comparison. All prior exposures and exceptional parameters are explicit:

| ID | Undirected edges | Prior beliefs | Prior exposures | Parameter change | Required next beliefs | Required next exposures |
| --- | --- | --- | --- | --- | --- | --- |
| `attach-source` | `[[0,1],[0,2]]` | `[1,0,0]` | `[0,0,0]` | none | `[1,1,1]` | `[0,1,1]` |
| `attach-relay` | `[[0,1],[1,2]]` | `[1,0,0]` | `[0,0,0]` | none | `[1,1,0]` | `[0,1,0]` |
| `relay-next-round` | `[[0,1],[1,2]]` | `[1,1,0]` | `[0,1,0]` | none | `[1,1,1]` | `[1,2,1]` |
| `second-birth` | `[[0,1],[1,2],[2,3]]` | `[1,1,0,0]` | `[0,1,0,0]` | none | `[1,1,1,0]` | `[1,2,1,0]` |
| `half-receptive` | `[[0,1]]` | `[1,0]` | `[0,0]` | receiver 1 receptivity `1/2` | `[1,1/2]` | `[0,1]` |
| `zero-receptive` | `[[0,1]]` | `[1,0]` | `[0,0]` | receiver 1 receptivity `0` | `[1,0]` | `[0,1]` |
| `silent` | `[[0,1]]` | `[0,0]` | `[0,0]` | none | `[0,0]` | `[0,0]` |
| `zero-threshold` | `[[0,1]]` | `[0,0]` | `[0,0]` | source 0 threshold `0` | `[0,0]` | `[0,1]` |

Broadcasting is derived from the next beliefs and unchanged thresholds. Transmissions are derived from the input snapshot, never the next beliefs. Assert these exact input/output pairs in `NarrativeDynamics/Tests/NetworkPropagation.lean` or `NarrativeDynamics/Tests/FitnessABMReplay.lean` as applicable, so the exporter cannot silently redefine expected semantics by generating a new corpus.

- [ ] **Implement the exporter from actual computed state.** In namespace `NarrativeDynamics.Conformance.BBPropagation`, define `VectorInput` with fields `id : String`, `seed : FitnessAttachment.RawSeed`, and `agents : Array FitnessABM.RawAgent`; `vectorInputs : List VectorInput` contains the eight rows. Define `renderCase : VectorInput → Except String Lean.Json`: call joint `replay` with the provided seed/agents and no ticks, then compute `advance` and `transmissions` on the parsed joint state. An error becomes an exporter error, not an omitted vector.

Serialize `input.edges` once per undirected pair in ascending numeric order, and profiles/states in ascending `Fin` order. Rational values are JSON strings from `toString` of the actual `Rat`; exposure counts and IDs are JSON numbers; broadcasting values are JSON booleans. Use `Lean.Json` constructors and `Lean.Json.mkObj`, not manual string escaping. A row has precisely this shape:

```json
{"id":"attach-relay","input":{"edges":[[0,1],[1,2]],"beliefs":["1","0","0"],"exposures":[0,0,0],"receptivity":["1","1","1"],"thresholds":["1/2","1/2","1/2"]},"expected":{"beliefs":["1","1","0"],"exposures":[0,1,0],"broadcasting":[true,true,false],"transmissions":[[0,1]]}}
```

Define `renderCorpus : Except String Lean.Json` by traversing every vector through `renderCase` and wrapping the array with `schema = "bb-abm-v1"`. `main : IO Unit` prints its compressed JSON plus one newline on success and throws `IO.userError` on failure. Keep all top-level axiom reports in test modules, not in exporter stdout.

- [ ] **Generate and compare GREEN.** The first generation creates the checked-in corpus. Subsequent verification always generates to a temporary path and compares, rather than silently updating the golden file:

```bash
lake build NarrativeDynamics.Core.FitnessABMReplay
lake env lean --run NarrativeDynamics/Conformance/FitnessABMVectors.lean > conformance/bb_abm_v1.json
python -m unittest tests.test_network_abm_bb_conformance tests.test_network_abm_simulation tests.test_network_abm_contracts -v
```

- [ ] **Check that the comparison can detect disagreement.** Temporarily change `attach-relay`'s expected receiver-2 belief from zero to one in the corpus; the Python test must fail on that receiver. Regenerate the original corpus from Lean and require the test to pass. Record both outputs. This checks the cross-language comparison without modifying either production kernel.
- [ ] **Commit only the conformance surface and associated exact fixture assertions:**

```bash
git diff --check
git add NarrativeDynamics/Conformance/FitnessABMVectors.lean conformance/bb_abm_v1.json tests/test_network_abm_bb_conformance.py NarrativeDynamics/Tests/NetworkPropagation.lean NarrativeDynamics/Tests/FitnessABMReplay.lean
git commit -m "test: compare BB agent propagation with shared Lean vectors"
```

## Task 6: Public proof surface, trust audit, and complete CI

**Files:** Create `tools/check_fitness_abm.sh`; modify `.github/workflows/proof.yml`, `NarrativeDynamics.lean`, `README.md`, and the four new Lean test modules. The existing auditor does not need an API change: it already accepts explicit source paths and required theorem names.

**Consumes:** All earlier task modules, exact fixture assertions, generated corpus, and the existing source/log trust checker.

**Produces:** One bounded proof gate that fails on missing proofs, missing dependency reports, timeouts, or corpus disagreement; root imports for the four new core modules; and README claims supported by the completed proof and comparison evidence.

- [ ] **Write the integration RED.** Change the first import of `NarrativeDynamics/Tests/FitnessABMDistribution.lean` to `import NarrativeDynamics` and add `#check NarrativeDynamics.FitnessABM.replay_projection`. Before new root imports, a fresh build/test must fail to resolve that public surface through the root. Also run the existing log auditor with an absent required new theorem report and confirm rejection; retain the existing audit regression tests rather than adding a duplicate auditor implementation.
- [ ] **Add the root imports after the corresponding modules exist:**

```lean
import NarrativeDynamics.Core.NetworkPropagation
import NarrativeDynamics.Core.FitnessABM
import NarrativeDynamics.Core.FitnessABMReplay
import NarrativeDynamics.Core.FitnessABMDistribution
```

Preserve the existing imports and a final newline. Do not import tests or the conformance executable into the public root.

- [ ] **Append actual dependency reports to each test module.** For every name in the script's `proof_names` array below, add `#print axioms NarrativeDynamics.<name>` to the test that owns that theorem. Keep helper lemmas private unless they are part of the specified public API. If an additional public theorem is necessary, add both its printed report and its required name; do not weaken the allowlist or remove a requirement to make a log pass.
- [ ] **Create the following bounded check script.** It runs after `lake build`, from any directory, and reuses the existing audit CLI. The four proof test suites run sequentially to avoid concurrent peak-memory demand:

```bash
#!/usr/bin/env bash
set -euo pipefail
bb_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$bb_repo_root"
bb_audit_log="$(mktemp)"
bb_vectors="$(mktemp)"
trap 'rm -f "$bb_audit_log" "$bb_vectors"' EXIT

python3 tools/audit_fitness_trust.py source \
  NarrativeDynamics/Core/Fitness*.lean \
  NarrativeDynamics/Tests/Fitness*.lean \
  NarrativeDynamics/Core/NetworkPropagation.lean \
  NarrativeDynamics/Tests/NetworkPropagation.lean \
  NarrativeDynamics/Conformance/FitnessABMVectors.lean

for bb_suite in NetworkPropagation FitnessABM FitnessABMReplay FitnessABMDistribution; do
  /usr/bin/time -f "$bb_suite elapsed=%e s peak_rss=%M KiB" \
    timeout --kill-after=10s 240s \
    lake env lean -DmaxErrors=1 "NarrativeDynamics/Tests/$bb_suite.lean" \
    2>&1 | tee -a "$bb_audit_log"
done

proof_names=(
  NetworkPropagation.propagate_valid
  NetworkPropagation.transmission_iff
  NetworkPropagation.nextAgent_no_incoming
  NetworkPropagation.exposures_mono
  NetworkPropagation.nextAgent_locality
  FitnessABM.extendPopulation_valid
  FitnessABM.grow_projection
  FitnessABM.grow_old_state
  FitnessABM.grow_new_state
  FitnessABM.grow_old_profile
  FitnessABM.advance_projection
  FitnessABM.advance_profiles
  FitnessABM.runTyped_counts
  FitnessABM.runTyped_probability_pos
  FitnessABM.grow_order_irrelevant
  FitnessABM.parseAgent_sound
  FitnessABM.parseAgent_complete
  FitnessABM.parseAgents_sound
  FitnessABM.parseAgents_complete
  FitnessABM.checkedBirth_spec
  FitnessABM.replay_success_iff_valid
  FitnessABM.replay_projection
  FitnessABM.replay_counts
  FitnessABM.replay_probability_pos
  FitnessABM.runInputs_append_error
  FitnessABM.jointProbability_eq_runTyped
  FitnessABM.jointFinal_projection
  FitnessABM.jointProbability_sum_one
  FitnessABM.eventProbability_true
  FitnessABM.eventProbability_false
  FitnessABM.eventProbability_nonneg
  FitnessABM.eventProbability_le_one
  FitnessABM.DistributionFixtures.newborn_mass_unit
  FitnessABM.DistributionFixtures.newborn_mass_weighted
  FitnessABM.DistributionFixtures.newborn_mass_after_idle
)
bb_required=()
for bb_name in "${proof_names[@]}"; do
  bb_required+=(--require "NarrativeDynamics.$bb_name")
done
python3 tools/audit_fitness_trust.py log "$bb_audit_log" "${bb_required[@]}"

/usr/bin/time -f 'FitnessABMVectors elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean --run NarrativeDynamics/Conformance/FitnessABMVectors.lean > "$bb_vectors"
cmp conformance/bb_abm_v1.json "$bb_vectors"
```

The emitted JSON is executable comparison evidence, not a proof oracle. Kernel-checked fixture equalities and actual theorem axiom reports remain separate obligations. `pipefail` must preserve a failing Lean/timeout exit even when `tee` succeeds.

- [ ] **Wire the script into the existing Lean proof job.** Add this step after `Build Lean library`, leaving checkout, event selection, the old BB gates, and the full theorem suite intact:

```yaml
      - name: BB ABM joint evolution contracts
        timeout-minutes: 25
        run: bash tools/check_fitness_abm.sh
```

The job already adds Elan to `GITHUB_PATH`. The 25-minute outer step contains four independently limited proof suites and one independently limited exporter; it does not increase the 240-second limit of any individual check. Python's full discovery job already collects `test_network_abm_bb_conformance.py` after installing its locked dependencies, so the Lean-only job does not need those Python runtime imports or another dependency installation.

- [ ] **Write the verified README addition.** State the concrete model, fixed fitness, unit channel influence, birth-then-synchronous-propagation order, idle ticks, persistent old states, and the exact probability experiment. Show the two one-birth branches and probabilities `1/2` and `1/4`, linking the new theorem/test modules. State that Python comparison is finite single-round example agreement, and that V19 creation, adaptive fitness, and fixed-hop guarantees remain outside this proof. Use completed results only; do not present this planning document as proof evidence.
- [ ] **Run the local final gates available in the execution environment:**

```bash
bash -n tools/check_fitness_abm.sh
lake build
bash tools/check_fitness_abm.sh
python -m unittest tests.test_fitness_trust_audit tests.test_network_abm_bb_conformance tests.test_network_abm_simulation tests.test_network_abm_contracts -v
git diff --check
```

Do not repeat the complete Python discovery locally if the current revision's required CI will run it and no additional local risk requires that repetition. Run earlier BB acceptance gates through the unchanged workflow; compare their elapsed time and memory with the prior baseline if a regression appears.

- [ ] **Commit the integration and publish the implementation branch for review:**

```bash
git add tools/check_fitness_abm.sh .github/workflows/proof.yml NarrativeDynamics.lean README.md NarrativeDynamics/Tests/NetworkPropagation.lean NarrativeDynamics/Tests/FitnessABM.lean NarrativeDynamics/Tests/FitnessABMReplay.lean NarrativeDynamics/Tests/FitnessABMDistribution.lean
git commit -m "ci: verify BB ABM proofs and cross-language examples"
```

Open a draft PR against `proof/narrative-dynamics-v0`, describing the concrete missing composition, the new behavior, the proof scope, and the unchanged production Python boundary. Keep all task commits. Record the final implementation SHA and verify that the Lean proof and complete Python discovery jobs check that SHA; verify the required World Studio workflow according to its existing checkout/event semantics. A skipped job or a docs-only workflow result is not implementation evidence. Do not merge without the user's merge authorization.

- [ ] **Review the final diff and evidence.** Use the requesting-code-review workflow before declaring merge readiness. Check the generic proofs, concrete delayed-relay and probability examples, error precedence, root imports, corpus provenance, and absence of production Python contract edits. Resolve material findings and rerun only affected checks plus required CI. Report actual pass/failure/skip status, execution revision, and any remaining scope limits.

## Spec coverage and handoff

| Approved spec section | Implementation coverage |
| --- | --- |
| 1: Concrete ABM claim and limits | Tasks 1–4 proofs; Task 6 README/review |
| 2: Existing BB/V1/V5/V19 boundaries | Global constraints; Tasks 2–3 reuse; Task 5 independent fixed-roster comparisons |
| 3: State, fixed profiles/fitness, stable IDs | Task 1 contracts; Task 2 extension and preservation proofs |
| 4: Exact interaction semantics | Task 1 kernel, locality, numeric/direction examples; Task 5 shared vectors |
| 5: Growth, idle ticks, scheduling and atomic result | Task 2 typed evaluator; Task 3 raw evaluator and failure stability |
| 6: Validation, precedence and two indices | Task 3 soundness/completeness and complete failure matrix |
| 7: Required public proof surface | Tasks 1–3 theorem interfaces; Task 6 requires their actual dependency reports |
| 8: Conditional finite law and pushforward | Task 4 exact law, event bounds, duplicate-outcome and empty-calendar cases |
| 9: Acceptance scenarios | Tasks 1–5 concrete assertions and one shared corpus |
| 10: Modules, resources, conformance and CI | File ownership table; Tasks 5–6; pinned environment preparation |
| 11: Deferred extensions | Global constraints and Task 6 diff/scope review |
| 12: Plan before implementation | This plan; all implementation task checkboxes remain unchecked |

Before execution, read the approved spec and this plan together. Normal implementation choices within these contracts need no new design approval. If the work requires changing fitness based on behavior, creating V19 agents, weakening raw atomicity, or treating a floating-point comparison as a formal proof, that is outside the approved boundary and needs a revised design.

Execution can use a fresh implementer/reviewer per task under the subagent-driven-development skill, or proceed inline with executing-plans and task checkpoints. Select the execution mode at handoff; this planning commit contains no Lean or production implementation.
