# BB replay-to-idle-tail implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a reusable checked-replay → idle-tail → observation boundary from the existing BB Path4 fixtures without creating a second propagation abstraction or changing runtime semantics.

**Architecture:** Add a proof-neutral `IdleTailModel` for repeated single-step iteration plus observation, then a Fitness ABM adapter whose runtime transition is exactly `advance`. Prove append-idle theorems directly over the existing `runInputs`/`replay` recursion so node count, round index, underlying `JointState`, and accumulated replay probability are preserved exactly. Refactor the Path4 fixture to consume this boundary while leaving `FitnessABMPathN`/`FiniteConsensus` as the canonical propagation/convergence layers.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, exact `Rat`, existing `FitnessABM`, `FitnessABMReplay`, Path4/Path5/PathN gates, `tools/audit_fitness_trust.py`, GitHub proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-15-bb-replay-idle-tail-design.md`, approved at `2f716319748502d2d3d9cede273665352814162c`.

## Global Constraints

- Implementation starts from the committed spec/plan history based on `proof/narrative-dynamics-v0@798f868a5327d6dd48de4f7a2b54cdf7a1ffed2a`.
- `FitnessABMPathN` remains the canonical reusable finite-path propagation surface; do not add `PropagationModel` or another graph/population/step abstraction.
- `FiniteConsensus` remains the canonical generic convergence layer; do not add convergence mathematics to idle-tail modules.
- One idle runtime tick is exactly the existing `none` branch of `runInputs`: node count unchanged, round index incremented by one, underlying state updated with `FitnessABM.advance`, and no birth mass introduced.
- Appending idle ticks to a successful replay prefix preserves the prefix's accumulated probability exactly; it must not reset probability to `1`.
- Do not duplicate `checkedBirth`, parsing, `runInputs`, replay recursion, probability multiplication, or network update logic.
- Keep BBII/BIBI/IIBB fixtures, activation offsets, and concrete history names in Path4 test code only.
- Preserve theorem names and expected values for `replay_baseline`, `raw_tail_bridge`, activation theorems, common-clock convergence, Path5 generated replay, and PathN convergence.
- Do not modify `FitnessABMPathN.lean` or `FiniteConsensus.lean`.
- Do not use `sorry`, `admit`, `native_decide`, a new `axiom`, `unsafe`, `unlock_limits`, `set_option maxHeartbeats 0`, or unbounded proof-resource settings.
- Every direct focused Lean build/test in the final gate uses `timeout --kill-after=10s 240s`.
- Mandatory theorem reports must stay within `propext`, `Classical.choice`, and `Quot.sound`.
- A malformed consumer or missing toolchain is not valid RED evidence; repair the harness before production changes.

## File Map

| File | Responsibility | Tasks |
| --- | --- | --- |
| `NarrativeDynamics/Core/IdleTail.lean` | Proof-neutral repeated-step + observation interface | 1 |
| `NarrativeDynamics/Tests/IdleTail.lean` | Generic RED/GREEN consumers | 1, 5 |
| `NarrativeDynamics/Core/FitnessABMIdleTail.lean` | Exact `advance`-backed `JointState` and `RunState` adapters | 2 |
| `NarrativeDynamics/Tests/FitnessABMIdleTail.lean` | Adapter/replay consumers and replay axiom reports | 2–3, 5 |
| `NarrativeDynamics/Core/FitnessABMReplay.lean` | Append-idle theorems over existing `runInputs`/`replay` | 3 |
| `NarrativeDynamics/Tests/FitnessABMPath4.lean` | Existing concrete fixture; compatibility consumer | 4–5 |
| `tools/check_fitness_abm_path4.sh` | Bounded source/build/test/trust regression gate | 5 |

## Preparation Before Task 1

- [ ] Create the implementation branch from the plan head.

```bash
git checkout design/bb-replay-idle-tail-v1
git pull --ff-only
git checkout -b feature/bb-replay-idle-tail-v1
git rev-parse HEAD
```

- [ ] Confirm the starting diff contains only the approved design and plan.

```bash
git merge-base HEAD proof/narrative-dynamics-v0
git diff --stat proof/narrative-dynamics-v0...HEAD
```

- [ ] Re-read the real runtime/replay/fixture surfaces before editing.

```text
NarrativeDynamics/Core/FitnessABM.lean
NarrativeDynamics/Core/FitnessABMReplay.lean
NarrativeDynamics/Core/FitnessABMPath4.lean
NarrativeDynamics/Tests/FitnessABMPath4.lean
tools/check_fitness_abm_path4.sh
tools/audit_fitness_trust.py
```

---

### Task 1: Proof-neutral IdleTail interface

**Files:**
- Create: `NarrativeDynamics/Tests/IdleTail.lean`
- Create: `NarrativeDynamics/Core/IdleTail.lean`

**Interfaces:**
- Consumes: only function iteration.
- Produces exactly:

```lean
namespace NarrativeDynamics

structure IdleTailModel (State Obs : Type*) where
  step : State → State
  observe : State → Obs

namespace IdleTailModel

def trajectory (m : IdleTailModel State Obs) (s : State) (k : Nat) : State :=
  (m.step)^[k] s

def observedTrajectory
    (m : IdleTailModel State Obs) (s : State) (k : Nat) : Obs :=
  m.observe (trajectory m s k)

@[simp] theorem trajectory_zero
    (m : IdleTailModel State Obs) (s : State) :
    trajectory m s 0 = s

theorem trajectory_succ
    (m : IdleTailModel State Obs) (s : State) (k : Nat) :
    trajectory m s (k + 1) = m.step (trajectory m s k)

theorem trajectory_add
    (m : IdleTailModel State Obs) (s : State) (a b : Nat) :
    trajectory m s (a + b) = trajectory m (trajectory m s a) b

@[simp] theorem observedTrajectory_zero
    (m : IdleTailModel State Obs) (s : State) :
    observedTrajectory m s 0 = m.observe s

theorem observedTrajectory_succ
    (m : IdleTailModel State Obs) (s : State) (k : Nat) :
    observedTrajectory m s (k + 1) =
      m.observe (m.step (trajectory m s k))
```

- [ ] **Step 1: Write the failing consumer before the core module exists.**

```lean
import NarrativeDynamics.Core.IdleTail

open NarrativeDynamics

private def incrementModel : IdleTailModel Nat Nat :=
  { step := Nat.succ
    observe := id }

example : IdleTailModel.trajectory incrementModel 3 0 = 3 := by
  simpa using IdleTailModel.trajectory_zero incrementModel 3

example : IdleTailModel.trajectory incrementModel 3 2 = 5 := by
  decide

example (a b : Nat) :
    IdleTailModel.trajectory incrementModel 3 (a + b) =
      IdleTailModel.trajectory incrementModel
        (IdleTailModel.trajectory incrementModel 3 a) b := by
  exact IdleTailModel.trajectory_add incrementModel 3 a b

example (k : Nat) :
    IdleTailModel.observedTrajectory incrementModel 3 (k + 1) =
      incrementModel.observe
        (incrementModel.step
          (IdleTailModel.trajectory incrementModel 3 k)) := by
  exact IdleTailModel.observedTrajectory_succ incrementModel 3 k
```

- [ ] **Step 2: Run RED.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/IdleTail.lean
```

Expected RED: missing `NarrativeDynamics.Core.IdleTail` / `IdleTailModel` declarations.

- [ ] **Step 3: Implement the minimal core.**

Use only `Function.iterate_succ_apply'` and `Function.iterate_add_apply` (with an `add_comm` rewrite if required for the chosen orientation). Do not add BB/path/replay imports.

- [ ] **Step 4: Build and test.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.IdleTail
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/IdleTail.lean
```

- [ ] **Step 5: Audit the generic boundary.**

```bash
git grep -n -E 'Fitness|BB|Path|NetworkPropagation|advance|replay' -- \
  NarrativeDynamics/Core/IdleTail.lean
```

Expected: no matches.

- [ ] **Step 6: Commit.**

```bash
git add NarrativeDynamics/Core/IdleTail.lean \
        NarrativeDynamics/Tests/IdleTail.lean
git commit -m "feat(lean): add generic idle-tail iteration interface"
```

---

### Task 2: Exact Fitness ABM idle adapters

**Files:**
- Create: `NarrativeDynamics/Core/FitnessABMIdleTail.lean`
- Create: `NarrativeDynamics/Tests/FitnessABMIdleTail.lean`

**Interfaces:**
- Consumes: `IdleTailModel`, `JointState`, `RunState`, `advance`.
- Produces exactly:

```lean
namespace NarrativeDynamics.FitnessABM

open NarrativeDynamics

def jointIdleTail (n : Nat) :
    IdleTailModel (JointState n) (JointState n) :=
  { step := advance
    observe := id }

def observeJointWith {n : Nat} {Obs : Type*}
    (project : JointState n → Obs) :
    IdleTailModel (JointState n) Obs :=
  { step := advance
    observe := project }

def idleRunStep (s : RunState) : RunState :=
  ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩

def runIdleTail : IdleTailModel RunState RunState :=
  { step := idleRunStep
    observe := id }

def runIdleTrajectory (s : RunState) (k : Nat) : RunState :=
  IdleTailModel.trajectory runIdleTail s k

theorem runIdleTrajectory_eq (s : RunState) (k : Nat) :
    runIdleTrajectory s k =
      ⟨s.nodeCount, s.roundIndex + k, (advance^[k]) s.state⟩
```

The whole-record theorem is the authoritative bookkeeping statement. Derive these public corollaries from it:

```lean
@[simp] theorem idleRunStep_nodeCount (s : RunState) :
  (idleRunStep s).nodeCount = s.nodeCount

@[simp] theorem idleRunStep_roundIndex (s : RunState) :
  (idleRunStep s).roundIndex = s.roundIndex + 1

@[simp] theorem idleRunStep_state (s : RunState) :
  (idleRunStep s).state = advance s.state

@[simp] theorem runIdleTrajectory_nodeCount (s : RunState) (k : Nat) :
  (runIdleTrajectory s k).nodeCount = s.nodeCount

@[simp] theorem runIdleTrajectory_roundIndex (s : RunState) (k : Nat) :
  (runIdleTrajectory s k).roundIndex = s.roundIndex + k
```

No separate cast-heavy state theorem is required because `runIdleTrajectory_eq` already proves the dependent state field as part of the `RunState` equality.

- [ ] **Step 1: Add RED consumers with arbitrary well-typed states; no concrete fixture is needed.**

```lean
import NarrativeDynamics.Core.FitnessABMIdleTail

open NarrativeDynamics NarrativeDynamics.FitnessABM

example (s : RunState) :
    (idleRunStep s).nodeCount = s.nodeCount :=
  idleRunStep_nodeCount s

example (s : RunState) :
    (idleRunStep s).roundIndex = s.roundIndex + 1 :=
  idleRunStep_roundIndex s

example (s : RunState) (k : Nat) :
    runIdleTrajectory s k =
      ⟨s.nodeCount, s.roundIndex + k, (advance^[k]) s.state⟩ :=
  runIdleTrajectory_eq s k

example {n : Nat} (s : JointState n) (k : Nat) :
    IdleTailModel.trajectory (jointIdleTail n) s k = (advance^[k]) s := by
  rfl
```

- [ ] **Step 2: Run RED.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Expected RED: missing adapter declarations.

- [ ] **Step 3: Implement adapters and `runIdleTrajectory_eq`.**

`FitnessABMIdleTail.lean` imports only `IdleTail` and `FitnessABM`; it must not import `FitnessABMReplay`. Prove `runIdleTrajectory_eq` by induction on `k`, rewriting the successor case with `IdleTailModel.trajectory_succ` and `Nat.add_assoc`/`Nat.add_comm` only as needed.

- [ ] **Step 4: Build and test.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMIdleTail
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

- [ ] **Step 5: Audit semantic duplication.**

```bash
git grep -n -E 'propagate|checkedBirth|runInputs|parseAgent|orderedMass' -- \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean
```

Expected: no matches.

- [ ] **Step 6: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMIdleTail.lean \
        NarrativeDynamics/Tests/FitnessABMIdleTail.lean
git commit -m "feat(lean): add Fitness ABM idle-tail adapters"
```

---

### Task 3: Append-idle theorems over existing replay recursion

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMReplay.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMIdleTail.lean`

**Interfaces:**
- Consumes: existing `runInputs`, existing `replay`, `runIdleTrajectory`.
- Produces exactly:

```lean
theorem runInputs_replicate_idle
    (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩

theorem runInputs_append_idle
    (m tickIndex birthIndex : Nat)
    (s : RunState) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runInputs m tickIndex birthIndex s
        (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩

theorem replay_append_idle
    (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : replay seed m agents ticks = .ok out) :
    replay seed m agents (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩
```

`runInputs_replicate_idle` is the production extraction of the existing Path4 test-local `private theorem run_idle`.

- [ ] **Step 1: Add RED consumers for all three theorem signatures.**

```lean
example (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩ :=
  runInputs_replicate_idle m tickIndex birthIndex k s

example (m tickIndex birthIndex : Nat)
    (s : RunState) (ticks : List RawTick) (out : Result) (k : Nat)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runInputs m tickIndex birthIndex s
        (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ :=
  runInputs_append_idle m tickIndex birthIndex s ticks out k h

example (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : replay seed m agents ticks = .ok out) :
    replay seed m agents (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩ :=
  replay_append_idle seed m agents ticks out k h
```

The second and third consumers encode probability preservation in their theorem types; no fixture can accidentally weaken it to state-only equality.

- [ ] **Step 2: Run RED.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Expected RED: unknown append-idle theorem declarations while Task 2 adapter consumers remain GREEN.

- [ ] **Step 3: Implement `runInputs_replicate_idle`.**

Import `FitnessABMIdleTail` into `FitnessABMReplay.lean`. Induct on `k`, generalizing `s` and `tickIndex`; each successor unfolds exactly one existing `none` branch and uses `runIdleTrajectory_eq`/`IdleTailModel.trajectory_succ`. Do not define another runner.

- [ ] **Step 4: Implement `runInputs_append_idle`.**

Induct on `ticks`, generalizing `s`, `tickIndex`, `birthIndex`, and `out`.

```text
nil:
  successful h fixes out to ⟨s,1⟩;
  reduce the suffix to runInputs_replicate_idle.

none :: rest:
  unfold one existing none branch;
  feed the recursive success equality into the induction hypothesis.

some raw :: rest:
  split only on existing checkedBirth and recursive runInputs results;
  h eliminates every error branch;
  apply the induction hypothesis to the successful recursive tail;
  rebuild the existing result with the unchanged next.2 * tail.probability factor.
```

The final RHS must remain `out.probability`, never `1`.

- [ ] **Step 5: Implement `replay_append_idle`.**

Unfold `replay` in `h` and the goal. Case-split on the existing `parseSeed`, `0 < m ∧ m ≤ seed.nodeCount`, and `parseAgents` branches. Use `h` to eliminate failure branches; in the unique successful branch apply `runInputs_append_idle`.

- [ ] **Step 6: Add mandatory axiom reports.**

```lean
#print axioms NarrativeDynamics.FitnessABM.runInputs_append_idle
#print axioms NarrativeDynamics.FitnessABM.replay_append_idle
```

- [ ] **Step 7: Build and run GREEN.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMReplay
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Expected: all Task 2/3 consumers pass; the two reports contain only the standard axiom allowlist.

- [ ] **Step 8: Review for replay duplication.**

Inspect the Task 3 diff. There must be no new `def` that recursively consumes `List RawTick`; only the existing `runInputs` remains the replay state machine.

- [ ] **Step 9: Commit.**

```bash
git add NarrativeDynamics/Core/FitnessABMReplay.lean \
        NarrativeDynamics/Tests/FitnessABMIdleTail.lean
git commit -m "feat(lean): bridge checked replay to idle tails"
```

---

### Task 4: Convert Path4 tail fixtures into compatibility consumers

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPath4.lean`

**Interfaces:**
- Consumes: `jointIdleTail`, `observeJointWith`, `runIdleTrajectory`, `replay_append_idle`.
- Preserves existing names/results for `replay_baseline`, `raw_tail_bridge`, activation theorems, and common-clock convergence.
- Adds exactly:

```lean
theorem tailState_idleTail (h : History) (k : Nat) :
    tailState h k =
      IdleTailModel.trajectory (jointIdleTail 4) (baseState h) k

theorem tailBelief_idleTail (h : History) (k : Nat) :
    tailBelief h k =
      IdleTailModel.observedTrajectory
        (observeJointWith
          (fun s : JointState 4 =>
            NarrativeDynamics.FitnessABMPath4.project s.population))
        (baseState h) k
```

- [ ] **Step 1: Import the adapter and prove both compatibility theorems against the existing `tailState`/`tailBelief` definitions before changing replay proofs.**

Expected proof: `rfl` or a single unfold/simp using the adapter definitions. If more is required, fix the adapter interface rather than changing Path4 values.

- [ ] **Step 2: Run the full Path4 test once.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
```

Expected: GREEN.

- [ ] **Step 3: Remove only the test-local recursive `private theorem run_idle`.**

The production `runInputs_replicate_idle` now owns that generic fact.

- [ ] **Step 4: Move the existing fixed four-tick fixture proof into a prefix-only theorem.**

```lean
private theorem replay_prefix (h : History) :
    FitnessABM.replay rawSeed 1 rawAgents (ticks h) =
      .ok ⟨⟨4,4,baseState h⟩,1/8⟩ := by
  -- use the existing checked fixture lemmas for the four scheduled ticks
```

The body is obtained by taking the current `raw_tail_bridge` fixture calculation and deleting its generic `List.replicate k none` tail step. No new birth/network proof is introduced.

- [ ] **Step 5: Preserve `replay_baseline` with the same exact statement.**

```lean
theorem replay_baseline (h : History) :
    rawTail h 0 = .ok ⟨⟨4,4,baseState h⟩,1/8⟩ := by
  simpa [rawTail] using replay_prefix h
```

- [ ] **Step 6: Reprove `raw_tail_bridge` exclusively through the production append theorem.**

```lean
theorem raw_tail_bridge (h : History) (k : Nat) :
    rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩ := by
  have hp := replay_append_idle
    rawSeed 1 rawAgents (ticks h)
    ⟨⟨4,4,baseState h⟩,1/8⟩ k (replay_prefix h)
  rw [runIdleTrajectory_eq] at hp
  simpa [rawTail, tailState] using hp
```

This is the concrete non-unit probability regression: `1/8` must remain `1/8` for every idle suffix length.

- [ ] **Step 7: Keep downstream activation/common-clock theorem bodies unchanged.**

If a downstream proof no longer resolves, add a compatibility rewrite using `tailState_idleTail`/`tailBelief_idleTail`; do not change expected activation states, offsets, means, or limits.

- [ ] **Step 8: Add a compatibility axiom report.**

```lean
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.tailState_idleTail
```

Keep all existing `replay_baseline` and `raw_tail_bridge` reports.

- [ ] **Step 9: Run Path4 regression.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
```

Expected: all existing exact activation and convergence values pass unchanged.

- [ ] **Step 10: Commit.**

```bash
git add NarrativeDynamics/Tests/FitnessABMPath4.lean
git commit -m "refactor(lean): route Path4 replay tails through idle interface"
```

**Review gate:** `History`, `.bbii`, `.bibi`, and `.iibb` remain test-only and do not appear in the new production modules.

---

### Task 5: Add bounded trust/CI coverage to the existing Path4 gate

**Files:**
- Modify: `tools/check_fitness_abm_path4.sh`

**Interfaces:**
- Preserve the existing Path4/Path5 builds, generated replay comparison, Path4 test log, and existing required theorem list.
- Add a separate `idle_tail_log` for the new replay theorem reports.

- [ ] **Step 1: Extend the source audit with these exact files.**

```text
NarrativeDynamics/Core/IdleTail.lean
NarrativeDynamics/Core/FitnessABMIdleTail.lean
NarrativeDynamics/Core/FitnessABMReplay.lean
NarrativeDynamics/Tests/IdleTail.lean
NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Do not remove any current Path4/Path5 audit input.

- [ ] **Step 2: Extend the temporary-file setup.**

Add:

```bash
idle_tail_log="$(mktemp)"
```

and include it in the existing `trap` cleanup alongside `path4_log` and `path5_vectors`.

- [ ] **Step 3: Add exact bounded module builds before the existing Path4/Path5 test executions.**

```bash
"$path4_time" -f 'NarrativeDynamics.Core.IdleTail elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.IdleTail

"$path4_time" -f 'NarrativeDynamics.Core.FitnessABMIdleTail elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMIdleTail

"$path4_time" -f 'NarrativeDynamics.Core.FitnessABMReplay elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMReplay
```

- [ ] **Step 4: Add the two new bounded consumers.**

```bash
"$path4_time" -f 'IdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/IdleTail.lean

"$path4_time" -f 'FitnessABMIdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  2>&1 | tee "$idle_tail_log"
```

- [ ] **Step 5: Add mandatory trust-log checks for the new production bridge.**

```bash
python3 tools/audit_fitness_trust.py log "$idle_tail_log" \
  --require NarrativeDynamics.FitnessABM.runInputs_append_idle \
  --require NarrativeDynamics.FitnessABM.replay_append_idle
```

Add `NarrativeDynamics.Tests.FitnessABMPath4.tailState_idleTail` to the existing `path4_required` list; do not remove any current requirement.

- [ ] **Step 6: Validate shell syntax and run the entire Path4 gate.**

```bash
bash -n tools/check_fitness_abm_path4.sh
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
```

Expected:
- source audit GREEN;
- each new direct Lean command stays within 240s;
- Path5 generated replay still byte-compares with `conformance/bb_path5_runtime_v1.json`;
- existing Path4 theorem/replay reports remain present;
- new theorem reports pass the standard axiom allowlist.

- [ ] **Step 7: Run the merged PathN gate unchanged.**

```bash
timeout --kill-after=10s 900s bash tools/check_fitness_abm_pathn.sh
```

Expected: GREEN with no PathN/FiniteConsensus source changes.

- [ ] **Step 8: Commit.**

```bash
git add tools/check_fitness_abm_path4.sh
git commit -m "ci: audit replay idle-tail interfaces"
```

---

### Task 6: Final exact-head verification and #81 review readiness

**Files:** no planned production edits.

- [ ] **Step 1: Run fresh focused verification on the exact candidate head.**

```bash
git rev-parse HEAD
bash -n tools/check_fitness_abm_path4.sh
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
timeout --kill-after=10s 900s bash tools/check_fitness_abm_pathn.sh
```

- [ ] **Step 2: Run source-boundary searches.**

```bash
git grep -n -E 'PropagationModel|TailModel' -- \
  NarrativeDynamics/Core NarrativeDynamics/Tests || true

git grep -n -E 'History\.(bbii|bibi|iibb)|\.bbii|\.bibi|\.iibb' -- \
  NarrativeDynamics/Core/IdleTail.lean \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean \
  NarrativeDynamics/Core/FitnessABMReplay.lean || true

git grep -n -E 'sorry|admit|native_decide|set_option[[:space:]]+maxHeartbeats[[:space:]]+0|axiom ' -- \
  NarrativeDynamics/Core/IdleTail.lean \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean \
  NarrativeDynamics/Core/FitnessABMReplay.lean \
  NarrativeDynamics/Tests/IdleTail.lean \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
```

Expected: no historical abstraction, no fixture leakage, no proof escape.

- [ ] **Step 3: Verify the diff is within the approved file/scope boundary.**

```bash
git diff --stat proof/narrative-dynamics-v0...HEAD
git diff --name-only proof/narrative-dynamics-v0...HEAD
```

Reject the candidate if `NarrativeDynamics/Core/FitnessABMPathN.lean` or `NarrativeDynamics/Core/FiniteConsensus.lean` changed.

- [ ] **Step 4: Request code review using this exact scope.**

```text
Base: 798f868a5327d6dd48de4f7a2b54cdf7a1ffed2a
Head: current feature head
Spec: docs/superpowers/specs/2026-09-15-bb-replay-idle-tail-design.md
Review focus:
- whole-RunState bookkeeping equality;
- accumulated replay probability preservation;
- no duplicate replay recursion;
- unchanged Path4 activation/common-clock semantics;
- bounded source/log trust gates.
```

Fix every Critical/Important finding and rerun Steps 1–3 after any code change.

- [ ] **Step 5: Open a draft PR against `proof/narrative-dynamics-v0`.**

The PR body must state:
- `FitnessABMPathN` remains canonical propagation/convergence;
- this PR adds replay→idle-tail reuse only;
- `runInputs_append_idle`/`replay_append_idle` preserve full `Result` bookkeeping;
- old #81 SHAs `72bfdda`/`fd2517c` are historical comments and do not resolve in the current repository;
- exact-head CI is required before ready-for-review.

- [ ] **Step 6: Require exact-head PR CI before ready-for-review.**

Verify on the PR head SHA:
- proof workflow = `completed/success`;
- World Studio = `completed/success`;
- `BB path-four convergence` step = success;
- `BB finite-path convergence` step = success;
- Python tests = success;
- no skipped/missing Lean proof job is being counted as proof evidence.

- [ ] **Step 7: Update #81 with factual evidence.**

Record:
- exact feature head SHA;
- PR number;
- exact proof and World Studio run IDs;
- Path4 gate elapsed/RSS lines for the new modules/tests;
- trust axiom results;
- unchanged `replay_baseline`/`raw_tail_bridge` and activation/convergence results;
- remaining non-goals: no `(α,τ)` theorem, no exposure-dependent learning, no universal replay equivalence.

- [ ] **Step 8: Mark the PR ready for review only after Step 6 is GREEN, then stop.**

Do not merge and do not close #81 until the user explicitly requests those actions.

---

## Plan Self-Review Checklist

- [ ] Generic iteration/observation is isolated in Task 1.
- [ ] Exact `RunState` bookkeeping is locked by `runIdleTrajectory_eq` in Task 2.
- [ ] `runInputs_replicate_idle` explicitly extracts the existing Path4 private `run_idle` fact in Task 3.
- [ ] `runInputs_append_idle` and `replay_append_idle` preserve `out.probability` in their exact signatures.
- [ ] Concrete non-unit probability preservation (`1/8`) is exercised by the refactored Path4 `raw_tail_bridge` in Task 4.
- [ ] BBII/BIBI/IIBB remain test-only.
- [ ] Existing Path4/Path5 generated replay and PathN gates remain explicit regressions.
- [ ] No task changes `FitnessABMPathN.lean` or `FiniteConsensus.lean`.
- [ ] No task creates another replay runner or propagation abstraction.
- [ ] Every code/test action has an exact file, interface, command, and expected result; no placeholder or alternate implementation branch remains.
