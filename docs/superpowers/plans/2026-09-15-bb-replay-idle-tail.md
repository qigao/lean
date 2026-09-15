# BB replay-to-idle-tail implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract a reusable checked-replay → idle-tail → observation boundary from the existing BB Path4 fixtures without creating a second propagation abstraction or changing runtime semantics.

**Architecture:** Add a proof-neutral `IdleTailModel` for repeated single-step iteration plus observation, then a Fitness ABM adapter whose runtime transition is exactly `advance`. Prove append-idle theorems directly over the existing `runInputs`/`replay` recursion so node count, round index, underlying `JointState`, and accumulated replay probability are preserved exactly. Refactor the Path4 fixture to consume this boundary while leaving `FitnessABMPathN`/`FiniteConsensus` as the canonical propagation/convergence layers.

**Tech Stack:** Lean 4.32.0, mathlib v4.32.0, exact `Rat`, existing `FitnessABM`, `FitnessABMReplay`, Path4/Path5/PathN gates, `tools/audit_fitness_trust.py`, GitHub proof workflow.

**Spec:** `docs/superpowers/specs/2026-09-15-bb-replay-idle-tail-design.md`, approved on branch `design/bb-replay-idle-tail-v1` at `2f716319748502d2d3d9cede273665352814162c`.

## Global Constraints

- Base implementation work on `proof/narrative-dynamics-v0@798f868a5327d6dd48de4f7a2b54cdf7a1ffed2a` plus the approved spec/plan history.
- `FitnessABMPathN` remains the canonical reusable finite-path propagation surface; do not add `PropagationModel` or any second graph/population/step abstraction.
- `FiniteConsensus` remains the canonical generic convergence layer; do not add convergence mathematics to idle-tail modules.
- One idle runtime tick is the existing `none` branch of `runInputs`: node count unchanged, round index incremented by one, underlying state advanced with `FitnessABM.advance`, no birth mass introduced.
- Appending idle ticks to a successful prefix preserves the prefix's accumulated replay probability exactly; it must not reset probability to `1`.
- Do not duplicate `checkedBirth`, parsing, `runInputs`, replay recursion, probability multiplication, or network update logic.
- Keep BBII/BIBI/IIBB fixtures, activation offsets, and concrete history names in Path4 test code only.
- Preserve existing theorem names and expected values for `replay_baseline`, `raw_tail_bridge`, activation theorems, common-clock convergence, Path5 generated replay, and PathN convergence.
- No changes to `FitnessABMPathN.lean` or `FiniteConsensus.lean`.
- No `sorry`, `admit`, `native_decide`, new `axiom`, `unsafe`, `unlock_limits`, `set_option maxHeartbeats 0`, or unbounded proof-resource settings.
- Keep direct build/test invocations bounded by `timeout --kill-after=10s 240s` unless fresh evidence demonstrates a need to change the bound.
- Trust reports must remain within the existing standard axiom allowlist (`propext`, `Classical.choice`, `Quot.sound`) unless a reviewed exception is justified by fresh evidence.
- If a RED fails because of missing toolchain/dependencies or a malformed consumer rather than the intended missing declaration/theorem, repair the RED before writing production code.

## File Map

| File | Responsibility | Tasks |
| --- | --- | --- |
| `NarrativeDynamics/Core/IdleTail.lean` | Proof-neutral repeated-step + observation interface | 1 |
| `NarrativeDynamics/Tests/IdleTail.lean` | Generic RED/GREEN consumers | 1, 5 |
| `NarrativeDynamics/Core/FitnessABMIdleTail.lean` | `advance`-backed `JointState` and `RunState` adapters | 2 |
| `NarrativeDynamics/Tests/FitnessABMIdleTail.lean` | Runtime adapter and replay bridge consumers + axiom reports | 2–3, 5 |
| `NarrativeDynamics/Core/FitnessABMReplay.lean` | Production append-idle theorems over existing replay recursion | 3 |
| `NarrativeDynamics/Tests/FitnessABMPath4.lean` | Existing fixture; refactor tail wrappers to consume new boundary | 4–5 |
| `tools/check_fitness_abm_path4.sh` | Bounded source/build/test/trust regression gate | 5 |
| `docs/superpowers/specs/2026-09-15-bb-replay-idle-tail-design.md` | Approved architecture | reference only |
| `docs/superpowers/plans/2026-09-15-bb-replay-idle-tail.md` | This execution plan | reference only |

## Preparation Before Task 1

- [ ] Create `feature/bb-replay-idle-tail-v1` from the committed plan head, not from `master`.

```bash
git checkout design/bb-replay-idle-tail-v1
git pull --ff-only
git checkout -b feature/bb-replay-idle-tail-v1
git rev-parse HEAD
```

- [ ] Confirm the implementation base and starting diff.

```bash
git merge-base HEAD proof/narrative-dynamics-v0
git diff --stat proof/narrative-dynamics-v0...HEAD
```

Expected starting diff: only the approved design spec and this implementation plan.

- [ ] Re-read these files before editing:

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
- Consumes: only core Lean/function iteration.
- Produces:

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

@[simp] theorem trajectory_zero ...
theorem trajectory_succ ...
theorem trajectory_add ...
@[simp] theorem observedTrajectory_zero ...
theorem observedTrajectory_succ ...
```

Use the orientation:

```lean
trajectory m s (k + 1) = m.step (trajectory m s k)
trajectory m s (a + b) = trajectory m (trajectory m s a) b
```

- [ ] **Step 1: Write the failing generic consumer before the core module exists.**

Create `NarrativeDynamics/Tests/IdleTail.lean`:

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

- [ ] **Step 2: Run the consumer and verify genuine RED.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/IdleTail.lean
```

Expected RED: missing `NarrativeDynamics.Core.IdleTail` / `IdleTailModel` declarations only.

- [ ] **Step 3: Implement the minimal proof-neutral core.**

Create `NarrativeDynamics/Core/IdleTail.lean` with only the structure, the two definitions, and the five elementary iteration/observation laws above. Prefer `Function.iterate_succ_apply'` / `Function.iterate_add_apply`; do not introduce model-specific imports.

- [ ] **Step 4: Build and run the consumer.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.IdleTail
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/IdleTail.lean
```

Expected: both commands succeed.

- [ ] **Step 5: Review the Task 1 boundary.**

```bash
git grep -n -E 'Fitness|BB|Path|NetworkPropagation|advance|replay' -- \
  NarrativeDynamics/Core/IdleTail.lean
```

Expected: no model-specific surface in the generic module.

- [ ] **Step 6: Commit Task 1.**

```bash
git add NarrativeDynamics/Core/IdleTail.lean \
        NarrativeDynamics/Tests/IdleTail.lean
git commit -m "feat(lean): add generic idle-tail iteration interface"
```

---

### Task 2: Fitness ABM idle adapters with exact RunState bookkeeping

**Files:**
- Create: `NarrativeDynamics/Core/FitnessABMIdleTail.lean`
- Create: `NarrativeDynamics/Tests/FitnessABMIdleTail.lean`

**Interfaces:**
- Consumes: `IdleTailModel`, `FitnessABM.JointState`, `FitnessABM.RunState`, `FitnessABM.advance`.
- Produces:

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
```

Required theorems:

```lean
@[simp] theorem idleRunStep_nodeCount (s : RunState) :
  (idleRunStep s).nodeCount = s.nodeCount

@[simp] theorem idleRunStep_roundIndex (s : RunState) :
  (idleRunStep s).roundIndex = s.roundIndex + 1

@[simp] theorem idleRunStep_state (s : RunState) :
  (idleRunStep s).state = advance s.state

theorem runIdleTrajectory_nodeCount (s : RunState) (k : Nat) :
  (runIdleTrajectory s k).nodeCount = s.nodeCount

theorem runIdleTrajectory_roundIndex (s : RunState) (k : Nat) :
  (runIdleTrajectory s k).roundIndex = s.roundIndex + k
```

For the dependent underlying state projection, prefer the strongest statement Lean accepts without casts. First try:

```lean
theorem runIdleTrajectory_state (s : RunState) (k : Nat) :
  (runIdleTrajectory s k).state = (advance^[k]) s.state
```

If elaboration requires transporting across the separately proved node-count equality, use `Fin.cast`/`Eq.ndrec` only at the dependent record boundary; do not weaken the theorem to beliefs-only equality.

- [ ] **Step 1: Add RED consumers to `FitnessABMIdleTail.lean` test file.**

Use a concrete `JointState 2` fixture already constructible from the test surface, or define a tiny valid one locally. Require:

```lean
example (s : RunState) :
    (idleRunStep s).nodeCount = s.nodeCount :=
  idleRunStep_nodeCount s

example (s : RunState) :
    (idleRunStep s).roundIndex = s.roundIndex + 1 :=
  idleRunStep_roundIndex s

example (s : RunState) (k : Nat) :
    (runIdleTrajectory s k).roundIndex = s.roundIndex + k :=
  runIdleTrajectory_roundIndex s k

example {n : Nat} (s : JointState n) (k : Nat) :
    IdleTailModel.trajectory (jointIdleTail n) s k = (advance^[k]) s := by
  rfl
```

- [ ] **Step 2: Run RED.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Expected RED: missing `FitnessABMIdleTail` declarations.

- [ ] **Step 3: Implement adapters only.**

`FitnessABMIdleTail.lean` imports `IdleTail` and `FitnessABM`, not `FitnessABMReplay`. `idleRunStep` must be a literal packaging of the existing `advance`; do not duplicate the body of `advance`.

Prove `runIdleTrajectory_roundIndex` by induction on `k` via `IdleTailModel.trajectory_succ`; prove state/node-count projections the same way if they are not definitional.

- [ ] **Step 4: Build and test.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMIdleTail
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

- [ ] **Step 5: Review for semantic duplication.**

```bash
git grep -n 'propagate\|checkedBirth\|runInputs\|parseAgent\|orderedMass' -- \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean
```

Expected: none of these implementations appear in the adapter.

- [ ] **Step 6: Commit Task 2.**

```bash
git add NarrativeDynamics/Core/FitnessABMIdleTail.lean \
        NarrativeDynamics/Tests/FitnessABMIdleTail.lean
git commit -m "feat(lean): add Fitness ABM idle-tail adapters"
```

---

### Task 3: Extract append-idle replay theorems from the existing recursion

**Files:**
- Modify: `NarrativeDynamics/Core/FitnessABMReplay.lean`
- Modify: `NarrativeDynamics/Tests/FitnessABMIdleTail.lean`

**Interfaces:**
- Consumes: `runInputs`, `replay`, `runIdleTrajectory`.
- Produces:

```lean
theorem runInputs_replicate_idle
    (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩
```

This is the production extraction of the existing Path4 test-local `private theorem run_idle`.

Then:

```lean
theorem runInputs_append_idle
    (m tickIndex birthIndex : Nat)
    (s : RunState) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
    runInputs m tickIndex birthIndex s
        (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩
```

Finally:

```lean
theorem replay_append_idle
    (seed : FitnessAttachment.RawSeed) (m : Nat)
    (agents : Array RawAgent) (ticks : List RawTick)
    (out : Result) (k : Nat)
    (h : replay seed m agents ticks = .ok out) :
    replay seed m agents (ticks ++ List.replicate k none) =
      .ok ⟨runIdleTrajectory out.final k, out.probability⟩
```

The public theorem must preserve the whole `Result`, not just the projected beliefs.

- [ ] **Step 1: Add a RED consumer for the pure idle suffix.**

In `NarrativeDynamics/Tests/FitnessABMIdleTail.lean`, add:

```lean
example (m tickIndex birthIndex k : Nat) (s : RunState) :
    runInputs m tickIndex birthIndex s (List.replicate k none) =
      .ok ⟨runIdleTrajectory s k, 1⟩ :=
  runInputs_replicate_idle m tickIndex birthIndex k s
```

Run the test. Expected RED: unknown `runInputs_replicate_idle` while the adapter tests remain GREEN.

- [ ] **Step 2: Implement `runInputs_replicate_idle`.**

Import `FitnessABMIdleTail` into `FitnessABMReplay.lean`. Prove by induction on `k`, generalizing `s`, `tickIndex`, and `birthIndex`:

```lean
induction k generalizing s tickIndex with
| zero => rfl
| succ k ih =>
    simp only [List.replicate_succ, runInputs]
    rw [ih]
    -- finish by `runIdleTrajectory` successor law / record equality
```

Do not introduce a second recursive runner.

- [ ] **Step 3: Run GREEN for the pure suffix theorem.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMReplay
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

- [ ] **Step 4: Add a RED consumer for successful-prefix append semantics.**

Use a concrete valid replay fixture in the test file that has non-unit probability, so the consumer can catch accidental probability reset. The target assertion must reduce to a result of the form:

```lean
replay seed 1 agents ticks = .ok out
→ replay seed 1 agents (ticks ++ List.replicate 2 none) =
    .ok ⟨runIdleTrajectory out.final 2, out.probability⟩
```

Choose a fixture whose prefix contains at least one birth and whose known `out.probability ≠ 1`; reuse the simplest existing Path4 checked data rather than inventing another attachment implementation.

- [ ] **Step 5: Run RED.**

Expected RED: missing `runInputs_append_idle` / `replay_append_idle`, not parser/fixture failure.

- [ ] **Step 6: Implement `runInputs_append_idle`.**

Prove by induction on `ticks`, generalizing `s`, indices, and `out`.

Required proof shape:

```text
[]:
  h fixes out = ⟨s,1⟩; reduce to runInputs_replicate_idle.

none :: rest:
  unfold one existing none branch;
  apply induction hypothesis to the recursive successful result.

some raw :: rest:
  split only on existing checkedBirth/runInputs results;
  use h to eliminate error branches;
  apply IH to the successful tail;
  preserve the existing `next.2 * tail.probability` factor exactly.
```

No probability algebra may replace `out.probability` with `1`.

- [ ] **Step 7: Implement the thin `replay_append_idle` wrapper.**

Unfold `replay` only enough to expose its successful `runInputs` call. Split on existing `parseSeed`, `m` validity, and `parseAgents`; contradictory branches are eliminated by `h`. In the success branch, apply `runInputs_append_idle`.

- [ ] **Step 8: Add axiom reports in the test.**

```lean
#print axioms NarrativeDynamics.FitnessABM.runInputs_append_idle
#print axioms NarrativeDynamics.FitnessABM.replay_append_idle
```

- [ ] **Step 9: Build and test.**

```bash
timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMReplay
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Expected: consumers GREEN; axiom reports contain only the standard allowlist.

- [ ] **Step 10: Commit Task 3.**

```bash
git add NarrativeDynamics/Core/FitnessABMReplay.lean \
        NarrativeDynamics/Tests/FitnessABMIdleTail.lean
git commit -m "feat(lean): bridge checked replay to idle tails"
```

**Review gate:** search the diff for a new recursive replay implementation. Only proofs over the existing `runInputs`/`replay` definitions are allowed.

---

### Task 4: Refactor Path4 tail fixture into a compatibility consumer

**Files:**
- Modify: `NarrativeDynamics/Tests/FitnessABMPath4.lean`

**Interfaces:**
- Consumes: `jointIdleTail`, `observeJointWith`, `runIdleTrajectory`, `replay_append_idle`.
- Preserves public/test theorem names: `replay_baseline`, `raw_tail_bridge`, activation theorems, common-clock theorems.
- Produces compatibility theorems:

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

These should be `rfl`/small simp proofs if the adapter is aligned correctly.

- [ ] **Step 1: Add compatibility consumers before changing old definitions.**

Import `NarrativeDynamics.Core.FitnessABMIdleTail` into the Path4 test and add `tailState_idleTail` / `tailBelief_idleTail` theorems against the current definitions.

Run:

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
```

Expected: GREEN if the adapter is definitionally aligned. If not, fix only the adapter theorem interface; do not change Path4 expected values.

- [ ] **Step 2: Replace the test-local generic `private theorem run_idle`.**

Delete the private recursive theorem:

```lean
private theorem run_idle ...
```

because Task 3 now supplies `runInputs_replicate_idle` / `replay_append_idle` in production.

- [ ] **Step 3: Separate prefix replay from idle suffix proof.**

Introduce a fixture-local theorem for the exact four-tick prefix only:

```lean
private theorem replay_prefix (h : History) :
    FitnessABM.replay rawSeed 1 rawAgents (ticks h) =
      .ok ⟨⟨4, 4, baseState h⟩, 1/8⟩ := by
  -- retain the existing checked fixture proof for the finite prefix only
```

Move the existing fixed birth/idle computation from `raw_tail_bridge` into this prefix theorem, but stop once the four scheduled ticks are proved.

- [ ] **Step 4: Rebuild `replay_baseline` as a zero-tail compatibility theorem.**

```lean
theorem replay_baseline (h : History) :
    rawTail h 0 = .ok ⟨⟨4,4,baseState h⟩,1/8⟩ := by
  simpa [rawTail] using replay_prefix h
```

Keep the theorem name and exact result unchanged.

- [ ] **Step 5: Rebuild `raw_tail_bridge` from `replay_append_idle`.**

Target remains unchanged:

```lean
theorem raw_tail_bridge (h : History) (k : Nat) :
    rawTail h k = .ok ⟨⟨4,4+k,tailState h k⟩,1/8⟩ := by
  have hp := replay_append_idle
    rawSeed 1 rawAgents (ticks h)
    ⟨⟨4,4,baseState h⟩,1/8⟩ k (replay_prefix h)
  -- rewrite runIdleTrajectory projections with the adapter laws
  -- rewrite underlying state via `tailState_idleTail`
  simpa [rawTail, tailState, runIdleTrajectory,
    NarrativeDynamics.IdleTailModel.trajectory] using hp
```

Use whatever minimal simp set the actual definitions require; do not reintroduce recursive idle replay in the fixture.

- [ ] **Step 6: Keep all downstream activation/common-clock proofs untouched if possible.**

Run the full Path4 test and inspect the diff. If downstream theorem bodies change solely because of a renamed helper, prefer compatibility wrappers so the old proof text stays stable.

- [ ] **Step 7: Add one compatibility axiom report.**

```lean
#print axioms NarrativeDynamics.Tests.FitnessABMPath4.tailState_idleTail
```

Keep the existing `replay_baseline` and `raw_tail_bridge` reports.

- [ ] **Step 8: Run Path4 regression.**

```bash
timeout --kill-after=10s 240s lake env lean -DmaxErrors=1 \
  NarrativeDynamics/Tests/FitnessABMPath4.lean
```

Expected: all existing exact values and convergence tests pass unchanged.

- [ ] **Step 9: Commit Task 4.**

```bash
git add NarrativeDynamics/Tests/FitnessABMPath4.lean
git commit -m "refactor(lean): route Path4 replay tails through idle interface"
```

**Review gate:** `History.bbii`, `.bibi`, `.iibb` appear only in test/fixture code, never in `IdleTail.lean`, `FitnessABMIdleTail.lean`, or production replay bridge declarations.

---

### Task 5: Integrate bounded Path4 trust/CI coverage

**Files:**
- Modify: `tools/check_fitness_abm_path4.sh`
- Test: all new and existing Path4/Path5 modules/tests

**Interfaces:**
- Preserve the existing Path4/Path5 generated replay gate and existing required theorem audit list.
- Add a separate log/audit for the new interface so the old trust boundary remains readable.

- [ ] **Step 1: Extend source audit inputs.**

Add:

```text
NarrativeDynamics/Core/IdleTail.lean
NarrativeDynamics/Core/FitnessABMIdleTail.lean
NarrativeDynamics/Core/FitnessABMReplay.lean
NarrativeDynamics/Tests/IdleTail.lean
NarrativeDynamics/Tests/FitnessABMIdleTail.lean
```

Do not remove any existing Path4/Path5 files from the audit.

- [ ] **Step 2: Add bounded builds for the two new core modules.**

Extend the module loop or add explicit calls:

```bash
"$path4_time" -f 'NarrativeDynamics.Core.IdleTail elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.IdleTail

"$path4_time" -f 'NarrativeDynamics.Core.FitnessABMIdleTail elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s lake build NarrativeDynamics.Core.FitnessABMIdleTail
```

`FitnessABMReplay` is already part of the project, but explicitly build it here if needed to ensure the new theorem surface is checked before tests.

- [ ] **Step 3: Add bounded new consumers with a dedicated trust log.**

Create a temporary `idle_tail_log` in the script and run:

```bash
"$path4_time" -f 'IdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/IdleTail.lean

"$path4_time" -f 'FitnessABMIdleTail tests elapsed=%e s peak_rss=%M KiB' \
  timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  2>&1 | tee "$idle_tail_log"
```

- [ ] **Step 4: Audit mandatory new theorem reports.**

```bash
python3 tools/audit_fitness_trust.py log "$idle_tail_log" \
  --require NarrativeDynamics.FitnessABM.runInputs_append_idle \
  --require NarrativeDynamics.FitnessABM.replay_append_idle
```

The Path4 log still requires its existing reports and additionally:

```text
NarrativeDynamics.Tests.FitnessABMPath4.tailState_idleTail
```

Do not remove `replay_baseline` or `raw_tail_bridge` from the old required list.

- [ ] **Step 5: Run shell syntax check and the complete gate.**

```bash
bash -n tools/check_fitness_abm_path4.sh
timeout --kill-after=10s 900s bash tools/check_fitness_abm_path4.sh
```

Expected:
- source trust audit passes;
- new modules/tests pass within the per-command 240s bound;
- existing Path5 generated JSON still exactly matches `conformance/bb_path5_runtime_v1.json`;
- existing Path4 theorem/replay reports remain present;
- new append-idle reports use only the standard axiom allowlist.

- [ ] **Step 6: Run the already-merged PathN gate unchanged.**

```bash
timeout --kill-after=10s 900s bash tools/check_fitness_abm_pathn.sh
```

Expected: GREEN; no PathN source changes.

- [ ] **Step 7: Commit CI integration.**

```bash
git add tools/check_fitness_abm_path4.sh
git commit -m "ci: audit replay idle-tail interfaces"
```

---

### Task 6: Final exact-head verification, review, and #81 merge-readiness

**Files:** no production changes expected unless review finds a blocker.

**Interfaces:** final evidence only.

- [ ] **Step 1: Run focused final verification on the exact candidate head.**

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

Expected: no duplicate historical abstraction; no fixture leakage into the new core modules; no proof escapes.

- [ ] **Step 3: Review the final diff against the approved spec.**

```bash
git diff --stat proof/narrative-dynamics-v0...HEAD
git diff proof/narrative-dynamics-v0...HEAD -- \
  NarrativeDynamics/Core/IdleTail.lean \
  NarrativeDynamics/Core/FitnessABMIdleTail.lean \
  NarrativeDynamics/Core/FitnessABMReplay.lean \
  NarrativeDynamics/Tests/IdleTail.lean \
  NarrativeDynamics/Tests/FitnessABMIdleTail.lean \
  NarrativeDynamics/Tests/FitnessABMPath4.lean \
  tools/check_fitness_abm_path4.sh
```

Reject the candidate if it:
- adds another propagation abstraction;
- duplicates replay recursion;
- ignores `roundIndex` or probability;
- changes Path4/Path5 expected values;
- changes PathN/FiniteConsensus production files.

- [ ] **Step 4: Request code review.**

Review scope:

```text
Base: 798f868a5327d6dd48de4f7a2b54cdf7a1ffed2a
Head: current feature head
Requirements: docs/superpowers/specs/2026-09-15-bb-replay-idle-tail-design.md
Focus: dependent RunState equality, probability preservation, no replay duplication, Path4 compatibility, bounded trust gate.
```

Fix every Critical/Important finding before proceeding; rerun exact-head gates after any fix.

- [ ] **Step 5: Open a draft PR against `proof/narrative-dynamics-v0`.**

PR description must state:
- `FitnessABMPathN` remains canonical propagation/convergence;
- new surface is replay→idle-tail only;
- `runInputs_append_idle` / `replay_append_idle` preserve full result bookkeeping;
- old unresolvable #81 SHAs are historical comments, not evidence;
- exact head required before ready-for-review.

- [ ] **Step 6: Require exact-head PR CI.**

On the exact PR head, verify:
- proof workflow: success;
- World Studio: success;
- existing `BB path-four convergence`: success;
- existing `BB finite-path convergence`: success;
- no skipped/newly missing required proof job is being mistaken for success.

- [ ] **Step 7: Update #81 with factual evidence and remaining boundaries.**

The closing/ready comment must include:
- exact head SHA;
- PR number;
- exact proof/World Studio run IDs;
- bounded elapsed/RSS evidence from the Path4 gate;
- trust axiom results;
- preserved theorem names/values;
- explicit non-goals: no `(α,τ)` parameter theorem, no exposure-dependent learning, no universal replay equivalence.

- [ ] **Step 8: Stop at merge-readiness unless the user explicitly requests merge.**

Do not auto-merge or close #81 merely because CI passes.

---

## Plan Self-Review Checklist

Before execution begins, confirm:

- [ ] Every spec requirement maps to a task: generic interface (Task 1), exact runtime bookkeeping adapter (Task 2), append-idle replay/probability bridge (Task 3), Path4 compatibility (Task 4), bounded trust/CI (Task 5), exact-head review evidence (Task 6).
- [ ] No task introduces `PropagationModel`, a second graph/operator abstraction, or a second replay runner.
- [ ] `runInputs_replicate_idle` extracts the existing Path4 private `run_idle` theorem into production rather than creating new semantics.
- [ ] `runInputs_append_idle` and `replay_append_idle` preserve `out.probability` exactly.
- [ ] `RunState.roundIndex` is part of the authoritative bridge; `JointState` convenience models are projection-only.
- [ ] Path4 theorem names and expected values remain unchanged.
- [ ] No changes are planned for `FitnessABMPathN.lean` or `FiniteConsensus.lean`.
- [ ] Existing Path5 generated replay and PathN gates are explicit final regressions.
- [ ] No placeholder language (`TBD`, `TODO`, “similar to”, unspecified tests) remains in this plan.
