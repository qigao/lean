# BB replay-to-idle-tail interface design

Issue: #81  
Parent roadmap: #80  
Base: `proof/narrative-dynamics-v0@798f868a5327d6dd48de4f7a2b54cdf7a1ffed2a`

## Status correction

Issue #81 contains an older comment that refers to commits `72bfdda` and `fd2517c` and describes `PropagationModel n` / `TailModel State`. Those commits do not resolve in the current repository, and the merged base does not contain those abstractions. Issue #82 similarly refers to an unresolvable `1e12766` parameterization commit. These comments are historical intent only and are not current implementation evidence.

The current merged architecture is different and stronger on the propagation side:

- `NarrativeDynamics.Core.FitnessABMPathN` is the canonical reusable finite-path propagation surface.
- `NarrativeDynamics.Core.FiniteConsensus` is the proof-only generic averaging/convergence layer.
- Existing Path4 and Path5 modules remain regression authorities.
- The remaining reuse problem in #81 is the checked replay → idle tail → observation boundary that is still embedded in the Path4 fixture tests.

This design therefore does **not** recreate `PropagationModel` or a second generic propagation abstraction.

## Goal

Extract a small reusable interface for the phase after a checked replay has reached a concrete `JointState n` and subsequent global ticks are idle (`none`). The interface must let proof consumers:

1. iterate the real runtime idle step without copying replay logic;
2. observe a projection of each state, especially beliefs;
3. connect `ticks ++ replicate k none` replay results to an iterative tail trajectory;
4. keep activation histories and concrete BBII/BIBI/IIBB fixtures outside the generic layer;
5. preserve the existing Path4 exact theorem suite and Path5/PathN evidence unchanged.

The interface is intended to become the stable boundary consumed by later #82 parameter studies and #84 exposure-dependent alternatives.

## Non-goals

This change will not:

- replace or wrap `FitnessABMPathN` with another propagation model;
- change `NetworkPropagation.propagate`, `FitnessABM.advance`, checked birth semantics, or replay parsing;
- change Path4 graph topology, profiles, exact limits, activation fixtures, or theorem statements;
- prove new convergence theorems;
- parameterize receptivity/threshold (`α`, `τ`);
- introduce exposure-dependent learning;
- generalize checked replay across arbitrary event schemas;
- claim that all histories have a common activation time or the same limit.

## Existing coupling to remove

`NarrativeDynamics/Tests/FitnessABMPath4.lean` currently mixes four different concerns:

1. raw replay fixture construction (`rawSeed`, `rawAgents`, `rawBirth`, `ticks`, `rawTail`);
2. proof-bearing checked replay adapters and birth witnesses;
3. concrete post-replay states (`baseState`, `tailState`);
4. belief observation and history-specific activation/common-clock convergence arguments (`tailBelief`, BBII/BIBI/IIBB).

The generic part hidden inside that fixture is simply:

```text
state s
  |
  | real idle runtime step
  v
advance s
  |
  | repeated k times
  v
advance^[k] s
  |
  | observation
  v
belief / other projection
```

The design extracts only this boundary.

## Architecture

### 1. Generic iteration/observation layer

Add a proof-neutral module, tentatively:

`NarrativeDynamics/Core/IdleTail.lean`

It should know nothing about BB, graphs, births, or beliefs.

Proposed structure:

```lean
structure IdleTailModel (State Obs : Type*) where
  step : State → State
  observe : State → Obs

namespace IdleTailModel

def trajectory (m : IdleTailModel State Obs) (s : State) (k : Nat) : State :=
  (m.step)^[k] s


def observedTrajectory
    (m : IdleTailModel State Obs) (s : State) (k : Nat) : Obs :=
  m.observe (m.trajectory s k)
```

Only elementary iteration lemmas belong here: zero, successor, addition/composition, and observation after zero/successor. No convergence or model-specific theorem belongs in this module.

The name `IdleTailModel` is intentionally narrower than the historical `TailModel`: the semantics are explicitly “repeat one idle transition”.

### 2. Fitness ABM idle adapter

Add a small adapter module, tentatively:

`NarrativeDynamics/Core/FitnessABMIdleTail.lean`

For each fixed `n`, define the canonical real-runtime model:

```lean
def jointIdleTail (n : Nat) :
    IdleTailModel (JointState n) (JointState n) :=
  { step := advance
    observe := id }
```

and a belief projection adapter when a belief projection is supplied by a concrete model:

```lean
def observeWith
    (project : JointState n → Obs) :
    IdleTailModel (JointState n) Obs :=
  { step := advance
    observe := project }
```

This adapter must call the existing `advance`; it must not reproduce the propagation/update body.

### 3. Replay-to-tail bridge

The important proof is not a new replay implementation. It is a bridge showing that appending idle ticks to an already checked replay is equivalent to applying the existing idle runtime step repeatedly.

Prefer a theorem at the existing typed/runtime level with the narrowest stable statement available from `FitnessABMReplay`, conceptually:

```lean
replay_append_idle
  (h : replay seed m agents ticks = .ok out) :
  replay seed m agents (ticks ++ List.replicate k none) =
    .ok (... iterate advance k from out.final ...)
```

The exact theorem type should follow the current `Result` / `runInputs` representation rather than introducing a parallel result type. If the public `replay` return object makes direct equality awkward, split the bridge into:

- a theorem about `runInputs` / the checked internal state transition;
- a thin theorem projecting the public replay `final` state.

The proof must reuse the existing recursion and checked-replay semantics. No duplicated `checkedBirth`, parsing, or probability proof is allowed.

### 4. Path4 fixture adapter

Keep `History`, `rawSeed`, births, exact checked fixtures, and activation witnesses in the Path4 test namespace.

Replace the conceptual role of:

```lean
def tailState (h) (k) := advance^[k] (baseState h)
def tailBelief (h) (k) := Path4.project (tailState h k).population
```

with the canonical idle-tail adapter. Existing names may remain as compatibility wrappers so downstream theorems do not churn.

The adapter should prove definitionally or by one small theorem that the old `tailState` / `tailBelief` surface agrees with the extracted model.

### 5. Relationship to PathN

`FitnessABMPathN` remains the canonical propagation/convergence layer. This change does not import `FitnessABMPathN` into the generic idle-tail module.

Path4 can connect the two layers in tests:

```text
checked replay
   ↓
JointState 4 at activation/base point
   ↓  idle-tail bridge
advance^[k]
   ↓  belief projection
Path4/PathN belief trajectory
   ↓
existing convergence theorem
```

The bridge from `advance`-based idle tails to `FitnessABMPath4.beliefStep` should remain a Path4-specific theorem because it depends on the concrete network/profile state and activation region.

## Invariants and proof obligations

The implementation should establish the following in order.

### Generic idle-tail laws

- `trajectory_zero`
- `trajectory_succ`
- `trajectory_add`
- `observedTrajectory_zero`
- `observedTrajectory_succ`

These are theorem-level API, not just simp accidents, so later models can consume them explicitly.

### Fitness runtime adapter laws

- adapter step is exactly `FitnessABM.advance`;
- generic `trajectory` is exactly `advance^[k]`;
- observation is pure projection and cannot mutate runtime state.

### Replay bridge

For successful checked replay prefixes, appending `k` idle ticks yields the same final state as `k` iterations of the existing idle transition.

Probability handling must remain explicit. If idle ticks preserve the replay probability, prove it from the existing replay/runInputs semantics; do not merely project the state and silently ignore probability.

### Path4 compatibility

- current `tailState h k` equals the extracted idle-tail state trajectory;
- current `tailBelief h k` equals the extracted observed trajectory;
- all existing activation/common-clock/convergence tests continue to pass without changing their expected values.

## TDD sequence

### RED 1 — generic interface

Create `NarrativeDynamics/Tests/IdleTail.lean` first with consumers for zero/succ/add laws before `IdleTail.lean` exists.

Expected RED: missing module/declarations only.

### GREEN 1

Implement the proof-neutral interface and elementary iteration laws. Run a bounded focused consumer.

### RED 2 — real Fitness adapter

Add consumers requiring an adapter whose `step` is definitionally/equationally `advance` and whose observation projection is preserved.

Expected RED: missing Fitness idle-tail adapter declarations.

### GREEN 2

Implement only the adapter; no replay theorem yet.

### RED 3 — replay append-idle bridge

Add a small deterministic checked fixture consuming the desired append-idle theorem.

The RED is valid only if the fixture itself parses/builds and fails because the bridge theorem is missing or the target equality is unproved.

### GREEN 3

Prove the bridge by reusing current replay/runInputs recursion. Do not copy checked replay logic.

### RED/GREEN 4 — Path4 compatibility

Refactor Path4 test-only `tailState` / `tailBelief` through the new adapter and prove compatibility before changing downstream uses. Then run the complete existing Path4 gate.

## CI and trust boundary

Add the new generic/core/test files to the existing Path4 bounded gate rather than creating a permanent second workflow.

Every direct Lean invocation remains bounded by the existing `240s` convention. Do not use:

- `set_option maxHeartbeats 0`;
- unbounded recursion/heartbeat settings to mask proof structure;
- `sorry` / `admit`;
- native proof oracles;
- duplicated replay code to make proofs easier.

Mandatory trust reports should cover the replay append-idle bridge and at least one Path4 compatibility theorem. The allowlist remains the existing standard Lean/Mathlib axioms unless fresh evidence justifies otherwise.

## File boundary

Expected production additions:

- `NarrativeDynamics/Core/IdleTail.lean`
- `NarrativeDynamics/Core/FitnessABMIdleTail.lean`

Expected tests:

- `NarrativeDynamics/Tests/IdleTail.lean`
- either `NarrativeDynamics/Tests/FitnessABMIdleTail.lean` or focused additions to `FitnessABMPath4.lean` for the concrete replay bridge

Expected CI change:

- `tools/check_fitness_abm_path4.sh` only, adding source/build/test/audit coverage while preserving its existing Path4/Path5 replay and theorem reports.

No changes are expected in `FitnessABMPathN.lean` or `FiniteConsensus.lean`.

## Failure modes to guard against

1. **Second propagation abstraction.** If the design starts packaging graph/population/step again, stop: that duplicates `FitnessABMPathN`.
2. **Replay reimplementation.** If the proof creates a second recursive replay function, stop and bridge the existing one instead.
3. **State-only equality hiding probability drift.** The append-idle theorem must account for the public replay result/probability boundary explicitly.
4. **Fixture leakage.** `History.bbii/bibi/iibb` must not appear in generic production modules.
5. **Semantic widening.** “Idle tail” means repeated existing idle runtime transitions, not arbitrary future events.
6. **Unnecessary PathN dependency.** Generic idle-tail code must stay independent of convergence mathematics.

## Acceptance criteria

#81 is complete only when all are true:

- reusable idle-tail iteration/observation interface exists;
- it is proof-neutral and contains no BB/path fixture semantics;
- Fitness adapter uses the real existing `advance` transition;
- checked replay + appended idle ticks is proved equivalent to iterative idle-tail execution on successful prefixes;
- probability/result semantics are preserved explicitly;
- existing Path4 `tailState` / `tailBelief` are compatibility consumers of the new interface;
- existing BBII/BIBI/IIBB activation and common-clock convergence facts remain unchanged;
- existing Path4/Path5 generated replay and PathN gates remain green;
- source trust audit and mandatory theorem reports pass under bounded resources;
- no `PropagationModel` replacement or duplicate propagation abstraction is added;
- exact-head proof and World Studio CI pass before merge readiness;
- issue #81 is updated with factual exact-head evidence and the historical unresolvable SHA caveat.

## Follow-on boundary

After #81 lands:

- #82 may introduce rational `(α, τ)` parameters against the real propagation/idle-tail boundaries and prove only the parameter domains actually justified.
- #84 may define a separate exposure-dependent model while reusing the same idle-tail observation interface, without changing baseline `NetworkPropagation`.
- #85 may use the replay boundary to define deterministic rational conformance cases without claiming universal Python↔Lean equivalence.
