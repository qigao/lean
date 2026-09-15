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

Extract a small reusable interface for the phase after a checked replay has reached a concrete runtime state and subsequent global ticks are idle (`none`). The interface must let proof consumers:

1. iterate the real runtime idle step without copying replay logic;
2. observe a projection of each runtime state, especially the underlying `JointState` and beliefs;
3. connect `ticks ++ List.replicate k none` replay results to an iterative tail trajectory;
4. preserve runtime bookkeeping exactly: node count, round index, and replay probability;
5. keep activation histories and concrete BBII/BIBI/IIBB fixtures outside the generic layer;
6. preserve the existing Path4 exact theorem suite and Path5/PathN evidence unchanged.

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

The generic state evolution hidden inside that fixture is:

```text
RunState { nodeCount, roundIndex, state }
        |
        | one `none` tick
        v
RunState { nodeCount,
           roundIndex + 1,
           advance state }
        |
        | repeated k times
        v
runtime idle tail
        |
        | observation
        v
JointState / belief / other projection
```

For a pure idle suffix, no birth mass is introduced. Therefore appending idle ticks must leave the already accumulated replay probability unchanged.

The design extracts only this boundary.

## Architecture

### 1. Generic iteration/observation layer

Add a proof-neutral module, tentatively:

`NarrativeDynamics/Core/IdleTail.lean`

It should know nothing about BB, graphs, births, probabilities, or beliefs.

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

### 2. Fitness ABM runtime idle adapter

Add a small adapter module, tentatively:

`NarrativeDynamics/Core/FitnessABMIdleTail.lean`

The primary replay-facing transition must operate on `FitnessABM.RunState`, because one idle tick changes both the underlying `JointState` and `roundIndex`:

```lean
def idleRunStep (s : RunState) : RunState :=
  ⟨s.nodeCount, s.roundIndex + 1, advance s.state⟩


def runIdleTail : IdleTailModel RunState RunState :=
  { step := idleRunStep
    observe := id }
```

Useful observations are then projections, not alternate state machines:

```lean
def observeRunWith (project : RunState → Obs) :
    IdleTailModel RunState Obs :=
  { step := idleRunStep
    observe := project }
```

For Path4 belief consumers, the observation can project through `s.state.population` and the existing Path4 belief projection.

A secondary `JointState n` convenience model may exist only if it eliminates test duplication:

```lean
def jointIdleTail (n : Nat) : IdleTailModel (JointState n) (JointState n) :=
  { step := advance
    observe := id }
```

It is not the replay bridge authority because it does not carry `roundIndex`.

All adapters must call the existing `advance`; none may reproduce the propagation/update body.

### 3. Replay-to-tail bridge

The important proof is not a new replay implementation. It is an append theorem for the existing `runInputs` recursion.

The preferred core statement is conceptually:

```lean
runInputs_append_idle
  (h : runInputs m tickIndex birthIndex s ticks = .ok out) :
  runInputs m tickIndex birthIndex s
      (ticks ++ List.replicate k none) =
    .ok
      { final := (runIdleTail.trajectory out.final k)
        probability := out.probability }
```

This statement preserves all public runtime semantics:

- `final.nodeCount` is unchanged across the idle suffix;
- `final.roundIndex = out.final.roundIndex + k`;
- `final.state = advance^[k] out.final.state`;
- `probability = out.probability`.

The theorem should be proved from the existing structural recursion of `runInputs`, using the fact that the `none` branch performs `idleRunStep` and contributes no birth mass.

Then add a thin public wrapper:

```lean
replay_append_idle
  (h : replay seed m agents ticks = .ok out) :
  replay seed m agents (ticks ++ List.replicate k none) =
    .ok
      { final := runIdleTail.trajectory out.final k
        probability := out.probability }
```

The exact Lean syntax should follow the current `Result` and dependent `RunState` representation; no parallel result type is allowed.

If dependent equality makes the public theorem awkward, the implementation may prove field-level equalities (`nodeCount`, `roundIndex`, underlying `JointState`, probability) instead of forcing a brittle whole-record equality. The field-level theorem set must still establish the four bullets above.

No duplicated `checkedBirth`, parsing, probability, or replay recursion is allowed.

### 4. Path4 fixture adapter

Keep `History`, `rawSeed`, births, exact checked fixtures, and activation witnesses in the Path4 test namespace.

Replace the conceptual role of:

```lean
def tailState (h) (k) := advance^[k] (baseState h)
def tailBelief (h) (k) := Path4.project (tailState h k).population
```

with observation of the canonical idle-tail transition. Existing names should remain as compatibility wrappers so downstream theorems do not churn.

Because the old `tailState` intentionally forgets runtime round indices, Path4 compatibility should compare it with the **underlying `JointState` projection** of the runtime idle trajectory, not redefine replay bookkeeping away.

The adapter should prove definitionally or by small bridge theorems that the old `tailState` / `tailBelief` surface agrees with those projections.

### 5. Relationship to PathN

`FitnessABMPathN` remains the canonical propagation/convergence layer. This change does not import `FitnessABMPathN` into the generic idle-tail module.

Path4 can connect the two layers in tests:

```text
checked replay
   ↓
RunState at activation/base point
   ↓  replay/idle-tail bridge
idleRunStep^[k]
   ↓  underlying JointState
advance^[k]
   ↓  belief projection
Path4/PathN belief trajectory
   ↓
existing convergence theorem
```

The bridge from `advance`-based idle tails to `FitnessABMPath4.beliefStep` remains Path4-specific because it depends on the concrete network/profile state and activation region.

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

For every `RunState s` and `k`:

- one adapter step has the same `nodeCount`;
- one adapter step has `roundIndex + 1`;
- one adapter step has underlying state `advance s.state`;
- `k` adapter steps have `roundIndex + k`;
- `k` adapter steps have underlying state `advance^[k] s.state`;
- observation is a pure projection of the resulting runtime state.

### Replay bridge

For successful checked replay prefixes, appending `k` idle ticks yields the same runtime final state as `k` iterations of `idleRunStep` and preserves the already accumulated replay probability exactly.

The proof must derive this from current `runInputs`/`replay`; it may not silently discard `roundIndex` or probability.

### Path4 compatibility

- current `tailState h k` equals the underlying `JointState` of the extracted idle-tail trajectory from the corresponding base runtime state;
- current `tailBelief h k` equals the extracted observed belief trajectory;
- all existing activation/common-clock/convergence tests continue to pass without changing their expected values.

## TDD sequence

### RED 1 — generic interface

Create `NarrativeDynamics/Tests/IdleTail.lean` first with consumers for zero/succ/add laws before `IdleTail.lean` exists.

Expected RED: missing module/declarations only.

### GREEN 1

Implement the proof-neutral interface and elementary iteration laws. Run a bounded focused consumer.

### RED 2 — real Fitness runtime adapter

Add consumers requiring `idleRunStep` and proving one-step node-count, round-index, and `advance` projections.

Expected RED: missing Fitness idle-tail adapter declarations.

### GREEN 2

Implement only the adapter and its iteration projection laws; no replay append theorem yet.

### RED 3 — `runInputs_append_idle`

Add a small deterministic checked fixture consuming the desired append-idle theorem. The consumer must assert final runtime bookkeeping **and probability**, not only beliefs.

The RED is valid only if the fixture itself parses/builds and fails because the bridge theorem is missing or the target equality is unproved.

### GREEN 3

Prove the append theorem by reusing current `runInputs` recursion. Add the public `replay_append_idle` wrapper only after the internal theorem is green. Do not copy checked replay logic.

### RED/GREEN 4 — Path4 compatibility

Refactor Path4 test-only `tailState` / `tailBelief` through projections of the new adapter and prove compatibility before changing downstream uses. Then run the complete existing Path4 gate.

## CI and trust boundary

Add the new generic/core/test files to the existing Path4 bounded gate rather than creating a permanent second workflow.

Every direct Lean invocation remains bounded by the existing `240s` convention. Do not use:

- `set_option maxHeartbeats 0`;
- unbounded recursion/heartbeat settings to mask proof structure;
- `sorry` / `admit`;
- native proof oracles;
- duplicated replay code to make proofs easier.

Mandatory trust reports should cover `runInputs_append_idle` / `replay_append_idle` and at least one Path4 compatibility theorem. The allowlist remains the existing standard Lean/Mathlib axioms unless fresh evidence justifies otherwise.

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
3. **Bookkeeping loss.** A theorem that proves only `JointState` equality while ignoring `roundIndex` or replay probability is insufficient.
4. **Probability reset.** A pure idle suffix has local mass `1`; appended to a successful prefix it must preserve the prefix's accumulated probability, not replace it with `1`.
5. **Fixture leakage.** `History.bbii/bibi/iibb` must not appear in generic production modules.
6. **Semantic widening.** “Idle tail” means repeated existing `none` runtime transitions, not arbitrary future events.
7. **Unnecessary PathN dependency.** Generic idle-tail code must stay independent of convergence mathematics.

## Acceptance criteria

#81 is complete only when all are true:

- reusable idle-tail iteration/observation interface exists;
- it is proof-neutral and contains no BB/path fixture semantics;
- Fitness runtime adapter uses the real existing `advance` transition and exact `RunState` bookkeeping;
- checked `runInputs` + appended idle ticks is proved equivalent to iterative `idleRunStep` execution on successful prefixes;
- the public replay-level bridge preserves node count, round index, underlying `JointState`, and accumulated probability explicitly;
- existing Path4 `tailState` / `tailBelief` are compatibility consumers of projections from the new interface;
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
