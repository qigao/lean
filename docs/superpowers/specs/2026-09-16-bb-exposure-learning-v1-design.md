# BB exposure-dependent learning v1 design

Issue: #84  
Parent roadmap: #80  
Base: `proof/narrative-dynamics-v0@b8766d3cc9767bbd8636f9adc0d8114f8efabb1e`

## Context

The merged baseline is intentionally exposure-independent at the belief level: `NetworkPropagation.broadcasting` depends on threshold and belief, while exposure counts are accumulated as bookkeeping. #83 supplies the canonical finite-path baseline and #81 supplies the replay/idle-tail boundary. #82 is defining a separate rational response-parameter layer.

#84 must not reinterpret those baseline semantics. It introduces a separately named model whose learning rate is explicitly a function of exposure count.

## Goal

Define a finite-path exposure-dependent variant with a declared rational schedule

```text
α : Nat → Rat
```

and an inclusive rational broadcast threshold `τ`. One step must:

1. determine broadcasters from the pre-step beliefs using `τ ≤ belief`;
2. count incoming broadcasts for each vertex;
3. update cumulative exposure from `e` to `e' = e + incoming`;
4. use the declared `α(e')` as the learning rate for that same step;
5. update belief toward the mean belief of the broadcasting neighbors;
6. leave belief unchanged when there are no broadcasting neighbors.

This ordering is part of the model contract: the current round's received broadcasts are recorded before selecting the learning rate.

## Non-goals

This model does not claim:

- that exposure counts have a psychological interpretation;
- that receptivity must increase or decrease with exposure;
- that the baseline model was incomplete or incorrect;
- arbitrary-graph convergence;
- stochastic exposure dynamics;
- equivalence with the production Python float runtime;
- replacement of `NetworkPropagation.propagate`.

## Architecture

### 1. Separate module and namespace

Add, after the design is approved and #82's parameter surface is stable:

- `NarrativeDynamics/Core/FitnessABMPathNExposure.lean`
- `NarrativeDynamics/Tests/FitnessABMPathNExposure.lean`

Namespace:

`NarrativeDynamics.FitnessABMPathNExposure`

No baseline definition in `NetworkPropagation`, `FitnessABMPathN`, or the #82 parameter module is mutated.

### 2. Parameters

```lean
structure ExposureParameters where
  receptivityAt : Nat → Rat
  threshold : Rat

def ExposureParameters.Valid (p : ExposureParameters) : Prop :=
  (∀ e, 0 ≤ p.receptivityAt e ∧ p.receptivityAt e ≤ 1) ∧
  0 ≤ p.threshold ∧ p.threshold ≤ 1
```

Monotonicity of `receptivityAt` is deliberately not part of `Valid`. Separate optional predicates may state increasing/decreasing schedules when a theorem actually needs them.

### 3. State

Reuse the canonical finite-path topology and represent each vertex by exact rational belief plus cumulative exposure:

```lean
structure AgentState where
  belief : Rat
  exposure : Nat

abbrev State (n : Nat) := Fin n → AgentState
```

The model does not duplicate graph storage. The graph is always `FitnessABMPathN.pathAdj n`.

### 4. Broadcast set and incoming exposure

```lean
def broadcasting (p : ExposureParameters) (s : State n) (i : Fin n) : Bool :=
  decide (p.threshold ≤ (s i).belief)
```

For vertex `i`, collect canonical path neighbors that broadcast under the pre-step state. Let `incoming i` be their count and `mean i` their exact rational mean when nonempty.

Inclusive equality at `threshold` is required.

### 5. One-step semantics

For every vertex `i`:

```text
e' = s[i].exposure + incoming(i)

if incoming(i) = 0:
    belief' = s[i].belief
else:
    a = receptivityAt(e')
    belief' = (1-a) * s[i].belief + a * mean(i)

exposure' = e'
```

The executable definition must be direct exact-rational finite-path semantics. It is a new model, not a wrapper that silently changes baseline `NetworkPropagation.propagate`.

### 6. Constant-schedule bridge to the baseline operator

Define a constant schedule constructor:

```lean
def constant (alpha tau : Rat) : ExposureParameters :=
  ⟨fun _ => alpha, tau⟩
```

For a constant schedule, projected beliefs must agree with the corresponding exposure-independent parameterized finite-path operator from #82, while cumulative exposure may continue to differ as state bookkeeping.

The strongest required bridge is therefore belief projection, not whole-state equality.

For `alpha = tau = 1/2`, compose this with #82's exact specialization theorem to recover the current fixed `FitnessABMPathN.beliefStep`.

### 7. Positive invariant theorem

Under:

- `ExposureParameters.Valid p`;
- all current beliefs lie in `[0,1]`;

prove one step preserves every belief in `[0,1]`.

This follows because each update is either unchanged or a convex combination of values in `[0,1]` with `α(e') ∈ [0,1]`.

Also prove exposure monotonicity:

```text
(s i).exposure ≤ (step p s i).exposure
```

No monotonicity of beliefs is claimed without stronger assumptions.

### 8. Required non-equivalence counterexample

Construct a small fixed-path fixture with:

- identical graph;
- identical beliefs;
- identical threshold;
- different starting exposure at one receiving vertex;
- a nonconstant schedule such as `α(0)=1/4` and `α(e)=3/4` for positive `e`;
- at least one broadcasting neighbor whose belief differs from the receiver.

Prove the two next-step projected beliefs differ at that vertex.

This explicitly shows why the baseline theorem `FitnessABMPathN.propagate_independent_exposures` does not extend to this new model. The baseline theorem itself remains true and unchanged.

### 9. Optional schedule-shape theorems

Only if small and directly justified, define predicates such as:

```lean
def Nondecreasing (p : ExposureParameters) : Prop :=
  Monotone p.receptivityAt
```

A theorem may then compare effective learning rates after larger exposure. Do not infer monotonicity of final beliefs unless the broadcaster mean is on a known side of the receiver belief.

## TDD sequence

### RED 1 — model surface

Create a consumer first requiring `ExposureParameters`, `Valid`, `AgentState`, `broadcasting`, and `step`.

Expected RED: missing module/declarations.

### GREEN 1 — exact one-step model

Implement only the finite-path state, broadcast collection, exposure accumulation, exact mean, and step ordering above.

### RED/GREEN 2 — constant schedule bridge

Add exact consumers showing constant `α` agrees in belief projection with #82's parameterized baseline and, at `1/2`, with the current fixed model.

### RED/GREEN 3 — invariants

Prove bounded beliefs and nondecreasing cumulative exposures under explicit assumptions.

### RED/GREEN 4 — counterexample

Add and prove the explicit same-belief/different-exposure fixture whose next belief differs. This is mandatory acceptance evidence, not an informal example.

### Final gate

Run the new bounded source/build/test/trust checks plus the unchanged PathN and Path4/Path5 regression gates.

## File and dependency boundary

#84 production implementation is gated on #82's response-parameter surface being stable enough to state the constant-schedule bridge. The design itself may be reviewed independently.

Expected changes when implemented:

- add `NarrativeDynamics/Core/FitnessABMPathNExposure.lean`;
- add `NarrativeDynamics/Tests/FitnessABMPathNExposure.lean`;
- minimally extend the existing bounded proof gate;
- no edits to baseline runtime semantics.

## Trust and resource boundary

- exact `Rat` arithmetic only for theorem-bearing state;
- no `sorry`, `admit`, new user `axiom`, `unsafe`, or `native_decide` proof oracle;
- no unbounded heartbeat/resource settings;
- direct focused Lean commands retain the repository's 240-second timeout convention;
- theorem reports remain within the existing trusted allowlist unless fresh evidence establishes a narrower dependency set.

## Acceptance

The #84 implementation is merge-ready only when:

- the exposure-dependent model has its own namespace and module;
- baseline `NetworkPropagation` and `FitnessABMPathN` semantics remain unchanged;
- current-round incoming broadcasts are accumulated before `α(e')` is selected;
- the constant-schedule belief projection bridges to #82/baseline semantics;
- at least one positive invariant theorem is proved;
- exposure monotonicity is proved;
- an explicit theorem-level counterexample shows exposure independence fails for a nonconstant schedule;
- baseline Path4/Path5/PathN gates remain green;
- exact-head CI and independent review have no blocker.
