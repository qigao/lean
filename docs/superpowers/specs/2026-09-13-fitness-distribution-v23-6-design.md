# Fitness Distribution V23.6 — exact finite trace law over BB growth

Status: design for review; no production implementation, new theorem compilation, or CI success is claimed by this document.
User direction: after merging the finite BB replay and seven-hop counterexample, proceed to the next layer: promote exact per-trace replay probabilities into a first-class finite probability law over all legal target traces, then support exact event probabilities and expectations.

## 1. Decision and repository boundary

Add a new typed finite-distribution layer on top of the merged V23.5 BB model. The selected architecture is a **typed finite target-trace space** whose members enumerate every legal ordered target choice for a fixed initial state, fixed attachment count `m`, and fixed positive newborn-fitness schedule. Each trace receives the exact rational product of the existing conditional `orderedMass` terms, and each trace is evaluated by the existing `applyBirth` transition.

Start `feature/fitness-distribution-v23-6` from merge commit `b432cd21eb8bac04bae8f676b7321734b90ed5c5`, which merged PR #66 into `feature/fitness-validation-v23-5`. This is a successor layer to V23.5, not an alternative replay implementation. Do not rewrite `FitnessAttachment`, `FitnessBirth`, `FitnessValidation`, or `FitnessReplay` semantics unless a missing helper is required by the same contract.

The main production surface is a new module:

- `NarrativeDynamics/Core/FitnessDistribution.lean`

with tests in:

- `NarrativeDynamics/Tests/FitnessDistribution.lean`

`NarrativeDynamics.lean` may import the new core module only after the implementation exists. `README.md` should be updated only after verification. Existing `FitnessReplay.lean` remains the deterministic raw-input boundary; distribution code must reuse `Targets`, `orderedMass`, `applyBirth`, `State`, and the fixed-schedule semantics already proved there.

Alternatives considered:

1. **Typed finite target-trace distribution** — selected. It preserves exact `Rat`, reuses the proved BB kernel directly, has a finite `Fintype` carrier, and is suitable for exact event probabilities and expectations.
2. **Mathlib PMF/Measure first** — deferred. It would introduce `ENNReal`, countable-support/measure interfaces, and conversion obligations before they provide value for the current finite exact model.
3. **Only add recursive `eventMass` functions** — rejected as the primary design. It would answer individual questions but would not expose a reusable trace space, making expectation, conditioning, exhaustive fixtures, and future sampling bridges duplicate the same recursion.

No general-purpose probability library is introduced. The abstraction is intentionally scoped to this BB trace model.

## 2. Exact probability experiment

A V23.6 experiment fixes all of the following before target randomness is considered:

- an initial valid `State n`;
- one fixed attachment count `m` with `0 < m` and `m <= n`;
- a finite schedule `schedule : List PosFitness`, one positive newborn fitness for each future birth.

Only the ordered target selections are random in this layer. The fitness schedule is conditioned on and is not sampled. This distinction is essential: V23.6 is a finite probability law **conditional on the supplied positive fitness schedule**.

At birth `t`, the current state is the result of the preceding target choices. A target choice is one `Targets currentN m`, i.e. an injective ordered embedding of `Fin m` into the existing vertices `Fin currentN`. Its conditional mass is the already-proved `orderedMass currentState T`.

The next state is exactly:

    applyBirth currentState T hm eta

where `eta` is the schedule entry for that birth. Therefore later conditional probabilities are computed from the actual updated graph and updated degree function. V23.6 must not freeze the initial graph for the entire schedule and must not multiply independent initial rows.

## 3. Typed trace carrier

Define a dependent finite trace carrier whose type records the changing number of available old vertices. The intended shape is equivalent to:

```lean
def TargetTrace (n m : Nat) : List PosFitness → Type
  | [] => PUnit
  | _eta :: rest => Σ T : Targets n m, TargetTrace (n + 1) m rest
```

The concrete Lean syntax may be adjusted for elaboration, but the semantic contract is fixed:

- empty schedule has exactly one empty trace;
- a nonempty trace begins with one legal `Targets n m`;
- its tail is a trace from carrier size `n+1`;
- the schedule entry affects transition/probability, not the finite carrier shape;
- the carrier is finite and executable without choice-based enumeration.

Provide a `Fintype` instance recursively from the existing computable `Fintype (Targets n m)` and finite sigma/product instances. Avoid `Fintype.ofFinite` or other choice-based executable enumeration. `DecidableEq` may be supplied if useful for fixtures and event enumeration, but it is not a mathematical requirement of the law.

The carrier enumerates every legal ordered target sequence exactly once. Because `Targets n m` is injective, duplicate targets within one birth are absent by construction. Because the carrier at each layer is `Fin currentN`, selecting a not-yet-created future vertex is unrepresentable.

## 4. Exact trace probability

Define the exact mass of a typed trace recursively using the existing BB kernel:

```lean
traceProbability
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule) : Rat
```

Semantics:

```text
P(empty) = 1
P(T :: rest) = orderedMass(s,T)
                 * P(rest from applyBirth(s,T,eta))
```

The recursive bound proof for later states uses only `m <= n -> m <= n+1`; the positive `m` proof remains fixed. No new probability formula is permitted: the first factor must be definitionally or theorem-equivalent to V23.5 `orderedMass`, and the successor state must be the real `applyBirth` result.

Required generic theorems:

- `traceProbability_pos`: every typed trace has strictly positive probability;
- `traceProbability_nonneg`: immediate corollary for finite sums;
- `traceProbability_le_one`: every trace mass is at most one, derived from the normalized finite law rather than assumed;
- `traceProbability_sum_continuationMass`: the sum over all typed traces equals the existing `continuationMass s m hm hb schedule`;
- `traceProbability_sum_one`: total trace mass is exactly one, derived from the previous theorem and `continuationMass_one`.

The key design choice is to prove equality with `continuationMass`, not create an independent normalization proof tree. `continuationMass` is already the recursively normalized total conditional mass over all future `Targets`; V23.6 makes the same tree explicit as a finite carrier.

For a nonempty schedule, the proof should reduce the finite sum over `TargetTrace` to a sum over the sigma carrier:

```text
Σ trace
= Σ T : Targets n m, Σ tail : TargetTrace (n+1) m rest,
    orderedMass s T * traceProbability next tail
```

then use the induction hypothesis to recover the `continuationMass` recurrence. Prefer standard finite-sum/sigma equivalences over ad hoc list enumeration.

## 5. Trace evaluation and outcome identity

Define deterministic evaluation of a typed trace using the same transition:

```lean
traceFinal
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule) : RunState
```

or an equivalent typed result if the implementation can retain the exact final size conveniently. `RunState` is acceptable and avoids forcing arithmetic casts into every consumer. Required facts include:

- final node count is `n + schedule.length`;
- final state is valid;
- each step is exactly `applyBirth` with the trace's target and schedule fitness;
- actual edge count increases by `m * schedule.length`;
- stored fitness extends by exactly the schedule values in order.

Do not store a parallel graph, degree vector, or probability in the trace carrier. A trace is target choices only; all outcomes are derived from the authoritative state transition.

The final-state evaluator and trace probability must traverse the same trace structure. If implementation convenience introduces a combined evaluator, projections must recover the standalone final state and probability exactly.

## 6. First-class finite event probability

Define exact event mass over final outcomes. The event is a decidable predicate on the evaluated final `RunState` (or on the selected equivalent typed final-state representation):

```lean
eventProbability
  ...
  (event : RunState → Prop) [DecidablePred event] : Rat
```

with semantics:

```text
eventProbability E
= Σ trace, if E (traceFinal ... trace)
           then traceProbability ... trace
           else 0
```

Required first-layer laws:

- `eventProbability_true = 1`;
- `eventProbability_false = 0`;
- `eventProbability_nonneg`;
- `eventProbability_le_one`;
- complement law: `P(E) + P(not E) = 1` for the same finite experiment;
- monotonicity: if `E x -> F x` for every reachable final state (or every `RunState`, whichever is simpler and stronger), then `P(E) <= P(F)`.

Do not introduce events over arbitrary malformed raw replay requests. This distribution is over typed legal traces. Raw replay errors remain API validation behavior, not probability mass in the valid typed law.

An event may identify multiple traces with the same final graph. Their probabilities must be **summed**, not deduplicated by final state. This matters because ordered choices can yield the same graph with different probabilities, as already demonstrated by the triangle `(2,1)` versus `(1,2)` fixture.

## 7. Exact rational expectation

Define finite expectation for exact rational observables of the final result:

```lean
expectation
  ...
  (observable : RunState → Rat) : Rat
```

with semantics:

```text
E[f] = Σ trace, traceProbability(trace) * f(traceFinal(trace))
```

The first implementation slice should prove only the small algebraic basis required for later network statistics:

- expectation of a constant equals that constant;
- linearity for addition;
- scalar multiplication by a rational constant;
- expectation of a `0/1` indicator equals the corresponding `eventProbability`.

Variance, covariance, moment-generating functions, concentration bounds, and asymptotic limits are explicitly deferred. They can be layered later once the finite expectation base is stable.

Observables are initially `Rat`-valued. Natural graph metrics can be embedded with `Nat.cast`. Do not introduce real-number completion merely to state a finite average.

## 8. Distribution-level BB invariances

The distribution layer should lift the already-proved local/replay compatibility properties instead of leaving them only at individual traces.

### Common fitness scaling

If every stored initial fitness and every entry of the future schedule are multiplied by the same positive rational `c`, then for the same typed target trace:

- every conditional target mass is unchanged;
- `traceProbability` is unchanged;
- the final graph topology is unchanged;
- stored fitness is scaled by `c`;
- therefore any topology-only event has the same event probability.

Do not claim invariance when only initial fitness is scaled. V23.5 already contains a negative fixture showing this changes a multi-birth trace probability.

### Constant-fitness BA specialization

For an experiment in which the initial state and every future schedule entry have one common positive fitness, the typed target-trace law has the same target probabilities as the corresponding unit-fitness BA experiment. Distribution-level topology-event probabilities therefore agree for the same schedule length and `m`.

The implementation plan may split these invariance lifts into a later task if the normalized trace law is already independently useful, but the V23.6 design boundary includes them.

## 9. Network events and observables

The generic distribution layer must not hard-code one network property. After the core event/expectation API is proved, tests should demonstrate that existing finite graph metrics can be lifted as ordinary predicates/observables of `traceFinal`.

Initial examples should include small bounded fixtures only:

- event that final diameter is at most a supplied bound;
- event that a chosen endpoint pair is reachable within `k` hops;
- observable equal to actual edge count;
- observable equal to a selected vertex degree when the final carrier size makes that ID valid.

Use existing `state_bounded`, `shortestHopCount`, `meshDiameter`, `ReachWithin`, and actual adjacency. Do not define parallel distance or diameter functions for the distribution module.

The previously proved eight-node seven-hop path remains a **single positive-probability witness**, not by itself an estimate of the distribution of diameters. V23.6 enables the next class of statements, e.g. exact `P(diameter <= k)` on small finite schedules, but does not assert any large-network or empirical six-degrees result.

## 10. Acceptance fixtures

Use fixtures small enough for exhaustive exact enumeration under existing resource limits.

### Empty schedule

For any valid state and admissible `m`:

- `TargetTrace ... []` has one member;
- its probability is `1`;
- its final state is the initial state;
- `P(True)=1`, `P(False)=0`;
- expectation of any constant `q` is `q`.

### One birth from the unit triangle, m=2

The six ordered target traces are the six ordered pairs of distinct vertices. With unit fitness and triangle degree 2 at every vertex:

- each ordered pair has probability `1/6`;
- all six sum to `1`;
- a target-set event such as “selected set is {1,2}” has mass `2/6 = 1/3` even though the two orders are separate traces.

For the nonconstant triangle fitness `(1,2,4)`, retain the already verified masses including:

- `(2,1)` has `8/21`;
- `(1,2)` has `8/35`;

and verify the sum of all six ordered traces is exactly one through the new trace carrier, not only by `orderedMass_sum_one` at one birth.

### Two births, small carrier

Use the two-node edge, `m=1`, unit fitness, and a two-entry unit schedule. Enumerate the complete typed two-birth trace space. Required checks:

- all trace masses are positive;
- total mass is one;
- at least two traces produce different final topology/degree outcomes;
- an event probability computed by `eventProbability` equals an independently simplified exact rational sum of the matching trace masses;
- expectation of final degree for a selected stable old ID equals the explicit weighted rational sum.

Do not use the seven-hop six-birth fixture as the first exhaustive distribution test; its trace space is unnecessarily large for the foundational acceptance gate.

## 11. Raw replay bridge

V23.5 raw `replay` remains important as an input-validation/executable boundary, but it should not be duplicated inside the distribution layer.

The minimum required semantic bridge is that typed distribution evaluation uses the exact same `orderedMass` and `applyBirth`, plus the theorem equating its total mass to `continuationMass` from `FitnessReplay`. This is sufficient to prevent a second stochastic kernel.

A later implementation task may add a conversion from a typed trace plus schedule into raw `RawBirth` inputs and prove:

```text
replay convertedRawTrace
= ok(final = traceFinal, probability = traceProbability)
```

if that conversion can be implemented cleanly without introducing array-cast complexity or performance regressions. This bridge is desirable but must not block the foundational finite-law implementation if the direct shared-kernel theorem is already exact.

No invalid raw request receives probability mass. Parser/validator failure probabilities are out of scope because V23.6 conditions on a valid typed initial state and valid positive schedule.

## 12. Resource and trust boundary

The feature must remain finite and exact. Preserve the existing pinned Lean 4.32.0 and mathlib revision used by the repository. No dependency update is part of V23.6.

Generic proofs must not use:

- `sorry` or `admit`;
- new user axioms;
- `native_decide` as a proof oracle;
- unsafe escape hatches;
- unlimited heartbeats/recursion as a substitute for proof structure;
- test skips or `continue-on-error` to hide a failing distribution contract.

Concrete finite fixtures may use equation-based `decide_cbv` where it remains within normal resource budgets. If exhaustive enumeration causes large reductions, restructure proofs through the normalized generic theorems and small opaque arithmetic lemmas rather than raising limits blindly. V23.5 demonstrated that fieldwise/theorem-guided proof structure is preferable to evaluating large proof-carrying replay terms wholesale.

Audit new generic laws with `#print axioms`. Expected dependencies may include the repository's existing `propext`, `Classical.choice`, and `Quot.sound`; the review must report actual outputs rather than promise an axiom-free result.

## 13. CI and review protocol

Add a named distribution contract step only when the implementation/test module exists, for example:

```yaml
- name: Fitness distribution contract tests
  run: |
    export PATH="$HOME/.elan/bin:$PATH"
    lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
```

Keep the current exact-head checkout/provenance behavior introduced in the V23.5 line. Do not reintroduce duplicate push/PR Lean runs.

The branch inherits the current BB-scope workflow policy under which Python discovery and World Studio are intentionally skipped on `feature/fitness-*` branches. Those skips are not test successes and must continue to be reported as scope exclusions, not green verification. Do not broaden or further weaken workflow coverage as part of V23.6.

For every implementation task, follow RED/GREEN discipline: new contract first, observe an expected missing declaration or failed mathematical contract, then implement the minimum production change. A dependency outage or unrelated workflow error is not a valid RED.

Final verification must include the exact branch head, full root build, prior fitness Attachment/Birth/Validation/Replay/Scope checks, the new Distribution check, general Lean theorem tests, Story, and Testimony. Existing warnings remain visible.

## 14. Explicit non-goals

V23.6 does **not** provide or claim:

- a PRNG or random sampler;
- Monte Carlo simulation;
- random fitness generation;
- a continuous fitness distribution;
- an infinite stochastic process or projective-limit construction;
- asymptotic degree distribution;
- power-law exponent estimation;
- condensation threshold/classification;
- concentration inequalities;
- central limit behavior;
- expected logarithmic diameter at large `n`;
- high-probability or universal six-hop behavior;
- empirical calibration to a real social network;
- automatic integration with narrative society, belief, observation, or transport layers.

A future sampler should sample from the proved finite conditional rows or a proven-equivalent integer ticket representation; entropy mechanics must remain separate from the mathematical law. A future asymptotic layer should begin only after finite events/expectations are stable and should state its additional assumptions explicitly.

## 15. Success criterion

V23.6 is successful when the repository can state and prove, for any valid finite BB experiment `(initial state, m, fixed positive fitness schedule)`:

1. there is an executable finite type enumerating every legal ordered target trace exactly once;
2. every trace has an exact positive rational probability computed from the real evolving graph;
3. the total probability of all traces is exactly one and equals the existing `continuationMass` semantics;
4. every trace deterministically evaluates to the real final BB state;
5. decidable final-state events have exact rational probabilities with the basic probability laws;
6. rational final-state observables have exact finite expectations with basic linearity;
7. constant-fitness BA and whole-schedule common scaling lift from per-step laws to trace-distribution/topology-event laws;
8. all claims stay within the finite conditional scope and preserve the existing trust/resource boundary.

The implementation should make the conceptual progression explicit:

```text
V23.5
single supplied trace -> exact probability + final state

V23.6
all legal typed traces -> normalized finite law
                        -> exact event probability
                        -> exact expectation
```

This is the intended foundation for later questions such as exact finite `P(diameter <= k)` or expected finite hop count. It is not yet an asymptotic network-science theorem.

## 16. Implementation handoff boundary

This committed design is the review artifact only. After human review/approval, write a separate implementation plan under `docs/superpowers/plans/` using the repository's task-by-task RED/GREEN convention. The plan should decompose at least:

1. typed trace carrier + computable finite enumeration;
2. trace probability + equality to `continuationMass` + normalization;
3. trace final-state evaluation + structural invariants;
4. event probability laws;
5. expectation laws;
6. distribution-level scaling/BA compatibility;
7. small exhaustive network-event fixtures + final trust/CI audit.

Do not begin production implementation from this design commit. Keep any PR Draft through the design and plan review gates; no merge or auto-merge is authorized by this document.