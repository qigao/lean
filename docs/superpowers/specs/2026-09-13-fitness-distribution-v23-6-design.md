# Fitness Distribution V23.6 — exact finite trace law over BB growth

Status: revised design approved for implementation planning. BB is the only attachment-model identity in the maintained repository tree.

## 1. Decision and repository boundary

V23.6 adds a typed finite-distribution layer on top of the verified V23.5 BB kernel. It does not introduce a second attachment model or a second stochastic kernel.

The main new surfaces are:

- `NarrativeDynamics/Core/FitnessDistribution.lean`
- `NarrativeDynamics/Tests/FitnessDistribution.lean`

The distribution layer must reuse:

```text
Targets
orderedMass
applyBirth
State
RunState
continuationMass
continuationMass_one
```

`FitnessReplay.lean` remains the raw validation and deterministic supplied-trace boundary. `NarrativeDynamics.lean` imports the new distribution module only after the implementation exists. README documentation is updated only after verified results.

## 2. Repository-wide BB-only invariant

The maintained tree exposes BB only. No alternate attachment-model namespace, API family, theorem family, fixture family, compatibility alias, or documentation identity is retained.

Useful mathematics survives as BB properties:

- unit-fitness normalization of one common positive constant-fitness profile;
- common positive scaling of current fitness;
- whole-schedule common scaling across seed and all future newborn fitness values;
- invariance of topology-only events under that complete scaling.

The maintained BB-native naming includes surfaces such as:

```text
unitFitnessState
unitFitnessRow
attachment_constant_fitness
orderedMass_constant_fitness
replay_constant_fitness_topology
replay_constant_fitness_probability
```

The implementation must include a static current-tree naming audit for the removed alternate-model surface. The audit applies to current source, tests, README, maintained specs/plans, and workflow files. It does not authorize rewriting Git history or editing already-closed review discussions.

## 3. Exact probability experiment

A V23.6 experiment fixes:

- one valid initial `State n`;
- one fixed attachment count `m` with `0 < m` and `m <= n`;
- one finite `schedule : List PosFitness`, containing the positive newborn fitness for each future birth.

Only ordered target selections are random in this layer. The fitness schedule is conditioned on; it is not sampled.

At each birth, target probability is computed from the actual current state. After a target tuple `T : Targets n m` is selected with exact mass `orderedMass s T`, the next state is exactly:

```text
applyBirth s T hm eta
```

where `eta` is the scheduled fitness for that birth. Later probabilities must use that updated graph and its updated degrees.

The distribution must not freeze the initial graph across the schedule and must not multiply independent copies of the initial target row.

## 4. Typed finite trace carrier

Expose a finite executable trace carrier indexed by current vertex count, attachment count, and remaining number of births. The intended semantic shape is:

```lean
def TargetTrace (n m : Nat) : Nat → Type
  | 0 => PUnit
  | steps + 1 => Targets n m × TargetTrace (n + 1) m steps
```

The exact Lean syntax may be adjusted for elaboration, but these invariants are fixed:

- zero future births has exactly one trace;
- each nonempty trace begins with one legal ordered injective `Targets n m`;
- the tail starts from carrier size `n+1`;
- newborn fitness does not affect carrier shape;
- duplicate targets within one birth are impossible by type;
- a future vertex cannot be selected before it exists;
- enumeration is computable without choice-based `Fintype.ofFinite`.

The schedule values are deliberately not part of the carrier type. Distribution functions pair `TargetTrace n m schedule.length` with the actual schedule.

## 5. Exact trace probability

Define:

```lean
traceProbability
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule.length) : Rat
```

with recurrence:

```text
P(empty) = 1
P(T :: rest) = orderedMass(s,T)
                 * P(rest from applyBirth(s,T,eta))
```

Required generic theorems:

```text
traceProbability_pos
traceProbability_nonneg
traceProbability_le_one
traceProbability_sum_continuationMass
traceProbability_sum_one
```

The central proof must identify the explicit finite trace sum with the already-proved recursive `continuationMass`:

```text
Σ trace, traceProbability(trace)
  = continuationMass(s,m,schedule)
  = 1
```

This requirement prevents a second normalization implementation. For a nonempty schedule, the finite sum decomposes as a product-carrier sum over the first `Targets n m` and the tail trace from `n+1`.

Every typed trace has strictly positive exact rational mass because every `orderedMass` on a valid BB state is positive.

## 6. Authoritative trace evaluation

Define deterministic trace evaluation:

```lean
traceFinal
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule.length) : RunState
```

Evaluation uses the same `applyBirth` recurrence as trace probability. A trace stores only target choices; it does not store a graph, degree vector, fitness vector, or probability cache.

Required facts:

```text
final node count = n + schedule.length
final state is valid
actual edge count = initial edges + m * schedule.length
final fitness list = initial fitness list ++ schedule values
```

The birth-level fitness-list theorem should live at the graph-growth layer so replay and distribution share one statement rather than duplicating it.

## 7. Exact event probability

For a decidable predicate on final `RunState`, define:

```lean
eventProbability ... (event : RunState → Prop) [DecidablePred event] : Rat
```

as:

```text
Σ trace,
  if event(traceFinal(trace))
  then traceProbability(trace)
  else 0
```

Required laws:

```text
P(True) = 1
P(False) = 0
0 <= P(E)
P(E) <= 1
P(E) + P(not E) = 1
E implies F  ->  P(E) <= P(F)
```

Events are over valid typed traces only. Raw parser/validator errors are not probability outcomes.

Multiple ordered traces may reach the same final graph. Their masses must be summed. Event probability must never deduplicate final states.

## 8. Exact rational expectation

For `observable : RunState → Rat`, define:

```lean
expectation ... observable : Rat
```

by:

```text
Σ trace,
  traceProbability(trace) * observable(traceFinal(trace))
```

Required first-layer algebra:

```text
expectation of a constant
linearity under addition
rational scalar multiplication
expectation of a 0/1 indicator = eventProbability
```

Natural graph metrics may be embedded with `Nat.cast`. V23.6 does not introduce real completion, variance, covariance, moment-generating functions, concentration bounds, or asymptotic expectation laws.

## 9. BB distribution invariances

### Common scaling

If one positive rational factor multiplies every initial fitness and every future scheduled fitness, then for the same target trace:

- every conditional target mass is unchanged;
- exact `traceProbability` is unchanged;
- final graph topology is unchanged;
- final stored fitness is scaled by that factor;
- every scale-invariant topology event has the same event probability.

The existing V23.5 negative fixture remains authoritative: scaling only the seed while leaving future newborn fitness unchanged is not an invariance and changes the tested multi-birth trace probability.

### Constant-fitness normalization

If the initial state and every scheduled newborn share one common positive fitness value, changing that common value to another common positive value does not change the target-trace law or topology-event probabilities.

This is stated only as a BB scaling/normalization theorem. No alternate model identity is introduced.

## 10. Network events and observables

The generic distribution layer does not hard-code one network statistic. Tests may lift existing verified finite metrics as ordinary predicates and observables of `traceFinal`, including:

- final diameter bounded by a supplied `k`;
- reachability of a chosen endpoint pair within `k` hops;
- actual edge count;
- degree of a stable old vertex.

Reuse:

```text
state_bounded
ReachWithin
shortestHopCount
meshDiameter
degree
actualEdgeCount
```

Do not define a parallel notion of distance or diameter.

The V23.5 seven-hop path remains one positive-probability witness. It is not an estimate of typical diameter. V23.6 enables exact finite event questions such as `P(diameter <= k)` on small schedules without asserting an empirical or asymptotic six-hop law.

## 11. Acceptance fixtures

### Empty schedule

For any valid initial state and admissible `m`:

```text
Fintype.card(TargetTrace ... 0) = 1
traceProbability(empty) = 1
traceFinal(empty) = initial state
P(True) = 1
P(False) = 0
E[constant q] = q
```

### One birth from a unit-fitness triangle, m=2

There are six ordered target traces. Because degree and fitness are equal, each ordered pair has mass `1/6`; all six sum to one.

A final-graph event whose selected target set is `{1,2}` receives mass `2/6 = 1/3`, because both target orders contribute.

For the nonconstant triangle fitness `(1,2,4)`, retain the verified exact masses:

```text
P[(2,1)] = 8/21
P[(1,2)] = 8/35
```

and verify total trace mass one through the new trace carrier.

### Two births from a two-node edge, m=1

Use unit positive fitness for both births. Exhaustively verify:

- all trace masses are positive;
- total mass is one;
- at least two traces produce different final degree/topology outcomes;
- one event probability equals an independently simplified rational sum;
- expected final degree of one stable old vertex equals the explicit weighted rational sum.

Keep this small fixture as the foundational exhaustive gate. Do not use the six-birth seven-hop fixture as the first distribution enumeration.

## 12. Raw replay bridge

Raw `replay` stays the validation/executable boundary. The mandatory semantic bridge is shared use of:

```text
orderedMass
applyBirth
continuationMass
```

A later task may convert a typed target trace plus fitness schedule to `RawBirth` inputs and prove equality between replay output and `(traceFinal, traceProbability)`. That bridge is desirable but not required to establish the finite law if it introduces unnecessary dependent array casts or proof cost.

Invalid raw requests receive no probability mass in V23.6.

## 13. Trust and resource boundary

Preserve Lean 4.32.0 and pinned mathlib revision `81a5d257c8e410db227a6665ed08f64fea08e997`.

Generic proofs must not use:

- `sorry` or `admit`;
- new user axioms;
- `native_decide` as a proof oracle;
- unsafe escape hatches;
- unlimited resource settings;
- skipped or `continue-on-error` proof gates.

Concrete finite fixtures may use equation-based `decide_cbv` under normal bounded resources. Expensive reductions must be restructured through generic theorems and small arithmetic lemmas rather than hidden by larger global limits.

Audit new generic theorems with `#print axioms` and report actual dependencies.

## 14. CI and review protocol

The implementation adds a named bounded `Fitness distribution contract tests` step once the module exists. It preserves:

- exact-head checkout/provenance;
- push/PR proof deduplication;
- conformance comparison;
- full root build;
- existing fitness attachment/birth/validation/replay/scope checks;
- general Lean theorem tests;
- Story and Testimony.

The branch retains the existing fitness-branch peripheral-job scope policy. A skipped peripheral job is an exclusion, not a success.

Every implementation task follows RED/GREEN discipline and stops at its review gate.

## 15. Explicit non-goals

V23.6 does not provide or claim:

- another attachment model alongside BB;
- compatibility aliases for removed model vocabulary;
- a PRNG or random sampler;
- Monte Carlo simulation;
- random fitness generation;
- a continuous fitness distribution;
- an infinite stochastic process;
- asymptotic degree distribution;
- power-law exponent estimation;
- condensation classification;
- concentration inequalities;
- central-limit behavior;
- expected logarithmic diameter at large `n`;
- high-probability or universal six-hop behavior;
- empirical calibration to a real social network;
- automatic coupling to society, belief, observation, or transport layers.

## 16. Success criterion

V23.6 succeeds when, for any valid finite BB experiment `(initial state, m, fixed positive fitness schedule)`:

1. an executable finite type enumerates every legal ordered target trace exactly once;
2. every trace has an exact positive rational mass from the real evolving graph;
3. total trace mass equals existing `continuationMass` and exactly one;
4. every trace evaluates to the authoritative final BB state;
5. decidable final-state events have exact rational probabilities with basic finite probability laws;
6. rational final-state observables have exact expectations with basic linearity;
7. whole-experiment common scaling and constant-fitness normalization lift to trace/event laws;
8. the maintained repository tree exposes BB only and preserves the established trust/resource boundary.

Conceptually:

```text
V23.5
supplied trace -> exact probability + final state

V23.6
all legal typed traces -> normalized finite law
                        -> exact event probability
                        -> exact expectation
```

This finite layer is the foundation for later exact questions such as `P(diameter <= k)` and expected finite hop metrics. It is not an asymptotic network-science theorem.
