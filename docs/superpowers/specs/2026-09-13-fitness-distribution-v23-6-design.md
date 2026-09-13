# Fitness Distribution V23.6 — exact finite trace law over BB growth

Status: revised design for review; no production implementation, new theorem compilation, or CI success is claimed by this document.
User direction: after merging the finite BB replay and seven-hop counterexample, promote exact per-trace replay probabilities into a first-class finite probability law. The repository model vocabulary is now simplified further: **BB is the only attachment model name. No BA model/API/theorem/fixture/documentation surface remains in the current repository tree.**

## 1. Decision and repository boundary

Add a typed finite-distribution layer on top of the merged V23.5 BB model. The selected architecture is a **typed finite target-trace space** whose members enumerate every legal ordered target choice for a fixed initial state, fixed attachment count `m`, and fixed positive newborn-fitness schedule. Each trace receives the exact rational product of the existing conditional `orderedMass` terms, and each trace is evaluated by the existing `applyBirth` transition.

Start `feature/fitness-distribution-v23-6` from merge commit `b432cd21eb8bac04bae8f676b7321734b90ed5c5`, which merged PR #66 into `feature/fitness-validation-v23-5`. This is a successor layer to V23.5, not an alternative replay implementation.

The main new production surface is:

- `NarrativeDynamics/Core/FitnessDistribution.lean`
- `NarrativeDynamics/Tests/FitnessDistribution.lean`

`NarrativeDynamics.lean` may import the new core module only after implementation exists. `README.md` is updated only after verification. Existing `FitnessReplay.lean` remains the deterministic raw-input boundary. Distribution code must reuse `Targets`, `orderedMass`, `applyBirth`, `State`, and the fixed-schedule semantics already proved there.

A deliberate cleanup is also part of V23.6: remove all legacy BA naming from the current tree while preserving the underlying BB constant-fitness theorems under BB-native names. Do not maintain two model identities.

Alternatives considered:

1. **Typed finite BB target-trace distribution** — selected. It preserves exact `Rat`, reuses the proved BB kernel directly, has a finite `Fintype` carrier, and supports exact event probabilities and expectations.
2. **Mathlib PMF/Measure first** — deferred. It introduces `ENNReal`, countable-support/measure interfaces, and conversion obligations before they add value to this finite exact model.
3. **Only recursive `eventMass` functions** — rejected as the primary design. They answer isolated questions but do not expose a reusable trace space for expectation, conditioning, exhaustive fixtures, and future sampling bridges.

No general-purpose probability library is introduced. The abstraction is intentionally scoped to BB.

## 2. Repository-wide BB-only naming invariant

The current repository tree must expose **BB only**. The implementation plan must begin with a complete search and purge of BA model terminology and symbols.

The cleanup includes, at minimum, currently known surfaces such as:

- `asBA`;
- `baRow`;
- `attachment_ba`;
- `orderedMass_ba` and any similarly named target-law theorem;
- `replay_ba_topology`;
- `replay_ba_probability`;
- comments describing a BA specialization;
- BA-named tests or fixtures;
- README text presenting BA as a separate or recovered model;
- current-tree design/plan documents that still present BA as a model identity.

The mathematical content is not discarded when it is useful. Instead, it is expressed as BB properties. For example:

- a constant positive fitness profile has the same attachment law as its unit-fitness normalization;
- multiplying all current and future fitness values by one positive constant leaves target probabilities unchanged;
- constant-fitness topology probabilities are invariant under the common fitness value.

Use BB-native names such as `constantFitnessRow`, `attachment_constant_fitness`, `orderedMass_constant_fitness`, `replay_constant_fitness_topology`, or equivalent concise names chosen during implementation. Do not retain aliases with BA names for compatibility: aliases would violate the BB-only invariant.

Acceptance requires a static current-tree search proving no model-facing BA token remains. At minimum search for case-sensitive/case-insensitive variants of `BA`, `Barabasi-Albert`, `Barabási–Albert`, `asBA`, `baRow`, and `_ba_`-style identifiers. False positives must be reviewed rather than blindly ignored.

This invariant applies to the checked-out repository tree, public Lean API, tests, README, current specs, and current plans. It does **not** authorize destructive Git-history rewriting or editing already-closed PR discussions. Historical commits remain archival evidence; the maintained tree must be BB-only.

## 3. Exact probability experiment

A V23.6 experiment fixes:

- an initial valid `State n`;
- one fixed attachment count `m` with `0 < m` and `m <= n`;
- a finite `schedule : List PosFitness`, one positive newborn fitness for each future birth.

Only ordered target selections are random in this layer. The fitness schedule is conditioned on and is not sampled. V23.6 therefore defines a finite probability law **conditional on the supplied positive fitness schedule**.

At birth `t`, the current state is the result of preceding target choices. A target choice is one `Targets currentN m`, an injective ordered embedding of `Fin m` into existing vertices `Fin currentN`. Its conditional mass is the existing `orderedMass currentState T`.

The next state is exactly:

```text
applyBirth currentState T hm eta
```

where `eta` is the schedule entry for that birth. Later conditional probabilities are computed from the actual updated graph and updated degrees. V23.6 must not freeze the initial graph for the whole schedule and must not multiply independent initial rows.

## 4. Typed trace carrier

Define a dependent finite trace carrier whose type records the changing number of available old vertices. The intended shape is equivalent to:

```lean
def TargetTrace (n m : Nat) : List PosFitness → Type
  | [] => PUnit
  | _eta :: rest => Σ T : Targets n m, TargetTrace (n + 1) m rest
```

The concrete Lean syntax may change for elaboration, but the semantic contract is fixed:

- empty schedule has exactly one empty trace;
- a nonempty trace begins with one legal `Targets n m`;
- its tail starts from carrier size `n + 1`;
- schedule fitness affects transition/probability, not carrier shape;
- the carrier is finite and executable without choice-based enumeration.

Provide a recursive `Fintype` instance from the existing computable `Fintype (Targets n m)` and finite sigma/product instances. Avoid `Fintype.ofFinite` or another choice-based executable enumeration.

The carrier enumerates every legal ordered target sequence exactly once. Duplicate targets in one birth are absent by construction. Future vertices are unrepresentable before they exist.

## 5. Exact trace probability

Define:

```lean
traceProbability
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule) : Rat
```

with semantics:

```text
P(empty) = 1
P(T :: rest) = orderedMass(s,T)
                 * P(rest from applyBirth(s,T,eta))
```

No new probability formula is permitted. The first factor must be the existing BB `orderedMass`; the successor must be the real `applyBirth` result.

Required generic theorems:

- `traceProbability_pos`;
- `traceProbability_nonneg`;
- `traceProbability_le_one`;
- `traceProbability_sum_continuationMass`;
- `traceProbability_sum_one`.

The key proof is equality with the existing `continuationMass`, not an independent normalization tree. For a nonempty schedule, finite summation over the sigma carrier should reduce to:

```text
Σ T : Targets n m,
  Σ tail : TargetTrace (n+1) m rest,
    orderedMass s T * traceProbability next tail
```

then use induction to recover the existing `continuationMass` recurrence and finally `continuationMass_one`.

## 6. Trace evaluation and outcome identity

Define deterministic evaluation using the same transition:

```lean
traceFinal
  (s : State n) (m : Nat) (hm : 0 < m) (hb : m ≤ n)
  (schedule : List PosFitness)
  (trace : TargetTrace n m schedule) : RunState
```

or an equivalent typed final result if it reduces casts.

Required facts:

- final node count is `n + schedule.length`;
- final state is valid;
- every step is exactly `applyBirth` with the trace target and scheduled fitness;
- actual edge count increases by `m * schedule.length`;
- stored fitness extends by schedule values in order.

Do not store a parallel graph, degree vector, or probability in the trace carrier. A trace contains target choices only; outcomes derive from the authoritative BB transition.

## 7. First-class finite event probability

Define exact event mass over final outcomes:

```lean
eventProbability
  ...
  (event : RunState → Prop) [DecidablePred event] : Rat
```

with semantics:

```text
P(E) = Σ trace,
  if E (traceFinal trace)
  then traceProbability trace
  else 0
```

Required laws:

- `eventProbability_true = 1`;
- `eventProbability_false = 0`;
- nonnegativity;
- upper bound by one;
- complement: `P(E) + P(not E) = 1`;
- monotonicity under event implication.

The law is over typed legal traces. Invalid raw replay requests are validation behavior, not probability mass.

Multiple ordered traces may reach the same graph. Their masses must be summed, never deduplicated by final state.

## 8. Exact rational expectation

Define:

```lean
expectation
  ...
  (observable : RunState → Rat) : Rat
```

with semantics:

```text
E[f] = Σ trace,
  traceProbability(trace) * f(traceFinal(trace))
```

Required first-layer laws:

- expectation of a constant;
- addition linearity;
- rational scalar multiplication;
- indicator expectation equals the corresponding event probability.

Observables are initially `Rat`-valued. Natural graph metrics use `Nat.cast`. Variance, covariance, concentration, and asymptotics remain deferred.

## 9. BB distribution invariances

V23.6 lifts existing BB invariances to the full trace distribution.

### Common fitness scaling

If every initial fitness and every future schedule value are multiplied by the same positive rational `c`, then for the same target trace:

- every conditional target mass is unchanged;
- `traceProbability` is unchanged;
- final graph topology is unchanged;
- stored fitness is scaled by `c`;
- every topology-only event has the same probability.

Do not claim invariance when only initial fitness is scaled; the existing multi-birth negative fixture already shows that changing only part of the schedule changes trace mass.

### Constant-fitness normalization

If the initial state and every schedule entry have one common positive fitness, changing that common value to another positive common value does not change the target-trace law or topology-event probabilities. This is a BB scaling theorem, not a separate model specialization.

There is no BA compatibility section, model, module, namespace, public theorem family, fixture family, or documentation concept in V23.6.

## 10. Network events and observables

The generic distribution layer must not hard-code one network property. Tests should demonstrate existing finite graph metrics as ordinary predicates/observables of `traceFinal`.

Initial examples may include:

- final diameter at most a supplied bound;
- a chosen endpoint pair reachable within `k` hops;
- actual edge count;
- degree of a selected stable old vertex when its ID is valid.

Reuse `state_bounded`, `shortestHopCount`, `meshDiameter`, `ReachWithin`, and actual adjacency. Do not define parallel distance or diameter functions.

The previous eight-node seven-hop path remains a single positive-probability witness, not an estimate of the diameter distribution. V23.6 enables exact finite statements such as `P(diameter <= k)` on small schedules without making a large-network six-degrees claim.

## 11. Acceptance fixtures

Use fixtures small enough for exhaustive exact enumeration under existing resource limits.

### Empty schedule

For any valid state and admissible `m`:

- one trace;
- probability `1`;
- final state equals initial state;
- `P(True)=1` and `P(False)=0`;
- expectation of constant `q` is `q`.

### One birth from a unit-fitness triangle, m=2

The six ordered pairs of distinct vertices are six distinct traces. With equal fitness and equal degree, each ordered pair has probability `1/6`; all six sum to one. A target-set event such as `{1,2}` has mass `2/6 = 1/3` because both orders contribute separately.

For nonconstant triangle fitness `(1,2,4)`, retain the verified masses including `(2,1)=8/21` and `(1,2)=8/35`, and verify through the new trace carrier that all six ordered traces sum exactly to one.

### Two births, small carrier

Use the two-node edge, `m=1`, unit fitness, and a two-entry unit schedule. Exhaustively verify:

- all trace masses are positive;
- total mass is one;
- at least two traces produce different final degree/topology outcomes;
- one event probability equals an independently simplified rational sum;
- expected final degree of one stable old vertex equals its explicit weighted rational sum.

Do not use the seven-hop six-birth fixture as the first exhaustive distribution acceptance test.

## 12. Raw replay bridge

V23.5 raw `replay` remains the input-validation/executable boundary. Do not duplicate it inside the distribution layer.

The mandatory bridge is shared semantics: distribution evaluation uses the same `orderedMass`, `applyBirth`, and `continuationMass` already used by replay.

A later task may convert a typed trace plus schedule into `RawBirth` inputs and prove:

```text
replay convertedRawTrace
= ok(final = traceFinal, probability = traceProbability)
```

if this remains clean and performant. This bridge is desirable but must not block the foundational finite law.

## 13. Resource and trust boundary

Preserve Lean 4.32.0 and the pinned mathlib revision. No dependency update is part of V23.6.

Generic proofs must not use:

- `sorry` or `admit`;
- new user axioms;
- `native_decide` as a proof oracle;
- unsafe escape hatches;
- unlimited heartbeats/recursion as a substitute for proof structure;
- test skips or `continue-on-error` to hide a failed contract.

Concrete finite fixtures may use equation-based `decide_cbv` within normal budgets. If exhaustive reduction becomes expensive, restructure through generic normalized theorems and small opaque arithmetic lemmas rather than raising limits blindly.

Audit new generic laws with `#print axioms` and report actual dependencies.

## 14. CI and review protocol

Add a named distribution contract step only when the module exists:

```yaml
- name: Fitness distribution contract tests
  run: |
    export PATH="$HOME/.elan/bin:$PATH"
    lake env lean NarrativeDynamics/Tests/FitnessDistribution.lean
```

Keep current exact-head checkout/provenance behavior and avoid duplicate push/PR Lean work.

The branch inherits the current fitness-branch scope policy where Python discovery and World Studio are intentionally skipped. Those skips are not successes. Do not broaden or further weaken workflow coverage.

For every implementation task, follow RED/GREEN discipline: add the contract first, observe an expected missing declaration or mathematical failure, then implement the minimum production change.

Final verification includes exact branch head, full root build, prior fitness Attachment/Birth/Validation/Replay/Scope checks, the new Distribution check, general Lean theorem tests, Story, Testimony, axiom audit, and the BB-only static naming audit.

## 15. Explicit non-goals

V23.6 does not provide or claim:

- a second attachment model alongside BB;
- compatibility aliases for removed BA names;
- a PRNG or random sampler;
- Monte Carlo simulation;
- random fitness generation;
- a continuous fitness distribution;
- an infinite stochastic process;
- asymptotic degree distribution;
- power-law exponent estimation;
- condensation classification;
- concentration inequalities;
- central limit behavior;
- expected logarithmic diameter at large `n`;
- high-probability or universal six-hop behavior;
- empirical calibration to a real social network;
- automatic integration with narrative society, belief, observation, or transport layers.

A future sampler must sample from the proved finite conditional rows or a proven-equivalent representation. Entropy mechanics remain separate from the mathematical law.

## 16. Success criterion

V23.6 succeeds when the repository can state and prove, for any valid finite BB experiment `(initial state, m, fixed positive fitness schedule)`:

1. an executable finite type enumerates every legal ordered target trace exactly once;
2. every trace has an exact positive rational probability from the real evolving BB graph;
3. total trace probability is exactly one and equals existing `continuationMass` semantics;
4. every trace deterministically evaluates to the real final BB state;
5. decidable final-state events have exact rational probabilities and basic probability laws;
6. rational final-state observables have exact finite expectations and basic linearity;
7. common whole-schedule fitness scaling and constant-fitness normalization hold at distribution level as BB invariances;
8. current production API, tests, README, specs, and plans expose BB only, with no BA aliases or model terminology;
9. all claims remain finite, conditional, exact, and inside the existing trust/resource boundary.

Conceptually:

```text
V23.5
single supplied BB trace -> exact probability + final state

V23.6
all legal typed BB traces -> normalized finite law
                           -> exact event probability
                           -> exact expectation
```

## 17. Implementation handoff boundary

This committed design is the review artifact only. After human approval, write a separate implementation plan under `docs/superpowers/plans/`.

The plan must decompose at least:

1. **BB-only cleanup:** remove BA-named APIs/theorems/tests/docs and replace useful mathematics with BB constant-fitness/scaling names; add static no-BA audit;
2. typed trace carrier + computable finite enumeration;
3. trace probability + equality to `continuationMass` + normalization;
4. trace final-state evaluation + structural invariants;
5. event probability laws;
6. expectation laws;
7. distribution-level BB scaling/constant-fitness invariance;
8. small exhaustive network-event fixtures + final trust/CI/current-tree naming audit.

Do not begin production implementation from this design commit. Keep PR #70 Draft through the design and plan review gates; no merge or auto-merge is authorized by this document.
