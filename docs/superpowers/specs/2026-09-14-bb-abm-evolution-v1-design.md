# BB-driven ABM Evolution V1 Design

Status: approved for implementation planning by the user's subsequent `go`. The implementation plan is `docs/superpowers/plans/2026-09-14-bb-abm-evolution-v1.md`; implementation has not started.

Base: `proof/narrative-dynamics-v0` at `0a41462a37064a2a3ded3676ebdfb24b9243fc8c`.

## 1. Objective and meaning of completion

Compose the existing finite BB growth kernel with a concrete, synchronous agent propagation rule. A successful run must actually create agents, preserve their identities and retained state through growth, update their beliefs through adjacent broadcasting agents, and produce one joint final state with the existing exact BB target-trace probability.

The claim after implementation will be:

> For the specified rational scalar-belief ABM, fixed positive fitness, and a valid finite birth calendar, every legal BB target trace induces a well-defined finite joint evolution. The transition preserves the stated graph and agent invariants, and its graph projection and target-trace probability agree with the existing BB model.

This is a proof for one concrete network ABM. It is not a proof for arbitrary agent programs, the full situated cognitive runtime, realistic human behavior, or a universal four-hop or six-hop bound. Small-world measurements remain observations of generated graphs, not admission requirements for growth.

## 2. Repository evidence and integration boundary

The relevant existing components are:

| Component | Existing contract | Consequence for this design |
| --- | --- | --- |
| `NarrativeDynamics/Core/FitnessBirth.lean` | `applyBirth` creates a new `Fin (n + 1)` vertex, with `oldId` and `newId`; old edges and fitness persist | Reuse the same birth and ID embeddings |
| `NarrativeDynamics/Core/FitnessValidation.lean` | `parseSeed`, `validateBirth`, and `step` check raw inputs and return exact errors | Delegate BB validation and growth; do not copy the attachment algorithm |
| `NarrativeDynamics/Core/FitnessReplay.lean` | Replay preserves graph validity and computes exact ordered trace mass | Require projection agreement for the filtered birth sequence |
| `NarrativeDynamics/Core/FitnessDistribution.lean` | `TargetTrace` and `traceProbability_sum_one` define a normalized finite law | Push that law through the joint evaluator |
| `narrative_dynamics/abm/simulation.py` | `simulate_round` reads one prior snapshot; `NetworkRoundResult` requires the same roster and model on both sides | Reuse its behavioral specification, with test-only single-round comparisons |
| `narrative_dynamics/abm/evolving.py` and its V5 design | Entry and exit operate on a fixed catalog of possible agents and edges | Catalog entry is not BB creation of a fresh identity |
| `narrative_dynamics/abm/situated_network.py` and its V19 design | World, perception, cognition, social state, and memory bind exact fixed rosters; network snapshots are derived views | Do not insert new BB agents into these hashes or turn projected edges into observations |

Three approaches were considered:

1. **Lean composition with a concrete propagation kernel, recommended.** Proves actual agent updates and graph growth together while keeping the first population model small. The rational rule specializes the existing V1 propagation semantics.
2. **Directly extend the Python V1 runtime first.** Produces a live growing simulation, but requires a separate changing-model lineage contract and an explicit numeric conformance boundary before a formal proof can cover it.
3. **Integrate directly into V19.** Requires a coordinated migration of embodied rosters, world state, perception, private cognition, social state, and database checkpoints. This is a separate architecture project.

V1 selects approach 1. The test comparisons with Python do not constitute a production birth adapter or a proof of Python floating-point arithmetic.

## 3. Joint state and identity

At a given node count `n`, the joint state contains:

- `network : FitnessAttachment.State n`;
- `profiles : Fin n -> AgentProfile`;
- `agents : Fin n -> AgentState`.

The run wrapper also records a propagation-round count. The global attachment count `m` belongs to the validated run configuration, with `0 < m` and `m <= seed.nodeCount` even for an empty schedule.

An `AgentProfile` contains rational receptivity `r` and broadcast threshold `theta`, both in `[0, 1]`. An `AgentState` contains rational belief `b` in `[0, 1]` and a natural-number cumulative exposure count `e`. Broadcasting is derived as `b >= theta`; it is not a second mutable field that can disagree with belief.

Fitness is stored only in the existing BB state. It is strictly positive, immutable after the corresponding birth, and is not reused as receptivity, trust, belief, or edge influence. Profiles also remain fixed after birth.

`Fin n` is the canonical agent identity. Growth transports every old profile and state through the existing `oldId n`; the new agent receives `newId n`, whose numerical ID is `n`. Existing IDs are never sorted again or reassigned according to degree or fitness.

The fixed-horizon design needs no unbounded identity catalog and no retired-agent state. External string-ID bindings are a later runtime-adapter concern.

## 4. Network and interaction semantics

The BB graph remains an undirected simple graph. Each undirected edge permits two possible directed transmissions, one in each direction. Whether a transmission occurs is determined by the source's broadcasting status in the shared prior snapshot.

Every permitted channel in V1 has influence one. There are no extra channels, parallel relations, inactive edges, or independently weighted edges. BB degree is computed from the undirected graph, not from a doubled transmission list.

For receiver `i`, let `R_i` be the finite set of adjacent agents that are broadcasting in the propagation input snapshot. Let `c_i` be its cardinality. Then:

- If `c_i = 0`, preserve `b_i` and `e_i`.
- Otherwise, compute `q_i = (sum of b_j over j in R_i) / c_i`, set `b_i' = (1 - r_i) * b_i + r_i * q_i`, and set `e_i' = e_i + c_i`.
- Next-round broadcasting is derived from `b_i' >= theta_i`.

This is the exact rational specialization of the V1 Python formula for unit-influence edges: a nonempty incoming set has total influence at least one, so its assimilation is exactly receptivity.

Every next state is calculated from the same snapshot. A newly changed receiver cannot relay that change during this propagation round. Source and receiver may both update, but neither update is read by the other until a later round.

Zero receptivity preserves belief, but received transmissions still increase exposure count. A threshold of zero permits an agent with zero belief to broadcast a zero signal. These are deliberate consequences of the existing V1 rule.

The rule is a scalar belief and exposure model. It does not create testimony, private evidence, percepts, memory records, or semantic claims in the situated runtime.

## 5. One joint tick and finite schedules

A tick supplies either no birth or exactly one birth, followed by exactly one propagation round.

A birth supplies the existing ordered BB targets and positive newborn fitness, plus the newborn receptivity, threshold, and initial belief. Newborn exposure is always zero before propagation. The initial belief is an explicit exogenous seed; it is not inferred from fitness or silently copied from a neighbor.

The tick executes in this order:

1. Validate the whole tick input.
2. For a birth tick, call the existing BB checked step, transport old profiles and agent states, and initialize the new agent. For a no-birth tick, retain the current network and population.
3. Freeze this complete post-growth state as the propagation input.
4. Compute every permitted transmission and next agent state from that frozen state.
5. Return the new joint state and advance the propagation-round count by one.

The new agent can receive immediately. It can also transmit immediately if its explicitly supplied initial belief meets its threshold. It cannot transmit an update that it receives during the same tick.

A no-birth tick has probability factor one. A birth tick has exactly the `orderedMass` returned by the existing BB step on the pre-growth network. Agent updates are deterministic and contribute no additional probability factor.

The scheduler therefore supports continued propagation after growth stops. After `T` ticks containing `B` births, the population has `n0 + B` agents and the round count has advanced by `T`.

The whole raw replay returns either one complete successful result or one error. Failure does not expose an accumulated probability, a partial graph, a partial population, or successful-prefix results. It has no external side effects.

## 6. Checked input and error contract

The raw seed pairs the existing `RawSeed` with an array of agent records in vertex-ID order. Seed records contain receptivity, threshold, belief, and a natural-number exposure count. The array must have exactly `nodeCount` records.

Validation precedence is fixed:

1. Existing BB seed parsing, preserving its error.
2. Existing initial `m` requirement.
3. Seed agent-array length.
4. Seed agent records in ascending ID order, checking receptivity, threshold, then belief bounds.
5. Ticks in authored order. At a birth tick, existing BB birth validation precedes newborn receptivity, threshold, and belief validation in that order.

No-birth ticks require no newborn input. They cannot carry a hidden partial birth request. Rational values are already typed `Rat` values; this layer introduces no JSON parser or floating-point conversion. Natural-number counts cannot encode negative values.

Errors distinguish seed-network, initial-attachment-count, seed-agent-count, indexed seed-agent field, and indexed tick field failures. A BB birth failure retains the original BB cause plus both the zero-based tick index and the zero-based birth index. No-birth ticks increment only the tick index.

Prove soundness and completeness of the added agent validation, success for every well-formed finite schedule, and first-failure precedence. Do not accept graph or agent invariant claims from raw callers as trusted assertions.

## 7. Required proof surface

The public proof surface must establish the following properties about the concrete functions, not take them as callback assumptions:

| Property | Required statement |
| --- | --- |
| Growth projection | Projecting the typed joint birth gives the existing `applyBirth` result |
| Retained agents at birth | Every old profile and agent state is unchanged under `oldId` before propagation |
| Newborn initialization | `newId` receives exactly the supplied profile and belief, with zero initial exposure |
| Propagation projection | Propagation changes neither the graph, fitness, IDs, nor profiles |
| Agent validity | The computed rational mean and convex update preserve belief bounds; exposure stays a natural number and never decreases |
| Exact edge-local transmission | A source-receiver transmission occurs exactly once if and only if the endpoints are adjacent and the source broadcasts in the input snapshot |
| No-input preservation | A receiver with no incoming transmissions retains its complete local state |
| Snapshot locality | Equal receiver state/profile and equal adjacent source beliefs/broadcast status yield the same next receiver state; unrelated agent state cannot affect that round |
| Finite execution | Every validated finite schedule has a successful joint result; replay composes the concrete tick |
| Graph replay agreement | Filtering births out of a successful joint schedule and running existing BB replay yields the same node count, adjacency, and fitness |
| Probability agreement | Joint replay probability equals the probability from that filtered BB replay |
| Count preservation | Agent count equals node count; final node and edge counts depend only on the number of births, with edges increasing by `m` per birth |

The old-agent preservation claim applies only to the growth stage. During propagation, old agents are expected to change. Equality of graph projections is observational equality of node count, adjacency, and fitness, without requiring equality of proof fields or decidability instances.

Single-round locality applies to the propagation operator with its graph fixed. The BB target selection itself depends on global degree and fitness normalization. No theorem may mislabel that selection as a local decision.

A static final-graph distance is not, by itself, a proof about when information arrived in a graph whose edges are being created over time. V1 proves snapshot locality and concrete delayed-relay fixtures; general temporal-path propagation theorems are deferred.

## 8. Exact finite probability of agent outcomes

Fix the validated seed, initial agent state, `m`, finite birth calendar, and the newborn fitness/profile/initial-belief sequence. Only the ordered target choices vary.

Evaluate the existing `TargetTrace` through the joint tick schedule. The population updates do not feed back into fitness, the birth calendar, or later target-choice rules. Consequently each trace has its existing `traceProbability`, and all trace masses still sum to one by `traceProbability_sum_one`.

Define the probability of a decidable joint final-state event as the finite sum of those existing masses whose joint evaluations satisfy the event. This is a pushforward of the existing law, not a new sampler or a normalization over selected successful outcomes. Prove the probability lies in `[0, 1]`, the always-true event has mass one, and the always-false event has mass zero.

Every trace is evaluated exactly once. Distinct ordered traces that lead to the same joint final state retain their separate masses and are both counted. A typed calendar with no births has the existing one-element empty-trace carrier and deterministic propagation.

The probability statement is conditional on the supplied birth calendar and agent parameters. It does not give a probability to real-world behavior, fitness values, or the author choosing those parameters.

## 9. Concrete acceptance scenarios

These are proof obligations and regression targets for implementation, not results already established by this design document.

### 9.1 One birth with a behavioral outcome probability

Use seed edge `0--1`, fitness `[1, 1]`, `m = 1`, receptivity one, threshold `1/2`, initial beliefs `[1, 0]`, and zero exposures. New agent `2` has fitness one, receptivity one, threshold `1/2`, and initial belief zero.

| Birth target | BB mass | Beliefs after birth and one propagation round | Exposures |
| --- | --- | --- | --- |
| `0` | `1/2` | `[1, 1, 1]` | `[0, 1, 1]` |
| `1` | `1/2` | `[1, 1, 0]` | `[0, 1, 0]` |

Therefore the exact probability that newborn `2` broadcasts after that tick is `1/2`. Both final graphs have diameter two, but the agent outcome differs because the source and attachment endpoint differ.

Adding one no-birth tick makes newborn `2` broadcast on both traces. The target-`1` trace must not achieve that result one tick earlier. Its final exposures after the extra tick are `[1, 2, 1]`.

Changing only seed fitness to `[1, 3]` changes the one-tick newborn-broadcast probability to `1/4`; the conditional agent updates for the two chosen graphs remain the same. This demonstrates the effect of the actual fitness law on an agent outcome, not only a topology count.

### 9.2 Successive births use retained agent state

Start from the unit-fitness setup above. Tick one attaches `2` to `1`; tick two attaches new agent `3`, initially at belief zero, to `2`. The trace mass is `(1/2) * (1/4) = 1/8`.

After tick two, beliefs are `[1, 1, 1, 0]` and exposures are `[1, 2, 1, 0]`. Agent `3` must wait for a later propagation round to receive the signal that `2` acquired in tick two. Reinitializing older beliefs at the second birth must fail this test.

### 9.3 Kernel boundaries and validation

- A receiver with receptivity `1/2`, prior belief zero, and a single unit signal updates to belief `1/2` and broadcasts at threshold `1/2`.
- A zero-receptivity receiver preserves belief while its exposure count increases.
- With no broadcasting sources, the entire population is unchanged by propagation.
- Reversing the enumeration of vertices or incoming transmissions cannot change the mathematical result. This is enumeration independence; the raw seed array remains indexed by stable vertex ID and cannot be arbitrarily reordered.
- Reordering an ordered BB target tuple can change its mass. For the same selected target set, growth and the subsequent propagation result must be identical; the raw target array is never sorted as a repair.
- Empty joint replay returns the seed state, round zero, and mass one after validation.
- No-birth-only replay changes agent state as appropriate while its BB projection is the empty birth replay with mass one.
- Bad seed dimensions, invalid agent bounds, invalid `m`, invalid newborn fitness, wrong target count, repeated targets, out-of-range targets, and conflicting errors exercise the specified precedence.
- An invalid birth after no-birth ticks reports the correct distinct tick and birth indices and returns no partial result.

## 10. Module and verification boundaries

Proposed Lean modules:

- `NarrativeDynamics/Core/NetworkPropagation.lean`: rational agent contracts, synchronous propagation, and local invariants.
- `NarrativeDynamics/Core/FitnessABM.lean`: typed joint state, identity transport, growth, tick, and finite schedule composition.
- `NarrativeDynamics/Core/FitnessABMReplay.lean`: raw agent validation, checked schedule replay, errors, and agreement with BB replay.
- `NarrativeDynamics/Core/FitnessABMDistribution.lean`: evaluation over the existing target-trace carrier and probabilities of agent outcomes.

Tests mirror these responsibilities under `NarrativeDynamics/Tests/`. Public names use one BB model identity and reuse existing graph, probability, and ID machinery. Update `NarrativeDynamics.lean` and README only when the corresponding implementation is proved and tested.

Add a focused Python test module, `tests/test_network_abm_bb_conformance.py`, to compare the specified per-round scenarios with the existing public `simulate_round`. Each comparison supplies an explicit post-growth graph and snapshot under a valid fixed-roster V1 model. It compares beliefs, exposures, broadcasting, and transmission endpoints; it does not forge a changing-roster `PopulationTrajectory` or claim hash-lineage agreement across births. Use exactly representable fixture values where possible and state any numeric tolerance explicitly. This is finite example agreement, not a proof for all floating-point executions.

The maintained proof boundary remains the Lean kernel. Extend the existing source/log trust audit to the new public theorem surfaces and their actual dependency reports. Permit only the current logical axiom allowlist; no unproved declarations, native proof oracle, unsafe bypass, or unlimited resource settings.

Run the new theorem and concrete acceptance modules with explicit resource limits and elapsed-time/peak-memory reporting. Start with the existing 240-second fixture limit; a failure requires diagnosis before changing the limit. Keep symbolic invariant proofs separate from concrete replay reduction, and keep enumerated distribution examples small. Do not expand the finite trace space merely to produce a large demonstration.

Required implementation gates are the new Lean modules and theorem fixtures, exact joint probability examples, BB regression and trust audits, the focused Python comparisons, and the existing complete proof/Python CI on the implementation revision. World Studio is outside the change surface; any repository-mandated job still applies. A design-only commit under the existing ignored spec path does not provide implementation CI evidence.

## 11. Explicitly deferred extensions

- Production Python BB sampling, mutable-roster adapters, and changing-model hash lineage.
- V19 embodied population creation, database migration, or insertion of topology edges into perceptual evidence.
- Fitness changed by learning, belief, success, or other agent behavior.
- Removal of nodes or edges, endogenous rewiring, adaptive trust, and multiple channel types.
- Full semantic cognition, narrative planning, and agent-generated birth schedules.
- General temporal-path, consensus-convergence, or asymptotic network theorems.
- Claims that BB necessarily generates a small-world certificate, universal fixed-hop bound, or a measured real-world separation value.

The first follow-up after this foundation should be a separately specified runtime adapter if a live growing Python simulation is needed. Agent-dependent fitness or topology feedback changes the stochastic assumptions and requires its own model and proof boundary.

## 12. Design review and next handoff

This design records the approved direction of BB-driven network growth with persistent agent updates and proposes its first concrete proof scope. Its source references, acceptance arithmetic, and compatibility statements must be checked before commit.

After written design review, create an implementation plan covering the propagation kernel, joint growth, checked replay, joint outcome probabilities, and conformance/audit integration. The plan must identify an executable failing example for each new behavior before implementing it. No production or Lean proof implementation belongs to this design-only change.
