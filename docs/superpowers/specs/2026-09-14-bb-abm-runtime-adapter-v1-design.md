# BB-driven ABM Runtime Adapter V1 Design

Status: proposed design for review. No Python runtime implementation or new proof result is claimed by this document.

Foundation: [PR #73](https://github.com/qigao/lean/pull/73), pinned at `fb17c2b7d5ac721fdae90051832c9bb703a7c230`. Its six implementation tasks, independent reviews, and final CI are complete. This proposal follows section 11 of [BB-driven ABM Evolution V1](2026-09-14-bb-abm-evolution-v1-design.md) and gives the deferred Python adapter its own changing-population contract.

The design branch is based on the verified foundation commit. While #73 remains open, the design PR targets `feature/bb-abm-evolution-v1` so its review diff contains this proposal only. Integration of the foundation and authorization to implement this new design are separate decisions.

## 1. Outcome and first executable boundary

Build a pure Python finite replay that actually creates a fresh agent for each supplied legal BB birth, preserves every old agent's current state at growth, then invokes the existing synchronous V1 propagation rule once. An idle tick invokes one propagation round without a birth.

The first runtime accepts an authored sequence of ordered target choices. It computes the exact BB probability of that trace; it does not choose targets randomly. This is sufficient to execute and inspect the proved successive-birth and delayed-relay scenarios in Python. A seeded stochastic sampler can later call the same checked birth kernel, with its RNG ownership and replay contract specified separately.

Completion means an executable Python growth trajectory and actual finite agreement checks against the Lean reference. It does not mean that arbitrary Python floating-point executions have acquired a Lean proof.

## 2. Repository evidence and architectural choice

Source references below are pinned to the foundation commit.

| Existing source | Observed contract | Adapter requirement |
| --- | --- | --- |
| [contracts.py](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/narrative_dynamics/abm/contracts.py) | `NetworkABMModel` binds an exact roster and topology; agents and states are sorted by string ID | Keep a separate append-only numeric-ID registry; never recover BB IDs from sorted tuple positions |
| [simulation.py](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/narrative_dynamics/abm/simulation.py) | `NetworkRoundResult` requires the same model hash and roster before and after a round; `PopulationTrajectory` has one model hash and rejects an empty round list | Represent births outside a V1 round, and introduce a distinct joint replay result |
| [simulation.py:228](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/narrative_dynamics/abm/simulation.py#L228) | `simulate_round` computes all transmissions and updates from the prior snapshot | Reuse that function; do not add a second Python propagation rule |
| [contracts.py](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/narrative_dynamics/abm/contracts.py) | `initialize_population` assigns exposure one to explicitly seeded beliefs | Construct the validated birth-stage snapshot directly; a newborn must start with exposure zero |
| [evolving_contracts.py](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/narrative_dynamics/abm/evolving_contracts.py) | V5 membership is drawn from a fixed base catalog | A V5 entry event is not creation of a new BB identity |
| [FitnessAttachment.lean](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/NarrativeDynamics/Core/FitnessAttachment.lean) | Weight is fitness times undirected degree; an ordered tuple uses a frozen pre-birth graph and masks earlier choices | Implement one exact checked Python birth kernel with the same law, and compare its outputs with actual Lean execution |
| [FitnessValidation.lean](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/NarrativeDynamics/Core/FitnessValidation.lean) | Seed and birth errors have deterministic precedence; invalid requests return no successor | Preserve the shared semantic precedence and make Python-specific shape/identity errors explicit |
| [shared V1 comparisons](https://github.com/qigao/lean/blob/fb17c2b7d5ac721fdae90051832c9bb703a7c230/tests/test_network_abm_bb_conformance.py) | Existing comparisons create independent fixed-roster snapshots for eight rounds | Retain them and add full birth-trace comparisons; do not reinterpret the old corpus as a changing-roster trajectory |

Three approaches were considered:

1. **Separate BB replay wrapper around unchanged V1 rounds — selected.** Explicitly represents the growth stage and model boundary, while reusing current propagation.
2. Relax `NetworkRoundResult` or `PopulationTrajectory` to accept changed rosters. Rejected because their exact-model lineage is an existing public contract used elsewhere.
3. Launch Lean for each Python tick or integrate directly with V19. Deferred: the former requires a general runtime process/JSON protocol, and the latter requires coordinated world, perception, cognition, memory and persistence migrations.

The Python BB kernel is a new implementation checked against the existing Lean reference. It is not a foreign-function call to the proved kernel and must not be described as universally verified by conformance examples.

## 3. Identity, topology and numeric ownership

The BB numeric identity is an index in an append-only tuple `agent_ids`. Initial index `i` maps to the corresponding raw seed agent's external string ID. Birth appends one previously unused string ID at numeric index `n`. Sorting V1 models or snapshots cannot change this registry.

The authoritative topology contains the node count, positive immutable fitness values, and canonical undirected edges. Fitness and BB trace masses use `fractions.Fraction`. Accept exact integer or Fraction fitness values, rejecting booleans and implicit float-to-fitness conversion. Canonical serialized rationals use an integer string when the denominator is one, otherwise `numerator/denominator`.

Validate seed edge endpoints and detect duplicates, including reversed duplicates, before canonicalizing the accepted edge list. Never silently drop a duplicate. Derive degrees from undirected adjacency; the two V1 channels representing an edge must not double its BB degree. No separately mutable degree cache is introduced.

Generate exactly two active `SocialEdge` channels with influence `1.0` and one fixed relation type `"bb"` for each undirected edge. Generate no other channels.

Agent receptivity, broadcast threshold and belief use the existing finite Python numeric domain and V1 normalization. Keep every old `NetworkAgentSpec` field fixed, including role and adoption threshold. Role and adoption threshold are runtime metadata for this adapter; neither participates in BB selection or in the proved scalar propagation rule.

Explicit seed exposure counts are nonnegative integers. A newborn request contains no exposure-count override. Broadcasting is derived from the actual V1 belief and threshold, never accepted as an independent raw flag.

## 4. Runtime records and two clocks

Proposed frozen records are deliberately separate from `PopulationTrajectory`:

| Record | Authoritative contents and validation |
| --- | --- |
| `BBRuntimeSeed` | Raw BB seed, raw agent records in numeric-ID order, and run/model identity; all values remain untrusted until replay validation |
| `BBRuntimeTick` | Tagged idle, or one ordered target tuple plus newborn fitness, profile metadata and initial belief; idle cannot carry birth data |
| `BBRuntimeFrame` | Run configuration including fixed `m`, global tick/birth counts, numeric-ID registry, BB topology/fitness, current V1 model and population, cumulative exact trace mass, parent frame hash and applied tick |
| `BBRuntimeTransition` | Prior frame, validated tick, frozen post-growth V1 input, actual `NetworkRoundResult`, exact tick mass and resulting frame |
| `BBRuntimeReplay` | Validated initial frame, ordered transition tuple and final frame; permits zero transitions |
| `BBRuntimeError` | Stage, stable error code, relevant agent/field or tick/birth indices, and preserved underlying BB cause; no partial trajectory or probability |

There are two explicitly different clocks:

- Global `tick_count` increases once for every tick; `birth_count` increases only for births.
- Each V1 model epoch has its own ordinary `PopulationState.round_index`. A birth creates a new model epoch with a post-growth input at local round zero and parent `None`. Its propagation result is local round one. Idle ticks continue that model's local round chain.

The epoch number is derived from `birth_count`; it is not another independently mutable counter. Run/model ID and adapter schema version remain stable, while the exact V1 model content hash reflects its new roster and topology.

Resetting the local round at a model boundary does not reset beliefs, exposures or profiles. Cross-model ancestry belongs to the BB wrapper. Do not manufacture a V1 parent hash linking two different model epochs or reinterpret a same-roster V1 round as a birth.

Transmissions retain their existing local round indices. Their containing transition supplies global round `tick_index + 1`; consumers must not compare local round numbers across epochs as global time.

## 5. One birth or idle tick

For a birth with ordered targets `t_0, ..., t_(m-1)`, compute from the frozen pre-birth BB graph:

```text
w_i = fitness_i * undirected_degree_i
tick_mass = product over k:
    w_(t_k) / sum(w_j for j not in {t_0, ..., t_(k-1)})
```

Do not add edges, update degree, or include the newborn while calculating those factors. Retain the supplied target order for mass and trace identity even though the final edge set is unordered.

Execution order:

1. Validate the entire current request before constructing any successor.
2. For a birth, compute its exact mass and apply the checked graph extension. Append the ID, fitness and newborn profile. Copy each old agent's complete current state by external ID; do not call `initialize_population` to reseed it.
3. Construct the new model's local-round-zero post-growth population. The newborn receives exactly the supplied belief and exposure zero. Derive its broadcasting from its initial belief.
4. For an idle, reuse the current model and population as the propagation input, with exact tick mass one.
5. Call the existing `simulate_round(model, post_growth_population)` exactly once.
6. Construct the successor frame and transition, increment global counts, and multiply cumulative mass by the tick mass.

An initially broadcasting newborn can transmit immediately. A newborn or old receiver cannot relay a belief acquired during this same round. Agent updates, profile metadata, idle ticks and local clock resets introduce no extra probability factor.

After `T` ticks with `B` births, the node count is `n0 + B`, the undirected edge count is `E0 + m * B`, and global tick count is `T`.

## 6. Checked entry point and failure contract

The first public execution API is a finite, side-effect-free `replay_bb_population(seed, m, ticks)`. It returns a complete replay or one structured error. A private checked tick helper handles the sequential loop. Public resume/import, streaming callbacks and an API accepting caller-authored cumulative probability are deferred.

Validate on entry in this order:

1. BB seed: node count at least two; fitness dimension; strictly positive exact fitness; valid edge endpoints/no self-edges; duplicate edges; connectedness.
2. Initial `0 < m <= seed.nodeCount`, including empty or idle-only schedules.
3. Seed agent-record count.
4. Seed agents in numeric-ID order: receptivity, broadcast threshold, belief, then exposure count. Validate that record's external ID, role and adoption threshold afterwards; reject duplicate IDs at their second occurrence.
5. Ticks in authored order. At a birth, preserve BB order: valid `m`, positive exact newborn fitness, target count, bounds, distinctness; only then validate newborn receptivity, broadcast threshold, belief and runtime metadata.

Python-specific shape/type failures are checked at the field's position in that sequence, not by eagerly scanning the complete schedule. Reject booleans where integers/numerics are expected and reject nonfinite agent values. Do not eagerly construct validating V1 profiles before the BB-first checks. Shared semantic failures should map to the Lean causes; additional Python shape, exposure and external-ID failures are explicitly runtime-specific.

Tick indices and birth indices are zero-based and distinct. An invalid birth after two idles reports tick index two and birth index zero. No later malformed request can displace the first failure. Failure returns no successful prefix, partial topology, population, allocated ID or accumulated mass, and input records remain unchanged.

An empty replay still validates the seed, `m` and agents, then returns the initial frame with counts zero and mass one. It must not call `simulate_population(rounds=0)`, whose existing rejection remains correct.

## 7. Lineage and content hashes

Use the existing `stable_content_hash` on canonical JSON-compatible values, with an explicit `bb-abm-runtime-v1` schema discriminator. Convert Fractions to the canonical rational strings before hashing.

A genesis frame has no parent and no applied tick. Each later frame binds its immediate prior frame hash and its canonical applied tick. The applied birth retains target order and full newborn input, so distinct ordered histories remain distinguishable even when their final graph, agent state and probability coincide.

The frame hash covers configuration, registry, exact topology/fitness, the current model and population, counts and cumulative mass. The replay binds all transitions in authored order. The transition contains the post-growth input and actual V1 round so newborn initialization and state retention are inspectable.

Do not put a transition's hash into a frame that the same transition hashes; that would create a cycle. Never reuse an old model hash after changing its roster. Hashes identify content; they are not evidence that a transition was computed correctly.

Checked execution constructs results from the seed and request sequence, computes masses internally, and validates model/roster/registry consistency. A caller-supplied model hash, degree, normalizer, cumulative mass or successful-prefix claim is not trusted input. Importing serialized replay certificates and resuming an arbitrary frame are outside the first API.

## 8. Exact BB arithmetic and floating propagation

BB topology, positive fitness, ordered target masses and cumulative trace probability use exact arithmetic. The scalar agent update uses the existing V1 floating-point operator, including its clamping and threshold comparison.

Consequently:

- BB mass comparisons against Lean are exact.
- Existing dyadic fixtures compare beliefs, exposure counts, broadcasting and transmissions exactly.
- Non-dyadic and threshold-near Python tests establish consistency with `simulate_round`; they do not imply universal agreement with rational Lean decisions.
- Never hide a broadcasting/transmission disagreement behind a numeric tolerance, relabel a Python result as kernel-checked, or multiply BB mass by a floating behavioral factor.

The probability is conditional on the supplied birth calendar, fitness, profiles and initial beliefs. This first replay returns an individual ordered trace mass. It does not supply a probability for authoring that trace, sample targets, or implement a new event-distribution enumerator.

## 9. Acceptance scenarios

Unless a row overrides them, use seed edge `0--1`, numeric IDs mapped to string IDs `"0"` and `"1"`, unit fitness, `m=1`, receptivity one, broadcast/adoption thresholds `1/2`, role `"peer"`, initial beliefs `[1,0]` and zero exposures. Each newborn has unit fitness, the same profile and initial belief zero.

| Scenario | Required observation |
| --- | --- |
| Unit seed edge, one birth targeting source 0 | Exact mass `1/2`; beliefs `[1,1,1]`; exposures `[0,1,1]` |
| Same seed, birth targeting relay 1 | Exact mass `1/2`; beliefs `[1,1,0]`; exposures `[0,1,0]` |
| Relay branch followed by idle | Mass remains `1/2`; beliefs `[1,1,1]`; exposures `[1,2,1]`; newborn did not relay one tick early |
| Successive births to 1, then 2 | Exact mass `1/8`; beliefs `[1,1,1,0]`; exposures `[1,2,1,0]`; counts four nodes, three edges, two global ticks |
| Extra idle after those births | Agent 3 receives on global round three; prior states are retained |
| Seed fitness `[1,3]`, one birth targeting source | Exact trace mass `1/4`; the authored branch's scalar update remains the same |
| `m=2`, seed fitness `[1,3]`, targets `(0,1)` versus `(1,0)` | Masses `1/4` versus `3/4`; equal final graph/agent observations, distinct ordered history records |
| Zero receptivity, silent population, zero threshold, initially broadcasting newborn | Match the existing V1 cases; exposure and delayed-relay rules remain explicit |
| Empty and idle-only requests | Validate initial `m`; correct counts, unchanged BB graph/fitness and mass one |
| Numeric IDs crossing 9 to 10 | Registry remains numeric append order despite lexicographically sorted V1 IDs; state lookup cannot use tuple position |
| Model-boundary lineage | New birth epoch starts at local zero, propagation at local one, global time advances once; idle continues local ancestry |
| Conflicting invalid fields and late invalid birth | Shared validation precedence, correct two indices, first failure, no partial result or input mutation |
| Invalid duplicate/reversed edges and invalid target order data | Reject rather than deduplicate, truncate or sort as repair |
| Existing V1 contract regressions | Mixed-roster/model V1 rounds and empty `PopulationTrajectory` still reject exactly as before |

Use a separate Lean-generated `conformance/bb_abm_runtime_v1.json` for full birth traces, including each required prefix observation and exact mass, plus stable shared error cases. Its producer must evaluate actual `FitnessABM.replay` calls; it cannot copy Python outputs or hand-author the expected JSON. Compare the Lean replay round count with the wrapper's global tick count, and separately check the V1 epoch-local clocks. Preserve the existing eight-vector `bb_abm_v1.json` unchanged.

Tie new reference cases to kernel-checked literal assertions or the already proved concrete fixtures. Fresh exporter output must compare byte-for-byte with the committed corpus. Tests must include a deliberate wrong retained belief or wrong ordered mass that fails, followed by restoration and successful regeneration/comparison.

## 10. Proposed implementation units and sequence

The detailed implementation plan should pin this reviewed design before production edits.

| Unit | Responsibility |
| --- | --- |
| `narrative_dynamics/abm/bb_runtime_contracts.py` | Raw inputs, immutable result/error records, canonical identity and lineage serialization |
| `narrative_dynamics/abm/bb_runtime.py` | One checked exact BB birth kernel, V1 model-boundary projection, private tick loop and public finite replay |
| Focused runtime contract/replay tests | Invalid input matrix, identity, retained states, clocks, masses and failure atomicity |
| `NarrativeDynamics/Conformance/FitnessABMRuntimeVectors.lean` plus owning fixtures | Actual reference birth-trace observations and errors |
| New runtime corpus and its Python consumer | Compare full computed histories with reference data |
| A small deterministic Python example | Print actual births, beliefs, exposure counts and exact trace mass for the successive-birth scenario |

Suggested task order:

1. Establish failing contract tests for fresh IDs, epoch boundaries, empty replay and rejection of forged model/roster bindings; implement the records.
2. Establish failing exact birth/validation tests; implement the single BB kernel, including ordered `m=2` behavior.
3. Establish failing retained-state, newborn-zero, delayed-relay and idle tests; compose one actual V1 round after growth.
4. Establish failing atomic multi-tick replay and two-index tests; implement the finite entry point and lineage.
5. Generate actual Lean reference histories, complete conformance comparisons and the prescribed mutation/restoration check.
6. Add the executable example, document the verified numeric boundary, run required CI and review the complete implementation.

Keep existing V1/V5/V19 contracts, production persistence, all 35 foundation audit requirements, toolchain/dependencies and workflow selection/checkout behavior unchanged. Any new Lean fixture must retain bounded execution and the existing trust allowlist; do not disable old gates to admit a new exporter. The implementation plan must specify the new exporter build/run/audit/comparison wiring before code is added.

## 11. Verification and limits

Implementation acceptance requires actual current-revision Lean reference execution and trust checks, focused runtime tests, complete Python discovery and applicable World Studio CI. Observe exact checked revisions. A skipped or absent proof run for this design-only path provides no evidence for an unimplemented adapter.

This proposal was prepared from the pinned repository contracts. Local execution was unavailable because the selected environment's exec-server connection failed. No local test pass or new runtime behavior is claimed here.

Deferred: random target sampling and RNG replay, interactive partial-result delivery, public resume/import, node removal, fitness feedback, independently weighted channels, V19/database/World Studio adapters, UI integration, large-network performance claims, universal Python/Lean floating-point equivalence, temporal-path/consensus theorems and universal hop bounds.
