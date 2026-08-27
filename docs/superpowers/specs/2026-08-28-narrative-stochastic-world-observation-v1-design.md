# Narrative Stochastic World / Observation V1 Design

Status: ready for review

Date: 2026-08-28

Integrated base: `proof/narrative-dynamics-v0@e2b505230211a38390e8eda9e92ba25c347703aa`

Roadmap: #27, P2 — Stochastic World / Observation Models

## 1. Objective

Add explicit, seeded, replayable aleatoric stochasticity to the narrative runtime without weakening the deterministic trust boundaries established by World Transition V1/V2, Observation Projection V1, Runtime Percept Admission, Generic Runtime Decision Dispatch, or the multi-step scheduler.

This design completes only the five P2 stochastic-world/observation roadmap items:

- explicit RNG/seed boundary;
- seed included in transition/observation lineage;
- exact replay under fixed seed;
- separation of aleatoric world/observation stochasticity from decision stochasticity;
- compatibility with the existing batch/seed diagnostics infrastructure.

It does not complete the separate P2 Identification and Model Comparison workstream.

## 2. Design principles

### 2.1 Preserve deterministic V1 semantics exactly

Existing deterministic APIs remain valid and their semantic payloads remain unchanged when no stochastic configuration is present.

In particular:

- `ActionTransitionSpec.transition_hook(snapshot, decision, action)` remains unchanged;
- `ObserverProjectionSpec.projection_hook(prior_visible, next_visible, observer, step_index)` remains unchanged;
- `run_runtime_decision(...)` remains RNG-free;
- conflict resolution remains deterministic;
- deterministic `WorldState`, `ActionTransitionRecord`, `WorldStepResult`, `ProjectedObservation`, `ObservationProjectionResult`, runtime evidence, simulation state, and trajectory payloads omit all new optional stochastic fields;
- deterministic model hashes and result hashes must remain exact where the pre-P2 payload is unchanged.

The P2 implementation is an additive stochastic lane, not an in-place mutation of V1 hooks.

### 2.2 Distribution first, sampling second

Model hooks declare finite probability distributions. They never receive or own an RNG.

The trusted runtime:

1. validates the complete finite distribution;
2. attests the hook/spec identity;
3. derives a component-local deterministic random substream from the frozen root seed and canonical lineage;
4. samples exactly one outcome;
5. records the complete distribution identity and sampling lineage;
6. only then executes the existing world/projection validation path.

This keeps stochastic assumptions inspectable instead of hiding random draws inside arbitrary hook code.

### 2.3 Stateless component-local pseudorandomness

The runtime must not use one mutable RNG shared across actors, observers, or phases. Shared mutable RNG would make results depend on evaluation order and the number of draws performed by unrelated components.

V1 instead uses deterministic, stateless, versioned hash derivation. Each stochastic component gets its own derived stream key from canonical semantic inputs. A component consumes exactly one categorical draw in V1.

Therefore actor order, projection-spec order, and unrelated stochastic components cannot perturb one another's samples.

### 2.4 Truth-preserving stochastic observation

Observation stochasticity in V1 controls whether and which true post-step facts are emitted. It does not permit an `equals` observation to contradict the actual post-step world state.

This preserves the existing meaning of `ObservationFact` as certified world-derived evidence.

False-positive measurements, corrupted values, hallucinated readings, and sensor-likelihood evidence require a separate future measurement-evidence contract. They are explicitly out of scope here.

## 3. Randomness core

Add:

`narrative_dynamics/narrative/randomness.py`

This module owns seed validation, substream derivation, finite categorical sampling, and sampling-lineage records. World and observation modules consume this capability but do not implement their own random algorithms.

### 3.1 Root seed

A narrative simulation branch has one optional `root_seed`.

Rules:

- type must be `int`, excluding `bool`;
- `None` means deterministic/unseeded branch;
- any stochastic world or observation execution requires a non-`None` root seed;
- a branch cannot change root seed after initialization;
- root seed is an execution input, not part of `SimulationModelSpec` identity.

`WorldState` gains an optional root-seed field appended to the existing public record. Its serialized form is omitted when `None`, preserving deterministic V1 payloads and hashes.

Every next world state must inherit the exact root seed from its prior state.

### 3.2 Versioned derivation

Define a closed implementation constant such as:

`RANDOM_DERIVATION_VERSION = "narrative-hash-categorical-v1"`

For each sample, derive a stream key from a canonical payload containing:

- derivation version;
- root seed;
- namespace;
- step index;
- source hash;
- stochastic component/spec hash;
- semantic component key.

Canonical namespaces in this V1 are:

- `world.transition`;
- `observation.projection`.

World component key:

`(actor_id, decision_id, action_id)`

Observation component key:

`(observer_id, channel)`

World source hash is the exact prior `WorldState.content_hash`.

Observation source hash is the exact `WorldStepResult.content_hash`.

The stream key is derived with the repository's canonical content-hash machinery. A second versioned hash of the stream key and draw index `0` yields an unsigned 64-bit draw integer. The categorical sampler compares `draw_u64 / 2**64` against the canonical cumulative probabilities, with the final outcome acting as the closed upper interval.

The implementation must not depend on Python's process-global RNG or shared mutable RNG state.

### 3.3 `RandomSampleRecord`

Introduce one frozen, self-validating sampling record containing at least:

- derivation version;
- root seed;
- namespace;
- step index;
- source hash;
- component hash;
- canonical component key;
- derived stream key/hash;
- draw index (`0` in V1);
- `draw_u64`;
- complete distribution hash;
- selected outcome id;
- selected outcome hash.

The constructor must recompute the stream derivation and draw from its declared inputs and reject forged lineage.

The domain-specific world/observation records additionally validate that the selected outcome exists in the declared distribution and that the selected outcome hash matches the resulting state delta or observation-fact tuple.

## 4. Finite distribution contract

All stochastic distributions are finite and explicit.

For every distribution:

- at least two outcomes are required;
- outcome ids are non-empty trimmed strings and unique;
- probabilities are finite numeric values, not bools;
- each probability is strictly greater than zero and at most one;
- canonical probability mass must sum to exactly `1.0` under the implementation's canonical summation routine;
- outcomes are canonicalized lexically by outcome id;
- input order must not affect distribution hash or sampling result;
- duplicate semantic outcome ids fail closed;
- distribution payloads are immutable and content-hashed.

V1 does not silently renormalize malformed distributions.

## 5. Stochastic world transition lane

### 5.1 New records

Add domain-specific records in `world.py`:

- `StateDeltaOutcome(outcome_id, probability, delta)`;
- `StateDeltaDistribution(outcomes)`;
- `StochasticTransitionSample(distribution, sample_record)`;
- `StochasticActionTransitionSpec(action_type, effects, parameters, distribution_hook)`.

`parameters` is a frozen canonical mapping of finite numeric values. It exists so stochastic-model assumptions such as failure probabilities are explicit model identity rather than hidden mutable hook instance state.

The distribution hook signature is additive and separate from deterministic hooks:

`distribution_hook(snapshot, decision, action, parameters) -> StateDeltaDistribution`

It receives no RNG.

### 5.2 Model integration

`WorldTransitionModelSpec` gains an optional `stochastic_transitions` tuple appended after existing fields so current positional construction remains compatible.

Rules:

- deterministic and stochastic transition action types form one uniqueness namespace;
- one action type cannot be declared in both lanes;
- stochastic transition specs bind domain/effect capability, explicit parameters, distribution-hook implementation identity, and randomness derivation version into model identity;
- when `stochastic_transitions` is empty, the field is omitted from serialized model identity so deterministic V1 model hashes remain exact.

### 5.3 Execution order

For a stochastic selected action:

1. perform the same narrative/domain/prior/intent/spec/effect-capability preflight used by deterministic execution;
2. require a root seed before calling the stochastic distribution hook;
3. snapshot the same immutable prior world state used by all actions in the simultaneous batch;
4. call the distribution hook;
5. validate every candidate `StateDelta` against the action's declared effect capability before sampling;
6. derive the component-local world sample;
7. select one outcome;
8. create the `ActionTransitionRecord` using the sampled delta and attach the self-validating `StochasticTransitionSample`;
9. continue through the existing conflict-component, atomic-resolution, collision, and next-state logic.

This order ensures invalid unsampled outcomes cannot hide outside capability checks.

### 5.4 `ActionTransitionRecord` compatibility

Append an optional stochastic sample field.

For deterministic records:

- field is `None`;
- field is omitted from `to_dict()`;
- hash remains exactly V1-compatible.

For stochastic records:

- record delta must equal the selected distribution outcome delta;
- sample root seed must equal the prior state's root seed;
- sample source hash must equal exact prior state hash;
- sample component hash must equal exact stochastic transition spec hash;
- sample component key must equal actor/decision/action identity;
- transition batch hash therefore binds seed, distribution, sample, selected outcome, and sampled delta automatically through the transition-record payload.

### 5.5 Conflict resolution

Conflict Resolution V2 is not made stochastic in this workstream.

The sequence is:

`sample each stochastic transition against the shared prior snapshot -> obtain concrete transition records -> derive actual write-set conflict components -> apply existing deterministic conflict resolver`

This is deliberate. Conflict detection must operate on the realized writes, not the union of every possible stochastic outcome.

The resolver receives the same concrete participant transition records it receives today. Its public hook/API does not gain an RNG.

## 6. Stochastic observation projection lane

### 6.1 New records

Add in `observation_projection.py`:

- `ObservationOutcome(outcome_id, probability, facts)`;
- `ObservationOutcomeDistribution(outcomes)`;
- `StochasticObservationSample(observer_id, channel, projection_spec_hash, distribution, sample_record)`;
- `StochasticObserverProjectionSpec(observer_type, channel, read_capabilities, emit_capabilities, parameters, distribution_hook)`.

The stochastic projection hook signature is:

`distribution_hook(prior_visible, next_visible, observer, step_index, parameters) -> ObservationOutcomeDistribution`

It receives the same capability-limited immutable world views as the deterministic projection hook and no RNG.

An observation outcome contains an exact tuple of `ObservationFact` values. The empty tuple is valid, enabling stochastic dropout/blackout.

### 6.2 Model integration

`ObservationProjectionModelSpec` gains optional `stochastic_projections` after the existing deterministic tuple.

Rules:

- `(observer_type, channel)` is unique across both deterministic and stochastic projection lanes;
- stochastic spec identity binds capabilities, channel, explicit parameters, distribution-hook implementation identity, and randomness derivation version;
- when stochastic projections are absent, no new serialized field is emitted and V1 deterministic model hashes remain exact.

### 6.3 Execution order

For every stochastic observer/spec pair:

1. perform the same source-world and domain validation as deterministic projection;
2. require the world branch root seed before the stochastic hook runs;
3. build exact immutable read-capability views;
4. call the distribution hook;
5. validate every outcome's facts for record shape, canonical cells, emit capability, and post-step truth semantics before sampling;
6. derive one observer/channel-local observation sample from the source world-step hash;
7. select one outcome;
8. emit the selected facts as normal `ProjectedObservation` records;
9. append the stochastic sample record to the enclosing projection result even when the selected fact tuple is empty.

### 6.4 Truth semantics

Every candidate outcome is subject to the existing acceptance rules before sampling:

- `equals` must equal the exact post-step typed world value;
- `clear` requires post-step absence plus one explicit effective current-step clear;
- facts must remain inside emit capability;
- capability-limited hooks cannot see objective cells they are not allowed to read.

Stochasticity may therefore model visibility, missingness, dropout, or selection among simultaneously true facts.

It may not emit a false world value in V1.

### 6.5 Projection lineage

`ObservationProjectionResult` gains a tuple of stochastic sample records, omitted when empty.

Each `ProjectedObservation` emitted by a stochastic sample gains an optional sampling-record hash, omitted for deterministic observations.

The result constructor validates:

- every stochastic sample binds the exact result/world-step/step/spec/observer/channel identities;
- sampling-record hashes referenced by observations exist in the result;
- an emitted stochastic observation belongs to the selected outcome fact set;
- deterministic observations do not claim stochastic sample lineage;
- a sampled empty outcome is still represented by the result's stochastic sample tuple.

## 7. Runtime percept and evidence lineage

`runtime_perception.py` remains responsible for admitting already validated projected observations; cognition semantics do not change.

Narrow additive changes:

- `RuntimeEpistemicEvidence` may carry an optional projection sampling-record hash copied from its `ProjectedObservation`;
- `RuntimeEvidenceBatch` carries the canonical tuple of projection sampling-record hashes for the complete projection result, including stochastic samples that emitted zero evidence;
- both fields are omitted when empty/`None`, preserving deterministic V1 payloads;
- admission verifies the batch's sampling hashes exactly equal those in the projection result.

This ensures stochastic blackout remains inspectable in the evidence ledger instead of disappearing merely because no percept was admitted.

The ledger's existing projection-result hash and world-state hash chain remain authoritative.

## 8. Scheduler seed plumbing and stochasticity separation

### 8.1 Initialization

Extend:

`simulation_state_from_story(..., at_time=None, seed=None)`

The seed is validated once and passed to both initial world-state construction and runtime-evidence-ledger construction so they bind the same initial world hash.

`runtime_evidence_ledger_from_story(..., at_time=None, seed=None)` gains the same optional seed only to reconstruct the exact seeded initial world state.

### 8.2 Step execution

`simulate_step(...)` does not accept a new seed.

It derives all stochastic execution from `prior_state.world_state.root_seed`. A caller therefore cannot silently change the random branch at step 2.

`simulate_trajectory(...)` also does not accept per-round seeds.

Same initial state + same model + same root seed must replay to the exact same:

- action transition records;
- stochastic samples;
- world-step hashes;
- world-state hashes;
- projection samples;
- projected observations;
- evidence batches/ledger hashes;
- simulation-step hashes;
- final trajectory hash.

### 8.3 Decision stochasticity remains separate

`run_runtime_decision(...)` and Reactive/Intentional/Planning public APIs remain RNG-free.

Current decision models continue to emit policies and deterministic lexical/MAP selections as before.

The P2 seed must not affect a decision result unless stochastic world/observation history has first changed the model-visible evidence/state.

If behavioral action sampling is later introduced, it must use a distinct future namespace such as `decision.choice` and its own explicit contract. It must never consume the world or observation stream.

## 9. Failure semantics

All new failures are fail-closed and preserve prior immutable state.

A stochastic operation rejects before returning a successful next state/result if any of the following occur:

- stochastic spec exists but branch seed is missing;
- malformed seed;
- duplicate deterministic/stochastic action type or observer/channel declaration;
- invalid/non-finite/unnormalized distribution;
- duplicate outcome id;
- any candidate world delta exceeds action capability;
- any candidate observation fact exceeds read/emit/truth boundaries;
- distribution hook raises;
- hook implementation attestation is unavailable;
- forged stream derivation/draw/sample record;
- selected outcome does not match recorded world delta/facts;
- stochastic sample lineage does not bind exact prior world step/spec/component;
- root seed changes across world-state chain;
- projection/evidence sampling hashes disagree.

No partial world mutation, partial evidence-ledger append, or hidden RNG advancement is observable on failure.

Because sampling is stateless and component-local, failed components do not consume random state that could perturb retries or siblings.

## 10. Deterministic compatibility requirements

The implementation must add explicit regression locks showing that a no-stochastic-configuration execution remains V1-exact.

Required compatibility assertions include:

- existing deterministic `WorldTransitionModelSpec.to_dict()` shape unchanged;
- existing deterministic model content hash unchanged for the same model;
- `WorldState.to_dict()` unchanged when seed is `None`;
- deterministic `ActionTransitionRecord.to_dict()` unchanged;
- deterministic world transition batch hash unchanged;
- deterministic `ObservationProjectionModelSpec.to_dict()` unchanged;
- deterministic `ProjectedObservation` and `ObservationProjectionResult` payloads unchanged;
- deterministic runtime evidence batch/ledger payloads unchanged;
- deterministic simulation state/step/trajectory payloads unchanged;
- all existing exact replay and conflict-resolution tests remain green without edits that weaken their assertions.

New optional fields must be omitted, not serialized as `null` or empty arrays, wherever omission is required to preserve V1 hashes.

## 11. Testing strategy

The implementation must follow strict test-only RED -> exact-head RED evidence -> minimal GREEN -> exact-head GREEN.

### 11.1 Randomness core tests

Lock:

- seed type validation;
- canonical stream derivation;
- same derivation inputs -> exact same stream/draw/sample;
- one changed derivation input -> changed stream identity;
- component input order does not affect canonical result;
- malformed/forged `RandomSampleRecord` rejects;
- distribution input order does not affect hash or selection;
- malformed probabilities/outcomes reject before sampling.

### 11.2 World stochastic tests

Lock:

- same seed exact replay;
- multiple seeds produce declared outcome variation over a controlled fixture;
- actor input order does not change per-actor samples or result hash;
- adding an unrelated stochastic actor does not perturb an existing actor's sample;
- every candidate delta is capability-validated before sampling;
- stochastic action without seed fails before hook;
- sampled transition lineage binds root seed/distribution/spec/prior/actor/decision/action;
- realized stochastic write sets feed the existing conflict resolver correctly;
- deterministic conflict behavior remains exact.

### 11.3 Observation stochastic tests

Lock:

- same seed/world-step exact projection replay;
- observer/spec order does not alter samples;
- one observer's existence does not perturb another observer's sample;
- controlled seed set exercises emit vs dropout outcomes;
- stochastic empty outcome still leaves sampling lineage in projection result and runtime evidence batch;
- every candidate fact is validated before sampling;
- false `equals` candidates reject even if their probability is tiny or they would not be selected;
- stochastic projection without seed fails before hook;
- deterministic projection payload/hash remains exact.

### 11.4 Scheduler trajectory tests

Lock:

- same initial seeded state + model -> exact multi-round trajectory hash;
- a different root seed changes stochastic lineage and can change realized state/observations;
- a different root seed must change stochastic lineage even if the extensional sampled outcome happens to be identical;
- step-level API cannot replace the root seed;
- decisions remain seed-invariant until stochastic world/observation evidence diverges;
- world and observation streams are independent namespaces.

### 11.5 Existing uncertainty/batch compatibility

Use the existing outer `SimulationRunner` and seed-block diagnostics without changing their core APIs.

A narrow test adapter may map the outer runner's seeded `random.Random` deterministically to one narrative root seed (for example one fixed-width `getrandbits` call before narrative initialization). The outer experiment manifest continues to record the external simulation seed; the narrative result records the derived narrative root seed and full inner sample lineage.

Tests must show existing seed-block variation machinery can observe seed-dependent aleatoric outcomes without any changes to calibration/uncertainty core.

No production comparison or calibration API changes are required by this workstream.

## 12. Production scope

Expected production files:

- new `narrative_dynamics/narrative/randomness.py`;
- modify `narrative_dynamics/narrative/world.py`;
- modify `narrative_dynamics/narrative/observation_projection.py`;
- narrow additive changes to `narrative_dynamics/narrative/runtime_perception.py`;
- narrow seed plumbing in `narrative_dynamics/narrative/simulation.py`.

A package-local export may be added only if existing narrative package conventions require it. No package-root export is required by default.

Expected tests are new focused stochastic world/observation/scheduler tests plus deterministic compatibility assertions. Existing tests should not be weakened.

## 13. Explicit non-goals

This V1 does not add:

- RNG arguments to deterministic world hooks;
- RNG arguments to deterministic projection hooks;
- RNG to Reactive/Intentional/Planning decision APIs;
- sampled behavioral action choice;
- stochastic conflict resolution;
- continuous distributions;
- unbounded or lazy distributions;
- false sensor values or measurement-likelihood evidence;
- changes to planning-model transition/observation probability semantics (those remain decision-model assumptions, not realized world noise);
- changes to comparison/preregistration/release protocols;
- new empirical fixtures;
- changes to existing held-out comparison conclusions;
- P2 intentional-vs-reactive or intentional-vs-POMDP identification claims;
- parameter recovery or non-identifiability work;
- Lean changes.

## 14. Roadmap completion criteria

The P2 Stochastic World / Observation Models workstream can be marked complete only after merge and post-merge exact-head proof demonstrates all of the following:

1. **Explicit RNG/seed boundary** — one frozen narrative root seed and runtime-owned component-local derivation.
2. **Seed lineage** — every stochastic world transition and observation sample is provenance-bound to root seed, source, component/spec, distribution, and selected outcome.
3. **Replay exactness** — same seeded initial state/model replays to an identical multi-round trajectory hash.
4. **Aleatoric separation** — world and observation stochasticity use separate namespaces and decision APIs remain RNG-free.
5. **Batch diagnostics compatibility** — existing outer seed/uncertainty machinery can exercise and detect narrative aleatoric variation without core uncertainty changes.
6. Deterministic V1/V2 payloads and hashes remain exact when stochastic configuration is absent.
7. All existing Lean/Python/story/testimony proof gates remain green.

Issue #27 remains open after this workstream because P2 Identification and Model Comparison remains separate and unfinished.
