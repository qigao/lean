# Narrative Runtime Percept Admission V1 Design

## Status

Approved architectural direction for the next P0 increment after Observation Projection V1.

Exact base: `2a45278bfaee75c69cdafc8c7abd9516cd41c1c7` on `proof/narrative-dynamics-v0`.

Feature branch: `work/narrative-runtime-percept-admission-v1`.

Target bridge:

```text
WorldStepResult
  -> ObservationProjectionResult
  -> Runtime percept admission
  -> Runtime evidence ledger
  -> Runtime epistemic state
  -> Runtime uncertain belief state
```

This phase does not add the multi-step scheduler or runtime action selection.

## Core semantic decision

Runtime cognition inherits authored cognition exactly at the simulation branch point:

```text
B_runtime,0^i = B_authored^i(source_at_time)
```

Then only admitted runtime percepts update cognition:

```text
B_runtime,k+1^i = Update(B_runtime,k^i, E_runtime,k+1^i)
```

The clocks remain distinct:

```text
authored: logical_time
runtime:  step_index
```

No V1 API maps one clock into the other.

## Architecture

Use two sidecar modules and leave authored replay records unchanged.

### `runtime_perception.py`

Responsibilities:

- initialize an immutable runtime evidence ledger from one authored branch point;
- call `project_world_observations()` internally;
- admit only engine-produced projected observations;
- append one evidence batch per runtime world step, including blackout steps;
- enforce ledger/world continuity;
- preserve exact world/projection provenance;
- never update belief directly.

### `runtime_cognition.py`

Responsibilities:

- recompute authored epistemic and uncertain-belief seeds at `source_at_time`;
- replay only admitted runtime evidence after the seed;
- expose deterministic runtime epistemic state;
- expose runtime finite Bayesian belief updates through a separate likelihood model;
- never read objective `WorldState` or `WorldStepResult` during cognition materialization.

The future scheduler consumes these outputs. Existing authored `run_intentional_decision()` remains unchanged in this V1.

## Rejected alternatives

### Extend `EpistemicEvidence` with runtime time

Rejected because existing authored replay, sorting, identity, and hashes are already defined around `logical_time`.

### Synthesize logical time from `step_index`

Rejected because simulated percepts would masquerade as authored evidence.

### Mutate `GenericNarrative.observations`

Rejected because runtime simulation must not mutate canonical authored narrative.

### Let Observation Projection update belief directly

Rejected because world visibility and cognitive interpretation are different mechanisms and should remain independently testable.

## Runtime Perception records

### `RuntimeEpistemicEvidence`

One admitted runtime percept with full provenance.

Fields:

```text
observer_id
channel
cell
relation: equals | clear
value
step_index
projected_observation_hash
projection_result_hash
projection_model_hash
source_world_state_hash
source_world_step_hash
source_transition_hashes
projection_spec_hash
```

Rules:

- `equals` requires one `TypedValue`;
- `clear` requires `value is None`;
- `step_index` is runtime time only;
- provenance hashes are canonical SHA-256 content hashes;
- transition hashes are unique and sorted;
- all lineage fields are engine-derived by admission;
- direct construction is data, not independent certification.

### `RuntimePerceptView`

A provenance-free semantic view passed to runtime cognition hooks.

Fields:

```text
observer_id
channel
cell
relation
value
step_index
```

This record deliberately omits:

```text
projected_observation_hash
projection_result_hash
projection_model_hash
source_world_state_hash
source_world_step_hash
source_transition_hashes
projection_spec_hash
```

Reason: a likelihood hook must not infer hidden objective branch identity from provenance hashes. Runtime model behavior may depend on what was perceived, its channel, and runtime step; it may not depend on opaque world/projection lineage identifiers.

`RuntimeEpistemicEvidence.percept_view` is derived from the semantic fields above.

### `RuntimeEvidenceBatch`

One admitted runtime step, even when no percept is emitted.

Fields:

```text
prior_ledger_hash
step_index
source_prior_world_state_hash
source_world_state_hash
source_world_step_hash
projection_model_hash
projection_result_hash
evidence: tuple[RuntimeEpistemicEvidence, ...]
```

Invariants:

- evidence order is canonical by `(observer_id, channel, cell)`;
- Observation Projection already guarantees uniqueness for one `(observer, channel, cell)` key, so provenance hashes are not used as semantic ordering keys;
- all evidence binds the batch step/world/projection identities exactly;
- multiple channels may observe the same cell;
- blackout is `evidence == ()`, not absence of a batch;
- the batch binds the exact prior ledger hash.

### `RuntimeEvidenceLedger`

Immutable simulation-level evidence history for all agents.

Fields:

```text
domain_id
domain_version
domain_spec_hash
source_story_hash
source_at_time
initial_world_state_hash
current_world_state_hash
batches
```

Derived:

```text
current_step_index = 0 if batches == () else batches[-1].step_index
```

Required continuity:

1. first batch step is `1`;
2. later batches increment by exactly one;
3. first batch prior-world hash equals `initial_world_state_hash`;
4. each later prior-world hash equals the previous next-world hash;
5. `current_world_state_hash` equals the latest next-world hash, or initial hash when empty;
6. each batch `prior_ledger_hash` equals the stable hash of the exact ledger prefix before it.

The ledger constructor can verify internal chain consistency, but a directly constructed ledger is still data rather than proof that historic world/projection execution occurred.

### `RuntimePerceptAdmissionResult`

Fields:

```text
prior_ledger_hash
projection_result: ObservationProjectionResult
evidence_batch: RuntimeEvidenceBatch
next_ledger: RuntimeEvidenceLedger
```

The result verifies that the batch is the exact final extension represented by `next_ledger`, and that batch projection/world identities match the embedded projection result.

## Runtime Perception APIs

### `runtime_evidence_ledger_from_story`

```text
runtime_evidence_ledger_from_story(
    story,
    domain,
    *,
    at_time=None,
) -> RuntimeEvidenceLedger
```

Behavior:

1. validate story/domain;
2. call `world_state_from_story(..., at_time=at_time)`;
3. return an empty ledger bound to exact domain, story, cutoff, and initial world-state hash;
4. create no evidence batch and no belief state.

### `admit_world_percepts`

```text
admit_world_percepts(
    story,
    domain,
    world_step,
    projection_model,
    prior_ledger,
) -> RuntimePerceptAdmissionResult
```

Pre-projection checks:

- ledger domain/story identity matches exactly;
- `prior_ledger.current_world_state_hash == world_step.prior_state.content_hash`;
- ledger current step equals `world_step.prior_state.step_index`;
- ledger/world `source_at_time` match exactly.

Then:

1. call `project_world_observations()` internally;
2. convert only that engine-produced projection result to runtime evidence;
3. create exactly one batch for the next world step;
4. append immutably;
5. return projection result, batch, and next ledger.

Consequences:

- no API admits a naked caller-supplied `ProjectedObservation`;
- the same world step cannot be admitted twice to the same ledger;
- skipped or reordered world steps reject;
- an old immutable ledger may be reused to form a counterfactual branch;
- projection model may vary by step, with each batch retaining its exact model hash.

Operational failures use `RuntimePerceptAdmissionError`.

## Blackout semantics

A projection result with no observations still appends an empty batch.

```text
step 2 existed, no percept was received
```

must remain distinguishable from:

```text
step 2 never occurred
```

A blackout batch advances ledger/world/step provenance but does not change deterministic epistemic content or uncertain posterior content.

## Runtime deterministic epistemic replay

### Authored seed

For agent `i`:

```text
seed_state = epistemic_state(
    story,
    domain,
    i,
    at_time=ledger.source_at_time,
)
```

The seed is recomputed from canonical story/domain state each time; no mutable caller-owned belief snapshot is trusted.

### `RuntimeEpistemicCellView`

Fields:

```text
cell
status: resolved | unknown | conflicted
resolved_value
constraints
basis: authored_seed | runtime_perception
supporting_runtime_evidence_hashes
last_runtime_step
```

Semantics:

- no runtime percept for a cell => preserve authored seed status/value/constraints;
- runtime percept is direct perception and supersedes stale authored content for that cell;
- latest runtime `equals` => resolved to observed value, stale authored constraints cleared;
- latest runtime `clear` => unknown, no resolved value, stale authored constraints cleared;
- if several channels perceive the same cell in the same latest step, all are truth-compatible because projection was validated against one next world state; retain all evidence hashes as support;
- runtime V1 has no testimony and no runtime `not_equals` relation.

### `RuntimeEpistemicState`

Fields:

```text
agent_id
source_at_time
step_index
ledger_hash
seed_state: EpistemicState
runtime_evidence_history
cells
resolved_values
```

Per-agent evidence order is canonical by:

```text
(step_index, channel, cell)
```

The state contains only evidence where `observer_id == agent_id`.

### `runtime_epistemic_state`

```text
runtime_epistemic_state(
    story,
    domain,
    agent_id,
    ledger,
) -> RuntimeEpistemicState
```

It accepts no objective world-state argument.

Operational failures use `RuntimeEpistemicResolutionError`.

## Runtime uncertain belief replay

### Separate model identity

Existing `UncertainBeliefModelSpec` continues to define authored seed belief.

Runtime evidence uses:

```text
RuntimeBeliefModelSpec(
    model_id,
    version,
    seed_model,
    runtime_parameters,
    runtime_likelihood_hook,
)
```

Identity binds:

- model id/version;
- exact `seed_model.content_hash`;
- canonical runtime parameters;
- measured implementation identity of the runtime likelihood hook.

### Runtime likelihood hook boundary

The hook signature is:

```text
runtime_likelihood_hook(
    agent_id,
    percept_view: RuntimePerceptView,
    hypotheses,
    runtime_parameters,
) -> Mapping[hypothesis_hash, likelihood]
```

The hook does **not** receive full `RuntimeEpistemicEvidence`.

Therefore it cannot inspect world-state hashes, world-step hashes, projection hashes, transition hashes, or projection-spec hashes.

This is the runtime analogue of capability-filtered Observation Projection: provenance is retained for audit, but hidden lineage is not model-visible cognition input.

### `RuntimeBeliefUpdateStep`

Fields:

```text
evidence: RuntimeEpistemicEvidence
prior: BeliefDistribution
likelihoods: tuple[BeliefLikelihood, ...]
posterior: BeliefDistribution
```

The stored update keeps full provenance evidence for audit, while the likelihood hook sees only `evidence.percept_view`.

Rules:

- evidence/prior/posterior describe one cell;
- finite hypothesis sets match exactly;
- likelihoods are finite probabilities in `[0,1]`;
- zero posterior mass rejects;
- posterior uses the existing canonical `posterior_distribution` mechanism.

### `RuntimeUncertainBeliefCellView`

Fields:

```text
cell
seed_posterior
posterior
updates
```

The first runtime update prior equals the authored seed posterior. Later updates form an exact chain. With no runtime evidence, posterior equals seed posterior.

### `RuntimeUncertainBeliefState`

Fields:

```text
agent_id
model_id
model_hash
source_at_time
step_index
ledger_hash
seed_belief_state: UncertainBeliefState
runtime_evidence_history
cells
```

The full authored seed belief state is retained for inspectability.

### `runtime_uncertain_belief_state`

```text
runtime_uncertain_belief_state(
    story,
    domain,
    agent_id,
    ledger,
    model,
    tracked_cells,
) -> RuntimeUncertainBeliefState
```

Behavior:

1. validate story/domain/agent/ledger/model/tracked cells;
2. compute the authored seed with existing `uncertain_epistemic_state()` at `ledger.source_at_time` using `model.seed_model`;
3. filter runtime evidence to the selected agent and tracked cell;
4. call runtime likelihood hook only with provenance-free `RuntimePerceptView`;
5. validate likelihood vectors and posterior continuity;
6. return a state at ledger current step.

It accepts no objective world-state argument.

Operational failures use `RuntimeBeliefResolutionError`.

## Clear evidence in uncertain belief

Runtime `clear` means the projected cell is absent after the step and Observation Projection verified an explicit current-step clear transition.

V1 does not invent an `absent` hypothesis.

`clear` is passed as a semantic relation in `RuntimePerceptView`; the runtime likelihood hook specifies how that observation changes probability over the existing finite hypotheses.

## No-objective-leakage invariant

There are two different equality notions and V1 must not confuse them.

### Cognitive semantic equality

If two branches give agent `i` the same authored seed and the same semantic runtime percept sequence:

```text
(step_index, channel, cell, relation, value)
```

then the agent's:

- deterministic cell statuses/resolved values/constraints;
- uncertain posterior distributions;
- likelihood-driven cognitive semantics

must be identical.

### Artifact/provenance identity

The full runtime state artifacts may have different content hashes when the underlying world/projection lineage differs, because `ledger_hash` and stored `RuntimeEpistemicEvidence` intentionally preserve provenance.

Thus:

```text
same percept semantics -> same cognition semantics
```

but not necessarily:

```text
same percept semantics -> same provenance artifact hash
```

This distinction is required for both scientific non-leakage and auditability.

The runtime likelihood hook sees only percept semantics, so provenance differences cannot alter posterior behavior.

## Multi-agent isolation

The ledger is simulation-wide; cognition is agent-specific.

```text
RuntimeEvidenceLedger
  -> filter observer_id == i
  -> RuntimeEpistemicState_i
  -> RuntimeUncertainBeliefState_i
```

Agent A's evidence cannot enter agent B's cognition in V1. Communication/testimony requires a future explicit mechanism.

## Determinism

Canonical ordering is semantic, not provenance-driven:

```text
batch evidence:
  observer_id, channel, cell

per-agent replay:
  step_index, channel, cell

tracked cells:
  entity_type, entity_id, state_variable
```

Equivalent caller insertion order cannot change cognitive semantics or stable artifact hashes within one fixed provenance branch.

## Trust boundary

`admit_world_percepts()` certifies the current append operation by executing existing Observation Projection against the supplied world step and by deriving runtime evidence itself.

Public records remain ordinary deterministic data records. Direct construction does not prove historic execution.

Future Scheduler V1 can retain complete world-step plus admission artifacts for stronger end-to-end trajectory revalidation without changing these V1 records.

## Public errors

Constructors use ordinary `TypeError` / `ValueError`.

Runtime entrypoints use:

```text
RuntimePerceptAdmissionError
RuntimeEpistemicResolutionError
RuntimeBeliefResolutionError
```

Admission rejects ledger/world continuity errors before projection execution.

Cognition rejects story/domain/agent/ledger/model/tracked-cell errors before any runtime likelihood hook call.

## Proposed narrative-scoped public API

```text
RuntimeEpistemicEvidence
RuntimePerceptView
RuntimeEvidenceBatch
RuntimeEvidenceLedger
RuntimePerceptAdmissionResult
RuntimePerceptAdmissionError
runtime_evidence_ledger_from_story
admit_world_percepts

RuntimeEpistemicCellView
RuntimeEpistemicState
RuntimeEpistemicResolutionError
RuntimeBeliefModelSpec
RuntimeBeliefUpdateStep
RuntimeUncertainBeliefCellView
RuntimeUncertainBeliefState
RuntimeBeliefResolutionError
runtime_epistemic_state
runtime_uncertain_belief_state
```

These names are exported only from `narrative_dynamics.narrative`, never root `narrative_dynamics`.

Existing authored `Observation`, `EpistemicEvidence`, `EpistemicState`, `UncertainBeliefState`, `UncertainBeliefModelSpec`, and authored intention APIs remain unchanged.

## Required V1 invariants

1. empty ledger binds exact domain/story/cutoff;
2. initial ledger hash binds `world_state_from_story` at the cutoff;
3. first admitted batch is step 1;
4. later batches are strictly consecutive;
5. world-state hashes chain exactly;
6. each batch binds the exact prior ledger prefix hash;
7. blackout creates an explicit empty batch;
8. blackout advances step/world lineage without changing cognition semantics;
9. duplicate admission of the same step rejects;
10. skipped or reordered world steps reject;
11. source/domain/story/cutoff mismatches reject before projection;
12. admission executes projection internally and accepts no naked projected observation;
13. runtime evidence binds exact projected observation/projection/world/transition lineage;
14. evidence/ledger identity is insertion-order invariant;
15. public record construction is data, not certification;
16. runtime epistemic seed equals authored `epistemic_state(..., at_time=source_at_time)`;
17. runtime uncertain seed equals authored `uncertain_epistemic_state(..., at_time=source_at_time)`;
18. authored records/replay/hashes remain unchanged;
19. runtime `step_index` never populates authored `logical_time`;
20. runtime equals supersedes stale authored cell semantics;
21. runtime clear makes deterministic runtime cell state unknown;
22. unperceived cells preserve authored seed semantics;
23. multi-channel same-cell percepts remain deterministic and provenance-preserving;
24. one agent's evidence never appears in another agent's cognition;
25. equal semantic percept sequences produce equal cognition semantics even when provenance branches differ;
26. provenance-different branches may retain different artifact hashes;
27. runtime belief model identity binds seed model, parameters, and runtime likelihood implementation;
28. runtime likelihood hook receives `RuntimePerceptView`, never provenance-bearing runtime evidence;
29. changing only hidden provenance hashes cannot change likelihood inputs or posterior semantics;
30. invalid likelihood shape/probabilities reject typed;
31. zero posterior mass rejects typed;
32. runtime updates form an exact posterior chain from the authored seed posterior;
33. clear evidence does not invent an `absent` hypothesis;
34. no runtime evidence leaves posterior equal to seed posterior;
35. blackout-only steps preserve posterior while advancing runtime step metadata;
36. runtime cognition APIs accept no objective world-state argument;
37. public surface exports exactly the approved narrative-scoped names and preserves root isolation;
38. all pre-existing replay, uncertain belief, intention, world transition, observation projection, compiler, intervention, movie, model-comparison, and Lean gates remain green.

## Future Scheduler contract

After this V1, Scheduler V1 can implement a round as:

```text
L_k
 -> materialize runtime cognition per eligible agent
 -> runtime-compatible decision execution
 -> ActionIntent batch
 -> advance_world_step
 -> admit_world_percepts
 -> L_k+1
```

Scheduler V1 must define an explicit runtime decision consumer. Existing authored `run_intentional_decision()` is not silently repurposed.

The runtime decision model must receive a provenance-free cognition semantic view (for example posterior distributions, resolved cells, goals, and declared decision context), not `ledger_hash`, world/projection hashes, or runtime evidence lineage. Full cognition artifact hashes may be attached to the decision result for audit, but they must not be model-visible inputs that can change action policy.

## Non-goals

This V1 does not implement:

- scheduler/trajectory loop;
- runtime action-selection API;
- modification of `run_intentional_decision()`;
- authored IR mutation;
- runtime testimony/communication;
- attention or selective admission after projection;
- memory decay/forgetting;
- sensor noise;
- stochastic world transitions;
- new hidden `absent` hypotheses;
- planning/POMDP integration;
- parameter learning;
- RNG/seed management.

## Expected implementation scope

```text
docs/superpowers/specs/2026-08-25-narrative-runtime-percept-admission-v1-design.md
docs/superpowers/plans/2026-08-25-narrative-runtime-percept-admission-v1.md
narrative_dynamics/narrative/runtime_perception.py
narrative_dynamics/narrative/runtime_cognition.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_perception.py
tests/test_narrative_runtime_cognition.py
tests/test_narrative_trust_api.py
```

Any required change to authored replay/uncertain/intention/world/observation-projection modules, GenericNarrative IR, DomainSpec, root exports, or Lean sources is an architectural scope expansion and stops implementation for review.

## Success criterion

For any agent and any admitted runtime history through step `k`:

```text
Authored cognition at source_at_time
  + admitted semantic runtime percepts through step k
  = runtime cognition semantics at step k
```

while exact world/projection provenance remains attached for audit and remains unavailable as a hidden input to the cognition model.

That is the perception-to-belief bridge required before Multi-Step Scheduler V1.