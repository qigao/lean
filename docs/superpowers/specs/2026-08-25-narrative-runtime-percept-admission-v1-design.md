# Narrative Runtime Percept Admission V1 Design

## Status

Approved architectural direction for the next P0 increment after Observation Projection V1.

Exact base: `2a45278bfaee75c69cdafc8c7abd9516cd41c1c7` on `proof/narrative-dynamics-v0`.

Feature branch: `work/narrative-runtime-percept-admission-v1`.

This design closes the next missing link in the runtime cognitive loop:

```text
WorldStepResult
  -> ObservationProjectionResult
  -> Runtime percept admission
  -> Runtime evidence ledger
  -> Runtime epistemic state
  -> Runtime uncertain belief state
```

It does not add the multi-step scheduler, action scheduling, runtime intentional decision execution, communication, noisy sensors, stochastic transitions, planning, or learning.

## Motivation

The integrated engine now has a clean distinction between authored narrative semantics and runtime simulation semantics:

```text
Authored narrative:
  events / observations / claims / receptions / decisions
  -> authored epistemic replay
  -> authored uncertain belief
  -> authored intentional decision

Runtime simulation:
  ActionIntent batch
  -> WorldStepResult
  -> ObservationProjectionResult
```

The remaining break is that a `ProjectedObservation` is currently a runtime percept artifact only. It does not legally enter `epistemic_state()` or `uncertain_epistemic_state()`.

That separation is intentional. Existing authored replay is indexed by authored `logical_time`; Observation Projection is indexed by simulation `step_index`. Reusing the authored `Observation` or `EpistemicEvidence` records for runtime projection would collapse two different clocks and would blur authored evidence with simulated evidence.

The approved initial-cognition semantics are:

```text
B_runtime,0^i = B_authored^i(source_at_time)
```

and then:

```text
B_runtime,k+1^i = Update(B_runtime,k^i, E_runtime,k+1^i)
```

with the invariant:

```text
logical_time != step_index
```

Runtime cognition therefore inherits authored cognition exactly at the simulation branch point and then evolves only through runtime evidence.

## Architectural decision

Use two sidecar units rather than modifying authored replay records.

### Unit A: Runtime Perception

File:

```text
narrative_dynamics/narrative/runtime_perception.py
```

Responsibilities:

- initialize an empty runtime evidence ledger from one authored branch point;
- call `project_world_observations()` internally;
- convert engine-produced `ProjectedObservation` records into runtime epistemic evidence;
- append one immutable evidence batch for every world step, including blackout steps;
- enforce world-step and ledger continuity;
- preserve exact projection/world provenance;
- never update belief directly.

### Unit B: Runtime Cognition

File:

```text
narrative_dynamics/narrative/runtime_cognition.py
```

Responsibilities:

- reconstruct the authored epistemic seed at `source_at_time`;
- replay runtime evidence into a deterministic runtime epistemic state;
- reconstruct the authored uncertain-belief seed using the existing `UncertainBeliefModelSpec`;
- apply runtime likelihood updates using a separate runtime likelihood model;
- produce deterministic, provenance-bound runtime belief state artifacts;
- never read objective `WorldState` or `WorldStepResult` during cognition materialization.

The scheduler will consume these interfaces later. This V1 deliberately does not decide how runtime belief is fed into the next action-selection model.

## Rejected alternatives

### Modify `EpistemicEvidence` to carry both clocks

Rejected because it would change the semantics, public identity, ordering, and hashes of an already stable authored replay record. Existing authored replay is explicitly organized around `logical_time`.

### Synthesize authored logical times from simulation step indices

Rejected because it makes runtime-generated evidence appear authored. It also creates an arbitrary ordering relationship between story times and simulation steps.

### Append projected observations directly to `GenericNarrative.observations`

Rejected because runtime simulation must not mutate or masquerade as canonical authored narrative.

### Let Observation Projection update belief directly

Rejected because perception and cognition are distinct mechanisms. Observation Projection answers what an agent is permitted to perceive from the world; belief update answers how that percept changes an internal hypothesis distribution.

## Clock model

There are two independent time coordinates.

### Authored time

```text
logical_time: int
```

Used only by authored events, observations, claims, receptions, decisions, authored epistemic replay, and the authored seed belief.

### Runtime time

```text
step_index: int
```

Used only by world transitions, projected observations, runtime evidence batches, runtime epistemic state, and runtime uncertain belief state.

The simulation branch point is identified by `source_at_time: int | None`.

An empty runtime ledger always corresponds to simulation world step `0` produced by:

```text
world_state_from_story(story, domain, at_time=source_at_time)
```

The first admitted runtime world step therefore produces batch step `1`.

No API in this V1 maps `step_index` to `logical_time`.

## Runtime Perception records

### `RuntimeEpistemicEvidence`

Represents one admitted runtime percept.

Fields:

```text
observer_id: str
channel: str
cell: StateCellRef
relation: equals | clear
value: TypedValue | None
step_index: int
projected_observation_hash: str
projection_result_hash: str
projection_model_hash: str
source_world_state_hash: str
source_world_step_hash: str
source_transition_hashes: tuple[str, ...]
projection_spec_hash: str
```

Rules:

- `observer_id`, `channel`, and state cell identity must be structurally valid;
- only `equals` and `clear` are supported in V1;
- `equals` requires one `TypedValue`;
- `clear` requires `value is None`;
- `step_index` is non-negative and never interpreted as authored time;
- all hash fields are syntactically valid canonical SHA-256 content hashes;
- transition hashes are unique and canonically sorted;
- all fields come from the engine-produced projection records during admission, not from hook return values;
- direct construction creates data, not independent certification.

`content_hash` is the stable content hash of the canonical payload.

### `RuntimeEvidenceBatch`

Represents admission for exactly one runtime world step, even if no percept was emitted.

Fields:

```text
prior_ledger_hash: str
step_index: int
source_prior_world_state_hash: str
source_world_state_hash: str
source_world_step_hash: str
projection_model_hash: str
projection_result_hash: str
evidence: tuple[RuntimeEpistemicEvidence, ...]
```

Rules:

- evidence is canonicalized by `(observer_id, channel, cell, content_hash)`;
- every evidence record must bind the batch step, world-state hash, world-step hash, projection model hash, and projection-result hash exactly;
- evidence records may contain different observer ids;
- the same observer/cell may appear on different channels in one batch;
- blackout is represented by `evidence == ()`, never by omitting the batch;
- the batch binds the exact hash of the ledger prefix it extends.

### `RuntimeEvidenceLedger`

Represents the immutable runtime evidence history for all agents in one simulation branch.

Fields:

```text
domain_id: str
domain_version: str
domain_spec_hash: str
source_story_hash: str
source_at_time: int | None
initial_world_state_hash: str
current_world_state_hash: str
batches: tuple[RuntimeEvidenceBatch, ...]
```

Derived:

```text
current_step_index = 0 if batches == () else batches[-1].step_index
```

Ledger invariants:

1. domain identity is exact and immutable;
2. `source_story_hash` and `source_at_time` are immutable;
3. the empty ledger binds exactly one initial world-state hash;
4. the first batch has `step_index == 1`;
5. each later batch increments step by exactly one;
6. the first batch prior-world hash equals `initial_world_state_hash`;
7. each later batch prior-world hash equals the previous batch next-world hash;
8. `current_world_state_hash` equals the initial state hash when empty, otherwise the last batch next-world hash;
9. each batch `prior_ledger_hash` equals the stable hash of the exact ledger prefix before that batch;
10. batch order is part of ledger identity and cannot be reordered without changing or invalidating the chain.

The constructor verifies all internal hash-chain and continuity invariants it can derive from the stored data.

A directly constructed ledger is still only a self-consistent data artifact; it is not proof that historic world transitions or historic projections actually executed. Runtime truth trust comes from retaining admission results or, later, the scheduler trajectory that contains the corresponding world/projection artifacts.

### `RuntimePerceptAdmissionResult`

Represents one successful append operation.

Fields:

```text
prior_ledger_hash: str
projection_result: ObservationProjectionResult
evidence_batch: RuntimeEvidenceBatch
next_ledger: RuntimeEvidenceLedger
```

Constructor invariants:

- `evidence_batch.prior_ledger_hash == prior_ledger_hash`;
- batch projection-result hash equals `projection_result.content_hash`;
- batch projection-model hash equals `projection_result.model_hash`;
- all source hashes and step index equal the projection result;
- `next_ledger` equals the exact one-batch extension of the supplied prior ledger lineage.

As with other public runtime records, direct construction is data, not certification.

## Runtime Perception APIs

### `runtime_evidence_ledger_from_story`

Signature:

```text
runtime_evidence_ledger_from_story(
    story,
    domain,
    *,
    at_time=None,
) -> RuntimeEvidenceLedger
```

Behavior:

1. validate the narrative/domain;
2. call `world_state_from_story(story, domain, at_time=at_time)`;
3. return an empty ledger bound to exact domain identity, story hash, source cutoff, and initial world-state hash;
4. do not create any evidence batch;
5. do not materialize belief.

### `admit_world_percepts`

Signature:

```text
admit_world_percepts(
    story,
    domain,
    world_step,
    projection_model,
    prior_ledger,
) -> RuntimePerceptAdmissionResult
```

Behavior:

1. validate story/domain and ledger source identity;
2. require `prior_ledger.current_world_state_hash == world_step.prior_state.content_hash`;
3. require `prior_ledger.current_step_index == world_step.prior_state.step_index`;
4. require ledger and world-step `source_at_time` to match exactly;
5. call `project_world_observations(story, domain, world_step, projection_model)` internally;
6. convert only the returned engine-produced projected observations into runtime evidence;
7. create exactly one batch for `world_step.next_state.step_index`, even when the projection result is empty;
8. append the batch immutably and return the prior hash, projection result, batch, and next ledger.

Consequences:

- a caller cannot pass a naked `ProjectedObservation` to admission;
- a caller cannot skip a world step without breaking prior-world hash/step continuity;
- a caller cannot admit the same world step twice to the same ledger;
- a caller can branch simulation by reusing an earlier immutable ledger with a different valid next world step;
- different projection models may be used on different steps, with each batch retaining its exact model hash.

`admit_world_percepts` raises `RuntimePerceptAdmissionError` for typed runtime rejection. It preserves lower-level projection errors as causes.

## Blackout semantics

Blackout is first-class runtime history.

If:

```text
ObservationProjectionResult.observations == ()
```

then admission still appends:

```text
RuntimeEvidenceBatch(step_index=k, evidence=())
```

This is required because:

```text
no percept at step 2
```

is not equivalent to:

```text
step 2 did not exist
```

A blackout batch advances the ledger/world continuity and the runtime cognition state's `step_index`, but does not change any agent's epistemic or uncertain-belief content.

## Runtime deterministic epistemic replay

### Seed semantics

For an agent `i`, runtime epistemic replay begins from:

```text
seed_state = epistemic_state(
    story,
    domain,
    i,
    at_time=ledger.source_at_time,
)
```

The authored seed is recomputed deterministically from canonical story/domain data. It is not copied from mutable caller state.

### `RuntimeEpistemicCellView`

Fields:

```text
cell: StateCellRef
status: resolved | unknown | conflicted
resolved_value: TypedValue | None
constraints: tuple[TypedValue, ...]
basis: authored_seed | runtime_perception
supporting_runtime_evidence_hashes: tuple[str, ...]
last_runtime_step: int | None
```

Semantics:

- when a cell has no runtime percept, copy the semantic seed status/value/constraints and use `basis=authored_seed`;
- runtime percepts are direct-perception evidence and supersede prior authored epistemic content for that cell;
- a latest-step runtime `equals` resolves the cell to the observed value and clears stale authored constraints;
- a latest-step runtime `clear` makes the cell unknown with no resolved value and clears stale authored constraints;
- if multiple channels emit the same cell on the same latest step, all truth-validated percepts have the same world truth; their hashes are retained canonically as joint support;
- runtime V1 has no testimony and no `not_equals` percepts.

### `RuntimeEpistemicState`

Fields:

```text
agent_id: str
source_at_time: int | None
step_index: int
ledger_hash: str
seed_state: EpistemicState
runtime_evidence_history: tuple[RuntimeEpistemicEvidence, ...]
cells: Mapping[StateCellRef, RuntimeEpistemicCellView]
resolved_values: Mapping[StateCellRef, TypedValue]
```

The state contains the full authored seed plus the filtered runtime evidence history for that agent.

Runtime evidence history is canonicalized by:

```text
(step_index, channel, cell, content_hash)
```

The state never contains another agent's runtime evidence.

### `runtime_epistemic_state`

Signature:

```text
runtime_epistemic_state(
    story,
    domain,
    agent_id,
    ledger,
) -> RuntimeEpistemicState
```

Behavior:

1. validate story/domain/agent and ledger source identity;
2. recompute the authored seed at `ledger.source_at_time`;
3. filter ledger evidence to exactly `observer_id == agent_id`;
4. apply runtime direct-perception semantics step by step;
5. return a state whose step index equals the ledger current step, including blackout-only advances;
6. never accept or inspect `WorldState`, `WorldStepResult`, or objective state values.

Failures are typed as `RuntimeEpistemicResolutionError`.

## Runtime uncertain belief replay

### Why a separate runtime likelihood model is required

Existing `UncertainBeliefModelSpec.likelihood_hook` consumes authored `EpistemicEvidence`. Existing models may legitimately depend on authored fields such as `supporting_id`, authored source agent, and authored provenance.

Runtime evidence has different semantics and a different clock. It therefore gets a separate likelihood hook rather than being disguised as authored `EpistemicEvidence`.

### `RuntimeBeliefModelSpec`

Fields:

```text
model_id: str
version: str
seed_model: UncertainBeliefModelSpec
runtime_parameters: Mapping[str, canonical value]
runtime_likelihood_hook: callable
```

Identity binds:

- model id/version;
- exact `seed_model.content_hash`;
- canonical runtime parameters;
- measured implementation identity of the runtime likelihood hook.

The seed model remains responsible for:

```text
authored evidence -> B_runtime,0
```

The runtime likelihood hook is responsible only for:

```text
RuntimeEpistemicEvidence -> likelihood vector over the current finite hypotheses
```

Hook signature:

```text
runtime_likelihood_hook(
    agent_id,
    runtime_evidence,
    hypotheses,
    runtime_parameters,
) -> Mapping[hypothesis_hash, likelihood]
```

The hook receives no objective world state, world step, or projection model object.

### `RuntimeBeliefUpdateStep`

Fields:

```text
evidence: RuntimeEpistemicEvidence
prior: BeliefDistribution
likelihoods: tuple[BeliefLikelihood, ...]
posterior: BeliefDistribution
```

Rules match existing finite uncertain-belief update rules:

- evidence/prior/posterior describe one cell;
- hypothesis sets match exactly;
- likelihood values are finite probabilities in `[0,1]`;
- zero total posterior mass rejects;
- posterior is computed through the existing canonical `posterior_distribution` mechanism.

### `RuntimeUncertainBeliefCellView`

Fields:

```text
cell: StateCellRef
seed_posterior: BeliefDistribution
posterior: BeliefDistribution
updates: tuple[RuntimeBeliefUpdateStep, ...]
```

Rules:

- `seed_posterior` is the exact posterior produced by the authored seed model at `source_at_time`;
- the first runtime update prior equals `seed_posterior`;
- each later update prior equals the previous posterior;
- final `posterior` equals the update-chain tail;
- no runtime evidence means `posterior == seed_posterior` and `updates == ()`.

### `RuntimeUncertainBeliefState`

Fields:

```text
agent_id: str
model_id: str
model_hash: str
source_at_time: int | None
step_index: int
ledger_hash: str
seed_belief_state: UncertainBeliefState
runtime_evidence_history: tuple[RuntimeEpistemicEvidence, ...]
cells: Mapping[StateCellRef, RuntimeUncertainBeliefCellView]
```

The full authored seed belief state is retained, not merely its hash, so the resulting artifact is inspectable without hiding the inherited cognition basis.

### `runtime_uncertain_belief_state`

Signature:

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

1. validate story/domain/agent, ledger, model, and unique tracked cells;
2. call the existing `uncertain_epistemic_state()` using `model.seed_model` and `at_time=ledger.source_at_time`;
3. use each tracked cell's authored seed posterior as the runtime prior;
4. filter runtime evidence to the selected agent and tracked cell;
5. apply the runtime likelihood hook in canonical evidence order;
6. validate vectors and posterior continuity exactly;
7. return a state at `ledger.current_step_index`;
8. never read objective world state or call Observation Projection.

Failures are typed as `RuntimeBeliefResolutionError`.

## Clear evidence in uncertain belief

A runtime `clear` percept means the projected world cell is absent after the step and the clear was explicitly written by the current world transition.

V1 does not add a synthetic `absent` hypothesis to finite state variables.

Instead, `clear` is an evidence relation passed to `runtime_likelihood_hook`, exactly as a likelihood-bearing observation. The model author decides how absence changes probability over the existing finite hypotheses.

This preserves current finite-hypothesis machinery and avoids silently changing domain value spaces.

## No objective-state leakage invariant

The central cognition boundary is:

```text
runtime epistemic/belief state depends only on
  authored seed + runtime evidence ledger
```

It does not depend on objective world state directly.

A required discriminating invariant is:

```text
Given two simulation branches with different objective world states,
if one agent has the same authored seed and an identical runtime evidence ledger,
then that agent's runtime epistemic state and runtime uncertain-belief state
must be identical.
```

Formally:

```text
Seed_i = Seed'_i
L_i,k = L'_i,k
--------------------------------
Epi_i,k = Epi'_i,k
Belief_i,k = Belief'_i,k
```

No runtime cognition API accepts `WorldState` or `WorldStepResult`, making direct objective leakage impossible through the framework interface.

## Multi-agent isolation

One ledger stores all agents' admitted runtime evidence because it is a simulation-level history artifact.

Cognition materialization is per agent:

```text
RuntimeEvidenceLedger
  -> filter observer_id == i
  -> RuntimeEpistemicState_i
  -> RuntimeUncertainBeliefState_i
```

Evidence for agent A must never appear in agent B's state unless a later explicit communication/testimony mechanism is added. V1 has no such mechanism.

## Determinism and canonical ordering

Canonical order rules:

```text
RuntimeEpistemicEvidence within a batch:
  observer_id, channel, cell, content_hash

RuntimeEvidenceBatch:
  step_index only; ledger chain fixes total order

Per-agent runtime evidence replay:
  step_index, channel, cell, content_hash

Runtime tracked cells:
  entity_type, entity_id, state_variable
```

Equivalent caller insertion order for evidence/model mappings must not affect hashes.

Same canonical story, domain, branch cutoff, world/projection sequence, ledger, runtime belief model, and tracked cells must produce the same final content hashes.

## Trust boundary

### Certified-by-execution claims

`admit_world_percepts()` can guarantee for the current step that:

- the world step passed Observation Projection's existing source validation;
- the projection model was measured and validated;
- projected observations were truth-checked against the next world state;
- admitted runtime evidence was derived from those exact projected observations;
- the new batch extends the exact supplied ledger and exact world-state lineage.

### Public-record limitation

All public runtime records remain ordinary deterministic data records.

Direct construction of:

- `RuntimeEpistemicEvidence`;
- `RuntimeEvidenceBatch`;
- `RuntimeEvidenceLedger`;
- `RuntimePerceptAdmissionResult`;
- runtime cognition states

does not independently certify historical execution.

This matches the project's existing provenance philosophy: content identity plus explicit lineage, with runtime entrypoints responsible for semantic validation.

A future scheduler trajectory can retain full world-step and admission artifacts to make end-to-end trajectory revalidation stronger without changing these V1 records.

## Error handling

Public constructors use ordinary `TypeError` / `ValueError` for malformed record construction.

Runtime entrypoints expose typed operational failures:

```text
RuntimePerceptAdmissionError(ValueError)
RuntimeEpistemicResolutionError(ValueError)
RuntimeBeliefResolutionError(ValueError)
```

Existing lower-level errors are chained as causes where useful.

Admission must reject before projection execution when ledger/world continuity is already invalid.

Runtime cognition replay must reject before any likelihood-hook call when story/domain/agent/ledger/model/tracked-cell identity is invalid.

## Public API

V1 public names are narrative-scoped only and must not leak to root `narrative_dynamics`.

Proposed exact names:

```text
RuntimeEpistemicEvidence
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

`GenericNarrative`, authored `Observation`, authored `EpistemicEvidence`, `EpistemicState`, `UncertainBeliefState`, `UncertainBeliefModelSpec`, and their existing hashes/public behavior remain unchanged.

## Required V1 invariants

The implementation tests must lock all of the following.

1. empty ledger is exact-domain/story/cutoff bound;
2. initial ledger world hash equals `world_state_from_story` at the cutoff;
3. first admitted batch is step 1;
4. later batches are strictly consecutive;
5. current world-state hash chains exactly through batches;
6. batch prior-ledger hash binds the exact prefix;
7. blackout produces an empty batch rather than no batch;
8. blackout advances step/world lineage but not cognition content;
9. re-admitting one world step to the same ledger rejects;
10. skipping or reordering world steps rejects;
11. admission rejects mismatched domain/story/source cutoff before projection;
12. admission internally executes projection and does not accept naked projected observations;
13. each runtime evidence record binds projected observation, projection result/model/spec, world state/step, and transition hashes;
14. evidence ordering and ledger hashes are insertion-order invariant;
15. direct public-record construction is data, not certification;
16. runtime epistemic seed equals authored `epistemic_state(... at_time=source_at_time)`;
17. runtime uncertain seed equals authored `uncertain_epistemic_state(... at_time=source_at_time)`;
18. authored records and authored replay remain unchanged;
19. runtime `step_index` never populates authored `logical_time`;
20. runtime equals percept supersedes stale authored state for the observed cell;
21. runtime clear percept makes deterministic runtime cell state unknown;
22. cells with no runtime percept retain authored seed semantics;
23. multi-channel same-cell percepts are deterministic and provenance-preserving;
24. one agent's evidence never appears in another agent's cognition state;
25. same seed + same agent ledger yields identical cognition even when external objective branches differ;
26. runtime belief model identity binds seed model, runtime parameters, and runtime likelihood implementation;
27. runtime likelihood hook receives only agent/evidence/hypotheses/runtime parameters;
28. invalid likelihood shape/probabilities reject typed;
29. zero posterior mass rejects typed;
30. runtime belief updates form an exact posterior chain from authored seed posterior;
31. clear evidence does not invent a new hidden `absent` hypothesis;
32. no runtime evidence leaves posterior equal to seed posterior;
33. blackout-only steps preserve posterior while advancing runtime step metadata;
34. runtime cognition APIs accept no objective world-state argument;
35. public surface exports exactly the approved narrative-scoped names and preserves root isolation;
36. all pre-existing replay, uncertain belief, intention, world transition, observation projection, compiler, intervention, movie-conformance, runtime/model-comparison, and Lean gates remain green.

## Interaction with the future scheduler

This V1 intentionally stops at runtime cognition materialization.

The next Scheduler V1 can then define a round without inventing any new perception semantics:

```text
1. start with RuntimeEvidenceLedger L_k
2. materialize RuntimeEpistemicState_i / RuntimeUncertainBeliefState_i
3. run a runtime-compatible decision model for eligible agents
4. collect ActionIntent batch
5. advance_world_step
6. admit_world_percepts -> L_k+1
7. repeat
```

The scheduler must later define an explicit runtime decision consumer. Existing `run_intentional_decision()` remains authored-replay based and is unchanged by this V1.

## Non-goals

Runtime Percept Admission V1 does not implement:

- the multi-step scheduler;
- runtime action-selection APIs;
- modifying `run_intentional_decision()`;
- authored narrative mutation;
- synthetic authored observations/events/claims/receptions;
- communication or testimony between runtime agents;
- attention or selective admission after projection;
- memory decay or forgetting;
- sensor noise or stochastic observation;
- stochastic world transitions;
- hidden `absent` hypotheses;
- planning/lookahead/POMDP integration;
- learning model parameters from runtime history;
- conflict resolution beyond existing World Transition V1 semantics;
- RNG or seed management.

## Expected implementation scope

The implementation plan should be able to remain within these paths:

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

If implementation requires changes to authored replay/uncertain/intention/world/observation-projection modules, GenericNarrative IR, DomainSpec, root package exports, or Lean sources, that is an architectural scope expansion and must stop for review rather than being folded into V1 silently.

## Success criterion

The phase is complete when the engine can take one or more validated runtime world steps and produce, for any agent, a deterministic cognition state satisfying:

```text
Authored cognition at source_at_time
  +
Admitted runtime percept history through step k
  =
Runtime cognition at step k
```

with no objective-state read path and exact provenance back through projection and world-transition artifacts.

This establishes the missing runtime perception-to-belief bridge required before Multi-Step Scheduler V1.