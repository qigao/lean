# Narrative Multi-Step Scheduler V1 Design

## Status

Approved architecture design for the next P0 stage of issue #27.

This design starts from the integrated Runtime Percept Admission V1 research head:

```text
ea45389354bf9b8dff4c18917e2bca3e3e83a939
```

Research base:

```text
proof/narrative-dynamics-v0
```

Feature branch:

```text
work/narrative-multi-step-scheduler-v1
```

## Goal

Close the deterministic multi-step cognitive simulation loop without mutating authored narrative history and without exposing objective-world or provenance identities to runtime cognition or action selection.

The target closed loop is:

```text
World
  -> Observation
  -> Runtime Evidence
  -> Runtime Belief
  -> Goal
  -> Choice
  -> Action
  -> World
  -> ...
```

Across simulation rounds:

\[
S_k=(X_k,L_k)
\]

\[
L_k
\to
\{B_k^i,G_k^i,A_k^i\}_{i=1}^{N}
\to
X_{k+1}
\to
O_{k+1}^{i}
\to
L_{k+1}
\]

where:

- \(X_k\) is the immutable `WorldState` at runtime step \(k\),
- \(L_k\) is the immutable `RuntimeEvidenceLedger` at runtime step \(k\),
- every configured agent decides exactly once per round,
- every decision in one round reads the same prior ledger,
- all selected actions execute in one existing atomic World Transition batch,
- next-step observations are produced only through Runtime Percept Admission, which internally invokes Observation Projection,
- a new `SimulationState` exists only after the entire round succeeds.

## Architectural classification

This is an architectural change. It introduces a new runtime action-selection boundary and a scheduler that composes four already-separated subsystems:

1. Runtime Cognition,
2. Intentional choice mathematics,
3. World Transition,
4. Runtime Percept Admission.

It deliberately does not collapse these layers into an opaque agent-state object.

## Existing baseline and the two hard interface gaps

The integrated baseline already provides:

```text
Authored cognition(source_at_time)
    + admitted runtime percepts
    -> RuntimeUncertainBeliefState
```

and:

```text
ActionIntent batch
    -> advance_world_step(...)
    -> WorldStepResult
    -> admit_world_percepts(...)
    -> RuntimeEvidenceLedger
```

Two existing interfaces prevent a naive scheduler from closing the loop.

### Gap 1: authored intentional execution cannot consume runtime belief

Existing:

```python
run_intentional_decision(
    story,
    domain,
    decision_id,
    model,
)
```

always resolves its own belief through:

```python
uncertain_epistemic_state(
    story,
    domain,
    decision.actor_id,
    model.belief_model,
    decision.context_cells,
    at_time=decision.logical_time,
)
```

Therefore it cannot consume `RuntimeUncertainBeliefState` and must not be repurposed for runtime rounds.

### Gap 2: World Transition only executes canonical authored actions

Existing `advance_world_step()` accepts `ActionIntent`, but resolves every intent back to a canonical `Decision` in `story.decisions`, then resolves `selected_action` from that canonical decision's action options.

Therefore Scheduler V1 must not fabricate temporary `Decision` or `ActionOption` values.

## Chosen architecture

The design adds two sidecar modules:

```text
narrative_dynamics/narrative/runtime_intention.py
narrative_dynamics/narrative/simulation.py
```

`runtime_intention.py` owns runtime belief-to-goal-to-choice evaluation.

`simulation.py` owns deterministic static scheduling and round/trajectory composition.

Existing authored IR, authored intentional execution, World Transition, Observation Projection, Runtime Percept Admission, and Runtime Cognition remain authoritative in their current scopes.

## Rejected alternatives

### Rejected: synthesize a new authored `Decision` every runtime step

This would overload authored `logical_time` with runtime `step_index`, mutate or shadow canonical narrative identity, and weaken World Transition's canonical action trust boundary.

### Rejected: make Scheduler reimplement intentional mathematics

Duplicating goal scoring, softmax policy formation, goal-conditioned action policies, and marginal action policy would create two implementations of the same mechanism that could drift scientifically and numerically.

### Rejected: change World Transition to accept dynamic runtime action IR

That would reopen the canonical action/type/argument trust boundary and is a separate architectural expansion.

### Rejected: dynamic eligibility in V1

An objective-world scheduler hook deciding which agents are active would introduce another capability, identity, and attestation boundary. V1 uses static round scheduling instead.

## Static scheduling semantics

V1 uses a fixed configured agent set.

For every round and every configured `RuntimeAgentSpec`:

\[
\text{exactly one runtime decision is evaluated}
\]

There is no implicit skip.

If an agent does nothing in a round, its authored decision template must contain an explicit `wait`/`noop` action that is selected normally.

The configured agent set must be non-empty.

One configured agent may contribute at most one action per round.

Every configured agent must have a distinct `agent_id` and a distinct authored `decision_template_id`.

Input order of agent specs is semantically irrelevant; canonical order is lexical `agent_id` order.

## Authored Decision as immutable runtime template

Scheduler V1 reuses an authored `Decision` as an immutable action-space template across any number of runtime rounds.

The template contributes only canonical authored configuration:

```text
actor_id
decision type
context_cells
canonical ActionOption values
```

Its `logical_time` is not the runtime clock.

It means only:

> this canonical decision/action template existed by this authored time.

If simulation `source_at_time` is numeric, every configured template must satisfy:

\[
decision.logical\_time \le source\_at\_time
\]

The same template may be reused at runtime steps 0, 1, 2, ... without changing its `logical_time`.

Runtime clock remains exclusively:

```text
step_index
```

and never becomes authored `logical_time`.

## Runtime intentional action-selection boundary

### `RuntimeIntentionalDecisionModelSpec`

Fields:

```python
RuntimeIntentionalDecisionModelSpec(
    model_id: str,
    version: str,
    supported_decision_types: tuple[str, ...],
    belief_model: RuntimeBeliefModelSpec,
    goal_model: GoalModelSpec,
    choice_model: ChoiceModelSpec,
)
```

Requirements:

- at least one supported decision type,
- decision types unique and canonicalized,
- `belief_model` is exactly `RuntimeBeliefModelSpec`,
- `goal_model` is exactly `GoalModelSpec`,
- `choice_model` is exactly `ChoiceModelSpec`,
- configured choice-model goal ids equal goal-model goal ids exactly,
- model identity binds all nested model identities and the runtime selection implementation identity.

### Compatibility-preserving reuse of existing intentional mathematics

A self-review of the current attestation layer shows `measure_implementation(...)` measures the entire backing Python module bytes of a class implementation.

Therefore modifying `narrative_dynamics/narrative/intention.py` merely to extract a shared kernel would change the measured identity of existing authored `IntentionalDecisionModelSpec` values.

Scheduler V1 must preserve existing authored intentional implementation identity and hashes.

Accordingly:

- `intention.py` is not modified in V1,
- `run_intentional_decision()` is unchanged,
- runtime intention reuses the existing pure private mathematical helpers from `intention.py` as an internal dependency,
- runtime model identity binds both:
  - the measured `runtime_intention.py` implementation, and
  - the measured existing `IntentionalDecisionModelSpec` implementation/module identity.

This avoids mathematical duplication while preserving authored implementation identity.

If a future version wants a formally shared public/private kernel module, that migration must explicitly version the authored intentional model identity rather than silently changing it.

## Provenance-free action-selection semantics

The runtime intentional model must not receive a full provenance-bearing cognition artifact as a model-visible input.

`run_runtime_intentional_decision()` first materializes the full runtime belief artifact for audit:

```text
RuntimeEvidenceLedger
    -> RuntimeUncertainBeliefState
```

It then extracts a private semantic view containing only the current decision context's posterior distributions.

Conceptually:

```text
RuntimeUncertainBeliefState
    |
    | full audit artifact retained in result
    |
    +--> private semantic view
            context cell -> posterior distribution
                 |
                 +--> goal/choice mathematics
```

The semantic action-selection view omits:

```text
ledger_hash
runtime evidence hashes
world state hashes
world-step hashes
transition hashes
projection hashes
projection model/spec hashes
source provenance
step provenance identity
```

It also does not expose objective `WorldState` or `WorldStepResult`.

The model-visible action inputs are exactly:

1. canonical authored decision context cells,
2. canonical authored action options,
3. posterior distributions for those context cells,
4. `GoalModelSpec`,
5. `ChoiceModelSpec`.

`step_index` is audit metadata in the final runtime result and is not an intentional choice input in V1.

Therefore, with the same decision template, posterior semantics, goal model, and choice model:

\[
\pi(A)=\pi'(A)
\]

and:

\[
A_{MAP}=A'_{MAP}
\]

regardless of hidden provenance identity differences.

## Runtime belief materialization

For one runtime decision template `decision` at prior ledger \(L_k\), runtime intention calls:

```python
runtime_uncertain_belief_state(
    story,
    domain,
    decision.actor_id,
    ledger,
    model.belief_model,
    decision.context_cells,
)
```

The authored decision's `context_cells` are therefore exactly the runtime tracked belief cells for action selection.

No additional scheduler-owned context-cell configuration exists in V1.

## Runtime intentional mathematical semantics

Runtime intention preserves the existing authored mathematical mechanism:

1. expected goal instrumentality from posterior mass,
2. goal score:

\[
score(g)=pressure_g\,E[instrumentality_g]-cost_g-risk_g
\]

3. goal softmax using `beta_goal`,
4. goal-conditioned action softmax using `beta_action`,
5. marginal action policy:

\[
P(a)=\sum_g P(g)P(a\mid g)
\]

6. lexical deterministic MAP action for exact ties.

The runtime path must use the same existing numerical validation and normalization behavior as the authored path.

## Runtime semantic belief hash for `GoalState`

Existing `GoalState` is reusable because it does not require an authored belief type; it stores a `belief_state_hash` plus goal scores/policy.

For runtime action selection, `GoalState.belief_state_hash` binds the canonical provenance-free posterior semantic view used by the goal/choice mathematics, not the full `RuntimeUncertainBeliefState.content_hash`.

This makes `GoalState` itself a semantic cognition artifact:

\[
\text{same posterior semantics}
\Rightarrow
\text{same GoalState semantics and hash}
\]

The full runtime belief artifact remains attached separately to `RuntimeIntentionalDecisionResult` for audit.

This distinction prevents hidden ledger/provenance identities from entering goal identity while retaining exact upstream audit lineage in the enclosing runtime result.

## `RuntimeIntentionalDecisionResult`

Fields:

```python
RuntimeIntentionalDecisionResult(
    model_id: str,
    model_hash: str,
    decision_id: str,
    step_index: int,
    belief_state: RuntimeUncertainBeliefState,
    goal_state: GoalState,
    conditional_action_policies: Mapping[str, Mapping[str, float]],
    action_scores: Mapping[str, float],
    action_policy: Mapping[str, float],
    selected_action: str,
)
```

Validation requires:

- model/decision ids canonical,
- nonnegative runtime `step_index`,
- `belief_state` is exactly `RuntimeUncertainBeliefState`,
- `step_index == belief_state.step_index`,
- `goal_state.belief_state_hash` equals the recomputed provenance-free posterior semantic hash,
- conditional action policies cover exactly the goal policy,
- all conditional policies cover one exact common action set,
- scores and final policy cover that same action set,
- all policies are normalized and finite using existing intentional validation semantics,
- selected action is the lexical deterministic MAP action.

The result's full `content_hash` includes the complete runtime belief artifact and therefore may differ across provenance-distinct branches even when action semantics are identical.

Correct invariant:

```text
same posterior semantics
    -> same goal policy
    -> same action policy
    -> same selected action
```

but not necessarily:

```text
same posterior semantics
    -> same RuntimeIntentionalDecisionResult hash
```

## `run_runtime_intentional_decision`

Signature:

```python
def run_runtime_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeIntentionalDecisionModelSpec,
) -> RuntimeIntentionalDecisionResult:
```

Execution order:

1. validate story/domain,
2. validate runtime model type/identity,
3. resolve canonical authored decision template,
4. validate supported decision type,
5. validate non-empty unique context cells,
6. validate non-empty unique actions,
7. enforce decision-template cutoff,
8. materialize runtime uncertain belief using exactly `decision.context_cells`,
9. extract provenance-free posterior semantic view,
10. evaluate existing intentional goal/choice mathematics,
11. build runtime result with full audit belief state and semantic `GoalState` binding.

Typed outer error:

```python
RuntimeIntentionalDecisionResolutionError
```

Runtime belief, goal, choice, validation, or numerical failures are preserved as chained causes.

## Runtime action to canonical World Transition

Scheduler converts runtime selection mechanically:

```python
ActionIntent(
    decision_id=result.decision_id,
    selected_action=result.selected_action,
    selection_model_id=result.model_id,
    selection_result_hash=result.content_hash,
)
```

Scheduler does not copy or synthesize action type/arguments.

Existing World Transition remains responsible for re-resolving:

```text
decision_id -> canonical Decision
selected_action -> canonical ActionOption
```

and for validating executable action type/effect capability.

This preserves the current canonical action trust boundary.

## RuntimeAgentSpec

Fields:

```python
RuntimeAgentSpec(
    agent_id: str,
    decision_template_id: str,
    intentional_model: RuntimeIntentionalDecisionModelSpec,
)
```

Data-level validation:

- canonical non-empty ids,
- intentional model exact type.

Execution-level certification additionally resolves story/domain and requires:

- `agent_id` names a canonical story entity,
- `decision_template_id` names a canonical story decision,
- decision actor equals `agent_id`,
- decision type is supported by the runtime intentional model,
- decision template is legal at simulation source cutoff.

## SimulationModelSpec

Fields:

```python
SimulationModelSpec(
    model_id: str,
    version: str,
    domain_id: str,
    domain_version: str,
    domain_spec_hash: str,
    agents: tuple[RuntimeAgentSpec, ...],
    world_model: WorldTransitionModelSpec,
    observation_model: ObservationProjectionModelSpec,
)
```

Requirements:

- canonical model/domain ids and version,
- valid domain content hash,
- non-empty agent tuple,
- exact `RuntimeAgentSpec` values,
- unique `agent_id`,
- unique `decision_template_id`,
- canonical lexical agent order,
- exact `WorldTransitionModelSpec`,
- exact `ObservationProjectionModelSpec`,
- nested world and observation model domain identity equals simulation domain identity.

Model identity binds:

```text
model id/version
domain id/version/hash
canonical agent specs and runtime intentional model hashes
world transition model hash
observation projection model hash
simulation scheduler implementation identity
```

Changing any nested cognitive, world, observation, or scheduler implementation identity changes the simulation model identity.

Public construction is data validation, not proof that the model matches a particular story.

Every execution revalidates story/domain/template bindings.

## Observation model and passive observers

Observation Projection is type-based and may produce projected observations for canonical story entities that are not configured as scheduled runtime agents.

Scheduler V1 allows such passive observer evidence to remain in the simulation-wide runtime evidence ledger.

Only configured `RuntimeAgentSpec.agent_id` values are materialized into runtime cognition and decisions.

Therefore:

```text
projected observer != scheduled actor
```

is allowed.

This avoids adding a new observer-filtering trust boundary to Scheduler V1.

A configured agent may receive no percepts in a round; that is an ordinary blackout for that agent. No projection coverage requirement is imposed on scheduled agents.

## SimulationState

Fields:

```python
SimulationState(
    model_id: str,
    model_hash: str,
    step_index: int,
    world_state: WorldState,
    evidence_ledger: RuntimeEvidenceLedger,
)
```

Required internal invariants:

\[
step\_index
=
world\_state.step\_index
=
evidence\_ledger.current\_step\_index
\]

and:

\[
evidence\_ledger.current\_world\_state\_hash
=
hash(world\_state)
\]

World and ledger must also agree exactly on:

```text
domain id
domain version
domain spec hash
source story hash
source_at_time
```

`SimulationState` does not add another world parent hash or ledger parent hash. Existing `WorldState` and `RuntimeEvidenceLedger` already own those lineages.

Direct public construction can establish internal consistency but is not certification that historical execution occurred.

## `simulation_state_from_story`

Signature:

```python
def simulation_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    model: SimulationModelSpec,
    *,
    at_time: int | None = None,
) -> SimulationState:
```

Execution:

1. validate story/domain/model,
2. validate simulation/world/observation model domain identities,
3. validate every configured agent/template binding and cutoff,
4. create initial `WorldState` with `world_state_from_story(..., at_time=at_time)`,
5. create initial `RuntimeEvidenceLedger` with `runtime_evidence_ledger_from_story(..., at_time=at_time)`,
6. require exact initial world hash equality,
7. create `SimulationState(step_index=0, ...)` bound to the exact simulation model hash.

No runtime percept batch exists at step zero.

Initial runtime cognition therefore inherits authored cognition exactly at `source_at_time`.

## SimulationAgentStep

Fields:

```python
SimulationAgentStep(
    agent_id: str,
    decision_template_id: str,
    decision_result: RuntimeIntentionalDecisionResult,
    action_intent: ActionIntent,
)
```

It validates exact bridge identity:

```text
agent_id == canonical template actor at execution

decision_template_id == decision_result.decision_id

action_intent.decision_id == decision_result.decision_id

action_intent.selected_action == decision_result.selected_action

action_intent.selection_model_id == decision_result.model_id

action_intent.selection_result_hash == decision_result.content_hash
```

The record is audit data. Canonical story/template actor validation is repeated by scheduler execution.

## One canonical simulation round

`simulate_step()` signature:

```python
def simulate_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: SimulationState,
    model: SimulationModelSpec,
) -> SimulationStepResult:
```

Execution order is strict.

### Phase 1: validate shared prior state

Before any cognition/model hook executes:

- validate story/domain,
- validate simulation model identity,
- validate prior simulation state identity,
- require `prior_state.model_id/hash == model.model_id/content_hash`,
- validate exact world/ledger alignment,
- re-resolve all configured agent/template bindings,
- validate decision-template cutoff,
- canonicalize schedule by `agent_id`.

### Phase 2: evaluate all agents against one prior ledger

For every configured agent, in canonical agent order:

```python
result = run_runtime_intentional_decision(
    story,
    domain,
    agent.decision_template_id,
    prior_state.evidence_ledger,
    agent.intentional_model,
)
```

Every runtime decision result must satisfy:

```text
result.step_index == prior_state.step_index
result.belief_state.ledger_hash == prior_state.evidence_ledger.content_hash
```

No agent sees another current-round action, transition, next world state, or current-round percept.

All agents therefore read one shared cognitive snapshot:

\[
B_k^i = F_i(L_k)
\]

### Phase 3: construct all ActionIntents only after every selection succeeds

No world transition hook executes until all configured runtime decisions have returned valid results.

Each result is converted mechanically to one `ActionIntent`.

The resulting tuple is canonicalized by `agent_id` through `SimulationAgentStep` order.

### Phase 4: execute one atomic World Transition

Call exactly once:

```python
world_step = advance_world_step(
    story,
    domain,
    prior_state.world_state,
    model.world_model,
    tuple(agent_step.action_intent for agent_step in agent_steps),
)
```

World Transition retains all existing semantics:

- one immutable prior snapshot,
- one action per actor,
- canonical authored action resolution,
- capability-limited deltas,
- overlap rejection,
- no priority,
- no last-writer-wins,
- atomic commit.

### Phase 5: project and admit next percepts exactly once

Scheduler must not call `project_world_observations()` directly.

Call exactly once:

```python
admission = admit_world_percepts(
    story,
    domain,
    world_step,
    model.observation_model,
    prior_state.evidence_ledger,
)
```

Runtime Percept Admission already owns the projection trust boundary and internally invokes Observation Projection exactly once.

This prevents duplicate projection hook execution and duplicate projection lineage.

### Phase 6: construct next SimulationState

Construct:

```text
S_{k+1} = (
    world_step.next_state,
    admission.next_ledger,
)
```

with:

\[
step(S_{k+1}) = step(S_k)+1
\]

and all SimulationState alignment invariants revalidated.

Only after this succeeds does `simulate_step()` return a result.

## Round transaction semantics

A simulation round is atomic at the Scheduler artifact boundary.

If any runtime cognition or selection fails:

```text
no World Transition execution
no Observation Projection execution
no admission
no next SimulationState
```

If World Transition fails:

```text
no Observation Projection execution
no admission
no next SimulationState
```

If projection/admission fails:

```text
no next SimulationState
```

An immutable temporary `WorldStepResult` may have been computed internally before admission fails, but it is not committed as a simulation state and is not returned as a successful partial step.

The prior `SimulationState`, `WorldState`, and `RuntimeEvidenceLedger` remain immutable and unchanged on every failure path.

## SimulationStepResult

Fields:

```python
SimulationStepResult(
    model_id: str,
    model_hash: str,
    step_index: int,
    prior_state: SimulationState,
    agent_steps: tuple[SimulationAgentStep, ...],
    world_step: WorldStepResult,
    admission_result: RuntimePerceptAdmissionResult,
    next_state: SimulationState,
)
```

`step_index` is the next-state step index.

Required bindings:

```text
model id/hash == prior and next simulation model id/hash

step_index == next_state.step_index
step_index == prior_state.step_index + 1

agent_steps sorted by agent_id
agent_steps cover configured agent set exactly at execution
all agent decision results bind prior ledger hash and prior step

world_step.prior_state == prior_state.world_state
world_step.next_state == next_state.world_state

admission_result.prior_ledger_hash == prior_state.evidence_ledger.content_hash
admission_result.next_ledger == next_state.evidence_ledger
admission_result.projection_result.source_world_step_hash == world_step.content_hash

one transition exists for every SimulationAgentStep action_intent
no extra transition exists outside the scheduled agent set
```

The complete prior state is intentionally stored so each step is a self-contained audit artifact.

## SimulationTrajectory

Fields:

```python
SimulationTrajectory(
    model_id: str,
    model_hash: str,
    initial_state: SimulationState,
    steps: tuple[SimulationStepResult, ...],
    final_state: SimulationState,
)
```

V1 trajectories are finite and non-empty.

Each consecutive step must satisfy:

\[
R_k.next\_state = R_{k+1}.prior\_state
\]

The first prior state equals `initial_state`.

The final next state equals `final_state`.

Step indices are exactly consecutive from `initial_state.step_index + 1`.

Trajectory identity binds:

```text
simulation model identity
initial state
ordered step results
final state
```

No behavioral-summary hash replaces the full audit identity.

## `simulate_trajectory`

Signature:

```python
def simulate_trajectory(
    story: GenericNarrative,
    domain: DomainSpec,
    initial_state: SimulationState,
    model: SimulationModelSpec,
    *,
    rounds: int,
) -> SimulationTrajectory:
```

`rounds` must be an exact positive integer; booleans and zero are rejected.

Execution is repeated `simulate_step()`:

\[
S_0 \xrightarrow{R_1} S_1 \xrightarrow{R_2} \cdots \xrightarrow{R_n} S_n
\]

No terminal predicate, event callback, dynamic stop rule, or implicit horizon exists in V1.

For deterministic configured model hooks:

```text
same story
+ same source cutoff
+ same simulation model identity
+ same initial SimulationState
+ same rounds
=> same complete trajectory hash
```

## Provenance versus behavioral semantics

The Scheduler preserves the distinction established by Runtime Percept Admission V1.

Full artifact identities bind provenance.

Therefore two branches may have identical extensional world values, posterior semantics, action policies, and selected actions but different:

```text
ActionIntent selection_result_hash
WorldStepResult hash
WorldState hash
RuntimeEvidenceLedger hash
RuntimeIntentionalDecisionResult hash
SimulationStepResult hash
SimulationTrajectory hash
```

if their hidden provenance lineages differ.

This is correct.

At the same time, hidden provenance must not alter model-visible intentional inputs.

Therefore:

```text
same canonical decision template
+ same posterior semantics
+ same goal/choice configuration
=> same goal policy
=> same action policy
=> same selected action
```

This invariant must be tested separately from full artifact identity.

## Typed errors

### Runtime intention

```python
class RuntimeIntentionalDecisionResolutionError(ValueError):
    ...
```

It wraps runtime belief, goal, choice, template, or numerical failures while preserving the underlying exception as `__cause__`.

### Scheduler

```python
class SimulationError(ValueError):
    ...

class SimulationStepError(SimulationError):
    ...

class SimulationTrajectoryError(SimulationError):
    ...
```

`simulate_step()` wraps stage failures as `SimulationStepError` with chained typed cause.

Messages identify the failed deterministic stage, and cognition failures may name the canonical `agent_id`.

`simulate_trajectory()` wraps the failing round as `SimulationTrajectoryError` and preserves its `SimulationStepError` cause.

No successful partial trajectory is returned.

## Public API

V1 adds exactly 16 narrative-scoped public names.

### Runtime intention — 4

```text
RuntimeIntentionalDecisionModelSpec
RuntimeIntentionalDecisionResult
RuntimeIntentionalDecisionResolutionError
run_runtime_intentional_decision
```

### Simulation — 12

```text
RuntimeAgentSpec
SimulationModelSpec
SimulationState
SimulationAgentStep
SimulationStepResult
SimulationTrajectory
SimulationError
SimulationStepError
SimulationTrajectoryError
simulation_state_from_story
simulate_step
simulate_trajectory
```

These names are exported only from:

```text
narrative_dynamics.narrative
```

They are not exported from the root `narrative_dynamics` package.

Existing public APIs remain present and unchanged.

## Required invariants

Implementation tests must lock at least the following V1 invariants.

### Runtime intentional selection

1. runtime model requires exact Runtime Belief / Goal / Choice model types,
2. runtime model identity binds nested model identities,
3. runtime model identity binds runtime implementation identity,
4. runtime model identity binds existing authored intentional implementation identity used as mathematical dependency,
5. `intention.py` remains byte-for-byte unchanged in this feature,
6. existing authored intentional model hashes/results remain unchanged,
7. canonical authored decision template must exist,
8. template actor/model type must be valid,
9. numeric source cutoff rejects templates authored after the branch point,
10. context cells are exactly runtime tracked belief cells,
11. action selection receives posterior semantics only,
12. no ledger/world/projection/transition hash is model-visible,
13. no objective WorldState/WorldStepResult is an action-selection input,
14. hidden provenance changes cannot change goal/action policy when posteriors are equal,
15. exact authored intentional goal/action mathematics is preserved,
16. action coverage mismatch rejects typed,
17. goal hypothesis coverage mismatch rejects typed,
18. selected action is deterministic lexical MAP,
19. runtime result binds current ledger-derived belief artifact for audit,
20. runtime result step index equals runtime belief step index,
21. GoalState binds provenance-free posterior semantic hash,
22. full runtime result hash may differ across provenance-distinct but semantically equal branches.

### Simulation model/state

23. simulation agents are non-empty,
24. agent ids are unique,
25. template ids are unique,
26. agent input order does not alter model identity,
27. every agent resolves to a canonical story entity,
28. every template resolves to a canonical story decision,
29. template actor equals configured agent id,
30. runtime model supports template decision type,
31. template cutoff is valid,
32. world model domain identity matches simulation domain,
33. observation model domain identity matches simulation domain,
34. SimulationState step equals world step equals ledger step,
35. ledger current world hash equals exact WorldState hash,
36. state world/ledger story/domain/cutoff identities match,
37. state model hash must match current simulation model at execution,
38. direct public state construction is data, not historical certification.

### One simulation step

39. all scheduled agents decide exactly once,
40. there is no silent skip,
41. explicit wait/noop is the only V1 no-action representation,
42. every agent decision binds the same prior ledger hash,
43. every agent decision binds the same prior step index,
44. no world hook executes before every agent selection succeeds,
45. all ActionIntents derive exactly from runtime selection results,
46. one atomic `advance_world_step()` call executes the complete intent batch,
47. scheduler never calls Observation Projection directly,
48. exactly one `admit_world_percepts()` call follows successful world transition,
49. world failure causes zero projection/admission hook execution,
50. cognition failure causes zero world/projection execution,
51. admission failure produces no successful next SimulationState,
52. prior simulation/world/ledger artifacts remain unchanged on failure,
53. next simulation state binds exact world-step next state,
54. next simulation state binds exact admission next ledger,
55. each successful round increments runtime step exactly once,
56. same authored decision template can be reused in consecutive rounds,
57. authored `logical_time` never changes or becomes runtime `step_index`,
58. agent input order does not alter step semantics or hash,
59. passive observer evidence may exist without scheduling that observer,
60. scheduled-agent blackout is legal and does not skip the agent's next decision.

### Trajectory

61. rounds must be a positive non-bool integer,
62. each trajectory step's prior state equals previous next state,
63. initial/final state bindings are exact,
64. trajectory steps are consecutive,
65. same deterministic initial inputs/models/rounds replay to exact same trajectory hash,
66. full trajectory hash binds provenance and may differ across provenance-distinct branches,
67. no authored observations/claims/decisions are injected between runtime rounds,
68. story content hash remains unchanged throughout simulation.

### API/regression

69. exactly 16 new narrative-scoped names are exported,
70. none of the 16 names leak to package root,
71. existing authored intention public API remains unchanged,
72. existing World Transition public API/semantics remain unchanged,
73. existing Observation Projection public API/semantics remain unchanged,
74. existing Runtime Perception/Cognition public API/semantics remain unchanged,
75. all pre-existing Python tests remain green,
76. all Lean build/theorem gates remain green.

## Required two-round causal closure test

The central acceptance test must prove a true cross-round causal path rather than two independent one-step runs.

Use at least two agents A and B and a typed world cell such as `service.alert`.

### Initial cognition

At `source_at_time`:

- A's authored/runtime-seed cognition supports taking an action that raises the alert,
- B's current posterior favors waiting.

No authored runtime-intermediate observation exists.

### Round 0

Static schedule evaluates both from the exact same \(L_0\).

Expected selections:

```text
A -> raise_alert
B -> wait
```

One atomic world transition produces:

\[
X_1(alert=true)
\]

Observation Projection exposes the new fact to B through its declared observation capability/channel.

Runtime Percept Admission creates the next ledger:

\[
X_1
\to
O_1^B(alert=true)
\to
L_1
\]

### Round 1

Without changing the canonical story, decisions, observations, claims, or receptions:

\[
L_1
\to
B_1^B
\]

B's posterior changes and the same authored decision template now selects:

```text
B -> respond
```

The test must demonstrate the causal chain:

\[
A_0
\to X_1
\to O_1^B
\to B_1^B
\to A_1^B
\]

and assert exact story immutability:

```python
story.content_hash == original_story_hash
story.decisions == original_decisions
story.observations == original_observations
story.claims == original_claims
story.receptions == original_receptions
```

This is the primary proof that the runtime cognitive loop closes across at least two consecutive rounds without authored intermediate-state injection.

## Failure-order tests

Tests must use sentinel hooks/counters to prove transaction ordering, not infer it only from returned errors.

Required examples:

- one agent runtime belief/choice failure => world transition hooks called zero times,
- one agent failure late in canonical agent order => no earlier successful decision causes world execution,
- world write conflict => observation projection hooks called zero times,
- projection failure => no successful `SimulationState` is returned,
- malformed/mismatched prior SimulationState rejects before any agent model hook.

## Determinism tests

Required examples:

- reverse `RuntimeAgentSpec` input order => same `SimulationModelSpec.content_hash`,
- equivalent canonical agent ordering => same `SimulationStepResult.content_hash`,
- two full runs from independently reconstructed identical initial state => exact same trajectory hash,
- multiple consecutive explicit wait/noop rounds remain deterministic,
- same semantic posterior branch with altered hidden provenance => same runtime goal policy, action policy, and selected action even when enclosing audit hashes differ.

## Expected implementation scope

Expected feature diff is exactly these eight paths:

```text
docs/superpowers/specs/2026-08-26-narrative-multi-step-scheduler-v1-design.md
docs/superpowers/plans/2026-08-26-narrative-multi-step-scheduler-v1.md
narrative_dynamics/narrative/runtime_intention.py
narrative_dynamics/narrative/simulation.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_intention.py
tests/test_narrative_simulation.py
tests/test_narrative_trust_api.py
```

If implementation appears to require modifying any of the following, stop and treat it as architecture expansion rather than silently folding it into Scheduler V1:

```text
narrative_dynamics/narrative/intention.py
narrative_dynamics/narrative/runtime_cognition.py
narrative_dynamics/narrative/runtime_perception.py
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/ir.py
narrative_dynamics/narrative/domain.py
narrative_dynamics/__init__.py
Lean sources
```

## Explicit non-goals

V1 does not add:

- dynamic eligibility,
- terminal predicates,
- stop conditions based on world/cognition state,
- callbacks or event buses,
- scheduler priorities,
- actor priority,
- last-writer-wins,
- conflict resolution V2,
- runtime-generated Decision or ActionOption IR,
- authored IR mutation,
- runtime testimony or communication,
- attention or selective admission after projection,
- forgetting,
- stochastic action selection,
- stochastic world transition,
- stochastic observation,
- RNG or seeds,
- planning or POMDP control,
- learning or parameter updates,
- reactive baseline integration,
- step-index-visible intentional policies,
- step-dependent world transition hooks,
- new root-package exports.

## Scientific interpretation

After Scheduler V1, the engine can support a deterministic, inspectable cognitive multi-agent runtime loop:

\[
World
\to Observation
\to Belief
\to Goal
\to Choice
\to Action
\to World
\]

across multiple rounds.

This remains a simulation mechanism, not evidence that real agents use the modeled cognition.

The scientific value is that competing mechanisms can now be evaluated on trajectories where actions alter future information and future beliefs rather than only on isolated one-step choices.

The next architectural expansion after this feature should not be more scheduler machinery. The intended next research stage is the discriminating multi-agent benchmark that compares mechanisms on history-sensitive trajectories.

## Success criterion

Scheduler V1 succeeds when a deterministic two-agent fixture executes at least two consecutive rounds such that a first-round action changes the world, that change produces a capability-limited runtime percept for another agent, the admitted percept changes that agent's runtime posterior, and the changed posterior changes its second-round action, all without authored intermediate-state injection or objective/provenance leakage into cognition.

Formally:

\[
A_0^A
\to X_1
\to O_1^B
\to B_1^B
\to A_1^B
\]

with:

\[
GenericNarrative_{after}=GenericNarrative_{before}
\]

and, for deterministic configured implementations:

\[
SameInitialState + SameModels + SameRounds
\Rightarrow
SameTrajectoryHash
\]
