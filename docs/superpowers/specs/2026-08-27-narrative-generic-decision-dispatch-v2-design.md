# Generic Runtime Decision Dispatch / Multi-Model Scheduler V2 Design

Date: 2026-08-27
Status: design for review
Base branch: `proof/narrative-dynamics-v0`
Base commit: `8317339cde03d46e0500798ebb30dfa27ca43e38`
Roadmap: issue #27

## 1. Purpose

The runtime now has three narrative-native decision families:

- **reactive**: current admitted cue semantics -> action scores -> policy;
- **intentional**: runtime posterior -> goals -> choice -> policy;
- **planning**: runtime posterior -> joint hidden-state belief -> hypothetical transition/observation branches -> finite-horizon value -> policy.

All three already produce a complete action policy, deterministic lexical MAP action, typed failure, and provenance-bound content identity. The remaining scheduler gap is that `simulation.py` is still hard-wired to the intentional family.

This feature introduces one closed typed decision-dispatch boundary so a single simulation may contain reactive, intentional, and planning agents in the same round while preserving the existing simultaneous world-step semantics.

Target flow:

```text
shared prior SimulationState
        |
        +--> reactive agent ------+
        +--> intentional agent ---+--> typed dispatch results
        +--> planning agent ------+
                                      |
                                      v
                              ActionIntent batch
                                      |
                                      v
                            atomic world transition
                                      |
                                      v
                              observation projection
                                      |
                                      v
                                percept admission
                                      |
                                      v
                              next SimulationState
```

This is an explicit architecture V2 boundary change. All-intentional simulations must preserve behavioral semantics, but pre-V2 simulation/model/step/trajectory content hashes are not compatibility targets because the new dispatch envelope and dispatch implementation identity are intentionally provenance-bound.

## 2. Non-goals

This feature does **not**:

- add conflict-resolution semantics;
- weaken existing overlapping-write rejection;
- add stochastic world/observation transitions;
- change reactive, intentional, or planning algorithms;
- add arbitrary callback/protocol model plugins;
- change `GenericNarrative` or `DomainSpec`;
- change world transition, observation projection, or percept admission semantics;
- change held-out model-comparison infrastructure;
- add RNG to dispatch or scheduler;
- preserve pre-V2 simulation hashes.

Conflict Resolution V2 remains separate. If heterogeneous agents produce conflicting writes, the existing typed world-transition conflict remains required; no family has priority.

## 3. Existing coupling

At base `8317339c...`, the scheduler-family coupling is concentrated in three places:

1. `RuntimeAgentSpec.intentional_model: RuntimeIntentionalDecisionModelSpec`;
2. `SimulationAgentStep.decision_result: RuntimeIntentionalDecisionResult`;
3. `simulate_step()` directly calls `run_runtime_intentional_decision()` and catches only the intentional resolution error.

Everything after decision resolution is already model-family neutral:

- all agents read the same immutable `prior_state.evidence_ledger`;
- all `ActionIntent`s are completed before world execution;
- `advance_world_step()` performs one atomic simultaneous batch;
- projection occurs only after the world step;
- admission extends the immutable ledger;
- the next state binds the new world and new ledger.

Therefore V2 adds a narrow dispatch sidecar and migrates only the scheduler's decision boundary.

## 4. Architecture choice

### Rejected: dispatch directly inside `simulation.py`

Direct scheduler `isinstance` branches would make the simulator own cognition-family details and force future family changes into world orchestration code.

### Rejected: arbitrary protocol / duck typing

A generic `run()` protocol would weaken the current closed typed trust boundary and make model identity/attestation less explicit.

### Selected: closed typed dispatch sidecar

Add:

```text
narrative_dynamics/narrative/runtime_decision_dispatch.py
```

This module owns the only three-family branch. The scheduler imports only the generic wrapper/result/error/runner.

## 5. Generic model wrapper

### 5.1 Public record

```python
RuntimeDecisionModelSpec(
    model_kind: str,
    model: RuntimeReactiveDecisionModelSpec
         | RuntimeIntentionalDecisionModelSpec
         | RuntimePlanningDecisionModelSpec,
)
```

Allowed kinds are exactly:

```text
reactive
intentional
planning
```

Exact tag/type relation:

```text
reactive    -> RuntimeReactiveDecisionModelSpec
intentional -> RuntimeIntentionalDecisionModelSpec
planning    -> RuntimePlanningDecisionModelSpec
```

Unknown kinds, mismatched tag/type pairs, and arbitrary model-like objects fail during construction before any family hook executes.

The wrapper is frozen/canonical and exposes read-only derived properties:

```python
model_id -> str
model_version -> str
supported_decision_types -> tuple[str, ...]
nested_model_hash -> str
```

### 5.2 Identity

`RuntimeDecisionModelSpec.to_dict()` binds exactly:

```text
model_kind
nested model id
nested model version
nested model content hash
dispatch model-wrapper implementation identity
dispatch runner implementation identity
```

The two implementation identities are explicit:

```python
measure_implementation(RuntimeDecisionModelSpec).manifest_identity()
measure_implementation(run_runtime_decision).manifest_identity()
```

There is no alternative or unspecified attestation boundary.

Consequences:

- changing family-specific model configuration changes wrapper identity;
- changing model family changes wrapper identity even if one scenario's policy is equal;
- changing wrapper semantics changes wrapper identity;
- changing `run_runtime_decision` implementation changes wrapper identity;
- simulation identity transitively binds all of the above through each agent wrapper.

## 6. Common dispatch result

### 6.1 Principle

Only genuinely common semantics are projected into the generic envelope.

The design explicitly does **not** pretend that:

- a reactive cue snapshot is a belief state;
- an intentional goal state is a planning value state;
- reactive/intentional action scores are the same concept as planning action values.

The exact family-specific result is retained in full.

### 6.2 Public record

```python
RuntimeDecisionDispatchResult(
    model_kind: str,
    decision_model_hash: str,
    model_id: str,
    model_hash: str,
    decision_id: str,
    actor_id: str,
    step_index: int,
    ledger_hash: str,
    action_policy: Mapping[str, float],
    selected_action: str,
    model_result_hash: str,
    model_result: RuntimeReactiveDecisionResult
                | RuntimeIntentionalDecisionResult
                | RuntimePlanningDecisionResult,
)
```

Meaning:

- `decision_model_hash`: generic `RuntimeDecisionModelSpec.content_hash`;
- `model_id` / `model_hash`: exact nested scientific model identity;
- `decision_id`: authored decision template;
- `actor_id`: resolved actor;
- `step_index`: runtime step used by the decision;
- `ledger_hash`: exact prior runtime ledger used by the decision;
- `action_policy`: complete normalized family policy;
- `selected_action`: exact lexical MAP family action;
- `model_result_hash`: exact nested result hash;
- `model_result`: complete exact nested typed result.

### 6.3 Family-specific extraction

Reactive:

```text
actor_id    = result.actor_id
ledger_hash = result.ledger_hash
```

Intentional:

```text
actor_id    = result.belief_state.agent_id
ledger_hash = result.belief_state.ledger_hash
```

Planning:

```text
actor_id    = result.actor_id
ledger_hash = result.ledger_hash
```

All families:

```text
model_id         = result.model_id
model_hash       = result.model_hash
decision_id      = result.decision_id
step_index       = result.step_index
action_policy    = result.action_policy
selected_action  = result.selected_action
model_result_hash = result.content_hash
```

### 6.4 Constructor self-validation

Standalone `RuntimeDecisionDispatchResult` construction must validate every binding it can determine without an external wrapper:

- exact allowed model kind;
- nested result type matches kind;
- model id/hash match nested result;
- decision id matches nested result;
- actor matches family-specific extraction;
- step matches nested result;
- ledger hash matches family-specific extraction;
- nested result hash is exact;
- common policy equals nested policy exactly;
- common selected action equals nested selected action exactly;
- policy values are finite/non-negative, cover at least one action, and sum to one within the existing probability tolerance;
- selected action is the deterministic lexical MAP.

`decision_model_hash` can be syntax-validated in the constructor, but only the public runner can prove it equals the external validated wrapper's `content_hash`.

### 6.5 Result identity

`to_dict()` includes all common fields and the complete nested result payload.

Lineage:

```text
RuntimeDecisionDispatchResult.content_hash
  -> generic decision model hash
  -> common policy/action
  -> exact nested result hash
  -> exact nested family result payload
  -> family-specific cue/belief/goal/planning audit state
```

No family audit data is discarded.

## 7. Generic runner

Public entry point:

```python
run_runtime_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeDecisionModelSpec,
) -> RuntimeDecisionDispatchResult
```

It has no world state, scheduler state, observation projection, RNG, or explicit step input.

### 7.1 Generic preflight

Before any family runner:

- validate/reconstruct `RuntimeDecisionModelSpec` so constructor-bypass wrapper forgeries fail closed;
- validate the narrative/domain at the generic boundary;
- require the decision id to exist;
- require the authored decision type to be supported by the wrapper.

Family runners retain their stricter family-specific validation.

### 7.2 Closed dispatch

```text
reactive
  -> run_runtime_reactive_decision(...)
intentional
  -> run_runtime_intentional_decision(...)
planning
  -> run_runtime_planning_decision(...)
```

No arbitrary callable path exists.

### 7.3 Error normalization

Public error:

```python
RuntimeDecisionDispatchError(ValueError)
```

Ordinary family resolution/validation errors are normalized to this type while preserving the exact underlying exception as `__cause__`.

`BaseException` is never caught.

Expected cause examples:

```text
RuntimeDecisionDispatchError
  caused by RuntimeReactiveDecisionResolutionError
RuntimeDecisionDispatchError
  caused by RuntimeIntentionalDecisionResolutionError
RuntimeDecisionDispatchError
  caused by RuntimePlanningDecisionResolutionError
```

### 7.4 Runner-dependent binding

Before return, verify at minimum:

```text
result.decision_model_hash == validated model.content_hash
result.model_kind          == model.model_kind
result.model_id            == model.model_id
result.model_hash          == model.nested_model_hash
```

Then revalidate all common/nested result bindings so an internal constructor-bypass forgery cannot escape.

## 8. Capability isolation

`runtime_decision_dispatch.py` may import:

- the three family model/result/error/runner modules;
- `GenericNarrative` / `DomainSpec` validation;
- runtime evidence ledger contract;
- stable hashing / attestation helpers;
- standard-library dataclass/typing/mapping utilities.

It must **not** import:

```text
world
simulation
observation_projection
```

Thus generic dispatch has no objective-world capability.

## 9. Scheduler V2

### 9.1 `RuntimeAgentSpec`

Change to:

```python
RuntimeAgentSpec(
    agent_id: str,
    decision_template_id: str,
    decision_model: RuntimeDecisionModelSpec,
)
```

Remove `intentional_model`; do not keep a compatibility alias. This is an explicit V2 API migration, and retaining both fields would create two competing sources of model identity.

`RuntimeAgentSpec.to_dict()` binds:

```text
agent_id
decision_template_id
decision_model_hash
```

### 9.2 `SimulationModelSpec`

Structure remains otherwise unchanged:

```text
simulation model/version
domain identity
canonical agents
world model
observation model
scheduler implementation identity
```

Each agent now transitively binds model kind, nested model identity, and generic dispatch implementation identities.

### 9.3 Execution binding validation

Scheduler preflight becomes family-neutral:

```text
story declares configured agent
story declares configured decision template
decision actor == agent id
decision type in agent.decision_model.supported_decision_types
source cutoff admits decision
```

Scheduler code must not inspect cue cells, goals, belief models, hidden states, planning horizon, or other family-specific internals.

### 9.4 `SimulationAgentStep`

Change to:

```python
SimulationAgentStep(
    agent_id: str,
    decision_template_id: str,
    decision_result: RuntimeDecisionDispatchResult,
    action_intent: ActionIntent,
)
```

Validate only common semantics:

```text
agent_id == decision_result.actor_id
decision_template_id == decision_result.decision_id
action_intent.decision_id == decision_result.decision_id
action_intent.selected_action == decision_result.selected_action
action_intent.selection_model_id == decision_result.model_id
action_intent.selection_result_hash == decision_result.content_hash
```

The scheduler no longer dereferences intentional-specific belief state.

### 9.5 `simulate_step()`

For each canonical configured agent:

```python
result = run_runtime_decision(
    story,
    domain,
    agent.decision_template_id,
    prior_state.evidence_ledger,
    agent.decision_model,
)
```

Require:

```text
result.step_index == prior_state.step_index
result.ledger_hash == prior_state.evidence_ledger.content_hash
result.actor_id == agent.agent_id
result.decision_model_hash == agent.decision_model.content_hash
```

Then create the unchanged `ActionIntent` schema:

```python
ActionIntent(
    decision_id=result.decision_id,
    selected_action=result.selected_action,
    selection_model_id=result.model_id,
    selection_result_hash=result.content_hash,
)
```

All agents finish decision resolution against the same immutable prior ledger before any world hook runs.

### 9.6 World/projection/admission sequence stays unchanged

```text
all generic decisions
 -> complete ActionIntent tuple
 -> advance_world_step
 -> admit_world_percepts
 -> next SimulationState
```

A same-round agent cannot observe another agent's selected action through the evidence ledger before world execution.

## 10. Action provenance

`ActionIntent` schema does not change.

V2 lineage becomes:

```text
ActionIntent.selection_result_hash
  -> RuntimeDecisionDispatchResult.content_hash
       -> decision_model_hash
       -> exact nested result hash/payload
       -> family-specific cognition provenance
```

`selection_model_id` remains the nested scientific model id, not a generic dispatcher id.

The simulation model hash separately binds the generic decision wrapper and dispatch implementation identities.

This preserves the distinction between:

- **scientific model identity**: which behavioral/cognitive model produced the choice;
- **runtime certification identity**: which generic dispatch/scheduler boundary certified and executed it.

## 11. Required invariants and experiments

### 11.1 All-intentional behavioral regression

When every `RuntimeAgentSpec` wraps an intentional model, V2 must preserve:

- selected actions for fixed inputs;
- world deltas and extensional next-world values;
- projected percept semantics;
- nested intentional posterior/goal/policy semantics;
- final extensional world values across the same number of rounds.

The following hashes are allowed and expected to change because V2 provenance is different:

```text
SimulationModelSpec.content_hash
SimulationAgentStep.content_hash
SimulationStepResult.content_hash
SimulationTrajectory.content_hash
```

### 11.2 Direct dispatch equivalence

For each family on fixed inputs:

```text
dispatch.model_result      == direct family result
dispatch.model_result_hash == direct family result.content_hash
dispatch.action_policy     == direct family result.action_policy
dispatch.selected_action   == direct family result.selected_action
```

Generic dispatch must not alter probability math.

### 11.3 Heterogeneous same-round snapshot isolation

A synthetic round must configure at least:

```text
Agent A = reactive
Agent B = intentional
Agent C = planning
```

All three dispatch results must bind exactly the same prior ledger hash.

No world/projection/admission hook may run until all three decision results and intents are complete.

### 11.4 Heterogeneous two-round closure

A deterministic synthetic fixture must execute at least two rounds:

```text
round t prior ledger
 -> three family decisions
 -> simultaneous intent batch
 -> atomic world transition
 -> projection
 -> admission
 -> round t+1 prior ledger
 -> three family decisions
```

No authored intermediate world, belief, or decision state may be injected.

### 11.5 Deterministic replay

Fixed story/domain/models/initial state/round count must replay to the exact same **V2** trajectory content hash.

Agent input order must not affect simulation identity, decision semantics, canonical result order, or trajectory identity.

### 11.6 Conflict semantics remain V1

Overlapping world writes continue to fail through the existing typed world conflict path.

There is no priority relation:

```text
planning > intentional   -- forbidden
intentional > reactive   -- forbidden
lexical agent order wins -- forbidden
```

## 12. Failure boundaries

### Generic wrapper

Fail closed on:

- unknown model kind;
- kind/type mismatch;
- arbitrary model object;
- constructor-bypass wrapper with noncanonical/mismatched nested identity.

### Generic dispatch

Fail closed on:

- unknown decision id;
- unsupported decision type;
- family typed failure;
- wrong nested result type;
- forged model id/hash;
- forged actor;
- forged ledger hash;
- forged step;
- forged nested result hash;
- forged common policy/action;
- forged generic `decision_model_hash`.

### Scheduler

Fail closed on:

- configured agent absent from story;
- decision actor mismatch;
- unsupported decision type;
- dispatch actor mismatch;
- dispatch prior-ledger mismatch;
- dispatch generic-model-hash mismatch;
- duplicate agents/templates;
- existing world transition conflict;
- existing projection/admission failure.

Any decision/preflight failure must occur before world/projection/admission hooks execute.

## 13. Public API

Add exactly four narrative-scoped names:

```text
RuntimeDecisionModelSpec
RuntimeDecisionDispatchResult
RuntimeDecisionDispatchError
run_runtime_decision
```

Export them from:

```text
narrative_dynamics.narrative
```

Do not export them from package root:

```text
narrative_dynamics
```

Existing family public APIs remain intact.

Existing simulation public type names remain intact, but `RuntimeAgentSpec` and `SimulationAgentStep` receive the V2 field/type changes described above.

## 14. Testing strategy

### 14.1 New dispatch tests

Add `tests/test_narrative_runtime_decision_dispatch.py` covering at least:

1. canonical frozen wrapper and exact identity binding;
2. exact wrapper + runner implementation attestation identities;
3. wrong tag/type pairs reject before family hooks;
4. direct reactive dispatch exact nested-result equivalence;
5. direct intentional dispatch exact nested-result equivalence and actor/ledger derivation;
6. direct planning dispatch exact nested-result equivalence;
7. common policy/action exact binding;
8. nested/common result forgery rejection;
9. runner rejection of forged `decision_model_hash`;
10. typed family causes preserved under `RuntimeDecisionDispatchError`;
11. module capability isolation;
12. public runner signature excludes world/RNG/explicit step.

### 14.2 Scheduler V2 tests

Extend `tests/test_narrative_simulation.py` for:

1. migrated all-intentional behavior;
2. direct intentional nested-result regression;
3. one mixed reactive/intentional/planning round;
4. mixed two-round causal closure;
5. exact mixed-trajectory replay;
6. heterogeneous agent input-order invariance;
7. every same-round result binds one shared prior ledger;
8. dispatch failure blocks world/projection/admission;
9. existing atomic world conflict rejection under mixed families;
10. simulation model identity changes if only one agent's model kind changes.

### 14.3 Trust API

Extend `tests/test_narrative_trust_api.py` with exactly the four new narrative exports and root isolation.

### 14.4 RED requirement

The first implementation-bearing CI must be a **test-only RED** commit.

Expected RED causes are limited to:

- missing generic dispatch module/types/runner;
- scheduler still requiring intentional model/result and direct intentional execution;
- missing four narrative-scoped exports.

Unrelated baseline tests must remain green.

Pure `docs/superpowers/specs/**` and `docs/superpowers/plans/**` commits must not trigger proof under the current workflow configuration.

## 15. TDD implementation slices

The implementation plan should keep these responsibilities separate:

1. test-only RED: dispatch contracts + scheduler mixed-family behavior + trust exports;
2. generic wrapper/result records + runner stub;
3. closed three-family runner + typed cause normalization;
4. model-dependent dispatch-result binding/forgery rejection;
5. migrate `RuntimeAgentSpec` / `SimulationAgentStep` to generic types;
6. switch `simulate_step()` to `run_runtime_decision` without changing world/projection/admission logic;
7. heterogeneous one-round/two-round fixtures and deterministic replay;
8. exact four-name narrative exports;
9. final exact-head proof.

No production slice starts before authoritative test-only RED.

## 16. Expected file scope

Expected feature paths:

```text
docs/superpowers/specs/2026-08-27-narrative-generic-decision-dispatch-v2-design.md
docs/superpowers/plans/2026-08-27-narrative-generic-decision-dispatch-v2.md
narrative_dynamics/narrative/runtime_decision_dispatch.py
narrative_dynamics/narrative/simulation.py
narrative_dynamics/narrative/__init__.py
tests/test_narrative_runtime_decision_dispatch.py
tests/test_narrative_simulation.py
tests/test_narrative_trust_api.py
```

No other production path is expected.

Do not modify:

```text
narrative_dynamics/narrative/runtime_reactive.py
narrative_dynamics/narrative/runtime_intention.py
narrative_dynamics/narrative/runtime_planning.py
narrative_dynamics/narrative/runtime_cognition.py
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/model_comparison.py
narrative_dynamics/prison_*.py
NarrativeDynamics/**/*.lean
```

If a family-specific runtime change becomes necessary, stop and reclassify rather than silently expanding scope.

## 17. Roadmap effect

After V2, the same deterministic multi-round simulation can contain heterogeneous cognitive agents:

```text
Reactive agent
Intentional agent
Planning agent
       |
       v
same shared prior snapshot
       |
       v
atomic world update
       |
       v
capability-limited observations
       |
       v
next-round heterogeneous cognition
```

This closes the scheduler integration gap among the three existing decision families.

It does not by itself close issue #27. Remaining major work includes:

- Conflict Resolution V2;
- Stochastic World / Observation Models;
- released identification/model-comparison experiments across reactive, intentional, and planning families.

## 18. Acceptance criteria

Complete only when one exact feature head satisfies all of the following:

- wrapper accepts exactly the three supported typed model families;
- wrapper identity binds nested model + wrapper implementation + runner implementation;
- generic dispatch returns a provenance-bound common envelope containing the exact nested family result;
- generic dispatch is probability/action-equivalent to each direct family runner;
- scheduler has no family-specific decision runner call or family-specific cognition dereference;
- one simulation can contain reactive, intentional, and planning agents simultaneously;
- every same-round agent result binds the same prior ledger hash;
- heterogeneous two-round execution closes without authored intermediate state injection;
- fixed heterogeneous inputs replay to the exact same V2 trajectory hash;
- existing world conflicts remain fail-closed with no model-kind priority;
- package root remains unchanged;
- final diff stays inside declared scope;
- exact-head GitHub Actions proof completes successfully across full Python and Lean gates.
