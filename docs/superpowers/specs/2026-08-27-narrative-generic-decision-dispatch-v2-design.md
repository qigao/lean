# Generic Runtime Decision Dispatch / Multi-Model Scheduler V2 Design

Date: 2026-08-27
Status: design for review
Base branch: `proof/narrative-dynamics-v0`
Base commit: `8317339cde03d46e0500798ebb30dfa27ca43e38`
Roadmap: issue #27

## 1. Purpose

The runtime now has three narrative-native decision families with distinct cognitive semantics:

- reactive: current admitted cue semantics -> action scores -> policy;
- intentional: runtime posterior -> goals -> choice -> policy;
- planning: runtime posterior -> joint hidden-state belief -> hypothetical future transitions/observations -> finite-horizon value -> policy.

All three already produce a complete action policy, a deterministic lexical MAP action, stable content identity, and typed failure semantics. The multi-step scheduler, however, is still hard-wired to `RuntimeIntentionalDecisionModelSpec` and `RuntimeIntentionalDecisionResult`.

This feature removes that scheduler-family coupling without collapsing the three cognitive models into one opaque agent state.

The target capability is a heterogeneous multi-agent runtime round in which, against one shared prior runtime snapshot, different agents may use different decision families:

```text
shared prior SimulationState
        |
        +--> reactive agent ------+
        +--> intentional agent ---+--> typed generic decision results
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

The feature is an architecture V2 boundary change. It preserves behavioral semantics for all-intentional simulations but does not promise old simulation/model/trajectory content hashes, because the new dispatch envelope and dispatch implementation identity are intentionally provenance-bound.

## 2. Non-goals

This feature does not:

- add conflict resolution semantics;
- weaken existing overlapping-write rejection;
- add stochastic world or observation transitions;
- change any family-specific reactive, intentional, or planning algorithm;
- add a fourth pluggable arbitrary callback/protocol model family;
- generalize `GenericNarrative` or `DomainSpec`;
- change objective-world transition semantics;
- change observation projection or runtime evidence admission semantics;
- change held-out model-comparison infrastructure;
- add RNG to the generic dispatcher or scheduler;
- preserve pre-V2 simulation content hashes.

Conflict Resolution V2 remains a separate roadmap item. If heterogeneous agents produce overlapping world writes, the current typed world-transition conflict remains the required result.

## 3. Existing coupling to remove

At the integrated base, `simulation.py` is model-family neutral after decision resolution. The hard coupling is concentrated in three places:

1. `RuntimeAgentSpec.intentional_model: RuntimeIntentionalDecisionModelSpec`;
2. `SimulationAgentStep.decision_result: RuntimeIntentionalDecisionResult`;
3. `simulate_step()` directly calls `run_runtime_intentional_decision()` and catches only `RuntimeIntentionalDecisionResolutionError`.

The rest of the scheduler already has the desired architecture:

- all agents decide against the same `prior_state.evidence_ledger`;
- `ActionIntent`s are accumulated before the world changes;
- `advance_world_step()` receives one complete simultaneous intent batch;
- projection happens only after the atomic world transition;
- admission extends the immutable runtime ledger;
- the next simulation state binds the resulting world and ledger.

The design therefore keeps the world/projection/admission pipeline unchanged and introduces a narrow decision dispatch sidecar.

## 4. Architectural choice

### 4.1 Rejected: scheduler-local union dispatch

The scheduler could import all three model/result types and branch with `isinstance` directly.

This is rejected because every future model family would require more scheduler cognition-specific code, and `simulation.py` would become responsible for extracting actor/ledger/policy semantics from each family.

### 4.2 Rejected: arbitrary protocol / duck typing

A broad protocol such as `model.run(...) -> result` would make future extension easy.

This is rejected for V2 because the runtime trust boundary is intentionally closed and typed. Arbitrary objects with compatible attributes would weaken model identity, type rejection, implementation attestation, and fail-closed behavior.

### 4.3 Selected: closed typed dispatch sidecar

Add `narrative_dynamics/narrative/runtime_decision_dispatch.py`.

The sidecar owns the only three-family branch. It accepts a canonical tagged model wrapper and returns a canonical common envelope that still contains the exact family-specific result.

The scheduler imports only the generic wrapper/result/error/runner from this sidecar.

## 5. Public dispatch model contract

### 5.1 `RuntimeDecisionModelSpec`

Public record:

```python
RuntimeDecisionModelSpec(
    model_kind: str,
    model: RuntimeReactiveDecisionModelSpec
         | RuntimeIntentionalDecisionModelSpec
         | RuntimePlanningDecisionModelSpec,
)
```

Allowed `model_kind` values are exactly:

```text
reactive
intentional
planning
```

The tag/type relation is closed and exact:

```text
reactive    -> RuntimeReactiveDecisionModelSpec
intentional -> RuntimeIntentionalDecisionModelSpec
planning    -> RuntimePlanningDecisionModelSpec
```

A mismatched pair fails during wrapper construction before any decision hook can run.

Examples that must fail:

```text
model_kind="planning" + RuntimeReactiveDecisionModelSpec
model_kind="reactive" + RuntimeIntentionalDecisionModelSpec
unknown model_kind
arbitrary object with model-like fields
```

The wrapper must be frozen and canonical.

It exposes read-only derived properties sufficient for scheduler validation:

```python
model_id -> str
model_version -> str
supported_decision_types -> tuple[str, ...]
nested_model_hash -> str
```

These are delegated from the validated nested typed model; they are not independently writable state.

### 5.2 Wrapper identity

`RuntimeDecisionModelSpec.to_dict()` binds:

```text
model_kind
nested model id
nested model version
nested model content hash
dispatch implementation identity
```

The dispatch implementation identity is measured from `RuntimeDecisionModelSpec` or the dispatch module's designated stable implementation boundary using existing attestation infrastructure.

Consequences:

- changing a family-specific model changes the wrapper hash;
- changing model family with an otherwise similar policy changes the wrapper hash;
- changing the generic dispatch implementation changes the wrapper hash;
- input order cannot affect identity because the wrapper has no unordered model-family collection.

## 6. Common dispatch result contract

### 6.1 Principle

The common result must contain only semantics that genuinely exist for every family.

It must not invent a common latent representation. In particular:

- reactive cue state is not a belief state;
- intentional goal state is not a planning value function;
- planning action values are not reactive/intentional action scores.

Family-specific internals remain inside the exact nested result.

### 6.2 `RuntimeDecisionDispatchResult`

Public frozen record:

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

Field meanings:

- `model_kind`: exact family tag;
- `decision_model_hash`: `RuntimeDecisionModelSpec.content_hash`;
- `model_id`: nested family model id;
- `model_hash`: nested family model content hash;
- `decision_id`: authored decision template id;
- `actor_id`: actor whose decision was resolved;
- `step_index`: runtime ledger/simulation step used for the decision;
- `ledger_hash`: exact shared prior ledger used for the decision;
- `action_policy`: complete normalized family policy copied into the common boundary;
- `selected_action`: exact family lexical MAP projection;
- `model_result_hash`: exact nested family result content hash;
- `model_result`: exact typed nested family result.

### 6.3 Family-specific common-field extraction

The dispatcher normalizes common fields as follows.

Reactive:

```text
actor_id   = result.actor_id
ledger_hash = result.ledger_hash
```

Intentional:

```text
actor_id   = result.belief_state.agent_id
ledger_hash = result.belief_state.ledger_hash
```

Planning:

```text
actor_id   = result.actor_id
ledger_hash = result.ledger_hash
```

For all families:

```text
model_id       = result.model_id
model_hash     = result.model_hash
decision_id    = result.decision_id
step_index     = result.step_index
action_policy  = result.action_policy
selected_action = result.selected_action
model_result_hash = result.content_hash
```

The common envelope must validate these equalities exactly.

### 6.4 Result constructor validation

Standalone result construction must fail closed on all facts it can determine without the external wrapper object:

- valid exact `model_kind`;
- nested result type matches `model_kind`;
- `model_id == model_result.model_id`;
- `model_hash == model_result.model_hash`;
- `decision_id == model_result.decision_id`;
- `actor_id` equals the family-specific actor extraction;
- `step_index == model_result.step_index`;
- `ledger_hash` equals the family-specific ledger extraction;
- `model_result_hash == model_result.content_hash`;
- common `action_policy` exactly equals nested `action_policy`;
- common `selected_action` exactly equals nested `selected_action`;
- policy is a finite non-negative complete simplex;
- selected action belongs to the policy and is the deterministic lexical MAP.

The constructor can validate hash syntax for `decision_model_hash` but cannot prove that it equals an external `RuntimeDecisionModelSpec.content_hash`; the public runner must perform that model-dependent binding check before returning.

### 6.5 Result identity

`to_dict()` includes both common fields and the complete nested result payload.

This makes the lineage explicit:

```text
RuntimeDecisionDispatchResult.content_hash
  binds decision_model_hash
  binds common policy/action
  binds model_result_hash
  binds complete exact nested model_result
```

No family-specific audit trace is discarded.

## 7. Generic dispatch runner

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

The function has no world state, observation projection, scheduler state, RNG, or explicit step input.

### 7.1 Preflight

Before a family runner executes, the dispatcher validates/reconstructs the generic wrapper so constructor-bypassing wrapper forgeries fail before family hooks.

It also validates that the requested authored decision exists and that its type is in `model.supported_decision_types` before family model execution.

The family runner retains responsibility for its stricter family-specific preflight.

### 7.2 Closed dispatch

Exact branch:

```text
reactive
  -> run_runtime_reactive_decision(...)

intentional
  -> run_runtime_intentional_decision(...)

planning
  -> run_runtime_planning_decision(...)
```

No generic callable is accepted.

### 7.3 Error normalization

Public error:

```python
class RuntimeDecisionDispatchError(ValueError):
    ...
```

The runner catches only ordinary family resolution failures / validation failures and normalizes them to `RuntimeDecisionDispatchError`, preserving the exact underlying exception as `__cause__`.

It must not catch `BaseException`.

Family failures remain inspectable:

```text
RuntimeDecisionDispatchError
  caused by RuntimeReactiveDecisionResolutionError

RuntimeDecisionDispatchError
  caused by RuntimeIntentionalDecisionResolutionError

RuntimeDecisionDispatchError
  caused by RuntimePlanningDecisionResolutionError
```

### 7.4 Runner-dependent result binding

Before return, the runner verifies:

```text
result.decision_model_hash == validated RuntimeDecisionModelSpec.content_hash
result.model_kind == model.model_kind
result.model_id == model.model_id
result.model_hash == model.nested_model_hash
```

It also rechecks nested-result/common-field equality so a constructor-bypass internal forgery cannot escape through the runner.

## 8. Capability isolation

`runtime_decision_dispatch.py` may import:

- reactive runtime decision types/runner;
- intentional runtime decision types/runner;
- planning runtime decision types/runner;
- `GenericNarrative`, `DomainSpec`, runtime ledger contract;
- stable hashing / attestation helpers;
- standard-library typing/dataclass/mapping utilities.

It must not import:

```text
world
simulation
observation_projection
```

The dispatcher therefore has no objective-world capability and cannot use a real world state to resolve cognition.

It also does not import family-internal world/projection modules indirectly by adding new code; family modules retain their existing isolation contracts.

## 9. Scheduler V2 contract

### 9.1 `RuntimeAgentSpec`

Change:

```python
RuntimeAgentSpec(
    agent_id: str,
    decision_template_id: str,
    decision_model: RuntimeDecisionModelSpec,
)
```

The old `intentional_model` field is removed rather than retained as a compatibility alias.

This is an explicit architecture V2 migration. Keeping an `intentional_model` alias would make the public scheduler contract misleading and would create two sources of model identity.

`RuntimeAgentSpec.to_dict()` binds:

```text
agent_id
decision_template_id
decision_model_hash
```

No family-specific field appears in scheduler provenance.

### 9.2 `SimulationModelSpec`

The existing model remains structurally the same:

```text
simulation identity
domain identity
agents
world model
observation model
scheduler implementation identity
```

Because each agent now binds `decision_model_hash`, the simulation model identity transitively binds:

- model family;
- family-specific model identity;
- generic dispatch implementation identity.

Changing one agent from intentional to planning changes the simulation model hash even if both happen to produce the same policy on one scenario.

### 9.3 Execution binding validation

`_validate_execution_bindings()` becomes family-neutral.

For each agent:

```text
story declares agent
story declares decision template
decision actor == agent id
decision type in agent.decision_model.supported_decision_types
source cutoff admits decision
```

The scheduler does not inspect nested family configuration such as cue cells, goals, hidden states, or planning horizon.

### 9.4 `SimulationAgentStep`

Change:

```python
SimulationAgentStep(
    agent_id: str,
    decision_template_id: str,
    decision_result: RuntimeDecisionDispatchResult,
    action_intent: ActionIntent,
)
```

Validation is common and family-neutral:

```text
agent_id == decision_result.actor_id
decision_template_id == decision_result.decision_id
action_intent.decision_id == decision_result.decision_id
action_intent.selected_action == decision_result.selected_action
action_intent.selection_model_id == decision_result.model_id
action_intent.selection_result_hash == decision_result.content_hash
```

The old intentional-specific `decision_result.belief_state.agent_id` access disappears from scheduler code.

### 9.5 `simulate_step()`

For each canonical agent:

```python
result = run_runtime_decision(
    story,
    domain,
    agent.decision_template_id,
    prior_state.evidence_ledger,
    agent.decision_model,
)
```

The scheduler validates:

```text
result.step_index == prior_state.step_index
result.ledger_hash == prior_state.evidence_ledger.content_hash
result.actor_id == agent.agent_id
result.decision_model_hash == agent.decision_model.content_hash
```

Then it constructs:

```python
ActionIntent(
    decision_id=result.decision_id,
    selected_action=result.selected_action,
    selection_model_id=result.model_id,
    selection_result_hash=result.content_hash,
)
```

All agents complete decision resolution against the same immutable prior ledger before `advance_world_step()` is called.

### 9.6 World/projection/admission unchanged

After `agent_steps` are built, existing V1 sequencing remains unchanged:

```text
all decision dispatches
  -> complete ActionIntent tuple
  -> advance_world_step
  -> admit_world_percepts
  -> next SimulationState
```

No family can observe another same-round agent's selected action through the runtime evidence ledger before world transition.

This preserves simultaneous-snapshot semantics.

## 10. Action lineage

The existing `ActionIntent` schema is not changed.

V2 changes what `selection_result_hash` points to:

```text
ActionIntent.selection_result_hash
  -> RuntimeDecisionDispatchResult.content_hash
       -> exact family model result hash + payload
       -> family-specific cognition/audit provenance
```

`selection_model_id` remains the nested family model id, not a new generic dispatcher id.

The simulation model hash separately binds the generic `RuntimeDecisionModelSpec` wrapper and dispatch implementation identity.

This split answers two provenance questions cleanly:

- which scientific decision model produced the action? -> `selection_model_id` / nested `model_hash`;
- which generic scheduler-dispatch boundary certified the action? -> dispatch result hash and simulation model hash.

## 11. Behavioral and scientific invariants

### 11.1 Homogeneous intentional regression

A simulation configured with only intentional decision wrappers must preserve the pre-V2 behavioral semantics:

- same selected actions at each round;
- same world deltas and next world state values;
- same projected percept semantics;
- same runtime posterior/goal semantics inside nested intentional results;
- same final extensional world values for the same number of rounds.

It is not required to preserve:

- old `SimulationModelSpec.content_hash`;
- old `SimulationAgentStep.content_hash`;
- old `SimulationStepResult.content_hash`;
- old trajectory content hash.

Those identities must change because dispatch provenance is new.

### 11.2 Direct family dispatch equivalence

For each family, running the generic dispatcher on fixed inputs must embed the exact direct family runner result:

```text
dispatch.model_result == direct_family_result
dispatch.model_result_hash == direct_family_result.content_hash
dispatch.action_policy == direct_family_result.action_policy
dispatch.selected_action == direct_family_result.selected_action
```

The generic layer must not alter family probability math.

### 11.3 Heterogeneous same-round snapshot isolation

A synthetic round with at least three agents must configure:

```text
agent A -> reactive
agent B -> intentional
agent C -> planning
```

All three generic dispatch results must bind exactly the same prior ledger hash.

Their family-specific results may differ in internal cognition and policies.

No world transition hook runs until all three results and action intents have been constructed.

### 11.4 Heterogeneous multi-round closure

A deterministic synthetic fixture must run at least two rounds with the three families present.

Required causal structure:

```text
round t prior ledger
 -> three family decisions
 -> simultaneous action batch
 -> one world transition
 -> projected observations
 -> admitted evidence
 -> round t+1 prior ledger
 -> three family decisions
```

No authored intermediate world state or authored intermediate belief injection is allowed.

### 11.5 Deterministic replay

Fixed story/domain/models/initial state/round count must produce the exact same V2 trajectory content hash on repeated execution.

Input agent order must not affect simulation model identity, decision execution semantics, step result ordering, or trajectory identity.

### 11.6 Existing world conflicts remain typed

If heterogeneous agents emit intents whose world-effect write sets conflict, execution must fail through the existing typed world transition conflict path.

There is no family priority and no model-kind priority:

```text
planning does not beat intentional
intentional does not beat reactive
lexical agent order does not resolve overlapping writes
```

## 12. Failure boundaries

Required fail-closed cases include:

### Generic model wrapper

- unknown model kind;
- kind/type mismatch;
- arbitrary model object;
- constructor-bypass wrapper with mismatched nested identity.

### Generic dispatch

- unknown decision id;
- unsupported decision type;
- family runner typed failure;
- family runner result/common envelope mismatch;
- forged nested result hash;
- forged nested model id/hash;
- forged actor id;
- forged ledger hash;
- forged step index;
- forged policy/action binding;
- forged generic decision model hash.

### Scheduler

- configured agent not in story;
- decision actor != configured agent;
- decision type unsupported by generic decision model;
- dispatch result actor != configured agent;
- dispatch result does not bind shared prior ledger;
- dispatch result does not bind configured decision model wrapper;
- duplicated agents/templates;
- existing world transition conflicts;
- existing projection/admission failures.

Every failure before world execution must prevent all world/projection/admission hooks from running.

## 13. Public API

Add exactly four narrative-scoped public names:

```text
RuntimeDecisionModelSpec
RuntimeDecisionDispatchResult
RuntimeDecisionDispatchError
run_runtime_decision
```

They are exported from:

```text
narrative_dynamics.narrative
```

They are not exported from package root:

```text
narrative_dynamics
```

The existing simulation public names remain public, but `RuntimeAgentSpec` and `SimulationAgentStep` receive the V2 field-type changes described above.

No family-specific public surface is removed.

## 14. Testing strategy

### 14.1 New dispatch tests

Add `tests/test_narrative_runtime_decision_dispatch.py` covering:

1. wrapper is frozen/canonical and binds tag + nested identity + dispatch implementation identity;
2. wrong tag/type pairs fail closed;
3. direct reactive dispatch embeds exact direct reactive result;
4. direct intentional dispatch embeds exact direct intentional result and derives actor/ledger from belief state;
5. direct planning dispatch embeds exact direct planning result;
6. common policy/selected action are exact, complete, normalized and lexical-MAP bound;
7. model-result/common-result forgery rejects;
8. decision-model-hash forgery rejects at runner boundary;
9. family failures preserve typed `__cause__`;
10. dispatch module has no world/simulation/observation-projection capability imports;
11. public runner signature excludes world/RNG/explicit step inputs.

### 14.2 Scheduler migration/regression tests

Extend `tests/test_narrative_simulation.py` to cover:

1. all existing intentional scheduler behavior migrated through `RuntimeDecisionModelSpec("intentional", ...)`;
2. behavioral regression against direct intentional family results;
3. mixed reactive/intentional/planning one-round execution against one prior ledger;
4. mixed two-round causal closure;
5. repeated mixed trajectory exact replay;
6. agent input ordering invariance with heterogeneous model kinds;
7. same-round shared prior ledger binding for all model kinds;
8. generic dispatch failure blocks world/projection/admission;
9. existing atomic world conflict rejection remains unchanged under mixed families;
10. simulation model identity changes when only one agent's model kind changes.

### 14.3 Trust API

Extend `tests/test_narrative_trust_api.py` with exactly the four new narrative-scoped dispatch exports and root isolation.

### 14.4 RED shape

The first implementation-bearing CI must be a test-only RED commit.

Expected RED causes should be limited to:

- missing `runtime_decision_dispatch` module/public types/runner;
- scheduler still requiring intentional model/result fields and direct intentional runner.

Existing unrelated tests must remain green.

Pure design and implementation-plan commits must not trigger CI under the current `proof.yml` `paths-ignore` configuration.

## 15. Implementation sequencing constraints

The implementation plan must preserve atomic TDD slices rather than landing one large scheduler rewrite.

Recommended slices:

1. test-only RED: generic dispatch contracts + scheduler mixed-family expectations + trust surface;
2. generic model wrapper/result records with runner stub;
3. closed three-family dispatcher + typed error normalization;
4. model-dependent dispatch result binding / forgery rejection;
5. migrate `RuntimeAgentSpec` and `SimulationAgentStep` to generic types;
6. switch scheduler execution to `run_runtime_decision` while preserving world/projection/admission logic;
7. heterogeneous one-round and multi-round science/runtime fixtures;
8. exact four-name narrative exports;
9. final exact-head proof.

No production slice may begin before authoritative test-only RED is established.

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

In particular, do not modify:

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

If implementation discovers that a family-specific runtime must change to support dispatch, stop and reclassify that change instead of silently expanding this feature.

## 17. Roadmap effect

After this feature, the engine supports heterogeneous cognitive agents in one deterministic multi-round world simulation:

```text
Reactive agent
Intentional agent
Planning agent
       |
       v
same-round shared snapshot
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

This closes the remaining scheduler integration gap between the three existing decision families.

It does not close issue #27 by itself. The remaining major roadmap work still includes:

- Conflict Resolution V2;
- Stochastic World / Observation Models;
- released identification and model-comparison experiments across reactive, intentional, and planning families.

## 18. Acceptance criteria

The feature is complete only when all of the following hold on one exact feature head:

- generic wrapper accepts exactly the three supported typed model families;
- generic dispatcher returns a provenance-bound common envelope with the exact nested family result;
- direct generic dispatch is probability/action-equivalent to each direct family runner;
- scheduler contains no family-specific decision runner call;
- one simulation can contain reactive, intentional, and planning agents simultaneously;
- every same-round agent result binds the same prior ledger hash;
- heterogeneous two-round execution closes without authored intermediate state injection;
- fixed heterogeneous inputs replay to the exact same trajectory hash;
- existing world conflict behavior remains fail-closed with no model-kind priority;
- root package remains unchanged;
- final diff stays within the declared feature scope;
- exact-head GitHub Actions proof completes successfully, including full Python and Lean gates.
