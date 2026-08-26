# Narrative Generic Reactive Baseline V1 Design

## Status

Approved architecture design for issue #27 P1 Generic Reactive Baseline.

This design starts from the integrated Multi-Step Scheduler V1 research head:

```text
045f5b0602e30e8ec12e7c9fa73128e28cabebd8
```

Research base:

```text
proof/narrative-dynamics-v0
```

Feature branch:

```text
work/narrative-generic-reactive-baseline-v1
```

## Goal

Add a narrative-native reactive behavioral model family that maps only currently observable cues to a complete action probability policy, without constructing latent belief, goal, intention, or planning state.

The model exists as a scientifically meaningful lower-complexity baseline for later comparison against intentional and POMDP families.

The core runtime mapping is:

```text
Current admitted observation cues
    -> Reactive cue snapshot
    -> Reactive score hook
    -> Action scores
    -> Softmax action policy
    -> Deterministic lexical MAP action
```

The defining non-feature is equally important:

```text
No posterior belief
No goal state
No latent memory state
No future-state planning
No objective WorldState access
```

## Architectural classification

This is an architectural change because it adds a new model family and a new runtime decision capability boundary whose public result must later participate in cross-family model comparison.

It does not change GenericNarrative IR, DomainSpec, authored decision semantics, Runtime Cognition, Intentional Decision, World Transition, Observation Projection, Runtime Percept Admission, Scheduler V1, Lean sources, or root package exports.

V1 deliberately does not generalize Scheduler V1 to dispatch multiple decision-family types. Reactive V1 is first established as an isolated runtime sidecar. A shared scheduler dispatch boundary is deferred until both Reactive and Planning/POMDP family contracts exist.

## Existing baseline

The integrated runtime currently provides:

```text
WorldStepResult
    -> Observation Projection
    -> Runtime Percept Admission
    -> RuntimeEvidenceLedger
```

Each admitted percept contains provenance-bound runtime evidence, and each evidence row exposes a sanitized:

```python
RuntimePerceptView(
    observer_id,
    channel,
    cell,
    relation,
    value,
    step_index,
)
```

The same runtime also provides richer cognition:

```text
Authored epistemic seed
    + all admitted runtime evidence history
    -> RuntimeEpistemicState
    -> RuntimeUncertainBeliefState
```

Reactive V1 must not call that cognition path. Otherwise missing current cues could be silently filled by remembered belief state, collapsing the distinction between a reactive model and an intentional/belief model.

## Existing prison reactive model is a reference, not the implementation base

`narrative_dynamics/adapters/prison_reactive.py` is scientifically useful evidence that the repository already values a lower-complexity cue-reactive comparator. It is not generic enough to become the narrative model family because it:

- reads prison-specific scenario parameters,
- derives cue utilities from those parameters,
- owns stochastic RNG sampling,
- is not driven by narrative observation provenance,
- does not use the runtime evidence ledger,
- cannot be reused across arbitrary GenericNarrative domains.

Reactive V1 preserves its important scientific property:

> policy responds to declared observable cues, while latent/future variables that are not present in the cue capability cannot directly change the policy.

## Chosen architecture

Add one new sidecar module:

```text
narrative_dynamics/narrative/runtime_reactive.py
```

It owns:

1. reactive cue extraction from the runtime evidence ledger,
2. the sanitized hook-facing cue context,
3. reactive model specification and attested identity,
4. score validation,
5. framework-owned softmax policy construction,
6. deterministic lexical MAP action selection,
7. typed fail-closed runtime resolution.

No reactive logic is added to `runtime_cognition.py` or `runtime_intention.py`.

## Rejected alternatives

### Rejected: reuse authored `DecisionModelSpec(EvidenceAccess.DIRECT_ONLY)`

The authored decision boundary resolves at authored `logical_time` and returns a deterministic one-hot `DecisionResult`. It is not a runtime-step model and does not produce the complete probability simplex needed for proper-scoring comparison.

### Rejected: call `runtime_epistemic_state()` and ignore belief probabilities

`runtime_epistemic_state()` intentionally merges authored seed state with the full runtime evidence history. That creates memory semantics. A reactive model must treat a currently unavailable cue as unavailable, not as remembered state.

### Rejected: pass `RuntimeEvidenceLedger` directly to the score hook

The ledger contains provenance identities, historical batches, source world hashes, and observation-model identities. A hook that receives it could condition on undeclared history or provenance metadata. The hook receives only a sanitized cue snapshot.

### Rejected: pass `WorldState` or `WorldStepResult` to reactive selection

This would let the baseline read objective truth and destroy the observation-capability distinction needed for information-intervention identification.

### Rejected: sample actions in V1

Stochastic action sampling would introduce a new RNG/seed lineage problem that issue #27 assigns to a later stochastic workstream. V1 emits a probability policy and selects its action deterministically using lexical MAP.

### Rejected: generalize Scheduler V1 now

`RuntimeAgentSpec` is intentionally intentional-specific. Generalizing it before the POMDP family exists would force a premature common protocol. Reactive and POMDP first establish their own semantics; scheduler dispatch can then be extracted from proven common output requirements.

## Reactive observational semantics

Reactive V1 uses a strict **current-cue** semantics.

For an agent at runtime step `k`:

- if `k == 0`, cues come only from the authored direct-observation seed available to the decision actor at the simulation source cutoff;
- if `k > 0`, cues come only from the latest admitted runtime evidence batch whose `step_index == k` and whose `observer_id` equals the decision actor;
- earlier runtime batches are not searched to fill missing current cues;
- authored seed values are not used to fill a missing runtime cue after step zero;
- an expected cue that is unavailable at the current step becomes `unknown` rather than inheriting a prior value.

This is the scientific boundary that separates reactive behavior from belief-state persistence.

### Blackout semantics

A projection blackout is valid runtime evidence about observation access, not an execution error.

If a declared reactive cue has no current percept, its cue view is:

```text
status = "unknown"
value = None
```

The score hook may assign action scores for an unknown cue, but cannot recover prior values through any history API.

### Same-step conflict semantics

If the current batch contains more than one semantic row for the same actor/cell and those rows disagree in `(relation, value)`, reactive resolution fails closed.

The implementation must not choose a winner by channel order, evidence order, last-write order, or provenance hash.

## Cue declaration and capability

The model declares exactly which narrative cells it may inspect.

```python
RuntimeReactiveDecisionModelSpec(
    model_id: str,
    version: str,
    supported_decision_types: tuple[str, ...],
    cue_cells: tuple[StateCellRef, ...],
    parameters: Mapping[str, object],
    beta: float,
    score_hook: object,
)
```

Requirements:

- `model_id` and `version` are non-empty trimmed strings,
- supported decision types are non-empty, unique, and canonically ordered,
- `cue_cells` are non-empty, unique `StateCellRef` values in canonical cell order,
- every cue cell must be one of the authored decision template's `context_cells` when the model runs,
- `parameters` are recursively frozen canonical values,
- `beta` is finite and strictly positive,
- `score_hook` is callable and its measured implementation identity is part of model identity,
- model content identity binds supported decision types, cue cells, parameters, beta, and hook implementation identity.

The model does not declare belief parameters, goal parameters, transition parameters, horizon, discount, or hidden-state support.

## `ReactiveCueView`

The hook-facing atomic cue view is:

```python
ReactiveCueView(
    cell: StateCellRef,
    status: str,
    value: TypedValue | None,
    step_index: int,
)
```

Supported status values are exactly:

```text
resolved
unknown
```

Rules:

- `resolved` requires a `TypedValue`,
- `unknown` requires `value is None`,
- the view contains no observer id, channel, evidence hash, projection hash, world hash, provenance refs, ledger object, belief state, or goal state.

The decision actor and decision identity are represented by the containing context rather than duplicated into each cue.

## `RuntimeReactiveDecisionContext`

The score hook receives exactly one sanitized value object:

```python
RuntimeReactiveDecisionContext(
    decision_id: str,
    actor_id: str,
    decision_type: str,
    step_index: int,
    actions: tuple[str, ...],
    cues: Mapping[StateCellRef, ReactiveCueView],
    parameters: Mapping[str, object],
)
```

Requirements:

- actions are the canonical authored action ids for the selected decision template,
- actions are non-empty and unique,
- cue keys equal the model's declared `cue_cells` exactly,
- every cue view binds the same `step_index`,
- context values are immutable/frozen before crossing the hook boundary.

The context intentionally excludes:

```text
GenericNarrative
DomainSpec
RuntimeEvidenceLedger
RuntimeEvidenceBatch
RuntimeEpistemicEvidence
WorldState
WorldStepResult
RuntimeEpistemicState
RuntimeUncertainBeliefState
GoalState
ChoiceModelSpec
```

## Score hook contract

The hook signature is conceptually:

```python
def score_hook(
    context: RuntimeReactiveDecisionContext,
) -> Mapping[str, float]:
    ...
```

The returned mapping is interpreted only as action scores.

Framework validation requires:

- mapping keys equal the authored decision action ids exactly,
- every score is numeric, finite, and not boolean,
- no missing action,
- no undeclared action,
- the hook cannot provide its own selected action or probability policy.

This keeps probability semantics framework-owned and comparable across reactive models.

## Framework-owned softmax

For validated action scores `s(a)` and declared inverse temperature `beta`:

\[
P(a) = \frac{\exp(\beta(s(a)-m))}{\sum_b \exp(\beta(s(b)-m))}
\]

where:

\[
m = \max_b s(b)
\]

The implementation must use numerically stable max-shifted exponentiation.

The resulting policy must:

- contain exactly the authored action set,
- contain finite non-negative probabilities,
- sum to one within the repository probability tolerance,
- have deterministic canonical key ordering.

If numerical construction cannot produce a valid simplex, resolution fails closed.

## Deterministic action projection

V1 does not sample from the action policy.

The selected action is:

```text
maximum policy probability
then lexical action id as the exact tie-break
```

This matches the deterministic replay guarantees of the current narrative runtime while preserving the full probability policy for later proper-scoring evaluation.

## `RuntimeReactiveDecisionResult`

The result is:

```python
RuntimeReactiveDecisionResult(
    model_id: str,
    model_hash: str,
    decision_id: str,
    actor_id: str,
    step_index: int,
    ledger_hash: str,
    cue_snapshot_hash: str,
    cue_snapshot: RuntimeReactiveCueSnapshot,
    action_scores: Mapping[str, float],
    action_policy: Mapping[str, float],
    selected_action: str,
)
```

The result must validate its own internal consistency:

- model id/hash are valid and immutable,
- decision/actor ids are valid,
- step index equals cue snapshot step,
- ledger hash binds the exact source ledger,
- cue snapshot hash equals the exact sanitized snapshot content hash,
- action score and policy keys match exactly,
- selected action belongs to the policy,
- selected action equals deterministic lexical MAP,
- action policy is a complete probability simplex.

The result contains no belief state, goal state, posterior distribution, value function, transition model, or planning tree.

## `RuntimeReactiveCueSnapshot`

Cue extraction is represented explicitly rather than hidden inside the result.

```python
RuntimeReactiveCueSnapshot(
    actor_id: str,
    decision_id: str,
    step_index: int,
    ledger_hash: str,
    cues: Mapping[StateCellRef, ReactiveCueView],
)
```

The snapshot is immutable and content-hashed.

The ledger hash proves which admitted-evidence state produced the sanitized cues without exposing the ledger to the hook.

## Public execution function

The public execution boundary is:

```python
run_runtime_reactive_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeReactiveDecisionModelSpec,
) -> RuntimeReactiveDecisionResult
```

Execution order is fixed:

1. validate story/domain identity,
2. validate exact ledger story/domain identity,
3. resolve canonical authored decision template,
4. validate supported decision type,
5. validate model cue cells are a subset of decision `context_cells`,
6. validate decision action ids are non-empty and unique,
7. enforce source-cutoff/template logical-time compatibility,
8. construct current reactive cue snapshot,
9. construct sanitized hook context,
10. invoke score hook,
11. validate exact action score schema,
12. construct framework softmax policy,
13. select deterministic lexical MAP action,
14. construct and return the self-validating result.

No step may call Runtime Cognition or read objective world state.

## Typed failure boundary

Public resolution failures are normalized to:

```python
RuntimeReactiveDecisionResolutionError
```

The public function fails closed on:

- invalid story/domain/model/ledger types,
- forged or mismatched ledger identity,
- undeclared decision id,
- unsupported decision type,
- decision template after source cutoff,
- duplicate/empty decision actions,
- cue not declared by decision context,
- malformed current evidence,
- same-step semantic cue conflict,
- score hook exception from accepted validation families,
- hook returning wrong type,
- missing/extra/non-finite action scores,
- invalid softmax simplex,
- invalid result construction.

The original exception is retained as `__cause__` for diagnostics.

## Step-zero authored direct-cue seed

At runtime step zero, no admitted runtime batch exists. Reactive V1 therefore needs a direct-observation seed that still respects capability.

The implementation uses the existing authored direct replay semantics for the decision actor at `ledger.source_at_time`.

For each declared cue cell:

- if direct replay resolves that cell, the reactive cue is `resolved` with that value,
- otherwise it is `unknown`.

Reactive V1 must not use authored epistemic/testimony aggregation for this seed. The baseline is immediate cue response, not belief inference.

## Step-positive current-batch extraction

At step `k > 0`, the current cue snapshot is built only from:

```text
ledger.batches[-1]
```

with required consistency:

```text
ledger.batches[-1].step_index == ledger.current_step_index == k
```

Only rows with:

```text
item.observer_id == decision.actor_id
```

are eligible.

For each declared cue cell:

- no eligible row -> `unknown`,
- all eligible rows agree on `equals(value)` -> `resolved(value)`,
- all eligible rows agree on `clear` -> `unknown`,
- semantic disagreement -> typed failure.

Channel identity is intentionally not exposed to the hook in V1. Multiple channels may corroborate one semantic cue, but channel-specific policies are a separate feature.

## Scientific model-family distinction

Reactive V1 is intentionally nested below the intentional family in representational complexity.

Reactive:

\[
A_k \sim \pi(C_k)
\]

where `C_k` is only the current observable cue snapshot.

Intentional:

\[
L_{0:k}
\to B_k
\to G_k
\to \pi(A_k)
\]

where admitted evidence history may update latent belief and goal state.

Future POMDP:

\[
B_k
\to \text{future transition/observation enumeration}
\to Q(B_k,A)
\to \pi(A_k)
\]

Reactive V1 must remain unable to emulate those richer families by receiving hidden history or future-state parameters through its runtime capability.

## Observational-equivalence fixture

V1 tests must include a synthetic case where current cue information fully reveals the information that drives an intentional model and both families are configured to implement the same action score transformation.

The test requires exact equality of the final action probability policy, not merely equality of the selected MAP action.

Purpose:

> prove that richer latent cognition is not automatically identifiable when observable behavior is fully explained by current cues.

## Separating information intervention fixture

V1 tests must include a pair of simulations with identical objective world state and authored action space but different observation access.

Required separation pattern:

```text
full current cue:
    reactive policy = policy_from_current_cue
    intentional policy = policy_from_belief

current cue blackout after prior evidence:
    reactive sees unknown current cue
    intentional may retain/update belief from evidence history
```

The test must demonstrate a policy difference attributable to information access rather than a changed world state.

Purpose:

> provide an intervention that can distinguish a memoryless reactive model from a belief-based intentional model.

## Action-policy comparison semantics

Reactive V1 must produce a full action probability simplex because the repository already supports categorical Brier and categorical log loss.

This design does not add a new evaluation subsystem. Later identification/model-comparison work may adapt reactive and intentional decision policies into the existing held-out/preregistered evaluation layer.

V1 only guarantees that its result is sufficient for that adapter:

```text
decision id
action ids
action scores
action probability policy
model identity
runtime step identity
```

## Public API scope

New symbols are exported only from:

```text
narrative_dynamics.narrative
```

Root `narrative_dynamics` exports remain unchanged.

The intended new narrative-scoped public surface is exactly:

```text
ReactiveCueView
RuntimeReactiveCueSnapshot
RuntimeReactiveDecisionContext
RuntimeReactiveDecisionModelSpec
RuntimeReactiveDecisionResult
RuntimeReactiveDecisionResolutionError
run_runtime_reactive_decision
```

No helper functions, softmax helpers, extraction helpers, or internal validation types are public.

## Module isolation constraints

`runtime_reactive.py` must not import:

```text
narrative_dynamics.narrative.runtime_cognition
narrative_dynamics.narrative.runtime_intention
narrative_dynamics.narrative.intention
narrative_dynamics.narrative.world
```

It may import:

```text
attestation/contracts
domain/IR
replay direct-state capability
runtime_perception value/evidence types
```

Tests must source-inspect the module and lock these import exclusions.

This is not merely aesthetic isolation. It prevents accidental use of belief, goal, intentional, or objective-world semantics inside the baseline.

## Determinism and identity

For fixed:

```text
story
domain
decision template
runtime ledger
reactive model spec
```

`run_runtime_reactive_decision(...)` must produce an exactly equal result and content hash on replay.

Canonicalization rules include:

- supported decision types sorted,
- cue cells sorted by entity type/entity id/state variable,
- action ids canonicalized from authored declaration into a deterministic order,
- parameter mappings recursively key-sorted for identity payloads,
- score/policy mappings emitted in canonical action order,
- lexical tie-break for selected action.

No process RNG, wall clock, hash randomization order, or mutable external state may affect the result.

## Trust and forgery resistance

The public boundary must reconstruct or revalidate supplied public dataclasses where constructor-bypassing mutation could otherwise bypass invariants.

At minimum tests must forge and reject:

- ledger story hash,
- ledger domain identity,
- ledger current step/batch consistency,
- cue snapshot ledger hash,
- cue snapshot step index,
- result cue snapshot hash,
- result policy selected-action consistency.

The implementation must fail before the score hook runs when source/model/ledger prerequisites are invalid.

## Test architecture

Add one dedicated semantic suite:

```text
tests/test_narrative_runtime_reactive.py
```

and extend:

```text
tests/test_narrative_trust_api.py
```

The semantic suite must lock at least these contracts:

1. model spec validation and stable content identity,
2. step-zero direct cue extraction,
3. step-positive current-batch-only extraction,
4. blackout becomes unknown without history fill,
5. multiple agreeing channels collapse to one semantic cue,
6. same-step disagreement fails typed,
7. score hook receives only sanitized context,
8. hook cannot access ledger/world/belief/goal objects,
9. exact action score coverage is required,
10. non-finite scores fail typed,
11. stable softmax produces exact complete simplex,
12. lexical MAP tie-break is deterministic,
13. same inputs replay to exact result hash,
14. forged ledger/source identity fails before hook execution,
15. cue cells outside decision context fail closed,
16. authored decision template after cutoff fails closed,
17. observational-equivalence fixture matches intentional action policy exactly,
18. information-access intervention separates reactive from intentional policy,
19. production source does not import runtime cognition/intention/world,
20. exact narrative public surface gains only the seven approved names and root isolation remains unchanged.

## Scope constraints

V1 must not modify:

```text
GenericNarrative IR
DomainSpec
authored decision.py
authored intention.py
runtime_cognition.py
runtime_intention.py
world.py
observation_projection.py
runtime_perception.py
simulation.py scheduler behavior
root narrative_dynamics exports
Lean source
prison reactive/POMDP adapters
model_comparison.py
categorical loss implementations
```

The expected production code diff is limited to:

```text
narrative_dynamics/narrative/runtime_reactive.py
narrative_dynamics/narrative/__init__.py
```

plus design/plan and tests.

## TDD and commit discipline

Implementation must continue the repository's strict workflow:

```text
design -> plan -> test-only RED -> minimal GREEN slices -> exact public export GREEN -> exact-head proof
```

The first executable feature commit after the plan is test-only RED. It must not contain production implementation.

The RED run must demonstrate that prior tests/Lean gates remain green and that failures are attributable only to the absent reactive family contract.

Subsequent GREEN slices should separate:

1. reactive value types/model spec,
2. cue extraction and trust boundary,
3. score/policy/action resolution,
4. scientific equivalence/intervention fixtures,
5. narrative-scoped exports.

Each slice is atomic and independently CI-verifiable.

## Acceptance criteria

Reactive Baseline V1 is complete only when all of the following hold:

- a generic narrative runtime reactive model exists independent of prison adapters,
- the hook sees only declared current observable cues and immutable parameters,
- runtime step > 0 never fills missing current cues from prior evidence history,
- no belief, goal, intentional, planning, or objective-world object crosses the reactive hook boundary,
- every result contains a complete deterministic probability policy over the exact authored action set,
- selected action is deterministic lexical MAP,
- same inputs reproduce the exact result/content hash,
- typed fail-closed validation rejects malformed/forged capability inputs,
- an observational-equivalence case shows reactive and intentional policies can coincide,
- an information-access intervention case separates the two families without changing objective world state,
- exactly seven approved names are added to `narrative_dynamics.narrative`,
- no names are added at package root,
- GenericNarrative IR, DomainSpec, Lean, world, perception, cognition, intention, scheduler, old adapters, and evaluation infrastructure remain unchanged,
- full Python and Lean proof workflow is green on the exact final head.

## Follow-on boundary

After this V1 is merged, issue #27 proceeds to Generic Planning/POMDP Integration.

Only after both Reactive and POMDP family outputs are proven should a later design generalize Scheduler dispatch around a shared decision-family result protocol. That later protocol should be derived from observed commonality, not guessed in this V1.
