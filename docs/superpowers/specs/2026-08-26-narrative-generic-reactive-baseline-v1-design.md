# Narrative Generic Reactive Baseline V1 Design

## Status

Approved architecture design for issue #27 P1 Generic Reactive Baseline.

Integrated starting head:

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

Add a narrative-native reactive behavioral model family that maps only **currently observable cues** to a complete action probability policy, without constructing latent belief, goal, intention, memory, planning, or objective-world state.

Runtime mapping:

```text
current admitted cues
  -> sanitized reactive cue snapshot
  -> reactive score hook
  -> exact action scores
  -> shared finite softmax
  -> complete action policy
  -> deterministic lexical MAP action
```

This is the lower-complexity scientific baseline for later reactive-vs-intentional-vs-POMDP comparison.

## Architectural classification

This is an architectural change because it introduces a new runtime model family and hook capability boundary.

V1 is an isolated sidecar. It deliberately does **not** generalize Scheduler V1 to dispatch multiple decision-family types. Shared scheduler dispatch is deferred until both Reactive and Planning/POMDP family contracts are proven.

V1 must not change GenericNarrative IR, DomainSpec, authored decision/intention semantics, Runtime Cognition, Runtime Intention, World Transition, Observation Projection, Runtime Percept Admission, Scheduler V1, Lean sources, old prison adapters, evaluation infrastructure, or root package exports.

## Existing baseline

The runtime already provides:

```text
WorldStepResult
  -> Observation Projection
  -> Runtime Percept Admission
  -> RuntimeEvidenceLedger
```

Admitted evidence exposes sanitized percept semantics equivalent to:

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

The richer cognition path separately provides:

```text
authored epistemic seed + all runtime evidence history
  -> RuntimeEpistemicState
  -> RuntimeUncertainBeliefState
```

Reactive V1 must not call that cognition path. Missing current cues must remain missing rather than being filled from remembered belief state.

## Existing prison reactive model

`narrative_dynamics/adapters/prison_reactive.py` is a reference baseline, not the implementation base. It is prison-specific, derives cue utilities from scenario parameters, owns stochastic sampling, and does not consume narrative runtime observation provenance.

Reactive V1 preserves only its scientific principle:

> policy may respond to declared observable cues; hidden or future variables outside that cue capability cannot directly alter policy.

## Chosen module

Create:

```text
narrative_dynamics/narrative/runtime_reactive.py
```

It owns:

1. current-cue extraction,
2. sanitized hook-facing value types,
3. reactive model specification and identity,
4. score validation,
5. shared softmax projection,
6. deterministic lexical MAP selection,
7. typed fail-closed resolution.

No reactive logic is added to `runtime_cognition.py`, `runtime_intention.py`, or `simulation.py`.

## Rejected alternatives

### Reuse `DecisionModelSpec(EvidenceAccess.DIRECT_ONLY)`

Rejected. That boundary resolves at authored `logical_time` and returns a deterministic one-hot `DecisionResult`; it is not a runtime-step complete probability policy.

### Call `runtime_epistemic_state()` and ignore belief probabilities

Rejected. It merges evidence history and therefore introduces memory semantics.

### Pass `RuntimeEvidenceLedger` directly to the hook

Rejected. The ledger exposes history, provenance, source world hashes, and projection identities that a reactive hook must not condition on.

### Pass `WorldState` or `WorldStepResult`

Rejected. This leaks objective truth across the observation boundary.

### Sample actions in V1

Rejected. Sampling introduces an RNG/seed lineage owned by the later stochastic workstream. V1 emits a probability policy and deterministically selects lexical MAP.

### Generalize Scheduler V1 now

Rejected. The POMDP contract does not yet exist, so a common scheduler family protocol would be premature.

## Current-cue semantics

For runtime step `k`:

- `k == 0`: cues come only from authored **direct observation** for the decision actor at `ledger.source_at_time`;
- `k > 0`: cues come only from `ledger.batches[-1]` at the current step for that actor;
- earlier runtime batches are never searched to fill a missing cue;
- authored seed values are never used to fill missing runtime cues after step zero;
- missing current cues become `unknown`.

This rule is the defining separation from belief-state persistence.

### Blackout

A blackout is valid runtime state, not an execution error.

A declared cue with no current percept is:

```text
status = "unknown"
value = None
```

The hook may score unknown cues but receives no history API.

### Same-step semantic disagreement

If eligible rows for one actor/cell disagree in `(relation, value)`, resolution fails closed. No channel order, evidence order, last-write order, or provenance hash may choose a winner.

## Reactive model specification

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

- `model_id` and `version` are non-empty trimmed strings;
- supported decision types are non-empty, unique, and lexically sorted;
- `cue_cells` are non-empty, unique, and sorted by `(entity_type, entity_id, state_variable)`;
- at execution, every cue cell must belong to the authored decision template's `context_cells`;
- `beta` is finite and strictly positive;
- `score_hook` is callable.

### Canonical parameter values

`parameters` accept exactly recursively canonical values:

```text
None
bool
int
str
finite float
Mapping[str, canonical-value]
list[canonical-value]
tuple[canonical-value]
```

Rules:

- mapping keys are non-empty strings;
- mappings are frozen and serialized with lexical key order;
- list and tuple values are frozen to tuples internally;
- non-finite floats are rejected;
- bytes, sets, arbitrary objects, and callables are rejected as parameter values.

### Model identity

Model content identity binds:

```text
model id/version
supported decision types
cue cells
canonical parameters
beta
score hook measured implementation identity
RuntimeReactiveDecisionModelSpec implementation identity
finite_softmax implementation identity
```

The explicit `finite_softmax` identity is required because the shared numerical kernel lives outside `runtime_reactive.py`; changing that kernel must change reactive model identity.

The model does not declare belief parameters, goals, transition parameters, horizon, discount, or hidden-state support.

## `ReactiveCueView`

```python
ReactiveCueView(
    cell: StateCellRef,
    status: str,
    value: TypedValue | None,
    step_index: int,
)
```

Statuses are exactly:

```text
resolved
unknown
```

Rules:

- `resolved` requires a `TypedValue`;
- `unknown` requires `value is None`;
- `step_index` is a non-negative integer and not bool;
- no observer/channel/evidence/projection/world/provenance/ledger/belief/goal identity is exposed.

## `RuntimeReactiveCueSnapshot`

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

Requirements:

- actor/decision ids are non-empty trimmed strings;
- ledger hash is a canonical `sha256:` content hash;
- cue keys are canonical and unique;
- every cue view binds the snapshot step;
- cue serialization uses canonical cell order.

The ledger hash binds which admitted-evidence state produced the sanitized cues without exposing the ledger to the hook.

## `RuntimeReactiveDecisionContext`

The score hook receives exactly:

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

- `actions` are the exact authored action ids sorted lexically;
- action ids are non-empty and unique;
- cue keys equal model `cue_cells` exactly;
- every cue binds the same step;
- parameters are the already-frozen model parameters;
- all values are immutable before crossing the hook boundary.

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

Conceptual signature:

```python
def score_hook(
    context: RuntimeReactiveDecisionContext,
) -> Mapping[str, float]:
    ...
```

Framework validation requires:

- return value is a mapping;
- keys equal the authored action ids exactly;
- every score is numeric, finite, and not bool;
- missing and extra actions fail closed.

The hook cannot return a selected action or policy.

Any exception derived from `Exception` during score-hook invocation is normalized to `RuntimeReactiveDecisionResolutionError` with the original exception retained as `__cause__`. `BaseException` subclasses are not caught.

## Shared softmax numerical semantics

Reactive V1 reuses:

```python
from grounded_goal_softmax import finite_softmax
```

This is the same numerical softmax kernel already used by authored/runtime intentional action choice. Reactive V1 does **not** reimplement exponentiation/normalization.

Call:

```python
raw_policy = finite_softmax(action_scores, beta=model.beta)
```

Then validate the returned policy independently.

Requirements:

- exact authored action set;
- finite non-negative probabilities;
- `math.fsum(policy.values())` sums to one within absolute tolerance `1e-12` and zero relative tolerance;
- canonical lexical action ordering;
- invalid numerical output fails typed.

If the shared softmax implementation raises `TypeError`, `ValueError`, or `OverflowError`, wrap it as typed reactive resolution failure.

## Deterministic action projection

V1 consumes no RNG.

Selected action is:

```text
maximum policy probability
then lexical action id as exact tie-break
```

## `RuntimeReactiveDecisionResult`

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

Self-validation requires:

- valid model/content hash shape;
- decision/actor ids valid;
- step equals cue snapshot step;
- ledger hash equals cue snapshot ledger hash;
- cue snapshot hash equals `cue_snapshot.content_hash`;
- score/policy keys match exactly;
- policy is a complete validated simplex;
- selected action belongs to policy;
- selected action equals deterministic lexical MAP.

The result contains no belief, goal, posterior, value function, transition model, or planning tree.

## Public execution function

```python
run_runtime_reactive_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeReactiveDecisionModelSpec,
) -> RuntimeReactiveDecisionResult
```

Fixed execution order:

1. validate story/domain;
2. validate exact ledger domain/story identity;
3. reconstruct/revalidate ledger invariants before hook execution;
4. resolve canonical authored decision template;
5. validate supported decision type;
6. validate cue cells are a subset of decision `context_cells`;
7. validate non-empty unique authored actions;
8. enforce `decision.logical_time <= ledger.source_at_time` when cutoff is numeric;
9. construct current cue snapshot;
10. construct sanitized hook context;
11. invoke score hook;
12. validate exact action scores;
13. call shared `finite_softmax`;
14. validate complete policy simplex;
15. select lexical MAP;
16. construct self-validating result.

No step may call Runtime Cognition or read objective world state.

## Typed error boundary

Public resolution failures normalize to:

```python
RuntimeReactiveDecisionResolutionError
```

Fail closed on malformed/mismatched story, domain, ledger, model, decision template, cue capability, current evidence, hook result, numerical policy, or result construction.

Malformed prerequisites must fail before the score hook runs.

The public function retains original underlying exceptions as `__cause__` where normalization occurs.

## Step-zero direct seed

At step zero, use existing authored **direct** replay for the decision actor at `ledger.source_at_time`.

For each cue cell:

- resolved direct cell -> `resolved(value)`;
- unresolved/missing direct cell -> `unknown`.

Do not use authored epistemic/testimony aggregation.

## Step-positive extraction

At `k > 0`, use only:

```text
ledger.batches[-1]
```

Require:

```text
ledger.batches[-1].step_index == ledger.current_step_index == k
```

Filter evidence to:

```text
item.observer_id == decision.actor_id
```

For each declared cue cell:

- no eligible row -> `unknown`;
- all eligible rows agree on `equals(value)` -> `resolved(value)`;
- all eligible rows agree on `clear` -> `unknown`;
- semantic disagreement -> typed failure.

Channel identity is not exposed to the hook in V1. Multiple channels may corroborate one semantic cue.

## Scientific model-family distinction

Reactive:

\[
A_k \sim \pi(C_k)
\]

where `C_k` contains only current cues.

Intentional:

\[
L_{0:k}\to B_k\to G_k\to \pi(A_k)
\]

Future POMDP:

\[
B_k\to \text{future transition/observation enumeration}\to Q(B_k,A)\to \pi(A_k)
\]

Reactive V1 must remain unable to emulate richer families by receiving hidden history or future-state parameters through its runtime capability.

## Observational-equivalence fixture

Tests must include a synthetic case where current cue information fully reveals what drives an intentional model.

To make numerical equality rigorous rather than approximate:

- configure the intentional choice layer so its conditional action values equal the reactive hook action scores;
- use identical `beta_action == reactive.beta`;
- construct the intentional fixture so the same conditional action policy is used under every supported goal, making marginalization preserve that policy exactly;
- both families therefore call the same `finite_softmax` implementation over the same ordered score mapping.

Require exact equality of the final action probability mapping and selected action.

Purpose:

> prove richer latent cognition is not identifiable when current cues already fully explain behavior.

## Separating information intervention fixture

Tests must include paired runtime states with identical objective world state and authored action space but different observation access.

Required pattern:

```text
prior step: informative cue admitted
current step A: cue visible
current step B: cue blackout
```

Reactive at B must see current cue as `unknown` because it cannot consult prior batches. Intentional cognition may retain/use prior admitted evidence through its history-driven belief state.

Require a full action-policy difference attributable only to observation access; world state and decision template remain unchanged.

Purpose:

> distinguish memoryless cue reaction from belief-based behavior.

## Evaluation compatibility

Reactive V1 emits a complete probability simplex compatible with existing categorical Brier/log losses and preregistered held-out comparison.

V1 does not modify evaluation infrastructure. It only guarantees sufficient result coordinates for later adapters:

```text
decision id
action ids
action scores
action policy
model identity
runtime step identity
```

## Public API

Exactly seven names are added to `narrative_dynamics.narrative`:

```text
ReactiveCueView
RuntimeReactiveCueSnapshot
RuntimeReactiveDecisionContext
RuntimeReactiveDecisionModelSpec
RuntimeReactiveDecisionResult
RuntimeReactiveDecisionResolutionError
run_runtime_reactive_decision
```

Root `narrative_dynamics` exports remain unchanged. No helper/extraction/validation function is public.

## Module isolation

`runtime_reactive.py` must not import:

```text
narrative_dynamics.narrative.runtime_cognition
narrative_dynamics.narrative.runtime_intention
narrative_dynamics.narrative.intention
narrative_dynamics.narrative.world
```

It may import:

```text
grounded_goal_softmax.finite_softmax
narrative_dynamics.attestation
narrative_dynamics.contracts
narrative_dynamics.narrative.domain
narrative_dynamics.narrative.ir
narrative_dynamics.narrative.replay
narrative_dynamics.narrative.runtime_perception
```

Source-inspection tests lock these exclusions.

## Determinism and identity

Fixed story/domain/decision/ledger/model inputs must produce exactly equal result and content hash.

Canonicalization rules:

- supported decision types lexical;
- cue cells by entity type/id/state variable;
- action ids lexical;
- parameter mapping serialization lexical;
- score/policy mappings lexical;
- lexical MAP tie-break.

No RNG, wall clock, hash iteration order, or mutable external state may affect results.

## Forgery resistance

The public boundary reconstructs or revalidates public values where constructor-bypassing mutation could violate invariants.

Tests must forge and reject at minimum:

- ledger story hash;
- ledger domain identity;
- ledger current step/batch consistency;
- cue snapshot ledger hash;
- cue snapshot step;
- result cue snapshot hash;
- result selected-action/policy consistency.

Prerequisite forgeries fail before score-hook execution.

## Test architecture

Create:

```text
tests/test_narrative_runtime_reactive.py
```

Extend:

```text
tests/test_narrative_trust_api.py
```

Semantic suite locks at least:

1. model validation and stable identity;
2. exact canonical parameter acceptance/rejection;
3. step-zero direct cue extraction;
4. step-positive current-batch-only extraction;
5. blackout -> unknown without history fill;
6. agreeing channels collapse semantically;
7. same-step disagreement typed failure;
8. hook receives sanitized context only;
9. no ledger/world/belief/goal objects cross hook boundary;
10. exact action-score coverage;
11. non-finite score rejection;
12. shared `finite_softmax` complete simplex;
13. softmax implementation identity affects model identity;
14. lexical MAP tie-break;
15. exact replay/result hash;
16. forged source/ledger identity rejected before hook;
17. cue outside decision context rejected;
18. decision template after cutoff rejected;
19. observational-equivalence exact policy match;
20. information-access intervention separates reactive/intentional policy;
21. production module import isolation;
22. exact seven-name narrative export and unchanged root isolation.

## Scope constraints

Production code is limited to:

```text
narrative_dynamics/narrative/runtime_reactive.py
narrative_dynamics/narrative/__init__.py
```

plus design, implementation plan, and tests.

Do not modify:

```text
GenericNarrative IR
DomainSpec
decision.py
intention.py
runtime_cognition.py
runtime_intention.py
runtime_perception.py
world.py
observation_projection.py
simulation.py scheduler behavior
root narrative_dynamics exports
Lean source
prison reactive/POMDP adapters
model_comparison.py
loss implementations
```

## TDD discipline

Required sequence:

```text
design -> plan -> test-only RED -> minimal GREEN slices -> export GREEN -> exact-head proof
```

The first executable feature commit after the plan is test-only RED with no production implementation. RED must leave prior Python/Lean gates green except failures attributable to the absent reactive contract.

Recommended GREEN slices:

1. value types/model spec;
2. cue extraction/trust boundary;
3. score/policy/action resolution;
4. scientific equivalence/intervention fixtures;
5. narrative-scoped exports.

Each slice is atomic and independently CI-verifiable.

## Acceptance criteria

V1 completes only when:

- generic narrative runtime reactive model exists independent of prison adapters;
- hook sees only declared current cues and immutable canonical parameters;
- step > 0 never fills missing current cues from history;
- no belief/goal/intentional/planning/objective-world object crosses the hook boundary;
- every result exposes a complete policy over the exact authored action set;
- reactive and intentional choice use the same measured `finite_softmax` numerical kernel;
- selected action is deterministic lexical MAP;
- fixed inputs replay to exact result/content hash;
- typed fail-closed validation rejects malformed/forged capability inputs;
- observational-equivalence fixture matches intentional policy exactly;
- information-access intervention separates the families without world-state change;
- exactly seven narrative-scoped names are added and root exports remain unchanged;
- all protected existing modules remain unchanged;
- full Python and Lean proof workflow passes on exact final head.

## Follow-on boundary

After Reactive V1 is merged, issue #27 proceeds to Generic Planning/POMDP Integration.

Shared scheduler family dispatch is designed only after both Reactive and POMDP contracts are proven.