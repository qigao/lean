# Narrative Generic Planning / POMDP V1 Design

Date: 2026-08-27
Status: approved design, pre-implementation
Tracking issue: #27 — P1 Generic Planning / POMDP Integration
Integrated base: `proof/narrative-dynamics-v0` at `8aae284fe02f27a76b6d00a651be152f49a9e6ed`

## 1. Objective

Add a narrative-native finite-horizon planning model family that can value future information and future state transitions without reading the actual runtime `WorldState` or reusing the production world-transition engine as an oracle.

The model must expose the same comparison-relevant decision boundary as the existing reactive and intentional families:

```text
complete action policy + deterministic lexical MAP action
```

while making its additional assumptions explicit and auditable:

```text
runtime posterior semantics
  -> declared joint hidden-state belief
  -> hypothetical transition model
  -> hypothetical observation model
  -> Bayesian belief update
  -> finite-horizon value recursion
  -> shared finite_softmax
  -> complete action policy
  -> lexical MAP action
```

V1 is an exact finite reference implementation. It is deterministic and uses no RNG.

## 2. Non-goals

This feature does not:

- generalize `SimulationModelSpec`, `RuntimeAgentSpec`, or scheduler dispatch;
- call the real `WorldTransitionModelSpec`, `ObservationProjectionModelSpec`, or `WorldState` during planning;
- modify `GenericNarrative`, `DomainSpec`, authored decisions, or action IR;
- replace or refactor the existing prison-specific POMDP adapter;
- introduce stochastic runtime execution or seeded sampling;
- change the released held-out/model-comparison infrastructure;
- add Conflict Resolution V2 or stochastic world/observation semantics;
- add Lean sources or theorem obligations.

Scheduler support for multiple decision model families is a later architectural change after the planning contract is stable.

## 3. Existing constraints

The integrated code already supplies the boundaries this design must preserve:

1. `RuntimeEvidenceLedger` is the admitted runtime information history.
2. `runtime_uncertain_belief_state(...)` converts that history into finite posterior semantics without objective-state leakage.
3. `run_runtime_intentional_decision(...)` already demonstrates a public runtime decision signature that excludes `WorldState` and explicit step input.
4. `run_runtime_reactive_decision(...)` establishes a narrative-native lower-complexity sidecar and binds the shared `grounded_goal_softmax.finite_softmax` implementation.
5. Scheduler V1 is intentionally typed to runtime intentional decisions. This feature does not broaden that type.
6. Plain Python function implementation attestation is now supported and can bind the four planning hooks plus shared softmax implementation bytes.

## 4. Rejected approaches

### 4.1 Roll out the real runtime world model

Rejected.

Passing `WorldState`, `WorldTransitionModelSpec`, or `ObservationProjectionModelSpec` into the planner would make the production objective state reachable from a belief-based model. Even if the current implementation used those objects carefully, the capability boundary would be wrong and future model hooks could turn the real world into an oracle.

Hypothetical planning state must therefore be a separate declared finite state space.

### 4.2 Generalize the prison POMDP into scalar/string state

Rejected.

The prison adapter proves finite POMDP behavior, but its hidden state and signals are scenario-specific scalar values. Promoting that representation would create a second opaque state language beside `StateCellRef` and `TypedValue`.

Planning state and observation atoms must instead be narrative-grounded typed data.

### 4.3 Generalize scheduler dispatch in the same feature

Rejected.

Planning integration and scheduler polymorphism are independent architectural changes. Coupling them would make failures ambiguous and would force scheduler API choices before the planning result contract is stable.

V1 therefore adds an isolated decision sidecar only.

## 5. Module boundary

Create:

```text
narrative_dynamics/narrative/runtime_planning.py
```

The module may depend on:

```text
narrative_dynamics.attestation
narrative_dynamics.contracts
narrative_dynamics.grounded_goal_softmax
narrative_dynamics.narrative.domain
narrative_dynamics.narrative.ir
narrative_dynamics.narrative.runtime_cognition
narrative_dynamics.narrative.runtime_perception
narrative_dynamics.narrative.uncertain
```

The module must not import:

```text
narrative_dynamics.narrative.world
narrative_dynamics.narrative.observation_projection
narrative_dynamics.narrative.simulation
narrative_dynamics.narrative.runtime_intention
narrative_dynamics.narrative.runtime_reactive
```

The import ban is a tested capability boundary, not merely a style rule.

## 6. Narrative-grounded finite state and observation spaces

### 6.1 Planning hidden state

```python
@dataclass(frozen=True)
class PlanningHiddenState:
    state_id: str
    cells: Mapping[StateCellRef, TypedValue]
```

Model-level validation requires:

- non-empty trimmed unique `state_id` values;
- every state assigns exactly the declared `planning_cells`;
- every assigned `TypedValue` is valid for the corresponding domain state variable;
- no two hidden states have identical semantic cell assignments under different IDs;
- states are canonicalized by lexical `state_id`.

A hidden state is a model hypothesis. It is never certified as the actual runtime world.

### 6.2 Planning observation

```python
@dataclass(frozen=True)
class PlanningObservation:
    observation_id: str
    cues: Mapping[StateCellRef, TypedValue | None]
```

`None` means an explicit unknown cue.

Model-level validation requires:

- non-empty trimmed unique `observation_id` values;
- every observation assigns exactly the declared `observation_cells`;
- non-`None` values are domain-valid typed values;
- no two observation atoms have identical cue payloads under different IDs;
- observations are canonicalized by lexical `observation_id`.

`observation_cells` may be empty only when the model declares exactly one observation atom with an empty cue mapping. That is the canonical no-observation/no-information token.

### 6.3 Declared cells

`planning_cells` and `observation_cells` are tuples of unique `StateCellRef` values.

At execution time both sets must be subsets of the selected authored decision's `context_cells`. This prevents planning from creating a hidden read capability outside the authored decision boundary.

`planning_cells` must be non-empty. `observation_cells` may be empty under the single-empty-observation rule above.

## 7. Sanitized hook contexts

Planning hooks receive only immutable hypothetical/model-semantic inputs. They never receive `GenericNarrative`, `DomainSpec`, `RuntimeEvidenceLedger`, runtime provenance, or objective world objects.

### 7.1 Joint belief context

```python
@dataclass(frozen=True)
class RuntimePlanningBeliefContext:
    posterior: Mapping[StateCellRef, BeliefDistribution]
    hidden_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]
```

The `posterior` mapping contains only the semantic finite posterior distributions for `planning_cells`; runtime support/provenance records are excluded.

Hook:

```python
joint_belief_hook(
    context: RuntimePlanningBeliefContext,
) -> Mapping[str, float]
```

The hook must return a probability for exactly every declared hidden-state ID.

This hook is mandatory because the existing runtime belief is cell-marginal. V1 must not silently multiply marginal distributions and thereby assume conditional independence.

### 7.2 Transition context

```python
@dataclass(frozen=True)
class PlanningTransitionContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    candidate_next_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]
```

Hook:

```python
transition_hook(
    context: PlanningTransitionContext,
) -> Mapping[str, float]
```

The result is `T(s' | s, a, depth)` and must exactly cover all declared hidden-state IDs.

### 7.3 Observation context

```python
@dataclass(frozen=True)
class PlanningObservationContext:
    depth: int
    next_state: PlanningHiddenState
    action: ActionOption
    observations: tuple[PlanningObservation, ...]
    parameters: Mapping[str, object]
```

Hook:

```python
observation_hook(
    context: PlanningObservationContext,
) -> Mapping[str, float]
```

The result is `O(o | s', a, depth)` and must exactly cover all declared observation IDs.

### 7.4 Reward context

```python
@dataclass(frozen=True)
class PlanningRewardContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    next_state: PlanningHiddenState
    parameters: Mapping[str, object]
```

Hook:

```python
reward_hook(context: PlanningRewardContext) -> float
```

The result is the immediate reward `R(s, a, s', depth)` and must be finite numeric and non-boolean.

V1 reward does not depend directly on the observation atom. Information value enters through posterior-dependent future action value.

## 8. Planning belief record

```python
@dataclass(frozen=True)
class PlanningBeliefState:
    probabilities: Mapping[str, float]
```

Validation requires:

- exact hidden-state coverage at the model/result boundary;
- finite non-negative non-boolean probabilities;
- `math.fsum(...)` equal to 1 with `abs_tol=1e-12`, `rel_tol=0`;
- lexical key ordering.

Its content hash is the canonical memoization and trace identity for a belief vector.

## 9. Model specification

```python
@dataclass(frozen=True)
class RuntimePlanningDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    planning_cells: tuple[StateCellRef, ...]
    observation_cells: tuple[StateCellRef, ...]
    belief_model: RuntimeBeliefModelSpec
    hidden_states: tuple[PlanningHiddenState, ...]
    observations: tuple[PlanningObservation, ...]
    action_schedule: tuple[tuple[str, ...], ...]
    discount: float
    beta: float
    parameters: Mapping[str, object]
    joint_belief_hook: object
    transition_hook: object
    observation_hook: object
    reward_hook: object
```

### 9.1 Horizon

There is no redundant `horizon` field.

```text
horizon = len(action_schedule)
```

`action_schedule` must contain at least one depth.

At depth zero its action IDs must equal exactly the selected authored decision's full action set.

Every later depth must be a non-empty subset of that same authored action set. V1 does not introduce state-dependent action availability or future-only action declarations.

Each depth is canonicalized to lexical action-ID ordering and rejects duplicates.

### 9.2 Discount and inverse temperature

- `discount` is finite numeric, non-boolean, and in `[0, 1]`.
- `beta` is finite numeric, non-boolean, and strictly positive.

### 9.3 Parameters

Planning parameters use the same canonical-value language as Reactive V1:

- `None`, `bool`, `int`, `str`, finite `float`;
- mappings with non-empty string keys and canonical values;
- list/tuple of canonical values, frozen to tuples;
- lexical mapping order.

Reject bytes, sets, non-finite floats, arbitrary objects, and callables inside parameters.

### 9.4 Model identity

The model content identity binds:

- model ID and version;
- supported decision types;
- planning and observation cells;
- runtime belief model hash;
- hidden-state and observation spaces;
- action schedule;
- discount and beta;
- canonical parameters;
- measured identities of all four hooks;
- measured `RuntimePlanningDecisionModelSpec` implementation identity;
- measured shared `finite_softmax` implementation identity.

Input tuple ordering that is declared semantically irrelevant is canonicalized before hashing.

## 10. Runtime execution contract

Public entry point:

```python
run_runtime_planning_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimePlanningDecisionModelSpec,
) -> RuntimePlanningDecisionResult
```

There is intentionally no `WorldState`, `WorldTransitionModelSpec`, observation projection model, RNG, or explicit `step_index` argument.

Execution order is fail-closed:

1. validate story against domain;
2. reconstruct/revalidate the planning model so constructor-bypass forgeries cannot skip invariants;
3. reconstruct/revalidate the evidence ledger and require exact domain/story/source identity;
4. resolve the authored decision template;
5. require model decision-type support;
6. require non-empty unique authored actions and valid context cells;
7. enforce source cutoff;
8. require `planning_cells` and `observation_cells` to stay within authored decision context;
9. validate hidden states and observations against domain types and the selected decision;
10. validate the root/future action schedule against the selected authored actions;
11. compute `RuntimeUncertainBeliefState` for exactly `planning_cells` through `runtime_uncertain_belief_state(...)`;
12. project that belief to a provenance-free semantic posterior mapping;
13. call `joint_belief_hook` and validate the exact root `PlanningBeliefState`;
14. solve the finite planning recursion;
15. convert root action values to a policy using the shared `finite_softmax(..., beta=model.beta)`;
16. validate exact action coverage, finite non-negative probability mass, and simplex sum;
17. choose deterministic lexical MAP action;
18. construct a self-validating result that binds the exact ledger, runtime belief, planning belief, values, trace, and model identity.

All structural checks that can be completed without executing user hooks occur before the first hook call.

## 11. Bayesian prediction and update

For belief `b`, action `a`, and depth `d`, predicted next-state mass is:

```text
p(s' | b,a,d) = sum_s b(s) T(s' | s,a,d)
```

Observation evidence mass is:

```text
p(o | b,a,d) = sum_s' p(s' | b,a,d) O(o | s',a,d)
```

For an observation with positive evidence mass:

```text
b'(s') = O(o | s',a,d) p(s' | b,a,d) / p(o | b,a,d)
```

Use `math.fsum` for probability aggregation and validate all intermediate masses as finite and non-negative.

Observation atoms with exactly zero evidence mass are impossible branches. They are not recursed into and do not produce a posterior record. A positive evidence mass that cannot yield a valid normalized posterior is a typed planning resolution failure.

No posterior clipping is permitted. Numerical drift is handled only by canonical simplex validation within the existing `1e-12` tolerance.

## 12. Finite-horizon soft Bellman recursion

Let `H = len(action_schedule)` and let `A_d` be the actions declared for depth `d`.

Immediate expected reward:

```text
r_d(b,a)
  = sum_s b(s)
      sum_s' T(s'|s,a,d) R(s,a,s',d)
```

At terminal depth `d = H - 1`:

```text
Q_d(b,a) = r_d(b,a)
```

At earlier depths:

```text
Q_d(b,a)
  = r_d(b,a)
    + discount * sum_o p(o|b,a,d) V_{d+1}(b'_o)
```

At every depth, including future depths:

```text
pi_d(a | b) = finite_softmax(Q_d(b,·), beta=model.beta)
V_d(b)      = sum_a pi_d(a|b) Q_d(b,a)
```

This is a soft stochastic-choice value recursion, matching the behavioral interpretation already used by the finite prison POMDP instead of silently replacing future choice with a hard `max` operator.

The public result exposes only the root policy for runtime action selection, but the audit trace records future value calculations.

## 13. Determinism and memoization

V1 uses exact finite enumeration over:

```text
beliefs reached by recursion
x current hidden states
x declared actions
x next hidden states
x declared observations
x finite depth
```

No random draw occurs.

All iteration order is canonical:

- hidden states by `state_id`;
- observations by `observation_id`;
- actions by lexical action ID;
- belief probability keys by hidden-state ID.

Memoize value calculations by:

```text
(depth, PlanningBeliefState.content_hash)
```

Memoization is an optimization only; disabling it must not change result semantics or hashes.

## 14. Audit trace

Planning must not collapse to an opaque root `Q(a)` vector.

### 14.1 Value records

```python
@dataclass(frozen=True)
class PlanningValueRecord:
    depth: int
    belief_hash: str
    action_id: str
    expected_immediate_reward: float
    expected_future_value: float
    total_value: float
```

For terminal depth, `expected_future_value` is exactly `0.0`.

### 14.2 Belief-update records

```python
@dataclass(frozen=True)
class PlanningBeliefUpdate:
    depth: int
    prior_belief_hash: str
    action_id: str
    observation_id: str
    observation_probability: float
    posterior: PlanningBeliefState
```

Only positive-probability observation branches receive update records.

The complete trace is sorted canonically by depth, belief hash, action ID, and observation ID as applicable. It is data, not certification; result validation rechecks its binding to the root solution.

## 15. Result contract

```python
@dataclass(frozen=True)
class RuntimePlanningDecisionResult:
    model_id: str
    model_hash: str
    decision_id: str
    actor_id: str
    step_index: int
    ledger_hash: str
    belief_state: RuntimeUncertainBeliefState
    planning_belief: PlanningBeliefState
    action_values: Mapping[str, float]
    action_policy: Mapping[str, float]
    selected_action: str
    value_records: tuple[PlanningValueRecord, ...]
    belief_updates: tuple[PlanningBeliefUpdate, ...]
```

Validation requires:

- exact model identity and valid hashes;
- actor/decision/step agreement with the embedded runtime belief state;
- ledger hash equals the embedded runtime belief state's upstream ledger binding where exposed by that type;
- root planning belief exactly covers model hidden states;
- `action_values` and `action_policy` exactly cover authored root actions;
- action values are finite numeric non-booleans;
- policy probabilities are finite, non-negative, and sum to 1 within `1e-12` absolute tolerance;
- `selected_action` is the lexical MAP action;
- trace records are canonical, finite, depth-valid, and bind the root belief/action values.

The comparison-relevant common boundary across reactive, intentional, and planning results is:

```text
action_policy: Mapping[action_id, probability]
selected_action: action_id
```

Model-specific latent/audit fields remain explicit rather than forced into one opaque common result type.

## 16. Error semantics

Public typed error:

```python
class RuntimePlanningDecisionResolutionError(ValueError):
    ...
```

The public runner normalizes ordinary `Exception` failures from:

- runtime belief resolution;
- joint-belief hook;
- transition hook;
- observation hook;
- reward hook;
- shared softmax/numerical validation;

into `RuntimePlanningDecisionResolutionError` while preserving the original exception as `__cause__`.

Do not catch `BaseException`.

Structural domain/story/ledger/model violations are also exposed through the planning typed boundary after preflight, consistent with runtime decision fail-closed behavior.

## 17. Scientific fixtures

The feature is not complete with schema tests alone. It must include the following behavior-level fixtures.

### 17.1 Horizon-one observational equivalence

Construct a one-depth planning model with:

- identity transition;
- one no-information observation;
- reward equal to a declared immediate action-value table;
- the same beta as a reactive score model;
- a single-goal intentional fixture whose conditional action scores are the same table and whose final policy reduces to the same softmax.

Require exact action-policy equality across planning, reactive, and intentional results and the same lexical MAP action.

This demonstrates that richer latent structure is not identifiable when it contributes no distinct observable behavior.

### 17.2 State-independent observation carries no information

Construct a multi-depth model whose observation likelihood is identical for every hidden state.

For every positive-probability observation branch, require the posterior planning belief to equal the predicted next-state belief exactly within the canonical probability representation.

This locks the Bayesian information boundary without making an incorrect claim that an extra delayed action must have the same utility as a myopic action.

### 17.3 Value-of-information separation

Construct a finite two-state case with an information-gathering root action and informative future observation.

Require:

- reactive and myopic intentional models to use only current admitted information;
- planning to value the observation branch through future posterior-dependent action value;
- planning policy to differ from both lower-complexity families;
- the difference to disappear when the observation kernel is replaced with the state-independent/no-information kernel while other declared reward/transition assumptions are held fixed as required by the fixture.

### 17.4 Objective-state non-leakage

Lock both of these:

- `run_runtime_planning_decision` has no world-state argument;
- `runtime_planning.py` contains no import of world, observation projection, or simulation modules.

Hook-context tests must prove no story/domain/ledger/provenance/world objects are reachable through the four hook context records.

### 17.5 Exact replay and forgery rejection

Fixed inputs must produce exactly identical result payload/content hash across runs.

Constructor-bypass or replaced nested records that break model, ledger, belief, trace, or selected-action bindings must fail closed.

## 18. Public API

`narrative_dynamics.narrative` gains exactly these 13 names:

1. `PlanningHiddenState`
2. `PlanningObservation`
3. `PlanningBeliefState`
4. `RuntimePlanningBeliefContext`
5. `PlanningTransitionContext`
6. `PlanningObservationContext`
7. `PlanningRewardContext`
8. `PlanningValueRecord`
9. `PlanningBeliefUpdate`
10. `RuntimePlanningDecisionModelSpec`
11. `RuntimePlanningDecisionResult`
12. `RuntimePlanningDecisionResolutionError`
13. `run_runtime_planning_decision`

The package root `narrative_dynamics` gains no new exports.

## 19. Expected implementation scope

Expected production changes:

```text
narrative_dynamics/narrative/runtime_planning.py        new
narrative_dynamics/narrative/__init__.py                exact exports only
```

Expected tests:

```text
tests/test_narrative_runtime_planning.py                new
tests/test_narrative_trust_api.py                       exact surface extension
```

Expected docs:

```text
docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md
docs/superpowers/plans/2026-08-27-narrative-generic-planning-pomdp-v1.md
```

No other path is in planned scope. Any discovered prerequisite must be isolated as its own RED -> GREEN slice and documented explicitly rather than silently broadening the feature.

## 20. TDD and verification gates

Implementation follows strict atomic TDD:

1. design spec commit;
2. implementation-plan commit after design approval;
3. test-only RED commit(s);
4. authoritative exact-head RED CI evidence;
5. minimal GREEN implementation slices;
6. exact public export slice;
7. exact-head final CI;
8. final diff-scope review before merge.

No final GREEN claim is valid without a completed/success GitHub Actions proof on the exact final feature head.

## 21. Success criterion

P1 Generic Planning / POMDP V1 is complete when the repository can express and exactly solve a narrative-grounded finite planning decision where:

```text
runtime admitted evidence
  -> finite posterior semantics
  -> explicit joint hidden-state belief
  -> hypothetical transition
  -> hypothetical observation
  -> Bayesian future belief
  -> finite-horizon soft value recursion
  -> complete root action policy
```

with no objective-world capability, deterministic replay, explicit audit lineage, and behavior-level fixtures showing both observational equivalence and value-of-information separation from reactive/intentional baselines.
