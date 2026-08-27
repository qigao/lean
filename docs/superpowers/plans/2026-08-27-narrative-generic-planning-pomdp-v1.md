# Narrative Generic Planning / POMDP V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native deterministic finite-horizon POMDP decision sidecar that starts from admitted runtime belief, preserves those posterior marginals through an explicit joint coupling, values hypothetical transitions and future observations without objective-world access, and returns a complete root action policy plus lexical MAP action.

**Architecture:** Implement one isolated `runtime_planning.py` sidecar. The sidecar reuses `runtime_uncertain_belief_state(...)` for the only runtime information input, projects its cell-marginal posterior into a validated joint hidden-state coupling, solves a finite soft Bellman recursion over declared hidden states/observations/actions, and uses shared `finite_softmax` for every decision depth. Scheduler polymorphism, real world rollout, and released model-comparison integration stay out of scope.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `MappingProxyType`, `unittest`), existing `narrative_dynamics` typed IR/runtime cognition/attestation/content hashing, shared `grounded_goal_softmax.finite_softmax`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md`

## Global Constraints

- Integrated base is `proof/narrative-dynamics-v0` at `8aae284fe02f27a76b6d00a651be152f49a9e6ed`.
- Current feature branch is `work/narrative-generic-planning-pomdp-v1`.
- Branch already contains the isolated CI trigger refactor commit `b328091c81217c12467bcf4610a04304a5bf3eae`; do not modify CI again in this feature unless a concrete workflow defect appears.
- Pure changes under `docs/superpowers/specs/**` and `docs/superpowers/plans/**` do not trigger proof CI. The first authoritative planning CI must therefore be the test-only RED commit, not this plan commit.
- Production planning scope is limited to `narrative_dynamics/narrative/runtime_planning.py` plus exact exports in `narrative_dynamics/narrative/__init__.py`.
- Test scope is `tests/test_narrative_runtime_planning.py` plus the exact public-surface extension in `tests/test_narrative_trust_api.py`.
- Do not modify `GenericNarrative`, `DomainSpec`, authored decisions, `runtime_cognition.py`, `runtime_intention.py`, `runtime_reactive.py`, `world.py`, `observation_projection.py`, `simulation.py`, released model-comparison infrastructure, prison adapters, or Lean sources.
- `runtime_planning.py` must not import `narrative_dynamics.narrative.world`, `observation_projection`, `simulation`, `runtime_intention`, or `runtime_reactive`.
- The public runner signature is exactly `run_runtime_planning_decision(story, domain, decision_id, ledger, model)`; no `WorldState`, RNG, explicit step, real transition model, or observation projection model argument is allowed.
- Planning hidden states and observation atoms are finite, narrative-grounded, typed declarations. No dynamic state/observation token creation is allowed during recursion.
- The joint-belief hook is mandatory and its output must be a legal coupling: marginalizing the joint over every `planning_cell` must reproduce the corresponding runtime posterior exactly within absolute tolerance `1e-12`, relative tolerance `0`.
- Never silently multiply marginals in production code. A product coupling is acceptable only as an explicit test/model hook.
- Transition/observation distributions must exactly cover their declared spaces, be finite/non-negative/non-boolean, and sum to one within absolute tolerance `1e-12`, relative tolerance `0`.
- Reward values and all derived Q/value terms must be finite numeric non-booleans.
- Bayesian zero-evidence observations are impossible branches: skip recursion and emit no `PlanningBeliefUpdate` for them.
- No probability clipping. Invalid normalization is a typed planning resolution failure.
- The Bellman recursion is soft at every depth: `pi = finite_softmax(Q, beta)` and `V = sum(pi[a] * Q[a])`; never replace future choice with hard `max`.
- V1 uses no RNG. Same inputs must produce byte-for-byte equivalent canonical result payloads/content hashes.
- Selected runtime action is deterministic lexical MAP of the root policy.
- Model identity binds runtime belief model, hidden/observation spaces, schedule, parameters, discount/beta, all four hook implementation attestations, planning model implementation identity, and shared softmax implementation identity.
- `narrative_dynamics.narrative` gains exactly 13 planning names; package root `narrative_dynamics` gains none.
- Use strict RED -> GREEN -> atomic commit discipline. Do not claim final GREEN without completed/success proof CI on the exact final feature head.

## File Structure

- `narrative_dynamics/narrative/runtime_planning.py` — all V1 planning value records, sanitized hook contexts, model/result contracts, preflight validation, joint-coupling validation, finite Bayesian solver, audit trace, public runner.
- `narrative_dynamics/narrative/__init__.py` — exact 13 planning imports/exports only.
- `tests/test_narrative_runtime_planning.py` — complete RED contract suite, scientific fixtures, leakage/import isolation, replay/forgery tests.
- `tests/test_narrative_trust_api.py` — exact public API set gains exactly the 13 planning names; package-root isolation remains locked.
- `docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md` — approved contract source of truth.
- `docs/superpowers/plans/2026-08-27-narrative-generic-planning-pomdp-v1.md` — this implementation plan.

---

### Task 1: Establish the Complete Test-Only RED Boundary

**Files:**
- Create: `tests/test_narrative_runtime_planning.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: existing `RuntimeEvidenceLedger`, `RuntimeBeliefModelSpec`, `RuntimeUncertainBeliefState`, `BeliefDistribution`, `StateCellRef`, `TypedValue`, `ActionOption`, `finite_softmax`; existing test helpers in `tests/test_narrative_runtime_cognition.py`, `tests/test_narrative_runtime_intention.py`, and `tests/test_narrative_runtime_reactive.py`.
- Produces: the authoritative RED contract for all 13 public planning symbols and all V1 behavior before any planning production module exists.

- [ ] **Step 1: Create the guarded planning import and reusable planning test fixtures**

At the top of `tests/test_narrative_runtime_planning.py`, use the same missing-module RED pattern as the reactive suite:

```python
from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import unittest

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import ActionOption, TypedValue
from narrative_dynamics.narrative.runtime_perception import runtime_evidence_ledger_from_story
from tests.test_narrative_runtime_cognition import (
    alert_cell,
    empty_runtime_case,
    phase_cell,
)
from tests.test_narrative_runtime_intention import make_runtime_intentional_model

_PLANNING_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_planning import (
        PlanningBeliefState,
        PlanningBeliefUpdate,
        PlanningHiddenState,
        PlanningObservation,
        PlanningObservationContext,
        PlanningRewardContext,
        PlanningTransitionContext,
        PlanningValueRecord,
        RuntimePlanningBeliefContext,
        RuntimePlanningDecisionModelSpec,
        RuntimePlanningDecisionResolutionError,
        RuntimePlanningDecisionResult,
        run_runtime_planning_decision,
    )
except ImportError as error:
    _PLANNING_IMPORT_ERROR = error


def _forge(instance, **changes):
    forged = object.__new__(type(instance))
    for item in fields(instance):
        object.__setattr__(
            forged,
            item.name,
            changes.get(item.name, getattr(instance, item.name)),
        )
    return forged


def _probability_map(distribution):
    return {mass.value: mass.probability for mass in distribution.masses}


class ProductCouplingHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        result = {}
        for state in context.hidden_states:
            probability = 1.0
            for cell, distribution in context.posterior.items():
                probability *= distribution.probability_of(state.cells[cell])
            result[state.state_id] = probability
        return result


class IdentityTransitionHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return {
            state.state_id: 1.0 if state.state_id == context.state.state_id else 0.0
            for state in context.candidate_next_states
        }


class NoInformationObservationHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return {
            observation.observation_id: (
                1.0 if observation.observation_id == "none" else 0.0
            )
            for observation in context.observations
        }


class ParameterRewardHook:
    def __init__(self):
        self.calls = []

    def __call__(self, context):
        self.calls.append(context)
        return float(context.parameters["action_rewards"][context.action.id])
```

Add `require_planning(self)` that fails with the captured import error, and a `make_model(...)` helper that creates a one-cell two-state baseline by default:

```python
def make_model(self, *, joint=None, transition=None, observation=None, reward=None,
               planning_cells=None, observation_cells=(), hidden_states=None,
               observations=None, schedule=None, discount=1.0, beta=2.0,
               parameters=None, belief_model=None):
    self.require_planning()
    if planning_cells is None:
        planning_cells = (phase_cell(),)
    if hidden_states is None:
        hidden_states = (
            PlanningHiddenState(
                "active",
                {phase_cell(): TypedValue("PhaseState", "active")},
            ),
            PlanningHiddenState(
                "ready",
                {phase_cell(): TypedValue("PhaseState", "ready")},
            ),
        )
    if observations is None:
        observations = (PlanningObservation("none", {}),)
    if schedule is None:
        schedule = (("a1-active", "a1-ready"),)
    if parameters is None:
        parameters = {
            "action_rewards": {"a1-active": 2.0, "a1-ready": 0.0},
            "nested": {"flags": [True, None]},
        }
    if belief_model is None:
        belief_model = make_runtime_intentional_model().belief_model
    return RuntimePlanningDecisionModelSpec(
        "runtime-planning",
        "1",
        ("phase-choice",),
        tuple(planning_cells),
        tuple(observation_cells),
        belief_model,
        tuple(hidden_states),
        tuple(observations),
        tuple(tuple(row) for row in schedule),
        discount,
        beta,
        parameters,
        joint or ProductCouplingHook(),
        transition or IdentityTransitionHook(),
        observation or NoInformationObservationHook(),
        reward or ParameterRewardHook(),
    )
```

- [ ] **Step 2: Add record/model/public-signature tests**

Add these exact tests and assertions:

```python
def test_hidden_state_observation_and_belief_records_are_canonical(self):
    self.require_planning()
    active = PlanningHiddenState(
        "active", {phase_cell(): TypedValue("PhaseState", "active")}
    )
    none = PlanningObservation("none", {})
    belief = PlanningBeliefState({"ready": 0.25, "active": 0.75})
    self.assertEqual(tuple(belief.probabilities), ("active", "ready"))
    self.assertEqual(math.fsum(belief.probabilities.values()), 1.0)
    with self.assertRaises((TypeError, ValueError)):
        PlanningHiddenState("", {phase_cell(): TypedValue("PhaseState", "active")})
    with self.assertRaises((TypeError, ValueError)):
        PlanningObservation("", {})
    for bad in (
        {"active": True, "ready": 0.0},
        {"active": math.nan, "ready": 1.0},
        {"active": -0.1, "ready": 1.1},
        {"active": 0.4, "ready": 0.4},
    ):
        with self.subTest(bad=bad):
            with self.assertRaises((TypeError, ValueError)):
                PlanningBeliefState(bad)


def test_model_parameters_and_identity_bind_all_planning_assumptions(self):
    self.require_planning()
    first = self.make_model()
    reordered = self.make_model(
        parameters={
            "nested": {"flags": (True, None)},
            "action_rewards": {"a1-ready": 0.0, "a1-active": 2.0},
        },
        schedule=(("a1-ready", "a1-active"),),
    )
    changed_beta = replace(first, beta=3.0)
    changed_discount = replace(first, discount=0.5)
    self.assertEqual(first.content_hash, reordered.content_hash)
    self.assertNotEqual(first.content_hash, changed_beta.content_hash)
    self.assertNotEqual(first.content_hash, changed_discount.content_hash)
    payload = first.to_dict()
    self.assertEqual(
        payload["softmax_implementation_identity"],
        measure_implementation(finite_softmax).manifest_identity(),
    )
    self.assertEqual(
        payload["runtime_implementation_identity"],
        measure_implementation(RuntimePlanningDecisionModelSpec).manifest_identity(),
    )
    for key in (
        "joint_belief_hook_identity",
        "transition_hook_identity",
        "observation_hook_identity",
        "reward_hook_identity",
    ):
        self.assertIn(key, payload)


def test_model_parameters_accept_only_canonical_values(self):
    self.require_planning()
    accepted = {
        "none": None,
        "bool": True,
        "int": 3,
        "str": "x",
        "float": 1.25,
        "nested": {"b": [1, 2], "a": (False, None)},
    }
    model = self.make_model(parameters=accepted)
    self.assertEqual(model.parameters["nested"]["a"], (False, None))
    self.assertEqual(model.parameters["nested"]["b"], (1, 2))
    for parameters in (
        {"x": math.nan},
        {"x": math.inf},
        {"x": b"bytes"},
        {"x": {1, 2}},
        {"x": object()},
        {"x": lambda: None},
        {"": 1},
    ):
        with self.subTest(parameters=parameters):
            with self.assertRaises((TypeError, ValueError)):
                self.make_model(parameters=parameters)


def test_public_signature_excludes_world_rng_and_explicit_step(self):
    self.require_planning()
    parameters = inspect.signature(run_runtime_planning_decision).parameters
    self.assertEqual(
        tuple(parameters),
        ("story", "domain", "decision_id", "ledger", "model"),
    )
```

Also construct `RuntimePlanningBeliefContext`, `PlanningTransitionContext`, `PlanningObservationContext`, `PlanningRewardContext`, `PlanningValueRecord`, and `PlanningBeliefUpdate` directly in this group and assert frozen/canonical behavior plus invalid negative depth, invalid IDs, invalid probability, and non-finite reward/value rejection.

- [ ] **Step 3: Add preflight, coupling, kernel, solver, trace, and error tests**

Add these exact test methods:

```python
def test_preflight_rejects_story_ledger_context_cutoff_and_schedule_before_hooks(self): ...
def test_joint_belief_must_be_a_coupling_of_runtime_cell_marginals(self): ...
def test_hook_contexts_are_sanitized_and_module_has_no_world_capability(self): ...
def test_transition_observation_and_reward_hook_schemas_fail_typed(self): ...
def test_zero_evidence_observation_is_skipped_and_no_information_preserves_prediction(self): ...
def test_finite_horizon_soft_bellman_matches_hand_computed_values(self): ...
def test_shared_softmax_produces_complete_root_policy_and_lexical_map(self): ...
def test_hook_exceptions_are_wrapped_with_typed_causes(self): ...
def test_result_and_trace_forgery_is_rejected(self): ...
def test_fixed_inputs_replay_to_exact_result_and_content_hash(self): ...
```

Implement the test data with these locked semantics:

1. **Preflight:** use hook objects with `calls=[]`; forge ledger domain/story/chain data, choose a planning cell outside authored context, use a decision after source cutoff, give root schedule missing one authored action, and give a later schedule containing an undeclared action. Each case must raise `RuntimePlanningDecisionResolutionError` and leave every hook call list empty.
2. **Coupling:** use two planning cells `(phase_cell(), alert_cell())` and four hidden states covering the Cartesian value combinations. A valid `ProductCouplingHook` must pass. A bad hook returning all mass on one state must fail whenever either runtime marginal is non-degenerate. Verify failure occurs before transition/observation/reward hooks.
3. **Sanitization/import isolation:** inspect every hook context from a successful run and assert it has no attributes named `story`, `domain`, `ledger`, `world`, `world_state`, `evidence`, `provenance`, or `runtime_evidence_history`. Read `inspect.getsource(runtime_planning_module)` and assert it contains none of `narrative.world`, `narrative.observation_projection`, `narrative.simulation`, `narrative.runtime_intention`, `narrative.runtime_reactive`.
4. **Kernel schemas:** parameterize transition/observation hooks returning non-mapping, missing key, extra key, bool, NaN, infinity, negative mass, or non-unit total. Parameterize reward returning bool/NaN/infinity. Every case must become typed planning resolution failure with a cause.
5. **No-information update:** for state-independent observation likelihoods, every positive-probability branch posterior equals the predicted next-state belief exactly by canonical payload; zero-evidence observations create no `PlanningBeliefUpdate`.
6. **Soft Bellman:** use a two-depth, two-state deterministic-transition fixture where hand-computed Q values differ from a hard-max recursion. Assert each root `PlanningValueRecord.total_value`, `expected_immediate_reward`, and `expected_future_value` to `places=12`; assert future value is `sum(pi * Q)` from `finite_softmax`, not `max(Q)`.
7. **Root policy/MAP:** root values with an exact tie must return equal policy coordinates and lexical smallest action as `selected_action` independent of input action ordering.
8. **Hook exceptions:** each of the four hooks raises `RuntimeError("planning <hook> boom")` in a separate subtest; public error must be `RuntimePlanningDecisionResolutionError` and `__cause__` must be that runtime error.
9. **Forgery:** replace nested planning belief, ledger hash, value record total, belief update posterior, model hash, and selected action using `_forge`; result constructor/revalidation must reject each mismatch.
10. **Replay:** two calls with the same story/domain/ledger/model must have equal result objects, equal `to_dict()`, and equal `content_hash`.

- [ ] **Step 4: Add the three scientific fixtures**

Add these exact tests:

```python
def test_horizon_one_policy_is_exactly_equivalent_across_planning_reactive_and_intentional(self): ...
def test_state_independent_observation_has_zero_bayesian_information_gain(self): ...
def test_value_of_information_separates_planning_from_reactive_and_intentional(self): ...
```

Lock the fixtures as follows:

- Horizon one: identity transition, one no-information observation, same immediate action-score table and same beta. Planning, reactive, and a single-goal intentional fixture must have exactly equal `action_policy` mappings and the same lexical MAP action.
- State-independent observation: use at least two observation atoms but identical likelihood vectors for every hidden state. For every emitted `PlanningBeliefUpdate`, compare posterior payload to the predicted prior-to-observation belief; they must be equal. Do not assert equality to a myopic utility because an extra delayed action can still change value without information gain.
- Value of information: use root actions `inspect`, `act-a`, `act-b`; future schedule contains only `act-a`, `act-b`. Informative observation identifies the two hidden states sufficiently that `inspect` has positive continuation advantage net of inspection cost. Reactive and intentional fixtures only evaluate current information and do not gain that future branch. Assert planning policy differs from both. Replace only the observation kernel with state-independent likelihoods and assert the inspection advantage disappears while reward/transition declarations remain fixed.

- [ ] **Step 5: Extend the exact narrative public-surface RED**

In `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`, add exactly:

```python
    # Runtime planning selection.
    "PlanningHiddenState",
    "PlanningObservation",
    "PlanningBeliefState",
    "RuntimePlanningBeliefContext",
    "PlanningTransitionContext",
    "PlanningObservationContext",
    "PlanningRewardContext",
    "PlanningValueRecord",
    "PlanningBeliefUpdate",
    "RuntimePlanningDecisionModelSpec",
    "RuntimePlanningDecisionResult",
    "RuntimePlanningDecisionResolutionError",
    "run_runtime_planning_decision",
```

Do not change the root-isolation assertion: these names must remain absent from `dir(narrative_dynamics)`.

- [ ] **Step 6: Run the RED suite locally or via available execution environment**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
python3 -m unittest tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected before production exists:

```text
planning tests fail because narrative_dynamics.narrative.runtime_planning is missing
trust public-surface test fails with exactly the 13 planned names missing
```

If the local environment cannot run the repository, do not weaken the gate; commit the test-only RED and use GitHub Actions as the authoritative execution proof.

- [ ] **Step 7: Commit the complete test-only RED**

```bash
git add tests/test_narrative_runtime_planning.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative planning pomdp contract"
```

This commit changes non-ignored `tests/**`, so it must trigger proof CI.

- [ ] **Step 8: Create/update a draft PR and capture authoritative exact-head RED**

Create a draft PR against `proof/narrative-dynamics-v0` if one does not yet exist. Fetch the proof run for the exact test-only head and require:

```text
status = completed
conclusion = failure
head_sha = exact test-only RED commit
```

Inspect the failing job log. Accept RED only if all pre-existing gates remain green up to the new planning failures and the failures are caused by the missing planning module/public names, not unrelated regressions.

---

### Task 2: Add Canonical Planning Records and Model Identity

**Files:**
- Create: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: `StateCellRef`, `TypedValue`, `ActionOption`, `BeliefDistribution`, `RuntimeBeliefModelSpec`, `measure_implementation`, `stable_content_hash`, `finite_softmax`.
- Produces: all 13 module-local public symbols so imports work; canonical data/model contracts; runner exists but may still fail semantic solver tests.

- [ ] **Step 1: Implement scalar/canonical helper functions and module constants**

Start `runtime_planning.py` with only allowed imports and define:

```python
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12


def _text(value: object, *, label: str) -> str: ...
def _hash(value: object, *, label: str) -> str: ...
def _depth(value: object, *, label: str) -> int: ...
def _finite(value: object, *, label: str) -> float: ...
def _probability(value: object, *, label: str) -> float: ...
def _cell_key(cell: StateCellRef) -> tuple[str, str, str]: ...
def _freeze_parameter(value: object, *, label: str) -> object: ...
def _freeze_parameters(value: object) -> Mapping[str, object]: ...
def _parameter_payload(value: object) -> object: ...
```

Match Reactive V1 canonical parameter rules exactly. Reject bool where numeric is required, non-finite floats, bytes, sets, arbitrary objects, callable parameter values, and empty mapping keys.

- [ ] **Step 2: Implement finite-space data records**

Implement:

```python
@dataclass(frozen=True)
class PlanningHiddenState:
    state_id: str
    cells: Mapping[StateCellRef, TypedValue]

@dataclass(frozen=True)
class PlanningObservation:
    observation_id: str
    cues: Mapping[StateCellRef, TypedValue | None]

@dataclass(frozen=True)
class PlanningBeliefState:
    probabilities: Mapping[str, float]
```

Structural constructor rules:

- freeze mappings with `MappingProxyType`;
- canonicalize cells by `_cell_key` and belief IDs lexically;
- record constructors validate only type/shape facts they can know without `DomainSpec`;
- `PlanningBeliefState` requires non-empty probabilities, finite/non-negative/non-bool entries, and normalized sum within `1e-12`.

Domain typing, exact declared cell coverage, duplicate semantic hidden states, and exact hidden-state ID coverage belong in runner/model preflight where `DomainSpec` and the selected decision are available.

- [ ] **Step 3: Implement sanitized hook contexts and audit records**

Implement exactly:

```python
@dataclass(frozen=True)
class RuntimePlanningBeliefContext:
    posterior: Mapping[StateCellRef, BeliefDistribution]
    hidden_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]

@dataclass(frozen=True)
class PlanningTransitionContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    candidate_next_states: tuple[PlanningHiddenState, ...]
    parameters: Mapping[str, object]

@dataclass(frozen=True)
class PlanningObservationContext:
    depth: int
    next_state: PlanningHiddenState
    action: ActionOption
    observations: tuple[PlanningObservation, ...]
    parameters: Mapping[str, object]

@dataclass(frozen=True)
class PlanningRewardContext:
    depth: int
    state: PlanningHiddenState
    action: ActionOption
    next_state: PlanningHiddenState
    parameters: Mapping[str, object]

@dataclass(frozen=True)
class PlanningValueRecord:
    depth: int
    belief_hash: str
    action_id: str
    expected_immediate_reward: float
    expected_future_value: float
    total_value: float

@dataclass(frozen=True)
class PlanningBeliefUpdate:
    depth: int
    prior_belief_hash: str
    action_id: str
    observation_id: str
    observation_probability: float
    posterior: PlanningBeliefState
```

Every nested mapping/tuple is detached and canonicalized. Audit numeric fields are finite; observation probability is in `(0, 1]`; depths are non-negative.

- [ ] **Step 4: Implement `RuntimePlanningDecisionModelSpec`**

Use this exact field order/signature:

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
    joint_belief_hook: object = field(compare=False, repr=False)
    transition_hook: object = field(compare=False, repr=False)
    observation_hook: object = field(compare=False, repr=False)
    reward_hook: object = field(compare=False, repr=False)
```

Constructor validation:

- trimmed model/version;
- non-empty unique sorted supported decision types;
- non-empty unique canonical planning cells;
- unique canonical observation cells;
- `RuntimeBeliefModelSpec` required;
- at least one hidden state with unique IDs;
- at least one observation with unique IDs;
- at least one action-schedule row; every row non-empty, duplicate-free, lexicalized;
- `discount in [0,1]`, `beta > 0`, finite non-bool;
- canonical parameters;
- all four hooks callable.

`to_dict()` must bind:

```python
{
    "model_id": self.model_id,
    "version": self.version,
    "supported_decision_types": list(self.supported_decision_types),
    "planning_cells": [cell.to_dict() for cell in self.planning_cells],
    "observation_cells": [cell.to_dict() for cell in self.observation_cells],
    "belief_model_hash": self.belief_model.content_hash,
    "hidden_states": [state.to_dict() for state in self.hidden_states],
    "observations": [item.to_dict() for item in self.observations],
    "action_schedule": [list(row) for row in self.action_schedule],
    "discount": self.discount,
    "beta": self.beta,
    "parameters": _parameter_payload(self.parameters),
    "joint_belief_hook_identity": measure_implementation(self.joint_belief_hook).manifest_identity(),
    "transition_hook_identity": measure_implementation(self.transition_hook).manifest_identity(),
    "observation_hook_identity": measure_implementation(self.observation_hook).manifest_identity(),
    "reward_hook_identity": measure_implementation(self.reward_hook).manifest_identity(),
    "runtime_implementation_identity": measure_implementation(RuntimePlanningDecisionModelSpec).manifest_identity(),
    "softmax_implementation_identity": measure_implementation(finite_softmax).manifest_identity(),
}
```

- [ ] **Step 5: Define the result/error/public runner symbols so the module imports cleanly**

Define `RuntimePlanningDecisionResolutionError(ValueError)` and the complete `RuntimePlanningDecisionResult` field shape from the spec. Implement basic type/hash/simplex/lexical-MAP validation now; deeper solver/trace binding is added in Task 5.

Define the public runner with its final signature, but make it fail typed until preflight/solver tasks land:

```python
def run_runtime_planning_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimePlanningDecisionModelSpec,
) -> RuntimePlanningDecisionResult:
    raise RuntimePlanningDecisionResolutionError(
        "runtime planning decision solver is not implemented"
    )
```

Add module-local `__all__` containing exactly the 13 names from the spec.

- [ ] **Step 6: Run the records/model subset**

Run the planning test file and verify the constructor/model/public-signature tests pass while runner/solver tests remain RED:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: planning import succeeds; record/model tests pass; semantic runner tests fail with the typed solver-not-implemented error.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: add narrative planning model records"
```

---

### Task 3: Add Preflight, Runtime Posterior Projection, and Joint-Coupling Validation

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: `RuntimePlanningDecisionModelSpec`, `RuntimeEvidenceLedger`, `runtime_uncertain_belief_state(...)`, `RuntimeUncertainBeliefState`, selected authored `Decision`.
- Produces: validated decision/model/ledger boundary and exact root `PlanningBeliefState` whose marginals equal the runtime posterior semantics.

- [ ] **Step 1: Reconstruct model and ledger to reject constructor-bypass forgeries**

Implement `_validated_model(model)` by constructing a fresh `RuntimePlanningDecisionModelSpec` from every public field and returning it only if the reconstructed value has the same canonical payload/content hash.

Implement `_validated_ledger(story, domain, ledger)` by constructing a fresh `RuntimeEvidenceLedger` from:

```python
RuntimeEvidenceLedger(
    ledger.domain_id,
    ledger.domain_version,
    ledger.domain_spec_hash,
    ledger.source_story_hash,
    ledger.source_at_time,
    ledger.initial_world_state_hash,
    ledger.current_world_state_hash,
    tuple(ledger.batches),
)
```

Then require exact:

```text
ledger.domain_id == domain.domain_id
ledger.domain_version == domain.version
ledger.domain_spec_hash == domain.content_hash
ledger.source_story_hash == story.content_hash
```

- [ ] **Step 2: Add decision/domain finite-space preflight**

Before any planning hook call:

1. `validate_narrative(story, domain)`;
2. resolve exact decision ID;
3. require supported decision type;
4. require non-empty unique authored context cells/actions;
5. enforce `decision.logical_time <= ledger.source_at_time` when cutoff is not `None`;
6. require planning/observation cells subsets of authored context;
7. require every hidden state assign exactly `planning_cells`;
8. require every observation assign exactly `observation_cells`;
9. validate every non-`None` typed value using the same domain state-variable typing helper/path used by existing narrative validation rather than inventing a second type system;
10. reject duplicate hidden-state semantic cell payloads under different IDs;
11. reject duplicate observation semantic cue payloads under different IDs;
12. if observation cells are empty, require exactly one observation with empty cues;
13. root action schedule must equal the complete authored action ID set;
14. every later schedule row must be a non-empty subset of the same authored action IDs.

Build a lexical `action_by_id` mapping from the authored `ActionOption` records for later hook contexts.

- [ ] **Step 3: Resolve the runtime belief for exactly the planning cells**

Call:

```python
belief_state = runtime_uncertain_belief_state(
    story,
    domain,
    decision.actor_id,
    ledger,
    model.belief_model,
    model.planning_cells,
)
```

Build a provenance-free semantic mapping:

```python
posterior = MappingProxyType({
    cell: belief_state.cells[cell].posterior
    for cell in sorted(model.planning_cells, key=_cell_key)
})
```

Do not pass `belief_state`, ledger evidence, support hashes, story, or domain into the joint hook.

- [ ] **Step 4: Validate the joint hook as an exact coupling**

Call:

```python
raw_joint = model.joint_belief_hook(
    RuntimePlanningBeliefContext(
        posterior=posterior,
        hidden_states=model.hidden_states,
        parameters=model.parameters,
    )
)
```

Validate the returned mapping has exactly every hidden-state ID and is a valid `PlanningBeliefState`.

Then marginalize the joint for every planning cell/value present in the runtime posterior:

```python
for cell, distribution in posterior.items():
    for mass in distribution.masses:
        joint_mass = math.fsum(
            root_belief.probabilities[state.state_id]
            for state in model.hidden_states
            if state.cells[cell] == mass.value
        )
        if not math.isclose(
            joint_mass,
            mass.probability,
            rel_tol=0.0,
            abs_tol=_PROBABILITY_TOLERANCE,
        ):
            raise ValueError("planning joint belief must preserve runtime marginals")
```

Also reject a hidden-state space that cannot represent a runtime posterior hypothesis with positive mass: for every positive `BeliefMass`, at least one hidden state must carry that value for that cell.

- [ ] **Step 5: Normalize public failures at the planning typed boundary**

Wrap ordinary `Exception` from runtime belief resolution and the joint hook as `RuntimePlanningDecisionResolutionError` with `raise ... from error`. Do not catch `BaseException`.

Structural preflight and coupling violations also surface as `RuntimePlanningDecisionResolutionError`, and transition/observation/reward hooks must remain uncalled on these failures.

- [ ] **Step 6: Run the preflight/coupling subset**

Run:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected now: preflight, coupling, sanitized joint-context, and runtime-belief binding tests pass; solver/value/trace/science tests remain RED.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: bind narrative planning belief"
```

---

### Task 4: Implement Kernel Validation and the Deterministic Soft Bellman Solver

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: validated model, authored action map, root `PlanningBeliefState`.
- Produces: deterministic root action values, future soft values, canonical `PlanningValueRecord` and `PlanningBeliefUpdate` collections.

- [ ] **Step 1: Add one shared exact-distribution validator**

Implement:

```python
def _distribution(
    raw: object,
    *,
    expected_ids: tuple[str, ...],
    label: str,
) -> Mapping[str, float]:
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if set(raw) != set(expected_ids):
        raise ValueError(f"{label} must cover exactly the declared ids")
    values = {
        item_id: _probability(raw[item_id], label=f"{label} {item_id}")
        for item_id in expected_ids
    }
    if not math.isclose(
        math.fsum(values.values()),
        1.0,
        rel_tol=0.0,
        abs_tol=_PROBABILITY_TOLERANCE,
    ):
        raise ValueError(f"{label} must sum to 1")
    return MappingProxyType(values)
```

Use it for transition and observation outputs. Bool, NaN, infinity, negative values, missing/extra IDs, and non-unit totals must fail.

- [ ] **Step 2: Implement kernel-call helpers with sanitized contexts**

Implement `_transition_distribution(...)`, `_observation_distribution(...)`, and `_reward(...)` that construct only the approved context records and call the corresponding model hook.

Transition receives one current hidden state, one authored action, depth, candidate next states, parameters.

Observation receives one hypothetical next state, one authored action, depth, declared observation atoms, parameters.

Reward receives one current state/action/next-state/depth/parameters and must return `_finite(...)`.

Catch only ordinary `Exception` at the public runner boundary, preserving each hook exception as cause.

- [ ] **Step 3: Implement predicted belief and observation-conditioned Bayes update**

For one belief/action/depth:

```python
predicted = {
    next_state.state_id: math.fsum(
        belief.probabilities[state.state_id]
        * transition[(state.state_id, next_state.state_id)]
        for state in model.hidden_states
    )
    for next_state in model.hidden_states
}
```

For each observation:

```python
evidence = math.fsum(
    predicted[state.state_id]
    * observation_likelihood[(state.state_id, observation.observation_id)]
    for state in model.hidden_states
)
```

If `evidence == 0.0`, skip it completely.

Otherwise:

```python
posterior = PlanningBeliefState({
    state.state_id: (
        observation_likelihood[(state.state_id, observation.observation_id)]
        * predicted[state.state_id]
        / evidence
    )
    for state in model.hidden_states
})
```

Do not clip. Emit a `PlanningBeliefUpdate` for each positive-evidence observation branch.

- [ ] **Step 4: Implement immediate expected reward**

Use the exact formula:

```python
immediate = math.fsum(
    belief.probabilities[state.state_id]
    * transition[(state.state_id, next_state.state_id)]
    * reward[(state.state_id, next_state.state_id)]
    for state in model.hidden_states
    for next_state in model.hidden_states
)
```

Validate the resulting value is finite.

- [ ] **Step 5: Implement recursive Q/V solving with memoized semantic results**

Implement an internal solver whose memo key is:

```python
(depth, belief.content_hash)
```

For every action in `model.action_schedule[depth]`, compute immediate reward. If terminal depth, future contribution is `0.0`.

At non-terminal depth, for each positive-evidence observation branch recursively obtain `V_{depth+1}(posterior)` and compute:

```python
undiscounted_continuation = math.fsum(
    observation_probability * child_value
    for observation_probability, child_value in branches
)
expected_future_value = model.discount * undiscounted_continuation
total_value = immediate + expected_future_value
```

Store `expected_future_value` in `PlanningValueRecord` as the **already discounted** continuation contribution, so the record invariant is exactly:

```python
record.total_value == record.expected_immediate_reward + record.expected_future_value
```

At each depth:

```python
policy = finite_softmax(action_values, beta=model.beta)
policy = _validated_policy(policy, expected_actions=current_action_ids)
value = math.fsum(policy[action] * action_values[action] for action in current_action_ids)
```

Never use `max(action_values.values())` for the future value.

- [ ] **Step 6: Canonicalize trace accumulation independently of memo execution order**

Do not append records directly into a traversal-order list that changes when memoization hits. Accumulate by semantic keys:

```text
value key = (depth, belief_hash, action_id)
update key = (depth, prior_belief_hash, action_id, observation_id)
```

If a key is recomputed, require the new record equals the existing record. At the end, return tuples sorted by those exact keys.

This guarantees memoization changes performance only, not result payload/hash.

- [ ] **Step 7: Run solver/kernel tests**

Run the planning test file. Require kernel-schema, zero-evidence, no-information update, hand-computed soft-Bellman, and root-policy/MAP tests to pass.

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

- [ ] **Step 8: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: solve finite narrative planning"
```

---

### Task 5: Complete Result Binding, Audit Integrity, Replay, and Typed Failure Semantics

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: runtime belief state, root planning belief, solver root values/policy, canonical value/update records.
- Produces: self-validating `RuntimePlanningDecisionResult` and full public `run_runtime_planning_decision(...)` implementation.

- [ ] **Step 1: Finish `RuntimePlanningDecisionResult.__post_init__`**

Lock these invariants:

```text
model_id/model_hash/decision_id/actor_id are valid trimmed/hash values
step_index >= 0
ledger_hash is a valid hash
belief_state is RuntimeUncertainBeliefState
actor_id == belief_state.agent_id
step_index == belief_state.step_index
ledger_hash == belief_state.ledger_hash
planning_belief is PlanningBeliefState
action_values/action_policy share one exact non-empty root action set
action_values are finite non-bools
action_policy is finite/non-negative and sums to 1 within 1e-12
selected_action is lexical MAP of action_policy
value_records and belief_updates are tuples of the exact audit record types
```

Freeze all mappings/tuples canonically.

- [ ] **Step 2: Bind trace records back to the root solution**

The result must contain exactly one root `PlanningValueRecord` for each root action with:

```text
depth == 0
belief_hash == planning_belief.content_hash
action_id in action_values
record.total_value == action_values[action_id]
record.total_value == record.expected_immediate_reward + record.expected_future_value
```

Reject duplicate audit keys.

Every value record depth must be `< len(model.action_schedule)` when validated by the public runner before construction.

Every belief update must reference a positive observation probability and a posterior `PlanningBeliefState`.

Because the standalone result record does not carry the full model state/observation declarations, model-dependent audit validation remains in a private `_validated_result_against_solution(...)` call inside the runner; constructor-only validation must still reject internal inconsistencies it can determine from its own fields.

- [ ] **Step 3: Construct the final result only after all solver checks pass**

The public runner returns:

```python
RuntimePlanningDecisionResult(
    model_id=model.model_id,
    model_hash=model.content_hash,
    decision_id=decision.id,
    actor_id=decision.actor_id,
    step_index=belief_state.step_index,
    ledger_hash=belief_state.ledger_hash,
    belief_state=belief_state,
    planning_belief=root_belief,
    action_values=root_values,
    action_policy=root_policy,
    selected_action=_map_choice(root_policy),
    value_records=value_records,
    belief_updates=belief_updates,
)
```

Then immediately run private model-dependent result revalidation before returning, so constructor-bypass nested forgeries cannot masquerade as solver output.

- [ ] **Step 4: Normalize public error behavior exactly once**

Structure the public runner as:

```python
try:
    ... full preflight / belief / hooks / solver / result ...
except RuntimePlanningDecisionResolutionError:
    raise
except Exception as error:
    raise RuntimePlanningDecisionResolutionError(
        "runtime planning decision could not be resolved"
    ) from error
```

Do not catch `BaseException`.

- [ ] **Step 5: Run replay, forgery, hook-error, and complete planning tests**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: all non-scientific planning tests now pass, including exact replay/content hash and nested forgery rejection.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: bind narrative planning audit results"
```

---

### Task 6: Close the Scientific Model-Comparison Fixtures

**Files:**
- Modify only if implementation correction is required: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: complete planning runner plus existing reactive/intentional public runners used only by tests.
- Produces: behavior-level evidence for horizon-one equivalence, no-information Bayesian invariance, and value-of-information separation.

- [ ] **Step 1: Run only the scientific tests and inspect exact failures**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_horizon_one_policy_is_exactly_equivalent_across_planning_reactive_and_intentional \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_state_independent_observation_has_zero_bayesian_information_gain \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_value_of_information_separates_planning_from_reactive_and_intentional \
  -v
```

- [ ] **Step 2: If horizon-one equivalence fails, fix planning math rather than loosening equality**

The equality is intentionally exact. Ensure planning root values feed the same shared `finite_softmax` with the same beta and same lexical action IDs. Do not change the test to tolerance-based policy comparison.

- [ ] **Step 3: If no-information posterior invariance fails, fix Bayesian prediction/update**

State-independent observation likelihood must algebraically cancel in Bayes update. Do not special-case the fixture; correct distribution aggregation/normalization so every positive branch posterior equals predicted belief under canonical representation.

- [ ] **Step 4: If value-of-information separation fails, distinguish fixture mistakes from solver mistakes**

The informative and no-information variants must share the same root belief, transition kernel, rewards, discount, beta, and action schedule; only observation likelihood changes. If the fixture satisfies that condition and inspection still has no information value, inspect the continuation recursion. Do not add ad hoc inspection bonuses.

- [ ] **Step 5: Run the complete planning test file again**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: all planning tests pass.

- [ ] **Step 6: Commit only if production/test fixture changes were needed**

If a real correction was required:

```bash
git add narrative_dynamics/narrative/runtime_planning.py tests/test_narrative_runtime_planning.py
git commit -m "fix: close narrative planning science fixtures"
```

If no changes were needed, do not create an empty commit.

---

### Task 7: Export Exactly 13 Planning Names and Run Final Exact-Head Proof

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`
- Verify: all final scoped paths relative to integrated base.

**Interfaces:**
- Consumes: complete `runtime_planning.py` public API.
- Produces: exact narrative-scoped public surface, root isolation, final proof evidence, merge-ready PR.

- [ ] **Step 1: Add one planning import block to `narrative_dynamics/narrative/__init__.py`**

Add exactly:

```python
from narrative_dynamics.narrative.runtime_planning import (
    PlanningBeliefState,
    PlanningBeliefUpdate,
    PlanningHiddenState,
    PlanningObservation,
    PlanningObservationContext,
    PlanningRewardContext,
    PlanningTransitionContext,
    PlanningValueRecord,
    RuntimePlanningBeliefContext,
    RuntimePlanningDecisionModelSpec,
    RuntimePlanningDecisionResolutionError,
    RuntimePlanningDecisionResult,
    run_runtime_planning_decision,
)
```

- [ ] **Step 2: Add the same 13 names to `__all__` and nothing else**

Keep package root `narrative_dynamics/__init__.py` unchanged.

- [ ] **Step 3: Run the exact public-surface test**

```bash
python3 -m unittest tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: PASS; all 13 names are in `narrative_dynamics.narrative`, none are exported from package root.

- [ ] **Step 4: Run the full Python suite before final commit when local execution is available**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`, with test count equal to the prior integrated count plus the new planning tests.

- [ ] **Step 5: Commit exact exports**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative planning surface"
```

- [ ] **Step 6: Verify final diff scope before trusting CI**

Compare integrated base `8aae284fe02f27a76b6d00a651be152f49a9e6ed` to current head. Expected feature/runtime paths are:

```text
.github/workflows/proof.yml                                      modified (pre-existing isolated CI trigger refactor)
docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md  added
docs/superpowers/plans/2026-08-27-narrative-generic-planning-pomdp-v1.md         added
narrative_dynamics/narrative/runtime_planning.py                  added
narrative_dynamics/narrative/__init__.py                          modified
tests/test_narrative_runtime_planning.py                          added
tests/test_narrative_trust_api.py                                 modified
```

No scheduler, world, observation projection, cognition, intention, reactive, prison, model-comparison, root package, or Lean path may appear. Any discovered prerequisite requires its own explicit RED -> GREEN slice and PR documentation.

- [ ] **Step 7: Obtain authoritative exact-head GitHub Actions GREEN**

Fetch the proof workflow for the exact final head. Require:

```text
status = completed
conclusion = success
head_sha = exact final feature head
```

Inspect the verify job and require every step success:

```text
Resolve Lean dependencies
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

Do not infer GREEN from a previous head or from local tests.

- [ ] **Step 8: Final code review and PR readiness**

Review the final diff against the design spec section-by-section. Check PR review threads/comments. Fix Critical/Important findings before proceeding and rerun exact-head CI after any code change.

Update the draft PR body with:

```text
verified RED run/head
final GREEN run/head
full Python test count
exact final changed-path list
explicit note that scheduler dispatch remains intentionally unchanged
explicit note that the CI path-ignore refactor is an isolated prerequisite commit
```

Mark the PR ready for review only after the exact final proof is green and review has no blocker.

- [ ] **Step 9: Integration remains an explicit finishing decision**

Do not merge solely because this plan finished. At the finishing gate, present/execute the chosen integration action with `expected_head_sha` pinned to the verified final feature head. After merge, verify `proof/narrative-dynamics-v0` points to the returned merge commit and that its parents are the prior base and verified feature head. Then update issue #27 P1 Generic Planning / POMDP checklist while keeping the roadmap open for remaining workstreams.
