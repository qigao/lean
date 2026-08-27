# Narrative Generic Planning / POMDP V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrative-native deterministic finite-horizon POMDP decision sidecar that starts from admitted runtime belief, preserves those posterior marginals through an explicit joint coupling, values hypothetical transitions and future observations without objective-world access, and returns a complete root action policy plus lexical MAP action.

**Architecture:** Implement one isolated `runtime_planning.py` sidecar. It reuses `runtime_uncertain_belief_state(...)` for the only runtime information input, projects cell-marginal posterior semantics into a validated joint hidden-state coupling, solves a finite soft Bellman recursion over declared hidden states/observations/actions, and uses shared `finite_softmax` at every decision depth. Scheduler polymorphism, real world rollout, and released model-comparison integration stay out of scope.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `MappingProxyType`, `unittest`), existing `narrative_dynamics` typed IR/runtime cognition/attestation/content hashing, shared `grounded_goal_softmax.finite_softmax`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md`

## Global Constraints

- Integrated base is `proof/narrative-dynamics-v0` at `8aae284fe02f27a76b6d00a651be152f49a9e6ed`.
- Current feature branch is `work/narrative-generic-planning-pomdp-v1`.
- Branch already contains isolated CI trigger refactor commit `b328091c81217c12467bcf4610a04304a5bf3eae`; do not modify CI again unless a concrete workflow defect appears.
- Pure changes under `docs/superpowers/specs/**` and `docs/superpowers/plans/**` do not trigger proof CI. The first authoritative planning CI is therefore the test-only RED commit, not design/plan commits.
- Production planning scope is limited to `narrative_dynamics/narrative/runtime_planning.py` plus exact exports in `narrative_dynamics/narrative/__init__.py`.
- Test scope is `tests/test_narrative_runtime_planning.py` plus the exact public-surface extension in `tests/test_narrative_trust_api.py`.
- Do not modify `GenericNarrative`, `DomainSpec`, authored decisions, `runtime_cognition.py`, `runtime_intention.py`, `runtime_reactive.py`, `world.py`, `observation_projection.py`, `simulation.py`, released model-comparison infrastructure, prison adapters, or Lean sources.
- `runtime_planning.py` must not import `narrative_dynamics.narrative.world`, `observation_projection`, `simulation`, `runtime_intention`, or `runtime_reactive`.
- Public runner signature is exactly `run_runtime_planning_decision(story, domain, decision_id, ledger, model)`; no `WorldState`, RNG, explicit step, real transition model, or observation projection model argument is allowed.
- Planning hidden states and observation atoms are finite, narrative-grounded typed declarations. Recursion cannot dynamically create state or observation IDs.
- Joint-belief hook is mandatory. Marginalizing its joint distribution over every `planning_cell` must reproduce that cell's runtime posterior within absolute tolerance `1e-12`, relative tolerance `0`.
- Never silently multiply marginals in production. Product coupling exists only as an explicit test/model hook.
- Transition/observation distributions exactly cover their declared spaces, contain finite non-negative non-boolean values, and sum to one within absolute tolerance `1e-12`, relative tolerance `0`.
- Reward values and all derived Q/value terms are finite numeric non-booleans.
- Zero-evidence observations are impossible branches: do not recurse and do not emit `PlanningBeliefUpdate`.
- Do not clip probabilities. Invalid normalization is a typed planning resolution failure.
- Bellman recursion is soft at every depth: `pi = finite_softmax(Q, beta)` and `V = sum(pi[a] * Q[a])`; never replace future choice with hard `max`.
- V1 uses no RNG. Same inputs produce identical canonical result payloads/content hashes.
- Selected runtime action is deterministic lexical MAP of the root policy.
- Model identity binds runtime belief model, hidden/observation spaces, schedule, parameters, discount/beta, all four hook attestations, planning implementation identity, and shared softmax implementation identity.
- `narrative_dynamics.narrative` gains exactly 13 planning names; package root `narrative_dynamics` gains none.
- Use strict RED -> GREEN -> atomic commit discipline. No final GREEN claim without completed/success proof CI on the exact final feature head.

## File Structure

- `narrative_dynamics/narrative/runtime_planning.py` — V1 planning records, sanitized contexts, model/result contracts, preflight, joint coupling, finite Bayesian solver, audit trace, public runner.
- `narrative_dynamics/narrative/__init__.py` — exact 13 planning imports/exports only.
- `tests/test_narrative_runtime_planning.py` — complete RED contract suite, scientific fixtures, import/leakage isolation, replay/forgery tests.
- `tests/test_narrative_trust_api.py` — exact public API gains exactly 13 planning names; package-root isolation remains locked.
- `docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md` — approved source contract.
- `docs/superpowers/plans/2026-08-27-narrative-generic-planning-pomdp-v1.md` — this implementation plan.

---

### Task 1: Establish the Complete Test-Only RED Boundary

**Files:**
- Create: `tests/test_narrative_runtime_planning.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: existing `RuntimeEvidenceLedger`, `RuntimeBeliefModelSpec`, `RuntimeUncertainBeliefState`, `BeliefDistribution`, `StateCellRef`, `TypedValue`, `ActionOption`, `finite_softmax`; existing test helpers in `tests/test_narrative_runtime_cognition.py`, `tests/test_narrative_runtime_intention.py`, and `tests/test_narrative_runtime_reactive.py`.
- Produces: authoritative RED contract for all 13 public planning symbols and all V1 behavior before planning production exists.

- [ ] **Step 1: Create guarded imports and reusable planning fixtures**

Start `tests/test_narrative_runtime_planning.py` with:

```python
from __future__ import annotations

from dataclasses import fields, replace
import inspect
import math
import unittest

from grounded_goal_softmax import finite_softmax
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.ir import TypedValue
from tests.test_narrative_runtime_cognition import alert_cell, empty_runtime_case, phase_cell
from tests.test_narrative_runtime_intention import make_runtime_intentional_model

_PLANNING_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.narrative.runtime_planning as runtime_planning_module
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

Inside `NarrativeRuntimePlanningTests`, add:

```python
def require_planning(self):
    if _PLANNING_IMPORT_ERROR is not None:
        self.fail(
            "narrative runtime planning boundary is missing: "
            f"{_PLANNING_IMPORT_ERROR}"
        )


def make_model(
    self,
    *,
    joint=None,
    transition=None,
    observation=None,
    reward=None,
    planning_cells=None,
    observation_cells=(),
    hidden_states=None,
    observations=None,
    schedule=None,
    discount=1.0,
    beta=2.0,
    parameters=None,
    belief_model=None,
):
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

- [ ] **Step 2: Add record/model/public-signature RED tests**

Add these concrete tests:

```python
def test_hidden_state_observation_and_belief_records_are_canonical(self):
    self.require_planning()
    active = PlanningHiddenState(
        "active", {phase_cell(): TypedValue("PhaseState", "active")}
    )
    none = PlanningObservation("none", {})
    belief = PlanningBeliefState({"ready": 0.25, "active": 0.75})
    self.assertEqual(active.state_id, "active")
    self.assertEqual(none.observation_id, "none")
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

Also directly construct every context/audit record with valid values, then assert invalid negative depth, empty IDs, non-finite values, and zero/negative `PlanningBeliefUpdate.observation_probability` are rejected.

- [ ] **Step 3: Add preflight/coupling/kernel/solver/trace/error RED tests**

Add test methods with these exact names:

```text
test_preflight_rejects_story_ledger_context_cutoff_and_schedule_before_hooks
test_joint_belief_must_be_a_coupling_of_runtime_cell_marginals
test_hook_contexts_are_sanitized_and_module_has_no_world_capability
test_transition_observation_and_reward_hook_schemas_fail_typed
test_zero_evidence_observation_is_skipped_and_no_information_preserves_prediction
test_finite_horizon_soft_bellman_matches_hand_computed_values
test_shared_softmax_produces_complete_root_policy_and_lexical_map
test_hook_exceptions_are_wrapped_with_typed_causes
test_result_and_trace_forgery_is_rejected
test_fixed_inputs_replay_to_exact_result_and_content_hash
```

Lock each method to these concrete assertions:

1. **Preflight:** hook objects have `calls=[]`; forge ledger domain/story/chain data, choose a planning cell outside authored context, use a decision after source cutoff, give root schedule missing one authored action, and give a later schedule containing an undeclared action. Every case raises `RuntimePlanningDecisionResolutionError`; every hook call list stays empty.
2. **Coupling:** use planning cells `(phase_cell(), alert_cell())` and four hidden states covering all value combinations. `ProductCouplingHook` passes. A hook returning all mass on one state fails whenever either runtime marginal is non-degenerate; transition/observation/reward hooks remain uncalled.
3. **Sanitization/import isolation:** after a successful run, every captured context lacks attributes `story`, `domain`, `ledger`, `world`, `world_state`, `evidence`, `provenance`, and `runtime_evidence_history`. `inspect.getsource(runtime_planning_module)` contains none of `narrative.world`, `narrative.observation_projection`, `narrative.simulation`, `narrative.runtime_intention`, `narrative.runtime_reactive`.
4. **Kernel schemas:** transition/observation variants return non-mapping, missing key, extra key, bool, NaN, infinity, negative mass, and non-unit total. Reward variants return bool, NaN, infinity. Every case raises typed planning resolution error with a cause.
5. **No-information update:** for state-independent observation likelihoods, every positive branch posterior equals the predicted next-state belief by canonical payload; zero-evidence observations create no `PlanningBeliefUpdate`.
6. **Soft Bellman:** use two depths/two hidden states and deterministic transition. Choose values where hard-max continuation differs from `sum(pi * Q)`. Assert root `PlanningValueRecord.total_value`, `expected_immediate_reward`, `expected_future_value` to 12 decimal places and verify `expected_future_value` equals discounted soft continuation.
7. **Root policy/MAP:** exact tie gives equal root policy coordinates and lexical smallest action independent of input action ordering.
8. **Hook exceptions:** each of four hooks raises a distinct `RuntimeError`; public error is `RuntimePlanningDecisionResolutionError` and `__cause__` is the original runtime error.
9. **Forgery:** standalone result validation rejects a forged `ledger_hash` that disagrees with embedded belief state, forged root value record total, forged belief update/posterior internal inconsistency, and forged selected action. A successful runner result separately asserts `result.model_hash == model.content_hash`; do not require standalone result construction to infer the correct model from an arbitrary valid-format hash.
10. **Replay:** two identical calls produce equal result objects, equal `to_dict()`, and equal `content_hash`.

- [ ] **Step 4: Add the three scientific RED fixtures**

Add test methods with these exact names:

```text
test_horizon_one_policy_is_exactly_equivalent_across_planning_reactive_and_intentional
test_state_independent_observation_has_zero_bayesian_information_gain
test_value_of_information_separates_planning_from_reactive_and_intentional
```

Fixture contracts:

- Horizon one: identity transition, one no-information observation, same immediate action-score table and same beta. Planning, reactive, and a single-goal intentional fixture have exactly equal `action_policy` mappings and same lexical MAP action.
- State-independent observation: at least two observation atoms with identical likelihood vectors under every hidden state. Every positive `PlanningBeliefUpdate.posterior` equals the predicted next-state belief. Do not assert equality to a myopic utility because delayed actions can change value without information gain.
- Value of information: root actions `inspect`, `act-a`, `act-b`; future schedule only `act-a`, `act-b`. Informative observation makes inspection valuable net of cost. Reactive/intentional fixtures use only current admitted information. Planning policy differs from both. Replacing only observation likelihood with state-independent likelihood removes inspection's information advantage while reward, transition, discount, beta, belief, and schedule remain fixed.

- [ ] **Step 5: Extend the exact narrative public-surface RED**

Add exactly these names to `_EXPECTED_PUBLIC_API` in `tests/test_narrative_trust_api.py`:

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

Keep the existing root-isolation assertion unchanged; these names remain absent from `dir(narrative_dynamics)`.

- [ ] **Step 6: Run the RED suite**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
python3 -m unittest tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected before production exists:

```text
planning tests fail because narrative_dynamics.narrative.runtime_planning is missing
trust public-surface test fails with exactly the 13 planned names missing
```

If local execution is unavailable, keep the test-only commit and use GitHub Actions as authoritative RED; do not weaken tests.

- [ ] **Step 7: Commit test-only RED**

```bash
git add tests/test_narrative_runtime_planning.py tests/test_narrative_trust_api.py
git commit -m "test: define narrative planning pomdp contract"
```

This changes non-ignored `tests/**`, so proof CI must trigger.

- [ ] **Step 8: Create/update a draft PR and capture authoritative exact-head RED**

Create a draft PR against `proof/narrative-dynamics-v0` if none exists. Require the proof run for the exact test-only head to reach:

```text
status = completed
conclusion = failure
head_sha = exact test-only RED commit
```

Inspect job logs. Accept RED only when prior gates remain green up to the new planning failures and failures are caused by missing planning module/public names rather than unrelated regressions.

---

### Task 2: Add Canonical Planning Records and Model Identity

**Files:**
- Create: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: `StateCellRef`, `TypedValue`, `ActionOption`, `BeliefDistribution`, `RuntimeBeliefModelSpec`, `measure_implementation`, `stable_content_hash`, `finite_softmax`.
- Produces: all 13 module-local public symbols; canonical data/model contracts; runner symbol with final signature.

- [ ] **Step 1: Implement scalar/canonical helpers with only allowed imports**

Use:

```python
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_PROBABILITY_TOLERANCE = 1e-12


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _hash(value: object, *, label: str) -> str:
    if not isinstance(value, str) or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be a sha256 content hash")
    return value


def _depth(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


def _finite(value: object, *, label: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _probability(value: object, *, label: str) -> float:
    number = _finite(value, label=label)
    if number < 0.0 or number > 1.0:
        raise ValueError(f"{label} must be in [0, 1]")
    return number


def _cell_key(cell: StateCellRef) -> tuple[str, str, str]:
    return (
        cell.subject.entity_type,
        cell.subject.entity_id,
        cell.state_variable,
    )


def _freeze_parameter(value: object, *, label: str) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} floats must be finite")
        return value
    if isinstance(value, Mapping):
        frozen = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"{label} mapping keys must be non-empty strings")
            frozen[key] = _freeze_parameter(item, label=f"{label}.{key}")
        return MappingProxyType({key: frozen[key] for key in sorted(frozen)})
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_parameter(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(
        f"{label} values must be canonical scalars, mappings, lists, or tuples"
    )


def _freeze_parameters(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("runtime planning model parameters must be a mapping")
    frozen = _freeze_parameter(value, label="runtime planning model parameters")
    if not isinstance(frozen, Mapping):
        raise TypeError("runtime planning model parameters must be a mapping")
    return frozen


def _parameter_payload(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _parameter_payload(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_parameter_payload(item) for item in value]
    return value
```

- [ ] **Step 2: Implement finite-space data records**

Implement exactly these records:

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
- constructors validate type/shape facts knowable without `DomainSpec`;
- `PlanningBeliefState` requires non-empty probabilities, finite non-negative non-bool entries, and normalized sum within `1e-12`.

Domain typing, exact declared cell coverage, duplicate semantic state/observation payloads, and exact model hidden-state coverage are runner/model preflight responsibilities because they require `DomainSpec` or model declarations.

- [ ] **Step 3: Implement sanitized hook contexts and audit records**

Implement:

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

Detach/canonicalize nested mappings/tuples. Audit numbers are finite. `PlanningBeliefUpdate.observation_probability` must be strictly positive and at most one. Depths are non-negative.

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

`to_dict()` binds:

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

- [ ] **Step 5: Define result/error/public runner symbols with final types**

Define:

```python
class RuntimePlanningDecisionResolutionError(ValueError):
    """Runtime posterior semantics could not produce a valid planning decision."""


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

At this task, implement constructor validations that are model-independent: valid IDs/hashes, actor/step/ledger match embedded `belief_state`, finite `action_values`, normalized `action_policy`, exact shared action keys, lexical MAP selection, typed/canonical audit tuples, root value-record consistency where determinable from result fields. Model-dependent exact hidden-state/schedule validation is added at the runner boundary in Task 5.

Define the public runner with final signature and temporary typed RED behavior:

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

Add module-local `__all__` containing exactly the 13 spec names.

- [ ] **Step 6: Run records/model subset**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: import succeeds; record/model/public-signature tests pass; semantic runner tests remain RED with typed solver-not-implemented error.

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
- Produces: validated decision/model/ledger boundary and exact root `PlanningBeliefState` whose marginals equal runtime posterior semantics.

- [ ] **Step 1: Reconstruct model and ledger to reject constructor-bypass forgeries**

Implement `_validated_model(model)` by constructing a fresh `RuntimePlanningDecisionModelSpec` from every public field and requiring equal canonical payload/content hash.

Implement `_validated_ledger(story, domain, ledger)` by reconstructing:

```python
validated = RuntimeEvidenceLedger(
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

Then require:

```python
if validated.domain_id != domain.domain_id:
    raise ValueError("runtime planning ledger domain id mismatch")
if validated.domain_version != domain.version:
    raise ValueError("runtime planning ledger domain version mismatch")
if validated.domain_spec_hash != domain.content_hash:
    raise ValueError("runtime planning ledger domain spec hash mismatch")
if validated.source_story_hash != story.content_hash:
    raise ValueError("runtime planning ledger source story mismatch")
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
9. validate every non-`None` typed value through the existing domain state-variable typing path used by narrative validation;
10. reject duplicate hidden-state semantic cell payloads under different IDs;
11. reject duplicate observation semantic cue payloads under different IDs;
12. if observation cells are empty, require exactly one observation with empty cues;
13. root schedule equals complete authored action ID set;
14. every later schedule is non-empty subset of authored action IDs.

Build lexical `action_by_id` from authored `ActionOption` values.

- [ ] **Step 3: Resolve runtime belief for exactly planning cells**

```python
belief_state = runtime_uncertain_belief_state(
    story,
    domain,
    decision.actor_id,
    ledger,
    model.belief_model,
    model.planning_cells,
)
posterior = MappingProxyType({
    cell: belief_state.cells[cell].posterior
    for cell in sorted(model.planning_cells, key=_cell_key)
})
```

Do not pass `belief_state`, ledger evidence, support hashes, story, or domain into `joint_belief_hook`.

- [ ] **Step 4: Validate joint hook as exact coupling**

```python
raw_joint = model.joint_belief_hook(
    RuntimePlanningBeliefContext(
        posterior=posterior,
        hidden_states=model.hidden_states,
        parameters=model.parameters,
    )
)
root_belief = _planning_belief_from_raw(
    raw_joint,
    expected_ids=tuple(state.state_id for state in model.hidden_states),
)
```

`_planning_belief_from_raw` requires mapping shape, exact IDs, and `PlanningBeliefState` normalization.

Then preserve every runtime marginal:

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

For each positive runtime `BeliefMass`, require at least one hidden state carrying that value for the cell.

- [ ] **Step 5: Normalize preflight/belief/joint failures at typed boundary**

Public runner will eventually own one outer exception normalization. During this task, implement an internal `_resolve_root_planning_belief(...)` that raises ordinary `TypeError`/`ValueError`/runtime belief errors; the temporary public runner catches ordinary `Exception` and wraps it as `RuntimePlanningDecisionResolutionError` with the original exception as `__cause__`. Do not catch `BaseException`.

- [ ] **Step 6: Run preflight/coupling subset**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: preflight, coupling, sanitized joint-context, runtime-belief binding tests pass; solver/value/trace/science tests remain RED.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: bind narrative planning belief"
```

---

### Task 4: Implement Kernel Validation and Deterministic Soft Bellman Solver

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: validated model, authored action map, root `PlanningBeliefState`.
- Produces: deterministic root values, future soft values, canonical `PlanningValueRecord` and `PlanningBeliefUpdate` collections.

- [ ] **Step 1: Add one exact distribution validator**

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

Use for transition and observation output. Bool, NaN, infinity, negative values, missing/extra IDs, non-unit total fail.

- [ ] **Step 2: Implement sanitized kernel helpers**

Implement three helpers:

```text
_transition_distribution(model, depth, state, action) -> Mapping[state_id, probability]
_observation_distribution(model, depth, next_state, action) -> Mapping[observation_id, probability]
_reward(model, depth, state, action, next_state) -> finite float
```

Each helper constructs only its approved context record and passes `model.parameters`. It cannot receive story/domain/ledger/world/provenance values.

- [ ] **Step 3: Implement predicted belief and Bayesian observation update**

Cache all transition rows needed for one action/depth. Compute and immediately validate predicted belief:

```python
predicted = PlanningBeliefState({
    next_state.state_id: math.fsum(
        belief.probabilities[state.state_id]
        * transition[(state.state_id, next_state.state_id)]
        for state in model.hidden_states
    )
    for next_state in model.hidden_states
})
```

For one observation:

```python
evidence = math.fsum(
    predicted.probabilities[state.state_id]
    * observation_likelihood[(state.state_id, observation.observation_id)]
    for state in model.hidden_states
)
```

If `evidence == 0.0`, skip branch. Otherwise:

```python
posterior = PlanningBeliefState({
    state.state_id: (
        observation_likelihood[(state.state_id, observation.observation_id)]
        * predicted.probabilities[state.state_id]
        / evidence
    )
    for state in model.hidden_states
})
```

Emit one `PlanningBeliefUpdate` per positive-evidence observation.

- [ ] **Step 4: Implement immediate expected reward**

```python
immediate = _finite(
    math.fsum(
        belief.probabilities[state.state_id]
        * transition[(state.state_id, next_state.state_id)]
        * reward[(state.state_id, next_state.state_id)]
        for state in model.hidden_states
        for next_state in model.hidden_states
    ),
    label="planning expected immediate reward",
)
```

- [ ] **Step 5: Implement recursive Q/V solving with memoization**

Memo key is `(depth, belief.content_hash)`.

For every action in `model.action_schedule[depth]`, compute immediate reward. Terminal depth has future contribution `0.0`.

At earlier depth:

```python
undiscounted_continuation = math.fsum(
    observation_probability * child_value
    for observation_probability, child_value in branches
)
expected_future_value = _finite(
    model.discount * undiscounted_continuation,
    label="planning expected future value",
)
total_value = _finite(
    immediate + expected_future_value,
    label="planning total action value",
)
```

Store **discounted** continuation in `PlanningValueRecord.expected_future_value`, so invariant is:

```python
record.total_value == record.expected_immediate_reward + record.expected_future_value
```

At every depth:

```python
raw_policy = finite_softmax(action_values, beta=model.beta)
policy = _validated_policy(raw_policy, expected_actions=current_action_ids)
value = _finite(
    math.fsum(policy[action] * action_values[action] for action in current_action_ids),
    label="planning soft belief value",
)
```

Never use hard-max continuation.

- [ ] **Step 6: Make audit trace independent of traversal/memo hit order**

Accumulate in dictionaries keyed by:

```text
value key = (depth, belief_hash, action_id)
update key = (depth, prior_belief_hash, action_id, observation_id)
```

If recomputed key differs from existing record, fail. Final tuples are sorted by those exact keys.

- [ ] **Step 7: Run solver/kernel tests**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Require kernel-schema, zero-evidence, no-information update, hand-computed soft-Bellman, root-policy/MAP tests to pass.

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
- Produces: self-validating `RuntimePlanningDecisionResult` and full public runner.

- [ ] **Step 1: Finish model-independent result invariants**

Require:

```text
model_id/model_hash/decision_id/actor_id are valid IDs/hashes
step_index >= 0
ledger_hash valid
belief_state is RuntimeUncertainBeliefState
actor_id == belief_state.agent_id
step_index == belief_state.step_index
ledger_hash == belief_state.ledger_hash
planning_belief is PlanningBeliefState
action_values/action_policy share one exact non-empty action set
action_values finite non-bools
action_policy finite/non-negative and sums to 1 within 1e-12
selected_action is lexical MAP
value_records and belief_updates are exact audit record tuples
```

Freeze mappings/tuples canonically.

- [ ] **Step 2: Bind root trace to root values**

Result contains exactly one root `PlanningValueRecord` per root action with:

```text
depth == 0
belief_hash == planning_belief.content_hash
action_id in action_values
record.total_value == action_values[action_id]
record.total_value == record.expected_immediate_reward + record.expected_future_value
```

Reject duplicate audit keys. Every belief update has positive observation probability and typed posterior.

Standalone result cannot know correct model content hash from hash syntax alone. The successful runner test must assert `result.model_hash == model.content_hash`; do not add a constructor test expecting arbitrary valid-format forged model hash rejection.

- [ ] **Step 3: Add private model-dependent result validation**

Inside runner, before returning, validate:

```text
result.model_id == model.model_id
result.model_hash == model.content_hash
result.planning_belief keys == exact hidden-state IDs
root action keys == model.action_schedule[0]
every value record depth < len(model.action_schedule)
every value record action belongs to schedule at that depth
every belief update observation ID belongs to model observations
```

Also compare result payload fields against solver-produced root values/policy/trace so constructor-bypass nested values cannot masquerade as solver output.

- [ ] **Step 4: Construct final result and normalize errors exactly once**

Public runner structure becomes:

```python
def run_runtime_planning_decision(story, domain, decision_id, ledger, model):
    try:
        validated_model = _validated_model(model)
        validated_ledger = _validated_ledger(story, domain, ledger)
        decision, action_by_id = _preflight_decision(
            story,
            domain,
            decision_id,
            validated_ledger,
            validated_model,
        )
        belief_state, root_belief = _resolve_root_planning_belief(
            story,
            domain,
            decision,
            validated_ledger,
            validated_model,
        )
        root_values, root_policy, value_records, belief_updates = _solve_planning(
            validated_model,
            action_by_id,
            root_belief,
        )
        result = RuntimePlanningDecisionResult(
            model_id=validated_model.model_id,
            model_hash=validated_model.content_hash,
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
        _validate_result_against_solution(result, validated_model, root_values, root_policy)
        return result
    except RuntimePlanningDecisionResolutionError:
        raise
    except Exception as error:
        raise RuntimePlanningDecisionResolutionError(
            "runtime planning decision could not be resolved"
        ) from error
```

Do not catch `BaseException`.

- [ ] **Step 5: Run replay/forgery/error tests**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: all non-scientific planning tests pass, including exact replay/content hash and nested forgery rejection.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/narrative/runtime_planning.py
git commit -m "feat: bind narrative planning audit results"
```

---

### Task 6: Close Scientific Fixtures

**Files:**
- Modify only when a demonstrated solver defect requires it: `narrative_dynamics/narrative/runtime_planning.py`
- Test: `tests/test_narrative_runtime_planning.py`

**Interfaces:**
- Consumes: complete planning runner plus existing reactive/intentional runners used only by tests.
- Produces: behavior evidence for horizon-one equivalence, no-information Bayesian invariance, value-of-information separation.

- [ ] **Step 1: Run only scientific tests**

```bash
python3 -m unittest \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_horizon_one_policy_is_exactly_equivalent_across_planning_reactive_and_intentional \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_state_independent_observation_has_zero_bayesian_information_gain \
  tests.test_narrative_runtime_planning.NarrativeRuntimePlanningTests.test_value_of_information_separates_planning_from_reactive_and_intentional \
  -v
```

- [ ] **Step 2: Preserve exact horizon-one equivalence**

If it fails, fix planning root math so the same score table, beta, and action IDs reach shared `finite_softmax`. Do not loosen equality to tolerance-based policy comparison.

- [ ] **Step 3: Preserve no-information Bayes invariance**

If it fails, correct distribution aggregation/normalization so state-independent likelihood cancels algebraically and every positive branch posterior equals predicted belief. Do not special-case fixture IDs.

- [ ] **Step 4: Preserve value-of-information contrast**

Informative/no-information variants share root belief, transition, reward, discount, beta, schedule; only observation likelihood changes. If inspection still lacks information value with a valid fixture, inspect continuation recursion. Do not add an inspection bonus.

- [ ] **Step 5: Run complete planning file**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_runtime_planning.py' -v
```

Expected: all planning tests pass.

- [ ] **Step 6: Commit only if a real correction was needed**

```bash
git add narrative_dynamics/narrative/runtime_planning.py tests/test_narrative_runtime_planning.py
git commit -m "fix: close narrative planning science fixtures"
```

If no file changed, do not create an empty commit.

---

### Task 7: Export Exactly 13 Names and Run Final Exact-Head Proof

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`
- Test: `tests/test_narrative_trust_api.py`
- Verify: all final scoped paths relative to integrated base.

**Interfaces:**
- Consumes: complete `runtime_planning.py` API.
- Produces: exact narrative-scoped surface, root isolation, final proof evidence, merge-ready PR.

- [ ] **Step 1: Add one planning import block**

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

- [ ] **Step 3: Run exact public-surface test**

```bash
python3 -m unittest tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation -v
```

Expected: PASS; all 13 names are narrative-scoped and absent from package root.

- [ ] **Step 4: Run full Python suite when local execution is available**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`, with count equal to integrated baseline plus new planning tests.

- [ ] **Step 5: Commit exact exports**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative planning surface"
```

- [ ] **Step 6: Verify final diff scope**

Compare integrated base `8aae284fe02f27a76b6d00a651be152f49a9e6ed` to final head. Expected paths:

```text
.github/workflows/proof.yml                                      modified (isolated CI trigger refactor)
docs/superpowers/specs/2026-08-27-narrative-generic-planning-pomdp-v1-design.md  added
docs/superpowers/plans/2026-08-27-narrative-generic-planning-pomdp-v1.md         added
narrative_dynamics/narrative/runtime_planning.py                  added
narrative_dynamics/narrative/__init__.py                          modified
tests/test_narrative_runtime_planning.py                          added
tests/test_narrative_trust_api.py                                 modified
```

No scheduler, world, observation projection, cognition, intention, reactive, prison, model-comparison, root package, or Lean path may appear. Any prerequisite gets an explicit isolated RED -> GREEN slice and PR documentation.

- [ ] **Step 7: Obtain authoritative exact-head GitHub Actions GREEN**

Require proof run:

```text
status = completed
conclusion = success
head_sha = exact final feature head
```

Require verify job success for:

```text
Resolve Lean dependencies
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

Do not infer GREEN from previous head or local tests.

- [ ] **Step 8: Final code review and PR readiness**

Review diff against spec section-by-section and inspect review threads/comments. Fix Critical/Important findings and rerun exact-head CI after code changes.

Update draft PR body with:

```text
verified RED run/head
final GREEN run/head
full Python test count
exact final changed-path list
scheduler dispatch intentionally unchanged
CI path-ignore refactor is isolated prerequisite commit
```

Mark ready only after exact final proof is green and review has no blocker.

- [ ] **Step 9: Integration remains explicit finishing decision**

At finishing gate, execute the chosen integration action with `expected_head_sha` pinned to verified final head. After merge, verify `proof/narrative-dynamics-v0` points to returned merge commit and its parents are prior base plus verified feature head. Update issue #27 P1 Generic Planning / POMDP checklist and keep roadmap open for remaining workstreams.
