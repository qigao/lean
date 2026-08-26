# Narrative Multi-Step Scheduler V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic static multi-agent scheduler that closes the runtime loop across repeated rounds from admitted percepts through runtime belief, intentional choice, atomic world transition, and next-round percept admission.

**Architecture:** Add two narrative sidecars. `runtime_intention.py` consumes `RuntimeEvidenceLedger`, materializes `RuntimeUncertainBeliefState`, strips it to a provenance-free posterior semantic view, and reuses the existing intentional mathematical helpers without modifying `intention.py`; `simulation.py` statically schedules one canonical authored decision template per configured agent, executes all selections against one prior ledger, performs one atomic world transition, admits next percepts exactly once, and chains immutable simulation states into trajectories.

**Tech Stack:** Python standard library (`dataclasses`, `MappingProxyType`, `unittest`, `inspect`), existing `stable_content_hash`, `measure_implementation`, Runtime Cognition V1, authored intention helpers, World Transition V1, Observation Projection V1 through Runtime Percept Admission V1, and GitHub Actions `proof` CI.

**Spec:** `docs/superpowers/specs/2026-08-26-narrative-multi-step-scheduler-v1-design.md`

## Global Constraints

- Integrated research base before this feature: `ea45389354bf9b8dff4c18917e2bca3e3e83a939` on `proof/narrative-dynamics-v0`.
- Approved spec commit: `e10ed369d1908b12a2592c07aa8ffb241e526cb2` on `work/narrative-multi-step-scheduler-v1`.
- Work only on `work/narrative-multi-step-scheduler-v1`; do not modify `proof/narrative-dynamics-v0` or `master` during implementation.
- Static scheduling only: every configured agent decides exactly once per round; no silent skip. Inactivity is an explicit authored wait/noop action.
- Reuse authored `Decision` only as immutable actor/type/context/action-space template. Never synthesize runtime `Decision` or `ActionOption` values.
- Authored `logical_time` and runtime `step_index` remain distinct. If `source_at_time` is numeric, every configured decision template must satisfy `decision.logical_time <= source_at_time`.
- `decision.context_cells` are exactly the tracked runtime belief cells used for that runtime decision.
- Runtime intentional action selection may depend only on canonical decision template semantics, posterior distributions, `GoalModelSpec`, and `ChoiceModelSpec`; it must not read ledger/world/projection/transition identities or objective world objects.
- `intention.py` must remain byte-for-byte unchanged. `measure_implementation()` measures complete module bytes, so moving helpers out of it would silently change existing authored intentional model identity.
- `runtime_intention.py` may import exactly `_goal_scores`, `_goal_state`, `_conditional_action_policies`, and `_marginal_action_policy` from `intention.py`; these helpers receive a private posterior-only semantic adapter, never a full `RuntimeUncertainBeliefState`.
- Runtime intentional model identity binds both the runtime sidecar implementation identity and the existing authored `IntentionalDecisionModelSpec` implementation identity.
- Scheduler never calls `project_world_observations()` directly. It calls `admit_world_percepts()` exactly once after a successful world transition; admission owns the projection trust boundary.
- Observation Projection may emit evidence for passive observers not configured as scheduled actors. The simulation-wide ledger keeps it; only configured agents are materialized for cognition/decision.
- Every successful `SimulationState` must satisfy `state.step_index == world_state.step_index == evidence_ledger.current_step_index` and `evidence_ledger.current_world_state_hash == world_state.content_hash` plus exact domain/story/cutoff identity equality.
- A scheduler round is atomic at the returned-artifact boundary: any cognition/selection failure causes zero world/projection execution; world failure causes zero projection/admission; admission failure returns no successful next simulation state.
- V1 adds exactly 16 narrative-scoped public names and none at package root.
- Do not modify authored IR, DomainSpec, `intention.py`, Runtime Cognition, Runtime Perception, World Transition, Observation Projection, root package exports, or Lean sources. If implementation appears to require one of those paths, STOP and treat it as architecture expansion.
- Final feature diff must contain exactly eight paths: spec, this plan, `runtime_intention.py`, `simulation.py`, narrative `__init__.py`, two dedicated test modules, and `tests/test_narrative_trust_api.py`.
- Baseline at integrated base is exactly 513 Python tests. This plan adds 12 runtime-intention methods and 16 simulation methods, so final discovery must report exactly **541 tests**. Any other count requires investigation.
- Follow complete test-only RED -> exact-head CI RED -> staged GREEN commits -> export-only RED -> exact final CI.
- Do not merge the final PR without explicit user instruction.

## File Map

- `docs/superpowers/specs/2026-08-26-narrative-multi-step-scheduler-v1-design.md`: approved architecture and 76 required invariants.
- `docs/superpowers/plans/2026-08-26-narrative-multi-step-scheduler-v1.md`: this executable TDD plan.
- `narrative_dynamics/narrative/runtime_intention.py`: runtime intentional model/result, posterior-only semantic adapter, and runtime belief -> goal -> choice execution.
- `narrative_dynamics/narrative/simulation.py`: runtime agent/simulation records, initialization, one-step scheduling, and finite trajectory execution.
- `narrative_dynamics/narrative/__init__.py`: final narrative-scoped export of exactly 16 approved names.
- `tests/test_narrative_runtime_intention.py`: 12 methods locking runtime action-selection semantics and no-provenance boundary.
- `tests/test_narrative_simulation.py`: 16 methods locking scheduler/static-round/trajectory semantics and two-round causal closure.
- `tests/test_narrative_trust_api.py`: exact 16-name API addition and root-isolation lock.

## Planned Test Inventory

### `NarrativeRuntimeIntentionTests` — 12 methods

1. `test_runtime_intentional_model_identity_binds_nested_models_and_both_implementation_identities`
2. `test_runtime_intentional_result_record_binds_semantic_goal_hash_and_full_audit_belief`
3. `test_runtime_intentional_public_signatures_exclude_world_and_step_inputs`
4. `test_empty_ledger_runtime_selection_matches_authored_intentional_math`
5. `test_runtime_percept_changes_posterior_goal_policy_and_selected_action`
6. `test_same_posterior_semantics_ignore_hidden_provenance_for_policy`
7. `test_decision_template_cutoff_actor_type_and_context_fail_closed_before_likelihood`
8. `test_context_cells_are_exact_runtime_tracked_cells`
9. `test_action_and_goal_hypothesis_coverage_fail_typed`
10. `test_exact_ties_use_lexical_runtime_action_without_input_order_effect`
11. `test_runtime_selection_wraps_belief_goal_choice_failures_with_typed_causes`
12. `test_authored_intentional_runner_remains_semantically_unchanged`

### `NarrativeSimulationTests` — 16 methods

1. `test_simulation_model_identity_canonicalizes_agents_and_binds_nested_models`
2. `test_simulation_model_rejects_duplicate_agents_templates_and_domain_mismatch`
3. `test_simulation_state_from_story_binds_exact_world_ledger_model_and_cutoff`
4. `test_initialization_rejects_story_agent_template_type_and_cutoff_mismatch_before_hooks`
5. `test_static_round_runs_every_agent_once_against_same_prior_ledger_and_builds_exact_intents`
6. `test_cognition_failure_blocks_all_world_projection_and_next_state`
7. `test_world_failure_blocks_projection_and_preserves_prior_state`
8. `test_projection_or_admission_failure_returns_no_successful_next_state`
9. `test_agent_input_order_does_not_change_step_hash_or_semantics`
10. `test_same_authored_template_repeats_across_rounds_without_logical_time_mutation`
11. `test_passive_observer_evidence_and_scheduled_agent_blackout_are_legal`
12. `test_two_round_action_world_percept_belief_action_causal_closure_without_story_mutation`
13. `test_trajectory_chain_and_final_state_are_exact`
14. `test_deterministic_replay_produces_exact_same_trajectory_hash`
15. `test_round_validation_and_trajectory_failure_report_exact_step_with_typed_cause`
16. `test_forged_prior_simulation_state_rejects_before_any_agent_hook`

---

### Task 1: Commit the complete test-only RED contract

**Files:**
- Create: `tests/test_narrative_runtime_intention.py`
- Create: `tests/test_narrative_simulation.py`
- Modify: `tests/test_narrative_trust_api.py`

**Interfaces:**
- Consumes: merged Runtime Perception/Cognition V1, existing authored intention records/helpers, World Transition V1, Observation Projection V1, and current trust API surface.
- Produces: the exact 28 new behavior methods above plus the exact 16-name scoped public API expectation.

- [ ] **Step 1: Add guarded runtime-intention imports and exact existing-fixture helpers**

At the top of `tests/test_narrative_runtime_intention.py`:

```python
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    GoalModelSpec,
    GoalSpec,
    GoalState,
    IntentionalDecisionModelSpec,
    run_intentional_decision,
)
from narrative_dynamics.narrative.ir import TypedValue
from tests.test_narrative_runtime_cognition import (
    SemanticRuntimeLikelihood,
    empty_runtime_case,
    extend_ledger,
    make_runtime_belief_model,
    phase_cell,
)

_RUNTIME_INTENTION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_intention import (
        RuntimeIntentionalDecisionModelSpec,
        RuntimeIntentionalDecisionResult,
        RuntimeIntentionalDecisionResolutionError,
        run_runtime_intentional_decision,
    )
except ImportError as error:
    _RUNTIME_INTENTION_IMPORT_ERROR = error
```

Every method begins with `self.require_runtime_intention()`, implemented as:

```python
def require_runtime_intention(self) -> None:
    if _RUNTIME_INTENTION_IMPORT_ERROR is not None:
        self.fail(
            "narrative runtime intention boundary is missing: "
            f"{_RUNTIME_INTENTION_IMPORT_ERROR}"
        )
```

The existing canonical runtime story uses `d-a1-phase`, decision type `phase-choice`, context cell `phase_cell()`, and actions `a1-ready|a1-active`. Use those exact values.

Define:

```python
def _value_hash(value: TypedValue) -> str:
    return stable_content_hash(value.to_dict())


def make_goal_model(*, reverse: bool = False, active_pressure: float = 2.0):
    cell = phase_cell()
    ready = _value_hash(TypedValue("PhaseState", "ready"))
    active = _value_hash(TypedValue("PhaseState", "active"))
    stay = GoalSpec(
        "stay",
        1.0,
        {cell: 1.0},
        {cell: {ready: 1.0, active: 0.0}},
    )
    engage = GoalSpec(
        "engage",
        active_pressure,
        {cell: 1.0},
        {cell: {ready: 0.0, active: 1.0}},
    )
    goals = (engage, stay) if reverse else (stay, engage)
    return GoalModelSpec("runtime-goals", "1", 8.0, goals)


def make_choice_model(*, reverse: bool = False):
    values = {
        "stay": {"a1-ready": 3.0, "a1-active": 0.0},
        "engage": {"a1-ready": 0.0, "a1-active": 3.0},
    }
    if reverse:
        values = {
            key: dict(reversed(tuple(value.items())))
            for key, value in reversed(tuple(values.items()))
        }
    return ChoiceModelSpec("runtime-choice", "1", 8.0, values)


def make_runtime_intentional_model(*, belief=None, goal=None, choice=None):
    return RuntimeIntentionalDecisionModelSpec(
        "runtime-intentional",
        "1",
        ("phase-choice",),
        make_runtime_belief_model() if belief is None else belief,
        make_goal_model() if goal is None else goal,
        make_choice_model() if choice is None else choice,
    )
```

- [ ] **Step 2: Add all 12 runtime-intention RED methods**

Lock the exact public signature:

```python
self.assertEqual(
    tuple(inspect.signature(run_runtime_intentional_decision).parameters),
    ("story", "domain", "decision_id", "ledger", "model"),
)
for forbidden in ("world_state", "world_step", "projection_result", "step_index"):
    self.assertNotIn(
        forbidden,
        inspect.signature(run_runtime_intentional_decision).parameters,
    )
```

Model identity test must compare nested model changes and exact measured identities:

```python
payload = model.to_dict()
self.assertEqual(
    payload["runtime_implementation_identity"],
    measure_implementation(RuntimeIntentionalDecisionModelSpec).manifest_identity(),
)
self.assertEqual(
    payload["authored_intentional_implementation_identity"],
    measure_implementation(IntentionalDecisionModelSpec).manifest_identity(),
)
```

Hidden-provenance test builds two ledgers with the same semantic row:

```python
row = (
    "a1",
    "vision",
    phase_cell(),
    "equals",
    TypedValue("PhaseState", "active"),
)
left_ledger = extend_ledger(initial, (row,), branch="left")
right_ledger = extend_ledger(initial, (row,), branch="right")
```

and asserts:

```python
self.assertNotEqual(left.belief_state.ledger_hash, right.belief_state.ledger_hash)
self.assertEqual(left.goal_state.policy, right.goal_state.policy)
self.assertEqual(left.action_policy, right.action_policy)
self.assertEqual(left.selected_action, right.selected_action)
self.assertNotEqual(left.content_hash, right.content_hash)
```

For the record-constructor test, build an empty-ledger runtime belief using `runtime_uncertain_belief_state()` and compute its exact semantic hash in the test with:

```python
semantic_payload = {
    "cells": [
        {
            "cell": phase_cell().to_dict(),
            "posterior": belief.cells[phase_cell()].posterior.to_dict(),
        }
    ]
}
semantic_hash = stable_content_hash(semantic_payload)
```

Construct `GoalState` with that hash and valid two-goal score/policy values, then construct `RuntimeIntentionalDecisionResult`; a GoalState bound to `belief.content_hash` instead of `semantic_hash` must reject.

The authored compatibility test constructs:

```python
authored_model = IntentionalDecisionModelSpec(
    "authored-intentional-regression",
    "1",
    ("phase-choice",),
    make_runtime_belief_model().seed_model,
    make_goal_model(),
    make_choice_model(),
)
result = run_intentional_decision(
    story,
    domain,
    "d-a1-phase",
    authored_model,
)
self.assertEqual(result.selected_action, "a1-ready")
self.assertEqual(
    measure_implementation(IntentionalDecisionModelSpec)
    .manifest_identity()["artifacts"][0]["locator"],
    "python-module:narrative_dynamics.narrative.intention",
)
```

The repository-level byte identity is additionally enforced in implementation tasks with:

```bash
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- \
  narrative_dynamics/narrative/intention.py
```

- [ ] **Step 3: Add guarded simulation imports and the exact two-agent causal fixture**

Reuse the mature projection test domain as a base but create a scheduler-specific extension inside `tests/test_narrative_simulation.py`.

Imports must include:

```python
from dataclasses import fields, replace
from narrative_dynamics.narrative.domain import (
    ActionTypeSpec,
    DecisionTypeSpec,
    ParameterSpec,
    StateDelta,
    StateDeltaOp,
)
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    Entity,
    EntityRef,
    TypedValue,
)
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
)
from narrative_dynamics.narrative.uncertain import UncertainBeliefModelSpec
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionModelSpec,
)
from tests.test_narrative_observation_projection import (
    make_projection_domain,
    make_projection_story,
)
```

Guard new modules exactly as in Task 1 runtime intention. Every simulation method begins with `self.require_simulation()`.

Build the extended domain exactly:

```python
def make_scheduler_domain():
    base = make_projection_domain()
    return replace(
        base,
        action_types=base.action_types
        + (
            ActionTypeSpec(
                "alert-control-action",
                (
                    ParameterSpec("service", "ServiceRef"),
                    ParameterSpec("raise", "AlertState"),
                ),
            ),
            ActionTypeSpec(
                "response-action",
                (ParameterSpec("respond", "AlertState"),),
            ),
        ),
        decision_types=base.decision_types
        + (
            DecisionTypeSpec(
                "scheduler-alert-choice",
                "Agent",
                "alert-control-action",
            ),
            DecisionTypeSpec(
                "scheduler-response-choice",
                "Agent",
                "response-action",
            ),
        ),
    )
```

Create the story with source cutoff **9**. Start from `make_projection_story(domain)`, change seed event `e6` from `service.alert=true` to `false`, append passive Agent `a3`, remove authored observations/claims/receptions, and replace decisions with exactly:

```python
Decision(
    "d-a1-scheduler",
    8,
    "a1",
    "scheduler-alert-choice",
    (alert_cell(),),
    (
        ActionOption(
            "a1-raise-alert",
            "alert-control-action",
            {
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "raise": TypedValue("AlertState", True),
            },
        ),
        ActionOption(
            "a1-wait",
            "alert-control-action",
            {
                "service": TypedValue("ServiceRef", EntityRef("svc", "Service")),
                "raise": TypedValue("AlertState", False),
            },
        ),
    ),
),
Decision(
    "d-a2-scheduler",
    9,
    "a2",
    "scheduler-response-choice",
    (alert_cell(),),
    (
        ActionOption(
            "a2-respond",
            "response-action",
            {"respond": TypedValue("AlertState", True)},
        ),
        ActionOption(
            "a2-wait",
            "response-action",
            {"respond": TypedValue("AlertState", False)},
        ),
    ),
),
```

Define `alert_cell()` as `StateCellRef(EntityRef("svc", "Service"), "service.alert")`.

World hooks use action arguments so wait is an explicit action with an empty delta under the same canonical action type:

```python
class AlertControlTransition:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, decision, action):
        self.calls += 1
        if action.arguments["raise"].value is False:
            return StateDelta(())
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    "svc",
                    "service.alert",
                    TypedValue("AlertState", True),
                ),
            )
        )


class ResponseTransition:
    def __init__(self):
        self.calls = 0

    def __call__(self, snapshot, decision, action):
        self.calls += 1
        if action.arguments["respond"].value is False:
            return StateDelta(())
        return StateDelta(
            (
                StateDeltaOp(
                    "set",
                    decision.actor_id,
                    "agent.phase",
                    TypedValue("PhaseState", "active"),
                ),
            )
        )
```

World model contains exactly these transitions:

```python
ActionTransitionSpec(
    "alert-control-action",
    (ActionEffectSpec("service.alert", "argument", "service"),),
    alert_hook,
)
ActionTransitionSpec(
    "response-action",
    (ActionEffectSpec("agent.phase", "actor"),),
    response_hook,
)
```

Projection emits the persistent alert only to B and passive observer a3:

```python
class ObserveAlertForBAndPassive:
    def __init__(self):
        self.calls = []

    def __call__(self, prior_visible, next_visible, observer, step_index):
        self.calls.append((observer.id, step_index))
        if observer.id not in {"a2", "a3"}:
            return ()
        value = next_visible.get(alert_cell())
        return () if value is None else (
            ObservationFact(alert_cell(), "equals", value),
        )
```

Use one `ObserverProjectionSpec` for Agent/vision with `service.alert` read+emit capability scoped `any`.

Define a scheduler seed prior that makes B initially favor `false`:

```python
class SchedulerPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        result = {}
        for value in hypotheses:
            if agent_id == "a2" and value.value is False:
                result[_value_hash(value)] = 0.9
            elif agent_id == "a2":
                result[_value_hash(value)] = 0.1
            else:
                result[_value_hash(value)] = 0.5
        return result
```

There is no authored evidence for `service.alert`, so this prior is B's round-0 posterior. Runtime alert likelihood uses `0.99` for the observed boolean value and `0.01` for the other hypothesis.

A's goal model must prefer an `act` goal regardless of alert hypothesis by assigning equal instrumentality to both hypotheses but a larger pressure; A's choice model maps `act -> a1-raise-alert` and `idle -> a1-wait`.

B's goal model uses `respond` instrumentality `{false: 0, true: 1}` and `idle` instrumentality `{false: 1, true: 0}`; B's choice model maps `respond -> a2-respond` and `idle -> a2-wait`. Use `beta_goal=8.0` and `beta_action=8.0`. These exact settings must yield round-0 `a2-wait` from the 0.9 false prior and round-1 `a2-respond` after the admitted true percept.

Scheduled agents are exactly a1 and a2. Agent a3 is never scheduled.

- [ ] **Step 4: Add all 16 simulation RED methods**

Lock signatures:

```python
self.assertEqual(
    tuple(inspect.signature(simulation_state_from_story).parameters),
    ("story", "domain", "model", "at_time"),
)
self.assertEqual(
    tuple(inspect.signature(simulate_step).parameters),
    ("story", "domain", "prior_state", "model"),
)
self.assertEqual(
    tuple(inspect.signature(simulate_trajectory).parameters),
    ("story", "domain", "initial_state", "model", "rounds"),
)
```

The shared-snapshot method asserts every returned runtime belief binds `prior.evidence_ledger.content_hash` and every runtime decision step equals `prior.step_index`. It also checks `ActionIntent` fields exactly equal their decision-result source fields.

The two-round acceptance method uses `simulation_state_from_story(..., at_time=9)` and `simulate_trajectory(..., rounds=2)`. Convert each `agent_steps` tuple to a local mapping by `agent_id` and assert:

```text
round 0 a1 == a1-raise-alert
round 0 a2 == a2-wait
round 0 next world service.alert == true
round 0 admission ledger contains step-1 equals-true alert evidence for a2
round 0 admission ledger also contains evidence for passive a3
round 1 a2 runtime posterior differs from its seed posterior
round 1 a2 == a2-respond
```

Snapshot and compare `story.content_hash`, `story.decisions`, `story.observations`, `story.claims`, and `story.receptions` before/after both rounds.

Failure-order tests use hook call counters. A cognition failure must leave both world hook counters and projection calls at zero. A world write-conflict variant must leave projection calls at zero. A raising projection hook must make `simulate_step()` raise without returning a next `SimulationState` while the prior state remains equal to its pre-call value.

- [ ] **Step 5: Extend the exact narrative API expectation by 16 names**

Add exactly:

```text
RuntimeIntentionalDecisionModelSpec
RuntimeIntentionalDecisionResult
RuntimeIntentionalDecisionResolutionError
run_runtime_intentional_decision
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

Keep all 16 absent from `narrative_dynamics.__all__` and absent as root attributes.

- [ ] **Step 6: Verify the test-only RED**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_simulation \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
python3 -m unittest discover -s tests -v
```

Expected full discovery: exactly **541 tests** with exactly **29 feature failures**: 12 runtime-intention module-gate failures, 16 simulation module-gate failures, and one exact API failure. All prior 513 methods remain `ok`.

- [ ] **Step 7: Commit only the RED contract and open a Draft PR**

```bash
git add \
  tests/test_narrative_runtime_intention.py \
  tests/test_narrative_simulation.py \
  tests/test_narrative_trust_api.py
git commit -m "test: define narrative multi-step scheduler v1"
```

Verify this commit changes exactly those three paths relative to the plan head. Open a Draft PR to `proof/narrative-dynamics-v0` and record exact RED head/run. Do not write production until full CI proves the intended RED boundary.

---

### Task 2: Implement runtime intention records, semantic adapter, and model identity

**Files:**
- Create: `narrative_dynamics/narrative/runtime_intention.py`

**Interfaces:**
- Consumes: `RuntimeBeliefModelSpec`, `RuntimeUncertainBeliefState`, `GoalModelSpec`, `ChoiceModelSpec`, `GoalState`, `IntentionalDecisionModelSpec`, `stable_content_hash`, and `measure_implementation`.
- Produces: `RuntimeIntentionalDecisionModelSpec`, `RuntimeIntentionalDecisionResult`, `RuntimeIntentionalDecisionResolutionError`, private posterior-semantic records/helpers, and a typed stage boundary for `run_runtime_intentional_decision()`.

- [ ] **Step 1: Add canonical validation helpers and the typed error**

```python
class RuntimeIntentionalDecisionResolutionError(ValueError):
    """Runtime belief could not produce a valid intentional action selection."""


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty trimmed string")
    return value


def _step(value: object, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value
```

Add local finite-vector/policy freezing using authored intention tolerance `1e-12` so public runtime result constructors validate independently.

- [ ] **Step 2: Add the private posterior-only semantic adapter**

```python
@dataclass(frozen=True)
class _PosteriorSemanticCell:
    cell: StateCellRef
    posterior: BeliefDistribution


@dataclass(frozen=True)
class _PosteriorSemanticView:
    cells: Mapping[StateCellRef, _PosteriorSemanticCell]

    def __post_init__(self) -> None:
        frozen = dict(self.cells)
        if any(
            not isinstance(cell, StateCellRef)
            or not isinstance(view, _PosteriorSemanticCell)
            or view.cell != cell
            for cell, view in frozen.items()
        ):
            raise TypeError("posterior semantic cells must be canonical")
        object.__setattr__(self, "cells", MappingProxyType(frozen))

    def to_dict(self) -> dict[str, object]:
        ordered = tuple(sorted(self.cells, key=_cell_key))
        return {
            "cells": [
                {
                    "cell": cell.to_dict(),
                    "posterior": self.cells[cell].posterior.to_dict(),
                }
                for cell in ordered
            ]
        }

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.to_dict())
```

Build it only from runtime posterior values:

```python
def _posterior_semantic_view(
    belief_state: RuntimeUncertainBeliefState,
) -> _PosteriorSemanticView:
    return _PosteriorSemanticView(
        {
            cell: _PosteriorSemanticCell(cell, view.posterior)
            for cell, view in belief_state.cells.items()
        }
    )
```

This adapter has no ledger/evidence/world/projection/transition/step provenance attributes.

- [ ] **Step 3: Implement `RuntimeIntentionalDecisionModelSpec` identity**

Use exact fields from the spec. Constructor requires exact nested types, non-empty unique supported types sorted lexically, and choice-goal id equality with goal model ids.

`to_dict()` is exactly:

```python
return {
    "model_id": self.model_id,
    "version": self.version,
    "supported_decision_types": list(self.supported_decision_types),
    "belief_model_hash": self.belief_model.content_hash,
    "goal_model_hash": self.goal_model.content_hash,
    "choice_model_hash": self.choice_model.content_hash,
    "runtime_implementation_identity": measure_implementation(
        RuntimeIntentionalDecisionModelSpec
    ).manifest_identity(),
    "authored_intentional_implementation_identity": measure_implementation(
        IntentionalDecisionModelSpec
    ).manifest_identity(),
}
```

- [ ] **Step 4: Implement `RuntimeIntentionalDecisionResult` as audit + semantic record**

Use exact spec fields. Validate:

```python
semantic = _posterior_semantic_view(self.belief_state)
if self.goal_state.belief_state_hash != semantic.content_hash:
    raise ValueError("runtime goal state must bind posterior semantics exactly")
if self.step_index != self.belief_state.step_index:
    raise ValueError("runtime intentional result step must match belief step")
```

Validate conditional policies, scores, final policy, common action coverage, finite normalization, and lexical MAP. `to_dict()` includes full `belief_state.to_dict()` for audit identity.

- [ ] **Step 5: Add the exact typed execution stage boundary**

```python
def run_runtime_intentional_decision(
    story: GenericNarrative,
    domain: DomainSpec,
    decision_id: str,
    ledger: RuntimeEvidenceLedger,
    model: RuntimeIntentionalDecisionModelSpec,
) -> RuntimeIntentionalDecisionResult:
    raise RuntimeIntentionalDecisionResolutionError(
        "runtime intentional execution is unavailable in this stage"
    )
```

- [ ] **Step 6: Verify staged GREEN/RED and commit**

Expected Task 2 runtime-intention GREEN methods: model identity, result record binding, public signature, authored intentional compatibility. Remaining eight runtime-execution methods fail only at the exact stage sentinel. All 16 simulation methods remain module-missing RED; API remains RED. Full discovery must show 541 tests and exactly **25 feature failures**.

Run:

```bash
python3 -m unittest tests.test_narrative_runtime_intention -v
python3 -m unittest tests.test_narrative_intention -v
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- \
  narrative_dynamics/narrative/intention.py
```

Commit only:

```bash
git add narrative_dynamics/narrative/runtime_intention.py
git commit -m "feat: add runtime intentional records"
```

Obtain exact-head CI before Task 3.

---

### Task 3: Implement runtime posterior -> goal -> action selection

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_intention.py`

**Interfaces:**
- Consumes: Task 2 records, `runtime_uncertain_belief_state`, canonical authored `Decision`, and exactly four existing private authored mathematical helpers.
- Produces: complete `run_runtime_intentional_decision()`; all 12 runtime-intention methods GREEN.

- [ ] **Step 1: Import only the allowed authored intention dependency surface**

```python
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    ChoiceResolutionError,
    GoalModelSpec,
    GoalResolutionError,
    GoalState,
    IntentionalDecisionModelSpec,
    _conditional_action_policies,
    _goal_scores,
    _goal_state,
    _marginal_action_policy,
)
```

Do not call `run_intentional_decision()` from the runtime path.

- [ ] **Step 2: Resolve and validate the canonical template before runtime likelihood execution**

Perform in this order:

```python
validate_narrative(story, domain)
if not isinstance(model, RuntimeIntentionalDecisionModelSpec):
    raise TypeError(
        "runtime intentional execution requires RuntimeIntentionalDecisionModelSpec"
    )
decision_id = _text(decision_id, label="runtime decision id")
decision = next((item for item in story.decisions if item.id == decision_id), None)
if decision is None:
    raise ValueError("runtime decision template is not declared")
if decision.type_name not in model.supported_decision_types:
    raise ValueError("runtime decision template type is not supported")
if not decision.context_cells or len(set(decision.context_cells)) != len(decision.context_cells):
    raise ValueError("runtime decision context cells must be non-empty and unique")
if not decision.actions or len({item.id for item in decision.actions}) != len(decision.actions):
    raise ValueError("runtime decision actions must be non-empty and unique")
if ledger.source_at_time is not None and decision.logical_time > ledger.source_at_time:
    raise ValueError("runtime decision template occurs after source cutoff")
```

`runtime_uncertain_belief_state()` then performs exact ledger/story/domain identity validation. Invalid template/type/cutoff cases must produce zero runtime likelihood calls.

- [ ] **Step 3: Materialize runtime belief with exactly `decision.context_cells`**

```python
belief_state = runtime_uncertain_belief_state(
    story,
    domain,
    decision.actor_id,
    ledger,
    model.belief_model,
    decision.context_cells,
)
```

No scheduler/runtime-intention parameter may override tracked cells.

- [ ] **Step 4: Strip to posterior semantics before goal evaluation**

```python
semantic = _posterior_semantic_view(belief_state)
scores = _goal_scores(
    decision.context_cells,
    semantic,
    model.goal_model,
)
goal_state = _goal_state(
    semantic,
    model.goal_model,
    scores,
)
```

Never pass the full runtime belief to `_goal_scores` or `_goal_state`.

- [ ] **Step 5: Reuse existing action-policy helpers exactly**

```python
declared_actions = {action.id for action in decision.actions}
conditional = _conditional_action_policies(
    model.choice_model,
    goal_state.policy,
    declared_actions,
)
action_scores = {
    action_id: math.fsum(
        goal_state.policy[goal_id]
        * model.choice_model.values[goal_id][action_id]
        for goal_id in sorted(goal_state.policy)
    )
    for action_id in sorted(declared_actions)
}
action_policy = _marginal_action_policy(
    goal_state.policy,
    conditional,
    declared_actions,
)
maximum = max(action_policy.values())
selected_action = min(
    action_id
    for action_id, probability in action_policy.items()
    if probability == maximum
)
```

- [ ] **Step 6: Return the full audit result and preserve typed causes**

```python
return RuntimeIntentionalDecisionResult(
    model_id=model.model_id,
    model_hash=model.content_hash,
    decision_id=decision.id,
    step_index=belief_state.step_index,
    belief_state=belief_state,
    goal_state=goal_state,
    conditional_action_policies=conditional,
    action_scores=action_scores,
    action_policy=action_policy,
    selected_action=selected_action,
)
```

Re-raise existing `RuntimeIntentionalDecisionResolutionError`. Wrap runtime-belief, goal, choice, structural, and numerical failures as `RuntimeIntentionalDecisionResolutionError("runtime intentional decision could not be resolved")` with exception chaining.

- [ ] **Step 7: Verify all runtime-intention semantics GREEN and commit**

```bash
python3 -m unittest tests.test_narrative_runtime_intention -v
python3 -m unittest \
  tests.test_narrative_intention \
  tests.test_narrative_runtime_cognition \
  -v
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- \
  narrative_dynamics/narrative/intention.py
```

Expected: all 12 runtime-intention methods `ok`; authored intention/runtime cognition regressions `ok`; no `intention.py` diff. Full CI should now have exactly **17 feature failures**: 16 simulation module gates plus API.

Commit:

```bash
git add narrative_dynamics/narrative/runtime_intention.py
git commit -m "feat: select runtime intentional actions"
```

Obtain exact-head CI before Task 4.

---

### Task 4: Implement simulation records, model identity, and exact initialization

**Files:**
- Create: `narrative_dynamics/narrative/simulation.py`

**Interfaces:**
- Consumes: complete runtime intention, `WorldState`, `RuntimeEvidenceLedger`, `WorldTransitionModelSpec`, `ObservationProjectionModelSpec`, `world_state_from_story`, and `runtime_evidence_ledger_from_story`.
- Produces: six immutable simulation records, three scheduler errors, complete `simulation_state_from_story()`, and typed stage boundaries for step/trajectory execution.

- [ ] **Step 1: Add scheduler errors and canonical helpers**

```python
class SimulationError(ValueError):
    """A deterministic runtime simulation contract could not be satisfied."""


class SimulationStepError(SimulationError):
    """One atomic runtime simulation round failed."""


class SimulationTrajectoryError(SimulationError):
    """A finite runtime simulation trajectory failed."""
```

Add local `_text`, `_hash`, `_step`, `_cutoff`, and agent-sort helpers.

- [ ] **Step 2: Implement `RuntimeAgentSpec` and `SimulationModelSpec`**

`RuntimeAgentSpec.to_dict()` is:

```python
return {
    "agent_id": self.agent_id,
    "decision_template_id": self.decision_template_id,
    "intentional_model_hash": self.intentional_model.content_hash,
}
```

`SimulationModelSpec` validates non-empty agents, unique agent ids, unique template ids, lexical agent order, exact nested types, and exact world/observation domain identity equality. Its payload includes each agent spec, nested world/observation model hashes, and:

```python
"scheduler_implementation_identity": measure_implementation(
    SimulationModelSpec
).manifest_identity()
```

- [ ] **Step 3: Implement `SimulationState` exact alignment**

Validate exact world/ledger types and:

```python
if self.step_index != self.world_state.step_index:
    raise ValueError("simulation step must match world step")
if self.step_index != self.evidence_ledger.current_step_index:
    raise ValueError("simulation step must match evidence ledger step")
if self.evidence_ledger.current_world_state_hash != self.world_state.content_hash:
    raise ValueError("simulation ledger must bind exact current world state")
```

Require exact equality of world/ledger domain id, domain version, domain spec hash, story hash, and source cutoff.

- [ ] **Step 4: Implement `SimulationAgentStep`, `SimulationStepResult`, and `SimulationTrajectory` record validation**

`SimulationAgentStep` validates exact result -> `ActionIntent` bridge fields.

`SimulationStepResult` validates model/step/prior/next/world/admission bindings and lexical agent order. Compare intent payload multisets exactly:

```python
scheduled = tuple(
    sorted(
        (item.action_intent.to_dict() for item in self.agent_steps),
        key=lambda value: (value["decision_id"], value["selected_action"]),
    )
)
executed = tuple(
    sorted(
        (item.intent.to_dict() for item in self.world_step.transitions),
        key=lambda value: (value["decision_id"], value["selected_action"]),
    )
)
if scheduled != executed:
    raise ValueError("simulation step transitions must equal scheduled intents exactly")
```

`SimulationTrajectory` requires non-empty tuple steps and exact prior->next chain ending at `final_state`.

- [ ] **Step 5: Implement execution-level story/template certification**

Before model hooks, validate story/domain and exact simulation model domain identity. Resolve every `RuntimeAgentSpec` to a canonical entity and decision. Require template actor == agent id, template type supported by runtime model, and template logical time <= numeric source cutoff. Do not require projected observers to equal scheduled agents.

- [ ] **Step 6: Implement `simulation_state_from_story()`**

```python
def simulation_state_from_story(
    story: GenericNarrative,
    domain: DomainSpec,
    model: SimulationModelSpec,
    *,
    at_time: int | None = None,
) -> SimulationState:
    cutoff = _cutoff(at_time, label="simulation source cutoff")
    _validate_execution_bindings(story, domain, model, cutoff)
    world = world_state_from_story(story, domain, at_time=cutoff)
    ledger = runtime_evidence_ledger_from_story(story, domain, at_time=cutoff)
    return SimulationState(
        model_id=model.model_id,
        model_hash=model.content_hash,
        step_index=0,
        world_state=world,
        evidence_ledger=ledger,
    )
```

- [ ] **Step 7: Add exact step/trajectory stage boundaries**

```python
def simulate_step(
    story: GenericNarrative,
    domain: DomainSpec,
    prior_state: SimulationState,
    model: SimulationModelSpec,
) -> SimulationStepResult:
    raise SimulationStepError(
        "simulation step execution is unavailable in this stage"
    )


def simulate_trajectory(
    story: GenericNarrative,
    domain: DomainSpec,
    initial_state: SimulationState,
    model: SimulationModelSpec,
    *,
    rounds: int,
) -> SimulationTrajectory:
    raise SimulationTrajectoryError(
        "simulation trajectory execution is unavailable in this stage"
    )
```

- [ ] **Step 8: Verify staged GREEN/RED and commit**

The first four simulation methods (model identity, model rejection, initial state binding, initialization rejection) must be GREEN; remaining 12 must fail only at the stage boundaries. Runtime intention 12 remain GREEN; API remains RED. Full discovery expected exactly **13 feature failures**.

Run scope guard against all forbidden production paths, then commit only `simulation.py`:

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: add narrative simulation state records"
```

Obtain exact-head CI before Task 5.

---

### Task 5: Implement atomic simulation steps and finite trajectories

**Files:**
- Modify: `narrative_dynamics/narrative/simulation.py`

**Interfaces:**
- Consumes: Task 4 records/init, complete runtime intention, existing `advance_world_step`, and existing `admit_world_percepts`.
- Produces: complete `simulate_step()` and `simulate_trajectory()`; all 28 feature behavior methods GREEN.

- [ ] **Step 1: Reject mismatched/forged prior state before every agent hook**

Require exact `SimulationState`/`SimulationModelSpec` types, exact model id/hash equality, internal state alignment, story/domain/model bindings, and template cutoff before calling `run_runtime_intentional_decision()`.

The forged-prior test must bypass dataclass construction with the existing `fields()` pattern, change one of `step_index`, ledger world hash, story hash, or model hash, and prove every runtime likelihood/world/projection counter remains zero.

- [ ] **Step 2: Evaluate every configured agent against the exact same prior ledger**

```python
agent_steps = []
for agent in model.agents:
    try:
        result = run_runtime_intentional_decision(
            story,
            domain,
            agent.decision_template_id,
            prior_state.evidence_ledger,
            agent.intentional_model,
        )
    except RuntimeIntentionalDecisionResolutionError as error:
        raise SimulationStepError(
            f"simulation cognition failed for agent {agent.agent_id}"
        ) from error
    if result.step_index != prior_state.step_index:
        raise SimulationStepError(
            "runtime decision step does not match simulation prior"
        )
    if result.belief_state.ledger_hash != prior_state.evidence_ledger.content_hash:
        raise SimulationStepError(
            "runtime decision does not bind shared prior ledger"
        )
    intent = ActionIntent(
        decision_id=result.decision_id,
        selected_action=result.selected_action,
        selection_model_id=result.model_id,
        selection_result_hash=result.content_hash,
    )
    agent_steps.append(
        SimulationAgentStep(
            agent_id=agent.agent_id,
            decision_template_id=agent.decision_template_id,
            decision_result=result,
            action_intent=intent,
        )
    )
```

No world hook executes inside this loop.

- [ ] **Step 3: Execute one atomic world transition only after all selections succeed**

```python
try:
    world_step = advance_world_step(
        story,
        domain,
        prior_state.world_state,
        model.world_model,
        tuple(item.action_intent for item in agent_steps),
    )
except WorldTransitionError as error:
    raise SimulationStepError("simulation world transition failed") from error
```

A world failure must leave projection/admission counters at zero.

- [ ] **Step 4: Admit next percepts exactly once through admission**

```python
try:
    admission = admit_world_percepts(
        story,
        domain,
        world_step,
        model.observation_model,
        prior_state.evidence_ledger,
    )
except RuntimePerceptAdmissionError as error:
    raise SimulationStepError("simulation percept admission failed") from error
```

`simulation.py` must not import `project_world_observations`.

- [ ] **Step 5: Construct next state and exact result only after successful admission**

```python
next_state = SimulationState(
    model_id=model.model_id,
    model_hash=model.content_hash,
    step_index=prior_state.step_index + 1,
    world_state=world_step.next_state,
    evidence_ledger=admission.next_ledger,
)
return SimulationStepResult(
    model_id=model.model_id,
    model_hash=model.content_hash,
    step_index=next_state.step_index,
    prior_state=prior_state,
    agent_steps=tuple(agent_steps),
    world_step=world_step,
    admission_result=admission,
    next_state=next_state,
)
```

Constructor failures wrap as `SimulationStepError("simulation next-state construction failed")` with cause.

- [ ] **Step 6: Implement positive-integer trajectory execution**

Reject non-int, bool, zero, and negative rounds before step execution. Then repeatedly call `simulate_step()` exactly `rounds` times. If a round fails, raise:

```python
SimulationTrajectoryError(
    f"simulation trajectory failed at step {current.step_index + 1}"
)
```

with the `SimulationStepError` as cause. Successful return is `SimulationTrajectory(model_id, model_hash, initial_state, tuple(steps), current)`.

- [ ] **Step 7: Prove the two-round causal closure in isolation**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_simulation.NarrativeSimulationTests.test_two_round_action_world_percept_belief_action_causal_closure_without_story_mutation \
  -v
```

Require `ok` and inspect the assertions proving the exact causal chain:

```text
a1-raise-alert
-> WorldState service.alert=true
-> admitted a2 equals-true alert percept at step 1
-> a2 posterior changes from 0.9 false prior to true-favoring posterior
-> a2-respond in round 1
```

Also require passive a3 evidence in the ledger and unchanged authored story artifacts.

- [ ] **Step 8: Verify all scheduler semantics, regressions, and forbidden-path scope**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_simulation \
  tests.test_narrative_runtime_cognition \
  tests.test_narrative_runtime_perception \
  tests.test_narrative_intention \
  tests.test_narrative_world_transition \
  tests.test_narrative_observation_projection \
  -v
```

All must pass. Then require no diff from spec head on:

```text
narrative_dynamics/narrative/intention.py
narrative_dynamics/narrative/runtime_cognition.py
narrative_dynamics/narrative/runtime_perception.py
narrative_dynamics/narrative/world.py
narrative_dynamics/narrative/observation_projection.py
narrative_dynamics/narrative/ir.py
narrative_dynamics/narrative/domain.py
narrative_dynamics/__init__.py
```

- [ ] **Step 9: Commit semantic GREEN and obtain export-only RED**

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: run narrative multi-step simulation"
```

Exact-head CI must have all 12 runtime-intention + 16 simulation methods GREEN and exactly one feature failure: the API/root-isolation gate. Full discovery count remains 541.

---

### Task 6: Export the exact 16-name API and close full regression

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

**Interfaces:**
- Consumes: complete runtime intention and simulation sidecars.
- Produces: exactly 16 new narrative-scoped names; root isolation preserved; final 541-test GREEN.

- [ ] **Step 1: Add only runtime-intention imports**

```python
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
```

- [ ] **Step 2: Add only simulation imports**

```python
from narrative_dynamics.narrative.simulation import (
    RuntimeAgentSpec,
    SimulationAgentStep,
    SimulationError,
    SimulationModelSpec,
    SimulationState,
    SimulationStepError,
    SimulationStepResult,
    SimulationTrajectory,
    SimulationTrajectoryError,
    simulate_step,
    simulate_trajectory,
    simulation_state_from_story,
)
```

Add exactly these 16 strings to narrative `__all__`; do not edit root `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run API and dedicated feature gates**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
python3 -m unittest \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_simulation \
  -v
```

Expected: API/root isolation `ok`; all 28 new behavior methods `ok`.

- [ ] **Step 4: Run full Python discovery and require exact count**

```bash
python3 -m unittest discover -s tests -v
```

The summary must report exactly 541 tests and final status `OK`. Any other count or nonzero failure/error blocks PR readiness.

- [ ] **Step 5: Commit only the export change**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative simulation api"
```

This commit must touch only narrative `__init__.py`.

- [ ] **Step 6: Inspect final exact-head `proof` CI**

Require success for every workflow step:

```text
Resolve Lean dependencies
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

Fetch the complete job log. Confirm the unittest summary count is exactly 541 and status is `OK`. Confirm every new runtime-intention/simulation method and exact API test is individually `ok`.

- [ ] **Step 7: Verify exact final eight-path diff**

Compare final head to `ea45389354bf9b8dff4c18917e2bca3e3e83a939`; changed paths must be exactly:

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

`intention.py`, root package, Runtime Cognition/Perception, World Transition, Observation Projection, IR, Domain, and Lean sources must have no diff.

- [ ] **Step 8: Update Draft PR to review-ready only after fresh evidence**

PR body records exact RED head/run and 29-failure split; Task 2 25-failure split; Task 3 17-failure split; Task 4 13-failure split; Task 5 export-only one-failure split; final exact head/run with 541 tests `OK`; and exact eight-path diff. Mark Ready only after final CI and read-only diff review. Do not merge without explicit user instruction.

## Invariant Coverage Map

- Invariants 1-6: Task 2 model/result/identity tests plus byte-for-byte `intention.py` scope gate.
- Invariants 7-10: Task 3 canonical template/cutoff/context tests.
- Invariants 11-15: Task 3 posterior semantic adapter, hidden-provenance equality, and authored-math equivalence tests.
- Invariants 16-22: Task 2 result validation plus Task 3 action/goal coverage, lexical tie, audit/semantic identity tests.
- Invariants 23-33: Task 4 runtime-agent/simulation-model data and execution-binding tests.
- Invariants 34-38: Task 4 `SimulationState` alignment/initialization and forged-data tests.
- Invariants 39-48: Task 5 shared-snapshot/static-schedule/action-intent/world/admission tests.
- Invariants 49-52: Task 5 explicit call-counter failure-order tests.
- Invariants 53-60: Task 5 next-state binding, repeated-template, order invariance, passive-observer, and blackout tests.
- Invariants 61-68: Task 5 trajectory validation, deterministic replay, two-round story immutability, and exact chain tests.
- Invariants 69-76: Task 6 exact API/root isolation, forbidden-path scope checks, full 541-test regression, and all Lean gates.

## Final Scientific Acceptance Gate

Before PR readiness, inspect the two-round test and verify this concrete audit chain exists in the returned artifacts:

```text
a1 round-0 RuntimeIntentionalDecisionResult selecting a1-raise-alert
-> matching ActionIntent
-> WorldStepResult with service.alert set true
-> RuntimePerceptAdmissionResult containing a2 alert=true percept
-> a2 round-1 RuntimeUncertainBeliefState with changed posterior
-> a2 round-1 RuntimeIntentionalDecisionResult selecting a2-respond
```

The same test must prove the canonical `GenericNarrative` content hash and authored decisions/observations/claims/receptions are unchanged. Passing unrelated unit tests is not a substitute for this causal closure proof.