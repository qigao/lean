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
- Static scheduling only: every configured agent decides exactly once per round; no silent skip. Inactivity is an explicit authored `wait`/`noop` action.
- Reuse authored `Decision` only as immutable actor/type/context/action-space template. Never synthesize runtime `Decision` or `ActionOption` values.
- Authored `logical_time` and runtime `step_index` remain distinct. If `source_at_time` is numeric, every configured decision template must satisfy `decision.logical_time <= source_at_time`.
- `decision.context_cells` are exactly the tracked runtime belief cells used for that runtime decision.
- Runtime intentional action selection may depend only on canonical decision template semantics, posterior distributions, `GoalModelSpec`, and `ChoiceModelSpec`; it must not read ledger/world/projection/transition identities or objective world objects.
- `intention.py` must remain byte-for-byte unchanged. `measure_implementation()` measures complete module bytes, so moving helpers out of it would silently change existing authored intentional model identity.
- `runtime_intention.py` may import the existing private helpers `_goal_scores`, `_goal_state`, `_conditional_action_policies`, and `_marginal_action_policy` from `intention.py`; it must pass them a private posterior-only semantic adapter, never a full `RuntimeUncertainBeliefState`.
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

- [ ] **Step 1: Add guarded runtime-intention imports and stable fixture helpers**

At the top of `tests/test_narrative_runtime_intention.py`, import existing fixtures rather than rebuilding the runtime evidence ledger:

```python
from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.narrative.intention import (
    ChoiceModelSpec,
    GoalModelSpec,
    GoalSpec,
    IntentionalDecisionModelSpec,
    run_intentional_decision,
)
from narrative_dynamics.narrative.ir import StateCellRef, TypedValue
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

Every test begins with:

```python
def require_runtime_intention(self) -> None:
    if _RUNTIME_INTENTION_IMPORT_ERROR is not None:
        self.fail(
            "narrative runtime intention boundary is missing: "
            f"{_RUNTIME_INTENTION_IMPORT_ERROR}"
        )
```

Define deterministic helper constructors in this test module:

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
        values = {key: dict(reversed(tuple(value.items()))) for key, value in reversed(tuple(values.items()))}
    return ChoiceModelSpec("runtime-choice", "1", 8.0, values)


def make_runtime_intentional_model(*, belief=None, goal=None, choice=None):
    return RuntimeIntentionalDecisionModelSpec(
        "runtime-intentional",
        "1",
        ("phase_choice",),
        make_runtime_belief_model() if belief is None else belief,
        make_goal_model() if goal is None else goal,
        make_choice_model() if choice is None else choice,
    )
```

Use the actual decision type from `make_runtime_story()` when implementing the test; if its canonical type is not `phase_choice`, set the constructor's supported type to the exact `story.decisions` entry type instead of altering the story.

- [ ] **Step 2: Add all 12 runtime-intention RED methods**

The tests must make the following exact assertions:

```python
self.assertEqual(
    tuple(inspect.signature(run_runtime_intentional_decision).parameters),
    ("story", "domain", "decision_id", "ledger", "model"),
)
for forbidden in ("world_state", "world_step", "projection_result", "step_index"):
    self.assertNotIn(forbidden, inspect.signature(run_runtime_intentional_decision).parameters)
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

Hidden-provenance test must build two ledgers with `extend_ledger(..., branch="left")` and `branch="right"`, identical semantic percept rows, and assert:

```python
self.assertNotEqual(left.belief_state.ledger_hash, right.belief_state.ledger_hash)
self.assertEqual(left.goal_state.policy, right.goal_state.policy)
self.assertEqual(left.action_policy, right.action_policy)
self.assertEqual(left.selected_action, right.selected_action)
self.assertNotEqual(left.content_hash, right.content_hash)
```

The cutoff/context validation method must attach a counting runtime likelihood hook and prove invalid template/domain/actor/cutoff conditions reject before its first call.

The authored compatibility method must run the existing `run_intentional_decision()` on the unchanged story fixture and assert its known selected action/policy plus:

```python
self.assertEqual(
    measure_implementation(IntentionalDecisionModelSpec).manifest_identity()["artifacts"][0]["locator"],
    "python-module:narrative_dynamics.narrative.intention",
)
```

The repository-level byte-for-byte guarantee is additionally enforced in every implementation task by `git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- narrative_dynamics/narrative/intention.py`.

- [ ] **Step 3: Add guarded simulation imports and two-agent scheduling fixtures**

At the top of `tests/test_narrative_simulation.py` use:

```python
from dataclasses import replace
import inspect
import unittest

from narrative_dynamics.narrative.domain import StateDelta, StateDeltaOp
from narrative_dynamics.narrative.ir import (
    ActionOption,
    Decision,
    EntityRef,
    StateCellRef,
    TypedValue,
)
from narrative_dynamics.narrative.observation_projection import (
    ObservationCapabilitySpec,
    ObservationFact,
    ObservationProjectionModelSpec,
    ObserverProjectionSpec,
)
from narrative_dynamics.narrative.runtime_cognition import RuntimeBeliefModelSpec
from narrative_dynamics.narrative.world import (
    ActionEffectSpec,
    ActionTransitionSpec,
    WorldTransitionModelSpec,
)
from tests.test_narrative_runtime_cognition import (
    SeedLikelihoodHook,
    SeedPriorHook,
)

_SIMULATION_IMPORT_ERROR: ImportError | None = None
try:
    from narrative_dynamics.narrative.runtime_intention import (
        RuntimeIntentionalDecisionModelSpec,
        RuntimeIntentionalDecisionResolutionError,
    )
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
        simulation_state_from_story,
        simulate_step,
        simulate_trajectory,
    )
except ImportError as error:
    _SIMULATION_IMPORT_ERROR = error
```

Create a dedicated fixture rather than mutating production fixtures. It must contain:

```text
agents: a1, a2
service: svc
state: service.alert : bool
A decision template: raise_alert | wait
B decision template: respond | wait
source_at_time: after both templates exist
```

Both authored templates are present before the simulation cutoff and are reused unchanged each round.

The world transition hooks are deterministic:

```python
class RaiseAlert:
    def __call__(self, snapshot, decision, action):
        return StateDelta((StateDeltaOp("set", "svc", "service.alert", TypedValue("AlertState", True)),))


class NoWorldChange:
    def __call__(self, snapshot, decision, action):
        return StateDelta(())
```

B's observation hook reads and emits only `service.alert`:

```python
class ObserveAlert:
    def __call__(self, prior_visible, next_visible, observer, step_index):
        cell = StateCellRef(EntityRef("svc", "Service"), "service.alert")
        value = next_visible.get(cell)
        return () if value is None else (ObservationFact(cell, "equals", value),)
```

The fixture's runtime likelihood for B favors the observed alert strongly. Goal/choice configuration must make B choose `wait` from the authored seed and `respond` after admitted `alert=true`. A's configuration must choose `raise_alert` in round 0 and may choose explicit `wait` in later rounds; the acceptance test only relies on A's first-round action.

Use counting wrapper hooks for runtime likelihood, world transition, and projection so failure-order tests assert exact call counts.

- [ ] **Step 4: Add all 16 simulation RED methods**

Lock the exact public signatures:

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

The shared-snapshot test must assert for every returned agent step:

```python
self.assertEqual(
    agent_step.decision_result.belief_state.ledger_hash,
    prior.evidence_ledger.content_hash,
)
self.assertEqual(agent_step.decision_result.step_index, prior.step_index)
```

and exact intent bridging:

```python
self.assertEqual(agent_step.action_intent.decision_id, agent_step.decision_result.decision_id)
self.assertEqual(agent_step.action_intent.selected_action, agent_step.decision_result.selected_action)
self.assertEqual(agent_step.action_intent.selection_result_hash, agent_step.decision_result.content_hash)
```

The two-round acceptance test must assert all of:

```python
original_hash = story.content_hash
original_decisions = story.decisions
original_observations = story.observations
original_claims = story.claims
original_receptions = story.receptions
trajectory = simulate_trajectory(story, domain, initial, model, rounds=2)
self.assertEqual(len(trajectory.steps), 2)
self.assertEqual(trajectory.steps[0].agent_steps_by_id["a1"].decision_result.selected_action, "raise_alert")
self.assertEqual(trajectory.steps[0].agent_steps_by_id["a2"].decision_result.selected_action, "wait")
self.assertEqual(trajectory.steps[1].agent_steps_by_id["a2"].decision_result.selected_action, "respond")
self.assertEqual(story.content_hash, original_hash)
self.assertEqual(story.decisions, original_decisions)
self.assertEqual(story.observations, original_observations)
self.assertEqual(story.claims, original_claims)
self.assertEqual(story.receptions, original_receptions)
```

Do not add `agent_steps_by_id` to the public spec merely for this assertion. In the actual test use:

```python
step0 = {item.agent_id: item for item in trajectory.steps[0].agent_steps}
step1 = {item.agent_id: item for item in trajectory.steps[1].agent_steps}
```

and assert through those mappings.

The test must additionally prove B's round-1 belief contains admitted runtime evidence at step 1 and that its posterior differs from its round-0 seed posterior.

- [ ] **Step 5: Extend the exact narrative API expectation by 16 names**

In `tests/test_narrative_trust_api.py`, add exactly:

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

- [ ] **Step 6: Verify the test-only RED locally or in exact-head CI**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_simulation \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
```

Expected before production exists:

```text
12 runtime-intention methods fail only at require_runtime_intention()
16 simulation methods fail only at require_simulation()
1 exact API method fails only for the 16 missing approved names
```

Then run full discovery:

```bash
python3 -m unittest discover -s tests -v
```

Expected count: exactly **541 tests**. Expected feature failures: exactly **29**. All prior 513 tests must remain `ok`.

- [ ] **Step 7: Commit only the RED contract and open a Draft PR**

```bash
git add \
  tests/test_narrative_runtime_intention.py \
  tests/test_narrative_simulation.py \
  tests/test_narrative_trust_api.py
git commit -m "test: define narrative multi-step scheduler v1"
```

Verify the commit changes exactly those three paths relative to the plan head. Open a Draft PR to `proof/narrative-dynamics-v0` and record the exact RED head/run. Do not write production until full CI proves the intended RED boundary.

---

### Task 2: Implement runtime intention records, semantic adapter, and model identity

**Files:**
- Create: `narrative_dynamics/narrative/runtime_intention.py`

**Interfaces:**
- Consumes: `RuntimeBeliefModelSpec`, `RuntimeUncertainBeliefState`, `GoalModelSpec`, `ChoiceModelSpec`, `GoalState`, `IntentionalDecisionModelSpec`, `stable_content_hash`, and `measure_implementation`.
- Produces: `RuntimeIntentionalDecisionModelSpec`, `RuntimeIntentionalDecisionResult`, `RuntimeIntentionalDecisionResolutionError`, private posterior-semantic records/helpers, and a typed stage boundary for `run_runtime_intentional_decision()`.

- [ ] **Step 1: Add canonical validation helpers and the typed error**

Use local helpers rather than importing private validation helpers from unrelated modules:

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

Also add local finite-policy freezing matching authored intention tolerance `1e-12` so public runtime-result constructors validate independently rather than trusting private authored constructors.

- [ ] **Step 2: Add private posterior-only semantic adapter**

Define private immutable records:

```python
@dataclass(frozen=True)
class _PosteriorSemanticCell:
    cell: StateCellRef
    posterior: BeliefDistribution


@dataclass(frozen=True)
class _PosteriorSemanticView:
    cells: Mapping[StateCellRef, _PosteriorSemanticCell]

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

Build it only from `RuntimeUncertainBeliefState.cells`:

```python
def _posterior_semantic_view(
    belief_state: RuntimeUncertainBeliefState,
) -> _PosteriorSemanticView:
    return _PosteriorSemanticView(
        MappingProxyType(
            {
                cell: _PosteriorSemanticCell(cell, view.posterior)
                for cell, view in belief_state.cells.items()
            }
        )
    )
```

This type intentionally has no ledger/evidence/world/projection/step provenance fields.

- [ ] **Step 3: Implement `RuntimeIntentionalDecisionModelSpec` identity**

Fields and exact `to_dict()` payload:

```python
@dataclass(frozen=True)
class RuntimeIntentionalDecisionModelSpec:
    model_id: str
    version: str
    supported_decision_types: tuple[str, ...]
    belief_model: RuntimeBeliefModelSpec
    goal_model: GoalModelSpec
    choice_model: ChoiceModelSpec

    def to_dict(self) -> dict[str, object]:
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

Constructor validation must require exact nested model types, non-empty unique supported types canonicalized lexically, and exact choice-goal id equality with `goal_model.goals`.

- [ ] **Step 4: Implement `RuntimeIntentionalDecisionResult` as audit + semantic record**

Use the exact public fields from the spec. Validate:

```python
semantic = _posterior_semantic_view(self.belief_state)
if self.goal_state.belief_state_hash != semantic.content_hash:
    raise ValueError("runtime goal state must bind posterior semantics exactly")
if self.step_index != self.belief_state.step_index:
    raise ValueError("runtime intentional result step must match belief step")
```

Validate conditional policies, scores, final policy, and lexical MAP action with the same normalization tolerance as authored intention. `to_dict()` includes the full `belief_state.to_dict()` so audit hashes differ when hidden provenance differs.

- [ ] **Step 5: Add the exact typed execution stage boundary**

Expose the final signature now but deliberately leave selection unavailable until Task 3:

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

- [ ] **Step 6: Verify staged GREEN/RED**

Run the 12 runtime-intention methods. Expected Task 2 split:

```text
GREEN:
- model identity
- result record semantic/audit binding
- public signature/no world inputs
- authored intentional runner compatibility

RED at exact stage sentinel:
- remaining 8 runtime execution methods
```

All 16 simulation tests remain module-missing RED; API remains RED. Full discovery remains 541 tests with expected **25 feature failures** (`8 + 16 + 1`).

Also run:

```bash
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- narrative_dynamics/narrative/intention.py
python3 -m unittest tests.test_narrative_intention -v
```

Both must succeed.

- [ ] **Step 7: Commit only `runtime_intention.py`**

```bash
git add narrative_dynamics/narrative/runtime_intention.py
git commit -m "feat: add runtime intentional records"
```

Obtain exact-head CI staged evidence before Task 3.

---

### Task 3: Implement runtime posterior -> goal -> action selection

**Files:**
- Modify: `narrative_dynamics/narrative/runtime_intention.py`

**Interfaces:**
- Consumes: Task 2 records, `runtime_uncertain_belief_state`, canonical authored `Decision`, and exactly four existing private authored mathematical helpers.
- Produces: complete `run_runtime_intentional_decision()`; all 12 runtime-intention methods GREEN.

- [ ] **Step 1: Import only the allowed authored intention dependency surface**

Use:

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

Do not import or call `run_intentional_decision()` from the runtime path.

- [ ] **Step 2: Resolve and validate the canonical authored template before runtime likelihood execution**

Implement a private validator that performs, in order:

```python
validate_narrative(story, domain)
if not isinstance(model, RuntimeIntentionalDecisionModelSpec):
    raise TypeError("runtime intentional execution requires RuntimeIntentionalDecisionModelSpec")
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

`runtime_uncertain_belief_state()` performs exact story/domain/ledger identity validation next. No runtime likelihood hook may be called before the checks above finish.

- [ ] **Step 3: Materialize runtime belief using exactly the template context cells**

Call exactly:

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

No alternate tracked-cell input exists.

- [ ] **Step 4: Convert full audit belief to posterior-only semantic view before intentional mathematics**

```python
semantic = _posterior_semantic_view(belief_state)
```

Then pass `semantic`, not `belief_state`, to the existing goal helpers:

```python
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

The semantic adapter's `to_dict()` produces exactly the payload hashed into `GoalState.belief_state_hash`.

- [ ] **Step 5: Reuse authored conditional/marginal action policy helpers exactly**

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

Do not pass `step_index`, ledger hash, evidence hashes, or world objects to these helpers.

- [ ] **Step 6: Return full audit result and wrap failures with typed cause**

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

Re-raise `RuntimeIntentionalDecisionResolutionError`. Wrap runtime belief, `GoalResolutionError`, `ChoiceResolutionError`, structural `TypeError`/`ValueError`, and numerical `OverflowError` as `RuntimeIntentionalDecisionResolutionError("runtime intentional decision could not be resolved")` with chaining.

- [ ] **Step 7: Verify all runtime-intention semantics GREEN**

Run:

```bash
python3 -m unittest tests.test_narrative_runtime_intention -v
python3 -m unittest \
  tests.test_narrative_intention \
  tests.test_narrative_runtime_cognition \
  -v
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- narrative_dynamics/narrative/intention.py
```

Expected: all 12 new runtime-intention methods pass; authored intention and runtime cognition regression pass; `intention.py` has no diff.

Full exact-head CI should now have only 16 simulation module-gate failures plus one API failure: **17 feature failures**.

- [ ] **Step 8: Commit semantic GREEN**

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

Add `_text`, `_hash`, `_step`, `_cutoff`, and agent-key helpers locally. Do not import private helpers from world/runtime modules.

- [ ] **Step 2: Implement `RuntimeAgentSpec` and `SimulationModelSpec` data validation**

`RuntimeAgentSpec` validates canonical ids and exact runtime intentional model type. `SimulationModelSpec` validates:

```text
non-empty agents
unique agent_id
unique decision_template_id
agents sorted by agent_id
exact world/observation model types
world domain identity == simulation domain identity
observation domain identity == simulation domain identity
```

Its `to_dict()` includes nested model hashes plus:

```python
"scheduler_implementation_identity": measure_implementation(
    SimulationModelSpec
).manifest_identity()
```

- [ ] **Step 3: Implement `SimulationState` internal alignment invariants**

Require exact types and:

```python
if self.step_index != self.world_state.step_index:
    raise ValueError("simulation step must match world step")
if self.step_index != self.evidence_ledger.current_step_index:
    raise ValueError("simulation step must match evidence ledger step")
if self.evidence_ledger.current_world_state_hash != self.world_state.content_hash:
    raise ValueError("simulation ledger must bind exact current world state")
```

Compare world/ledger tuples exactly:

```python
(
    world.domain_id,
    world.domain_version,
    world.domain_spec_hash,
    world.source_story_hash,
    world.source_at_time,
) == (
    ledger.domain_id,
    ledger.domain_version,
    ledger.domain_spec_hash,
    ledger.source_story_hash,
    ledger.source_at_time,
)
```

`SimulationState` stores model id/hash but does not certify that those fields correspond to a supplied model until execution.

- [ ] **Step 4: Implement `SimulationAgentStep`, `SimulationStepResult`, and `SimulationTrajectory` record validation**

`SimulationAgentStep` validates the mechanical bridge between `RuntimeIntentionalDecisionResult` and `ActionIntent`.

`SimulationStepResult` validates exact prior/next/world/admission binding and canonical agent-step order. It must verify the set of `ActionIntent.to_dict()` payloads in `agent_steps` equals the set of transition-record intent payloads in `world_step.transitions`; no missing or extra transition is allowed.

`SimulationTrajectory` requires a non-empty exact tuple of step results and validates exact state chain:

```python
current = self.initial_state
for result in self.steps:
    if result.prior_state != current:
        raise ValueError("simulation trajectory prior state chain is discontinuous")
    current = result.next_state
if current != self.final_state:
    raise ValueError("simulation trajectory final state must equal chain tail")
```

- [ ] **Step 5: Add execution-level simulation model/story/template certification**

Implement `_validate_execution_bindings(story, domain, model, source_at_time)` that validates before any runtime likelihood hook:

```text
story/domain canonical validation
simulation model domain identity
world model domain identity
observation model domain identity
agent ids resolve to canonical story entities
template ids resolve to canonical decisions
template actor == configured agent id
template decision type supported by configured runtime model
numeric source cutoff contains every template logical_time
```

Passive observers are not validated against the scheduled set; they remain legal.

- [ ] **Step 6: Implement `simulation_state_from_story()` completely**

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
    raise SimulationStepError("simulation step execution is unavailable in this stage")


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

Expected simulation split:

```text
GREEN:
- model canonical identity
- model duplicate/domain rejection
- exact initial state binding
- initialization story/agent/template/type/cutoff rejection

RED at stage boundaries:
- remaining 12 step/trajectory methods
```

Runtime-intention 12 stay green; API stays RED. Full discovery expected **13 feature failures** (`12 + 1`).

Run regression and scope guard:

```bash
python3 -m unittest tests.test_narrative_simulation -v
python3 -m unittest tests.test_narrative_runtime_intention -v
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- \
  narrative_dynamics/narrative/intention.py \
  narrative_dynamics/narrative/world.py \
  narrative_dynamics/narrative/observation_projection.py \
  narrative_dynamics/narrative/runtime_cognition.py \
  narrative_dynamics/narrative/runtime_perception.py
```

Commit only:

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: add narrative simulation state records"
```

Obtain exact-head CI before Task 5.

---

### Task 5: Implement atomic simulation steps and finite deterministic trajectories

**Files:**
- Modify: `narrative_dynamics/narrative/simulation.py`

**Interfaces:**
- Consumes: Task 4 records/init, complete runtime intention, existing `advance_world_step`, and existing `admit_world_percepts`.
- Produces: complete `simulate_step()` and `simulate_trajectory()`; all 28 feature behavior methods GREEN.

- [ ] **Step 1: Validate the exact prior simulation snapshot before any model hook**

At the start of `simulate_step()`:

```python
if not isinstance(prior_state, SimulationState):
    raise SimulationStepError("simulation step requires SimulationState")
if not isinstance(model, SimulationModelSpec):
    raise SimulationStepError("simulation step requires SimulationModelSpec")
if prior_state.model_id != model.model_id or prior_state.model_hash != model.content_hash:
    raise SimulationStepError("simulation prior state model identity mismatch")
_validate_execution_bindings(
    story,
    domain,
    model,
    prior_state.world_state.source_at_time,
)
```

Then reconstruct a `SimulationState` from its own fields or call a private alignment validator so forged records produced by bypassing dataclass construction still reject before agent hooks.

- [ ] **Step 2: Evaluate every configured agent against the exact same prior ledger**

Iterate `model.agents`, already canonicalized by `agent_id`:

```python
agent_steps = []
for agent in model.agents:
    result = run_runtime_intentional_decision(
        story,
        domain,
        agent.decision_template_id,
        prior_state.evidence_ledger,
        agent.intentional_model,
    )
    if result.step_index != prior_state.step_index:
        raise SimulationStepError("runtime decision step does not match simulation prior")
    if result.belief_state.ledger_hash != prior_state.evidence_ledger.content_hash:
        raise SimulationStepError("runtime decision does not bind shared prior ledger")
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

Do not call world transition inside this loop.

If any agent fails, wrap with `SimulationStepError(f"simulation cognition failed for agent {agent.agent_id}")` and chain the original typed runtime-intention error. Previously computed immutable agent results are discarded and no world hook runs.

- [ ] **Step 3: Execute exactly one atomic world transition after every selection succeeds**

```python
world_step = advance_world_step(
    story,
    domain,
    prior_state.world_state,
    model.world_model,
    tuple(item.action_intent for item in agent_steps),
)
```

Catch `WorldTransitionError` and wrap as `SimulationStepError("simulation world transition failed")` with chaining. Do not invoke admission in the exception path.

- [ ] **Step 4: Admit next percepts exactly once through the existing admission boundary**

```python
admission = admit_world_percepts(
    story,
    domain,
    world_step,
    model.observation_model,
    prior_state.evidence_ledger,
)
```

Do not import or invoke `project_world_observations()` in `simulation.py`.

Catch `RuntimePerceptAdmissionError` and wrap as `SimulationStepError("simulation percept admission failed")` with chaining.

- [ ] **Step 5: Construct next state only after successful admission**

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

Any constructor failure is wrapped as `SimulationStepError("simulation next-state construction failed")` with chaining.

- [ ] **Step 6: Implement positive-integer finite trajectory execution**

Validate `rounds` before calling `simulate_step()`:

```python
if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
    raise SimulationTrajectoryError("simulation rounds must be a positive integer")
```

Then:

```python
current = initial_state
steps = []
for offset in range(rounds):
    try:
        result = simulate_step(story, domain, current, model)
    except SimulationStepError as error:
        raise SimulationTrajectoryError(
            f"simulation trajectory failed at step {current.step_index + 1}"
        ) from error
    steps.append(result)
    current = result.next_state
return SimulationTrajectory(
    model_id=model.model_id,
    model_hash=model.content_hash,
    initial_state=initial_state,
    steps=tuple(steps),
    final_state=current,
)
```

No stop predicate or terminal condition is consulted.

- [ ] **Step 7: Prove the central two-round causal closure**

Run only:

```bash
python3 -m unittest \
  tests.test_narrative_simulation.NarrativeSimulationTests.test_two_round_action_world_percept_belief_action_causal_closure_without_story_mutation \
  -v
```

Expected `ok`. Inspect the test artifacts and require the causal sequence:

```text
round 0 a1 selected_action == raise_alert
round 0 a2 selected_action == wait
round 0 next world service.alert == true
round 0 admission ledger contains step-1 alert percept for a2
round 1 a2 belief posterior differs from its seed posterior
round 1 a2 selected_action == respond
canonical story hash/decisions/observations/claims/receptions unchanged
```

- [ ] **Step 8: Verify failure-order, passive-observer, determinism, and trajectory semantics**

Run the complete simulation module:

```bash
python3 -m unittest tests.test_narrative_simulation -v
```

Expected all 16 methods `ok`.

Then run all dedicated behavior modules and regressions:

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

Expected all pass.

- [ ] **Step 9: Verify forbidden paths remain unchanged**

```bash
git diff --exit-code e10ed369d1908b12a2592c07aa8ffb241e526cb2 -- \
  narrative_dynamics/narrative/intention.py \
  narrative_dynamics/narrative/runtime_cognition.py \
  narrative_dynamics/narrative/runtime_perception.py \
  narrative_dynamics/narrative/world.py \
  narrative_dynamics/narrative/observation_projection.py \
  narrative_dynamics/narrative/ir.py \
  narrative_dynamics/narrative/domain.py \
  narrative_dynamics/__init__.py
```

Any diff is architecture expansion and blocks continuation.

- [ ] **Step 10: Commit full scheduler semantics and obtain export-only RED**

```bash
git add narrative_dynamics/narrative/simulation.py
git commit -m "feat: run narrative multi-step simulation"
```

On exact-head CI require all 12 runtime-intention + 16 simulation methods GREEN. The only feature-related failure must be `test_exact_public_surface_and_root_isolation` because the 16 exports are still absent. All prior 513 tests and all Lean gates remain green. Expected full discovery: **541 tests, exactly 1 failure**.

---

### Task 6: Export the exact 16-name API and close full regression

**Files:**
- Modify: `narrative_dynamics/narrative/__init__.py`

**Interfaces:**
- Consumes: complete runtime intention and simulation sidecars.
- Produces: exactly 16 new narrative-scoped names; root isolation preserved; final 541-test GREEN.

- [ ] **Step 1: Add only runtime-intention scoped imports**

Add:

```python
from narrative_dynamics.narrative.runtime_intention import (
    RuntimeIntentionalDecisionModelSpec,
    RuntimeIntentionalDecisionResolutionError,
    RuntimeIntentionalDecisionResult,
    run_runtime_intentional_decision,
)
```

- [ ] **Step 2: Add only simulation scoped imports**

Add:

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

Add exactly these 16 strings to narrative `__all__`. Do not edit `narrative_dynamics/__init__.py`.

- [ ] **Step 3: Run the exact API and dedicated semantic gates**

```bash
python3 -m unittest \
  tests.test_narrative_trust_api.NarrativeTrustTests.test_exact_public_surface_and_root_isolation \
  -v
python3 -m unittest \
  tests.test_narrative_runtime_intention \
  tests.test_narrative_simulation \
  -v
```

Expected: API/root-isolation `ok`; all 28 new behavior methods `ok`.

- [ ] **Step 4: Run full Python discovery and require exact count**

```bash
python3 -m unittest discover -s tests -v
```

Expected summary begins:

```text
Ran 541 tests in
```

and ends:

```text
OK
```

Any count other than 541 requires explicit investigation before PR readiness.

- [ ] **Step 5: Commit only the export change**

```bash
git add narrative_dynamics/narrative/__init__.py
git commit -m "feat: export narrative simulation api"
```

The commit must touch only narrative `__init__.py`.

- [ ] **Step 6: Run and inspect final exact-head `proof` CI**

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

Fetch the complete job log and confirm:

```text
Ran 541 tests in ...
OK
```

Also confirm all 12 `NarrativeRuntimeIntentionTests`, all 16 `NarrativeSimulationTests`, and `test_exact_public_surface_and_root_isolation` are individually `ok`.

- [ ] **Step 7: Verify exact final eight-path diff and forbidden-path preservation**

Compare final head to `ea45389354bf9b8dff4c18917e2bca3e3e83a939`. The changed paths must be exactly:

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

Require `intention.py` byte-for-byte unchanged by compare and no root-package diff.

- [ ] **Step 8: Update the Draft PR to review-ready only after all evidence is fresh**

Update PR description with:

```text
RED exact head/run and 29-failure distribution
Task 2 staged 25-failure distribution
Task 3 staged 17-failure distribution
Task 4 staged 13-failure distribution
Task 5 export-only 1-failure distribution
final exact head/run with 541 tests OK
exact eight-path diff
```

Mark Ready only after final CI and read-only diff review. Do not merge without explicit user instruction.

## Invariant Coverage Map

The spec's 76 invariants are covered as follows.

- Invariants 1-6: Task 2 model/result/identity tests plus byte-for-byte `intention.py` scope gate.
- Invariants 7-10: Task 3 canonical template/cutoff/context tests.
- Invariants 11-15: Task 3 posterior semantic adapter, hidden-provenance equality, and authored-math equivalence tests.
- Invariants 16-22: Task 2 result validation + Task 3 action/goal coverage, lexical tie, audit/semantic identity tests.
- Invariants 23-33: Task 4 runtime-agent/simulation-model data and execution-binding tests.
- Invariants 34-38: Task 4 `SimulationState` alignment/initialization and forged-data tests.
- Invariants 39-48: Task 5 shared-snapshot/static-schedule/action-intent/world/admission tests.
- Invariants 49-52: Task 5 explicit call-counter failure-order tests.
- Invariants 53-60: Task 5 next-state binding, repeated-template, order invariance, passive-observer, and blackout tests.
- Invariants 61-68: Task 5 trajectory validation, deterministic replay, two-round story immutability, and exact chain tests.
- Invariants 69-76: Task 6 exact API/root isolation, forbidden-path scope checks, full 541-test regression, and all Lean gates.

## Final Scientific Acceptance Gate

Do not treat “all tests pass” alone as proof of loop closure. Before PR readiness, explicitly inspect the two-round acceptance test and verify the causal artifact chain:

```text
A round-0 runtime decision result
  -> ActionIntent
  -> WorldStepResult that sets service.alert
  -> RuntimePerceptAdmissionResult with B's alert percept
  -> B round-1 RuntimeUncertainBeliefState with updated posterior
  -> B round-1 RuntimeIntentionalDecisionResult selecting respond
```

The same test must show the canonical `GenericNarrative` hash and authored decisions/observations/claims/receptions are unchanged. This is the feature's primary semantic success criterion.