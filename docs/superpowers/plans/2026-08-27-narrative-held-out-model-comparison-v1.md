# Narrative Held-Out Three-Model Comparison V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the generic narrative Reactive, Intentional, and Planning/POMDP runtime families on the existing preregistered prison train/selection/final-test protocol without changing the comparison core, fixture, or generic runtime implementations.

**Architecture:** Add one benchmark adapter module that translates each existing prison `Scenario` into one canonical generic narrative benchmark, wraps each family in a fresh-per-batch `ModelSource`, dispatches all three through `RuntimeDecisionModelSpec` / `run_runtime_decision()`, and emits the existing `initial_policy` metric contract. The preregistration and comparison layers stay untouched and continue to own dataset/partition/metric/loss/seed/candidate identity enforcement.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `random`, `MappingProxyType`, `unittest`), existing `narrative_dynamics` GenericNarrative/DomainSpec/runtime cognition/decision dispatch/planning contracts, `grounded_goal_softmax.finite_softmax`, existing observational training/selection/final comparison APIs, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-held-out-model-comparison-v1-design.md`

## Global Constraints

- Integrated base: `proof/narrative-dynamics-v0@c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765`.
- Feature branch: `work/narrative-held-out-model-comparison-v1`.
- Approved written-spec head before planning: `0b7f7f98f13685253accf26394e0777ff57e3974`.
- Docs-only spec/plan commits do not trigger proof CI; the first authoritative feature CI must be the test-only RED commit.
- Production scope is exactly one new file: `narrative_dynamics/adapters/narrative_prison.py`.
- Test scope is exactly one new file: `tests/test_narrative_held_out_model_comparison.py`.
- Do not modify `narrative_dynamics/adapters/__init__.py`, package-root exports, `model_comparison.py`, any `observations/**` implementation, generic Reactive/Intentional/Planning runtime modules, scheduler/world/conflict/projection/perception code, the fixture, or Lean sources.
- Reuse `fixtures/observations/prison_initial_choice_v1.json` byte-for-byte; it remains synthetic protocol-integration data, not empirical human behavior.
- Outer candidate parameters are exactly `{beta}` with positive finite `beta`.
- Beta grid: `(0.5, 1.0, 2.0, 4.0)`.
- Training seeds: `(101, 102)`; selection seeds: `(201, 202)`; final seeds: `(301, 302)`.
- Metric: `prison_initial_action_metrics`. Loss: categorical Brier on `initial.scout`, `initial.escape`, `initial.submit`.
- All families execute through `RuntimeDecisionModelSpec` + `run_runtime_decision()`; adapter production code must not call family-specific runtime runners directly.
- Generic Reactive must match `FinitePrisonReactiveModel.initial_policy` for every committed fixture case × beta-grid point to absolute tolerance `1e-12`, relative tolerance `0`.
- Generic Planning must match `FinitePrisonPOMDPModel.initial_policy` for every committed fixture case × beta-grid point to the same tolerance.
- Reactive and Intentional root policies are invariant to a change in only `guard_persistence`; Planning may change.
- Intentional `beta_goal` is fixed at `1.0`; goal pressure/instrumentality are fixed benchmark semantics and are not fitted.
- Adapter output is a predicted policy, not a sampled behavioral action. The supplied RNG is accepted but unused; `SimulationRunner` still records seeds in lineage.
- Horizon 1 authored actions are `escape, submit`; outer `initial_policy` must still contain `scout: 0.0`.
- Strict test-only RED -> minimal GREEN -> exact-head proof. No final GREEN claim without `completed/success` proof CI on the exact final feature head.
- Do not mark roadmap #27's held-out comparison complete until merge plus post-merge exact-head proof on `proof/narrative-dynamics-v0`.

## File Structure

- `narrative_dynamics/adapters/narrative_prison.py` — source identity, scenario translation, shared belief hooks, three family builders, unified dispatch, canonical `ModelRun` output.
- `tests/test_narrative_held_out_model_comparison.py` — authoritative RED, reference equivalence, persistence discriminator, seed behavior, and full preregistered comparison integration.
- Approved spec and this plan remain docs-only evidence.

---

### Task 1: Complete Test-Only RED Boundary

**Files:**
- Create: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Consumes existing fixture, finite prison adapters, `SimulationRunner`, training/selection/final comparison APIs.
- Produces all P1 acceptance tests before `narrative_prison.py` exists.

- [ ] **Step 1: Create guarded imports, constants, helpers, and the test class**

```python
from __future__ import annotations

import inspect
import math
from pathlib import Path
import random
import unittest

from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model
from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.model_comparison import ComparisonModel, compare_models_on_final_partition
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    construct_categorical_targets,
    fit_training_target_grid,
    load_observation_dataset,
)
from narrative_dynamics.report_artifact import attest_report
from narrative_dynamics.simulation import SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import EvaluationRole, HeldOutCase, HeldOutSuite, select_on_validation_suite

_ADAPTER_IMPORT_ERROR: ImportError | None = None
try:
    import narrative_dynamics.adapters.narrative_prison as narrative_prison_module
    from narrative_dynamics.adapters.narrative_prison import (
        NarrativePrisonModelSource,
        create_narrative_prison_intentional_source,
        create_narrative_prison_planning_source,
        create_narrative_prison_reactive_source,
    )
except ImportError as error:
    _ADAPTER_IMPORT_ERROR = error

FIXTURE = Path("fixtures/observations/prison_initial_choice_v1.json")
GRID = {"beta": (0.5, 1.0, 2.0, 4.0)}
TRAIN_SEEDS = (101, 102)
SELECTION_SEEDS = (201, 202)
FINAL_SEEDS = (301, 302)
TOL = 1e-12


def target_spec():
    return CategoricalTargetSpec(
        name="prison-initial-choice-target",
        version="1",
        categories=("scout", "escape", "submit"),
        metric_prefix="initial",
    )


def brier_loss():
    return CategoricalBrierLoss((
        CategoricalMetricGroup(
            "initial-action",
            ("initial.scout", "initial.escape", "initial.submit"),
        ),
    ))


def policy(trace):
    return {
        action: float(trace.outcome["initial_policy"][action])
        for action in ("scout", "escape", "submit")
    }


class NarrativeHeldOutModelComparisonTests(unittest.TestCase):
    def require_adapter(self):
        if _ADAPTER_IMPORT_ERROR is not None:
            self.fail(
                "generic narrative prison benchmark adapter is missing: "
                f"{_ADAPTER_IMPORT_ERROR}"
            )

    def assert_policy_close(self, left, right):
        self.assertEqual(set(left), set(right))
        for action in left:
            self.assertTrue(
                math.isclose(left[action], right[action], rel_tol=0.0, abs_tol=TOL),
                (action, left[action], right[action]),
            )
```

- [ ] **Step 2: Lock source identity and adapter isolation**

Add `test_sources_are_fresh_per_batch_and_bind_family_implementation_identity`. Create sources in order Reactive, Intentional, Planning; assert families exactly `("reactive", "intentional", "planning")`, names unique, `lifecycle == "fresh_per_batch"`, two `instantiate()` calls return distinct model objects with the source name, and `component_identity(source)` contains:

```python
for key in (
    "family",
    "lifecycle",
    "source_implementation_identity",
    "model_implementation_identity",
    "benchmark_builder_implementation_identity",
    "family_builder_implementation_identity",
    "dispatch_implementation_identity",
):
    self.assertIn(key, identity)
```

Add `test_adapter_uses_only_unified_dispatch_and_not_comparison_protocol`:

```python
source = inspect.getsource(narrative_prison_module)
for banned in (
    "narrative_dynamics.model_comparison",
    "narrative_dynamics.observations",
    "compare_models_on_final_partition",
    "PreregisteredEvaluationProtocol",
    "fit_training_target_grid",
    "run_runtime_reactive_decision",
    "run_runtime_intentional_decision",
    "run_runtime_planning_decision",
):
    self.assertNotIn(banned, source)
self.assertIn("run_runtime_decision", source)
```

- [ ] **Step 3: Lock horizon-1 normalization and dispatch-owned policy**

Use train record `train-2` (`horizon == 1`). For each source, run with `beta=2.0`, seed 17 and assert:

```python
self.assertEqual(set(trace.outcome["initial_policy"]), {"scout", "escape", "submit"})
self.assertEqual(trace.outcome["initial_policy"]["scout"], 0.0)
dispatch = trace.outcome["runtime_dispatch"]
self.assertEqual(trace.outcome["selected_action"], dispatch["selected_action"])
self.assertEqual(
    {k: trace.outcome["initial_policy"][k] for k in ("escape", "submit")},
    {k: dispatch["action_policy"][k] for k in ("escape", "submit")},
)
```

- [ ] **Step 4: Lock Reactive and Planning reference equivalence**

For every record in every fixture partition and every beta-grid value, compare generic source output against the existing finite adapter's deterministic `initial_policy`:

```python
actual = policy(runner.run_once(generic, record.scenario, {"beta": beta}, seed=1))
expected_run = legacy.simulate(record.scenario, {"beta": beta}, random.Random(1))
expected = {
    key: float(expected_run.outcome["initial_policy"][key])
    for key in ("scout", "escape", "submit")
}
self.assert_policy_close(actual, expected)
```

Implement this once for Reactive vs `create_prison_reactive_model()` and once for Planning vs `create_prison_pomdp_model()`.

- [ ] **Step 5: Lock persistence discrimination and seed lineage**

For final records `final-1` and `final-2`, run `beta=2.0`, seed 301. Assert Reactive and Intentional policies are equal to tolerance; assert Planning differs in at least one coordinate by more than `1e-12`.

Also for each family run one final scenario with seeds 301 and 302 and assert:

```python
self.assertEqual(policy(first), policy(second))
self.assertNotEqual(first.manifest.content_hash, second.manifest.content_hash)
```

- [ ] **Step 6: Lock train -> selection -> preregistered final comparison**

Implement `prepare_three_family_protocol()` exactly with existing APIs:

```python
dataset = load_observation_dataset(FIXTURE)
sources = {
    source.name: source
    for source in (
        create_narrative_prison_reactive_source(),
        create_narrative_prison_intentional_source(),
        create_narrative_prison_planning_source(),
    )
}
runner = SimulationRunner()
train_targets = construct_categorical_targets(dataset, role=ObservationPartitionRole.TRAIN, spec=target_spec())
selection_targets = construct_categorical_targets(dataset, role=ObservationPartitionRole.SELECTION_VALIDATION, spec=target_spec())
selection_suite = HeldOutSuite(
    name="generic-narrative-prison-selection-v1",
    role=EvaluationRole.SELECTION_VALIDATION,
    cases=tuple(
        HeldOutCase(
            scenario=case.scenario,
            seeds=SELECTION_SEEDS,
            target=case.target_map,
            name=case.name,
        )
        for case in selection_targets.cases
    ),
)
```

For each source:

```python
training = fit_training_target_grid(
    runner=runner,
    model=source,
    target_report=train_targets,
    parameter_grid=GRID,
    simulation_seeds=TRAIN_SEEDS,
    extractor=prison_initial_action_metrics,
    loss=brier_loss(),
)
accepted = ParameterAcceptanceSet.from_parameters(
    training.candidate_parameters,
    source_manifest_hashes=(training.manifest.content_hash,),
)
selection = select_on_validation_suite(
    runner=runner,
    model=source,
    accepted_parameters=accepted,
    suite=selection_suite,
    extractor=prison_initial_action_metrics,
    loss=brier_loss(),
)
frozen[name] = FrozenModelCandidate.from_selection(name, source, selection)
```

Create `PreregisteredEvaluationProtocol` with final seeds exactly `FINAL_SEEDS`, baseline equal to the Planning source name, all three frozen candidates, and `AdequacyThresholds(2.0, 2.0)`. Construct FINAL_TEST targets and call `compare_models_on_final_partition(...)` with exactly the preregistered candidates/sources, metric, loss, and seeds.

Assert the report contains exactly all three names, baseline is Planning, every loss is finite, every final-test case manifest records `FINAL_SEEDS`, and `attest_report(report).require_integrity()` returns the report. Do not assert a predetermined winner.

- [ ] **Step 7: Verify authoritative RED locally**

```bash
python3 -m unittest tests.test_narrative_held_out_model_comparison -v
python3 -m unittest discover -s tests -v
```

Expected: new tests fail because `narrative_dynamics.adapters.narrative_prison` is absent; unrelated existing tests stay green.

- [ ] **Step 8: Commit and capture exact-head RED CI**

```bash
git add tests/test_narrative_held_out_model_comparison.py
git commit -m "test: specify held-out narrative model comparison"
git push origin work/narrative-held-out-model-comparison-v1
```

Record exact RED SHA and proof run. Authoritative RED requires Lean stages success and Python failure only from the missing new adapter boundary.

---

### Task 2: Shared Adapter Boundary + Generic Reactive GREEN

**Files:**
- Create: `narrative_dynamics/adapters/narrative_prison.py`
- Test: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Public: `NarrativePrisonModelSource`, `create_narrative_prison_reactive_source()`, `create_narrative_prison_intentional_source()`, `create_narrative_prison_planning_source()`.
- Internal: `_ScenarioValues`, `_BenchmarkCase`, `_NarrativePrisonBenchmarkModel`, `_build_benchmark_case`, `_build_reactive_model`, `_build_intentional_model`, `_build_planning_model`, `_build_runtime_decision_model`.

- [ ] **Step 1: Add constants, scenario validation, and source contract**

Use:

```python
_BENCHMARK_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "narrative-prison-held-out-v1"
_FAMILIES = ("reactive", "intentional", "planning")
_NAMES = {
    "reactive": "generic-narrative-prison-reactive",
    "intentional": "generic-narrative-prison-intentional",
    "planning": "generic-narrative-prison-planning",
}
_ACTIONS = ("scout", "escape", "submit")
_TERMINAL_ACTIONS = ("escape", "submit")
```

`_ScenarioValues` contains the existing nine prison fields. `_scenario_values` must enforce exactly the existing finite prison schema/ranges. `_beta(parameters)` must require exactly one positive finite `beta`.

Implement:

```python
@dataclass(frozen=True)
class NarrativePrisonModelSource:
    family: str

    def __post_init__(self):
        if self.family not in _FAMILIES:
            raise ValueError("narrative prison family must be reactive, intentional, or planning")

    @property
    def name(self): return _NAMES[self.family]
    @property
    def version(self): return _BENCHMARK_VERSION
    @property
    def implementation_revision(self): return _IMPLEMENTATION_REVISION
    @property
    def lifecycle(self): return "fresh_per_batch"

    def instantiate(self):
        return _NarrativePrisonBenchmarkModel(self.family)

    def manifest_identity(self):
        family_builder = {
            "reactive": _build_reactive_model,
            "intentional": _build_intentional_model,
            "planning": _build_planning_model,
        }[self.family]
        return {
            "name": self.name,
            "version": self.version,
            "family": self.family,
            "implementation_revision": self.implementation_revision,
            "lifecycle": self.lifecycle,
            "source_implementation_identity": measure_implementation(NarrativePrisonModelSource).manifest_identity(),
            "model_implementation_identity": measure_implementation(_NarrativePrisonBenchmarkModel).manifest_identity(),
            "benchmark_builder_implementation_identity": measure_implementation(_build_benchmark_case).manifest_identity(),
            "family_builder_implementation_identity": measure_implementation(family_builder).manifest_identity(),
            "dispatch_implementation_identity": measure_implementation(run_runtime_decision).manifest_identity(),
        }
```

The three public factory functions return this source with the matching family.

- [ ] **Step 2: Build the canonical generic narrative case**

`_build_benchmark_case(scenario, values)` must create one deterministic DomainSpec/GenericNarrative with:

```text
entities: prisoner:Prisoner, guard:Guard
cells:
  guard.status           -> GuardStatus {weak,strong}
  prisoner.episode_phase -> EpisodePhase {active,terminal}
  prisoner.signal        -> Signal {none,clear,alarm}
authored seed facts:
  episode_phase=active
  signal=none
  no resolved guard.status
decision:
  id/type = prison-choice
  actor = prisoner
  context = all three cells
  horizon 1 actions = escape,submit
  horizon 2 actions = scout,escape,submit
```

Validate the narrative against the domain, create a step-zero `RuntimeEvidenceLedger`, and return `_BenchmarkCase(values, domain, story, decision_id, ledger, guard_cell, phase_cell, signal_cell)`.

- [ ] **Step 3: Add shared belief hooks**

Seed prior semantics:

```text
guard.status: weak=prior_weak, strong=1-prior_weak
episode_phase: active=1, terminal=0
signal: none=1, clear=0, alarm=0
```

Use complete hypothesis-hash mappings. Seed likelihood is neutral (`1.0` for every hypothesis). Runtime likelihood implements exact equals/clear semantics for admitted evidence. Build one `RuntimeBeliefModelSpec` used by Intentional and Planning; bind only parameters actually required by that belief model.

- [ ] **Step 4: Implement exact finite-reactive root values in `_build_reactive_model`**

Direct values:

```python
escape = values.prior_weak * values.escape_reward - (1.0 - values.prior_weak) * values.capture_cost
submit = values.submit_reward
```

For horizon 2, compute the existing finite-reactive scout value exactly: `cue_strength = 2*signal_accuracy-1`, `cue_scale=(escape_reward+capture_cost)/2`, clear/alarm terminal value maps, terminal `finite_softmax(..., beta=beta)`, expected soft terminal value, then `-scout_cost + discount * terminal_value`. Do not read `guard_persistence`.

Construct `RuntimeReactiveDecisionModelSpec` with `cue_cells=(guard_cell,)`, supported type `("prison-choice",)`, fitted `beta`, sanitized scenario parameters required by the hook, and the benchmark score hook.

- [ ] **Step 5: Add outer unified dispatch/output**

`_NarrativePrisonBenchmarkModel.simulate(...)` must:

```python
values = _scenario_values(scenario)
beta = _beta(parameters)
case = _build_benchmark_case(scenario, values)
nested = _build_runtime_decision_model(self.family, case, beta)
dispatch = run_runtime_decision(
    case.story,
    case.domain,
    case.decision_id,
    case.ledger,
    RuntimeDecisionModelSpec(self.family, nested),
)
initial_policy = {action: 0.0 for action in _ACTIONS}
for action, probability in dispatch.action_policy.items():
    initial_policy[action] = probability
return ModelRun(
    events=(),
    outcome={
        "model_kind": self.family,
        "initial_policy": initial_policy,
        "selected_action": dispatch.selected_action,
        "runtime_dispatch_hash": dispatch.content_hash,
        "runtime_dispatch": dispatch.to_dict(),
        "translated_story_hash": case.story.content_hash,
        "translated_domain_hash": case.domain.content_hash,
        "runtime_ledger_hash": case.ledger.content_hash,
    },
)
```

Do not sample from `rng`.

- [ ] **Step 6: Verify shared/source/Reactive tests**

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_sources_are_fresh_per_batch_and_bind_family_implementation_identity \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_adapter_uses_only_unified_dispatch_and_not_comparison_protocol \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas -v
```

Expected: PASS. Intentional/Planning acceptance tests may remain red.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison reactive adapter"
```

---

### Task 3: Generic Intentional GREEN

**Files:**
- Modify: `narrative_dynamics/adapters/narrative_prison.py`

**Interfaces:**
- Consumes `_BenchmarkCase`, shared runtime belief model, fitted `beta`.
- Produces `RuntimeIntentionalDecisionModelSpec` with fixed benchmark goals and `beta_goal=1.0`.

- [ ] **Step 1: Build exact goal instrumentality**

Create `freedom` and `safety` goals. Both use cell weights:

```python
{case.guard_cell: 1.0, case.phase_cell: 0.0, case.signal_cell: 0.0}
```

Instrumentality must cover every posterior hypothesis hash for every context cell. Guard values are `weak:+1,strong:-1` for freedom and reversed for safety; all phase/signal values are `0.0`. Both goals use pressure `1.0`, cost/risk `0.0`.

Create `GoalModelSpec("generic-narrative-prison-goals", _BENCHMARK_VERSION, 1.0, (freedom, safety))`.

- [ ] **Step 2: Build exact goal-conditional action values**

Freedom:

```python
escape = values.escape_reward
submit = 0.0
scout = (2.0 * values.signal_accuracy - 1.0) * values.escape_reward - values.scout_cost
```

Safety:

```python
escape = -values.capture_cost
submit = values.submit_reward
scout = (2.0 * values.signal_accuracy - 1.0) * values.capture_cost - values.scout_cost
```

Drop `scout` completely when horizon is 1. `guard_persistence` must not participate.

Use `ChoiceModelSpec(..., beta_action=beta, values={"freedom": ..., "safety": ...})`, then construct `RuntimeIntentionalDecisionModelSpec` with the shared runtime belief model and supported type `("prison-choice",)`.

- [ ] **Step 3: Verify Intentional invariance and horizon contract**

Run the horizon-1 test and persistence discriminator test. Reactive and Intentional subcases must pass; only Planning may still be red.

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_persistence_pair_separates_planning_from_non_lookahead_families -v
```

- [ ] **Step 4: Commit**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison intentional adapter"
```

---

### Task 4: Generic Planning/POMDP GREEN + Reference Equality

**Files:**
- Modify: `narrative_dynamics/adapters/narrative_prison.py`

**Interfaces:**
- Consumes `_BenchmarkCase`, shared runtime belief model, scenario `discount`, fitted `beta`.
- Produces a four-state `RuntimePlanningDecisionModelSpec` whose root policy matches the finite POMDP reference.

- [ ] **Step 1: Declare hidden states and observations**

Hidden states exactly:

```text
weak-active
strong-active
weak-terminal
strong-terminal
```

Each assigns both guard and phase cells. Observations exactly `clear`, `alarm`, `none`, each assigning the signal cell. Joint-belief hook must preserve guard and phase marginals exactly, producing root mass `(prior_weak, 1-prior_weak, 0, 0)`.

- [ ] **Step 2: Implement transition distribution**

Exact semantics:

```text
depth0 scout: active -> same active state
depth0 escape/submit: preserve guard, phase -> terminal
depth1 escape from weak-active: weak-terminal=persistence, strong-terminal=1-persistence
depth1 escape from strong-active: strong-terminal=persistence, weak-terminal=1-persistence
depth1 submit from active: preserve guard, phase -> terminal
any action from terminal: same terminal state
```

Every returned mapping covers all four hidden-state IDs and sums to 1.

- [ ] **Step 3: Implement observation distribution**

After depth-0 scout from active states:

```text
weak-active: clear=accuracy, alarm=1-accuracy, none=0
strong-active: clear=1-accuracy, alarm=accuracy, none=0
```

After direct escape/submit or from terminal: `none=1`, `clear=alarm=0`.

- [ ] **Step 4: Implement reward semantics**

```text
depth0 scout -> -scout_cost
depth0 escape weak-active -> +escape_reward
depth0 escape strong-active -> -capture_cost
depth0 submit active -> submit_reward
depth1 escape -> reward from NEXT terminal guard state (+escape_reward weak, -capture_cost strong)
depth1 submit -> submit_reward
any action from terminal -> 0
```

Using the next-state guard at depth 1 is mandatory; this is where `guard_persistence` enters expected escape value.

- [ ] **Step 5: Construct exact planning schedule**

Horizon 1:

```python
(("escape", "submit"),)
```

Horizon 2:

```python
(("scout", "escape", "submit"), ("escape", "submit"))
```

Planning cells are `(guard_cell, phase_cell)`, observation cells `(signal_cell,)`, with shared belief model, four states, three observations, scenario discount, fitted beta, and the four benchmark hooks.

- [ ] **Step 6: Require reference equality; do not loosen tolerance**

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_planning_matches_finite_pomdp_on_all_fixture_cases_and_betas \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_persistence_pair_separates_planning_from_non_lookahead_families \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch -v
```

Expected: PASS. If POMDP equality fails, correct transition/observation/reward ordering; do not change the `1e-12` contract.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison planning adapter"
```

---

### Task 5: Full Preregistered Held-Out GREEN

**Files:**
- Production remains `narrative_dynamics/adapters/narrative_prison.py`.
- Test remains `tests/test_narrative_held_out_model_comparison.py`.

**Interfaces:**
- Produces a valid `ModelComparisonReport` over exactly three frozen generic narrative candidates after train and selection-validation.

- [ ] **Step 1: Run the complete new module**

```bash
python3 -m unittest tests.test_narrative_held_out_model_comparison -v
```

Expected: PASS.

- [ ] **Step 2: Run the full Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: PASS. Record actual test count/runtime; do not predict them.

- [ ] **Step 3: Enforce exact approved diff scope**

```bash
git diff --name-only c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765...HEAD
```

Allowed files only:

```text
docs/superpowers/specs/2026-08-27-narrative-held-out-model-comparison-v1-design.md
docs/superpowers/plans/2026-08-27-narrative-held-out-model-comparison-v1.md
narrative_dynamics/adapters/narrative_prison.py
tests/test_narrative_held_out_model_comparison.py
```

Any other path is a blocker requiring scope review.

- [ ] **Step 4: If the test implementation itself needs a semantics-preserving correction, commit it separately**

```bash
git add tests/test_narrative_held_out_model_comparison.py
git commit -m "test: tighten held-out narrative comparison evidence"
```

This step is used only for correcting the RED test implementation to match the approved spec; it must not weaken reference equality, data-role separation, or identity checks.

---

### Task 6: Exact-Head Proof, Review, and Integration Handoff

**Files:**
- No code changes expected after final GREEN.

**Interfaces:**
- Produces authoritative proof evidence and a review-ready branch. Roadmap completion remains pending until post-merge proof.

- [ ] **Step 1: Capture and push exact final head**

```bash
git rev-parse HEAD
git push origin work/narrative-held-out-model-comparison-v1
```

- [ ] **Step 2: Require proof CI `completed/success` on that exact SHA**

All existing workflow stages must pass, including the exact Python command:

```bash
python3 -m unittest discover -s tests -v
```

Verify run `head_sha == git rev-parse HEAD`.

- [ ] **Step 3: Final review checklist**

Explicitly verify:

```text
comparison core unchanged
observation protocol/release unchanged
fixture unchanged
generic runtime family implementations unchanged
production uses only run_runtime_decision
Reactive == finite Reactive reference grid
Planning == finite POMDP reference grid
Intentional persistence-invariant
Planning persistence-sensitive
three candidates selected then frozen before FINAL_TEST
FINAL_TEST uses exactly (301,302)
no P2 stochasticity/recovery/identifiability/log-score work added
```

- [ ] **Step 4: Open/update PR evidence**

Base is `proof/narrative-dynamics-v0`. PR body records exact integrated base, authoritative RED SHA/run/failure, final GREEN SHA/run/Python count/OK, both reference witnesses, three-family train->selection->final protocol, final seeds, and P2 exclusions.

- [ ] **Step 5: Invoke finishing-development-branch after exact-head GREEN**

Do not auto-merge as implementation. Present integration choices. If merge is selected, require a new proof run `completed/success` on the exact new `proof/narrative-dynamics-v0` head before checking off #27's held-out comparison item.
