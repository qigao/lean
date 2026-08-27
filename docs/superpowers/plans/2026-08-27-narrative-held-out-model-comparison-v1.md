# Narrative Held-Out Three-Model Comparison V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Compare the generic narrative Reactive, Intentional, and Planning/POMDP runtime families on the existing preregistered prison train/selection/final-test protocol without changing the comparison core, fixture, or generic runtime implementations.

**Architecture:** Add one benchmark adapter module that translates each existing prison `Scenario` into one canonical generic narrative benchmark, wraps each family in a fresh-per-batch `ModelSource`, dispatches all three through `RuntimeDecisionModelSpec` / `run_runtime_decision()`, and emits the existing `initial_policy` metric contract. Keep all protocol identity checks in `PreregisteredEvaluationProtocol` / `compare_models_on_final_partition()` unchanged; the only production code is the adapter.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `math`, `MappingProxyType`, `unittest`), existing `narrative_dynamics` GenericNarrative/DomainSpec/runtime cognition/decision dispatch/planning contracts, `grounded_goal_softmax.finite_softmax`, existing observation dataset / training / held-out validation / model comparison pipeline, GitHub Actions `.github/workflows/proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-27-narrative-held-out-model-comparison-v1-design.md`

## Global Constraints

- Integrated base is `proof/narrative-dynamics-v0` at `c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765`.
- Feature branch is `work/narrative-held-out-model-comparison-v1`.
- Approved spec head before this plan is `0b7f7f98f13685253accf26394e0777ff57e3974`.
- Pure changes under `docs/superpowers/specs/**` and `docs/superpowers/plans/**` do not trigger proof CI. The first authoritative feature CI must be the test-only RED commit.
- Production scope is exactly one new module: `narrative_dynamics/adapters/narrative_prison.py`. Do not modify `narrative_dynamics/adapters/__init__.py` unless a concrete import requirement appears in the RED tests; direct module imports are sufficient for V1.
- Test scope is one new file: `tests/test_narrative_held_out_model_comparison.py`.
- Do not modify `narrative_dynamics/model_comparison.py`, any `narrative_dynamics/observations/**` implementation, generic runtime Reactive/Intentional/Planning modules, `narrative_dynamics/narrative/simulation.py`, world/conflict/projection/perception code, the committed fixture, package root exports, or Lean sources.
- Reuse `fixtures/observations/prison_initial_choice_v1.json` byte-for-byte. It remains explicitly synthetic protocol-integration data, not empirical human behavior.
- Outer candidate parameter schema is exactly `{beta}` with positive finite `beta`.
- Benchmark beta grid remains exactly `(0.5, 1.0, 2.0, 4.0)`.
- Training seeds are `(101, 102)`, selection seeds are `(201, 202)`, final-test seeds are `(301, 302)`.
- Final metric remains `prison_initial_action_metrics`; loss remains categorical Brier over `initial.scout`, `initial.escape`, `initial.submit`.
- All three families must execute through `RuntimeDecisionModelSpec` and `run_runtime_decision()`. Production adapter code must not directly call the three family-specific runtime runner functions.
- Generic Reactive must reproduce `FinitePrisonReactiveModel` root `initial_policy` for every committed fixture scenario and beta-grid value within absolute tolerance `1e-12`, relative tolerance `0`.
- Generic Planning must reproduce `FinitePrisonPOMDPModel` root `initial_policy` for every committed fixture scenario and beta-grid value within absolute tolerance `1e-12`, relative tolerance `0`.
- Reactive and Intentional root policies must be invariant when only `guard_persistence` changes; Planning must be allowed to change because it models future guard transition structure.
- `beta_goal` is fixed at `1.0`; goal pressures and instrumentality are fixed benchmark semantics. Do not fit or recover them in P1.
- No model samples an action for the metric. The supplied runner RNG is ignored by the adapter; `SimulationRunner` still records and preregisters seeds in lineage.
- Horizon 1 emits a three-coordinate outer policy with `scout == 0.0`, even though the authored generic decision contains only `escape` and `submit`.
- Use strict test-only RED -> minimal GREEN -> exact-head proof discipline. Do not claim GREEN until proof CI is `completed/success` on the exact final feature head.
- Do not mark roadmap #27's held-out comparison checkbox complete before merge plus post-merge exact-head proof on `proof/narrative-dynamics-v0`.

## File Structure

- `narrative_dynamics/adapters/narrative_prison.py` — scenario validation/translation, fresh-per-batch source identity, shared runtime belief hooks, family-specific benchmark builders, unified dispatch, canonical `ModelRun` output.
- `tests/test_narrative_held_out_model_comparison.py` — authoritative RED boundary, legacy-reference equivalence, persistence discriminator, source-identity checks, seed behavior, and full train/selection/final comparison integration.
- `docs/superpowers/specs/2026-08-27-narrative-held-out-model-comparison-v1-design.md` — approved architecture and scientific semantics.
- `docs/superpowers/plans/2026-08-27-narrative-held-out-model-comparison-v1.md` — this execution plan.

---

### Task 1: Establish the Complete Test-Only RED Boundary

**Files:**
- Create: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Consumes: existing prison fixture, legacy finite prison Reactive/POMDP adapters, `SimulationRunner`, training/selection/final-test protocol APIs, `attest_report`, `component_identity`.
- Produces: one authoritative test-only RED commit defining all P1 acceptance behavior before `narrative_prison.py` exists.

- [ ] **Step 1: Add guarded imports, constants, and protocol helpers**

Create the test file with these imports and fixed protocol constants:

```python
from __future__ import annotations

import inspect
import math
from pathlib import Path
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
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    select_on_validation_suite,
)

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
```

Inside the test class add:

```python
def require_adapter(self):
    if _ADAPTER_IMPORT_ERROR is not None:
        self.fail(
            "generic narrative prison benchmark adapter is missing: "
            f"{_ADAPTER_IMPORT_ERROR}"
        )
```

- [ ] **Step 2: Lock the public source and identity boundary**

Add:

```python
def test_sources_are_fresh_per_batch_and_bind_family_implementation_identity(self):
    self.require_adapter()
    sources = (
        create_narrative_prison_reactive_source(),
        create_narrative_prison_intentional_source(),
        create_narrative_prison_planning_source(),
    )
    self.assertTrue(all(isinstance(source, NarrativePrisonModelSource) for source in sources))
    self.assertEqual(
        tuple(source.family for source in sources),
        ("reactive", "intentional", "planning"),
    )
    self.assertEqual(len({source.name for source in sources}), 3)
    for source in sources:
        self.assertEqual(source.lifecycle, "fresh_per_batch")
        first = source.instantiate()
        second = source.instantiate()
        self.assertIsNot(first, second)
        self.assertEqual(first.name, source.name)
        identity = component_identity(source)
        self.assertEqual(identity["family"], source.family)
        self.assertEqual(identity["lifecycle"], "fresh_per_batch")
        for key in (
            "source_implementation_identity",
            "model_implementation_identity",
            "benchmark_builder_implementation_identity",
            "family_builder_implementation_identity",
            "dispatch_implementation_identity",
        ):
            self.assertIn(key, identity)
```

Also lock the module boundary:

```python
def test_adapter_does_not_depend_on_comparison_or_observation_protocol_modules(self):
    self.require_adapter()
    source = inspect.getsource(narrative_prison_module)
    for banned in (
        "narrative_dynamics.model_comparison",
        "narrative_dynamics.observations",
        "compare_models_on_final_partition",
        "PreregisteredEvaluationProtocol",
        "fit_training_target_grid",
    ):
        self.assertNotIn(banned, source)
    self.assertIn("run_runtime_decision", source)
    for banned_runner in (
        "run_runtime_reactive_decision",
        "run_runtime_intentional_decision",
        "run_runtime_planning_decision",
    ):
        self.assertNotIn(banned_runner, source)
```

- [ ] **Step 3: Lock horizon normalization and dispatch-owned policy output**

Add:

```python
def test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    scenario = dataset.partition(ObservationPartitionRole.TRAIN).records[1].scenario
    runner = SimulationRunner()
    for source in (
        create_narrative_prison_reactive_source(),
        create_narrative_prison_intentional_source(),
        create_narrative_prison_planning_source(),
    ):
        trace = runner.run_once(source, scenario, {"beta": 2.0}, seed=17)
        self.assertEqual(set(trace.outcome["initial_policy"]), {"scout", "escape", "submit"})
        self.assertEqual(trace.outcome["initial_policy"]["scout"], 0.0)
        dispatch = trace.outcome["runtime_dispatch"]
        self.assertEqual(
            trace.outcome["selected_action"],
            dispatch["selected_action"],
        )
        self.assertEqual(
            {
                key: trace.outcome["initial_policy"][key]
                for key in ("escape", "submit")
            },
            {
                key: dispatch["action_policy"][key]
                for key in ("escape", "submit")
            },
        )
```

- [ ] **Step 4: Lock Reactive and Planning equivalence to the existing finite references**

Add:

```python
def assert_policy_close(self, left, right):
    self.assertEqual(set(left), set(right))
    for action in left:
        self.assertTrue(
            math.isclose(left[action], right[action], rel_tol=0.0, abs_tol=TOL),
            (action, left[action], right[action]),
        )


def test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    runner = SimulationRunner()
    generic = create_narrative_prison_reactive_source()
    legacy = create_prison_reactive_model()
    for partition in dataset.partitions:
        for record in partition.records:
            for beta in GRID["beta"]:
                with self.subTest(record=record.id, beta=beta):
                    actual = policy(runner.run_once(generic, record.scenario, {"beta": beta}, seed=1))
                    expected_run = legacy.simulate(record.scenario, {"beta": beta}, __import__("random").Random(1))
                    expected = {
                        key: float(expected_run.outcome["initial_policy"][key])
                        for key in ("scout", "escape", "submit")
                    }
                    self.assert_policy_close(actual, expected)


def test_generic_planning_matches_finite_pomdp_on_all_fixture_cases_and_betas(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    runner = SimulationRunner()
    generic = create_narrative_prison_planning_source()
    legacy = create_prison_pomdp_model()
    for partition in dataset.partitions:
        for record in partition.records:
            for beta in GRID["beta"]:
                with self.subTest(record=record.id, beta=beta):
                    actual = policy(runner.run_once(generic, record.scenario, {"beta": beta}, seed=1))
                    expected_run = legacy.simulate(record.scenario, {"beta": beta}, __import__("random").Random(1))
                    expected = {
                        key: float(expected_run.outcome["initial_policy"][key])
                        for key in ("scout", "escape", "submit")
                    }
                    self.assert_policy_close(actual, expected)
```

- [ ] **Step 5: Lock the held-out persistence discriminator and seed semantics**

Add:

```python
def test_persistence_pair_separates_planning_from_non_lookahead_families(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    records = {
        record.id: record
        for record in dataset.partition(ObservationPartitionRole.FINAL_TEST).records
    }
    first = records["final-1"].scenario
    second = records["final-2"].scenario
    runner = SimulationRunner()
    for source in (
        create_narrative_prison_reactive_source(),
        create_narrative_prison_intentional_source(),
    ):
        left = policy(runner.run_once(source, first, {"beta": 2.0}, seed=301))
        right = policy(runner.run_once(source, second, {"beta": 2.0}, seed=301))
        self.assert_policy_close(left, right)
    planning = create_narrative_prison_planning_source()
    left = policy(runner.run_once(planning, first, {"beta": 2.0}, seed=301))
    right = policy(runner.run_once(planning, second, {"beta": 2.0}, seed=301))
    self.assertTrue(any(abs(left[key] - right[key]) > TOL for key in left))


def test_seed_changes_lineage_but_not_predicted_policy(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    scenario = dataset.partition(ObservationPartitionRole.FINAL_TEST).records[0].scenario
    runner = SimulationRunner()
    for source in (
        create_narrative_prison_reactive_source(),
        create_narrative_prison_intentional_source(),
        create_narrative_prison_planning_source(),
    ):
        first = runner.run_once(source, scenario, {"beta": 2.0}, seed=301)
        second = runner.run_once(source, scenario, {"beta": 2.0}, seed=302)
        self.assertEqual(policy(first), policy(second))
        self.assertNotEqual(first.manifest.content_hash, second.manifest.content_hash)
```

- [ ] **Step 6: Lock the complete train -> selection -> preregistered final comparison**

Add a helper inside the test class:

```python
def prepare_three_family_protocol(self):
    self.require_adapter()
    dataset = load_observation_dataset(FIXTURE)
    spec = target_spec()
    loss = brier_loss()
    sources = {
        source.name: source
        for source in (
            create_narrative_prison_reactive_source(),
            create_narrative_prison_intentional_source(),
            create_narrative_prison_planning_source(),
        )
    }
    runner = SimulationRunner()
    train_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.TRAIN,
        spec=spec,
    )
    selections = {}
    frozen = {}
    selection_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.SELECTION_VALIDATION,
        spec=spec,
    )
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
    for name, source in sources.items():
        training = fit_training_target_grid(
            runner=runner,
            model=source,
            target_report=train_targets,
            parameter_grid=GRID,
            simulation_seeds=TRAIN_SEEDS,
            extractor=prison_initial_action_metrics,
            loss=loss,
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
            loss=loss,
        )
        selections[name] = selection
        frozen[name] = FrozenModelCandidate.from_selection(name, source, selection)

    planning_name = create_narrative_prison_planning_source().name
    protocol = PreregisteredEvaluationProtocol.create(
        name="generic-narrative-prison-three-family-v1",
        version="1",
        dataset=dataset,
        target_spec=spec,
        extractor=prison_initial_action_metrics,
        loss=loss,
        simulation_seeds=FINAL_SEEDS,
        baseline_name=planning_name,
        candidates=tuple(frozen[name] for name in sorted(frozen)),
        thresholds=AdequacyThresholds(max_mean_loss=1.0, max_worst_loss=1.0),
    )
    final_targets = construct_categorical_targets(
        dataset,
        role=ObservationPartitionRole.FINAL_TEST,
        spec=spec,
    )
    return runner, sources, frozen, protocol, final_targets, loss
```

Then add:

```python
def test_three_generic_families_complete_preregistered_final_comparison(self):
    runner, sources, frozen, protocol, final_targets, loss = self.prepare_three_family_protocol()
    report = compare_models_on_final_partition(
        runner=runner,
        models=tuple(
            ComparisonModel(frozen=frozen[name], model=sources[name])
            for name in sorted(sources)
        ),
        target_set=final_targets,
        extractor=prison_initial_action_metrics,
        loss=loss,
        simulation_seeds=FINAL_SEEDS,
        protocol=protocol,
    )
    self.assertEqual(
        {entry.name for entry in report.ranking},
        set(sources),
    )
    self.assertEqual(report.baseline_name, create_narrative_prison_planning_source().name)
    self.assertIs(attest_report(report).require_integrity(), report)
    for entry in report.ranking:
        self.assertTrue(math.isfinite(entry.mean_loss))
        self.assertTrue(math.isfinite(entry.worst_loss))
        cases = entry.final_test.validation.manifest.inputs["cases"]
        self.assertTrue(cases)
        self.assertTrue(all(case["seeds"] == FINAL_SEEDS for case in cases))
```

Do **not** assert a predetermined winner; this task is to perform a valid held-out comparison, not manufacture a ranking.

- [ ] **Step 7: Run the new test file and the full Python suite to verify RED**

Run:

```bash
python3 -m unittest tests.test_narrative_held_out_model_comparison -v
python3 -m unittest discover -s tests -v
```

Expected new-test result: FAIL because `narrative_dynamics.adapters.narrative_prison` does not exist. Existing unrelated tests must remain green.

- [ ] **Step 8: Commit the complete test-only RED boundary**

```bash
git add tests/test_narrative_held_out_model_comparison.py
git commit -m "test: specify held-out narrative model comparison"
```

Push the exact commit. Because tests are not path-ignored, `.github/workflows/proof.yml` must start. Record the exact branch head SHA and proof run number. Authoritative RED requires Lean gates to remain green and Python to fail only because the new adapter/public API is missing.

---

### Task 2: Implement the Fresh Model Source, Scenario Translation, and Generic Reactive Candidate

**Files:**
- Create: `narrative_dynamics/adapters/narrative_prison.py`
- Test: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Produces public `NarrativePrisonModelSource`, `create_narrative_prison_reactive_source()`, `create_narrative_prison_intentional_source()`, `create_narrative_prison_planning_source()`.
- Produces internal `_ScenarioValues`, `_BenchmarkCase`, `_NarrativePrisonBenchmarkModel`, `_build_benchmark_case(...)`, `_build_runtime_decision_model(...)`.
- Reactive path must already make source/identity/horizon/reference tests pass; Intentional and Planning tests may remain red until later tasks.

- [ ] **Step 1: Add canonical constants, scenario validation, and source identity**

Start the module with the exact public/family constants:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import random

from narrative_dynamics.attestation import measure_implementation
from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.narrative.runtime_decision_dispatch import (
    RuntimeDecisionModelSpec,
    run_runtime_decision,
)

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


@dataclass(frozen=True)
class _ScenarioValues:
    prior_weak: float
    signal_accuracy: float
    guard_persistence: float
    escape_reward: float
    capture_cost: float
    submit_reward: float
    scout_cost: float
    discount: float
    horizon: int
```

Implement `_finite`, `_probability`, `_positive_beta`, and `_scenario_values(scenario)` with exactly the existing finite prison schema/ranges. `_positive_beta(parameters)` must reject any key set other than `{"beta"}`.

Add the source:

```python
@dataclass(frozen=True)
class NarrativePrisonModelSource:
    family: str

    def __post_init__(self):
        if self.family not in _FAMILIES:
            raise ValueError("narrative prison family must be reactive, intentional, or planning")

    @property
    def name(self):
        return _NAMES[self.family]

    @property
    def version(self):
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self):
        return _IMPLEMENTATION_REVISION

    @property
    def lifecycle(self):
        return "fresh_per_batch"

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

The three creation functions return `NarrativePrisonModelSource("reactive")`, `NarrativePrisonModelSource("intentional")`, and `NarrativePrisonModelSource("planning")` respectively.

- [ ] **Step 2: Build one canonical generic narrative case per outer Scenario**

Use existing `DomainSpec`, GenericNarrative IR constructors, and `runtime_evidence_ledger_from_story(...)`. The builder must create these typed cells exactly:

```text
guard.status             : GuardStatus {weak,strong}
prisoner.episode_phase   : EpisodePhase {active,terminal}
prisoner.signal          : Signal {none,clear,alarm}
```

The story contains one `Prisoner` actor `prisoner`, one `Guard` subject `guard`, deterministic authored facts `episode_phase=active` and `signal=none`, no resolved authored `guard.status`, and one decision `prison-choice` of type `prison-choice` at the source cutoff.

For horizon 1 authored actions are exactly `escape`, `submit`; for horizon 2 they are exactly `scout`, `escape`, `submit`. Decision context cells are all three cells. Build the step-zero runtime evidence ledger from the translated story/domain and return:

```python
@dataclass(frozen=True)
class _BenchmarkCase:
    values: _ScenarioValues
    domain: DomainSpec
    story: GenericNarrative
    decision_id: str
    ledger: RuntimeEvidenceLedger
    guard_cell: StateCellRef
    phase_cell: StateCellRef
    signal_cell: StateCellRef
```

Validate the translated story against the domain before returning it.

- [ ] **Step 3: Implement the shared seed/runtime belief hooks**

Define a seed prior hook whose guard prior is scenario-driven and whose deterministic authored cells remain degenerate:

```python
class _PrisonSeedPriorHook:
    def __call__(self, agent_id, cell, hypotheses, parameters):
        values = {stable_content_hash(item.to_dict()): item for item in hypotheses}
        if cell.state_variable == "guard.status":
            return {
                key: parameters["prior_weak"] if item.value == "weak" else 1.0 - parameters["prior_weak"]
                for key, item in values.items()
            }
        expected = {
            "prisoner.episode_phase": "active",
            "prisoner.signal": "none",
        }[cell.state_variable]
        return {key: 1.0 if item.value == expected else 0.0 for key, item in values.items()}
```

Use a seed likelihood hook returning `1.0` for every hypothesis because no authored observation should distort the configured prior. Use a runtime likelihood hook that respects exact equals/clear semantics for any admitted evidence even though this benchmark executes at step zero. Construct one `RuntimeBeliefModelSpec` shared by Intentional and Planning and bind `prior_weak` in its parameters.

- [ ] **Step 4: Implement the Reactive score hook and model builder**

Use `RuntimeReactiveDecisionModelSpec`. The score hook receives only sanitized context plus model parameters. Compute the exact legacy finite-reactive values:

```python
def _direct_values(values):
    return {
        "escape": values.prior_weak * values.escape_reward - (1.0 - values.prior_weak) * values.capture_cost,
        "submit": values.submit_reward,
    }
```

For horizon 2 compute cue terminal values using `cue_strength = 2 * signal_accuracy - 1`, `cue_scale = (escape_reward + capture_cost) / 2`, form clear/alarm terminal policies with `finite_softmax(..., beta=beta)`, integrate the expected terminal soft value, and return `scout = -scout_cost + discount * terminal_value`. Never read `guard_persistence` in this hook.

Build:

```python
RuntimeReactiveDecisionModelSpec(
    model_id="generic-narrative-prison-reactive-runtime",
    version=_BENCHMARK_VERSION,
    supported_decision_types=("prison-choice",),
    cue_cells=(case.guard_cell,),
    parameters={...scenario values needed by the score hook...},
    beta=beta,
    score_hook=_PrisonReactiveScoreHook(),
)
```

- [ ] **Step 5: Implement unified outer dispatch/output for the Reactive path**

The outer model is:

```python
@dataclass(frozen=True)
class _NarrativePrisonBenchmarkModel:
    family: str

    @property
    def name(self):
        return _NAMES[self.family]

    @property
    def version(self):
        return _BENCHMARK_VERSION

    @property
    def implementation_revision(self):
        return _IMPLEMENTATION_REVISION

    def simulate(self, scenario: Scenario, parameters: Mapping[str, float], rng: random.Random) -> ModelRun:
        values = _scenario_values(scenario)
        beta = _positive_beta(parameters)
        case = _build_benchmark_case(scenario, values)
        nested = _build_runtime_decision_model(self.family, case, beta)
        dispatch_model = RuntimeDecisionModelSpec(self.family, nested)
        dispatch = run_runtime_decision(
            case.story,
            case.domain,
            case.decision_id,
            case.ledger,
            dispatch_model,
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

Do not use `rng` beyond accepting the protocol signature.

- [ ] **Step 6: Run the Reactive-focused tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_sources_are_fresh_per_batch_and_bind_family_implementation_identity \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_adapter_does_not_depend_on_comparison_or_observation_protocol_modules \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas -v
```

Expected: PASS. Intentional/Planning/full-pipeline tests are still allowed to fail because those builders are not complete yet.

- [ ] **Step 7: Commit Reactive/shared GREEN**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison reactive adapter"
```

---

### Task 3: Implement the Generic Intentional Candidate

**Files:**
- Modify: `narrative_dynamics/adapters/narrative_prison.py`
- Test: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Consumes `_BenchmarkCase`, shared runtime belief model, fitted outer `beta`.
- Produces `RuntimeIntentionalDecisionModelSpec` with fixed `beta_goal=1.0`, fixed benchmark goal semantics, and fitted `ChoiceModelSpec.beta_action=beta`.

- [ ] **Step 1: Add complete fixed goal semantics**

Build two `GoalSpec` values: `freedom` and `safety`. For both, cell weights are exactly:

```python
{
    case.guard_cell: 1.0,
    case.phase_cell: 0.0,
    case.signal_cell: 0.0,
}
```

For every context cell, construct complete instrumentality mappings over every domain hypothesis hash. Guard instrumentality is `weak:+1,strong:-1` for `freedom` and the reverse for `safety`; all phase/signal instrumentality values are `0.0`.

Create:

```python
GoalModelSpec(
    "generic-narrative-prison-goals",
    _BENCHMARK_VERSION,
    1.0,
    (freedom, safety),
)
```

- [ ] **Step 2: Add exact goal-conditional choice values**

For horizon 1 use only `escape`, `submit`. For horizon 2 include `scout`.

Freedom values:

```python
{
    "escape": values.escape_reward,
    "submit": 0.0,
    "scout": (2.0 * values.signal_accuracy - 1.0) * values.escape_reward - values.scout_cost,
}
```

Safety values:

```python
{
    "escape": -values.capture_cost,
    "submit": values.submit_reward,
    "scout": (2.0 * values.signal_accuracy - 1.0) * values.capture_cost - values.scout_cost,
}
```

Drop `scout` entirely from both maps when horizon is 1. `guard_persistence` must not be used.

Construct `ChoiceModelSpec(..., beta_action=beta, values={...})` and then `RuntimeIntentionalDecisionModelSpec(...)` using the shared runtime belief model.

- [ ] **Step 3: Run Intentional behavior and horizon tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_persistence_pair_separates_planning_from_non_lookahead_families \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_seed_changes_lineage_but_not_predicted_policy -v
```

At this point the persistence test may still fail only on the Planning branch because Planning is not implemented; Reactive and Intentional subcases must pass.

- [ ] **Step 4: Commit Intentional GREEN**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison intentional adapter"
```

---

### Task 4: Implement the Generic Planning/POMDP Candidate and Legacy Equivalence

**Files:**
- Modify: `narrative_dynamics/adapters/narrative_prison.py`
- Test: `tests/test_narrative_held_out_model_comparison.py`

**Interfaces:**
- Consumes `_BenchmarkCase`, shared runtime belief model, scenario `discount`, fitted `beta`.
- Produces a four-state `RuntimePlanningDecisionModelSpec` whose root policy numerically matches the existing finite prison POMDP.

- [ ] **Step 1: Add four joint hidden states and three observation atoms**

Create hidden states exactly:

```text
weak-active
strong-active
weak-terminal
strong-terminal
```

Each assigns both `guard.status` and `prisoner.episode_phase`. Observations are `clear`, `alarm`, `none`, each assigning `prisoner.signal` to the matching typed value.

The joint-belief hook must compute each hidden-state mass from the guard and phase marginals supplied in `RuntimePlanningBeliefContext` and therefore produce root mass `(prior_weak, 1-prior_weak, 0, 0)` without reading objective world state.

- [ ] **Step 2: Implement the transition hook**

Use exact action/depth/state semantics:

```text
depth 0 scout: active state -> same active state with probability 1
depth 0 escape/submit: preserve guard status, phase -> terminal
depth 1 escape from weak-active: weak-terminal=persistence, strong-terminal=1-persistence
depth 1 escape from strong-active: strong-terminal=persistence, weak-terminal=1-persistence
depth 1 submit from active: preserve guard status, phase -> terminal
any action from terminal: same terminal state with probability 1
```

Every returned distribution must cover all four state IDs exactly and sum to one.

- [ ] **Step 3: Implement the observation hook**

At depth 0 after `scout` from active states:

```text
weak-active   -> clear=signal_accuracy, alarm=1-signal_accuracy, none=0
strong-active -> clear=1-signal_accuracy, alarm=signal_accuracy, none=0
```

After direct `escape`/`submit`, and from any terminal state, return `none=1`, `clear=alarm=0`. Cover all three observation IDs exactly.

- [ ] **Step 4: Implement the reward hook**

Use:

```text
depth 0 scout                    -> -scout_cost
depth 0 escape from weak active  -> +escape_reward
depth 0 escape from strong active-> -capture_cost
depth 0 submit from active       -> submit_reward
depth 1 escape -> reward by resulting terminal guard status
depth 1 submit -> submit_reward
any action from terminal -> 0
```

The depth-1 escape reward must use the **next** state's guard value so the transition hook's `guard_persistence` distribution determines expected escape value.

- [ ] **Step 5: Build the planning spec with the exact schedule**

For horizon 1:

```python
action_schedule = (("escape", "submit"),)
```

For horizon 2:

```python
action_schedule = (
    ("scout", "escape", "submit"),
    ("escape", "submit"),
)
```

Construct `RuntimePlanningDecisionModelSpec` with planning cells `(guard_cell, phase_cell)`, observation cells `(signal_cell,)`, shared belief model, four hidden states, three observations, scenario `discount`, fitted `beta`, and parameters containing the scenario semantics needed by the hooks.

- [ ] **Step 6: Run the complete legacy-reference and discriminator tests**

Run:

```bash
python3 -m unittest \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_reactive_matches_finite_reactive_on_all_fixture_cases_and_betas \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_generic_planning_matches_finite_pomdp_on_all_fixture_cases_and_betas \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_persistence_pair_separates_planning_from_non_lookahead_families \
  tests.test_narrative_held_out_model_comparison.NarrativeHeldOutModelComparisonTests.test_horizon_one_zero_fills_scout_and_output_binds_runtime_dispatch -v
```

Expected: all PASS. Do not relax tolerance to make reference equality pass; fix the translation/transition/reward semantics instead.

- [ ] **Step 7: Commit Planning GREEN**

```bash
git add narrative_dynamics/adapters/narrative_prison.py
git commit -m "feat: add generic narrative prison planning adapter"
```

---

### Task 5: Close the Preregistered Held-Out Comparison Loop

**Files:**
- Modify only if a test bug is found: `tests/test_narrative_held_out_model_comparison.py`
- Production should already be complete in `narrative_dynamics/adapters/narrative_prison.py`.

**Interfaces:**
- Consumes all three fresh model sources and the existing observational pipeline.
- Produces a valid `ModelComparisonReport` over exactly three frozen generic narrative candidates using existing train, selection-validation, and final-test roles.

- [ ] **Step 1: Run the full new test module**

```bash
python3 -m unittest tests.test_narrative_held_out_model_comparison -v
```

Expected: PASS with all source identity, horizon, reference-equivalence, persistence, seed-lineage, and preregistered comparison tests green.

- [ ] **Step 2: Run the entire Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests PASS. Record the exact total test count and runtime; do not predict the count in advance.

- [ ] **Step 3: Confirm the feature branch diff is within approved scope**

Run:

```bash
git diff --name-only c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765...HEAD
```

Allowed output is only:

```text
docs/superpowers/specs/2026-08-27-narrative-held-out-model-comparison-v1-design.md
docs/superpowers/plans/2026-08-27-narrative-held-out-model-comparison-v1.md
narrative_dynamics/adapters/narrative_prison.py
tests/test_narrative_held_out_model_comparison.py
```

If any other file appears, stop and review scope before continuing.

- [ ] **Step 4: Commit any final test-only correction separately**

Only if the integration test itself required correction without changing the approved scientific semantics:

```bash
git add tests/test_narrative_held_out_model_comparison.py
git commit -m "test: tighten held-out narrative comparison evidence"
```

Do not combine unrelated cleanup.

---

### Task 6: Exact-Head Proof, Review, and Integration Handoff

**Files:**
- No production changes expected.
- PR/roadmap metadata only after code verification.

**Interfaces:**
- Consumes final feature branch head.
- Produces authoritative proof evidence and a review-ready branch; roadmap completion still waits for merge plus post-merge proof.

- [ ] **Step 1: Push the exact final feature head and capture its SHA**

```bash
git rev-parse HEAD
git push origin work/narrative-held-out-model-comparison-v1
```

Record the exact SHA. Do not use a prior intermediate GREEN run as final evidence.

- [ ] **Step 2: Require `.github/workflows/proof.yml` success on that exact head**

Authoritative final GREEN requires all workflow stages success:

```text
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
Narrative story theorem tests
Narrative testimony theorem tests
```

The workflow's Python command is exactly:

```bash
python3 -m unittest discover -s tests -v
```

Verify the workflow run is `completed` / `success` and its `head_sha` equals the exact feature head from Step 1.

- [ ] **Step 3: Perform final scope/code review before claiming readiness**

Review these invariants explicitly:

```text
comparison core unchanged
observation protocol/release unchanged
fixture unchanged
generic runtime family implementations unchanged
all production dispatch goes through run_runtime_decision
reactive reference equality preserved
planning reference equality preserved
intentional persistence invariance preserved
planning persistence sensitivity demonstrated
all three candidates frozen after selection before final test
final test uses exactly FINAL_SEEDS
no P2 parameter recovery/identifiability/stochasticity added
```

- [ ] **Step 4: Open/update the PR with RED and exact-head GREEN evidence**

PR base: `proof/narrative-dynamics-v0`.

PR body must record:

```text
Integrated base: c9f1b97309ffd3c40fc1e2f6d9cc706dd715f765
Authoritative test-only RED: <exact RED SHA + proof run + expected missing-adapter failure>
Final feature GREEN: <exact final SHA + proof run + full Python count / OK>
Reference witnesses: generic Reactive == finite Reactive; generic Planning == finite POMDP across all committed fixture cases x beta grid
Held-out protocol: train -> selection-validation -> frozen candidates -> final-test, three families, FINAL_SEEDS=(301,302)
Out of scope: P2 stochasticity, parameter recovery, identifiability, log-score robustness
```

- [ ] **Step 5: Invoke finishing-development-branch only after exact-head GREEN**

Do not merge automatically as part of implementation. Use the finishing-development-branch workflow to present integration choices. If merge is chosen, require a new post-merge proof run on the exact new `proof/narrative-dynamics-v0` head before updating #27's held-out comparison checkbox to complete.
