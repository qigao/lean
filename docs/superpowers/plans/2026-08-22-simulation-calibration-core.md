# Simulation–Calibration Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-free Python core for deterministic simulation, metric aggregation, finite-grid calibration, synthetic recovery, and adaptation of the existing grounded-goal decision model.

**Architecture:** A trusted runner owns seeds and canonical trace metadata; pluggable models return only events and outcomes. Calibration observes models exclusively through traces and metric extractors. The existing grounded-goal softmax is wrapped, not duplicated.

**Tech Stack:** Python 3.12 standard library, `dataclasses`, `typing.Protocol`, `random`, `itertools`, `math`, and `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md`

## Global Constraints

- Add no third-party Python dependency.
- Keep existing root-level reference modules and their public behavior unchanged.
- Use explicit integer seeds and deterministic ordering.
- Reject non-finite numeric inputs at public boundaries.
- Continue using `python3 -m unittest discover -s tests -v` in CI.

---

### Task 1: Canonical simulation contracts and deterministic runner

**Files:**
- Create: `tests/test_simulation_runner.py`
- Create: `narrative_dynamics/__init__.py`
- Create: `narrative_dynamics/contracts.py`
- Create: `narrative_dynamics/simulation.py`

**Interfaces:**
- Produces: `Scenario`, `TraceEvent`, `ModelRun`, `SimulationTrace`, `SimulatorModel`, `SimulationRunner.run_once`, and `SimulationRunner.run_batch`.

- [ ] **Step 1: Write failing tests**

Test a tiny stochastic model that consumes the supplied RNG. Assert that identical seed/parameters produce equal traces, different seeds can differ, parameter metadata is sorted, and batch order equals seed order. Also assert rejection of empty seed lists and non-finite parameters.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_simulation_runner -v
```

Expected: import failure for `narrative_dynamics`.

- [ ] **Step 3: Implement contracts and runner**

Use this model boundary:

```python
class SimulatorModel(Protocol):
    name: str

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        ...
```

`SimulationRunner` constructs `random.Random(seed)`, canonicalizes parameters, invokes the model, and wraps its result in `SimulationTrace`.

- [ ] **Step 4: Verify GREEN**

Run the focused test and then the full Python suite.

- [ ] **Step 5: Commit**

Commit the test and implementation together after the observed RED run.

---

### Task 2: Metric aggregation and deterministic grid calibration

**Files:**
- Create: `tests/test_calibration.py`
- Create: `narrative_dynamics/metrics.py`
- Create: `narrative_dynamics/calibration.py`

**Interfaces:**
- Consumes: `SimulationTrace`, `SimulationRunner`, and `SimulatorModel`.
- Produces: `MetricExtractor`, `aggregate_metrics`, `weighted_squared_error`, `CandidateEvaluation`, `CalibrationResult`, and `calibrate_grid`.

- [ ] **Step 1: Write failing tests**

Use a toy model whose numeric outcome depends on one parameter plus seeded noise. Assert metric means, key consistency checks, deterministic Cartesian grid order, correct best candidate, finite losses, and lexicographic tie breaking.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_calibration -v
```

Expected: missing `narrative_dynamics.metrics` or `narrative_dynamics.calibration`.

- [ ] **Step 3: Implement metrics and grid search**

`aggregate_metrics` averages each metric over nonempty traces and rejects key drift. `calibrate_grid` expands sorted grid keys with `itertools.product`, executes every candidate over the supplied seeds, computes weighted squared error, and sorts by `(loss, parameters)`.

- [ ] **Step 4: Verify GREEN**

Run the focused test and full suite.

- [ ] **Step 5: Commit**

Commit after all tests pass.

---

### Task 3: Synthetic recovery

**Files:**
- Create: `tests/test_synthetic_recovery.py`
- Create: `narrative_dynamics/validation.py`

**Interfaces:**
- Consumes: `calibrate_grid` and simulation/metric interfaces.
- Produces: `SyntheticRecoveryReport` and `synthetic_recovery`.

- [ ] **Step 1: Write failing tests**

Assert recovery when the true parameter is in an identifiable grid, explicit non-recovery when metrics are non-identifying, and rejection when observation or calibration seed sets are empty.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_synthetic_recovery -v
```

Expected: missing `narrative_dynamics.validation`.

- [ ] **Step 3: Implement synthetic recovery**

Generate target metrics with true parameters and observation seeds, then invoke `calibrate_grid` with separate calibration seeds. Compare canonical true parameters with the best candidate and preserve the full ranked calibration result.

- [ ] **Step 4: Verify GREEN**

Run the focused test and full suite.

- [ ] **Step 5: Commit**

Commit after all tests pass.

---

### Task 4: Existing grounded-goal model adapter and recovery experiment

**Files:**
- Create: `tests/test_grounded_goal_model.py`
- Create: `narrative_dynamics/adapters/__init__.py`
- Create: `narrative_dynamics/adapters/grounded_goal.py`

**Interfaces:**
- Consumes: root-level `grounded_goal_softmax.grounded_goal_choice_probabilities` and simulation contracts.
- Produces: `GroundedGoalScenario`, `GroundedGoalDecisionModel`, and `selected_goal_metrics`.

- [ ] **Step 1: Write failing tests**

Construct the existing three-hypothesis grounded space used by current tests. Assert that the adapter records a normalized policy, seeded selected goal, canonical trace metadata, and rejects empty goal maps or invalid beta. Run a recovery experiment over a finite beta grid using choice-frequency metrics.

- [ ] **Step 2: Verify RED**

Run:

```bash
python3 -m unittest tests.test_grounded_goal_model -v
```

Expected: missing adapter module.

- [ ] **Step 3: Implement adapter**

Call the existing grounded goal choice function, sample from sorted goal labels using the supplied RNG, emit `policy_computed` and `goal_selected` events, and return the selected goal plus policy in the outcome. Do not reproduce the score formula.

- [ ] **Step 4: Verify GREEN**

Run the focused test, then:

```bash
python3 -m unittest discover -s tests -v
```

- [ ] **Step 5: Commit and update PR description**

Record the simulation–calibration boundary, verified head, test count, and next target: posterior uncertainty and held-out scenario validation.
