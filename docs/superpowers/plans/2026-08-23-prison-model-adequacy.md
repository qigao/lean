# Prison POMDP Model-Adequacy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add proper-scoring losses, explicit finite-grid misspecification diagnosis, mixed parameter/scenario sensitivity, and final-test scenario/seed coverage diagnostics around the finite prison POMDP.

**Architecture:** Keep the prison adapter unchanged. Add a reusable metric-loss module, thread an optional loss object through existing calibration/validation entry points, and add a focused `narrative_dynamics.adequacy` package for misspecification, interaction, and final-test coverage reports. Every new report is immutable, manifest-backed, and compatible with `attest_report()`.

**Tech Stack:** Python 3 standard library, `unittest`, existing `narrative_dynamics` manifests/contracts/runner; Lean remains unchanged and is used only by the existing CI conformance/build/test gates.

**Spec:** `docs/superpowers/specs/2026-08-23-prison-model-adequacy-design.md`

## Global Constraints

- Python-only; do not modify any `.lean` file.
- Do not add dependencies.
- Do not add prison states, actions, parameters, or horizons.
- Preserve weighted-squared-error behavior when `loss` is omitted.
- Every production behavior is introduced by a failing test first.
- New reports must carry `ExperimentManifest` lineage and work with `attest_report()`.
- Coverage means declared scenario-stratum and seed-plan coverage only.

---

### Task 1: Proper-scoring loss objects and calibration plumbing

**Files:**
- Create: `narrative_dynamics/losses.py`
- Modify: `narrative_dynamics/calibration.py`
- Modify: `narrative_dynamics/validation.py`
- Modify: `narrative_dynamics/__init__.py`
- Test: `tests/test_model_adequacy_losses.py`

**Interfaces:**
- Produces: `CategoricalMetricGroup`, `WeightedSquaredErrorLoss`, `CategoricalBrierLoss`, `CategoricalLogLoss`, `MetricLoss`, `evaluate_metric_loss`, `metric_loss_identity`, `DEFAULT_METRIC_LOSS`.
- Extends: `calibrate_grid(..., loss: MetricLoss | None = None)` and all validation wrappers with the same optional argument.

- [ ] **Step 1: Write the failing proper-scoring tests**

```python
class AdequacyProbabilityModel:
    name = "adequacy-probability"
    def simulate(self, scenario, parameters, rng):
        p = float(parameters["p"])
        return ModelRun(events=(), outcome={"policy": {"a": p, "b": 1.0 - p}})


def policy_metrics(trace):
    return {
        "choice.a": float(trace.outcome["policy"]["a"]),
        "choice.b": float(trace.outcome["policy"]["b"]),
    }


def test_brier_and_log_losses_rank_true_probability_and_enter_manifest():
    group = CategoricalMetricGroup("choice", ("choice.a", "choice.b"))
    target = {"choice.a": 0.8, "choice.b": 0.2}
    for loss in (CategoricalBrierLoss((group,)), CategoricalLogLoss((group,))):
        report = calibrate_grid(
            runner=SimulationRunner(),
            model=AdequacyProbabilityModel(),
            scenario=Scenario(id="probabilities", payload={}),
            parameter_grid={"p": (0.2, 0.8)},
            seeds=(1,),
            extractor=policy_metrics,
            target=target,
            loss=loss,
        )
        self.assertEqual(report.best.parameters, (("p", 0.8),))
        self.assertEqual(
            report.manifest.inputs["loss"]["content_hash"],
            loss.content_hash,
        )
```

Add separate tests for duplicate keys across groups, invalid probability sums, positive target mass with zero predicted mass under log loss, and categorical loss rejecting `weights`.

- [ ] **Step 2: Run the targeted tests and verify RED**

Run:

```bash
python3 -m unittest tests.test_model_adequacy_losses -v
```

Expected: import/API failures for `narrative_dynamics.losses` and the missing `loss` parameter.

- [ ] **Step 3: Implement immutable loss identities**

Implement:

```python
@dataclass(frozen=True)
class CategoricalMetricGroup:
    name: str
    keys: tuple[str, ...]
    weight: float = 1.0


@dataclass(frozen=True)
class CategoricalBrierLoss:
    groups: tuple[CategoricalMetricGroup, ...]
    def __call__(self, observed, target, *, weights=None) -> float: ...
    def manifest_identity(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class CategoricalLogLoss:
    groups: tuple[CategoricalMetricGroup, ...]
    def __call__(self, observed, target, *, weights=None) -> float: ...
    def manifest_identity(self) -> dict[str, object]: ...
```

Target masses are normalized within each group. Observed group probabilities must be finite, non-negative, and sum to one within `1e-12`. Brier returns weighted squared probability error. Log loss returns weighted negative log score and raises when positive target mass receives zero predicted probability.

- [ ] **Step 4: Thread the optional loss through calibration and validation**

Use:

```python
selected_loss = DEFAULT_METRIC_LOSS if loss is None else loss
value = evaluate_metric_loss(
    selected_loss,
    metrics,
    target,
    weights=weights,
)
```

Replace hard-coded manifest loss dictionaries with `metric_loss_identity(selected_loss)`. Pass `loss` through `synthetic_recovery`, `validate_held_out`, `validate_acceptance_set_held_out`, `select_on_validation_suite`, `evaluate_on_final_test_suite`, and `local_sensitivity_report`.

- [ ] **Step 5: Verify targeted and complete Python GREEN**

Run:

```bash
python3 -m unittest tests.test_model_adequacy_losses -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass and existing default-loss tests remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/losses.py narrative_dynamics/calibration.py \
  narrative_dynamics/validation.py narrative_dynamics/__init__.py \
  tests/test_model_adequacy_losses.py
git commit -m "feat: add proper-scoring metric losses"
```

---

### Task 2: Shared-grid misspecification report

**Files:**
- Create: `narrative_dynamics/adequacy/__init__.py`
- Create: `narrative_dynamics/adequacy/misspecification.py`
- Modify: `narrative_dynamics/contracts.py`
- Test: `tests/test_model_misspecification.py`

**Interfaces:**
- Consumes: `HeldOutCase`, `validate_held_out`, `MetricLoss`, and calibration's canonical finite-grid candidate generation.
- Produces: `MisspecificationCandidate`, `MisspecificationReport`, `assess_misspecification`.

- [ ] **Step 1: Write failing adequate-versus-misspecified prison tests**

Build prison targets by running the existing policy-bound adapter and projecting the three initial action probabilities. Use `CategoricalBrierLoss` over `initial.scout`, `initial.escape`, and `initial.submit`.

Adequate suite: both cases generated with `beta=2.0`; grid `(0.5, 2.0, 5.0)`; zero thresholds; expect `adequate is True` and best beta `2.0`.

Misspecified suite: one case generated with beta `0.5`, another with beta `5.0`, while fitting one shared beta from the same grid; zero thresholds; expect `misspecified is True`, positive residual floor, and a manifest with `ExperimentStage.MODEL_MISSPECIFICATION`.

- [ ] **Step 2: Run targeted RED**

```bash
python3 -m unittest tests.test_model_misspecification -v
```

Expected: missing `narrative_dynamics.adequacy` APIs and experiment stage.

- [ ] **Step 3: Implement candidate ranking and thresholds**

```python
@dataclass(frozen=True)
class MisspecificationCandidate:
    parameters: tuple[tuple[str, float], ...]
    case_losses: tuple[tuple[str, float], ...]
    mean_loss: float
    worst_loss: float
    validation_manifest_hash: str


@dataclass(frozen=True)
class MisspecificationReport:
    suite_name: str
    best: MisspecificationCandidate
    ranking: tuple[MisspecificationCandidate, ...]
    max_mean_loss: float
    max_worst_loss: float
    adequate: bool
    manifest: ExperimentManifest

    @property
    def misspecified(self) -> bool:
        return not self.adequate
```

For each canonical parameter candidate, call `validate_held_out(..., loss=loss)` over the complete case tuple. Rank by `(mean_loss, worst_loss, parameters)`. Validate finite non-negative thresholds and require nonempty cases/grid.

- [ ] **Step 4: Add manifest lineage and report-artifact check**

Add `ExperimentStage.MODEL_MISSPECIFICATION`. Parent hashes are the candidate held-out validation manifests. Manifest inputs include suite name, canonical grid, loss identity, thresholds, and best parameters. Assert `attest_report(report).require_integrity() is report`.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_model_misspecification -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/adequacy narrative_dynamics/contracts.py \
  tests/test_model_misspecification.py
git commit -m "feat: diagnose finite-grid model misspecification"
```

---

### Task 3: Mixed parameter/scenario interaction sensitivity

**Files:**
- Create: `narrative_dynamics/adequacy/interaction.py`
- Modify: `narrative_dynamics/adequacy/__init__.py`
- Modify: `narrative_dynamics/contracts.py`
- Test: `tests/test_factor_interaction_sensitivity.py`

**Interfaces:**
- Produces: `FactorSource`, `LocalFactor`, `InteractionCorner`, `MetricInteraction`, `FactorInteractionReport`, `local_factor_interaction_report`.

- [ ] **Step 1: Write the failing prison interaction test**

```python
report = local_factor_interaction_report(
    runner=runner,
    model=bound_prison_model,
    scenario=prison_scenario("interaction", signal_accuracy=0.8),
    parameters={"beta": 2.0},
    seeds=(31, 32),
    extractor=prison_policy_metrics,
    first=LocalFactor(FactorSource.PARAMETER, "beta", 0.25),
    second=LocalFactor(FactorSource.SCENARIO, "signal_accuracy", 0.05),
)
self.assertNotEqual(report.interaction_map["initial.scout"], 0.0)
self.assertEqual(len(report.corners), 4)
self.assertIs(attest_report(report).require_integrity(), report)
```

Also test duplicate factors, missing fields, non-positive steps, and metric-schema drift.

- [ ] **Step 2: Run targeted RED**

```bash
python3 -m unittest tests.test_factor_interaction_sensitivity -v
```

Expected: missing interaction APIs and stage.

- [ ] **Step 3: Implement four-corner common-seed evaluation**

For each sign pair `(-1,-1), (-1,+1), (+1,-1), (+1,+1)`:

1. copy the canonical parameter mapping;
2. copy the canonical `Scenario.payload` when a scenario factor is present;
3. apply signed perturbations;
4. run the same ordered seed tuple;
5. aggregate a stable metric schema;
6. retain all run manifest hashes.

Compute each mixed finite difference with denominator `4 * first.step * second.step`.

- [ ] **Step 4: Add lineage, verify, and commit**

Add `ExperimentStage.INTERACTION_SENSITIVITY`. Parent hashes are every corner trace manifest. Manifest inputs include base scenario identity, base parameters, factors, seeds, and metric identity.

```bash
python3 -m unittest tests.test_factor_interaction_sensitivity -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/adequacy narrative_dynamics/contracts.py \
  tests/test_factor_interaction_sensitivity.py
git commit -m "feat: report local factor interactions"
```

---

### Task 4: Final-test scenario and seed coverage diagnostics

**Files:**
- Create: `narrative_dynamics/adequacy/coverage.py`
- Modify: `narrative_dynamics/adequacy/__init__.py`
- Modify: `narrative_dynamics/contracts.py`
- Test: `tests/test_final_test_coverage.py`

**Interfaces:**
- Produces: `CoverageBin`, `CoverageAxis`, `AxisCoverage`, `FinalTestCoverageReport`, `diagnose_final_test_coverage`.

- [ ] **Step 1: Write failing complete/incomplete coverage tests**

Create one actual `FinalTestReport` from the prison adapter and its `HeldOutSuite`. Declare axes for:

```python
CoverageAxis(
    name="prior",
    field="prior_weak",
    bins=(
        CoverageBin("low", 0.0, 0.34),
        CoverageBin("middle", 0.34, 0.67),
        CoverageBin("high", 0.67, 1.0, include_upper=True),
    ),
)
CoverageAxis(
    name="horizon",
    field="horizon",
    bins=(
        CoverageBin("one", 1.0, 1.0, include_upper=True),
        CoverageBin("two", 2.0, 2.0, include_upper=True),
    ),
)
```

A three-prior/two-horizon suite must report all bins represented. A reduced suite must list missing bins. Reused seeds across cases must be reported explicitly. Mismatched suite/report names or non-final roles must fail.

- [ ] **Step 2: Run targeted RED**

```bash
python3 -m unittest tests.test_final_test_coverage -v
```

Expected: missing coverage APIs and stage.

- [ ] **Step 3: Implement non-overlapping bin validation and diagnostics**

Intervals are lower-inclusive. Upper bounds are exclusive unless `include_upper=True`. Point bins require `minimum == maximum` and `include_upper=True`. Reject overlapping bins after sorting by minimum/maximum.

Match the final report's evaluated case names and scenario ids to the supplied suite. Count each case into exactly one bin or record it in `uncovered_cases`. Compute total seeds, unique seeds, and sorted reused seeds.

- [ ] **Step 4: Add manifest lineage, verify, and commit**

Add `ExperimentStage.FINAL_TEST_COVERAGE`. The final-test report manifest is the sole parent. Manifest inputs include the suite name, canonical axis definitions, and case/seed identities.

```bash
python3 -m unittest tests.test_final_test_coverage -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/adequacy narrative_dynamics/contracts.py \
  tests/test_final_test_coverage.py
git commit -m "feat: diagnose final-test scenario coverage"
```

---

### Task 5: Full review, CI, and integration

**Files:**
- Modify only if review finds a tested defect: files introduced above.
- Update: PR description and main PR #2 description.

- [ ] **Step 1: Run static syntax and focused adequacy suite**

```bash
python3 -m compileall -q narrative_dynamics tests
python3 -m unittest \
  tests.test_model_adequacy_losses \
  tests.test_model_misspecification \
  tests.test_factor_interaction_sensitivity \
  tests.test_final_test_coverage -v
```

- [ ] **Step 2: Run the complete Python suite**

```bash
python3 -m unittest discover -s tests -v
```

- [ ] **Step 3: Push and require feature-PR workflow GREEN**

Require all existing workflow steps:

```text
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
```

- [ ] **Step 4: Review for boundary overclaims**

Confirm documentation says:

- misspecification is relative to supplied grid, cases, extractor, loss, and thresholds;
- categorical scores evaluate declared probability groups only;
- interaction is a local finite-difference description, not causal identification;
- coverage is scenario-stratum and seed-plan coverage, not confidence coverage.

Add a new failing regression before any review-driven production fix.

- [ ] **Step 5: Fast-forward into `proof/narrative-dynamics-v0` and verify PR #2 merge context**

Do not force-update. Require the target head to pass the complete workflow again before reporting completion.
