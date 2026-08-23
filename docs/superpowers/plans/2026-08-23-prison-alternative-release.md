# Prison Alternative Model and Witnessed Protocol Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently specified one-parameter reactive prison model, a versioned synthetic observational fixture, an explicit train→selection→final evaluation path, and an externally witnessed protocol-release gate without changing the frozen prison POMDP or any Lean source.

**Architecture:** Keep the current `finite-prison-pomdp` untouched. Add a separate `finite-prison-reactive` adapter that shares only the scenario surface and `initial_policy` output shape; add one shared categorical metric extractor; compose multi-case training from existing single-case grid calibration; represent release and witness evidence as canonical immutable identities; require a caller-supplied verifier before delegating to the existing `compare_models_on_final_partition` preflight and execution path.

**Tech Stack:** Python 3 standard library only, `unittest`, existing `narrative_dynamics` runtime/contracts/manifests/schema validation/subprocess execution, GitHub Actions `proof` workflow, Lean 4 only as an unchanged CI gate.

**Spec:** `docs/superpowers/specs/2026-08-23-prison-alternative-release-design.md`

## Global Constraints

- Implementation scope is Python-only; no `.lean` source file may change.
- Existing `finite-prison-pomdp` state, action, observation, reward, transition, horizon, policy, parameter, contract, and implementation semantics remain frozen.
- The reactive alternative has exactly one fitted parameter: `beta > 0`.
- The reactive policy must not import or call `narrative_dynamics.adapters.prison_pomdp` or reuse its posterior/planning helpers.
- Reactive cue semantics are exactly those in the approved spec: direct prior-based escape value, `cue_strength = 2 * signal_accuracy - 1`, `cue_scale = (escape_reward + capture_cost) / 2`, additive/subtractive clear/alarm cue values, and no posterior state probability.
- `guard_persistence` may affect realized post-scout environment outcomes but must not affect the reactive model's policy calculation.
- Do not add cryptographic dependencies or implement Ed25519, X.509, RFC 3161, Rekor, or another signature/timestamp protocol in core code.
- A syntactically valid witness receipt is not trusted until a caller-supplied verifier accepts it.
- The committed observational fixture is explicitly synthetic, non-empirical, and non-population-representative.
- Train, selection-validation, and final-test data remain role-separated. Training must not call final-test APIs; final comparison must not recalibrate or reselect parameters.
- Final comparison must continue to use existing target, extractor, loss, seed, candidate, runtime identity, and mutable-source preflight checks.
- Production adapter smoke tests must continue through `TrustedExecutionPolicy`, measured implementation pinning, `SubprocessModel`, all four executable schemas, result artifacts, and the canonical `SimulationRunner`.
- Every production change follows RED → verify RED → minimal GREEN → verify GREEN → commit.

---

## File Structure

Create or modify only the focused units below unless a test proves a narrower compatibility change is necessary.

```text
narrative_dynamics/adapters/prison_metrics.py
    Shared `prison_initial_action_metrics` extractor only.

narrative_dynamics/adapters/prison_reactive.py
    Independent reactive simulator, deterministic policy math, stochastic episode execution,
    model contract, subprocess source.

narrative_dynamics/observations/training.py
    Multi-case training-grid aggregation composed from existing `calibrate_grid` calls.

narrative_dynamics/observations/release.py
    ProtocolRelease / WitnessReceipt / VerifiedProtocolRelease identities,
    external-verifier gate, released comparison wrapper/report.

narrative_dynamics/contracts.py
    Add only `TRAINING_TARGET_FIT` and `RELEASED_MODEL_COMPARISON` experiment stages.

narrative_dynamics/observations/__init__.py
narrative_dynamics/__init__.py
    Export observational training/release APIs following existing package convention.
    Do not export model-specific reactive adapter symbols from package root because the
    existing prison POMDP is also module-scoped rather than root-exported.

fixtures/observations/prison_initial_choice_v1.json
    Fixed synthetic record-oriented dataset with declared content hash.

tests/test_prison_reactive_model.py
    Reactive equations, independence from POMDP implementation, replay, schema/process pinning.

tests/test_observational_training_fit.py
    Train-only multi-case finite-grid aggregation and manifest lineage.

tests/test_protocol_release.py
    Canonical release/receipt identity and trusted verifier boundary.

tests/test_observational_fixture_pipeline.py
    Full fixture → train → selection → freeze → preregister → witness → final path.
```

---

### Task 1: Shared Initial-Action Metric Extractor

**Files:**
- Create: `narrative_dynamics/adapters/prison_metrics.py`
- Create: `tests/test_prison_reactive_model.py`

**Interfaces:**
- Consumes: `SimulationTrace.outcome["initial_policy"]`
- Produces: `prison_initial_action_metrics(trace: SimulationTrace) -> dict[str, float]`
- Stable callable metadata: `prison_initial_action_metrics.version = "1.0.0"`

- [ ] **Step 1: Write the failing extractor test**

Add this first test to `tests/test_prison_reactive_model.py` before creating the module:

```python
from __future__ import annotations

import math
import unittest

from narrative_dynamics.contracts import SimulationTrace


class PrisonInitialActionMetricTests(unittest.TestCase):
    def _trace(self, policy):
        return SimulationTrace(
            model_name="fixture-model",
            scenario_id="fixture-scenario",
            parameters=(("beta", 1.0),),
            seed=1,
            events=(),
            outcome={"initial_policy": policy},
        )

    def test_shared_extractor_returns_exact_three_initial_action_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        metrics = prison_initial_action_metrics(
            self._trace({"scout": 0.25, "escape": 0.5, "submit": 0.25})
        )
        self.assertEqual(
            metrics,
            {
                "initial.scout": 0.25,
                "initial.escape": 0.5,
                "initial.submit": 0.25,
            },
        )
        self.assertEqual(prison_initial_action_metrics.version, "1.0.0")

    def test_shared_extractor_rejects_missing_or_non_finite_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import (
            prison_initial_action_metrics,
        )

        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": 0.5, "escape": 0.5})
            )
        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": math.nan, "escape": 0.5, "submit": 0.5})
            )
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
python3 -m unittest tests.test_prison_reactive_model.PrisonInitialActionMetricTests -v
```

Expected: import failure for `narrative_dynamics.adapters.prison_metrics`.

- [ ] **Step 3: Implement the minimal extractor**

Create `narrative_dynamics/adapters/prison_metrics.py` with exactly one public behavior:

```python
from __future__ import annotations

from collections.abc import Mapping
import math

from narrative_dynamics.contracts import SimulationTrace


_ACTIONS = ("scout", "escape", "submit")


def prison_initial_action_metrics(trace: SimulationTrace) -> dict[str, float]:
    policy = trace.outcome.get("initial_policy")
    if not isinstance(policy, Mapping) or set(policy) != set(_ACTIONS):
        raise ValueError("prison trace must contain exactly the three initial-policy actions")
    result: dict[str, float] = {}
    for action in _ACTIONS:
        raw = policy[action]
        if isinstance(raw, bool):
            raise ValueError("prison initial-policy coordinates must be numeric")
        try:
            value = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("prison initial-policy coordinates must be numeric") from error
        if not math.isfinite(value):
            raise ValueError("prison initial-policy coordinates must be finite")
        result[f"initial.{action}"] = value
    return result


prison_initial_action_metrics.version = "1.0.0"

__all__ = ["prison_initial_action_metrics"]
```

Do not enforce normalization here; `CategoricalBrierLoss` / `CategoricalLogLoss` already own that trust boundary.

- [ ] **Step 4: Verify GREEN**

```bash
python3 -m unittest tests.test_prison_reactive_model.PrisonInitialActionMetricTests -v
```

Expected: all extractor tests pass.

- [ ] **Step 5: Commit**

```bash
git add narrative_dynamics/adapters/prison_metrics.py tests/test_prison_reactive_model.py
git commit -m "feat: add shared prison initial-action metrics"
```

---

### Task 2: Independent Reactive Prison Adapter

**Files:**
- Create: `narrative_dynamics/adapters/prison_reactive.py`
- Modify: `tests/test_prison_reactive_model.py`

**Interfaces:**
- Produces: `FinitePrisonReactiveModel`
- Produces: `create_prison_reactive_model() -> FinitePrisonReactiveModel`
- Produces: `prison_reactive_contract() -> ModelContract`
- Produces: `prison_reactive_source(*, limits: ProcessLimits | None = None) -> SubprocessModel`
- Model identity:
  - name: `finite-prison-reactive`
  - version: `1.0.0`
  - implementation revision: `prison-reactive-v1`

- [ ] **Step 1: Add failing behavioral tests**

Append tests that lock the independent equations before implementation:

```python
from narrative_dynamics.attestation import RepositoryIdentity, measure_implementation
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.execution_policy import TrustedExecutionPolicy, TrustedModelPin
from narrative_dynamics.process_execution import ProcessLimits
from narrative_dynamics.simulation import SimulationRunner


def prison_scenario(
    scenario_id: str,
    *,
    prior_weak: float = 0.5,
    signal_accuracy: float = 0.75,
    guard_persistence: float = 0.9,
    horizon: int = 2,
) -> Scenario:
    return Scenario(
        id=scenario_id,
        payload={
            "prior_weak": prior_weak,
            "signal_accuracy": signal_accuracy,
            "guard_persistence": guard_persistence,
            "escape_reward": 8.0,
            "capture_cost": 10.0,
            "submit_reward": 1.0,
            "scout_cost": 0.25,
            "discount": 0.95,
            "horizon": horizon,
        },
    )


class PrisonReactiveModelTests(unittest.TestCase):
    def setUp(self):
        self.runner = SimulationRunner(
            repository_identity=RepositoryIdentity(provider="test")
        )

    def test_reactive_values_follow_declared_cue_rule_and_replay_seed(self):
        from narrative_dynamics.adapters.prison_reactive import (
            create_prison_reactive_model,
        )

        model = create_prison_reactive_model()
        scenario = prison_scenario("reactive-equations")
        first = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)
        replay = self.runner.run_once(model, scenario, {"beta": 2.0}, seed=17)

        self.assertEqual(first, replay)
        self.assertAlmostEqual(first.outcome["initial_values"]["escape"], -1.0)
        self.assertAlmostEqual(first.outcome["cue_values"]["clear"]["escape"], 3.5)
        self.assertAlmostEqual(first.outcome["cue_values"]["alarm"]["escape"], -5.5)
        self.assertAlmostEqual(sum(first.outcome["initial_policy"].values()), 1.0)
        self.assertNotIn("posterior_weak", first.outcome)

    def test_guard_persistence_does_not_change_reactive_policy(self):
        from narrative_dynamics.adapters.prison_reactive import (
            create_prison_reactive_model,
        )

        model = create_prison_reactive_model()
        low = self.runner.run_once(
            model,
            prison_scenario("low-persistence", guard_persistence=0.1),
            {"beta": 2.0},
            seed=11,
        )
        high = self.runner.run_once(
            model,
            prison_scenario("high-persistence", guard_persistence=0.9),
            {"beta": 2.0},
            seed=11,
        )
        self.assertEqual(low.outcome["initial_policy"], high.outcome["initial_policy"])
        self.assertEqual(low.outcome["initial_values"], high.outcome["initial_values"])

    def test_reactive_module_does_not_import_prison_pomdp(self):
        import inspect
        import narrative_dynamics.adapters.prison_reactive as reactive

        source = inspect.getsource(reactive)
        self.assertNotIn("adapters.prison_pomdp", source)
        self.assertNotIn("_posterior_weak", source)
        self.assertNotIn("_future_weak_probability", source)
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_prison_reactive_model.PrisonReactiveModelTests -v
```

Expected: module import failure for `prison_reactive`.

- [ ] **Step 3: Implement exact reactive mathematics and episode execution**

In `prison_reactive.py`, independently implement:

```python
V_escape = p * escape_reward - (1.0 - p) * capture_cost
V_submit = submit_reward
cue_strength = 2.0 * signal_accuracy - 1.0
cue_scale = (escape_reward + capture_cost) / 2.0
V_clear_escape = V_escape + cue_strength * cue_scale
V_alarm_escape = V_escape - cue_strength * cue_scale
```

Use a model-local max-shifted stable softmax. Normalize by placing floating residual on the largest mass coordinate, matching the numerical safety requirement but not importing baseline helpers.

For horizon 2:

```python
p_clear = (
    prior_weak * signal_accuracy
    + (1.0 - prior_weak) * (1.0 - signal_accuracy)
)

clear_policy = softmax({"escape": V_clear_escape, "submit": V_submit}, beta)
alarm_policy = softmax({"escape": V_alarm_escape, "submit": V_submit}, beta)

V_scout = (
    -scout_cost
    + discount * (
        p_clear * sum(clear_policy[a] * clear_values[a] for a in TERMINAL_ACTIONS)
        + (1.0 - p_clear)
        * sum(alarm_policy[a] * alarm_values[a] for a in TERMINAL_ACTIONS)
    )
)
```

The initial policy is softmax over `scout/escape/submit`; horizon 1 sets scout probability to exactly `0.0` and softmaxes only `escape/submit`.

Episode realization may sample latent guard weakness and signal using the scenario probabilities. If scouting then escaping, use `guard_persistence` only to realize the future environment state/outcome, never in policy/value functions.

Expose at minimum these trusted outcome fields:

```text
initial_policy
initial_values
cue_values
initial_action
terminal_action
signal
escaped
utility
steps
```

- [ ] **Step 4: Add and verify four-schema contract**

Mirror the baseline scenario and parameter schema ranges, but define reactive-specific event/outcome schemas. Required event kinds:

```text
initial_decision
observation_received
terminal_decision
episode_ended
```

No `belief_state` event is required because the alternative does not claim an internal Bayesian belief representation.

The `outcome_schema` must include the exact `initial_policy` three-action shape and `cue_values` clear/alarm terminal values.

- [ ] **Step 5: Add production subprocess/pin RED before source implementation is considered complete**

Add:

```python
    def test_reactive_production_source_is_pinned_isolated_and_schema_attested(self):
        from narrative_dynamics.adapters.prison_reactive import (
            prison_reactive_contract,
            prison_reactive_source,
        )

        source = prison_reactive_source(
            limits=ProcessLimits(
                timeout_seconds=3.0,
                max_output_bytes=64 * 1024,
                max_trace_bytes=128 * 1024,
            )
        )
        contract = prison_reactive_contract()
        policy = TrustedExecutionPolicy(
            name="reactive-test-policy",
            version="1",
            pins=(TrustedModelPin(
                model_name=source.name,
                declared_contract_hash=contract.content_hash,
                expected_implementation_hash=measure_implementation(source).content_hash,
            ),),
        )
        bound = policy.bind(source, contract=contract)
        trace = self.runner.run_once(
            bound,
            prison_scenario("reactive-production"),
            {"beta": 2.0},
            seed=23,
        )
        self.assertTrue(trace.execution.isolated)
        for boundary in ("parameters", "scenario", "events", "outcome"):
            self.assertEqual(
                trace.manifest.inputs["schema_validation"][boundary]["status"],
                "validated",
            )
        self.assertEqual(
            trace.manifest.inputs["model"]["implementation_attestation"]["verification"],
            "matched",
        )
        self.assertIn("result_artifact", trace.manifest.inputs)
```

Run it once before implementing/finalizing `prison_reactive_source`; expected RED is a missing/incorrect source or contract boundary.

- [ ] **Step 6: Implement `prison_reactive_source` and verify GREEN**

Use:

```python
SubprocessModel(
    name="finite-prison-reactive",
    factory="narrative_dynamics.adapters.prison_reactive:create_prison_reactive_model",
    version="1.0.0",
    implementation_revision="prison-reactive-v1",
    limits=ProcessLimits() if limits is None else limits,
)
```

Run:

```bash
python3 -m unittest tests.test_prison_reactive_model -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/adapters/prison_reactive.py tests/test_prison_reactive_model.py
git commit -m "feat: add independent reactive prison model"
```

---

### Task 3: Train-Only Multi-Case Grid Fit

**Files:**
- Create: `narrative_dynamics/observations/training.py`
- Modify: `narrative_dynamics/contracts.py`
- Create: `tests/test_observational_training_fit.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class TrainingCaseFit:
    name: str
    loss: float
    calibration_manifest_hash: str

@dataclass(frozen=True)
class TrainingCandidateFit:
    parameters: tuple[tuple[str, float], ...]
    cases: tuple[TrainingCaseFit, ...]
    mean_loss: float
    worst_loss: float

@dataclass(frozen=True)
class TrainingFitReport:
    target_report_hash: str
    ranking: tuple[TrainingCandidateFit, ...]
    manifest: ExperimentManifest

    @property
    def candidate_parameters(self) -> tuple[tuple[tuple[str, float], ...], ...]: ...


def fit_training_target_grid(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    target_report: TargetConstructionReport,
    parameter_grid: Mapping[str, Iterable[float]],
    simulation_seeds: Iterable[int],
    extractor: object,
    loss: MetricLoss,
) -> TrainingFitReport: ...
```

- [ ] **Step 1: Write RED tests**

Use a simple record-oriented `ObservationDataset` with two train cases. Test these obligations:

```python
class TrainingTargetFitTests(unittest.TestCase):
    def test_train_fit_ranks_one_shared_grid_across_all_train_cases(self):
        report = fit_training_target_grid(
            runner=SimulationRunner(),
            model=ModelFactory(
                name=ProbabilityModel.name,
                create=create_probability_model,
                version="1.0.0",
                implementation_revision="training-test-v1",
            ),
            target_report=train_targets(),
            parameter_grid={"p": (0.2, 0.5, 0.8)},
            simulation_seeds=(101, 102),
            extractor=policy_metrics,
            loss=brier_loss(),
        )
        self.assertIs(report.manifest.stage, ExperimentStage.TRAINING_TARGET_FIT)
        self.assertEqual(len(report.ranking), 3)
        self.assertEqual(
            set(report.candidate_parameters),
            {
                (("p", 0.2),),
                (("p", 0.5),),
                (("p", 0.8),),
            },
        )
        self.assertTrue(all(len(candidate.cases) == 2 for candidate in report.ranking))
        self.assertIs(attest_report(report).require_integrity(), report)

    def test_train_fit_rejects_selection_or_final_targets_before_execution(self):
        model = CountingModel()
        for role in (
            ObservationPartitionRole.SELECTION_VALIDATION,
            ObservationPartitionRole.FINAL_TEST,
        ):
            with self.subTest(role=role):
                before = model.calls
                with self.assertRaises(ValueError):
                    fit_training_target_grid(
                        runner=SimulationRunner(),
                        model=model,
                        target_report=targets_for(role),
                        parameter_grid={"p": (0.2, 0.8)},
                        simulation_seeds=(101,),
                        extractor=policy_metrics,
                        loss=brier_loss(),
                    )
                self.assertEqual(model.calls, before)
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_observational_training_fit -v
```

Expected: missing `training` API / missing `ExperimentStage.TRAINING_TARGET_FIT`.

- [ ] **Step 3: Add the dedicated stage**

In `ExperimentStage`, add exactly:

```python
TRAINING_TARGET_FIT = "training_target_fit"
```

Do not reuse `HELD_OUT_VALIDATION`, `SELECTION_VALIDATION`, or `FINAL_TEST`.

- [ ] **Step 4: Implement by composing existing single-case calibration**

Do not duplicate finite-grid expansion. For each `target_report.cases` entry, call existing `calibrate_grid` with the same `parameter_grid`, same `simulation_seeds`, same extractor, same loss, and that case's scenario/target.

Then require every per-case calibration to expose the same candidate parameter set. Aggregate candidate losses by parameter tuple:

```python
mean_loss = fmean(case.loss for case in cases)
worst_loss = max(case.loss for case in cases)
```

Sort by:

```python
(mean_loss, worst_loss, parameters)
```

The `TRAINING_TARGET_FIT` manifest must record:

```text
model identity
target_report.content_hash
training seed tuple
extractor identity
loss identity
canonical parameter-grid candidate set
per-case target names/scenario identities
```

and use every child `GRID_CALIBRATION` manifest as a parent.

- [ ] **Step 5: Verify GREEN and report attestation**

```bash
python3 -m unittest tests.test_observational_training_fit -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/contracts.py narrative_dynamics/observations/training.py tests/test_observational_training_fit.py
git commit -m "feat: add train-only observational grid fitting"
```

---

### Task 4: Commit the Versioned Synthetic Prison Fixture

**Files:**
- Create: `fixtures/observations/prison_initial_choice_v1.json`
- Modify/Create: `tests/test_observational_fixture_pipeline.py`

**Interfaces:**
- Consumes: existing `load_observation_dataset`
- Produces no new runtime API.

- [ ] **Step 1: Write RED fixture test before the JSON file exists**

```python
from pathlib import Path
import unittest

from narrative_dynamics.observations import (
    ObservationPartitionRole,
    load_observation_dataset,
)

FIXTURE = Path("fixtures/observations/prison_initial_choice_v1.json")


class PrisonObservationFixtureTests(unittest.TestCase):
    def test_committed_fixture_is_synthetic_versioned_and_role_complete(self):
        dataset = load_observation_dataset(FIXTURE)
        self.assertEqual(dataset.name, "prison-initial-choice")
        self.assertEqual(dataset.version, "1.0.0")
        self.assertEqual(dataset.source["kind"], "synthetic_fixture")
        self.assertEqual(dataset.source["purpose"], "protocol_integration_test")
        self.assertFalse(dataset.provenance["empirical_human_data"])
        self.assertFalse(dataset.provenance["population_representative"])
        self.assertEqual(
            {partition.role for partition in dataset.partitions},
            set(ObservationPartitionRole),
        )
        for partition in dataset.partitions:
            for record in partition.records:
                self.assertEqual(set(record.counts), {"scout", "escape", "submit"})
```

Run and verify RED: file-not-found.

- [ ] **Step 2: Build the fixture from fixed constants, not model output**

Use these fixed records. All reward/cost fields remain `escape_reward=8.0`, `capture_cost=10.0`, `submit_reward=1.0`, `scout_cost=0.25`, `discount=0.95`.

```text
TRAIN
train-1: prior=.45, accuracy=.80, persistence=.80, horizon=2, counts scout=52 escape=28 submit=20
train-2: prior=.70, accuracy=.70, persistence=.50, horizon=1, counts scout=0  escape=61 submit=39

SELECTION_VALIDATION
selection-1: prior=.35, accuracy=.85, persistence=.75, horizon=2, counts scout=58 escape=17 submit=25
selection-2: prior=.75, accuracy=.65, persistence=.50, horizon=1, counts scout=0  escape=69 submit=31

FINAL_TEST
final-1: prior=.80, accuracy=.75, persistence=.20, horizon=2, counts scout=18 escape=66 submit=16
final-2: prior=.80, accuracy=.75, persistence=.90, horizon=2, counts scout=43 escape=45 submit=12
final-3: prior=.25, accuracy=.90, persistence=.60, horizon=1, counts scout=0  escape=14 submit=86
```

The `final-1` / `final-2` pair deliberately changes only persistence and scenario/record IDs so the frozen planning vs reactive contrast is exercised.

- [ ] **Step 3: Generate the declared content hash using existing runtime identity code**

Use a one-off repository-root script, not model simulation:

```python
import json
from pathlib import Path
from narrative_dynamics.contracts import Scenario
from narrative_dynamics.observations import (
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
)

# Construct exactly the fixed records above.
dataset = ObservationDataset(...)
payload = dataset.to_payload()
Path("fixtures/observations/prison_initial_choice_v1.json").parent.mkdir(
    parents=True, exist_ok=True
)
Path("fixtures/observations/prison_initial_choice_v1.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
```

The test must load the committed JSON through `load_observation_dataset`, which rechecks the declared hash.

- [ ] **Step 4: Add anti-tautology assertions**

Assert exact committed counts in the test, e.g.:

```python
final = dataset.partition(ObservationPartitionRole.FINAL_TEST)
self.assertEqual(final.records[0].count_map, {"scout": 18, "escape": 66, "submit": 16})
```

Do not call either simulator to produce expected counts.

- [ ] **Step 5: Verify GREEN**

```bash
python3 -m unittest tests.test_observational_fixture_pipeline.PrisonObservationFixtureTests -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add fixtures/observations/prison_initial_choice_v1.json tests/test_observational_fixture_pipeline.py
git commit -m "test: add versioned synthetic prison observations"
```

---

### Task 5: Canonical Protocol Release and External Witness Verification

**Files:**
- Create: `narrative_dynamics/observations/release.py`
- Create: `tests/test_protocol_release.py`

**Interfaces:**

```python
PROTOCOL_RELEASE_SCHEMA_VERSION = 1
WITNESS_RECEIPT_SCHEMA_VERSION = 1

class ProtocolReleaseVerificationError(ValueError): ...

@dataclass(frozen=True)
class ProtocolRelease:
    name: str
    version: str
    protocol_hash: str
    dataset_hash: str
    target_spec_hash: str
    candidate_hashes: tuple[str, ...]
    source_revision: Mapping[str, object]
    declared_content_hash: str
    schema_version: int = 1

    @classmethod
    def create(..., protocol: PreregisteredEvaluationProtocol, source_revision: Mapping[str, object]) -> "ProtocolRelease": ...
    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolRelease": ...
    def to_payload(self) -> dict[str, object]: ...
    def identity_payload(self) -> dict[str, object]: ...
    @property
    def content_hash(self) -> str: ...

@dataclass(frozen=True)
class WitnessReceipt:
    provider: str
    authority: str
    subject_hash: str
    reference: str
    claimed_at: str
    proof: Mapping[str, object]
    declared_content_hash: str
    schema_version: int = 1

    @classmethod
    def create(...) -> "WitnessReceipt": ...
    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "WitnessReceipt": ...
    @property
    def content_hash(self) -> str: ...

@dataclass(frozen=True)
class VerifiedProtocolRelease:
    release_hash: str
    protocol_hash: str
    verifier_identity: Mapping[str, object]
    verified_receipt_hashes: tuple[str, ...]
    status: str = "verified"

    def require_matches(self, protocol: PreregisteredEvaluationProtocol) -> None: ...
    @property
    def content_hash(self) -> str: ...


def verify_protocol_release(
    release: ProtocolRelease,
    *,
    protocol: PreregisteredEvaluationProtocol,
    receipts: tuple[WitnessReceipt, ...],
    verifier: object,
) -> VerifiedProtocolRelease: ...
```

Verifier structural contract:

```python
verifier.name: str
verifier.version: str
verifier.verify(release: ProtocolRelease, receipt: WitnessReceipt) -> bool
```

- [ ] **Step 1: Write release/receipt identity RED tests**

Test exact round-trip and stale hash rejection:

```python
class ProtocolReleaseIdentityTests(unittest.TestCase):
    def test_release_and_receipt_roundtrip_recompute_declared_hashes(self):
        protocol = protocol_fixture()
        release = ProtocolRelease.create(
            name="prison-comparison-release",
            version="1",
            protocol=protocol,
            source_revision={
                "kind": "declared_revision",
                "repository": "qigao/lean",
                "revision": "prison-alternative-release-v1",
            },
        )
        self.assertEqual(
            ProtocolRelease.from_payload(release.to_payload()),
            release,
        )
        receipt = WitnessReceipt.create(
            provider="test-fixture",
            authority="unit-test",
            subject_hash=release.content_hash,
            reference="fixture-receipt-1",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        self.assertEqual(WitnessReceipt.from_payload(receipt.to_payload()), receipt)

        forged = dict(release.to_payload())
        forged["version"] = "2"
        with self.assertRaises(ValueError):
            ProtocolRelease.from_payload(forged)
```

Verify RED: import/module missing.

- [ ] **Step 2: Implement canonical immutable release and receipt values**

Reuse `stable_content_hash`; use strict non-empty text and `sha256:<64 lowercase hex>` validation. Freeze `source_revision` and `proof` recursively using existing canonical mapping helpers rather than mutable dict retention.

`declared_content_hash` is excluded from `identity_payload()` and must equal `stable_content_hash(identity_payload())` in `__post_init__`.

Candidate hashes are canonicalized as the protocol's ordered `tuple(candidate.content_hash for candidate in protocol.candidates)`; do not accept caller-supplied candidate order in `ProtocolRelease.create`.

- [ ] **Step 3: Write trusted-verifier RED tests**

Test-only verifier:

```python
class FixtureWitnessVerifier:
    name = "fixture-witness-verifier"
    version = "1"

    def __init__(self, *, accept: bool = True):
        self.accept = accept
        self.calls = 0

    def verify(self, release, receipt):
        self.calls += 1
        return (
            self.accept
            and receipt.provider == "test-fixture"
            and receipt.proof == {"nonce": "release-v1"}
        )
```

Required tests:

```python
    def test_verification_requires_at_least_one_external_receipt_acceptance(self):
        verifier = FixtureWitnessVerifier(accept=False)
        with self.assertRaises(ProtocolReleaseVerificationError):
            verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(receipt,),
                verifier=verifier,
            )

    def test_subject_and_duplicate_receipt_drift_fail_before_verifier_call(self):
        verifier = FixtureWitnessVerifier()
        wrong_subject = WitnessReceipt.create(
            provider="test-fixture",
            authority="unit-test",
            subject_hash=stable_content_hash({"other": True}),
            reference="wrong",
            claimed_at="2026-08-23T00:00:00Z",
            proof={"nonce": "release-v1"},
        )
        with self.assertRaises(ProtocolReleaseVerificationError):
            verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(wrong_subject,),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 0)

        with self.assertRaises(ProtocolReleaseVerificationError):
            verify_protocol_release(
                release,
                protocol=protocol,
                receipts=(receipt, receipt),
                verifier=verifier,
            )
        self.assertEqual(verifier.calls, 0)
```

- [ ] **Step 4: Implement pre-verifier integrity checks and verification**

Before any verifier call, require:

```text
release.protocol_hash == protocol.content_hash
release.dataset_hash == protocol.dataset_hash
release.target_spec_hash == protocol.target_spec_hash
release.candidate_hashes == tuple(candidate.content_hash for candidate in protocol.candidates)
every receipt.subject_hash == release.content_hash
receipt identities unique
```

Call `verifier.verify` once per receipt. Convert verifier exceptions into `ProtocolReleaseVerificationError` with the original exception chained. Collect only accepted receipt hashes. If zero accepted, raise.

Record verifier identity with `component_identity(verifier)` and return a `VerifiedProtocolRelease` whose `content_hash` covers release hash, protocol hash, verifier identity, verified receipt hashes, and status.

- [ ] **Step 5: Verify GREEN**

```bash
python3 -m unittest tests.test_protocol_release -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/observations/release.py tests/test_protocol_release.py
git commit -m "feat: add externally witnessed protocol release"
```

---

### Task 6: Release-Gated Final Comparison

**Files:**
- Modify: `narrative_dynamics/observations/release.py`
- Modify: `narrative_dynamics/contracts.py`
- Modify: `tests/test_protocol_release.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ReleasedModelComparisonReport:
    release_hash: str
    verification_hash: str
    comparison: ModelComparisonReport
    manifest: ExperimentManifest

    @property
    def best(self) -> ModelComparisonEntry: ...
    @property
    def entry_map(self): ...


def compare_released_models(
    *,
    runner: SimulationRunner,
    verified_release: VerifiedProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    models: tuple[ComparisonModel, ...],
    target_set: TargetConstructionReport,
    extractor: object,
    loss: MetricLoss,
) -> ReleasedModelComparisonReport: ...
```

- [ ] **Step 1: Write RED that raw release cannot unlock execution**

Use two counting models and assert zero calls:

```python
    def test_raw_or_mismatched_release_fails_before_model_execution(self):
        before = baseline_model.calls + alternative_model.calls
        with self.assertRaises(TypeError):
            compare_released_models(
                runner=SimulationRunner(),
                verified_release=release,  # raw ProtocolRelease, intentionally wrong type
                protocol=protocol,
                models=models,
                target_set=final_targets,
                extractor=policy_metrics,
                loss=loss,
            )
        self.assertEqual(baseline_model.calls + alternative_model.calls, before)
```

Also construct a `VerifiedProtocolRelease` for another protocol and require `ProtocolReleaseVerificationError` before any model calls.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_protocol_release.ProtocolReleaseComparisonTests -v
```

Expected: missing comparison API / report stage.

- [ ] **Step 3: Add exact experiment stage**

```python
RELEASED_MODEL_COMPARISON = "released_model_comparison"
```

- [ ] **Step 4: Implement wrapper without duplicating final preflight**

Order:

```python
if not isinstance(verified_release, VerifiedProtocolRelease):
    raise TypeError(...)
verified_release.require_matches(protocol)
comparison = compare_models_on_final_partition(
    runner=runner,
    models=models,
    target_set=target_set,
    extractor=extractor,
    loss=loss,
    simulation_seeds=protocol.simulation_seeds,
    protocol=protocol,
)
```

Then build:

```python
manifest = ExperimentManifest(
    stage=ExperimentStage.RELEASED_MODEL_COMPARISON,
    inputs={
        "verified_release_hash": verified_release.content_hash,
        "release_hash": verified_release.release_hash,
        "protocol_hash": protocol.content_hash,
        "verifier_identity": verified_release.verifier_identity,
        "verified_receipt_hashes": verified_release.verified_receipt_hashes,
        "comparison_manifest_hash": comparison.manifest.content_hash,
    },
    parent_hashes=(comparison.manifest.content_hash,),
)
```

Return `ReleasedModelComparisonReport`. Do not modify `ModelComparisonReport` or its existing manifest semantics.

- [ ] **Step 5: Prove report artifact compatibility**

Test:

```python
report = compare_released_models(...)
self.assertIs(report.manifest.stage, ExperimentStage.RELEASED_MODEL_COMPARISON)
self.assertEqual(report.best.name, report.comparison.best.name)
self.assertIs(attest_report(report).require_integrity(), report)
```

- [ ] **Step 6: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_protocol_release -v
git add narrative_dynamics/contracts.py narrative_dynamics/observations/release.py tests/test_protocol_release.py
git commit -m "feat: gate final comparison on verified release"
```

---

### Task 7: Full Fixture Train → Selection → Witness → Final Pipeline and Public Observational API

**Files:**
- Modify: `tests/test_observational_fixture_pipeline.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Modify: `narrative_dynamics/__init__.py`

**Interfaces:**
- Observations package/root exports:
  - `TrainingCaseFit`
  - `TrainingCandidateFit`
  - `TrainingFitReport`
  - `fit_training_target_grid`
  - `ProtocolRelease`
  - `WitnessReceipt`
  - `VerifiedProtocolRelease`
  - `ProtocolReleaseVerificationError`
  - `ReleasedModelComparisonReport`
  - `verify_protocol_release`
  - `compare_released_models`
- Model-specific adapter symbols remain under `narrative_dynamics.adapters.prison_reactive` and `prison_metrics`, matching existing prison-adapter convention.

- [ ] **Step 1: Write the end-to-end RED/acceptance test**

The integration test must use real model code with fresh factories:

```python
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model
from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model
from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalMetricGroup
from narrative_dynamics.model_comparison import ComparisonModel
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    ProtocolRelease,
    WitnessReceipt,
    compare_released_models,
    construct_categorical_targets,
    fit_training_target_grid,
    load_observation_dataset,
    verify_protocol_release,
)
from narrative_dynamics.simulation import ModelFactory, SimulationRunner
from narrative_dynamics.uncertainty import ParameterAcceptanceSet
from narrative_dynamics.validation import (
    EvaluationRole,
    HeldOutCase,
    HeldOutSuite,
    select_on_validation_suite,
)
```

Factories:

```python
baseline_factory = ModelFactory(
    name="finite-prison-pomdp",
    create=create_prison_pomdp_model,
    version="1.0.0",
    implementation_revision="prison-pomdp-v1",
)
reactive_factory = ModelFactory(
    name="finite-prison-reactive",
    create=create_prison_reactive_model,
    version="1.0.0",
    implementation_revision="prison-reactive-v1",
)
```

Shared target/loss:

```python
spec = CategoricalTargetSpec(
    name="prison-initial-choice-target",
    version="1",
    categories=("scout", "escape", "submit"),
    metric_prefix="initial",
)
loss = CategoricalBrierLoss((
    CategoricalMetricGroup(
        "initial-action",
        ("initial.scout", "initial.escape", "initial.submit"),
    ),
))
```

Use three independent seed plans:

```text
training:  (101, 102)
selection: (201, 202)
final:     (301, 302)
```

- [ ] **Step 2: Fit the finite grid on TRAIN only**

Use the same candidate grid for both one-parameter models:

```python
parameter_grid = {"beta": (0.5, 1.0, 2.0, 4.0)}
```

Construct `TRAIN` targets only, then call `fit_training_target_grid` independently for baseline/reactive. Convert `report.candidate_parameters` to `ParameterAcceptanceSet` using only the corresponding training report manifest as provenance.

- [ ] **Step 3: Select frozen beta on SELECTION_VALIDATION only**

Construct selection targets. Convert each constructed case to:

```python
HeldOutCase(
    scenario=case.scenario,
    seeds=(201, 202),
    target=case.target_map,
    name=case.name,
)
```

Build one `HeldOutSuite(role=EvaluationRole.SELECTION_VALIDATION)` and call `select_on_validation_suite` independently for both models, using only each model's training-derived acceptance set.

Freeze:

```python
baseline_frozen = FrozenModelCandidate.from_selection(
    "finite-prison-pomdp", baseline_factory, baseline_selection
)
reactive_frozen = FrozenModelCandidate.from_selection(
    "finite-prison-reactive", reactive_factory, reactive_selection
)
```

Assert each frozen `selection_manifest_hash` equals the real selection report manifest.

- [ ] **Step 4: Create final protocol before executing FINAL_TEST**

```python
protocol = PreregisteredEvaluationProtocol.create(
    name="prison-model-comparison-v1",
    version="1",
    dataset=dataset,
    target_spec=spec,
    extractor=prison_initial_action_metrics,
    loss=loss,
    simulation_seeds=(301, 302),
    baseline_name="finite-prison-pomdp",
    candidates=(baseline_frozen, reactive_frozen),
    thresholds=AdequacyThresholds(
        max_mean_loss=0.35,
        max_worst_loss=0.50,
    ),
)
```

No final model execution may occur before this protocol and the release verification in the next step.

- [ ] **Step 5: Create and externally verify release**

Use the test-only verifier from `tests/test_protocol_release.py` or move it into a test helper if importing a test module would create a cycle. Do not place it in production code.

```python
release = ProtocolRelease.create(
    name="prison-model-comparison-release",
    version="1",
    protocol=protocol,
    source_revision={
        "kind": "declared_revision",
        "repository": "qigao/lean",
        "revision": "prison-alternative-release-v1",
    },
)
receipt = WitnessReceipt.create(
    provider="test-fixture",
    authority="integration-test",
    subject_hash=release.content_hash,
    reference="prison-release-v1",
    claimed_at="2026-08-23T00:00:00Z",
    proof={"nonce": "release-v1"},
)
verified = verify_protocol_release(
    release,
    protocol=protocol,
    receipts=(receipt,),
    verifier=FixtureWitnessVerifier(),
)
```

- [ ] **Step 6: Execute untouched FINAL_TEST through release gate**

Construct final targets only now, then:

```python
report = compare_released_models(
    runner=SimulationRunner(),
    verified_release=verified,
    protocol=protocol,
    models=(
        ComparisonModel(frozen=baseline_frozen, model=baseline_factory),
        ComparisonModel(frozen=reactive_frozen, model=reactive_factory),
    ),
    target_set=final_targets,
    extractor=prison_initial_action_metrics,
    loss=loss,
)
```

Assertions must prove fairness without asserting which scientific model must win forever:

```python
self.assertEqual(
    {entry.name for entry in report.comparison.ranking},
    {"finite-prison-pomdp", "finite-prison-reactive"},
)
self.assertEqual(
    report.manifest.inputs["release_hash"],
    release.content_hash,
)
for entry in report.comparison.ranking:
    self.assertEqual(entry.final_test.validation.manifest.inputs["loss"], protocol.loss_identity)
self.assertIs(attest_report(report).require_integrity(), report)
```

Also assert every final test child uses the same `protocol.simulation_seeds` by reading its held-out case manifest inputs.

- [ ] **Step 7: Add pipeline role-drift RED cases before declaring acceptance GREEN**

At minimum prove all fail before final model execution:

```text
selection target passed as final target
changed extractor
changed loss
changed final seed tuple through a forged/reconstructed protocol attempt
substituted frozen beta
raw ProtocolRelease instead of VerifiedProtocolRelease
release from a different protocol
```

Reuse existing comparison preflight behavior; do not add duplicate validation if the existing exception already satisfies the obligation.

- [ ] **Step 8: Export only observational/runtime public APIs**

Update `narrative_dynamics/observations/__init__.py` and root `narrative_dynamics/__init__.py` with training/release APIs. Add a public API test:

```python
expected = (
    "TrainingFitReport",
    "fit_training_target_grid",
    "ProtocolRelease",
    "WitnessReceipt",
    "VerifiedProtocolRelease",
    "ProtocolReleaseVerificationError",
    "ReleasedModelComparisonReport",
    "verify_protocol_release",
    "compare_released_models",
)
self.assertEqual(
    tuple(name for name in expected if not hasattr(narrative_dynamics, name)),
    (),
)
```

Do not add model-specific adapter names to root exports; callers use `narrative_dynamics.adapters.prison_reactive` and `prison_metrics` just as the existing POMDP adapter remains module-scoped.

- [ ] **Step 9: Verify targeted GREEN**

```bash
python3 -m unittest \
  tests.test_prison_reactive_model \
  tests.test_observational_training_fit \
  tests.test_protocol_release \
  tests.test_observational_fixture_pipeline \
  -v
```

Expected: all pass.

- [ ] **Step 10: Run complete Python regression suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all existing and new Python tests pass.

- [ ] **Step 11: Verify no Lean source changed and diff is clean**

```bash
git diff --check
git diff --name-only proof/narrative-dynamics-v0...HEAD | grep '\.lean$' && exit 1 || true
python3 -m compileall -q narrative_dynamics tests
```

Expected: no `.lean` paths, no whitespace errors, compileall success.

- [ ] **Step 12: Commit**

```bash
git add \
  narrative_dynamics/observations/__init__.py \
  narrative_dynamics/__init__.py \
  tests/test_observational_fixture_pipeline.py
git commit -m "feat: exercise released prison model comparison pipeline"
```

---

### Task 8: Remote RED/GREEN Evidence, Review, and Integration

**Files:**
- No production file should be changed merely to satisfy this task unless CI exposes a real defect.
- Update PR body only after each observed RED/GREEN state.

**Interfaces:**
- Feature branch: `work/prison-alternative-release-v1`
- Base: `proof/narrative-dynamics-v0`
- Main integration PR: `#2` against `master`

- [ ] **Step 1: Create/open the feature PR while tests are intentionally RED**

The first implementation commit should contain tests that fail for the intended missing API before corresponding production code. Keep PR draft. Record exact workflow run ID and failure reason.

- [ ] **Step 2: Require RED precision**

A valid RED has:

```text
Lean conformance: success
full Lean build: success
Lean theorem tests: success
Python: failures only in newly introduced alternative/release/fixture obligations
```

If an unrelated existing test fails, invoke systematic debugging before implementing more production code.

- [ ] **Step 3: Complete tasks in order with one GREEN commit per coherent boundary**

Recommended commit sequence:

```text
feat: add shared prison initial-action metrics
feat: add independent reactive prison model
feat: add train-only observational grid fitting
test: add versioned synthetic prison observations
feat: add externally witnessed protocol release
feat: gate final comparison on verified release
feat: exercise released prison model comparison pipeline
```

Tests may be committed separately as explicit RED commits where useful; never squash away the only observable RED evidence before CI records it.

- [ ] **Step 4: Run full GitHub Actions proof workflow on final feature head**

Require:

```text
Lean-generated reference vectors: success
full lake build: success (expected 8685 jobs unless repository base changes)
every Lean theorem test: success
complete Python unittest discovery: success
```

- [ ] **Step 5: Request code review and inspect the exact diff**

Review for:

```text
no import from prison_reactive → prison_pomdp
no new dependency files
no .lean changes
no test-only verifier exported from production package
no final-data access in training helper
no final recalibration after release creation
no bypass of existing compare_models_on_final_partition preflight
no raw release accepted where VerifiedProtocolRelease is required
no receipt provider string treated as trust by itself
```

Any review defect gets a new failing test before a fix.

- [ ] **Step 6: Fast-forward integrate only after final feature GREEN**

Before moving `proof/narrative-dynamics-v0`, compare base and feature head and require `behind_by == 0`; update the ref with `force=false` only.

- [ ] **Step 7: Verify PR #2 merge context**

After fast-forward, require a fresh `proof` workflow for PR #2 against `master`. Do not claim integration complete until that merge-context run is `completed / success` with all Lean gates and the complete Python suite green.

- [ ] **Step 8: Update PR #2 documentation**

Record:

```text
independent reactive alternative model
synthetic fixture only — not empirical human data
training/selection/final role separation
externally witnessed release interface, not built-in cryptographic trust
release-gated final comparison
exact final head SHA
feature workflow run
PR #2 merge-context workflow run
Python test count
no Lean source change
```

Retain explicit limits: witness authority is external, no trusted timestamp/signature backend is bundled, extractor identity is not a transitive implementation hash, and neither model comparison nor proper score establishes empirical validity.
