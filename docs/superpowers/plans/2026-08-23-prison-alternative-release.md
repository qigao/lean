# Prison Alternative Model and Witnessed Protocol Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independently specified one-parameter reactive prison model, a versioned synthetic observational fixture, an explicit train → selection → final evaluation path, and an externally witnessed protocol-release gate without changing the frozen prison POMDP or any Lean source.

**Architecture:** Keep `finite-prison-pomdp` untouched. Add a separate `finite-prison-reactive` adapter that shares only the scenario surface and `initial_policy` output shape; add one shared categorical extractor; compose multi-case training from existing `calibrate_grid`; represent release and witness evidence as canonical immutable identities; require a caller-supplied verifier before delegating to the existing `compare_models_on_final_partition` preflight and execution path.

**Tech Stack:** Python 3 standard library only, `unittest`, existing `narrative_dynamics` runtime/contracts/manifests/schema validation/subprocess execution, GitHub Actions `proof`, Lean 4 only as an unchanged CI gate.

**Spec:** `docs/superpowers/specs/2026-08-23-prison-alternative-release-design.md`

## Global Constraints

- Python-only. No `.lean` source may change.
- Existing `finite-prison-pomdp` semantics and API remain frozen.
- Reactive alternative has exactly one fitted parameter: `beta > 0`.
- Reactive code must not import or call `narrative_dynamics.adapters.prison_pomdp` or reuse its posterior/planning helpers.
- Exact cue rule: `V_escape = p*reward - (1-p)*cost`; `cue_strength = 2*signal_accuracy - 1`; `cue_scale = (escape_reward + capture_cost)/2`; clear adds the cue term, alarm subtracts it.
- `guard_persistence` may affect realized post-scout environment outcomes but must not affect reactive policy calculation.
- No cryptographic dependency or built-in signature/timestamp implementation.
- Witness receipts remain untrusted until a caller-supplied verifier accepts them.
- Fixture is explicitly synthetic, non-empirical, and non-population-representative.
- Train, selection-validation, and final-test roles remain separated.
- Final comparison continues to use existing target/extractor/loss/seed/candidate/runtime identity and mutable-source preflight checks.
- Production adapter smoke tests continue through `TrustedExecutionPolicy`, measured implementation pinning, `SubprocessModel`, four executable schemas, result artifacts, and `SimulationRunner`.
- Every production change follows RED → verify RED → minimal GREEN → verify GREEN → commit.

---

## File Structure

```text
narrative_dynamics/adapters/prison_metrics.py
    Shared initial-action extractor only.

narrative_dynamics/adapters/prison_reactive.py
    Independent reactive simulator, policy math, stochastic episode execution,
    contract, subprocess source.

narrative_dynamics/observations/training.py
    Train-only multi-case grid aggregation using existing calibrate_grid.

narrative_dynamics/observations/release.py
    Protocol release/receipt identities, verifier gate, released comparison wrapper.

narrative_dynamics/contracts.py
    Add TRAINING_TARGET_FIT and RELEASED_MODEL_COMPARISON only.

narrative_dynamics/observations/__init__.py
narrative_dynamics/__init__.py
    Export observational training/release APIs only.

fixtures/observations/prison_initial_choice_v1.json
    Fixed synthetic record-oriented dataset with declared content hash.

tests/test_prison_reactive_model.py
tests/test_observational_training_fit.py
tests/test_protocol_release.py
tests/test_observational_fixture_pipeline.py
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

- [ ] **Step 1: Write the failing extractor tests**

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

    def test_shared_extractor_returns_exact_three_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics

        self.assertEqual(
            prison_initial_action_metrics(
                self._trace({"scout": 0.25, "escape": 0.5, "submit": 0.25})
            ),
            {
                "initial.scout": 0.25,
                "initial.escape": 0.5,
                "initial.submit": 0.25,
            },
        )
        self.assertEqual(prison_initial_action_metrics.version, "1.0.0")

    def test_shared_extractor_rejects_missing_or_non_finite_coordinates(self):
        from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics

        with self.assertRaises(ValueError):
            prison_initial_action_metrics(self._trace({"scout": 0.5, "escape": 0.5}))
        with self.assertRaises(ValueError):
            prison_initial_action_metrics(
                self._trace({"scout": math.nan, "escape": 0.5, "submit": 0.5})
            )
```

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_prison_reactive_model.PrisonInitialActionMetricTests -v
```

Expected: module import failure for `narrative_dynamics.adapters.prison_metrics`.

- [ ] **Step 3: Implement the extractor**

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

Normalization remains owned by the categorical loss boundary.

- [ ] **Step 4: Verify GREEN**

```bash
python3 -m unittest tests.test_prison_reactive_model.PrisonInitialActionMetricTests -v
```

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
- `FinitePrisonReactiveModel`
- `create_prison_reactive_model() -> FinitePrisonReactiveModel`
- `prison_reactive_contract() -> ModelContract`
- `prison_reactive_source(*, limits: ProcessLimits | None = None) -> SubprocessModel`
- Model name `finite-prison-reactive`
- Version `1.0.0`
- Implementation revision `prison-reactive-v1`

- [ ] **Step 1: Add failing behavioral tests**

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
        self.runner = SimulationRunner(repository_identity=RepositoryIdentity(provider="test"))

    def test_reactive_values_follow_declared_cue_rule_and_replay_seed(self):
        from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model

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
        from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model

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

Expected: missing `prison_reactive` module.

- [ ] **Step 3: Implement exact reactive math and stable softmax**

The production implementation must calculate exactly:

```python
v_escape = prior_weak * escape_reward - (1.0 - prior_weak) * capture_cost
v_submit = submit_reward
cue_strength = 2.0 * signal_accuracy - 1.0
cue_scale = (escape_reward + capture_cost) / 2.0
clear_escape = v_escape + cue_strength * cue_scale
alarm_escape = v_escape - cue_strength * cue_scale
```

Use a model-local max-shifted softmax. Put floating normalization residual on the largest-mass coordinate and reject non-finite or negative resulting mass.

For horizon 2:

```python
p_clear = (
    prior_weak * signal_accuracy
    + (1.0 - prior_weak) * (1.0 - signal_accuracy)
)
clear_values = {"escape": clear_escape, "submit": v_submit}
alarm_values = {"escape": alarm_escape, "submit": v_submit}
clear_policy = stable_softmax(clear_values, beta=beta, order=("escape", "submit"))
alarm_policy = stable_softmax(alarm_values, beta=beta, order=("escape", "submit"))
v_scout = (
    -scout_cost
    + discount * (
        p_clear * sum(clear_policy[a] * clear_values[a] for a in ("escape", "submit"))
        + (1.0 - p_clear)
        * sum(alarm_policy[a] * alarm_values[a] for a in ("escape", "submit"))
    )
)
```

Initial horizon-2 policy uses `(v_scout, v_escape, v_submit)`. Horizon 1 sets `scout` probability exactly `0.0` and softmaxes `escape/submit` only.

Episode realization may sample latent guard weakness and a signal. If scouting then escaping, `guard_persistence` is used only to realize the future environment outcome.

Required outcome fields:

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

- [ ] **Step 4: Implement the four-schema contract**

Parameter schema: exact `beta` only, minimum `1e-12`.

Scenario schema: exact baseline scenario fields and ranges.

Event schema kinds:

```text
initial_decision
observation_received
terminal_decision
episode_ended
```

Outcome schema includes the exact three-action `initial_policy`, three-action `initial_values`, clear/alarm `cue_values`, action/signal strings, escaped boolean, utility number, and steps integer 1–2.

- [ ] **Step 5: Add source/pinning test and verify RED if source boundary is still missing**

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
    trace = self.runner.run_once(
        policy.bind(source, contract=contract),
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

- [ ] **Step 6: Implement subprocess source**

```python
return SubprocessModel(
    name="finite-prison-reactive",
    factory="narrative_dynamics.adapters.prison_reactive:create_prison_reactive_model",
    version="1.0.0",
    implementation_revision="prison-reactive-v1",
    limits=ProcessLimits() if limits is None else limits,
)
```

- [ ] **Step 7: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_prison_reactive_model -v
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
    def candidate_parameters(self) -> tuple[tuple[tuple[str, float], ...], ...]:
        return tuple(candidate.parameters for candidate in self.ranking)
```

Function signature:

```python
def fit_training_target_grid(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    target_report: TargetConstructionReport,
    parameter_grid: Mapping[str, Iterable[float]],
    simulation_seeds: Iterable[int],
    extractor: object,
    loss: MetricLoss,
) -> TrainingFitReport:
    """Rank one finite parameter grid across all TRAIN target cases."""
```

- [ ] **Step 1: Write RED tests**

Use a record-oriented dataset with two train cases and a simple probability model. Assert:

```python
report = fit_training_target_grid(
    runner=SimulationRunner(),
    model=factory,
    target_report=train_targets,
    parameter_grid={"p": (0.2, 0.5, 0.8)},
    simulation_seeds=(101, 102),
    extractor=policy_metrics,
    loss=brier_loss,
)
self.assertIs(report.manifest.stage, ExperimentStage.TRAINING_TARGET_FIT)
self.assertEqual(len(report.ranking), 3)
self.assertEqual(
    set(report.candidate_parameters),
    {(("p", 0.2),), (("p", 0.5),), (("p", 0.8),)},
)
self.assertTrue(all(len(candidate.cases) == 2 for candidate in report.ranking))
self.assertIs(attest_report(report).require_integrity(), report)
```

Also pass SELECTION_VALIDATION and FINAL_TEST reports to the helper with a counting model and assert `ValueError` plus zero model calls.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_observational_training_fit -v
```

Expected: missing training API and `TRAINING_TARGET_FIT` stage.

- [ ] **Step 3: Add dedicated stage**

```python
TRAINING_TARGET_FIT = "training_target_fit"
```

- [ ] **Step 4: Implement by composing `calibrate_grid`**

Validate `target_report.role is ObservationPartitionRole.TRAIN` before any execution. Materialize `simulation_seeds` and `parameter_grid` dimensions once.

For every train case, call:

```python
calibration = calibrate_grid(
    runner=runner,
    model=model,
    scenario=case.scenario,
    parameter_grid=replayable_grid,
    seeds=seed_tuple,
    extractor=extractor,
    target=case.target_map,
    loss=loss,
)
```

Require the same candidate parameter set in every child calibration. Build each `TrainingCandidateFit` by matching the same parameter tuple across children, then calculate `fmean(losses)` and `max(losses)`. Sort by `(mean_loss, worst_loss, parameters)`.

Manifest stage is `TRAINING_TARGET_FIT`, inputs include model identity, target report hash, training seed tuple, extractor identity, loss identity, canonical parameter candidates, and per-case scenario identities. Parents are all child `GRID_CALIBRATION` manifests.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_observational_training_fit -v
git add narrative_dynamics/contracts.py narrative_dynamics/observations/training.py tests/test_observational_training_fit.py
git commit -m "feat: add train-only observational grid fitting"
```

---

### Task 4: Commit Versioned Synthetic Prison Fixture

**Files:**
- Create: `fixtures/observations/prison_initial_choice_v1.json`
- Create/Modify: `tests/test_observational_fixture_pipeline.py`

- [ ] **Step 1: Write fixture RED test before the JSON exists**

```python
from pathlib import Path
import unittest

from narrative_dynamics.observations import ObservationPartitionRole, load_observation_dataset

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

Run and verify file-not-found RED.

- [ ] **Step 2: Use these fixed records**

Common scenario fields: `escape_reward=8.0`, `capture_cost=10.0`, `submit_reward=1.0`, `scout_cost=0.25`, `discount=0.95`.

```text
TRAIN
train-1: prior=.45 accuracy=.80 persistence=.80 horizon=2 counts 52/28/20
train-2: prior=.70 accuracy=.70 persistence=.50 horizon=1 counts 0/61/39

SELECTION_VALIDATION
selection-1: prior=.35 accuracy=.85 persistence=.75 horizon=2 counts 58/17/25
selection-2: prior=.75 accuracy=.65 persistence=.50 horizon=1 counts 0/69/31

FINAL_TEST
final-1: prior=.80 accuracy=.75 persistence=.20 horizon=2 counts 18/66/16
final-2: prior=.80 accuracy=.75 persistence=.90 horizon=2 counts 43/45/12
final-3: prior=.25 accuracy=.90 persistence=.60 horizon=1 counts 0/14/86
```

Counts are ordered in text as scout/escape/submit. `final-1` and `final-2` deliberately differ in persistence.

- [ ] **Step 3: Generate the exact JSON payload without simulator calls**

Use this complete script from repository root:

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

COMMON = {
    "escape_reward": 8.0,
    "capture_cost": 10.0,
    "submit_reward": 1.0,
    "scout_cost": 0.25,
    "discount": 0.95,
}


def record(record_id, prior, accuracy, persistence, horizon, scout, escape, submit):
    return ObservationRecord(
        id=record_id,
        scenario=Scenario(
            id=f"{record_id}-scenario",
            payload={
                **COMMON,
                "prior_weak": prior,
                "signal_accuracy": accuracy,
                "guard_persistence": persistence,
                "horizon": horizon,
            },
        ),
        counts={"scout": scout, "escape": escape, "submit": submit},
        metadata={"fixture_role": record_id.split("-")[0]},
    )


dataset = ObservationDataset(
    name="prison-initial-choice",
    version="1.0.0",
    source={
        "kind": "synthetic_fixture",
        "purpose": "protocol_integration_test",
    },
    provenance={
        "empirical_human_data": False,
        "population_representative": False,
    },
    partitions=(
        ObservationPartition(
            name="train",
            role=ObservationPartitionRole.TRAIN,
            records=(
                record("train-1", .45, .80, .80, 2, 52, 28, 20),
                record("train-2", .70, .70, .50, 1, 0, 61, 39),
            ),
        ),
        ObservationPartition(
            name="selection_validation",
            role=ObservationPartitionRole.SELECTION_VALIDATION,
            records=(
                record("selection-1", .35, .85, .75, 2, 58, 17, 25),
                record("selection-2", .75, .65, .50, 1, 0, 69, 31),
            ),
        ),
        ObservationPartition(
            name="final_test",
            role=ObservationPartitionRole.FINAL_TEST,
            records=(
                record("final-1", .80, .75, .20, 2, 18, 66, 16),
                record("final-2", .80, .75, .90, 2, 43, 45, 12),
                record("final-3", .25, .90, .60, 1, 0, 14, 86),
            ),
        ),
    ),
)

path = Path("fixtures/observations/prison_initial_choice_v1.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    json.dumps(dataset.to_payload(), indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
```

- [ ] **Step 4: Add anti-tautology assertions**

```python
final = dataset.partition(ObservationPartitionRole.FINAL_TEST)
self.assertEqual(final.records[0].count_map, {"scout": 18, "escape": 66, "submit": 16})
self.assertEqual(final.records[1].count_map, {"scout": 43, "escape": 45, "submit": 12})
```

Do not call either simulator to generate expected counts.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_observational_fixture_pipeline.PrisonObservationFixtureTests -v
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


class ProtocolReleaseVerificationError(ValueError):
    """Release or witness evidence failed trusted verification."""


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
    schema_version: int = PROTOCOL_RELEASE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        name: str,
        version: str,
        protocol: PreregisteredEvaluationProtocol,
        source_revision: Mapping[str, object],
    ) -> "ProtocolRelease":
        identity = {
            "schema_version": PROTOCOL_RELEASE_SCHEMA_VERSION,
            "name": name,
            "version": version,
            "protocol_hash": protocol.content_hash,
            "dataset_hash": protocol.dataset_hash,
            "target_spec_hash": protocol.target_spec_hash,
            "candidate_hashes": tuple(candidate.content_hash for candidate in protocol.candidates),
            "source_revision": source_revision,
        }
        return cls(
            **identity,
            declared_content_hash=stable_content_hash(identity),
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "ProtocolRelease":
        return cls(
            schema_version=payload["schema_version"],
            name=payload["name"],
            version=payload["version"],
            protocol_hash=payload["protocol_hash"],
            dataset_hash=payload["dataset_hash"],
            target_spec_hash=payload["target_spec_hash"],
            candidate_hashes=tuple(payload["candidate_hashes"]),
            source_revision=payload["source_revision"],
            declared_content_hash=payload["content_hash"],
        )


@dataclass(frozen=True)
class WitnessReceipt:
    provider: str
    authority: str
    subject_hash: str
    reference: str
    claimed_at: str
    proof: Mapping[str, object]
    declared_content_hash: str
    schema_version: int = WITNESS_RECEIPT_SCHEMA_VERSION


@dataclass(frozen=True)
class VerifiedProtocolRelease:
    release_hash: str
    protocol_hash: str
    verifier_identity: Mapping[str, object]
    verified_receipt_hashes: tuple[str, ...]
    status: str = "verified"

    def require_matches(self, protocol: PreregisteredEvaluationProtocol) -> None:
        if self.status != "verified" or self.protocol_hash != protocol.content_hash:
            raise ProtocolReleaseVerificationError("verified release does not match protocol")
```

`ProtocolRelease.__post_init__`, `WitnessReceipt.__post_init__`, and `VerifiedProtocolRelease.__post_init__` must validate schema version, non-empty text, canonical hashes, recursively frozen mappings, non-empty/unique verified receipt hashes, exact status `verified`, and declared-vs-recomputed content hashes where applicable.

`ProtocolRelease.identity_payload()` excludes `declared_content_hash`; `content_hash` recomputes from the identity payload; `to_payload()` adds `content_hash`. `WitnessReceipt` follows the same pattern.

Verifier structural contract:

```python
verifier.name: str
verifier.version: str
verifier.verify(release: ProtocolRelease, receipt: WitnessReceipt) -> bool
```

- [ ] **Step 1: Write release/receipt round-trip RED tests**

Create a small `PreregisteredEvaluationProtocol` fixture using existing test probability models, then assert:

```python
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
self.assertEqual(ProtocolRelease.from_payload(release.to_payload()), release)

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

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_protocol_release.ProtocolReleaseIdentityTests -v
```

Expected: missing release module/API.

- [ ] **Step 3: Implement canonical release/receipt values**

Use existing `stable_content_hash` and canonical mapping freezer. Candidate hashes are always derived from the protocol's canonical candidate order.

- [ ] **Step 4: Add trusted-verifier RED tests**

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
            and dict(receipt.proof) == {"nonce": "release-v1"}
        )
```

Required assertions:
- rejecting verifier → `ProtocolReleaseVerificationError`
- wrong subject → error before verifier call
- duplicate receipt → error before verifier call
- valid receipt → one accepted receipt hash and `status == "verified"`

- [ ] **Step 5: Implement `verify_protocol_release`**

Exact pre-verifier checks:

```text
release.protocol_hash == protocol.content_hash
release.dataset_hash == protocol.dataset_hash
release.target_spec_hash == protocol.target_spec_hash
release.candidate_hashes == tuple(candidate.content_hash for candidate in protocol.candidates)
every receipt.subject_hash == release.content_hash
receipt content hashes are unique
```

Then call `verifier.verify(release, receipt)` once per receipt. Chain verifier exceptions into `ProtocolReleaseVerificationError`. At least one receipt must be accepted. Record verifier identity with `component_identity(verifier)`.

- [ ] **Step 6: Verify GREEN and commit**

```bash
python3 -m unittest tests.test_protocol_release -v
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
    def best(self) -> ModelComparisonEntry:
        return self.comparison.best

    @property
    def entry_map(self):
        return self.comparison.entry_map


def compare_released_models(
    *,
    runner: SimulationRunner,
    verified_release: VerifiedProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    models: tuple[ComparisonModel, ...],
    target_set: TargetConstructionReport,
    extractor: object,
    loss: MetricLoss,
) -> ReleasedModelComparisonReport:
    if not isinstance(verified_release, VerifiedProtocolRelease):
        raise TypeError("released comparison requires VerifiedProtocolRelease")
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
    return ReleasedModelComparisonReport(
        release_hash=verified_release.release_hash,
        verification_hash=verified_release.content_hash,
        comparison=comparison,
        manifest=manifest,
    )
```

- [ ] **Step 1: Write RED that raw release and mismatched verification fail before execution**

Use counting models. Passing a raw `ProtocolRelease` must raise `TypeError` with zero model calls. A verified release whose protocol hash differs from the supplied protocol must raise `ProtocolReleaseVerificationError` before any call.

- [ ] **Step 2: Verify RED**

```bash
python3 -m unittest tests.test_protocol_release.ProtocolReleaseComparisonTests -v
```

- [ ] **Step 3: Add exact stage**

```python
RELEASED_MODEL_COMPARISON = "released_model_comparison"
```

- [ ] **Step 4: Implement wrapper exactly as above**

Do not modify `ModelComparisonReport`; do not reproduce its preflight logic.

- [ ] **Step 5: Verify artifact compatibility**

```python
report = compare_released_models(
    runner=runner,
    verified_release=verified,
    protocol=protocol,
    models=models,
    target_set=final_targets,
    extractor=policy_metrics,
    loss=loss,
)
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

### Task 7: Full Fixture Train → Selection → Witness → Final Pipeline and Public API

**Files:**
- Modify: `tests/test_observational_fixture_pipeline.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Modify: `narrative_dynamics/__init__.py`

**Public observational/root exports:**
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

Model-specific reactive APIs remain module-scoped under `narrative_dynamics.adapters.prison_reactive` and `prison_metrics`.

- [ ] **Step 1: Write end-to-end acceptance test imports and factories**

```python
from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.adapters.prison_pomdp import create_prison_pomdp_model
from narrative_dynamics.adapters.prison_reactive import create_prison_reactive_model
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

Seed plans:
- training `(101, 102)`
- selection `(201, 202)`
- final `(301, 302)`

- [ ] **Step 2: Fit TRAIN grid only**

Load fixture and construct TRAIN targets. Use `parameter_grid={"beta": (0.5, 1.0, 2.0, 4.0)}` for both models. Convert each `TrainingFitReport.candidate_parameters` to `ParameterAcceptanceSet` with only that training report manifest as provenance.

- [ ] **Step 3: Select beta on SELECTION_VALIDATION only**

Construct selection targets and convert each target case to:

```python
HeldOutCase(
    scenario=case.scenario,
    seeds=(201, 202),
    target=case.target_map,
    name=case.name,
)
```

Build `HeldOutSuite(role=EvaluationRole.SELECTION_VALIDATION)` and run `select_on_validation_suite` independently for each model. Freeze with `FrozenModelCandidate.from_selection` and assert the selection manifest hashes are real selection report manifests.

- [ ] **Step 4: Create preregistered protocol before final execution**

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
    thresholds=AdequacyThresholds(max_mean_loss=0.35, max_worst_loss=0.50),
)
```

- [ ] **Step 5: Create and verify release**

Use a test-only `FixtureWitnessVerifier` defined in test code, never production exports.

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

Assert:

```python
self.assertEqual(
    {entry.name for entry in report.comparison.ranking},
    {"finite-prison-pomdp", "finite-prison-reactive"},
)
self.assertEqual(report.manifest.inputs["release_hash"], release.content_hash)
self.assertIs(attest_report(report).require_integrity(), report)
```

For every child final-test report, inspect held-out manifest inputs and confirm every case uses `(301, 302)`.

Do not assert which scientific model must rank first forever.

- [ ] **Step 7: Add role/identity drift RED cases**

Each must fail before final model execution:
- selection target supplied as final target
- changed extractor
- changed loss
- changed protocol final seed tuple through a reconstructed protocol
- substituted frozen beta
- raw `ProtocolRelease`
- verified release from another protocol

Reuse existing comparison preflight when it already enforces the boundary.

- [ ] **Step 8: Add root/package exports and public API test**

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

- [ ] **Step 9: Verify targeted GREEN**

```bash
python3 -m unittest \
  tests.test_prison_reactive_model \
  tests.test_observational_training_fit \
  tests.test_protocol_release \
  tests.test_observational_fixture_pipeline \
  -v
```

- [ ] **Step 10: Run complete Python regression suite**

```bash
python3 -m unittest discover -s tests -v
```

- [ ] **Step 11: Verify diff and no Lean changes**

```bash
git diff --check
git diff --name-only proof/narrative-dynamics-v0...HEAD | grep '\.lean$' && exit 1 || true
python3 -m compileall -q narrative_dynamics tests
```

- [ ] **Step 12: Commit**

```bash
git add narrative_dynamics/observations/__init__.py narrative_dynamics/__init__.py tests/test_observational_fixture_pipeline.py
git commit -m "feat: exercise released prison model comparison pipeline"
```

---

### Task 8: Remote RED/GREEN Evidence, Review, and Integration

**Feature branch:** `work/prison-alternative-release-v1`

**Base:** `proof/narrative-dynamics-v0`

**Main integration PR:** `#2` against `master`

- [ ] **Step 1: Open a draft feature PR while a deliberate implementation RED exists**

Record exact workflow run and failure reason.

- [ ] **Step 2: Require precise RED**

Valid RED:

```text
Lean conformance: success
full Lean build: success
Lean theorem tests: success
Python: failures only in new alternative/release/fixture obligations
```

Use systematic debugging if unrelated existing tests fail.

- [ ] **Step 3: Complete coherent RED/GREEN commits**

Recommended sequence:

```text
feat: add shared prison initial-action metrics
feat: add independent reactive prison model
feat: add train-only observational grid fitting
test: add versioned synthetic prison observations
feat: add externally witnessed protocol release
feat: gate final comparison on verified release
feat: exercise released prison model comparison pipeline
```

Keep observable RED evidence in remote CI before its GREEN fix.

- [ ] **Step 4: Run final feature `proof` workflow**

Require conformance success, full `lake build` success, all theorem tests success, complete Python suite success.

- [ ] **Step 5: Review exact diff**

Confirm:
- no reactive → POMDP import
- no new dependency file
- no `.lean` change
- no test verifier in production exports
- training helper never reads final data
- no final recalibration after release creation
- release wrapper delegates to `compare_models_on_final_partition`
- raw release cannot unlock comparison
- provider string alone never grants trust

Every review defect gets a failing test before a fix.

- [ ] **Step 6: Fast-forward only after feature GREEN**

Compare base/head and require `behind_by == 0`. Move `proof/narrative-dynamics-v0` with `force=false` only.

- [ ] **Step 7: Verify fresh PR #2 merge context**

Require a new PR #2 `proof` run against `master` to complete successfully before claiming integration complete.

- [ ] **Step 8: Update PR #2 body**

Record independent alternative model, synthetic-only fixture, train/selection/final separation, externally witnessed release interface, release-gated final comparison, final SHA, feature workflow, merge-context workflow, Python test count, and no Lean source change.

Retain explicit limits: witness authority is external; no trusted timestamp/signature backend is bundled; extractor identity is not a transitive implementation hash; neither proper scoring nor model comparison establishes empirical validity.
