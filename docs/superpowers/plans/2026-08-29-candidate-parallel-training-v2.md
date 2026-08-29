# Candidate-Parallel Training V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic candidate-level sharding and bounded CPU process execution for TRAIN and SELECTION while keeping sequential and parallel scientific reports hash-identical.

**Architecture:** Keep the existing sequential functions as the reference implementation. Add immutable training and selection candidate shard records plus canonical assemblers that reconstruct the exact existing calibration/validation manifests independent of worker completion order. Add an operational executor layer with sequential and spawn-based process backends, then add an explicit Feher/Hare parallel wrapper that reconstructs immutable worker context and never exposes FINAL execution through the candidate executor.

**Tech Stack:** Python 3 standard library (`dataclasses`, `concurrent.futures`, `multiprocessing`), existing `SimulationRunner`, observation/training/validation contracts, `unittest`, GitHub Actions Linux runners.

**Spec:** `docs/superpowers/specs/2026-08-29-candidate-parallel-training-v2-design.md`

## Global Constraints

- Base revision is frozen V1 `de09518d5d034b14a7a22545af02117769297517`; V1 must remain unchanged.
- This V2 is performance-only: no model-family, parameter-grid, partition, seed, metric, loss, threshold, separation-rule, or claim-scope changes.
- Existing public sequential APIs remain valid and remain the reference path.
- TRAIN seeds remain exactly `(101, 102)` for Feher/Hare.
- SELECTION seeds remain exactly `(201, 202)` for Feher/Hare.
- FINAL seeds `(301, 302)` are forbidden at the candidate executor boundary.
- Worker count, PID, CPU core, timestamps, queue order, and completion order must never enter scientific identity.
- Missing, duplicate, extra, malformed, cancelled, or failed shards return no authoritative assembled report.
- Parallel workers must measure/use the same repository identity and implementation bytes as the parent scientific run.
- Do not rely on mutable state inherited only through `fork`; process workers must be reconstructable under `spawn`.
- Ordinary CI must not access the real Feher/Hare human checkout. All process-equivalence tests use synthetic/test fixtures.
- No Docker and no GPU changes.

---

## File Structure

- Create `narrative_dynamics/observations/training_shards.py` — training candidate shard schema, one-candidate evaluator, exact calibration reconstruction, exact `TrainingFitReport` assembly.
- Create `narrative_dynamics/observations/selection_shards.py` — selection candidate shard schema, one-candidate held-out evaluator, exact acceptance/selection report assembly.
- Create `narrative_dynamics/candidate_execution.py` — operational executor protocol, sequential executor, bounded spawn-based process executor, typed execution failure.
- Create `narrative_dynamics/studies/feher_hare_two_stage_parallel.py` — explicit Feher/Hare TRAIN/SELECTION orchestration over the shard/executor APIs with immutable worker initialization and seed firewall.
- Modify `narrative_dynamics/observations/__init__.py` — export the new generic shard APIs without changing existing sequential exports.
- Modify `narrative_dynamics/studies/__init__.py` — export the explicit Feher/Hare parallel wrapper.
- Create `tests/test_candidate_parallel_training.py` — training shard/assembler RED→GREEN equivalence and fail-closed coverage.
- Create `tests/test_candidate_parallel_selection.py` — selection shard/assembler RED→GREEN equivalence and fail-closed coverage.
- Create `tests/test_candidate_process_executor.py` — real process backend, bounded worker behavior, result ordering, failure semantics.
- Create `tests/test_feher_hare_parallel_training.py` — synthetic study integration, sequential/process equality, repository identity, seed firewall, no FINAL.
- Create `tools/benchmark_candidate_parallel.py` — deterministic non-CI benchmark for sequential versus bounded-process candidate evaluation.

---

### Task 1: Training Candidate Shard Contract and Exact Assembly

**Files:**
- Create: `tests/test_candidate_parallel_training.py`
- Create: `narrative_dynamics/observations/training_shards.py`
- Modify: `narrative_dynamics/observations/__init__.py`

**Interfaces:**
- Consumes: `TrainingFitReport`, `TrainingCandidateFit`, `TrainingCaseFit`, `CandidateEvaluation`, `CalibrationResult`, `SimulationRunner`, `TargetConstructionReport`, existing metric/loss identity helpers.
- Produces:
  - `TrainingCandidateShard`
  - `evaluate_training_candidate(*, runner, model, target_report, parameters, simulation_seeds, extractor, loss) -> TrainingCandidateShard`
  - `assemble_training_fit_report(*, model, target_report, parameter_candidates, simulation_seeds, extractor, loss, shards) -> TrainingFitReport`
  - `TrainingCandidateShard.to_payload() -> dict[str, object]`
  - `TrainingCandidateShard.from_payload(payload: Mapping[str, object]) -> TrainingCandidateShard`

- [ ] **Step 1: Add test-only RED for exact sequential/shard equality**

Create `tests/test_candidate_parallel_training.py` using the existing fixture from `tests.test_observational_training_fit`:

```python
from narrative_dynamics.adapters.prison_metrics import prison_initial_action_metrics
from narrative_dynamics.observations.dataset import ObservationPartitionRole
from narrative_dynamics.simulation import SimulationRunner
from tests.test_observational_training_fit import (
    TrainingProbabilityModel,
    brier_loss,
    targets_for,
)


def _reference():
    from narrative_dynamics.observations.training import fit_training_target_grid
    return fit_training_target_grid(
        runner=SimulationRunner(),
        model=TrainingProbabilityModel(),
        target_report=targets_for(ObservationPartitionRole.TRAIN),
        parameter_grid={"p": (0.2, 0.5, 0.8)},
        simulation_seeds=(101, 102),
        extractor=prison_initial_action_metrics,
        loss=brier_loss(),
    )


class CandidateParallelTrainingTests(unittest.TestCase):
    def test_candidate_shards_reassemble_exact_reference_report(self):
        from narrative_dynamics.observations.training_shards import (
            assemble_training_fit_report,
            evaluate_training_candidate,
        )
        target_report = targets_for(ObservationPartitionRole.TRAIN)
        model = TrainingProbabilityModel()
        candidates = ((("p", 0.2),), (("p", 0.5),), (("p", 0.8),))
        shards = tuple(
            evaluate_training_candidate(
                runner=SimulationRunner(),
                model=model,
                target_report=target_report,
                parameters=parameters,
                simulation_seeds=(101, 102),
                extractor=prison_initial_action_metrics,
                loss=brier_loss(),
            )
            for parameters in candidates
        )
        assembled = assemble_training_fit_report(
            model=model,
            target_report=target_report,
            parameter_candidates=candidates,
            simulation_seeds=(101, 102),
            extractor=prison_initial_action_metrics,
            loss=brier_loss(),
            shards=tuple(reversed(shards)),
        )
        reference = _reference()
        self.assertEqual(assembled.ranking, reference.ranking)
        self.assertEqual(assembled.manifest, reference.manifest)
        self.assertEqual(assembled.manifest.content_hash, reference.manifest.content_hash)
```

Also add RED tests requiring payload round-trip integrity and rejection of missing, duplicate, and undeclared candidate shards.

- [ ] **Step 2: Run focused RED**

Run:

```bash
python3 -m unittest tests.test_candidate_parallel_training -v
```

Expected: FAIL because `narrative_dynamics.observations.training_shards` does not exist. No existing test should be modified to make RED pass.

- [ ] **Step 3: Implement immutable training shard schema**

Create `training_shards.py` with focused immutable records:

```python
@dataclass(frozen=True)
class TrainingShardCase:
    name: str
    metrics: tuple[tuple[str, float], ...]
    loss: float
    run_manifest_hashes: tuple[str, ...]


@dataclass(frozen=True)
class TrainingCandidateShard:
    parameters: ParameterTuple
    target_report_hash: str
    model_identity: Mapping[str, object]
    repository_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    cases: tuple[TrainingShardCase, ...]
```

`identity_payload()` and `content_hash` must bind every scientific field above but no worker metadata. `to_payload()`/`from_payload()` must include a declared `content_hash` and reject schema drift or tampering.

- [ ] **Step 4: Implement one-candidate evaluation using existing run semantics**

`evaluate_training_candidate` must:

1. require `target_report.role is TRAIN`;
2. canonicalize the one parameter tuple using the same numeric/name rules used by existing training/calibration;
3. iterate `target_report.cases` in existing order;
4. call `runner.run_batch(... seeds=simulation_seeds)` exactly once per case;
5. compute metrics with `aggregate_metrics` and loss with `evaluate_metric_loss`;
6. preserve run-manifest hashes in seed order;
7. bind `component_identity(model)`, `runner.repository_identity.manifest_identity()`, `callable_identity(extractor)`, and `metric_loss_identity(loss)`.

Do not construct ranking or a `TrainingFitReport` in the worker.

- [ ] **Step 5: Reconstruct the exact existing per-case `CalibrationResult` manifests**

`assemble_training_fit_report` must first canonicalize `parameter_candidates` exactly as `calibrate_grid` does. For each target case, gather the corresponding case result from every candidate shard and reconstruct:

```python
ranking = tuple(sorted(candidate_evaluations, key=lambda item: (item.loss, item.parameters)))
manifest = ExperimentManifest(
    stage=ExperimentStage.GRID_CALIBRATION,
    inputs={
        "model": component_identity(model),
        "scenario": scenario_identity(case.scenario),
        "parameter_grid": canonical_candidates,
        "seeds": seeds,
        "metric": callable_identity(extractor),
        "target_hash": stable_content_hash(case.target_map),
        "weights_hash": None,
        "loss": metric_loss_identity(loss),
    },
    parent_hashes=tuple(
        run_hash
        for parameters in canonical_candidates
        for run_hash in case_result_by_parameters[parameters].run_manifest_hashes
    ),
)
```

Then construct `CalibrationResult(best=ranking[0], ranking=ranking, manifest=manifest)` and reproduce the existing `fit_training_target_grid` aggregation exactly: candidate case fits, mean/worst loss, `(mean_loss, worst_loss, parameters)` ranking, and `TRAINING_TARGET_FIT` manifest parent ordering.

Validate before assembly that every shard agrees on target hash, model identity, repository identity, seeds, metric identity, loss identity, and exact case names. Reject missing/duplicate/extra candidates before constructing any report.

- [ ] **Step 6: Run focused GREEN and reference regression**

Run:

```bash
python3 -m unittest tests.test_candidate_parallel_training tests.test_observational_training_fit -v
```

Expected: all tests PASS and the new assembled manifest hash equals the untouched sequential reference hash.

- [ ] **Step 7: Export generic training shard APIs and commit**

Update `narrative_dynamics/observations/__init__.py` to export the shard type/evaluator/assembler while retaining all existing exports.

Commit:

```bash
git add tests/test_candidate_parallel_training.py narrative_dynamics/observations/training_shards.py narrative_dynamics/observations/__init__.py
git commit -m "feat: add deterministic training candidate shards"
```

---

### Task 2: Selection Candidate Shard Contract and Exact Assembly

**Files:**
- Create: `tests/test_candidate_parallel_selection.py`
- Create: `narrative_dynamics/observations/selection_shards.py`
- Modify: `narrative_dynamics/observations/__init__.py`

**Interfaces:**
- Consumes: `HeldOutSuite`, `HeldOutValidationReport`, `HeldOutAcceptanceReport`, `SelectionValidationReport`, `ParameterAcceptanceSet`, `validate_held_out`.
- Produces:
  - `SelectionCandidateShard`
  - `evaluate_selection_candidate(*, runner, model, accepted_parameter_set, parameters, suite, extractor, loss) -> SelectionCandidateShard`
  - `assemble_selection_validation_report(*, model, accepted_parameters, suite, extractor, loss, max_mean_loss=None, max_worst_loss=None, shards) -> SelectionValidationReport`
  - payload round-trip with declared hash verification.

- [ ] **Step 1: Add test-only RED against untouched `select_on_validation_suite`**

Use a module-level deterministic `ScaledLevelModel`, `value_metrics`, a `HeldOutSuite` with `EvaluationRole.SELECTION_VALIDATION`, and a `ParameterAcceptanceSet` of three candidates. Compute the untouched sequential reference with `select_on_validation_suite`; evaluate one shard per candidate; pass shards to the assembler in reversed order; assert exact equality of:

```python
self.assertEqual(assembled.candidate_report, reference.candidate_report)
self.assertEqual(assembled.selected_parameters, reference.selected_parameters)
self.assertEqual(assembled.manifest, reference.manifest)
self.assertEqual(assembled.manifest.content_hash, reference.manifest.content_hash)
```

Add missing/duplicate/extra/tampered shard rejection tests.

- [ ] **Step 2: Run focused RED**

```bash
python3 -m unittest tests.test_candidate_parallel_selection -v
```

Expected: FAIL because `selection_shards` is absent.

- [ ] **Step 3: Implement selection shard by reusing `validate_held_out`**

`evaluate_selection_candidate` must validate `suite.role is SELECTION_VALIDATION` and candidate membership in the exact accepted set, then call:

```python
validation = validate_held_out(
    runner=runner,
    model=model,
    parameters=dict(parameters),
    cases=suite.cases,
    extractor=extractor,
    loss=loss,
)
```

The shard binds accepted-set hash, suite name/role/case identity, model/repository identity, metric/loss identity, parameters, and the complete `HeldOutValidationReport` identity. No retained/selected decision occurs in a worker.

- [ ] **Step 4: Implement exact acceptance and selection assembly**

Reproduce the current `validate_acceptance_set_held_out` and `select_on_validation_suite` logic exactly:

1. sort candidate evaluations by `(mean_loss, worst_loss, parameters)`;
2. apply the current `max_mean_loss` / `max_worst_loss` filters;
3. build validation parent hashes in sorted evaluation order;
4. build `ExperimentStage.ACCEPTANCE_VALIDATION` manifest with unchanged inputs;
5. build retained `ParameterAcceptanceSet` with the acceptance manifest hash;
6. select the first ranked retained candidate;
7. build `ExperimentStage.SELECTION_VALIDATION` manifest with the acceptance report hash as its only parent.

The assembler must reject any shard whose embedded held-out report does not match the candidate, suite case set, repository/model identity, metric/loss, or accepted-set identity.

- [ ] **Step 5: Run focused GREEN and commit**

```bash
python3 -m unittest tests.test_candidate_parallel_selection tests.test_runtime_trust_boundary -v
```

Expected: PASS and exact manifest equality with the sequential reference.

Commit:

```bash
git add tests/test_candidate_parallel_selection.py narrative_dynamics/observations/selection_shards.py narrative_dynamics/observations/__init__.py
git commit -m "feat: add deterministic selection candidate shards"
```

---

### Task 3: Operational Sequential and Spawn-Process Executors

**Files:**
- Create: `tests/test_candidate_process_executor.py`
- Create: `narrative_dynamics/candidate_execution.py`

**Interfaces:**
- Produces:
  - `CandidateExecutionError(RuntimeError)`
  - `CandidateExecutor` protocol with `execute(worker, tasks, *, initializer=None, initargs=()) -> tuple`
  - `SequentialCandidateExecutor`
  - `ProcessCandidateExecutor(max_workers: int | None = None)`

- [ ] **Step 1: Add RED for a real process backend**

Define only module-level picklable worker functions in the test file:

```python
def process_probe(value: int) -> tuple[int, int]:
    return value * value, os.getpid()


def failing_probe(value: int) -> int:
    if value == 2:
        raise ValueError("candidate boom")
    return value
```

Tests must assert:

- sequential and process executors return task results in input task order;
- `ProcessCandidateExecutor(max_workers=2)` uses a child PID for at least one result;
- `max_workers=1` is valid;
- non-positive worker counts are rejected;
- a worker exception raises `CandidateExecutionError` and no result tuple is returned;
- initializer/initargs run under both backends.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_candidate_process_executor -v
```

Expected: module/API missing.

- [ ] **Step 3: Implement executor layer with `spawn`**

Use `concurrent.futures.ProcessPoolExecutor` with:

```python
context = multiprocessing.get_context("spawn")
effective = min(
    requested if requested is not None else (os.cpu_count() or 1),
    os.cpu_count() or 1,
    len(tasks),
)
```

For process execution, submit all tasks but collect `future.result()` in canonical task/input order, not completion order. On any exception, cancel remaining futures where possible and raise `CandidateExecutionError` from the original exception. Do not return partial results.

`SequentialCandidateExecutor` must call the initializer exactly once before evaluating tasks and preserve the same task ordering/error contract.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest tests.test_candidate_process_executor -v
```

Commit:

```bash
git add tests/test_candidate_process_executor.py narrative_dynamics/candidate_execution.py
git commit -m "feat: add bounded candidate process executor"
```

---

### Task 4: End-to-End Generic Sequential vs Process Scientific Equivalence

**Files:**
- Modify: `tests/test_candidate_parallel_training.py`
- Modify: `tests/test_candidate_parallel_selection.py`

**Interfaces:**
- Consumes Task 1/2 shard evaluators and Task 3 executors.
- Produces proof that process scheduling does not alter scientific identity.

- [ ] **Step 1: Add module-level picklable synthetic task records/workers**

The worker task must contain only explicit serializable scientific inputs needed by the fixture. The worker reconstructs a fresh `SimulationRunner` and deterministic model; it does not capture closures or rely on inherited globals.

- [ ] **Step 2: Add equality tests for executor worker counts**

For training and selection separately, compare:

```text
untouched sequential reference
== shard assembly via SequentialCandidateExecutor
== shard assembly via ProcessCandidateExecutor(max_workers=1)
== shard assembly via ProcessCandidateExecutor(max_workers=2 or 4)
```

Require exact report/manifest equality, not approximate loss equality.

- [ ] **Step 3: Add completion-order independence test without timing assertions**

Use a fake executor that deliberately returns valid shard values in reverse order and prove the assemblers still produce the exact reference manifests. Do not use sleeps or wall-clock ordering as a test oracle.

- [ ] **Step 4: Run focused process tests**

```bash
python3 -m unittest \
  tests.test_candidate_parallel_training \
  tests.test_candidate_parallel_selection \
  tests.test_candidate_process_executor -v
```

Expected: PASS under Linux GitHub Actions and local spawn-capable Python.

- [ ] **Step 5: Commit**

```bash
git add tests/test_candidate_parallel_training.py tests/test_candidate_parallel_selection.py
git commit -m "test: prove candidate process scientific equivalence"
```

---

### Task 5: Feher/Hare Explicit Parallel TRAIN/SELECTION Wrapper and Seed Firewall

**Files:**
- Create: `tests/test_feher_hare_parallel_training.py`
- Create: `narrative_dynamics/studies/feher_hare_two_stage_parallel.py`
- Modify: `narrative_dynamics/studies/__init__.py`

**Interfaces:**
- Consumes: existing `prepare_feher_hare_two_stage_v1`, `_family_sources`, `REACTIVE_GRID`, `HISTORY_GRID`, `TRAIN_SEEDS`, `SELECTION_SEEDS`, `FINAL_SEEDS`, shard evaluators/assemblers, executors, `FrozenModelSpec`, `FeherHareFamilyFreeze`, `FrozenFeherHareModels`.
- Produces:
  - `fit_and_freeze_feher_hare_models_parallel(*, root: Path, manifest: TwoStageSourceManifest, repository_identity: RepositoryIdentity, executor: CandidateExecutor) -> FrozenFeherHareModels`
  - internal serializable `FeherHareCandidateTask`
  - explicit spawn initializer reconstructing immutable study/source/runner context.

- [ ] **Step 1: Add synthetic RED comparing old and new study-level APIs**

Build a synthetic two-stage checkout with enough participants for nonempty TRAIN/SELECTION/FINAL partitions. Use one explicit deterministic `RepositoryIdentity` for both reference and parallel runs:

```python
identity = RepositoryIdentity(
    provider="test",
    repository="qigao/lean",
    checkout_commit="a" * 40,
    source_commit="a" * 40,
    ref="test",
    dirty=False,
)
reference = study.fit_and_freeze_feher_hare_models(
    runner=SimulationRunner(repository_identity=identity),
    prepared=prepared,
)
parallel = fit_and_freeze_feher_hare_models_parallel(
    root=root,
    manifest=manifest,
    repository_identity=identity,
    executor=SequentialCandidateExecutor(),
)
self.assertEqual(parallel.frozen_candidates, reference.frozen_candidates)
self.assertEqual(parallel.training_manifest_hashes, reference.training_manifest_hashes)
self.assertEqual(parallel.selection_manifest_hashes, reference.selection_manifest_hashes)
```

- [ ] **Step 2: Add RED for FINAL isolation**

The public parallel wrapper must expose no final role/task. Add a runner/executor audit seam in the test that records requested shard phase/seed plans and assert exactly:

```python
self.assertEqual(training_seed_set, {101, 102})
self.assertEqual(selection_seed_set, {201, 202})
self.assertTrue({301, 302}.isdisjoint(all_requested_seeds))
```

Protocol bundle construction after freezing may be tested, but no call to any final evaluation API is allowed.

- [ ] **Step 3: Run RED**

```bash
python3 -m unittest tests.test_feher_hare_parallel_training -v
```

Expected: parallel study module/API missing.

- [ ] **Step 4: Implement explicit worker context reconstruction**

Use a module-level initializer, invoked explicitly by the executor, that reconstructs:

- the supplied `TwoStageSourceManifest`;
- the pinned/local source snapshot at `root`;
- `PreparedFeherHareTwoStageV1`;
- the selected family source by canonical family name;
- `SimulationRunner(repository_identity=repository_identity)`.

The context may live in a module-level worker variable only after explicit initializer execution. It must not depend on state inherited through `fork`, and process tests must use the spawn backend.

`FeherHareCandidateTask` supports only two closed phases: `train` and `selection_validation`; there is no `final_test` value. Seeds are derived internally from phase and cannot be supplied by a task payload.

- [ ] **Step 5: Implement family-by-family training and selection assembly**

For each family from the frozen `_family_sources()` declaration:

1. derive the exact canonical candidate set from its existing grid;
2. execute one TRAIN task per candidate with the executor;
3. assemble exact `TrainingFitReport`;
4. create `ParameterAcceptanceSet` exactly as current V1 does;
5. execute one SELECTION task per accepted candidate;
6. assemble exact `SelectionValidationReport`;
7. freeze `FrozenModelSpec.from_selection`;
8. create `FeherHareFamilyFreeze` and finally `FrozenFeherHareModels`.

Do not call `build_feher_hare_protocol_bundle` inside worker processes and never call any FINAL execution API.

- [ ] **Step 6: Prove process backend equality on synthetic study**

Run the same fixture with `ProcessCandidateExecutor(max_workers=2)` and require exact equality with the old sequential reference for all three family training manifests, selection manifests, frozen parameters, and frozen candidate hashes.

- [ ] **Step 7: Export and commit**

```bash
git add \
  tests/test_feher_hare_parallel_training.py \
  narrative_dynamics/studies/feher_hare_two_stage_parallel.py \
  narrative_dynamics/studies/__init__.py
git commit -m "feat: add parallel feher hare train selection"
```

---

### Task 6: Failure, Forgery, and Repository-Attestation Regression Gates

**Files:**
- Modify: `tests/test_candidate_parallel_training.py`
- Modify: `tests/test_candidate_parallel_selection.py`
- Modify: `tests/test_feher_hare_parallel_training.py`

**Interfaces:**
- Consumes all V2 shard/executor APIs.
- Produces fail-closed evidence required before performance use.

- [ ] **Step 1: Add forged training shard tests**

Use `dataclasses.replace` to mutate one field at a time and require rejection before report construction for:

- target report hash;
- model identity;
- repository identity;
- seed plan;
- metric identity;
- loss identity;
- case name/coverage;
- parameters;
- run-manifest lineage.

- [ ] **Step 2: Add forged selection shard tests**

Require rejection for accepted-set hash, suite identity, repository/model identity, candidate parameters, validation case coverage, and validation manifest drift.

- [ ] **Step 3: Add process failure atomicity test**

Inject one task that raises and assert the caller gets `CandidateExecutionError`; no assembler is called with a partial shard tuple and no authoritative report exists.

- [ ] **Step 4: Add repository identity equality test across process workers**

Use an explicit test `RepositoryIdentity` and assert every returned shard binds exactly `identity.manifest_identity()`, independent of worker PID.

- [ ] **Step 5: Run all V2 focused tests and commit**

```bash
python3 -m unittest \
  tests.test_candidate_parallel_training \
  tests.test_candidate_parallel_selection \
  tests.test_candidate_process_executor \
  tests.test_feher_hare_parallel_training -v
```

Commit:

```bash
git add tests/test_candidate_parallel_training.py tests/test_candidate_parallel_selection.py tests/test_feher_hare_parallel_training.py
git commit -m "test: harden candidate parallel failure boundaries"
```

---

### Task 7: Full Verification and Performance Evidence

**Files:**
- Create: `tools/benchmark_candidate_parallel.py`
- No scientific production changes after the exact-head verification candidate is chosen.

**Interfaces:**
- Consumes the final V2 public/explicit APIs.
- Produces non-authoritative operational timing evidence only.

- [ ] **Step 1: Add deterministic benchmark driver**

The benchmark must use a synthetic CPU-bound model and a fixed finite candidate set. It reports elapsed time for:

- sequential reference;
- `ProcessCandidateExecutor(max_workers=2)`;
- `ProcessCandidateExecutor(max_workers=min(4, os.cpu_count() or 1))`.

Before printing any timing, assert the assembled scientific manifest hashes are identical across all runs. Timing, PID, and worker count are printed only and never written into a scientific manifest.

- [ ] **Step 2: Run focused V2 suite**

```bash
python3 -m unittest \
  tests.test_candidate_parallel_training \
  tests.test_candidate_parallel_selection \
  tests.test_candidate_process_executor \
  tests.test_feher_hare_parallel_training -v
```

Expected: PASS.

- [ ] **Step 3: Run complete repository verification**

```bash
python3 -m unittest discover -s tests -v
lake build
```

Expected: full Python suite `OK`; Lean build success.

- [ ] **Step 4: Run benchmark outside ordinary CI acceptance**

```bash
python3 tools/benchmark_candidate_parallel.py
```

Require hash equality first. Record observed wall-clock speedup as operational evidence; do not fail ordinary CI on a fixed timing ratio.

- [ ] **Step 5: Exact-head CI and scope review**

Push the exact implementation head and require both Python and Lean jobs success. Compare against `de09518d5d034b14a7a22545af02117769297517` and verify no changes to:

- Feher/Hare source manifest;
- participant split rules;
- model parameter grids;
- TRAIN/SELECTION/FINAL seeds;
- Brier/Log metrics or thresholds;
- claim scope;
- FINAL release/witness gates.

- [ ] **Step 6: Commit benchmark if not already committed**

```bash
git add tools/benchmark_candidate_parallel.py
git commit -m "perf: add candidate parallel benchmark"
```

After this task is exact-head GREEN, the performance implementation is complete. A new real empirical study revision may then deliberately freeze this implementation and rerun source/readiness/preregistration before using it for real TRAIN/SELECTION; do not reuse the V1 registration or V1 repository revision as if no scientific-code revision occurred.
