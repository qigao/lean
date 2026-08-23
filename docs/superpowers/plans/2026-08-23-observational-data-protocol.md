# Versioned Observational-Data Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add immutable observational datasets, role-separated target construction, content-hashed preregistration, and exact fair final-test comparison around the existing Python model runtime.

**Architecture:** Create a focused `narrative_dynamics.observations` package with separate dataset, target, preregistration, and comparison modules. Reuse existing scenarios, losses, metric identities, selection/final-test reports, manifests, and report artifacts; do not change the prison model or Lean code.

**Tech Stack:** Python 3 standard library, `dataclasses`, `enum`, `json`, `unittest`, existing `narrative_dynamics` runtime.

**Spec:** `docs/superpowers/specs/2026-08-23-observational-data-protocol-design.md`

## Global Constraints

- Python-only; do not modify any `.lean` file.
- Do not add dependencies.
- Do not add prison states, actions, parameters, or horizons.
- Dataset source observations must be disjoint across train, selection-validation, and final-test roles.
- Target construction must retain raw counts, source observation ids, dataset/partition hashes, grouping identity, and seed plan.
- Final comparison must execute only frozen parameters selected before final evaluation.
- Every new production behavior must be preceded by a failing test.
- Every new aggregate report must carry `ExperimentManifest` lineage and work with `attest_report()`.

---

### Task 1: Immutable versioned datasets and partitions

**Files:**
- Create: `narrative_dynamics/observations/__init__.py`
- Create: `narrative_dynamics/observations/dataset.py`
- Test: `tests/test_observation_dataset.py`

**Interfaces:**
- Produces: `OBSERVATION_DATASET_SCHEMA_VERSION`, `ObservationPartitionRole`, `ObservationCase`, `ObservationPartition`, `ObservationDataset`, `load_observation_dataset`.

- [ ] **Step 1: Write failing dataset tests**

Test a round trip with one case in each role, mutation detachment, stable hash under input ordering, and exact payload reproduction. Add separate tests for missing roles, duplicate case names, duplicate observation ids across roles, unknown JSON fields, invalid counts, and a declared content hash mismatch.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest discover -s tests -p 'test_observation_dataset.py' -v
```

Expected: import failure for `narrative_dynamics.observations`.

- [ ] **Step 3: Implement dataset values**

Implement strict immutable dataclasses and typed SHA-256 identities. Canonicalize cases by `(role.value, name)`. Require nonempty train, selection-validation, and final-test partitions. Reject globally repeated observation ids.

- [ ] **Step 4: Implement strict payload I/O**

`ObservationDataset.to_payload()` emits only schema version, name, version, source, cases, and content hash. `from_payload()` rejects unknown or missing fields, rebuilds `Scenario`, verifies a declared content hash, and never trusts stored ordering. `load_observation_dataset(path)` reads UTF-8 JSON and wraps decode/file errors with explicit `ValueError` messages.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest discover -s tests -p 'test_observation_dataset.py' -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/observations tests/test_observation_dataset.py
git commit -m "feat: add versioned observational datasets"
```

---

### Task 2: Provenance-preserving categorical targets

**Files:**
- Create: `narrative_dynamics/observations/targets.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Modify: `narrative_dynamics/contracts.py`
- Test: `tests/test_observation_targets.py`

**Interfaces:**
- Consumes: dataset values and `CategoricalMetricGroup`.
- Produces: `CategoricalTargetPlan`, `ConstructedTargetCase`, `TargetConstructionReport`, `construct_categorical_targets`.

- [ ] **Step 1: Write failing target tests**

Construct selection and final partitions from raw counts. Assert exact normalized targets, raw-count preservation, source ids, plan/dataset/partition hashes, manifest stage, suite role mapping, and `attest_report(report).require_integrity()`. Add failure tests for incomplete/duplicate metric groups, wrong group totals, missing/extra/empty seed plans, and train-to-held-out conversion.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest discover -s tests -p 'test_observation_targets.py' -v
```

Expected: missing target-construction APIs and stage.

- [ ] **Step 3: Implement plan and report**

Add `ExperimentStage.TARGET_CONSTRUCTION`. Validate exact count-key coverage and require each group total to equal `len(observation_ids)`. Normalize counts by group total and retain raw values plus source ids in each constructed case.

- [ ] **Step 4: Convert only evaluation roles**

`TargetConstructionReport.as_held_out_suite()` maps selection-validation and final-test roles to existing `EvaluationRole` values and creates exact `HeldOutCase` values. It rejects train-role reports.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest discover -s tests -p 'test_observation_targets.py' -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/observations narrative_dynamics/contracts.py tests/test_observation_targets.py
git commit -m "feat: construct observation targets with provenance"
```

---

### Task 3: Frozen models and preregistered thresholds

**Files:**
- Create: `narrative_dynamics/observations/preregistration.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Test: `tests/test_comparison_preregistration.py`

**Interfaces:**
- Produces: `FrozenModelSpec`, `AdequacyThresholds`, `ComparisonPreregistration`.

- [ ] **Step 1: Write failing preregistration tests**

Use real `SelectionValidationReport` values to freeze two model specifications. Assert selected parameters, model identity, selection manifest lineage, threshold identity, deterministic model ordering, and registration hash. Add failures for non-selection reports, mismatched explicit parameters, duplicate model names, invalid hashes, negative thresholds, non-final target reports, and target/dataset mismatch.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest discover -s tests -p 'test_comparison_preregistration.py' -v
```

Expected: missing preregistration APIs.

- [ ] **Step 3: Implement frozen model specs**

`FrozenModelSpec.from_selection(name, model, report)` requires `EvaluationRole.SELECTION_VALIDATION`, copies `report.selected_parameters`, captures `component_identity(model)`, and retains the selection manifest hash. All mappings and parameter values are canonical and finite.

- [ ] **Step 4: Implement registration identity**

`ComparisonPreregistration.create(...)` requires a final `TargetConstructionReport`, recomputes metric and loss identities, binds the dataset and partition hashes, thresholds, and sorted model specs, and exposes a stable content hash plus fixed ranking rule.

- [ ] **Step 5: Verify GREEN and commit**

```bash
python3 -m unittest discover -s tests -p 'test_comparison_preregistration.py' -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/observations tests/test_comparison_preregistration.py
git commit -m "feat: preregister frozen model comparisons"
```

---

### Task 4: Exact fair alternative-model comparison

**Files:**
- Create: `narrative_dynamics/observations/comparison.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Modify: `narrative_dynamics/contracts.py`
- Modify: `narrative_dynamics/__init__.py`
- Test: `tests/test_alternative_model_comparison.py`

**Interfaces:**
- Produces: `AlternativeModelEvaluation`, `AlternativeModelComparisonReport`, `compare_registered_models`.

- [ ] **Step 1: Write failing fair-comparison test**

Create one versioned dataset, construct selection/final targets, freeze two independently named model sources from selection reports, preregister thresholds, and compare on final data. Assert the better model ranks first, both child final-test reports use the same target suite/loss/metric, adequacy uses only registered thresholds, manifest lineage is complete, and `attest_report()` succeeds.

- [ ] **Step 2: Add preflight failure tests**

Use a counting model to prove that target-report, loss, metric, model-name-set, or model-identity mismatch fails before any model executes. Also test missing and extra models and input mapping order independence.

- [ ] **Step 3: Run RED**

```bash
python3 -m unittest discover -s tests -p 'test_alternative_model_comparison.py' -v
```

Expected: missing comparison APIs and stage.

- [ ] **Step 4: Implement complete preflight and evaluation**

Add `ExperimentStage.ALTERNATIVE_MODEL_COMPARISON`. Validate every registration boundary before the first run. Evaluate each frozen spec through `evaluate_on_final_test_suite()` with identical suite, extractor, and loss. Rank by `(mean_loss, worst_loss, name)` and retain child final-test manifests plus selection lineage.

- [ ] **Step 5: Export public API, verify, and commit**

```bash
python3 -m unittest discover -s tests -p 'test_alternative_model_comparison.py' -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics tests/test_alternative_model_comparison.py
git commit -m "feat: compare preregistered alternative models"
```

---

### Task 5: Review, CI, and integration

**Files:**
- Modify only when a newly written regression demonstrates a defect.
- Update: feature PR description and main PR #2 description.

- [ ] **Step 1: Run syntax and focused protocol tests**

```bash
python3 -m compileall -q narrative_dynamics tests
python3 -m unittest discover -s tests -p 'test_observation_*.py' -v
python3 -m unittest discover -s tests -p 'test_*preregistration.py' -v
python3 -m unittest discover -s tests -p 'test_alternative_model_comparison.py' -v
```

- [ ] **Step 2: Run the complete Python suite**

```bash
python3 -m unittest discover -s tests -v
```

- [ ] **Step 3: Review leakage and identity claims**

Confirm that source observation ids are disjoint across roles, target construction is bound to exact dataset/partition/plan identities, all comparison preflight happens before model execution, and documentation does not claim temporal preregistration proof or population representativeness.

- [ ] **Step 4: Require feature workflow GREEN**

Require:

```text
Lean Python conformance vectors
Build Lean library
Lean theorem tests
Python numerical tests
```

- [ ] **Step 5: Integrate only by non-force fast-forward**

Fast-forward `proof/narrative-dynamics-v0` only after feature CI is green, then require a separate PR #2 merge-context workflow to pass on the same head. Update both PR descriptions with exact RED/GREEN run numbers, final SHA, test count, and remaining limits.
