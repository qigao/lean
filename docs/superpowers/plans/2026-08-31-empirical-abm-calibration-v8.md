# Empirical ABM Calibration V8 Implementation Plan

> **For agentic workers:** Execute task-by-task with strict red/green tests and frequent commits. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit V7 learning and rewiring parameters to frozen empirical trajectories with exhaustive training-only selection and untouched holdout reporting.

**Architecture:** Immutable dataset contracts carry paired V7 schedules and macro observations. A canonical finite candidate replaces only three declared nested model parameters. The evaluator replays every candidate/case, projects existing system metrics, scores weighted trajectory MSE, ranks only by mean training loss, and emits a fully validated report containing holdout results and snapshot hashes.

**Tech Stack:** Python 3 standard library, frozen dataclasses, V7 runtime/metrics, stable content hashing, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-empirical-abm-calibration-v8-design.md`

## Global Constraints

- TRAIN selects; HOLDOUT only reports.
- Every candidate runs every case exactly once.
- Only learning rate and formation/dissolution thresholds vary.
- Candidate application preserves all other model values and never mutates input.
- Observed schedules and snapshots are frozen, canonical, and complete.
- Replay mismatch fails with candidate/case identity; no hidden imputation or penalty.
- Loss is finite weighted trajectory MSE with at least one positive declared weight.
- All ranking, case, candidate, and report identities are canonical and hash-stable.

---

### Task 1: Observation, dataset, candidate, and weight contracts

**Files:**
- Create: `narrative_dynamics/abm/calibration_contracts.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/abm_calibration_fixtures.py`
- Test: `tests/test_network_abm_calibration_contracts.py`

**Interfaces:**
- Consumes: V7 events/truth values, V5/V6/V7 metric contracts, validation helpers, and stable hashing.
- Produces: `CalibrationSplit`, `ObservedABMSnapshot`, `EmpiricalABMCase`, `EmpiricalABMDataset`, `ABMCalibrationCandidate`, and `ABMCalibrationWeights`.

- [ ] Write failing bounded-observation, paired-schedule, split, threshold, weight, canonicalization, and immutability tests.
- [ ] Run `python -m unittest tests.test_network_abm_calibration_contracts -v` and confirm import RED.
- [ ] Implement frozen validated contracts, canonical tuples, exact round coverage, `to_dict()`, and `content_hash`.
- [ ] Rerun contract tests and confirm GREEN.
- [ ] Commit with `git commit -m "feat: define empirical ABM calibration contracts"`.

### Task 2: Snapshot projection, candidate replay, scoring, and report

**Files:**
- Create: `narrative_dynamics/abm/calibration.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Test: `tests/test_network_abm_calibration.py`

**Interfaces:**
- Consumes: Task 1 contracts, V7 simulation, existing V5/V6/V7 metrics, and nested immutable model constructors.
- Produces: `observe_dynamic_role_state`, `apply_calibration_candidate`, `ABMCaseCalibrationFit`, `ABMCandidateCalibrationFit`, `EmpiricalABMCalibrationReport`, and `calibrate_dynamic_role_model`.

- [ ] Write failing synthetic recovery, holdout isolation, lexical tie, complete coverage, reconstruction, replay-failure, and input-order tests.
- [ ] Run `python -m unittest tests.test_network_abm_calibration -v` and confirm import RED.
- [ ] Implement one canonical snapshot projection by composing existing metrics.
- [ ] Implement immutable nested candidate reconstruction and exact case replay.
- [ ] Implement weighted trajectory MSE and validated complete fit/report contracts.
- [ ] Rank by `(training_loss, candidate tuple)` only and preserve holdout scores separately.
- [ ] Run V8 runtime plus V7 regressions and confirm GREEN.
- [ ] Commit with `git commit -m "feat: calibrate dynamic-role ABM"`.

### Task 3: Public API, executable example, regression, and delivery

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `README.md`
- Modify: `tests/test_network_abm_public_api.py`

- [ ] Extend the exact V8 public surface.
- [ ] Add an independent README example that generates train/holdout observations from a known candidate, calibrates a two-candidate grid, and asserts recovery plus holdout reporting.
- [ ] Execute every README Python block independently.
- [ ] Run all `test_network_abm*.py`, the 40 targeted narrative regressions, `compileall`, and base diff checks.
- [ ] Commit with `git commit -m "docs: expose empirical ABM calibration"`.
- [ ] Push the feature branch and update PR #44 with V8 semantics and fresh evidence.
