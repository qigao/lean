# Calibrated ABM Experiment Suite V9 Implementation Plan

> **For agentic workers:** Execute with strict red/green tests, then run the complete V1–V9 audit before delivery.

**Goal:** Orchestrate paired, calibrated holdout experiments with a locked baseline, declared primary metric/direction, auditable case outcomes, and effect deltas.

**Architecture:** A protocol binds named parameter arms and a mandatory V8-selected baseline. The runner replays every arm over every holdout case, records final snapshots/state hashes, aggregates the primary metric, computes signed baseline deltas, and validates objective-specific ranking and complete pairing.

**Spec:** `docs/superpowers/specs/2026-08-31-calibrated-abm-experiment-suite-v9-design.md`

### Task 1: Protocol and experiment report

**Files:** Create `narrative_dynamics/abm/experiments.py`; create `tests/test_network_abm_experiments.py`; modify public API files.

- [ ] Write failing protocol, paired execution, baseline lock, objective ranking, replay error, canonicalization, and report-tamper tests.
- [ ] Confirm import RED with `python -m unittest tests.test_network_abm_experiments -v`.
- [ ] Implement `ExperimentObjective`, `ABMExperimentArm`, `ABMExperimentProtocol`, `ABMExperimentCaseResult`, `ABMExperimentArmResult`, `CalibratedABMExperimentReport`, and `run_calibrated_abm_experiment`.
- [ ] Validate exact V8 model/dataset/report binding, HOLDOUT-only pairing, aggregates, deltas, and ranking.
- [ ] Run V9 and V8 regression tests and commit `feat: run calibrated ABM experiments`.

### Task 2: Documentation and full delivery audit

**Files:** Modify `README.md`; modify exact public API test.

- [ ] Add an independent V9 example calibrating a baseline and comparing a treatment on holdout cases.
- [ ] Execute all README Python blocks independently.
- [ ] Run all network ABM tests, 40 targeted narrative regressions, compileall, diff check, placeholder scan, git status, and commit audit.
- [ ] Commit `docs: expose calibrated ABM experiments`, push, and update PR #44 with V1–V9 evidence.
