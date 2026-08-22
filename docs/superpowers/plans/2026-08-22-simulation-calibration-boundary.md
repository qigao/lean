# Simulation–Calibration Boundary Implementation Plan

> **Status:** Completed on `proof/narrative-dynamics-v0`.

**Goal:** Provide a dependency-free Python research runtime that can execute, calibrate, externally validate, and diagnose existing or black-box models while Lean remains the formal semantic and invariant layer.

**Architecture:** Models implement the canonical `SimulatorModel` protocol and are executed only by `SimulationRunner`, which owns seeds and trace metadata. Calibration and diagnostics consume canonical traces and metric extractors. Model registration provides discovery only and never executes an adapter.

**Tech Stack:** Python standard library, `unittest`, Lean 4.32.0, Mathlib.

**Spec:** `docs/superpowers/specs/2026-08-22-simulation-calibration-boundary-design.md`

## Completed deliverables

- [x] Immutable scenario, event, model-run, and trace contracts.
- [x] Deterministic seeded single and batch simulation.
- [x] Stable metric aggregation and weighted loss.
- [x] Exhaustive finite-grid calibration with deterministic tie breaking.
- [x] Synthetic recovery with explicit non-recovery for non-identifiable metrics.
- [x] Grounded-goal adapter that reuses the existing formally aligned softmax.
- [x] Repeated calibration across independent seed blocks.
- [x] Explicit parameter acceptance sets and coordinate-wise identifiability.
- [x] Seed-block variation with accepted-set union/intersection.
- [x] Held-out scenario validation with per-case, mean, and worst loss.
- [x] External filtering of accepted candidates.
- [x] Common-seed central-difference metric sensitivity.
- [x] Held-out loss slope and curvature diagnostics.
- [x] Explicit model registry for generic, planner, POMDP, rule-engine, and population adapters.
- [x] Top-level package exports and documentation.
- [x] Full Lean build, theorem regression, and Python suite in GitHub Actions.

## Boundary retained

- Lean proves formal identities, admissibility, provenance, and invariants.
- Python runs numerical or third-party models and produces validation evidence.
- Registration and calibration do not make a model empirically true.
- Local sensitivity does not establish global identifiability.
- Held-out validation never recalibrates the candidate under evaluation.
- Third-party adapters still execute only through `SimulationRunner`.

## Next independent milestone

Create a model-comparison protocol and integrate the first mature algorithm adapters. Candidate first integrations are a finite POMDP policy adapter and a planner adapter. Each integration must use canonical traces, register explicitly, support seeded replay where applicable, and be compared on held-out scenarios rather than only in-sample fit.
