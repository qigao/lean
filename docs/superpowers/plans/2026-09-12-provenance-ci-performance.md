# Provenance CI Performance Implementation Plan

> For agentic workers: use superpowers:executing-plans task by task.

**Goal:** Remove per-proposal whole-graph copying and repeated unchanged control generation.
**Architecture:** Keep the current sampler/RNG semantics; make provenance generation a cached, separately verified step after tests. A corrupt/stale exact cache is an error, never a substitute graph.
**Tech stack:** Python, pytest, GitHub Actions.
**Spec:** Issue #68 and `protocols/v0-real-ntu120-preflight.json`.

## Global constraints

Preserve seeds 7/11/19/23/31; 10 successful swaps per off-diagonal edge;
exact diagonal edge+weight set; directed degrees; weight multiset; attempt budget.
No real-data training, final-test access, threshold changes or compatibility path.

## Task 1: Exact-output performance regression

Files: `tests/test_rewiring_performance.py`, `python/yolo_flywire/graphs.py`.
- [ ] Submit test-only oracle of the current copying sampler and verify RED for bulk copying/progress.
- [ ] Replace `edge_set - {old_one, old_two}` with membership checks excluding those two tuples.
- [ ] Keep RNG calls and acceptance order unchanged; log initial, periodic (100,000 attempts), and final accepted/attempt counts.
- [ ] Run `python -m pytest tests -q`; compare exact ordered output and fingerprints against the test oracle for all five seeds on sparse/dense diagonal graphs.

## Task 2: Generate once, verify every run

Files: `python/yolo_flywire/provenance.py`, `.github/workflows/yolo-flywire-ci.yml`, same regression test file.
- [ ] RED: cache verification must not call the sampler; stale identities, missing seeds, wrong fingerprints, changed weights and altered diagonals must fail.
- [ ] Add `generate_controls(graph, identity)` and `verify_controls(bundle, graph, identity)` with complete graph serialization, fingerprint and invariant checks.
- [ ] Add CLI `key`, `generate`, `verify`, each validating source bytes and selected graph against the frozen protocol. Identity binds canonical protocol, graph fingerprint, generator source hashes, full Python/NumPy/platform identity.
- [ ] Use exact cache keys only, save only after validation, verify on every hit, and log per-seed durations. No partial-key reuse.
- [ ] Make provenance depend on Lean/Python success, set a ten-minute timeout and cancel superseded runs in the same workflow/ref group.
- [ ] Require exact-head CI with actual pinned FlyWire CSV and all five full-budget controls. Exercise a warm-cache run and record evidence in #68.

Verification commands:
```sh
python -m pytest tests -q
PYTHONPATH=python python -m yolo_flywire.provenance generate --config protocols/v0-real-ntu120-preflight.json --source flywire-connectivity.csv --output .cache/flywire-controls
PYTHONPATH=python python -m yolo_flywire.provenance verify --config protocols/v0-real-ntu120-preflight.json --source flywire-connectivity.csv --output .cache/flywire-controls
lake build
```
Only claim the checks actually executed; the local DNS failure does not substitute for GitHub CI.
