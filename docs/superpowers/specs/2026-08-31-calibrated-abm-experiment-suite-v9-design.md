# Calibrated ABM Experiment Suite V9 Design

## Objective

Run preregistered, paired experiment arms on the exact V8 holdout cases after calibration. The V8 selected candidate is the mandatory baseline. Every arm replays every holdout case, reports the same declared final-snapshot metric, and stores its signed mean difference from baseline. Training cases never enter experiment outcomes.

## Protocol

`ABMExperimentProtocol` declares a stable id/version, a unique baseline arm id, one primary metric from the V8 snapshot schema, an objective (`MINIMIZE` or `MAXIMIZE`), and at least two canonical `ABMExperimentArm` values. Each arm binds an id to one V8 candidate; arm and candidate identities are unique.

The baseline arm candidate must equal the selected candidate in the exact V8 report. This prevents silently replacing the calibrated reference model.

## Execution

For every arm and every HOLDOUT case, V9 rebuilds the candidate model, initializes the frozen beliefs, and replays the exact event/truth schedules. It records the final `ObservedABMSnapshot` and final dynamic-state hash. Replay failures name both arm and case and stop the suite.

`ABMExperimentArmResult` contains the complete canonical holdout case results, arithmetic mean primary outcome, and signed `mean_outcome - baseline_mean`. The baseline delta is exactly zero.

`CalibratedABMExperimentReport` binds the base-model, dataset, V8 calibration, and protocol hashes; exact selected candidate; complete arm results; and ranking. Ranking follows the declared objective and breaks ties by arm id. The report validates paired case coverage, outcome aggregates, deltas, and the baseline identity.

## Acceptance scenarios

1. Baseline and treatment run every holdout case exactly once and no training case.
2. Baseline candidate must be the V8 selected candidate.
3. Signed treatment effect equals treatment mean minus baseline mean.
4. `MAXIMIZE` and `MINIMIZE` produce opposite rankings from the same outcomes.
5. Arm/case input order cannot change report content or hash.
6. Invalid metric, missing baseline, duplicate arm/candidate, mismatched model/dataset/calibration, incomplete pairing, or inconsistent aggregate fails closed.
7. Candidate replay failure names arm and case.
8. Final snapshot and final state hash remain available per paired case.

## Deferred work

- stochastic replicates and uncertainty intervals;
- factorial interaction estimators and multiple-testing correction;
- distributed/parallel execution;
- intervention-specific model transformations beyond calibrated parameter arms;
- causal interpretation without randomized or otherwise identified designs.
