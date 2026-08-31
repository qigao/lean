# Empirical ABM Calibration V8 Design

## Objective

Calibrate the V7 dynamic-role network ABM against observed multi-round macro trajectories with a finite, preregistered parameter grid and a strict training/holdout boundary. V8 estimates the V5 learning rate, dissolution threshold, and formation threshold while keeping catalog identity, role policies, transition rules, initial trust, event schedules, and truth-observation schedules fixed.

The output is an immutable audit report containing every candidate, every case loss, the training-only ranking, the selected candidate, and its untouched holdout loss. V8 is deterministic exhaustive calibration, not stochastic optimization or causal identification.

## Empirical observation contract

`ObservedABMSnapshot` stores one positive round index and ten bounded macro observables:

- active population share;
- mean active belief, defined as zero when no agent is active;
- mean edge trust;
- learned-edge rate;
- effective active-edge rate;
- ever-rewired edge rate;
- cumulative verification rate;
- current active sharing rate;
- cumulative role-transition rate;
- normalized active-role entropy.

`observe_dynamic_role_state` derives the exact same projection from a V7 model/state by joining existing V5, V6, and V7 metrics. This prevents calibration code from maintaining a second definition of the observables.

`EmpiricalABMCase` contains a stable case id, a `TRAIN` or `HOLDOUT` split, canonical initial beliefs, paired environment/truth schedules, and one observed snapshot per round. Schedules must be nonempty, equal length, and match consecutive observed rounds starting at one.

`EmpiricalABMDataset` contains a stable id/version and unique canonical cases. It requires at least one training case and one holdout case. Case order does not affect its hash.

## Parameter and weight contract

`ABMCalibrationCandidate` contains:

- `learning_rate` in `[0, 1]`;
- `dissolution_similarity` in `[0, 1]`;
- `formation_similarity` in `[0, 1]`, strictly greater than dissolution.

Candidate application rebuilds the nested evolving, autonomy, and dynamic-role model values with the same ids, versions, catalogs, policies, and rules. It never mutates the supplied base model.

`ABMCalibrationWeights` has one finite non-negative weight per observed metric and requires at least one positive weight. Weights are recorded exactly; normalization is unnecessary because every candidate is scored with the same denominator.

## Evaluation and selection

For each candidate and case:

1. Rebuild the candidate model.
2. Initialize it from the case's frozen initial beliefs.
3. Replay the exact paired environment and truth-observation schedules.
4. Project each resulting state to `ObservedABMSnapshot`.
5. Compute weighted mean squared error across all rounds and all positively weighted metrics.
6. Store observed/predicted snapshot hashes and the case loss.

If a candidate cannot replay a frozen schedule, calibration fails closed with the candidate and case identity in the error. V8 does not synthesize missing truth, discard rounds, or assign an arbitrary penalty.

Candidate training loss is the arithmetic mean of its training case losses. Holdout loss is the arithmetic mean of its holdout case losses. The ranking key is `(training_loss, candidate canonical tuple)`. Holdout loss is never part of selection or tie-breaking.

## Audit report

`ABMCaseCalibrationFit` binds candidate, case id/split, loss, and exact observed/predicted snapshot hashes.

`ABMCandidateCalibrationFit` binds one candidate, canonical complete case-fit tuple, training loss, and holdout loss. It validates that both aggregates exactly match the declared splits.

`EmpiricalABMCalibrationReport` binds base model hash, dataset hash, weights, complete unique candidate grid, complete candidate fits, training ranking, and selected candidate. The report rejects missing/extra candidates, duplicate evaluations, any ranking influenced by holdout scores, and inconsistent identities.

## Acceptance scenarios

1. Synthetic observations generated from an in-grid candidate recover that candidate from training loss.
2. A candidate with worse training loss cannot win even when it has better holdout loss.
3. Equal training losses are broken by the canonical candidate tuple, independent of grid input order.
4. Every candidate is evaluated on every training and holdout case exactly once.
5. Observed and predicted snapshot hashes are retained per case for audit.
6. Dataset cases, belief inputs, candidates, and case fits are canonical and hash-stable under input reordering.
7. Missing training/holdout splits, duplicate candidates/cases, invalid thresholds, negative/all-zero weights, and schedule length drift fail closed.
8. Candidate reconstruction preserves all non-calibrated model configuration and does not mutate the base model.
9. A truth schedule incompatible with a candidate raises a candidate/case-specific replay error.
10. A zero-active snapshot remains finite and uses zero for active-belief, sharing, and role-diversity observables.

## Deferred work

- calibrating role-policy and role-transition thresholds;
- noisy or partially observed snapshots and missing-data likelihoods;
- uncertainty intervals, posterior inference, and stochastic optimization;
- cross-validation or rolling-origin temporal splits;
- micro-level agent/edge observations;
- causal claims from observational fit;
- large-grid parallel execution, addressed as experiment orchestration in V9.
