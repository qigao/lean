# Prison POMDP Model-Adequacy Design

## Status

Approved by the user's explicit `go` after the finite prison-scenario POMDP adapter and its trusted execution boundary were merged. This increment is Python-only. It does not modify `NarrativeDynamics/Core`, any Lean theorem, or the prison adapter's state/action model.

## Purpose

Add an explicit model-adequacy layer around the finite prison POMDP. The layer must distinguish parameter fitting from model adequacy and make four failure modes observable:

1. no single parameter candidate can explain deliberately misspecified external cases;
2. probability predictions are evaluated with proper scoring rules rather than only raw squared error over arbitrary summaries;
3. local interactions between a fitted parameter and an environment/scenario factor are reported instead of being hidden by one-at-a-time sensitivity;
4. an untouched final-test suite reports which declared scenario strata it covers and whether seeds are reused.

These diagnostics do not establish empirical truth, causal validity, or population coverage. They make specific inadequacies and test-suite gaps explicit and reproducible.

## Design Choice

The recommended design adds reusable metric-loss and adequacy primitives, then exercises them with the prison adapter.

Rejected alternatives:

- **Add more prison parameters first.** This could reduce residuals while hiding the original beta-only model's inadequacy and would confound model expansion with diagnosis.
- **Add prison-only analysis helpers.** This would duplicate calibration and validation logic and would not improve the runtime boundary for later adapters.
- **Treat final-test coverage as a confidence interval guarantee.** The available data do not support that claim. V1 reports scenario-stratum and seed-plan coverage only.

## Proper-Scoring Loss Boundary

Create `narrative_dynamics/losses.py` with an explicit `MetricLoss` protocol and three immutable loss objects:

- `WeightedSquaredErrorLoss`: compatibility wrapper around the existing weighted squared-error behavior;
- `CategoricalBrierLoss`: finite strictly proper score over one or more declared categorical metric groups;
- `CategoricalLogLoss`: negative categorical log score over one or more declared categorical metric groups.

A `CategoricalMetricGroup` declares a stable group name, ordered metric keys, and non-negative group weight. Target values are non-negative masses and are normalized within each group. Predicted values must be finite probabilities whose group sum is one within a fixed tolerance. A positive target mass assigned zero predicted probability is rejected explicitly by log loss rather than silently clipped.

`calibrate_grid`, held-out validation, selection validation, final testing, synthetic recovery, and local sensitivity gain an optional `loss` argument. The default remains weighted squared error, preserving existing behavior. Every report manifest records the selected loss identity. Case-level `weights` remain supported by `WeightedSquaredErrorLoss`; categorical losses reject per-key weights because they would change the scoring-rule semantics.

## Explicit Misspecification

Create `narrative_dynamics/adequacy/misspecification.py`.

`assess_misspecification` evaluates one shared finite parameter grid across multiple existing `HeldOutCase` values using the chosen metric extractor and loss. For each candidate it records per-case losses, mean loss, worst loss, and the child held-out validation manifests. Candidates are ranked by mean loss, then worst loss, then canonical parameters.

The report declares the model adequate only when the best shared candidate satisfies both caller-supplied non-negative mean-loss and worst-loss limits. A misspecified suite can therefore show an irreducible residual floor even after the best allowed candidate is chosen.

The prison RED uses two cases with identical environment mechanics but incompatible beta-generated target policies. No single shared beta in the finite grid can match both exactly. A companion adequate suite uses a common generating beta and must pass the same thresholds.

## Local Factor Interaction

Create `narrative_dynamics/adequacy/interaction.py`.

A `LocalFactor` names either:

- a model parameter; or
- a numeric top-level `Scenario.payload` field.

`local_factor_interaction_report` accepts exactly two distinct factors, positive finite step sizes, one scenario, fixed seeds, parameters, and a metric extractor. It evaluates the four common-random-number corners:

```text
(-h1, -h2)  (-h1, +h2)
(+h1, -h2)  (+h1, +h2)
```

For every stable metric key it reports the central mixed finite difference:

```text
[f(+,+) - f(+,-) - f(-,+) + f(-,-)] / (4 h1 h2)
```

The prison consumer measures the interaction between fitted `beta` and scenario `signal_accuracy`. The report is descriptive local sensitivity, not a causal interaction estimate.

## Final-Test Coverage Diagnostics

Create `narrative_dynamics/adequacy/coverage.py`.

A `CoverageAxis` declares one numeric top-level scenario field and ordered, non-overlapping `CoverageBin` intervals. `diagnose_final_test_coverage` requires both the actual `FinalTestReport` and the `HeldOutSuite` used to produce it. It verifies final-test role, suite name, case names, and scenario ids before reporting:

- count per declared bin for every axis;
- missing bins;
- cases outside every declared bin;
- total and unique seed counts;
- seeds reused across cases;
- an overall `complete` flag that is true only when every bin is represented and every case falls into exactly one bin.

This is scenario-space and seed-plan coverage. It is not posterior predictive coverage, confidence-interval coverage, or evidence that the suite is representative.

## Manifests and Artifacts

Add three experiment stages:

- `model_misspecification`;
- `interaction_sensitivity`;
- `final_test_coverage`.

Every new report carries an `ExperimentManifest` and parent lineage to the held-out validations, simulation runs, or final-test report it diagnoses. Existing `attest_report()` must work without special cases because the new reports are immutable dataclasses with canonical fields.

## Error Boundaries

The implementation fails closed on:

- empty or duplicate categorical groups;
- duplicate metric keys across groups;
- non-finite or negative group weights;
- invalid probability vectors or target masses;
- custom categorical weights;
- empty misspecification grids or cases;
- negative adequacy thresholds;
- duplicate interaction factors, missing fields, non-finite perturbations, or changing metric schemas;
- non-final suites, mismatched final reports, overlapping coverage bins, uncovered cases, or non-numeric coverage values.

Model and schema failures from the existing trusted runner are not caught or reclassified.

## Testing

Strict RED -> GREEN:

1. proper-scoring tests fail because the loss objects and loss-pluggable calibration do not exist;
2. misspecification tests fail because no shared-grid adequacy report exists;
3. interaction tests fail because no mixed-factor sensitivity report exists;
4. final coverage tests fail because no scenario-stratum report exists;
5. implement the minimum production code;
6. run targeted tests, full Python suite, Lean-generated conformance gate, full Lean build, and every Lean theorem test;
7. merge only after feature and PR #2 merge contexts are green.

## Non-goals

- no new prison hidden state, action, parameter, or solver horizon;
- no generic Bayesian posterior inference engine;
- no bootstrap confidence region or posterior predictive interval;
- no empirical dataset ingestion;
- no Lean modification;
- no change to subprocess sandbox capabilities;
- no claim that threshold choice is statistically calibrated.
