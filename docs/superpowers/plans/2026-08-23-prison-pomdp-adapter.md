# Finite Prison-Scenario POMDP Adapter Implementation Plan

1. Add RED tests for the public adapter API, finite-horizon behavior, exact replay, trusted subprocess registration, finite-grid calibration, selection/final-test separation, and report-artifact integrity.
2. Implement the exact two-state, one-observation-step POMDP and deterministic expectation coordinates in a focused Python adapter module.
3. Add the importable subprocess factory, unpinned four-schema `ModelContract`, source builder, and stable metrics extractor.
4. Bind the source through an external `TrustedExecutionPolicy`, register it as `ModelKind.POMDP`, and verify pre-execution implementation matching plus post-execution event/outcome validation.
5. Run finite-grid `beta` recovery, selection validation, untouched final testing, and aggregate final-report attestation.
6. Review numerical boundaries, add a focused RED for underflowed softmax tails, and assign the normalization residual to the largest coordinate.
7. Run the complete Python suite, Lean-generated conformance gate, full Lean build, and all Lean theorem tests. Merge only after the feature and target merge contexts are green.
