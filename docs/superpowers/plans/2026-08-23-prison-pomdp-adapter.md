# Finite Prison-Scenario POMDP Adapter Implementation Plan

1. Add RED tests for the public adapter API, Bayesian update, finite policy behavior, trusted subprocess registration, calibration/validation workflow, and report artifact integrity.
2. Implement the pure finite POMDP mathematics and deterministic expectation calculations in a focused adapter module.
3. Add the importable subprocess factory, unpinned four-schema `ModelContract`, source builder, and metrics extractor.
4. Bind the source in tests through an external `TrustedExecutionPolicy`, register it as `ModelKind.POMDP`, and verify pre-execution implementation matching plus post-execution event/outcome validation.
5. Run synthetic finite-grid recovery, repeated seed blocks, selection validation, final testing, and aggregate report attestation.
6. Run the full Python suite, Lean-generated conformance gate, full Lean build, and all Lean theorem tests. Merge only after review and two green contexts.
