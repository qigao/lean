# Lean–Python Conformance Gate Design

## Scope

Add an explicit cross-language numerical conformance gate without changing any definition under `NarrativeDynamics/Core/`. The first gate covers only closed-form operations that Lean can certify exactly over rational inputs:

- `NarrativeDynamics.bayesPosterior` ↔ `motivation.bayes_posterior`;
- `NarrativeDynamics.learnInstrumentality` ↔ `motivation.learn_instrumentality`;
- `NarrativeDynamics.effectivePressure` ↔ `motivation.effective_pressure` with weights `(1, boundary)`;
- `NarrativeDynamics.goalScore` ↔ `AgentMotivation.score` on one nonnegative drive.

`Real.exp`-based softmax definitions are deliberately excluded from v1. A separately written Float mirror would not by itself establish conformance with the existing noncomputable `Real.exp` definitions.

## Architecture

A sidecar Lean module defines a versioned corpus of exact rational reference vectors. Every emitted expected value is shared with a Lean theorem that unfolds the corresponding existing core definition and proves the closed result with `norm_num`.

The same Lean module has a root `main : IO Unit` that emits one canonical JSON document. The JSON encodes every number as a reduced `numerator/denominator` string, avoiding decimal and platform ambiguity.

The generated JSON is committed at `conformance/lean_reference_vectors.json`. CI executes the Lean generator into a temporary file and requires byte-for-byte equality with the committed file before running Python tests. This gives Python-only local test runs a stable fixture while preventing the fixture from drifting away from the Lean-checked source.

A dependency-free Python module loads and strictly validates the fixture, dispatches each vector to the existing production implementation, and raises a typed mismatch containing vector id, operation, expected value, and actual value.

## Reference-vector contract

The document has:

```json
{
  "schema_version": 1,
  "generator": "NarrativeDynamics.Conformance.ReferenceVectors",
  "numeric_encoding": "reduced_fraction_v1",
  "definitions": [
    "NarrativeDynamics.bayesPosterior",
    "NarrativeDynamics.learnInstrumentality",
    "NarrativeDynamics.effectivePressure",
    "NarrativeDynamics.goalScore"
  ],
  "vectors": []
}
```

Each vector has a unique stable id, a supported operation, an exact-rational input mapping, and an exact-rational expected result. Vector and input order are canonical and therefore part of the golden-file identity.

Bayesian vectors are restricted to cases with a strictly positive denominator. Lean's totalized division at zero and Python's `ZeroDivisionError` remain an explicit domain-boundary issue rather than being hidden by this gate.

## Python boundary

`narrative_dynamics.conformance` provides:

- `ExactRational` and `ReferenceVector` immutable values;
- `ReferenceSuite` with strict schema/version/ordering validation;
- `load_reference_suite(path)`;
- `default_operation_evaluators()`;
- `assert_reference_conformance(suite, evaluators=None)`;
- typed `ConformanceDefinitionError` and `ConformanceMismatch` failures.

All evaluator inputs are converted from exact fractions to Python floats only at the call into the existing numerical implementation. Results must be finite and match the exact expected fraction converted to float under fixed `rel_tol=1e-12` and `abs_tol=1e-12`.

## RED behavior

The first test-only commit requires the module, fixture, supported operation set, successful corpus verification, and detection of an injected perturbed implementation. It must fail before any generator or Python checker is added.

## Verification

The final workflow must pass:

1. full `lake build`;
2. every existing Lean theorem test;
3. Lean reference-vector generation and byte-for-byte golden comparison;
4. the complete Python suite, including a mutation-style test proving that a deliberately drifted Python evaluator is rejected.

## Non-claims

Passing this gate does not prove empirical validity, model adequacy, global numerical equivalence, softmax equivalence, or conformance outside the emitted admissible vectors. It establishes an executable regression link between named Lean definitions and named Python implementations for the exact v1 corpus.