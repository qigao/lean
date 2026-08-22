# Lean–Python Conformance Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CI-enforced, Lean-certified exact-rational reference corpus that detects numerical drift in the corresponding Python implementations.

**Architecture:** A new Lean sidecar imports existing core definitions, proves a finite exact corpus with `norm_num`, and emits canonical JSON. A committed golden JSON is regenerated and compared in CI. A dependency-free Python checker strictly parses that corpus and dispatches vectors to existing production functions.

**Tech Stack:** Lean 4.32, mathlib `norm_num`, Python 3 standard library (`dataclasses`, `fractions`, `json`, `math`, `pathlib`, `unittest`), GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-22-lean-python-conformance-design.md`

## Global Constraints

- Do not modify any file under `NarrativeDynamics/Core/`.
- Keep the Python runtime dependency-free.
- RED must be observed before adding generator, fixture, checker, or workflow support.
- V1 covers only Bayes posterior, prediction-error learning, two-person effective pressure, and one-drive goal score.
- All emitted numbers use canonical reduced `numerator/denominator` strings.
- Bayesian vectors must have strictly positive denominators.
- Softmax conformance is out of scope.

---

### Task 1: Define and verify the RED contract

**Files:**
- Create: `tests/test_lean_python_conformance.py`

**Interfaces:**
- Consumes: future `narrative_dynamics.conformance` public API and `conformance/lean_reference_vectors.json`.
- Produces: behavior-level obligations for corpus loading, production conformance, and injected-drift detection.

- [ ] **Step 1: Write the failing tests**

Create tests that dynamically import `narrative_dynamics.conformance`, require `ConformanceDefinitionError`, `ConformanceMismatch`, `load_reference_suite`, `default_operation_evaluators`, and `assert_reference_conformance`, then:

```python
FIXTURE = Path(__file__).resolve().parents[1] / "conformance" / "lean_reference_vectors.json"

suite = api.load_reference_suite(FIXTURE)
self.assertEqual(suite.schema_version, 1)
self.assertEqual(
    {vector.operation for vector in suite.vectors},
    {
        "bayes_posterior",
        "learn_instrumentality",
        "effective_pressure2",
        "goal_score_single_drive",
    },
)
api.assert_reference_conformance(suite)
```

Add a mutation-style check:

```python
evaluators = api.default_operation_evaluators()
original = evaluators["learn_instrumentality"]
evaluators["learn_instrumentality"] = lambda inputs: original(inputs) + 0.125
with self.assertRaises(api.ConformanceMismatch) as raised:
    api.assert_reference_conformance(suite, evaluators=evaluators)
self.assertEqual(raised.exception.operation, "learn_instrumentality")
```

- [ ] **Step 2: Run the full suite and verify RED**

Run:

```bash
python3 -m unittest discover -s tests -v
```

Expected: existing tests pass; only the new conformance tests fail because the Python module and fixture do not exist.

- [ ] **Step 3: Commit the RED**

```bash
git add tests/test_lean_python_conformance.py
git commit -m "test: define Lean Python conformance gate"
```

---

### Task 2: Add the Lean-certified reference corpus

**Files:**
- Create: `NarrativeDynamics/Conformance/ReferenceVectors.lean`
- Create: `conformance/lean_reference_vectors.json`

**Interfaces:**
- Consumes: `bayesPosterior`, `learnInstrumentality`, `effectivePressure`, and `goalScore` from existing core modules.
- Produces: root `main : IO Unit` that emits canonical schema-v1 JSON and compile-time soundness theorems for every emitted expected value.

- [ ] **Step 1: Define exact rational and typed case structures**

Use an immutable exact rational representation:

```lean
structure ExactRat where
  numerator : Int
  denominator : Nat

def ExactRat.toReal (value : ExactRat) : ℝ :=
  (value.numerator : ℝ) / (value.denominator : ℝ)
```

Define typed `BayesCase`, `LearningCase`, `EffectivePressureCase`, and `GoalScoreCase` values. Convert them to a common JSON-facing `ReferenceVector` structure with stable ids and input order.

- [ ] **Step 2: Prove each emitted vector against the core definition**

For each named case, prove a theorem of this form:

```lean
theorem bayesPositiveEvidence_sound :
    bayesPosterior
        bayesPositiveEvidence.prior.toReal
        bayesPositiveEvidence.likelihoodH.toReal
        bayesPositiveEvidence.likelihoodNotH.toReal =
      bayesPositiveEvidence.expected.toReal := by
  norm_num [bayesPositiveEvidence, ExactRat.toReal, bayesPosterior]
```

Use equivalent `norm_num` proofs for every learning, effective-pressure, and goal-score case. Do not emit a case that lacks a theorem.

- [ ] **Step 3: Emit deterministic JSON**

Implement string-only JSON emission with fixed field order, reduced fraction strings, and a root:

```lean
def main : IO Unit :=
  IO.println NarrativeDynamics.Conformance.referenceSuiteJson
```

Run:

```bash
lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean
```

Expected: one JSON document and exit status 0.

- [ ] **Step 4: Commit the generated golden file**

Capture the exact generator output at `conformance/lean_reference_vectors.json`, including its final newline.

- [ ] **Step 5: Commit the Lean corpus**

```bash
git add NarrativeDynamics/Conformance/ReferenceVectors.lean conformance/lean_reference_vectors.json
git commit -m "prove: emit exact conformance reference vectors"
```

---

### Task 3: Implement the Python conformance checker

**Files:**
- Create: `narrative_dynamics/conformance.py`
- Modify: `narrative_dynamics/__init__.py`
- Test: `tests/test_lean_python_conformance.py`

**Interfaces:**
- Consumes: schema-v1 JSON and existing functions/classes in `motivation.py`.
- Produces: `ExactRational`, `ReferenceVector`, `ReferenceSuite`, `ConformanceDefinitionError`, `ConformanceMismatch`, `load_reference_suite`, `default_operation_evaluators`, and `assert_reference_conformance`.

- [ ] **Step 1: Parse canonical exact rationals strictly**

Use `fractions.Fraction` but accept only strings matching canonical reduced `-?N/D` syntax. Reject zero/negative denominators, leading zeros, unreduced fractions, `-0`, booleans, floats, and non-strings with `ConformanceDefinitionError` carrying a JSON-style path.

- [ ] **Step 2: Validate the complete suite contract**

Require exact top-level keys, `schema_version == 1`, the declared generator and numeric encoding, unique sorted vector ids, supported operations, exact required input names per operation, and nonempty vectors.

- [ ] **Step 3: Dispatch to existing production implementations**

Implement evaluators:

```python
"bayes_posterior" -> motivation.bayes_posterior
"learn_instrumentality" -> motivation.learn_instrumentality
"effective_pressure2" -> motivation.effective_pressure((1.0, boundary), (p_self, p_other))
"goal_score_single_drive" -> AgentMotivation(...one drive...).score("goal")
```

Convert exact fractions to floats only when invoking these functions.

- [ ] **Step 4: Compare and raise typed drift**

Reject non-finite actual results. Compare with `math.isclose(rel_tol=1e-12, abs_tol=1e-12)`. On mismatch raise `ConformanceMismatch` with `vector_id`, `operation`, `expected`, and `actual` attributes.

- [ ] **Step 5: Run focused and full GREEN tests**

Run:

```bash
python3 -m unittest tests.test_lean_python_conformance -v
python3 -m unittest discover -s tests -v
```

Expected: all tests pass, including the injected-drift test.

- [ ] **Step 6: Commit the checker**

```bash
git add narrative_dynamics/conformance.py narrative_dynamics/__init__.py tests/test_lean_python_conformance.py
git commit -m "feat: check Lean reference vectors in Python"
```

---

### Task 4: Enforce generation and golden identity in CI

**Files:**
- Modify: `.github/workflows/proof.yml`

**Interfaces:**
- Consumes: Lean generator and committed golden fixture.
- Produces: a required workflow step that prevents stale or hand-edited reference fixtures.

- [ ] **Step 1: Add the generation comparison step**

Insert after Lean theorem tests and before Python tests:

```yaml
      - name: Lean Python conformance vectors
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          mkdir -p .generated-conformance
          lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean \
            > .generated-conformance/lean_reference_vectors.json
          cmp \
            conformance/lean_reference_vectors.json \
            .generated-conformance/lean_reference_vectors.json
```

- [ ] **Step 2: Run complete verification**

Run or observe the PR merge-context workflow. Required evidence:

- `lake build` succeeds;
- every existing Lean theorem test succeeds;
- the generator compiles, runs, and `cmp` succeeds;
- the complete Python suite succeeds.

- [ ] **Step 3: Commit the workflow gate**

```bash
git add .github/workflows/proof.yml
git commit -m "ci: enforce Lean Python conformance vectors"
```

---

### Task 5: Review, document boundaries, and integrate

**Files:**
- Update: pull request body for the feature branch.
- Update after merge: PR #2 body.

**Interfaces:**
- Consumes: final diff and fresh workflow logs.
- Produces: auditable RED evidence, GREEN evidence, exact covered operations, and explicit non-claims.

- [ ] **Step 1: Review the complete diff**

Check specifically for an unproved emitted vector, a second source of expected values, a Python evaluator that reimplements rather than calls production code, noncanonical fraction acceptance, fixture generation not enforced by CI, and accidental changes under `NarrativeDynamics/Core/`.

- [ ] **Step 2: Record RED and GREEN evidence**

The PR body must identify the RED run and its expected new failures, the final head, the successful merge-context run, total Python test count, Lean build count, and the exact operation set.

- [ ] **Step 3: Merge only after fresh verification**

Fast-forward the feature head into `proof/narrative-dynamics-v0` only after the final feature PR workflow is green, then require a fresh PR #2 merge-context workflow to pass before claiming completion.