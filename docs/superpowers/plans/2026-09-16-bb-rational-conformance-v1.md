# BB Rational Conformance V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic Python→JSONL→Lean exact-rational conformance pipeline for a finite declared set of BB replay/propagation cases, with permanent schema/provenance/output drift rejection.

**Architecture:** Python owns canonical schema serialization and a small exact `Fraction` evaluator; Lean independently checks the same declared cases with `Rat` and existing BB/network proof surfaces where applicable. The committed JSONL golden is immutable CI input: normal CI generates to a temporary file and performs unconditional byte comparison before running the Lean checker.

**Tech Stack:** Python 3.13, `fractions.Fraction`, `json`, `hashlib`, Lean 4.32.0, mathlib, existing `FitnessAttachment`/`FitnessABMReplay`/`FitnessABMPathN`, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-rational-conformance-v1-design.md`

## Global Constraints

- Schema is exactly `bb-rational-conformance/v1`.
- Theorem-bearing rationals are reduced `{ "num": Int, "den": Nat }` with `den > 0`; no decimal rational fields.
- Python exact semantics use `fractions.Fraction`; production float beliefs are not proof evidence.
- Lean checks exact declared fields using `Rat` and existing proof surfaces; do not add a second general replay implementation.
- Normal CI never rewrites the tracked golden; it writes a temp file and byte-compares.
- Permanent negative tests cover schema drift, provenance drift, and expected-rational drift.
- Keep existing Path4/Path5/PathN fixtures and gates unchanged except minimal bounded conformance wiring.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, or unbounded Lean resource settings.
- Direct Lean invocations use `timeout --kill-after=10s 240s`.

## File Map

| File | Responsibility |
| --- | --- |
| `tools/bb_rational_conformance.py` | schema types, canonical rational/JSON encoding, exact finite evaluator, case definitions, provenance hash |
| `tools/generate_bb_rational_conformance.py` | deterministic CLI generator only |
| `tests/test_bb_rational_conformance.py` | Python schema/evaluator/generator/drift tests |
| `conformance/bb_rational_v1.jsonl` | reviewed immutable golden corpus |
| `NarrativeDynamics/Conformance/BBRationalConformance.lean` | JSONL parser + exact checker executable |
| `NarrativeDynamics/Tests/BBRationalConformance.lean` | theorem/trust consumers for exact case adapters if needed |
| `tools/check_bb_rational_conformance.sh` | bounded generation/compare/Lean/drift gate |
| `.github/workflows/proof.yml` | invoke the bounded gate under relevant path filters |
| `docs/superpowers/plans/2026-09-16-bb-rational-conformance-v1.md` | execution record/checklist |

---

### Task 1: Canonical schema and rational codec

**Files:**
- Create: `tools/bb_rational_conformance.py`
- Create: `tests/test_bb_rational_conformance.py`

**Interfaces:**
- Produces `SCHEMA = "bb-rational-conformance/v1"`.
- Produces `rat(value: Fraction | int) -> dict[str, int]`.
- Produces `parse_rat(obj: object) -> Fraction` with strict reduced/positive-denominator validation.
- Produces `canonical_json_line(case: dict[str, object]) -> str` using sorted keys and compact separators.

- [ ] **Step 1: Write schema/codec RED tests.**

```python
from fractions import Fraction
import json
import pytest

from tools.bb_rational_conformance import SCHEMA, canonical_json_line, parse_rat, rat


def test_rational_codec_is_structural_and_reduced():
    assert SCHEMA == "bb-rational-conformance/v1"
    assert rat(Fraction(-6, 16)) == {"num": -3, "den": 8}
    assert rat(0) == {"num": 0, "den": 1}
    assert parse_rat({"num": -3, "den": 8}) == Fraction(-3, 8)


@pytest.mark.parametrize("bad", [0.5, {"num": 1, "den": 0}, {"num": 2, "den": 4}])
def test_parse_rat_rejects_float_nonpositive_and_nonreduced(bad):
    with pytest.raises((TypeError, ValueError)):
        parse_rat(bad)


def test_canonical_json_line_is_stable():
    line = canonical_json_line({"schema": SCHEMA, "z": 1, "a": rat(Fraction(1, 2))})
    assert line.endswith("\n")
    assert json.loads(line)["a"] == {"den": 2, "num": 1}
    assert " " not in line
```

- [ ] **Step 2: Run RED.**

Run: `python -m pytest tests/test_bb_rational_conformance.py -q`
Expected: import/module failure because `tools.bb_rational_conformance` does not exist.

- [ ] **Step 3: Implement the minimal strict codec.**

`rat()` must normalize via `Fraction`; `parse_rat()` must reject booleans, floats, missing/extra keys, nonpositive denominator, and nonreduced pairs (`Fraction(num, den)` must round-trip to the same pair).

- [ ] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_bb_rational_conformance.py -q`
Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit.**

```bash
git add tools/bb_rational_conformance.py tests/test_bb_rational_conformance.py
git commit -m "test(conformance): define exact rational schema"
```

---

### Task 2: Exact finite propagation and BB case evaluator

**Files:**
- Modify: `tools/bb_rational_conformance.py`
- Modify: `tests/test_bb_rational_conformance.py`

**Interfaces:**
- Produce immutable authored `CASES` containing ids:
  - `inclusive-threshold`
  - `no-broadcast`
  - `nontrivial-mean`
  - `bb-successive-births`
  - `duplicate-target-error`
- Produce `evaluate_case(case) -> dict[str, object]`.
- Propagation uses exact `Fraction` equation `(1-alpha)*belief + alpha*mean` and increments exposure by number of broadcasting neighbors.
- BB attachment uses ordered without-replacement exact fitness×degree mass.

- [ ] **Step 1: Add RED assertions for all five case families.**

Use exact expectations:

```python
def test_inclusive_threshold_broadcasts():
    out = evaluate_named("inclusive-threshold")
    assert out["expected"]["broadcasting"] == [True, False]


def test_no_broadcast_preserves_beliefs():
    out = evaluate_named("no-broadcast")
    assert out["expected"]["beliefs"] == [rat(Fraction(1, 4)), rat(Fraction(1, 8))]


def test_nontrivial_mean_is_exact():
    out = evaluate_named("nontrivial-mean")
    assert all(isinstance(parse_rat(x), Fraction) for x in out["expected"]["beliefs"])


def test_successive_birth_mass_is_one_eighth():
    out = evaluate_named("bb-successive-births")
    assert parse_rat(out["expected"]["trace_mass"]) == Fraction(1, 8)


def test_duplicate_target_has_stable_error():
    out = evaluate_named("duplicate-target-error")
    assert out["expected"]["error"] == {
        "stage": "tick_network", "code": "duplicateTarget", "field": "targets"
    }
```

- [ ] **Step 2: Run RED.**

Run: `python -m pytest tests/test_bb_rational_conformance.py -q`
Expected: missing case/evaluator declarations.

- [ ] **Step 3: Implement exact evaluator only for declared v1 contract.**

Represent graph edges canonically as sorted undirected `(min,max)` integer pairs. Validate target range/count/distinctness before computing mass. Do not import the production float simulation engine.

- [ ] **Step 4: Run GREEN and production-runtime discrete cross-checks.**

Run:
```bash
python -m pytest tests/test_bb_rational_conformance.py -q
python -m examples.bb_abm_runtime
```
Expected: tests pass; existing example still prints final exact mass `1/8` and exits 0.

- [ ] **Step 5: Commit.**

```bash
git add tools/bb_rational_conformance.py tests/test_bb_rational_conformance.py
git commit -m "feat(conformance): evaluate exact BB rational cases"
```

---

### Task 3: Deterministic generator, semantic provenance, and golden

**Files:**
- Create: `tools/generate_bb_rational_conformance.py`
- Create: `conformance/bb_rational_v1.jsonl`
- Modify: `tests/test_bb_rational_conformance.py`

**Interfaces:**
- Produce `contract_hash() -> str` as lowercase SHA-256 over explicitly normalized semantic source bytes.
- Generator CLI accepts `--output PATH` and never mutates tracked golden unless that path is explicitly supplied.

- [ ] **Step 1: Add deterministic/provenance RED tests.**

Generate twice into two temp files and assert byte equality; assert every case has `provenance.schema == SCHEMA`, 64-hex `generator_contract`, fixed `case_set == "v1"`, and stable generator/checker path strings.

- [ ] **Step 2: Run RED.**

Run: `python -m pytest tests/test_bb_rational_conformance.py -q`
Expected: missing generator/provenance functions.

- [ ] **Step 3: Implement generator and create reviewed golden.**

Run:
```bash
python tools/generate_bb_rational_conformance.py --output /tmp/bb-rational-v1.jsonl
cat /tmp/bb-rational-v1.jsonl
cp /tmp/bb-rational-v1.jsonl conformance/bb_rational_v1.jsonl
python tools/generate_bb_rational_conformance.py --output /tmp/bb-rational-v1-second.jsonl
cmp conformance/bb_rational_v1.jsonl /tmp/bb-rational-v1-second.jsonl
```
Expected: five canonical case lines; `cmp` exit 0.

- [ ] **Step 4: Run GREEN.**

Run: `python -m pytest tests/test_bb_rational_conformance.py -q`
Expected: pass.

- [ ] **Step 5: Commit.**

```bash
git add tools/generate_bb_rational_conformance.py conformance/bb_rational_v1.jsonl tests/test_bb_rational_conformance.py tools/bb_rational_conformance.py
git commit -m "feat(conformance): generate deterministic BB rational golden"
```

---

### Task 4: Lean exact checker

**Files:**
- Create: `NarrativeDynamics/Conformance/BBRationalConformance.lean`
- Optionally create only if trust reports require named theorem consumers: `NarrativeDynamics/Tests/BBRationalConformance.lean`

**Interfaces:**
- Executable command: `lake env lean --run NarrativeDynamics/Conformance/BBRationalConformance.lean -- conformance/bb_rational_v1.jsonl`.
- Exit 0 only after every case schema/provenance/input/expected field passes.
- Diagnostics include stable `case_id` and field/category.

- [ ] **Step 1: Create a RED checker consumer.**

The first version imports `Lean.Data.Json`, reads the path from argv, parses nonempty JSONL lines, and calls not-yet-defined `checkCase`; compilation must fail specifically on the missing checker declaration after JSON APIs are corrected.

- [ ] **Step 2: Run bounded RED.**

Run:
```bash
timeout --kill-after=10s 240s lake env lean --run NarrativeDynamics/Conformance/BBRationalConformance.lean -- conformance/bb_rational_v1.jsonl
```
Expected: compile failure at missing `checkCase`/case adapter, not a toolchain/import failure.

- [ ] **Step 3: Implement strict JSON/rational parsing and case adapters.**

Reject JSON numbers used directly as rational values; require object `num`/`den`, positive denominator, and normalized equality after constructing `Rat`. Use existing `FitnessABM`/`FitnessAttachment`/`FitnessABMPathN` functions where the authored case matches them; keep any schema adapter finite and case-contract-specific.

- [ ] **Step 4: Run GREEN.**

Run the bounded command above.
Expected: exit 0 and a stable summary containing `5 cases checked`.

- [ ] **Step 5: Commit.**

```bash
git add NarrativeDynamics/Conformance/BBRationalConformance.lean NarrativeDynamics/Tests/BBRationalConformance.lean
git commit -m "feat(lean): check BB rational conformance corpus"
```

If no separate test file is required, omit that nonexistent path from `git add`.

---

### Task 5: Permanent drift rejection

**Files:**
- Modify: `tests/test_bb_rational_conformance.py`
- Create: `tools/check_bb_rational_conformance.sh`

**Interfaces:**
- Gate runs canonical generation/byte compare, canonical Lean check, then three temp-file mutations.

- [ ] **Step 1: Add three RED mutation tests/helpers.**

Mutations:
1. schema → `bb-rational-conformance/v999`;
2. `generator_contract` first hex digit changed;
3. first expected rational numerator changed by `+1` while preserving structural validity.

Each mutated corpus must be rejected by the same checker entry point; canonical corpus must still pass afterward.

- [ ] **Step 2: Run RED.**

Run: `bash tools/check_bb_rational_conformance.sh`
Expected: missing gate or missing mutation rejection.

- [ ] **Step 3: Implement bounded shell gate.**

Required sequence:
```bash
python -m pytest tests/test_bb_rational_conformance.py -q
python tools/generate_bb_rational_conformance.py --output "$tmp_generated"
cmp conformance/bb_rational_v1.jsonl "$tmp_generated"
timeout --kill-after=10s 240s lake env lean --run NarrativeDynamics/Conformance/BBRationalConformance.lean -- conformance/bb_rational_v1.jsonl
# make each temp mutation and assert the same checker exits nonzero
```

Use `mktemp` and `trap` cleanup. Never edit the tracked golden during mutation tests.

- [ ] **Step 4: Run GREEN twice.**

Run: `bash tools/check_bb_rational_conformance.sh && bash tools/check_bb_rational_conformance.sh`
Expected: both complete successfully and produce identical canonical generation.

- [ ] **Step 5: Commit.**

```bash
git add tools/check_bb_rational_conformance.sh tests/test_bb_rational_conformance.py
git commit -m "test(conformance): reject BB schema and rational drift"
```

---

### Task 6: CI integration, full regression, review handoff

**Files:**
- Modify: `.github/workflows/proof.yml`
- Modify: this plan only to check completed boxes / record exact evidence after execution

**Interfaces:**
- Conformance gate runs on changes to its Python/Lean/golden/gate paths and remains bounded.
- Existing Path4/Path5/PathN gates remain unchanged and still run under their existing triggers.

- [ ] **Step 1: Add the conformance gate to proof workflow with exact path filters.**

Invoke only `bash tools/check_bb_rational_conformance.sh`; do not duplicate its internals into YAML.

- [ ] **Step 2: Syntax/source checks.**

Run:
```bash
bash -n tools/check_bb_rational_conformance.sh
python -m compileall tools/bb_rational_conformance.py tools/generate_bb_rational_conformance.py tests/test_bb_rational_conformance.py
git diff --check
```
Expected: all exit 0.

- [ ] **Step 3: Run focused local gate and existing relevant tests.**

Run:
```bash
bash tools/check_bb_rational_conformance.sh
bash tools/check_fitness_abm_path4.sh
python -m pytest tests/test_bb_rational_conformance.py -q
```
Expected: all pass within bounded resources.

- [ ] **Step 4: Commit CI wiring.**

```bash
git add .github/workflows/proof.yml docs/superpowers/plans/2026-09-16-bb-rational-conformance-v1.md
git commit -m "ci: gate BB rational conformance"
```

- [ ] **Step 5: Open PR against `proof/narrative-dynamics-v0`.**

PR body must state:
- finite generated-case conformance only;
- exact `Fraction`/`Rat` fields;
- production float belief exclusion;
- five case families;
- three permanent drift negatives;
- exact-head CI evidence only after CI actually completes.

- [ ] **Step 6: Independent review and exact-head verification.**

Review code/diff for blockers, inspect review threads, and require PR-event proof/World Studio statuses applicable to the exact head. Fix Critical/Important findings before merge. Do not merge while any required exact-head workflow is failing/in progress.
