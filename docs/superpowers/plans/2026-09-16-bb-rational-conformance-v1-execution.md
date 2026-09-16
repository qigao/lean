# BB Rational Conformance V1 Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved #85 exact-rational Python→JSONL→Lean conformance pipeline using the repository's real `unittest` discovery surface.

**Architecture:** Keep the design from `docs/superpowers/specs/2026-09-16-bb-rational-conformance-v1-design.md`. Python uses `Fraction` and deterministic canonical JSONL; Lean independently checks declared exact fields with `Rat`. The tracked golden is never rewritten by normal CI.

**Tech Stack:** Python 3.13 `unittest`, `fractions`, `json`, `hashlib`; Lean 4.32.0/mathlib; existing BB exact proof/runtime surfaces; GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-16-bb-rational-conformance-v1-design.md`

## Global Constraints

- The full Python acceptance command is `python -m unittest discover -s tests -v`; every Python test added here must be discovered by it.
- Schema is exactly `bb-rational-conformance/v1`.
- Exact rational JSON uses reduced `{ "num": Int, "den": positive Nat }`; float rational fields are rejected.
- Production Python float beliefs are excluded from theorem/conformance rational expectations.
- CI generates to temporary paths and byte-compares to `conformance/bb_rational_v1.jsonl`.
- Drift negatives cover schema, provenance hash, and one expected rational.
- Existing Path4/Path5/PathN semantics stay unchanged.
- Lean commands stay bounded by `timeout --kill-after=10s 240s`.
- No `sorry`, `admit`, new user `axiom`, `unsafe`, `native_decide`, or unbounded resource settings.

---

### Task 1: Strict rational schema codec

**Files:**
- Create: `tests/test_bb_rational_conformance.py`
- Create: `tools/bb_rational_conformance.py`

**Interfaces:**
- `SCHEMA: str = "bb-rational-conformance/v1"`
- `rat(value: int | Fraction) -> dict[str, int]`
- `parse_rat(value: object) -> Fraction`
- `canonical_json_line(case: dict[str, object]) -> str`

- [ ] Create `unittest.TestCase` RED tests for reduced `-3/8`, canonical zero `0/1`, float rejection, denominator-zero rejection, nonreduced `2/4` rejection, and compact sorted JSON ending in newline.
- [ ] Push the test-only RED and verify the exact branch Python workflow fails because `tools.bb_rational_conformance` is missing; a dependency/setup failure is not valid RED.
- [ ] Add the minimal codec. `parse_rat` rejects bools, floats, missing/extra keys, non-int numerator/denominator, `den <= 0`, and pairs that do not equal the normalized `Fraction` numerator/denominator.
- [ ] Verify `python -m unittest tests.test_bb_rational_conformance -v` and full `python -m unittest discover -s tests -v` through CI.
- [ ] Commit as `feat(conformance): define exact rational schema`.

### Task 2: Exact authored evaluator

**Files:**
- Modify: `tests/test_bb_rational_conformance.py`
- Modify: `tools/bb_rational_conformance.py`

**Interfaces:**
- `CASES` contains exactly stable ids `inclusive-threshold`, `no-broadcast`, `nontrivial-mean`, `bb-successive-births`, `duplicate-target-error`.
- `evaluate_named(case_id: str) -> dict[str, object]` returns canonical `input` + `expected` data.

- [ ] Add RED tests asserting inclusive equality broadcasts, no-broadcast preserves beliefs/exposure, nontrivial mean uses exact fractions, successive births accumulate exact mass `1/8`, and duplicate targets return stable `{stage:"tick_network",code:"duplicateTarget",field:"targets"}`.
- [ ] Verify RED fails only on missing evaluator/cases.
- [ ] Implement finite exact propagation: broadcaster iff `threshold <= belief`; if no incoming broadcaster preserve belief; otherwise `(1-alpha)*belief + alpha*mean`; exposure increments by broadcaster count.
- [ ] Implement exact ordered-without-replacement BB mass from `fitness * degree`, target validation, graph growth, and accumulated trace mass for the authored birth case. Do not import the production float simulation engine.
- [ ] Verify focused + full unittest discovery and existing `python -m examples.bb_abm_runtime` in CI; the production example must still report final exact mass `1/8`.
- [ ] Commit as `feat(conformance): evaluate exact BB rational cases`.

### Task 3: Deterministic generator and golden

**Files:**
- Create: `tools/generate_bb_rational_conformance.py`
- Create: `conformance/bb_rational_v1.jsonl`
- Modify: `tools/bb_rational_conformance.py`
- Modify: `tests/test_bb_rational_conformance.py`

**Interfaces:**
- `contract_hash() -> str` is lowercase SHA-256 over explicitly named semantic source bytes only.
- generator accepts `--output PATH`.
- every line contains schema, case id, kind, provenance, input, expected.

- [ ] Add RED unittest using `tempfile.TemporaryDirectory`: generate twice, compare bytes, inspect five lines, verify 64-hex contract hash and fixed generator/checker/case-set provenance strings.
- [ ] Implement provenance/hash and generator CLI.
- [ ] Generate a temporary corpus, decode/review every line, then intentionally create `conformance/bb_rational_v1.jsonl`; regenerate and require byte equality.
- [ ] Verify focused + full unittest discovery.
- [ ] Commit as `feat(conformance): generate deterministic BB rational golden`.

### Task 4: Lean exact checker

**Files:**
- Create: `NarrativeDynamics/Conformance/BBRationalConformance.lean`
- Create only if named trust consumers are required: `NarrativeDynamics/Tests/BBRationalConformance.lean`

**Interface:**
`lake env lean --run NarrativeDynamics/Conformance/BBRationalConformance.lean -- conformance/bb_rational_v1.jsonl`
returns exit 0 only after all five cases and provenance pass.

- [ ] Commit a RED Lean consumer that successfully imports the repository JSON API, reads argv/file/JSONL, then intentionally references the not-yet-implemented `checkCase`; verify bounded CI fails at that declaration rather than import/toolchain.
- [ ] Implement strict rational-object parsing (`num`,`den`, positive denominator, normalized pair) and stable case diagnostics.
- [ ] Reuse existing exact Lean model functions where case inputs match; keep schema adapters case-contract-specific and do not add a second general replay engine.
- [ ] Verify canonical corpus prints a stable `5 cases checked` summary and exits 0 under the 240-second bound.
- [ ] Commit as `feat(lean): check BB rational conformance corpus`.

### Task 5: Permanent drift rejection gate

**Files:**
- Create: `tools/check_bb_rational_conformance.sh`
- Modify: `tests/test_bb_rational_conformance.py`

**Interface:** gate performs unittest, temp generation, unconditional `cmp`, canonical Lean check, then three temp mutations through the same Lean checker.

- [ ] Add mutation helpers/tests for unsupported schema version, one changed provenance hash nibble, and `+1` to one expected rational numerator while keeping structurally valid JSON.
- [ ] Implement shell gate with `set -euo pipefail`, `mktemp`, `trap`, `cmp`, bounded Lean invocation, and explicit assertion that each mutated corpus exits nonzero.
- [ ] Run gate twice in CI-equivalent environment; canonical golden must still pass after all mutations because mutations are temp-only.
- [ ] `bash -n tools/check_bb_rational_conformance.sh` and `git diff --check` must pass.
- [ ] Commit as `test(conformance): reject BB schema and rational drift`.

### Task 6: Proof workflow, PR, independent review

**Files:**
- Modify: `.github/workflows/proof.yml`
- Update execution record only after facts are observed.

- [ ] Add one bounded workflow step `bash tools/check_bb_rational_conformance.sh`; do not duplicate gate internals in YAML.
- [ ] Ensure path-ignore/path triggering still causes both proof and full Python jobs for any non-doc #85 change.
- [ ] Verify exact-head PR proof and full Python workflows. Also require existing Path4/Path5/PathN gates from the same proof run to remain green.
- [ ] Review exact diff for blocker findings and inspect GitHub review threads.
- [ ] PR body states finite generated-case evidence only, exact `Fraction`/`Rat`, production float exclusion, five case families, three drift negatives, and exact-head evidence only after completion.
- [ ] Merge only after exact-head required checks are green and no Critical/Important review finding remains.

## Execution correction from the earlier draft

The earlier plan draft used pytest commands. Repository proof CI actually runs `python -m unittest discover -s tests -v`, so this execution plan supersedes those pytest commands. No semantic/design requirement changed.
