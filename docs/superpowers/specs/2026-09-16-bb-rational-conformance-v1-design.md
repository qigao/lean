# BB Python↔Lean rational conformance v1 design

Issue: #85  
Parent roadmap: #80  
Base: `proof/narrative-dynamics-v0@b8766d3cc9767bbd8636f9adc0d8114f8efabb1e`

## Context

The repository already has two useful but distinct surfaces:

1. Python production BB runtime in `narrative_dynamics.abm.bb_runtime`, where BB attachment mass is exact `fractions.Fraction` but agent propagation is executed through the production network runtime using floating-point agent fields.
2. Lean exact-rational BB replay / finite-path proofs, plus the Path5 Lean→JSONL generator and byte-for-byte golden comparison in the bounded Path4/Path5 gate.

#85 must not convert a Python float into theorem evidence. Instead it adds a deliberately narrow exact-rational conformance contract: Python generates deterministic cases using `Fraction`, Lean recomputes the declared rational semantics using `Rat`, and CI compares the immutable machine-readable corpus plus provenance.

This is finite case conformance evidence, not a universal language/runtime equivalence theorem.

## Goal

Create a versioned deterministic JSONL corpus containing exact finite BB/replay/propagation cases. Each case carries exact inputs, exact expected rational fields, discrete outputs/errors, and provenance. Python owns deterministic generation. Lean owns an independent exact-rational checker over the same declared contract.

CI must fail when:

- a declared expected rational changes;
- schema/version changes unexpectedly;
- generator/checker provenance drifts without an intentional corpus update;
- generated output differs byte-for-byte from the committed golden.

## Non-goals

V1 does not establish:

- universal equivalence between Python and Lean;
- equality of arbitrary Python float beliefs and Lean rationals;
- stochastic/Monte-Carlo equivalence;
- performance equivalence;
- a replacement for existing runtime tests;
- a replacement for existing Lean proof theorems;
- automatic acceptance of regenerated goldens.

## Architecture

### 1. Schema

Use JSON Lines with one canonical JSON object per case. Top-level schema:

```text
schema: "bb-rational-conformance/v1"
case_id: stable string
kind: "propagation" | "replay" | "error"
provenance: {...}
input: {...}
expected: {...}
```

Canonical JSON serialization:

- UTF-8;
- one object per line;
- keys sorted;
- compact separators `,` and `:`;
- final newline required;
- case ordering lexicographic by stable authored case order/id, never hash-map iteration.

### 2. Rational encoding

Every theorem-bearing rational is serialized structurally as:

```json
{"num": -3, "den": 8}
```

Requirements:

- denominator strictly positive;
- numerator/denominator reduced by the generator;
- zero encoded as `0/1`;
- no decimal or JSON floating-point value is accepted in a rational field.

Lean parses these directly into exact `Rat`; Python constructs them from `fractions.Fraction`.

### 3. Provenance

Each case contains:

```text
provenance.schema = "bb-rational-conformance/v1"
provenance.generator = "tools/generate_bb_rational_conformance.py"
provenance.generator_contract = <stable SHA-256>
provenance.lean_checker = "NarrativeDynamics/Conformance/BBRationalConformance.lean"
provenance.case_set = "v1"
```

The stable contract hash must be computed from explicitly named contract source files, not from the whole Git checkout and not from timestamps. V1 should hash the normalized bytes of the generator's exact-semantics module plus the schema/case-definition file(s). If the checker needs a separate declared hash, record it separately.

No volatile commit SHA is embedded in the golden because that would force regeneration on unrelated commits. Exact-head Git evidence belongs in CI/reporting, while the corpus provenance identifies the semantic generator/checker contract.

### 4. Exact Python reference semantics

Add a tool module dedicated to conformance, separate from the production float runtime, for example:

- `tools/bb_rational_conformance.py`
- `tools/generate_bb_rational_conformance.py`

It may reuse pure discrete validation helpers only when doing so does not import float belief arithmetic into the rational contract. Otherwise keep the exact evaluator small and explicit.

Use `Fraction` for all declared rational fields:

- fitness;
- attachment weight and trace mass;
- receptivity;
- threshold;
- belief;
- neighbor means;
- expected belief vectors.

The propagation contract for v1 mirrors the current exposure-independent semantics over a finite authored graph:

1. broadcaster iff `threshold ≤ belief` (inclusive);
2. collect incoming broadcasting neighbors;
3. if none, preserve belief;
4. otherwise update `belief' = (1-α) * belief + α * exact_mean`;
5. exposure increments by incoming broadcaster count.

Replay cases additionally use exact BB ordered attachment mass and graph growth. They need not run the production float propagation runtime to establish the rational expectation.

The production Python runtime may be exercised in separate tests for compatible discrete fields/error tags, but its float beliefs are not copied into the golden rational expectations.

### 5. Lean checker

Add:

`NarrativeDynamics/Conformance/BBRationalConformance.lean`

The checker reads the committed JSONL corpus and, for every case:

- checks schema and provenance fields;
- rejects malformed/non-reduced rational encodings;
- reconstructs exact graph/profiles/state/ticks;
- recomputes the declared exact semantics with existing Lean BB/network functions wherever the model matches existing production proof surfaces;
- checks all declared expected rational vectors and discrete fields;
- checks expected error category for negative runtime inputs;
- exits nonzero at the first mismatching case with stable case id/field diagnostics.

Do not implement a second general proof-bearing replay state machine merely for parsing the corpus. Reuse the existing Lean `FitnessABMReplay`, `FitnessAttachment`, finite-path, and #81 bridge functions where their input surface matches. Thin case adapters/parsers are allowed.

### 6. Case families

V1 contains a small authored set, not combinatorial generation.

#### A. Inclusive broadcast

A path case where a source belief equals its threshold exactly. Expected broadcaster flag is true and the receiver's exact updated belief/exposure are checked.

#### B. No-broadcast

A case where all neighboring beliefs are below threshold. Expected beliefs are unchanged and incoming exposure increment is zero.

#### C. Exact propagation with nontrivial mean

At least three vertices, mixed rational beliefs, and multiple broadcasting neighbors so the exact mean and receptivity multiplication are nontrivial.

#### D. Successful BB birth/replay

A small connected seed and authored ordered target list. Check:

- resulting canonical edge set;
- exact ordered birth mass;
- accumulated trace mass;
- node/birth/tick counters represented by the declared replay schema.

Include the established successive-birth `1/8` family or a smaller exact prefix compatible with current Lean replay fixtures.

#### E. Error case

At least one deterministic invalid input such as duplicate target or out-of-range target. Expected stable stage/code/field is checked, not an exception string.

#### F. Immutable Path4 regression identities

The existing fixed BBII/BIBI/IIBB Path4 fixtures remain authoritative. V1 should reference/copy only their declared stable rational outputs needed for conformance (for example prefix mass/base belief vectors) and verify they remain unchanged. Do not rewrite or regenerate the existing Path4 proof fixture from Python.

### 7. Golden corpus

Commit:

`conformance/bb_rational_v1.jsonl`

Generation writes to a temporary path in CI, then performs unconditional byte comparison to the committed golden:

```text
python3 tools/generate_bb_rational_conformance.py --output "$tmp"
cmp conformance/bb_rational_v1.jsonl "$tmp"
```

The generator must not overwrite the golden in normal CI.

### 8. Negative drift tests

Permanent tests must demonstrate rejection of at least:

1. schema version changed to an unsupported value;
2. provenance contract hash changed;
3. one expected rational numerator/denominator changed.

Prefer mutation of an in-memory/temp copy and run the same checker path. Tests restore/avoid mutation of the tracked golden.

### 9. CI ownership

Extend the existing proof workflow/gate rather than adding another broad workflow.

Bounded phases:

1. Python schema/generator unit tests;
2. generate temporary JSONL;
3. unconditional byte compare with committed golden;
4. build/run Lean checker with the repository's 240-second timeout convention;
5. negative drift tests;
6. existing source/trust audits and Path4/Path5/PathN gates remain intact.

### 10. Relationship to current Python production runtime

Current `narrative_dynamics.abm.bb_runtime` remains unchanged by v1 unless a tiny pure exact helper can be safely shared without changing behavior.

Important boundary:

- BB topology/attachment probability code already uses `Fraction` and is appropriate for exact cross-checks.
- Agent/profile belief fields in that runtime are normalized to Python floats and propagated through the production simulation engine; those values are runtime evidence only and must not be serialized as if they were exact rational proof values.

This separation must be stated in the implementation report and PR body.

## TDD sequence

### RED 1 — schema parser/serializer

Write Python tests requiring canonical rational encoding, canonical JSONL formatting, stable schema, and malformed/float-rational rejection before the generator exists.

### GREEN 1

Implement the schema types/helpers and canonical serializer.

### RED/GREEN 2 — exact Python evaluator

Add authored cases for inclusive broadcast, no-broadcast, nontrivial mean, successful BB mass, and one error. Implement the smallest `Fraction` evaluator needed for these declared semantics.

### RED/GREEN 3 — deterministic generator and golden

Generate the corpus twice and prove byte identity. Commit the first intentional golden only after reviewing the decoded cases.

### RED/GREEN 4 — Lean checker

Create a failing Lean consumer for the schema/cases, then implement the parser/adapters/checks using existing exact Lean surfaces. Check every authored case.

### RED/GREEN 5 — drift rejection

Mutate schema, provenance hash, and one rational expected value in temporary fixtures. All must fail for the expected reason; restored canonical corpus must pass.

### Final gate

Run full bounded conformance gate plus existing Path4/Path5/PathN proof workflow and Python discovery on the exact feature head.

## Expected file boundary

Likely additions:

- `tools/bb_rational_conformance.py`
- `tools/generate_bb_rational_conformance.py`
- Python tests for schema/generation/drift
- `conformance/bb_rational_v1.jsonl`
- `NarrativeDynamics/Conformance/BBRationalConformance.lean`
- `NarrativeDynamics/Tests/BBRationalConformance.lean` if theorem/trust consumers are separate
- minimal bounded CI wiring
- implementation/report documentation

Avoid modifying production Python BB runtime or baseline Lean dynamics unless a demonstrably behavior-preserving shared helper is necessary.

## Trust and resource boundary

- exact `Fraction`/`Rat` fields only;
- no float-to-rational coercion in the conformance corpus;
- no `sorry`, `admit`, new user `axiom`, `unsafe`, or `native_decide` proof oracle;
- no unbounded Lean resource settings;
- direct Lean invocations use `timeout --kill-after=10s 240s`;
- corpus changes are code-reviewed artifacts, never auto-accepted output.

## Acceptance

#85 is merge-ready only when:

- stable `bb-rational-conformance/v1` JSONL schema exists;
- all rational fields use reduced numerator/positive-denominator encoding;
- Python deterministically generates the committed cases with `Fraction`;
- Lean independently checks declared exact outputs with `Rat` and existing model surfaces;
- cases include inclusive broadcast, no-broadcast, nontrivial exact propagation, successful BB replay/mass, and stable error behavior;
- fixed Path4 identities remain immutable regression references;
- CI performs unconditional generated-vs-golden byte comparison;
- schema drift, provenance drift, and rational-output mutation each fail permanently tested checks;
- production Python float beliefs are explicitly excluded from proof claims;
- exact-head proof/Python CI passes and independent review has no blocker;
- PR/report states that the result is finite generated-case conformance, not universal Python↔Lean equivalence.
