# Receptivity Schedule Classifier Phase 2 Design

Issue: #99  
Parent roadmap: #96  
Phase 1: #97 / PR #98  
Base: `proof/narrative-dynamics-v0@b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b`

## Status

Approved architecture, committed design specification.

This document freezes the Phase 2 product contract, theorem boundary, schedule-family language, result semantics, and acceptance criteria. It does not authorize production implementation until the committed-spec review gate is passed and a separate implementation plan is written and reviewed.

## Goal

Extend the Phase 1 proof-backed `narrative-analyze model.toml` analyzer with a **receptivity schedule classifier** that recognizes common exact families `alpha(e)` and determines which exact consensus/non-consensus theorem route applies.

The classifier must distinguish:

1. a schedule being syntactically recognized as a family;
2. an asymptotic/product fact being formally proved;
3. a Path2 consensus or non-convergence conclusion being formally proved;
4. theorem coverage being insufficient, which remains `UNKNOWN`.

Python remains orchestration. Lean remains proof authority.

## Non-negotiable trust boundary

Phase 2 preserves the Phase 1 architecture:

```text
model.toml
   |
   v
exact Python parser + closed schedule AST
   |
   v
family recognition + candidate theorem route
   |
   v
closed generated Lean certificate
   |
   v
production executable semantics + theorem modules
   |
   v
bounded Lean compilation
   |
   v
PROVED / DISPROVED / UNKNOWN + provenance
```

Python may recognize family syntax, normalize exact parameters, select candidate routes, and compute finite exact witnesses used by a certificate.

Python must not:

- numerically truncate an infinite product and label the result `PROVED` or `DISPROVED`;
- infer asymptotic convergence from floats;
- duplicate `FitnessABMPathNExposure.step`;
- implement a second authoritative belief/exposure dynamics;
- emit arbitrary user-controlled Lean source;
- generalize a Path2 iff theorem to PathN;
- convert certificate compilation failure into `UNKNOWN`.

Every theorem-bearing positive or negative classification is accepted only after the generated Lean certificate compiles.

## Verified current theorem surface

The Phase 2 design is based on the exact theorem surface at base `b9a5e1fc3ccb6c3a253b01eb18c5296e3973a24b`.

The existing convergence module already provides the generic Path2 machinery:

```text
path2MultiplierProduct
path2_equal_exposure_iterate
path2_disagreement_product
path2_mean_iterate
path2_consensus_iff_product_tendsto_zero
```

The current iff route requires the actual model assumptions, including:

```text
ExposureParameters.Valid
initial equal Path2 exposures
initial allBroadcast
initial beliefs unequal
```

The current module also contains three exact fixed schedules and their proof artifacts:

```text
slowZeroSchedule
nearOneSchedule
harmonicSchedule
```

with existing exact results including:

```text
slowZeroSchedule_valid
slowZero_product
slowZero_product_tendsto_half
slowZero_not_consensus

nearOneSchedule_valid
nearOne_product
nearOne_not_convergent

harmonicSchedule_valid
harmonic_product
harmonic_consensus
```

These are valuable regression fixtures, but they are not generic parameterized family theorems.

At the approved base there is no generic theorem family for:

```text
polynomial decay
exponential decay
periodic schedules
alternating schedules
arbitrary finite-prefix + constant-tail product classification
```

Therefore Phase 2 includes new Lean theorem work. It is not only a Python routing change.

## Core mathematical object

For Path2 with equal initial exposure `e0`, the existing exact disagreement formula is governed by:

```text
P(k) = product_{r < k} (1 - 2 * alpha(e0 + r + 1))
```

and the current iff theorem uses:

```text
|P(k)| -> 0
```

as the exact criterion for nontrivial Path2 consensus under its assumptions.

Phase 2 must preserve the **post-incoming** lookup index:

```text
e0 + r + 1
```

A classifier using `e0 + r` is incorrect.

## Unifying schedule quantity: mixing mass

Phase 2 introduces the theorem-level concept:

```text
mixingMass(alpha) = min(alpha, 1 - alpha)
```

for valid `0 <= alpha <= 1`.

For a valid receptivity value:

```text
|1 - 2 * alpha| = 1 - 2 * mixingMass(alpha)
```

This is the correct unifying quantity because schedules approaching `0` and schedules approaching `1` can have the same absolute disagreement contraction.

Consequences:

- `alpha(e) -> 0` is not itself a consensus verdict;
- `alpha(e) -> 1` is not itself a non-convergence verdict;
- direct decay and near-one/complement decay can share the same absolute-product classification while having different signed-product behavior;
- finite zero factors (`alpha = 1/2`) must be handled explicitly because they make the disagreement product exactly zero from that point onward.

The new theorem layer should prove reusable product facts in terms of exact mixing mass where that reduces duplication, then expose family-specific corollaries.

## Schedule language

Phase 1 forms remain valid without semantic changes:

```text
constant
piecewise
named
```

Phase 2 adds closed parameterized forms. No arbitrary expression language is introduced.

### Common exact types

All theorem-bearing rational parameters remain string-encoded exact rationals in TOML and normalize to `ExactRat`.

Natural-number parameters remain TOML integers.

No float, decimal, scientific notation, user predicate, user code, or Lean expression is accepted.

### Decay target

Polynomial, harmonic, and exponential families use a closed target enum:

```text
zero
one
```

For a positive base decay `d(e)`:

```text
target = zero : alpha(e) = d(e)
target = one  : alpha(e) = 1 - d(e)
```

This keeps near-one schedules inside the same audited closed AST rather than introducing a generic symbolic `1 - expression` operator.

### Harmonic

Canonical syntax:

```toml
[schedule]
kind = "harmonic"
c = "1/2"
offset = 1
target = "zero"
```

Semantics:

```text
d(e) = c / (e + offset)
```

Structural family constraints:

```text
c > 0
offset >= 1
target in {zero, one}
```

Parameter validity (`0 <= alpha(e) <= 1`) remains a theorem-bearing claim and is not silently assumed merely because the family syntax is valid.

Harmonic is kept as a first-class family for UX/provenance even though its theorem implementation may reuse the polynomial exponent-1 route.

### Polynomial decay

Canonical syntax:

```toml
[schedule]
kind = "polynomial"
c = "1/2"
p = 2
offset = 1
target = "zero"
```

Semantics:

```text
d(e) = c / (e + offset)^p
```

Phase 2 restricts:

```text
p : Nat
p >= 1
c > 0
offset >= 1
```

Real-valued or rational exponents are explicitly outside Phase 2.

### Exponential decay

Canonical syntax:

```toml
[schedule]
kind = "exponential"
c = "1/2"
base = "1/2"
offset = 0
target = "zero"
```

Semantics:

```text
d(e) = c * base^(e + offset)
```

Structural family constraints:

```text
c > 0
0 < base < 1
offset : Nat
target in {zero, one}
```

### Periodic

Canonical syntax:

```toml
[schedule]
kind = "periodic"
values = ["1/4", "3/4"]
```

Semantics:

```text
alpha(e) = values[e mod period]
period = len(values)
```

The value list must be nonempty. Parameter validity is still formally certified.

### Alternating

Canonical syntax:

```toml
[schedule]
kind = "alternating"
a = "1/4"
b = "3/4"
```

Semantics:

```text
alpha(2m)   = a
alpha(2m+1) = b
```

Alternating is an ergonomic period-2 family. The theorem implementation should reuse periodic infrastructure rather than duplicate it.

### Finite piecewise + constant tail

The existing Phase 1 `piecewise` AST is retained:

```text
finite exact exposure -> value overrides
constant exact default branch
```

Phase 2 additionally classifies its Path2 product behavior as a finite prefix/finite exceptions followed by a constant tail.

The classifier must respect the concrete Path2 starting exposure `e0`: overrides that can never be reached after `e0` are not allowed to influence the concrete product classification.

### Named schedules

The existing trusted named schedule registry remains supported.

Initial theorem-backed named fixtures remain:

```text
slowZeroSchedule
nearOneSchedule
harmonicSchedule
```

An unknown/untrusted named identifier is an input error, not generated Lean text and not an `UNKNOWN` theorem result.

## Family recognition versus theorem classification

Family recognition is a data fact:

```text
family_id = polynomial
exact_parameters = {c=1/2, p=2, offset=1, target=zero}
```

It does not prove any convergence result.

The structured result gains:

```text
ScheduleFamilyInfo
  status: RECOGNIZED | UNRECOGNIZED
  family_id
  canonical_family_id
  exact_parameters
```

For example:

```text
alternating
  canonical_family_id = periodic
```

and:

```text
harmonic
  canonical_family_id = polynomial
  p = 1
```

The original user-visible family is preserved even when theorem routing delegates to a canonical family.

A trusted named schedule may be recognized as `named` while having a more specific canonical family only when that equivalence is itself fixed by trusted registry metadata or Lean proof. Python must not guess algebraic equivalence from source text.

## Theorem-bearing claim model

Phase 2 keeps the Phase 1 truth states unchanged:

```text
PROVED
DISPROVED
UNKNOWN
```

It does not introduce a fourth truth status.

The analyzer adds classification claims where applicable:

```text
alpha_tends_to_zero
alpha_tends_to_one
multiplier_abs_product_tends_to_zero
multiplier_nonzero_limit
path2_consensus
path2_nodewise_convergence
path2_oscillatory_nonconvergence
```

Claim semantics:

### `alpha_tends_to_zero`

The exact receptivity schedule tends to zero.

### `alpha_tends_to_one`

The exact receptivity schedule tends to one.

These are descriptive schedule facts. Neither one alone is a consensus verdict.

### `multiplier_abs_product_tends_to_zero`

For the concrete Path2 starting exposure `e0`:

```text
|path2MultiplierProduct p e0 k| -> 0
```

Under the existing Path2 assumptions, this is the necessary-and-sufficient product criterion for nontrivial consensus.

### `multiplier_nonzero_limit`

The signed multiplier product tends to some certified `L != 0`.

This is stronger than merely disproving zero-product convergence and is useful to distinguish stable disagreement from oscillation.

### `path2_consensus`

There exists one common real limit for both node beliefs.

For unequal initial beliefs, when the existing iff assumptions are discharged, the product-to-zero route is tagged:

```text
criterion_strength = NECESSARY_AND_SUFFICIENT
```

### `path2_nodewise_convergence`

Each node has a real limit, not necessarily the same limit:

```text
exists c0 c1,
  belief_0(k) -> c0 and belief_1(k) -> c1
```

This distinguishes a slow-zero stable split from oscillatory non-convergence.

### `path2_oscillatory_nonconvergence`

The certificate proves that nodewise convergence fails because the signed disagreement/product has incompatible subsequential behavior.

This is stronger than `path2_consensus = DISPROVED`.

## Criterion strength metadata

`ClaimResult` gains optional theorem-route metadata:

```text
criterion_strength:
  SUFFICIENT
  NECESSARY_AND_SUFFICIENT
  COUNTEREXAMPLE
  or absent
```

This field describes the logical role of the theorem route, not the truth value.

Examples:

- generic PathN global/reachable-interior consensus: `SUFFICIENT`;
- Path2 product iff route: `NECESSARY_AND_SUFFICIENT`;
- fixed exact negative witness theorem: `COUNTEREXAMPLE`;
- `alpha_tends_to_zero`: no consensus criterion strength required.

Existing Phase 1 callers remain valid by making the new field optional/defaulted.

## Path2 applicability rules

### Equal initial exposures

The generic product criterion is used only when the certificate proves:

```text
initial exposure 0 = initial exposure 1
```

If exposures differ, family recognition and schedule asymptotic facts may still be reported, but the existing Path2 product iff route is unavailable. `path2_consensus` then falls back to another valid theorem route, such as the generic PathN sufficient route, or remains `UNKNOWN`.

### Unequal initial beliefs

The current iff theorem assumes initial beliefs are unequal.

Phase 2 must not pretend this assumption is present when beliefs are equal.

The new Lean theorem layer should include a direct exact equal-belief preservation/consensus theorem so trivial equal-belief Path2 models do not become falsely `UNKNOWN` merely because the nontrivial product iff theorem is inapplicable.

### All-broadcast and validity

The existing `ExposureParameters.Valid` and `allBroadcast` assumptions remain explicit and certificate-checked.

A recognized family with invalid concrete parameters may produce:

```text
parameters_valid = DISPROVED
```

while dependent consensus claims remain `UNKNOWN` unless an independent theorem applies.

## Required new Lean theorem layer

Phase 2 adds a focused theorem module rather than expanding Python into proof logic.

Recommended module boundary:

```text
NarrativeDynamics/Core/FitnessABMPathNExposureScheduleClassifier.lean
```

It imports the existing exposure convergence module and does not modify the executable `step` semantics.

The module should expose reusable theorem groups.

### 1. Mixing-mass/product bridge

Public exact lemmas should connect valid receptivity to:

```text
|1 - 2 * alpha| = 1 - 2 * min(alpha, 1-alpha)
```

and provide reusable sufficient/exact routes for absolute product behavior under divergent versus summable mixing mass, including finite-prefix invariance and explicit zero-factor handling.

The theorem implementation may use Mathlib infinite-product/series infrastructure or an equivalent exact proof strategy. Numerical truncation is not an admissible substitute.

### 2. Polynomial/harmonic family theorems

For valid polynomial schedules, Phase 2 must cover at least these exact regimes for concrete Path2 starting exposure:

```text
p = 1:
  absolute multiplier product tends to zero
  therefore nontrivial Path2 consensus is proved

p >= 2:
  if a reached factor is exactly zero, consensus is proved
  otherwise the absolute multiplier product has a nonzero asymptotic magnitude
```

For `target = zero`, the signed product is eventually nonnegative/stable and the no-zero-factor `p >= 2` regime should prove nodewise convergence to distinct limits for unequal initial beliefs.

For `target = one`, the no-zero-factor `p >= 2` regime should prove repeated sign alternation with nonzero asymptotic magnitude and therefore nodewise non-convergence.

The `harmonic` family is the `p = 1` specialization and should reuse these theorems.

### 3. Exponential family theorems

For valid `0 < base < 1`, exponential mixing mass is summable.

Phase 2 must classify:

```text
reached zero multiplier factor -> consensus
otherwise -> absolute product does not tend to zero
```

For `target = zero`, the signed product should converge to a nonzero limit and produce stable non-consensus for unequal initial beliefs.

For `target = one`, the signed factors are eventually negative; with nonzero asymptotic magnitude the theorem route should establish oscillatory nodewise non-convergence.

### 4. Periodic/alternating family theorems

For a valid finite periodic schedule:

```text
if any repeated period value is strictly inside (0,1)
  -> repeated strict contraction
  -> absolute product tends to zero
  -> Path2 consensus

if all period values are in {0,1}
  -> absolute multiplier magnitude stays 1
  -> nontrivial Path2 consensus is disproved
```

When all period values are endpoints:

- all-zero schedule yields a constant positive signed product and stable disagreement;
- any periodically repeated `1` factor yields recurring sign changes and nodewise non-convergence.

`alternating` delegates to these period-2 results.

### 5. Finite piecewise + constant-tail theorem

For the existing finite piecewise schedule, the certificate may discharge the concrete finite prefix exactly, then use a generic constant-tail theorem.

Required behavior for the reached tail:

```text
any reached alpha = 1/2
  -> product becomes exactly zero
  -> consensus

tail q strictly inside (0,1)
  -> repeated |1-2q| < 1
  -> absolute product tends to zero
  -> consensus

tail q = 0 with no zero factor
  -> signed product converges to a nonzero constant
  -> stable non-consensus

tail q = 1 with no zero factor
  -> signed product alternates with nonzero magnitude
  -> nodewise non-convergence
```

The finite-prefix proof remains generated exact arithmetic from the closed AST; it must not become a numerical simulation.

### 6. Equal-belief Path2 theorem

Add a direct theorem proving persistence/consensus of equal Path2 beliefs under the required executable assumptions, so the nontrivial product iff theorem remains restricted to unequal initial beliefs without losing trivial consensus coverage.

## Named fixture compatibility

The existing fixed named theorems remain authoritative regression evidence:

```text
harmonicSchedule -> consensus
slowZeroSchedule -> no common consensus
nearOneSchedule -> no common nodewise limit
```

Phase 2 should prove that generalized routes agree with these fixtures where the family mapping is supported, but it must not delete or rewrite the existing fixture theorems merely to force the new abstraction.

The fixed fixtures remain useful independent regression tests for theorem drift.

## PathN boundary

The schedule classifier may derive exact schedule facts useful to PathN, but the new Path2 family classifier does not create a PathN iff theorem.

For `n > 2`:

- family recognition is still available;
- schedule limit facts may be available;
- existing global/reachable interior remains a sufficient consensus route;
- a family failing global interior is not automatically PathN non-consensus;
- Path2 non-consensus results must never be projected to PathN by analogy.

Generic PathN consensus value remains existential unless another exact theorem identifies it.

## Analyzer component changes

The implementation plan should preserve these boundaries.

### `model.py`

Extend the closed schedule AST and TOML parser with exact Phase 2 forms. It does not own theorem truth.

### `schedule_families.py`

New focused registry for:

- family descriptors;
- canonical family mapping;
- structural parameter checks;
- theorem-route identifiers selected from trusted static metadata.

This is separate from `named_schedules.py`, which continues to own trusted named fixture resolution.

### Lean family theorem module

Owns generic exact family/product theorems and model-independent mathematical facts.

It does not duplicate executable dynamics.

### Certificate generation

Add focused schedule-classifier templates, preferably separated from the already dense Phase 1 certificate generator.

Generated source may instantiate exact family definitions, finite-prefix witnesses, theorem assumptions, and theorem applications only.

### `analyze.py`

Orchestrates:

1. Phase 1 structural claims;
2. family recognition;
3. schedule-level theorem candidates;
4. Path2 family theorem candidates when assumptions apply;
5. fallback to existing PathN sufficient routes;
6. result merge with provenance conflict checks.

### `result.py`

Add:

```text
ScheduleFamilyInfo
CriterionStrength
optional ClaimResult.criterion_strength
```

Existing Phase 1 result construction remains source-compatible through defaults.

### CLI

Keep one command:

```text
narrative-analyze model.toml
```

Do not create a second classifier executable.

Add a schedule-classification section while preserving the existing claim-by-claim output and consensus summary.

## Conflict and precedence rules

Independent compiled claims may coexist.

Examples:

```text
alpha_tends_to_zero = PROVED
path2_consensus = DISPROVED
```

is valid and important for slow polynomial decay.

A stronger exact Path2 theorem may determine the Path2 consensus summary even when generic PathN sufficient coverage is `UNKNOWN`.

If two successfully compiled routes produce contradictory truth statuses for the same formal claim, the analyzer must raise an internal provenance/invariant failure. It must not pick one by priority.

`UNKNOWN` never overrides compiled proof evidence.

## Error taxonomy

Phase 1 error semantics remain unchanged.

Additional Phase 2 input errors include:

```text
unsupported target enum
harmonic/polynomial offset < 1
polynomial p < 1
nonpositive decay c
exponential base <= 0 or >= 1
empty periodic value list
malformed exact family parameter
```

The following remain analyzer failures rather than theorem statuses:

```text
certificate generation failure
Lean compile failure
certificate timeout/resource failure
certificate digest/provenance mismatch
internal contradictory compiled claims
```

Recognized family with no theorem coverage is not an error; the affected theorem claim is `UNKNOWN`.

## Bounded execution and source safety

All Phase 1 safety requirements continue to apply:

- deterministic generated identifiers;
- no user-controlled import/theorem/tactic text;
- no `sorry`, `admit`, user `axiom`, `unsafe`, `native_decide`, or external proof oracle in generated certificates;
- bounded `timeout --kill-after=10s 240s` focused Lean execution;
- ephemeral generated files by default;
- fail closed on nonzero exit or timeout;
- certificate digest verified before accepting result provenance.

Family parameter strings are data and are rendered only through exact trusted rational/Nat templates.

## Testing strategy

### Parser/model tests

Cover:

- every new family form;
- exact canonicalization;
- target enum;
- polynomial Nat exponent restriction;
- exponential base restriction;
- periodic nonempty restriction;
- float/scientific/source-injection rejection;
- Phase 1 `constant`, `piecewise`, and `named` compatibility.

### Family-recognition tests

Verify recognized family ID, canonical family ID, and exact parameters independently of theorem verdicts.

### Lean theorem tests

Add focused permanent Lean tests for:

- mixing-mass identity/bridge;
- harmonic/polynomial p=1 zero-product route;
- polynomial p>=2 nonzero-product route;
- direct-decay stable split;
- complement-decay oscillatory non-convergence;
- exponential direct/complement cases;
- periodic interior consensus;
- periodic endpoint stable/oscillatory cases;
- alternating as period-2 reuse;
- finite-prefix + tail reduction;
- equal-belief direct consensus;
- arbitrary starting exposure where the generic theorem claims support it.

### Analyzer real-certificate golden tests

Every `PROVED` or `DISPROVED` family classification requires at least one real generated Lean certificate test.

Mandatory paired regressions include:

```text
alpha(e) -> 0 + consensus
alpha(e) -> 0 + non-consensus

same polynomial family with p=1 versus p>=2

direct p>=2 stable non-consensus
complement p>=2 oscillatory non-convergence

periodic interior versus endpoint-only

piecewise interior tail versus endpoint tail
```

Keep the existing fixed named harmonic/slow-zero/near-one golden tests.

### Failure tests

- theorem route intentionally broken -> analyzer failure, not `UNKNOWN`;
- compile timeout -> analyzer failure;
- conflicting compiled evidence -> provenance/invariant failure;
- unsupported assumptions -> `UNKNOWN`;
- numerical diagnostic product behavior never authorizes proof status.

## CI design

CI changes are additive.

The implementation should add a focused schedule-classifier theorem gate and extend the existing analyzer gate, while preserving the existing finite-Path gate and full Lean build.

Recommended order inside the Lean proof job:

```text
Build Lean library
BB finite-path convergence
PathN exposure schedule classifier theorem gate
PathN exposure consensus analyzer
existing downstream trust/theorem gates
```

The classifier analyzer gate must not internally rerun the entire repository proof suite.

Exact-head PR CI remains the acceptance authority before merge.

## Acceptance criteria

Phase 2 is complete only when all of the following hold:

- Phase 1 schedule forms and CLI remain compatible;
- the new schedule language is closed and exact;
- harmonic, polynomial, exponential, periodic, alternating, piecewise-tail, and trusted named families are recognized;
- polynomial exponent is Nat-only with `p >= 1` in Phase 2;
- direct-to-zero and complement-to-one decay are represented without arbitrary symbolic expressions;
- family recognition is clearly separated from theorem truth;
- every theorem-bearing `PROVED` / `DISPROVED` classification compiles a Lean certificate;
- the generic Path2 product iff route is used only under its real assumptions;
- equal-belief Path2 models have an exact direct theorem route rather than abusing the unequal-belief iff theorem;
- `alpha -> 0` has both certified consensus and certified non-consensus examples;
- `alpha -> 1` complement schedules can distinguish consensus from oscillatory non-convergence according to product behavior;
- p=1 polynomial/harmonic and p>=2 polynomial regimes are formally distinguished;
- exponential summable-decay behavior is formally classified;
- periodic/alternating interior versus endpoint-only regimes are formally classified;
- finite piecewise + constant-tail schedules are classified with exact finite-prefix handling;
- Path2 stable non-consensus and nodewise oscillatory non-convergence are not collapsed into one label;
- PathN remains conservative and does not inherit a false iff theorem;
- no numerical truncation or floating-point asymptotic result authorizes a proof verdict;
- all new real certificate golden tests pass;
- existing Phase 1 analyzer tests remain green;
- full repository Lean/trust/resource gates remain green.

## Explicit non-goals

Not Phase 2:

- arbitrary symbolic schedule expressions;
- arbitrary real/rational polynomial exponents;
- numerical infinite-product guessing as proof;
- automated theorem synthesis;
- automated counterexample search;
- convergence-rate bounds (Phase 3);
- proof + simulation visualization (Phase 4);
- threshold-entry analysis (Phase 6);
- arbitrary connected graphs;
- dynamic graph topology;
- BB topology/belief co-evolution;
- a generic closed-form PathN consensus value.

## Implementation gate

After this specification is committed:

1. perform a self-review for placeholders, contradictions, theorem overclaiming, and trust-boundary drift;
2. user reviews the committed spec;
3. only after explicit spec approval, write a separate implementation plan under `docs/superpowers/plans/`;
4. review and commit that plan;
5. only then begin TDD production/theorem implementation.

No production Python, Lean theorem module, tests, or workflow changes are authorized by this document alone.