# PathN exposure consensus analyzer MVP design

Issue: #97  
Parent roadmap: #96  
Foundation: #94 / PR #95  
Base: `proof/narrative-dynamics-v0@94089c2820e0dd18fc3b831ef15c640601a5f6da`

## Status

Approved architecture for the Phase 1 analyzer MVP.

This document freezes the product and trust boundary before an implementation plan is written. It does **not** approve production implementation by itself.

## Goal

Build the first usable **PathN Exposure Consensus Analyzer** on top of the already merged executable semantics and proof layer.

Primary user interface:

```text
narrative-analyze model.toml
```

The analyzer returns proof-backed claim statuses:

```text
PROVED
DISPROVED
UNKNOWN
```

Every `PROVED` or `DISPROVED` claim must identify the exact Lean theorem/proof source and the assumptions discharged for the concrete model.

The MVP is limited to finite paths and the theorem surface already available from PR #95. It must not present simulation, duplicated Python dynamics, failed sufficient conditions, or certificate-compilation failures as formal proof results.

## Normative semantic source

The theorem-bearing semantic source remains:

```text
NarrativeDynamics.FitnessABMPathNExposure.step
```

The analyzer must preserve the merged same-step ordering:

1. determine incoming broadcasters;
2. accumulate incoming exposure;
3. query `receptivityAt` using the post-incoming exposure;
4. update belief with that receptivity.

Inside the all-broadcast region, PR #95 proves:

```text
exposure_i(k) = exposure_i(0) + k * degree(i)
```

and therefore the transition from time `k` to `k+1` uses:

```text
alpha_i(k) = receptivityAt(exposure_i(0) + (k+1) * degree(i))
```

The analyzer must never replace this with a pre-incoming or `k`-indexed lookup.

The existing theorem module is:

```text
NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
```

with proof authority in namespace:

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence
```

Relevant existing theorem surface includes, among others:

```text
incoming_eq_degree
exposure_iterate
beliefs_iterate_eq_varyingTrajectory
path2_disagreement_product
path2_consensus_iff_product_tendsto_zero
slowZero_not_consensus
nearOne_not_convergent
harmonic_consensus
path_window_common_mass
varyingTrajectory_consensus_exists
trajectory_consensus_exists
trajectory_consensus_exists_of_global_interior
```

The analyzer consumes these results; it does not fork or mirror their dynamics.

## Architectural decision

### Python is orchestration; Lean is certificate authority

The frozen architecture is:

```text
model.toml
   |
   v
Python parser / normalization / UX
   |
   v
closed exact model AST
   |
   v
candidate applicability facts + generated Lean certificate module
   |
   v
existing executable semantics + existing theorem modules
   |
   v
bounded Lean compilation
   |
   v
structured AnalysisResult
   |
   +--> human-readable CLI output
```

Python may:

- parse TOML;
- parse exact rational strings;
- normalize the schedule AST;
- perform exact arithmetic needed to propose certificate values;
- select candidate theorem paths;
- generate closed Lean source from trusted templates;
- invoke Lean under bounded resource policy;
- assemble and format results after successful certificate compilation.

Python must not:

- independently reimplement `FitnessABMPathNExposure.step` and call its output `PROVED`;
- infer a formal theorem from floating-point simulation;
- upgrade a failed sufficient-condition check to `DISPROVED`;
- treat a Lean compile failure as `UNKNOWN`;
- accept user-controlled Lean source fragments.

### Rejected MVP architectures

A pure-Python theorem mirror is rejected because it creates a second semantic authority that can drift from Lean.

A Lean-only TOML/CLI stack is rejected for Phase 1 because it adds parsing and UX infrastructure before validating the analyzer product.

## Exact input model

The canonical normalized model contains only:

```text
Path n
Rat beliefs[n]
Nat exposures[n]
Rat threshold
Schedule AST
```

Phase 1 requires:

```text
n >= 2
len(beliefs)  = n
len(exposures) = n
```

All theorem-bearing rational values are supplied as strings and parsed exactly. JSON/TOML floating-point numbers are forbidden for beliefs, threshold, and schedule values.

Example:

```toml
[topology]
kind = "path"
n = 7

[initial]
beliefs = ["1", "3/4", "1/2", "1/2", "1/4", "0", "0"]
exposures = [0, 0, 0, 0, 0, 0, 0]

[dynamics]
threshold = "0"

[schedule]
kind = "constant"
value = "1/4"
```

Malformed rational syntax, length mismatches, unsupported topology, and unsupported schedule syntax are analyzer input errors, not theorem verdicts.

## Phase 1 schedule DSL

The schedule input is a closed AST. The accepted forms are:

```text
constant(q)
piecewise([(e0,q0), ...], default=q_tail)
named(schedule_id)
```

where every `e` is a natural exposure index and every `q` is an exact rational.

### Constant

`constant(q)` means:

```text
forall e, receptivityAt(e) = q
```

For `0 < q < 1`, the analyzer may propose an exact global interior witness such as:

```text
eps = min(q, 1-q)
```

but Lean must certify the required inequalities before the claim becomes `PROVED`.

### Finite piecewise plus default

The piecewise form is a finite exact lookup table plus a constant default branch:

```text
receptivityAt(e) = q_i     if e = e_i
                 = q_tail  otherwise
```

Rules:

- exposure keys are natural numbers;
- duplicate keys are invalid input;
- canonicalization sorts entries by exposure key;
- every branch value and the default remain exact rationals;
- no user predicate or source expression is accepted.

For the MVP global-interior path, Python may compute the exact finite set of branch/default values and propose:

```text
eps = min({q, 1-q | q is any branch/default value})
```

when positive. Lean must certify branch coverage and the global bound.

Failure to obtain a positive global-interior witness does **not** prove failure of `ReachableInterior`; it yields no positive certificate and therefore cannot by itself produce `DISPROVED`.

### Named schedules

`named(schedule_id)` resolves only through a fixed whitelist to theorem-backed Lean definitions already present in the repository or added in a separately reviewed theorem module.

Initial named schedules are intended for regression/proof demonstrations such as the already merged Path2 examples:

```text
slowZeroSchedule
nearOneSchedule
harmonicSchedule
```

The user-supplied ID never becomes Lean source text. It selects a trusted registry entry containing the Lean definition and permitted theorem paths.

## Certificate boundary

For one normalized model, the analyzer generates a bounded Lean module that:

1. imports production theorem modules;
2. declares the concrete exact model from trusted templates;
3. proves or instantiates only the assumptions needed for requested claims;
4. applies existing theorems;
5. contains no duplicated implementation of the executable transition or proof library.

Generated certificates must not copy or redefine:

```text
FitnessABMPathNExposure.step
kernelSchedule
ReachableInterior
trajectory_consensus_exists
path2_consensus_iff_product_tendsto_zero
```

A certificate may contain exact finite arithmetic facts, finite-vector definitions, schedule definitions from the closed DSL, and theorem instantiations.

A certificate compile succeeds only if all claims marked `PROVED` or `DISPROVED` in that certificate are accepted by Lean.

If compilation fails, the whole certificate attempt is an analyzer failure. The CLI must report diagnostics and must not downgrade the failure to `UNKNOWN`.

## Claim model

The analyzer evaluates claims independently. The Phase 1 claim set is:

```text
parameters_valid
initial_all_broadcast
exposure_law
effective_alpha_lookup
reachable_interior
path2_consensus          # only when n = 2
pathn_consensus_exists
consensus_value_known
```

Each emitted claim has exactly one status:

```text
PROVED | DISPROVED | UNKNOWN
```

and conceptually carries:

```text
claim_id
status
theorem
assumptions[]
exact_values{}
note
```

`theorem` is required for `PROVED` and `DISPROVED`. `UNKNOWN` may instead include the missing/unavailable proof condition.

### PROVED

Use `PROVED` only when the generated certificate mechanically discharges all assumptions of a concrete theorem proving the claim.

For generic PathN consensus this means, at minimum, a successful composition of the current theorem assumptions:

```text
ExposureParameters.Valid
2 <= n
initial allBroadcast
0 < eps
ReachableInterior p n s0 eps
```

followed by:

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence.trajectory_consensus_exists
```

or the global-interior convenience theorem when its stronger premise is certified.

### DISPROVED

Use `DISPROVED` only when Lean certifies the negation of the requested claim or an exact theorem composition that directly refutes it.

Examples include theorem-backed Path2 negative schedules such as:

```text
slowZero_not_consensus
nearOne_not_convergent
```

A failed sufficient condition is never enough.

### UNKNOWN

Use `UNKNOWN` when no currently admitted theorem proves or refutes the concrete claim.

Examples:

```text
no positive global-interior witness found
Path2 theorem assumptions do not hold
schedule is valid but outside supported theorem families
consensus value has no current theorem-backed closed form
```

`UNKNOWN` means theorem coverage is incomplete for this model. It is not evidence that the claim is false.

## Applicability and theorem composition

### Parameter validity

The normalized schedule and threshold are translated into the existing `ExposureParameters.Valid` contract.

A syntactically valid exact model may still violate parameter validity. If Lean certifies the negation for the concrete exact model, `parameters_valid` may be `DISPROVED`. Dependent convergence claims then remain `UNKNOWN` unless an independent theorem applies.

### Initial all-broadcast

The analyzer uses the existing `allBroadcast` predicate from the convergence module. It must not substitute a Python-only approximation.

When `allBroadcast` is certified, the analyzer may prove the exact exposure law and effective lookup formula through the existing bridge theorems.

When it is not established, the all-broadcast exposure/lookup claims and the generic PathN theorem path remain `UNKNOWN` rather than being applied outside their hypotheses.

### Reachable interior

The target theorem hypothesis is the existing:

```text
ReachableInterior p n s0 eps
```

The MVP supports a conservative global-interior proof path for `constant` and finite `piecewise` schedules. It may also use a named theorem-backed proof path.

Failure of this conservative checker does not establish `not ReachableInterior`.

### Path2 stronger analysis

When `n = 2`, the analyzer may use the stronger exact Path2 theorem family.

The current product/iff machinery has assumptions that must be surfaced and mechanically discharged, including equal initial exposures where required by the existing theorem.

The exact disagreement object is:

```text
d_k = d_0 * product_{r<k}(1 - 2 * alpha(e0+r+1))
```

The analyzer must preserve the distinction:

```text
PathN reachable-interior theorem = sufficient condition
Path2 product-to-zero theorem    = iff criterion under its own assumptions
```

They are not equivalent in strength.

### Consensus value

Generic exposure-dependent PathN consensus currently proves existence of a common real limit but does not provide a closed-form value.

The analyzer must not import the constant-alpha degree-weighted value from the separate exposure-independent model. PR #95 includes a Path3 theorem showing that the old degree-weighted mean is not generally invariant under row/time-dependent receptivity.

For the generic PathN theorem path:

```text
consensus_value_known = UNKNOWN
```

unless a separate exact theorem applicable to the concrete model identifies the value.

For supported Path2 equal-exposure cases, the existing mean/product theorem family may identify the exact arithmetic mean when its assumptions are certified.

## Result and CLI semantics

The structured result is the core contract. Text output is a rendering of it.

Example:

```text
Topology: Path 7
Exposure semantics: post-incoming lookup

parameters_valid          PROVED
initial_all_broadcast     PROVED
exposure_law              PROVED
effective_alpha_lookup    PROVED
reachable_interior        PROVED
pathn_consensus_exists    PROVED
consensus_value_known     UNKNOWN

Consensus: PROVED
Theorem:
  NarrativeDynamics.FitnessABMPathNExposureConvergence.trajectory_consensus_exists
Assumptions:
  parameters valid ✓
  n >= 2 ✓
  initial allBroadcast ✓
  ReachableInterior eps ✓
Certificate values:
  eps = 1/4
  beta = 1/8
  block = 6
  delta = (1/8)^6
Consensus value:
  unknown in closed form
```

The summary `Consensus` line is derived from the strongest applicable exact consensus claim:

- `PROVED` when a certified theorem establishes common consensus;
- `DISPROVED` when a certified theorem directly refutes common consensus/convergence for the concrete case;
- otherwise `UNKNOWN`.

The per-claim result remains authoritative and must always be available even when a summary is printed.

## Component boundaries

The implementation plan must preserve these responsibilities even if exact file names change.

### Analyzer model/schema

Owns:

- TOML schema;
- exact rational parser;
- normalized Path model;
- closed schedule AST;
- input validation/canonicalization.

Does not own theorem semantics.

### Analyzer certificate generator

Owns:

- trusted Lean templates;
- exact model/schedule rendering;
- theorem-path selection already decided by orchestration;
- generation of assumption proofs and theorem instantiations.

Does not reimplement executable dynamics.

### Analyzer runner

Owns:

- ephemeral certificate workspace;
- bounded Lean invocation;
- timeout/resource handling;
- compiler diagnostics;
- cleanup/debug retention policy.

A runner failure is an analyzer error, not a theorem result.

### Analyzer result/provenance

Owns:

- claim statuses;
- theorem names;
- discharged assumptions;
- exact certificate values;
- explanatory notes.

It must not invent proof provenance that was not present in the successfully compiled certificate path.

### CLI

Owns:

- input path handling;
- invocation of analyzer orchestration;
- human-readable output;
- nonzero exit behavior for input/internal/certificate failures.

The CLI is not the semantic core.

## Security and bounded execution

The generated certificate is data-driven, not source-injection driven.

Required invariants:

- user input is parsed into a closed AST before generation;
- user text never becomes an identifier, import path, theorem name, tactic, or arbitrary Lean term;
- named schedules resolve through a trusted whitelist;
- generated identifiers are deterministic internal names;
- certificate compilation uses the repository's existing bounded timeout/resource policy;
- generated files are ephemeral by default;
- optional debug retention never changes proof semantics;
- generated certificates contain no `sorry`, `admit`, user `axiom`, `unsafe`, or external proof oracle.

## Failure taxonomy

The following are analyzer failures, not claim statuses:

```text
invalid TOML
invalid rational syntax
length mismatch
unsupported topology
unsupported schedule form
duplicate piecewise exposure key
certificate generation failure
Lean certificate compile failure
certificate timeout/resource failure
internal theorem/provenance mismatch
```

A failure exits nonzero and does not emit a successful `AnalysisResult` pretending the affected claim is `UNKNOWN`.

## Testing strategy

### Positive golden cases

- constant-alpha Path2 consensus with exact certificate;
- constant-alpha PathN consensus through global interior;
- finite-piecewise global-interior PathN consensus;
- exact post-incoming lookup regression proving the `(k+1) * degree(i)` lookup convention;
- named harmonic Path2 theorem-backed consensus.

### Negative golden cases

- named `slowZeroSchedule` theorem-backed non-consensus;
- named `nearOneSchedule` theorem-backed non-convergence;
- exact invalid parameter fixture only when Lean certifies the negated validity claim.

### Unknown golden cases

- a valid supported schedule for which no current positive or negative theorem path applies;
- failure of the conservative global-interior checker without a proof of `not ReachableInterior`;
- generic PathN consensus value after existential consensus has been proved;
- Path2 model that does not satisfy the assumptions of the current iff theorem and has no alternate theorem path.

### Invalid-input cases

- TOML float where an exact rational string is required;
- malformed rational;
- `n < 2`;
- belief/exposure length mismatch;
- unsupported topology;
- unsupported schedule kind;
- duplicate piecewise exposure key;
- source-like schedule payload or identifier injection attempt.

### Trust regressions

Tests must assert that:

- no duplicate Python belief/exposure transition is used to authorize proof verdicts;
- every `PROVED`/`DISPROVED` fixture records a theorem name;
- intentionally broken generated Lean produces analyzer failure, not `UNKNOWN`;
- generated source contains no forbidden proof escape hatch;
- the existing full Lean proof gate remains green.

## Acceptance criteria

Phase 1 is complete only when all of the following hold:

- the analyzer uses the actual merged theorem modules on `proof/narrative-dynamics-v0`;
- `FitnessABMPathNExposure.step` remains the single executable semantic source for theorem-bearing behavior;
- all theorem-bearing numeric input remains exact rational;
- only the closed Phase 1 schedule DSL is accepted;
- the generated Lean certificate independently compiles under bounded execution;
- every `PROVED`/`DISPROVED` claim names its theorem/proof source and discharged assumptions;
- insufficient theorem coverage yields `UNKNOWN`, never fabricated certainty;
- Path2 iff analysis is exposed only when its actual assumptions are discharged;
- generic PathN consensus remains existential unless a separate theorem identifies its value;
- no arbitrary graph, dynamic graph, floating-point proof, automatic counterexample search, or BB co-evolution scope enters the MVP;
- analyzer tests cover positive, negative, unknown, invalid-input, certificate-failure, and source-injection cases;
- the existing Lean proof/trust/resource gates remain green.

## Explicit non-goals

Not Phase 1:

- arbitrary user functions for `alpha(e)`;
- symbolic theorem synthesis for schedule families;
- arbitrary finite connected graphs;
- dynamic topology;
- automatic counterexample search;
- plots or simulation UI;
- convergence-rate UX;
- proving entry into all-broadcast from arbitrary initial states;
- floating-point theorem claims;
- BB topology/belief co-evolution;
- a closed-form generic PathN consensus value.

These remain later phases under #96.

## Implementation gate

After this spec is committed and reviewed, the next allowed step is to write a separate implementation plan.

No production code should be written from #97 until the committed spec review gate is explicitly passed.
