# PathN Exposure Consensus Analyzer MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase 1 `narrative-analyze model.toml` analyzer from issue #97 so exact finite-Path exposure models receive claim-by-claim `PROVED`, `DISPROVED`, or `UNKNOWN` results backed by compiled Lean certificates and exact theorem provenance.

**Architecture:** Python owns TOML parsing, exact normalization, theorem-route selection, certificate templating, bounded Lean invocation, result assembly, and CLI rendering. Lean remains the only proof authority: generated certificates import the merged `FitnessABMPathNExposure` / convergence theorem modules, instantiate exact data, discharge assumptions, and apply existing theorems. Python must never duplicate executable belief/exposure dynamics or infer proof verdicts from simulation.

**Tech Stack:** Python 3 standard library (`tomllib`, `fractions`, `dataclasses`, `enum`, `subprocess`, `tempfile`, `pathlib`, `unittest`), Lean 4.32.0, Mathlib, exact `Rat`, existing `NarrativeDynamics.FitnessABMPathNExposureConvergence`, repository 240-second Lean timeout convention, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-17-pathn-exposure-consensus-analyzer-mvp-design.md`

**Issue:** #97, parent roadmap #96.

## Global Constraints

- Base architecture is already approved. Do not reopen the Python-orchestration / Lean-certificate-authority decision while implementing this plan.
- Implementation branch starts from `design/pathn-exposure-consensus-analyzer-mvp` after the spec and this plan are reviewed.
- `NarrativeDynamics.FitnessABMPathNExposure.step` is the normative executable semantics and must remain unchanged.
- Same-step lookup is fixed: incoming broadcasters -> accumulated exposure -> `receptivityAt` post-incoming exposure -> belief update.
- Under all-broadcast, transition `k -> k+1` uses `receptivityAt (e0(i) + (k+1) * degree(i))`. A pre-incoming or `k * degree(i)` lookup is a correctness failure.
- The Python package must not implement an independent belief-update function, exposure-transition function, kernel trajectory, or numerical consensus simulation for theorem verdicts.
- All theorem-bearing numeric input is exact. TOML floats are rejected; rationals arrive as strings and are normalized with exact integer numerator/denominator arithmetic.
- Phase 1 topology is only finite `path` with `n >= 2`.
- Phase 1 schedule forms are only `constant`, finite `piecewise` plus exact default, and trusted `named` schedules.
- `PROVED` requires a successfully compiled concrete Lean proof for that claim.
- `DISPROVED` requires a successfully compiled concrete Lean proof of the negated claim or a directly applicable exact negative theorem.
- Failure of a sufficient condition is never `DISPROVED`.
- Lack of a supported theorem route is `UNKNOWN`; a generated-certificate compile error, timeout, resource failure, or provenance mismatch is an analyzer failure and must never be converted to `UNKNOWN`.
- Existing Path2 product/iff theorems are stronger than the generic PathN sufficient theorem but only under their actual assumptions, including equal initial exposure where required.
- Generic exposure-dependent PathN consensus is existential in the consensus value. Do not reuse the separate constant-alpha degree-weighted invariant.
- No arbitrary graph, dynamic topology, automatic counterexample search, plotting/simulation UI, convergence-rate UX, floating-point proof, all-broadcast-entry proof, or BB co-evolution enters this implementation.
- Generated Lean source is closed-template output. User input must never become an identifier, import path, theorem name, tactic, or arbitrary Lean expression.
- No generated or production certificate may contain `sorry`, `admit`, user `axiom`, `unsafe`, `native_decide`, or an external proof oracle.
- Focused Lean certificate commands use the repository convention `timeout --kill-after=10s 240s` and fail closed.
- Every GREEN task must keep all earlier analyzer tests green before moving to the next task.

---

## Planned File Structure

Create a focused Python package instead of adding more root-level scripts:

```text
narrative_analyzer/
  __init__.py
  model.py
  result.py
  named_schedules.py
  certificate.py
  runner.py
  analyze.py
  cli.py

tools/
  narrative-analyze
  check_narrative_analyzer.sh

tests/
  test_narrative_analyzer_model.py
  test_narrative_analyzer_result.py
  test_narrative_analyzer_named_schedules.py
  test_narrative_analyzer_certificate.py
  test_narrative_analyzer_runner.py
  test_narrative_analyzer_analysis.py
  test_narrative_analyzer_cli.py
  test_narrative_analyzer_golden.py
  fixtures/narrative_analyzer/
    constant_path7.toml
    piecewise_path7.toml
    harmonic_path2.toml
    slow_zero_path2.toml
    near_one_path2.toml
    unknown_path2.toml
```

Modify only after the focused implementation is green:

```text
.github/workflows/proof.yml
```

Do not modify the merged theorem modules merely to make the Python implementation easier. If an actually missing theorem blocks an acceptance criterion, stop at that point and open a separate theorem-gap review rather than silently broadening this implementation.

---

### Task 1: Exact model types and strict TOML normalization

**Files:**
- Create: `narrative_analyzer/__init__.py`
- Create: `narrative_analyzer/model.py`
- Create: `tests/test_narrative_analyzer_model.py`

**Public interfaces:**

```python
class ModelInputError(ValueError): ...

@dataclass(frozen=True)
class ExactRat:
    numerator: int
    denominator: int

@dataclass(frozen=True)
class ConstantSchedule: ...
@dataclass(frozen=True)
class PiecewiseSchedule: ...
@dataclass(frozen=True)
class NamedSchedule: ...
Schedule = ConstantSchedule | PiecewiseSchedule | NamedSchedule

@dataclass(frozen=True)
class PathModel:
    n: int
    beliefs: tuple[ExactRat, ...]
    exposures: tuple[int, ...]
    threshold: ExactRat
    schedule: Schedule

def load_model(path: Path) -> PathModel: ...
def parse_model(document: Mapping[str, object]) -> PathModel: ...
```

The normalized piecewise contract is:

```toml
[schedule]
kind = "piecewise"
default = "1/4"
points = [
  { exposure = 0, value = "1/3" },
  { exposure = 4, value = "2/5" },
]
```

Named schedule contract is:

```toml
[schedule]
kind = "named"
id = "harmonicSchedule"
```

- [ ] **Step 1: Write RED tests for exact rational parsing and model shape**

Test at minimum:

```python
class ExactRatTests(unittest.TestCase):
    def test_accepts_integer_and_fraction_strings(self): ...
    def test_canonicalizes_sign_and_gcd(self): ...
    def test_rejects_python_or_toml_float(self): ...
    def test_rejects_decimal_string(self): ...
    def test_rejects_zero_denominator(self): ...

class PathModelTests(unittest.TestCase):
    def test_constant_schedule_parses(self): ...
    def test_piecewise_points_are_sorted(self): ...
    def test_duplicate_piecewise_exposure_is_rejected(self): ...
    def test_named_schedule_parses_as_data_only(self): ...
    def test_n_less_than_two_is_rejected(self): ...
    def test_belief_length_mismatch_is_rejected(self): ...
    def test_exposure_length_mismatch_is_rejected(self): ...
    def test_negative_exposure_is_rejected(self): ...
    def test_unsupported_topology_is_rejected(self): ...
    def test_unsupported_schedule_kind_is_rejected(self): ...
```

Use a strict grammar for theorem-bearing rational strings, equivalent to:

```text
[+-]?DIGITS
[+-]?DIGITS/[1-9]DIGITS*
```

Do not allow decimal strings, scientific notation, whitespace-bearing expressions, or arbitrary `Fraction(...)` syntax.

- [ ] **Step 2: Run RED and verify only missing package/declaration failures**

```bash
python3 -m unittest tests.test_narrative_analyzer_model -v
```

Expected: failure because `narrative_analyzer.model` does not exist. Test syntax/import harness errors are not acceptable RED evidence.

- [ ] **Step 3: Commit the valid RED**

```bash
git add tests/test_narrative_analyzer_model.py
git commit -m "test(analyzer): add exact model parser RED"
```

- [ ] **Step 4: Implement only the exact normalized model layer**

Use `tomllib` for TOML decoding and integer numerator/denominator arithmetic for normalized rationals. Keep the exact type independent of Lean rendering; `model.py` must not emit Lean syntax.

Reject TOML float objects explicitly before any conversion. Enforce `n >= 2`, exact belief/exposure lengths, nonnegative natural exposures, schedule AST closure, duplicate piecewise-key rejection, and deterministic point sorting.

- [ ] **Step 5: Run GREEN**

```bash
python3 -m unittest tests.test_narrative_analyzer_model -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer tests/test_narrative_analyzer_model.py
git commit -m "feat(analyzer): add exact Path model schema"
```

---

### Task 2: Claim/result/provenance model and failure taxonomy

**Files:**
- Create: `narrative_analyzer/result.py`
- Create: `tests/test_narrative_analyzer_result.py`

**Public interfaces:**

```python
class ClaimStatus(Enum):
    PROVED = "PROVED"
    DISPROVED = "DISPROVED"
    UNKNOWN = "UNKNOWN"

@dataclass(frozen=True)
class ClaimResult:
    claim_id: str
    status: ClaimStatus
    theorem: str | None
    assumptions: tuple[str, ...]
    exact_values: Mapping[str, str]
    note: str | None

@dataclass(frozen=True)
class AnalysisResult:
    model_summary: Mapping[str, object]
    claims: tuple[ClaimResult, ...]

class AnalyzerError(RuntimeError): ...
class CertificateGenerationError(AnalyzerError): ...
class CertificateCompileError(AnalyzerError): ...
class CertificateTimeoutError(AnalyzerError): ...
class ProvenanceMismatchError(AnalyzerError): ...
```

- [ ] **Step 1: Write RED tests for invariants**

Required invariants:

- `PROVED` and `DISPROVED` require nonempty theorem/proof provenance.
- `UNKNOWN` may have no theorem but must carry an explanatory note.
- claim IDs are unique inside `AnalysisResult`.
- the consensus summary is derived from the strongest applicable exact consensus claim; it is never an independent mutable field.
- analyzer failures are exceptions, not `ClaimStatus` members.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_result -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_result.py
git commit -m "test(analyzer): add verdict provenance RED"
```

- [ ] **Step 4: Implement the minimal immutable result layer**

Keep text rendering out of this task. Do not add theorem-selection logic here.

- [ ] **Step 5: Run GREEN and all prior tests**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/result.py tests/test_narrative_analyzer_result.py
git commit -m "feat(analyzer): add proof verdict model"
```

---

### Task 3: Trusted named-schedule registry and theorem-route metadata

**Files:**
- Create: `narrative_analyzer/named_schedules.py`
- Create: `tests/test_narrative_analyzer_named_schedules.py`

**Purpose:** Keep user-provided `named(schedule_id)` as a lookup key only. It must never become a Lean identifier directly.

**Initial whitelist:**

```text
slowZeroSchedule
nearOneSchedule
harmonicSchedule
```

All three resolve to definitions in namespace:

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence
```

The registry stores trusted static metadata, including:

- canonical user-facing ID;
- exact Lean definition chosen by the implementation, hard-coded by developers;
- whether the existing theorem is tied to a specific fixture/model;
- positive or negative theorem names available for that fixture;
- exact preconditions the orchestrator must check before selecting that route.

Important: the existing `slowZero_not_consensus`, `nearOne_not_convergent`, and `harmonic_consensus` theorems are concrete Path2 fixtures. Do not apply them to an arbitrary user model merely because the schedule ID matches.

- [ ] **Step 1: Write RED tests**

Cover:

- all three approved names resolve;
- an unknown name returns an input error or unsupported-name error before certificate generation;
- source-looking names such as `"slowZeroSchedule; axiom hacked : False"` do not resolve;
- the registry returns only predeclared Lean identifiers;
- fixed-fixture theorem routes are selected only when `n`, beliefs, exposures, threshold, and schedule requirements match exactly.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_named_schedules -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_named_schedules.py
git commit -m "test(analyzer): add trusted schedule registry RED"
```

- [ ] **Step 4: Implement the closed registry**

Do not use dynamic imports, `getattr` on user strings, string concatenation into theorem names, or a generic “Lean name” field populated from TOML.

- [ ] **Step 5: Run GREEN with prior suites**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result \
  tests.test_narrative_analyzer_named_schedules -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/named_schedules.py tests/test_narrative_analyzer_named_schedules.py
git commit -m "feat(analyzer): add trusted theorem schedule registry"
```

---

### Task 4: Deterministic Lean certificate generator for exact model and structural claims

**Files:**
- Create: `narrative_analyzer/certificate.py`
- Create: `tests/test_narrative_analyzer_certificate.py`

**Public interfaces:**

```python
@dataclass(frozen=True)
class CertificateClaim:
    claim_id: str
    theorem: str
    assumptions: tuple[str, ...]
    exact_values: Mapping[str, str]

@dataclass(frozen=True)
class Certificate:
    source: str
    claims: tuple[CertificateClaim, ...]

class CertificateBuilder:
    def build_structural(self, model: PathModel) -> Certificate: ...
    def build_pathn_consensus(self, model: PathModel, eps: ExactRat) -> Certificate: ...
    def build_named_path2(self, model: PathModel, route: ...) -> Certificate: ...
```

Keep certificate construction deterministic: the same normalized model and route produce byte-identical source.

**Lean source boundary:** Every generated certificate imports the existing production theorem module:

```lean
import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
```

and uses aliases from the existing namespaces. It must not reimplement `step`, `kernelSchedule`, `ReachableInterior`, or consensus proofs.

For `constant` / `piecewise`, render only the exact `ExposureParameters` definition and concrete `State n` data. For `named`, reference only hard-coded registry definitions; if the model threshold differs from the theorem fixture threshold, construct a local parameter value with the trusted `receptivityAt` and the exact input threshold, then use fixture-specific theorems only if a generated equality proof identifies it with the existing named definition.

- [ ] **Step 1: Write RED source-generation tests**

Assert:

- exact rationals render canonically without floats;
- negative rationals are parenthesized unambiguously;
- generated finite vectors have exactly `n` elements;
- piecewise schedules render deterministic nested `if` branches from sorted natural keys;
- only the production convergence module is imported for this feature;
- generated source contains the exact post-incoming provenance theorems for structural claims;
- source never contains user-supplied identifiers;
- source never contains forbidden tokens `sorry`, `admit`, `axiom`, `unsafe`, or `native_decide`;
- source-like schedule payloads have already been rejected and cannot reach rendering.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_certificate.py
git commit -m "test(analyzer): add Lean certificate generation RED"
```

- [ ] **Step 4: Implement exact rendering primitives**

Add small private renderers such as:

```python
_render_rat(q: ExactRat) -> str
_render_state(model: PathModel) -> str
_render_schedule(model: PathModel) -> str
_internal_name(prefix: str, normalized_bytes: bytes) -> str
```

Internal names may be deterministic hashes of normalized data but never raw user text.

Structural certificate facts should prove, where exact pre-checks show the route is expected to hold:

- `parameters_valid` using `ExposureParameters.Valid`;
- `initial_all_broadcast` using the existing convergence `allBroadcast` predicate;
- `exposure_law` by instantiating `exposure_iterate`;
- `effective_alpha_lookup` using the existing executable-to-kernel bridge and/or a generated lemma unfolding the existing `kernelSchedule` / `exposureKernel` definitions so the `(k+1) * degree(i)` lookup is explicit.

The proof source recorded for generated helper lemmas must include both the generated lemma name and the production theorem/definition dependency it instantiates.

Do not compile certificates in this task; generation and execution remain separate boundaries.

- [ ] **Step 5: Run GREEN**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/certificate.py tests/test_narrative_analyzer_certificate.py
git commit -m "feat(analyzer): generate closed Lean certificates"
```

---

### Task 5: Generic PathN global-interior certificate path

**Files:**
- Modify: `narrative_analyzer/certificate.py`
- Modify: `tests/test_narrative_analyzer_certificate.py`
- Create: `tests/fixtures/narrative_analyzer/constant_path7.toml`
- Create: `tests/fixtures/narrative_analyzer/piecewise_path7.toml`

**Route contract:** Python may compute a candidate exact `eps` for the closed constant/piecewise DSL, but Lean must certify the global interior inequalities and the actual consensus theorem application.

For a finite set of branch/default values `q`, candidate:

```text
eps = min(q, 1-q for every reachable branch representation)
```

is only a route-selection/certificate-value calculation. It is not itself a proof verdict.

If the candidate is not positive, do not attempt the generic global-interior consensus certificate. That is absence of this sufficient route, not `DISPROVED` consensus.

**Production theorem target:** Prefer the existing convenience theorem when its stronger premise is exactly certified:

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence.trajectory_consensus_exists_of_global_interior
```

Its certificate must discharge:

```text
p.Valid
2 <= n
allBroadcast p s0
0 < eps
forall e, eps <= p.receptivityAt e and p.receptivityAt e <= 1 - eps
```

- [ ] **Step 1: Add RED tests for constant and piecewise certificates**

Expected exact certificate metadata includes:

```text
claim: reachable_interior
status after successful compile: PROVED
exact_values: eps=<exact rational>

claim: pathn_consensus_exists
production theorem: ...trajectory_consensus_exists_of_global_interior

claim: consensus_value_known
no positive theorem route -> UNKNOWN at orchestration layer
```

Add a non-interior schedule test showing `candidate_global_interior(...) is None` and explicitly asserting that this function does not return a negative consensus conclusion.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 3: Implement the exact candidate and Lean proof template**

For constant schedules, generate direct `norm_num`/arithmetic discharge.

For piecewise schedules, generate a finite `by_cases` / `split_ifs` proof over trusted integer keys plus default, with exact rational inequalities. Do not generate a Python-computed Boolean and inject it as an axiom.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 5: Commit**

```bash
git add narrative_analyzer/certificate.py \
  tests/test_narrative_analyzer_certificate.py \
  tests/fixtures/narrative_analyzer
git commit -m "feat(analyzer): certify PathN global-interior consensus"
```

---

### Task 6: Path2 stronger theorem routes and theorem-backed negative evidence

**Files:**
- Modify: `narrative_analyzer/named_schedules.py`
- Modify: `narrative_analyzer/certificate.py`
- Modify: `tests/test_narrative_analyzer_named_schedules.py`
- Modify: `tests/test_narrative_analyzer_certificate.py`
- Create: `tests/fixtures/narrative_analyzer/harmonic_path2.toml`
- Create: `tests/fixtures/narrative_analyzer/slow_zero_path2.toml`
- Create: `tests/fixtures/narrative_analyzer/near_one_path2.toml`
- Create: `tests/fixtures/narrative_analyzer/unknown_path2.toml`

**Existing theorem authority:**

```text
path2_disagreement_product
path2_consensus_iff_product_tendsto_zero
path2_mean_iterate
slowZero_not_consensus
nearOne_not_convergent
harmonic_consensus
```

- [ ] **Step 1: Write RED tests for theorem-route eligibility**

Cases:

1. exact existing `split2` + `harmonicSchedule` fixture selects theorem-backed positive Path2 route;
2. exact `split2` + `slowZeroSchedule` selects theorem-backed `DISPROVED` consensus route;
3. exact `split2` + `nearOneSchedule` selects theorem-backed `DISPROVED` convergence/consensus route;
4. same named schedule with changed beliefs, exposure, threshold, or `n` does not reuse fixed-fixture theorem and falls back to another valid route or `UNKNOWN`;
5. generic Path2 iff route is only eligible when its explicit assumptions, including equal initial exposures where required, are satisfied;
6. a Path2 route never reports the generic PathN sufficient theorem and the Path2 iff theorem as equivalent-strength evidence.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_named_schedules \
  tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 3: Implement fixed theorem-backed certificate paths**

Generated certificates must instantiate the concrete input and prove equality with the existing fixture before applying fixed-fixture theorems. The certificate provenance for negative verdicts must name the production theorem directly.

For other Path2 inputs, expose product/iff machinery only if a supported exact product-limit theorem path exists. Do not attempt numerical product sampling.

For `consensus_value_known`, only mark `PROVED` when a concrete Lean proof identifies the limit. It is valid for generic PathN to remain `UNKNOWN`. Do not import the separate exposure-independent degree-weighted mean.

- [ ] **Step 4: Run GREEN**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_named_schedules \
  tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 5: Commit**

```bash
git add narrative_analyzer \
  tests/test_narrative_analyzer_named_schedules.py \
  tests/test_narrative_analyzer_certificate.py \
  tests/fixtures/narrative_analyzer
git commit -m "feat(analyzer): add exact Path2 theorem routes"
```

---

### Task 7: Bounded Lean certificate runner with fail-closed diagnostics

**Files:**
- Create: `narrative_analyzer/runner.py`
- Create: `tests/test_narrative_analyzer_runner.py`

**Public interface:**

```python
@dataclass(frozen=True)
class CompileEvidence:
    certificate_digest: str
    returncode: int
    stdout: str
    stderr: str

class LeanCertificateRunner:
    def compile(self, certificate: Certificate) -> CompileEvidence: ...
```

Default execution from repository root must be semantically equivalent to:

```bash
timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 -DstderrAsMessages=false \
  /absolute/path/to/ephemeral/AnalyzerCertificate.lean
```

The implementation may use Python `subprocess.run(..., timeout=...)` in addition to the external `timeout` wrapper, but it must not weaken the existing 240-second upper bound. Keep the process working directory at the repository root so the generated certificate imports the checked-out project.

- [ ] **Step 1: Write RED unit tests with an injected subprocess adapter**

Test:

- exit 0 returns immutable compile evidence;
- nonzero exit raises `CertificateCompileError` with captured diagnostics;
- timeout raises `CertificateTimeoutError`;
- generated file is deleted after success and failure by default;
- optional debug-retention mode returns the retained path but does not alter verdict semantics;
- runner never maps failure to `UNKNOWN`;
- command contains the bounded timeout and exact Lean flags;
- certificate digest binds evidence to the exact generated source.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_runner -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_runner.py
git commit -m "test(analyzer): add certificate runner RED"
```

- [ ] **Step 4: Implement the runner**

Use `tempfile.TemporaryDirectory()` by default. No shell interpolation of model data. Pass command arguments as an argv list.

- [ ] **Step 5: Add one real Lean smoke certificate**

The test creates a tiny generated certificate importing the production convergence module and proving a trivial exact fact. Gate it as an integration test but keep it in the same unittest module.

Run:

```bash
timeout --kill-after=10s 240s \
  python3 -m unittest tests.test_narrative_analyzer_runner -v
```

Expected: unit paths and real Lean smoke compile pass.

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/runner.py tests/test_narrative_analyzer_runner.py
git commit -m "feat(analyzer): run Lean certificates fail closed"
```

---

### Task 8: Analyzer orchestration and claim-by-claim decision logic

**Files:**
- Create: `narrative_analyzer/analyze.py`
- Create: `tests/test_narrative_analyzer_analysis.py`

**Public interface:**

```python
def analyze_model(
    model: PathModel,
    *,
    runner: LeanCertificateRunner,
) -> AnalysisResult: ...
```

**Decision discipline:**

1. exact Python checks may determine whether a supported certificate route is worth generating;
2. a positive/negative verdict is emitted only after the associated certificate compiles;
3. unsupported theorem coverage becomes `UNKNOWN` without invoking a doomed certificate;
4. any certificate that was expected to prove its route but fails compilation is an analyzer failure;
5. theorem/provenance metadata emitted to the user must exactly match the claims bound into the compiled certificate.

- [ ] **Step 1: Write RED orchestration tests with a fake runner**

Cover the full claim set:

```text
parameters_valid
initial_all_broadcast
exposure_law
effective_alpha_lookup
reachable_interior
path2_consensus          # n == 2 only
pathn_consensus_exists
consensus_value_known
```

Required scenarios:

- valid constant Path7 -> structural claims + reachable interior + consensus become `PROVED` only after runner success; generic consensus value remains `UNKNOWN`;
- valid piecewise global-interior Path7 -> same proof shape with exact `eps`;
- failure to find a positive global-interior candidate -> `reachable_interior`/consensus remain `UNKNOWN` unless another theorem route exists;
- invalid parameter exact model can mark `parameters_valid` `DISPROVED` only if a negative certificate route is compiled; dependent consensus remains `UNKNOWN` absent an independent theorem;
- `slowZero` exact Path2 fixture -> theorem-backed `DISPROVED` consensus;
- `nearOne` exact Path2 fixture -> theorem-backed non-convergence/consensus `DISPROVED`;
- `harmonic` exact Path2 fixture -> theorem-backed `PROVED`;
- any fake compile error propagates as `AnalyzerError`, never `UNKNOWN`;
- provenance mismatch between requested claim and compiled certificate raises `ProvenanceMismatchError`.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_analysis -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_analysis.py
git commit -m "test(analyzer): add proof orchestration RED"
```

- [ ] **Step 4: Implement the smallest route orchestrator**

Keep theorem selection explicit and finite; do not build a generic theorem search engine.

Prefer separate candidate functions such as:

```python
_structural_route(model)
_global_interior_route(model)
_named_path2_route(model)
```

The orchestrator merges claim evidence deterministically and rejects conflicting proven statuses rather than silently choosing one.

- [ ] **Step 5: Run GREEN plus all analyzer unit suites**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/analyze.py tests/test_narrative_analyzer_analysis.py
git commit -m "feat(analyzer): compose certificate-backed verdicts"
```

---

### Task 9: CLI contract and stable human-readable output

**Files:**
- Create: `narrative_analyzer/cli.py`
- Create: `tools/narrative-analyze`
- Create: `tests/test_narrative_analyzer_cli.py`

**CLI contract:**

```bash
tools/narrative-analyze model.toml
```

The wrapper may execute:

```bash
exec python3 -m narrative_analyzer.cli "$@"
```

Keep the structured `AnalysisResult` authoritative. CLI text is only a renderer.

Exit classes:

```text
0  analyzer completed and emitted claim statuses, including UNKNOWN claims
2  invalid/unsupported user input
3  certificate generation/compile/timeout failure
4  internal provenance/invariant failure
```

Do not use the exit code to encode `PROVED` vs `DISPROVED`; those are valid analysis outcomes represented in the result.

- [ ] **Step 1: Write RED CLI tests**

Test:

- usage requires exactly one model path for MVP;
- claim table has deterministic ordering;
- every `PROVED`/`DISPROVED` detail shows theorem provenance;
- generic PathN prints `Consensus value: unknown in closed form` when appropriate;
- input error exits 2 without traceback by default;
- certificate failure exits 3 and is not rendered as `UNKNOWN`;
- internal provenance mismatch exits 4;
- valid results containing `DISPROVED` or `UNKNOWN` still exit 0.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_cli -v
```

- [ ] **Step 3: Commit RED**

```bash
git add tests/test_narrative_analyzer_cli.py
git commit -m "test(analyzer): add CLI contract RED"
```

- [ ] **Step 4: Implement CLI and wrapper**

No JSON mode, plotting, simulation output, automatic certificate retention, or extra topology flags in Phase 1.

- [ ] **Step 5: Run GREEN**

```bash
python3 -m unittest tests.test_narrative_analyzer_cli -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/cli.py tools/narrative-analyze \
  tests/test_narrative_analyzer_cli.py
git commit -m "feat(analyzer): add narrative-analyze CLI"
```

---

### Task 10: Real certificate golden matrix and trust regressions

**Files:**
- Create: `tests/test_narrative_analyzer_golden.py`
- Modify/add fixtures under: `tests/fixtures/narrative_analyzer/`
- Create: `tools/check_narrative_analyzer.sh`

This task is the first acceptance-level test of the complete Python -> generated Lean -> theorem -> structured result pipeline. Do not mock the Lean runner here.

- [ ] **Step 1: Add positive golden models**

Required real-certificate cases:

- constant-alpha Path2 consensus;
- constant-alpha PathN consensus;
- finite-piecewise global-interior PathN consensus;
- exact post-incoming lookup regression showing transition `k -> k+1` uses `(k+1) * degree(i)`;
- exact existing harmonic Path2 theorem-backed consensus.

- [ ] **Step 2: Add negative golden models**

Required real-certificate cases:

- exact existing `slowZeroSchedule` fixture -> theorem-backed non-consensus;
- exact existing `nearOneSchedule` fixture -> theorem-backed non-convergence.

- [ ] **Step 3: Add UNKNOWN models**

Required cases:

- valid schedule outside available proof criteria -> consensus `UNKNOWN`;
- conservative global-interior candidate failure does not become `DISPROVED`;
- generic PathN proven consensus keeps `consensus_value_known = UNKNOWN`;
- Path2 model failing current iff assumptions and lacking another theorem route remains `UNKNOWN` for that stronger claim.

- [ ] **Step 4: Add invalid/security cases**

Required cases:

- TOML float where exact rational string is required;
- malformed rational;
- state length mismatch;
- unsupported schedule kind;
- duplicate piecewise key;
- source-like schedule payload;
- deliberately corrupted generated certificate -> analyzer failure, not `UNKNOWN`;
- generated source scan confirms no forbidden proof escape hatch.

- [ ] **Step 5: Run the real golden suite under the repository timeout convention**

```bash
timeout --kill-after=10s 240s \
  python3 -m unittest tests.test_narrative_analyzer_golden -v
```

Expected: all positive, negative, unknown, and invalid/security expectations pass with real Lean certificate compilation where a proof verdict is claimed.

- [ ] **Step 6: Add permanent analyzer gate script**

Create `tools/check_narrative_analyzer.sh` with `set -euo pipefail` and, at minimum:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v
bash tools/check_fitness_abm_pathn.sh
```

The script may split fast unit tests and bounded real Lean golden tests for clearer logs, but every proof-bearing golden must remain in the permanent gate.

- [ ] **Step 7: Run permanent gate locally**

```bash
bash tools/check_narrative_analyzer.sh
```

Do not proceed on partial green. Record failing test/certificate if any.

- [ ] **Step 8: Commit acceptance gate**

```bash
git add tests/test_narrative_analyzer_golden.py \
  tests/fixtures/narrative_analyzer \
  tools/check_narrative_analyzer.sh
git commit -m "test(analyzer): add proof-backed golden gate"
```

---

### Task 11: Additive CI integration and final scope audit

**Files:**
- Modify: `.github/workflows/proof.yml`
- Modify only if required by final trust audit: `tools/check_narrative_analyzer.sh`

- [ ] **Step 1: Add one additive proof workflow step**

After the existing finite-Path proof gate, add a bounded step equivalent to:

```yaml
- name: PathN exposure consensus analyzer
  timeout-minutes: 20
  run: bash tools/check_narrative_analyzer.sh
```

Do not remove, bypass, or weaken `BB finite-path convergence`. Do not add a second full `lake build` unless evidence shows the existing build is insufficient for the analyzer gate.

- [ ] **Step 2: Run Python analyzer suites**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v
```

Expected: zero failures/errors.

- [ ] **Step 3: Run existing PathN proof gate fresh**

```bash
bash tools/check_fitness_abm_pathn.sh
```

Expected: all existing exact theorem builds/tests/trust audits pass, including:

```text
exposure_iterate
beliefs_iterate_eq_varyingTrajectory
path2_disagreement_product
path2_consensus_iff_product_tendsto_zero
slowZero_not_consensus
nearOne_not_convergent
harmonic_consensus
path_window_common_mass
trajectory_consensus_exists
trajectory_consensus_exists_of_global_interior
```

- [ ] **Step 4: Run analyzer permanent gate fresh**

```bash
bash tools/check_narrative_analyzer.sh
```

Expected: zero failures; real certificates compile within bounded execution.

- [ ] **Step 5: Run full repository Lean build before claiming merge readiness**

```bash
timeout --kill-after=10s 240s lake build
```

If the repository's normal full build legitimately exceeds this single-command budget, use the exact existing CI `lake build` step and record the GitHub Actions exact-head result instead of raising the limit locally.

- [ ] **Step 6: Scope audit**

```bash
git diff --name-status proof/narrative-dynamics-v0...HEAD
git grep -n -E 'sorry|admit|axiom|unsafe|native_decide' -- \
  narrative_analyzer tools/narrative-analyze tools/check_narrative_analyzer.sh \
  tests/test_narrative_analyzer_*.py || true
```

Review every match manually; string literals in negative security tests are allowed only when they explicitly verify rejection. There must be no production proof escape hatch.

Also verify there is no Python implementation with semantics equivalent to `FitnessABMPathNExposure.step`. Search likely names such as `belief_step`, `update_belief`, `simulate`, `incoming`, and `exposure_step`; any hit must be reviewed for orchestration-only behavior.

- [ ] **Step 7: Commit CI integration**

```bash
git add .github/workflows/proof.yml tools/check_narrative_analyzer.sh
git commit -m "ci: gate PathN exposure consensus analyzer"
```

- [ ] **Step 8: Push and require exact-head GitHub evidence**

```bash
git push
```

Required before merge-readiness claim:

- exact-head `proof.yml` has a real non-skipped `Lean proof` job;
- existing `BB finite-path convergence` is green;
- new `PathN exposure consensus analyzer` step is green;
- full `Build Lean library` is green;
- no workflow startup failure or skipped-only run is counted as proof evidence.

Record workflow run ID, job ID, and exact head SHA on #97.

---

## Acceptance Checklist

Do not mark #97 implementation complete until every item below has fresh evidence:

- [ ] `model.toml` accepts only finite Path `n >= 2` and exact theorem-bearing rational strings.
- [ ] Constant, finite-piecewise+default, and trusted named schedule ASTs are closed and deterministic.
- [ ] TOML floats and source-like schedule payloads fail before certificate generation.
- [ ] Python contains no duplicate executable belief/exposure dynamics used to authorize verdicts.
- [ ] Generated certificates import the actual merged theorem modules and compile independently.
- [ ] Every `PROVED`/`DISPROVED` claim records theorem/proof provenance bound to the compiled certificate.
- [ ] `UNKNOWN` means unavailable theorem coverage, not certificate failure.
- [ ] Certificate compile/timeout/provenance failures are nonzero analyzer errors.
- [ ] Post-incoming `(k+1) * degree(i)` lookup has a permanent real-certificate regression.
- [ ] Generic PathN global-interior consensus is theorem-backed and keeps generic consensus value unknown.
- [ ] Path2 theorem routes expose stronger exact machinery only under actual theorem assumptions.
- [ ] `slowZeroSchedule` and `nearOneSchedule` negative fixtures are theorem-backed, not simulation-backed.
- [ ] Failure of global interior alone never produces `DISPROVED` consensus.
- [ ] No arbitrary graph, dynamic topology, counterexample search, convergence-rate UX, simulation UI, floating-point proof, all-broadcast-entry proof, or BB co-evolution scope entered Phase 1.
- [ ] Existing PathN proof/trust/resource gate remains green.
- [ ] Analyzer unit/golden/security tests are green.
- [ ] Full Lean build is green on exact head.
- [ ] GitHub Actions exact-head evidence is recorded on #97.

## Stop Conditions / Review Gates

Stop and request architecture/theorem review instead of improvising if any of these occurs:

1. an acceptance claim cannot be expressed using the existing merged theorem surface without changing theorem semantics;
2. a named negative theorem applies only to a narrower fixture than expected;
3. generated certificate compilation requires copying `step`, `kernelSchedule`, or consensus proof logic;
4. a desired `DISPROVED` result is available only from failed sufficient conditions or numerical evidence;
5. actual CI resource limits require weakening the repository's current fail-fast policy;
6. implementing exact TOML support would require accepting theorem-bearing floating-point input;
7. Path2 consensus-value identification requires importing the exposure-independent degree-weighted invariant.

At any stop condition, preserve the last green commit and open a focused design/theorem-gap issue. Do not hide the gap inside Python orchestration.

## Completion Boundary

This plan authorizes implementation only after the plan review gate is explicitly approved. It does not itself authorize merging.

After implementation reaches the final acceptance checklist, use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`, and only then evaluate PR/merge readiness.