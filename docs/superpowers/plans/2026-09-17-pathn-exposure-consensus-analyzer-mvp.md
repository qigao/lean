# PathN Exposure Consensus Analyzer MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Phase 1 `narrative-analyze model.toml` analyzer from issue #97 so exact finite-Path exposure models receive claim-by-claim `PROVED`, `DISPROVED`, or `UNKNOWN` results backed by compiled Lean certificates and exact theorem provenance.

**Architecture:** Python owns TOML parsing, exact normalization, theorem-route selection, closed certificate templating, bounded Lean invocation, result assembly, and CLI rendering. Lean remains the only proof authority: generated certificates import the merged `FitnessABMPathNExposure` / convergence theorem modules, instantiate exact data, discharge assumptions, and apply existing theorems. Python must never duplicate executable belief/exposure dynamics or infer proof verdicts from simulation.

**Tech Stack:** Python >= 3.11 standard library (`tomllib`, `fractions`, `dataclasses`, `enum`, `subprocess`, `tempfile`, `pathlib`, `unittest`), Lean 4.32.0, Mathlib, exact `Rat`, existing `NarrativeDynamics.FitnessABMPathNExposureConvergence`, repository 240-second focused Lean timeout convention, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-09-17-pathn-exposure-consensus-analyzer-mvp-design.md`

**Issue:** #97, parent roadmap #96.

## Global Constraints

- Base architecture is already approved. Do not reopen the Python-orchestration / Lean-certificate-authority decision while implementing this plan.
- Implementation starts from `design/pathn-exposure-consensus-analyzer-mvp` only after this committed plan review gate is approved.
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
- The existing `tools/check_fitness_abm_pathn.sh` gate stays independent. The new analyzer gate must not re-run that whole gate internally; final verification runs them separately so CI does not duplicate expensive proof work.
- Every GREEN task must keep all earlier analyzer tests green before moving to the next task.

---

## Planned File Structure

Create a focused package instead of adding more root-level implementation scripts:

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

Modify only after focused implementation is green:

```text
.github/workflows/proof.yml
```

Do not modify merged theorem modules merely to make Python implementation easier. If a genuinely missing theorem blocks an acceptance criterion, stop and open a separate theorem-gap review instead of silently broadening this implementation.

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

- [ ] **Step 1: Write the Task 1 RED tests**

Cover at minimum:

```python
class ExactRatTests(unittest.TestCase):
    def test_accepts_integer_and_fraction_strings(self): ...
    def test_canonicalizes_sign_and_gcd(self): ...
    def test_rejects_toml_float(self): ...
    def test_rejects_decimal_string(self): ...
    def test_rejects_scientific_notation(self): ...
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

Use a strict rational grammar equivalent to:

```text
[+-]?DIGITS
[+-]?DIGITS/[1-9]DIGITS*
```

Do not allow decimal strings, scientific notation, whitespace-bearing expressions, or arbitrary `Fraction(...)` syntax.

- [ ] **Step 2: Verify Python preflight and RED**

```bash
python3 -c 'import tomllib; print("tomllib-ok")'
python3 -m unittest tests.test_narrative_analyzer_model -v
```

Expected RED: missing `narrative_analyzer.model`, not a malformed test harness.

- [ ] **Step 3: Commit valid RED**

```bash
git add tests/test_narrative_analyzer_model.py
git commit -m "test(analyzer): add exact model parser RED"
```

- [ ] **Step 4: Implement only exact normalized model parsing**

Use `tomllib` for decoding and exact integer numerator/denominator normalization. `model.py` must not emit Lean syntax.

Reject TOML `float` values explicitly. Enforce `n >= 2`, exact belief/exposure lengths, nonnegative natural exposures, schedule AST closure, duplicate piecewise-key rejection, and deterministic point sorting.

- [ ] **Step 5: Run GREEN**

```bash
python3 -m unittest tests.test_narrative_analyzer_model -v
```

- [ ] **Step 6: Commit GREEN**

```bash
git add narrative_analyzer/__init__.py narrative_analyzer/model.py \
  tests/test_narrative_analyzer_model.py
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

- [ ] **Step 1: Write RED tests for result invariants**

Required invariants:

- `PROVED` and `DISPROVED` require nonempty theorem/proof provenance.
- `UNKNOWN` may have no theorem but requires an explanatory note.
- claim IDs are unique inside `AnalysisResult`.
- consensus summary is derived from the strongest applicable exact consensus claim; it is not an independently mutable field.
- analyzer failures are exceptions, never `ClaimStatus` values.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_result -v
git add tests/test_narrative_analyzer_result.py
git commit -m "test(analyzer): add verdict provenance RED"
```

- [ ] **Step 3: Implement the immutable result layer**

Keep text rendering and theorem selection out of this task.

- [ ] **Step 4: Run GREEN with prior suite**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result -v
```

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_analyzer/result.py tests/test_narrative_analyzer_result.py
git commit -m "feat(analyzer): add proof verdict model"
```

---

### Task 3: Trusted named-schedule registry and theorem-route metadata

**Files:**
- Create: `narrative_analyzer/named_schedules.py`
- Create: `tests/test_narrative_analyzer_named_schedules.py`

**Initial whitelist:**

```text
slowZeroSchedule
nearOneSchedule
harmonicSchedule
```

All resolve through hard-coded developer metadata to definitions in:

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence
```

The registry stores only trusted static metadata:

- canonical user-facing ID;
- exact predeclared Lean definition;
- fixed-fixture requirements, where applicable;
- positive/negative theorem names available for that fixture;
- theorem assumptions the orchestrator must mechanically discharge.

The existing `slowZero_not_consensus`, `nearOne_not_convergent`, and `harmonic_consensus` theorems are concrete Path2 fixtures. Matching only the schedule ID is never sufficient to apply them.

- [ ] **Step 1: Write RED tests**

Cover:

- all three approved IDs resolve;
- unknown IDs fail before certificate generation;
- source-like IDs such as `slowZeroSchedule; axiom hacked : False` do not resolve;
- registry returns only predeclared Lean names;
- fixed-fixture routes require exact `n`, beliefs, exposures, threshold, and named schedule match.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_named_schedules -v
git add tests/test_narrative_analyzer_named_schedules.py
git commit -m "test(analyzer): add trusted schedule registry RED"
```

- [ ] **Step 3: Implement the closed registry**

No dynamic imports, `getattr` on user strings, user-populated Lean names, or theorem-name string concatenation.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_model \
  tests.test_narrative_analyzer_result \
  tests.test_narrative_analyzer_named_schedules -v

git add narrative_analyzer/named_schedules.py \
  tests/test_narrative_analyzer_named_schedules.py
git commit -m "feat(analyzer): add trusted theorem schedule registry"
```

---

### Task 4: Deterministic Lean certificate generator for structural claims

**Files:**
- Create: `narrative_analyzer/certificate.py`
- Create: `tests/test_narrative_analyzer_certificate.py`

**Public interfaces:**

```python
@dataclass(frozen=True)
class CertificateClaim:
    claim_id: str
    expected_status: ClaimStatus
    theorem: str
    assumptions: tuple[str, ...]
    exact_values: Mapping[str, str]

@dataclass(frozen=True)
class Certificate:
    source: str
    claims: tuple[CertificateClaim, ...]

class CertificateBuilder:
    def build_structural_positive(self, model: PathModel, claims: ...) -> Certificate: ...
    def build_structural_negative(self, model: PathModel, claims: ...) -> Certificate: ...
    def build_pathn_consensus(self, model: PathModel, eps: ExactRat) -> Certificate: ...
    def build_named_path2(self, model: PathModel, route: ...) -> Certificate: ...
```

The same normalized model + route must produce byte-identical source.

Every certificate imports production theorem code:

```lean
import NarrativeDynamics.Core.FitnessABMPathNExposureConvergence
```

It must not reimplement `step`, `kernelSchedule`, `ReachableInterior`, or consensus proofs.

For `constant` / `piecewise`, render only exact `ExposureParameters` and concrete `State n` data. For `named`, use only hard-coded registry definitions. If model threshold differs from a named fixture definition, construct a local parameter using the trusted named `receptivityAt`; fixture-specific theorem use remains forbidden unless the certificate also proves exact equality to the production fixture parameter.

- [ ] **Step 1: Write RED source-generation tests**

Assert:

- exact rationals render canonically without floats;
- negative rationals are unambiguous;
- generated vectors have exactly `n` elements;
- piecewise schedules render deterministic nested `if` branches from sorted natural keys;
- production convergence module is imported;
- user text never becomes an identifier/import/theorem/tactic;
- source never contains `sorry`, `admit`, user `axiom`, `unsafe`, or `native_decide`;
- generated helper names are deterministic internal names;
- structural positive certificates can prove `parameters_valid`, `initial_all_broadcast`, `exposure_law`, and `effective_alpha_lookup` when exact prechecks say they hold;
- structural negative certificates can prove `¬ p.Valid` or `¬ allBroadcast p s0` for concrete exact models when an exact counter-witness exists;
- a negative structural certificate never implies a negative consensus verdict by itself.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
git add tests/test_narrative_analyzer_certificate.py
git commit -m "test(analyzer): add Lean certificate generation RED"
```

- [ ] **Step 3: Implement exact rendering primitives**

Private helpers may include:

```python
_render_rat(q: ExactRat) -> str
_render_state(model: PathModel) -> str
_render_schedule(model: PathModel) -> str
_internal_name(prefix: str, normalized_bytes: bytes) -> str
```

Internal names may use deterministic hashes of normalized data, never raw user text.

Structural certificate facts use the existing proof surface:

- `parameters_valid`: `ExposureParameters.Valid`;
- `initial_all_broadcast`: convergence-module `allBroadcast`;
- `exposure_law`: `exposure_iterate`;
- `effective_alpha_lookup`: existing executable-to-kernel bridge and/or a generated lemma unfolding existing `kernelSchedule` / `exposureKernel` so the post-incoming `(k+1) * degree(i)` lookup is explicit.

For `parameters_valid = DISPROVED`, generate a concrete Lean proof of `¬ p.Valid`, typically by choosing the exact failing schedule value or threshold bound. For `initial_all_broadcast = DISPROVED`, generate a concrete index witness and prove `¬ allBroadcast p s0`. These structural negatives may block dependent theorem routes but must not themselves set consensus `DISPROVED`.

Recorded provenance for generated helper lemmas includes the generated lemma name plus the production theorem/definition dependency.

Do not compile certificates in this task; generation and execution remain separate.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v

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

**Route contract:** Python may compute an exact candidate `eps` for the closed constant/piecewise DSL, but Lean certifies both the global interior inequalities and the actual consensus theorem application.

For all constant/piecewise branch and default values `q`, candidate:

```text
eps = min(q, 1-q for every branch/default value q)
```

is route-selection data only. If candidate `eps <= 0`, do not attempt this sufficient theorem route. That is not `DISPROVED` consensus and does not prove `¬ ReachableInterior`.

**Production theorem target:**

```text
NarrativeDynamics.FitnessABMPathNExposureConvergence.trajectory_consensus_exists_of_global_interior
```

The certificate must discharge:

```text
p.Valid
2 <= n
allBroadcast p s0
0 < eps
forall e, eps <= p.receptivityAt e and p.receptivityAt e <= 1 - eps
```

- [ ] **Step 1: Add RED tests**

Required metadata after successful compilation will include:

```text
reachable_interior      PROVED
pathn_consensus_exists  PROVED
consensus_value_known   UNKNOWN   # orchestration, absent another exact theorem
```

with exact `eps` and theorem provenance.

Also test non-interior exact schedules return no candidate route and never synthesize negative consensus evidence.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 3: Implement exact candidate + Lean proof template**

For constant schedules, use direct exact arithmetic. For piecewise schedules, generate finite branch proofs (`by_cases` / `split_ifs`) over trusted integer keys plus default. Never inject a Python Boolean as proof authority.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest tests.test_narrative_analyzer_certificate -v

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

- [ ] **Step 1: Write RED eligibility tests**

Cases:

1. exact production split2 + `harmonicSchedule` fixture selects theorem-backed positive Path2 route;
2. exact split2 + `slowZeroSchedule` selects theorem-backed negative route;
3. exact split2 + `nearOneSchedule` selects theorem-backed negative convergence route;
4. same named schedule with changed beliefs, exposures, threshold, or `n` does not reuse a fixed-fixture theorem;
5. generic Path2 product/iff route is eligible only when its explicit assumptions, including equal initial exposures where required, are satisfied and a supported exact product-limit proof route exists;
6. no numerical product sampling is accepted as proof;
7. result/provenance distinguishes generic PathN sufficient evidence from Path2 iff evidence.

- [ ] **Step 2: Run RED**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_named_schedules \
  tests.test_narrative_analyzer_certificate -v
```

- [ ] **Step 3: Implement fixed theorem-backed certificate paths**

Generated certificates instantiate concrete input and prove equality with the existing fixture before applying a fixture-specific theorem. Negative verdict provenance names the production theorem directly.

For other Path2 inputs, expose product/iff machinery only when a supported exact product-limit theorem path exists. Otherwise that stronger claim is `UNKNOWN`; do not sample or approximate the infinite product.

For `consensus_value_known`, mark `PROVED` only if a concrete Lean proof identifies the limit. Generic PathN remains `UNKNOWN`. Never import the exposure-independent degree-weighted consensus value.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest \
  tests.test_narrative_analyzer_named_schedules \
  tests.test_narrative_analyzer_certificate -v

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

Default execution from repository root is semantically equivalent to:

```bash
timeout --kill-after=10s 240s \
  lake env lean -DmaxErrors=1 -DstderrAsMessages=false \
  /absolute/path/to/ephemeral/AnalyzerCertificate.lean
```

The implementation may also use Python subprocess timeout as defense in depth but must not weaken the 240-second upper bound. Working directory stays at repository root so imports resolve against the checked-out project.

- [ ] **Step 1: Write RED unit tests with injected process adapter**

Test:

- exit 0 returns immutable compile evidence;
- nonzero exit raises `CertificateCompileError` with diagnostics;
- timeout raises `CertificateTimeoutError`;
- ephemeral certificate is deleted after success/failure by default;
- optional debug retention changes retention only, never verdict semantics;
- runner never maps failure to `UNKNOWN`;
- command contains bounded timeout and exact Lean flags;
- source digest binds evidence to the exact certificate.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_runner -v
git add tests/test_narrative_analyzer_runner.py
git commit -m "test(analyzer): add certificate runner RED"
```

- [ ] **Step 3: Implement runner**

Use `tempfile.TemporaryDirectory()` by default. No shell interpolation of model data; pass argv as a list.

- [ ] **Step 4: Add one real Lean smoke certificate**

Compile a generated certificate importing the production convergence module and proving a trivial exact fact.

```bash
timeout --kill-after=10s 240s \
  python3 -m unittest tests.test_narrative_analyzer_runner -v
```

- [ ] **Step 5: Commit GREEN**

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

1. exact Python checks decide whether a supported certificate route is worth generating;
2. positive/negative verdict is emitted only after associated certificate compilation succeeds;
3. unsupported theorem coverage becomes `UNKNOWN` without invoking a certificate known not to match its assumptions;
4. an attempted certificate expected to prove its route that fails compilation is an analyzer failure;
5. theorem/provenance metadata emitted to user exactly matches claims bound into compiled certificate.

- [ ] **Step 1: Write RED orchestration tests with fake runner**

Cover full claim set:

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

- valid constant Path7 -> structural/global-interior/consensus claims become `PROVED` only after runner success; generic value remains `UNKNOWN`;
- valid piecewise global-interior Path7 -> same proof shape with exact `eps`;
- nonpositive global-interior candidate -> reachable-interior/consensus stay `UNKNOWN` unless another theorem route applies;
- exact invalid `ExposureParameters.Valid` -> `parameters_valid DISPROVED` only after negative structural certificate compiles; dependent consensus remains `UNKNOWN` absent independent theorem;
- exact false all-broadcast -> `initial_all_broadcast DISPROVED` only after negative structural certificate compiles; dependent all-broadcast theorem claims are `UNKNOWN`, not `DISPROVED`;
- slow-zero exact Path2 fixture -> theorem-backed `DISPROVED` consensus;
- near-one exact Path2 fixture -> theorem-backed non-convergence / consensus `DISPROVED`;
- harmonic exact Path2 fixture -> theorem-backed `PROVED`;
- any compile error propagates as analyzer failure, never `UNKNOWN`;
- provenance mismatch raises `ProvenanceMismatchError`.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_analysis -v
git add tests/test_narrative_analyzer_analysis.py
git commit -m "test(analyzer): add proof orchestration RED"
```

- [ ] **Step 3: Implement smallest explicit route orchestrator**

Prefer finite candidate functions:

```python
_structural_routes(model)
_global_interior_route(model)
_named_path2_route(model)
```

Do not build a generic theorem-search engine. Merge evidence deterministically; conflicting compiled statuses are an invariant error, not a priority choice.

- [ ] **Step 4: Run GREEN + all analyzer suites and commit**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v

git add narrative_analyzer/analyze.py tests/test_narrative_analyzer_analysis.py
git commit -m "feat(analyzer): compose certificate-backed verdicts"
```

---

### Task 9: CLI contract and deterministic human-readable output

**Files:**
- Create: `narrative_analyzer/cli.py`
- Create: `tools/narrative-analyze`
- Create: `tests/test_narrative_analyzer_cli.py`

**CLI:**

```bash
tools/narrative-analyze model.toml
```

Wrapper may be:

```bash
exec python3 -m narrative_analyzer.cli "$@"
```

Structured `AnalysisResult` is authoritative; text is a rendering.

Exit classes:

```text
0  analyzer completed and emitted valid claim statuses, including UNKNOWN/DISPROVED
2  invalid/unsupported user input
3  certificate generation/compile/timeout failure
4  internal provenance/invariant failure
```

- [ ] **Step 1: Write RED CLI tests**

Test deterministic claim ordering and that:

- every `PROVED`/`DISPROVED` detail shows provenance;
- generic PathN reports unknown closed-form value when appropriate;
- input error exits 2 without traceback by default;
- certificate failure exits 3 and is never printed as `UNKNOWN`;
- provenance/invariant failure exits 4;
- valid analyses containing `DISPROVED` or `UNKNOWN` still exit 0.

- [ ] **Step 2: Run RED and commit it**

```bash
python3 -m unittest tests.test_narrative_analyzer_cli -v
git add tests/test_narrative_analyzer_cli.py
git commit -m "test(analyzer): add CLI contract RED"
```

- [ ] **Step 3: Implement CLI and wrapper**

No JSON mode, plots, simulation output, automatic certificate retention, or extra topology flags in Phase 1.

- [ ] **Step 4: Run GREEN and commit**

```bash
python3 -m unittest tests.test_narrative_analyzer_cli -v

git add narrative_analyzer/cli.py tools/narrative-analyze \
  tests/test_narrative_analyzer_cli.py
git commit -m "feat(analyzer): add narrative-analyze CLI"
```

---

### Task 10: Real-certificate golden matrix and analyzer-only permanent gate

**Files:**
- Create: `tests/test_narrative_analyzer_golden.py`
- Modify/add: `tests/fixtures/narrative_analyzer/`
- Create: `tools/check_narrative_analyzer.sh`

This is the first acceptance-level test of Python -> generated Lean -> theorem -> structured result. Do not mock Lean here.

- [ ] **Step 1: Add positive golden models**

Required real-certificate cases:

- constant-alpha Path2 consensus;
- constant-alpha PathN consensus;
- finite-piecewise global-interior PathN consensus;
- exact post-incoming lookup regression proving `(k+1) * degree(i)` convention;
- existing harmonic Path2 theorem-backed consensus.

- [ ] **Step 2: Add negative golden models**

- exact `slowZeroSchedule` fixture -> theorem-backed non-consensus;
- exact `nearOneSchedule` fixture -> theorem-backed non-convergence.

- [ ] **Step 3: Add UNKNOWN golden models**

- valid schedule outside available proof criteria -> consensus `UNKNOWN`;
- global-interior sufficient-route failure never becomes `DISPROVED`;
- generic PathN proven consensus keeps `consensus_value_known = UNKNOWN`;
- Path2 model outside current iff/theorem assumptions remains `UNKNOWN` for stronger claim.

- [ ] **Step 4: Add invalid/security cases**

- TOML float instead of exact rational string;
- malformed rational;
- state length mismatch;
- unsupported schedule kind;
- duplicate piecewise key;
- source-like schedule payload;
- deliberately corrupted generated certificate -> analyzer failure, not `UNKNOWN`;
- generated source has no forbidden proof escape hatch.

- [ ] **Step 5: Run real golden suite**

```bash
timeout --kill-after=10s 240s \
  python3 -m unittest tests.test_narrative_analyzer_golden -v
```

If the complete golden matrix legitimately needs more than 240 seconds because it compiles multiple individually bounded certificates, keep each certificate bounded at 240 seconds and let the outer CI step use its own workflow-level timeout. Do not increase any individual certificate limit.

- [ ] **Step 6: Add analyzer-only gate script**

Create `tools/check_narrative_analyzer.sh` with `set -euo pipefail`. It runs only analyzer tests, for example:

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v
```

Do **not** call `tools/check_fitness_abm_pathn.sh` from this script. The existing PathN proof gate already runs independently and remains authoritative for theorem-library regressions.

- [ ] **Step 7: Run analyzer gate and commit**

```bash
bash tools/check_narrative_analyzer.sh

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

Immediately after existing `BB finite-path convergence`, add:

```yaml
- name: PathN exposure consensus analyzer
  timeout-minutes: 20
  run: bash tools/check_narrative_analyzer.sh
```

Do not remove, bypass, weaken, or duplicate the existing PathN gate. The new step is analyzer-only.

- [ ] **Step 2: Run analyzer suites fresh**

```bash
python3 -m unittest discover -s tests -p 'test_narrative_analyzer_*.py' -v
```

Expected: zero failures/errors.

- [ ] **Step 3: Run existing PathN theorem/trust gate fresh and separately**

```bash
bash tools/check_fitness_abm_pathn.sh
```

Expected existing required theorem evidence includes:

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

Expected: zero failures; proof-bearing golden cases compile real Lean certificates under bounded execution.

- [ ] **Step 5: Run full Lean build before merge-readiness claim**

Use the repository's normal full build:

```bash
lake build
```

Do not substitute a partial module build for this acceptance evidence. The focused 240-second convention remains on individual proof/certificate commands; the existing CI full-build step is the authoritative fallback if local full build duration is longer.

- [ ] **Step 6: Scope and trust audit**

```bash
git diff --name-status proof/narrative-dynamics-v0...HEAD

git grep -n -E 'sorry|admit|axiom|unsafe|native_decide' -- \
  narrative_analyzer tools/narrative-analyze tools/check_narrative_analyzer.sh \
  tests/test_narrative_analyzer_*.py || true
```

Review every match manually; negative security-test literals are allowed only when explicitly verifying rejection. There must be no production proof escape hatch.

Also audit for forbidden duplicate dynamics:

```bash
git grep -n -E 'belief_step|update_belief|simulate|incoming|exposure_step' -- \
  narrative_analyzer tools/narrative-analyze
```

Any hit must be orchestration/naming only, not an implementation equivalent to `FitnessABMPathNExposure.step`.

- [ ] **Step 7: Commit CI integration**

```bash
git add .github/workflows/proof.yml tools/check_narrative_analyzer.sh
git commit -m "ci: gate PathN exposure consensus analyzer"
```

- [ ] **Step 8: Push and require exact-head GitHub evidence**

```bash
git push
```

Before any merge-readiness claim require:

- exact-head `proof.yml` has a real non-skipped `Lean proof` job;
- `Build Lean library` is green;
- existing `BB finite-path convergence` is green;
- new `PathN exposure consensus analyzer` step is green;
- no workflow startup failure or skipped-only run is counted as proof evidence.

Record exact head SHA, workflow run ID, and job ID on #97.

---

## Acceptance Checklist

Do not mark #97 implementation complete until every item has fresh evidence:

- [ ] `model.toml` accepts only finite Path `n >= 2` and exact theorem-bearing rational strings.
- [ ] Constant, finite-piecewise+default, and trusted named schedule ASTs are closed and deterministic.
- [ ] TOML floats and source-like schedule payloads fail before certificate generation.
- [ ] Python contains no duplicate executable belief/exposure dynamics used to authorize verdicts.
- [ ] Generated certificates import actual merged theorem modules and compile independently.
- [ ] Every `PROVED`/`DISPROVED` claim records provenance bound to the compiled certificate.
- [ ] `parameters_valid` and `initial_all_broadcast` can be theorem-backed negative structural claims without incorrectly turning dependent consensus into `DISPROVED`.
- [ ] `UNKNOWN` means unavailable theorem coverage, not certificate failure.
- [ ] Certificate compile/timeout/provenance failures are nonzero analyzer errors.
- [ ] Post-incoming `(k+1) * degree(i)` lookup has a permanent real-certificate regression.
- [ ] Generic PathN global-interior consensus is theorem-backed and keeps generic consensus value unknown.
- [ ] Path2 routes expose stronger exact machinery only under actual theorem assumptions.
- [ ] `slowZeroSchedule` and `nearOneSchedule` negative fixtures are theorem-backed, not simulation-backed.
- [ ] Failure of global interior alone never produces `DISPROVED` consensus.
- [ ] No arbitrary graph, dynamic topology, counterexample search, convergence-rate UX, simulation UI, floating-point proof, all-broadcast-entry proof, or BB co-evolution scope entered Phase 1.
- [ ] Existing PathN proof/trust/resource gate remains green independently.
- [ ] Analyzer unit/golden/security gate is green independently.
- [ ] Full Lean build is green on exact head.
- [ ] GitHub Actions exact-head evidence is recorded on #97.

## Stop Conditions / Review Gates

Stop and request architecture/theorem review instead of improvising if any occurs:

1. an acceptance claim cannot be expressed using existing merged theorem surface without changing theorem semantics;
2. a named negative theorem applies only to a narrower fixture than expected;
3. generated certificate compilation requires copying `step`, `kernelSchedule`, or consensus proof logic;
4. a desired `DISPROVED` result is available only from failed sufficient conditions or numerical evidence;
5. actual CI resource limits require weakening current fail-fast policy;
6. exact TOML support would require accepting theorem-bearing floating-point input;
7. Path2 consensus-value identification requires importing the exposure-independent degree-weighted invariant.

At any stop condition, preserve the last green commit and open a focused design/theorem-gap issue. Do not hide the gap inside Python orchestration.

## Completion Boundary

This plan authorizes implementation only after the committed-plan review gate is explicitly approved. It does not authorize merging.

After implementation reaches the acceptance checklist, use `superpowers:verification-before-completion`, then `superpowers:requesting-code-review`, and only then evaluate PR/merge readiness.