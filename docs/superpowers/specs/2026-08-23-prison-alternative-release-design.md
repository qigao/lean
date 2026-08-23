# Alternative Prison Model and Witnessed Protocol Release — Design

Date: 2026-08-23
Status: approved design for implementation planning
Target branch: `proof/narrative-dynamics-v0`
Implementation scope: Python-only

## Purpose

This increment freezes the existing finite prison POMDP baseline and adds three empirical-control components around it:

1. an independently specified alternative prison model that does not reuse the baseline's posterior/planning equations;
2. an externally witnessed protocol-release format whose trust decision is supplied by an injected verifier rather than by home-grown cryptography;
3. a committed, versioned synthetic observational fixture that exercises the complete train → selection-validation → final-test path.

The goal is model comparison under a preregistered and externally witnessable protocol. This increment does not claim that either model is a true description of human behavior, does not add a real observational dataset, and does not extend the Lean semantic kernel.

## Existing boundaries that remain unchanged

The current runtime already provides:

- immutable scenario, trace, manifest, dataset, target, loss, and report identities;
- explicit train / selection-validation / final-test roles;
- proper-scoring categorical losses;
- frozen model candidates and final-comparison preflight checks;
- executable parameter, scenario, event, and outcome schemas;
- expected implementation hashes supplied by contracts or `TrustedExecutionPolicy`;
- fresh-process execution through `SubprocessModel`;
- result and aggregate-report artifacts;
- PR-level Lean/Python conformance and full-suite CI.

This work must use these boundaries rather than duplicate or weaken them.

The existing `finite-prison-pomdp` model is frozen. No existing state, action, observation, parameter, horizon, reward, policy, or transition semantics are changed.

## Non-goals

This increment does not:

- modify any `.lean` source;
- add new formal theorems;
- widen the prison baseline state/action space;
- add parameters to the prison POMDP baseline;
- implement a generic POMDP framework;
- implement Ed25519, X.509, RFC 3161, Rekor, or any other cryptographic protocol in the core package;
- claim a witness receipt is trustworthy merely because it is syntactically valid;
- bundle or validate real human observational data;
- perform causal identification, population inference, or representative-sampling analysis;
- prove source-observation independence beyond the existing conservative duplicate detection.

## Component 1: independently specified reactive prison model

### Scientific contrast

The baseline model uses a latent weak/strong guard state, Bayesian signal updating, guard-state prediction, and finite-horizon action-value planning. The alternative model must not be a renamed copy of that logic.

The alternative is a signal-reactive decision model:

- it accepts the same prison scenario surface so the two models can be evaluated on exactly the same dataset;
- it does not compute a Bayesian posterior;
- it does not propagate a posterior through guard persistence;
- it does not reuse the baseline's `_posterior_weak`, `_future_weak_probability`, `_scout_value`, or related implementation helpers;
- after scouting, it maps the observed signal directly to a reactive escape-vs-submit preference;
- the environment may still use `guard_persistence` when realizing an escape outcome, so ignoring transition planning is a real model restriction rather than an environment change.

The alternative therefore embodies a different behavioral hypothesis: actions react directly to cues rather than to an explicitly represented posterior belief and predicted future latent state.

### Shared scenario contract

The alternative accepts the same scenario fields as the baseline:

```text
prior_weak
signal_accuracy
guard_persistence
escape_reward
capture_cost
submit_reward
scout_cost
discount
horizon
```

Validation ranges remain the same:

- probabilities in `[0, 1]`;
- `signal_accuracy >= 0.5`;
- horizon is `1 | 2`;
- reward/cost fields finite, with non-negative escape reward, capture cost, and scout cost.

Sharing a scenario schema is an evaluation-control decision; it does not imply shared internal inference semantics.

### Free parameter

The alternative has exactly one fitted parameter:

```text
beta > 0
```

`beta` is an inverse-temperature / choice-sharpness parameter. No additional signal-weight, bias, persistence, or utility parameter is introduced in V1. This keeps candidate freedom directly comparable to the one-parameter baseline.

### Policy semantics

For horizon 1:

- scouting is unavailable;
- direct escape value is computed from `prior_weak`, `escape_reward`, and `capture_cost`;
- submit value is `submit_reward`;
- a numerically stable softmax with `beta` produces escape/submit probabilities.

For horizon 2:

- direct escape and submit retain the horizon-1 values;
- the scout action is evaluated by expected signal-reactive terminal behavior minus `scout_cost`, discounted by `discount`;
- `clear` is treated as favorable evidence for escape and `alarm` as unfavorable evidence, but the model does not normalize these cues into a posterior state probability;
- the cue-responsive terminal values are derived directly from the scenario's cue reliability and reward/cost surface;
- initial `scout | escape | submit` probabilities are produced by the same model-local stable softmax.

The model must expose deterministic expected policy coordinates for calibration and stochastic seeded episode traces for execution-boundary validation, as the baseline does.

### Runtime API

New module:

```text
narrative_dynamics/adapters/prison_reactive.py
```

Public API:

```python
FinitePrisonReactiveModel
create_prison_reactive_model
prison_reactive_contract
prison_reactive_source
```

The production source is a `SubprocessModel` with its own model name, version, implementation revision, four executable schemas, and external implementation pin through `TrustedExecutionPolicy`.

The implementation must not import `narrative_dynamics.adapters.prison_pomdp`.

## Component 2: shared prison categorical metric extractor

The observational fixture records initial action counts. Comparison therefore uses only a shared three-coordinate categorical extractor:

```text
initial.scout
initial.escape
initial.submit
```

New public function:

```python
prison_initial_action_metrics(trace)
```

It may live in a small shared adapter-metrics module or another focused module, but must not alter the existing `prison_policy_metrics` API.

The extractor must:

- read only the trusted `initial_policy` outcome coordinates;
- require all three coordinates;
- require finite numeric values;
- rely on the existing categorical-loss boundary to enforce non-negative normalized probability mass;
- have a stable callable/version identity suitable for preregistration.

Both baseline and alternative must emit the same trusted `initial_policy` schema so one extractor is used for both models.

## Component 3: versioned synthetic observational fixture

### Location and format

Add a committed JSON fixture under:

```text
fixtures/observations/prison_initial_choice_v1.json
```

It uses the existing record-oriented `ObservationDataset` JSON schema and loader. The fixture's declared content hash must be verified by the existing strict round-trip loader.

### Required provenance labels

The fixture must explicitly identify itself as synthetic and non-empirical. Its source/provenance payload must include semantics equivalent to:

```text
kind: synthetic_fixture
purpose: protocol_integration_test
empirical_human_data: false
population_representative: false
```

No documentation or test may refer to the fixture as observed human behavior.

### Partitions

The fixture contains exactly one partition for each required role:

```text
train
selection_validation
final_test
```

Each partition contains distinct record IDs, distinct scenario IDs, and no duplicate source-observation fingerprint according to the existing dataset boundary.

Scenarios span multiple values of at least:

- `prior_weak`;
- `signal_accuracy`;
- horizon 1 and horizon 2 where useful.

`guard_persistence` may vary as well because it distinguishes the planning and reactive hypotheses, but the fixture remains small enough for full CI execution.

### Counts

Each record contains integer counts for exactly:

```text
scout
escape
submit
```

Counts are committed constants. They are not generated at test runtime from either candidate model. This prevents the integration test from tautologically defining its target by whichever model is under test.

A `CategoricalTargetSpec` maps those counts to:

```text
initial.scout
initial.escape
initial.submit
```

using the existing deterministic target-construction path.

## Component 4: witnessed protocol release

### Motivation

`PreregisteredEvaluationProtocol.content_hash` currently provides an immutable, recomputable protocol identity but not proof that the protocol was published before final-test disclosure. The release layer makes external evidence attachable without pretending the core package can establish external trust by itself.

### Data types

New module:

```text
narrative_dynamics/observations/release.py
```

Primary immutable types:

```python
ProtocolRelease
WitnessReceipt
VerifiedProtocolRelease
ProtocolReleaseVerificationError
```

### `ProtocolRelease`

A release canonically binds:

```text
schema_version
release name
release version
protocol content hash
dataset content hash
target-spec content hash
ordered candidate content hashes
source repository identity or explicitly declared source revision identity
release content hash
```

The release must be reconstructible from its payload and reject a stale or forged declared content hash.

The release does not contain mutable execution results.

### `WitnessReceipt`

A receipt is evidence supplied by an external witnessing system. Canonical fields:

```text
schema_version
provider
authority
subject_hash
reference
claimed_at
proof payload
receipt content hash
```

Rules:

- `subject_hash` must be a canonical SHA-256 content hash;
- the receipt's own content hash is recomputed and validated;
- `claimed_at` is data supplied by the witness, not a trusted time merely because it parses;
- `proof` is canonical detached metadata whose interpretation belongs to the external verifier;
- duplicate receipt identities are rejected in a release-verification request.

### Trusted verifier boundary

The core package does not decide that a receipt is trustworthy by inspecting provider names or proof strings.

Verification is injected:

```python
verify_protocol_release(
    release,
    *,
    protocol,
    receipts,
    verifier,
) -> VerifiedProtocolRelease
```

The verifier is a trusted callable/component supplied by the caller. It receives the canonical release and receipt and returns a positive verification result only after checking the actual external evidence according to its provider-specific policy.

Before calling the verifier, core code must fail closed if:

- the release does not match the supplied protocol;
- dataset, target spec, or candidate identities differ;
- a receipt does not target `release.content_hash`;
- a receipt's declared content hash is invalid;
- receipts contain duplicate identities.

A `VerifiedProtocolRelease` records:

```text
release identity
protocol identity
verifier identity
verified receipt identities
verification status
```

At least one receipt must be successfully verified. A syntactically valid but unverified receipt never produces `VerifiedProtocolRelease`.

### External-witness scope

V1 deliberately specifies a verifier interface, not a concrete network backend. Tests use a deterministic test verifier that validates fixture evidence; production callers can later implement GitHub, transparency-log, timestamp-authority, institutional-registry, or signature-backed verifiers without changing the release identity model.

The deterministic test verifier must be clearly named and documented as test-only and must never be exported as a production trust provider.

## Component 5: released final comparison gate

New orchestration entry point:

```python
compare_released_models(...)
```

It composes, rather than replaces, the existing final-comparison implementation.

Preflight order:

```text
VerifiedProtocolRelease type/integrity
        ↓
release ↔ protocol identity match
        ↓
existing preregistered final-comparison preflight
        ↓
model execution
```

The function must reject release/protocol drift before any model call. It then delegates to the existing final comparison, preserving the existing checks for:

- final target payload and lineage;
- metric extractor identity;
- proper-scoring loss identity;
- simulation seed plan;
- frozen candidate set and parameters;
- runtime model identity;
- mutable-source reuse.

The resulting comparison manifest/report records the verified release identity and verified witness/verifier identity in addition to the existing comparison lineage.

No execution path may accept a raw `ProtocolRelease` where `VerifiedProtocolRelease` is required.

## Complete train → selection → final pipeline

The integration fixture test follows this exact order.

### 1. Load fixture

```text
load_observation_dataset
```

Assert the committed declared hash and synthetic provenance.

### 2. Construct training targets

Use only the train partition to construct categorical targets.

Training targets are used to rank a small finite `beta` grid independently for each model. The implementation may use the existing grid-calibration primitive over the train cases or add one thin multi-case helper if necessary, but final-test APIs must not be reused for training.

### 3. Selection-validation

Use only the selection-validation partition to choose one frozen `beta` per model from the train-derived candidate set.

The result must be a real `SelectionValidationReport`, not a fabricated manifest hash.

### 4. Freeze candidates

Create `FrozenModelCandidate` / `FrozenModelSpec` values from the actual selection reports for:

```text
finite-prison-pomdp
finite-prison-reactive
```

### 5. Create preregistered protocol

Freeze:

- full dataset identity and partition hashes;
- final target payload + construction manifest;
- shared extractor;
- categorical proper-scoring loss;
- common simulation seeds;
- baseline name;
- both frozen candidates;
- adequacy thresholds.

### 6. Create and externally witness release

Create `ProtocolRelease`, attach at least one `WitnessReceipt`, and pass it through an injected test verifier to obtain `VerifiedProtocolRelease`.

### 7. Execute untouched final comparison

Construct final targets from the final-test partition and call `compare_released_models` using the frozen candidates. No calibration, candidate selection, threshold change, seed change, extractor change, or loss change is allowed after the release identity is created.

### 8. Attest report

The final model-comparison report must remain compatible with `attest_report(...).require_integrity()` and retain parent lineage to the underlying final evaluations.

## Failure semantics

The new layer fails closed on identity and trust drift.

The following conditions must fail before any candidate model executes:

- release content hash mismatch;
- protocol hash mismatch;
- dataset hash mismatch;
- target-spec hash mismatch;
- candidate-set/hash mismatch;
- receipt subject does not equal release hash;
- forged/stale receipt content hash;
- duplicate receipt identity;
- verifier rejects all receipts;
- verifier identity drift when compared against a previously verified release payload;
- raw/unverified release supplied to final comparison;
- existing final-comparison preflight mismatch for targets, extractor, loss, seeds, frozen candidate, or runtime model.

Model/schema/process failures after successful preflight continue to use the existing typed runtime errors; the release layer does not wrap or erase them.

## Manifest and report identity

New release identities are canonical data objects. They do not create a circular hash graph.

A released comparison report includes the verified release identity in its report payload/manifest inputs and continues to inherit final evaluation manifests through the existing comparison report.

The release itself references protocol/dataset/spec/candidate hashes but not the eventual final result artifact.

## Public API boundaries

Expected package exports, subject to implementation-plan confirmation:

```python
# alternative adapter
FinitePrisonReactiveModel
create_prison_reactive_model
prison_reactive_contract
prison_reactive_source
prison_initial_action_metrics

# release layer
ProtocolRelease
WitnessReceipt
VerifiedProtocolRelease
ProtocolReleaseVerificationError
verify_protocol_release
compare_released_models
```

Test-only witness implementations are not exported from `narrative_dynamics`.

## Expected files

Production and fixture scope should remain focused around:

```text
narrative_dynamics/adapters/prison_reactive.py
narrative_dynamics/adapters/prison_metrics.py          # if a dedicated shared-metrics module is useful
narrative_dynamics/observations/release.py
narrative_dynamics/observations/__init__.py
narrative_dynamics/__init__.py                         # public exports only if project convention requires it
fixtures/observations/prison_initial_choice_v1.json
```

Tests should be split by obligation, for example:

```text
tests/test_prison_reactive_model.py
tests/test_protocol_release.py
tests/test_observational_fixture_pipeline.py
```

No `.lean` file belongs in the change set.

## TDD and CI acceptance criteria

Implementation follows strict RED → GREEN.

The initial RED must establish at minimum:

1. the independent reactive adapter/API does not yet exist;
2. release verification requires an external verifier and at least one verified receipt;
3. release/receipt/protocol identity drift is rejected before model execution;
4. raw release values cannot unlock final comparison;
5. committed fixture identity and partition roles are enforced;
6. train data is used only for candidate generation, selection-validation only for parameter freezing, and final-test only after release verification;
7. both models use the same final targets, seeds, extractor, loss, and adequacy thresholds;
8. alternative production source still passes external implementation pinning, fresh subprocess execution, and parameter/scenario/event/outcome schema validation;
9. report attestation remains valid;
10. no `.lean` source change is introduced.

Final feature CI and the subsequent PR #2 merge-context CI must both pass:

```text
Lean-generated conformance vectors
full Lean build
all Lean theorem tests
complete Python unittest suite
```

## Trust claims and explicit limits

After this increment the project may accurately claim:

- two independently specified finite prison behavioral hypotheses can be compared under the same preregistered observational protocol;
- candidate parameters are frozen before final evaluation;
- the protocol release has a canonical identity;
- final execution can be conditioned on at least one receipt accepted by a caller-supplied trusted verifier;
- the verified witness identity is carried into final comparison provenance;
- committed synthetic data exercises the full protocol deterministically in CI.

It must not claim:

- the committed fixture is empirical human evidence;
- receipt timestamps are trusted without an external verifier that actually establishes that property;
- the project implements a cryptographic signature or transparency-log verifier in V1;
- either prison model is scientifically validated by the synthetic fixture;
- alternative-model comparison proves causal or population-level adequacy;
- process isolation is a complete sandbox;
- implementation hashing proves the full transitive software supply chain.
