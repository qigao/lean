# Narrative Empirical / External Validation V1 Design

Date: 2026-08-28

Status: written for review

Roadmap: #38 — P3 Empirical / External Validation V1

Integrated base: `proof/narrative-dynamics-v0@f6dbe9e67d6dda064fa332989187e7e8a604039a`

Design branch: `work/narrative-empirical-external-validation-v1-design`

## 1. Purpose

Narrative Empirical / External Validation V1 is the first P3 research layer after the completed P2 synthetic-identification program.

The existing runtime already has rigorous machinery for immutable observational datasets, target construction, train/selection/final partitioning, finite calibration, held-out evaluation, proper scoring, model comparison, preregistration, release witnessing, and synthetic identification. P3 does not replace any of those components. It adds a thin, explicit evidence bridge that permits a frozen **external observational** dataset to enter the existing evaluation pipeline while preserving a hard scientific boundary between predictive evidence and claims about latent cognition.

The primary P3 question is:

> Given a frozen external observational dataset and a preregistered, externally witnessed evaluation protocol, do the existing Reactive, Intentional, and Planning/POMDP narrative model families achieve external predictive adequacy, and are their observable predictions separated by the data under the frozen protocol?

P3 V1 is deliberately not a causal-identification phase and not a new cognitive-model phase.

## 2. Scientific claim firewall

The central contract is:

> **Predictive fit does not identify or confirm real latent cognition.**

This is a type and report boundary, not only documentation guidance.

### 2.1 Permitted external-validation language

P3 may report only protocol-scoped observable conclusions such as:

- `PREDICTIVE_ADEQUACY_MET`;
- `PREDICTIVE_ADEQUACY_NOT_MET`;
- `PREDICTIVELY_SEPARATED_UNDER_PROTOCOL`;
- `NOT_PREDICTIVELY_SEPARATED_UNDER_PROTOCOL`;
- `CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`;
- `NOT_CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`.

A report may also state which frozen model has lower Brier or Log loss, the magnitude and direction of a preregistered score difference, and which finite parameter values remain compatible with the external selection protocol.

### 2.2 Forbidden conclusions

External validation must never emit successful statuses or canonical conclusions equivalent to:

- `EMPIRICALLY_IDENTIFIED`;
- `STRUCTURALLY_IDENTIFIED`;
- `COGNITIVE_MECHANISM_CONFIRMED`;
- `LATENT_COGNITION_IDENTIFIED`;
- `REAL_PERSON_GOAL_RECOVERED`.

P2 `IdentificationStatus` remains synthetic-protocol-only and is not extended.

A singleton externally compatible parameter set is **constrained under the frozen external protocol**, not identified.

### 2.3 No free-text canonical conclusion field

The canonical P3 report contains typed statuses and a fixed claim scope rather than an arbitrary narrative conclusion string. Human-authored discussion can exist outside the canonical report, but the runtime itself must not serialize an empirical result using synthetic-identification vocabulary.

The fixed report claim scope is:

```text
external_observational_predictive_only
```

## 3. Existing foundations to reuse

P3 composes the current system rather than creating a parallel empirical framework.

Authoritative existing components include:

- `narrative_dynamics.observations.dataset.ObservationDataset`;
- `ObservationPartition` / `ObservationPartitionRole` and their existing cross-partition uniqueness checks;
- `narrative_dynamics.observations.targets` target construction and target lineage;
- `narrative_dynamics.observations.training.fit_training_target_grid()`;
- selection-validation and final-test infrastructure in `narrative_dynamics.validation`;
- `FrozenModelSpec` and `PreregisteredEvaluationProtocol` in `narrative_dynamics.observations.preregistration`;
- `ProtocolRelease`, `WitnessReceipt`, `VerifiedProtocolRelease`, `verify_protocol_release()`, and `compare_released_models()` in `narrative_dynamics.observations.release`;
- `ComparisonModel` / `ModelComparisonReport` and the existing final-partition comparison path;
- categorical Brier and categorical Log losses;
- `ExperimentManifest`, stable content hashes, and report attestation;
- the existing Reactive, Intentional, and Planning/POMDP runtime families;
- P2 `SyntheticIdentificationReport` as method-validation evidence only.

The existing observation dataset schema already includes canonical `source`, `provenance`, record/case identities, partitions, and a content hash. P3 therefore does **not** require an `ObservationDataset` schema-version change.

The existing repository fixture `fixtures/observations/prison_initial_choice_v1.json` explicitly declares synthetic, non-empirical provenance. It is valid for protocol-integration tests but must be rejected by the P3 external-evidence path.

## 4. Architectural decision

### 4.1 Chosen approach: thin external-evidence bridge

Add two focused modules:

```text
narrative_dynamics/observations/external.py
narrative_dynamics/external_validation.py
```

Responsibilities remain separated:

```text
external source snapshot
        |
        v
ExternalEvidenceDeclaration
        |
        v
existing ObservationDataset / target construction
        |
        v
TRAIN -> selection-validation
        |
        v
existing FrozenModelSpec / PreregisteredEvaluationProtocol
        |
        v
ExternalValidationPreregistration
        |
        +---- Brier ProtocolRelease -> WitnessReceipt -> VerifiedProtocolRelease
        |
        +---- Log   ProtocolRelease -> WitnessReceipt -> VerifiedProtocolRelease
        |
        v
locked dual final evaluation
        |
        v
ExternalValidationReport
```

The external layer validates composition and claim scope. It does not duplicate the dataset, calibration, validation, release, or model-comparison engines.

### 4.2 Rejected: empirical mode inside `identification.py`

Rejected because P2 has a deliberately narrow synthetic truth boundary. Extending `IdentificationStatus` to observed data would make the type system imply that predictive fit can inherit synthetic generator-truth semantics.

P3 must not import `IdentificationStatus` or call P2 identification-status constructors.

### 4.3 Rejected: ObservationDataset V2

Rejected because the existing schema already content-hashes source, provenance, records/cases, scenarios, partitions, and observation identity. External eligibility is a research-protocol concern, not a reason to rewrite the dataset container.

### 4.4 Rejected: raw event-level empirical platform

Rejected for V1. Dataframes, databases, missing-data pipelines, survey weights, hierarchical models, and event-level likelihoods are separate research problems and would obscure the external-validation claim boundary.

## 5. External evidence declaration

`narrative_dynamics.observations.external` owns only the positive external-evidence declaration and source-lineage checks.

### 5.1 `ExternalEvidenceDeclaration`

Add a frozen dataclass with responsibilities equivalent to:

```python
@dataclass(frozen=True)
class ExternalEvidenceDeclaration:
    name: str
    version: str
    dataset_hash: str
    source_snapshot_hash: str
    source_reference: str
    source_revision: Mapping[str, object]
    transform_identity: Mapping[str, object]
    record_namespace: str
    partition_assignment_hash: str
    evidence_origin: str = "external_observational"
    claim_scope: str = "external_observational_predictive_only"
```

All fields are canonical and content-hashed.

`source_snapshot_hash` is a SHA-256 identity for the frozen source snapshot used to create the repository dataset payload. P3 does not prescribe how that snapshot is stored; it may be a public release, repository artifact, or controlled external snapshot. The declaration binds the snapshot identity, not storage technology.

`source_revision` records immutable source-version metadata such as publication/release revision, export version, or equivalent stable identity.

`transform_identity` binds the declared transformation from source records into the existing observation schema. It is canonical metadata and must be sufficiently specific to detect transform-version drift. P3 does not introduce a generic ETL engine.

`record_namespace` names the source record-id namespace so that record identities cannot silently change meaning across dataset versions.

`partition_assignment_hash` is computed from the complete canonical mapping of source observation identities to `TRAIN`, `SELECTION_VALIDATION`, and `FINAL_TEST`. It supplements, rather than replaces, the existing dataset hash and partition hashes.

### 5.2 Positive origin requirement

The external path requires an affirmative external-observational origin declaration. Absence of a synthetic marker is not sufficient.

`ExternalEvidenceDeclaration.from_dataset(...)` must require the dataset's canonical `source` / `provenance` to contain a positive external-observational marker defined by this P3 contract. A conforming external dataset uses canonical metadata equivalent to:

```text
source.kind = "external_observational"
provenance.external_observational = true
```

The exact persisted values above are part of V1 and are not inferred from filenames.

### 5.3 Known synthetic rejection

The external constructor fails closed if the dataset carries known non-empirical provenance including any of:

```text
source.kind = "synthetic_fixture"
provenance.synthetic_non_empirical = true
provenance.empirical_human_data = false
```

This explicitly rejects the existing prison protocol-integration fixture and P2 synthetic-identification fixture from the P3 scientific path.

### 5.4 What the declaration does not prove

The declaration makes source lineage and the asserted evidence origin reproducible. It does not independently prove that:

- a source publisher is truthful;
- a sample is representative;
- measurement is unbiased;
- collection occurred before a specified date;
- the chosen variables are scientifically sufficient;
- the data reflect a particular latent cognitive mechanism.

These limitations are preserved in the final report claim scope.

## 6. External validation preregistration

`narrative_dynamics.external_validation` adds one top-level P3 preregistration that binds the external evidence declaration to the existing Brier and Log evaluation protocols.

### 6.1 Score roles

Use a closed enum:

```text
BRIER
LOG
```

V1 requires exactly one categorical Brier sibling protocol and exactly one categorical Log sibling protocol.

### 6.2 `ExternalValidationPreregistration`

Add a frozen dataclass equivalent in responsibility to:

```python
@dataclass(frozen=True)
class ExternalValidationPreregistration:
    name: str
    version: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    strata: tuple[ExternalStratum, ...]
    separation_rule: PairwiseSeparationRule
    constraint_plans: tuple[ExternalConstraintPlan, ...]
    method_validation_hashes: tuple[str, ...]
    claim_scope: str = "external_observational_predictive_only"
```

Its constructor receives the actual `ExternalEvidenceDeclaration` and the two actual `PreregisteredEvaluationProtocol` objects, validates every sibling invariant, and stores their hashes.

### 6.3 Brier / Log sibling invariants

Before any final-test release can be accepted, the Brier and Log protocols must match exactly on all non-loss scientific inputs:

- dataset hash;
- train partition hash;
- selection-validation partition hash;
- final-test partition hash;
- target-spec hash;
- final-target hash;
- target-construction manifest hash;
- metric extractor identity;
- simulation seed plan;
- baseline model name;
- complete frozen candidate set;
- each candidate model identity;
- each candidate selected parameter tuple;
- each candidate selection-manifest lineage;
- adequacy thresholds unless a threshold is inherently loss-specific and is represented explicitly inside the P3 score role;
- protocol version.

Permitted differences are limited to:

- protocol name suffix needed to distinguish score role;
- categorical loss identity;
- the protocol's derived precommitment/content hash.

One sibling must use the repository's canonical categorical Brier loss identity and the other the canonical categorical Log loss identity. Supplying two Brier protocols, two Log protocols, or an unknown loss family fails closed.

### 6.4 No post-final sibling construction

The P3 aggregate preregistration hash binds both sibling protocol hashes before either final test is allowed to execute. It is invalid to run Brier final evaluation and construct the Log sibling afterward.

## 7. External release and witness gating

P3 reuses the existing release machinery rather than creating a new witness format.

### 7.1 Two existing `ProtocolRelease` values

Create one existing `ProtocolRelease` for the Brier `PreregisteredEvaluationProtocol` and one for the Log protocol.

Each release's canonical `source_revision` mapping must include at least:

```text
repository_revision
external_evidence_declaration_hash
external_validation_preregistration_hash
score_role
```

The first three identities must match across the sibling releases. `score_role` is `brier` or `log` as appropriate.

Because `ProtocolRelease.content_hash` includes `source_revision`, existing `WitnessReceipt` values indirectly witness the complete P3 sibling preregistration and external evidence declaration without modifying the release schema.

### 7.2 Both siblings verified before final execution

The P3 orchestration preflights **both** `VerifiedProtocolRelease` values before calling `compare_released_models()` for either score.

Required checks include:

- Brier release matches Brier protocol;
- Log release matches Log protocol;
- each verified release matches its release/protocol identity;
- both releases bind the same external evidence hash;
- both releases bind the same P3 preregistration hash;
- both releases bind the expected score role;
- release dataset/candidate/target identities remain exact;
- all sibling invariants remain exact.

If any preflight check fails, neither final comparison executes.

This prevents final-result peeking from becoming an implicit second-stage preregistration mechanism.

## 8. External training and model freezing

P3 does not change the train/selection/final semantics already encoded in the observation protocol.

### 8.1 Training

Parameter fitting uses only the `TRAIN` partition and existing finite-grid/calibration infrastructure.

### 8.2 Selection-validation

Model or parameter selection uses only `SELECTION_VALIDATION` after train fitting. Existing `SelectionValidationReport` lineage remains authoritative for `FrozenModelSpec`.

### 8.3 Final test

The `FINAL_TEST` partition is not used to choose model families, parameters, thresholds, score siblings, strata, separation thresholds, or external parameter-constraint rules.

Final evaluation begins only after both score siblings have verified releases.

## 9. Predictive adequacy findings

P3 wraps existing final score results in external claim-safe findings.

### 9.1 `ExternalPredictiveAdequacyFinding`

For one model and one score role, retain at least:

- model name and frozen model-spec hash;
- score role and loss identity;
- global mean final loss;
- global worst-case final loss;
- frozen threshold identity;
- per-score adequacy result;
- child released-comparison manifest hash.

Per-score status is:

```text
PREDICTIVE_ADEQUACY_MET
PREDICTIVE_ADEQUACY_NOT_MET
```

using the existing frozen adequacy thresholds.

### 9.2 Aggregate model adequacy

A model is `PREDICTIVE_ADEQUACY_MET` at the P3 aggregate level only when both the Brier and Log sibling evaluations meet their frozen adequacy requirements.

If either sibling fails, aggregate adequacy is `PREDICTIVE_ADEQUACY_NOT_MET` while preserving both child results.

This conservative rule prevents cherry-picking the proper score that favors a preferred conclusion.

## 10. Pairwise predictive separation

P3 predictive separation is a deterministic preregistered discrimination rule, not a statistical significance test.

### 10.1 `PairwiseSeparationRule`

Freeze strictly positive finite thresholds:

```text
min_mean_loss_delta_brier
min_mean_loss_delta_log
```

V1 fixes `require_direction_agreement = true`.

### 10.2 Separation semantics

For a canonical model pair `(A, B)`:

1. compute the signed mean-loss difference under Brier;
2. compute the signed mean-loss difference under Log;
3. require both scores to prefer the same model;
4. require the absolute Brier difference to meet the Brier threshold;
5. require the absolute Log difference to meet the Log threshold.

Only then may the finding be:

```text
PREDICTIVELY_SEPARATED_UNDER_PROTOCOL
```

Otherwise it is:

```text
NOT_PREDICTIVELY_SEPARATED_UNDER_PROTOCOL
```

The report retains both raw score differences and the preferred model when separation succeeds.

This status means only that the frozen dual-score rule discriminates the observable predictions. It does not mean the preferred model's latent mechanism is true.

## 11. Preregistered strata

External validation reports both global and preregistered stratum results.

### 11.1 `ExternalStratum`

A frozen stratum contains:

```text
name
final_case_names
```

Strata refer to final target-case names, not runtime predicates.

### 11.2 V1 stratum invariants

- stratum names are unique and non-empty;
- each stratum is non-empty;
- every final case appears in exactly one stratum;
- no final case may be omitted;
- no final case may appear in multiple strata;
- stratum definitions are frozen in `ExternalValidationPreregistration` before final execution.

Global scores are always reported independently of strata.

For each score role and model, the report retains per-stratum mean and worst loss. Pairwise score deltas are also reported per stratum, but P3 V1 does not create a separate stratum-specific winner status unless the preregistered global separation rule is explicitly evaluated on that stratum and stored as a separate finding.

## 12. External parameter constraints

Parameter constraints are supported as an optional, separately declared P3 analysis. They never consume final-test data and never emit identification statuses.

### 12.1 `ExternalConstraintPlan`

A frozen plan binds:

- plan name;
- frozen model family/source identity;
- finite candidate parameter grid;
- target parameter coordinates;
- selection-validation case identities;
- Brier acceptance-loss delta;
- Log acceptance-loss delta;
- simulation seeds;
- metric identity;
- both score identities.

The plan is part of the P3 preregistration and cannot be added after final evaluation.

### 12.2 Compatible-set rule

For each score sibling on the declared selection-validation cases:

```text
candidate_loss <= best_loss + preregistered_delta
```

A parameter tuple is externally compatible only if it is retained under **both** the Brier and Log selection rules.

The resulting compatible set is explicit and canonical.

### 12.3 Constraint semantics

For one declared parameter coordinate:

- if every retained tuple has the same coordinate value, status is `CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`;
- if retained tuples contain multiple coordinate values, status is `NOT_CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`.

There is no generator-truth requirement because external observational data do not expose latent ground truth.

An empty compatible set is an execution/protocol failure, not `NOT_CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`.

For a multi-coordinate plan, aggregate status is `CONSTRAINED_UNDER_EXTERNAL_PROTOCOL` only when every declared coordinate is constrained. The complete compatible set remains visible in the finding.

### 12.4 Isolation from P2 identification

P3 implements constraint interpretation directly in external-validation types. It must not call `diagnose_identifiability()` or construct `ParameterIdentificationFinding`.

This intentionally duplicates only the tiny interpretation rule needed to preserve a different scientific meaning.

## 13. P2 synthetic method evidence

P2 synthetic identification answers whether the machinery can recover truth, expose non-recovery, certify observational equivalence, and discriminate controlled information interventions when generator truth and synthetic construction are known.

P3 may bind one or more P2 method-evidence hashes as lineage:

```text
method_validation_hashes
```

These hashes are opaque evidence identities in P3. `external_validation.py` does not import P2 identification statuses or translate their findings.

The dependency is therefore:

```text
P2 synthetic method evidence hash
            |
            v
P3 external-validation preregistration/report lineage
            |
            X
   no status/claim inheritance
```

A P2 `IDENTIFIED_UNDER_PROTOCOL` result never changes an external constraint into empirical identification.

## 14. `ExternalValidationReport`

The final aggregate report is a frozen, attested artifact containing at least:

- external validation preregistration hash;
- external evidence declaration hash;
- fixed claim scope;
- Brier protocol/release/verification identities;
- Log protocol/release/verification identities;
- both released model-comparison child manifest hashes;
- global predictive adequacy findings;
- global pairwise predictive-separation findings;
- preregistered stratum scores;
- zero or more external parameter-constraint findings;
- method-validation hashes;
- complete canonical parent lineage;
- `ExperimentManifest(stage=EXTERNAL_VALIDATION, ...)`.

The report exposes no `identification_status` field.

The report must be compatible with existing aggregate attestation and must pass:

```text
attest_report(report).require_integrity()
```

without introducing a parallel artifact system.

## 15. Manifest stage

Add exactly one new aggregate experiment stage:

```text
EXTERNAL_VALIDATION = "external_validation"
```

Existing target-construction, training, selection, final-test, model-comparison, and released-model-comparison stages remain authoritative for child evidence.

No additional manifest-schema version is required.

## 16. Canonical identity rules

All new public declaration, preregistration, rule, finding, and report dataclasses are frozen.

Canonical ordering rules:

- strata sort by stratum name;
- case names inside strata sort lexically;
- constraint plans sort by plan name;
- parameter names sort lexically;
- parameter tuples sort by parameter name and then tuple value;
- model pairs are stored in lexical model-name order;
- score roles have fixed Brier-then-Log order;
- evidence and method hashes are validated SHA-256 identities and stored deterministically;
- mappings are recursively frozen through existing canonical helpers or equivalent trusted helpers.

No identity may depend on object address, set iteration, dictionary insertion order, filesystem order, or runtime timestamps.

## 17. Error boundaries

Use one P3 root error:

```python
class ExternalValidationError(ValueError):
    pass
```

Subclasses may separate evidence, protocol, sibling-score, constraint, claim, and report failures. Existing `ProtocolReleaseVerificationError` continues to propagate for existing release-verification failures.

P3 fails closed on at least:

### 17.1 Evidence failures

- missing positive external-observational origin marker;
- known synthetic/non-empirical provenance;
- invalid source snapshot hash;
- empty source reference or record namespace;
- non-canonical source revision or transform identity;
- dataset hash mismatch;
- partition-assignment hash drift;
- source snapshot / transform / record namespace drift after declaration.

### 17.2 Protocol sibling failures

- Brier/Log dataset mismatch;
- partition mismatch;
- target-spec/target-manifest mismatch;
- metric mismatch;
- seed mismatch;
- baseline mismatch;
- candidate-set mismatch;
- frozen parameter mismatch;
- selection lineage mismatch;
- threshold drift outside explicitly represented loss-specific fields;
- duplicate score family;
- unknown score family;
- P3 preregistration created from already divergent siblings.

### 17.3 Release failures

- missing Brier or Log release;
- missing Brier or Log verified release;
- release bound to wrong P3 preregistration;
- release bound to wrong external evidence declaration;
- wrong score role;
- sibling release repository revision mismatch where the P3 release contract requires the same revision;
- any existing release/protocol/dataset/candidate mismatch.

### 17.4 Stratum failures

- unknown final case;
- missing final case;
- duplicate final case across strata;
- empty stratum;
- duplicate stratum name.

### 17.5 Constraint failures

- empty candidate grid;
- changed candidate schema across score siblings;
- undeclared parameter coordinate;
- final-test case used in a constraint plan;
- Brier/Log selection case drift;
- empty compatible set;
- score identity drift.

### 17.6 Claim-boundary failures

- external finding constructed with a P2 identification status;
- aggregate claim scope changed from `external_observational_predictive_only`;
- canonical payload containing a successful empirical/structural/cognitive identification status;
- external report exposing a synthetic-identification result as its own empirical finding.

### 17.7 Report failures

- missing child released-comparison lineage;
- forged release/verification hash;
- missing external evidence hash;
- missing P3 preregistration hash;
- finding values inconsistent with child scores;
- payload tampering detected by attestation.

## 18. Production file boundaries

The intended production surface is deliberately small.

### 18.1 `narrative_dynamics/observations/external.py`

Owns:

- `ExternalEvidenceDeclaration`;
- positive external-origin validation;
- known synthetic-origin rejection;
- partition-assignment identity;
- external evidence errors.

It does not run models or score predictions.

### 18.2 `narrative_dynamics/external_validation.py`

Owns:

- score-role enum;
- P3 claim/status enums;
- Brier/Log sibling validation;
- `ExternalValidationPreregistration`;
- stratum declarations;
- pairwise separation rule/findings;
- optional external parameter-constraint plans/findings;
- dual-release preflight;
- orchestration over existing `compare_released_models()`;
- aggregate `ExternalValidationReport` construction;
- P3-specific typed errors.

It does not implement cognitive dynamics, target normalization, calibration algorithms, release signatures, or general model comparison.

### 18.3 `narrative_dynamics/contracts.py`

Add only:

```text
ExperimentStage.EXTERNAL_VALIDATION
```

### 18.4 Package exports

Expose only the intended public P3 types and entry points. No existing P1/P2 public API is renamed or reinterpreted.

## 19. External data source policy for V1

This design freezes the **validation machinery and scientific contract**, not a particular scientific dataset.

No existing repository synthetic fixture may be relabeled as empirical evidence to satisfy P3.

A real scientific P3 run becomes externally meaningful only when a concrete external dataset release is supplied with:

- positive external-observational provenance;
- a frozen source snapshot hash;
- record namespace and transform identity;
- train/selection/final assignments;
- a P3 external evidence declaration;
- witnessed Brier and Log sibling protocol releases.

Selecting a particular external source changes the empirical research question and therefore belongs in a separate data-release/study decision, not hidden inside this architecture increment.

Unit tests may construct minimal in-memory external declarations to exercise contracts. Such test data are not themselves a scientific empirical result and must never be published by P3 as one.

## 20. Testing strategy

Implementation follows strict RED -> GREEN -> atomic-commit discipline after this written design and a separate implementation plan are approved.

The first REDs should protect scientific boundaries before happy-path orchestration.

### 20.1 Claim-firewall RED

Require that:

- P3 exposes no `IdentificationStatus` result;
- a singleton external compatible set produces `CONSTRAINED_UNDER_EXTERNAL_PROTOCOL` rather than identification;
- forbidden empirical/cognitive-identification status strings cannot enter a canonical P3 report;
- claim scope is immutable.

### 20.2 Evidence-origin RED

Require that:

- the existing `prison_initial_choice_v1.json` synthetic fixture is rejected by the P3 external path;
- the P2 narrative-identification fixture is rejected;
- absence of a synthetic marker is insufficient without positive external provenance;
- source snapshot, transform, namespace, dataset, and partition drift are rejected.

### 20.3 Sibling-protocol RED

Require exact Brier/Log sibling equality on every non-loss identity and explicit rejection of target, seed, model, selected-parameter, threshold, metric, or selection-lineage drift.

### 20.4 Dual-release gate RED

Require that no final model execution occurs unless both sibling releases and both verified releases pass complete preflight.

Use execution-counting fakes/spies to prove fail-before-execute behavior.

### 20.5 Adequacy/separation RED

Require:

- both proper scores are preserved;
- aggregate adequacy requires both siblings;
- pairwise separation requires same-direction preference and both preregistered deltas;
- a score disagreement yields `NOT_PREDICTIVELY_SEPARATED_UNDER_PROTOCOL`;
- no separation result is described as statistical significance or mechanism truth.

### 20.6 Stratum RED

Require complete exactly-once final-case coverage and deterministic global/stratum score aggregation.

### 20.7 External-constraint RED

Require:

- selection-only candidate evaluation;
- intersection of Brier and Log compatible sets;
- constrained and non-constrained cases;
- empty compatible set as typed failure;
- no generator-truth or P2-identification dependency.

### 20.8 Aggregate lineage / attestation RED

Require both released comparison parents, both verification identities, P3 preregistration, external evidence declaration, method-evidence hashes when present, deterministic payload identity, and tamper rejection.

### 20.9 Regression gate

Existing Lean conformance, full Lean build, narrative theorem/story tests, and the complete Python suite must remain GREEN.

P3 V1 is Python research-protocol work and does not require Docker acceptance.

## 21. Non-goals

P3 V1 does not:

- infer causal effects from observational association;
- implement instrumental variables, regression discontinuity, propensity weighting, or natural-experiment estimators;
- claim population representativeness;
- claim measurement validity;
- add event-level likelihoods or a raw-data database;
- add missing-data imputation;
- add bootstrap confidence intervals or asymptotic significance tests;
- add hierarchical Bayesian inference, MCMC, variational inference, or continuous optimization;
- alter Reactive, Intentional, or Planning/POMDP semantics;
- add new latent cognitive parameters;
- modify P2 `IdentificationStatus`;
- modify stochastic world/observation semantics;
- modify the Lean kernel;
- change GenericNarrative IR/domain identity;
- use final-test data for model/parameter selection;
- treat predictive superiority as proof that real agents use the winning model's latent mechanism.

## 22. Definition of done

P3 Empirical / External Validation V1 is architecturally complete when the implementation can, under one frozen external-observational protocol:

1. bind an existing `ObservationDataset` to an explicit external source snapshot and transform lineage;
2. reject known synthetic/non-empirical datasets and missing positive external provenance;
3. freeze exactly one Brier and one Log sibling `PreregisteredEvaluationProtocol` with all non-loss identities equal;
4. bind both siblings and the external evidence declaration into one P3 preregistration;
5. require witnessed, verified releases for both siblings before either final test executes;
6. evaluate the same frozen model candidates on the same final targets and seeds under both proper scores;
7. report global and preregistered stratum predictive adequacy and pairwise separation;
8. optionally report finite external parameter constraints from selection-validation data only;
9. preserve optional P2 method-evidence hashes without inheriting P2 identification statuses;
10. produce an attested `ExternalValidationReport` whose canonical claim scope is `external_observational_predictive_only`;
11. make attempts to upgrade predictive fit into empirical latent-cognition identification fail at the typed/report boundary;
12. keep all existing P0-P2 behavior and tests unchanged.

A successful implementation establishes a reproducible external **predictive-validation** pipeline. It does not establish empirical identification of latent cognition.

## 23. Implementation gate

This document is the written architecture spec for issue #38.

No production implementation begins from this design branch until the user reviews and approves this written spec. After approval, the next artifact is a separate RED -> GREEN implementation plan; only after that plan is approved does implementation start.
