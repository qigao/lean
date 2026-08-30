# Narrative Cross-Dataset Transfer V1 Governance Design

Date: 2026-08-30

Status: written for human review; not approved; no implementation or execution authorized

Tracker: #43 — Cross-dataset Transfer V1

Roadmap: #39 — Real external validation study and post-P3 research frontiers

Scientific base: `work/narrative-measurement-validity-v1@c979eafe650a506bf30f78ab5b35078742b54087`

Design branch: `work/narrative-cross-dataset-transfer-v1-governance-design`

Depends on:

- completed Feher/Hare Study V1 and immutable R3 FINAL evidence;
- Measurement Validity V1 protocol `sha256:c5c40911aed2a6e2af97c39f702f656330530290420f8316a3af80019d3eaad4`;
- Measurement Validity V1 GREEN report `sha256:15bbc3a1dae20fc34cdb76dcdd90563bc997d3a85baaf8ce35f0c15e6c8f2d28`;
- Measurement Validity V1 exact-head and real-audit evidence recorded append-only in #39.

This document replaces the unapproved provisional Cross-Dataset Transfer V1 design. The provisional implementation branch remains evidence only. Nothing in that branch is approved, frozen, or executable by virtue of this document.

## 1. Decision and purpose

Cross-Dataset Transfer V1 asks whether the unchanged Reactive, Intentional, and Planning two-stage model families retain predictive value on one independently collected public dataset from the same broad task family.

V1 distinguishes three claims:

1. **Code portability** — the same frozen implementations and canonical semantic interfaces can process a qualified second source without changing model-family meaning.
2. **Zero-shot parameter transfer** — the three Feher/Hare R3 candidate tuples predict new participants without parameter fitting or selection on the new source.
3. **Model-family portability after fresh refit** — the unchanged finite model families can freeze candidates using only new TRAIN and SELECTION_VALIDATION participants.

These claims remain predictive and observational. They do not establish a true latent cognitive mechanism, causal transport, population invariance, or universal parameter values.

## 2. Alternatives considered

### 2.1 Selected design: staged dual-candidate protocol

V1 compares a frozen zero-shot path and a fresh-refit path against the same sealed FINAL_TEST participants. Both candidate sets and both score releases are frozen before FINAL_TEST is opened. One sealed prediction artifact feeds both Brier and Log scoring.

This design separates parameter transfer from family portability while preserving a paired FINAL comparison.

### 2.2 Rejected: zero-shot-only V1

A zero-shot-only study has the cleanest parameter-transfer claim but cannot distinguish universal-parameter failure from model-family portability. It remains a valid future simplification if the dual-path protocol cannot satisfy the capability and governance gates below.

### 2.3 Rejected: sequential zero-shot then refit FINAL

Observing zero-shot FINAL before refit candidate freeze would leak held-out evidence into the secondary protocol. Separate or sequential FINAL subsets are therefore prohibited.

### 2.4 Rejected: pooled or trial-randomized study

Trial-level partitioning leaks participant history. Pooling away task, score, aggregation, or participant-influence strata would hide the exact Measurement Validity V1 uncertainties that motivate this transfer study.

## 3. Fixed claim boundary

The canonical scope is:

```text
external_observational_cross_dataset_predictive_transfer_only
```

Permitted conclusions are limited to typed statements that code portability, zero-shot transfer, or family portability was met, not met, mixed, or inconclusive under the frozen protocol.

Forbidden conclusions include:

- a family is the true human cognitive mechanism;
- a parameter tuple is universal or psychologically identified;
- one same-family source establishes cross-task, cross-domain, cultural, causal, hierarchical, participant-specific, or longitudinal generality;
- refit success erases a zero-shot failure;
- rank one without predictive adequacy is transfer evidence;
- an internal repository lock is external preregistration.

Every report serializes `external_registration=false` and fixed limitation identifiers rather than optional free text.

## 4. Frozen scientific anchor

The study base is exactly:

```text
c979eafe650a506bf30f78ab5b35078742b54087
```

The zero-shot candidates are reconstructed from the immutable R3 lock and must retain their source hashes:

| Family | Parameters | Source candidate hash |
| --- | --- | --- |
| Reactive | `beta=0.5` | `sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456` |
| Intentional | `beta=2.0`, `memory_decay=0.5` | `sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839` |
| Planning | `beta=4.0`, `memory_decay=0.75` | `sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c` |

The refit search spaces remain exactly the R3 finite grids:

```text
Reactive:
  beta in {0.5, 1.0, 2.0, 4.0}

Intentional and Planning:
  beta in {0.5, 1.0, 2.0, 4.0}
  memory_decay in {0.5, 0.75, 0.9, 1.0}
```

The seed roles remain:

```text
TRAIN:                (101, 102)
SELECTION_VALIDATION: (201, 202)
FINAL_TEST:           (301, 302)
```

The model implementations, builders, simulation semantics, proper-score implementations, and tie-breaking rules must retain exact callable identities from the approved base. V1 adds no family, parameter, optimizer, participant-specific parameter, or task-specific fitted parameter.

## 5. Reproducible dataset-search protocol

Dataset selection is a scientific operation even when it runs zero models. V1 therefore freezes the search universe and stopping rule before selecting a source.

### 5.1 Search protocol record

`DatasetSearchProtocol` binds:

- public-release cutoff `2026-08-30`;
- exact bibliographic query strings for two-stage sequential decision datasets;
- exact repository families searched, including DOI-linked archives and the originating paper's declared public repository;
- repository query date, API or user-interface version, pagination limit, and sort order;
- deduplication key `(study DOI, canonical repository locator, immutable release/revision)`;
- inclusion/exclusion fields required below;
- a rule that candidate-model code is unavailable to the search process.

The search-protocol schema and stopping rule are reviewed as part of the design freeze. The resulting append-only `DatasetCandidateCatalog` is a separate pre-model source gate: it must be independently reviewed and frozen before deterministic selection. A candidate discovered after catalog freeze belongs to a new revision.

### 5.2 Catalog stopping rule

The search terminates only when every frozen query page through the predeclared pagination limit has been recorded and every DOI-linked repository discovered from those records has been resolved. The catalog retains every candidate and a typed exclusion reason. The process may not stop after finding a favorable eligible source.

If the catalog cannot be reproduced from its frozen metadata and evidence receipts, V1 is `INFRASTRUCTURE_INCOMPLETE`; it does not select a source.

### 5.3 Eligibility

An eligible source must:

- be independently collected from Feher/Hare and publicly released by the cutoff;
- have explicit scientific-analysis rights or license evidence;
- expose immutable repository/release identity and a complete consumed-file inventory;
- freeze each consumed file by canonical locator, path, byte size, and SHA-256;
- contain a documented binary first-stage choice, chronological participant sequence, second-stage transition/state information, and reward feedback;
- expose source-declared practice, timeout, no-response, abort, and exclusion semantics when present;
- retain at least 150 eligible participants after fixed source-declared exclusions;
- contain no file derived from the Feher/Hare snapshot;
- require zero candidate-model executions during search and qualification.

Mappings that depend on model predictions, FINAL performance, published winner labels, or case-by-case outcome interpretation make a source ineligible.

### 5.4 Deterministic selection

Among eligible catalog entries select by:

1. greatest eligible participant count;
2. earliest public release date;
3. lexically smallest canonical source locator.

If no entry qualifies, the protocol ends `SCIENTIFIC_RED` with finding `NO_ELIGIBLE_DATASET`. Criteria cannot be relaxed inside V1.

No dataset is selected by this design document. Selection occurs only after the search protocol, catalog, evidence receipts, and zero-model-execution attestation are frozen and approved.

## 6. Canonical endpoint and semantic harmonization

The only confirmatory endpoint is the observed first-stage binary choice on each retained trial.

Source actions map to `action_0` and `action_1` using only the source data dictionary and pre-choice task semantics. Reward, transitions, prior retained history, task version, and prior outcomes may be model inputs where the existing family semantics permit them; they are never alternate targets.

The transform must:

- remove source-declared excluded trials before retained-history construction;
- reject included malformed rows rather than impute them;
- preserve chronological order within participant;
- use the previous retained trial, not the previous raw row;
- exclude same-trial post-choice information from model-visible state;
- freeze source, transform, endpoint, row, participant, and snapshot commitments;
- preserve every pre-choice source task/condition stratum;
- reject duplicate trials, ambiguous ordering, unknown labels, and post-split exclusion.

Semantic invariance is end-to-end, not an involution check. Synthetic relabel variants must run through source parsing, canonical transform, scenario construction, unchanged model execution, inverse probability mapping, scoring, and report assembly. Coherent action/state/history relabeling must preserve inverse-mapped losses, findings, and report identities.

Failure of any required mapping or invariance gate yields `SCIENTIFIC_RED` with `HARMONIZATION_INVALID` and zero FINAL_TEST opening.

## 7. Participant-disjoint roles

Partitioning occurs only after fixed exclusions and canonical participant construction.

Within every predeclared source stratum, participants are ordered by a stable commitment over the study namespace, source snapshot, stratum, and an ephemeral participant token. Assignment uses exact counts:

```text
TRAIN:                floor(0.60 * n)
SELECTION_VALIDATION: floor(0.20 * n)
FINAL_TEST:           all remaining participants
```

Every participant belongs to one role; every retained trial inherits that role. Empty roles, duplicate participants, cross-role overlap, missing strata, post-split exclusion, or outcome-dependent reassignment fail closed.

### 7.1 Participant privacy threat model

Direct participant identifiers and deterministic unsalted hashes are prohibited in durable evidence because public-source identifiers may be enumerable.

Role assignment uses `HMAC-SHA256` over the study namespace, source snapshot, stratum, and canonical participant identifier with a precommitted 256-bit study secret. The secret is generated before source parsing and retained only in an access-restricted governance vault through the independent audit. The durable public split manifest retains only:

- aggregate role/stratum counts;
- a Merkle root over private commitments;
- the HMAC algorithm identifier and a one-way key commitment;
- an independent transform attestation.

The HMAC key, direct participant identifiers, and participant-level commitments do not enter Git, logs, public artifacts, or canonical reports. Authorized independent audit occurs inside the restricted source environment. The secret is destroyed after terminal evidence is archived, which permanently closes participant-level replay for that revision.

## 8. Capability architecture and sealed FINAL vault

Role isolation is an object-capability and process boundary, not a naming convention.

### 8.1 Preparation outputs

Preparation returns only:

- verified source/snapshot/transform identities;
- aggregate role/stratum counts;
- TRAIN capability;
- SELECTION_VALIDATION capability;
- `FinalProjectionCommitment` containing hashes and counts only;
- a non-iterable `FinalVaultHandle` usable only by the locked FINAL worker.

The public prepared record must not contain a general dataset, FINAL scenarios, FINAL targets, a filesystem path to FINAL bytes, or a method that can enumerate FINAL cases.

### 8.2 Vault isolation

The transform worker writes role-segregated material to isolated stores. The refit worker receives no credentials, path, handle, or API for the FINAL vault. The orchestration process can inspect the commitment but cannot open the vault.

The vault opens only when supplied with:

- the exact current authoritative ledger head;
- a completed dual preflight;
- an immutable authorization receipt bound to that preflight;
- both approved score releases and internal lock;
- the one approved scientific revision.

Unlock returns a single-use projection directly to the locked FINAL worker. It never returns the general dataset or a reusable collection.

### 8.3 Zero-shot and refit inputs

Zero-shot construction receives only frozen R3 candidate records and lineage identities. It has no TRAIN, SELECTION_VALIDATION, or FINAL capability.

Refit construction receives only TRAIN and SELECTION_VALIDATION projections. It has no general dataset, FINAL commitment opener, FINAL cases, targets, predictions, scores, or prior zero-shot FINAL finding.

Reflection and hostile-capability tests must prove the absence of these fields and accesses.

## 9. Candidate freezes

`ZeroShotCandidateFreeze` binds the scientific base, immutable R3 lock, exact source candidate hashes, new-source lineage, execution aliases, and explicit false assertions for new-data training and selection.

`RefitCandidateFreeze` binds:

- the exact ordered finite grids;
- TRAIN and SELECTION_VALIDATION split and target manifest receipts;
- TRAIN/SELECTION seeds;
- Brier loss callable identity;
- model-builder and simulation callable identities;
- lexical tie-break identity;
- proof that every grid point was evaluated;
- one selected candidate per family and its source report hash;
- `final_projection_opened=false` and `final_model_execution=false`.

Selected values merely belonging to a permitted set is insufficient. The freeze must prove the complete search protocol was executed unchanged.

## 10. Baseline, scores, aggregation, and uncertainty

### 10.1 TRAIN-only baseline

The primary comparator is a no-history population base-rate predictor estimated from TRAIN only. For each frozen source stratum:

```text
p(action_1) = (n1 + 1) / (n0 + n1 + 2)
p(action_0) = 1 - p(action_1)
```

Unknown or missing FINAL strata fail closed; there is no global or uniform fallback. Uniform `(0.5, 0.5)` remains a sanity diagnostic only.

### 10.2 Co-primary scores

Brier and clipped Log use the existing implementations and consume the same sealed prediction identity. A confirmatory claim requires both scores to support the same direction. Disagreement yields `MIXED_OR_INCONCLUSIVE_EVIDENCE`.

### 10.3 Primary aggregation

Participant-equal loss is primary: average retained-trial loss within each FINAL participant, then average participant means equally. Trial-equal pooling is diagnostic and cannot override a terminal.

### 10.4 Paired participant-block bootstrap

Use exactly 10,000 paired participant resamples with seed `43001`, 2.5th/97.5th percentile endpoints using frozen linear interpolation, and complete participant blocks.

For lower-is-better loss:

```text
relative_improvement = (loss_comparator - loss_candidate) / loss_comparator
```

A candidate beats a comparator on one score only when point relative improvement is at least `0.01` and the lower 95% endpoint is greater than `0.0`.

Predictive adequacy requires beating the TRAIN-only base rate on both scores. Material refit gain requires refit to beat the same family's zero-shot candidate on both scores.

### 10.5 Ephemeral and durable evidence

Participant-level blocks, commitments, counts, and losses are ephemeral scoring state. They are not serializable in durable score or report payloads.

For every confirmatory comparison, durable `TransferInferenceEvidence` contains only:

- candidate and comparator identities;
- score and aggregation identifiers;
- aggregate candidate/comparator losses;
- point relative improvement;
- lower and upper 95% endpoints;
- bootstrap seed, replicate count, percentile rule, and participant count;
- threshold and pass status;
- source-stratum identity when applicable;
- canonical content hash.

This aggregate record is sufficient to audit every terminal without disclosing participant-level contributions.

## 11. Frozen carry-forward sensitivity requirements

Measurement Validity V1's inconclusive cells are mandatory lineage, not optional commentary. The transfer protocol contains these exact requirement IDs:

1. `MV1_RI_SELECTION_SPACESHIP_TASK_LOG_BOTH_AGGREGATIONS` — Reactive vs Intentional, SELECTION_VALIDATION Spaceship, Log, participant-equal and trial-equal.
2. `MV1_RI_SELECTION_SPACESHIP_SCORE_BOTH_AGGREGATIONS` — Reactive vs Intentional, SELECTION_VALIDATION Spaceship, Brier-near-tie versus Log sign reversal, both aggregations.
3. `MV1_RI_SELECTION_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS` — Reactive vs Intentional leave-one-participant-out sensitivity.
4. `MV1_IP_TRAIN_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS` — Intentional vs Planning leave-one-participant-out sensitivity.

The new source need not use the label `Spaceship`. Each requirement preserves its original Measurement Validity lineage and requires the corresponding model-pair, score, aggregation, task/condition-stratum, and participant-influence diagnostics on every semantically comparable new-source stratum. A non-comparable stratum is reported `NOT_ESTABLISHED`; it is not silently dropped or averaged away.

Aggregation stability, 5/5 semantic invariance, and stay/switch completeness remain positive controls. Cross-family pairwise losses, task strata, trial-equal rankings, stay/switch cells, and deterministic leave-one-participant-out ranges remain diagnostic and cannot select candidates or alter family terminals.

## 12. Family findings and study-level terminals

### 12.1 Family transfer findings

Each family receives exactly one confirmatory finding:

| Zero-shot adequacy | Refit adequacy | Finding |
| --- | --- | --- |
| met | met | `PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE` |
| not met | met | `FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED` |
| met | not met | `ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED` |
| not met | not met | `NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL` |
| either path has score disagreement | any | `MIXED_OR_INCONCLUSIVE_EVIDENCE` |

Refit material gain is reported separately. Rank order alone is never adequacy.

### 12.2 Study-level terminal

Every attempt ends in exactly one top-level terminal. Canonical enum identifiers use underscores; the corresponding tracker labels are `GREEN`, `SCIENTIFIC RED`, and `INFRASTRUCTURE INCOMPLETE`:

- `GREEN` — the frozen protocol completed, the authoritative ledger and artifact are complete, all identities and privacy gates passed, and every family received a typed finding. `GREEN` may contain scientifically negative or inconclusive family findings; it means the protocol completed validly.
- `SCIENTIFIC_RED` — a reproducible scientific hard gate such as no eligible dataset, invalid harmonization, invalid split, or unfrozen candidate protocol prevented valid FINAL inference, with complete aggregate/hash-only evidence. It is not infrastructure failure and does not authorize criterion relaxation.
- `INFRASTRUCTURE_INCOMPLETE` — required identity, persistence, execution, or artifact evidence is missing or unverifiable before scientific classification. It is not a scientific result.

These top-level terminals are distinct from attempt events such as `FINAL_STARTED`, `FINAL_COMPLETED`, `REVISION_REQUIRED`, and `EXACT_REPLAY_ALLOWED`.

## 13. Immutable identity and release chain

The study freezes canonical records for search protocol/catalog, source manifest, snapshot, transform, split, candidate freezes, TRAIN baseline, preregistration, Brier/Log sibling releases, dual preflight, authorization receipt, attempt ledger, sealed prediction, aggregate score evidence, family findings, and final report.

Unknown fields, duplicate identities, role drift, callable drift, grid drift, baseline drift, score-sibling drift, threshold drift, or lock drift fail closed.

The Brier and Log sibling releases bind the same source, split, six candidates, baseline, FINAL commitment, seeds, aggregation, bootstrap, thresholds, family vocabulary, study terminal vocabulary, limitations, and carry-forward sensitivity requirements. Only score-specific identity may differ.

An internal lock is committed to `lock/narrative-cross-dataset-transfer-v1` after exact-head Phase A proof and before FINAL authorization. It records `external_registration=false`, `final_projection_opened=false`, `final_model_execution=false`, and the exact scientific revision.

## 14. Authoritative attempt ledger and authorization

### 14.1 External attempt store

Exactly-once governance cannot depend on a caller-supplied in-memory ledger. `TransferAttemptStore` is external to the scientific process and supports atomic:

```text
compare_and_append(expected_head_hash, event) -> new_head_hash
```

The manual workflow implementation uses a dedicated append-only Git branch plus non-force fast-forward updates under a non-cancelling concurrency group. Every event is a canonical signed/attested record whose parent is the previously verified ledger head. A stale or empty replacement ledger is rejected.

The current remote ledger head, complete history, and zero prior FINAL openings/executions are verified during preflight. The preflight binds that exact head hash.

### 14.2 Authorization receipt

Design approval, source approval, TRAIN/SELECTION approval, lock creation, and preflight are not FINAL authorization.

After completed dual preflight, the authorized repository owner must add a new #43 comment containing exactly:

```text
I authorize Cross-dataset Transfer V1 locked FINAL_TEST execution.
scientific_sha=<exact scientific SHA>
lock_commit=<exact lock commit>
preflight_hash=<exact dual-preflight hash>
brier_release_hash=<exact Brier release hash>
log_release_hash=<exact Log release hash>
```

`TransferAuthorizationReceipt` binds the issue/comment ID and URL, author login, immutable creation timestamp, exact text, all five identities, and the pre-authorization ledger head. A comment predating preflight, edited after capture, authored by another account, or mismatching any identity is invalid.

Authorization is consumed by the atomic `FINAL_STARTED` append and cannot authorize another attempt or revision.

## 15. FINAL state machine and replay rule

The locked FINAL workflow executes in this order:

1. verify scientific, source, transform, split, candidate, release, lock, ledger, and authorization identities;
2. atomically append and persist `FINAL_STARTED`, consuming the authorization receipt;
3. open the single-use FINAL vault projection;
4. execute the six candidates × two seeds matrix exactly once per FINAL case;
5. seal one prediction artifact;
6. derive both score siblings and diagnostics from that artifact;
7. atomically append `FINAL_COMPLETED` or a typed failure event;
8. persist aggregate/hash-only evidence under `if: always()`.

The STARTED event must be durably visible before the vault can open. Process termination after STARTED cannot erase it.

### 15.1 Failure and replay

- Any schema, scientific, prediction, scoring, privacy, or report failure after STARTED is `REVISION_REQUIRED`.
- Any failure after `final_projection_opened=true` is `REVISION_REQUIRED`, even if no model run completed, because FINAL values may have been exposed.
- Any failure after one execution manifest or prediction exists is `REVISION_REQUIRED`.
- `EXACT_REPLAY_ALLOWED` is possible only when independent evidence proves `final_projection_opened=false`, `completed_model_runs=0`, no prediction or score artifact exists, and the failure was infrastructure-only before vault unlock.
- Exact replay requires a new post-preflight authorization receipt and a new atomic ledger event. There is no automatic retry.

Infrastructure exception class alone never proves replay safety. Progress, vault-open state, and available hashes must be persisted in the authoritative attempt record.

## 16. Artifact and privacy policy

Raw source rows, direct or reversibly hashed participant identifiers, participant commitments, source paths tied to individuals, per-case predictions, and FINAL targets never enter Git, logs, release payloads, durable artifacts, or canonical reports.

The sealed prediction artifact exists only in the isolated FINAL/scoring environment. It contains opaque ephemeral case tokens, target coordinates, candidate probabilities, and execution identities needed for single-pass scoring. It is destroyed after independent score/report attestation.

Durable evidence contains only:

- immutable identity hashes and receipts;
- aggregate role/stratum participant and trial counts;
- aggregate inference evidence from section 10.5;
- aggregate diagnostics and carry-forward sensitivity statuses;
- ledger events and typed findings;
- explicit privacy and no-training/no-FINAL-boundary assertions.

An independent forbidden-key/value scan and archive inventory check are acceptance gates. Any participant-level or per-case leakage rejects the artifact.

## 17. Manual execution gates

Ordinary CI uses only committed compact synthetic fixtures. It must not fetch a public dataset, run real TRAIN/SELECTION, open FINAL, or contain a manual FINAL entry point.

Real work is separated into explicit gates:

1. freeze and approve search protocol and candidate catalog with zero model executions;
2. qualify and freeze one source/snapshot;
3. run real TRAIN/SELECTION once and freeze zero-shot/refit candidates;
4. freeze baseline, preregistration, score siblings, and internal lock;
5. complete dual preflight with authoritative zero-execution ledger head;
6. obtain the exact post-preflight authorization receipt;
7. run one sealed FINAL pass;
8. independently audit and durably archive aggregate/hash-only evidence.

Each gate records exact branch, commit, run, job, artifact, and content identities. Earlier approval never authorizes a later gate.

## 18. Test strategy

Implementation must follow test-only RED evidence before each minimal GREEN change.

### 18.1 Source and harmonization

Tests cover search catalog completeness/stopping, typed exclusions, deterministic selection, exact source inventory, local-only byte verification, malformed rows, retained-history semantics, duplicate/order rejection, and end-to-end inverse relabeling invariance.

### 18.2 Split, privacy, and capabilities

Tests cover participant-disjoint roles, HMAC key-secrecy and commitment privacy, non-enumerable durable manifests, hostile FINAL properties, absence of general dataset/FINAL fields, separate process/store credentials, forged handle rejection, and preflight-bound single-use vault unlock.

### 18.3 Candidates and inference

Tests cover exact R3 candidates, complete refit-grid evidence, callable/tie-break identities, TRAIN-only baseline, participant-equal versus trial-equal references, deterministic 10,000-block bootstrap, threshold boundaries, durable interval schema, and every family finding.

### 18.4 Governance and failure injection

Tests cover stale/fresh-empty ledger attacks, atomic STARTED persistence, authorization mismatch/edit/reuse, concurrent attempts, crash after STARTED, crash before/after vault unlock, partial execution timeout, prediction/score failure, forbidden replay, new authorization for safe replay, and always-run evidence staging.

### 18.5 Carry-forward sensitivities and terminals

Tests require all four exact Measurement Validity lineage IDs, new-source comparable/not-established mappings, both aggregations/scores where frozen, all family findings, and top-level `GREEN`, `SCIENTIFIC_RED`, and `INFRASTRUCTURE_INCOMPLETE` classification.

### 18.6 Repository gates

The exact Phase A head must pass full Python discovery, Lean conformance/build/theorem/Story/Testimony, `git diff --check`, public API review, frozen-model byte-identity checks, offline CI inspection, and a scope diff proving no real data or workflow was added.

## 19. #43 design-gate coverage

| #43 gate | Frozen design section |
| --- | --- |
| independent public dataset eligibility/provenance | §5 search protocol, catalog, eligibility, and deterministic selection |
| harmonized endpoint and semantic coordinate/task invariance | §6 end-to-end transform and inverse-relabel proof |
| separate zero-shot and fresh-refit protocols/claims | §§1–2, §8.3, §9, and §12.1 |
| participant-level TRAIN/SELECTION_VALIDATION/sealed FINAL assignment | §7 role split and §8 vault isolation |
| Brier/Log, both aggregations, task strata, and participant influence | §10 and §11 |
| all Measurement Validity sensitivity strata carried forward | §11's four exact lineage identifiers |
| candidates, seeds, thresholds, and immutable identities | §4, §9, §10, and §13 |
| aggregate/hash-only artifacts and privacy firewall | §7.1, §10.5, and §16 |
| GREEN / SCIENTIFIC RED / INFRASTRUCTURE INCOMPLETE | §12.2 |
| representativeness, external-validity, predictive-only, non-causal limits | §3 and §21 |
| explicit design and plan approval before execution | §20 acceptance gates and §22 approval boundary |

## 20. Acceptance gates

Phase A may be frozen only when:

1. this design is explicitly approved in #43;
2. a separate implementation plan is written and approved;
3. every implementation task has a reproducible test-only RED commit and minimal GREEN commit;
4. the FINAL vault, authoritative ledger, authorization receipt, replay rule, and durable aggregate evidence pass hostile tests;
5. every #43 design gate and carry-forward sensitivity requirement is represented explicitly;
6. full exact-head Python and Lean proof is GREEN;
7. ordinary CI proves zero public-data retrieval and zero real execution.

Phase B cannot begin until Phase A is frozen. FINAL readiness additionally requires an approved source/catalog, frozen candidates/releases/lock, zero-execution dual preflight, and a fresh exact authorization receipt.

The study is complete only when one top-level terminal is independently established, aggregate/hash-only evidence is durably archived, #43 records the exact result without expanding claims, and #39 records the new empirical baseline without rewriting Study V1.

## 21. Limitations and non-goals

The design deliberately does not establish:

- cross-task, cross-domain, cultural, causal, mechanistic, hierarchical, participant-specific, longitudinal, or population-weighted generality;
- universal parameters or a true family;
- repeated-split uncertainty or simultaneous population-wide intervals;
- external preregistration;
- new families, larger grids, continuous optimization, or automatic replay;
- any change to the completed R3 FINAL or Measurement Validity V1 result.

One same-family dataset and one participant split have limited external-validity reach. Public availability and deterministic eligibility may favor larger, better-documented cohorts. The 150-participant floor is not a power guarantee. These limitations are frozen report identifiers.

## 22. Approval boundary

Approval of this design authorizes only creation of a separate implementation plan. It does not authorize implementation, source retrieval, real TRAIN/SELECTION, lock creation, preflight, FINAL opening, or any GitHub push.

The unapproved provisional branch `work/narrative-cross-dataset-transfer-v1-design@b32e9c9ac5e4a9f7771125bc25ff8d7770f3c9a3` must remain untouched until an approved implementation plan specifies whether any logic may be rederived or cherry-picked. Its history cannot be cited as RED/GREEN evidence for the approved revision.
