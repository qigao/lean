# Empirical Revision 3 — Internal Locked Final Design

## Status

Approved direction: replace Empirical Revision 2's external OSF witness requirement with an explicitly **internal immutable repository lock**, while preserving the already-frozen real TRAIN/SELECTION outputs and all predictive scientific semantics.

This is a new empirical governance revision. It is **not externally preregistered** and must never be described as such.

## Motivation

Empirical Revision 2 completed real Brier TRAIN + SELECTION_VALIDATION and froze three model candidates, but its remaining FINAL gate required a public OSF Registration. The study will instead continue under a new internal-lock governance model so the FINAL protocol remains precommitted and auditable without claiming an external immutable witness.

The change is governance-only. It does not permit model, parameter, split, seed, threshold, metric, loss, stratum, target, or claim-scope changes.

## Frozen scientific inputs retained from Revision 2

The following identities are carried forward without recomputation or tuning:

- source repository: `carolfs/muddled_models`
- source revision: `4567763780a2c596fd6510af720ec468a8214a8f`
- source manifest: `sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d`
- source snapshot: `sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186`
- transform: `sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541`
- participant assignment: `sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf`
- dataset: `sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781`
- FINAL target: `sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2`

Frozen candidates from authoritative real TRAIN/SELECTION run `33254738541`:

- Reactive: `beta=0.5`
  - TRAIN `sha256:5b607a4bc0a8082809c2446a8248fb8659b0d9ba178687ddad96dc74e3f19622`
  - SELECTION `sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e`
  - candidate `sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456`
- Intentional: `beta=2.0`, `memory_decay=0.5`
  - TRAIN `sha256:47057311fe7450469c1710be49ba0e3b42aa7416fa2129f40761dbf8d237d4fc`
  - SELECTION `sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d`
  - candidate `sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839`
- Planning: `beta=4.0`, `memory_decay=0.75`
  - TRAIN `sha256:ef0f3caf4f2a02b2d719bc302d7409fc8c2d937eb13517d906d83b38c29b76ac`
  - SELECTION `sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74`
  - candidate `sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c`

The TRAIN/SELECTION artifact identity remains part of R3 provenance:

- run `33254738541`
- artifact ID `9717258303`
- payload digest `sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5`
- artifact ZIP digest `sha256:f4eb40a12a8b0ae62ec5c9e4baacf744a9311da491b60ac488ec24956a660616`

No real TRAIN or SELECTION is rerun for Revision 3.

## Scientific protocol continuity

Revision 3 keeps the exact Study V1 predictive protocol semantics:

- TRAIN seeds `(101, 102)`
- SELECTION seeds `(201, 202)`
- FINAL seeds `(301, 302)`
- Brier-only TRAIN/SELECTION
- sibling Brier + Log FINAL scores
- probability floor `1e-12`
- Planning bookkeeping baseline
- task strata `magic_carpet` and `spaceship`
- Brier separation delta `0.005`
- Log separation delta `0.006931471805599453`
- `constraint_plans=()`
- claim scope `external_observational_predictive_only`
- one sealed FINAL prediction artifact shared by Brier, Log, and the secondary diagnostic
- append-only FINAL attempt semantics

The Brier/Log protocol and preregistration objects are rebuilt and content-hash checked on the exact Revision 3 scientific head. Because their scientific content is intentionally unchanged, their hashes may remain equal to Revision 2. That equality is evidence of protocol continuity, not reuse of the Revision 2 OSF witness.

## Governance change

### Revision 2

`frozen scientific revision -> ProtocolRelease -> OSF Registration -> two WitnessReceipt values -> VerifiedProtocolRelease -> dual preflight -> manual FINAL authorization`

### Revision 3

`frozen scientific revision -> new R3 ProtocolRelease values -> internal lock payload -> immutable Git commit on a dedicated lock branch -> two internal WitnessReceipt values -> InternalRepositoryLockVerifier -> VerifiedProtocolRelease -> dual preflight -> manual FINAL authorization`

The Git commit is an **internal immutable audit lock**, not an independent external witness.

## New internal-lock boundary

Add `narrative_dynamics/studies/two_stage_internal_lock.py` containing study-neutral immutable contracts for this study family:

### `TwoStageInternalLockBundle`

Content-hashed metadata that binds:

- governance mode `internal_locked_final`
- external registration status `false`
- exact Revision 3 scientific repository revision
- R2 TRAIN/SELECTION artifact/run identities
- source manifest/snapshot/transform/split/dataset/final-target hashes
- all three frozen candidate hashes
- Brier/Log protocol hashes
- external-validation preregistration hash
- new R3 Brier/Log release hashes
- thresholds, floor, strata, separation rule, FINAL seeds, claim scope
- `final_model_execution=false`

No raw human rows are permitted.

### `InternalRepositoryLockProof`

Derived after the lock payload is committed on a dedicated lock branch. It binds:

- immutable Git commit SHA containing the exact lock payload
- lock commit timestamp
- repository identifier `qigao/lean`
- lock payload content hash
- scientific repository revision
- both release hashes

The lock payload itself does not contain its own commit SHA, avoiding circular identity construction.

### `InternalRepositoryLockVerifier`

Verifies existing `WitnessReceipt` values without changing the generic receipt schema. A valid receipt must use:

- provider: `internal-repository-lock`
- authority: `qigao/lean`
- subject hash: exactly one frozen R3 release hash
- reference: canonical GitHub commit URL for the lock commit
- claimed time: exact lock commit timestamp
- proof: exact internal-lock payload hash and scientific revision

The verifier must reject:

- another repository
- another lock commit/reference/time
- a different lock payload hash
- a different scientific revision
- a release not listed in the proof
- OSF/provider substitution
- duplicate or cross-role receipt substitution

## R3 releases

Create new Brier and Log `ProtocolRelease` values after the Revision 3 scientific implementation head is exact-head GREEN.

Their required generic bindings remain:

- `repository_revision`
- `external_evidence_declaration_hash`
- `external_validation_preregistration_hash`
- `score_role`

Revision 3 additionally records:

- `governance_mode = internal_locked_final`
- `external_registration = false`
- R2 TRAIN/SELECTION freeze artifact digest

The new exact Revision 3 repository revision and governance metadata ensure release hashes differ from Revision 2 even if scientific protocol hashes remain unchanged.

## Internal lock commit

After the new Revision 3 releases and internal-lock bundle are generated:

1. create a dedicated branch `lock/feher-hare-r3-internal-final` from the exact Revision 3 scientific head;
2. add only a metadata JSON lock payload under `research-locks/`;
3. commit once with no scientific code changes;
4. treat that commit SHA and timestamp as the immutable internal lock reference;
5. do not merge the lock commit into the scientific execution branch.

FINAL execution checks out the exact Revision 3 scientific revision, not the lock branch.

Any scientific code/data/protocol change after this lock invalidates the R3 lock and requires a new empirical revision. A metadata-only re-serialization mismatch requires a new lock commit before FINAL but does not permit scientific retuning.

## Locked FINAL integration

Do **not** create a second FINAL execution implementation.

The existing `run_locked_feher_hare_final()` remains authoritative. Revision 3 supplies standard `VerifiedProtocolRelease` objects produced by `verify_protocol_release()` with `InternalRepositoryLockVerifier`. Therefore all existing safety properties remain inherited:

- dual release preflight before any model execution
- repository/release/protocol/final-target identity checks
- exact retry after infrastructure-only failure
- STARTED attempt immediately before first FINAL model execution
- one sealed prediction pass
- Brier + Log scoring from the same prediction artifact
- secondary diagnostic without runner calls
- attested predictive-only report
- COMPLETED / infrastructure_failed / revision_required append-only statuses

No FINAL model execution is part of Revision 3 implementation CI.

## Manual authorization gate

A successful internal-lock preflight is necessary but not sufficient to run FINAL.

After both R3 releases verify and `preflight_external_releases()` succeeds with zero model executions, the study must stop and request explicit manual authorization for FINAL. No workflow commit, issue comment, or previous authorization implicitly authorizes FINAL.

## Claim language

Revision 3 may say:

- internally locked before FINAL
- protocol and candidates frozen before FINAL
- exact Git-repository lock committed before FINAL
- external observational predictive validation

Revision 3 must not say:

- externally preregistered
- OSF preregistered
- independently witnessed
- external immutable witness
- latent cognition identified or confirmed

Predictive adequacy/separation remains the strongest permitted scientific conclusion.

## TDD / verification gates

Implementation must be RED -> GREEN and include at minimum:

1. test-only RED for internal lock bundle/proof/verifier and R3 release metadata;
2. focused GREEN for the new module;
3. synthetic verification that two internal receipts verify two sibling releases and `preflight_external_releases()` succeeds;
4. assert zero `SimulationRunner` calls during internal lock verification/preflight;
5. assert OSF receipts do not satisfy the internal verifier and internal receipts do not satisfy `OSFRegistrationVerifier`;
6. assert R2 candidate hashes reconstruct exactly without TRAIN/SELECTION execution;
7. full Python regression suite;
8. Lean/Python conformance + full Lean build/theorem gates via exact-head `proof` CI;
9. only after exact-head GREEN, generate new real metadata-only R3 releases/internal-lock bundle;
10. commit the lock payload on the dedicated lock branch and verify both real receipts/preflight;
11. stop for explicit manual FINAL authorization.

## Non-goals

Revision 3 does not:

- rerun real TRAIN/SELECTION
- alter any model family
- add participant/task-specific parameters
- change source transform or split
- change FINAL target data
- change seeds, metrics, losses, thresholds, strata, or separation rules
- execute FINAL during CI
- preserve an external-preregistration claim
- weaken append-only FINAL attempt semantics

## Migration record

Revision 2 remains a valid historical record of completed TRAIN/SELECTION and the abandoned OSF-witness path. Its OSF bundle hash `sha256:cca12ced5bc1ba1643980a3c55cbc8bba7a5c22cedd7bf474f00c40f726aa520` is retained for audit only and is not used to authorize Revision 3 FINAL.
