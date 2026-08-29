# Empirical Revision 3 — Internal Locked Final Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the abandoned Revision 2 OSF-witness gate with an explicitly internal immutable Git-repository lock while preserving the completed real TRAIN/SELECTION freeze and every predictive scientific protocol choice.

**Architecture:** Add one study-specific internal-lock module that creates content-hashed lock bundles, R3 releases, Git-commit proofs, internal `WitnessReceipt` values, and a verifier compatible with the existing generic `verify_protocol_release()` / `preflight_external_releases()` chain. Do not modify the existing locked FINAL implementation. After exact-head GREEN, use metadata-only Actions workflows to construct real R3 release identities, commit the lock payload on a separate lock branch, verify both releases, and stop for explicit manual FINAL authorization.

**Tech Stack:** Python stdlib (`dataclasses`, `re`, `collections.abc`), existing `ProtocolRelease`, `WitnessReceipt`, `VerifiedProtocolRelease`, `verify_protocol_release`, `preflight_external_releases`, existing Feher/Hare Study V1 types, Git/GitHub commit identity, GitHub Actions, `unittest`, Lean/lake regression gates.

**Spec:** `docs/superpowers/specs/2026-08-30-empirical-r3-internal-locked-final-design.md`

## Global Constraints

- Revision 3 is **not externally preregistered** and must never be described as externally or independently witnessed.
- Do not rerun real TRAIN or SELECTION.
- Reconstruct exactly the R2 frozen candidates from run `33254738541`; candidate hashes must match before R3 metadata is accepted.
- Preserve source manifest `sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d`.
- Preserve source snapshot `sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186`.
- Preserve transform `sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541`.
- Preserve participant assignment `sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf`.
- Preserve dataset `sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781`.
- Preserve FINAL target `sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2`.
- Preserve TRAIN seeds `(101, 102)`, SELECTION seeds `(201, 202)`, FINAL seeds `(301, 302)`.
- Preserve Brier-only TRAIN/SELECTION, sibling Brier + Log FINAL, probability floor `1e-12`, Planning baseline, task strata, adequacy thresholds, separation deltas, `constraint_plans=()`, and `external_observational_predictive_only` claim scope.
- Do not modify `narrative_dynamics/studies/two_stage_locked_final.py` unless a RED test proves existing generic verified-release compatibility is insufficient. The expected implementation needs no change there.
- Never execute real FINAL in CI or in any implementation task.
- Existing R2 OSF bundle `sha256:cca12ced5bc1ba1643980a3c55cbc8bba7a5c22cedd7bf474f00c40f726aa520` is historical audit evidence only and cannot authorize R3 FINAL.
- After internal-lock preflight GREEN, stop for a fresh explicit manual FINAL authorization.

---

### Task 1: Commit the complete test-only internal-lock RED

**Files:**
- Create: `tests/test_two_stage_internal_lock.py`

**Interfaces:**
- Consumes: existing `ProtocolRelease`, `WitnessReceipt`, `verify_protocol_release`, `preflight_external_releases`, `FrozenModelSpec`, `OSFRegistrationProof`, `OSFRegistrationVerifier`, and synthetic external-validation fixtures.
- Produces: a complete RED contract for the production interfaces implemented in Task 2.

- [ ] **Step 1: Create guarded imports for the new production API.**

The test imports must expect exactly:

```python
from narrative_dynamics.studies.two_stage_internal_lock import (
    INTERNAL_LOCK_AUTHORITY,
    INTERNAL_LOCK_GOVERNANCE_MODE,
    INTERNAL_LOCK_PROVIDER,
    InternalRepositoryLockProof,
    InternalRepositoryLockVerifier,
    TwoStageInternalLockBundle,
    create_internal_locked_protocol_release,
    create_internal_repository_lock_receipt,
)
```

Use the same readable missing-import RED pattern as `tests/test_two_stage_osf_witness.py`.

- [ ] **Step 2: Add a synthetic helper that builds sibling protocols/releases and one internal-lock bundle.**

Use `tests.external_validation_fixtures.sibling_protocols()`, `build_external_declaration()`, and `build_preregistration()`; do not use real human data.

Construct R3 releases through the wished-for helper:

```python
brier_release = create_internal_locked_protocol_release(
    name="external-validation:brier:r3-release",
    version="3",
    protocol=brier_protocol,
    repository_revision="r3-scientific-revision",
    evidence_hash=evidence.content_hash,
    preregistration_hash=preregistration.content_hash,
    score_role="brier",
    train_selection_freeze_digest=stable_content_hash({"freeze": "r2"}),
)
```

Repeat for Log.

- [ ] **Step 3: Add lock-bundle identity tests.**

Lock these behaviors:

```text
test_internal_lock_bundle_is_content_hashed_and_immutable
test_internal_lock_bundle_requires_internal_governance_and_no_external_registration
test_internal_lock_bundle_requires_two_distinct_release_hashes
test_internal_lock_bundle_requires_final_model_execution_false
```

The synthetic bundle must bind source/snapshot/transform/split/dataset/final-target hashes, frozen candidate hashes, protocol/preregistration/release hashes, repository revision, claim scope, and a non-empty scientific contract.

- [ ] **Step 4: Add R3 release metadata tests.**

Assert both R3 releases contain the generic preflight keys plus:

```python
{
    "governance_mode": "internal_locked_final",
    "external_registration": False,
    "train_selection_freeze_digest": freeze_digest,
}
```

Assert Brier and Log releases are distinct and changing the R3 repository revision changes both release hashes.

- [ ] **Step 5: Add proof and receipt verification tests.**

Use a synthetic 40-hex lock commit and exact timestamp. Lock:

```text
test_internal_lock_proof_binds_repository_commit_payload_revision_and_releases
test_two_internal_receipts_verify_two_sibling_releases
test_receipt_for_other_release_is_rejected
test_wrong_lock_commit_reference_or_time_is_rejected
test_wrong_lock_payload_or_scientific_revision_is_rejected
```

Create receipts only through `create_internal_repository_lock_receipt()`.

- [ ] **Step 6: Add cross-governance rejection tests.**

Assert:

```text
test_osf_receipt_does_not_verify_as_internal_lock
test_internal_receipt_does_not_verify_with_osf_verifier
```

Use the existing `OSFRegistrationVerifier`; do not modify OSF code.

- [ ] **Step 7: Add generic dual-preflight compatibility test.**

Verify both internal receipts with generic `verify_protocol_release()`, then call:

```python
preflight = preflight_external_releases(
    preregistration=preregistration,
    evidence=evidence,
    brier_protocol=brier_protocol,
    brier_release=brier_release,
    brier_verified=brier_verified,
    log_protocol=log_protocol,
    log_release=log_release,
    log_verified=log_verified,
)
```

Assert stable `preflight.content_hash` and that synthetic model call counters remain zero. The test must not invoke any FINAL runner.

- [ ] **Step 8: Add exact R2 candidate reconstruction test.**

Create the three existing Feher/Hare model sources and rebuild `FrozenModelSpec.freeze(...)` from the frozen R2 parameters and SELECTION manifest hashes. Assert exact hashes:

```text
reactive    sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456
intentional sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839
planning    sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c
```

This test must never call TRAIN/SELECTION APIs.

- [ ] **Step 9: Run the focused RED and record the expected missing-module failure.**

```bash
python3 -m unittest tests.test_two_stage_internal_lock -v
```

Expected: RED only because `narrative_dynamics.studies.two_stage_internal_lock` does not exist.

- [ ] **Step 10: Commit the test-only RED atomically.**

```bash
git add tests/test_two_stage_internal_lock.py
git commit -m "test: require empirical r3 internal final lock"
```

---

### Task 2: Implement the internal-lock contracts

**Files:**
- Create: `narrative_dynamics/studies/two_stage_internal_lock.py`
- Modify: `narrative_dynamics/studies/__init__.py`
- Test: `tests/test_two_stage_internal_lock.py`

**Interfaces:**
- Consumes: `ProtocolRelease`, `WitnessReceipt`, `PreregisteredEvaluationProtocol`, `stable_content_hash`, `_freeze_mapping`, `EXTERNAL_CLAIM_SCOPE`.
- Produces:
  - `TwoStageInternalLockBundle`
  - `InternalRepositoryLockProof`
  - `InternalRepositoryLockVerifier`
  - `create_internal_locked_protocol_release(...) -> ProtocolRelease`
  - `create_internal_repository_lock_receipt(...) -> WitnessReceipt`

- [ ] **Step 1: Confirm Task 1 RED on the implementation branch.**

```bash
python3 -m unittest tests.test_two_stage_internal_lock -v
```

- [ ] **Step 2: Implement canonical validators and constants.**

Use:

```python
INTERNAL_LOCK_PROVIDER = "internal-repository-lock"
INTERNAL_LOCK_AUTHORITY = "qigao/lean"
INTERNAL_LOCK_GOVERNANCE_MODE = "internal_locked_final"
_INTERNAL_LOCK_BUNDLE_VERSION = "two-stage-internal-lock-v1"
_INTERNAL_LOCK_VERIFIER_VERSION = "two-stage-internal-lock-verifier-v1"
```

Validate sha256 hashes as `^sha256:[0-9a-f]{64}$`, Git commit SHA as `^[0-9a-f]{40}$`, non-empty trimmed text, non-empty unique hash tuples, and boolean `external_registration is False` / `final_model_execution is False` exactly.

- [ ] **Step 3: Implement `TwoStageInternalLockBundle`.**

Use this exact public field set:

```python
@dataclass(frozen=True)
class TwoStageInternalLockBundle:
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_protocol_hash: str
    log_protocol_hash: str
    evaluation_preregistration_hash: str
    brier_release_hash: str
    log_release_hash: str
    scientific_repository_revision: str
    train_selection_freeze_digest: str
    claim_scope: str
    scientific_contract: Mapping[str, object]
    governance_mode: str = INTERNAL_LOCK_GOVERNANCE_MODE
    external_registration: bool = False
    final_model_execution: bool = False
```

Its `identity_payload()` must include the fixed bundle version and every field; `content_hash` is `stable_content_hash(identity_payload())`.

Reject equal sibling protocol hashes or equal sibling release hashes.

- [ ] **Step 4: Implement R3 release creation.**

Use this exact signature:

```python
def create_internal_locked_protocol_release(
    *,
    name: str,
    version: str,
    protocol: PreregisteredEvaluationProtocol,
    repository_revision: str,
    evidence_hash: str,
    preregistration_hash: str,
    score_role: str,
    train_selection_freeze_digest: str,
) -> ProtocolRelease:
```

Require `score_role in {"brier", "log"}` and call existing `ProtocolRelease.create(...)` with:

```python
source_revision={
    "repository_revision": repository_revision,
    "external_evidence_declaration_hash": evidence_hash,
    "external_validation_preregistration_hash": preregistration_hash,
    "score_role": score_role,
    "governance_mode": INTERNAL_LOCK_GOVERNANCE_MODE,
    "external_registration": False,
    "train_selection_freeze_digest": train_selection_freeze_digest,
}
```

Do not change `ProtocolRelease` itself.

- [ ] **Step 5: Implement `InternalRepositoryLockProof`.**

Use:

```python
@dataclass(frozen=True)
class InternalRepositoryLockProof:
    repository: str
    lock_commit: str
    locked_at: str
    lock_payload_hash: str
    scientific_repository_revision: str
    release_hashes: tuple[str, ...]
```

Require `repository == "qigao/lean"`, exactly two distinct release hashes, canonical sorted release hashes, and expose:

```python
@property
def reference(self) -> str:
    return f"https://github.com/{self.repository}/commit/{self.lock_commit}"
```

`manifest_identity()` returns verifier-stable proof identity including `content_hash`.

- [ ] **Step 6: Implement the verifier and receipt helper.**

`InternalRepositoryLockVerifier.verify(release, receipt)` returns `False` unless all are true:

```python
receipt.provider == INTERNAL_LOCK_PROVIDER
receipt.authority == INTERNAL_LOCK_AUTHORITY
receipt.subject_hash == release.content_hash
receipt.subject_hash in proof.release_hashes
receipt.reference == proof.reference
receipt.claimed_at == proof.locked_at
set(receipt.proof) == {
    "lock_payload_hash",
    "scientific_repository_revision",
    "governance_mode",
    "external_registration",
}
receipt.proof["lock_payload_hash"] == proof.lock_payload_hash
receipt.proof["scientific_repository_revision"] == proof.scientific_repository_revision
receipt.proof["governance_mode"] == INTERNAL_LOCK_GOVERNANCE_MODE
receipt.proof["external_registration"] is False
release.source_revision["repository_revision"] == proof.scientific_repository_revision
release.source_revision["governance_mode"] == INTERNAL_LOCK_GOVERNANCE_MODE
release.source_revision["external_registration"] is False
```

`create_internal_repository_lock_receipt(release, proof)` creates the existing generic `WitnessReceipt` with those exact values.

- [ ] **Step 7: Export only the R3 internal-lock public API from `narrative_dynamics.studies`.**

Do not alter OSF exports or generic observations APIs.

- [ ] **Step 8: Run focused GREEN.**

```bash
python3 -m unittest tests.test_two_stage_internal_lock -v
```

Expected: all internal-lock tests PASS.

- [ ] **Step 9: Commit GREEN atomically.**

```bash
git add narrative_dynamics/studies/two_stage_internal_lock.py \
  narrative_dynamics/studies/__init__.py tests/test_two_stage_internal_lock.py
git commit -m "feat: add empirical r3 internal final lock"
```

---

### Task 3: Prove locked-FINAL compatibility and full regression GREEN

**Files:**
- Test: `tests/test_two_stage_internal_lock.py`
- Existing regression: `tests/test_external_validation_release_gate.py`
- Existing regression: `tests/test_two_stage_osf_witness.py`
- Existing regression: `tests/test_two_stage_final_attempts.py`
- Existing regression: `tests/test_feher_hare_two_stage_pipeline.py`

**Interfaces:**
- Consumes: Task 2 public API and unchanged `run_locked_feher_hare_final()` chain.
- Produces: evidence that internal verification feeds the same generic release/preflight boundary without weakening FINAL semantics.

- [ ] **Step 1: Run focused internal/external release gate tests together.**

```bash
python3 -m unittest \
  tests.test_two_stage_internal_lock \
  tests.test_two_stage_osf_witness \
  tests.test_external_validation_release_gate -v
```

Expected: GREEN; OSF and internal verifiers remain mutually non-substitutable.

- [ ] **Step 2: Run FINAL-attempt and full synthetic Feher/Hare pipeline regressions.**

```bash
python3 -m unittest \
  tests.test_two_stage_final_attempts \
  tests.test_feher_hare_two_stage_pipeline -v
```

Expected: GREEN with unchanged single-pass/append-only semantics.

- [ ] **Step 3: Run the complete Python suite.**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests GREEN. Record exact test count.

- [ ] **Step 4: Push exact implementation head and verify the `proof` workflow.**

Push branch `work/empirical-r3-internal-locked-final`. The authoritative `proof` run must bind exactly the pushed SHA and pass Python plus all Lean gates. Do not run Docker or any real empirical workflow.

- [ ] **Step 5: Freeze the exact Revision 3 scientific head only after exact-head proof GREEN.**

Record the implementation SHA as `R3_SCIENTIFIC_REVISION`. From this point until the internal lock is committed, do not change scientific code, source manifest, candidates, protocol semantics, or FINAL policy.

---

### Task 4: Build the real metadata-only R3 release and internal-lock payload

**Files:**
- Create on orchestration branch only: `.github/workflows/real-internal-lock-r3.yml`
- Artifact only: `artifacts/real-internal-lock-r3/internal_lock_payload.json`

**Interfaces:**
- Consumes: exact `R3_SCIENTIFIC_REVISION`, pinned Feher/Hare upstream, R2 TRAIN/SELECTION freeze identities, Task 2 production API.
- Produces: new R3 Brier/Log releases and one metadata-only `TwoStageInternalLockBundle`; no model execution.

- [ ] **Step 1: Create an isolated manual orchestration branch from `R3_SCIENTIFIC_REVISION`.**

Use branch `manual/empirical-r3-internal-final`. Workflow commits on this branch are orchestration-only and never become scientific repository identity.

- [ ] **Step 2: Write a workflow that explicitly checks out the exact scientific SHA and pinned upstream revision.**

The workflow must assert all retained source/transform/split/dataset/final-target hashes and reconstruct the three R2 candidates via `FrozenModelSpec.freeze(...)` before creating any R3 release.

- [ ] **Step 3: Rebuild Brier/Log protocols and scientific preregistration without model execution.**

Rebuild the same protocol semantics from the exact frozen candidates. Assert the scientific contract is unchanged from R2. Record whether protocol/prereg hashes remain identical; equality is expected and allowed.

- [ ] **Step 4: Create new R3 releases with Task 2 helper.**

Use `repository_revision=R3_SCIENTIFIC_REVISION`, version `3`, governance mode `internal_locked_final`, `external_registration=false`, and the R2 TRAIN/SELECTION payload digest.

Assert the new R3 release hashes are not equal to R2 releases:

```text
R2 Brier release sha256:6962db18f9e190e02dae844008a8add0de808a620cd1c6cb1030856be500fecb
R2 Log release   sha256:a019bdc7cd049401ca4075ad3e81077012839ff0e86ffdb982d4c49280b9f989
```

- [ ] **Step 5: Build `TwoStageInternalLockBundle`.**

Bind all required identities and explicitly serialize:

```json
{
  "governance_mode": "internal_locked_final",
  "external_registration": false,
  "final_model_execution": false,
  "raw_human_rows_in_artifact": false
}
```

Print only hashes/metadata, never raw human rows.

- [ ] **Step 6: Upload the metadata-only lock payload artifact.**

Use `actions/upload-artifact@v4`, retention 90 days, `if-no-files-found: error`. Record run ID, artifact ID, ZIP digest, payload hash, R3 release hashes, and exact scientific SHA.

---

### Task 5: Commit the immutable internal lock on a dedicated branch

**Files:**
- Create branch: `lock/feher-hare-r3-internal-final`
- Create: `research-locks/feher-hare-r3-internal-final.json`

**Interfaces:**
- Consumes: Task 4 exact metadata-only lock payload.
- Produces: immutable Git commit SHA and timestamp used by `InternalRepositoryLockProof`.

- [ ] **Step 1: Download Task 4 artifact and verify its recorded byte/content digest before commit.**

Reject any mismatch. Do not reformat or regenerate JSON after verification.

- [ ] **Step 2: Create `lock/feher-hare-r3-internal-final` from `R3_SCIENTIFIC_REVISION`.**

The branch starts at the exact scientific SHA but is never used as the FINAL execution checkout.

- [ ] **Step 3: Commit the exact verified JSON bytes once.**

Commit message:

```text
research: lock empirical revision 3 final protocol
```

No scientific code files may be changed in this commit.

- [ ] **Step 4: Record the lock commit SHA and commit timestamp.**

Construct `InternalRepositoryLockProof` from:

```python
repository="qigao/lean"
lock_commit=<actual lock commit SHA>
locked_at=<actual commit timestamp>
lock_payload_hash=<Task 4 bundle content hash>
scientific_repository_revision=R3_SCIENTIFIC_REVISION
release_hashes=(r3_brier_release_hash, r3_log_release_hash)
```

Do not create a fake/external registration reference.

---

### Task 6: Verify the real internal receipts and complete zero-FINAL preflight

**Files:**
- Create on orchestration branch only: `.github/workflows/real-internal-preflight-r3.yml`
- Artifact only: `artifacts/real-internal-preflight-r3/preflight.json`

**Interfaces:**
- Consumes: exact R3 scientific SHA, Task 4 release identities, Task 5 lock commit proof.
- Produces: two real internal `WitnessReceipt` values, two `VerifiedProtocolRelease` values, and `ExternalReleasePreflight`; no FINAL execution.

- [ ] **Step 1: Add a manual preflight workflow that checks out exact `R3_SCIENTIFIC_REVISION` and pinned upstream.**

Reconstruct source/dataset/candidates/protocols/releases/internal-lock bundle and reject any identity drift.

- [ ] **Step 2: Instantiate `InternalRepositoryLockProof` from the actual lock commit metadata.**

Verify the lock commit URL/reference and timestamp match GitHub commit metadata before creating receipts.

- [ ] **Step 3: Create one internal receipt per release.**

Use `create_internal_repository_lock_receipt()` for Brier and Log. Assert the receipts share the same lock reference/payload hash but have distinct subject hashes.

- [ ] **Step 4: Verify both releases with generic `verify_protocol_release()`.**

Use the same `InternalRepositoryLockVerifier(proof)` for both siblings. Assert both verification hashes are stable and distinct.

- [ ] **Step 5: Run `preflight_external_releases()` and prove zero FINAL model execution.**

Do not instantiate or call FINAL runtime models. The workflow must emit:

```json
{
  "governance_mode": "internal_locked_final",
  "external_registration": false,
  "brier_verified": true,
  "log_verified": true,
  "preflight_hash": "sha256:...",
  "final_model_execution": false
}
```

- [ ] **Step 6: Upload the metadata-only preflight artifact and update #40/#39.**

Mark R3 internal lock + dual preflight GREEN, explicitly state the study is not externally preregistered, preserve the R2 history, and keep FINAL `not authorized / not run`.

- [ ] **Step 7: STOP for explicit manual FINAL authorization.**

Do not create or trigger a FINAL workflow. The next action after Task 6 is a fresh human instruction explicitly authorizing the locked Revision 3 FINAL execution.
