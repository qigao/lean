# Narrative Cross-Dataset Transfer V1 Governance Phase A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the source-neutral, privacy-preserving, exactly-once Cross-Dataset Transfer V1 scientific and governance machinery using only synthetic fixtures, without selecting or retrieving a real dataset.

**Architecture:** Split the implementation into focused search, source, privacy, capability, candidate, inference, ledger, release, authorization, prediction, reporting, and locked-FINAL modules. Keep participant-level and FINAL material inside private capabilities; make durable evidence aggregate/hash-only; require an external append-only attempt store plus a post-preflight authorization receipt before a one-shot FINAL vault can open.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `hashlib`, `hmac`, `json`, `pathlib`, `random`, `secrets`, `subprocess`, `unittest`), existing `narrative_dynamics.contracts.stable_content_hash`, existing two-stage adapters/losses, Git for local append-only store tests, GitHub Actions `proof` workflow for later exact-head verification, and Lean/Lake for repository-wide gates.

**Spec:** `docs/superpowers/specs/2026-08-30-narrative-cross-dataset-transfer-v1-governance-design.md`

## Global Constraints

- Start from the human-approved commit containing this plan on `work/narrative-cross-dataset-transfer-v1-governance-design`, in an isolated worktree. Its ancestry must contain design commit `0378a40e934b3b241d6883df709aa056caec5f69`, whose parent is `c979eafe650a506bf30f78ab5b35078742b54087`.
- Keep the claim scope exactly `external_observational_cross_dataset_predictive_transfer_only`.
- Do not select, name, retrieve, parse, or commit any real candidate dataset in Phase A. All URLs use `https://example.invalid/`; all rows are compact synthetic fixtures.
- Do not modify `narrative_dynamics/adapters/narrative_two_stage.py`, existing model builders, simulation semantics, Brier/Log implementations, R3 evidence, or Measurement Validity V1 evidence.
- Do not cherry-pick any commit from `work/narrative-cross-dataset-transfer-v1-design@b32e9c9ac5e4a9f7771125bc25ff8d7770f3c9a3`. That branch may be read only as threat-model evidence; every approved RED and GREEN commit must be freshly produced from this plan.
- Keep the zero-shot candidate hashes, refit grids, seed roles, participant split proportions, bootstrap seed `43001`, bootstrap replicate count `10000`, minimum relative improvement `0.01`, and confidence interval rules exactly as frozen in the spec.
- Durable payloads must never contain direct participant identifiers, HMAC outputs, participant commitments, per-case predictions, FINAL targets, participant losses, source-local paths, or reusable FINAL access material.
- Ordinary CI must remain offline and synthetic. Do not add a real-data workflow, a manual FINAL workflow, dataset credentials, or a public-data download path during Phase A.
- Keep `external_registration=false`. An internal lock is not created by this plan; it belongs to a later gate after Phase A exact-head proof.
- Every implementation task has two commits: a test-only RED commit whose targeted command fails for the named missing behavior, followed by a minimal GREEN commit whose targeted and regression commands pass.
- After each RED or GREEN commit, record `git rev-parse HEAD`, `git show --stat --oneline HEAD`, and the exact command output in the task log. A GREEN commit may not rewrite or squash its RED parent.

## Scope Decomposition

This plan implements only Phase A: source-neutral contracts and synthetic proof. It deliberately ends before public catalog construction, deterministic real-source selection, source-specific parsing, real TRAIN/SELECTION, lock creation, preflight, authorization, or FINAL.

After Phase A exact-head GREEN, the next approved unit is a separate **Catalog and Source Adapter Plan**. That plan begins from the frozen search protocol and an independently reviewed `DatasetCandidateCatalog`; it names the deterministically selected dataset and contains the source-specific parser, inventory, harmonization proof, and manual TRAIN/SELECTION gates. No source-specific filename or dataset identity is guessed in this plan.

## File Responsibility Map

| File | Responsibility |
| --- | --- |
| `narrative_dynamics/cross_dataset_search.py` | Frozen search protocol, append-only catalog schema, eligibility audit, deterministic selection |
| `narrative_dynamics/cross_dataset_source.py` | Source manifests, local byte verification, canonical two-stage endpoint, end-to-end semantic invariance contract |
| `narrative_dynamics/cross_dataset_privacy.py` | Restricted HMAC secret, deterministic participant roles, public aggregate/Merkle split manifest |
| `narrative_dynamics/cross_dataset_capabilities.py` | TRAIN/SELECTION capabilities, hash-only FINAL commitment, non-iterable FINAL vault handle and unlock grant |
| `narrative_dynamics/cross_dataset_candidates.py` | Exact R3 zero-shot freeze and complete finite-grid refit freeze |
| `narrative_dynamics/cross_dataset_inference.py` | TRAIN-only baseline, ephemeral participant blocks, paired bootstrap, aggregate inference evidence |
| `narrative_dynamics/cross_dataset_ledger.py` | Canonical attempt events and external Git-backed compare-and-append store |
| `narrative_dynamics/cross_dataset_release.py` | Brier/Log sibling protocol releases and authoritative-ledger-bound dual preflight |
| `narrative_dynamics/cross_dataset_authorization.py` | Exact #43 authorization parsing, immutable receipt, identity and timing checks |
| `narrative_dynamics/cross_dataset_prediction.py` | Locked six-candidate × two-seed worker and ephemeral sealed prediction artifact |
| `narrative_dynamics/cross_dataset_reporting.py` | Aggregate scoring, four carry-forward sensitivity statuses, family findings, top-level terminal report |
| `narrative_dynamics/studies/cross_dataset_transfer_locked_final.py` | Ordered FINAL state machine, failure persistence, replay classification |
| `narrative_dynamics/__init__.py` | Deliberately small approved public surface; no FINAL opener or participant-level type |
| `tests/cross_dataset_transfer_fixtures.py` | Synthetic identities, rows, candidates, releases, fake vault, and local Git store builders |

## Spec Coverage Map

| Governance design section | Phase A implementation coverage |
| --- | --- |
| §1 decision and purpose | Global claim boundary; Tasks 5, 10, and 11 separate zero-shot, refit, and family findings |
| §2 staged dual-candidate protocol | Tasks 5, 8, 10, and 11; sequential or separate FINAL paths are rejected by tests |
| §3 fixed claim boundary | Task 1 constant, Task 8 protocol, Task 11 report limitations |
| §4 frozen scientific anchor | Task 5 exact hashes, grids, seeds, and callable identities |
| §5 reproducible dataset search | Task 1 protocol, catalog, stopping, eligibility, deterministic selection |
| §6 endpoint and harmonization | Task 2 canonical endpoint, information firewall, six-stage relabel invariance |
| §7 participant-disjoint roles | Task 3 exact stratified counts, HMAC secret, Merkle/count-only public manifest |
| §8 capability architecture | Task 4 role projections, non-iterable handle, service-owned single-use vault |
| §9 candidate freezes | Task 5 zero-shot assertions and complete 36-point refit evidence |
| §10 baseline, scores, uncertainty | Task 6 baseline/aggregations/bootstrap; Task 11 both-score terminal use |
| §11 carry-forward sensitivities | Task 8 freezes four IDs; Task 11 requires comparable or `NOT_ESTABLISHED` evidence |
| §12 findings and terminals | Task 11 exact family matrix and top-level vocabulary |
| §13 immutable release chain | Task 8 sibling releases and strict identity decoding |
| §14 ledger and authorization | Task 7 external CAS ledger; Tasks 8–9 preflight and exact receipt |
| §15 FINAL and replay | Task 12 ordered state machine, durable failures, replay predicate |
| §16 artifact/privacy policy | Tasks 3, 4, 6, 10, 11, and 13 forbidden-surface scans |
| §17 manual execution gates | Scope boundary and final gate stop before public/manual operations |
| §18 test strategy | Every Task 1–13 has a test-only RED commit and a minimal GREEN commit |
| §19 #43 coverage | This table plus exact targeted tests in Tasks 1–13 |
| §20 acceptance gates | Phase A Final Verification Gate |
| §21 limitations | Global constraints and Task 11 frozen report limitations |
| §22 approval boundary | Scope decomposition and mandatory stops before push or source work |

---

## Task 1: Freeze the Search Protocol and Candidate Catalog Contracts

**Files:**

- Create: `narrative_dynamics/cross_dataset_search.py`
- Create: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_search.py`

**Interfaces:**

- Produces: `DatasetSearchProtocol`, `DatasetCatalogEntry`, `DatasetCandidateCatalog`, `DatasetSelectionDecision`, `DatasetSelectionStatus`, `select_dataset_candidate()`.
- Consumes: `stable_content_hash()` only; it must not import any model, adapter, prediction, loss, or scoring module.

- [ ] **Step 1: Write the test-only RED contract**

```python
from narrative_dynamics.cross_dataset_search import (
    DatasetCandidateCatalog,
    DatasetCatalogEntry,
    DatasetSearchProtocol,
    DatasetSelectionStatus,
    select_dataset_candidate,
)

def test_catalog_selection_is_complete_zero_model_and_deterministic():
    protocol = search_protocol()
    catalog = DatasetCandidateCatalog.create(
        protocol=protocol,
        pages_recorded=(("doi_archive", 1), ("paper_repository", 1)),
        resolved_repository_receipts=(digest("archive"), digest("paper")),
        entries=(eligible_entry("b", 180, "2020-01-02"),
                 eligible_entry("a", 180, "2020-01-02")),
        candidate_model_executions=0,
    )
    decision = select_dataset_candidate(catalog)
    assert decision.status is DatasetSelectionStatus.SELECTED
    assert decision.selected_locator == "https://example.invalid/a"
    assert decision.catalog_hash == catalog.content_hash

def test_incomplete_catalog_is_infrastructure_incomplete():
    catalog = incomplete_catalog(missing_page=("doi_archive", 2))
    decision = select_dataset_candidate(catalog)
    assert decision.status is DatasetSelectionStatus.INFRASTRUCTURE_INCOMPLETE
    assert decision.reason_codes == ("CATALOG_INCOMPLETE",)

def test_no_eligible_entry_is_scientific_red_without_relaxation():
    catalog = complete_catalog(entries=(ineligible_entry("too_small"),))
    decision = select_dataset_candidate(catalog)
    assert decision.status is DatasetSelectionStatus.SCIENTIFIC_RED
    assert decision.reason_codes == ("NO_ELIGIBLE_DATASET",)
```

Add explicit tests named `test_search_protocol_rejects_empty_duplicate_or_unordered_queries`, `test_catalog_rejects_unresolved_doi_repository`, `test_catalog_rejects_model_execution`, `test_entry_requires_complete_inventory_license_and_same_family_endpoint`, `test_selection_prefers_count_then_release_then_locator`, and `test_all_payload_decoders_reject_unknown_or_missing_fields`.

- [ ] **Step 2: Run the RED test and commit it alone**

Run: `python3 -B -m unittest tests.test_cross_dataset_search -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'narrative_dynamics.cross_dataset_search'`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_search.py
git commit -m "test: require governed cross-dataset search"
```

- [ ] **Step 3: Implement the minimal strict contracts**

```python
CROSS_DATASET_TRANSFER_CLAIM_SCOPE = (
    "external_observational_cross_dataset_predictive_transfer_only"
)

class DatasetSelectionStatus(str, Enum):
    SELECTED = "SELECTED"
    SCIENTIFIC_RED = "SCIENTIFIC_RED"
    INFRASTRUCTURE_INCOMPLETE = "INFRASTRUCTURE_INCOMPLETE"

def select_dataset_candidate(catalog: DatasetCandidateCatalog) -> DatasetSelectionDecision:
    if not catalog.is_complete:
        return DatasetSelectionDecision.infrastructure_incomplete(
            catalog_hash=catalog.content_hash,
            reason_codes=("CATALOG_INCOMPLETE",),
        )
    if catalog.candidate_model_executions != 0:
        raise ValueError("candidate catalog must have zero model executions")
    eligible = tuple(entry for entry in catalog.entries if entry.is_eligible)
    if not eligible:
        return DatasetSelectionDecision.scientific_red(
            catalog_hash=catalog.content_hash,
            reason_codes=("NO_ELIGIBLE_DATASET",),
        )
    selected = min(
        eligible,
        key=lambda entry: (
            -entry.eligible_participant_count,
            entry.public_release_date,
            entry.canonical_source_locator,
        ),
    )
    return DatasetSelectionDecision.selected(
        catalog_hash=catalog.content_hash,
        selected_entry_hash=selected.content_hash,
        selected_locator=selected.canonical_source_locator,
    )
```

`DatasetSearchProtocol` must bind cutoff `2026-08-30`, exact ordered query strings, ordered repository families, query date, API/UI versions, pagination limit, sort order, deduplication key, eligibility fields, stopping rule, and `candidate_model_code_available=false`. `DatasetCandidateCatalog.is_complete` is true only when every required page is present and every DOI-linked repository has a resolution receipt. Every record uses strict `to_payload()` / `from_payload()` decoding and a canonical `content_hash`.

- [ ] **Step 4: Run GREEN and regression tests**

Run: `python3 -B -m unittest tests.test_cross_dataset_search -v`

Expected: PASS.

Run: `python3 -B -m unittest tests.test_external_validation_preregistration tests.test_measurement_validity_protocol_lock -v`

Expected: PASS.

- [ ] **Step 5: Commit the GREEN implementation**

```bash
git add narrative_dynamics/cross_dataset_search.py
git commit -m "feat: freeze governed dataset search"
```

## Task 2: Add Source Identity, Local Verification, and Semantic Pipeline Contracts

**Files:**

- Create: `narrative_dynamics/cross_dataset_source.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_source.py`

**Interfaces:**

- Consumes: selected `DatasetCatalogEntry.content_hash` from Task 1.
- Produces: `DatasetSourceFile`, `DatasetSourceManifest`, `VerifiedSourceSnapshot`, `CanonicalTransferTrial`, `TransformReceipt`, `SemanticPipelineResult`, `SemanticInvarianceEvidence`, `verify_source_snapshot()`, `evaluate_end_to_end_semantic_invariance()`.

- [ ] **Step 1: Write tests that lock the local-only and end-to-end boundary**

```python
def test_snapshot_verification_is_local_exact_and_hash_bound(tmp_path):
    root, manifest = synthetic_source_tree(tmp_path)
    verified = verify_source_snapshot(root, manifest)
    assert verified.source_manifest_hash == manifest.content_hash
    assert verified.selected_catalog_entry_hash == manifest.selected_catalog_entry_hash
    assert all(not Path(path).is_absolute() for path, _, _ in verified.files)

def test_semantic_invariance_runs_the_whole_pipeline():
    pipeline = recording_semantic_pipeline()
    evidence = evaluate_end_to_end_semantic_invariance(
        pipeline=pipeline,
        source_rows=synthetic_rows(),
        relabelings=five_coherent_relabelings(),
    )
    assert evidence.passed is True
    assert evidence.relabeling_count == 5
    assert pipeline.calls == ("parse", "transform", "scenario", "predict", "score", "report") * 6

def test_postchoice_mutation_does_not_change_model_visible_scenario():
    original, mutated = same_prechoice_different_postchoice_rows()
    assert synthetic_pipeline().scenario_hash(original) == synthetic_pipeline().scenario_hash(mutated)
```

Add explicit rejection tests for path traversal, non-HTTPS locators, duplicate paths, byte-size/hash drift, extra consumed files, malformed retained rows, duplicate trials, ambiguous order, unknown action/state labels, post-split exclusion, target drift, and an invariance evaluator that calls only a relabel function without parse/predict/score/report stages.

- [ ] **Step 2: Run and commit test-only RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_source -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_source`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_source.py
git commit -m "test: require source-neutral transfer boundary"
```

- [ ] **Step 3: Implement strict identity and semantic contracts**

```python
def verify_source_snapshot(root: Path, manifest: DatasetSourceManifest) -> VerifiedSourceSnapshot:
    resolved_root = root.resolve(strict=True)
    verified = []
    for source_file in manifest.files:
        candidate = (resolved_root / source_file.path).resolve(strict=True)
        if resolved_root not in candidate.parents:
            raise ValueError("source path escapes verified root")
        payload = candidate.read_bytes()
        actual_hash = "sha256:" + hashlib.sha256(payload).hexdigest()
        if len(payload) != source_file.byte_size or actual_hash != source_file.sha256:
            raise ValueError("source file identity mismatch")
        verified.append((source_file.path, len(payload), actual_hash))
    return VerifiedSourceSnapshot.create(manifest=manifest, files=tuple(verified))

def evaluate_end_to_end_semantic_invariance(
    *, pipeline: SemanticPipeline, source_rows: tuple[Mapping[str, str], ...],
    relabelings: tuple[SemanticRelabeling, ...],
) -> SemanticInvarianceEvidence:
    reference = pipeline.run(source_rows, SemanticRelabeling.identity())
    variants = tuple(pipeline.run(source_rows, relabeling) for relabeling in relabelings)
    passed = all(variant.inverse_mapped_identity == reference.identity for variant in variants)
    return SemanticInvarianceEvidence.create(reference, variants, passed=passed)
```

`CanonicalTransferTrial` retains a private participant key only in memory; its durable companion `TransformReceipt` contains source/snapshot/adapter/schema/endpoint hashes and aggregate counts only. `SemanticPipeline.run()` must return stage receipts for exactly `parse`, `transform`, `scenario`, `predict`, `score`, and `report`; missing or repeated stages reject evidence.

- [ ] **Step 4: Run GREEN and existing transform regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_source tests.test_two_stage_transform tests.test_feher_hare_measurement_invariance -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_source.py
git commit -m "feat: add source-neutral transfer contracts"
```

## Task 3: Build Participant-Disjoint Roles and the Public Privacy Manifest

**Files:**

- Create: `narrative_dynamics/cross_dataset_privacy.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_privacy.py`

**Interfaces:**

- Consumes: private canonical participant identifiers and source strata from Task 2.
- Produces: `ParticipantRole`, `RestrictedStudySecret`, private `RoleAssignmentIndex`, public `PublicSplitManifest`, `assign_participant_roles()`.

- [ ] **Step 1: Write privacy and split RED tests**

```python
def test_split_is_stratified_deterministic_disjoint_and_exact():
    secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
    private_index, public_manifest = assign_participant_roles(
        namespace="cross-dataset-transfer-v1",
        source_snapshot_hash=digest("snapshot"),
        participants=synthetic_stratified_participants((10, 15)),
        secret=secret,
    )
    assert private_index.role_counts == (("s0", 6, 2, 2), ("s1", 9, 3, 3))
    assert private_index.has_overlap is False
    assert public_manifest.role_counts == private_index.role_counts

def test_public_manifest_is_aggregate_only_and_non_enumerable():
    _, manifest = assigned_split()
    payload = json.dumps(manifest.to_payload(), sort_keys=True)
    for forbidden in ("participant_id", "participant_token", "hmac", "commitments"):
        assert forbidden not in payload.lower()
    assert manifest.merkle_root.startswith("sha256:")
    assert manifest.key_commitment.startswith("sha256:")

def test_secret_close_zeroizes_and_blocks_reuse():
    secret = RestrictedStudySecret.from_bytes(bytes(range(32)))
    secret.close()
    with self.assertRaisesRegex(RuntimeError, "secret is closed"):
        secret.commit("p-1", "s0")
```

Add tests for secret length exactly 32 bytes, no random nonce field, empty roles, duplicate participants, cross-stratum duplicates, deterministic `floor(0.60*n)` / `floor(0.20*n)` / remainder counts, missing strata, bool/non-string identifiers, post-assignment mutation, `repr()` redaction, pickling rejection, and absence of a public method returning the raw key or full commitment list.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_privacy -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_privacy`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_privacy.py
git commit -m "test: require private participant role split"
```

- [ ] **Step 3: Implement HMAC assignment and aggregate manifest**

```python
class ParticipantRole(str, Enum):
    TRAIN = "TRAIN"
    SELECTION_VALIDATION = "SELECTION_VALIDATION"
    FINAL_TEST = "FINAL_TEST"

def _participant_commitment(secret: RestrictedStudySecret, namespace: str,
                            snapshot_hash: str, stratum: str,
                            participant_id: str) -> bytes:
    message = "\x1f".join((namespace, snapshot_hash, stratum, participant_id)).encode()
    return secret.digest(message)

def _role_counts(n: int) -> tuple[int, int, int]:
    train = math.floor(0.60 * n)
    selection = math.floor(0.20 * n)
    return train, selection, n - train - selection
```

Order each stratum by the HMAC bytes and assign the exact counts. Build the Merkle root over domain-separated leaf commitments while retaining leaves only inside `RoleAssignmentIndex`. `PublicSplitManifest.to_payload()` exposes only namespace, snapshot hash, aggregate role/stratum counts, Merkle root, `HMAC-SHA256`, one-way key commitment, transform attestation hash, and its content hash. `RestrictedStudySecret.close()` overwrites its internal `bytearray` before marking the capability closed.

- [ ] **Step 4: Run GREEN and dataset partition regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_privacy tests.test_two_stage_transform -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_privacy.py
git commit -m "feat: protect transfer participant splits"
```

## Task 4: Enforce Role Capabilities and a Non-Iterable FINAL Vault Handle

**Files:**

- Create: `narrative_dynamics/cross_dataset_capabilities.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_capabilities.py`

**Interfaces:**

- Consumes: private `RoleAssignmentIndex`, canonical trials, and `PublicSplitManifest` from Task 3.
- Produces: `TrainProjection`, `SelectionProjection`, `FinalProjectionCommitment`, `FinalVaultHandle`, `PreparedCrossDatasetTransferV1`, `FinalUnlockGrant`, `FinalVaultBackend` protocol, `provision_transfer_capabilities()`.

- [ ] **Step 1: Write hostile capability tests**

```python
def test_prepared_surface_has_no_general_dataset_or_final_rows():
    prepared, _backend = prepared_transfer()
    assert set(prepared.__dataclass_fields__) == {
        "source_identity_hash", "split_manifest", "train",
        "selection_validation", "final_commitment", "final_vault_handle",
    }
    payload = json.dumps(prepared.identity_payload(), sort_keys=True).lower()
    assert "target" not in payload
    assert "scenario" not in payload
    assert "participant" not in payload

def test_final_handle_is_non_iterable_non_serializable_and_path_free():
    prepared, _backend = prepared_transfer()
    handle = prepared.final_vault_handle
    with self.assertRaises(TypeError):
        iter(handle)
    with self.assertRaises(TypeError):
        pickle.dumps(handle)
    assert "path" not in repr(handle).lower()

def test_vault_unlock_requires_exact_single_use_grant():
    prepared, backend = prepared_transfer()
    grant = final_unlock_grant(prepared)
    projection = backend.unlock(prepared.final_vault_handle, grant)
    assert projection.commitment_hash == prepared.final_commitment.content_hash
    with self.assertRaisesRegex(RuntimeError, "already consumed"):
        backend.unlock(prepared.final_vault_handle, grant)
```

Add tests proving zero-shot receives no role capability; refit receives TRAIN and SELECTION only; no object reachable from either constructor has a FINAL opener/path/credential; forged, stale-ledger, wrong-preflight, wrong-release, wrong-lock, or wrong-revision grants fail before unlock; and vault backend state is process/service-owned rather than embedded in `PreparedCrossDatasetTransferV1`.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_capabilities -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_capabilities`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_capabilities.py
git commit -m "test: require sealed transfer capabilities"
```

- [ ] **Step 3: Implement opaque role projections and grant verification**

```python
@dataclass(frozen=True)
class FinalUnlockGrant:
    scientific_revision: str
    ledger_head_hash: str
    preflight_hash: str
    authorization_receipt_hash: str
    brier_release_hash: str
    log_release_hash: str
    lock_commit: str

class FinalVaultBackend(Protocol):
    def unlock(self, handle: FinalVaultHandle,
               grant: FinalUnlockGrant) -> FinalWorkerProjection:
        raise NotImplementedError
```

Keep projection row containers module-private and expose only task-specific iteration methods on TRAIN/SELECTION. `FinalVaultHandle` contains an unguessable opaque capability ID and the commitment hash only, defines `__reduce_ex__` to raise `TypeError`, and exposes no iterator, length, path, target, or scenario API. The backend validates every `FinalUnlockGrant` field against its provisioned immutable policy before setting `final_projection_opened=true` and consuming the handle.

- [ ] **Step 4: Run GREEN and observation boundary regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_capabilities tests.test_external_validation_constraints tests.test_two_stage_final_attempts -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_capabilities.py
git commit -m "feat: seal transfer role capabilities"
```

## Task 5: Freeze Exact Zero-Shot and Complete Refit Candidates

**Files:**

- Create: `narrative_dynamics/cross_dataset_candidates.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_candidates.py`

**Interfaces:**

- Consumes: R3 source identities, TRAIN/SELECTION capabilities, builder/loss/tie-break identities.
- Produces: `TransferFamily`, `FrozenTransferCandidate`, `ZeroShotCandidateFreeze`, `GridCandidateEvaluation`, `RefitCandidateFreeze`, `freeze_zero_shot_candidates()`, `freeze_refit_candidates()`.

- [ ] **Step 1: Write exact-candidate and full-grid tests**

```python
def test_zero_shot_freeze_matches_r3_exactly():
    freeze = freeze_zero_shot_candidates(r3_anchor(), new_source_lineage())
    assert tuple((c.family.value, c.parameters, c.source_candidate_hash) for c in freeze.candidates) == (
        ("reactive", (("beta", 0.5),), "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456"),
        ("intentional", (("beta", 2.0), ("memory_decay", 0.5)), "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839"),
        ("planning", (("beta", 4.0), ("memory_decay", 0.75)), "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c"),
    )
    assert freeze.new_data_training is False
    assert freeze.new_data_selection is False

def test_refit_freeze_requires_every_grid_point_and_lexical_tie_break():
    freeze = freeze_refit_candidates(
        train=train_projection(), selection=selection_projection(),
        evaluations=complete_36_point_evaluations(with_tie=True),
        brier_loss_identity=digest("brier"), builder_identities=builder_identities(),
        simulation_identity=digest("simulation"), tie_break_identity="lexical_parameters_v1",
    )
    assert len(freeze.evaluations) == 36
    assert freeze.final_projection_opened is False
    assert freeze.final_model_execution is False
```

Add tests for exact grids, TRAIN seeds `(101, 102)`, SELECTION seeds `(201, 202)`, duplicate/missing/extra grid points, score other than Brier, builder drift, simulation drift, tie-break drift, target/split manifest drift, non-finite loss, and any FINAL field or prior FINAL result supplied to refit.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_candidates -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_candidates`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_candidates.py
git commit -m "test: require exact transfer candidate freezes"
```

- [ ] **Step 3: Implement exact anchors and complete-grid validation**

```python
REACTIVE_GRID = ((("beta", 0.5),), (("beta", 1.0),),
                 (("beta", 2.0),), (("beta", 4.0),))
MEMORY_GRID = tuple(
    (("beta", beta), ("memory_decay", decay))
    for beta in (0.5, 1.0, 2.0, 4.0)
    for decay in (0.5, 0.75, 0.9, 1.0)
)

def _select_candidate(rows: tuple[GridCandidateEvaluation, ...]) -> GridCandidateEvaluation:
    return min(rows, key=lambda row: (row.selection_brier_loss, row.parameters))
```

Validate the complete expected set separately for each family, bind every evaluation to the same split/target manifests and callable identities, and serialize the complete evaluation receipts in the refit freeze. The selected candidates are derived only after set equality succeeds.

- [ ] **Step 4: Run GREEN and R3 anchor regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_candidates tests.test_feher_hare_two_stage_protocol tests.test_feher_hare_measurement_validity_pipeline -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_candidates.py
git commit -m "feat: freeze transfer candidate paths"
```

## Task 6: Implement TRAIN-Only Baselines and Aggregate Bootstrap Evidence

**Files:**

- Create: `narrative_dynamics/cross_dataset_inference.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_inference.py`

**Interfaces:**

- Consumes: TRAIN projection and ephemeral FINAL case losses.
- Produces: `FrozenBaseRateCell`, `FrozenBaseRateComparator`, private `_ParticipantLossBlock`, `TransferInferenceEvidence`, `fit_train_base_rate()`, `participant_equal_loss()`, `trial_equal_loss()`, `paired_participant_bootstrap()`.

- [ ] **Step 1: Write numerical and privacy RED tests**

```python
def test_train_base_rate_uses_laplace_one_per_stratum():
    baseline = fit_train_base_rate(train_projection(actions=("action_1", "action_1", "action_0")))
    assert baseline.cells[0].probability_action_1 == 3 / 5

def test_paired_bootstrap_is_exact_and_aggregate_only():
    evidence = paired_participant_bootstrap(
        candidate=participant_blocks((0.10, 0.20, 0.30)),
        comparator=participant_blocks((0.20, 0.30, 0.40)),
        score="BRIER", aggregation="PARTICIPANT_EQUAL",
        candidate_identity=digest("candidate"), comparator_identity=digest("baseline"),
        seed=43001, replicates=10000,
    )
    assert evidence.bootstrap_seed == 43001
    assert evidence.bootstrap_replicates == 10000
    assert evidence.pass_status == (evidence.relative_improvement >= 0.01 and evidence.lower_95 > 0.0)
    payload = json.dumps(evidence.to_payload(), sort_keys=True).lower()
    assert "participant" not in payload
    assert "blocks" not in payload
```

Add reference-vector tests for participant-equal versus trial-equal loss, paired resampling, linear percentile interpolation, both Brier and clipped Log, exact boundary `0.01`, lower endpoint exactly zero, unknown baseline stratum, empty participant block, duplicate token, non-finite loss, unequal candidate/comparator participants, and serialization rejection for participant-level objects.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_inference -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_inference`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_inference.py
git commit -m "test: require aggregate transfer inference"
```

- [ ] **Step 3: Implement exact aggregation and bootstrap rules**

```python
def relative_improvement(candidate_loss: float, comparator_loss: float) -> float:
    if comparator_loss <= 0.0:
        raise ValueError("comparator loss must be positive")
    return (comparator_loss - candidate_loss) / comparator_loss

def _linear_percentile(values: tuple[float, ...], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
```

Use `random.Random(43001)` and complete participant blocks for exactly 10,000 paired resamples. Construct `TransferInferenceEvidence` only after discarding participant rows; its payload contains aggregate losses, point improvement, interval endpoints, seed, replicates, percentile rule, aggregate participant count, threshold, pass status, identities, optional source stratum, and content hash.

- [ ] **Step 4: Run GREEN and loss regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_inference tests.test_two_stage_metrics tests.test_measurement_validity_robustness -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_inference.py
git commit -m "feat: add aggregate transfer inference"
```

## Task 7: Add the Authoritative Git-Backed Attempt Ledger

**Files:**

- Create: `narrative_dynamics/cross_dataset_ledger.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_ledger.py`

**Interfaces:**

- Produces: `TransferAttemptEventType`, `TransferAttemptEvent`, `TransferAttemptHistory`, `TransferAttemptStore` protocol, `GitTransferAttemptStore`, `ConcurrentAttemptError`.
- Store contract: `head()`, `history()`, and `compare_and_append(expected_head_hash, event) -> str`.

- [ ] **Step 1: Write local-bare-Git RED tests**

```python
def test_compare_and_append_rejects_stale_or_empty_replacement(tmp_path):
    store = local_git_attempt_store(tmp_path)
    first = store.compare_and_append(store.head(), final_started_event(parent=store.head()))
    with self.assertRaises(ConcurrentAttemptError):
        store.compare_and_append(store.genesis_hash, failure_event(parent=store.genesis_hash))
    assert store.head() == first

def test_two_writers_cannot_both_start_final(tmp_path):
    left, right = two_git_store_clients(tmp_path)
    expected = left.head()
    left.compare_and_append(expected, final_started_event(parent=expected))
    with self.assertRaises(ConcurrentAttemptError):
        right.compare_and_append(expected, final_started_event(parent=expected))

def test_history_proves_zero_prior_opening_and_execution(tmp_path):
    store = local_git_attempt_store(tmp_path)
    history = store.history()
    assert history.final_projection_openings == 0
    assert history.completed_model_runs == 0
    assert history.prediction_artifact_hashes == ()
```

Add tests for signed/attested canonical payloads, exact parent chain, non-force fast-forward behavior, corrupt branch history, duplicate STARTED, durable STARTED visibility from a fresh clone, run-progress persistence, vault-open persistence, prediction hash persistence, terminal uniqueness, and branch name exactly `ledger/narrative-cross-dataset-transfer-v1`.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_ledger -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_ledger`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_ledger.py
git commit -m "test: require authoritative transfer ledger"
```

- [ ] **Step 3: Implement canonical events and remote compare-and-append**

```python
class TransferAttemptStore(Protocol):
    def head(self) -> str:
        raise NotImplementedError

    def history(self) -> TransferAttemptHistory:
        raise NotImplementedError

    def compare_and_append(self, expected_head_hash: str,
                           event: TransferAttemptEvent) -> str:
        raise NotImplementedError

class ConcurrentAttemptError(RuntimeError):
    """Raised when the authoritative ledger head changes before append."""
```

`TransferAttemptEvent` binds `event_type`, `parent_ledger_head`, `scientific_revision`, `attempt_id`, immutable UTC timestamp, repository/run/job receipts, a canonical event payload hash, and an attestation hash. `TransferAttemptHistory` recomputes every event hash and parent link before deriving vault-opening, completed-run, prediction/score, authorization-consumption, and terminal counts.

The concrete store writes one canonical JSON event as one Git commit whose parent is the verified branch head, then pushes `new_commit:refs/heads/ledger/narrative-cross-dataset-transfer-v1` without force. Before returning, fetch and verify that the remote head equals the new commit. A rejected non-fast-forward push becomes `ConcurrentAttemptError`; it never retries automatically. Tests use a local bare remote and subprocess environment with fixed author metadata and no network.

- [ ] **Step 4: Run GREEN including concurrency tests**

Run: `python3 -B -m unittest tests.test_cross_dataset_ledger -v`

Expected: PASS, including exactly one winner in the two-writer test.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_ledger.py
git commit -m "feat: add authoritative transfer ledger"
```

## Task 8: Freeze Sibling Score Releases and an Authoritative Dual Preflight

**Files:**

- Create: `narrative_dynamics/cross_dataset_release.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_release.py`

**Interfaces:**

- Consumes: source/split/candidate/baseline identities, current `TransferAttemptStore` head/history.
- Produces: `TransferScore`, `TransferProtocol`, `TransferScoreRelease`, `DualTransferPreflight`, `build_transfer_protocol()`, `preflight_transfer_releases()`.

- [ ] **Step 1: Write sibling and zero-execution RED tests**

```python
def test_sibling_releases_differ_only_by_score_identity():
    brier, log = sibling_releases()
    assert brier.score is TransferScore.BRIER
    assert log.score is TransferScore.LOG
    assert brier.shared_identity_payload() == log.shared_identity_payload()
    assert brier.prediction_artifact_identity == log.prediction_artifact_identity

def test_preflight_binds_current_authoritative_head_and_zero_final_history(tmp_path):
    store = local_git_attempt_store(tmp_path)
    brier, log = sibling_releases()
    preflight = preflight_transfer_releases(brier, log, store=store)
    assert preflight.ledger_head_hash == store.head()
    assert preflight.final_projection_openings == 0
    assert preflight.completed_model_runs == 0
    assert preflight.prediction_artifact_hashes == ()
```

Add drift tests for source, transform, split, six candidates, baseline, FINAL commitment, seeds, aggregation, bootstrap, thresholds, family vocabulary, top-level terminal vocabulary, limitations, `external_registration`, carry-forward IDs, scientific revision, prediction identity, stale ledger head, prior FINAL_STARTED, prior vault opening, prior run, prior prediction, and an in-memory caller ledger.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_release -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_release`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_release.py
git commit -m "test: require governed transfer releases"
```

- [ ] **Step 3: Implement score sibling and preflight identity checks**

```python
class TransferScore(str, Enum):
    BRIER = "BRIER"
    LOG = "LOG"

CARRY_FORWARD_REQUIREMENT_IDS = (
    "MV1_RI_SELECTION_SPACESHIP_TASK_LOG_BOTH_AGGREGATIONS",
    "MV1_RI_SELECTION_SPACESHIP_SCORE_BOTH_AGGREGATIONS",
    "MV1_RI_SELECTION_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS",
    "MV1_IP_TRAIN_SPACESHIP_LOPO_BRIER_LOG_BOTH_AGGREGATIONS",
)

def preflight_transfer_releases(brier: TransferScoreRelease,
                                log: TransferScoreRelease,
                                *, store: TransferAttemptStore) -> DualTransferPreflight:
    _require_exact_siblings(brier, log)
    head = store.head()
    history = store.history()
    if history.head_hash != head or history.has_any_final_activity:
        raise ValueError("authoritative ledger is not zero-FINAL at current head")
    return DualTransferPreflight.create(brier, log, ledger_head_hash=head, history=history)
```

The protocol freezes both aggregations, both scores, task/condition strata, participant influence diagnostics, exact thresholds, all four carry-forward IDs, `external_registration=false`, and the three top-level terminal identifiers.

- [ ] **Step 4: Run GREEN and release regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_release tests.test_external_validation_release_gate tests.test_measurement_validity_protocol_lock -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_release.py
git commit -m "feat: freeze transfer score releases"
```

## Task 9: Parse and Verify the Exact Post-Preflight Authorization Receipt

**Files:**

- Create: `narrative_dynamics/cross_dataset_authorization.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_authorization.py`

**Interfaces:**

- Consumes: GitHub #43 comment metadata, `DualTransferPreflight`, release hashes, lock commit, authorized owner login.
- Produces: `TransferAuthorizationReceipt`, `parse_transfer_authorization()`.

- [ ] **Step 1: Write exact-text, identity, and timing RED tests**

```python
def test_exact_post_preflight_owner_comment_builds_receipt():
    receipt = parse_transfer_authorization(
        issue_number=43, comment=authorization_comment(),
        authorized_owner="qigao", preflight=dual_preflight(),
        lock_commit=sha("lock"),
    )
    assert receipt.issue_number == 43
    assert receipt.author_login == "qigao"
    assert receipt.pre_authorization_ledger_head == dual_preflight().ledger_head_hash

def test_edited_predated_or_identity_drifted_comment_is_rejected():
    for comment in (edited_comment(), predated_comment(), wrong_sha_comment()):
        with self.assertRaises(ValueError):
            parse_transfer_authorization(
                issue_number=43, comment=comment, authorized_owner="qigao",
                preflight=dual_preflight(), lock_commit=sha("lock"),
            )
```

Add tests for issue other than 43, author mismatch, deleted comment, surrounding text, reordered lines, missing line, duplicate line, CRLF normalization attempt, non-canonical SHA/hash, timestamp equal to or earlier than preflight, stale ledger head, wrong Brier/Log release, wrong lock, wrong scientific revision, and receipt reuse after STARTED.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_authorization -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_authorization`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_authorization.py
git commit -m "test: require exact transfer authorization"
```

- [ ] **Step 3: Implement exact six-line parsing**

```python
AUTHORIZATION_HEADER = "I authorize Cross-dataset Transfer V1 locked FINAL_TEST execution."

def _expected_text(scientific_sha: str, lock_commit: str, preflight_hash: str,
                   brier_release_hash: str, log_release_hash: str) -> str:
    return "\n".join((
        AUTHORIZATION_HEADER,
        f"scientific_sha={scientific_sha}",
        f"lock_commit={lock_commit}",
        f"preflight_hash={preflight_hash}",
        f"brier_release_hash={brier_release_hash}",
        f"log_release_hash={log_release_hash}",
    ))
```

Require immutable comment ID/URL, `created_at == updated_at`, exact author login, exact UTF-8 text, creation strictly after preflight completion, and all five identities. Bind the pre-authorization ledger head into the receipt hash. The receipt exposes no method to mutate or refresh its fields.

- [ ] **Step 4: Run GREEN**

Run: `python3 -B -m unittest tests.test_cross_dataset_authorization tests.test_cross_dataset_release -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_authorization.py
git commit -m "feat: verify exact transfer authorization"
```

## Task 10: Execute One Ephemeral Six-Candidate FINAL Prediction Matrix

**Files:**

- Create: `narrative_dynamics/cross_dataset_prediction.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_prediction.py`

**Interfaces:**

- Consumes: worker-only `FinalWorkerProjection`, three zero-shot and three refit candidates, FINAL seeds `(301, 302)`, progress callback.
- Produces: private case predictions, `SealedTransferPredictionArtifact`, `execute_transfer_final_predictions()`.

- [ ] **Step 1: Write exact-matrix and secrecy RED tests**

```python
def test_every_final_case_runs_six_candidates_by_two_seeds_once():
    calls = []
    artifact = execute_transfer_final_predictions(
        projection=final_worker_projection(case_count=4),
        zero_shot=zero_shot_freeze(), refit=refit_freeze(),
        seeds=(301, 302), evaluator=recording_evaluator(calls),
        progress=recording_progress(),
    )
    assert len(calls) == 4 * 6 * 2
    assert len(set(calls)) == len(calls)
    assert artifact.case_count == 4

def test_prediction_artifact_has_no_durable_serializer():
    artifact = sealed_prediction_artifact()
    assert not hasattr(artifact, "to_payload")
    with self.assertRaises(TypeError):
        pickle.dumps(artifact)
```

Add tests for seed drift, duplicate/missing candidate, wrong candidate path, case reuse, FINAL commitment mismatch, evaluator exception, progress persistence before the next run, probabilities not finite/simplex-complete, target leakage into evaluator input, and one sealed artifact identity shared by both score siblings.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_prediction -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_prediction`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_prediction.py
git commit -m "test: require one sealed transfer prediction"
```

- [ ] **Step 3: Implement worker-only deterministic execution**

```python
FINAL_SEEDS = (301, 302)

def execute_transfer_final_predictions(*, projection: FinalWorkerProjection,
        zero_shot: ZeroShotCandidateFreeze, refit: RefitCandidateFreeze,
        seeds: tuple[int, ...], evaluator: TransferEvaluator,
        progress: Callable[[TransferRunProgress], None]) -> SealedTransferPredictionArtifact:
    if seeds != FINAL_SEEDS:
        raise ValueError("FINAL seeds must be exactly (301, 302)")
    candidates = _exact_six_candidates(zero_shot, refit)
    rows = []
    for case in projection.consume_once():
        for candidate in candidates:
            for seed in seeds:
                prediction = evaluator(case.model_input(), candidate, seed)
                rows.append(_validated_private_prediction(case, candidate, seed, prediction))
                progress(TransferRunProgress.from_row(rows[-1]))
    return SealedTransferPredictionArtifact.seal(projection, candidates, seeds, tuple(rows))
```

The evaluator receives only model-visible pre-choice state. Targets remain attached inside private artifact rows for scoring and never enter evaluator arguments, logs, progress events, exceptions, or durable payloads.

- [ ] **Step 4: Run GREEN and adapter identity regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_prediction tests.test_narrative_two_stage_adapter tests.test_function_implementation_attestation -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_prediction.py
git commit -m "feat: seal transfer final predictions"
```

## Task 11: Score Aggregate Evidence and Assemble Typed Transfer Reports

**Files:**

- Create: `narrative_dynamics/cross_dataset_reporting.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_reporting.py`

**Interfaces:**

- Consumes: one sealed prediction artifact, TRAIN baseline, sibling releases, semantic evidence, aggregate diagnostics.
- Produces: `TransferAdequacyStatus`, `TransferFamilyFinding`, `CarryForwardSensitivityStatus`, `TransferStudyTerminal`, `CrossDatasetTransferReport`, `score_and_report_transfer()`.

- [ ] **Step 1: Write terminal vocabulary and privacy RED tests**

```python
def test_family_finding_matrix_is_exact():
    assert classify_family_finding(True, True, False) == "PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE"
    assert classify_family_finding(False, True, False) == "FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED"
    assert classify_family_finding(True, False, False) == "ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED"
    assert classify_family_finding(False, False, False) == "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL"
    assert classify_family_finding(True, True, True) == "MIXED_OR_INCONCLUSIVE_EVIDENCE"

def test_valid_negative_family_results_still_form_green_study():
    report = score_and_report_transfer(valid_negative_scoring_input())
    assert report.terminal is TransferStudyTerminal.GREEN
    assert all(f.finding == "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL" for f in report.family_findings)

def test_report_is_aggregate_hash_only():
    payload = json.dumps(score_and_report_transfer(green_input()).to_payload(), sort_keys=True).lower()
    for forbidden in ("participant_id", "participant_token", "case_prediction", "final_target", "source_path"):
        assert forbidden not in payload
```

Add tests for Brier/Log disagreement, participant-equal primary rule, trial-equal diagnostic non-override, task/condition strata, participant influence, refit material gain, all four exact carry-forward IDs with comparable/`NOT_ESTABLISHED` status, semantic gate failure, aggregate interval completeness, terminal exclusivity, `SCIENTIFIC_RED`, `INFRASTRUCTURE_INCOMPLETE`, unknown fields, and forbidden-key/value archive scan.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_reporting -v`

Expected: FAIL with missing `narrative_dynamics.cross_dataset_reporting`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_reporting.py
git commit -m "test: require typed aggregate transfer reports"
```

- [ ] **Step 3: Implement scoring and terminal assembly**

```python
class TransferStudyTerminal(str, Enum):
    GREEN = "GREEN"
    SCIENTIFIC_RED = "SCIENTIFIC_RED"
    INFRASTRUCTURE_INCOMPLETE = "INFRASTRUCTURE_INCOMPLETE"

def classify_family_finding(zero_shot: bool, refit: bool, mixed: bool) -> str:
    if mixed:
        return "MIXED_OR_INCONCLUSIVE_EVIDENCE"
    return {
        (True, True): "PARAMETER_AND_FAMILY_TRANSFER_EVIDENCE",
        (False, True): "FAMILY_TRANSFER_ONLY_RETRAINING_REQUIRED",
        (True, False): "ZERO_SHOT_TRANSFER_REFIT_NOT_ESTABLISHED",
        (False, False): "NO_TRANSFER_EVIDENCE_UNDER_PROTOCOL",
    }[(zero_shot, refit)]
```

Score Brier and clipped Log from the same private prediction rows, immediately reduce case losses to ephemeral participant blocks, build `TransferInferenceEvidence`, and discard row/block state before report construction. A study is GREEN when the frozen protocol completes validly and every family has one typed finding, even when all family findings are negative.

- [ ] **Step 4: Run GREEN and reporting regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_reporting tests.test_external_validation_reporting tests.test_measurement_validity_reporting -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/cross_dataset_reporting.py
git commit -m "feat: report aggregate transfer evidence"
```

## Task 12: Enforce the Locked FINAL State Machine and Safe Replay Rule

**Files:**

- Create: `narrative_dynamics/studies/cross_dataset_transfer_locked_final.py`
- Modify: `tests/cross_dataset_transfer_fixtures.py`
- Create: `tests/test_cross_dataset_locked_final.py`

**Interfaces:**

- Consumes: authoritative store, dual preflight, authorization receipt, FINAL vault backend/handle, candidate freezes, baseline, evaluator.
- Produces: `LockedCrossDatasetTransferResult`, `TransferFinalInfrastructureError`, `run_locked_cross_dataset_transfer_final()`.

- [ ] **Step 1: Write failure-injection RED tests for every boundary**

```python
def test_started_is_persisted_before_vault_unlock(tmp_path):
    order = []
    result = run_locked_cross_dataset_transfer_final(
        **locked_final_inputs(tmp_path, order=order)
    )
    assert order.index("ledger:FINAL_STARTED") < order.index("vault:open")
    assert result.terminal_event == "FINAL_COMPLETED"

def test_crash_after_vault_open_requires_revision(tmp_path):
    inputs = locked_final_inputs(tmp_path, fail_at="after_vault_open")
    with self.assertRaises(TransferFinalInfrastructureError):
        run_locked_cross_dataset_transfer_final(**inputs)
    history = inputs["store"].history()
    assert history.last_event.event_type.value == "REVISION_REQUIRED"
    assert history.final_projection_openings == 1

def test_preunlock_infrastructure_failure_can_only_record_replay_allowed(tmp_path):
    inputs = locked_final_inputs(tmp_path, fail_at="before_vault_open")
    with self.assertRaises(TransferFinalInfrastructureError):
        run_locked_cross_dataset_transfer_final(**inputs)
    history = inputs["store"].history()
    assert history.last_event.event_type.value == "EXACT_REPLAY_ALLOWED"
    assert history.final_projection_openings == 0
    assert history.completed_model_runs == 0
    assert history.prediction_artifact_hashes == ()
```

Add failure tests before STARTED, infrastructure failure after STARTED/before unlock, schema failure after STARTED/before unlock, during unlock, after first model run, during prediction seal, during Brier scoring, during Log scoring, during report assembly, during FINAL_COMPLETED append, during always-run evidence persistence, authorization reuse, concurrent STARTED, stale preflight, and attempted automatic replay. Assert the pre-unlock schema failure is `REVISION_REQUIRED`, and every post-STARTED path appends a durable typed event and emits aggregate/hash-only evidence.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_locked_final -v`

Expected: FAIL with missing `narrative_dynamics.studies.cross_dataset_transfer_locked_final`.

```bash
git add tests/cross_dataset_transfer_fixtures.py tests/test_cross_dataset_locked_final.py
git commit -m "test: require durable locked transfer final"
```

- [ ] **Step 3: Implement the ordered state machine**

```python
def run_locked_cross_dataset_transfer_final(*, store, preflight, authorization,
        vault_backend, vault_handle, candidates, baseline, evaluator,
        evidence_sink) -> LockedCrossDatasetTransferResult:
    _verify_locked_inputs(store, preflight, authorization, candidates, baseline)
    head = store.compare_and_append(
        preflight.ledger_head_hash,
        _final_started(preflight, authorization),
    )
    opened = False
    completed_runs = 0
    prediction_hash = None
    try:
        projection = vault_backend.unlock(vault_handle, _unlock_grant(preflight, authorization))
        opened = True
        head = store.compare_and_append(head, _vault_opened(projection))

        def persist_progress(row: TransferRunProgress) -> None:
            nonlocal head, completed_runs
            head = store.compare_and_append(head, _run_progress(row))
            completed_runs += 1

        artifact = execute_transfer_final_predictions(
            projection=projection, zero_shot=candidates.zero_shot,
            refit=candidates.refit, seeds=(301, 302), evaluator=evaluator,
            progress=persist_progress,
        )
        completed_runs = artifact.completed_model_runs
        prediction_hash = artifact.content_hash
        head = store.compare_and_append(head, _prediction_sealed(artifact))
        report = score_and_report_transfer(_report_input(artifact, baseline, preflight))
        head = store.compare_and_append(head, _final_completed(report))
        return LockedCrossDatasetTransferResult(report=report, terminal_event="FINAL_COMPLETED")
    except Exception as exc:
        _persist_failure(store, opened=opened, completed_runs=completed_runs,
                         prediction_hash=prediction_hash, cause=exc)
        raise TransferFinalInfrastructureError("locked FINAL did not complete") from exc
    finally:
        evidence_sink.persist_aggregate_state(store.history())
```

`_persist_failure()` may choose `EXACT_REPLAY_ALLOWED` only when the cause is a typed infrastructure-only failure before vault unlock and authoritative history independently proves the vault never opened, completed runs are zero, and no prediction/score hash exists. A schema, scientific, privacy, prediction, scoring, or reporting failure is `REVISION_REQUIRED` even before unlock; every failure after vault opening is also `REVISION_REQUIRED`. No branch catches an exception and returns a successful report; no retry loop exists. If always-run evidence persistence itself fails, the function returns no successful result and appends `REVISION_REQUIRED` when the authoritative store remains reachable.

- [ ] **Step 4: Run GREEN and locked-FINAL regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_locked_final tests.test_two_stage_final_attempts tests.test_external_validation_final -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/studies/cross_dataset_transfer_locked_final.py
git commit -m "feat: govern locked transfer final"
```

## Task 13: Freeze the Public Surface and Prove Ordinary CI Is Offline

**Files:**

- Modify: `narrative_dynamics/__init__.py`
- Create: `tests/test_cross_dataset_public_api.py`
- Create: `tests/test_cross_dataset_ci_boundary.py`

**Interfaces:**

- Public: search/source identity records, public split manifest, candidate freezes, aggregate inference evidence, releases/preflight, authorization receipt, aggregate report enums/report.
- Private: raw secret, role assignment index, FINAL vault backend/opener, worker projection, participant blocks, per-case prediction rows, sealed artifact, and locked FINAL runner.

- [ ] **Step 1: Write RED surface and repository scans**

```python
def test_root_exports_only_durable_non_final_contracts():
    required = {
        "DatasetCandidateCatalog", "DatasetSourceManifest", "VerifiedSourceSnapshot",
        "PublicSplitManifest", "ZeroShotCandidateFreeze", "RefitCandidateFreeze",
        "TransferInferenceEvidence", "DualTransferPreflight",
        "TransferAuthorizationReceipt", "CrossDatasetTransferReport",
    }
    forbidden = {
        "RestrictedStudySecret", "RoleAssignmentIndex", "FinalVaultBackend",
        "FinalWorkerProjection", "_ParticipantLossBlock",
        "SealedTransferPredictionArtifact", "run_locked_cross_dataset_transfer_final",
    }
    assert required.issubset(set(narrative_dynamics.__all__))
    assert forbidden.isdisjoint(set(narrative_dynamics.__all__))

def test_phase_a_has_no_real_source_or_network_workflow():
    tracked = phase_a_runtime_text_files(
        base="0378a40e934b3b241d6883df709aa056caec5f69",
        roots=("narrative_dynamics", ".github/workflows"),
    )
    forbidden_patterns = (
        "osf.io/", "doi.org/", "requests.get(", "urllib.request",
        "workflow_dispatch:",
    )
    for path, text in tracked.items():
        assert not any(pattern in text for pattern in forbidden_patterns), path

def test_frozen_two_stage_adapter_is_byte_identical_to_base():
    assert git_blob("narrative_dynamics/adapters/narrative_two_stage.py") == \
        git_blob_at("c979eafe650a506bf30f78ab5b35078742b54087",
                    "narrative_dynamics/adapters/narrative_two_stage.py")
```

Also assert there is no source-specific module/manifest, no raw `.csv`/`.tsv`/`.parquet`, no participant-like key in durable payload fixtures, no new workflow file, no import-time network access, and no public callable that opens FINAL.

- [ ] **Step 2: Run and commit RED**

Run: `python3 -B -m unittest tests.test_cross_dataset_public_api tests.test_cross_dataset_ci_boundary -v`

Expected: FAIL because the approved public exports are absent.

```bash
git add tests/test_cross_dataset_public_api.py tests/test_cross_dataset_ci_boundary.py
git commit -m "test: freeze transfer phase a boundary"
```

- [ ] **Step 3: Add only the approved durable exports**

```python
from narrative_dynamics.cross_dataset_search import (
    CROSS_DATASET_TRANSFER_CLAIM_SCOPE,
    DatasetCandidateCatalog,
    DatasetCatalogEntry,
    DatasetSearchProtocol,
    DatasetSelectionDecision,
    DatasetSelectionStatus,
    select_dataset_candidate,
)
from narrative_dynamics.cross_dataset_source import (
    DatasetSourceFile,
    DatasetSourceManifest,
    SemanticInvarianceEvidence,
    TransformReceipt,
    VerifiedSourceSnapshot,
)
from narrative_dynamics.cross_dataset_privacy import PublicSplitManifest
from narrative_dynamics.cross_dataset_candidates import (
    RefitCandidateFreeze,
    ZeroShotCandidateFreeze,
)
from narrative_dynamics.cross_dataset_inference import TransferInferenceEvidence
from narrative_dynamics.cross_dataset_release import (
    DualTransferPreflight,
    TransferProtocol,
    TransferScore,
    TransferScoreRelease,
)
from narrative_dynamics.cross_dataset_authorization import TransferAuthorizationReceipt
from narrative_dynamics.cross_dataset_reporting import (
    CrossDatasetTransferReport,
    TransferStudyTerminal,
)
```

Add the exact names to `__all__`. Do not root-export capability constructors, ledger writers, authorization parsers, prediction/scoring internals, or the locked FINAL runner. `narrative_dynamics.studies.__init__` remains byte-identical to the approved base.

- [ ] **Step 4: Run GREEN and import regressions**

Run: `python3 -B -m unittest tests.test_cross_dataset_public_api tests.test_cross_dataset_ci_boundary tests.test_narrative_trust_api -v`

Expected: PASS.

- [ ] **Step 5: Commit GREEN**

```bash
git add narrative_dynamics/__init__.py
git commit -m "feat: freeze transfer phase a api"
```

## Phase A Final Verification Gate

This gate creates no implementation commit and does not authorize a push, catalog search, real source retrieval, lock, or FINAL activity.

- [ ] Confirm the diff contains only the Phase A files listed in the File Responsibility Map and this approved plan/spec lineage:

```bash
git diff --name-status 0378a40e934b3b241d6883df709aa056caec5f69...HEAD
git log --oneline --reverse 0378a40e934b3b241d6883df709aa056caec5f69..HEAD
```

- [ ] Confirm every implementation task has an adjacent test-only RED commit followed by its GREEN commit, and no commit from the provisional branch is an ancestor:

```bash
git merge-base --is-ancestor b32e9c9ac5e4a9f7771125bc25ff8d7770f3c9a3 HEAD
```

Expected: exit status `1`.

- [ ] Run the full Python suite without writing bytecode:

```bash
python3 -B -m unittest discover -s tests -v
```

Expected: all tests PASS.

- [ ] Run exact Lean proof gates:

```bash
lake build
lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean conformance/lean_reference_vectors.json
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Expected: every command exits `0`.

- [ ] Run formatting, privacy, source, workflow, and frozen-model scans:

```bash
git diff --check 0378a40e934b3b241d6883df709aa056caec5f69...HEAD
python3 -B -m unittest tests.test_cross_dataset_ci_boundary tests.test_cross_dataset_public_api -v
git diff --exit-code 0378a40e934b3b241d6883df709aa056caec5f69 -- narrative_dynamics/adapters/narrative_two_stage.py .github/workflows
```

Expected: no diff-check findings, boundary tests PASS, and frozen paths have no diff.

- [ ] Stop and request explicit authorization before any push or GitHub exact-head proof. After authorized proof is GREEN, stop again for approval of the separate Catalog and Source Adapter Plan.

## Completion Criteria

Phase A is ready for exact-head proof only when all 13 task pairs exist as fresh RED→GREEN commits, the full local verification gate is GREEN, the worktree is clean, the provisional branch remains untouched, and the diff contains no real dataset, source-specific adapter, workflow, lock, or execution artifact.

Phase A completion establishes only that the governed machinery works on synthetic fixtures. It does not establish that an eligible public dataset exists, that harmonization succeeds for any real source, that any candidate transfers, or that FINAL execution is authorized.
