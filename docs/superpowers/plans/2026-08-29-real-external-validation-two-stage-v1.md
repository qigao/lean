# Real External Validation Study V1 — Feher da Silva & Hare Two-Stage Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not skip RED evidence, atomic commits, or exact-head verification.

**Goal:** Implement the approved Real External Validation Study V1 so the frozen Feher da Silva & Hare two-stage behavioral source can be transformed reproducibly, participant-split without leakage, evaluated by the existing Reactive / Intentional / Planning runtime families, externally preregistered and witnessed, and scored on unseen participants with one sealed FINAL prediction artifact consumed by both Brier and Log without promoting predictive fit to latent-cognition identification.

**Architecture:** Add a study-specific source/transform/orchestration layer under `narrative_dynamics.studies`, a task adapter under `narrative_dynamics.adapters`, and one additive generic `external_prediction` seam that executes FINAL models once and recreates existing released-comparison semantics from attested predictions. Existing P3 evidence declarations, protocol releases, witness receipts, predictive statuses, separation logic, and final report remain authoritative. `external_validation.py` gains only an additive assembly function for already-scored child reports; the existing runner-based API remains backwards compatible.

**Tech Stack:** Python stdlib (`csv`, `hashlib`, `json`, `math`, `pathlib`, `subprocess`, `dataclasses`, `enum`, `statistics`, `unittest`), existing `narrative_dynamics` runtime/P3 APIs, Git repositories for source identity, GitHub Actions `.github/workflows/proof.yml`, and existing Lean/lake gates for regression verification only.

**Spec:** `docs/superpowers/specs/2026-08-29-real-external-validation-two-stage-v1-design.md`

**Approved spec commit:** `639ed8db68909dfca2f45cb7558a3cf328849542`

**Integrated research base:** `proof/narrative-dynamics-v0@e529167eddc07efd6b2bbd90039b89ed1cbeb03d`

## Global Constraints

- [ ] Create implementation branch `work/real-external-validation-two-stage-v1` from the committed plan head, not directly from the old base.
- [ ] The first implementation commit is a complete **test-only RED**. No production symbol for this study is added before that RED is committed and its intended failure is observed.
- [ ] No Lean source changes.
- [ ] No Docker acceptance. The authoritative repository workflow remains `.github/workflows/proof.yml`.
- [ ] Ordinary CI never downloads, vendors, or reads the real human-data checkout.
- [ ] Raw human CSV rows are never committed to `qigao/lean`.
- [ ] Frozen upstream identity is exactly `carolfs/muddled_models@4567763780a2c596fd6510af720ec468a8214a8f`.
- [ ] Scientific evidence is Magic Carpet `results/magic_carpet/choices/*_game.csv` plus Spaceship `results/spaceship/choices/*.csv` excluding `*_practice.csv`; matching `*_config.txt` / `*_info.txt` are transform/provenance metadata only.
- [ ] `slow == 1` is the only behavioral trial-quality exclusion; it is applied before target/history construction. Malformed included rows fail the complete transform closed.
- [ ] Split unit is `(task_variant, source_participant_id)` and all participant trials/history remain in one partition.
- [ ] Deterministic split is task-stratified 60/20/20 with stable-hash ordering and deterministic largest-remainder apportionment. The pinned inventory must resolve to 24 Magic Carpet + 21 Spaceship = 45 eligible participants and 27/9/9 TRAIN/SELECTION/FINAL.
- [ ] Study-specific `participant_assignment_hash` remains distinct from existing P3 record-level `ExternalEvidenceDeclaration.partition_assignment_hash`.
- [ ] Primary target is the first-stage choice only. Current-trial transition/final-state/second-stage choice/reward/latent reward probabilities never enter current model-visible inputs.
- [ ] No participant-specific or task-specific fitted parameters.
- [ ] TRAIN and SELECTION use Brier only; Log is a locked FINAL confirmatory score.
- [ ] Frozen parameter grids are exactly those in the approved spec.
- [ ] Frozen seed plans are TRAIN `(101, 102)`, SELECTION `(201, 202)`, FINAL `(301, 302)`.
- [ ] Metric floor is exactly `1e-12`, followed by simplex renormalization, and its implementation identity is protocol-bound.
- [ ] Strict-better-than-uniform mean thresholds are `math.nextafter(0.5, -math.inf)` for Brier and `math.nextafter(math.log(2.0), -math.inf)` for Log. Do not change generic P3 comparator semantics.
- [ ] Pairwise separation requires same score direction and deltas `>= 0.005` Brier and `>= 0.006931471805599453` Log.
- [ ] `constraint_plans=()` for Study V1.
- [ ] FINAL model predictions execute exactly once and are sealed before either score is computed. Brier, Log, task strata, and stay/switch diagnostic consume that same artifact and cannot call the runner.
- [ ] One immutable OSF registration may witness both sibling releases, but create two existing `WitnessReceipt` values, one per release. Do not modify `WitnessReceipt` schema.
- [ ] Canonical claim scope remains exactly `external_observational_predictive_only`. Do not import or expose P2 `IdentificationStatus` in the study path.
- [ ] A disappointing FINAL result is a valid study result, never a test failure or retry reason.
- [ ] No code or source-manifest commit may occur after the repository revision is frozen into the real preregistration bundle. If code changes after that point, create a new study revision and new OSF registration.

## Planned Files

**Production — create**

- `narrative_dynamics/studies/__init__.py`
- `narrative_dynamics/studies/two_stage_source.py`
- `narrative_dynamics/studies/two_stage_transform.py`
- `narrative_dynamics/studies/two_stage_osf.py`
- `narrative_dynamics/studies/feher_hare_two_stage_v1.py`
- `narrative_dynamics/adapters/narrative_two_stage.py`
- `narrative_dynamics/adapters/two_stage_metrics.py`
- `narrative_dynamics/external_prediction.py`
- generated metadata-only `narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json`

**Production — modify**

- `narrative_dynamics/contracts.py:100-130` — add `ExperimentStage.EXTERNAL_PREDICTION`
- `narrative_dynamics/external_validation.py:1170-1310` — additive assembly from already-scored released reports; keep `evaluate_external_final()` backwards compatible
- `narrative_dynamics/adapters/__init__.py` — additive task-adapter exports if the existing package convention uses this file
- `narrative_dynamics/__init__.py` — additive generic external-prediction exports only; keep study-specific APIs under `narrative_dynamics.studies`

**Tests — create**

- `tests/two_stage_test_support.py`
- `tests/test_two_stage_source.py`
- `tests/test_two_stage_transform.py`
- `tests/test_two_stage_metrics.py`
- `tests/test_narrative_two_stage_adapter.py`
- `tests/test_external_prediction.py`
- `tests/test_feher_hare_two_stage_protocol.py`
- `tests/test_two_stage_osf_witness.py`
- `tests/test_two_stage_secondary_diagnostic.py`
- `tests/test_two_stage_final_attempts.py`
- `tests/test_feher_hare_two_stage_pipeline.py`

**Existing regression tests used directly**

- `tests/test_external_validation_preregistration.py`
- `tests/test_external_validation_release_gate.py`
- `tests/test_external_validation_final.py`
- `tests/test_external_validation_reporting.py`
- `tests/test_narrative_held_out_model_comparison.py`
- `tests/test_preregistered_model_comparison.py`
- `tests/test_protocol_release.py`

Normative signatures below define public boundaries; implementation bodies follow the behavior specified in each step and its RED tests.

---

## Task 1 — Commit the complete test-only Study V1 RED boundary

**Files:** create all eleven new test/support files listed above. Do not modify production files.

- [ ] **1.1 Add synthetic raw-source builders in `tests/two_stage_test_support.py`.**

Use actual frozen source schemas, but only synthetic rows. Keep helpers non-collected by `unittest` naming.

```python
MAGIC_HEADER = (
    "trial,common,reward.1.1,reward.1.2,reward.2.1,reward.2.2,"
    "isymbol_lft,isymbol_rgt,rt1,choice1,final_state,fsymbol_lft,"
    "fsymbol_rgt,rt2,choice2,reward,slow"
)
SPACESHIP_HEADER = (
    "trial,rwrd_prob0,rwrd_prob1,rwrd_prob2,rwrd_prob3,symbol0,symbol1,"
    "common,choice1,rt1,final_state,choice2,rt2,reward,slow"
)
```

Provide deterministic helpers that create a temporary git checkout, synthetic participant files, synthetic config/info metadata, a matching test-only source manifest, and simple counting model sources. Synthetic source metadata must be unmistakably test-only; never use real participant rows.

- [ ] **1.2 Add source-boundary RED tests in `tests/test_two_stage_source.py`.**

Lock:

```text
test_source_manifest_is_content_hashed_and_order_canonical
test_snapshot_requires_exact_repository_revision
test_snapshot_requires_every_declared_path_and_git_blob_identity
test_unlisted_main_task_file_cannot_silently_enter_scientific_evidence
test_magic_game_and_spaceship_main_file_patterns_are_distinct
test_metadata_only_source_identity_is_not_an_eligible_participant
test_verification_uses_local_checkout_and_never_network
```

Guard imports so the first RED is readable while modules are missing.

- [ ] **1.3 Add transform/split/leakage RED tests in `tests/test_two_stage_transform.py`.**

Lock:

```text
test_magic_carpet_and_spaceship_rows_normalize_to_one_canonical_trial_schema
test_slow_trials_are_removed_before_history_construction
test_malformed_included_row_fails_entire_transform
test_spaceship_target_decoding_can_use_current_transition_only_outside_model_input
test_same_trial_postchoice_mutations_do_not_change_scenario_hash
test_current_latent_reward_probability_mutation_does_not_change_scenario_hash
test_prior_retained_reward_mutation_changes_history_identity
test_participant_split_is_task_stratified_deterministic_and_disjoint
test_pinned_24_21_inventory_apportions_to_14_5_5_and_13_4_4
test_participant_assignment_hash_changes_on_any_role_change
test_observation_dataset_has_positive_external_origin_and_record_level_partitions
```

Use archived encoding contracts in synthetic vectors: Magic Carpet `choice1` is 1/2; Spaceship canonical first-stage option uses the source loader rule `final_state + 1 if common else 2 - final_state` strictly outside model input.

- [ ] **1.4 Add metric/scoring RED tests in `tests/test_two_stage_metrics.py`.**

Lock extractor schema, exact floor, raw-policy immutability, uniform Brier/Log references, downward-adjacent mean thresholds, and 1% separation margins.

- [ ] **1.5 Add family-boundary RED tests in `tests/test_narrative_two_stage_adapter.py`.**

Lock fresh source lifecycle, unified dispatch/no protocol imports, exact Reactive latest-cue rule, Intentional full reward-history + Beta(1,1) smoothing without planning hooks, shared Intentional beta, Planning fixed 0.7/0.3 transition and second-stage reward beliefs, fixed discount 1.0, and seed-invariant predicted policy with seed-bound lineage.

- [ ] **1.6 Add generic single-pass prediction RED tests in `tests/test_external_prediction.py`.**

Lock:

```text
test_external_prediction_stage_exists
test_dual_protocol_preflight_happens_before_any_model_execution
test_final_prediction_artifact_executes_each_model_case_seed_once
test_prediction_artifact_binds_preflight_both_protocols_target_candidate_metric_seed_and_run_lineage
test_brier_and_log_scoring_consume_same_artifact_without_runner
test_precomputed_scoring_matches_direct_comparator_losses_on_synthetic_fixture
test_artifact_or_protocol_identity_drift_is_rejected_before_scoring
test_external_final_can_be_assembled_from_precomputed_released_reports
test_precomputed_child_release_or_verification_drift_is_rejected
```

Use counting sources and assert call count is unchanged by both score calls and P3 assembly.

- [ ] **1.7 Add Study protocol RED tests in `tests/test_feher_hare_two_stage_protocol.py`.**

Lock exact global parameter grids, Brier-only TRAIN/SELECTION, Log sibling without reselection, exact seed plans, Planning bookkeeping baseline with all pairwise comparisons, exact task strata, strict thresholds, positive external evidence, and `constraint_plans=()`.

- [ ] **1.8 Add OSF witness RED tests in `tests/test_two_stage_osf_witness.py`.**

Lock immutable bundle identity, two receipts against one registration, same bundle digest, distinct per-release subject hashes, and no `WitnessReceipt` schema change.

- [ ] **1.9 Add secondary-diagnostic RED tests in `tests/test_two_stage_secondary_diagnostic.py`.**

Lock adjacent retained trial pairs, previous reward × previous transition cells, observed stay rate, predicted stay probability derived from sealed current-trial policy, task strata, and zero runner calls.

- [ ] **1.10 Add FINAL-attempt RED tests in `tests/test_two_stage_final_attempts.py`.**

Lock:

```text
test_preflight_failure_creates_no_started_attempt
test_first_model_execution_starts_attempt
test_infrastructure_failure_is_append_only_and_exact_retry_requires_all_scientific_identities
test_scientific_failure_marks_revision_required
test_completed_attempt_cannot_be_overwritten_or_reopened
test_disappointing_result_is_completed_not_revision_required
```

- [ ] **1.11 Add complete synthetic study RED in `tests/test_feher_hare_two_stage_pipeline.py`.**

Exercise synthetic source -> transform -> split -> dataset -> TRAIN -> SELECTION -> dual protocols/releases/fake OSF witness -> dual preflight -> one FINAL artifact -> Brier + Log -> P3 aggregate report -> secondary diagnostic -> completed attempt. Pipeline correctness must not depend on a preferred real-study winner.

- [ ] **1.12 Run the complete focused RED.**

```bash
python3 -m unittest \
  tests.test_two_stage_source \
  tests.test_two_stage_transform \
  tests.test_two_stage_metrics \
  tests.test_narrative_two_stage_adapter \
  tests.test_external_prediction \
  tests.test_feher_hare_two_stage_protocol \
  tests.test_two_stage_osf_witness \
  tests.test_two_stage_secondary_diagnostic \
  tests.test_two_stage_final_attempts \
  tests.test_feher_hare_two_stage_pipeline -v
```

Expected: failures are only missing Study V1 modules/symbols and missing `ExperimentStage.EXTERNAL_PREDICTION`; existing P3 imports stay healthy.

- [ ] **1.13 Commit the test-only RED atomically.**

```bash
git add tests/two_stage_test_support.py tests/test_two_stage_*.py \
  tests/test_narrative_two_stage_adapter.py tests/test_external_prediction.py \
  tests/test_feher_hare_two_stage_protocol.py tests/test_feher_hare_two_stage_pipeline.py
git commit -m "test: require real external two-stage validation v1"
```

Do not add production files.

---

## Task 2 — Implement manifest-pinned local source verification

**Files:** create `narrative_dynamics/studies/__init__.py`, `narrative_dynamics/studies/two_stage_source.py`; green `tests/test_two_stage_source.py`.

- [ ] **2.1 Confirm focused RED.**

```bash
python3 -m unittest tests.test_two_stage_source -v
```

- [ ] **2.2 Implement canonical source contracts.**

```python
@dataclass(frozen=True)
class TwoStageSourceFile:
    path: str
    git_blob_sha: str
    task_variant: str
    source_participant_id: str | None
    purpose: str

@dataclass(frozen=True)
class TwoStageSourceManifest:
    name: str
    version: str
    repository: str
    revision: str
    license_reference: str
    files: tuple[TwoStageSourceFile, ...]

@dataclass(frozen=True)
class VerifiedTwoStageSnapshot:
    root: str
    manifest_hash: str
    repository_revision: str
    source_snapshot_hash: str
    verified_file_identities: tuple[tuple[str, str], ...]
```

Validate task variants exactly `magic_carpet`/`spaceship`, purposes exactly `scientific_evidence`/`transform_metadata`, blob IDs as 40 lowercase hex chars, canonical unique paths, and participant identities. Generic synthetic manifests may use test repository identities; the Feher-Hare freezer in Task 12 pins the real repository/revision.

- [ ] **2.3 Implement local verification without network.**

Use `git -C <root> rev-parse HEAD` and Git blob hashing `sha1(b"blob <len>\0" + bytes)`. Reject wrong HEAD, missing/changed files, duplicate paths, invalid evidence filename patterns, and any unmanifested main-task file matching an approved scientific pattern. No HTTP/GitHub calls.

- [ ] **2.4 Add JSON load/write with declared stable hash.**

Public functions:

```python
def load_two_stage_source_manifest(path: Path) -> TwoStageSourceManifest: ...
def write_two_stage_source_manifest(path: Path, manifest: TwoStageSourceManifest) -> None: ...
def verify_two_stage_snapshot(root: Path, manifest: TwoStageSourceManifest) -> VerifiedTwoStageSnapshot: ...
```

`source_snapshot_hash` is the stable hash of the verified manifest identity, which includes exact upstream revision and every included blob identity.

- [ ] **2.5 Run GREEN/regression and commit.**

```bash
python3 -m unittest tests.test_two_stage_source tests.test_external_evidence -v
git add narrative_dynamics/studies/__init__.py narrative_dynamics/studies/two_stage_source.py
git commit -m "feat: add pinned two-stage source verification"
```

---

## Task 3 — Implement causal transform, structural eligibility, deterministic split, and ObservationDataset construction

**Files:** create `narrative_dynamics/studies/two_stage_transform.py`; green `tests/test_two_stage_transform.py`.

- [ ] **3.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_transform -v
```

- [ ] **3.2 Separate pre-choice input from post-choice outcome in the type boundary.**

```python
@dataclass(frozen=True)
class TwoStagePreChoiceView:
    task_variant: str
    source_participant_id: str
    trial_id: int
    first_stage_configuration: tuple[tuple[str, object], ...]

@dataclass(frozen=True)
class TwoStageObservedOutcome:
    first_stage_action: str
    transition_common: bool
    final_state: str
    second_stage_action: str
    reward: int

@dataclass(frozen=True)
class CanonicalTwoStageTrial:
    pre_choice: TwoStagePreChoiceView
    outcome: TwoStageObservedOutcome
    source_path: str
    audit_latent_reward_probabilities: tuple[float, ...]
```

The current Scenario builder accepts only current `pre_choice` plus prior retained outcomes; it never accepts the current outcome object.

- [ ] **3.3 Implement exact raw parsers and target decoding.**

Magic required columns are exactly the archived header:

```text
trial,common,reward.1.1,reward.1.2,reward.2.1,reward.2.2,isymbol_lft,isymbol_rgt,rt1,choice1,final_state,fsymbol_lft,fsymbol_rgt,rt2,choice2,reward,slow
```

Magic `choice1` is 1/2 -> `action_0`/`action_1`. Parse/validate matching config text such as `Common transitions: 1 -> blue -> (5, 6); 2 -> pink -> (3, 4);` as transform metadata.

Spaceship required columns are exactly:

```text
trial,rwrd_prob0,rwrd_prob1,rwrd_prob2,rwrd_prob3,symbol0,symbol1,common,choice1,rt1,final_state,choice2,rt2,reward,slow
```

Observed first-stage canonical option is decoded outside predictor input:

```python
relative = final_state + 1 if common else 2 - final_state
canonical = "action_0" if relative == 1 else "action_1"
```

Validate binary `common/reward/slow`, legal state/action ranges, finite required numbers, and strictly ordered unique trial IDs. Any malformed included main-task row fails the whole transform.

- [ ] **3.4 Apply slow exclusion before retained history and structural eligibility.**

`TwoStageTransformReport` binds source manifest/snapshot hashes, transform implementation identity, structural exclusions, slow counts, retained counts, and canonical retained trials. A source participant is eligible only with an included main-task file and at least one retained scorable trial.

- [ ] **3.5 Implement deterministic participant assignment.**

```python
@dataclass(frozen=True)
class TwoStageParticipantSplitPlan:
    namespace: str = "feher-hare-two-stage-v1"
    version: str = "1"
    ratios: tuple[tuple[str, int], ...] = (
        ("train", 60),
        ("selection_validation", 20),
        ("final_test", 20),
    )

@dataclass(frozen=True)
class TwoStageParticipantAssignment:
    assignments: tuple[tuple[str, str, str], ...]
```

Order within each task by `stable_content_hash((namespace, version, task, participant))`. Use deterministic largest-remainder apportionment with tie-break order TRAIN, SELECTION_VALIDATION, FINAL_TEST. `content_hash` covers the complete task/participant/role mapping. Pinned 24/21 counts must yield 14/5/5 and 13/4/4.

- [ ] **3.6 Build record-oriented `ObservationDataset`.**

Each retained trial gets one unique record id `task/participant/trial`, causal pre-choice Scenario, one-hot `action_0/action_1` counts, task/participant/trial metadata, causal-history hash, and transform lineage. Dataset source/provenance use the existing positive external markers. Transform identity binds `participant_assignment_hash`, source identities, slow-rule version, choice-normalization version, information-firewall version, and implementation identity. Do not alter generic P3 record-level assignment hashing.

- [ ] **3.7 Run GREEN/regression and commit.**

```bash
python3 -m unittest tests.test_two_stage_transform tests.test_observation_dataset tests.test_observation_targets -v
git add narrative_dynamics/studies/two_stage_transform.py
git commit -m "feat: add causal two-stage observational transform"
```

---

## Task 4 — Implement shared first-stage metric extractor, probability floor, losses, and thresholds

**Files:** create `narrative_dynamics/adapters/two_stage_metrics.py`; green `tests/test_two_stage_metrics.py`.

- [ ] **4.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_metrics -v
```

- [ ] **4.2 Implement exact metric boundary.**

Freeze `FIRST_STAGE_PROBABILITY_FLOOR = 1e-12`; require raw policy keys exactly `action_0/action_1`, finite probabilities in `[0,1]`, and raw sum 1. Apply floor then renormalize without mutating trace. Return exactly `first_stage.action_0` / `first_stage.action_1`. Set extractor version `1.0.0` so callable identity binds the implementation/version.

- [ ] **4.3 Add target/loss/threshold factories.**

```python
def two_stage_brier_thresholds() -> AdequacyThresholds:
    return AdequacyThresholds(math.nextafter(0.5, -math.inf), 2.0)

def two_stage_log_thresholds() -> AdequacyThresholds:
    return AdequacyThresholds(
        math.nextafter(math.log(2.0), -math.inf),
        -math.log(1e-12 / (1.0 + 1e-12)),
    )
```

Factories also return the binary `CategoricalTargetSpec`, one-group categorical Brier, and one-group categorical Log loss. Separation constants stay study-specific.

- [ ] **4.4 Run GREEN/regression and commit.**

```bash
python3 -m unittest tests.test_two_stage_metrics tests.test_preregistered_model_comparison -v
git add narrative_dynamics/adapters/two_stage_metrics.py
git commit -m "feat: add two-stage first-choice metrics"
```

---

## Task 5 — Implement the two-stage Reactive / Intentional / Planning adapter over unified dispatch

**Files:** create `narrative_dynamics/adapters/narrative_two_stage.py`; optionally modify `narrative_dynamics/adapters/__init__.py`; green `tests/test_narrative_two_stage_adapter.py`.

- [ ] **5.1 Confirm RED.**

```bash
python3 -m unittest tests.test_narrative_two_stage_adapter -v
```

- [ ] **5.2 Implement a fresh-per-batch source wrapper matching the existing narrative-prison pattern.**

Public builders:

```python
def create_narrative_two_stage_reactive_source() -> NarrativeTwoStageModelSource: ...
def create_narrative_two_stage_intentional_source() -> NarrativeTwoStageModelSource: ...
def create_narrative_two_stage_planning_source() -> NarrativeTwoStageModelSource: ...
```

Every run goes through `run_runtime_decision()`. Ban imports of observations/P3 orchestration and family-specific runtime entry points.

- [ ] **5.3 Build common canonical domain/story/history replay.**

Use first-stage actions `action_0/action_1`, second-stage states `state_0/state_1`, and prior retained outcomes as evidence before the current decision time. Current observed target/outcome must be absent.

- [ ] **5.4 Implement Reactive exactly.**

Empty history -> equal scores. Latest retained rewarded trial -> favor repeating prior first-stage action. Latest retained unrewarded -> favor switching. Ignore earlier history and previous transition identity. Fitted parameters exactly `{beta}`.

- [ ] **5.5 Implement Intentional exactly.**

With memory decay `d`, immediately previous retained trial weight is `d^0`, then `d^1`, etc. For each first-stage action, direct reward belief is `(1 + weighted_rewards)/(2 + weighted_observations)`. Feed those beliefs through genuine existing belief -> goal -> choice runtime objects. Bind `beta_goal = beta_action = beta`; fitted parameters exactly `{beta, memory_decay}`; no transition lookahead.

- [ ] **5.6 Implement Planning exactly.**

Estimate second-stage `(state, action)` rewards with the same decay/Beta smoothing. Fix common/rare probabilities 0.7/0.3 and discount 1.0. Use existing planning hidden-state/transition/reward/value hooks through unified dispatch. Fitted parameters exactly `{beta, memory_decay}`.

- [ ] **5.7 Expose `first_stage_policy`, `selected_action`, `runtime_dispatch`, and `family` in every run outcome; preserve implementation identities.**

- [ ] **5.8 Run GREEN/regressions and commit.**

```bash
python3 -m unittest tests.test_narrative_two_stage_adapter tests.test_narrative_held_out_model_comparison -v
git add narrative_dynamics/adapters/narrative_two_stage.py narrative_dynamics/adapters/__init__.py
git commit -m "feat: add narrative two-stage model adapter"
```

If `adapters/__init__.py` is not used by existing package convention, do not modify it and omit it from `git add`.

---

## Task 6 — Implement Brier-only TRAIN/SELECTION freezing and dual P3 protocol construction

**Files:** create `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green `tests/test_feher_hare_two_stage_protocol.py`.

- [ ] **6.1 Confirm RED.**

```bash
python3 -m unittest tests.test_feher_hare_two_stage_protocol -v
```

- [ ] **6.2 Freeze study constants.**

```python
UPSTREAM_REPOSITORY = "carolfs/muddled_models"
UPSTREAM_REVISION = "4567763780a2c596fd6510af720ec468a8214a8f"
TRAIN_SEEDS = (101, 102)
SELECTION_SEEDS = (201, 202)
FINAL_SEEDS = (301, 302)
REACTIVE_GRID = {"beta": (0.5, 1.0, 2.0, 4.0)}
HISTORY_GRID = {
    "beta": (0.5, 1.0, 2.0, 4.0),
    "memory_decay": (0.5, 0.75, 0.9, 1.0),
}
```

- [ ] **6.3 Implement prepared-study/frozen-model/protocol bundle dataclasses.**

Prepared study holds transform report, participant assignment, dataset, external evidence declaration, and train/selection/final target reports. Frozen-model bundle retains training and selection manifest hashes. Protocol bundle contains Brier protocol, Log protocol, and `ExternalValidationPreregistration`.

- [ ] **6.4 Fit and select each family with Brier only.**

Use `fit_training_target_grid` on TRAIN with exact grids/seeds, create `ParameterAcceptanceSet` with training lineage, then `select_on_validation_suite` on SELECTION with exact seeds and Brier. Freeze one global candidate per family with `FrozenModelSpec.from_selection()`. Tests use a raising Log test double to prove Log is absent from TRAIN/SELECTION.

- [ ] **6.5 Build both sibling protocols from the same frozen candidate objects.**

Planning is comparator bookkeeping baseline only. Both protocols share dataset, target spec, final target, extractor, FINAL seeds, candidate identities/params/selection lineage, and version. Score-specific differences are only loss identity/threshold/name hash. Build exact Magic/Spaceship FINAL strata, separation rule deltas, `constraint_plans=()`, and method lineage.

- [ ] **6.6 Run GREEN/P3 sibling regression and commit.**

```bash
python3 -m unittest tests.test_feher_hare_two_stage_protocol tests.test_external_validation_preregistration -v
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py
git commit -m "feat: freeze two-stage external study protocol"
```

---

## Task 7 — Implement immutable OSF preregistration bundle and existing-receipt verifier

**Files:** create `narrative_dynamics/studies/two_stage_osf.py`; green `tests/test_two_stage_osf_witness.py`.

- [ ] **7.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_osf_witness -v
```

- [ ] **7.2 Implement canonical bundle.**

`TwoStageOSFBundle` binds source manifest/snapshot, transform, participant assignment, dataset/final target, frozen candidate hashes, both protocol hashes, external preregistration hash, both release hashes, exact repository revision, claim scope, and canonical scientific contract covering exclusions/split/target/firewall/grids/seeds/floor/thresholds/strata/separation/claim language. Its `content_hash` is derived, never caller-supplied.

- [ ] **7.3 Implement immutable proof adapter without network or release-schema change.**

`OSFRegistrationProof` contains registration reference, registration time, and bundle hash. `OSFRegistrationVerifier` implements existing `verify(release, receipt) -> bool`. Receipt proof must carry the same registration reference/bundle hash and the receipt subject must equal the supplied release hash. Create two receipts against one proof, one per sibling release. Ordinary CI uses a fake immutable reference; actual registration happens only after exact-head freeze in Task 12.

- [ ] **7.4 Run GREEN/release regressions and commit.**

```bash
python3 -m unittest tests.test_two_stage_osf_witness tests.test_protocol_release -v
git add narrative_dynamics/studies/two_stage_osf.py
git commit -m "feat: add osf witness bundle for two-stage study"
```

---

## Task 8 — Add generic single-pass FINAL prediction artifact and precomputed released scoring

**Files:** create `narrative_dynamics/external_prediction.py`; modify `narrative_dynamics/contracts.py:100-130`, `narrative_dynamics/__init__.py`; green `tests/test_external_prediction.py`.

- [ ] **8.1 Confirm RED.**

```bash
python3 -m unittest tests.test_external_prediction -v
```

- [ ] **8.2 Add exactly `ExperimentStage.EXTERNAL_PREDICTION = "external_prediction"`.**

No other manifest stage and no Lean change.

- [ ] **8.3 Implement immutable prediction records.**

```python
@dataclass(frozen=True)
class ExternalSeedPrediction:
    case_name: str
    scenario_id: str
    seed: int
    metrics: tuple[tuple[str, float], ...]
    run_manifest_hash: str

@dataclass(frozen=True)
class ExternalModelPrediction:
    model_name: str
    frozen_model_hash: str
    parameters: tuple[tuple[str, float], ...]
    predictions: tuple[ExternalSeedPrediction, ...]

@dataclass(frozen=True)
class ExternalFinalPredictionArtifact:
    preregistration_hash: str
    preflight_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    repository_revision: str
    dataset_hash: str
    final_partition_hash: str
    final_target_hash: str
    metric_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    model_predictions: tuple[ExternalModelPrediction, ...]
    manifest: ExperimentManifest
```

`content_hash` is a property derived from all identity fields plus manifest hash. Validate exact model/case/seed coverage and metric schema.

- [ ] **8.4 Implement `predict_external_final_once()`.**

Inputs are runner, P3 preregistration/preflight, both sibling protocols, exact `ComparisonModel` tuple, final targets, extractor, and repository revision. Before execution verify sibling protocol hashes, final target, extractor, seeds, candidate identities/params and runtime component identities. Then call `runner.run_once()` exactly once per model/case/seed, apply extractor, store metric vector + run manifest hash. Artifact manifest parents every run manifest and records all scientific identities. Do not compute Brier or Log.

- [ ] **8.5 Implement `score_external_prediction_artifact()` with no runner argument.**

Validate artifact against one sibling protocol, verified release, and final target. For each model/case, average each metric coordinate across exact protocol seeds with `statistics.fmean`, compute case loss with existing `evaluate_metric_loss`, and construct existing `HeldOutCaseEvaluation`, `HeldOutValidationReport`, `FinalTestReport`, `ModelComparisonEntry`, `ModelComparisonReport`, and `ReleasedModelComparisonReport`. Preserve existing ranking and `<=` threshold semantics. Child comparison lineage must include the sealed prediction artifact manifest, so downstream report ancestry proves both scores share one prediction source.

- [ ] **8.6 Prove parity against current direct comparator on a deterministic synthetic fixture.**

Compare per-case metrics, mean/worst losses, adequacy, ranking, release/verification identities. Manifest hashes need not match because lineage intentionally differs.

- [ ] **8.7 Run GREEN/regression and commit.**

```bash
python3 -m unittest tests.test_external_prediction tests.test_preregistered_model_comparison tests.test_protocol_release -v
git add narrative_dynamics/external_prediction.py narrative_dynamics/contracts.py narrative_dynamics/__init__.py
git commit -m "feat: add sealed external final predictions"
```

---

## Task 9 — Add P3 assembly from precomputed sibling reports without changing existing semantics

**Files:** modify `narrative_dynamics/external_validation.py:1170-1310`; use already-created RED tests in `tests/test_external_prediction.py`; add only backwards-compat regression to `tests/test_external_validation_final.py` if absent.

- [ ] **9.1 Re-run the assembly RED from Task 1.**

```bash
python3 -m unittest tests.test_external_prediction tests.test_external_validation_final -v
```

Expected: sealed prediction/scoring is green after Task 8; additive P3 assembly tests remain RED until this task.

- [ ] **9.2 Implement `assemble_external_final_from_reports()`.**

Inputs are preregistration/evidence, both protocols/releases/verifications, both precomputed `ReleasedModelComparisonReport`s, and final targets. Call `preflight_external_releases()` first. Require each child report's release hash, verification hash, protocol/comparison manifest identity, candidate names, final target, and loss role to match its sibling. Reuse existing `_external_adequacy_findings`, `_external_separation_findings`, and `_external_stratum_scores` unchanged to construct `ExternalFinalEvaluation`.

Keep existing `evaluate_external_final()` runner-based behavior unchanged and public.

- [ ] **9.3 Run complete P3 external regressions and commit.**

```bash
python3 -m unittest discover -s tests -p 'test_external_validation_*.py' -v
python3 -m unittest tests.test_external_evidence tests.test_external_prediction -v
git add narrative_dynamics/external_validation.py tests/test_external_validation_final.py
git commit -m "feat: assemble external validation from sealed predictions"
```

If no existing test file change is necessary, commit only the production file.

---

## Task 10 — Implement secondary stay/switch diagnostic and append-only FINAL attempts

**Files:** extend `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green `tests/test_two_stage_secondary_diagnostic.py`, `tests/test_two_stage_final_attempts.py`.

- [ ] **10.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_secondary_diagnostic tests.test_two_stage_final_attempts -v
```

- [ ] **10.2 Implement diagnostic from sealed predictions only.**

`TwoStageStaySwitchCell` binds task, previous reward, previous common/rare transition, count, observed stay rate, and per-model mean predicted stay probability. `TwoStageStaySwitchDiagnostic` binds prediction artifact hash and canonical cells; `content_hash` is derived. Pair consecutive retained FINAL trials only. Predicted stay probability is the sealed current-trial probability assigned to the previous retained first-stage action. Never call runner/model.

- [ ] **10.3 Implement immutable append-only attempt ledger.**

Statuses are exactly `started`, `completed`, `infrastructure_failed`, `revision_required`. Attempt identity binds preregistration/preflight/repository/dataset/final-target/candidate/release hashes, start time, status, failure class, run manifest hashes, result hash. Pure transition functions return new ledger values and reject overwriting/reopening. `require_exact_retry()` compares every scientific identity in the spec.

- [ ] **10.4 Run GREEN and commit.**

```bash
python3 -m unittest tests.test_two_stage_secondary_diagnostic tests.test_two_stage_final_attempts -v
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py
git commit -m "feat: add two-stage final diagnostics and attempt lineage"
```

---

## Task 11 — Implement locked final orchestration and full synthetic end-to-end GREEN

**Files:** extend `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green `tests/test_feher_hare_two_stage_pipeline.py`.

- [ ] **11.1 Confirm RED.**

```bash
python3 -m unittest tests.test_feher_hare_two_stage_pipeline -v
```

- [ ] **11.2 Implement `run_locked_feher_hare_final()`.**

Return a frozen result bundle containing updated attempt ledger, sealed prediction artifact, Brier/Log released reports, `ExternalFinalEvaluation`, secondary diagnostic, and `ExternalValidationReport`.

Execution order is normative:

1. dual release preflight + final target/runtime-model identity checks; zero runner calls and no started attempt on failure;
2. append STARTED immediately before first FINAL model execution;
3. `predict_external_final_once()`;
4. precomputed Brier score;
5. precomputed Log score;
6. `assemble_external_final_from_reports()`;
7. secondary diagnostic;
8. `build_external_validation_report(... constraint_findings=())`;
9. `attest_report(report).require_integrity()`;
10. append COMPLETED with run/result hashes.

Use an explicit `FinalInfrastructureError` wrapper for environment-only execution failures. Catch only that type as `infrastructure_failed`. Any scientific/identity/model/transform/scoring defect after STARTED marks `revision_required` and re-raises. Never map predictive adequacy/separation outcome to failure.

- [ ] **11.3 Make full synthetic study pass and prove one-pass execution.**

Assert both score reports descend from the same prediction artifact, no runner calls occur during scoring/strata/diagnostic, report is predictive-only and attested, constraints are empty, and valid execution ends COMPLETED regardless of winner.

- [ ] **11.4 Run focused study suite and commit.**

```bash
python3 -m unittest \
  tests.test_two_stage_source tests.test_two_stage_transform \
  tests.test_two_stage_metrics tests.test_narrative_two_stage_adapter \
  tests.test_external_prediction tests.test_feher_hare_two_stage_protocol \
  tests.test_two_stage_osf_witness tests.test_two_stage_secondary_diagnostic \
  tests.test_two_stage_final_attempts tests.test_feher_hare_two_stage_pipeline -v
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py tests/test_feher_hare_two_stage_pipeline.py
git commit -m "feat: add locked feher hare external final workflow"
```

---

## Task 12 — Freeze real source manifest, establish exact implementation head, then prepare real preregistration through SELECTION only

**Files:** generate metadata-only `narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json`; extend source/study module only if the manifest freezer/conformance API is missing. Do **not** run real FINAL in this task.

This task requires an explicitly supplied local checkout:

```bash
export FEHER_HARE_SOURCE_ROOT=/absolute/path/to/muddled_models
```

It must be at exact upstream revision `4567763780a2c596fd6510af720ec468a8214a8f`.

- [ ] **12.1 Implement deterministic real-source manifest freezer and generate the metadata-only manifest.**

Expose `freeze_feher_hare_v1_source_manifest(root: Path) -> TwoStageSourceManifest`. It first verifies exact upstream HEAD, then includes only approved main-task evidence files plus matching config/info metadata for participants with main-task files. Store path + Git blob identity only, never raw row content.

Generate `narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json`, inspect it manually for metadata-only content, and run `verify_two_stage_snapshot()` against the local checkout.

- [ ] **12.2 Run real source/transform/split conformance without TRAIN/SELECTION yet.**

Required assertions from the pinned checkout:

```text
Magic Carpet eligible = 24
Spaceship eligible = 21
Total eligible = 45
Magic split = 14 / 5 / 5
Spaceship split = 13 / 4 / 4
Total split = 27 / 9 / 9
```

Record hashes/counts only: metadata-only structural exclusions, slow counts, retained counts, source snapshot hash, transform hash, participant assignment hash, dataset hash, final-target hash. Do not persist raw rows.

- [ ] **12.3 Commit source manifest before freezing the study repository revision.**

```bash
git add narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json \
  narrative_dynamics/studies/two_stage_source.py narrative_dynamics/studies/__init__.py
git commit -m "research: freeze feher hare two-stage source manifest"
```

If only the generated manifest changed, keep this commit manifest-only.

- [ ] **12.4 Run complete repository verification on the would-be preregistration head.**

```bash
python3 -m unittest discover -s tests -v
lake build
python3 -m unittest discover -s tests -p 'test_external_validation_*.py' -v
python3 -m unittest tests.test_external_prediction tests.test_feher_hare_two_stage_pipeline \
  tests.test_two_stage_transform tests.test_narrative_two_stage_adapter -v
```

Expected: all green. Ordinary test commands require no real checkout.

- [ ] **12.5 Final scope review, then push and establish exact-head CI evidence.**

Confirm no raw CSV/questionnaires, no Lean changes, no P2 identification change, no participant/task-specific fitted parameters, no second FINAL pass, and no latent-cognition claim. Then:

```bash
git rev-parse HEAD
git push -u origin work/real-external-validation-two-stage-v1
```

Record the exact SHA. The authoritative `.github/workflows/proof.yml` run must report that same `head_sha`; reject stale runs. Do not run Docker acceptance.

After this point, **do not change code or the source manifest** if this repository revision is going to be used in the real preregistration bundle. Any later scientific/code change requires a new exact head and repeats this verification step.

- [ ] **12.6 At that exact verified repository head, run real TRAIN and SELECTION only.**

Using the pinned local checkout and committed source manifest, run Brier-only TRAIN/SELECTION, record the three frozen parameter tuples and manifest hashes, then build `ExternalEvidenceDeclaration`, Brier protocol, Log protocol, `ExternalValidationPreregistration`, and both `ProtocolRelease` values. Do not call `predict_external_final_once()` and do not compute any FINAL behavioral summary.

The release `source_revision.repository_revision` must equal the exact verified implementation SHA from 12.5.

- [ ] **12.7 Build the canonical OSF bundle from that exact head and frozen TRAIN/SELECTION outputs.**

The bundle must bind the exact repository SHA, committed source manifest, source/transform/split/dataset/target hashes, frozen model/selection identities, both protocol/release hashes, thresholds, floor, task strata, separation rule, and claim language. Serialize the bundle for external registration without raw human rows.

- [ ] **12.8 Human external-witness gate — last step before any real FINAL.**

Submit that exact bundle to a public immutable OSF Registration. Record immutable registration reference/time/bundle digest, create two existing `WitnessReceipt`s, verify both, and run `preflight_external_releases()` with zero FINAL model executions.

If anything in code, source manifest, transform, candidates, protocols, releases, or bundle changes after registration, the old registration is no longer valid for the changed study: create a new study revision/registration.

Real FINAL is intentionally not an implementation/CI acceptance test. It is a later explicit locked research execution using `run_locked_feher_hare_final()` after this witness gate.

---

## Implementation Completion Criteria

Implementation is code-complete when the complete synthetic Study V1 suite and repository regressions are GREEN on one exact head, source/transform/split identities are fail-closed and content-hashed, three family adapters preserve generic runtime boundaries, Brier-only TRAIN/SELECTION freezes one global tuple per family, OSF bundle/receipt plumbing uses existing release schemas, FINAL prediction is single-pass and sealed, both score reports plus secondary diagnostic consume that artifact without runner calls, P3 report attestation remains valid and predictive-only, and attempt/retry lineage is append-only.

The empirical study is preregistration-ready only after Task 12 has generated/committed the metadata-only source manifest, established exact-head CI evidence, run real-source conformance plus TRAIN/SELECTION at that unchanged head, built both releases and the canonical bundle, and completed the immutable OSF witness gate. A real FINAL outcome is not required to merge the implementation and must never be an ordinary CI expectation.
