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
- generated metadata-only `narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json` after real-source conformance

**Production — modify**

- `narrative_dynamics/contracts.py:100-130` — add `ExperimentStage.EXTERNAL_PREDICTION`
- `narrative_dynamics/external_validation.py:1170-1310` — additive assembly from already-scored released reports; keep `evaluate_external_final()` backwards compatible
- `narrative_dynamics/adapters/__init__.py` — additive task-adapter exports if package convention requires them
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

# Use temporary git repositories with deterministic participant files.
def build_synthetic_two_stage_checkout(root: Path, *, magic_n=10, spaceship_n=10): ...
def build_synthetic_source_manifest(root: Path): ...
def build_two_stage_final_fixture(): ...
```

Synthetic source metadata must be unmistakably test-only; never use real participant rows.

- [ ] **1.2 Add source-boundary RED tests in `tests/test_two_stage_source.py`.**

Lock these names and contracts:

```text
test_source_manifest_is_content_hashed_and_order_canonical
test_snapshot_requires_exact_repository_revision
test_snapshot_requires_every_declared_path_and_git_blob_identity
test_unlisted_main_task_file_cannot_silently_enter_scientific_evidence
test_magic_game_and_spaceship_main_file_patterns_are_distinct
test_metadata_only_source_identity_is_not_an_eligible_participant
test_verification_uses_local_checkout_and_never_network
```

Imports must be guarded so the first RED stays readable:

```python
try:
    from narrative_dynamics.studies.two_stage_source import (
        TwoStageSourceManifest,
        verify_two_stage_snapshot,
    )
except ImportError as error:
    IMPORT_ERROR = error
```

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

Use the archived encoding contracts in test vectors: Magic Carpet `choice1` is 1/2 and its config expresses common first-stage transition mappings; Spaceship raw first-stage choice is decoded to relative option semantics with the source rule:

```python
def expected_spaceship_relative_choice(common: int, final_state: int) -> int:
    return final_state + 1 if common else 2 - final_state
```

- [ ] **1.4 Add metric/scoring RED tests in `tests/test_two_stage_metrics.py`.**

Lock:

```text
test_extractor_returns_exact_binary_metric_schema
test_probability_floor_is_exactly_1e_minus_12_and_renormalizes
test_extractor_does_not_mutate_raw_runtime_policy
test_uniform_binary_brier_is_exactly_point_5
test_uniform_binary_log_is_log_2
test_strict_adequacy_thresholds_are_downward_adjacent_representable_values
test_separation_margins_are_one_percent_of_uniform_references
```

- [ ] **1.5 Add family-boundary RED tests in `tests/test_narrative_two_stage_adapter.py`.**

Lock:

```text
test_three_sources_are_fresh_and_bind_family_identity
test_adapter_uses_unified_dispatch_and_not_observation_or_external_protocol_modules
test_reactive_is_neutral_with_empty_history
test_reactive_uses_latest_rewarded_stay_and_unrewarded_switch_only
test_reactive_ignores_earlier_history_and_previous_transition_identity
test_intentional_uses_full_reward_history_and_beta1_smoothing_without_planning_hooks
test_intentional_binds_beta_goal_and_beta_action_to_same_beta
test_planning_uses_fixed_point_7_transition_and_second_stage_reward_beliefs
test_planning_discount_is_fixed_one
test_seed_changes_lineage_but_not_policy_for_deterministic_adapter
```

- [ ] **1.6 Add generic single-pass prediction RED tests in `tests/test_external_prediction.py`.**

Lock:

```text
test_external_prediction_stage_exists
test_dual_protocol_preflight_happens_before_any_model_execution
test_final_prediction_artifact_executes_each_model_case_seed_once
test_prediction_artifact_binds_preflight_protocol_target_candidate_metric_seed_and_run_lineage
test_brier_and_log_scoring_consume_same_artifact_without_runner
test_precomputed_scoring_matches_direct_comparator_losses_on_synthetic_fixture
test_artifact_or_protocol_identity_drift_is_rejected_before_scoring
```

Use counting model sources. Assert call count after artifact sealing and prove both score calls leave it unchanged.

- [ ] **1.7 Add study protocol RED tests in `tests/test_feher_hare_two_stage_protocol.py`.**

Lock Brier-only fitting/selection, global parameter grids, fixed seeds, Planning bookkeeping baseline, dual sibling identity, task strata, strict thresholds, positive external evidence, and `constraint_plans=()`.

```text
test_parameter_grids_are_exact_and_global_across_tasks
test_training_and_selection_use_brier_only
test_log_protocol_freezes_the_same_selected_candidates_without_reselection
test_seed_plans_are_exact
test_planning_is_bookkeeping_baseline_but_all_pairs_are_scored
test_magic_and_spaceship_final_cases_form_exact_task_strata
test_external_preregistration_has_no_constraint_plans
```

- [ ] **1.8 Add OSF witness RED tests in `tests/test_two_stage_osf_witness.py`.**

Lock immutable bundle identity, two receipts against one registration, bundle digest equality, per-release subject hashes, and no `WitnessReceipt` schema changes.

- [ ] **1.9 Add secondary-diagnostic RED tests in `tests/test_two_stage_secondary_diagnostic.py`.**

Lock adjacent retained trial-pair construction, previous reward × transition cells, observed stay rate, predicted stay probability derived from sealed current-trial policy, task strata, and zero runner calls.

- [ ] **1.10 Add FINAL-attempt RED tests in `tests/test_two_stage_final_attempts.py`.**

Lock append-only statuses and retry rules:

```text
test_preflight_failure_creates_no_started_attempt
test_first_model_execution_starts_attempt
test_infrastructure_failure_is_append_only_and_exact_retry_requires_all_scientific_identities
test_scientific_failure_marks_revision_required
test_completed_attempt_cannot_be_overwritten_or_reopened
test_disappointing_result_is_completed_not_revision_required
```

- [ ] **1.11 Add complete synthetic study RED in `tests/test_feher_hare_two_stage_pipeline.py`.**

Exercise synthetic source -> transform -> split -> dataset -> TRAIN -> SELECTION -> dual protocols/releases/fake OSF witness -> dual preflight -> one FINAL artifact -> Brier + Log -> P3 aggregate report -> secondary diagnostic -> completed attempt. The test may choose a mathematically convenient synthetic winner, but the assertion of pipeline correctness must not require a particular real-study winner.

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

Expected: failures are only missing Study V1 modules/symbols and missing `ExperimentStage.EXTERNAL_PREDICTION`. Existing P3 imports remain healthy.

- [ ] **1.13 Commit the test-only RED atomically.**

```bash
git add tests/two_stage_test_support.py tests/test_two_stage_*.py \
  tests/test_narrative_two_stage_adapter.py tests/test_external_prediction.py \
  tests/test_feher_hare_two_stage_protocol.py tests/test_feher_hare_two_stage_pipeline.py
git commit -m "test: require real external two-stage validation v1"
```

Do not add production files to this commit.

---

## Task 2 — Implement manifest-pinned local source verification

**Files:** create `narrative_dynamics/studies/__init__.py`, `narrative_dynamics/studies/two_stage_source.py`; green `tests/test_two_stage_source.py`.

- [ ] **2.1 Confirm Task 1 source tests are RED on exact head.**

```bash
python3 -m unittest tests.test_two_stage_source -v
```

Expected: missing `narrative_dynamics.studies.two_stage_source` / required source symbols.

- [ ] **2.2 Implement canonical source types.**

Use these public contracts:

```python
@dataclass(frozen=True)
class TwoStageSourceFile:
    path: str
    git_blob_sha: str
    task_variant: str
    source_participant_id: str | None
    purpose: str  # "scientific_evidence" | "transform_metadata"

@dataclass(frozen=True)
class TwoStageSourceManifest:
    name: str
    version: str
    repository: str
    revision: str
    license_reference: str
    files: tuple[TwoStageSourceFile, ...]

    def identity_payload(self) -> dict[str, object]: ...
    @property
    def content_hash(self) -> str: ...

@dataclass(frozen=True)
class VerifiedTwoStageSnapshot:
    root: str
    manifest_hash: str
    repository_revision: str
    source_snapshot_hash: str
    verified_file_identities: tuple[tuple[str, str], ...]
```

Validate task variants exactly `magic_carpet` / `spaceship`, purposes exactly the two values above, Git blob IDs as 40 lowercase hex chars, canonical path ordering, unique paths, and source participant identities.

- [ ] **2.3 Implement local checkout verification with no network.**

```python
def _git_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()

def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()

def verify_two_stage_snapshot(
    root: Path,
    manifest: TwoStageSourceManifest,
) -> VerifiedTwoStageSnapshot: ...
```

Reject wrong HEAD, missing file, changed blob, duplicate/unknown scientific path pattern, and an unmanifested main-task file that would otherwise enter evidence. Do not call HTTP/GitHub APIs.

- [ ] **2.4 Add JSON manifest load/write helpers used later by the real conformance lane.**

```python
def load_two_stage_source_manifest(path: Path) -> TwoStageSourceManifest: ...
def write_two_stage_source_manifest(path: Path, manifest: TwoStageSourceManifest) -> None: ...
```

Serialized payload includes a declared `content_hash` and rejects inconsistent hash on load.

- [ ] **2.5 Run GREEN and regression.**

```bash
python3 -m unittest tests.test_two_stage_source -v
python3 -m unittest tests.test_external_evidence -v
```

Expected: both pass.

- [ ] **2.6 Commit.**

```bash
git add narrative_dynamics/studies/__init__.py narrative_dynamics/studies/two_stage_source.py
git commit -m "feat: add pinned two-stage source verification"
```

---

## Task 3 — Implement causal transform, structural eligibility, deterministic participant split, and ObservationDataset construction

**Files:** create `narrative_dynamics/studies/two_stage_transform.py`; green `tests/test_two_stage_transform.py`.

- [ ] **3.1 Confirm focused RED.**

```bash
python3 -m unittest tests.test_two_stage_transform -v
```

Expected: missing transform/split symbols.

- [ ] **3.2 Implement separated pre-choice and post-choice raw semantics so future leakage is structurally difficult.**

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

The Scenario builder receives `pre_choice` and prior retained `outcome` objects only; it never receives the current outcome object.

- [ ] **3.3 Implement exact raw schema parsers.**

Magic required columns:

```text
trial, common, reward.1.1, reward.1.2, reward.2.1, reward.2.2,
isymbol_lft, isymbol_rgt, rt1, choice1, final_state,
fsymbol_lft, fsymbol_rgt, rt2, choice2, reward, slow
```

Magic `choice1` is 1/2 and maps deterministically to `action_0` / `action_1`. Parse the matching config line `Common transitions: <first action> -> <label> -> (...); ...` as transform metadata and validate it before using participant rows.

Spaceship required columns:

```text
trial, rwrd_prob0, rwrd_prob1, rwrd_prob2, rwrd_prob3,
symbol0, symbol1, common, choice1, rt1, final_state,
choice2, rt2, reward, slow
```

Canonical first-stage observed option is decoded outside predictor input:

```python
relative = final_state + 1 if common else 2 - final_state
canonical = "action_0" if relative == 1 else "action_1"
```

Validate binary `common/reward/slow`, legal state/action ranges, finite numeric fields, strictly ordered unique trial IDs, and fail the entire transform on any invalid retained source row.

- [ ] **3.4 Apply `slow` exclusion before history construction.**

Create `TwoStageTransformReport` retaining source participant counts, structural exclusions, slow counts, retained counts, canonical trials, source manifest/snapshot hashes, and a transform implementation identity. Metadata-only IDs never become participants.

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
    assignments: tuple[tuple[str, str, str], ...]  # task, participant, role
    @property
    def content_hash(self) -> str: ...

def assign_two_stage_participants(
    report: TwoStageTransformReport,
    plan: TwoStageParticipantSplitPlan,
) -> TwoStageParticipantAssignment: ...
```

Within each task, order by `stable_content_hash((namespace, version, task, participant))`. Implement deterministic largest-remainder apportionment with role tie-break order TRAIN, SELECTION_VALIDATION, FINAL_TEST. Assert the pinned 24/21 counts yield 14/5/5 and 13/4/4.

- [ ] **3.6 Build record-oriented `ObservationDataset`.**

```python
def build_two_stage_observation_dataset(
    report: TwoStageTransformReport,
    assignment: TwoStageParticipantAssignment,
) -> ObservationDataset: ...
```

For retained trial `t`, Scenario payload includes task, current pre-choice configuration, and a canonical tuple of prior retained observable outcomes `< t`. Counts are exactly one-hot `{"action_0": 1, "action_1": 0}` or reverse. Metadata includes task, participant, source trial ID, causal-history hash, and transform lineage.

Dataset origin is exactly:

```python
source={"kind": "external_observational", ...}
provenance={"external_observational": True, ...}
```

Transform identity binds `participant_assignment_hash`, source manifest/snapshot identity, slow-rule version, target-normalization version, information-firewall version, and measured implementation identity.

- [ ] **3.7 Run GREEN and mutation tests.**

```bash
python3 -m unittest tests.test_two_stage_transform -v
python3 -m unittest tests.test_observation_dataset tests.test_observation_targets -v
```

Expected: pass.

- [ ] **3.8 Commit.**

```bash
git add narrative_dynamics/studies/two_stage_transform.py
git commit -m "feat: add causal two-stage observational transform"
```

---

## Task 4 — Implement shared first-stage metric extractor, probability floor, losses, and frozen thresholds

**Files:** create `narrative_dynamics/adapters/two_stage_metrics.py`; green `tests/test_two_stage_metrics.py`.

- [ ] **4.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_metrics -v
```

- [ ] **4.2 Implement exact metric boundary.**

```python
FIRST_STAGE_PROBABILITY_FLOOR = 1e-12
FIRST_STAGE_METRIC_KEYS = (
    "first_stage.action_0",
    "first_stage.action_1",
)

def two_stage_first_stage_policy_metrics(trace) -> dict[str, float]:
    raw = trace.outcome["first_stage_policy"]
    values = {key: float(raw[action]) for key, action in (
        ("first_stage.action_0", "action_0"),
        ("first_stage.action_1", "action_1"),
    )}
    floored = {key: max(value, FIRST_STAGE_PROBABILITY_FLOOR) for key, value in values.items()}
    total = math.fsum(floored.values())
    return {key: value / total for key, value in floored.items()}

two_stage_first_stage_policy_metrics.version = "1.0.0"
```

Validate raw policy schema, finite `[0,1]` probabilities, and raw simplex before flooring. Never mutate trace outcome.

- [ ] **4.3 Add target/loss/threshold factories.**

```python
def two_stage_target_spec() -> CategoricalTargetSpec: ...
def two_stage_brier_loss() -> CategoricalBrierLoss: ...
def two_stage_log_loss() -> CategoricalLogLoss: ...
def two_stage_brier_thresholds() -> AdequacyThresholds:
    return AdequacyThresholds(math.nextafter(0.5, -math.inf), 2.0)
def two_stage_log_thresholds() -> AdequacyThresholds:
    return AdequacyThresholds(
        math.nextafter(math.log(2.0), -math.inf),
        -math.log(FIRST_STAGE_PROBABILITY_FLOOR / (1.0 + FIRST_STAGE_PROBABILITY_FLOOR)),
    )
```

Freeze separation constants `0.005` and `0.006931471805599453` in the study protocol builder, not generic losses.

- [ ] **4.4 Run GREEN.**

```bash
python3 -m unittest tests.test_two_stage_metrics tests.test_model_adequacy_losses -v
```

- [ ] **4.5 Commit.**

```bash
git add narrative_dynamics/adapters/two_stage_metrics.py
git commit -m "feat: add two-stage first-choice metrics"
```

---

## Task 5 — Implement the two-stage Reactive / Intentional / Planning task adapter over unified dispatch

**Files:** create `narrative_dynamics/adapters/narrative_two_stage.py`; optionally modify `narrative_dynamics/adapters/__init__.py`; green `tests/test_narrative_two_stage_adapter.py`.

- [ ] **5.1 Confirm RED.**

```bash
python3 -m unittest tests.test_narrative_two_stage_adapter -v
```

- [ ] **5.2 Implement one source wrapper pattern matching `narrative_prison.py`.**

```python
@dataclass(frozen=True)
class NarrativeTwoStageModelSource:
    family: str
    name: str
    version: str = "1.0.0"
    lifecycle: str = "fresh_per_batch"

    def instantiate(self): ...
    def simulate(self, scenario, parameters, rng): ...

def create_narrative_two_stage_reactive_source() -> NarrativeTwoStageModelSource: ...
def create_narrative_two_stage_intentional_source() -> NarrativeTwoStageModelSource: ...
def create_narrative_two_stage_planning_source() -> NarrativeTwoStageModelSource: ...
```

All execution goes through `run_runtime_decision()`. The module must not import `narrative_dynamics.observations`, `narrative_dynamics.external_validation`, `run_runtime_reactive_decision`, `run_runtime_intentional_decision`, or `run_runtime_planning_decision`.

- [ ] **5.3 Implement common domain/story/history replay.**

Build one canonical participant decision story with first-stage actions `action_0/action_1`, second-stage states `state_0/state_1`, and retained past outcomes represented as narrative evidence before the current decision logical time. Current target/outcome remains absent.

- [ ] **5.4 Implement Reactive exactly.**

- empty history: equal action scores;
- latest retained rewarded trial: score latest first-stage action `+1`, other `0`;
- latest retained unrewarded trial: score latest first-stage action `0`, other `+1`;
- ignore all earlier history and previous common/rare identity;
- fitted parameter set exactly `{beta}`.

- [ ] **5.5 Implement Intentional exactly.**

For action `a`, with decay `d`, compute weighted direct reward belief from prior retained first-stage action outcomes:

```python
estimated_p[a] = (1.0 + weighted_rewards[a]) / (2.0 + weighted_observations[a])
```

Use these beliefs through actual existing runtime belief -> goal -> choice objects. Bind `beta_goal == beta_action == parameters["beta"]`; fitted parameters exactly `{beta, memory_decay}`. Do not use transition identity in lookahead.

- [ ] **5.6 Implement Planning exactly.**

Estimate second-stage `(state, action)` reward probabilities with the same decay and Beta(1,1) smoothing. Fix common transition `0.7`, rare `0.3`, discount `1.0`. Use existing planning hidden-state/transition/reward/value hooks and unified dispatch. Fitted parameters exactly `{beta, memory_decay}`.

- [ ] **5.7 Expose `first_stage_policy` and lineage in every run outcome.**

The trace outcome contains at least:

```python
{
    "first_stage_policy": {"action_0": ..., "action_1": ...},
    "selected_action": ...,
    "runtime_dispatch": ...,
    "family": ...,
}
```

- [ ] **5.8 Run GREEN plus generic-family regressions.**

```bash
python3 -m unittest \
  tests.test_narrative_two_stage_adapter \
  tests.test_narrative_held_out_model_comparison \
  tests.test_narrative_runtime_decision_dispatch -v
```

- [ ] **5.9 Commit.**

```bash
git add narrative_dynamics/adapters/narrative_two_stage.py narrative_dynamics/adapters/__init__.py
git commit -m "feat: add narrative two-stage model adapter"
```

---

## Task 6 — Implement Study V1 TRAIN/SELECTION freezing and dual P3 protocol construction

**Files:** create `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green protocol subset of `tests/test_feher_hare_two_stage_protocol.py`.

- [ ] **6.1 Confirm RED.**

```bash
python3 -m unittest tests.test_feher_hare_two_stage_protocol -v
```

- [ ] **6.2 Freeze study constants and parameter grids.**

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

- [ ] **6.3 Implement prepared-data and selection bundles.**

```python
@dataclass(frozen=True)
class FeherHarePreparedStudy:
    transform_report: TwoStageTransformReport
    assignment: TwoStageParticipantAssignment
    dataset: ObservationDataset
    evidence: ExternalEvidenceDeclaration
    train_targets: TargetConstructionReport
    selection_targets: TargetConstructionReport
    final_targets: TargetConstructionReport

@dataclass(frozen=True)
class FeherHareFrozenModels:
    frozen_candidates: tuple[FrozenModelSpec, ...]
    training_manifest_hashes: tuple[str, ...]
    selection_manifest_hashes: tuple[str, ...]
```

`prepare_feher_hare_two_stage_v1()` verifies snapshot, transforms, assigns participants, builds dataset/targets, and creates `ExternalEvidenceDeclaration.from_dataset()`. Its transform identity includes the study participant-assignment hash while generic P3 computes record-level assignment itself.

- [ ] **6.4 Fit and select with Brier only.**

For each family:

1. `fit_training_target_grid(... loss=two_stage_brier_loss(), simulation_seeds=TRAIN_SEEDS)`;
2. wrap candidate tuples in `ParameterAcceptanceSet` with training manifest lineage;
3. build SELECTION `HeldOutSuite` using `SELECTION_SEEDS`;
4. `select_on_validation_suite(... loss=two_stage_brier_loss())`;
5. `FrozenModelSpec.from_selection()`.

No Log call is allowed in training/selection; enforce via tests with a raising Log test double.

- [ ] **6.5 Build sibling protocols from the same frozen candidate set.**

```python
@dataclass(frozen=True)
class FeherHareProtocolBundle:
    brier_protocol: PreregisteredEvaluationProtocol
    log_protocol: PreregisteredEvaluationProtocol
    preregistration: ExternalValidationPreregistration
```

Planning is `baseline_name` only for comparator bookkeeping. Both protocols use the same final targets, extractor, `FINAL_SEEDS`, and exact frozen candidates. Use score-specific strict thresholds. Build task strata from FINAL record IDs by metadata. Freeze separation rule with exact deltas and `constraint_plans=()`.

- [ ] **6.6 Run GREEN and P3 sibling regression.**

```bash
python3 -m unittest \
  tests.test_feher_hare_two_stage_protocol \
  tests.test_external_validation_preregistration -v
```

- [ ] **6.7 Commit.**

```bash
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py
git commit -m "feat: freeze two-stage external study protocol"
```

---

## Task 7 — Implement immutable OSF preregistration bundle and existing-receipt verifier adapter

**Files:** create `narrative_dynamics/studies/two_stage_osf.py`; green `tests/test_two_stage_osf_witness.py`.

- [ ] **7.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_osf_witness -v
```

- [ ] **7.2 Implement canonical preregistration bundle.**

```python
@dataclass(frozen=True)
class TwoStageOSFBundle:
    source_manifest_hash: str
    source_snapshot_hash: str
    transform_hash: str
    participant_assignment_hash: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_protocol_hash: str
    log_protocol_hash: str
    external_preregistration_hash: str
    brier_release_hash: str
    log_release_hash: str
    repository_revision: str
    claim_scope: str
    scientific_contract: Mapping[str, object]

    def identity_payload(self) -> dict[str, object]: ...
    @property
    def content_hash(self) -> str: ...
```

`scientific_contract` must bind source/exclusion/split/target/firewall/parameter-grid/seeds/floor/thresholds/strata/separation and allowed/forbidden claim language in canonical JSON-like values.

- [ ] **7.3 Implement proof and verifier without changing `WitnessReceipt`.**

```python
@dataclass(frozen=True)
class OSFRegistrationProof:
    registration_reference: str
    registered_at: str
    bundle_hash: str

class OSFRegistrationVerifier:
    name = "osf-registration-verifier"
    version = "1"

    def __init__(self, proof: OSFRegistrationProof): ...
    def verify(self, release: ProtocolRelease, receipt: WitnessReceipt) -> bool: ...
```

Create two receipts with same registration reference/bundle hash and distinct `subject_hash` values equal to Brier vs Log release hashes. The verifier validates frozen proof fields only; it does not perform network access during ordinary CI.

- [ ] **7.4 Run GREEN and release regressions.**

```bash
python3 -m unittest tests.test_two_stage_osf_witness tests.test_protocol_release -v
```

- [ ] **7.5 Commit.**

```bash
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

Expected: missing stage/module/symbols.

- [ ] **8.2 Add exactly one generic manifest stage.**

```python
class ExperimentStage(str, Enum):
    ...
    EXTERNAL_VALIDATION = "external_validation"
    EXTERNAL_PREDICTION = "external_prediction"
```

No Lean mirror is required because this is Python research-manifest plumbing, not formal cognitive semantics.

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
    repository_revision: str
    dataset_hash: str
    final_partition_hash: str
    final_target_hash: str
    metric_identity: Mapping[str, object]
    simulation_seeds: tuple[int, ...]
    model_predictions: tuple[ExternalModelPrediction, ...]
    manifest: ExperimentManifest

    @property
    def content_hash(self) -> str: ...
```

- [ ] **8.4 Implement `predict_external_final_once()`.**

```python
def predict_external_final_once(
    *,
    runner: SimulationRunner,
    preregistration: ExternalValidationPreregistration,
    preflight: ExternalReleasePreflight,
    brier_protocol: PreregisteredEvaluationProtocol,
    log_protocol: PreregisteredEvaluationProtocol,
    models: tuple[ComparisonModel, ...],
    final_targets: TargetConstructionReport,
    extractor: object,
    repository_revision: str,
) -> ExternalFinalPredictionArtifact: ...
```

Preflight all sibling identities, target/metric/seeds/candidates, and model component identities before executing. For each frozen model, final case, and seed, call `runner.run_once()` exactly once, apply extractor, and store the metric vector plus run manifest hash. The artifact manifest uses `ExperimentStage.EXTERNAL_PREDICTION`, stores all scientific identities, and parents every run manifest hash.

Do not compute Brier or Log in this function.

- [ ] **8.5 Implement precomputed scoring with existing report types and no runner argument.**

```python
def score_external_prediction_artifact(
    *,
    artifact: ExternalFinalPredictionArtifact,
    verified_release: VerifiedProtocolRelease,
    protocol: PreregisteredEvaluationProtocol,
    target_set: TargetConstructionReport,
    loss: MetricLoss,
) -> ReleasedModelComparisonReport: ...
```

For each model/case, select artifact predictions for exactly `protocol.simulation_seeds`, average each metric coordinate with `statistics.fmean`, compute case loss using existing `evaluate_metric_loss`, and construct existing:

- `HeldOutCaseEvaluation` with preserved `run_manifest_hashes`;
- `HeldOutValidationReport`;
- `FinalTestReport`;
- `ModelComparisonEntry` / `ModelComparisonReport` using the same ranking and `<=` threshold semantics as current comparator;
- `ReleasedModelComparisonReport` with existing `RELEASED_MODEL_COMPARISON` stage.

The child model-comparison manifest must parent the sealed prediction artifact manifest so downstream P3 report lineage proves both score views originate from one prediction evidence artifact.

- [ ] **8.6 Prove semantic parity against the direct comparator.**

On a deterministic synthetic fixture, run direct Brier comparison once outside the sealed-study path and compare per-case metrics, mean/worst loss, adequacy, ranking, and release identities with `score_external_prediction_artifact()`. Do not require comparison manifest hashes to be equal because their execution lineage intentionally differs.

- [ ] **8.7 Run GREEN plus generic final-comparison regressions.**

```bash
python3 -m unittest \
  tests.test_external_prediction \
  tests.test_preregistered_model_comparison \
  tests.test_protocol_release -v
```

- [ ] **8.8 Commit.**

```bash
git add narrative_dynamics/external_prediction.py narrative_dynamics/contracts.py narrative_dynamics/__init__.py
git commit -m "feat: add sealed external final predictions"
```

---

## Task 9 — Add P3 assembly from precomputed sibling reports without changing existing P3 semantics

**Files:** modify `narrative_dynamics/external_validation.py:1170-1310`; extend `tests/test_external_prediction.py` and `tests/test_external_validation_final.py` only where needed.

- [ ] **9.1 Add RED for additive assembly.**

Add:

```text
test_external_final_can_be_assembled_from_precomputed_released_reports
test_precomputed_child_release_or_verification_drift_is_rejected
test_existing_runner_based_evaluate_external_final_remains_backwards_compatible
```

Run:

```bash
python3 -m unittest tests.test_external_prediction tests.test_external_validation_final -v
```

Expected: only new assembly symbol missing; existing P3 tests pass.

- [ ] **9.2 Implement one additive public function.**

```python
def assemble_external_final_from_reports(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    brier_report: ReleasedModelComparisonReport,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    log_report: ReleasedModelComparisonReport,
    final_targets: TargetConstructionReport,
) -> ExternalFinalEvaluation: ...
```

Call existing `preflight_external_releases()` first. Require child `release_hash`, `verification_hash`, candidate names, final target identity, and loss role to match exact sibling protocols. Then reuse existing private `_external_adequacy_findings`, `_external_separation_findings`, and `_external_stratum_scores` unchanged.

Do not change `evaluate_external_final()` behavior or existing typed findings/statuses.

- [ ] **9.3 Run all P3 external regression tests.**

```bash
python3 -m unittest discover -s tests -p 'test_external_validation_*.py' -v
python3 -m unittest tests.test_external_evidence tests.test_uncertainty_external_validation -v
```

- [ ] **9.4 Commit.**

```bash
git add narrative_dynamics/external_validation.py tests/test_external_prediction.py tests/test_external_validation_final.py
git commit -m "feat: assemble external validation from sealed predictions"
```

---

## Task 10 — Implement secondary stay/switch diagnostic and append-only FINAL attempt lineage

**Files:** extend `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green `tests/test_two_stage_secondary_diagnostic.py` and `tests/test_two_stage_final_attempts.py`.

- [ ] **10.1 Confirm RED.**

```bash
python3 -m unittest tests.test_two_stage_secondary_diagnostic tests.test_two_stage_final_attempts -v
```

- [ ] **10.2 Implement diagnostic from sealed predictions only.**

```python
@dataclass(frozen=True)
class TwoStageStaySwitchCell:
    task_variant: str
    previous_reward: int
    previous_transition_common: bool
    observation_count: int
    observed_stay_rate: float
    predicted_stay_probability_by_model: tuple[tuple[str, float], ...]

@dataclass(frozen=True)
class TwoStageStaySwitchDiagnostic:
    prediction_artifact_hash: str
    cells: tuple[TwoStageStaySwitchCell, ...]
    content_hash: str

def build_two_stage_stay_switch_diagnostic(
    *,
    dataset: ObservationDataset,
    artifact: ExternalFinalPredictionArtifact,
) -> TwoStageStaySwitchDiagnostic: ...
```

Pair consecutive retained trials within FINAL participants only. Predicted stay probability on current trial is the sealed probability assigned to previous trial's canonical first-stage action. Never call a model/runner.

- [ ] **10.3 Implement append-only attempt records.**

```python
class TwoStageFinalAttemptStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"
    REVISION_REQUIRED = "revision_required"

@dataclass(frozen=True)
class TwoStageFinalExecutionAttempt:
    attempt_id: str
    preregistration_hash: str
    preflight_hash: str
    repository_revision: str
    dataset_hash: str
    final_target_hash: str
    frozen_candidate_hashes: tuple[str, ...]
    brier_release_hash: str
    log_release_hash: str
    started_at: str
    status: TwoStageFinalAttemptStatus
    failure_class: str | None = None
    completed_run_manifest_hashes: tuple[str, ...] = ()
    result_hash: str | None = None

@dataclass(frozen=True)
class TwoStageFinalAttemptLedger:
    attempts: tuple[TwoStageFinalExecutionAttempt, ...]
    @property
    def content_hash(self) -> str: ...
```

Provide pure functions `start_final_attempt`, `mark_infrastructure_failed`, `mark_revision_required`, `complete_final_attempt`, `require_exact_retry`. They return new ledger values; no in-place mutation. Exact retry compares every frozen scientific identity listed in the spec.

- [ ] **10.4 Run GREEN.**

```bash
python3 -m unittest tests.test_two_stage_secondary_diagnostic tests.test_two_stage_final_attempts -v
```

- [ ] **10.5 Commit.**

```bash
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py
git commit -m "feat: add two-stage final diagnostics and attempt lineage"
```

---

## Task 11 — Implement locked Study V1 final orchestration and full synthetic end-to-end GREEN

**Files:** extend `narrative_dynamics/studies/feher_hare_two_stage_v1.py`; green `tests/test_feher_hare_two_stage_pipeline.py`.

- [ ] **11.1 Confirm end-to-end RED.**

```bash
python3 -m unittest tests.test_feher_hare_two_stage_pipeline -v
```

- [ ] **11.2 Implement result bundle and locked final entry point.**

```python
@dataclass(frozen=True)
class FeherHareFinalStudyResult:
    attempt_ledger: TwoStageFinalAttemptLedger
    prediction_artifact: ExternalFinalPredictionArtifact
    brier_report: ReleasedModelComparisonReport
    log_report: ReleasedModelComparisonReport
    external_evaluation: ExternalFinalEvaluation
    secondary_diagnostic: TwoStageStaySwitchDiagnostic
    report: ExternalValidationReport

class FinalInfrastructureError(RuntimeError):
    pass

def run_locked_feher_hare_final(
    *,
    runner: SimulationRunner,
    prepared: FeherHarePreparedStudy,
    protocols: FeherHareProtocolBundle,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    runtime_models: tuple[ComparisonModel, ...],
    repository_revision: str,
    attempt_ledger: TwoStageFinalAttemptLedger,
    attempt_id: str,
    started_at: str,
) -> FeherHareFinalStudyResult: ...
```

Execution order is normative:

1. dual `preflight_external_releases()` and final-target/model identity checks — no attempt started yet and zero runner calls on failure;
2. append STARTED attempt immediately before first FINAL model execution;
3. `predict_external_final_once()`;
4. `score_external_prediction_artifact()` Brier;
5. `score_external_prediction_artifact()` Log;
6. `assemble_external_final_from_reports()`;
7. `build_two_stage_stay_switch_diagnostic()`;
8. `build_external_validation_report(... constraint_findings=())`;
9. `attest_report(report).require_integrity()`;
10. append COMPLETED attempt with run/result hashes.

Catch only explicitly wrapped environmental execution failures as `FinalInfrastructureError` and append `INFRASTRUCTURE_FAILED`. Validation, transform, model-semantic, identity, protocol, or scoring defects after STARTED append `REVISION_REQUIRED` and re-raise. Never treat predictive status/outcome as an error.

- [ ] **11.3 Make the full synthetic mini-study pass.**

The test must assert:

- preflight before first runner call;
- exactly one prediction pass;
- Brier and Log reports both bind the same artifact lineage;
- no runner calls in scores/strata/diagnostic;
- external report claim scope remains predictive-only;
- `constraint_findings == ()`;
- report attestation passes;
- attempt is COMPLETED regardless of which synthetic family is best, provided execution is valid.

- [ ] **11.4 Run complete focused study suite.**

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

- [ ] **11.5 Commit.**

```bash
git add narrative_dynamics/studies/feher_hare_two_stage_v1.py tests/test_feher_hare_two_stage_pipeline.py
git commit -m "feat: add locked feher hare external final workflow"
```

---

## Task 12 — Freeze real source manifest, run real-source conformance through SELECTION only, and finish repository verification

**Files:** create generated metadata-only `narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json`; extend source/study module only if a minimal manifest-generation/conformance interface is missing. Do **not** run real FINAL in this task.

This task requires an explicitly supplied local checkout at the frozen upstream revision. Set:

```bash
export FEHER_HARE_SOURCE_ROOT=/absolute/path/to/muddled_models
```

The checkout must itself be at `4567763780a2c596fd6510af720ec468a8214a8f`.

- [ ] **12.1 Add a deterministic manifest freezer that emits metadata only.**

In `two_stage_source.py` expose:

```python
def freeze_feher_hare_v1_source_manifest(root: Path) -> TwoStageSourceManifest: ...
```

It enumerates only the approved evidence patterns plus matching metadata for participants with main-task files. It records path + Git blob SHA, not raw rows. It rejects wrong upstream HEAD before producing a manifest.

Generate and save:

```bash
python3 - <<'PY'
import os
from pathlib import Path
from narrative_dynamics.studies.two_stage_source import (
    freeze_feher_hare_v1_source_manifest,
    write_two_stage_source_manifest,
)
root = Path(os.environ["FEHER_HARE_SOURCE_ROOT"])
manifest = freeze_feher_hare_v1_source_manifest(root)
out = Path("narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json")
write_two_stage_source_manifest(out, manifest)
print(manifest.content_hash)
PY
```

Review the generated JSON to ensure it contains paths/identities only and no CSV content.

- [ ] **12.2 Verify the real frozen inventory and transform accounting.**

Run a conformance-only Python entry point or direct API call that stops before FINAL prediction and print a JSON-safe summary containing hashes/counts only. Required assertions:

```text
Magic Carpet structurally eligible participants = 24
Spaceship structurally eligible participants = 21
Total eligible = 45
Magic split = 14 TRAIN / 5 SELECTION / 5 FINAL
Spaceship split = 13 TRAIN / 4 SELECTION / 4 FINAL
Total split = 27 / 9 / 9
```

Also record structural metadata-only exclusions, slow-trial counts, retained-trial counts, source snapshot hash, transform hash, participant assignment hash, dataset hash, and final-target hash. Do not print or persist raw human rows.

- [ ] **12.3 Run real TRAIN and SELECTION only.**

Use Brier-only fitting/selection and record the three frozen parameter tuples plus training/selection manifest hashes. Build `ExternalEvidenceDeclaration`, Brier protocol, Log protocol, `ExternalValidationPreregistration`, and both `ProtocolRelease` values. Do **not** execute `predict_external_final_once()` and do not compute any FINAL behavioral summaries.

This output is the preregistration material to submit to OSF. The exact repository revision used for this preregistration must be frozen in the bundle.

- [ ] **12.4 Human external-witness gate.**

Before any real FINAL execution, submit the canonical bundle to a public immutable OSF Registration. Record the immutable registration reference, registration time, and bundle digest. Create the two existing `WitnessReceipt` values and verify both. If the bundle changes after registration, stop and create a new study revision/registration; never patch the old registration lineage.

Real FINAL is intentionally **not** an implementation/CI acceptance test. It is a later explicit locked research execution using `run_locked_feher_hare_final()` after this witness gate.

- [ ] **12.5 Run all Python regression tests.**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests pass. Real human checkout is not required by this ordinary test command.

- [ ] **12.6 Run Lean regression gates without modifying Lean sources.**

```bash
lake build
```

Expected: success.

- [ ] **12.7 Run focused P3 and study proof again.**

```bash
python3 -m unittest discover -s tests -p 'test_external_validation_*.py' -v
python3 -m unittest \
  tests.test_external_prediction \
  tests.test_feher_hare_two_stage_pipeline \
  tests.test_two_stage_transform \
  tests.test_narrative_two_stage_adapter -v
```

Expected: success.

- [ ] **12.8 Commit the generated source manifest and any final additive exports.**

```bash
git add narrative_dynamics/studies/feher_hare_two_stage_v1_source_manifest.json \
  narrative_dynamics/studies/two_stage_source.py narrative_dynamics/studies/__init__.py
git commit -m "research: freeze feher hare two-stage source manifest"
```

If the code was already complete and only the generated manifest changed, keep the commit manifest-only.

- [ ] **12.9 Push the exact implementation head and verify repository CI.**

```bash
git rev-parse HEAD
git push -u origin work/real-external-validation-two-stage-v1
```

Record the exact SHA. The authoritative GitHub Actions proof must report the same `head_sha`; do not accept a run for an earlier commit. Required green evidence is the existing proof workflow, full Python suite, Lean build/conformance gates, and the focused two-stage/P3 suite. Do not add or run Docker acceptance.

- [ ] **12.10 Final scope review before integration.**

Confirm the diff contains only the planned study/source/adapter/external-prediction files, additive P3 assembly/exports, tests, the approved design/plan, and metadata-only source manifest. Confirm no raw human CSV, no questionnaire data, no Lean changes, no P2 identification changes, no participant/task-specific fitted params, no second FINAL runner pass, and no canonical latent-cognition claim.

Commit any review-only corrections as their own atomic RED->GREEN changes and rerun exact-head verification.

---

## Implementation Completion Criteria

Implementation is code-complete only when the complete synthetic Study V1 suite and all repository regressions are GREEN on one exact head, source/transform/split identities are fail-closed and content-hashed, three family adapters preserve generic runtime boundaries, Brier-only TRAIN/SELECTION freezes one global tuple per family, OSF bundle/receipt plumbing uses existing release schemas, FINAL prediction is single-pass and sealed, both score reports plus secondary diagnostic consume that artifact without runner calls, P3 report attestation remains valid and predictive-only, and attempt/retry lineage is append-only.

The empirical study is preregistration-ready only after the separate real-source conformance lane has verified the pinned 24+21 inventory, generated the metadata-only source manifest, frozen TRAIN/SELECTION candidates and dual releases, and produced an immutable OSF bundle. A real FINAL outcome is not required to merge the implementation and must never be made an ordinary CI expectation.
