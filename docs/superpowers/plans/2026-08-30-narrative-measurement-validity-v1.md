# Narrative Measurement Validity V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Use `superpowers:test-driven-development` for every RED/GREEN pair, `superpowers:systematic-debugging` for any unexpected failure, and `superpowers:verification-before-completion` before every completion claim.

**Goal:** Build and run a bounded Measurement Validity V1 audit that tests exact representational invariance and TRAIN/SELECTION-only measurement robustness for the three already frozen Feher/Hare candidates, without training, selection, R3 FINAL access, or a new FINAL execution.

**Architecture:** Add one generic immutable measurement-audit module and one Feher/Hare composition module. A trusted provisioner may reconstruct and verify the full frozen source, but it emits a restricted TRAIN/SELECTION projection. The measurement engine predicts only the three frozen candidates, evaluates a hard exact-invariance gate, assembles aggregate robustness and diagnostic findings, and attests an aggregate-only report. A separate manual orchestration branch pins the scientific parent SHA, R3 lock commit, and upstream source commit for the one real audit.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `statistics`, `unittest`, `unittest.mock`), existing `narrative_dynamics` contracts/simulation/loss/report-attestation APIs, Git/GitHub Actions, and the existing Lean/Python exact-head proof workflow. No new Python package or external service is required.

**Spec:** `docs/superpowers/specs/2026-08-30-narrative-measurement-validity-v1-design.md`

**Global Constraints:**

- Work from the commit containing the approved design and this plan. Create `work/narrative-measurement-validity-v1`; do not modify the completed `final/feher-hare-r3` or `lock/feher-hare-r3-internal-final` branches.
- The audit claim scope is exactly `external_observational_measurement_audit_only`.
- The only empirical roles are `ObservationPartitionRole.TRAIN` and `ObservationPartitionRole.SELECTION_VALIDATION`; `FINAL_TEST` is represented only by the two excluded negative-lineage hashes.
- Never read the sealed R3 prediction/report archives, R3 Brier/Log values, diagnostic values, comparison results, or winner as analytic input.
- Never call `_grid_candidates`, `fit_training_target_grid`, `select_on_validation_suite`, acceptance-set construction, or `FrozenModelSpec.from_selection()`.
- Reconstruct the three selected candidates only with `FrozenModelSpec.freeze`, then verify their exact content hashes.
- Reuse seeds `(101, 102)` for TRAIN and `(201, 202)` for SELECTION_VALIDATION. Seeds `(301, 302)` are forbidden.
- Use the existing `two_stage_first_stage_policy_metrics`, `two_stage_brier_loss`, and `two_stage_log_loss`; do not create a second loss implementation or loosen the `1e-12` probability tolerance.
- Do not call `predict_external_final_once()` or reuse FINAL-named prediction/report types. Add measurement-specific prediction types and an `ExperimentStage.MEASUREMENT_AUDIT` manifest stage.
- Persist no raw human rows, direct participant identifiers, record-level choices, case targets, or per-case predictions. Persist only canonical hashes, aggregate rows/counts, execution-manifest hashes, report, attestation, and attempt metadata.
- Every task follows test-only RED commit, minimal GREEN commit, focused verification, then a clean `git status --short` check.
- Do not execute the real audit until ordinary exact-head CI is GREEN at the approved scientific SHA. Do not start Transfer V1 unless the exact hard gate reaches GREEN.

## Frozen identities used by every task

| Identity | Exact value |
| --- | --- |
| Completed R3 scientific revision | `d01232979cdfc9d902daab5f9e3e937079b56f69` |
| R3 lock branch | `lock/feher-hare-r3-internal-final` |
| R3 lock commit | `f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3` |
| R3 lock path | `research-locks/feher-hare-r3-internal-final.json` |
| Upstream source revision | `4567763780a2c596fd6510af720ec468a8214a8f` |
| Source manifest | `sha256:3609e980af172823cfef78290e7f2337fdb337d67f1f27641e17d473aeedd10d` |
| Source snapshot | `sha256:25bdc2e4bff4110f38b098b89e7d59aa38c9658727fcc89a38d50e107d243186` |
| Transform | `sha256:2fb8ab6dc796a9e4ece5653a5485d865ef44dd0f5f7dff92d94a904932d6e541` |
| Participant assignment | `sha256:fc144c35f6713e27f75141d678872cc04ca44c7a0fd8e109e8618c72b4111ccf` |
| Dataset | `sha256:17789130372d7eace05e1216a57bdae2ffbd519960333ffe814aee2d2d404781` |
| Target specification | `sha256:3134c9dc424418c87379f8e451420fb2defe81b8f30a45d67cd9b1f453349713` |
| TRAIN partition | `sha256:71ce56243338eb23b9dd5ad6dd901e4b0d0be4989742b66bbe7ea4f025f207da` |
| SELECTION partition | `sha256:ecdcfa1681b58888ebf4419372d5de7b75be9624cdd3307144371c5486eb1347` |
| Excluded FINAL partition | `sha256:936ffe872644e888111007b300ea847484698be8fbfd7c2d54a177505a411047` |
| Excluded FINAL target | `sha256:927e1727d37993e9a5c887f79ac8712d07a155622deacf0a3122d41f90b16ed2` |
| TRAIN/SELECTION freeze | `sha256:77fa1bb80ab0c7ac737a09a8001f1d0c1432ce3bd991fd7191d07fdd0d8e05d5` |
| Internal lock bundle | `sha256:96e553e557d6e7314eb0b9b1d0aaa8696e01d7a6ddec0abde949be8c8d45602f` |

Frozen candidates, in canonical family order:

```python
FROZEN_CANDIDATE_ROWS = (
    (
        "reactive",
        (("beta", 0.5),),
        "sha256:e8414e301d19fc6acfd5accf055402ac995790f785402186c67ff95a48ee0c2e",
        "sha256:42a84ef10c207159ff98397fe2bd86594df6c8be0f41b98e876f7ae33cba9456",
    ),
    (
        "intentional",
        (("beta", 2.0), ("memory_decay", 0.5)),
        "sha256:498a512cd931afe276f78bf135bd5c8269e051920d4b876177d0aa9ea250806d",
        "sha256:68a01b5496e20095c1a603b30f1384bbe6a3ea22aac8cf74b7b97463c4935839",
    ),
    (
        "planning",
        (("beta", 4.0), ("memory_decay", 0.75)),
        "sha256:e347a3d7d523e2a35d0bb6b0f662343f5a01ddd1ff7d2efe95a8a5efa1986e74",
        "sha256:c0ed956c1b285153075e5a22e7fb51f6d53dcf8ad326cce82e39e0221c2e877c",
    ),
)
```

## Spec coverage map

| Frozen design requirement | Implementation tasks |
| --- | --- |
| Partition-restricted information firewall | Tasks 1, 6, 9, 12, 13 |
| Categorical coordinate permutation | Tasks 2, 3, 8, 10 |
| Task canonicalization equivalence | Task 8 |
| Deterministic order and executor invariance | Tasks 7, 8 |
| Task-stratified, aggregation-unit, score-rule robustness | Task 4 |
| Deterministic leave-one-participant-out influence | Task 4 |
| Descriptive stay/switch diagnostic | Task 5 |
| Closed statuses and aggregate-only report | Tasks 1, 9 |
| Pinned protocol, real execution, durable artifact | Tasks 11, 12, 13 |
| GREEN / SCIENTIFIC RED / INFRASTRUCTURE INCOMPLETE terminal classes | Tasks 9, 12, 13 |
| Transfer V1 handoff without selecting a dataset | Task 13 |

## Task 1: Establish the branch and immutable input firewall

**Files:**

- Create: `tests/measurement_validity_fixtures.py`
- Create: `tests/test_measurement_validity_contracts.py`
- Create: `narrative_dynamics/measurement_validity.py`
- Modify: `narrative_dynamics/contracts.py`
- Modify: `narrative_dynamics/__init__.py`

**Interfaces:**

- `MEASUREMENT_CLAIM_SCOPE = "external_observational_measurement_audit_only"`
- `MeasurementScore`: `BRIER`, `LOG`
- `MeasurementAggregation`: `TRIAL_EQUAL`, `PARTICIPANT_EQUAL`
- `MeasurementValidityStatus`: the six closed scientific statuses from the spec
- `MeasurementTerminalClass`: `GREEN`, `SCIENTIFIC_RED`, `INFRASTRUCTURE_INCOMPLETE`
- `MeasurementAuditCase(role, scenario, target, task_variant, participant_group_hash, trial_index, record_hash)` with a derived `case_hash` property
- `MeasurementAuditInput(claim_scope, source_manifest_hash, source_snapshot_hash, transform_hash, participant_assignment_hash, dataset_hash, target_spec_hash, allowed_partition_hashes, allowed_target_report_hashes, allowed_case_commitments, frozen_candidates, excluded_final_partition_hash, excluded_final_target_hash, cases)`
- `MeasurementValidityProtocol(name, version, claim_scope, empirical_anchor_hash, allowed_roles, candidate_hashes, seeds_by_role, semantic_fixture_hashes, semantic_permutations, task_strata, aggregation_rules, scores, material_reversal_references, participant_influence_rule, stay_switch_definition, excluded_final_partition_hash, excluded_final_target_hash, implementation_identities)`
- `MeasurementValidityProtocol.to_payload()` and `MeasurementValidityProtocol.from_payload(payload)` with exact-field validation
- `ExperimentStage.MEASUREMENT_AUDIT = "measurement_audit"`

### Step 1.1: Create the implementation branch

Run:

```bash
design_sha="$(git rev-parse HEAD)"
git diff --quiet
git diff --cached --quiet
git switch -c work/narrative-measurement-validity-v1 "$design_sha"
git rev-parse HEAD
```

Expected: the new branch points to the exact plan-bearing commit and the worktree is clean.

### Step 1.2: Write the contract tests first

Create shared synthetic fixtures with two tasks, at least two opaque participants per task, and both allowed roles. The helper must construct scenarios using only `task_variant`, `first_stage_configuration`, and `history`; its participant values must already be SHA-256 hashes.

The contract test must include these exact boundary assertions:

```python
self.assertEqual(
    MEASUREMENT_CLAIM_SCOPE,
    "external_observational_measurement_audit_only",
)
self.assertEqual(
    {status.value for status in MeasurementValidityStatus},
    {
        "exact_invariance_met",
        "exact_invariance_failed",
        "stable_under_frozen_audit",
        "materially_measurement_dependent",
        "inconclusive_sensitivity",
        "not_established",
    },
)
self.assertEqual(
    tuple(case.role for case in audit_input.cases),
    tuple(sorted(
        (case.role for case in audit_input.cases),
        key=lambda role: role.value,
    )),
)
self.assertNotIn("participant_id", repr(audit_input.identity_payload()))
self.assertNotIn("source_participant_id", repr(audit_input.identity_payload()))
```

Also test that construction rejects:

- a `FINAL_TEST` case or seed map;
- a direct participant identifier or malformed opaque hash;
- duplicate case, record, scenario, or participant/trial identities;
- a task outside `magic_carpet`/`spaceship`;
- a target other than exactly `first_stage.action_0`/`first_stage.action_1` summing to one within `1e-12`;
- zero, two, or four frozen candidates;
- any candidate hash outside the canonical three;
- a missing TRAIN or SELECTION role;
- malformed or equal excluded FINAL identities;
- non-finite reversal references;
- duplicate semantic permutation, score, aggregation, task, role, or callable identity;
- protocol seeds containing `301` or `302`;
- unknown or missing fields in `MeasurementValidityProtocol.from_payload`;
- mutation of source mappings after dataclass construction;
- hash changes from input reordering after canonicalization.

### Step 1.3: Run the focused RED test

Run:

```bash
python3 -m unittest tests.test_measurement_validity_contracts -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `narrative_dynamics.measurement_validity`; no unrelated failure.

### Step 1.4: Commit the test-only RED

```bash
git add tests/measurement_validity_fixtures.py tests/test_measurement_validity_contracts.py
git commit -m "test: define measurement audit firewall"
```

### Step 1.5: Implement the minimal immutable contracts

In `contracts.py`, append the new enum member only; do not reorder existing stages.

In `measurement_validity.py`:

- use frozen dataclasses, tuples, `MappingProxyType`, `stable_content_hash`, and strict SHA-256 validation;
- canonicalize roles as `(TRAIN, SELECTION_VALIDATION)`, tasks as `("magic_carpet", "spaceship")`, candidates in reactive/intentional/planning order, and scores/aggregations in enum order;
- expose `identity_payload()` and `content_hash` for case, input, and protocol;
- inspect only declared fields; do not add a general dataset or partition lookup method;
- derive `MeasurementAuditCase.case_hash` from its canonical projection payload rather than accepting a human-readable record id;
- reject a scenario payload containing direct identity keys `participant_id` or `source_participant_id` at any mapping depth;
- validate each input per-role case commitment against `stable_content_hash(tuple(case.record_hash for case in role_cases))`; the provisioner must first verify the original full partition hash against the R3 anchor and only then compute this projection commitment.

Use this exact stage binding in protocol manifest construction:

```python
ExperimentManifest(
    stage=ExperimentStage.MEASUREMENT_AUDIT,
    inputs=protocol.identity_payload(),
    parent_hashes=(audit_input.content_hash,),
)
```

Export generic public types from `narrative_dynamics/__init__.py`; keep Feher/Hare-specific APIs out of the package root.

### Step 1.6: Run GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_contracts -v
git diff --check
git add narrative_dynamics/contracts.py narrative_dynamics/measurement_validity.py narrative_dynamics/__init__.py
git commit -m "feat: add measurement audit contracts"
git status --short
```

Expected: focused tests pass and status is clean.

## Task 2: Implement exact coordinate invariance and dependence classification

**Files:**

- Create: `tests/test_measurement_validity_invariance.py`
- Modify: `narrative_dynamics/measurement_validity.py`

**Interfaces:**

- `ExactInvarianceFinding(check_name, status, original_hash, transformed_hash, details_hash)`
- `MeasurementDependenceFinding(dimension, score, aggregation, task, model_pair, deltas, reference, status)`
- `permute_binary_metric_map(values)`
- `evaluate_categorical_coordinate_invariance(target, prediction)`
- `classify_material_reversal(deltas, references)`

The pairwise delta convention is frozen as `first_model_loss - second_model_loss`; negative prefers the first model and positive prefers the second.

### Step 2.1: Write pure-function RED tests

Use the existing two-stage loss objects to assert:

```python
target = {
    "first_stage.action_0": 1.0,
    "first_stage.action_1": 0.0,
}
prediction = {
    "first_stage.action_0": 0.8,
    "first_stage.action_1": 0.2,
}
permuted_target = permute_binary_metric_map(target)
permuted_prediction = permute_binary_metric_map(prediction)
self.assertEqual(
    evaluate_metric_loss(two_stage_brier_loss(), target, prediction),
    evaluate_metric_loss(
        two_stage_brier_loss(),
        permuted_target,
        permuted_prediction,
    ),
)
self.assertEqual(
    evaluate_metric_loss(two_stage_log_loss(), target, prediction),
    evaluate_metric_loss(
        two_stage_log_loss(),
        permuted_target,
        permuted_prediction,
    ),
)
```

Freeze classification vectors:

| Deltas | References | Expected status |
| --- | --- | --- |
| `(-0.006, -0.010)` | `(0.005, 0.005)` | `STABLE_UNDER_FROZEN_AUDIT` |
| `(-0.006, 0.006)` | `(0.005, 0.005)` | `MATERIALLY_MEASUREMENT_DEPENDENT` |
| `(-0.006, 0.004)` | `(0.005, 0.005)` | `INCONCLUSIVE_SENSITIVITY` |
| `(0.0, 0.0)` | `(0.005, 0.005)` | `STABLE_UNDER_FROZEN_AUDIT` |

Also reject an empty vector, unequal vector lengths, non-finite values, zero/negative references, unknown metric keys, unnormalized probabilities, and a coordinate map with more or fewer than two keys.

### Step 2.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_measurement_validity_invariance -v
git add tests/test_measurement_validity_invariance.py
git commit -m "test: define measurement invariance rules"
```

Expected: import errors for the new functions.

### Step 2.3: Implement the pure rules

Implementation rules:

- `permute_binary_metric_map` swaps only the two existing first-stage keys and returns a new canonical mapping.
- `evaluate_categorical_coordinate_invariance` checks normalization, calculates Brier and Log before/after the simultaneous swap, and emits `EXACT_INVARIANCE_MET` only when losses and inverse-mapped maps are identical.
- Coordinate-only operations use exact equality; the existing probability validator retains `1e-12` only for normalization.
- `classify_material_reversal` returns material dependence only when at least one strictly negative and one strictly positive delta both meet their corresponding positive reference; an opposing sign below either reference is inconclusive; otherwise it is stable.
- The finding hashes contain values/statuses only and no free-text conclusion.

### Step 2.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_contracts tests.test_measurement_validity_invariance -v
git diff --check
git add narrative_dynamics/measurement_validity.py
git commit -m "feat: add exact measurement invariance rules"
```

## Task 3: Add measurement-specific prediction and scoring artifacts

**Files:**

- Create: `tests/test_measurement_validity_prediction.py`
- Modify: `narrative_dynamics/measurement_validity.py`

**Interfaces:**

- `MeasurementSeedPrediction(case_hash, scenario_hash, role, model_name, seed, metrics, run_manifest_hash)`
- `MeasurementModelPrediction(model_name, candidate_hash, rows)`
- `MeasurementPredictionArtifact(protocol_hash, audit_input_hash, models, execution_manifest_hashes)`
- `MeasurementCaseLoss(case_hash, role, task_variant, participant_group_hash, model_name, score, value)`
- `score_measurement_predictions(audit_input, prediction_artifact, losses)`

### Step 3.1: Write RED tests around one sealed prediction set

Build a deterministic artifact for the synthetic cases and assert:

- every model has exactly `case_count × role_seed_count` rows;
- `(model, case, seed)` and run-manifest hashes are unique;
- rows canonicalize independently of input order;
- seed metrics are averaged per case before either loss is evaluated, matching `external_prediction.py` semantics;
- the same in-memory predictions produce both Brier and Log rows;
- score calculation rejects missing/extra cases, missing/extra seeds, an unknown model/candidate, duplicate manifests, non-finite metrics, or an artifact/input/protocol hash mismatch;
- `MeasurementPredictionArtifact` is not `ExternalFinalPredictionArtifact` and its manifest stage is not FINAL;
- artifact identity does not contain target values, participant ids, or a FINAL role.

Use this concrete averaging check:

```python
seed_metrics = (
    {"first_stage.action_0": 0.7, "first_stage.action_1": 0.3},
    {"first_stage.action_0": 0.9, "first_stage.action_1": 0.1},
)
self.assertEqual(
    average_seed_metrics(seed_metrics),
    (
        ("first_stage.action_0", 0.8),
        ("first_stage.action_1", 0.2),
    ),
)
```

### Step 3.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_measurement_validity_prediction -v
git add tests/test_measurement_validity_prediction.py
git commit -m "test: define measurement prediction artifacts"
```

### Step 3.3: Implement prediction contracts and scoring

- Keep prediction rows in memory; their `identity_payload()` may contain case hashes and metrics, but the persisted report serializer added later must not serialize them.
- Bind the artifact to input/protocol/candidate/run-manifest identities.
- Match cases by `case_hash`, never display names or direct ids.
- Use `two_stage_first_stage_policy_metrics` output unchanged.
- Call existing `evaluate_metric_loss`; identify losses with `metric_loss_identity`.
- Canonicalize `-0.0` to `0.0` using the existing `1e-15` numerical-zero convention, without changing material thresholds.

### Step 3.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_prediction tests.test_measurement_validity_invariance -v
git diff --check
git add narrative_dynamics/measurement_validity.py
git commit -m "feat: add measurement prediction scoring"
```

## Task 4: Build the frozen empirical robustness profile

**Files:**

- Create: `tests/test_measurement_validity_robustness.py`
- Modify: `narrative_dynamics/measurement_validity.py`

**Interfaces:**

- `MeasurementAggregateLoss(model_name, score, aggregation, task_variant, role, mean_loss, case_count, participant_count)`
- `MeasurementPairwiseDelta(first_model, second_model, score, aggregation, task_variant, role, delta)`
- `ParticipantInfluenceRange(role, task_variant, score, aggregation, first_model, second_model, minimum_delta, maximum_delta, omitted_participant_count, status)`
- `MeasurementRobustnessProfile(aggregate_losses, pairwise_deltas, task_findings, aggregation_findings, score_findings, participant_influence)`
- `aggregate_measurement_losses(rows, aggregation)`
- `build_measurement_robustness_profile(rows, protocol)`

### Step 4.1: Write formula-level RED tests

Freeze a reference table where unequal trial counts make the two aggregations differ:

```python
participant_losses = {
    "sha256:" + "1" * 64: (0.1, 0.1, 0.1, 0.1),
    "sha256:" + "2" * 64: (0.9,),
}
self.assertEqual(trial_equal_mean(participant_losses), 0.26)
self.assertEqual(participant_equal_mean(participant_losses), 0.5)
```

Tests must cover:

- task-stratified rows for both tasks and both roles;
- trial-equal and participant-equal formulas;
- all unordered model pairs in canonical family order;
- Brier and Log direction comparison with their own references (`0.005` and `0.006931471805599453`);
- material task reversal, material aggregation reversal, material score reversal, inconclusive sign change, and stable direction;
- deterministic leave-one-participant-out minimum/maximum ranges;
- a removal that creates a material reversal;
- one-participant support producing `NOT_ESTABLISHED`, not a division error;
- missing task strata, missing model rows, duplicate case loss rows, or non-finite values failing closed;
- identical profile hash under input row reordering.

### Step 4.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_measurement_validity_robustness -v
git add tests/test_measurement_validity_robustness.py
git commit -m "test: define measurement robustness profile"
```

### Step 4.3: Implement aggregation and influence

Apply these exact grouping keys:

```python
aggregate_key = (
    row.role.value,
    row.task_variant,
    row.score.value,
    aggregation.value,
    row.model_name,
)
pair_key = (
    role,
    task_variant,
    score,
    aggregation,
    first_model,
    second_model,
)
```

For participant-equal aggregation, average case losses within participant first, then average participant means. For each leave-one-participant-out result, remove that opaque participant's complete row set before recomputing the requested aggregation. Sort participant hashes before omission; introduce no bootstrap or random seed.

### Step 4.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_robustness tests.test_measurement_validity_prediction -v
git diff --check
git add narrative_dynamics/measurement_validity.py
git commit -m "feat: add frozen measurement robustness profile"
```

## Task 5: Add the descriptive stay/switch diagnostic

**Files:**

- Create: `tests/test_measurement_validity_diagnostic.py`
- Create: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`

**Interfaces:**

- `StaySwitchCell(task_variant, reward, transition_common, observed_stay_probability, model_expected_stay_probability, count, status)`
- `StaySwitchDiagnostic(task_variant, model_name, cells, observed_interaction, model_interaction, status)`
- `build_feher_hare_stay_switch_diagnostics(audit_input, prediction_artifact)`

### Step 5.1: Write RED diagnostic vectors

Use consecutive synthetic cases. The current case target gives the observed current action. The final row in `scenario.payload["history"]` gives prior `first_stage_action`, `transition_common`, and `reward`. For model expectation, select the current prediction probability assigned to the prior action.

Assert the frozen interaction formula exactly:

```python
interaction = (
    stay_rewarded_common
    - stay_unrewarded_common
    - stay_rewarded_rare
    + stay_unrewarded_rare
)
self.assertEqual(interaction, 0.5)
```

Cover both tasks, all three models, participant boundaries, non-consecutive retained trial ids, first retained trials without history, an empty reward/transition cell, and malformed history. Empty cells must produce a cell and summary with `NOT_ESTABLISHED`; they must never be dropped or imputed.

Also prove this API has no `SimulationRunner`, executor, parameter, candidate-selection, or threshold argument; it consumes only the frozen input and already-created predictions.

### Step 5.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_measurement_validity_diagnostic -v
git add tests/test_measurement_validity_diagnostic.py
git commit -m "test: define measurement stay switch diagnostic"
```

### Step 5.3: Implement the report-only diagnostic

- Sort cases by `(task_variant, participant_group_hash, trial_index)`.
- Use only the immediate final history entry already visible before the current choice.
- Treat missing history as no diagnostic observation, not as a stay/switch value.
- Keep observed and model-expected numerators/counts separate.
- Do not feed any diagnostic value into model ranking, hard-gate status, or dependence classification.

### Step 5.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_diagnostic -v
git diff --check
git add narrative_dynamics/studies/feher_hare_measurement_validity_v1.py
git commit -m "feat: add measurement stay switch diagnostic"
```

## Task 6: Bind the R3 anchor and trusted TRAIN/SELECTION provisioner

**Files:**

- Create: `tests/test_feher_hare_measurement_validity_input.py`
- Modify: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`

**Interfaces:**

- `FeherHareMeasurementAnchor(lock_commit, scientific_repository_revision, upstream_revision, source_manifest_hash, source_snapshot_hash, transform_hash, participant_assignment_hash, dataset_hash, target_spec_hash, train_partition_hash, selection_partition_hash, excluded_final_partition_hash, excluded_final_target_hash, train_selection_freeze_hash, internal_lock_bundle_hash, candidate_rows)`
- `FeherHareMeasurementAnchor.to_payload()` and `FeherHareMeasurementAnchor.from_payload(payload)` with exact-field validation
- `load_feher_hare_r3_measurement_anchor(lock_path, *, lock_commit)`
- `freeze_feher_hare_measurement_candidates(anchor)`
- `provision_feher_hare_measurement_input(root, manifest, anchor)`

### Step 6.1: Write anchor and firewall RED tests

Create a minimal lock JSON fixture using the exact production field paths:

- `scientific_repository_revision`
- `upstream_revision`
- `source_manifest_hash`
- `source_snapshot_hash`
- `transform_hash`
- `participant_assignment_hash`
- `dataset_hash`
- `final_target_hash`
- `train_selection_freeze.artifact_payload_digest`
- `train_selection_freeze.families`
- `frozen_candidates`
- `brier_protocol.train_partition_hash`
- `brier_protocol.selection_partition_hash`
- `brier_protocol.final_partition_hash`
- `brier_protocol.target_spec_hash`
- `internal_lock_bundle.content_hash`

Assert that loading succeeds only for lock commit `f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3` and all exact values in the frozen-identity table. Mutate each identity one at a time and assert a labelled failure. Reject unknown or missing fields in `FeherHareMeasurementAnchor.from_payload`.

For candidate reconstruction, patch `FrozenModelSpec.from_selection` to raise immediately, invoke `freeze_feher_hare_measurement_candidates`, and assert exact candidate hashes and parameters. Also patch these forbidden functions to raise:

```python
forbidden = (
    "narrative_dynamics.calibration._grid_candidates",
    "narrative_dynamics.observations.training.fit_training_target_grid",
    "narrative_dynamics.validation.select_on_validation_suite",
    "narrative_dynamics.uncertainty.ParameterAcceptanceSet.from_parameters",
    "narrative_dynamics.observations.preregistration.FrozenModelSpec.from_selection",
)
```

For provisioning, build a small synthetic `PreparedFeherHareTwoStageV1` and assert:

- only TRAIN/SELECTION target cases survive;
- direct participant ids and source paths are replaced by `stable_content_hash((task, participant))` and then discarded;
- cases keep scenario, target, task, opaque participant hash, trial index, and record hash only;
- FINAL partition/target hashes survive only as negative commitments;
- no `ObservationDataset`, `PreparedFeherHareTwoStageV1`, FINAL target report, or general partition method is reachable from the returned input;
- full dataset/assignment/transform/source identities are verified before projection;
- source preparation may touch the full verified snapshot but the projected payload has zero FINAL cases/values.

### Step 6.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_feher_hare_measurement_validity_input -v
git add tests/test_feher_hare_measurement_validity_input.py
git commit -m "test: define frozen measurement input provisioner"
```

### Step 6.3: Implement exact anchor loading and projection

Parse with `json.load`, require the expected field set used above, and cross-check duplicate identities wherever the lock repeats them. Build each candidate with:

```python
candidate = FrozenModelSpec.freeze(
    name=family,
    model=source,
    parameters=dict(parameters),
    selection_manifest_hash=selection_manifest_hash,
)
if candidate.content_hash != expected_candidate_hash:
    raise ValueError(f"{family} frozen candidate identity changed")
```

The public real provisioner must call `prepare_feher_hare_two_stage_v1`, verify all source/dataset/transform/assignment/target/partition hashes, then immediately project only the two allowed target reports. Keep a private `_project_prepared_measurement_input` helper so unit tests can exercise projection without fetching the human-data repository.

### Step 6.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_feher_hare_measurement_validity_input tests.test_measurement_validity_contracts -v
git diff --check
git add narrative_dynamics/studies/feher_hare_measurement_validity_v1.py
git commit -m "feat: provision frozen measurement audit input"
```

## Task 7: Execute only the three frozen candidates

**Files:**

- Create: `tests/test_feher_hare_measurement_prediction.py`
- Modify: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`

**Interfaces:**

- `FeherHareMeasurementPredictionTask(family, case_hash, seed)`
- `execute_feher_hare_measurement_predictions(repository_identity, audit_input, protocol, executor)`
- private spawn initializer and worker function following `feher_hare_two_stage_parallel.py`

### Step 7.1: Write selected-only execution RED tests

Use the synthetic audit input and a real `RepositoryIdentity`. Assert:

- exactly three model families execute;
- each allowed case uses only its role's two seeds;
- `SimulationRunner.run_once` is called exactly `3 × case_count × 2` times;
- no seed is `301` or `302`;
- no candidate-grid, training, selection, acceptance-set, or FINAL API is invoked;
- changing audit case order or executor batch order does not change prediction artifact hash;
- `SequentialCandidateExecutor` and `ProcessCandidateExecutor(max_workers=2)` produce identical artifact hashes on a small spawn-safe fixture;
- a worker family/candidate mismatch, missing row, duplicate row, or executor failure fails closed.

Patch `narrative_dynamics.external_prediction.predict_external_final_once` to raise and prove the measurement executor does not touch it.

### Step 7.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_feher_hare_measurement_prediction -v
git add tests/test_feher_hare_measurement_prediction.py
git commit -m "test: define selected only measurement execution"
```

### Step 7.3: Implement the deterministic executor

Follow the existing spawn pattern:

- loop once per canonical family;
- initialize a fresh worker context with the known source factory, repository identity, frozen parameters, and a mapping of restricted cases;
- create tasks only for `(case, seed)` pairs allowed by `protocol.seeds_by_role`;
- call `SimulationRunner.run_once(source, case.scenario, parameters, seed=seed)`;
- extract only `two_stage_first_stage_policy_metrics` and the run manifest hash;
- reassemble rows in canonical family/case/seed order and verify cardinality before constructing the artifact.

Do not pass `PreparedFeherHareTwoStageV1`, an observation dataset, or a partition object into a worker.

### Step 7.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_feher_hare_measurement_prediction tests.test_measurement_validity_prediction -v
git diff --check
git add narrative_dynamics/studies/feher_hare_measurement_validity_v1.py
git commit -m "feat: execute frozen measurement candidates"
```

## Task 8: Implement the semantic metamorphic hard gate

**Files:**

- Create: `tests/test_feher_hare_measurement_invariance.py`
- Modify: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`

**Interfaces:**

- `FeherHareSemanticFixture(name, original_scenario, transformed_scenario, inverse_metric_permutation)`
- `frozen_feher_hare_semantic_fixtures()`
- `evaluate_feher_hare_semantic_invariance(repository_identity, candidates)`
- `evaluate_feher_hare_empirical_invariance(audit_input, prediction_artifact, protocol)`
- `combine_feher_hare_exact_invariance(semantic_findings, empirical_findings)`

### Step 8.1: Write metamorphic RED tests

Freeze paired fixtures covering all of the following, with every affected history field permuted coherently:

- first-stage `action_0`/`action_1` relabel;
- second-stage action relabel;
- `state_0`/`state_1` relabel;
- Magic Carpet displayed positions and config identity;
- Spaceship symbol order and relative-choice reconstruction;
- current post-choice outcome mutation that leaves the model-visible pre-choice scenario unchanged;
- record-order and batch-boundary permutations;
- serial/process execution identity.

For each of Reactive, Intentional, and Planning, assert after inverse mapping:

- identical first-stage policy;
- identical Brier and Log loss;
- identical aggregate rows and hard-gate result hash;
- `EXACT_INVARIANCE_MET` only when every required fixture passes.

Using an allowed synthetic TRAIN/SELECTION prediction artifact, also assert that simultaneous target/prediction coordinate permutation, case reordering, and executor batch regrouping preserve every per-case loss, aggregate loss, pairwise delta, and canonical artifact hash after inverse mapping.

Intentionally break one retained-history action label and assert `EXACT_INVARIANCE_FAILED`, with no robustness conclusion assembled afterward.

### Step 8.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_feher_hare_measurement_invariance -v
git add tests/test_feher_hare_measurement_invariance.py
git commit -m "test: define two stage semantic invariance gate"
```

### Step 8.3: Implement frozen fixture builders and gate evaluation

- Construct fixtures entirely in code; include no human rows or upstream paths.
- Give each fixture a stable content hash and include all fixture hashes in the protocol.
- Use the same three frozen parameter tuples; fixture evaluation is method validation, not fitting.
- Run small fixture tasks under serial and process executors, canonicalize outputs, and compare artifact/report hashes.
- Evaluate categorical-coordinate and record-order invariance over the already-created allowed prediction artifact without rerunning the models.
- Return findings for categorical coordinates, task canonicalization, and deterministic order/executor invariance separately; the hard gate passes only if all are `EXACT_INVARIANCE_MET`.

### Step 8.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_feher_hare_measurement_invariance tests.test_measurement_validity_invariance -v
git diff --check
git add narrative_dynamics/studies/feher_hare_measurement_validity_v1.py
git commit -m "feat: add two stage semantic invariance gate"
```

## Task 9: Assemble, attest, and serialize the aggregate-only report

**Files:**

- Create: `tests/test_measurement_validity_reporting.py`
- Modify: `narrative_dynamics/measurement_validity.py`
- Modify: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`
- Modify: `narrative_dynamics/__init__.py`

**Interfaces:**

- `MeasurementValidityReport(terminal_class, claim_scope, protocol_hash, audit_input_hash, empirical_anchor_hash, allowed_partition_hashes, allowed_target_report_hashes, excluded_final_partition_hash, excluded_final_target_hash, candidate_hashes, prediction_artifact_hash, execution_manifest_hashes, exact_invariance_findings, robustness_profile, stay_switch_diagnostics, record_counts, participant_counts, diagnostic_cell_counts, parameter_training_performed, parameter_selection_performed, final_test_values_exposed_to_audit, final_test_outcomes_analyzed, final_model_execution, manifest)`, where `prediction_artifact_hash` and `robustness_profile` may be `None` only for a pre-execution synthetic hard-gate failure
- `MeasurementAuditAttempt(attempt_id, scientific_revision, orchestration_revision, started_at_utc, terminal_class, protocol_hash, report_hash, error_type, error_message_hash, artifact_file_hashes)`
- `assemble_feher_hare_measurement_validity_report`
- `measurement_report_payload(report)`

### Step 9.1: Write report-boundary RED tests

Assert that a GREEN report binds:

- protocol/input/anchor/candidate/prediction/execution-manifest hashes;
- all exact-invariance findings;
- aggregate losses, pairwise deltas, dependence findings, influence ranges, diagnostics, and counts;
- all five explicit false execution-boundary booleans;
- exact claim scope and `ExperimentStage.MEASUREMENT_AUDIT`;
- `AggregateReportArtifact.from_report(report)` and repository attestation.

Serialize the report and scan its canonical JSON for forbidden keys and known fixture values:

```python
encoded = json.dumps(
    measurement_report_payload(report),
    sort_keys=True,
    separators=(",", ":"),
)
for forbidden in (
    "source_participant_id",
    "participant_id",
    "source_path",
    "first_stage_action",
    "target_map",
    "prediction_rows",
    "final_test_cases",
):
    self.assertNotIn(forbidden, encoded)
```

Also assert:

- a failed exact finding yields `SCIENTIFIC_RED`, preserves an attested failure report, and omits the robustness conclusion;
- a pre-execution synthetic hard-gate failure permits no prediction hash or execution manifests, while a post-prediction empirical hard-gate failure must bind both;
- a material robustness dependence still yields GREEN with `MATERIALLY_MEASUREMENT_DEPENDENT` findings;
- an execution/identity exception is represented by `MeasurementAuditAttempt` as `INFRASTRUCTURE_INCOMPLETE`, not converted into a scientific report;
- the report has no free-text canonical scientific conclusion;
- attestation fails after any report-field mutation;
- raw prediction/case objects cannot be inserted into canonical report fields.

### Step 9.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_measurement_validity_reporting -v
git add tests/test_measurement_validity_reporting.py
git commit -m "test: define aggregate measurement report boundary"
```

### Step 9.3: Implement report assembly

- Build the manifest from aggregate identity hashes only.
- Store `execution_manifest_hashes` as a sorted unique tuple, never execution payloads.
- Store participant-group counts, never participant hashes, in the persisted report.
- Exclude `prediction_artifact` and `audit_input` fields through `report_artifact_exclude` metadata if they are temporarily attached for runtime validation; prefer not attaching them.
- Use `attest_report(report)` without creating a second attestation scheme.
- Make report construction check the five false booleans rather than merely populate them.

### Step 9.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_measurement_validity_reporting tests.test_measurement_validity_robustness tests.test_measurement_validity_diagnostic -v
git diff --check
git add narrative_dynamics/measurement_validity.py narrative_dynamics/studies/feher_hare_measurement_validity_v1.py narrative_dynamics/__init__.py
git commit -m "feat: attest aggregate measurement reports"
```

## Task 10: Compose the complete synthetic pipeline and public study API

**Files:**

- Create: `tests/test_feher_hare_measurement_validity_pipeline.py`
- Modify: `narrative_dynamics/studies/feher_hare_measurement_validity_v1.py`
- Modify: `narrative_dynamics/studies/__init__.py`

**Interfaces:**

- `FeherHareMeasurementValidityResult(audit_input_hash, protocol, prediction_hash, report, artifact, attestation)`
- `build_feher_hare_measurement_protocol(anchor)`
- `run_feher_hare_measurement_validity_v1(root, manifest, lock_path, lock_commit, repository_identity, executor)`

### Step 10.1: Write end-to-end RED tests

The synthetic integration test must patch only the trusted source preparation boundary, then run the real protocol builder, candidate reconstruction, selected-only execution, invariance gate, scoring, robustness, diagnostic, report, artifact, and attestation.

Assert:

- deterministic full result hash across two runs and reordered synthetic inputs;
- zero training, zero selection, zero FINAL access, and zero FINAL execution;
- the hard gate runs before empirical robustness assembly;
- exact-invariance failure stops robustness assembly and returns a scientific RED report;
- executor failure yields an infrastructure attempt record and no scientific report;
- every protocol callable identity matches the callable actually invoked;
- the only public study entry point that touches the source root is the trusted provisioner;
- `narrative_dynamics.studies` exports the study runner but package root exports only generic contracts.

### Step 10.2: Confirm RED and commit tests

```bash
python3 -m unittest tests.test_feher_hare_measurement_validity_pipeline -v
git add tests/test_feher_hare_measurement_validity_pipeline.py
git commit -m "test: define measurement validity pipeline"
```

### Step 10.3: Implement the orchestrator with explicit stop order

The function order is fixed:

1. load and verify anchor;
2. trusted provision restricted input;
3. build and hash protocol;
4. reconstruct exact frozen candidates;
5. run the frozen synthetic semantic/executor pre-gate;
6. if the synthetic pre-gate fails, assemble/attest SCIENTIFIC RED and stop;
7. execute selected candidates on allowed real cases;
8. run categorical-coordinate and record-order invariance over that single prediction artifact and combine the complete hard gate;
9. if the complete hard gate fails, assemble/attest SCIENTIFIC RED and stop before robustness;
10. score the single prediction artifact under Brier and Log;
11. build robustness and diagnostic profiles;
12. assemble/attest GREEN report.

The function must not catch arbitrary exceptions as scientific findings. The workflow layer records those as infrastructure attempts.

### Step 10.4: Verify GREEN and commit

```bash
python3 -m unittest tests.test_feher_hare_measurement_validity_pipeline -v
python3 -m unittest discover -s tests -v
git diff --check
git add narrative_dynamics/studies/feher_hare_measurement_validity_v1.py narrative_dynamics/studies/__init__.py
git commit -m "feat: compose measurement validity v1"
```

Expected: the full Python suite passes; no upstream repository is fetched.

## Task 11: Freeze the protocol payload and prove the scientific exact head

**Files:**

- Create: `tests/test_measurement_validity_protocol_lock.py`
- Create: `research-locks/feher-hare-measurement-validity-v1-protocol.json`
- Modify production code only if a failing existing test reveals a genuine compatibility defect in the new code.
- Do not create the real workflow yet.

### Step 11.1: Add a test-only RED for the immutable protocol payload

The test must load `research-locks/feher-hare-measurement-validity-v1-protocol.json`, require this exact top-level schema, reconstruct the protocol through `MeasurementValidityProtocol.from_payload`, and verify the declared hash:

```text
schema
design_path
design_file_sha256
r3_lock_commit
anchor
protocol
protocol_hash
parameter_training_performed
parameter_selection_performed
final_test_values_exposed_to_audit
final_test_outcomes_analyzed
final_model_execution
```

It must also rebuild `build_feher_hare_measurement_protocol` from the embedded frozen anchor and require the same protocol hash and canonical payload. All five execution booleans must be false.

Run and commit RED:

```bash
python3 -m unittest tests.test_measurement_validity_protocol_lock -v
git add tests/test_measurement_validity_protocol_lock.py
git commit -m "test: require frozen measurement protocol"
```

Expected: failure because the protocol-lock JSON does not exist.

### Step 11.2: Generate and commit the exact protocol lock

Read the R3 lock bytes directly from `lock/feher-hare-r3-internal-final@f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3`, load the exact anchor, build the final protocol, and print its canonical JSON and content hash. Use `apply_patch` to add those exact values to `research-locks/feher-hare-measurement-validity-v1-protocol.json`; do not reimplement protocol construction in a one-off script.

The embedded anchor must contain only the identities and candidate rows listed in this plan, not the full R3 lock or any FINAL result. The protocol lock must not contain its own repository commit, avoiding a self-referential hash.

Run and commit GREEN:

```bash
python3 -m unittest tests.test_measurement_validity_protocol_lock -v
git diff --check
git add research-locks/feher-hare-measurement-validity-v1-protocol.json
git commit -m "research: freeze measurement validity protocol"
```

### Step 11.3: Run local verification from a clean worktree

```bash
git status --short
python3 -m unittest discover -s tests -v
git diff --check
rg -n "predict_external_final_once|FINAL_SEEDS|_grid_candidates|fit_training_target_grid|select_on_validation_suite|from_selection" narrative_dynamics/measurement_validity.py narrative_dynamics/studies/feher_hare_measurement_validity_v1.py
```

Expected:

- clean status;
- full Python suite passes;
- `git diff --check` is silent;
- the source scan shows no forbidden execution call. References in explicit fail-closed validation messages are acceptable only when tests prove they are not invoked.

### Step 11.4: Push the scientific branch and record the exact SHA/protocol hash

```bash
measurement_scientific_sha="$(git rev-parse HEAD)"
measurement_protocol_hash="$(python3 -c 'import json; print(json.load(open("research-locks/feher-hare-measurement-validity-v1-protocol.json", encoding="utf-8"))["protocol_hash"])')"
git log -1 --format='%H %s'
```

Push `work/narrative-measurement-validity-v1`, then run the repository's existing exact-head proof workflow for `measurement_scientific_sha`.

### Step 11.5: Verify authoritative CI

Require all existing gates at the exact scientific SHA:

- Python unit tests;
- Lean conformance;
- Lean build;
- Lean theorem/story/testimony checks;
- exact-head equality.

If any gate fails, use `superpowers:systematic-debugging`; add a new test-only RED commit before a production fix. Do not create or run the real audit workflow until the exact head is GREEN.

### Step 11.6: Record the scientific proof in #39

Append the scientific SHA, frozen protocol hash, protocol-lock file hash, proof run/job IDs, test count, and explicit statements:

```text
parameter_training_performed=false
parameter_selection_performed=false
final_test_values_exposed_to_audit=false
final_test_outcomes_analyzed=false
final_model_execution=false
```

Do not claim a measurement result yet.

## Task 12: Add the one real audit workflow on an orchestration-only branch

**Files:**

- Create on `manual/measurement-validity-v1` only: `.github/workflows/real-measurement-validity-v1.yml`

### Step 12.1: Create an orchestration branch whose parent is the approved scientific SHA

```bash
git switch work/narrative-measurement-validity-v1
measurement_scientific_sha="$(git rev-parse HEAD)"
git switch -c manual/measurement-validity-v1 "$measurement_scientific_sha"
```

### Step 12.2: Write a workflow that self-derives and checks its scientific parent

Trigger only on a push to `manual/measurement-validity-v1` with commit message exactly:

```text
research: run measurement validity v1
```

The workflow must:

1. checkout the orchestration head with `fetch-depth: 2`;
2. derive `scientific_sha="$(git rev-parse HEAD^)"` and require the current commit has exactly one parent;
3. checkout that parent into `study/`;
4. checkout `lock/feher-hare-r3-internal-final@f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3` into `r3_lock/`;
5. checkout `carolfs/muddled_models@4567763780a2c596fd6510af720ec468a8214a8f` into `upstream/muddled_models/`;
6. verify the R3 lock file SHA/path and every frozen semantic identity;
7. load the scientific parent's `research-locks/feher-hare-measurement-validity-v1-protocol.json`, rebuild the protocol with current callables, and require byte-equivalent payload plus exact protocol hash;
8. create an immutable attempt-start JSON before importing the study runner;
9. invoke `run_feher_hare_measurement_validity_v1` with `ProcessCandidateExecutor(max_workers=4)`;
10. write aggregate report, attestation, attempt terminal state, and execution-manifest hash index;
11. on a scientific hard-gate failure, persist the SCIENTIFIC RED report and exit nonzero only after artifact staging;
12. on infrastructure failure, write `INFRASTRUCTURE_INCOMPLETE`, include the exception class/message hash, and do not create a scientific report;
13. upload the artifact under `if: always()` with 90-day retention.

Use a final artifact staging directory containing exactly:

```text
measurement_validity_attempt.json
measurement_validity_report.json
measurement_validity_report_artifact.json
measurement_validity_attestation.json
measurement_execution_manifest_index.json
```

For infrastructure incomplete, the directory may contain only `measurement_validity_attempt.json` plus any pre-existing hash-only manifest index. It must never contain the restricted input or prediction artifact.

### Step 12.3: Add workflow self-checks before the real invocation

The shell/Python preflight must fail unless:

```text
scientific parent equals approved exact-head proof SHA
rebuilt protocol hash equals the scientific protocol-lock hash
lock commit equals f09d6afc96f0a720dd5e8e810f440cda0fe9f8a3
upstream revision equals 4567763780a2c596fd6510af720ec468a8214a8f
claim scope equals external_observational_measurement_audit_only
allowed roles equal train,selection_validation
candidate count equals 3
TRAIN seeds equal 101,102
SELECTION seeds equal 201,202
FINAL seeds absent
```

### Step 12.4: Commit the orchestration-only workflow

```bash
git add .github/workflows/real-measurement-validity-v1.yml
git commit -m "research: run measurement validity v1"
test "$(git rev-list --count HEAD^..HEAD)" -eq 1
test "$(git diff --name-only HEAD^ HEAD)" = ".github/workflows/real-measurement-validity-v1.yml"
```

Expected: the workflow is the only file changed relative to the approved scientific parent.

### Step 12.5: Push once and monitor without automatic retry

Push `manual/measurement-validity-v1`. Record the run/job IDs immediately. Do not automatically dispatch a second run.

- `GREEN`: continue to Task 13.
- `SCIENTIFIC RED`: continue to Task 13 with Transfer V1 blocked.
- `INFRASTRUCTURE INCOMPLETE`: inspect the attempt artifact, fix orchestration only if the root cause is infrastructure, disclose the failed attempt, and replay the exact same protocol/scientific SHA manually.

## Task 13: Independently audit the real artifact and close the phase

**Files:**

- No scientific code changes.
- Update GitHub issue #39 only after independent artifact checks.
- Create a separate Transfer V1 design issue only when the hard gate is GREEN.

### Step 13.1: Download and hash the artifact

Record the Actions artifact ID, ZIP digest, every contained file SHA-256, and report artifact content hash. Verify the archive contains only the five allowed filenames and no nested unexpected file.

### Step 13.2: Perform a byte-level safety audit

Parse each JSON file and assert:

- scientific SHA equals the GREEN exact-head SHA;
- lock and upstream commits match the frozen values;
- source/data/partition/target/candidate/protocol hashes match;
- aggregate counts reconcile with allowed TRAIN/SELECTION cases;
- execution-manifest count equals `3 × 2 × allowed_case_count`;
- all execution-manifest hashes are unique;
- exact gate status and every required robustness/diagnostic dimension are present for a GREEN report;
- all five false booleans are present and false;
- no key/value exposes direct participant ids, source paths, raw choices, targets, per-case predictions, FINAL cases, R3 FINAL values, or FINAL seeds.

Run the forbidden-string scan against unpacked JSON:

```bash
rg -n 'source_participant_id|participant_id|source_path|prediction_rows|target_map|final_case_names|"seed":30[12]' measurement-validity-artifact
```

Expected: no match.

### Step 13.3: Classify the terminal state exactly

- Hard gate met plus complete attested profile: `GREEN`.
- Reproducible hard-gate failure plus attested failure report: `SCIENTIFIC RED`.
- Missing identity, execution, report, or artifact: `INFRASTRUCTURE INCOMPLETE`.

Material task/aggregation/score dependence is a substantive finding inside GREEN; it is not infrastructure failure and does not authorize choosing a favorable measurement.

### Step 13.4: Update #39 append-only

Record:

- design, plan, scientific, orchestration, run/job, and artifact identities;
- exact hard-gate findings;
- typed task/aggregation/score/influence findings;
- diagnostic status including every `NOT_ESTABLISHED` cell;
- fixed claim scope;
- explicit zero training/selection/FINAL assertions;
- a statement that Study V1 conclusions were not rewritten.

### Step 13.5: Apply the Transfer V1 handoff rule

- If hard gate is GREEN, open a separate architectural-design issue named `Cross-dataset Transfer V1`, carrying every materially dependent or inconclusive dimension forward as a frozen stratum/sensitivity requirement. Do not choose a second dataset or write transfer code in this task.
- If hard gate is SCIENTIFIC RED, keep Transfer V1 blocked and open a measurement-revision design issue instead.
- If infrastructure is incomplete, leave Measurement Validity V1 open and authorize only exact-input infrastructure replay.

## Final verification checklist

Before declaring Measurement Validity V1 complete, run or verify every item below against fresh output:

```bash
git status --short
git log --oneline --decorate -20
python3 -m unittest discover -s tests -v
git diff --check
rg -n "NotImplementedError|raise AssertionError\(\"unfinished\"" narrative_dynamics tests .github/workflows/real-measurement-validity-v1.yml
```

Then verify externally:

- exact-head CI is GREEN at the scientific SHA;
- the manual workflow changed only the workflow file relative to its scientific parent;
- the real run has a terminal artifact even on failure;
- the independent artifact audit matches all identities and firewall assertions;
- #39 contains the typed terminal result;
- no R3 FINAL branch, sealed artifact, result, threshold, candidate, or conclusion was modified;
- no model training, parameter selection, FINAL value exposure, FINAL outcome analysis, or FINAL model execution occurred.

Only after those checks may the phase be called GREEN or SCIENTIFIC RED. An infrastructure-incomplete run is not a scientific result.
