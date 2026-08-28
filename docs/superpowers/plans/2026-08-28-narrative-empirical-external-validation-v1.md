# Narrative Empirical / External Validation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to execute this plan task-by-task. Do not skip RED evidence or exact-head verification.

**Goal:** Implement issue #38, P3 Empirical / External Validation V1, as a thin fail-closed bridge from positively declared external observational evidence into the existing observation, calibration, preregistration, witnessed-release, proper-scoring, and attestation machinery while making it impossible for predictive fit to become a canonical claim that real latent cognition was identified.

**Architecture:** Add `narrative_dynamics.observations.external` for positive external-evidence declarations and source/partition lineage. Add `narrative_dynamics.external_validation` for claim-safe status vocabularies, Brier/Log sibling preregistration, dual verified-release preflight, predictive adequacy/separation/stratum findings, selection-only finite parameter constraints, and the final attested aggregate report. Reuse `ObservationDataset`, target construction, `calibrate_grid`, `PreregisteredEvaluationProtocol`, `ProtocolRelease`, `VerifiedProtocolRelease`, `compare_released_models()`, and `attest_report()` as lower layers. Do not add an empirical mode to P2 identification.

**Tech stack:** Python stdlib, existing `narrative_dynamics` research APIs, `unittest`, GitHub Actions `proof.yml`, and existing Lean/lake gates for regression verification only.

**Approved spec:** `docs/superpowers/specs/2026-08-28-narrative-empirical-external-validation-v1-design.md`

**Approved spec head:** `9758b8f7d6088093418826610768c8a73e13943f`

**Integrated research base:** `proof/narrative-dynamics-v0@f6dbe9e67d6dda064fa332989187e7e8a604039a`

## Non-negotiable boundaries

- Implementation branch after this plan is approved: `work/narrative-empirical-external-validation-v1`, cut from the committed plan head.
- First implementation commit is a complete **test-only RED**. No P3 production symbol is added before that RED is committed and observed failing for the intended missing boundary.
- Production changes are limited to:
  - create `narrative_dynamics/observations/external.py`;
  - create `narrative_dynamics/external_validation.py`;
  - add only `ExperimentStage.EXTERNAL_VALIDATION = "external_validation"` in `narrative_dynamics/contracts.py`;
  - additive exports in `narrative_dynamics/observations/__init__.py` and `narrative_dynamics/__init__.py`.
- Do not modify `narrative_dynamics/identification.py`, `narrative_dynamics/diagnostics.py`, `narrative_dynamics/calibration.py`, `narrative_dynamics/validation.py`, `narrative_dynamics/model_comparison.py`, `narrative_dynamics/observations/dataset.py`, `targets.py`, `preregistration.py`, `release.py`, narrative cognition/runtime semantics, stochastic world/observation semantics, existing observation fixtures, or Lean sources.
- No Docker acceptance. The authoritative workflow is `.github/workflows/proof.yml`.
- Canonical claim scope is exactly `external_observational_predictive_only`.
- P3 exposes no `IdentificationStatus`, no `identification_status` field, and no free-text canonical conclusion field.
- Allowed P3 result vocabularies are exactly:
  - `predictive_adequacy_met` / `predictive_adequacy_not_met`;
  - `predictively_separated_under_protocol` / `not_predictively_separated_under_protocol`;
  - `constrained_under_external_protocol` / `not_constrained_under_external_protocol`.
- Canonical P3 fields must reject values equivalent to `EMPIRICALLY_IDENTIFIED`, `STRUCTURALLY_IDENTIFIED`, `COGNITIVE_MECHANISM_CONFIRMED`, `LATENT_COGNITION_IDENTIFIED`, or `REAL_PERSON_GOAL_RECOVERED`.
- External eligibility requires both positive markers:
  - `source.kind == "external_observational"`;
  - `provenance.external_observational is True`.
- Known synthetic markers fail closed:
  - `source.kind == "synthetic_fixture"`;
  - `provenance.synthetic_non_empirical is True`.
- `provenance.empirical_human_data is False` alone is not a synthetic marker and cannot reject a positively declared non-human external dataset.
- Existing `fixtures/observations/prison_initial_choice_v1.json` and `fixtures/observations/narrative_identification_v1.json` are rejection fixtures only. Do not relabel them.
- Do not add a repository fixture presented as genuine empirical evidence. Positive external records used by unit tests are constructed in test code with an explicit `test_only` provenance marker.
- Brier and Log are one frozen sibling pair. All non-score-specific identities are exact. Score-specific adequacy thresholds may differ only because both threshold sets are frozen in the P3 preregistration before final evaluation.
- Both sibling `ProtocolRelease` values and both `VerifiedProtocolRelease` values pass P3 preflight before either child `compare_released_models()` call.
- P3 release `source_revision` binds at least `repository_revision`, `external_evidence_declaration_hash`, `external_validation_preregistration_hash`, and `score_role`.
- Stratum reporting uses existing final per-case losses and triggers no resimulation.
- External parameter constraints use only `SELECTION_VALIDATION`. They do not consume `FINAL_TEST`, generator truth, P2 `IdentificationStatus`, `diagnose_identifiability()`, or `ParameterIdentificationFinding`.
- For each score, compatible candidates satisfy `candidate_loss <= best_loss + preregistered_delta`. The external compatible set is Brier-compatible ∩ Log-compatible. Empty intersection is a typed protocol/execution failure.
- A singleton compatible set is `constrained_under_external_protocol`, never identified.
- `method_validation_hashes` are opaque lineage only. P2 statuses never propagate into P3 findings.
- Final `ExternalValidationReport` must pass `attest_report(report).require_integrity()` without changing `report_artifact.py`.

## Planned files

**Production**

- Create `narrative_dynamics/observations/external.py`
- Create `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/contracts.py`
- Modify `narrative_dynamics/observations/__init__.py`
- Modify `narrative_dynamics/__init__.py`

**Tests**

- Create `tests/external_validation_fixtures.py`
- Create `tests/test_external_validation_claims.py`
- Create `tests/test_external_evidence.py`
- Create `tests/test_external_validation_preregistration.py`
- Create `tests/test_external_validation_release_gate.py`
- Create `tests/test_external_validation_final.py`
- Create `tests/test_external_validation_constraints.py`
- Create `tests/test_external_validation_reporting.py`

---

## Task 1 — Commit the complete test-only P3 RED boundary

**Files:** create all eight test files listed above. Do not modify production files.

- [ ] **1.1 Create `tests/external_validation_fixtures.py`**

Reuse the exact existing observation/loss/protocol constructors from `tests/observational_data_fixtures.py`, `tests/test_observational_data_protocol.py`, `tests/test_preregistered_model_comparison.py`, and `tests/test_protocol_release.py`. The helper module owns only test data/builders and is not collected by the `test_*.py` pattern.

Create a positively declared test-only dataset with all three partitions. Its canonical source/provenance must include:

```python
source={
    "kind": "external_observational",
    "release": "test-snapshot-v1",
}
provenance={
    "external_observational": True,
    "test_only": True,
}
```

Use stable source-record ids with one TRAIN record, one SELECTION_VALIDATION record, and two FINAL_TEST records. Add a trivial model with a call counter:

```python
class ProbabilityModel:
    name = "external-probability-model"
    version = "1"
    implementation_revision = "external-probability-v1"

    def __init__(self) -> None:
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        p = float(parameters["p"])
        return ModelRun(
            events=(),
            outcome={"policy": {"a": p, "b": 1.0 - p}},
        )
```

Add a second model with a distinct component identity. Build Brier and Log sibling protocols using the same dataset, target spec, extractor, seeds, baseline, frozen candidate names/model identities/parameter tuples/selection-manifest hashes, and protocol version. Freeze Brier adequacy thresholds separately from Log thresholds. Use fresh runtime model instances per score during execution so mutable test counters cannot leak across siblings.

- [ ] **1.2 Create claim-firewall RED tests**

`tests/test_external_validation_claims.py` locks these exact tests:

```text
test_claim_scope_is_exact_and_immutable
test_public_status_vocabularies_are_closed
test_external_validation_exposes_no_identification_status
test_forbidden_empirical_identification_values_are_not_valid_statuses
```

Guard the new-module import so discovery continues and the RED is readable:

```python
try:
    import narrative_dynamics.external_validation as external_validation
except ImportError as error:
    external_validation = None
    IMPORT_ERROR = error
```

Every test first asserts the module exists; after GREEN it must also assert `not hasattr(external_validation, "IdentificationStatus")`.

- [ ] **1.3 Create external-evidence RED tests**

`tests/test_external_evidence.py` locks:

```text
test_positive_external_origin_builds_stable_declaration
test_absence_of_positive_external_origin_is_rejected
test_prison_synthetic_fixture_is_rejected
test_p2_synthetic_identification_fixture_is_rejected
test_empirical_human_false_alone_does_not_mark_positive_external_data_synthetic
test_partition_assignment_hash_is_order_stable_and_role_sensitive
test_dataset_source_transform_namespace_or_partition_drift_is_rejected
```

Load the two existing JSON fixtures through the existing observation dataset loader. The positive non-human test must keep `source.kind="external_observational"` and `provenance.external_observational=True` while setting `empirical_human_data=False`.

- [ ] **1.4 Create sibling-preregistration RED tests**

`tests/test_external_validation_preregistration.py` locks:

```text
test_score_roles_are_exactly_brier_then_log
test_preregistration_binds_evidence_both_protocols_and_score_specific_thresholds
test_preregistration_accepts_different_brier_log_threshold_numbers_when_each_matches_its_protocol
test_duplicate_or_unknown_score_family_is_rejected
test_target_seed_metric_baseline_candidate_parameter_or_selection_lineage_drift_is_rejected
test_score_specific_threshold_drift_is_rejected
test_strata_cover_every_final_case_exactly_once
test_strata_reject_missing_extra_or_overlapping_final_cases
test_pairwise_separation_rule_requires_positive_finite_deltas_and_direction_agreement
test_constraint_plan_is_finite_selection_scoped_and_content_hashed
```

- [ ] **1.5 Create dual-release-gate RED tests**

`tests/test_external_validation_release_gate.py` locks:

```text
test_release_source_revision_must_bind_evidence_preregistration_repository_and_score_role
test_brier_release_cannot_be_substituted_for_log_release
test_verified_release_hash_must_match_the_actual_release_hash
test_one_unverified_or_drifted_sibling_prevents_both_final_executions
test_both_verified_siblings_produce_stable_preflight_identity
```

The invalid-second-sibling test must assert zero calls on both score-side runtime models.

- [ ] **1.6 Create final-predictive RED tests**

`tests/test_external_validation_final.py` locks:

```text
test_dual_final_uses_existing_released_comparisons_for_both_scores
test_per_score_and_aggregate_predictive_adequacy_are_claim_safe
test_pairwise_separation_requires_same_direction_and_both_delta_thresholds
test_nonseparation_preserves_both_raw_score_deltas
test_stratum_scores_use_existing_case_losses_without_resimulation
test_global_scores_are_unchanged_by_stratum_declarations
```

The stratum test records all model call counts immediately after child released comparisons and proves stratum construction leaves those counts unchanged.

- [ ] **1.7 Create selection-only constraint RED tests**

`tests/test_external_validation_constraints.py` locks:

```text
test_constraint_evaluation_requires_selection_validation_targets
test_brier_and_log_compatible_sets_are_intersected
test_full_candidate_loss_table_and_parent_calibration_hashes_are_retained
test_singleton_compatible_set_is_constrained_not_identified
test_multi_value_coordinate_is_not_constrained
test_empty_compatible_intersection_is_typed_failure
test_final_test_targets_fail_before_any_model_execution
test_constraint_path_has_no_generator_truth_or_p2_identifiability_dependency
```

For the final test, patch `narrative_dynamics.diagnostics.diagnose_identifiability` to raise immediately. P3 constraint evaluation must still complete because it never calls that function.

- [ ] **1.8 Create aggregate-report RED tests**

`tests/test_external_validation_reporting.py` locks:

```text
test_manifest_stage_is_external_validation
test_report_binds_evidence_preregistration_both_release_verifications_and_child_manifests
test_report_preserves_method_validation_hashes_as_opaque_lineage_only
test_report_has_no_identification_status_or_free_text_conclusions_field
test_report_claim_scope_is_exact
test_report_requires_declared_constraint_findings_exactly
test_report_attests_and_tampering_changes_artifact_identity
test_forged_or_incomplete_parent_lineage_is_rejected
```

- [ ] **1.9 Run the authoritative focused RED**

```bash
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: failures are only the newly required P3 boundary: missing `narrative_dynamics.observations.external`, missing `narrative_dynamics.external_validation`, missing P3 symbols, and missing `ExperimentStage.EXTERNAL_VALIDATION`. Existing imported P0-P2 infrastructure remains healthy.

- [ ] **1.10 Commit and push the test-only RED**

```bash
git add \
  tests/external_validation_fixtures.py \
  tests/test_external_validation_claims.py \
  tests/test_external_evidence.py \
  tests/test_external_validation_preregistration.py \
  tests/test_external_validation_release_gate.py \
  tests/test_external_validation_final.py \
  tests/test_external_validation_constraints.py \
  tests/test_external_validation_reporting.py
git commit -m "test: require empirical external validation v1"
git push -u origin work/narrative-empirical-external-validation-v1
```

Record the exact failing feature-head `proof` run before any production commit.

---

## Task 2 — GREEN the claim and external-evidence firewall

**Files:**
- Create `narrative_dynamics/observations/external.py`
- Create `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/observations/__init__.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **2.1 Implement the shared constants and root error in `observations/external.py`**

```python
EXTERNAL_EVIDENCE_ORIGIN = "external_observational"
EXTERNAL_CLAIM_SCOPE = "external_observational_predictive_only"

class ExternalValidationError(ValueError):
    """External predictive-validation protocol or evidence is invalid."""
```

Placing the shared root here keeps dependency direction acyclic; `external_validation.py` imports and re-exports this root.

- [ ] **2.2 Implement `ExternalEvidenceDeclaration`**

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
    evidence_origin: str = EXTERNAL_EVIDENCE_ORIGIN
    claim_scope: str = EXTERNAL_CLAIM_SCOPE
```

Public operations:

```python
@classmethod
def from_dataset(
    cls,
    dataset: ObservationDataset,
    *,
    name: str,
    version: str,
    source_snapshot_hash: str,
    source_reference: str,
    source_revision: Mapping[str, object],
    transform_identity: Mapping[str, object],
    record_namespace: str,
) -> "ExternalEvidenceDeclaration": ...

def require_matches(self, dataset: ObservationDataset) -> None: ...
def identity_payload(self) -> dict[str, object]: ...
@property
def content_hash(self) -> str: ...
```

Implementation requirements:

1. require `dataset` is `ObservationDataset`;
2. require both positive origin markers;
3. reject either known synthetic marker;
4. do not treat `empirical_human_data=False` as synthetic by itself;
5. validate `source_snapshot_hash` as `sha256:<64 lowercase hex>`;
6. recursively freeze `source_revision` and `transform_identity` using the existing observation canonical-freeze seam;
7. build `partition_assignment_hash` from the complete sorted tuple `(role.value, source_observation_id)`;
8. for record-oriented partitions use every `ObservationRecord.id`;
9. for case-oriented partitions expand every `ObservationCase.observation_ids` entry;
10. `require_matches()` rechecks dataset hash and partition assignment and fails closed on drift.

- [ ] **2.3 Implement only the P3 claim/status vocabulary in `external_validation.py`**

```python
class ExternalScoreRole(str, Enum):
    BRIER = "brier"
    LOG = "log"

class PredictiveAdequacyStatus(str, Enum):
    MET = "predictive_adequacy_met"
    NOT_MET = "predictive_adequacy_not_met"

class PredictiveSeparationStatus(str, Enum):
    SEPARATED = "predictively_separated_under_protocol"
    NOT_SEPARATED = "not_predictively_separated_under_protocol"

class ExternalConstraintStatus(str, Enum):
    CONSTRAINED = "constrained_under_external_protocol"
    NOT_CONSTRAINED = "not_constrained_under_external_protocol"
```

Do not import `narrative_dynamics.identification`. Do not define or re-export `IdentificationStatus`.

- [ ] **2.4 Add intended additive exports**

Expose evidence types/constants through `narrative_dynamics.observations` and the intended P3 public constants/status types through `narrative_dynamics` root. Preserve every existing export.

- [ ] **2.5 Run focused GREEN**

```bash
python3 -m unittest \
  tests.test_external_validation_claims \
  tests.test_external_evidence -v
```

Expected: GREEN. The later P3 suites remain RED because preregistration/release/report symbols do not exist yet.

- [ ] **2.6 Run observation/release regressions**

```bash
python3 -m unittest \
  tests.test_observation_dataset \
  tests.test_observation_targets \
  tests.test_protocol_release -v
```

Expected: GREEN.

- [ ] **2.7 Commit**

```bash
git add \
  narrative_dynamics/observations/external.py \
  narrative_dynamics/external_validation.py \
  narrative_dynamics/observations/__init__.py \
  narrative_dynamics/__init__.py
git commit -m "feat: add external evidence claim firewall"
```

---

## Task 3 — GREEN frozen Brier/Log external preregistration

**Files:**
- Modify `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **3.1 Add protocol/constraint error types**

```python
class ExternalValidationProtocolError(ExternalValidationError):
    pass

class ExternalValidationConstraintError(ExternalValidationError):
    pass
```

- [ ] **3.2 Add immutable declaration types**

```python
@dataclass(frozen=True)
class PairwiseSeparationRule:
    min_mean_loss_delta_brier: float
    min_mean_loss_delta_log: float
    require_direction_agreement: bool = True

@dataclass(frozen=True)
class ExternalStratum:
    name: str
    final_case_names: tuple[str, ...]

@dataclass(frozen=True)
class ExternalConstraintPlan:
    name: str
    model_identity: Mapping[str, object]
    parameter_grid: tuple[tuple[str, tuple[float, ...]], ...]
    target_coordinates: tuple[str, ...]
    selection_case_names: tuple[str, ...]
    brier_acceptance_loss_delta: float
    log_acceptance_loss_delta: float
    simulation_seeds: tuple[int, ...]
    metric_identity: Mapping[str, object]
    brier_loss_identity: Mapping[str, object]
    log_loss_identity: Mapping[str, object]
```

Validate finite/non-negative acceptance deltas, strictly positive finite separation deltas, non-empty finite candidate dimensions, unique names/coordinates/case names, non-empty integer seeds, canonical ordering, and exact `require_direction_agreement=True` for V1.

- [ ] **3.3 Implement `ExternalValidationPreregistration`**

```python
@dataclass(frozen=True)
class ExternalValidationPreregistration:
    name: str
    version: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    adequacy_thresholds_by_score: tuple[
        tuple[ExternalScoreRole, AdequacyThresholds], ...
    ]
    strata: tuple[ExternalStratum, ...]
    separation_rule: PairwiseSeparationRule
    constraint_plans: tuple[ExternalConstraintPlan, ...]
    method_validation_hashes: tuple[str, ...]
    claim_scope: str = EXTERNAL_CLAIM_SCOPE
```

Provide a `create(...)` constructor receiving the actual `ExternalEvidenceDeclaration`, Brier protocol, Log protocol, final `TargetConstructionReport`, both score-specific `AdequacyThresholds`, strata, separation rule, constraint plans, and method hashes.

Constructor validation order:

1. evidence `dataset_hash` equals both sibling protocol dataset hashes;
2. final target report hash/manifest/partition identity equals both sibling frozen final target identity;
3. identify Brier only by canonical `categorical_brier` loss identity and Log only by canonical `categorical_log` loss identity;
4. require exact equality of train/selection/final partition hashes, target spec/final target/final target manifest, metric identity, simulation seeds, baseline name, protocol version, complete candidate names/model identities/selected parameters/selection manifest hashes;
5. require each sibling protocol's thresholds equal its own threshold set passed to the P3 preregistration;
6. allow Brier threshold values and Log threshold values to differ from each other;
7. reject duplicate or unknown score families;
8. require score ordering Brier then Log;
9. require strata cover every final target case exactly once with no missing/extra/overlap;
10. require unique constraint-plan names;
11. validate every `method_validation_hash` as a content hash;
12. compute deterministic `content_hash` from the semantic declaration.

- [ ] **3.4 Export preregistration types**

Add intended symbols to package root without removing or renaming any existing symbol.

- [ ] **3.5 Run preregistration GREEN**

```bash
python3 -m unittest tests.test_external_validation_preregistration -v
```

Expected: GREEN.

- [ ] **3.6 Run existing preregistration regressions**

```bash
python3 -m unittest \
  tests.test_observational_data_protocol \
  tests.test_preregistered_model_comparison \
  tests.test_protocol_release -v
```

Expected: GREEN.

- [ ] **3.7 Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: freeze external validation sibling protocols"
```

---

## Task 4 — GREEN dual verified-release preflight

**Files:**
- Modify `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **4.1 Add the preflight identity**

```python
@dataclass(frozen=True)
class ExternalReleasePreflight:
    preregistration_hash: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    brier_release_hash: str
    brier_verification_hash: str
    log_protocol_hash: str
    log_release_hash: str
    log_verification_hash: str

    def identity_payload(self) -> dict[str, object]: ...

    @property
    def content_hash(self) -> str: ...
```

- [ ] **4.2 Implement `preflight_external_releases()`**

```python
def preflight_external_releases(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
) -> ExternalReleasePreflight: ...
```

Exact validation sequence before any simulation:

1. validate argument types;
2. `evidence.content_hash == preregistration.evidence_declaration_hash`;
3. protocol hashes equal the preregistered sibling hashes;
4. call each `VerifiedProtocolRelease.require_matches(protocol)`;
5. additionally require `verified.release_hash == release.content_hash` for each sibling because the lower `require_matches()` checks protocol identity only;
6. require each release dataset/target/candidate identity matches its protocol;
7. require `source_revision` has exactly valid P3 binding values for `repository_revision`, evidence hash, P3 preregistration hash, and score role;
8. require sibling releases share exact repository/evidence/P3 preregistration identities;
9. require Brier role is `brier` and Log role is `log`;
10. return the frozen preflight identity.

Let lower-layer `ProtocolReleaseVerificationError` propagate unchanged. P3-only binding drift raises `ExternalValidationProtocolError`.

- [ ] **4.3 Run release-gate GREEN**

```bash
python3 -m unittest tests.test_external_validation_release_gate -v
```

The pure-preflight tests must be GREEN. The final-entry-point no-execution assertion becomes fully GREEN in Task 5 when the public dual-final orchestrator exists.

- [ ] **4.4 Run existing release regression**

```bash
python3 -m unittest tests.test_protocol_release -v
```

Expected: GREEN with no changes to `narrative_dynamics/observations/release.py`.

- [ ] **4.5 Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: gate external validation on dual verified releases"
```

---

## Task 5 — GREEN dual final predictive evaluation, separation, and strata

**Files:**
- Modify `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **5.1 Add claim-safe final finding types**

```python
@dataclass(frozen=True)
class ExternalPredictiveAdequacyFinding:
    model_name: str
    frozen_model_hash: str
    score_role: ExternalScoreRole
    loss_identity: Mapping[str, object]
    mean_loss: float
    worst_loss: float
    threshold_hash: str
    status: PredictiveAdequacyStatus
    parent_comparison_manifest_hash: str

@dataclass(frozen=True)
class ExternalPairwiseSeparationFinding:
    left_model: str
    right_model: str
    brier_mean_loss_delta: float
    log_mean_loss_delta: float
    preferred_model: str | None
    status: PredictiveSeparationStatus

@dataclass(frozen=True)
class ExternalStratumScore:
    stratum_name: str
    score_role: ExternalScoreRole
    model_name: str
    case_names: tuple[str, ...]
    mean_loss: float
    worst_loss: float

@dataclass(frozen=True)
class ExternalFinalEvaluation:
    preflight: ExternalReleasePreflight
    brier_report: ReleasedModelComparisonReport
    log_report: ReleasedModelComparisonReport
    adequacy_findings: tuple[ExternalPredictiveAdequacyFinding, ...]
    aggregate_adequacy: tuple[tuple[str, PredictiveAdequacyStatus], ...]
    separation_findings: tuple[ExternalPairwiseSeparationFinding, ...]
    stratum_scores: tuple[ExternalStratumScore, ...]
```

Canonical ordering is score-role, model name, stratum name, and lexical model pair.

- [ ] **5.2 Build adequacy from existing child report entries**

Use the existing released comparison entries. Per-score status is MET iff the child entry is adequate under the already frozen score-specific thresholds. Aggregate model adequacy is MET only when both Brier and Log are MET.

- [ ] **5.3 Build pairwise separation from existing mean losses**

For lexical pair `(A, B)` define:

```text
brier_delta = brier_loss(A) - brier_loss(B)
log_delta   = log_loss(A)   - log_loss(B)
```

Status is `SEPARATED` only when:

```text
sign(brier_delta) == sign(log_delta) != 0
abs(brier_delta) >= min_mean_loss_delta_brier
abs(log_delta) >= min_mean_loss_delta_log
```

The lower-loss model is the `preferred_model`. On non-separation retain both raw deltas; do not serialize a canonical winner claim.

- [ ] **5.4 Build strata without model execution**

Read per-case losses already present below each released comparison entry's final-test validation report. For every declared stratum and every model/score, calculate `fmean` and `max` over those existing case losses. Do not call `SimulationRunner` or any model from a stratum helper.

- [ ] **5.5 Implement `evaluate_external_final()`**

```python
def evaluate_external_final(
    *,
    runner: SimulationRunner,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    brier_protocol: PreregisteredEvaluationProtocol,
    brier_release: ProtocolRelease,
    brier_verified: VerifiedProtocolRelease,
    brier_models: tuple[ComparisonModel, ...],
    brier_loss: MetricLoss,
    log_protocol: PreregisteredEvaluationProtocol,
    log_release: ProtocolRelease,
    log_verified: VerifiedProtocolRelease,
    log_models: tuple[ComparisonModel, ...],
    log_loss: MetricLoss,
    final_targets: TargetConstructionReport,
    extractor: object,
) -> ExternalFinalEvaluation: ...
```

After argument materialization/type checks, the first semantic operation is a single `preflight_external_releases(...)`. Only after it succeeds may either `compare_released_models()` call run. Both children use the same final target report and extractor, with their own frozen score protocol/loss and fresh runtime model tuples.

- [ ] **5.6 Run release-gate + final GREEN**

```bash
python3 -m unittest \
  tests.test_external_validation_release_gate \
  tests.test_external_validation_final -v
```

Expected: GREEN, including zero Brier calls when the Log sibling fails preflight and zero additional calls during stratum aggregation.

- [ ] **5.7 Run exact existing final-comparison regressions**

```bash
python3 -m unittest \
  tests.test_preregistered_model_comparison \
  tests.test_protocol_release \
  tests.test_narrative_held_out_model_comparison -v
```

Expected: GREEN.

- [ ] **5.8 Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: evaluate external predictive evidence"
```

---

## Task 6 — GREEN selection-only external parameter constraints

**Files:**
- Modify `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **6.1 Add auditable constraint result types**

```python
@dataclass(frozen=True)
class ExternalConstraintCandidateLoss:
    parameters: tuple[tuple[str, float], ...]
    brier_loss: float
    log_loss: float
    brier_parent_manifest_hashes: tuple[str, ...]
    log_parent_manifest_hashes: tuple[str, ...]

@dataclass(frozen=True)
class ExternalCoordinateConstraint:
    parameter_name: str
    retained_values: tuple[float, ...]
    constrained: bool

@dataclass(frozen=True)
class ExternalConstraintFinding:
    plan_hash: str
    model_identity: Mapping[str, object]
    candidate_losses: tuple[ExternalConstraintCandidateLoss, ...]
    brier_compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    log_compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    compatible_parameters: tuple[tuple[tuple[str, float], ...], ...]
    coordinates: tuple[ExternalCoordinateConstraint, ...]
    status: ExternalConstraintStatus
    parent_manifest_hashes: tuple[str, ...]
```

No constraint type contains `true_parameters`, `generator_family`, `identification_status`, or a P2 finding.

- [ ] **6.2 Implement one score-grid evaluator using existing `calibrate_grid()`**

Before any model call:

1. require `selection_targets.role is ObservationPartitionRole.SELECTION_VALIDATION`;
2. require the target report lineage and metric/loss identities match the plan;
3. require the plan's declared selection case names exist exactly in the selection target report;
4. require runtime model component identity equals the plan model identity.

For each declared selection case, call existing `calibrate_grid()` using the same finite grid and plan seeds. Require every case returns the same canonical candidate set. Aggregate each candidate's case losses with `fmean`. Retain every candidate loss and child calibration manifest hash.

- [ ] **6.3 Implement dual-score compatible-set intersection**

For each score:

```python
best_loss = min(candidate.mean_loss for candidate in score_table)
compatible = {
    candidate.parameters
    for candidate in score_table
    if candidate.mean_loss <= best_loss + preregistered_delta
}
```

Then:

```python
compatible_parameters = brier_compatible & log_compatible
```

If empty, raise `ExternalValidationConstraintError`.

For each target coordinate, collect sorted distinct retained values. Exactly one value means the coordinate is constrained. Aggregate status is CONSTRAINED only when every declared target coordinate is constrained; otherwise NOT_CONSTRAINED.

Do not call `diagnose_identifiability()` and do not translate the result into P2 vocabulary.

- [ ] **6.4 Implement the public evaluator**

```python
def evaluate_external_constraint(
    *,
    runner: SimulationRunner,
    plan: ExternalConstraintPlan,
    model: ModelSource,
    selection_targets: TargetConstructionReport,
    extractor: object,
    brier_loss: MetricLoss,
    log_loss: MetricLoss,
) -> ExternalConstraintFinding: ...
```

All role/identity checks happen before the first model call.

- [ ] **6.5 Run constraint GREEN**

```bash
python3 -m unittest tests.test_external_validation_constraints -v
```

Expected: GREEN, including singleton constrained, multi-value not constrained, empty-intersection typed failure, final-test rejected before calls, and patched P2 diagnostics never invoked.

- [ ] **6.6 Run exact existing calibration/validation regressions**

```bash
python3 -m unittest \
  tests.test_calibration \
  tests.test_observational_training_fit \
  tests.test_heldout_validation \
  tests.test_narrative_validation -v
```

Expected: GREEN.

- [ ] **6.7 Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: constrain external parameters without identification"
```

---

## Task 7 — GREEN the attested aggregate external validation report

**Files:**
- Modify `narrative_dynamics/contracts.py`
- Modify `narrative_dynamics/external_validation.py`
- Modify `narrative_dynamics/__init__.py`

- [ ] **7.1 Add exactly one manifest stage**

```python
EXTERNAL_VALIDATION = "external_validation"
```

No manifest schema-version change.

- [ ] **7.2 Add report error and immutable report type**

```python
class ExternalValidationReportError(ExternalValidationError):
    pass

@dataclass(frozen=True)
class ExternalValidationReport:
    preregistration_hash: str
    evidence_declaration_hash: str
    claim_scope: str
    brier_protocol_hash: str
    brier_release_hash: str
    brier_verification_hash: str
    log_protocol_hash: str
    log_release_hash: str
    log_verification_hash: str
    brier_comparison_manifest_hash: str
    log_comparison_manifest_hash: str
    adequacy_findings: tuple[ExternalPredictiveAdequacyFinding, ...]
    aggregate_adequacy: tuple[tuple[str, PredictiveAdequacyStatus], ...]
    separation_findings: tuple[ExternalPairwiseSeparationFinding, ...]
    stratum_scores: tuple[ExternalStratumScore, ...]
    constraint_findings: tuple[ExternalConstraintFinding, ...]
    method_validation_hashes: tuple[str, ...]
    manifest: ExperimentManifest
```

There is no `conclusions` field and no identification field.

- [ ] **7.3 Implement `build_external_validation_report()`**

```python
def build_external_validation_report(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    final_evaluation: ExternalFinalEvaluation,
    constraint_findings: tuple[ExternalConstraintFinding, ...] = (),
) -> ExternalValidationReport: ...
```

Builder requirements:

1. exact evidence/preregistration hash match;
2. final preflight matches both preregistered sibling protocols/evidence;
3. constraint finding plan hashes equal exactly the declared constraint-plan hashes, with no missing/extra finding;
4. method-validation hashes copy exactly from preregistration and remain opaque;
5. claim scope is exact;
6. all P3 statuses are instances of the P3 enums;
7. both child released-comparison manifest hashes are present;
8. canonical parent lineage is the union of both child comparison manifest hashes, all constraint parent manifest hashes, and all method-validation hashes.

Create the manifest as:

```python
ExperimentManifest(
    stage=ExperimentStage.EXTERNAL_VALIDATION,
    inputs={
        "preregistration_hash": preregistration.content_hash,
        "evidence_declaration_hash": evidence.content_hash,
        "claim_scope": EXTERNAL_CLAIM_SCOPE,
        "brier_protocol_hash": final_evaluation.preflight.brier_protocol_hash,
        "log_protocol_hash": final_evaluation.preflight.log_protocol_hash,
        "preflight_hash": final_evaluation.preflight.content_hash,
        "method_validation_hashes": preregistration.method_validation_hashes,
        "constraint_plan_hashes": tuple(
            plan.content_hash for plan in preregistration.constraint_plans
        ),
    },
    parent_hashes=canonical_parent_hashes,
)
```

`ExternalValidationReport.__post_init__` validates fixed claim scope, content-hash syntax, P3 status types, deterministic collections, and manifest stage/input consistency that can be checked from fields. Cross-object semantic matching remains in the builder, where all source objects are available.

- [ ] **7.4 Reuse generic report attestation unchanged**

Do not modify `narrative_dynamics/report_artifact.py`. Require:

```python
attested = attest_report(report)
assert attested.require_integrity() is report
```

Construct a forged/tampered dataclass replacement and prove the original artifact identity no longer validates that altered payload.

- [ ] **7.5 Export intended final P3 API**

Add report/finding/build/evaluation/preflight/error symbols to package root. Do not export any P2 identification symbol as a P3 symbol.

- [ ] **7.6 Run report + full focused P3 GREEN**

```bash
python3 -m unittest tests.test_external_validation_reporting -v
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: all P3 tests GREEN.

- [ ] **7.7 Run exact existing artifact/P2/manifest regressions**

```bash
python3 -m unittest \
  tests.test_report_artifact_policy \
  tests.test_narrative_identification_reporting \
  tests.test_experiment_manifest -v
```

Expected: GREEN; P2 identification semantics remain unchanged.

- [ ] **7.8 Commit**

```bash
git add \
  narrative_dynamics/contracts.py \
  narrative_dynamics/external_validation.py \
  narrative_dynamics/__init__.py
git commit -m "feat: attest external predictive validation reports"
```

---

## Task 8 — Full regression, scope audit, exact-head proof, and review

**Files:** no planned production changes. Any regression defect caused by Tasks 2-7 gets its own focused RED before a fix.

- [ ] **8.1 Run focused P3 suite**

```bash
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: GREEN.

- [ ] **8.2 Run complete Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests GREEN. Record exact test count and `OK`.

- [ ] **8.3 Run Lean/Python conformance exactly as workflow**

```bash
lake build \
  NarrativeDynamics.Core.Belief \
  NarrativeDynamics.Core.Drive \
  NarrativeDynamics.Core.Learning
mkdir -p .generated-conformance
lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean \
  > .generated-conformance/lean_reference_vectors.json
cmp conformance/lean_reference_vectors.json \
  .generated-conformance/lean_reference_vectors.json
```

Expected: exact byte-for-byte conformance.

- [ ] **8.4 Run full Lean and theorem gates**

```bash
lake build
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Expected: GREEN. No Lean source changed.

- [ ] **8.5 Audit diff scope and forbidden production dependencies**

```bash
git diff --check
git diff --name-only f6dbe9e67d6dda064fa332989187e7e8a604039a...HEAD
git grep -n \
  "IdentificationStatus\|diagnose_identifiability\|ParameterIdentificationFinding" \
  -- narrative_dynamics/observations/external.py \
     narrative_dynamics/external_validation.py || true
git status --short
```

Expected production diff is limited to the five approved production files. The grep returns no P2 identification dependency in P3 production modules.

- [ ] **8.6 Prove no fake empirical fixture or Docker scope entered**

```bash
if git diff --name-only \
  f6dbe9e67d6dda064fa332989187e7e8a604039a...HEAD \
  | grep -E '^(fixtures/observations/|.*Dockerfile|.*docker|.*compose)'; then
  echo "unexpected fixture/docker scope"
  exit 1
fi
```

Expected: success with no matched path.

- [ ] **8.7 Push exact feature head and require `proof` GREEN**

```bash
git push origin work/narrative-empirical-external-validation-v1
```

Record:

```text
feature head SHA
proof run number
Lean conformance result
full Lean result
Python test count / OK
StoryState GREEN
Testimony GREEN
```

Do not claim P3 GREEN before the exact-head `proof` run passes. Do not run Docker acceptance.

- [ ] **8.8 Perform final scope/research review**

Verify explicitly:

```text
predictive fit never becomes latent identification
positive external origin is mandatory
known synthetic fixtures are rejected
Brier/Log siblings freeze all non-score identities
score-specific thresholds are frozen before final evaluation
both releases verify before either final execution
strata do not resimulate
constraints are selection-only and use Brier/Log compatible-set intersection
method-validation evidence is lineage-only
aggregate report is attested and claim-safe
no P0-P2 semantic drift
```

- [ ] **8.9 Invoke `superpowers:finishing-a-development-branch`**

Only after all gates above are GREEN. Choose the integration path there. Do not close #38 or call P3 integrated until the selected integration path completes and the integrated post-merge head has its own exact-head `proof` GREEN evidence.

## Definition of done

P3 V1 is done only when the integrated post-merge head satisfies all twelve conditions:

1. positive external observational origin is explicitly bound to source snapshot, transform identity, record namespace, dataset hash, and partition assignment;
2. known synthetic fixtures cannot enter the P3 external path;
3. one Brier and one Log sibling protocol are frozen with exact non-score identity and explicit score-specific thresholds;
4. both witnessed/verified releases pass P3 preflight before either final model execution;
5. the same frozen candidates are evaluated on the same final targets/seeds under both proper scores;
6. global adequacy and pairwise observable separation use protocol-scoped predictive vocabulary only;
7. preregistered strata reuse existing per-case final losses and trigger no extra simulation;
8. optional finite parameter constraints use selection-validation only, intersect Brier/Log compatible sets, retain full candidate-loss evidence, and never emit identification language;
9. optional P2 method-validation hashes remain opaque lineage and transfer no P2 status;
10. `ExternalValidationReport` has `claim_scope == "external_observational_predictive_only"`, no identification/free-text conclusion field, `ExperimentStage.EXTERNAL_VALIDATION`, complete lineage, and passes generic attestation;
11. existing P0-P2 APIs and semantics remain unchanged;
12. complete Python, Lean/conformance, StoryState, Testimony, and exact-head GitHub `proof` gates are GREEN, with Docker acceptance not run.
