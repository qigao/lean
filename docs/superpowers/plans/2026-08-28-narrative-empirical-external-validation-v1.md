# Narrative Empirical / External Validation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement #38 P3 Empirical / External Validation V1 as a thin, fail-closed bridge from positively declared external observational evidence into the existing train/selection/final, preregistration, witnessed-release, proper-scoring, and report-attestation machinery while making it impossible for predictive fit to become a canonical claim that real latent cognition was identified.

**Architecture:** Add `narrative_dynamics.observations.external` for positive external-evidence declarations and source/partition lineage, and `narrative_dynamics.external_validation` for claim-safe statuses, Brier/Log sibling preregistration, dual verified-release preflight, final predictive findings, strata, optional selection-only parameter constraints, and the aggregate attested report. Reuse `ObservationDataset`, target construction, `calibrate_grid`, `PreregisteredEvaluationProtocol`, `ProtocolRelease` / `VerifiedProtocolRelease`, `compare_released_models()`, and `attest_report()` exactly as existing lower layers. Do not add an empirical mode to P2 identification.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `enum`, `math`, `statistics`, `MappingProxyType`, `unittest`, `pathlib`), existing `narrative_dynamics` observation/calibration/validation/loss/model-comparison/release/manifest/report-artifact APIs, GitHub Actions `proof.yml`, Lean/lake only for regression verification.

**Spec:** `docs/superpowers/specs/2026-08-28-narrative-empirical-external-validation-v1-design.md`

## Global Constraints

- Issue: #38 — `research: P3 empirical / external validation v1`.
- Integrated research base: `proof/narrative-dynamics-v0@f6dbe9e67d6dda064fa332989187e7e8a604039a`.
- Approved written-spec head: `9758b8f7d6088093418826610768c8a73e13943f`.
- Implementation branch after this plan is approved: `work/narrative-empirical-external-validation-v1`, cut from the plan commit so the approved spec/plan travel with the feature branch.
- Docs-only spec/plan commits are ignored by `proof.yml`; the first authoritative feature evidence is the complete test-only RED commit.
- Strict sequence: test-only RED -> focused exact RED -> monotonic task-local GREEN commits -> focused P3 GREEN -> complete Python GREEN -> Lean/conformance/story/testimony regression GREEN -> exact-head GitHub proof GREEN -> review -> finishing-development-branch integration choice.
- Do not run Docker acceptance. P3 V1 is Python research-protocol work and the approved spec explicitly does not require Docker.
- Production changes are limited to:
  - create `narrative_dynamics/observations/external.py`;
  - create `narrative_dynamics/external_validation.py`;
  - add only `ExperimentStage.EXTERNAL_VALIDATION` to `narrative_dynamics/contracts.py`;
  - additive public exports in `narrative_dynamics/observations/__init__.py` and `narrative_dynamics/__init__.py`.
- Do not modify `narrative_dynamics/identification.py`, `narrative_dynamics/diagnostics.py`, `calibration.py`, `validation.py`, `model_comparison.py`, `observations/dataset.py`, `observations/targets.py`, `observations/preregistration.py`, `observations/release.py`, any Reactive/Intentional/Planning runtime implementation, stochastic world/observation semantics, existing observational fixtures, or Lean sources.
- The canonical external claim scope is exactly `external_observational_predictive_only`.
- P3 public results expose no `IdentificationStatus` and no `identification_status` field.
- Allowed P3 statuses are exactly:
  - `predictive_adequacy_met` / `predictive_adequacy_not_met`;
  - `predictively_separated_under_protocol` / `not_predictively_separated_under_protocol`;
  - `constrained_under_external_protocol` / `not_constrained_under_external_protocol`.
- Canonical P3 result/status/claim fields reject values equivalent to `EMPIRICALLY_IDENTIFIED`, `STRUCTURALLY_IDENTIFIED`, `COGNITIVE_MECHANISM_CONFIRMED`, `LATENT_COGNITION_IDENTIFIED`, or `REAL_PERSON_GOAL_RECOVERED`.
- `source.kind == "external_observational"` and `provenance.external_observational is True` are both required for the external path. Absence of a synthetic marker is not sufficient.
- Known synthetic markers `source.kind == "synthetic_fixture"` or `provenance.synthetic_non_empirical is True` fail closed.
- `provenance.empirical_human_data is False` alone is not a synthetic marker and must not reject a positively declared external non-human dataset.
- The existing `fixtures/observations/prison_initial_choice_v1.json` and `fixtures/observations/narrative_identification_v1.json` are rejection fixtures only. Do not relabel either as external or empirical.
- Do not commit a fabricated repository fixture presented as real empirical evidence. Positive external data used by unit tests is constructed in test code with an explicit `test_only` provenance marker.
- Brier and Log are sibling final protocols. All non-score-specific identities are exact; score-specific thresholds may differ only because both threshold sets were frozen in the P3 preregistration before final evaluation.
- Both sibling `ProtocolRelease` values and both `VerifiedProtocolRelease` values must pass P3 preflight before either call to `compare_released_models()` occurs.
- A P3 release `source_revision` contains at least `repository_revision`, `external_evidence_declaration_hash`, `external_validation_preregistration_hash`, and `score_role`.
- Global and stratum results reuse the per-case losses already produced by existing released final comparisons. Stratum reporting does not re-run a model and introduces no post-hoc weights.
- External parameter constraints consume only `SELECTION_VALIDATION` target cases. They never consume `FINAL_TEST`, generator truth, P2 `IdentificationStatus`, `diagnose_identifiability()`, or `ParameterIdentificationFinding`.
- Constraint compatibility is the intersection of Brier and Log compatible sets, where each score accepts `candidate_loss <= best_loss + preregistered_delta`. Empty intersection is a typed failure, not a non-constrained result.
- A singleton compatible parameter set is `CONSTRAINED_UNDER_EXTERNAL_PROTOCOL`; it is never called identified.
- `method_validation_hashes` are opaque lineage hashes only. P2 findings may be parents but no P2 status/conclusion is copied into a P3 finding.
- The final `ExternalValidationReport` has no free-text canonical conclusion field and must pass `attest_report(report).require_integrity()`.

## File Structure

- `narrative_dynamics/observations/external.py` — P3 root error, positive external evidence declaration, source/transform/namespace/partition-assignment identity, dataset match checks.
- `narrative_dynamics/external_validation.py` — claim/status enums; score role; sibling preregistration; strata/separation/constraint declarations; release preflight; dual final orchestration; adequacy/separation/stratum findings; selection-only constraint evaluation; aggregate report.
- `narrative_dynamics/contracts.py` — add only `ExperimentStage.EXTERNAL_VALIDATION = "external_validation"`.
- `narrative_dynamics/observations/__init__.py` — additive external evidence exports.
- `narrative_dynamics/__init__.py` — additive intended P3 public exports.
- `tests/external_validation_fixtures.py` — test-only external dataset/protocol/models/releases helper; not collected as a test module.
- `tests/test_external_validation_claims.py` — claim vocabulary/firewall REDs.
- `tests/test_external_evidence.py` — positive origin, synthetic rejection, deterministic lineage REDs.
- `tests/test_external_validation_preregistration.py` — sibling protocol, thresholds, strata, separation, constraint-plan REDs.
- `tests/test_external_validation_release_gate.py` — sibling release binding and no-execution-before-both-verified REDs.
- `tests/test_external_validation_final.py` — dual final evaluation, adequacy, separation, strata/no-resimulation REDs.
- `tests/test_external_validation_constraints.py` — selection-only compatible-set intersection REDs.
- `tests/test_external_validation_reporting.py` — aggregate lineage, manifest, attestation, method-evidence isolation, claim-firewall REDs.

---

### Task 1: Commit the Complete Test-Only P3 RED Boundary

**Files:**
- Create: `tests/external_validation_fixtures.py`
- Create: `tests/test_external_validation_claims.py`
- Create: `tests/test_external_evidence.py`
- Create: `tests/test_external_validation_preregistration.py`
- Create: `tests/test_external_validation_release_gate.py`
- Create: `tests/test_external_validation_final.py`
- Create: `tests/test_external_validation_constraints.py`
- Create: `tests/test_external_validation_reporting.py`

**Intent:** Freeze the entire P3 acceptance surface before any P3 production symbol exists. Guard new imports so unittest discovery reports the intended missing boundary rather than aborting after the first import error.

- [ ] **Step 1: Add shared test-only external fixtures**

Create `tests/external_validation_fixtures.py` with a positively declared, explicitly test-only `ObservationDataset`, two simple policy models, canonical target/loss builders, and witness helpers. Use code equivalent to:

```python
from __future__ import annotations

from narrative_dynamics.contracts import ModelRun, Scenario, stable_content_hash
from narrative_dynamics.losses import CategoricalBrierLoss, CategoricalLogLoss, CategoricalMetricGroup
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationDataset,
    ObservationPartition,
    ObservationPartitionRole,
    ObservationRecord,
    PreregisteredEvaluationProtocol,
)

FINAL_SEEDS = (41, 42)
SELECTION_SEEDS = (31, 32)
GROUP = CategoricalMetricGroup("choice", ("choice.a", "choice.b"))


def policy_metrics(trace):
    policy = trace.outcome["policy"]
    return {"choice.a": float(policy["a"]), "choice.b": float(policy["b"])}
policy_metrics.version = "external-validation-v1"


class ProbabilityModel:
    name = "external-probability-model"
    version = "1"
    implementation_revision = "external-probability-v1"

    def __init__(self):
        self.calls = 0

    def simulate(self, scenario, parameters, rng):
        self.calls += 1
        p = float(parameters["p"])
        return ModelRun(events=(), outcome={"policy": {"a": p, "b": 1.0 - p}})


class AlternativeProbabilityModel(ProbabilityModel):
    name = "external-alternative-model"
    implementation_revision = "external-alternative-v1"


def external_dataset() -> ObservationDataset:
    def record(record_id, scenario_id, a, b):
        return ObservationRecord(
            id=record_id,
            scenario=Scenario(id=scenario_id, payload={"condition": scenario_id}),
            counts={"a": a, "b": b},
            metadata={"test_only": True},
        )

    return ObservationDataset(
        name="external-observations-test-only",
        version="1",
        source={"kind": "external_observational", "release": "test-snapshot-v1"},
        provenance={"external_observational": True, "test_only": True},
        partitions=(
            ObservationPartition("train", ObservationPartitionRole.TRAIN, records=(record("ext:train:1", "train-1", 7, 3),)),
            ObservationPartition("selection", ObservationPartitionRole.SELECTION_VALIDATION, records=(record("ext:selection:1", "selection-1", 8, 2),)),
            ObservationPartition("final", ObservationPartitionRole.FINAL_TEST, records=(
                record("ext:final:1", "final-1", 9, 1),
                record("ext:final:2", "final-2", 6, 4),
            )),
        ),
    )


def target_spec():
    return CategoricalTargetSpec(name="external-choice", version="1", categories=("a", "b"), metric_prefix="choice")


def brier_loss():
    return CategoricalBrierLoss((GROUP,))


def log_loss():
    return CategoricalLogLoss((GROUP,))
```

Add helpers that create frozen candidates and sibling `PreregisteredEvaluationProtocol` values with the same dataset/partitions/target/extractor/seeds/baseline/candidate identities, Brier thresholds `AdequacyThresholds(1.0, 1.0)`, and Log thresholds `AdequacyThresholds(3.0, 3.0)`. Use separate runtime model instances when executing the two score siblings so score order cannot leak mutable test-model state.

- [ ] **Step 2: Add claim-firewall RED tests**

`tests/test_external_validation_claims.py` must lock:

```text
test_claim_scope_is_exact_and_immutable
test_public_status_vocabularies_are_closed
test_external_validation_exposes_no_identification_status
test_forbidden_empirical_identification_values_are_not_valid_statuses
```

Guard the new module import:

```python
try:
    import narrative_dynamics.external_validation as external_validation
except ImportError as error:
    external_validation = None
    _IMPORT_ERROR = error
```

Each test first fails with a clear message if the module is absent. Later tests must assert `not hasattr(external_validation, "IdentificationStatus")`.

- [ ] **Step 3: Add evidence-origin RED tests**

`tests/test_external_evidence.py` must lock:

```text
test_positive_external_origin_builds_stable_declaration
test_absence_of_positive_external_origin_is_rejected
test_prison_synthetic_fixture_is_rejected
test_p2_synthetic_identification_fixture_is_rejected
test_empirical_human_false_alone_does_not_mark_positive_external_data_synthetic
test_partition_assignment_hash_is_order_stable_and_role_sensitive
test_dataset_source_transform_namespace_or_partition_drift_is_rejected
```

Load the two existing synthetic fixtures using `load_observation_dataset()` and require `ExternalValidationError` on `ExternalEvidenceDeclaration.from_dataset(...)`.

- [ ] **Step 4: Add preregistration RED tests**

`tests/test_external_validation_preregistration.py` must lock:

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

- [ ] **Step 5: Add dual-release gate RED tests**

`tests/test_external_validation_release_gate.py` must reuse the existing `ProtocolRelease`, `WitnessReceipt`, and `verify_protocol_release()` seam and lock:

```text
test_release_source_revision_must_bind_evidence_preregistration_repository_and_score_role
test_brier_release_cannot_be_substituted_for_log_release
test_verified_release_hash_must_match_the_actual_release_hash
test_one_unverified_or_drifted_sibling_prevents_both_final_executions
test_both_verified_siblings_produce_stable_preflight_identity
```

The no-execution test uses `.calls` counters and requires the total to remain zero when the Log sibling is invalid, even if the Brier sibling is valid.

- [ ] **Step 6: Add final predictive-evaluation RED tests**

`tests/test_external_validation_final.py` must lock:

```text
test_dual_final_uses_existing_released_comparisons_for_both_scores
test_per_score_and_aggregate_predictive_adequacy_are_claim_safe
test_pairwise_separation_requires_same_direction_and_both_delta_thresholds
test_nonseparation_preserves_both_raw_score_deltas
test_stratum_scores_use_existing_case_losses_without_resimulation
test_global_scores_are_unchanged_by_stratum_declarations
```

The no-resimulation assertion records model `.calls` immediately after both released comparisons and asserts constructing all stratum findings leaves the call count unchanged.

- [ ] **Step 7: Add selection-only constraint RED tests**

`tests/test_external_validation_constraints.py` must lock:

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

For the last test, patch `narrative_dynamics.diagnostics.diagnose_identifiability` to raise if called; P3 constraint evaluation must still complete because it never invokes that function.

- [ ] **Step 8: Add aggregate-report RED tests**

`tests/test_external_validation_reporting.py` must lock:

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

- [ ] **Step 9: Run the complete P3 RED suite**

```bash
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: failures are only the newly required P3 boundary (missing `narrative_dynamics.observations.external`, missing `narrative_dynamics.external_validation`, missing `ExperimentStage.EXTERNAL_VALIDATION`, and downstream missing P3 symbols). Existing imported P0-P2 infrastructure must not fail.

- [ ] **Step 10: Commit the authoritative RED**

```bash
git add tests/external_validation_fixtures.py \
  tests/test_external_validation_claims.py \
  tests/test_external_evidence.py \
  tests/test_external_validation_preregistration.py \
  tests/test_external_validation_release_gate.py \
  tests/test_external_validation_final.py \
  tests/test_external_validation_constraints.py \
  tests/test_external_validation_reporting.py
git commit -m "test: require empirical external validation v1"
```

Push this exact test-only head and record the failing `proof` run before any production code is committed.

---

### Task 2: GREEN the Claim and External-Evidence Firewall

**Files:**
- Create: `narrative_dynamics/observations/external.py`
- Create: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/observations/__init__.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Implement the single P3 root error and external declaration**

Keep dependency direction acyclic by defining the shared root error in `observations.external`; `external_validation` imports and re-exports it.

```python
EXTERNAL_EVIDENCE_ORIGIN = "external_observational"
EXTERNAL_CLAIM_SCOPE = "external_observational_predictive_only"

class ExternalValidationError(ValueError):
    pass

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

Provide:

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

`from_dataset()` must require both positive markers, reject the two known synthetic markers, validate a SHA-256 snapshot hash, freeze mappings with the existing canonical helper from `observations.dataset`, and compute `partition_assignment_hash` from the complete sorted tuple:

```text
(role.value, source_observation_id)
```

For record-oriented partitions, source IDs are `ObservationRecord.id`. For case-oriented partitions, expand every `ObservationCase.observation_ids` value. `require_matches()` recomputes dataset and partition-assignment identities and rejects drift.

- [ ] **Step 2: Implement only the claim vocabulary in `external_validation.py`**

```python
from .observations.external import EXTERNAL_CLAIM_SCOPE, ExternalValidationError

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

Do not import `narrative_dynamics.identification` or define `IdentificationStatus`.

- [ ] **Step 3: Add intended exports only**

Expose `ExternalEvidenceDeclaration`, `ExternalValidationError`, and the evidence constants through `narrative_dynamics.observations`. Expose the claim/status enums and intended evidence type through package root without renaming any existing symbol.

- [ ] **Step 4: Run claim/evidence tests**

```bash
python3 -m unittest \
  tests.test_external_validation_claims \
  tests.test_external_evidence -v
```

Expected: GREEN. The remaining P3 modules/tests stay RED because preregistration/orchestration/report symbols are intentionally absent.

- [ ] **Step 5: Run existing observation/release regression**

```bash
python3 -m unittest \
  tests.test_observation_dataset \
  tests.test_observation_targets \
  tests.test_protocol_release -v
```

Expected: GREEN.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/observations/external.py \
  narrative_dynamics/external_validation.py \
  narrative_dynamics/observations/__init__.py \
  narrative_dynamics/__init__.py
git commit -m "feat: add external evidence claim firewall"
```

---

### Task 3: GREEN the Frozen Brier/Log External Preregistration

**Files:**
- Modify: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Add typed protocol errors and declaration types**

```python
class ExternalValidationProtocolError(ExternalValidationError):
    pass

class ExternalValidationConstraintError(ExternalValidationError):
    pass

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

Validation rules: finite/non-negative acceptance deltas, finite non-empty grid dimensions, unique parameter/coordinate/case names, non-empty seeds, canonical sorted parameter dimensions and coordinates, and no final-test case notion in the type.

- [ ] **Step 2: Implement `ExternalValidationPreregistration`**

```python
@dataclass(frozen=True)
class ExternalValidationPreregistration:
    name: str
    version: str
    evidence_declaration_hash: str
    brier_protocol_hash: str
    log_protocol_hash: str
    adequacy_thresholds_by_score: tuple[tuple[ExternalScoreRole, AdequacyThresholds], ...]
    strata: tuple[ExternalStratum, ...]
    separation_rule: PairwiseSeparationRule
    constraint_plans: tuple[ExternalConstraintPlan, ...]
    method_validation_hashes: tuple[str, ...]
    claim_scope: str = EXTERNAL_CLAIM_SCOPE
```

Use a `create(...)` constructor that receives the actual `ExternalEvidenceDeclaration`, Brier/Log `PreregisteredEvaluationProtocol`, the actual final `TargetConstructionReport`, the two threshold sets, strata, separation rule, optional constraint plans, and method hashes.

The constructor must:

1. require evidence hash == both protocol dataset lineage via `evidence.dataset_hash`;
2. require `final_targets.content_hash` == both sibling `final_target_hash`;
3. identify Brier by `loss_identity["name"] == "categorical_brier"` and Log by `"categorical_log"`;
4. compare sibling protocol identity payloads after removing only `name`, `loss_identity`, `thresholds`, and derived precommitment/hash fields;
5. separately require metric identity, dataset/partition/target hashes, seeds, baseline, candidate names/model identities/parameters/selection manifest hashes, and protocol version exact;
6. require each protocol threshold object equals its preregistered score-specific threshold set;
7. require exactly one Brier and one Log score role in canonical Brier-then-Log order;
8. require strata cover `tuple(case.name for case in final_targets.cases)` exactly once with no missing, extra, or overlap;
9. require unique constraint-plan names;
10. validate every method-validation hash as `sha256:<64 lowercase hex>`;
11. compute deterministic `content_hash` from semantic declaration only.

Do not compare Brier and Log threshold numbers to each other; compare each one only with its own P3 frozen threshold set.

- [ ] **Step 3: Export intended protocol types**

Add the new P3 declarations to package root `__all__` without changing existing exports.

- [ ] **Step 4: Run preregistration tests**

```bash
python3 -m unittest tests.test_external_validation_preregistration -v
```

Expected: GREEN.

- [ ] **Step 5: Run observation preregistration/model-comparison regressions**

```bash
python3 -m unittest \
  tests.test_observational_data_protocol \
  tests.test_preregistered_model_comparison \
  tests.test_protocol_release -v
```

Expected: GREEN.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: freeze external validation sibling protocols"
```

---

### Task 4: GREEN the Dual Verified-Release Preflight

**Files:**
- Modify: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Add a frozen preflight result**

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

    @property
    def content_hash(self) -> str:
        return stable_content_hash(self.identity_payload())
```

- [ ] **Step 2: Implement `preflight_external_releases()` before any orchestration**

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

Required order:

1. validate P3 object types;
2. require `evidence.content_hash == preregistration.evidence_declaration_hash`;
3. require sibling protocol hashes equal the preregistered hashes;
4. call `brier_verified.require_matches(brier_protocol)` and `log_verified.require_matches(log_protocol)`;
5. additionally require `brier_verified.release_hash == brier_release.content_hash` and same for Log because `VerifiedProtocolRelease.require_matches()` itself only checks protocol identity;
6. require each existing release's dataset/target/candidate identities match its protocol;
7. read `source_revision` and require the four P3 binding keys;
8. require both sibling bindings share exact `repository_revision`, `external_evidence_declaration_hash`, and `external_validation_preregistration_hash`;
9. require Brier role == `brier`, Log role == `log`;
10. return the frozen preflight identity.

Existing `ProtocolReleaseVerificationError` from the lower release-verification API must propagate unchanged. P3 binding/drift errors raise `ExternalValidationProtocolError`.

- [ ] **Step 3: Run release gate tests**

```bash
python3 -m unittest tests.test_external_validation_release_gate -v
```

Expected: GREEN for pure preflight cases; the orchestration no-execution test may still be RED until Task 5 if it targets the final entry point.

- [ ] **Step 4: Run existing release regression**

```bash
python3 -m unittest tests.test_protocol_release -v
```

Expected: GREEN with no modifications to `observations/release.py`.

- [ ] **Step 5: Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: gate external validation on dual verified releases"
```

---

### Task 5: GREEN Dual Final Predictive Evaluation, Separation, and Strata

**Files:**
- Modify: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Add claim-safe final finding dataclasses**

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

All collections use deterministic score/model/stratum ordering. These types contain no free-text conclusion and no identification field.

- [ ] **Step 2: Implement pure finding builders from existing child reports**

Adequacy uses each `ModelComparisonEntry.adequate`, because Task 3 guarantees the child's threshold set exactly equals the P3 score-specific threshold set. Aggregate adequacy is MET only if both sibling findings for a model are MET.

Pairwise separation uses `entry.mean_loss` from the two existing child reports. For lexical pair `(A, B)`, define signed delta as `loss(A) - loss(B)`. The scores separate only when:

```text
sign(brier_delta) == sign(log_delta) != 0
abs(brier_delta) >= min_mean_loss_delta_brier
abs(log_delta) >= min_mean_loss_delta_log
```

Lower loss is preferred. Otherwise status is NOT_SEPARATED and `preferred_model` is `None` unless both scores agree on direction but only magnitude fails; in that case preserving an observational preference field is allowed only if tests and spec treat it as non-status metadata. Do not call it a winner.

Stratum aggregation selects `HeldOutCaseEvaluation.loss` values already present at:

```python
released.comparison.entry_map[model].final_test.validation.cases
```

and calculates `fmean` / `max`. Do not call `SimulationRunner` from a stratum helper.

- [ ] **Step 3: Implement `evaluate_external_final()` with full preflight before first child execution**

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

The first executable line after type/materialization checks is one call to `preflight_external_releases(...)`. Only after it returns may either child `compare_released_models()` run. Then execute Brier and Log with the same `final_targets` and extractor, each using its own protocol/loss and a separate tuple of runtime model instances bound to the same frozen candidates.

- [ ] **Step 4: Prove an invalid second sibling prevents the first sibling from running**

Run:

```bash
python3 -m unittest \
  tests.test_external_validation_release_gate \
  tests.test_external_validation_final -v
```

Expected: GREEN. The invalid-Log test must observe zero calls on both Brier and Log model instances.

- [ ] **Step 5: Run existing final-comparison regressions**

```bash
python3 -m unittest \
  tests.test_preregistered_model_comparison \
  tests.test_protocol_release \
  tests.test_narrative_held_out_model_comparison -v
```

If the last module name differs in the repository, use the existing test module that owns Narrative Held-Out Three-Model Comparison V1; do not change production behavior to satisfy a guessed test path.

Expected: GREEN.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: evaluate external predictive evidence"
```

---

### Task 6: GREEN Selection-Only External Parameter Constraints

**Files:**
- Modify: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Add auditable constraint evidence types**

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

There is deliberately no `true_parameters`, `generator_family`, `identification_status`, or P2 finding field.

- [ ] **Step 2: Implement a score-grid helper using existing `calibrate_grid()`**

For one score role:

1. require `target_set.role is ObservationPartitionRole.SELECTION_VALIDATION` before any model execution;
2. require target-set dataset/spec lineage matches the P3 sibling protocol/plan;
3. select exactly the case names declared in the plan, with no missing/extra case execution;
4. for each selected case call existing `calibrate_grid()` with the plan's finite grid and seeds;
5. require every case returns the same canonical candidate set;
6. aggregate a candidate's case losses with `fmean` across the declared selection cases;
7. retain every child calibration manifest hash and every candidate loss.

Do this once with canonical Brier loss and once with canonical Log loss. The model source may be a fresh instance per score to avoid mutable state crossing scores.

- [ ] **Step 3: Implement compatible-set intersection and constraint interpretation**

For each score:

```python
best = min(candidate.mean_loss for candidate in score_table)
compatible = {
    candidate.parameters
    for candidate in score_table
    if candidate.mean_loss <= best + preregistered_delta
}
```

Then:

```python
intersection = brier_compatible & log_compatible
```

If empty, raise `ExternalValidationConstraintError`.

For each target coordinate, collect the sorted distinct values in `intersection`; one retained value means that coordinate is constrained. Aggregate status is `CONSTRAINED` only when every declared coordinate is constrained, otherwise `NOT_CONSTRAINED`.

Do not call `diagnose_identifiability()` and do not create a `ParameterAcceptanceSet` merely to borrow P2 terminology.

- [ ] **Step 4: Implement the public constraint evaluator**

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

Require `component_identity(model)` equals the plan's frozen model identity and both score/metric identities are exact before execution.

- [ ] **Step 5: Run constraint tests**

```bash
python3 -m unittest tests.test_external_validation_constraints -v
```

Expected: GREEN, including singleton->CONSTRAINED, multi-value->NOT_CONSTRAINED, empty intersection typed failure, final-test role rejected before calls, and patched P2 diagnostics never invoked.

- [ ] **Step 6: Run calibration/validation regressions**

```bash
python3 -m unittest \
  tests.test_calibration \
  tests.test_observational_training_fit \
  tests.test_validation -v
```

Use the repository's actual validation test module names if discovery shows a split suite; do not alter production APIs solely to match a guessed module path.

- [ ] **Step 7: Commit**

```bash
git add narrative_dynamics/external_validation.py narrative_dynamics/__init__.py
git commit -m "feat: constrain external parameters without identification"
```

---

### Task 7: GREEN the Attested Aggregate External Validation Report

**Files:**
- Modify: `narrative_dynamics/contracts.py`
- Modify: `narrative_dynamics/external_validation.py`
- Modify: `narrative_dynamics/__init__.py`

- [ ] **Step 1: Add exactly one manifest stage**

In `ExperimentStage` add only:

```python
EXTERNAL_VALIDATION = "external_validation"
```

No manifest schema version change.

- [ ] **Step 2: Add report-specific error and report dataclass**

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

The type has no `conclusions`, `identification_status`, or arbitrary status string field.

- [ ] **Step 3: Implement `build_external_validation_report()`**

```python
def build_external_validation_report(
    *,
    preregistration: ExternalValidationPreregistration,
    evidence: ExternalEvidenceDeclaration,
    final_evaluation: ExternalFinalEvaluation,
    constraint_findings: tuple[ExternalConstraintFinding, ...] = (),
) -> ExternalValidationReport: ...
```

Validation must require:

1. exact evidence/preregistration hashes;
2. final preflight matches both preregistered protocols/evidence;
3. exactly the declared constraint-plan hashes have findings — no missing or extra finding;
4. method-validation hashes copied exactly from preregistration and treated only as hashes;
5. claim scope exact and immutable;
6. no P2 status object accepted as any P3 status field;
7. child released-comparison manifest hashes present;
8. parent lineage is the canonical union of both child comparison manifest hashes, all constraint parent manifest hashes, and method-validation hashes.

Create:

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
        "constraint_plan_hashes": tuple(plan.content_hash for plan in preregistration.constraint_plans),
    },
    parent_hashes=canonical_parent_hashes,
)
```

The dataclass's `__post_init__` must reject forged claim scope, prereg/evidence/preflight mismatch, incomplete child identity, or incomplete parent set. Do not generate narrative prose conclusions.

- [ ] **Step 4: Verify generic report attestation without special-casing**

`attest_report()` already canonicalizes arbitrary dataclass reports and excludes `manifest`; do not modify `report_artifact.py`. Test:

```python
attested = attest_report(report)
assert attested.require_integrity() is report
```

Use `dataclasses.replace(report, evidence_declaration_hash=HASH0)` and confirm an artifact created from the original no longer matches the forged report. Construction-level invariants should also reject semantically inconsistent replacements when `__post_init__` can detect them.

- [ ] **Step 5: Export intended final P3 API**

Update package root exports for the report, findings, preregistration, preflight, evaluators, and errors. Do not export any P2 identification symbol through the P3 namespace.

- [ ] **Step 6: Run report and full focused P3 tests**

```bash
python3 -m unittest tests.test_external_validation_reporting -v
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: all P3 tests GREEN.

- [ ] **Step 7: Run report-artifact and P2 identification regressions**

```bash
python3 -m unittest \
  tests.test_report_artifact_policy \
  tests.test_narrative_identification_reporting -v
```

Expected: GREEN; P2 identification semantics are unchanged.

- [ ] **Step 8: Commit**

```bash
git add narrative_dynamics/contracts.py \
  narrative_dynamics/external_validation.py \
  narrative_dynamics/__init__.py
git commit -m "feat: attest external predictive validation reports"
```

---

### Task 8: Full Regression, Scope Audit, and Exact-Head Proof

**Files:**
- No planned production changes.
- Modify only if a regression exposes a defect caused by Tasks 2-7; any such correction must receive its own RED before the fix.

- [ ] **Step 1: Verify the focused P3 suite**

```bash
python3 -m unittest discover -s tests -p 'test_external_*.py' -v
```

Expected: all external validation tests GREEN.

- [ ] **Step 2: Verify the complete Python suite**

```bash
python3 -m unittest discover -s tests -v
```

Expected: all tests GREEN. Record the exact test count and `OK` output for the feature evidence.

- [ ] **Step 3: Verify Lean/Python conformance exactly as CI**

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

- [ ] **Step 4: Verify the Lean library and narrative theorem gates**

```bash
lake build
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

The normal CI theorem list remains authoritative; no Lean source should be changed by this feature.

- [ ] **Step 5: Audit scope and forbidden dependencies**

```bash
git diff --check
git diff --name-only f6dbe9e67d6dda064fa332989187e7e8a604039a...HEAD
git grep -n "IdentificationStatus\|diagnose_identifiability\|ParameterIdentificationFinding" -- \
  narrative_dynamics/observations/external.py \
  narrative_dynamics/external_validation.py || true
git status --short
```

Expected production diff is limited to the approved P3 files/exports/stage. The grep must find no P2 identification dependency in P3 production modules. The words may appear in negative tests/spec/plan and are not a production violation.

- [ ] **Step 6: Verify no fake empirical fixture or Docker change entered scope**

```bash
git diff --name-only f6dbe9e67d6dda064fa332989187e7e8a604039a...HEAD | \
  grep -E 'fixtures/observations|docker|Dockerfile|compose' && exit 1 || true
```

Expected: no production/external empirical fixture and no Docker changes.

- [ ] **Step 7: Push exact feature head and require `proof` GREEN**

Push `work/narrative-empirical-external-validation-v1`. The authoritative CI is `.github/workflows/proof.yml`; it runs Lean conformance, full Lean build/theorem tests, complete Python tests, StoryState, and Testimony. Do not run or wait for Docker acceptance.

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

Do not claim P3 GREEN before this exact-head run passes.

- [ ] **Step 8: Perform final review before integration**

Review specifically for:

```text
predictive fit never becomes latent identification
positive external origin is required
synthetic fixtures are rejected
Brier/Log siblings freeze all non-score identities
score-specific thresholds are preregistered
both releases verify before either final execution
strata do not resimulate
constraints are selection-only and dual-score intersection based
method evidence is lineage-only
aggregate report is attested and claim-safe
no P0-P2 semantic drift
```

If complete, invoke `superpowers:finishing-a-development-branch` for the integration choice. Do not close #38 or describe P3 as integrated until the chosen integration path and post-merge exact-head proof are complete.

## Definition of Done

P3 V1 is complete only when all of the following hold on the integrated post-merge head:

1. a positively declared external observational dataset can be bound to immutable source snapshot / transform / record namespace / partition lineage;
2. known synthetic fixtures cannot enter the external path;
3. one Brier and one Log sibling protocol are frozen with exact non-score identity and explicit score-specific thresholds;
4. both witnessed/verified releases preflight before any final model execution;
5. the same frozen candidates are evaluated on the same final targets/seeds under both proper scores;
6. global adequacy and pairwise observable separation are reported with protocol-scoped predictive vocabulary only;
7. preregistered strata reuse existing final per-case losses and trigger no extra simulation;
8. optional finite parameter constraints use selection-validation only, intersect Brier/Log compatible sets, retain the complete candidate-loss table, and never emit identification language;
9. optional P2 method-validation hashes remain opaque lineage and do not transfer P2 statuses;
10. `ExternalValidationReport` has `claim_scope == "external_observational_predictive_only"`, no identification status/conclusion field, `ExperimentStage.EXTERNAL_VALIDATION`, complete lineage, and passes generic attestation;
11. existing P0-P2 APIs and semantics remain unchanged;
12. complete Python, Lean/conformance, StoryState, Testimony, and exact-head GitHub `proof` gates are GREEN; Docker acceptance is not run.
