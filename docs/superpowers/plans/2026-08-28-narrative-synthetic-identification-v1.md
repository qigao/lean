# Narrative Synthetic Identification V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete #27 P2 Identification and Model Comparison with a preregistered synthetic protocol that can recover selected intentional latent parameters when the design identifies them, report explicit non-identifiability when observational equivalence remains, certify information-only interventions that separate model families, and compare frozen Reactive/Intentional/Planning candidates on the same held-out cases with Brier and log scores.

**Architecture:** Add one generic `narrative_dynamics.identification` orchestration/report module and one isolated synthetic benchmark adapter. The new layer reuses `SimulationRunner`, finite-grid calibration, accepted parameter sets, coordinate-wise identifiability, held-out selection/final evaluation, proper scoring, preregistration, unified narrative decision dispatch, and aggregate report attestation; it does not change those cores. Parameter identification is based on explicit accepted sets rather than `calibration.best`, model-family conclusions use equivalent/separated language rather than parameter-identification statuses, and the final Log protocol re-scores the exact Brier-selected candidates/targets/seeds rather than selecting a second model set.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `enum`, `math`, `statistics`, `MappingProxyType`, `unittest`, `json`, `pathlib`), existing `narrative_dynamics` simulation/calibration/uncertainty/validation/loss/preregistration/model-comparison/report-artifact APIs, GenericNarrative runtime Reactive/Intentional/Planning model families, unified `RuntimeDecisionModelSpec` / `run_runtime_decision()`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-28-narrative-synthetic-identification-v1-design.md`

## Global Constraints

- Integrated base is exactly `proof/narrative-dynamics-v0@f628dcea4176731e3fe7c5ec40170e63d0233759`.
- Feature branch is exactly `work/narrative-synthetic-identification-v1`.
- Approved written-spec head before planning is `5f703ec1da1faa96fdcd62e76cc7384387cb06a4`.
- Baseline post-merge proof #1158 is authoritative for the inherited state: exact head `f628dcea4176731e3fe7c5ec40170e63d0233759`, Python `659 tests` / `OK`, Lean and story/testimony gates GREEN.
- Docs-only spec/plan commits do not trigger `proof.yml`; the first authoritative feature proof is the complete test/fixture-only RED commit.
- Strict sequence: written plan -> complete test/fixture-only RED -> exact-head RED evidence -> task-local GREEN commits with monotonic failure reduction -> exact-head feature GREEN -> final scope/review -> finishing-development-branch integration choice -> PR synthetic-merge GREEN -> guarded merge -> post-merge exact-head GREEN -> roadmap #27 update.
- No production code is present in the authoritative RED commit.
- Do not run Docker. This feature requires no Docker acceptance.
- Production scope is limited to one additive `ExperimentStage` member in `narrative_dynamics/contracts.py`, one new generic module `narrative_dynamics/identification.py`, one new adapter `narrative_dynamics/adapters/narrative_prison_identification.py`, and only additive package exports if a public import test requires them.
- Test/data scope is exactly `fixtures/observations/narrative_identification_v1.json` plus five new test modules named in Task 1.
- Do not modify `narrative_dynamics/adapters/narrative_prison.py`, Reactive/Intentional/Planning runtime implementations, runtime decision dispatch, `calibration.py`, `uncertainty.py`, `losses.py`, `validation.py`, `model_comparison.py`, existing observational preregistration/target/dataset implementations, release code, stochastic runtime code, existing fixtures, or Lean sources.
- The scientific claim scope string is exactly `synthetic_protocol_only`.
- Parameter findings have exactly two successful statuses: `identified_under_protocol` and `not_identified_under_protocol`.
- Model-family findings never use those statuses; they use `equivalent`, `separated`, and `discriminating` semantics.
- Truth excluded from the accepted set or an empty accepted set is an experiment failure (`IdentificationRecoveryError`), never non-identifiability.
- Accepted-set membership uses the preregistered rule `candidate_loss <= block_best_loss + acceptance_loss_delta`, followed by the preregistered minimum accepted-block fraction.
- Coordinate-wise identification delegates to existing `diagnose_identifiability()` and never claims global structural identifiability.
- Observational equivalence is direct complete-policy equivalence with absolute tolerance exactly `1e-12`; equal ranked losses are not sufficient evidence.
- Information intervention certification requires the canonical scenario diff to contain exactly one preregistered information path; reward, objective dynamics, action set, and every other payload field remain exact.
- Joint latent-temperature grid is `beta_goal ∈ (0.5, 1.0, 2.0, 4.0)` × `beta_action ∈ (0.5, 1.0, 2.0, 4.0)` with truth `beta_goal=2.0`, `beta_action=1.0`, `goal_pressure_scale=1.0`, `instrumentality_scale=1.0`.
- Pressure and instrumentality scale grids are `(0.5, 1.0, 2.0)`; the deliberately confounded candidates include `(0.5,2.0)`, `(1.0,1.0)`, `(2.0,0.5)`.
- Final candidate grids: Reactive `beta_action=(0.5,1.0,2.0,4.0)`; Intentional `beta_goal=(0.5,1.0,2.0,4.0)`, `beta_action=(0.5,1.0,2.0,4.0)`, and both scale coordinates fixed to `(1.0,)`; Planning `beta_action=(0.5,1.0,2.0,4.0)`.
- Recovery generation seeds are `(11, 12)`; calibration seed blocks are `((101, 102), (111, 112))`; Brier training seeds are `(201, 202)`; Brier selection seeds are `(301, 302)`; final Brier/Log seeds are exactly `(401, 402)`.
- Recovery acceptance defaults are `acceptance_loss_delta=0.0`, `min_acceptance_fraction=1.0`; any experiment requiring a nonzero delta must preregister it explicitly in its `ParameterRecoveryExperiment`.
- Final target metric is the existing `prison_initial_action_metrics`, with categories `scout`, `escape`, `submit` and metric keys `initial.scout`, `initial.escape`, `initial.submit`.
- Brier is the only training/selection loss. Log is final re-scoring only; no candidate may be re-fit or re-selected under Log.
- Brier and Log sibling final protocols share exact dataset, train/selection/final partition hashes, target spec/hash/manifest, metric identity, final seeds, baseline, candidate identities/selected parameters, and ranking rule. Only protocol name, loss identity, loss-specific thresholds, declared precommitment hash, and derived protocol content hash may differ.
- Brier final thresholds are `AdequacyThresholds(2.0, 2.0)`. Log final thresholds are `AdequacyThresholds(20.0, 20.0)`; thresholds are reporting gates only and do not alter candidate selection.
- Final aggregate report must pass `attest_report(report).require_integrity()`.
- No #27 P2 Identification and Model Comparison checkbox is updated before merge plus post-merge exact-head GREEN. If all eight remaining checks are then evidenced, issue #27 may be closed; otherwise it remains open with failed/unproven checks unchanged.

## File Structure

### Production

- `narrative_dynamics/contracts.py` — add `ExperimentStage.SYNTHETIC_IDENTIFICATION` only.
- `narrative_dynamics/identification.py` — canonical protocol declarations, typed errors, accepted-set interpretation, recovery orchestration, equivalence certification, intervention certification/evaluation, Brier/Log sibling validation, per-stratum scoring aggregation, and final attested report construction.
- `narrative_dynamics/adapters/narrative_prison_identification.py` — isolated synthetic scenario translation, three family source/model builders, intentional latent parameterization, memory-evidence and future-information benchmark semantics, unified dispatch, canonical three-action output with `scout=0.0` zero-fill for binary decisions.
- Optional additive exports only if locked by a public API test.

### Synthetic data

- `fixtures/observations/narrative_identification_v1.json` — case-oriented `ObservationDataset` containing train, selection-validation, and final-test strata with explicit `synthetic_non_empirical` provenance and exact information-pair linkage.

### Tests

- `tests/test_narrative_identification_protocol.py`
- `tests/test_narrative_identification_recovery.py`
- `tests/test_narrative_identification_interventions.py`
- `tests/test_narrative_identification_model_comparison.py`
- `tests/test_narrative_identification_reporting.py`

---

### Task 1: Complete the Authoritative Test/Fixture-Only RED Boundary

**Files:**
- Create: `fixtures/observations/narrative_identification_v1.json`
- Create: `tests/test_narrative_identification_protocol.py`
- Create: `tests/test_narrative_identification_recovery.py`
- Create: `tests/test_narrative_identification_interventions.py`
- Create: `tests/test_narrative_identification_model_comparison.py`
- Create: `tests/test_narrative_identification_reporting.py`

**Interfaces:**
- Consumes: existing `SimulationRunner`, `calibrate_grid`, `ParameterAcceptanceSet`, `diagnose_identifiability`, held-out selection/final APIs, categorical Brier/Log losses, `PreregisteredEvaluationProtocol`, `compare_models_on_final_partition`, `attest_report`, `run_runtime_decision`, and the exact base behavior proven by #1158.
- Produces: the complete P2 acceptance boundary before any new production symbol exists.

- [ ] **Step 1: Generate and commit the exact synthetic fixture**

Use the existing case-oriented `ObservationDataset` serializer so the committed `content_hash` is generated by existing code, not hand-calculated. Run this one-off script; do not commit the script itself:

```python
from __future__ import annotations

import json
from pathlib import Path

from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.observations.dataset import (
    ObservationCase,
    ObservationDataset,
    ObservationPartitionRole,
)

OUT = Path("fixtures/observations/narrative_identification_v1.json")


def payload(*, mode, goal_gap, memory, future):
    return {
        "mode": mode,
        "goal_score_gap": float(goal_gap),
        "action_value_gap": 1.3,
        "prior_weak": 0.5,
        "memory_evidence_available": bool(memory),
        "memory_signal": "weak" if mode == "memory_evidence" else "none",
        "memory_signal_accuracy": 1.0 if mode == "memory_evidence" else 0.5,
        "future_information_available": bool(future),
        "future_signal_accuracy": 0.9 if mode == "future_information" else 0.5,
        "guard_persistence": 0.8,
        "escape_reward": 4.0,
        "capture_cost": 4.0,
        "submit_reward": 0.5,
        "scout_cost": 0.1 if mode == "future_information" else 2.0,
        "discount": 0.95,
    }


SPECS = (
    # name, role, stratum, mode, gap, memory, future, counts, pair
    ("train-temp-anchor", "train", "latent_temperature", "latent_temperature", 0.0, False, False, {"scout": 0, "escape": 79, "submit": 21}, None),
    ("train-goal-low", "train", "latent_temperature", "latent_temperature", 0.4, False, False, {"scout": 0, "escape": 61, "submit": 39}, None),
    ("train-memory-hidden", "train", "memory_evidence", "memory_evidence", 0.0, False, False, {"scout": 0, "escape": 45, "submit": 55}, None),
    ("train-future-off", "train", "future_information", "future_information", 0.0, False, False, {"scout": 5, "escape": 45, "submit": 50}, None),
    ("selection-goal-high", "selection_validation", "latent_temperature", "latent_temperature", 1.1, False, False, {"scout": 0, "escape": 73, "submit": 27}, None),
    ("selection-memory-revealed", "selection_validation", "memory_evidence", "memory_evidence", 0.0, True, False, {"scout": 0, "escape": 75, "submit": 25}, None),
    ("selection-future-on", "selection_validation", "future_information", "future_information", 0.0, False, True, {"scout": 55, "escape": 30, "submit": 15}, None),
    ("final-equivalence", "final_test", "observational_equivalence", "latent_temperature", 0.0, False, False, {"scout": 0, "escape": 50, "submit": 50}, None),
    ("final-memory-hidden", "final_test", "memory_evidence", "memory_evidence", 0.0, False, False, {"scout": 0, "escape": 45, "submit": 55}, "final-memory-revealed"),
    ("final-memory-revealed", "final_test", "memory_evidence", "memory_evidence", 0.0, True, False, {"scout": 0, "escape": 75, "submit": 25}, "final-memory-hidden"),
    ("final-future-off", "final_test", "future_information", "future_information", 0.0, False, False, {"scout": 5, "escape": 45, "submit": 50}, "final-future-on"),
    ("final-future-on", "final_test", "future_information", "future_information", 0.0, False, True, {"scout": 55, "escape": 30, "submit": 15}, "final-future-off"),
)

role_map = {
    "train": ObservationPartitionRole.TRAIN,
    "selection_validation": ObservationPartitionRole.SELECTION_VALIDATION,
    "final_test": ObservationPartitionRole.FINAL_TEST,
}

cases = []
for name, role, stratum, mode, gap, memory, future, counts, pair in SPECS:
    scenario = Scenario(name + "-scenario", payload(mode=mode, goal_gap=gap, memory=memory, future=future))
    generator_payload = {
        "generator_family": "synthetic-reference-policy-v1",
        "scenario_hash": scenario.content_hash,
        "counts": counts,
    }
    provenance = {
        "synthetic_non_empirical": True,
        "stratum": stratum,
        "generator_family": "synthetic-reference-policy-v1",
        "generator_parameters_hash": stable_content_hash(generator_payload),
        "information_condition": (
            "memory_revealed" if memory else
            "future_available" if future else
            "baseline"
        ),
    }
    if pair is not None:
        provenance["paired_case_id"] = pair
    cases.append(ObservationCase(
        name=name,
        role=role_map[role],
        scenario=scenario,
        counts=counts,
        observation_ids=(name + "-obs",),
        provenance=provenance,
    ))

dataset = ObservationDataset(
    name="narrative-synthetic-identification",
    version="1",
    source={"kind": "synthetic_reference_policy", "version": "1"},
    provenance={
        "synthetic_non_empirical": True,
        "claim_scope": "synthetic_protocol_only",
        "design_spec": "2026-08-28-narrative-synthetic-identification-v1-design",
    },
    cases=tuple(cases),
)
OUT.write_text(json.dumps(dataset.to_payload(), indent=2, sort_keys=True) + "\n")
print(dataset.content_hash)
```

Then verify the committed payload round-trips exactly:

```bash
python3 - <<'PY'
from narrative_dynamics.observations import load_observation_dataset
p = "fixtures/observations/narrative_identification_v1.json"
d = load_observation_dataset(p)
print(d.content_hash)
assert d.to_payload()["content_hash"] == d.content_hash
assert d.provenance["synthetic_non_empirical"] is True
PY
```

Expected: command exits 0 and prints one `sha256:` content hash.

- [ ] **Step 2: Create protocol RED tests with guarded imports**

Start `tests/test_narrative_identification_protocol.py` with:

```python
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import unittest

from narrative_dynamics.losses import (
    CategoricalBrierLoss,
    CategoricalLogLoss,
    CategoricalMetricGroup,
)
from narrative_dynamics.manifest import component_identity
from narrative_dynamics.observations import (
    AdequacyThresholds,
    CategoricalTargetSpec,
    FrozenModelCandidate,
    ObservationPartitionRole,
    PreregisteredEvaluationProtocol,
    load_observation_dataset,
)

_IDENTIFICATION_IMPORT_ERROR = None
try:
    from narrative_dynamics.contracts import ExperimentStage
    from narrative_dynamics.identification import (
        IdentificationProtocolError,
        IdentificationStatus,
        InformationInterventionPair,
        ParameterRecoveryExperiment,
        SyntheticIdentificationProtocol,
        validate_sibling_final_protocols,
    )
except (ImportError, AttributeError) as error:
    _IDENTIFICATION_IMPORT_ERROR = error

FIXTURE = Path("fixtures/observations/narrative_identification_v1.json")
TOL = 1e-12


class NarrativeIdentificationProtocolTests(unittest.TestCase):
    def require_api(self):
        if _IDENTIFICATION_IMPORT_ERROR is not None:
            self.fail(f"synthetic identification API is missing: {_IDENTIFICATION_IMPORT_ERROR}")
```

Add methods with these exact responsibilities:

1. `test_manifest_stage_is_additive_and_exact`: require `ExperimentStage.SYNTHETIC_IDENTIFICATION.value == "synthetic_identification"`.
2. `test_fixture_is_synthetic_partitioned_and_pair_linked`: load fixture; require top-level claim scope `synthetic_protocol_only`, all cases `synthetic_non_empirical is True`, roles exactly train/selection/final, and both final pair links are reciprocal.
3. `test_protocol_canonicalizes_experiments_pairs_and_hash`: reverse declaration order and require equal `identity_payload()` / `content_hash`.
4. `test_protocol_claim_scope_is_closed`: any claim scope other than `synthetic_protocol_only` raises `IdentificationProtocolError`.
5. `test_recovery_truth_must_be_in_grid`: a true coordinate absent from its finite grid raises.
6. `test_protocol_rejects_duplicate_experiment_and_pair_names`: duplicates fail closed.
7. `test_protocol_rejects_partition_overlap`: any scenario hash assigned to more than one role fails.
8. `test_sibling_final_protocols_allow_only_name_loss_threshold_and_derived_hash_difference`: one Brier and one Log protocol built from the same candidates/dataset/metric/seeds passes `validate_sibling_final_protocols`.
9. `test_sibling_final_protocols_reject_candidate_target_metric_seed_or_baseline_drift`: mutate one locked field per subtest and require `IdentificationProtocolError`.
10. `test_protocol_rejects_nonfinite_grid_tolerance_and_acceptance_values`: bool/NaN/inf/negative invalid values fail closed.

- [ ] **Step 3: Create recovery/equivalence RED tests**

In `tests/test_narrative_identification_recovery.py`, guarded-import the generic identification API and adapter factories. Reuse `prison_initial_action_metrics` and one Brier group over all three `initial.*` coordinates. Lock these constants:

```python
BETA_GRID = (0.5, 1.0, 2.0, 4.0)
SCALE_GRID = (0.5, 1.0, 2.0)
TRUTH = {
    "beta_goal": 2.0,
    "beta_action": 1.0,
    "goal_pressure_scale": 1.0,
    "instrumentality_scale": 1.0,
}
GENERATION_SEEDS = (11, 12)
CALIBRATION_BLOCKS = ((101, 102), (111, 112))
```

Add these tests:

1. `test_joint_beta_goal_beta_action_recovery_is_exact`: use `train-temp-anchor`, `train-goal-low`, and `selection-goal-high`; require accepted set exactly `(TRUTH,)` after fixing both scales to 1 and require `beta_goal` and `beta_action` identified.
2. `test_action_temperature_anchor_is_invariant_to_beta_goal`: same `beta_action`, varying `beta_goal`, exact policy equality on anchor.
3. `test_low_and_high_goal_gaps_change_with_beta_goal`: fixed `beta_action=1`, varying `beta_goal`, require at least one policy coordinate changes on both gap cases.
4. `test_equivalence_fixture_retains_multiple_latent_candidates_and_reports_nonidentification`: on `final-equivalence`, accepted set contains truth plus at least one distinct latent pair; status is `NOT_IDENTIFIED_UNDER_PROTOCOL`.
5. `test_pressure_recovery_with_fixed_instrumentality`: vary pressure grid only and require true pressure identified.
6. `test_instrumentality_recovery_with_fixed_pressure`: vary instrumentality grid only and require true scale identified.
7. `test_scale_confounded_design_retains_preregistered_product_equivalence_class`: candidate set includes the three `(pressure,instrumentality)` inverse pairs; require all three retained and both coordinates not identified.
8. `test_truth_excluded_is_recovery_error_not_nonidentification`: feed `interpret_parameter_identification` a complete candidate-loss table whose accepted candidates omit truth; require `IdentificationRecoveryError`.
9. `test_empty_accepted_set_is_recovery_error`: construct a semantically invalid/empty accepted interpretation input and require the same typed error.
10. `test_candidate_loss_table_retains_every_grid_candidate_and_ties`: require the finding exposes all evaluated tuples and a tie does not disappear behind lexical `best` ordering.
11. `test_observational_equivalence_certifies_complete_policy_not_equal_loss`: direct certification on equivalent policies succeeds; alter one policy coordinate by `2e-12` and require `equivalent is False` even if a supplied scalar score would tie.

- [ ] **Step 4: Create information-intervention RED tests**

In `tests/test_narrative_identification_interventions.py`, load final pair scenarios by case name and guard-import:

```python
from narrative_dynamics.identification import (
    InformationInterventionPair,
    InterventionCertificationError,
    certify_information_intervention,
    evaluate_information_intervention,
)
```

Add tests:

1. `test_memory_pair_diff_is_exactly_memory_availability`: canonical diff is exactly `("memory_evidence_available", False, True)`.
2. `test_future_pair_diff_is_exactly_future_information_availability`: canonical diff is exactly `("future_information_available", False, True)`.
3. `test_reward_drift_is_rejected`: change `escape_reward` in the intervention scenario and require `InterventionCertificationError`.
4. `test_action_semantic_drift_is_rejected`: change `mode` or another action-defining frozen field and require rejection.
5. `test_multiple_changed_information_fields_are_rejected`: change both allowed boolean and `memory_signal_accuracy`; reject.
6. `test_memory_evidence_intervention_separates_intentional_from_reactive`: with frozen comparison parameters, Reactive policy is unchanged across pair to `1e-12`, Intentional changes by more than `1e-12`, and finding is discriminating.
7. `test_future_information_intervention_separates_planning_from_intentional`: Intentional policy unchanged across future pair to `1e-12`, Planning changes by more than `1e-12`, and finding is discriminating.
8. `test_intervention_finding_replays_and_canonicalizes_family_order`: reverse family input order and require same finding payload/hash.

- [ ] **Step 5: Create model-comparison RED tests**

In `tests/test_narrative_identification_model_comparison.py`, guard-import the adapter and generic scoring helpers. Add `inspect.getsource()` isolation checks as in P1.

Use:

```python
REACTIVE_GRID = {"beta_action": (0.5, 1.0, 2.0, 4.0)}
INTENTIONAL_GRID = {
    "beta_goal": (0.5, 1.0, 2.0, 4.0),
    "beta_action": (0.5, 1.0, 2.0, 4.0),
    "goal_pressure_scale": (1.0,),
    "instrumentality_scale": (1.0,),
}
PLANNING_GRID = {"beta_action": (0.5, 1.0, 2.0, 4.0)}
TRAIN_SEEDS = (201, 202)
SELECTION_SEEDS = (301, 302)
FINAL_SEEDS = (401, 402)
```

Add tests:

1. `test_identification_adapter_uses_unified_dispatch_and_not_identification_or_comparison_protocol`: source text contains `run_runtime_decision` and excludes `narrative_dynamics.identification`, `model_comparison`, observational preregistration, and family-specific runtime runner names.
2. `test_source_parameter_schemas_match_real_family_semantics`: Reactive/Planning accept only `beta_action`; Intentional requires exactly all four declared latent/scaling coordinates; extra/missing keys reject.
3. `test_binary_modes_zero_fill_scout_without_changing_dispatch_policy`: temperature/memory cases emit `initial_policy` with scout 0 and preserve escape/submit dispatch probabilities exactly.
4. `test_brier_training_and_selection_freeze_one_candidate_per_family`: use existing `fit_training_target_grid` + `select_on_validation_suite`; create three `FrozenModelCandidate`s from Brier selection reports.
5. `test_brier_final_protocol_completes_on_exact_frozen_candidates`: compare the three sources on final targets with final seeds.
6. `test_log_sibling_protocol_reuses_exact_brier_candidates_targets_and_seeds`: create Log protocol only after Brier candidate freeze; sibling validation passes and Log comparison completes without any training/selection call under Log.
7. `test_log_candidate_or_seed_drift_is_rejected_before_final_scoring`: mutate frozen candidate or final seed and require typed comparison/protocol error.
8. `test_scored_findings_expose_equivalence_memory_and_future_strata`: per-stratum scores contain exactly `observational_equivalence`, `memory_evidence`, `future_information`; each family has finite mean/worst loss.
9. `test_comparison_does_not_require_one_family_to_win_every_stratum`: report construction succeeds with different per-stratum rankings and contains pairwise deltas rather than a universal-winner assertion.

- [ ] **Step 6: Create final-report RED tests**

In `tests/test_narrative_identification_reporting.py`, guard-import the aggregate report API and build findings through the same helper functions used by the other test modules.

Add tests:

1. `test_final_report_requires_all_eight_roadmap_evidence_categories`.
2. `test_final_report_manifest_stage_is_synthetic_identification_and_parents_are_complete`.
3. `test_attested_report_round_trip_requires_integrity`:

```python
attested = attest_report(report)
self.assertIs(attested.require_integrity(), report)
```

4. `test_parameter_nonidentification_language_is_protocol_scoped`: rendered conclusion contains `not identified under this frozen synthetic protocol` and does not contain `structurally identified`, `empirically identified`, or `human cognition`.
5. `test_model_family_language_uses_equivalent_or_separated_not_identified`: family conclusion contains model-family separation language and no parameter-identification status token.
6. `test_forged_protocol_hash_is_rejected`: `dataclasses.replace` one child finding/report protocol hash and require `IdentificationReportError`.
7. `test_forged_parent_manifest_set_is_rejected`: remove or add one parent hash and require rejection.
8. `test_missing_required_finding_prevents_completion`: omit one required recovery/intervention/score finding and require report construction failure rather than a partially complete success report.

- [ ] **Step 7: Run the complete test-only RED and preserve legacy GREEN**

Run the new modules first:

```bash
python3 -m unittest \
  tests.test_narrative_identification_protocol \
  tests.test_narrative_identification_recovery \
  tests.test_narrative_identification_interventions \
  tests.test_narrative_identification_model_comparison \
  tests.test_narrative_identification_reporting -v
```

Expected: failures/errors only because the new identification module, additive manifest stage, and identification adapter are missing; fixture-only assertions may already pass.

Then run the exact full Python gate used by CI:

```bash
python3 -m unittest discover -s tests -v
```

Expected: all inherited 659 tests remain GREEN; every failure/error belongs to the five new P2 test modules. Record the exact total/failure/error counts; do not predict or normalize them after the fact.

- [ ] **Step 8: Commit the authoritative RED atomically and require exact-head Actions evidence**

```bash
git add \
  fixtures/observations/narrative_identification_v1.json \
  tests/test_narrative_identification_protocol.py \
  tests/test_narrative_identification_recovery.py \
  tests/test_narrative_identification_interventions.py \
  tests/test_narrative_identification_model_comparison.py \
  tests/test_narrative_identification_reporting.py
git commit -m "test: define narrative synthetic identification red boundary"
```

Push the exact commit and require `proof.yml` to show Lean/conformance/theorem gates GREEN and Python RED only in the new boundary. Save run id/job id/exact checkout/failure counts for the future PR body.

---

### Task 2: Add the Generic Protocol, Error, and Accepted-Set Semantics Core

**Files:**
- Modify: `narrative_dynamics/contracts.py`
- Create: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_protocol.py`
- Test: `tests/test_narrative_identification_recovery.py`

**Interfaces:**
- Consumes: `stable_content_hash`, `ExperimentManifest`, `ParameterAcceptanceSet`, `diagnose_identifiability`, `component_identity`, `callable_identity`, `metric_loss_identity`.
- Produces: `IdentificationStatus`, all typed errors, `ParameterRecoveryExperiment`, `ParameterCandidateLoss`, `ParameterIdentificationFinding`, `InformationInterventionPair`, `SyntheticIdentificationProtocol`, `interpret_parameter_identification()`, `validate_sibling_final_protocols()`.

- [ ] **Step 1: Add only the additive manifest stage**

In `ExperimentStage` add:

```python
SYNTHETIC_IDENTIFICATION = "synthetic_identification"
```

No other `contracts.py` behavior changes.

- [ ] **Step 2: Implement validators, errors, and canonical parameter/grid types**

Start `narrative_dynamics/identification.py` with the exact error tree and status values:

```python
class SyntheticIdentificationError(ValueError):
    pass


class IdentificationProtocolError(SyntheticIdentificationError):
    pass


class IdentificationRecoveryError(SyntheticIdentificationError):
    pass


class InterventionCertificationError(SyntheticIdentificationError):
    pass


class IdentificationComparisonError(SyntheticIdentificationError):
    pass


class IdentificationReportError(SyntheticIdentificationError):
    pass


class IdentificationStatus(str, Enum):
    IDENTIFIED_UNDER_PROTOCOL = "identified_under_protocol"
    NOT_IDENTIFIED_UNDER_PROTOCOL = "not_identified_under_protocol"
```

Add private validators for trimmed text, finite non-bool floats, probabilities/fractions, unique integer seed tuples/blocks, sha256 hashes, canonical `ParameterTuple`, and canonical finite grids. Every public dataclass must freeze/canonicalize input in `__post_init__`.

- [ ] **Step 3: Implement the recovery declaration and candidate-loss table**

Use these interfaces:

```python
ParameterTuple = tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class ParameterRecoveryExperiment:
    name: str
    model_identity: Mapping[str, object]
    true_parameters: ParameterTuple
    parameter_grid: tuple[tuple[str, tuple[float, ...]], ...]
    target_coordinates: tuple[str, ...]
    cases: tuple[Scenario, ...]
    generation_seeds: tuple[int, ...]
    calibration_seed_blocks: tuple[tuple[int, ...], ...]
    acceptance_loss_delta: float
    min_acceptance_fraction: float
    metric_identity: Mapping[str, object]
    loss_identity: Mapping[str, object]
    expected_status: IdentificationStatus | None = None

    @property
    def content_hash(self) -> str: ...


@dataclass(frozen=True)
class ParameterCandidateLoss:
    parameters: ParameterTuple
    block_losses: tuple[float, ...]
    mean_loss: float
    accepted_blocks: int
    acceptance_fraction: float
```

`ParameterRecoveryExperiment.__post_init__` must require truth to be a member of the Cartesian grid, target coordinates to be a nonempty subset of parameter names, at least one case, nonempty generation seeds/blocks, and unique case ids.

- [ ] **Step 4: Implement accepted-set interpretation independently of lexical best**

Use:

```python
@dataclass(frozen=True)
class ParameterIdentificationFinding:
    experiment_hash: str
    true_parameters: ParameterTuple
    candidate_losses: tuple[ParameterCandidateLoss, ...]
    accepted_parameters: ParameterAcceptanceSet
    truth_retained: bool
    identifiability: IdentifiabilityReport
    status: IdentificationStatus
    parent_manifest_hashes: tuple[str, ...]

    @property
    def content_hash(self) -> str: ...
```

And:

```python
def interpret_parameter_identification(
    experiment: ParameterRecoveryExperiment,
    candidate_losses: tuple[ParameterCandidateLoss, ...],
    *,
    parent_manifest_hashes: tuple[str, ...] = (),
) -> ParameterIdentificationFinding:
    ...
```

Algorithm:

```python
accepted = tuple(
    item.parameters
    for item in candidate_losses
    if item.acceptance_fraction >= experiment.min_acceptance_fraction
)
if not accepted:
    raise IdentificationRecoveryError("synthetic identification accepted set is empty")
if experiment.true_parameters not in accepted:
    raise IdentificationRecoveryError("synthetic identification did not retain its true parameters")
report = diagnose_identifiability(accepted)
by_name = {item.name: item for item in report.parameters}
identified = all(by_name[name].identified for name in experiment.target_coordinates)
if identified:
    for name in experiment.target_coordinates:
        if by_name[name].values != (dict(experiment.true_parameters)[name],):
            raise IdentificationRecoveryError("identified coordinate does not equal synthetic truth")
status = (
    IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL
    if identified
    else IdentificationStatus.NOT_IDENTIFIED_UNDER_PROTOCOL
)
if experiment.expected_status is not None and status is not experiment.expected_status:
    raise IdentificationRecoveryError("synthetic identification result violates preregistered expectation")
```

Also require the candidate table to cover the complete Cartesian grid exactly once.

- [ ] **Step 5: Implement intervention declaration and top-level preregistration identity**

Use:

```python
@dataclass(frozen=True)
class InformationInterventionPair:
    name: str
    baseline: Scenario
    intervention: Scenario
    allowed_information_path: tuple[str, ...]
    frozen_paths: tuple[tuple[str, ...], ...]
    stratum: str
    expected_relationship: str | None = None

    @property
    def content_hash(self) -> str: ...
```

And a `SyntheticIdentificationProtocol` that binds exactly:

```python
@dataclass(frozen=True)
class SyntheticIdentificationProtocol:
    name: str
    version: str
    claim_scope: str
    fixture_hash: str
    adapter_identities: tuple[tuple[str, Mapping[str, object]], ...]
    recovery_experiments: tuple[ParameterRecoveryExperiment, ...]
    intervention_pairs: tuple[InformationInterventionPair, ...]
    train_partition_hash: str
    selection_partition_hash: str
    final_partition_hash: str
    target_spec_hash: str
    metric_identity: Mapping[str, object]
    brier_loss_identity: Mapping[str, object]
    log_loss_identity: Mapping[str, object]
    brier_training_seeds: tuple[int, ...]
    brier_selection_seeds: tuple[int, ...]
    final_seeds: tuple[int, ...]
    equivalence_tolerance: float = 1e-12

    def identity_payload(self) -> dict[str, object]: ...

    @property
    def content_hash(self) -> str: ...
```

Require claim scope exactly `synthetic_protocol_only`, unique recovery/pair names, disjoint partition case hashes, canonical sort by name, and exact `1e-12` tolerance.

- [ ] **Step 6: Implement sibling final-protocol validation**

```python
def validate_sibling_final_protocols(
    brier: PreregisteredEvaluationProtocol,
    log: PreregisteredEvaluationProtocol,
) -> None:
    ...
```

Compare these fields for exact equality:

```python
(
    "version",
    "dataset_hash",
    "train_partition_hash",
    "selection_partition_hash",
    "final_partition_hash",
    "target_spec_hash",
    "final_target_hash",
    "final_target_manifest_hash",
    "metric_identity",
    "simulation_seeds",
    "baseline_name",
    "candidates",
)
```

Require Brier and Log loss identities to be distinct and match their expected registered loss names. Permit only `name`, `loss_identity`, `thresholds`, `declared_precommitment_hash`, and derived protocol hash to differ.

- [ ] **Step 7: Run Task 2 targeted tests and commit**

```bash
python3 -m unittest \
  tests.test_narrative_identification_protocol \
  tests.test_narrative_identification_recovery -v
```

Expected: protocol/core negative tests GREEN; adapter/evaluation-dependent recovery tests remain RED only because adapter/evaluation execution is not implemented yet.

```bash
git add narrative_dynamics/contracts.py narrative_dynamics/identification.py
git commit -m "feat: add synthetic identification protocol core"
```

Require exact-head proof to show monotonic reduction from the authoritative RED; do not require full feature GREEN yet.

---

### Task 3: Add the Isolated Narrative Identification Adapter

**Files:**
- Create: `narrative_dynamics/adapters/narrative_prison_identification.py`
- Test: `tests/test_narrative_identification_recovery.py`
- Test: `tests/test_narrative_identification_interventions.py`
- Test: `tests/test_narrative_identification_model_comparison.py`

**Interfaces:**
- Consumes: GenericNarrative/DomainSpec runtime model constructors and unified `run_runtime_decision()`.
- Produces: `NarrativePrisonIdentificationSource`, three source factories, canonical three-coordinate `initial_policy` output, real family-specific parameter schemas.

- [ ] **Step 1: Implement exact external source/model contracts**

Use:

```python
_FAMILIES = ("reactive", "intentional", "planning")
_ACTIONS = ("scout", "escape", "submit")
_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "narrative-synthetic-identification-v1"


@dataclass(frozen=True)
class NarrativePrisonIdentificationSource:
    family: str

    @property
    def name(self) -> str:
        return f"generic-narrative-identification-{self.family}"

    @property
    def lifecycle(self) -> str:
        return "fresh_per_batch"

    def instantiate(self):
        return _NarrativePrisonIdentificationModel(self.family)

    def manifest_identity(self) -> dict[str, object]:
        ...
```

Factories:

```python
def create_narrative_identification_reactive_source(): ...
def create_narrative_identification_intentional_source(): ...
def create_narrative_identification_planning_source(): ...
```

Source identity must attest source class, model class, scenario builder, family builder, and `run_runtime_decision` implementation identity.

- [ ] **Step 2: Validate one closed scenario schema and family-specific parameter schemas**

All scenarios require exactly the payload keys emitted by Task 1's `payload()` helper. `mode` is one of:

```python
{"latent_temperature", "memory_evidence", "future_information"}
```

Intentional parameters are exactly:

```python
{"beta_goal", "beta_action", "goal_pressure_scale", "instrumentality_scale"}
```

Reactive/Planning parameters are exactly:

```python
{"beta_action"}
```

All betas/scales are positive finite non-bool floats.

- [ ] **Step 3: Build binary latent-temperature semantics without changing existing intentional math**

For `latent_temperature`, declare only `escape` and `submit` in the runtime decision. Construct two goals whose score gap is:

```text
scenario.goal_score_gap * goal_pressure_scale * instrumentality_scale
```

Split it symmetrically as `+gap/2` and `-gap/2` expected instrumentality. Use mirror action values with gap `action_value_gap`:

```python
freedom_values = {"escape": +action_gap / 2, "submit": -action_gap / 2}
safety_values = {"escape": -action_gap / 2, "submit": +action_gap / 2}
```

For the action-temperature anchor (`goal_score_gap == 0.0`), use identical conditional value vectors for both goals instead of mirrors so marginal action policy is exactly independent of `beta_goal`. For the explicit equivalence case, use equal goal scores and mirror conditional policies so `escape/submit == 0.5/0.5` for every tested latent pair.

Do not reimplement the softmax/marginal formula: instantiate existing `GoalModelSpec(beta_goal=...)`, `ChoiceModelSpec(beta_action=...)`, and execute the existing Intentional runtime through unified dispatch.

- [ ] **Step 4: Build memory-evidence semantics**

Construct one hidden `guard.status ∈ {weak,strong}` cell with prior 0.5. When `memory_evidence_available=False`, the prisoner's runtime ledger carries no diagnostic weak/strong evidence before the decision. When `True`, admit one prior weak signal with the scenario-declared `memory_signal_accuracy` through the existing belief/evidence path.

Reactive sees the same current declared cue in both paired scenarios and therefore its policy must be identical. Intentional goal instrumentality uses weak/strong with opposite signs, so posterior change alters goal scores/policy. `scout` is not authored in this binary mode and is zero-filled only in the outer `initial_policy`.

- [ ] **Step 5: Build future-information semantics**

For `future_information`, author all three actions. Reactive and Intentional treat `scout` myopically as the fixed immediate `-scout_cost` option and do not consume `future_information_available` in their decision semantics. Planning uses the existing finite planning runtime: when availability is false, scout's future observation is uninformative; when true, its observation hook has the declared `future_signal_accuracy`, creating value of information. Keep reward, transition, hidden-state, discount, persistence, and action set identical across the pair.

- [ ] **Step 6: Execute every family only through unified dispatch and normalize output**

```python
dispatch = run_runtime_decision(
    case.story,
    case.domain,
    case.decision_id,
    case.ledger,
    RuntimeDecisionModelSpec(self.family, nested),
)
policy = {action: 0.0 for action in _ACTIONS}
policy.update({action: float(prob) for action, prob in dispatch.action_policy.items()})
return ModelRun(
    events=(),
    outcome={
        "model_kind": self.family,
        "initial_policy": policy,
        "selected_action": dispatch.selected_action,
        "runtime_dispatch_hash": dispatch.content_hash,
        "runtime_dispatch": dispatch.to_dict(),
        "translated_story_hash": case.story.content_hash,
        "translated_domain_hash": case.domain.content_hash,
        "runtime_ledger_hash": case.ledger.content_hash,
    },
)
```

Accept the outer RNG argument but do not sample an action from the policy.

- [ ] **Step 7: Run adapter/recovery/intervention/model-source tests and commit**

```bash
python3 -m unittest \
  tests.test_narrative_identification_recovery \
  tests.test_narrative_identification_interventions \
  tests.test_narrative_identification_model_comparison -v
```

Expected: source schema/isolation, raw policy invariants, and mode semantics GREEN; recovery/intervention orchestration and final comparison aggregation may remain RED until later tasks.

```bash
git add narrative_dynamics/adapters/narrative_prison_identification.py
git commit -m "feat: add narrative synthetic identification adapter"
```

---

### Task 4: Implement Recovery and Observational-Equivalence Evaluation

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_recovery.py`

**Interfaces:**
- Consumes: `ParameterRecoveryExperiment`, `SimulationRunner`, `calibrate_grid`, `aggregate_metrics`, `MetricLoss`, adapter sources.
- Produces: `evaluate_parameter_recovery_experiment()`, `ObservationalEquivalenceFinding`, `certify_observational_equivalence()`.

- [ ] **Step 1: Implement multi-case target generation and block scoring by composing existing calibration**

Use this public signature:

```python
def evaluate_parameter_recovery_experiment(
    *,
    runner: SimulationRunner,
    model: ModelSource,
    experiment: ParameterRecoveryExperiment,
    extractor: MetricExtractor,
    loss: MetricLoss,
) -> ParameterIdentificationFinding:
    ...
```

Preflight exact source/extractor/loss identities against the experiment. For each case, generate a target from the true parameters with `runner.run_batch(..., seeds=experiment.generation_seeds)` and `aggregate_metrics()`.

For every calibration seed block, call existing `calibrate_grid()` once per case with the complete replayable grid and that case's generated target. Aggregate each candidate's per-case loss using `fmean` to obtain one block loss. A candidate is accepted in that block when:

```python
candidate_block_loss <= min_block_loss + experiment.acceptance_loss_delta
```

Build a complete `ParameterCandidateLoss` row per Cartesian candidate with all block losses, mean loss, accepted block count, and acceptance fraction. Parent hashes are the complete set of generated-trace and calibration report manifest hashes. Pass the result to `interpret_parameter_identification()`.

- [ ] **Step 2: Build the four preregistered recovery experiments in test helpers**

Joint beta experiment uses exact cases `train-temp-anchor`, `train-goal-low`, `selection-goal-high`, full 4×4 beta grid, fixed scales 1, truth `TRUTH`, expected `IDENTIFIED_UNDER_PROTOCOL`.

Pressure recovery varies only `goal_pressure_scale=(0.5,1.0,2.0)` with truth 1, fixes instrumentality 1 and both betas at truth, expected identified.

Instrumentality recovery mirrors pressure recovery.

Scale-confounded recovery uses only inverse-product candidate combinations rather than the full Cartesian scale grid. Represent this by permitting `ParameterRecoveryExperiment` to accept an optional explicit canonical candidate tuple set; when present, it replaces Cartesian expansion but must still contain truth and one stable schema. If this extension is necessary, add it now as `explicit_candidates: tuple[ParameterTuple, ...] = ()` and lock it with a protocol test. Do not change `calibrate_grid`; for explicit candidate mode, evaluate candidates through existing `SimulationRunner` + loss evaluation directly and produce the same `ParameterCandidateLoss` table.

- [ ] **Step 3: Implement direct complete-policy equivalence finding**

```python
@dataclass(frozen=True)
class ObservationalEquivalenceFinding:
    left_model_identity: Mapping[str, object]
    right_model_identity: Mapping[str, object]
    case_hashes: tuple[str, ...]
    action_keys: tuple[str, ...]
    policies: tuple[tuple[str, tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]], ...]
    max_abs_policy_delta: float
    tolerance: float
    equivalent: bool
    parent_manifest_hashes: tuple[str, ...]

    @property
    def content_hash(self) -> str: ...
```

Public evaluator:

```python
def certify_observational_equivalence(
    *,
    runner: SimulationRunner,
    left_model: ModelSource,
    right_model: ModelSource,
    left_parameters: Mapping[str, float],
    right_parameters: Mapping[str, float],
    cases: tuple[Scenario, ...],
    seeds: tuple[int, ...],
    extractor: MetricExtractor,
    action_keys: tuple[str, ...],
    tolerance: float = 1e-12,
) -> ObservationalEquivalenceFinding:
    ...
```

Aggregate each side's policy metrics over seeds, compare every declared action on every case, and set `equivalent = max_delta <= tolerance`. Never inspect scalar calibration loss for this conclusion.

- [ ] **Step 4: Make joint recovery and deliberate non-recovery GREEN**

Run:

```bash
python3 -m unittest tests.test_narrative_identification_recovery -v
```

Expected: every recovery/equivalence test GREEN, including exact joint beta recovery, isolated pressure/instrumentality recovery, retained confounded equivalence class, truth-excluded failure, and full policy equivalence.

- [ ] **Step 5: Commit**

```bash
git add narrative_dynamics/identification.py tests/test_narrative_identification_recovery.py
git commit -m "feat: add synthetic latent recovery and equivalence"
```

If test edits were only corrections to an already-approved RED assumption, keep them minimal and explain the correction in the commit body; do not weaken the scientific criterion to obtain GREEN.

---

### Task 5: Implement Information-Only Intervention Certification and Family Separation

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_interventions.py`

**Interfaces:**
- Consumes: `InformationInterventionPair`, adapter family sources, `SimulationRunner`, metric extractor.
- Produces: `certify_information_intervention()`, `InformationInterventionFinding`, `evaluate_information_intervention()`.

- [ ] **Step 1: Implement recursive canonical scenario diff**

Private helper returns sorted leaf diffs:

```python
def _scenario_payload_diff(
    left: Mapping[str, object],
    right: Mapping[str, object],
    prefix: tuple[str, ...] = (),
) -> tuple[tuple[tuple[str, ...], object, object], ...]:
    ...
```

Mappings recurse by sorted key. Missing key, list-length change, or noncanonical value is a diff at the corresponding path; the helper does not silently coerce values.

- [ ] **Step 2: Certify exactly one allowed information change**

```python
def certify_information_intervention(
    pair: InformationInterventionPair,
) -> tuple[tuple[str, ...], object, object]:
    diffs = _scenario_payload_diff(pair.baseline.payload, pair.intervention.payload)
    if len(diffs) != 1:
        raise InterventionCertificationError("information intervention must change exactly one payload leaf")
    path, before, after = diffs[0]
    if path != pair.allowed_information_path:
        raise InterventionCertificationError("information intervention changed an undeclared field")
    for frozen in pair.frozen_paths:
        if frozen == path:
            raise InterventionCertificationError("allowed information path cannot also be frozen")
    return diffs[0]
```

For this V1, protocol construction must populate `frozen_paths` with every top-level scenario payload key except the allowed boolean. Tests verify rewards/dynamics/action-defining fields are therefore frozen rather than relying only on convention.

- [ ] **Step 3: Implement family policy delta finding**

```python
@dataclass(frozen=True)
class InformationInterventionFinding:
    pair_hash: str
    certified_diff: tuple[tuple[str, ...], object, object]
    family_policies: tuple[
        tuple[str, tuple[tuple[str, float], ...], tuple[tuple[str, float], ...]], ...
    ]
    within_family_max_deltas: tuple[tuple[str, float], ...]
    cross_family_contrasts: tuple[tuple[str, str, float], ...]
    tolerance: float
    discriminating: bool
    parent_manifest_hashes: tuple[str, ...]

    @property
    def content_hash(self) -> str: ...
```

Evaluator:

```python
def evaluate_information_intervention(
    *,
    pair: InformationInterventionPair,
    runner: SimulationRunner,
    families: tuple[tuple[str, ModelSource, Mapping[str, float]], ...],
    seeds: tuple[int, ...],
    extractor: MetricExtractor,
    action_keys: tuple[str, ...],
    tolerance: float = 1e-12,
) -> InformationInterventionFinding:
    ...
```

Certify before any model run. Canonicalize families by name. Aggregate policies over seeds and compute max coordinate changes.

For `expected_relationship="intentional_vs_reactive"`, discriminating means Reactive delta `<=1e-12`, Intentional delta `>1e-12`.

For `expected_relationship="planning_vs_intentional"`, discriminating means Intentional delta `<=1e-12`, Planning delta `>1e-12`.

Raise `IdentificationProtocolError` for an unsupported expected relationship rather than guessing.

- [ ] **Step 4: Run the intervention suite and commit**

```bash
python3 -m unittest tests.test_narrative_identification_interventions -v
```

Expected: all eight tests GREEN.

```bash
git add narrative_dynamics/identification.py
git commit -m "feat: certify narrative information interventions"
```

---

### Task 6: Freeze Brier-Selected Candidates and Add Log Sibling Final Scoring

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_model_comparison.py`

**Interfaces:**
- Consumes: existing observational target construction, Brier training fit, selection-validation, `FrozenModelCandidate`, `PreregisteredEvaluationProtocol`, `compare_models_on_final_partition`.
- Produces: protocol construction helpers, `ScoredModelComparisonFinding`, per-stratum aggregation, strict Brier/Log sibling execution.

- [ ] **Step 1: Build Brier training/selection with existing APIs only**

In test/helper code, construct:

```python
spec = CategoricalTargetSpec(
    name="narrative-identification-initial-action",
    version="1",
    categories=("scout", "escape", "submit"),
    metric_prefix="initial",
)
group = CategoricalMetricGroup(
    "initial-action",
    ("initial.scout", "initial.escape", "initial.submit"),
)
brier = CategoricalBrierLoss((group,))
log = CategoricalLogLoss((group,))
```

Create train/selection targets from the committed fixture. For each source, call `fit_training_target_grid(... loss=brier)` with its exact family grid, convert `training.candidate_parameters` into `ParameterAcceptanceSet`, run `select_on_validation_suite(... loss=brier)`, then freeze with `FrozenModelCandidate.from_selection`.

No Log call occurs in training or selection.

- [ ] **Step 2: Create exact sibling final protocols**

Use the same frozen candidate tuple for both:

```python
brier_protocol = PreregisteredEvaluationProtocol.create(
    name="narrative-synthetic-identification-brier-v1",
    version="1",
    dataset=dataset,
    target_spec=spec,
    extractor=prison_initial_action_metrics,
    loss=brier,
    simulation_seeds=(401, 402),
    baseline_name=intentional_name,
    candidates=frozen_candidates,
    thresholds=AdequacyThresholds(2.0, 2.0),
)
log_protocol = PreregisteredEvaluationProtocol.create(
    name="narrative-synthetic-identification-log-v1",
    version="1",
    dataset=dataset,
    target_spec=spec,
    extractor=prison_initial_action_metrics,
    loss=log,
    simulation_seeds=(401, 402),
    baseline_name=intentional_name,
    candidates=frozen_candidates,
    thresholds=AdequacyThresholds(20.0, 20.0),
)
validate_sibling_final_protocols(brier_protocol, log_protocol)
```

- [ ] **Step 3: Execute both existing final comparisons**

Construct one `ComparisonModel` tuple from the exact frozen candidates and runtime sources. Call `compare_models_on_final_partition` once with Brier protocol/loss and once with Log protocol/loss against the same `final_targets` object and exact final seeds.

Do not re-instantiate or mutate frozen candidate parameters between calls.

- [ ] **Step 4: Add per-stratum scoring aggregation**

Use:

```python
@dataclass(frozen=True)
class StratumModelScore:
    stratum: str
    model_name: str
    mean_loss: float
    worst_loss: float
    case_names: tuple[str, ...]


@dataclass(frozen=True)
class ScoredModelComparisonFinding:
    protocol_hash: str
    loss_identity: Mapping[str, object]
    candidate_hashes: tuple[tuple[str, str], ...]
    final_partition_hash: str
    final_target_hash: str
    final_seeds: tuple[int, ...]
    global_scores: tuple[tuple[str, float, float], ...]
    stratum_scores: tuple[StratumModelScore, ...]
    pairwise_mean_deltas: tuple[tuple[str, str, float], ...]
    parent_comparison_manifest_hash: str

    @property
    def content_hash(self) -> str: ...
```

Public converter:

```python
def score_model_comparison_by_stratum(
    *,
    dataset: ObservationDataset,
    protocol: PreregisteredEvaluationProtocol,
    report: ModelComparisonReport,
) -> ScoredModelComparisonFinding:
    ...
```

Map final case names/scenario ids to fixture `provenance["stratum"]`, then aggregate the existing per-case losses in each model's `FinalTestReport`. Require every final case to have a nonempty stratum and require exactly the three final strata: `observational_equivalence`, `memory_evidence`, `future_information`.

- [ ] **Step 5: Run model-comparison tests and commit**

```bash
python3 -m unittest tests.test_narrative_identification_model_comparison -v
```

Expected: all tests GREEN; no test or implementation asserts one universal best family.

```bash
git add narrative_dynamics/identification.py
git commit -m "feat: add dual-score synthetic model comparison"
```

---

### Task 7: Build the Aggregate Identification Report and Attestation Boundary

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Optional Modify: `narrative_dynamics/__init__.py` only if the RED public API test requires additive exports.
- Test: `tests/test_narrative_identification_reporting.py`

**Interfaces:**
- Consumes: one frozen `SyntheticIdentificationProtocol`, all recovery/equivalence/intervention/scoring findings, child manifest hashes, existing `ExperimentManifest` and `attest_report`.
- Produces: `SyntheticIdentificationReport`, `build_synthetic_identification_report()`, protocol-scoped conclusion rendering.

- [ ] **Step 1: Add final report type with internal consistency checks**

Use:

```python
@dataclass(frozen=True)
class SyntheticIdentificationReport:
    protocol_hash: str
    claim_scope: str
    parameter_findings: tuple[ParameterIdentificationFinding, ...]
    equivalence_findings: tuple[ObservationalEquivalenceFinding, ...]
    intervention_findings: tuple[InformationInterventionFinding, ...]
    brier_comparison: ScoredModelComparisonFinding
    log_comparison: ScoredModelComparisonFinding
    conclusions: tuple[str, ...]
    manifest: ExperimentManifest

    def __post_init__(self) -> None:
        ...
```

Validate exact claim scope, sorted unique finding identities, Brier/Log candidate/final identities, protocol hash consistency, manifest stage, manifest input finding hashes, and complete parent hashes.

- [ ] **Step 2: Implement report builder with all eight roadmap evidence gates**

```python
def build_synthetic_identification_report(
    *,
    protocol: SyntheticIdentificationProtocol,
    parameter_findings: tuple[ParameterIdentificationFinding, ...],
    equivalence_findings: tuple[ObservationalEquivalenceFinding, ...],
    intervention_findings: tuple[InformationInterventionFinding, ...],
    brier_comparison: ScoredModelComparisonFinding,
    log_comparison: ScoredModelComparisonFinding,
) -> SyntheticIdentificationReport:
    ...
```

Require named evidence for:

```text
joint-beta-goal-action
pressure-recovery
instrumentality-recovery
scale-confounded-nonrecovery
observational-equivalence
intentional-vs-reactive-memory
intentional-vs-planning-future-information
brier-and-log-held-out
```

The first three parameter findings must be identified; scale-confounded must be not identified; equivalence must certify true; both interventions must be discriminating; both score findings must bind the same candidates/final identities.

- [ ] **Step 3: Build final manifest from child evidence, not prose**

```python
manifest = ExperimentManifest(
    stage=ExperimentStage.SYNTHETIC_IDENTIFICATION,
    inputs={
        "protocol_hash": protocol.content_hash,
        "claim_scope": protocol.claim_scope,
        "parameter_finding_hashes": tuple(item.content_hash for item in parameter_findings),
        "equivalence_finding_hashes": tuple(item.content_hash for item in equivalence_findings),
        "intervention_finding_hashes": tuple(item.content_hash for item in intervention_findings),
        "brier_comparison_hash": brier_comparison.content_hash,
        "log_comparison_hash": log_comparison.content_hash,
    },
    parent_hashes=tuple(sorted(set(
        hash_value
        for finding in parameter_findings
        for hash_value in finding.parent_manifest_hashes
    ) | {
        brier_comparison.parent_comparison_manifest_hash,
        log_comparison.parent_comparison_manifest_hash,
    })),
)
```

Include intervention/equivalence run parent hashes in the union as well.

- [ ] **Step 4: Render only protocol-scoped conclusions**

Use deterministic wording functions, for example:

```python
def parameter_conclusion(name: str, status: IdentificationStatus) -> str:
    if status is IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL:
        return f"{name} is identified under this frozen synthetic protocol."
    return f"{name} is not identified under this frozen synthetic protocol."
```

Family conclusions use wording such as:

```text
Reactive and Intentional are separated by the preregistered memory-evidence intervention under this synthetic protocol.
Intentional and Planning/POMDP are separated by the preregistered future-information intervention under this synthetic protocol.
```

Never generate `structurally identified`, `empirically identified`, or claims about real human cognition.

- [ ] **Step 5: Make attestation and forgery tests GREEN**

```bash
python3 -m unittest tests.test_narrative_identification_reporting -v
```

Expected: all report/attestation/forgery tests GREEN.

If public exports are needed, add only the explicitly tested names; otherwise leave package roots unchanged.

- [ ] **Step 6: Commit**

```bash
git add narrative_dynamics/identification.py narrative_dynamics/__init__.py
git commit -m "feat: attest narrative synthetic identification report"
```

Omit `narrative_dynamics/__init__.py` from `git add` when unchanged.

---

### Task 8: Exact-Head Verification, Scope Review, and Integration Evidence

**Files:**
- No new production files expected.
- Review all base→head changes against the approved scope.
- Update PR body and roadmap only after their respective gates.

**Interfaces:**
- Consumes: the complete feature implementation and all proof evidence.
- Produces: exact-head feature GREEN evidence, clean scope review, PR synthetic-merge evidence, guarded merge/post-merge evidence, and final #27 closure only if all eight P2 checks are proven.

- [ ] **Step 1: Run every new module together**

```bash
python3 -m unittest \
  tests.test_narrative_identification_protocol \
  tests.test_narrative_identification_recovery \
  tests.test_narrative_identification_interventions \
  tests.test_narrative_identification_model_comparison \
  tests.test_narrative_identification_reporting -v
```

Expected: all GREEN.

- [ ] **Step 2: Run the exact full Python CI gate**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`; record the exact final test count and elapsed time. Do not reuse the old 659 count as if it were the new total.

- [ ] **Step 3: Run/inspect the full `proof.yml` gate on the exact final feature head**

Authoritative success requires one workflow run with:

```text
head_sha == exact final feature commit
status == completed
conclusion == success
```

and GREEN steps for Lean dependency resolution, conformance vectors, full Lean build, Lean theorem tests, Python numerical tests, Narrative story theorem tests, and Narrative testimony theorem tests. Capture the exact checkout SHA and Python `Ran N tests ... / OK` log lines.

- [ ] **Step 4: Perform base→head scope review**

Compare exact base `f628dcea4176731e3fe7c5ec40170e63d0233759` to final feature head. Allowed changed paths are only:

```text
docs/superpowers/specs/2026-08-28-narrative-synthetic-identification-v1-design.md
docs/superpowers/plans/2026-08-28-narrative-synthetic-identification-v1.md
narrative_dynamics/contracts.py
narrative_dynamics/identification.py
narrative_dynamics/adapters/narrative_prison_identification.py
fixtures/observations/narrative_identification_v1.json
tests/test_narrative_identification_protocol.py
tests/test_narrative_identification_recovery.py
tests/test_narrative_identification_interventions.py
tests/test_narrative_identification_model_comparison.py
tests/test_narrative_identification_reporting.py
```

Optionally allow `narrative_dynamics/__init__.py` only when it contains additive exports required by a locked RED public API test. Any other changed production path blocks PR creation.

Explicitly verify unchanged: P1 `narrative_prison.py`, all runtime decision family modules, dispatch, calibration/uncertainty/loss/validation/model-comparison core, existing fixtures, stochastic world/observation modules, and Lean.

- [ ] **Step 5: Review scientific claims against actual findings**

Require no Critical/Important review findings. Check that:

- joint beta finding retains only truth and identifies both target coordinates;
- pressure/instrumentality separate recoveries identify truth;
- scale-confounded finding retains truth plus the preregistered equivalence class and reports not identified;
- observational equivalence uses direct policy vectors and `1e-12` tolerance;
- memory pair is information-only and separates Intentional from Reactive;
- future pair is information-only and separates Planning from Intentional;
- Brier and Log findings bind exact same frozen candidates/targets/seeds;
- Log did not participate in candidate selection;
- no conclusion claims structural/empirical/real-human identification.

- [ ] **Step 6: Invoke verification-before-completion and finishing-development-branch**

Do not claim DONE merely from local tests. Use the relevant Superpowers verification skill with fresh exact-head evidence, then finishing-development-branch for integration choice.

- [ ] **Step 7: If PR integration is chosen, create PR and require synthetic-merge GREEN**

PR title:

```text
feat: add narrative synthetic identification v1
```

PR body must record:

- integrated base SHA;
- design/plan paths;
- authoritative test-only RED SHA/run/failure counts;
- exact final feature GREEN SHA/run/full test count;
- accepted-set/non-identifiability claim boundary;
- exact information intervention results;
- Brier-selected / Log-rescored sibling protocol evidence;
- exact changed-file scope;
- statement that no runtime decision/calibration/uncertainty/loss/Lean core changed.

Before merge, require a `pull_request` `proof` run on the current feature head and current base to complete successfully. The synthetic checkout must be the GitHub merge of those exact SHAs.

- [ ] **Step 8: Guard merge and require post-merge exact-head GREEN**

Merge only with `expected_head_sha=<exact approved feature head>`. Then fetch the new `proof/narrative-dynamics-v0` integration SHA and require push-triggered `proof.yml` completed/success on that exact merge commit. Capture the post-merge Python `Ran N tests ... / OK` line and all Lean/story/testimony gates.

- [ ] **Step 9: Update roadmap #27 only after post-merge GREEN**

Check all eight remaining P2 boxes only if evidence supports every one:

```text
Intentional vs reactive comparison
Intentional vs POMDP comparison
Information intervention suite
Synthetic recovery for beta_goal and beta_action
Recovery / non-recovery cases for goal and instrumentality parameters
Observational-equivalence fixtures
Held-out Brier/log-score comparison
Explicit reporting when latent cognition is not identifiable
```

Update the integrated baseline and proof history with the merge SHA and post-merge run. If all eight are checked and no other workstream remains, close #27 as completed. If any criterion is not proven, leave that checkbox unchecked and keep #27 open.
