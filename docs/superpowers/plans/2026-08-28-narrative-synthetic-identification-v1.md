# Narrative Synthetic Identification V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete #27 P2 Identification and Model Comparison with a frozen synthetic protocol that distinguishes model-family equivalence/separation from latent-parameter identifiability, proves both recovery and deliberate non-recovery cases, certifies information-only interventions, and scores one Brier-selected candidate set under both held-out Brier and Log losses.

**Architecture:** Add one generic orchestration/report module, `narrative_dynamics.identification`, and one isolated synthetic benchmark adapter, `narrative_dynamics.adapters.narrative_prison_identification`. Reuse the existing simulation, calibration, accepted-set, identifiability, held-out validation, proper-scoring, preregistration, unified runtime dispatch, and report-attestation cores without modifying them. Parameter identification is decided from complete accepted candidate sets; model families are described only as equivalent/separated/discriminating; Log is final re-scoring of the exact Brier-selected candidates rather than a second selection path.

**Tech Stack:** Python 3 stdlib (`dataclasses`, `enum`, `math`, `statistics`, `MappingProxyType`, `unittest`, `json`, `pathlib`), existing `narrative_dynamics` simulation/calibration/uncertainty/validation/loss/preregistration/model-comparison/report-artifact APIs, GenericNarrative Reactive/Intentional/Planning runtimes, unified `RuntimeDecisionModelSpec` / `run_runtime_decision()`, GitHub Actions `proof.yml`.

**Spec:** `docs/superpowers/specs/2026-08-28-narrative-synthetic-identification-v1-design.md`

## Global Constraints

- Integrated base: `proof/narrative-dynamics-v0@f628dcea4176731e3fe7c5ec40170e63d0233759`.
- Feature branch: `work/narrative-synthetic-identification-v1`.
- Approved written-spec head before planning: `5f703ec1da1faa96fdcd62e76cc7384387cb06a4`.
- Inherited exact-head evidence: post-merge proof #1158, Python `659 tests` / `OK`, all Lean/story/testimony gates GREEN on the integrated base.
- Docs-only spec/plan commits do not trigger `proof.yml`; the first authoritative feature proof is the complete test/fixture-only RED commit.
- Strict sequence: complete RED -> exact-head RED proof -> monotonic task-local GREEN commits -> exact-head feature GREEN -> final scope/review -> finishing-development-branch integration choice -> PR synthetic-merge GREEN -> guarded merge -> post-merge exact-head GREEN -> #27 update.
- Do not run Docker.
- Production changes are limited to: one additive `ExperimentStage` member in `narrative_dynamics/contracts.py`; new `narrative_dynamics/identification.py`; new `narrative_dynamics/adapters/narrative_prison_identification.py`.
- Data/tests are limited to: `fixtures/observations/narrative_identification_v1.json` and five `tests/test_narrative_identification_*.py` modules named below.
- Do not modify package-root exports, P1 `narrative_prison.py`, any runtime decision-family implementation, runtime dispatch, `calibration.py`, `uncertainty.py`, `losses.py`, `validation.py`, `model_comparison.py`, observation dataset/target/preregistration/release core, stochastic world/observation code, existing fixtures, or Lean sources.
- Claim scope is exactly `synthetic_protocol_only`.
- Parameter statuses are exactly `identified_under_protocol` and `not_identified_under_protocol`.
- Model-family findings never use parameter-identification statuses.
- Empty accepted set or true tuple excluded from accepted set raises `IdentificationRecoveryError`; it is never reported as non-identifiability.
- Accepted-set membership is `candidate_block_loss <= block_best_loss + acceptance_loss_delta`, followed by the preregistered minimum accepted-block fraction.
- Coordinate-wise identification delegates to existing `diagnose_identifiability()` and is never described as global structural identifiability.
- Direct observational equivalence tolerance is absolute `1e-12` on the complete declared action-policy vector.
- Non-finite candidate loss or policy coordinates fail closed; they never become ties, accepted candidates, or equivalence evidence.
- Information intervention certification requires exactly one declared information leaf to change; all other scenario payload leaves are exact.
- Joint beta grid: `beta_goal=(0.5,1.0,2.0,4.0)` × `beta_action=(0.5,1.0,2.0,4.0)`, with `goal_pressure_scale=1.0`, `instrumentality_scale=1.0`, truth `(2.0,1.0,1.0,1.0)`.
- Pressure/instrumentality grids are `(0.5,1.0,2.0)`; the full 3×3 joint grid must retain exactly the product-one confounded scale pairs `(0.5,2.0)`, `(1.0,1.0)`, `(2.0,0.5)` when betas are fixed at truth.
- Recovery generation seeds: `(11,12)`. Calibration seed blocks: `((101,102),(111,112))`.
- Final candidate grids: Reactive `beta_action=(0.5,1.0,2.0,4.0)`; Intentional beta 4×4 with both scale coordinates fixed to `(1.0,)`; Planning `beta_action=(0.5,1.0,2.0,4.0)`.
- Brier training seeds: `(201,202)`. Brier selection seeds: `(301,302)`. Final Brier/Log seeds: `(401,402)`.
- Recovery defaults: `acceptance_loss_delta=0.0`, `min_acceptance_fraction=1.0`.
- Metric: existing `prison_initial_action_metrics`, keys `initial.scout`, `initial.escape`, `initial.submit`.
- Brier is the only training/selection loss. Log performs no fit and no selection.
- Brier and Log final protocols share exact dataset/partition/target/metric/seeds/baseline/candidate identities and selected parameters. Only protocol name, loss identity, loss-specific thresholds, declared precommitment hash, and derived protocol hash may differ.
- Brier thresholds: `AdequacyThresholds(2.0,2.0)`. Log thresholds: `AdequacyThresholds(20.0,20.0)`.
- Final aggregate report must pass `attest_report(report).require_integrity()`.
- No #27 P2 Identification and Model Comparison checkbox is updated before merge plus post-merge exact-head GREEN.

## File Structure

- `narrative_dynamics/contracts.py` — add only `ExperimentStage.SYNTHETIC_IDENTIFICATION`.
- `narrative_dynamics/identification.py` — typed protocol/errors/findings, accepted-set interpretation, recovery orchestration, equivalence/intervention certification, sibling final-protocol validation, per-stratum score aggregation, aggregate report.
- `narrative_dynamics/adapters/narrative_prison_identification.py` — closed synthetic scenario schema, three family builders/sources, intentional latent parameterization, memory-evidence and future-information semantics, unified dispatch, canonical three-action output.
- `fixtures/observations/narrative_identification_v1.json` — case-oriented synthetic train/selection/final fixture with explicit non-empirical provenance.
- Tests: `test_narrative_identification_protocol.py`, `test_narrative_identification_recovery.py`, `test_narrative_identification_interventions.py`, `test_narrative_identification_model_comparison.py`, `test_narrative_identification_reporting.py`.

---

### Task 1: Complete the Authoritative Test/Fixture-Only RED Boundary

**Files:**
- Create: `fixtures/observations/narrative_identification_v1.json`
- Create: all five new test modules listed above.

**Interfaces:**
- Consumes existing public infrastructure only.
- Produces the complete P2 acceptance boundary before any new production symbol exists.

- [ ] **Step 1: Generate the committed fixture with the existing serializer**

Run this one-off script; commit only the resulting JSON:

```python
from __future__ import annotations
import json
from pathlib import Path
from narrative_dynamics.contracts import Scenario, stable_content_hash
from narrative_dynamics.observations.dataset import ObservationCase, ObservationDataset, ObservationPartitionRole

OUT = Path("fixtures/observations/narrative_identification_v1.json")

def payload(*, mode, structure, goal_gap, memory, future):
    return {
        "mode": mode,
        "latent_structure": structure,
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
    # name, role, stratum, mode, structure, gap, memory, future, counts, pair
    ("train-temp-anchor", "train", "latent_temperature", "latent_temperature", "identical", 0.0, False, False, {"scout":0,"escape":79,"submit":21}, None),
    ("train-goal-low", "train", "latent_temperature", "latent_temperature", "mirror", 0.4, False, False, {"scout":0,"escape":61,"submit":39}, None),
    ("train-memory-hidden", "train", "memory_evidence", "memory_evidence", "belief_driven", 0.0, False, False, {"scout":0,"escape":45,"submit":55}, None),
    ("train-future-off", "train", "future_information", "future_information", "value_of_information", 0.0, False, False, {"scout":5,"escape":45,"submit":50}, None),
    ("selection-goal-high", "selection_validation", "latent_temperature", "latent_temperature", "mirror", 1.1, False, False, {"scout":0,"escape":73,"submit":27}, None),
    ("selection-memory-revealed", "selection_validation", "memory_evidence", "memory_evidence", "belief_driven", 0.0, True, False, {"scout":0,"escape":75,"submit":25}, None),
    ("selection-future-on", "selection_validation", "future_information", "future_information", "value_of_information", 0.0, False, True, {"scout":55,"escape":30,"submit":15}, None),
    ("final-equivalence", "final_test", "observational_equivalence", "latent_temperature", "mirror", 0.0, False, False, {"scout":0,"escape":50,"submit":50}, None),
    ("final-memory-hidden", "final_test", "memory_evidence", "memory_evidence", "belief_driven", 0.0, False, False, {"scout":0,"escape":45,"submit":55}, "final-memory-revealed"),
    ("final-memory-revealed", "final_test", "memory_evidence", "memory_evidence", "belief_driven", 0.0, True, False, {"scout":0,"escape":75,"submit":25}, "final-memory-hidden"),
    ("final-future-off", "final_test", "future_information", "future_information", "value_of_information", 0.0, False, False, {"scout":5,"escape":45,"submit":50}, "final-future-on"),
    ("final-future-on", "final_test", "future_information", "future_information", "value_of_information", 0.0, False, True, {"scout":55,"escape":30,"submit":15}, "final-future-off"),
)
roles = {"train":ObservationPartitionRole.TRAIN,"selection_validation":ObservationPartitionRole.SELECTION_VALIDATION,"final_test":ObservationPartitionRole.FINAL_TEST}
cases = []
for name, role, stratum, mode, structure, gap, memory, future, counts, pair in SPECS:
    scenario = Scenario(name + "-scenario", payload(mode=mode, structure=structure, goal_gap=gap, memory=memory, future=future))
    provenance = {
        "synthetic_non_empirical": True,
        "stratum": stratum,
        "generator_family": "synthetic-reference-policy-v1",
        "generator_parameters_hash": stable_content_hash({"scenario_hash":scenario.content_hash,"counts":counts}),
        "information_condition": "memory_revealed" if memory else "future_available" if future else "baseline",
    }
    if pair is not None:
        provenance["paired_case_id"] = pair
    cases.append(ObservationCase(name=name, role=roles[role], scenario=scenario, counts=counts, observation_ids=(name + "-obs",), provenance=provenance))
dataset = ObservationDataset(name="narrative-synthetic-identification", version="1", source={"kind":"synthetic_reference_policy","version":"1"}, provenance={"synthetic_non_empirical":True,"claim_scope":"synthetic_protocol_only","design_spec":"2026-08-28-narrative-synthetic-identification-v1-design"}, cases=tuple(cases))
OUT.write_text(json.dumps(dataset.to_payload(), indent=2, sort_keys=True) + "\n")
print(dataset.content_hash)
```

Verify:

```bash
python3 - <<'PY'
from narrative_dynamics.observations import load_observation_dataset
d = load_observation_dataset("fixtures/observations/narrative_identification_v1.json")
assert d.to_payload()["content_hash"] == d.content_hash
assert d.provenance["synthetic_non_empirical"] is True
print(d.content_hash)
PY
```

- [ ] **Step 2: Add protocol RED tests**

Guard imports so discovery continues when the new module is absent. Lock:

```text
test_manifest_stage_is_additive_and_exact
test_fixture_is_synthetic_partitioned_and_pair_linked
test_protocol_create_binds_case_hashes_and_canonical_hash
test_protocol_claim_scope_is_closed
test_recovery_truth_must_be_in_grid
test_protocol_rejects_duplicate_experiment_and_pair_names
test_protocol_rejects_partition_overlap
test_sibling_final_protocols_allow_only_name_loss_threshold_and_derived_hash_difference
test_sibling_final_protocols_reject_candidate_target_metric_seed_or_baseline_drift
test_protocol_rejects_nonfinite_grid_tolerance_and_acceptance_values
```

- [ ] **Step 3: Add recovery/equivalence RED tests**

Use:

```python
TRUTH = {"beta_goal":2.0,"beta_action":1.0,"goal_pressure_scale":1.0,"instrumentality_scale":1.0}
BETA_GRID = (0.5,1.0,2.0,4.0)
SCALE_GRID = (0.5,1.0,2.0)
GENERATION_SEEDS = (11,12)
CALIBRATION_BLOCKS = ((101,102),(111,112))
```

Lock:

```text
test_joint_beta_goal_beta_action_recovery_is_exact
test_action_temperature_anchor_is_invariant_to_beta_goal
test_low_and_high_goal_gaps_change_with_beta_goal
test_equivalence_fixture_retains_multiple_latent_candidates_and_reports_nonidentification
test_pressure_recovery_with_fixed_instrumentality
test_instrumentality_recovery_with_fixed_pressure
test_scale_confounded_full_grid_retains_exact_product_one_equivalence_class
test_truth_excluded_is_recovery_error_not_nonidentification
test_empty_accepted_set_is_recovery_error
test_candidate_loss_table_retains_every_grid_candidate_and_ties
test_observational_equivalence_certifies_complete_policy_not_equal_loss
test_nonfinite_candidate_loss_or_policy_coordinate_is_rejected
```

Scale-confounded uses the full 3×3 scale grid with betas fixed at truth and requires accepted scale pairs exactly `{(0.5,2.0),(1.0,1.0),(2.0,0.5)}`.

- [ ] **Step 4: Add intervention RED tests**

Lock:

```text
test_memory_pair_diff_is_exactly_memory_availability
test_future_pair_diff_is_exactly_future_information_availability
test_reward_drift_is_rejected
test_action_semantic_drift_is_rejected
test_multiple_changed_information_fields_are_rejected
test_memory_evidence_intervention_separates_intentional_from_reactive
test_future_information_intervention_separates_planning_from_intentional
test_intervention_finding_replays_and_canonicalizes_family_order
```

- [ ] **Step 5: Add model-comparison RED tests**

Lock:

```text
test_identification_adapter_uses_unified_dispatch_and_not_identification_or_comparison_protocol
test_source_parameter_schemas_match_real_family_semantics
test_all_sources_accept_every_fixture_mode
test_binary_modes_zero_fill_scout_without_changing_dispatch_policy
test_brier_training_and_selection_freeze_one_candidate_per_family
test_brier_final_protocol_completes_on_exact_frozen_candidates
test_log_sibling_protocol_reuses_exact_brier_candidates_targets_and_seeds
test_log_candidate_or_seed_drift_is_rejected_before_final_scoring
test_scored_findings_expose_equivalence_memory_and_future_strata
test_comparison_does_not_require_one_family_to_win_every_stratum
```

- [ ] **Step 6: Add reporting RED tests**

Lock:

```text
test_final_report_requires_all_eight_roadmap_evidence_categories
test_final_report_manifest_stage_is_synthetic_identification_and_parents_are_complete
test_attested_report_round_trip_requires_integrity
test_parameter_nonidentification_language_is_protocol_scoped
test_model_family_language_uses_equivalent_or_separated_not_identified
test_forged_protocol_hash_is_rejected
test_forged_parent_manifest_set_is_rejected
test_missing_required_finding_prevents_completion
```

- [ ] **Step 7: Run and commit authoritative RED**

```bash
python3 -m unittest tests.test_narrative_identification_protocol tests.test_narrative_identification_recovery tests.test_narrative_identification_interventions tests.test_narrative_identification_model_comparison tests.test_narrative_identification_reporting -v
python3 -m unittest discover -s tests -v
```

Expected: inherited 659 tests remain GREEN; all failures/errors are confined to the five new P2 modules. Record exact totals/failures/errors.

```bash
git add fixtures/observations/narrative_identification_v1.json tests/test_narrative_identification_*.py
git commit -m "test: define narrative synthetic identification red boundary"
```

Push and require exact-head Actions RED evidence before production work.

---

### Task 2: Add Protocol Types, Typed Errors, and Accepted-Set Interpretation

**Files:**
- Modify: `narrative_dynamics/contracts.py`
- Create: `narrative_dynamics/identification.py`

**Interfaces produced:** `IdentificationStatus`, typed errors, `ParameterRecoveryExperiment`, `ParameterCandidateLoss`, `ParameterIdentificationFinding`, `InformationInterventionPair`, `SyntheticIdentificationProtocol`, `interpret_parameter_identification()`, `validate_sibling_final_protocols()`.

- [ ] **Step 1: Add only the manifest stage**

```python
SYNTHETIC_IDENTIFICATION = "synthetic_identification"
```

- [ ] **Step 2: Add exact errors/status and canonical validators**

```python
class SyntheticIdentificationError(ValueError): pass
class IdentificationProtocolError(SyntheticIdentificationError): pass
class IdentificationRecoveryError(SyntheticIdentificationError): pass
class InterventionCertificationError(SyntheticIdentificationError): pass
class IdentificationComparisonError(SyntheticIdentificationError): pass
class IdentificationReportError(SyntheticIdentificationError): pass

class IdentificationStatus(str, Enum):
    IDENTIFIED_UNDER_PROTOCOL = "identified_under_protocol"
    NOT_IDENTIFIED_UNDER_PROTOCOL = "not_identified_under_protocol"
```

Validators reject untrimmed/empty text, bool-as-number, non-finite floats, malformed hashes, duplicate/non-int seeds, empty blocks, malformed grids, non-finite losses/policies.

- [ ] **Step 3: Implement recovery declaration and candidate rows**

`ParameterRecoveryExperiment` fields are exactly: `name`, `model_identity`, `true_parameters`, `parameter_grid`, `target_coordinates`, `cases`, `generation_seeds`, `calibration_seed_blocks`, `acceptance_loss_delta`, `min_acceptance_fraction`, `metric_identity`, `loss_identity`, `expected_status`. It exposes `candidate_parameters` as the exact Cartesian expansion and `content_hash` over the canonical declaration.

`ParameterCandidateLoss` fields are exactly: `parameters`, `block_losses`, `mean_loss`, `accepted_blocks`, `acceptance_fraction`.

Require truth in the Cartesian expansion; target coordinates are a nonempty subset of parameter names; cases/seeds/blocks are nonempty and unique where identity requires uniqueness.

- [ ] **Step 4: Implement accepted-set interpretation**

`ParameterIdentificationFinding` fields: `experiment_hash`, `true_parameters`, complete canonical `candidate_losses`, `accepted_parameters: ParameterAcceptanceSet`, `truth_retained`, `identifiability: IdentifiabilityReport`, `status`, `parent_manifest_hashes`; it has canonical `content_hash`.

`interpret_parameter_identification(experiment, candidate_losses, parent_manifest_hashes=())` must:

```python
expected = set(experiment.candidate_parameters)
actual = {row.parameters for row in candidate_losses}
if actual != expected or len(actual) != len(candidate_losses):
    raise IdentificationRecoveryError("candidate-loss table must cover the preregistered grid exactly once")
accepted = tuple(row.parameters for row in candidate_losses if row.acceptance_fraction >= experiment.min_acceptance_fraction)
if not accepted:
    raise IdentificationRecoveryError("synthetic identification accepted set is empty")
if experiment.true_parameters not in accepted:
    raise IdentificationRecoveryError("synthetic identification did not retain its true parameters")
identifiability = diagnose_identifiability(accepted)
```

Then derive status only from declared target coordinates; if identified, each singleton must equal truth. Enforce preregistered expected status.

- [ ] **Step 5: Implement intervention declaration**

`InformationInterventionPair` fields: `name`, `baseline`, `intervention`, `allowed_information_path`, `frozen_paths`, `stratum`, `expected_relationship`, plus canonical `content_hash`.

For V1 both scenario payload key sets must be identical, the allowed path must be one top-level key, and `frozen_paths` must equal every other top-level key exactly.

- [ ] **Step 6: Implement protocol with explicit case-hash binding and a single construction path**

`SyntheticIdentificationProtocol` fields are exactly:

```text
name, version, claim_scope, fixture_hash,
adapter_identities, recovery_experiments, intervention_pairs,
train_partition_hash, selection_partition_hash, final_partition_hash,
train_case_hashes, selection_case_hashes, final_case_hashes,
target_spec_hash, metric_identity, brier_loss_identity, log_loss_identity,
brier_training_seeds, brier_selection_seeds, final_seeds, equivalence_tolerance
```

Add `SyntheticIdentificationProtocol.create(...)` taking the actual `ObservationDataset`, adapter sources/identities, experiments, pairs, target spec/extractor/losses/seeds. It derives partition hashes and sorted scenario-content-hash tuples for each role. `__post_init__` independently verifies the three case-hash sets are pairwise disjoint, claim scope is exact, tolerance is `1e-12`, names are unique, and declarations canonicalize by name. Thus direct construction cannot bypass partition-overlap validation.

- [ ] **Step 7: Implement sibling final-protocol validation**

Require exact equality of `version,dataset_hash,train_partition_hash,selection_partition_hash,final_partition_hash,target_spec_hash,final_target_hash,final_target_manifest_hash,metric_identity,simulation_seeds,baseline_name,candidates`. Require Brier loss name `categorical_brier` and Log loss name `categorical_log`. Only name/loss/threshold/precommitment/derived hash may differ.

- [ ] **Step 8: Run targeted tests and commit**

```bash
python3 -m unittest tests.test_narrative_identification_protocol tests.test_narrative_identification_recovery -v
git add narrative_dynamics/contracts.py narrative_dynamics/identification.py
git commit -m "feat: add synthetic identification protocol core"
```

Expected: protocol/core tests GREEN; adapter/evaluation tests remain RED. Require exact-head monotonic failure reduction.

---

### Task 3: Add the Isolated Narrative Identification Adapter

**Files:**
- Create: `narrative_dynamics/adapters/narrative_prison_identification.py`

**Interfaces produced:** `NarrativePrisonIdentificationSource`, `create_narrative_identification_reactive_source()`, `create_narrative_identification_intentional_source()`, `create_narrative_identification_planning_source()`.

- [ ] **Step 1: Add closed source contracts**

Use family names `reactive,intentional,planning`, source version `1.0.0`, revision `narrative-synthetic-identification-v1`, lifecycle `fresh_per_batch`, and source names `generic-narrative-identification-{family}`. Source identity attests source class, model class, scenario builder, family builder, and `run_runtime_decision` implementation.

- [ ] **Step 2: Validate exact scenario/parameter schemas**

Allowed `(mode, latent_structure)` pairs:

```python
{("latent_temperature","identical"),("latent_temperature","mirror"),("memory_evidence","belief_driven"),("future_information","value_of_information")}
```

Scenario keys are exactly Task 1 payload keys. Intentional parameters are exactly `beta_goal,beta_action,goal_pressure_scale,instrumentality_scale`. Reactive/Planning parameters are exactly `beta_action`. All are positive finite non-bool floats.

- [ ] **Step 3: Implement latent-temperature mode for all families**

Binary authored actions are `escape,submit`; outer output zero-fills scout.

Intentional identical structure uses the same conditional values for both goals: `escape=+action_gap/2`, `submit=-action_gap/2`, making the anchor invariant to `beta_goal`. Intentional mirror structure uses opposite conditional values and a goal score gap `goal_score_gap * goal_pressure_scale * instrumentality_scale`, split `+gap/2,-gap/2`; mirror + zero gap is the deliberate equivalence case. Instantiate existing `GoalModelSpec`/`ChoiceModelSpec`; do not reimplement their softmax/marginal math.

Reactive latent-temperature uses its existing runtime reactive softmax over direct binary scores `escape=+action_gap/2`, `submit=-action_gap/2`; it ignores goal gap/structure beyond schema validation because it has no latent goals.

Planning latent-temperature uses its existing planning runtime as a one-step/no-future-information problem with the same direct terminal utilities; it must accept both `identical` and `mirror` fixture cases without inventing goal parameters.

- [ ] **Step 4: Implement memory-evidence mode for all families**

Use hidden `guard.status ∈ {weak,strong}`, prior 0.5. Hidden pair adds no diagnostic evidence; revealed pair admits the prior weak signal through existing evidence/belief semantics with `memory_signal_accuracy`.

Reactive declares only current cues and uses prior-based direct binary scores, so its policy is exact across the pair. Intentional uses opposite weak/strong goal instrumentality, so posterior evidence changes goal/action policy. Planning must support this binary mode as a one-step planning problem using its belief state; no future observation is available. Binary output zero-fills scout.

- [ ] **Step 5: Implement future-information mode for all families**

All three actions authored. Reactive and Intentional use the same immediate reward/cost semantics on both pair members and never branch on `future_information_available`; scout is myopic. Planning uses existing finite planning transition/observation/reward hooks: scout is uninformative when false and uses `future_signal_accuracy` when true. Reward, persistence, action set, and discount remain exact.

- [ ] **Step 6: Dispatch only through the unified boundary**

```python
dispatch = run_runtime_decision(case.story, case.domain, case.decision_id, case.ledger, RuntimeDecisionModelSpec(self.family, nested))
policy = {"scout":0.0,"escape":0.0,"submit":0.0}
policy.update({action:float(probability) for action,probability in dispatch.action_policy.items()})
return ModelRun(events=(), outcome={"model_kind":self.family,"initial_policy":policy,"selected_action":dispatch.selected_action,"runtime_dispatch_hash":dispatch.content_hash,"runtime_dispatch":dispatch.to_dict(),"translated_story_hash":case.story.content_hash,"translated_domain_hash":case.domain.content_hash,"runtime_ledger_hash":case.ledger.content_hash})
```

Outer RNG is accepted but never samples an action.

- [ ] **Step 7: Run targeted tests and commit**

```bash
python3 -m unittest tests.test_narrative_identification_recovery tests.test_narrative_identification_interventions tests.test_narrative_identification_model_comparison -v
git add narrative_dynamics/adapters/narrative_prison_identification.py
git commit -m "feat: add narrative synthetic identification adapter"
```

Expected: source/schema/all-mode/raw-policy tests GREEN; orchestration tests may remain RED.

---

### Task 4: Implement Recovery and Direct Observational Equivalence

**Files:**
- Modify: `narrative_dynamics/identification.py`

**Interfaces produced:** `evaluate_parameter_recovery_experiment()`, `ObservationalEquivalenceFinding`, `certify_observational_equivalence()`.

- [ ] **Step 1: Implement multi-case target generation and block scoring by composing existing calibration**

`evaluate_parameter_recovery_experiment(runner, model, experiment, extractor, loss)` preflights exact source/extractor/loss identities. For each case, generate target metrics from true parameters with `runner.run_batch(... generation_seeds)` + `aggregate_metrics()`. For every calibration block, call existing `calibrate_grid()` once per case with the exact full grid; aggregate same-candidate per-case losses by `fmean`; reject any non-finite loss; mark block acceptance using the preregistered delta. Build one complete candidate row per Cartesian candidate and call `interpret_parameter_identification()`.

- [ ] **Step 2: Lock four recovery experiments**

- Joint beta: anchor + low-gap + high-gap, 4×4 beta grid, scales fixed 1, expected identified, accepted set exactly truth.
- Pressure: mirror discriminating cases, pressure 3-grid, instrumentality 1, betas truth, expected identified.
- Instrumentality: mirror of pressure, expected identified.
- Scale-confounded: full pressure 3-grid × instrumentality 3-grid, betas truth, expected not identified, accepted set exactly the three product-one tuples.

- [ ] **Step 3: Implement direct complete-policy equivalence**

`ObservationalEquivalenceFinding` fields: `name,left_model_identity,right_model_identity,case_hashes,action_keys,policies,max_abs_policy_delta,tolerance,equivalent,parent_manifest_hashes`, plus canonical `content_hash`.

`certify_observational_equivalence(name, runner, left_model, right_model, left_parameters, right_parameters, cases, seeds, extractor, action_keys, tolerance=1e-12)` aggregates complete policy metrics over seeds, rejects non-finite/missing coordinates, and sets equivalence only from max absolute policy delta. Scalar loss is not an input. The same model source may appear on both sides with different latent parameters.

- [ ] **Step 4: Run recovery suite and commit**

```bash
python3 -m unittest tests.test_narrative_identification_recovery -v
git add narrative_dynamics/identification.py
git commit -m "feat: add synthetic latent recovery and equivalence"
```

Expected: all recovery/equivalence tests GREEN.

---

### Task 5: Implement Information-Only Intervention Certification

**Files:**
- Modify: `narrative_dynamics/identification.py`

**Interfaces produced:** `certify_information_intervention()`, `InformationInterventionFinding`, `evaluate_information_intervention()`.

- [ ] **Step 1: Implement sorted recursive payload diff**

`_scenario_payload_diff(left,right,prefix=())` recursively compares mappings by sorted key and returns canonical leaf triples `(path,before,after)` without coercion.

- [ ] **Step 2: Certify exactly one allowed leaf**

```python
diffs = _scenario_payload_diff(pair.baseline.payload, pair.intervention.payload)
if len(diffs) != 1:
    raise InterventionCertificationError("information intervention must change exactly one payload leaf")
path,before,after = diffs[0]
if path != pair.allowed_information_path:
    raise InterventionCertificationError("information intervention changed an undeclared field")
```

Because pair construction locks the complete complement in `frozen_paths`, this proves reward/dynamics/action fields unchanged.

- [ ] **Step 3: Implement family policy-delta finding/evaluator**

`InformationInterventionFinding` fields: `name,pair_hash,certified_diff,family_policies,within_family_max_deltas,cross_family_contrasts,tolerance,discriminating,parent_manifest_hashes`, plus canonical `content_hash`.

`evaluate_information_intervention(pair, runner, families, seeds, extractor, action_keys, tolerance=1e-12)` certifies first, canonicalizes family order, aggregates finite complete policies, computes max deltas. For `intentional_vs_reactive`: Reactive `<=1e-12`, Intentional `>1e-12`. For `planning_vs_intentional`: Intentional `<=1e-12`, Planning `>1e-12`. Unsupported relationship raises `IdentificationProtocolError`.

- [ ] **Step 4: Run/commit**

```bash
python3 -m unittest tests.test_narrative_identification_interventions -v
git add narrative_dynamics/identification.py
git commit -m "feat: certify narrative information interventions"
```

---

### Task 6: Freeze Brier Candidates and Add Log Sibling Final Scoring

**Files:**
- Modify: `narrative_dynamics/identification.py`

**Interfaces produced:** `StratumModelScore`, `ScoredModelComparisonFinding`, `score_model_comparison_by_stratum()`.

- [ ] **Step 1: Run Brier-only training/selection with existing APIs**

```python
spec = CategoricalTargetSpec(name="narrative-identification-initial-action", version="1", categories=("scout","escape","submit"), metric_prefix="initial")
group = CategoricalMetricGroup("initial-action", ("initial.scout","initial.escape","initial.submit"))
brier = CategoricalBrierLoss((group,))
log = CategoricalLogLoss((group,))
```

For each family: `fit_training_target_grid(... loss=brier)` using exact grid; build `ParameterAcceptanceSet` from training candidates; `select_on_validation_suite(... loss=brier)`; freeze with `FrozenModelCandidate.from_selection`. No Log call occurs in training or selection.

- [ ] **Step 2: Create exact sibling final protocols over the same frozen tuple**

Brier protocol name `narrative-synthetic-identification-brier-v1`, thresholds `(2.0,2.0)`; Log name `narrative-synthetic-identification-log-v1`, thresholds `(20.0,20.0)`; both version `1`, exact same dataset/spec/metric/seeds `(401,402)`, intentional baseline, exact same `frozen_candidates`. Call `validate_sibling_final_protocols()` before scoring.

- [ ] **Step 3: Execute both existing final comparisons**

Use one canonical `ComparisonModel` tuple and one `final_targets` object. Call `compare_models_on_final_partition` once under Brier and once under Log. No candidate fitting/selection occurs between calls.

- [ ] **Step 4: Aggregate existing per-case losses by stratum**

`StratumModelScore`: `stratum,model_name,mean_loss,worst_loss,case_names`.

`ScoredModelComparisonFinding`: `protocol_hash,loss_identity,candidate_hashes,final_partition_hash,final_target_hash,final_seeds,global_scores,stratum_scores,pairwise_mean_deltas,parent_comparison_manifest_hash`, plus canonical `content_hash`.

`score_model_comparison_by_stratum(dataset,protocol,report)` maps final cases to `provenance["stratum"]`, rejects missing/non-finite case losses, and requires exact final strata `observational_equivalence,memory_evidence,future_information`. It aggregates existing `FinalTestReport` case losses and never reruns models.

- [ ] **Step 5: Run/commit**

```bash
python3 -m unittest tests.test_narrative_identification_model_comparison -v
git add narrative_dynamics/identification.py
git commit -m "feat: add dual-score synthetic model comparison"
```

---

### Task 7: Build the Aggregate Report and Attestation Boundary

**Files:**
- Modify: `narrative_dynamics/identification.py`

**Interfaces produced:** `SyntheticIdentificationReport`, `build_synthetic_identification_report()`, deterministic scoped conclusions.

- [ ] **Step 1: Add final report type**

Fields: `protocol_hash,claim_scope,parameter_findings,equivalence_findings,intervention_findings,brier_comparison,log_comparison,conclusions,manifest`. `__post_init__` validates exact claim scope, canonical unique child identities, Brier/Log frozen identity equality, manifest stage/inputs, and exact complete parent-hash union.

- [ ] **Step 2: Require the complete evidence set in the builder**

`build_synthetic_identification_report(...)` maps finding hashes back through the frozen protocol and requires: joint beta identified; pressure identified; instrumentality identified; scale-confounded not identified; observational equivalence true; memory Intentional-vs-Reactive discriminating; future Intentional-vs-Planning discriminating; Brier/Log same frozen candidates/final identities. Missing or contradictory evidence raises `IdentificationReportError`.

- [ ] **Step 3: Build final manifest from child evidence**

Use `ExperimentStage.SYNTHETIC_IDENTIFICATION`. Manifest inputs contain protocol hash and every child finding hash. Parent hashes are exactly the sorted union of every parameter/equivalence/intervention run/calibration manifest plus both comparison manifests. Report construction recomputes and validates the union.

- [ ] **Step 4: Render only protocol-scoped language**

Parameter conclusions use exactly `X is identified under this frozen synthetic protocol.` or `X is not identified under this frozen synthetic protocol.` Family conclusions use equivalent/separated/discriminating wording only. Never emit structural/empirical/real-human cognition claims.

- [ ] **Step 5: Run/commit**

```bash
python3 -m unittest tests.test_narrative_identification_reporting -v
git add narrative_dynamics/identification.py
git commit -m "feat: attest narrative synthetic identification report"
```

---

### Task 8: Exact-Head Verification, Scope Review, and Integration Evidence

**Files:** No new production files.

- [ ] **Step 1: Run all five new modules together**

```bash
python3 -m unittest tests.test_narrative_identification_protocol tests.test_narrative_identification_recovery tests.test_narrative_identification_interventions tests.test_narrative_identification_model_comparison tests.test_narrative_identification_reporting -v
```

Expected: GREEN.

- [ ] **Step 2: Run exact full Python CI gate**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`; record exact new total/elapsed time rather than reusing 659.

- [ ] **Step 3: Require exact-head `proof.yml` completed/success**

Record feature SHA/run/job, exact checkout, Lean conformance/full build/theorems, Python `Ran N tests ... / OK`, story/testimony GREEN.

- [ ] **Step 4: Perform base→head scope review**

Allowed paths exactly:

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

Any other changed path blocks PR creation.

- [ ] **Step 5: Review actual scientific findings**

Require exact beta truth acceptance; identified pressure/instrumentality single-coordinate experiments; exact three-point scale-confounded set with truth retained; direct `1e-12` policy equivalence; certified information-only pairs; required model-family separation; exact Brier/Log frozen candidates/targets/seeds; no Log selection; no overclaim.

- [ ] **Step 6: Invoke verification-before-completion, then finishing-development-branch**

Do not claim DONE from local results. Use fresh exact-head evidence and present the finishing skill's integration choice.

- [ ] **Step 7: If PR integration is chosen, require PR synthetic-merge GREEN**

PR title: `feat: add narrative synthetic identification v1`. Body records base, spec/plan, RED proof, feature GREEN proof, accepted-set/non-identifiability boundary, intervention results, Brier-selected/Log-rescored evidence, exact scope, unchanged cores. Require pull-request proof completed/success for current exact head/base.

- [ ] **Step 8: Guard merge and require post-merge exact-head GREEN**

Merge with `expected_head_sha=<approved exact feature head>`. Require push-triggered `proof.yml` completed/success on exact new integrated merge SHA; capture `Ran N tests ... / OK` plus all Lean/story/testimony gates.

- [ ] **Step 9: Update #27 only after post-merge GREEN**

Evaluate all remaining checks together:

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

Check only proven items. If all eight are proven and no other workstream remains, update integrated baseline/proof history and close #27 as completed; otherwise keep unproven checks unchecked and issue open.
