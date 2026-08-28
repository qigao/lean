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
roles = {"train": ObservationPartitionRole.TRAIN, "selection_validation": ObservationPartitionRole.SELECTION_VALIDATION, "final_test": ObservationPartitionRole.FINAL_TEST}
cases = []
for name, role, stratum, mode, structure, gap, memory, future, counts, pair in SPECS:
    scenario = Scenario(name + "-scenario", payload(mode=mode, structure=structure, goal_gap=gap, memory=memory, future=future))
    provenance = {
        "synthetic_non_empirical": True,
        "stratum": stratum,
        "generator_family": "synthetic-reference-policy-v1",
        "generator_parameters_hash": stable_content_hash({"scenario_hash": scenario.content_hash, "counts": counts}),
        "information_condition": "memory_revealed" if memory else "future_available" if future else "baseline",
    }
    if pair is not None:
        provenance["paired_case_id"] = pair
    cases.append(ObservationCase(name=name, role=roles[role], scenario=scenario, counts=counts, observation_ids=(name + "-obs",), provenance=provenance))

dataset = ObservationDataset(
    name="narrative-synthetic-identification",
    version="1",
    source={"kind":"synthetic_reference_policy","version":"1"},
    provenance={"synthetic_non_empirical":True,"claim_scope":"synthetic_protocol_only","design_spec":"2026-08-28-narrative-synthetic-identification-v1-design"},
    cases=tuple(cases),
)
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

Guard imports so test discovery continues when `identification.py` is absent. Lock these methods:

```text
test_manifest_stage_is_additive_and_exact
test_fixture_is_synthetic_partitioned_and_pair_linked
test_protocol_canonicalizes_experiments_pairs_and_hash
test_protocol_claim_scope_is_closed
test_recovery_truth_must_be_in_grid
test_protocol_rejects_duplicate_experiment_and_pair_names
test_protocol_rejects_partition_overlap
test_sibling_final_protocols_allow_only_name_loss_threshold_and_derived_hash_difference
test_sibling_final_protocols_reject_candidate_target_metric_seed_or_baseline_drift
test_protocol_rejects_nonfinite_grid_tolerance_and_acceptance_values
```

Core guarded import:

```python
_IDENTIFICATION_IMPORT_ERROR = None
try:
    from narrative_dynamics.identification import IdentificationProtocolError, IdentificationStatus, InformationInterventionPair, ParameterRecoveryExperiment, SyntheticIdentificationProtocol, validate_sibling_final_protocols
except ImportError as error:
    _IDENTIFICATION_IMPORT_ERROR = error
```

- [ ] **Step 3: Add recovery/equivalence RED tests**

Use truth:

```python
TRUTH = {"beta_goal":2.0,"beta_action":1.0,"goal_pressure_scale":1.0,"instrumentality_scale":1.0}
BETA_GRID = (0.5,1.0,2.0,4.0)
SCALE_GRID = (0.5,1.0,2.0)
GENERATION_SEEDS = (11,12)
CALIBRATION_BLOCKS = ((101,102),(111,112))
```

Lock these methods:

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
```

The scale-confounded test uses the full 3×3 scale grid with betas fixed at truth and requires accepted scale pairs exactly `{(0.5,2.0),(1.0,1.0),(2.0,0.5)}`.

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

Use exact grids/seeds from Global Constraints and lock:

```text
test_identification_adapter_uses_unified_dispatch_and_not_identification_or_comparison_protocol
test_source_parameter_schemas_match_real_family_semantics
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
python3 -m unittest \
  tests.test_narrative_identification_protocol \
  tests.test_narrative_identification_recovery \
  tests.test_narrative_identification_interventions \
  tests.test_narrative_identification_model_comparison \
  tests.test_narrative_identification_reporting -v
python3 -m unittest discover -s tests -v
```

Expected: inherited 659 tests remain GREEN; every failure/error is confined to the new P2 boundary. Record exact new totals/failures/errors rather than predicting them.

```bash
git add fixtures/observations/narrative_identification_v1.json tests/test_narrative_identification_*.py
git commit -m "test: define narrative synthetic identification red boundary"
```

Push and require exact-head Actions evidence before production work.

---

### Task 2: Add Protocol Types, Typed Errors, and Accepted-Set Interpretation

**Files:**
- Modify: `narrative_dynamics/contracts.py`
- Create: `narrative_dynamics/identification.py`
- Test: protocol/recovery modules from Task 1.

**Interfaces:**
- Produces: `IdentificationStatus`, five typed error subclasses, `ParameterRecoveryExperiment`, `ParameterCandidateLoss`, `ParameterIdentificationFinding`, `InformationInterventionPair`, `SyntheticIdentificationProtocol`, `interpret_parameter_identification()`, `validate_sibling_final_protocols()`.

- [ ] **Step 1: Add only the manifest stage**

```python
SYNTHETIC_IDENTIFICATION = "synthetic_identification"
```

- [ ] **Step 2: Add exact errors/status**

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

Add private validators for trimmed text, finite non-bool floats, sha256 hashes, unique integer seed tuples/blocks, canonical parameter tuples, and finite nonempty grids.

- [ ] **Step 3: Implement recovery declaration/candidate rows**

```python
ParameterTuple = tuple[tuple[str,float], ...]

@dataclass(frozen=True)
class ParameterRecoveryExperiment:
    name: str
    model_identity: Mapping[str,object]
    true_parameters: ParameterTuple
    parameter_grid: tuple[tuple[str,tuple[float,...]], ...]
    target_coordinates: tuple[str,...]
    cases: tuple[Scenario,...]
    generation_seeds: tuple[int,...]
    calibration_seed_blocks: tuple[tuple[int,...], ...]
    acceptance_loss_delta: float
    min_acceptance_fraction: float
    metric_identity: Mapping[str,object]
    loss_identity: Mapping[str,object]
    expected_status: IdentificationStatus | None = None

    @property
    def candidate_parameters(self) -> tuple[ParameterTuple,...]: ...
    @property
    def content_hash(self) -> str: ...

@dataclass(frozen=True)
class ParameterCandidateLoss:
    parameters: ParameterTuple
    block_losses: tuple[float,...]
    mean_loss: float
    accepted_blocks: int
    acceptance_fraction: float
```

`candidate_parameters` is exactly the Cartesian expansion of the canonical finite grid. Require truth in that expansion, nonempty unique cases, target coordinates subset parameter names, generation seeds/blocks nonempty.

- [ ] **Step 4: Implement interpretation without lexical-best leakage**

```python
@dataclass(frozen=True)
class ParameterIdentificationFinding:
    experiment_hash: str
    true_parameters: ParameterTuple
    candidate_losses: tuple[ParameterCandidateLoss,...]
    accepted_parameters: ParameterAcceptanceSet
    truth_retained: bool
    identifiability: IdentifiabilityReport
    status: IdentificationStatus
    parent_manifest_hashes: tuple[str,...]
    @property
    def content_hash(self) -> str: ...
```

`interpret_parameter_identification()` must require candidate rows cover `experiment.candidate_parameters` exactly once, require nonempty acceptance and retained truth, call `diagnose_identifiability`, require identified target coordinates equal truth, and enforce any preregistered `expected_status`.

- [ ] **Step 5: Implement intervention/protocol declaration identities**

```python
@dataclass(frozen=True)
class InformationInterventionPair:
    name: str
    baseline: Scenario
    intervention: Scenario
    allowed_information_path: tuple[str,...]
    frozen_paths: tuple[tuple[str,...], ...]
    stratum: str
    expected_relationship: str | None = None
    @property
    def content_hash(self) -> str: ...
```

For V1, `frozen_paths` must equal every top-level baseline payload key except `allowed_information_path`; baseline/intervention key sets must match.

```python
@dataclass(frozen=True)
class SyntheticIdentificationProtocol:
    name: str
    version: str
    claim_scope: str
    fixture_hash: str
    adapter_identities: tuple[tuple[str,Mapping[str,object]], ...]
    recovery_experiments: tuple[ParameterRecoveryExperiment,...]
    intervention_pairs: tuple[InformationInterventionPair,...]
    train_partition_hash: str
    selection_partition_hash: str
    final_partition_hash: str
    target_spec_hash: str
    metric_identity: Mapping[str,object]
    brier_loss_identity: Mapping[str,object]
    log_loss_identity: Mapping[str,object]
    brier_training_seeds: tuple[int,...]
    brier_selection_seeds: tuple[int,...]
    final_seeds: tuple[int,...]
    equivalence_tolerance: float = 1e-12
    def identity_payload(self) -> dict[str,object]: ...
    @property
    def content_hash(self) -> str: ...
```

Canonicalize experiments/pairs/sources by name, require exact claim scope and tolerance, unique names, and disjoint train/selection/final case hashes.

- [ ] **Step 6: Implement sibling final-protocol validator**

Exact-equality fields:

```python
("version","dataset_hash","train_partition_hash","selection_partition_hash","final_partition_hash","target_spec_hash","final_target_hash","final_target_manifest_hash","metric_identity","simulation_seeds","baseline_name","candidates")
```

Require Brier loss identity name `categorical_brier`, Log loss identity name `categorical_log`; permit differences only in protocol name, loss identity, thresholds, declared precommitment hash, and derived protocol hash.

- [ ] **Step 7: Run targeted tests and commit**

```bash
python3 -m unittest tests.test_narrative_identification_protocol tests.test_narrative_identification_recovery -v
```

Expected: protocol/core semantics GREEN; adapter/evaluation-dependent tests remain RED.

```bash
git add narrative_dynamics/contracts.py narrative_dynamics/identification.py
git commit -m "feat: add synthetic identification protocol core"
```

Require exact-head monotonic failure reduction.

---

### Task 3: Add the Isolated Narrative Identification Adapter

**Files:**
- Create: `narrative_dynamics/adapters/narrative_prison_identification.py`
- Test: recovery/intervention/model-comparison modules.

**Interfaces:**
- Produces `NarrativePrisonIdentificationSource`, three source factories, and canonical `ModelRun.outcome["initial_policy"]`.

- [ ] **Step 1: Add closed source contracts**

```python
_FAMILIES = ("reactive","intentional","planning")
_ACTIONS = ("scout","escape","submit")
_VERSION = "1.0.0"
_IMPLEMENTATION_REVISION = "narrative-synthetic-identification-v1"

@dataclass(frozen=True)
class NarrativePrisonIdentificationSource:
    family: str
    @property
    def name(self) -> str: return f"generic-narrative-identification-{self.family}"
    @property
    def lifecycle(self) -> str: return "fresh_per_batch"
    def instantiate(self): return _NarrativePrisonIdentificationModel(self.family)
    def manifest_identity(self) -> dict[str,object]: ...

def create_narrative_identification_reactive_source(): ...
def create_narrative_identification_intentional_source(): ...
def create_narrative_identification_planning_source(): ...
```

Identity attests source/model/scenario-builder/family-builder/unified-dispatch implementations.

- [ ] **Step 2: Validate exact scenario/parameter schemas**

Scenario payload keys are exactly those emitted by Task 1 `payload()`. Allowed `(mode, latent_structure)` pairs are:

```python
{
    ("latent_temperature","identical"),
    ("latent_temperature","mirror"),
    ("memory_evidence","belief_driven"),
    ("future_information","value_of_information"),
}
```

Intentional parameters are exactly `beta_goal,beta_action,goal_pressure_scale,instrumentality_scale`. Reactive/Planning parameters are exactly `beta_action`. All are positive finite non-bool floats.

- [ ] **Step 3: Implement latent-temperature semantics with existing intentional math**

Binary actions are `escape,submit`; outer output later zero-fills scout.

For `latent_structure="identical"`, both goals use the same values:

```python
{"escape": +action_gap/2, "submit": -action_gap/2}
```

so the anchor depends on `beta_action` and is exactly invariant to `beta_goal`.

For `latent_structure="mirror"`, goal values are opposite:

```python
freedom = {"escape": +action_gap/2, "submit": -action_gap/2}
safety  = {"escape": -action_gap/2, "submit": +action_gap/2}
```

Goal score gap is `goal_score_gap * goal_pressure_scale * instrumentality_scale`, split `+gap/2,-gap/2`. Therefore `goal_score_gap=0` + mirror structure is the deliberate observational-equivalence fixture, while `0` + identical structure is the action-temperature anchor.

Instantiate existing `GoalModelSpec(beta_goal=...)` and `ChoiceModelSpec(beta_action=...)`; do not reproduce their softmax/marginal formulas.

- [ ] **Step 4: Implement memory-evidence semantics**

Use one hidden `guard.status ∈ {weak,strong}` belief with prior 0.5. `memory_evidence_available=False` leaves the ledger without diagnostic weak/strong evidence. `True` adds the prior weak signal through the existing evidence/belief path using `memory_signal_accuracy`. Reactive's declared current cues exclude that history and therefore remain exact across the pair. Intentional goal instrumentality is opposite on weak/strong, so posterior change alters goal policy/action policy. This mode is binary and zero-fills scout externally.

- [ ] **Step 5: Implement future-information semantics**

All three actions are authored. Reactive and Intentional treat scout myopically with the same immediate cost in both pair members and never branch on `future_information_available`. Planning uses the existing finite planning transition/observation/reward interfaces; scout is uninformative when availability is false and receives `future_signal_accuracy` when true. Reward, transition, persistence, action set, and discount remain exact across the pair.

- [ ] **Step 6: Dispatch only through the unified boundary**

```python
dispatch = run_runtime_decision(case.story, case.domain, case.decision_id, case.ledger, RuntimeDecisionModelSpec(self.family, nested))
policy = {action: 0.0 for action in _ACTIONS}
policy.update({action: float(p) for action,p in dispatch.action_policy.items()})
return ModelRun(events=(), outcome={
    "model_kind": self.family,
    "initial_policy": policy,
    "selected_action": dispatch.selected_action,
    "runtime_dispatch_hash": dispatch.content_hash,
    "runtime_dispatch": dispatch.to_dict(),
    "translated_story_hash": case.story.content_hash,
    "translated_domain_hash": case.domain.content_hash,
    "runtime_ledger_hash": case.ledger.content_hash,
})
```

Outer RNG is accepted but not used to sample a behavioral action.

- [ ] **Step 7: Run targeted tests and commit**

```bash
python3 -m unittest tests.test_narrative_identification_recovery tests.test_narrative_identification_interventions tests.test_narrative_identification_model_comparison -v
```

Expected: adapter/source/schema/raw-policy tests GREEN; orchestration tests may remain RED.

```bash
git add narrative_dynamics/adapters/narrative_prison_identification.py
git commit -m "feat: add narrative synthetic identification adapter"
```

---

### Task 4: Implement Recovery and Direct Observational Equivalence

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_recovery.py`

**Interfaces:**
- Produces `evaluate_parameter_recovery_experiment()`, `ObservationalEquivalenceFinding`, `certify_observational_equivalence()`.

- [ ] **Step 1: Implement multi-case target generation and block scoring by composing `calibrate_grid()`**

```python
def evaluate_parameter_recovery_experiment(*, runner: SimulationRunner, model: ModelSource, experiment: ParameterRecoveryExperiment, extractor: MetricExtractor, loss: MetricLoss) -> ParameterIdentificationFinding: ...
```

Preflight source/extractor/loss identities. For each case, generate target metrics from true parameters with `runner.run_batch(..., seeds=experiment.generation_seeds)` + `aggregate_metrics()`. For each calibration block, call existing `calibrate_grid()` once per case with the exact full grid; aggregate same-candidate per-case losses by `fmean`; mark block acceptance with the preregistered delta. Build one complete `ParameterCandidateLoss` row for every Cartesian candidate and call `interpret_parameter_identification()`.

- [ ] **Step 2: Lock the four recovery experiments**

- Joint beta: cases `train-temp-anchor`, `train-goal-low`, `selection-goal-high`; 4×4 beta grid, scales fixed 1; expected identified; accepted set exactly truth.
- Pressure: same discriminating mirror cases, pressure 3-grid, instrumentality 1, betas truth; expected identified.
- Instrumentality: mirror of pressure experiment; expected identified.
- Scale-confounded: full pressure 3-grid × instrumentality 3-grid, betas truth, `goal_score_gap` mirror cases; expected not identified; accepted set exactly the three product-one tuples.

- [ ] **Step 3: Add complete-policy equivalence finding**

```python
@dataclass(frozen=True)
class ObservationalEquivalenceFinding:
    name: str
    left_model_identity: Mapping[str,object]
    right_model_identity: Mapping[str,object]
    case_hashes: tuple[str,...]
    action_keys: tuple[str,...]
    policies: tuple[tuple[str,tuple[tuple[str,float],...],tuple[tuple[str,float],...]], ...]
    max_abs_policy_delta: float
    tolerance: float
    equivalent: bool
    parent_manifest_hashes: tuple[str,...]
    @property
    def content_hash(self) -> str: ...
```

```python
def certify_observational_equivalence(*, name: str, runner: SimulationRunner, left_model: ModelSource, right_model: ModelSource, left_parameters: Mapping[str,float], right_parameters: Mapping[str,float], cases: tuple[Scenario,...], seeds: tuple[int,...], extractor: MetricExtractor, action_keys: tuple[str,...], tolerance: float=1e-12) -> ObservationalEquivalenceFinding: ...
```

Aggregate policy metrics over seeds and compare every declared action on every case. Scalar loss is not an input.

- [ ] **Step 4: Run recovery suite and commit**

```bash
python3 -m unittest tests.test_narrative_identification_recovery -v
```

Expected: all recovery/equivalence tests GREEN.

```bash
git add narrative_dynamics/identification.py
git commit -m "feat: add synthetic latent recovery and equivalence"
```

---

### Task 5: Implement Information-Only Intervention Certification

**Files:**
- Modify: `narrative_dynamics/identification.py`
- Test: `tests/test_narrative_identification_interventions.py`

**Interfaces:**
- Produces `certify_information_intervention()`, `InformationInterventionFinding`, `evaluate_information_intervention()`.

- [ ] **Step 1: Implement canonical payload diff**

```python
def _scenario_payload_diff(left: Mapping[str,object], right: Mapping[str,object], prefix: tuple[str,...]=()) -> tuple[tuple[tuple[str,...],object,object], ...]: ...
```

Recurse mappings by sorted key and return sorted leaf diffs. No coercion.

- [ ] **Step 2: Certify exactly one allowed leaf**

```python
def certify_information_intervention(pair: InformationInterventionPair) -> tuple[tuple[str,...],object,object]:
    diffs = _scenario_payload_diff(pair.baseline.payload, pair.intervention.payload)
    if len(diffs) != 1:
        raise InterventionCertificationError("information intervention must change exactly one payload leaf")
    path,before,after = diffs[0]
    if path != pair.allowed_information_path:
        raise InterventionCertificationError("information intervention changed an undeclared field")
    return path,before,after
```

`InformationInterventionPair.__post_init__` already requires `frozen_paths` to be the complete complement, so exact one-diff certification proves all frozen reward/dynamics/action fields unchanged.

- [ ] **Step 3: Implement family policy-delta finding/evaluator**

```python
@dataclass(frozen=True)
class InformationInterventionFinding:
    name: str
    pair_hash: str
    certified_diff: tuple[tuple[str,...],object,object]
    family_policies: tuple[tuple[str,tuple[tuple[str,float],...],tuple[tuple[str,float],...]], ...]
    within_family_max_deltas: tuple[tuple[str,float], ...]
    cross_family_contrasts: tuple[tuple[str,str,float], ...]
    tolerance: float
    discriminating: bool
    parent_manifest_hashes: tuple[str,...]
    @property
    def content_hash(self) -> str: ...
```

```python
def evaluate_information_intervention(*, pair: InformationInterventionPair, runner: SimulationRunner, families: tuple[tuple[str,ModelSource,Mapping[str,float]], ...], seeds: tuple[int,...], extractor: MetricExtractor, action_keys: tuple[str,...], tolerance: float=1e-12) -> InformationInterventionFinding: ...
```

Certify before running. Canonicalize families by name. For `intentional_vs_reactive`, require Reactive delta `<=1e-12` and Intentional `>1e-12`; for `planning_vs_intentional`, require Intentional `<=1e-12` and Planning `>1e-12`.

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
- Test: `tests/test_narrative_identification_model_comparison.py`

**Interfaces:**
- Consumes existing target construction, Brier training fit, selection-validation, frozen candidates, preregistered final comparison.
- Produces `StratumModelScore`, `ScoredModelComparisonFinding`, `score_model_comparison_by_stratum()`.

- [ ] **Step 1: Build Brier-only train/selection**

```python
spec = CategoricalTargetSpec(name="narrative-identification-initial-action", version="1", categories=("scout","escape","submit"), metric_prefix="initial")
group = CategoricalMetricGroup("initial-action", ("initial.scout","initial.escape","initial.submit"))
brier = CategoricalBrierLoss((group,))
log = CategoricalLogLoss((group,))
```

For each source call existing `fit_training_target_grid(... loss=brier)` with its exact family grid, build `ParameterAcceptanceSet` from training candidates, run `select_on_validation_suite(... loss=brier)`, then `FrozenModelCandidate.from_selection`. No Log call occurs here.

- [ ] **Step 2: Create sibling final protocols over the exact same frozen candidate tuple**

```python
brier_protocol = PreregisteredEvaluationProtocol.create(name="narrative-synthetic-identification-brier-v1", version="1", dataset=dataset, target_spec=spec, extractor=prison_initial_action_metrics, loss=brier, simulation_seeds=(401,402), baseline_name=intentional_name, candidates=frozen_candidates, thresholds=AdequacyThresholds(2.0,2.0))
log_protocol = PreregisteredEvaluationProtocol.create(name="narrative-synthetic-identification-log-v1", version="1", dataset=dataset, target_spec=spec, extractor=prison_initial_action_metrics, loss=log, simulation_seeds=(401,402), baseline_name=intentional_name, candidates=frozen_candidates, thresholds=AdequacyThresholds(20.0,20.0))
validate_sibling_final_protocols(brier_protocol, log_protocol)
```

- [ ] **Step 3: Execute both final comparisons**

Use one canonical `ComparisonModel` tuple and one `final_targets` object. Call `compare_models_on_final_partition` once under Brier and once under Log. No candidate fit/selection occurs between these calls.

- [ ] **Step 4: Aggregate existing per-case losses by final fixture stratum**

```python
@dataclass(frozen=True)
class StratumModelScore:
    stratum: str
    model_name: str
    mean_loss: float
    worst_loss: float
    case_names: tuple[str,...]

@dataclass(frozen=True)
class ScoredModelComparisonFinding:
    protocol_hash: str
    loss_identity: Mapping[str,object]
    candidate_hashes: tuple[tuple[str,str], ...]
    final_partition_hash: str
    final_target_hash: str
    final_seeds: tuple[int,...]
    global_scores: tuple[tuple[str,float,float], ...]
    stratum_scores: tuple[StratumModelScore,...]
    pairwise_mean_deltas: tuple[tuple[str,str,float], ...]
    parent_comparison_manifest_hash: str
    @property
    def content_hash(self) -> str: ...
```

```python
def score_model_comparison_by_stratum(*, dataset: ObservationDataset, protocol: PreregisteredEvaluationProtocol, report: ModelComparisonReport) -> ScoredModelComparisonFinding: ...
```

Require final strata exactly `observational_equivalence`, `memory_evidence`, `future_information`. Aggregate the already-computed `FinalTestReport` case losses; do not rerun models.

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
- Test: `tests/test_narrative_identification_reporting.py`

**Interfaces:**
- Produces `SyntheticIdentificationReport`, `build_synthetic_identification_report()`, deterministic scoped conclusion rendering.

- [ ] **Step 1: Add final report type**

```python
@dataclass(frozen=True)
class SyntheticIdentificationReport:
    protocol_hash: str
    claim_scope: str
    parameter_findings: tuple[ParameterIdentificationFinding,...]
    equivalence_findings: tuple[ObservationalEquivalenceFinding,...]
    intervention_findings: tuple[InformationInterventionFinding,...]
    brier_comparison: ScoredModelComparisonFinding
    log_comparison: ScoredModelComparisonFinding
    conclusions: tuple[str,...]
    manifest: ExperimentManifest
    def __post_init__(self) -> None: ...
```

Validate exact claim scope, child finding hash uniqueness/order, Brier/Log frozen identity equality, manifest stage/inputs, and exact complete parent hash union.

- [ ] **Step 2: Require all eight roadmap evidence categories in the builder**

```python
def build_synthetic_identification_report(*, protocol: SyntheticIdentificationProtocol, parameter_findings: tuple[ParameterIdentificationFinding,...], equivalence_findings: tuple[ObservationalEquivalenceFinding,...], intervention_findings: tuple[InformationInterventionFinding,...], brier_comparison: ScoredModelComparisonFinding, log_comparison: ScoredModelComparisonFinding) -> SyntheticIdentificationReport: ...
```

Map experiment/pair hashes back through `protocol` and require:

```text
joint beta recovery = identified
pressure recovery = identified
instrumentality recovery = identified
scale-confounded recovery = not identified
observational equivalence finding = equivalent
memory Intentional-vs-Reactive finding = discriminating
future Intentional-vs-Planning finding = discriminating
Brier + Log findings share exact frozen candidates/final identities
```

These jointly evidence all eight #27 checklist statements; missing evidence raises `IdentificationReportError`.

- [ ] **Step 3: Construct the manifest from child evidence**

Use `ExperimentStage.SYNTHETIC_IDENTIFICATION`; manifest inputs contain protocol hash and every child finding hash. `parent_hashes` is exactly the sorted union of all child calibration/run/comparison manifest hashes. `SyntheticIdentificationReport.__post_init__` recomputes that expected union and rejects additions/removals.

- [ ] **Step 4: Render only scoped language**

```python
def parameter_conclusion(name: str, status: IdentificationStatus) -> str:
    if status is IdentificationStatus.IDENTIFIED_UNDER_PROTOCOL:
        return f"{name} is identified under this frozen synthetic protocol."
    return f"{name} is not identified under this frozen synthetic protocol."
```

Family conclusions use only `equivalent`, `separated`, or `discriminating` language. Never emit `structurally identified`, `empirically identified`, or claims about real human cognition.

- [ ] **Step 5: Run/commit**

```bash
python3 -m unittest tests.test_narrative_identification_reporting -v
git add narrative_dynamics/identification.py
git commit -m "feat: attest narrative synthetic identification report"
```

---

### Task 8: Exact-Head Verification, Scope Review, and Integration Evidence

**Files:** No new production files.

**Interfaces:** Produces authoritative feature/PR/post-merge evidence and roadmap closure only after all gates.

- [ ] **Step 1: Run all new tests together**

```bash
python3 -m unittest \
  tests.test_narrative_identification_protocol \
  tests.test_narrative_identification_recovery \
  tests.test_narrative_identification_interventions \
  tests.test_narrative_identification_model_comparison \
  tests.test_narrative_identification_reporting -v
```

Expected: GREEN.

- [ ] **Step 2: Run exact full Python gate**

```bash
python3 -m unittest discover -s tests -v
```

Expected: `OK`; record exact new total and elapsed time. Do not reuse 659 as the new total.

- [ ] **Step 3: Require exact-head `proof.yml` completed/success**

Record exact feature SHA/run/job, checkout SHA, Lean conformance/full-build/theorem gates, Python `Ran N tests ... / OK`, story/testimony GREEN.

- [ ] **Step 4: Perform base→head scope review**

Allowed paths are exactly:

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

- [ ] **Step 5: Review the actual scientific findings**

Require: exact joint-beta truth acceptance; identified pressure/instrumentality single-coordinate experiments; exact three-point scale-confounded equivalence set with truth retained; direct `1e-12` policy equivalence; certified information-only memory/future pairs; required family separation; identical Brier/Log frozen candidates/targets/seeds; no Log selection; no structural/empirical/human-cognition overclaim.

- [ ] **Step 6: Invoke verification-before-completion, then finishing-development-branch**

Do not claim DONE from local results. Use fresh exact-head evidence and present the integration choice required by the finishing skill.

- [ ] **Step 7: If PR integration is chosen, require PR synthetic-merge GREEN**

PR title: `feat: add narrative synthetic identification v1`.

PR body records base, spec/plan, authoritative RED, exact feature GREEN, accepted-set/non-identifiability boundary, intervention results, Brier-selected/Log-rescored evidence, changed-file scope, and unchanged core modules. Require `pull_request` proof completed/success for current exact head/base before merge.

- [ ] **Step 8: Guard merge and require post-merge exact-head GREEN**

Merge with `expected_head_sha=<approved exact feature head>`. Require push-triggered `proof.yml` completed/success on the exact new `proof/narrative-dynamics-v0` merge commit and capture `Ran N tests ... / OK` plus all Lean/story/testimony gates.

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
