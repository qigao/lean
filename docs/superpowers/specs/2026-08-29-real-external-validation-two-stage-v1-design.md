# Real External Validation Study V1 — Feher da Silva & Hare Two-Stage Task

Date: 2026-08-29

Status: written for review

Roadmap: #39 — Real External Validation Study and post-P3 frontiers

Study issue: #40 — Real External Validation Study V1 — Feher da Silva & Hare two-stage task

Integrated base: `proof/narrative-dynamics-v0@e529167eddc07efd6b2bbd90039b89ed1cbeb03d`

Design branch: `work/real-external-validation-two-stage-v1-design`

Depends on: `docs/superpowers/specs/2026-08-28-narrative-empirical-external-validation-v1-design.md`

## 1. Purpose

This study is the first real external-observational application of the completed P3 external-validation machinery. It evaluates the existing Reactive, Intentional, and Planning narrative cognitive model families on archived human choices from Feher da Silva & Hare's two-stage-task repository.

The scientific question is deliberately predictive:

> Given a frozen external behavioral dataset, a deterministic participant-disjoint split, globally frozen family-level parameters, and an externally witnessed preregistered protocol, do the Reactive, Intentional, and Planning families predict unseen participants' first-stage choices better than an uninformative 50/50 predictor, and are any model pairs predictively separated under both Brier and Log scoring?

This study does not claim that predictive fit identifies real latent cognition, validates a causal mechanism, or proves that a fitted model family is the true cognitive process used by participants.

The fixed claim scope remains:

```text
external_observational_predictive_only
```

## 2. Upstream source and evidence scope

The frozen upstream source is:

```text
repository: carolfs/muddled_models
revision: 4567763780a2c596fd6510af720ec468a8214a8f
```

Study V1 uses both task variants:

- `magic_carpet`
- `spaceship`

The two variants are reporting strata within one study, not separate studies.

### 2.1 Included scientific evidence

Magic Carpet scientific evidence is limited to:

```text
results/magic_carpet/choices/*_game.csv
```

Spaceship scientific evidence is limited to participant main-task CSV files:

```text
results/spaceship/choices/*.csv
```

excluding:

```text
*_practice.csv
```

Matching Magic Carpet `*_config.txt` and Spaceship `*_info.txt` files may be retained only as provenance / deterministic transform metadata, including counterbalancing or semantic-normalization information where needed. They are not predictive targets.

The following are excluded from Study V1 model comparison:

- Magic Carpet tutorial files;
- Spaceship practice files;
- questionnaires;
- published fitted parameters;
- published model labels;
- paper-level mechanism interpretations.

Published results are background literature only and must not affect participant exclusions, split construction, parameter grids, thresholds, FINAL_TEST decisions, or expected model winner.

### 2.2 Source inventory

At the frozen source revision, structural main-task eligibility yields:

```text
Magic Carpet: 24 participants with main-game CSV files
Spaceship:    21 participants with main-task CSV files
Total:        45 structurally eligible participants
```

Both task trees also contain source identities with config/info metadata but no main-task CSV. Those identities are not eligible participants under this study.

### 2.3 Licensing boundary

The upstream repository contains a root GNU GPL v3 license file. Study V1 does not vendor raw human CSV rows into `qigao/lean`. The source manifest records the upstream repository, revision, and license reference. This design does not make a broader legal conclusion about separate data-licensing questions.

## 3. Non-vendored source snapshot contract

Study V1 uses a non-vendored, manifest-pinned external source.

The repository stores only a canonical study source manifest that binds:

- repository identity;
- exact upstream commit SHA;
- every included scientific-evidence relative path;
- every included file's Git blob/content identity;
- task variant;
- source participant identity;
- provenance-only metadata file identities;
- license/source reference;
- source-manifest schema/version.

Raw human CSV rows are not committed to `qigao/lean`.

Study ingestion accepts only an explicitly supplied local source snapshot/checkout. Before parsing any trial rows, the source verifier must fail closed unless repository revision, required paths, blob/content identities, and manifest identity match exactly.

Ordinary CI must not require network access to the upstream repository.

The canonical `source_snapshot_hash` is the stable content hash of the verified source-manifest identity, including the exact upstream revision and included file identities.

## 4. Participant and trial eligibility

Eligibility is outcome-independent.

### 4.1 Participant eligibility

A source identity enters the study only if it has a frozen, included main-task file that passes source identity and schema verification and retains at least one scorable trial after the frozen trial-quality rule.

No participant may be excluded because of:

- reward rate;
- choice pattern;
- response-time distribution;
- apparent model-based/model-free behavior;
- paper model label;
- fitted parameter value;
- model fit;
- questionnaire response;
- expected or observed FINAL_TEST performance.

No minimum-completion percentage threshold is introduced in V1.

### 4.2 Trial eligibility

The source-defined `slow` marker is the only behavioral trial-quality exclusion in V1:

```text
slow == 1  -> exclude before target/history construction
slow == 0  -> retain if the row is otherwise structurally valid
```

A slow trial is not a prediction target and does not enter the retained history used to predict later choices.

### 4.3 Malformed included rows

Malformed rows are never silently dropped. An included main-task file causes the complete source transformation to fail closed if a required row contains, for example:

- missing required fields;
- invalid choice/state/reward codes;
- non-finite required numeric values;
- duplicate or non-monotone trial identity where monotonicity is required by the task schema;
- inconsistent task-specific canonicalization information.

Exclusion counts and reasons are part of the transform report identity.

## 5. Deterministic participant split

The split unit is:

```text
(task_variant, source_participant_id)
```

A numerically identical participant id in different task variants is a different source identity.

The split occurs after structural eligibility and frozen slow-trial filtering, but before any model fitting, model scoring, loss inspection, behavioral summary inspection, or FINAL_TEST analysis.

Participants are stratified by task variant and deterministically ordered with a versioned study namespace plus stable content hash. Python RNG state, filesystem ordering, and manual assignment are forbidden.

The frozen ratio is:

```text
60% TRAIN
20% SELECTION_VALIDATION
20% FINAL_TEST
```

Within each task, integer counts use deterministic largest-remainder apportionment. For the frozen 24/21 source inventory this gives:

```text
Magic Carpet: 14 TRAIN / 5 SELECTION_VALIDATION / 5 FINAL_TEST
Spaceship:    13 TRAIN / 4 SELECTION_VALIDATION / 4 FINAL_TEST
Total:        27 TRAIN / 9 SELECTION_VALIDATION / 9 FINAL_TEST
```

All retained trials of a participant belong to that participant's one partition. No participant contributes history or targets to more than one partition.

### 5.1 Participant-assignment identity versus generic P3 assignment identity

The study-specific split object owns a `participant_assignment_hash` over the complete canonical mapping from all 45 eligible participant identities to partition roles.

Existing P3 `ExternalEvidenceDeclaration.partition_assignment_hash` remains the generic downstream assignment hash over observation-record identities. Study V1 does not change that generic P3 contract. Instead:

- `participant_assignment_hash` is bound into the study transform identity and source revision metadata;
- deterministic trial-record construction expands that participant assignment into the existing generic record-level partition assignment;
- changing any participant assignment must therefore change both the study transform lineage and the downstream dataset / P3 assignment identities.

## 6. Measurement target and causal information firewall

The primary endpoint is the per-trial first-stage human choice probability.

Each retained human trial becomes one record-oriented `ObservationRecord` with a binary categorical target:

```text
first_stage.action_0
first_stage.action_1
```

The observed human first-stage choice is encoded as a one-hot categorical target.

### 6.1 Task-specific choice normalization

Magic Carpet first-stage choice uses the source task's provided choice semantics and frozen counterbalancing/config metadata where necessary to map the raw row to canonical `action_0` / `action_1`.

Spaceship first-stage choice is normalized to the source loader's relative option semantics. Current-trial realized `common` / `final_state` may be used only inside target decoding when required to reconstruct the observed canonical option. Those fields must not enter the model-visible input for that same trial.

### 6.2 Model-visible scenario

The current trial's `Scenario` contains only information available before the participant makes that trial's first-stage choice:

- task variant;
- current first-stage stimulus/configuration in canonical form;
- the participant's complete prior retained observable history.

A prior retained trial may contribute its already-observed:

- first-stage canonical choice;
- realized transition/final state;
- second-stage choice;
- reward.

The following same-trial fields are forbidden from the current prediction input because they occur after the first-stage choice or are latent environment state:

- current `common` realization;
- current `final_state`;
- current second-stage choice;
- current reward;
- current latent/drifting true reward-probability columns such as `reward.*` or `rwrd_prob*`.

True drifting reward probabilities remain provenance/audit-only and are never model-visible.

### 6.3 Causal construction invariant

For any retained trial, `scenario.content_hash` must be computable before reading that trial's observed first-stage target and post-choice outcome fields, except for task-specific target decoding performed strictly outside the predictor input.

Metamorphic tests must prove that mutating same-trial post-choice fields leaves the current model-visible scenario and prediction unchanged.

## 7. ObservationDataset construction

The study transform plus participant assignment produces the existing record-oriented `ObservationDataset`; no ObservationDataset schema change is required.

Each record id canonically encodes:

```text
task_variant / participant_source_identity / source_trial_identity
```

Each record stores:

- the causal pre-choice `Scenario`;
- binary one-hot counts for `action_0` / `action_1`;
- task variant metadata;
- participant source identity;
- source trial identity;
- causal-history hash;
- study transform lineage.

The dataset declares positive external origin using the existing P3 source/provenance markers:

```text
source.kind = "external_observational"
provenance.external_observational = true
```

No synthetic fixture may be relabeled as external evidence.

## 8. Two-stage task adapter and family semantics

Study V1 adds a task-specific two-stage narrative adapter. It does not alter generic Reactive, Intentional, or Planning runtime semantics.

All three families execute through the existing unified runtime decision dispatch.

The adapter owns only task semantics:

- canonical two-stage domain/story construction;
- canonical first-stage and second-stage action/state identities;
- history-to-evidence replay;
- task-specific family hooks/builders;
- task-known transition structure;
- first-stage policy extraction.

The adapter must not import observational comparison, external-validation orchestration, release, or witness logic.

### 8.1 Canonical task identities

The adapter normalizes both task variants to canonical semantic identities equivalent to:

```text
first-stage actions: action_0, action_1
second-stage states: state_0, state_1
second-stage actions within each state: action_0, action_1
```

Visual symbols/counterbalancing are transform metadata, not new model families.

### 8.2 Reactive family

Reactive is structurally latest-cue-only.

It may use only the immediately previous retained trial's canonical first-stage choice and reward-sensitive cue. It must not read earlier history or construct a transition model.

The fixed V1 rule is:

- no previous retained trial -> neutral equal action scores;
- previous retained trial rewarded -> favor repeating its first-stage canonical action;
- previous retained trial unrewarded -> favor switching away from its first-stage canonical action.

The score magnitude is fixed by the adapter; only `beta` is calibrated.

Reactive does not use previous `common/rare` transition identity as a planning variable.

### 8.3 Intentional family

Intentional reads the complete prior retained history and uses the existing belief -> goal -> choice runtime path.

Its task-specific belief summarizes first-stage option instrumentality directly from realized rewards conditional on canonical first-stage action. It does not use the two-stage common/rare transition structure for lookahead.

For memory decay `d`, the immediately previous retained trial has weight `d^0`, the next earlier retained trial `d^1`, and so on.

For each first-stage action, its smoothed reward belief is:

```text
estimated_p = (1 + weighted_rewards) / (2 + weighted_observations)
```

which is Beta(1,1) smoothing and is well-defined for empty history.

The goal/choice construction must remain a genuine use of the existing Intentional runtime. V1 binds `beta_goal = beta_action = beta` to avoid an extra Intentional temperature parameter.

### 8.4 Planning family

Planning reads the complete prior retained history and uses the existing planning runtime with explicit hidden-state, transition, reward, and value hooks.

It estimates second-stage state/action reward probabilities from prior retained outcomes using the same Beta(1,1) smoothing and the same memory-decay definition as Intentional.

The experiment's task-known transition structure is fixed in the adapter rather than fitted. The canonical common transition probability is 0.7 and rare transition probability is 0.3, with task/counterbalancing metadata determining the canonical state mapping.

Planning evaluates first-stage options by lookahead through that transition structure to the currently estimated second-stage values. V1 uses fixed planning discount `1.0`; discount is not a fitted parameter.

### 8.5 Family-boundary claim

These task adapters are operational comparison models, not claims that the labels correspond to participants' true internal algorithms.

## 9. Global parameterization and calibration

No participant-specific or task-specific parameter fitting is allowed.

Every family freezes one global parameter tuple shared across Magic Carpet and Spaceship.

Frozen candidate grids are:

```text
Reactive:
  beta in {0.5, 1.0, 2.0, 4.0}

Intentional:
  beta in {0.5, 1.0, 2.0, 4.0}
  memory_decay in {0.5, 0.75, 0.9, 1.0}
  beta_goal = beta_action = beta

Planning:
  beta in {0.5, 1.0, 2.0, 4.0}
  memory_decay in {0.5, 0.75, 0.9, 1.0}
  transition probabilities fixed by task semantics
  discount = 1.0
```

### 9.1 Training and selection score

Study V1 uses categorical Brier loss as the single frozen parameter-fitting and selection score. Log loss is not used to choose parameters; it remains an independent confirmatory proper-scoring view at locked FINAL_TEST.

This keeps one deterministic selection objective and prevents score-dependent candidate freezing.

TRAIN uses only TRAIN participant records. SELECTION_VALIDATION chooses one parameter tuple per family from the TRAIN-fitted finite grid. FINAL_TEST never adapts or refits parameters.

The existing finite-grid training and selection infrastructure remains authoritative.

### 9.2 Aggregate weighting

Primary training, selection, and final mean losses weight retained trial records equally. Task strata are reported separately so task imbalance remains visible.

### 9.3 Simulation seed plans

The frozen seed plans are:

```text
TRAIN:                 (101, 102)
SELECTION_VALIDATION:  (201, 202)
FINAL_TEST:            (301, 302)
```

The current family policies are expected to be seed-invariant in their predicted distributions under this adapter, but seeds remain part of execution lineage and parity checks.

## 10. Primary scoring and probability safety boundary

Study V1 uses both existing proper categorical scores:

```text
categorical_brier
categorical_log
```

The common first-stage metric extractor returns exactly:

```text
first_stage.action_0
first_stage.action_1
```

### 10.1 Probability floor

Before either score consumes a predicted policy, a versioned common extractor applies the fixed numerical safety rule:

```text
p_i := max(raw_p_i, 1e-12)
renormalize the binary simplex
```

The same normalized policy is consumed by both Brier and Log. Raw runtime policy remains available in the attested trace for audit.

The `1e-12` floor is not fitted, tuned, or participant-specific. Its implementation identity is bound into both sibling protocols.

## 11. Adequacy and predictive separation

For a binary uninformative 50/50 predictor:

```text
Brier reference = 0.5
Log reference   = ln(2)
```

Study V1 defines predictive adequacy as strictly better than those references on mean FINAL_TEST loss.

Because the existing generic comparator uses `<= max_mean_loss`, Study V1 encodes strictness without modifying generic P3 by freezing the immediately lower representable IEEE-754 values:

```text
Brier max_mean_loss = 0.49999999999999994
Log max_mean_loss   = 0.6931471805599452
```

Worst-loss thresholds are non-substantive safety bounds:

```text
Brier max_worst_loss = 2.0
Log max_worst_loss   = 27.63102111592955
```

The Log worst bound is derived from the fixed `1e-12` floor after binary renormalization.

A model is aggregate `predictive_adequacy_met` only if both Brier and Log child evaluations meet their frozen adequacy criteria.

### 11.1 Pairwise separation

For every model pair, `predictively_separated_under_protocol` requires:

1. Brier and Log prefer the same model by mean loss;
2. absolute Brier mean-loss delta >= `0.005`;
3. absolute Log mean-loss delta >= `0.006931471805599453`.

These margins are exactly 1% of the two uniform-reference loss scales.

No significance test or mechanism-identification interpretation is attached to this status.

### 11.2 Reporting strata

Magic Carpet and Spaceship are preregistered task strata. Each model reports Brier and Log mean/worst loss separately by task.

Stratum scores are descriptive reporting outputs and do not independently determine aggregate adequacy or pairwise separation.

## 12. External evidence declaration

Study V1 reuses the existing `ExternalEvidenceDeclaration`.

The declaration binds at least:

- final `ObservationDataset.content_hash`;
- frozen source snapshot hash;
- upstream repository/revision;
- study transform identity;
- record namespace;
- generic record-level P3 partition assignment hash;
- fixed external evidence origin;
- fixed predictive-only claim scope.

The study transform identity additionally binds the participant-assignment hash, source-manifest hash, exclusion policy, choice-normalization version, causal-history construction identity, information-firewall version, and implementation identity.

## 13. Dual P3 protocols and preregistration

After TRAIN and SELECTION_VALIDATION freeze the three global candidate parameter tuples, Study V1 builds two sibling `PreregisteredEvaluationProtocol` values:

- Brier sibling;
- Log sibling.

They must match on every non-score-specific scientific input required by P3, including dataset, partitions, targets, extractor, seeds, baseline, frozen candidates, selected parameters, and selection lineage.

V1 uses Planning as the existing comparator's bookkeeping `baseline_name`. Primary adequacy and pairwise-separation conclusions do not depend on baseline deltas; all three model pairs are evaluated.

The top-level `ExternalValidationPreregistration` binds both sibling protocol hashes, both frozen threshold sets, two task strata, the pairwise-separation rule, and the external evidence declaration.

V1 sets:

```text
constraint_plans = ()
```

No external parameter-constraint finding is produced in Study V1.

## 14. OSF external witness

The complete preregistration bundle is submitted to a public immutable OSF Registration before any FINAL_TEST model execution.

The bundle freezes at least:

- upstream source repository and exact revision;
- source manifest / blob identities;
- transform identity and exclusion policy;
- eligible participant inventory;
- deterministic participant assignment and hash;
- dataset and target identities;
- information firewall;
- three frozen model identities and parameter tuples;
- Brier and Log protocol hashes;
- adequacy thresholds;
- probability floor;
- task strata;
- separation rule;
- `ExternalValidationPreregistration.content_hash`;
- Brier and Log `ProtocolRelease.content_hash` values;
- exact `qigao/lean` repository revision;
- permitted and forbidden scientific claim language.

One OSF registration may witness both sibling releases, but Study V1 creates two existing `WitnessReceipt` values because each receipt has one `subject_hash`:

```text
Brier release -> WitnessReceipt -> same OSF registration
Log release   -> WitnessReceipt -> same OSF registration
```

The two receipts must bind the same OSF registration/bundle digest and their respective release hashes.

An OSF-specific verifier adapter uses the existing witness interface; Study V1 does not change `WitnessReceipt` schema.

## 15. Dual-release preflight

Before any FINAL_TEST model is instantiated, both sibling releases must be externally verified and pass existing `preflight_external_releases()`.

Any mismatch in release, protocol, dataset, target, candidate, evidence, preregistration, verifier, score role, or sibling identity fails closed before FINAL execution.

Preflight failure does not consume the FINAL protocol because no model prediction has executed.

## 16. Single-pass FINAL prediction artifact

Study V1 adds an additive generic external-prediction seam so Brier and Log do not independently rerun models.

The locked execution order is:

```text
verified dual-release preflight
        -> predict FINAL once
        -> seal prediction artifact
        -> score same artifact as Brier
        -> score same artifact as Log
        -> derive adequacy / separation / strata
        -> derive secondary diagnostic
        -> build final report
```

### 16.1 ExternalFinalPredictionArtifact

The attested prediction artifact binds at least:

- external preregistration hash;
- dual-release preflight hash;
- repository revision;
- dataset hash;
- final partition / final target identities;
- metric extractor identity;
- seed plan;
- complete frozen candidate identities and parameters;
- per-model / per-case / per-seed metric vectors;
- run-manifest hashes;
- artifact content hash and parent lineage.

The artifact contains prediction evidence, not raw external CSV rows.

### 16.2 Precomputed scoring seam

A generic additive scoring path accepts only an attested prediction artifact that exactly matches the supplied Brier or Log protocol and release lineage.

It constructs the same child comparison semantics as the existing final comparator but performs no model execution.

The existing final-comparison API remains available for other uses. Study V1 uses only the precomputed scoring seam after the prediction artifact is sealed.

Brier scoring, Log scoring, and secondary diagnostics must not call the model runner again.

This amendment changes execution plumbing only; it does not change P3 statuses, release schema, claim scope, model semantics, or P2 identification.

## 17. Secondary stay/switch diagnostic

Study V1 includes one preregistered secondary descriptive diagnostic.

For consecutive retained trials within each FINAL_TEST participant, construct previous-trial groups by:

```text
previous reward x previous common/rare transition
```

For each task stratum and model, report:

- observed human stay rate;
- mean predicted stay probability derived from the already sealed current-trial first-stage action policy.

The secondary diagnostic:

- uses no new model executions;
- has no adequacy threshold;
- has no separation status;
- does not affect parameter selection;
- does not affect primary conclusions;
- cannot be interpreted as confirmation of model-based cognition.

Filtering follows the retained-trial sequence, so an excluded slow trial is absent before adjacent retained pairs are constructed.

## 18. FINAL consumption and retry semantics

A study-specific append-only attempt record tracks every FINAL execution attempt.

Conceptually it contains:

```text
attempt_id
preregistration_hash
preflight_hash
repository_revision
dataset_hash
final_target_hash
frozen_candidate_hashes
brier_release_hash
log_release_hash
started_at
status
failure_class
completed_run_manifest_hashes
result_hash
```

The first actual FINAL model execution marks the protocol consumed for that attempt.

Allowed attempt statuses distinguish:

```text
started
completed
infrastructure_failed
revision_required
```

### 18.1 Exact infrastructure retry

A retry is allowed only after a documented infrastructure failure such as process crash, machine loss, CI executor cancellation, disk/resource failure, or equivalent non-scientific execution failure.

The retry must preserve exactly:

- repository revision;
- source / transform / split / dataset / target identities;
- frozen candidate identities and parameters;
- metric extractor;
- probability floor;
- seed plan;
- Brier/Log protocols and releases;
- OSF witness receipts;
- preflight identity.

The retry is a new append-only attempt, never an overwrite of the failed attempt.

### 18.2 Revision required

Any scientific data/code/protocol defect requires a new study revision and new external preregistration. Examples include:

- transform bug;
- same-trial information leakage;
- incorrect task choice normalization;
- adapter semantic bug;
- incorrect parameter grid;
- incorrect probability floor;
- incorrect threshold;
- incorrect participant split;
- model implementation change.

Even a one-line scientific fix cannot reuse the old OSF registration/release lineage.

A disappointing or surprising FINAL result is not a defect and is never a reason to rerun under changed scientific identities.

## 19. Component boundaries

Expected implementation boundaries are:

```text
narrative_dynamics/studies/two_stage_source.py
  source manifest + local snapshot verification

narrative_dynamics/studies/two_stage_transform.py
  schema normalization, exclusions, history construction,
  participant assignment, ObservationDataset construction

narrative_dynamics/adapters/narrative_two_stage.py
  task domain/story/evidence + Reactive/Intentional/Planning builders

narrative_dynamics/adapters/two_stage_metrics.py
  first-stage policy extractor + probability floor

narrative_dynamics/studies/feher_hare_two_stage_v1.py
  thin study orchestration through TRAIN/SELECTION/release/final report

narrative_dynamics/external_prediction.py
  generic attested single-pass FINAL prediction artifact
  + precomputed scoring seam
```

The exact public API names may be refined in the implementation plan, but the responsibility boundaries above are normative: source/transform logic must not leak into generic P3, and comparison/release logic must not leak into the task adapter.

No Lean source changes are required.

No Docker acceptance is required.

## 20. Canonical identity chain

The study must provide a fully auditable lineage equivalent to:

```text
upstream revision
-> source manifest hash
-> source snapshot hash
-> transform identity
-> participant assignment hash
-> ObservationDataset.content_hash
-> generic record partition-assignment hash
-> target-spec / target-construction hashes
-> TRAIN manifests
-> SELECTION manifests
-> FrozenModelSpec hashes
-> Brier protocol hash
-> Log protocol hash
-> ExternalValidationPreregistration hash
-> Brier release hash
-> Log release hash
-> OSF receipt hashes
-> ExternalReleasePreflight hash
-> ExternalFinalPredictionArtifact hash
-> Brier child comparison
-> Log child comparison
-> ExternalFinalEvaluation
-> secondary diagnostic
-> final ExternalValidationReport
```

No manual escape hatch may preserve an old scientific identity after a source, transform, split, adapter, parameter, extractor, floor, threshold, release, or witness change.

## 21. Testing and verification strategy

### 21.1 Ordinary CI uses synthetic evidence only

Ordinary CI must not download, vendor, or read the real human source snapshot.

It uses synthetic mini-files and a complete synthetic mini-study to verify contracts and execution plumbing.

Required unit / mutation coverage includes:

- source revision/path/blob mismatch rejection;
- Magic Carpet and Spaceship schema normalization;
- exact slow-trial exclusion;
- malformed included row fail-closed behavior;
- participant split determinism and disjointness;
- participant assignment hash mutation;
- dataset / target lineage mutation;
- probability floor and renormalization;
- Brier/Log uniform-reference values and strict thresholds;
- sibling-release identity drift rejection;
- zero FINAL executions on preflight failure;
- one sealed prediction artifact for both scores;
- no runner calls during Brier/Log precomputed scoring;
- no runner calls during secondary diagnostic;
- append-only retry lineage;
- scientific identity drift rejection during retry.

### 21.2 Information-leakage metamorphic tests

For a fixed prior retained history and current pre-choice configuration, mutating current-trial post-choice fields must not change the model-visible Scenario or prediction:

```text
current common
current final_state
current choice2
current reward
current latent reward-probability fields
```

Mutating allowed prior-history evidence must change the input identity where appropriate.

### 21.3 Family-boundary tests

Reactive tests prove that earlier-than-latest history cannot affect its prediction while the latest retained reward cue can.

Intentional tests prove that complete prior reward history can affect belief/choice but Planning transition/reward hooks are not imported or used.

Planning tests prove that prior transition and second-stage reward history can affect explicit planning values and that task transition/reward hook identities are bound.

All three must execute through unified runtime dispatch.

### 21.4 Full synthetic external-study integration

A synthetic two-stage mini-study must exercise:

```text
synthetic source snapshot
-> verification
-> transform
-> deterministic participant split
-> ObservationDataset
-> TRAIN
-> SELECTION
-> freeze three families
-> Brier + Log sibling releases
-> fake immutable witness receipts
-> dual-release preflight
-> single FINAL prediction artifact
-> Brier + Log scoring
-> adequacy / separation / strata
-> secondary diagnostic
-> final report integrity
```

Synthetic fixtures must retain explicit synthetic provenance and must not be accepted as real external evidence.

Tests may use a synthetic generator to obtain a known mathematical winner, but real Study V1 correctness must never depend on which family wins.

## 22. Real-source conformance lane

Real Feher-Hare data enter only through an explicit local conformance execution against the frozen upstream revision.

Before OSF registration, the real-source lane may produce:

- verified source manifest identity;
- eligible participant counts;
- structural exclusion counts/reasons;
- slow-trial counts;
- retained-trial counts;
- deterministic participant assignment;
- dataset/transform hashes;
- TRAIN fitting evidence;
- SELECTION evidence;
- frozen candidate parameter tuples.

It must not execute FINAL_TEST model predictions or compute FINAL outcome summaries before external witness completion.

The conformance result must verify the frozen 24 Magic Carpet + 21 Spaceship eligible participant inventory from the pinned source snapshot.

## 23. Locked real FINAL lane

The real FINAL execution is an explicit research workflow, not an ordinary push-CI expectation.

Its immutable prerequisites are:

- exact repository revision;
- verified source / transform / split / dataset identities;
- frozen model candidates;
- dual protocols;
- dual releases;
- OSF receipts;
- verified dual-release preflight.

Successful scientific completion means the frozen protocol executed with valid lineage. It does not mean a preferred model won.

The following are valid scientific outcomes rather than CI failures:

- all three models fail predictive adequacy;
- no pair is predictively separated;
- Brier and Log disagree on preferred model;
- task strata differ substantially.

Verification checks identity, accounting, mathematics, and claim scope, not a desired empirical conclusion.

## 24. Permitted and forbidden report language

Permitted canonical conclusions remain P3 typed statuses such as:

```text
predictive_adequacy_met
predictive_adequacy_not_met
predictively_separated_under_protocol
not_predictively_separated_under_protocol
```

Study V1 produces no external parameter-constraint status because `constraint_plans=()`.

The canonical report must not claim:

- empirical identification of latent cognition;
- structural identification;
- causal inference;
- cognitive mechanism confirmation;
- true participant goal recovery;
- true model-based/model-free latent labels.

A model with best external predictive score is only the best frozen predictor under this exact study protocol.

## 25. Non-goals

Study V1 does not:

- change P2 synthetic identification;
- change generic Reactive, Intentional, or Planning semantics;
- add participant-specific or hierarchical parameters;
- add task-specific fitted parameters;
- add full joint sequence likelihood;
- add continuous optimization, MCMC, variational inference, or asymptotic model evidence;
- add causal inference;
- use questionnaire covariates;
- use response time as a target;
- score second-stage choice as a primary endpoint;
- vendor raw human data;
- require network access in ordinary CI;
- require Docker;
- modify Lean proofs.

Population/hierarchical inference, longitudinal sequence likelihood, continuous/scalable inference, measurement-validity work, public benchmark packaging, and causal inference remain later roadmap frontiers.

## 26. Acceptance criteria for the written design

The implementation plan must preserve all of the following frozen decisions:

1. both Magic Carpet and Spaceship are one study with task strata;
2. behavioral-choice-only scientific evidence;
3. non-vendored exact-revision source manifest;
4. structural-only participant eligibility plus `slow == 1` trial exclusion;
5. per-trial first-stage choice as the primary target with strict same-trial information firewall;
6. deterministic task-stratified participant-disjoint 60/20/20 split;
7. task-specific adapter over unchanged Reactive/Intentional/Planning runtimes;
8. global low-dimensional parameters with no participant/task-specific refit;
9. Brier/Log primary external scoring with theory-referenced adequacy and pairwise separation;
10. common `1e-12` probability floor at the metric boundary;
11. immutable OSF registration as external witness before FINAL;
12. descriptive stay/switch secondary diagnostic only, with no external parameter-constraint analysis;
13. single-pass sealed FINAL prediction artifact consumed by both Brier and Log;
14. strict-better-than-uniform adequacy encoded via downward-adjacent floating thresholds without changing generic comparator semantics;
15. append-only FINAL attempt lineage with exact infrastructure retry and new-revision requirement for scientific defects.

Implementation may begin only after this written spec is reviewed and approved, followed by a separate implementation plan.