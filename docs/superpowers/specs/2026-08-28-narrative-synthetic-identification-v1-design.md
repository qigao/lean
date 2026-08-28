# Narrative Synthetic Identification V1 Design

Date: 2026-08-28

Status: approved design

Roadmap: #27 — P2 Identification and Model Comparison

Integrated base: `proof/narrative-dynamics-v0@f628dcea4176731e3fe7c5ec40170e63d0233759`

Feature branch: `work/narrative-synthetic-identification-v1`

## 1. Purpose

Narrative Synthetic Identification V1 completes the remaining P2 Identification and Model Comparison workstream by adding a first-class, preregistered synthetic protocol for determining when the existing reactive, intentional, and planning/POMDP narrative model families are observationally equivalent, when information interventions separate them, and when selected latent intentional parameters are recoverable or non-recoverable under a frozen finite design.

The scientific claim boundary is deliberately narrow:

> Results may state that a model family or latent parameter is identified or not identified **under the frozen synthetic protocol**. They must not claim global structural identifiability, empirical identification of real human cognition, or validation of a real-world psychological mechanism.

The objective is not to force a unique global winner. The objective is to make the information conditions that identify or fail to identify richer cognitive structure explicit, reproducible, provenance-bound, and testable.

## 2. Existing foundations to reuse

The implementation must compose existing infrastructure rather than replace it.

Existing capabilities include:

- `SimulationRunner` and seed-stable model execution;
- finite-grid calibration and training-target fitting;
- `ParameterAcceptanceSet`, repeated calibration, seed-block variation, and coordinate-wise `diagnose_identifiability()`;
- selection-validation and final-test held-out evaluation;
- categorical Brier and categorical log losses with stable manifest identities;
- frozen model candidates and preregistered final comparison;
- aggregate report attestation through `ExperimentManifest` and `attest_report()`;
- generic narrative reactive, intentional, and planning/POMDP runtime families executed through unified `run_runtime_decision()`;
- existing authored-narrative counterfactual intervention machinery.

These primitives remain authoritative. This feature adds orchestration, a synthetic identification benchmark, protocol validation, and typed findings.

## 3. Non-goals

This feature does not:

- modify the P1 shared-`beta` prison comparison adapter;
- redefine reactive, intentional, or planning/POMDP runtime semantics;
- add RNG or seed parameters to decision APIs;
- alter stochastic world/observation semantics from PR #36;
- change calibration, uncertainty, loss, held-out validation, or model-comparison algorithms;
- claim global structural identifiability;
- infer cognition of real people from synthetic fixtures;
- add continuous optimization, MCMC, variational inference, or asymptotic statistical theory;
- modify Lean sources;
- require Docker acceptance.

## 4. Architectural choice

### 4.1 Chosen architecture

Add an independent synthetic-identification layer:

- `narrative_dynamics/identification.py`
- `narrative_dynamics/adapters/narrative_prison_identification.py`
- `fixtures/observations/narrative_identification_v1.json`
- one additive experiment-manifest stage in `narrative_dynamics/contracts.py`

The P1 adapter `narrative_dynamics/adapters/narrative_prison.py` remains unchanged.

### 4.2 Rejected alternatives

#### Extend the P1 shared-`beta` adapter

Rejected because the existing held-out benchmark intentionally exposes one common `beta` across model families. Its intentional model fixes `beta_goal=1.0` and uses the shared `beta` as `beta_action`. Changing that adapter would mutate an already attested comparison boundary and conflate P1 fairness with P2 latent identification.

#### Implement the P2 only in tests

Rejected because tests alone cannot provide a first-class frozen protocol, typed non-identifiability findings, lineage-bound aggregate reports, or reusable fail-closed validation.

## 5. Scientific claim boundary

All public identification findings use exactly two successful statuses:

- `IDENTIFIED_UNDER_PROTOCOL`
- `NOT_IDENTIFIED_UNDER_PROTOCOL`

These statuses are meaningful only relative to the exact frozen protocol, fixture, candidate grid, acceptance rule, model implementations, information conditions, seeds, and scoring identities recorded in lineage.

The following are explicitly not permitted as successful statuses:

- `STRUCTURALLY_IDENTIFIED`
- `EMPIRICALLY_IDENTIFIED`
- `COGNITIVE_MECHANISM_CONFIRMED`

A failed generator self-recovery is not non-identifiability. If the true parameter tuple is excluded from the accepted set, the protocol fails with a typed recovery error.

## 6. Production file boundaries

### 6.1 `narrative_dynamics/identification.py`

This module owns only the research-protocol layer:

- frozen protocol types;
- parameter-recovery experiment declarations;
- accepted-set interpretation;
- coordinate-wise identification findings;
- observational-equivalence certification;
- information-intervention certification;
- dual proper-score comparison aggregation;
- final synthetic-identification report construction;
- typed identification errors.

It does not implement cognitive dynamics, probability kernels, optimization, or model-specific benchmark logic.

### 6.2 `narrative_dynamics/adapters/narrative_prison_identification.py`

This module owns the dedicated synthetic benchmark adapter.

It may build generic narrative/domain/runtime model objects, but execution must pass through unified `run_runtime_decision()` rather than family-specific direct runtime entry points.

The adapter must not import `narrative_dynamics.identification`, final comparison orchestration, observational preregistration, or report-release code. The dependency direction remains:

```text
identification protocol
        |
        v
identification adapter
        |
        v
unified narrative runtime
```

and never the reverse.

### 6.3 `narrative_dynamics/contracts.py`

Add exactly one `ExperimentStage` member:

```text
SYNTHETIC_IDENTIFICATION = "synthetic_identification"
```

No manifest schema redesign is required.

### 6.4 Synthetic fixture

Add:

`fixtures/observations/narrative_identification_v1.json`

The existing observation-dataset schema is reused. No dataset schema version change is required.

Every fixture record/case used by this feature must include canonical provenance sufficient to establish that the evidence is synthetic and non-empirical.

Required provenance fields include:

- `synthetic_non_empirical: true`
- `stratum`
- `generator_family`
- `generator_parameters_hash`
- `information_condition`
- `paired_case_id` where applicable

## 7. Public protocol and report types

The generic module should expose frozen dataclasses equivalent in responsibility to the following types.

### 7.1 `IdentificationStatus`

Enum values:

```text
IDENTIFIED_UNDER_PROTOCOL
NOT_IDENTIFIED_UNDER_PROTOCOL
```

### 7.2 `ParameterRecoveryExperiment`

One frozen finite recovery experiment. It binds:

- experiment name;
- model family/source identity;
- true parameter tuple;
- complete finite candidate grid;
- declared target coordinates to diagnose;
- synthetic case identities;
- observation/generation seeds where needed;
- calibration seed blocks;
- acceptance-loss delta;
- minimum acceptance fraction;
- metric extractor identity;
- loss identity;
- expected protocol outcome (`identified` or `not_identified`) only when this expectation is part of the preregistered synthetic design.

The true tuple must be present in the candidate grid.

### 7.3 `ParameterIdentificationFinding`

A completed recovery finding must retain:

- experiment identity/hash;
- true parameters;
- complete candidate losses or parent calibration report hashes sufficient to recover them;
- accepted candidates;
- whether truth is retained;
- coordinate-wise `IdentifiabilityReport`;
- typed `IdentificationStatus`;
- parent manifest hashes.

`calibration.best` alone is never sufficient evidence of identification.

### 7.4 `ObservationalEquivalenceFinding`

Binds:

- candidate model identities;
- exact case hashes;
- declared observable action keys;
- exact per-case policies;
- maximum absolute policy delta;
- equivalence tolerance;
- `equivalent` boolean;
- parent run/report hashes.

### 7.5 `InformationInterventionPair`

Declares a baseline/intervention pair with:

- pair name;
- baseline scenario hash;
- intervention scenario hash;
- exact information field/path allowed to change;
- frozen fields that must remain invariant;
- stratum;
- expected discriminating relationship where preregistered.

### 7.6 `InformationInterventionFinding`

Records:

- certified canonical scenario diff;
- family policies on both sides;
- within-family policy deltas;
- cross-family contrasts;
- whether the pair is discriminating under the preregistered tolerance;
- parent hashes.

### 7.7 `ScoredModelComparisonFinding`

Aggregates one frozen final comparison and records:

- loss identity;
- frozen candidate identities and parameters;
- final target/partition identity;
- final seeds;
- global mean/worst scores;
- per-stratum mean/worst scores;
- pairwise score deltas;
- parent comparison report hash.

### 7.8 `SyntheticIdentificationProtocol`

The top-level preregistration identity. It binds all declarations needed before result inspection.

### 7.9 `SyntheticIdentificationReport`

The final aggregate report. It contains:

- protocol hash;
- claim scope;
- parameter-recovery findings;
- goal/instrumentality recovery and non-recovery findings;
- observational-equivalence findings;
- information-intervention findings;
- Brier comparison finding;
- log comparison finding;
- parent manifest hashes;
- `ExperimentManifest(stage=SYNTHETIC_IDENTIFICATION, ...)`.

The report must be accepted by existing `attest_report(report).require_integrity()`.

## 8. Canonical identity rules

All public protocol/report dataclasses are frozen.

Identity-bearing collections must use deterministic canonical order.

Parameter tuples must be sorted by parameter name.

Case/model/finding collections must be sorted by stable declared key rather than input iteration order.

No result may use object identity, process address, set iteration order, or implicit dictionary insertion order as semantic identity.

The protocol itself must have a deterministic `content_hash` computed from its canonical semantic declaration.

Aggregate report identity continues to use the existing manifest-plus-attested-report mechanism rather than introducing a parallel artifact system.

## 9. Typed errors

Use one root error:

```python
class SyntheticIdentificationError(ValueError):
    pass
```

Subclasses:

### 9.1 `IdentificationProtocolError`

Raised for invalid or drifted protocol declarations, including:

- true tuple absent from grid;
- duplicate experiment/pair identifiers;
- partition overlap;
- unsupported claim scope;
- Brier/log sibling protocol drift beyond explicitly permitted loss-specific fields;
- changed metric identity, model candidate identity, target lineage, or seed plan;
- malformed finite grids;
- invalid expected-status declaration.

### 9.2 `IdentificationRecoveryError`

Raised when a recovery experiment cannot support either successful identification status, including:

- accepted set is empty;
- true tuple is not accepted;
- generator self-recovery fails;
- candidate schema changes across calibration blocks.

### 9.3 `InterventionCertificationError`

Raised when an intervention pair changes anything outside the preregistered information boundary, including reward, objective dynamics, action set, or other frozen fields.

### 9.4 `IdentificationComparisonError`

Raised for frozen final-comparison drift, including candidate identity, selected parameter, loss, seed, partition, target, or sibling-protocol inconsistencies.

### 9.5 `IdentificationReportError`

Raised for forged or internally inconsistent findings or aggregate lineage.

Non-identifiability is not an exception.

## 10. Parameter identification semantics

### 10.1 Accepted-set rule

For a finite candidate grid, acceptance is defined by the preregistered rule:

```text
candidate_loss <= best_loss + acceptance_loss_delta
```

When repeated seed blocks are declared, a candidate must satisfy the preregistered minimum acceptance fraction across blocks.

The resulting candidates form an explicit `ParameterAcceptanceSet`.

### 10.2 Identification criterion

A target coordinate is identified when every retained candidate has the same value for that coordinate.

A multi-coordinate experiment is `IDENTIFIED_UNDER_PROTOCOL` only when:

1. the accepted set is non-empty;
2. the true tuple is retained;
3. every declared target coordinate is coordinate-wise identified;
4. each identified coordinate equals the true coordinate value.

### 10.3 Non-identification criterion

An experiment is `NOT_IDENTIFIED_UNDER_PROTOCOL` only when:

1. the accepted set is non-empty;
2. the true tuple is retained;
3. at least one declared target coordinate retains multiple values.

The accepted set must remain visible in the report.

### 10.4 Failure criterion

If truth is not retained, the experiment raises `IdentificationRecoveryError` instead of producing `NOT_IDENTIFIED_UNDER_PROTOCOL`.

This distinguishes non-identifiability from model misspecification, incorrect fixture construction, an insufficient grid, or a broken calibration implementation.

## 11. Synthetic intentional parameterization

The dedicated intentional identification adapter exposes at least:

- `beta_goal`
- `beta_action`
- `goal_pressure_scale`
- `instrumentality_scale`

The underlying existing intentional equations remain unchanged.

For each goal:

```text
goal score = pressure * E[instrumentality | belief] - cost - risk
```

`beta_goal` controls the goal softmax.

`beta_action` controls each goal-conditional action softmax.

The final action policy is the marginal over the goal policy and conditional action policies.

No parameter is added merely to make model schemas symmetric across families.

Reactive and planning comparison sources expose only parameters with actual semantics in those model families.

## 12. Joint `beta_goal` / `beta_action` recovery

### 12.1 Candidate grid

Initial finite design:

```text
beta_goal   in {0.5, 1.0, 2.0, 4.0}
beta_action in {0.5, 1.0, 2.0, 4.0}
```

Reference synthetic truth:

```text
beta_goal = 2.0
beta_action = 1.0
```

### 12.2 Required complementary cases

The protocol includes at least three complementary cases.

#### Action-temperature anchor

Latent goals have identical action-value vectors. The marginal action policy is invariant to the goal mixture and therefore isolates `beta_action`.

#### Goal-temperature low-gap

Goals have opposed action preferences and a small nonzero goal-score gap. Once action temperature is constrained, this case is sensitive to `beta_goal`.

#### Goal-temperature high-gap

Same qualitative structure with a larger goal-score gap. It prevents accidental finite-grid aliasing from a single gap magnitude.

### 12.3 Success criterion

The accepted set contains exactly the reference truth pair and `diagnose_identifiability()` reports both coordinates identified.

The test must inspect the accepted set, not only the lexically ranked best candidate.

## 13. Observational equivalence fixture

A preregistered equivalence fixture deliberately makes latent variation observationally invisible.

Required structure:

- equal goal scores produce a symmetric goal mixture;
- goal-conditional action policies are mirror-symmetric;
- multiple latent parameter tuples generate the same declared observable action policy within tolerance.

The protocol must retain multiple accepted latent candidates including the true tuple.

The corresponding finding must be:

```text
NOT_IDENTIFIED_UNDER_PROTOCOL
```

and the observational-equivalence finding must certify policy equivalence directly.

The implementation must not treat equal final losses or lexical ranking ties as proof of observational equivalence.

## 14. Goal pressure and instrumentality experiments

### 14.1 Pressure recovery

Freeze instrumentality and vary:

```text
goal_pressure_scale in {0.5, 1.0, 2.0}
```

The frozen discriminating cases must uniquely recover the preregistered true pressure scale.

### 14.2 Instrumentality recovery

Freeze pressure and vary:

```text
instrumentality_scale in {0.5, 1.0, 2.0}
```

The frozen discriminating cases must uniquely recover the preregistered true instrumentality scale.

### 14.3 Scale-confounded non-recovery

Free both coordinates in a deliberately confounded design containing candidates such as:

```text
(goal_pressure_scale, instrumentality_scale)
(0.5, 2.0)
(1.0, 1.0)
(2.0, 0.5)
```

When the fixture leaves only the product `pressure * expected instrumentality` observable, the accepted set must retain the preregistered equivalence class, including truth.

The correct outcome is `NOT_IDENTIFIED_UNDER_PROTOCOL`, not a tie-broken recovery.

## 15. Observational-equivalence certification

Two predictions are equivalent only when complete declared action-policy vectors satisfy:

```text
max_action |P_left(action) - P_right(action)| <= 1e-12
```

This must hold on every case declared in the equivalence fixture.

The tolerance is fixed in the protocol and cannot be changed after result inspection.

The finding records the actual maximum absolute delta.

Equivalence is about declared observables only. Internal model hashes, latent goal states, or provenance hashes may differ without invalidating observable equivalence.

## 16. Information-intervention suite

The protocol uses paired synthetic information conditions rather than modifying the generic authored-narrative intervention API.

Each pair must prove that objective/reward structure is frozen and only one declared information factor changes.

### 16.1 Memory/evidence intervention

Current observable cue and reward structure remain fixed. Historical evidence availability changes.

Purpose:

- reactive behavior remains limited to current observable cues;
- intentional behavior may change because belief-mediated evidence history changes.

This is the required discriminating stratum for Intentional vs Reactive.

### 16.2 Future-information / value-of-information intervention

Current belief, immediate reward structure, and available initial actions remain fixed. The availability or informativeness of a future observation after an information-gathering action changes.

Purpose:

- myopic intentional behavior does not acquire planning value from a future observation channel;
- planning/POMDP behavior may change because future information alters expected continuation value.

This is the required discriminating stratum for Intentional vs Planning/POMDP.

### 16.3 Goal-evidence reveal/hide intervention

Objective state remains fixed. Evidence that creates a goal-score contrast is revealed or hidden.

Purpose:

- show a protocol in which `beta_goal` is identifiable under a discriminating evidence condition;
- show a paired symmetric blackout condition in which the latent goal-temperature coordinate is not identifiable.

## 17. Intervention certification

An intervention pair is valid only if its canonical scenario diff matches exactly the preregistered allowed information path/field set.

The certification must reject a pair if it also changes any frozen factor, including:

- rewards or costs;
- action set;
- objective transition dynamics;
- latent ground truth when it is declared frozen;
- horizon when it is declared frozen;
- unrelated observation parameters;
- model-family identity.

A pair that fails certification raises `InterventionCertificationError` before model comparison.

## 18. Synthetic dataset partitions

The fixture uses the existing partition roles:

```text
train
selection_validation
final_test
```

No scenario or observation record may appear in more than one role by semantic identity.

The protocol binds exact partition hashes before final evaluation.

The final-test partition contains at least the following strata:

- `observational_equivalence`
- `memory_evidence_intervention`
- `future_information_intervention`
- `latent_recovery_stress`

The exact case identities are preregistered.

## 19. Two-stage preregistration

### 19.1 Stage A — identification-design preregistration

Before recovery or comparison results are inspected, freeze `SyntheticIdentificationProtocol` with:

- claim scope;
- adapter implementation identity;
- fixture/dataset hash;
- partition hashes;
- parameter grids and truths;
- recovery cases;
- acceptance rules;
- equivalence tolerance;
- intervention pairs and allowed diffs;
- candidate families;
- metric identity;
- seed plans;
- final strata;
- proper-score plan;
- status criteria.

After Stage A is frozen, result-dependent changes require a new protocol version rather than mutating the existing one.

### 19.2 Stage B — final-model preregistration

Training and selection-validation may select one parameterization per candidate model family according to already frozen procedures.

After selection, freeze the exact candidate identities and selected parameters for final test.

No candidate refit is allowed on final-test observations.

## 20. Dual proper-score final comparison

The final synthetic comparison must report both:

- categorical Brier loss;
- categorical log loss.

Do not create one protocol with a runtime-switchable loss.

Create two sibling frozen final protocols:

```text
<name>-brier-v1
<name>-log-v1
```

The siblings must be identical in:

- dataset identity;
- final partition;
- target-construction identity except where the existing loss-specific final comparison requires no target difference;
- model candidate identities;
- selected parameters;
- metric extractor;
- final simulation seeds;
- model ordering/ranking rule;
- identification strata.

The only permitted differences are:

- loss identity;
- explicitly loss-specific adequacy threshold if the protocol defines one;
- sibling protocol name/hash derived from that declared difference.

Any other drift raises `IdentificationProtocolError` or `IdentificationComparisonError` before final interpretation.

## 21. Model-comparison interpretation

The report contains both global and stratum-level results.

### 21.1 Global frozen final score

For each family and each proper score:

- mean loss;
- worst loss;
- pairwise delta from declared baseline or pairwise peers.

### 21.2 Identification strata

Report the same families separately over:

- observational-equivalence cases;
- memory/evidence interventions;
- future-information interventions;
- latent-recovery stress cases.

The protocol does not require one family to win every stratum.

A valid synthetic result may show:

- Reactive and Intentional equivalent on an observational-equivalence stratum;
- Intentional separated from Reactive on an evidence-history stratum;
- Planning separated from Intentional on a future-information stratum.

This is preferable to interpreting one aggregate leaderboard as proof that one cognitive architecture is universally superior.

## 22. Aggregate lineage

The final `SyntheticIdentificationReport` uses:

```text
ExperimentManifest(stage=SYNTHETIC_IDENTIFICATION, ...)
```

Its manifest inputs bind at minimum:

- protocol hash;
- claim scope;
- fixture/dataset hash;
- recovery finding hashes or parent manifests;
- equivalence finding hashes;
- intervention finding hashes;
- Brier comparison report hash;
- log comparison report hash;
- sibling-consistency certification;
- final seed identities.

The report must successfully pass existing aggregate report attestation.

Forged parent hashes or inconsistent nested findings fail closed.

## 23. Textual reporting rules

Typed findings may be rendered only with bounded language.

Allowed examples:

> `beta_goal` is identified under this frozen synthetic protocol.

> Goal pressure and instrumentality scale are not separately identified under this frozen synthetic protocol; multiple preregistered candidates remain observationally compatible.

> This information intervention separates the intentional and reactive model families under the frozen synthetic benchmark.

Disallowed examples:

> `beta_goal` is structurally identifiable in general.

> The experiment proves human goal temperature exists.

> Planning is the true cognitive model.

## 24. Failure handling

All preflight/certification failures are deterministic and fail closed.

The implementation must reject at least:

- empty grids;
- non-finite candidate values;
- missing true tuple;
- duplicate experiment names;
- duplicate intervention-pair names;
- empty accepted set;
- accepted set excluding truth;
- candidate-schema drift across seed blocks;
- train/selection/final partition overlap;
- non-finite policies or losses;
- incomplete or non-normalized policy vectors;
- forged model identity;
- final candidate-parameter drift;
- final seed drift;
- loss identity drift;
- metric identity drift;
- target/partition lineage drift;
- Brier/log sibling drift outside permitted fields;
- intervention pairs that modify reward/dynamics/action set in addition to information;
- forged report parent lineage.

## 25. RED-to-GREEN acceptance design

### 25.1 Test-only RED files

Authoritative RED adds five test files before production implementation:

- `tests/test_narrative_identification_protocol.py`
- `tests/test_narrative_identification_recovery.py`
- `tests/test_narrative_identification_interventions.py`
- `tests/test_narrative_identification_model_comparison.py`
- `tests/test_narrative_identification_reporting.py`

At the RED commit, all pre-existing tests must remain GREEN. New failures/errors must be confined to the missing P2 identification APIs/fixture.

### 25.2 Monotonic implementation sequence

Implementation proceeds in atomic RED-to-GREEN layers:

1. protocol/report core;
2. dedicated identification adapter;
3. parameter recovery and equivalence;
4. information-intervention certification;
5. dual Brier/log final comparison;
6. aggregate reporting/attestation;
7. final exact-head full proof.

Future-layer tests may remain RED while an earlier task reaches its task-local GREEN boundary. Failure count/signature must reduce monotonically without unrelated regressions.

### 25.3 Final workflow gate

Before PR creation, the exact feature head must pass the complete repository `proof` workflow, including:

- dependencies;
- Lean/Python conformance;
- full Lean build;
- Lean theorem tests;
- all Python numerical tests, including the five new P2 files;
- narrative story theorem tests;
- narrative testimony theorem tests.

No Docker acceptance is required for this feature.

## 26. Required positive acceptance cases

Final GREEN must explicitly demonstrate all remaining #27 P2 items.

### 26.1 Intentional vs Reactive comparison

A certified memory/evidence intervention must produce a preregistered family-separating policy contrast while frozen objective/reward conditions remain unchanged.

### 26.2 Intentional vs POMDP comparison

A certified future-information/value-of-information intervention must produce a preregistered family-separating contrast while current information/reward conditions remain frozen.

### 26.3 Information intervention suite

Every declared pair must pass information-only canonical diff certification.

### 26.4 Synthetic `beta_goal` / `beta_action` recovery

The joint finite accepted set must contract to the true pair under the discriminating protocol, with both coordinates identified.

### 26.5 Goal and instrumentality recovery/non-recovery

Pressure-only and instrumentality-only experiments recover their true coordinate; the preregistered scale-confounded joint experiment retains multiple truth-containing candidates and reports non-identification.

### 26.6 Observational-equivalence fixtures

The frozen equivalence fixture must certify full policy-vector equivalence within `1e-12`, while a discriminating information intervention breaks that equivalence where declared.

### 26.7 Held-out Brier/log comparison

The same frozen candidates, final cases, metric extractor, and final seeds complete both sibling proper-score protocols. Per-stratum results are retained.

### 26.8 Explicit non-identifiability reporting

A successful non-identification case must emit typed `NOT_IDENTIFIED_UNDER_PROTOCOL`, preserve its multi-candidate accepted set, and remain attestable. Lexical best-candidate ranking must not change this finding.

## 27. Negative acceptance cases

Tests must fail closed for at least:

- truth absent from grid;
- truth excluded by accepted set;
- empty accepted set;
- duplicate pair/experiment identifiers;
- partition overlap;
- malformed/non-finite policy;
- intervention plus reward drift;
- intervention plus action-set drift;
- model identity drift;
- selected-parameter drift;
- seed drift;
- target/partition drift;
- metric drift;
- loss drift;
- Brier/log sibling drift outside loss/threshold/name;
- forged aggregate parent lineage.

## 28. Scope gate before PR

Relative to integrated base `f628dcea4176731e3fe7c5ec40170e63d0233759`, allowed changes are limited to:

- `docs/superpowers/specs/2026-08-28-narrative-synthetic-identification-v1-design.md`
- one implementation plan under `docs/superpowers/plans/`
- `narrative_dynamics/contracts.py`
- `narrative_dynamics/identification.py`
- `narrative_dynamics/adapters/narrative_prison_identification.py`
- `fixtures/observations/narrative_identification_v1.json`
- the five identification test files
- an optional additive package export only if required by the established public-API pattern

Forbidden scope includes:

- `narrative_dynamics/adapters/narrative_prison.py`
- runtime reactive/intentional/planning implementation files
- `narrative_dynamics/calibration.py`
- `narrative_dynamics/uncertainty.py`
- `narrative_dynamics/losses.py`
- `narrative_dynamics/model_comparison.py`
- existing empirical/observational fixtures
- stochastic world/observation implementation
- Lean production/test sources

Any forbidden production change blocks PR creation unless separately designed and approved.

## 29. Integration discipline

The branch starts from exactly:

`proof/narrative-dynamics-v0@f628dcea4176731e3fe7c5ec40170e63d0233759`

Before merge:

1. feature-head exact proof must be GREEN;
2. base-to-head scope must match Section 28;
3. PR synthetic-merge proof must be GREEN;
4. merge must be guarded by the exact expected feature head.

After merge:

1. observe a push-triggered proof on the exact integration commit;
2. require the full proof workflow GREEN;
3. only then update roadmap #27.

## 30. Roadmap completion criterion

The remaining eight P2 Identification and Model Comparison checkboxes are evaluated only after merge and post-merge exact-head GREEN.

They are:

1. Intentional vs reactive comparison;
2. Intentional vs POMDP comparison;
3. Information intervention suite;
4. Synthetic recovery for `beta_goal` and `beta_action`;
5. Recovery / non-recovery cases for goal and instrumentality parameters;
6. Observational-equivalence fixtures;
7. Held-out Brier/log-score comparison;
8. Explicit reporting when latent cognition is not identifiable.

If all eight are supported by the frozen final protocol and post-merge proof, mark all eight complete and close #27 because no roadmap workstream remains open.

If any item is not demonstrated, leave that checkbox open and keep #27 open.

## 31. Definition of done

Narrative Synthetic Identification V1 is complete only when all of the following hold:

- the design and implementation plan are committed;
- authoritative test-only RED is recorded on an exact head;
- all new tests pass without weakening pre-existing tests;
- the joint `beta_goal`/`beta_action` synthetic truth is recovered by accepted-set semantics;
- the scale-confounded goal/instrumentality experiment correctly reports non-identification;
- observational equivalence is certified from full policy vectors, not score ties;
- information-only interventions separate the declared family pairs;
- both frozen Brier and log final comparisons complete on the same candidates/cases/seeds;
- aggregate report attestation succeeds;
- feature-head proof is fully GREEN;
- PR synthetic merge is fully GREEN;
- guarded merge succeeds;
- post-merge exact-head proof is fully GREEN;
- roadmap #27 is updated only from this final integrated evidence.

The resulting scientific statement is intentionally limited:

> Under a frozen, provenance-bound synthetic protocol, the engine can identify which observation/intervention designs distinguish reactive, intentional, and planning/POMDP behavior, and can report when selected latent intentional parameters are or are not identifiable without overstating those synthetic findings as real-world cognitive identification.
