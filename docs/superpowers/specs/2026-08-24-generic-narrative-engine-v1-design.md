# Generic Narrative Engine V1 Design

Date: 2026-08-24

Status: design approved in chat; implementation not started

Base research SHA: `82096098dda907aec04330b5ddf357ac6f753d2a`

Target branch: `work/generic-narrative-engine-v1`

## 1. Purpose

Generic Narrative Engine V1 generalizes the existing location/testimony research path into a reusable structured narrative research engine without weakening the current provenance, fail-closed, deterministic replay, anti-leak, or mechanism-identification boundaries.

The engine is intended to study canonical chains of the form:

```text
world event
→ information exposure / communication
→ agent-specific evidence state
→ decision context
→ decision mechanism
→ action
→ temporal evolution
→ counterfactual divergence
```

The engine is not a free-form story generator and is not a general natural-language understanding system. It executes only validated canonical narrative semantics.

The main architectural goal is to make domain semantics pluggable while keeping information propagation, epistemic replay, counterfactual validation, provenance handling, and analysis semantics common across domains.

## 2. Scientific boundary

The strongest intended research claim is:

> Narrative Dynamics can execute a typed, provenance-preserving, domain-pluggable canonical narrative semantics in which objective state, direct evidence, communicated claims, recipient-specific epistemic state, decision mechanisms, and single-variable counterfactual divergences remain explicitly separable and reproducible.

This design does not establish:

- general Theory of Mind;
- empirical human validity;
- population representativeness;
- causal psychological mechanisms;
- deception intent or trust truthfulness;
- unique hidden-mechanism identification from observed action;
- unrestricted natural-language understanding;
- unrestricted causal discovery;
- arbitrary future-world prediction.

Final action alone must never be treated as proof of the internal information mechanism that produced it.

## 3. Architectural decision summary

The approved architecture has the following hard boundaries:

1. LLM/NLP extraction is external and non-canonical.
2. The compiler core is deterministic.
3. Domain semantics are expressed through a typed `DomainSpec` plus narrowly scoped pure semantic hooks.
4. The canonical narrative uses a typed event-sourced semantic IR.
5. Objective world state, direct agent evidence, and communicated testimony/claims remain separate replay paths.
6. Belief-like views are derived, not authored as truth.
7. Decision facts and decision mechanisms are separate.
8. Decision models receive capability-limited immutable views, not the full scenario.
9. Counterfactuals modify only upstream canonical history through typed single-variable interventions.
10. Derived world, epistemic, or decision state cannot be patched directly.
11. Analysis artifacts separate baseline trajectories, mechanism comparisons, and counterfactuals.
12. Source/compiler provenance identity is separate from canonical semantic identity.
13. Domain declaration identity and semantic-hook implementation identity are both bound into `DomainSpec` identity.
14. Legacy Narrative Testimony V2 remains unchanged and becomes the primary conformance reference domain.
15. A second synthetic non-location domain is required to prove the generic path is not location semantics with renamed fields.
16. The first Generic Engine increment does not require new Lean core semantics; all existing Lean proof gates remain mandatory regressions.

## 4. High-level dependency direction

```text
External extractor
      ↓
Candidate bundle + source provenance
      ↓
Deterministic Narrative Compiler
      ↕
   DomainSpec
      ↓
Generic Narrative IR
      ↓
Generic semantic validation
      ↓
Objective replay
      ↓
Direct / epistemic replay
      ↓
Decision-context projection
      ↓
DecisionModel
      ↓
Generic analysis
  ├─ baseline
  ├─ mechanism comparison
  └─ counterfactuals
```

Legacy compatibility path:

```text
NarrativeScenarioV2
      ↓
LocationTestimonyDomainAdapter
      ↓
Generic Narrative IR
      ↓
Generic replay / decision / analysis
```

The generic story layer must never import story adapters or model runtime adapters in order to implement semantic replay.

## 5. External extraction versus deterministic compilation

### 5.1 External candidate extractor

LLM/NLP systems may propose candidate entities, events, propositions, claims, observations, receptions, decisions, and temporal relations.

The extractor is not a truth authority and must not directly create canonical runtime semantics.

Extractor outputs may contain:

- source spans;
- extractor identity and version;
- confidence values;
- unresolved entity alternatives;
- unresolved coreference;
- unresolved temporal alternatives;
- candidate support relations.

These values live only in compiler provenance.

### 5.2 Deterministic compiler

The compiler performs only deterministic operations over:

- candidate records;
- resolution records;
- a selected `DomainSpec`;
- canonical validation rules.

The compiler must not:

- call an LLM;
- call the network;
- use RNG;
- depend on wall-clock time;
- silently choose the highest-confidence candidate when semantic ambiguity remains;
- infer canonical truth from extraction confidence.

Compilation either yields a fully validated canonical narrative or yields no canonical narrative.

## 6. Source and candidate provenance model

### 6.1 Source records

```text
SourceDocument
  document_id
  content_hash
  source_type
  source_locator
```

```text
SourceSpan
  document_id
  start
  end
  exact_text_hash
```

A source span must be verifiable against its source document content identity.

### 6.2 Candidate records

V1 candidate types:

```text
CandidateEntity
  proposed_id
  proposed_type
  source_refs
  confidence
```

```text
CandidateEvent
  proposed_type
  logical_time?
  actor?
  arguments
  source_refs
  confidence
```

```text
CandidateProposition
  subject
  state_variable
  relation
  value
  source_refs
  confidence
```

```text
CandidateClaim
  speaker
  proposition
  support_refs?
  source_refs
  confidence
```

```text
CandidateObservation
  agent
  observed_ref
  source_refs
```

```text
CandidateReception
  recipient
  claim_ref
  source_refs
```

```text
CandidateDecision
  actor
  logical_time?
  decision_type
  context
  candidate_actions
  source_refs
```

Confidence must be finite but must never affect canonical semantic validation.

### 6.3 Resolution records

Candidate bindings have exactly three semantic states:

```text
accepted
rejected
unresolved
```

A trusted resolver may provide structured decisions such as:

- accept a candidate;
- reject a candidate;
- bind a candidate entity to a canonical entity;
- choose one value from explicit alternatives;
- supply a required typed field;
- select a temporal ordering.

Each resolution is recorded in a `ResolutionRecord` containing at least:

```text
candidate_id
resolution_kind
selected_value?
resolver_identity
reason?
```

Resolution records are compiler provenance and do not enter model-visible canonical IR.

### 6.4 Fail closed on semantic ambiguity

If any unresolved candidate is required to determine canonical semantics, compilation returns:

```text
status = incomplete
canonical_scenario = None
```

The compiler may not silently guess.

Conflicting communicated claims are allowed if they are represented as distinct claims. Extractor uncertainty about what a source means is not the same as a canonical conflicting claim.

## 7. Generic typed event-sourced IR

The Generic IR stores canonical typed facts and event history. It does not store redundant derived world-state snapshots as input truth.

### 7.1 Entity types and entities

```text
EntityTypeSpec
  name
```

```text
Entity
  id
  type
```

Entity IDs must be unique inside one narrative.

### 7.2 Value types

```text
ValueTypeSpec
  name
  kind
  allowed_values?
  referenced_entity_type?
```

V1 `kind` values:

```text
enum
bool
integer
text
entity_ref
```

No arbitrary Python objects are valid canonical values.

### 7.3 State variables

```text
StateVariableSpec
  name
  subject_type
  value_type
  cardinality
```

V1 supports only:

```text
cardinality = one
```

A state cell is identified by:

```text
(subject_id, state_variable)
```

Examples:

```text
key.location
company.regulatory_status
service.health
territory.control
```

### 7.4 Event types and events

```text
EventTypeSpec
  name
  actor_type?
  parameters
  affected_state_variables
  intervenable_parameters
  transition_hook
```

```text
Event
  id
  logical_time
  type
  actor?
  arguments
```

Events are the only authored mechanism that changes objective world state in V1.

Events must be strictly ordered by canonical logical time under Generic Narrative validation.

### 7.5 Proposition

V1 propositions are deliberately restricted typed state assertions:

```text
Proposition
  subject_id
  state_variable
  relation
  value
```

V1 relations:

```text
equals
not_equals
```

V1 does not support arbitrary predicate code, quantifiers, free-form formulas, or user-defined logical operators.

An `equals` proposition may resolve a current value. A `not_equals` proposition is retained as a constraint and does not cause the engine to invent a unique value.

### 7.6 Observation

```text
Observation
  event_ref
  agent_id
  channel
```

V1 direct evidence derives only from the actual typed state delta produced by the observed event.

Observing an event does not grant access to all objective world state.

### 7.7 Claim

```text
Claim
  id
  logical_time
  speaker_id
  proposition
  support_refs
  claim_type
```

Claim support is provenance, not truth.

A supported claim may disagree with objective world state.

### 7.8 Reception

```text
Reception
  claim_ref
  recipient_id
  channel
```

Reception controls whether a communicated claim can enter a recipient's epistemic evidence history.

### 7.9 Decision and actions

Canonical IR stores only the decision fact and its allowed actions:

```text
Decision
  id
  logical_time
  actor_id
  decision_type
  context_targets
  action_options
```

The canonical decision must not contain `decision_model_ref`.

Decision mechanism selection belongs to the experiment/analysis layer.

## 8. DomainSpec V1

### 8.1 Structure

```text
DomainSpec
  domain_id
  version
  entity_types
  value_types
  state_variables
  event_types
  decision_types
  action_types
  semantic_hooks
```

The declarative portion must be canonicalizable and content-hashed.

### 8.2 Pure semantic hooks

V1 semantic hooks are limited to objective state transitions.

Conceptual contract:

```text
apply_event(prior_state, typed_event) -> StateDelta
```

`StateDelta` V1 operations:

```text
set(subject, state_variable, value)
clear(subject, state_variable)
```

Hooks may not:

- use files;
- use network access;
- use wall-clock time;
- use RNG;
- call an LLM;
- mutate global state;
- create entities;
- create events;
- create claims;
- decide observations;
- author beliefs;
- select actions;
- return arbitrary Python objects.

A hook answers only:

> What deterministic objective state delta does this already-valid event produce?

Python purity is not treated as mathematically proven. Formal trust is based on immutable contracts, deterministic conformance tests, and implementation identity.

### 8.3 Generic semantics that domains cannot redefine

A domain cannot override:

- strict canonical time ordering;
- observation ownership;
- claim speaker semantics;
- claim/reception relationship;
- objective state versus agent evidence separation;
- support provenance versus truth separation;
- no-objective-fallback behavior;
- counterfactual fail-closed reconstruction;
- analysis first-divergence semantics.

## 9. Generic semantic validation

Validation is split into four layers.

### 9.1 Candidate validation

Checks only source/extractor structural integrity:

- source span exists;
- source hashes match;
- confidence is finite;
- candidate reference structure is valid.

Candidate validation never certifies truth.

### 9.2 Compiler validation

Checks:

- ontology binding uniqueness;
- required field resolution;
- deterministic temporal ordering;
- typed value compatibility;
- absence of semantic ambiguity.

### 9.3 Canonical narrative validation

This validation ignores LLM confidence and source prose.

It checks:

- `DomainSpec` identity;
- entity typing;
- state variable typing;
- event argument typing;
- event transition legality;
- strict event/claim/decision ordering constraints;
- observation references;
- claim proposition typing;
- support references and provenance rules;
- reception references;
- decision/action typing;
- Generic Engine invariants.

### 9.4 Analysis validation

Checks:

- analysis scope;
- DecisionModel evidence-access contract;
- model identity;
- intervention validity;
- complete revalidation of every counterfactual candidate narrative.

No earlier validation layer may be used to bypass a later one.

## 10. Objective and epistemic replay

Generic replay remains event-sourced.

### 10.1 Objective world state

```text
WorldState
  StateCell -> CanonicalValue
```

Objective replay applies every valid event's deterministic `StateDelta` in logical-time order.

Objective replay does not read observations, claims, receptions, or decisions.

### 10.2 Direct evidence

An agent directly observing an event receives evidence only for the state delta produced by that event.

Conceptual record:

```text
EpistemicEvidence
  subject_id
  state_variable
  relation
  value
  evidence_kind
  supporting_id
  source_agent
  logical_time
  provenance_refs
```

For direct perception:

```text
evidence_kind = direct_perception
supporting_id = event.id
source_agent = querying agent
```

No objective fallback is permitted.

### 10.3 Communicated evidence

A received claim contributes testimony evidence to the recipient's evidence history.

For testimony:

```text
evidence_kind = testimony
supporting_id = claim.id
source_agent = claim.speaker
```

Claim truth is not inferred from claim support.

### 10.4 Evidence history and derived views

Generic replay retains full evidence history.

```text
EpistemicState
  evidence_history
  latest_evidence_by_state_cell
  resolved_values
  constraints
```

The standard replay views are:

```text
objective_state(story, at_time)
direct_state(story, agent, at_time)
epistemic_state(story, agent, at_time)
```

Domains cannot redefine these access paths.

### 10.5 Conflict handling

Conflicting evidence is preserved explicitly.

A generic epistemic view may report:

```text
status = resolved | unknown | conflicted
```

The Generic Engine does not choose which source is more trustworthy.

Decision models may implement explicit conflict-handling strategies.

## 11. Decision Model architecture

### 11.1 Separation from canonical narrative

`Decision` is a narrative fact.

`DecisionModel` is an experiment mechanism.

The same canonical narrative must be executable under multiple competing DecisionModels without changing narrative identity.

### 11.2 DecisionModelSpec

```text
DecisionModelSpec
  model_id
  version
  supported_decision_types
  evidence_access
  parameter_schema
  decision_hook
```

### 11.3 Evidence-access capability

V1 standard capabilities:

```text
direct_only
epistemic
omniscient
custom_epistemic
```

Definitions:

- `direct_only`: actor direct evidence only;
- `epistemic`: actor direct evidence plus received claims;
- `omniscient`: objective world state, allowed only as an explicit baseline;
- `custom_epistemic`: actor evidence history and provenance, but no objective world state.

A model receives a capability-limited immutable `DecisionContext`, not the raw scenario.

Conceptual context:

```text
DecisionContext
  decision
  actor
  logical_time
  visible_state
  evidence_history
  unresolved_constraints
  permitted_metadata
```

### 11.4 DecisionResult

```text
DecisionResult
  model_id
  selected_action
  action_scores
  policy
  basis
```

`basis` includes canonical evidence references and state-cell resolutions needed to audit the decision.

The selected action must be exactly one declared action option.

### 11.5 V1 determinism

Generic DecisionModel V1 is deterministic by default.

Probabilistic choice, seeded sampling, calibration, and fitted parameters remain future extensions that may reuse the repository's existing runtime/calibration infrastructure.

### 11.6 Mechanism-identification boundary

A model matching observed behavior establishes only behavioral compatibility under the declared scenario and model contract.

The engine must never emit `true_mechanism`, `unique_cause`, or equivalent hidden-mechanism claims.

## 12. Generic counterfactual architecture

### 12.1 Intervention principle

V1 interventions are typed, upstream, single-variable changes to canonical history.

Derived world, epistemic, and decision state must always be recomputed.

### 12.2 V1 intervention kinds

```text
remove_event
change_event_argument
remove_observation
change_claim_value
remove_reception
```

`change_event_argument` may modify only a parameter marked `intervenable` by the `EventTypeSpec`.

`change_claim_value` changes only proposition value while keeping subject, state variable, relation, speaker, support, and reception structure fixed.

### 12.3 Forbidden counterfactual mutations

V1 forbids:

- arbitrary JSON patching;
- replacing a whole event;
- changing actor, time, and payload together;
- injecting undeclared entities;
- modifying derived world state;
- modifying derived epistemic state;
- modifying `DecisionContext` directly;
- changing DecisionModel under the name of a narrative counterfactual.

Changing DecisionModel is mechanism comparison, not narrative intervention.

### 12.4 Counterfactual evaluation pipeline

```text
baseline canonical narrative
→ apply one intervention
→ reconstruct typed candidate narrative
→ DomainSpec validation
→ Generic narrative validation
→ replay
→ same DecisionModel
→ trajectory comparison
```

Rejected counterfactuals are retained as rejected records; they are never repaired into fake valid trajectories.

### 12.5 Rejection stages

V1 generic rejection stages:

```text
domain_validation
narrative_validation
epistemic_resolution
decision_resolution
```

`epistemic_resolution` is used only when a DecisionModel contract requires a unique resolved value and the valid epistemic state remains unknown or conflicted.

### 12.6 No automatic intervention powerset

V1 automatically generates or accepts only single interventions.

Multi-step `InterventionSequence` is explicitly deferred.

## 13. Generic analysis surface

Generic Analysis V1 is a typed research artifact, not a matrix-specific data format.

```text
NarrativeAnalysis
  baseline_trajectory
  mechanism_comparisons
  counterfactuals
```

### 13.1 AnalysisScope

```text
AnalysisScope
  decision_id
  tracked_state_cells
  tracked_agents
  snapshot_policy
```

Default scope derives from decision context targets and directly relevant event/claim paths.

Researchers may explicitly expand the scope, but the scope is immutable during one analysis.

### 13.2 Snapshot policy

V1 snapshots occur at:

- every relevant event time;
- every relevant claim time;
- the selected decision time.

If future receptions acquire independent timestamps, relevant reception times also become snapshot times.

### 13.3 EvolutionSnapshot

```text
EvolutionSnapshot
  logical_time
  triggers
  objective_cells
  agent_views
  decision_result?
```

```text
AgentEvolutionView
  direct_cells
  epistemic_cells
```

```text
EpistemicCellView
  status
  resolved_value?
  constraints
  evidence_kind?
  supporting_id?
  source_agent?
  evidence_logical_time?
  evidence_refs
```

### 13.4 First divergence

First divergence compares only canonical analysis fields such as:

```text
objective.<cell>.value
agent.<agent>.direct.<cell>.value
agent.<agent>.direct.<cell>.supporting_id
agent.<agent>.epistemic.<cell>.status
agent.<agent>.epistemic.<cell>.resolved_value
agent.<agent>.epistemic.<cell>.constraints
agent.<agent>.epistemic.<cell>.evidence_kind
agent.<agent>.epistemic.<cell>.supporting_id
agent.<agent>.epistemic.<cell>.source_agent
agent.<agent>.epistemic.<cell>.evidence_logical_time
decision.selected_action
```

It does not compare:

- source text;
- extractor confidence;
- compiler diagnostics;
- human notes;
- display labels;
- renderer metadata.

### 13.5 Mechanism comparison

Mechanism comparison is first-class:

```text
MechanismComparison
  decision_id
  models
  results
  pairwise_basis_differences
  pairwise_action_equal
  mechanism_uniqueness_claimed = False
```

The analysis may show:

```text
action_equal = True
basis_equal = False
```

without claiming either mechanism is the true hidden mechanism.

### 13.6 Matrix and graph renderers

Matrix, timeline, information-flow graph, and counterfactual diff are derived views over typed analysis records.

UI/rendering metadata is not canonical research semantics and does not affect analysis identity.

## 14. Identity and trust model

### 14.1 Source/compiler identity chain

```text
SourceBundleHash
→ CandidateBundleHash
→ ResolutionBundleHash
→ CompilationArtifactHash
```

This chain answers how a canonical narrative was produced.

### 14.2 Canonical semantic identity chain

```text
DomainSpecHash
+
GenericNarrativePayload
→ CanonicalNarrativeHash
```

This chain answers what executable semantics the narrative contains independent of source material.

Two different source/compiler lineages may legitimately produce the same canonical narrative hash.

### 14.3 DomainSpec identity

```text
DomainSpecIdentity
  domain_id
  domain_version
  schema_version
  declaration_hash
  semantic_hook_manifest
  content_hash
```

Each semantic hook identity includes at least:

```text
hook_name
declared_purpose
implementation_hash
```

Changing hook implementation changes DomainSpec identity even if declarations remain identical.

### 14.4 TrustedDomainPolicy

Formal/canonical runs may use a narrow external allow-list analogous in spirit to the existing trusted model execution policy:

```text
TrustedDomainPolicy
  allowed_domain_spec_hash
  expected_hook_implementation_hashes
```

A development domain may be unpinned, but resulting manifests must clearly report:

```text
trust_status = unpinned
```

A domain cannot self-authorize its own trusted identity.

### 14.5 Generic narrative identity exclusions

Canonical narrative identity excludes:

- source text;
- candidate confidence;
- extractor identity;
- human resolution notes;
- objective replay results;
- derived epistemic state;
- selected actions;
- analysis snapshots;
- counterfactual results;
- oracle/gold labels;
- renderer/UI state.

### 14.6 DecisionModel identity

DecisionModel identity binds at least:

```text
model_id
version
supported_decision_types
evidence_access
parameter_schema
implementation_hash
```

Evidence-access capability must be part of model identity.

### 14.7 Analysis artifact lineage

```text
NarrativeAnalysisArtifact
  canonical_narrative_hash
  domain_spec_hash
  decision_model_identity
  analysis_scope_hash
  intervention_manifest_hash
  analysis_payload_hash
  content_hash
```

If source compilation exists, `compilation_artifact_hash` is recorded as lineage but excluded from pure semantic analysis identity.

## 15. Legacy V2 compatibility and conformance

Narrative Testimony V2 remains unchanged.

The first Generic Engine must include a `LocationTestimonyDomainAdapter` that maps:

```text
Object                  → Entity(type=Object)
Location                → typed value/entity in the location domain
object.location          → StateVariable
RelocationEventV1       → Event(type=relocation)
DirectObservationV1     → Observation
LocationReportV2        → Claim(object.location == value)
ReportReceptionV2       → Reception
SearchDecisionV1        → Decision(type=search-location)
SearchActionV1          → typed action option
```

The adapter must not modify existing V1/V2 fixtures, schemas, runtime payloads, hashes, IDs, or APIs.

### 15.1 Required equivalence locks

For committed truthful and stale V2 cases, legacy and Generic paths must agree on:

- objective trajectory;
- direct trajectory;
- testimony/epistemic trajectory;
- evidence kind;
- supporting ID;
- source agent;
- selected decision action;
- remove-reception result;
- report-content-change result;
- remove-direct-observation rejection;
- remove-support-observation rejection;
- first divergence time;
- `action_changed`.

The following must remain exact:

- V1 fixture hashes;
- V1 runtime IDs;
- V2 fixture hashes;
- V2 runtime IDs;
- V1/V2 public API surfaces;
- all existing story tests.

## 16. Second conformance domain: Service Incident

Generic Engine V1 must include a small synthetic non-location domain to prove the engine is not location-specific under generic names.

Suggested domain:

```text
EntityType: Service
ValueType: HealthState = healthy | failed | recovered
StateVariable: service.health
```

Suggested event family:

```text
ServiceFailure
ServiceRecovery
```

Suggested story shape:

```text
e1: service fails
Alice observes e1
e2: service recovers
Bob does not observe e2
c1: Alice communicates service.health = recovered
Bob receives / does not receive c1
d1: Bob chooses a declared operational action
```

The domain is synthetic and is used only for generic conformance, not empirical systems reliability claims.

The concrete decision/action names should be chosen during implementation planning so that they have deterministic, domain-local semantics without introducing utility or probabilistic policy assumptions.

## 17. Public API boundary

Generic Engine APIs should live in a dedicated package namespace rather than expanding the legacy `narrative_dynamics.story` surface indefinitely.

The exact package path and export list will be fixed in the implementation plan, but the design requires separation among:

- generic IR types;
- DomainSpec types;
- compiler provenance types;
- replay types;
- DecisionModel contracts;
- analysis records;
- legacy story compatibility adapter.

Legacy V1/V2/V3 APIs remain source-compatible and identity-compatible.

Internal hook identities, trust policy internals, and compatibility implementation details should not leak into the package root unless an existing repository-wide trust API already requires them.

## 18. Error handling

All semantic errors fail closed.

Programming/type/runtime errors must not be silently converted into semantic counterfactual rejection records.

Only expected validation/resolution failures are represented as structured analysis rejections.

Compiler errors distinguish at least:

```text
incomplete
rejected
```

Canonical execution requires a fully validated narrative.

No canonical analysis is produced from an incomplete compilation.

## 19. Determinism and immutability

All public canonical records and analysis records are immutable.

Nested sequence/mapping fields must be defensively frozen or canonicalized.

`to_dict()` / serialization methods must be deterministic and JSON-serializable.

Stable identity must use the repository's typed canonical `stable_content_hash` semantics rather than ad hoc JSON hashing.

No generic semantic output may depend on dict insertion accidents, wall-clock time, process-local object IDs, or RNG in V1.

## 20. Lean boundary

Generic Narrative Engine V1 does not require new Lean core semantics.

Existing Lean proofs remain authoritative for their current formal boundaries, including information-path and testimony-related invariants.

Every Generic Engine increment must continue running the full existing Lean build and theorem suite.

Possible future formalization targets after Python generic semantics stabilize include:

- generic strict event ordering;
- observation visibility invariants;
- claim reception information-path invariants;
- no-objective-fallback properties.

The design intentionally avoids prematurely freezing the evolving Generic IR directly into Lean.

## 21. Implementation decomposition requirement

This architecture is larger than a single monolithic code drop.

The implementation plan should decompose V1 into strict RED→GREEN increments, likely along these boundaries:

1. generic value/type/state IR;
2. DomainSpec declaration + state-transition hook identity;
3. Generic narrative validation;
4. objective/direct/epistemic replay;
5. capability-limited DecisionModel context;
6. generic analysis trajectory;
7. generic interventions/counterfactual rejection;
8. Location/Testimony V2 compatibility adapter + equivalence vectors;
9. Service Incident conformance domain;
10. compiler provenance/candidate/resolution model;
11. deterministic compiler;
12. trust/identity lineage and public API finalization.

The final implementation plan may split these further but must not collapse the architectural boundaries above.

## 22. Testing requirements

At minimum, tests must cover:

- typed value and entity validation;
- event parameter typing;
- hook output legality;
- deterministic hook identity drift;
- strict logical time;
- objective replay independence from epistemic inputs;
- direct evidence no-objective-fallback;
- received versus unreceived claim behavior;
- supported stale/conflicting claims without truth coercion;
- `not_equals` remaining a constraint rather than invented resolution;
- conflicted epistemic status;
- DecisionModel capability isolation;
- explicit omniscient baseline isolation;
- action membership and exact resolution;
- single-variable intervention enforcement;
- rejection-stage correctness;
- first-divergence determinism;
- same action with different evidence basis;
- source/compiler provenance identity versus canonical semantic identity;
- DomainSpec declaration and hook implementation identities;
- V2 truthful/stale legacy equivalence;
- exact preservation of legacy V1/V2 identities;
- non-location Service Incident generic conformance;
- deterministic JSON serialization;
- package-root isolation rules;
- full Python regression;
- full Lean regression.

## 23. Non-goals for Generic Narrative Engine V1

V1 deliberately does not include:

- embedded LLM inference;
- automatic truth adjudication among sources;
- learned source trust;
- deception intent;
- recursive Theory of Mind;
- arbitrary logical theorem proving over propositions;
- arbitrary graph query language;
- utility optimization framework;
- probabilistic world transitions;
- stochastic DecisionModel API;
- automatic multi-intervention powersets;
- unrestricted branching story generation;
- UI/product visualization implementation;
- empirical domain calibration;
- complete OS sandboxing of semantic hooks;
- new Lean formalization of the entire Generic IR.

## 24. Success criteria

Generic Narrative Engine V1 is complete only when all of the following hold:

1. A typed non-location narrative can execute without importing or depending on location-specific story semantics.
2. Legacy V2 truthful/stale scenarios map through the compatibility adapter and produce required equivalence results.
3. Objective, direct, and communicated evidence remain structurally separated.
4. Decision models cannot read data outside their declared capability.
5. Counterfactuals recompute the full canonical pipeline from one typed upstream change.
6. Invalid counterfactuals remain rejected rather than repaired.
7. Source/compiler provenance and canonical semantic identity remain distinct.
8. Domain declaration and semantic hook implementation both affect DomainSpec identity.
9. Generic analysis can expose identical actions with different evidence bases without claiming a unique mechanism.
10. All existing V1/V2/V3 identities and APIs remain intact.
11. Full Python and existing Lean regression suites are green on the exact final feature head.

## 25. Final architecture statement

Generic Narrative Engine V1 turns the current canonical testimony research path into a domain-pluggable structured narrative engine while keeping the epistemic and scientific boundaries strict.

The intended architecture is:

```text
source material
    ↓
external candidate extractor
    ↓
source/candidate provenance
    ↓
deterministic compiler
    ↕
typed DomainSpec + narrow pure state hooks
    ↓
Generic typed event-sourced narrative IR
    ↓
canonical validation
    ↓
objective replay
    ↓
direct / testimony-aware epistemic replay
    ↓
capability-limited DecisionModel
    ↓
typed Generic Analysis
    ├─ baseline trajectory
    ├─ mechanism comparison
    └─ single-variable counterfactuals
```

The central invariant is that uncertainty, provenance, world truth, agent evidence, decision mechanism, and analysis result are different layers and must remain separately represented, validated, hashed, and auditable.
