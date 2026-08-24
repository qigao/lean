# Generic Narrative Engine V1 Design

Date: 2026-08-24

Status: approved architecture; implementation not started

Base research SHA: `82096098dda907aec04330b5ddf357ac6f753d2a`

Feature branch: `work/generic-narrative-engine-v1`

## 1. Purpose

Generic Narrative Engine V1 generalizes the existing location/testimony research path into a reusable structured narrative research engine without weakening the current provenance, fail-closed, deterministic replay, anti-leak, or mechanism-identification boundaries.

It studies canonical chains of the form:

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

The architectural goal is to make domain semantics pluggable while keeping information propagation, epistemic replay, counterfactual validation, provenance handling, and analysis semantics common across domains.

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

## 3. Approved architecture

The following are hard boundaries:

1. LLM/NLP extraction is external and non-canonical.
2. The compiler core is deterministic.
3. Domain semantics use a typed `DomainSpec` plus narrowly scoped pure state-transition hooks.
4. Canonical narrative semantics use a typed event-sourced IR.
5. Objective world state, direct agent evidence, and communicated claims are separate replay paths.
6. Belief-like views are derived, not authored as truth.
7. Decision facts and decision mechanisms are separate.
8. Decision models receive capability-limited immutable views, not the full scenario.
9. Counterfactuals modify upstream canonical history through typed single-variable interventions.
10. Derived world, epistemic, or decision state cannot be patched directly.
11. Analysis separates baseline trajectories, mechanism comparisons, and counterfactuals.
12. Source/compiler provenance identity is separate from canonical semantic identity.
13. Domain declarations and semantic-hook implementation identities are both bound into `DomainSpec` identity.
14. Legacy Narrative Testimony V2 remains unchanged and becomes the primary conformance reference domain.
15. A second synthetic non-location domain proves the generic path is not location semantics with renamed fields.
16. Generic Engine V1 does not require new Lean core semantics; all existing Lean proof gates remain mandatory regressions.

## 4. Dependency direction

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

The generic narrative package must not depend on runtime adapters to implement semantic replay.

## 5. Extraction versus compilation

### 5.1 External candidate extractor

LLM/NLP systems may propose candidate entities, events, propositions, claims, observations, receptions, decisions, and temporal relations.

The extractor is not a truth authority and cannot directly create canonical runtime semantics.

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

The compiler operates deterministically over:

- candidate records;
- resolution records;
- one selected `DomainSpec`;
- canonical validation rules.

The compiler must not:

- call an LLM;
- call the network;
- use RNG;
- depend on wall-clock time;
- silently select the highest-confidence candidate when semantic ambiguity remains;
- infer canonical truth from extraction confidence.

Compilation either yields a fully validated canonical narrative or yields no canonical narrative.

## 6. Source, candidate, and resolution provenance

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

A source span must be verifiable against the source document identity.

### 6.2 Candidate records

V1 candidate types:

```text
CandidateEntity
  proposed_id
  proposed_type
  source_refs
  confidence

CandidateEvent
  proposed_type
  logical_time?
  actor?
  arguments
  source_refs
  confidence

CandidateProposition
  subject
  state_variable
  relation
  value
  source_refs
  confidence

CandidateClaim
  speaker
  proposition
  support_refs?
  source_refs
  confidence

CandidateObservation
  agent
  observed_ref
  source_refs

CandidateReception
  recipient
  claim_ref
  source_refs

CandidateDecision
  actor
  logical_time?
  decision_type
  context
  candidate_actions
  source_refs
```

Confidence must be finite but never affects canonical semantic validation.

### 6.3 Resolution records

Candidate bindings have exactly three states:

```text
accepted
rejected
unresolved
```

A trusted resolver may:

- accept or reject a candidate;
- bind a candidate entity to a canonical entity;
- choose one value from explicit alternatives;
- supply a required typed field;
- select a temporal ordering.

Every resolution is recorded:

```text
ResolutionRecord
  candidate_id
  resolution_kind
  selected_value?
  resolver_identity
  reason?
```

Resolution records remain compiler provenance and never enter model-visible Generic IR.

### 6.4 Fail closed on semantic ambiguity

If any unresolved candidate is required to determine canonical semantics:

```text
status = incomplete
canonical_scenario = None
```

The compiler may not guess.

Conflicting communicated claims are allowed when represented as distinct claims. Extractor uncertainty about what a source means is ambiguity and is not a canonical conflicting claim.

### 6.5 CompilationResult

Successful compilation returns a versioned immutable artifact:

```text
CompilationResult
  status = canonical
  domain_id
  domain_version
  source_bundle_hash
  candidate_bundle_hash
  resolution_bundle_hash
  canonical_scenario
  canonical_hash
  diagnostics
```

Incomplete or rejected compilation returns:

```text
CompilationResult
  status = incomplete | rejected
  canonical_scenario = None
  canonical_hash = None
  diagnostics
```

No analysis may execute from an incomplete/rejected result.

## 7. Generic typed event-sourced IR

Generic IR stores canonical typed facts and event history. Derived world-state snapshots are never authored as input truth.

### 7.1 Entity types and entities

```text
EntityTypeSpec
  name

Entity
  id
  type
```

Entity IDs are unique inside one narrative.

### 7.2 Value types

```text
ValueTypeSpec
  name
  kind
  allowed_values?
  referenced_entity_type?
```

V1 kinds:

```text
enum
bool
integer
text
entity_ref
```

Arbitrary Python objects are invalid canonical values.

### 7.3 State variables

```text
StateVariableSpec
  name
  subject_type
  value_type
  cardinality
```

V1 supports only `cardinality = one`.

A state cell is `(subject_id, state_variable)`.

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

Canonical `logical_time` is a strict total order for state-changing events, claims, and decisions. Coarse source timestamps remain provenance; unresolved ordering required for semantics makes compilation incomplete. Partial-order/simultaneous-event semantics are a V1 non-goal.

### 7.5 Proposition

V1 propositions are restricted typed state assertions:

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

V1 excludes arbitrary predicate code, quantifiers, free-form formulas, and user-defined logical operators.

An `equals` proposition may resolve a current value. A `not_equals` proposition remains a constraint and does not cause the engine to invent a unique value.

### 7.6 Observation

```text
Observation
  event_ref
  agent_id
  channel
```

Direct evidence derives only from the typed state delta produced by the observed event.

Observing an event never grants access to all objective world state.

### 7.7 Claim

```text
Claim
  id
  logical_time
  speaker_id
  proposition
  support_refs
  claim_type = state_claim
```

`support_refs` is non-empty in V1 and may reference only earlier canonical events or earlier canonical claims.

Generic provenance validation requires the support to have been available to the speaker before the new claim:

- event support requires a direct observation of that event by the speaker;
- prior-claim support requires reception of that prior claim by the speaker;
- every support item must precede the new claim in canonical logical time.

Claim support establishes an information path, not claim truth. A supported claim may disagree with objective world state.

### 7.8 Reception

```text
Reception
  claim_ref
  recipient_id
  channel
```

V1 receptions do not have independent logical time; reception availability is tied to the supported claim time, matching the existing V2 boundary. Independent delayed-delivery semantics are deferred.

Reception controls whether a claim enters the recipient's epistemic evidence history.

### 7.9 Decision and actions

Canonical IR stores only decision facts and allowed actions:

```text
Decision
  id
  logical_time
  actor_id
  decision_type
  context_targets
  action_options
```

Canonical Decision never contains `decision_model_ref`.

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

The declarative portion is canonicalizable and content-hashed.

### 8.2 Pure semantic hooks

V1 semantic hooks are limited to objective state transitions:

```text
apply_event(prior_state, typed_event) -> StateDelta
```

V1 `StateDelta` operations:

```text
set(subject, state_variable, value)
clear(subject, state_variable)
```

A direct observation of `clear` produces evidence that the state cell became unresolved/absent, with the event as provenance; it does not synthesize an `equals` or `not_equals` proposition.

Hooks may not:

- use files;
- use network access;
- use wall-clock time;
- use RNG;
- call an LLM;
- mutate global state;
- create entities/events/claims;
- decide observations;
- author beliefs;
- select actions;
- return arbitrary Python objects.

A hook answers only:

> What deterministic objective state delta does this already-valid event produce?

Python purity is not treated as mathematically proven. Trust is based on immutable contracts, determinism tests, and implementation identity.

### 8.3 Generic semantics domains cannot redefine

A domain cannot override:

- canonical time ordering;
- observation ownership;
- claim speaker semantics;
- claim/reception information-path rules;
- objective versus agent-evidence separation;
- support provenance versus truth separation;
- no-objective-fallback behavior;
- counterfactual fail-closed reconstruction;
- first-divergence semantics.

## 9. Four validation layers

### 9.1 Candidate validation

Checks source/extractor structural integrity only:

- source span exists;
- source hashes match;
- confidence is finite;
- candidate references are structurally valid.

It never certifies truth.

### 9.2 Compiler validation

Checks:

- ontology binding uniqueness;
- required field resolution;
- deterministic temporal ordering;
- typed value compatibility;
- absence of semantic ambiguity.

### 9.3 Canonical narrative validation

Ignores LLM confidence and source prose. It checks:

- `DomainSpec` identity;
- entity/state/event typing;
- event argument typing;
- event transition legality;
- global canonical ordering;
- observation references;
- claim proposition typing;
- support accessibility/provenance;
- reception references;
- decision/action typing;
- Generic Engine invariants.

### 9.4 Analysis validation

Checks:

- analysis scope;
- DecisionModel evidence-access contract;
- model identity;
- intervention validity;
- complete revalidation of each counterfactual candidate narrative.

No validation layer may be used to bypass a later one.

## 10. Objective and epistemic replay

### 10.1 Objective state

```text
WorldState
  StateCell -> CanonicalValue
```

Objective replay applies valid event deltas in canonical order.

It never reads observations, claims, receptions, or decisions.

### 10.2 Direct evidence

An agent observing an event receives evidence only for state cells changed by that event.

Conceptual evidence record:

```text
EpistemicEvidence
  subject_id
  state_variable
  assertion_kind
  value?
  evidence_kind
  supporting_id
  source_agent
  logical_time
  provenance_refs
```

`assertion_kind` is `equals`, `not_equals`, or internal `cleared`. Canonical Claim propositions may use only `equals` and `not_equals`.

For direct perception:

```text
evidence_kind = direct_perception
supporting_id = event.id
source_agent = querying agent
```

No objective fallback is permitted.

### 10.3 Communicated evidence

A received Claim contributes testimony evidence:

```text
evidence_kind = testimony
supporting_id = claim.id
source_agent = claim.speaker
```

Claim truth is never inferred from claim support.

### 10.4 Evidence history and views

Generic replay retains full evidence history:

```text
EpistemicState
  evidence_history
  latest_evidence_by_state_cell
  resolved_values
  constraints
```

Standard replay views:

```text
objective_state(story, at_time)
direct_state(story, agent, at_time)
epistemic_state(story, agent, at_time)
```

Domains cannot redefine these access paths.

### 10.5 Conflict handling

Conflicting evidence remains explicit:

```text
status = resolved | unknown | conflicted
```

The Generic Engine never chooses which source is more trustworthy. Decision models may implement explicit conflict-handling strategies.

## 11. Decision Model architecture

### 11.1 Separation

`Decision` is a narrative fact. `DecisionModel` is an experiment mechanism.

The same canonical narrative executes under multiple competing DecisionModels without changing narrative identity.

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
- `omniscient`: objective state, explicit baseline only;
- `custom_epistemic`: actor evidence history/provenance, never objective state.

The engine constructs an immutable capability-limited context:

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

DecisionModel never receives the raw scenario.

### 11.4 DecisionResult

```text
DecisionResult
  model_id
  selected_action
  action_scores
  policy
  basis
```

`basis` records canonical evidence/state-cell references needed to audit the decision.

The selected action must be exactly one declared action option.

### 11.5 V1 determinism

Generic DecisionModel V1 is deterministic. Probabilistic choice, seeded sampling, calibration, and fitted parameters are deferred and may later reuse existing runtime/calibration infrastructure.

### 11.6 Mechanism-identification boundary

Behavioral agreement establishes only compatibility under the declared scenario and model contract. The engine never emits `true_mechanism`, `unique_cause`, or equivalent hidden-mechanism claims.

## 12. Generic counterfactual architecture

### 12.1 Principle

V1 interventions are typed, upstream, single-variable changes to canonical history. Derived world, epistemic, and decision state are always recomputed.

### 12.2 V1 intervention kinds

```text
remove_event
change_event_argument
remove_observation
change_claim_value
remove_reception
```

`change_event_argument` may modify only an `EventTypeSpec.intervenable_parameters` field.

`change_claim_value` changes only proposition value while holding subject, state variable, relation, speaker, support, and reception structure fixed.

### 12.3 Forbidden mutations

V1 forbids:

- arbitrary JSON patching;
- replacing a whole event;
- changing actor/time/payload together;
- injecting undeclared entities;
- modifying derived world/epistemic state;
- modifying `DecisionContext` directly;
- changing DecisionModel under the name of a narrative counterfactual.

Changing DecisionModel is mechanism comparison, not narrative intervention.

### 12.4 Evaluation pipeline

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

Rejected counterfactuals are retained and never repaired into fake valid trajectories.

### 12.5 Rejection stages

```text
domain_validation
narrative_validation
epistemic_resolution
decision_resolution
```

`epistemic_resolution` is used only when a DecisionModel requires unique resolution but the valid epistemic state remains unknown/conflicted.

### 12.6 No powerset

V1 accepts/generates single interventions only. Multi-step `InterventionSequence` is deferred.

## 13. Generic analysis surface

Generic Analysis V1 is a typed research artifact, not a matrix-specific data format:

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

Default scope derives from decision context targets and directly relevant event/claim paths. Researchers may explicitly expand it, but scope is immutable during one analysis.

### 13.2 Snapshot policy

V1 snapshots occur at every relevant event time, claim time, and selected decision time.

Because V1 receptions have no independent timestamp, they are visible at their Claim logical time.

### 13.3 Snapshot records

```text
EvolutionSnapshot
  logical_time
  triggers
  objective_cells
  agent_views
  decision_result?

AgentEvolutionView
  direct_cells
  epistemic_cells

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

First divergence compares only canonical analysis fields, including objective state-cell values, direct/epistemic cell values/status/provenance, and selected action.

It excludes source text, extractor confidence, compiler diagnostics, human notes, display labels, and renderer metadata.

### 13.5 Mechanism comparison

```text
MechanismComparison
  decision_id
  models
  results
  pairwise_basis_differences
  pairwise_action_equal
  mechanism_uniqueness_claimed = False
```

The analysis may report `action_equal = True` and `basis_equal = False` without identifying a true hidden mechanism.

### 13.6 Renderers

Matrix, timeline, information-flow graph, and counterfactual diff are derived views over typed analysis records. Rendering metadata is non-canonical and does not affect analysis identity.

## 14. Identity and trust model

### 14.1 Source/compiler chain

```text
SourceBundleHash
→ CandidateBundleHash
→ ResolutionBundleHash
→ CompilationArtifactHash
```

This answers how a canonical narrative was produced.

### 14.2 Semantic chain

```text
DomainSpecHash
+
GenericNarrativePayload
→ CanonicalNarrativeHash
```

Different source/compiler lineages may produce the same canonical semantic hash.

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

Each hook identity binds:

```text
hook_name
declared_purpose
implementation_hash
```

Changing hook implementation changes DomainSpec identity even if declarations remain identical.

### 14.4 TrustedDomainPolicy

V1 implements an external allow-list for trusted/canonical domain execution:

```text
TrustedDomainPolicy
  allowed_domain_spec_hash
  expected_hook_implementation_hashes
```

A development domain may be unpinned, but its manifest must state `trust_status = unpinned`. A domain cannot self-authorize trust.

### 14.5 Canonical narrative identity exclusions

Canonical identity excludes source text, candidate confidence, extractor identity, human resolution notes, replay results, derived epistemic state, selected action, analysis/counterfactual results, oracle/gold labels, and UI state.

### 14.6 DecisionModel identity

DecisionModel identity binds:

```text
model_id
version
supported_decision_types
evidence_access
parameter_schema
implementation_hash
```

Evidence-access capability is part of model identity.

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

`LocationTestimonyDomainAdapter` maps:

```text
Object                  → Entity(type=Object)
Location                → typed location value/entity
object.location          → StateVariable
RelocationEventV1       → Event(type=relocation)
DirectObservationV1     → Observation
LocationReportV2        → Claim(object.location == value)
ReportReceptionV2       → Reception
SearchDecisionV1        → Decision(type=search-location)
SearchActionV1          → typed action option
```

The adapter does not modify existing V1/V2 fixtures, schemas, runtime payloads, hashes, IDs, or APIs.

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

Exact legacy identities that must not change include V1/V2 fixture hashes, runtime IDs, public API surfaces, and all existing story regressions.

## 16. Second conformance domain: Service Incident

Generic Engine V1 includes one explicit non-location synthetic conformance domain.

### 16.1 Domain

```text
EntityType: Service
ValueType: HealthState = healthy | failed | recovered
StateVariable: service.health
```

Event types:

```text
ServiceFailure
  health: healthy → failed

ServiceRecovery
  health: failed → recovered
```

Decision type:

```text
service-response
```

Action options:

```text
restart_service
leave_running
```

The deterministic reference decision rule is:

```text
resolved health == failed    → restart_service
resolved health == recovered → leave_running
otherwise                    → decision_resolution rejection
```

### 16.2 Matched synthetic pair

Both scenarios share the same objective events, observations, speaker/support path, reception, decision, and actions:

```text
e1 @ 1: service fails
         Bob directly observes e1

e2 @ 2: service recovers
         Alice directly observes e2

c1 @ 3: Alice communicates one service.health proposition
         support = e2
         Bob receives c1

d1 @ 4: Bob decides restart_service / leave_running
```

The pair differs only in `c1` proposition value:

```text
recovered-claim: service.health == recovered
stale-claim:     service.health == failed
```

The stale claim remains admissible because support provenance is not a truth predicate.

Expected model contrast:

```text
                    recovered-claim   stale-claim
direct-only         restart_service   restart_service
omniscient           leave_running     leave_running
epistemic            leave_running     restart_service
```

This domain is synthetic and supports only generic conformance claims, not empirical systems-reliability conclusions.

## 17. Public package boundary

Generic Engine V1 lives under the dedicated namespace:

```text
narrative_dynamics.narrative
```

Required module boundaries:

```text
narrative_dynamics/narrative/
  __init__.py
  ir.py
  domain.py
  compiler.py
  replay.py
  decision.py
  interventions.py
  analysis.py
  trust.py
  compat_story_v2.py
  domains/
    __init__.py
    service_incident.py
```

The exact public symbol list will be enumerated in the implementation plan from the types/functions fixed by this design, but V1 has two API constraints:

1. canonical Generic Engine APIs are exported from `narrative_dynamics.narrative`;
2. the package root `narrative_dynamics` gains no Generic Narrative Engine exports in V1.

Legacy `narrative_dynamics.story` V1/V2/V3 APIs remain unchanged.

Internal hook identities, trust-policy helpers, and compatibility details do not leak into the package root.

## 18. Error handling

All semantic errors fail closed.

Programming/type/runtime errors are not silently converted into semantic counterfactual rejection records.

Only expected validation/resolution failures become structured rejections.

Compiler results distinguish `canonical`, `incomplete`, and `rejected`.

No canonical analysis is produced from an incomplete/rejected compilation.

## 19. Determinism and immutability

All public canonical and analysis records are immutable. Nested sequence/mapping fields are defensively frozen/canonicalized.

Serialization is deterministic and JSON-serializable.

Stable identity uses the repository's typed canonical `stable_content_hash` semantics, not ad hoc JSON hashing.

No Generic semantic output depends on dict insertion accidents, wall-clock time, process-local object IDs, or RNG in V1.

## 20. Lean boundary

Generic Narrative Engine V1 does not add new Lean core semantics.

Existing Lean proofs remain authoritative for current formal boundaries, including information-path and testimony invariants.

Every Generic Engine increment continues running the full existing Lean build and theorem suite.

Possible later formalization targets, after Python semantics stabilize, include generic event ordering, observation visibility, claim reception information paths, and no-objective-fallback properties.

## 21. Implementation decomposition requirement

The implementation plan must use strict RED→GREEN increments and preserve architecture boundaries. Expected increments are:

1. generic value/type/state IR;
2. DomainSpec declaration + state-transition hook identity;
3. Generic narrative validation;
4. objective/direct/epistemic replay;
5. capability-limited DecisionModel context;
6. generic analysis trajectory;
7. generic interventions/counterfactual rejection;
8. Location/Testimony V2 compatibility adapter + equivalence vectors;
9. Service Incident conformance domain;
10. compiler source/candidate/resolution records;
11. deterministic compiler;
12. trust/identity lineage and public API finalization.

The plan may split these further but must not collapse the approved boundaries.

## 22. Testing requirements

At minimum, tests cover:

- typed value/entity validation;
- event parameter typing;
- hook output legality;
- hook implementation identity drift;
- strict total canonical ordering;
- objective replay independence from epistemic inputs;
- direct evidence no-objective-fallback;
- received versus unreceived claims;
- claim support accessibility for event and prior-claim support;
- supported stale/conflicting claims without truth coercion;
- `not_equals` remaining a constraint;
- explicit `conflicted` epistemic state;
- DecisionModel capability isolation;
- explicit omniscient-baseline isolation;
- exact action membership/resolution;
- single-variable intervention enforcement;
- rejection-stage correctness;
- first-divergence determinism;
- same action with different evidence basis;
- source/compiler provenance identity versus semantic identity;
- DomainSpec declaration and hook implementation identities;
- V2 truthful/stale legacy equivalence;
- exact preservation of legacy V1/V2 identities;
- non-location Service Incident matched-pair conformance;
- deterministic JSON serialization;
- package-root isolation;
- full Python regression;
- full Lean regression.

## 23. Non-goals for V1

V1 deliberately excludes:

- embedded LLM inference;
- automatic truth adjudication among sources;
- learned source trust;
- deception intent;
- recursive Theory of Mind;
- arbitrary logical theorem proving;
- arbitrary graph query language;
- utility optimization framework;
- probabilistic world transitions;
- stochastic DecisionModel API;
- automatic multi-intervention powersets;
- unrestricted branching story generation;
- partial-order/simultaneous-event execution semantics;
- independent reception timestamps/delivery delays;
- UI/product visualization implementation;
- empirical domain calibration;
- complete OS sandboxing of semantic hooks;
- new Lean formalization of the entire Generic IR.

## 24. Success criteria

Generic Narrative Engine V1 is complete only when all of the following hold:

1. A typed non-location narrative executes without location-specific story semantics.
2. Legacy V2 truthful/stale scenarios map through the compatibility adapter and satisfy required equivalence locks.
3. Objective, direct, and communicated evidence remain structurally separated.
4. Decision models cannot read data outside their declared capability.
5. Counterfactuals recompute the full canonical pipeline from one typed upstream change.
6. Invalid counterfactuals remain rejected rather than repaired.
7. Source/compiler provenance and canonical semantic identity remain distinct.
8. Domain declarations and semantic hook implementation both affect DomainSpec identity.
9. Generic analysis exposes identical actions with different evidence bases without claiming a unique mechanism.
10. The Service Incident matched pair produces the declared three-model contrast.
11. Existing V1/V2/V3 identities and APIs remain intact.
12. Full Python and existing Lean regression suites are green on the exact final feature head.

## 25. Final architecture statement

Generic Narrative Engine V1 turns the current canonical testimony research path into a domain-pluggable structured narrative engine while keeping epistemic and scientific boundaries strict:

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

The central invariant is that uncertainty, provenance, world truth, agent evidence, decision mechanism, and analysis result are different layers and remain separately represented, validated, hashed, and auditable.
