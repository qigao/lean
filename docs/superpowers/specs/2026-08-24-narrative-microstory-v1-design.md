# Narrative Microstory V1 — Design

Date: 2026-08-24
Status: approved design for implementation planning
Target branch: `proof/narrative-dynamics-v0`
Implementation scope: Lean semantic extension + Python canonical story/runtime integration

## Purpose

This increment introduces the first non-prison narrative domain and the first explicit story-input layer for Narrative Dynamics.

The goal is to execute one minimal but scientifically discriminating chain:

```text
story
  → canonical events
  → objective world state
  → agent-specific epistemic state
  → decision context
  → competing action predictions
```

The V1 domain is deliberately tiny: two agents, one object, two locations, relocation events, direct perception, and one search decision. The primary test is false belief. The objective world may contain a later relocation that the decision actor did not observe; a belief-sensitive model must continue to predict from the actor's last admissible information rather than from narrator-level truth.

This increment is not a general natural-language story parser. The source story and the canonical annotation coexist in the fixture, but models consume only the canonical runtime projection.

## Existing boundaries that remain authoritative

The current repository already provides:

- `WorldGraph` with temporal causal and information-reachability invariants;
- proof-carrying observation admission and the No Epistemic Teleportation boundary;
- private per-agent belief/provenance graphs;
- symbolic belief support separated from Bayesian confidence;
- finite softmax order-preservation results;
- immutable Python scenarios, traces, manifests, and content hashes;
- `SimulatorModel`, `ModelRun`, `SimulationRunner`, model factories, subprocess execution, schema enforcement, and attestation;
- metric/loss/model-comparison infrastructure.

This work extends those boundaries rather than replacing them. In particular, V1 does not invent a second observation semantics or a second execution runtime.

## Scientific question

V1 asks one narrow question:

> When objective reality changes without entering an agent's information boundary, should the predicted action follow objective reality or the agent's last supported subjective state?

The two competing models answer differently:

- `agent-belief-search` predicts from the decision actor's latest admissible subjective object location;
- `omniscient-search` predicts from the objective latest object location.

A false-belief story separates them. A matched informed counterfactual makes them converge again.

This is the first repository-level contrast where:

```text
narrator knows P ≠ character knows P ≠ character acts on P
```

can be represented and tested end to end.

## V1 authored microstories

### Case A: false belief

Canonical source text:

> Alice and Bob are in a room. While Bob is watching, Alice puts a key in the drawer. Bob leaves the room. While Bob is away, Alice moves the key from the drawer to the box. Bob returns and wants to find the key. Where will Bob search first: the drawer or the box?

Canonical event history:

```text
e1 @ t=1: Alice relocates key from unknown → drawer
e2 @ t=2: Alice relocates key from drawer → box
```

Bob directly observes `e1` and does not observe `e2`.

At the decision:

```text
objective(key)      = box
subjective(Bob,key) = drawer
```

Required model ranking:

```text
agent-belief-search: search_drawer > search_box
omniscient-search:   search_box > search_drawer
```

### Case B: informed belief

Canonical source text:

> Alice and Bob are in a room. While Bob is watching, Alice puts a key in the drawer. Alice later moves the key from the drawer to the box, and Bob sees the move. Bob wants to find the key. Where will Bob search first: the drawer or the box?

The objective events are intentionally identical to Case A. The only discriminating semantic change is that Bob also directly observes `e2`.

At the decision:

```text
objective(key)      = box
subjective(Bob,key) = box
```

Required model ranking:

```text
agent-belief-search: search_box > search_drawer
omniscient-search:   search_box > search_drawer
```

The pair therefore isolates information availability while holding objective event history fixed.

## Canonical story schema

V1 introduces an immutable `NarrativeCaseV1` fixture representation. It is authored data, not model output.

### Entity surface

V1 admits exactly three entity families:

```text
Agent
Object
Location
```

The committed fixtures use:

```text
agents    = alice, bob
objects   = key
locations = drawer, box
```

Identifiers are non-empty canonical strings, unique inside their entity family. V1 does not infer entity aliases or coreference.

### Relocation event

The only state-changing event kind in V1 is:

```text
relocate_object
```

Each event contains:

```text
id
logical time
actor
object
from_location : optional Location
 to_location   : Location
```

Requirements:

- event IDs are unique;
- logical times are strictly increasing in fixture order;
- actor/object/location references must resolve to declared entities;
- `to_location` must be declared;
- `from_location`, when present, must equal the object's current objective location immediately before the event;
- `from_location = null` is allowed only when the object has no prior objective location in the history;
- one event always establishes the object's new objective location at `to_location`.

These validation rules make the authored event history itself internally checkable instead of trusting a later oracle field.

### Observation

A V1 observation contains:

```text
event
agent
channel = direct_perception
```

Requirements:

- referenced event and agent must exist;
- duplicate `(event, agent)` observations are rejected;
- V1 supports no other channel.

`direct_perception` is semantic, not decorative metadata: when projected to the Lean world graph it supplies the direct event→agent information connection required for observation admissibility.

No implicit rule says an actor automatically observes their own action. Every model-relevant observation is explicit in canonical data.

### Decision

V1 admits one decision kind:

```text
search_object
```

A decision contains:

```text
id
time
actor
object
actions[]
```

Each action contains:

```text
id
location
```

Requirements:

- decision ID is non-empty;
- decision time is strictly later than every preceding story event;
- decision actor/object/action locations resolve to declared entities;
- action IDs are unique;
- action locations are unique;
- the decision contains at least two actions;
- the target object has a known objective latest location;
- the decision actor has at least one directly observed relocation of the target object before the decision, so the deterministic V1 belief model has a known subjective location.

The committed pair uses exactly `search_drawer` and `search_box`.

## Fixture identity and oracle separation

The committed fixture stores both authored material and expected semantic results. A representative top-level shape is:

```json
{
  "schema_version": 1,
  "name": "key-location-false-belief",
  "version": "1.0.0",
  "source": {
    "kind": "authored_microstory",
    "language": "en"
  },
  "provenance": {
    "synthetic": true,
    "empirical_human_data": false,
    "population_representative": false
  },
  "source_text": "...",
  "entities": {
    "agents": ["alice", "bob"],
    "objects": ["key"],
    "locations": ["drawer", "box"]
  },
  "events": ["...canonical relocation objects..."],
  "observations": ["...canonical observation objects..."],
  "decision": {"...": "..."},
  "oracle": {
    "objective_location": "box",
    "actor_subjective_location": "drawer",
    "agent_belief_ranking": ["search_drawer", "search_box"],
    "omniscient_ranking": ["search_box", "search_drawer"]
  },
  "content_hash": "sha256:..."
}
```

The declared fixture `content_hash` is recomputed from all canonical fixture content except the declared hash itself. The fixture is immutable after construction, round-trips through strict JSON, and rejects a mismatched declared hash.

The fixture oracle is test metadata. It must never be exposed to a model.

## Runtime projection and anti-leakage boundary

Fixture and runtime scenario are distinct values.

A pure projection:

```text
NarrativeCaseV1 → NarrativeScenarioV1 → Scenario
```

constructs the only payload visible to a simulator.

The runtime payload contains exactly:

```text
entities
events
observations
decision
```

It excludes:

- `oracle`;
- `source_text`;
- any observed continuation after the decision;
- any gold selected action;
- fixture provenance fields that could encode the answer.

The projection is deterministic and itself covered by tests. The resulting ordinary runtime `Scenario` therefore receives the repository's existing recursive immutability and stable identity behavior.

V1 intentionally withholds the raw source text from the models. Natural-language parsing becomes a later upstream stage whose output can be checked against this manually authored canonical oracle.

## Story-state replay semantics

V1 introduces temporal object-location fluents rather than extending structural graph edges with mutable location state.

### Objective projection

`objective_state(events)` replays every valid relocation in time order.

For one object, each relocation replaces the current location with `to_location`. Historical facts remain historical; later movement does not make the earlier event false.

The latest objective location is the `to_location` of the most recent relocation for the object.

### Subjective projection

`subjective_state(events, observations, agent)` replays only relocations that the agent directly observed.

The latest subjective location is the `to_location` of the most recent observed relocation for that object.

The subjective replay does not require its prior subjective location to equal an observed relocation's `from_location`. Directly seeing a relocation is sufficient to learn its destination even if the observer lacked the earlier state. Objective `from_location` consistency is checked against objective history, not against an agent's private history.

This keeps observation semantics distinct from world-history consistency.

### Time-indexed facts

V1 treats relocation-derived propositions as time-indexed historical facts, conceptually:

```text
objectAt(key, drawer, t1)
objectAt(key, box, t2)
```

A later fact does not erase the existence of the earlier historical fact. Decision models query the latest supported state at or before the decision time.

This distinction is required for future narrative reasoning about what happened previously versus what an agent currently believes.

## Lean semantic extension

### New module

Add:

```text
NarrativeDynamics/Core/StoryState.lean
```

The module is generic over object and location types. It must not mention Alice, Bob, key, drawer, or box.

Its responsibility is only temporal fluent projection over relocation events and the connection from admissible observation to subjective projection.

A representative semantic surface is:

```text
StoryRelocation Object Location
objectiveLocation
subjectiveLocation
```

Exact implementation types may be refined in the implementation plan, but the semantics above are fixed.

### Reuse of existing epistemic kernel

The new module must use the existing `WorldGraph` observation boundary.

For a canonical direct observation, the story-to-Lean instantiation provides an event→agent information path. Existing `WorldInvariant` therefore continues to require that every recorded observation has information reachability.

The existing `ObservationEvidenceAdmissible` boundary remains authoritative: an event that has no information path to the agent cannot be admitted into the agent's private evidence state.

StoryState adds the missing fluent theorem layer; it does not redefine observation admission.

### Provenance interpretation

Relocation events may be represented as event proofs whose derived typed concept fact is the time-indexed `objectAt` proposition. The existing provenance invariant remains authoritative: derived facts must match the rule output and exact ordered parent facts.

The V1 design does not require adding a new `NodeKind` or mutating the meaning of existing structural `TypedEdge.locatedAt`, which currently describes an agent-location structural edge rather than an object fluent.

### Required Lean properties

The implementation must establish at least these semantic properties with executable theorem tests:

1. replaying a valid relocation history returns the latest objective destination;
2. an unobserved relocation can change objective location without changing an agent's subjective location;
3. an observed later relocation updates the agent's latest subjective location;
4. objective and subjective locations can differ in the false-belief case;
5. adding the missing admissible observation makes the informed actor's subjective location agree with objective location;
6. an event unavailable through the information graph cannot be used as the observation premise that updates the agent projection;
7. the concrete false-belief instantiation yields opposite strict action rankings for belief-sensitive versus omniscient decision semantics;
8. the informed counterfactual yields the same `search_box` ranking for both semantics.

The implementation plan may split these between generic core theorems and concrete theorem tests, but the behavioral claims may not be weakened.

## Python module architecture

### `narrative_dynamics/story/schema.py`

Owns canonical story values and fixture validation:

- `NarrativeCaseV1`;
- entity declarations;
- relocation events;
- direct observations;
- search decision/actions;
- oracle value;
- strict JSON load/dump;
- fixture content hash verification.

This module performs structural and objective-history validation only. It does not run a behavioral model.

### `narrative_dynamics/story/replay.py`

Owns pure semantic projection:

```text
objective_state(...)
subjective_state(..., agent)
latest_object_location(...)
```

It has no dependency on the simulation runner, metric/loss code, or model comparison.

### `narrative_dynamics/story/scenario.py`

Owns the anti-leakage projection:

```text
NarrativeCaseV1 → NarrativeScenarioV1 → contracts.Scenario
```

It is the only supported path from a V1 fixture into a story model.

### `narrative_dynamics/story/metrics.py`

Owns a common extractor for both story models. It returns the same categorical coordinates for both adapters:

```text
choice.search_box
choice.search_drawer
```

The extractor validates that the emitted policy is complete, finite, non-negative, and normalized.

### `narrative_dynamics/adapters/story_belief_search.py`

Owns `agent-belief-search`.

The model accepts exactly an empty parameter mapping in V1. It ignores RNG for its deterministic policy but still runs through `SimulationRunner`.

Algorithm:

1. derive the decision actor's subjective state from admissible canonical observations;
2. find the latest subjective location of the target object;
3. assign score `1` to the action targeting that location and `0` to every other action;
4. emit a deterministic normalized one-hot policy;
5. select the maximum-score action;
6. emit an epistemic-basis record identifying the subjective location and supporting observed event ID.

Unknown subjective target location is rejected; the V1 scenario contract is designed to prevent it.

### `narrative_dynamics/adapters/story_omniscient_search.py`

Owns `omniscient-search`.

It also accepts exactly an empty parameter mapping and uses no fitted free parameter.

Algorithm:

1. replay all objective events;
2. find the target object's latest objective location;
3. assign score `1` to the action targeting that location and `0` to every other action;
4. emit the same policy/output contract as the belief model;
5. identify the basis explicitly as objective-world state.

The baseline is intentionally epistemically unrealistic. Its purpose is to test whether agent-specific information adds explanatory content beyond narrator-level state.

## Prediction output contract

Both models emit the same result shape:

```text
epistemic_basis
action_scores
policy
selected_action
```

`epistemic_basis` must identify:

- basis kind: `subjective` or `objective`;
- target object;
- resolved location;
- supporting latest relocation event ID.

`action_scores` and `policy` must contain exactly the declared decision action IDs.

The deterministic V1 policy contains one action with probability `1.0` and every rival with probability `0.0`.

No model may read or emit the fixture oracle as its prediction basis.

## Why V1 is parameter-free

V1 tests semantic correctness, not behavioral noise.

Adding a free inverse-temperature parameter before the objective/subjective distinction is proven would make semantic bugs harder to diagnose and introduce calibration machinery that is unnecessary for the first story-domain milestone.

The current runtime accepts an empty parameter mapping, and the repository already has finite-softmax semantics that can be applied later without changing score ordering. A later V2 can add `beta > 0` when the research question becomes population choice frequency rather than deterministic false-belief competence.

## Tests and RED obligations

Implementation follows strict RED → GREEN. Each production increment must begin with a failing obligation and keep unrelated existing tests green.

### Python schema RED

Tests must initially fail because the story schema does not exist, then cover:

- immutable canonical values;
- strict JSON round-trip;
- stable content hash and rejection of a forged declared hash;
- duplicate IDs;
- undeclared entity references;
- non-increasing event time;
- invalid objective `from_location` continuity;
- invalid observation references or duplicate observations;
- invalid decision action identities/locations;
- decision time not after events;
- decision actor with no observed location for the target object.

### Python replay RED

Require exact projections:

```text
false belief:
  objective = box
  bob subjective = drawer

informed:
  objective = box
  bob subjective = box
```

Adding an unobserved relocation must not change Bob's subjective state. Adding Bob's direct observation of that relocation must change it.

### Python anti-leakage RED

Require that runtime scenario payload contains no:

```text
oracle
source_text
observed_choice
gold action
post-decision continuation
```

The model-facing scenario hash must therefore be independent of test oracle fields.

### Python model RED

Require:

```text
false belief:
  belief model     → search_drawer
  omniscient model → search_box

informed:
  belief model     → search_box
  omniscient model → search_box
```

Both models must reject non-empty parameters and malformed story scenarios.

### Trusted-runtime RED

Both story models must run through the canonical `SimulationRunner` rather than being called directly as the production acceptance path.

The tests must verify stable traces, manifest attachment, canonical empty parameter identity, common metric extraction, and seeded replay compatibility even though the deterministic model ignores RNG.

If production source factories/contracts are added in V1, they must use the repository's existing schema/pinning/isolation machinery instead of bespoke trust checks.

### Lean RED

Add concrete theorem obligations before `StoryState.lean` is implemented. The deliberate RED should arise from missing story-state definitions/theorems while the existing Lean library and unrelated theorem tests stay green.

## Committed fixtures

Add exactly two V1 authored fixtures:

```text
fixtures/stories/key_location_false_belief_v1.json
fixtures/stories/key_location_informed_v1.json
```

Both are synthetic authored examples, not empirical observations.

Their objective event histories are identical. They differ in Bob's observation of `e2` and in oracle expectations implied by that observation.

No train/selection/final partitioning is introduced in this increment.

## Public API

The canonical story schema/projection surface should be available from `narrative_dynamics.story`.

Model-specific adapters remain module-scoped under `narrative_dynamics.adapters` unless a later integration need justifies root-level exports.

The package root should not be expanded merely for convenience in V1.

## Error handling

Canonical authored-data errors fail during fixture/schema construction before model execution.

Examples include invalid references, impossible relocation continuity, invalid time order, duplicate observations, and malformed decisions.

Semantic model applicability errors such as an unresolved target location also fail closed rather than silently choosing an arbitrary action.

No fallback may substitute objective state when the belief-sensitive model lacks subjective information; that would erase the scientific distinction under test.

## Explicit non-goals

V1 does not implement:

- automatic natural-language parsing;
- entity extraction or coreference resolution;
- dialogue or speech-act parsing;
- testimony, hearsay, or indirect communication;
- lying or deception;
- source credibility or trust parameters;
- noisy perception;
- Bayesian confidence updates for story facts;
- memory decay, forgetting, or recall errors;
- second-order beliefs such as `Alice believes that Bob believes ...`;
- emotion or personality;
- social norms or institutions as behavioral mechanisms;
- arbitrary planning;
- arbitrary action vocabularies;
- a generic fluent logic language;
- parameter fitting;
- train/selection/final observational evaluation;
- empirical human-behavior claims;
- causal identification or population inference.

These are later extensions only after the canonical information-state boundary is proven correct.

## Success criteria

Narrative Microstory V1 is complete only when all of the following hold:

1. the two authored fixtures load, validate, round-trip, and verify their content hashes;
2. the model-facing scenario projection excludes all oracle/source-answer material;
3. objective replay returns `key@box` for both fixtures;
4. Bob's subjective replay returns `key@drawer` in false belief and `key@box` in informed belief;
5. the Lean semantic layer proves that unobserved relocation cannot update the agent projection while admissible observation can;
6. `agent-belief-search` and `omniscient-search` disagree only where the information contrast predicts they should;
7. both models expose one common categorical prediction contract and run through the existing trusted simulation runtime;
8. the complete pre-existing Lean and Python test suites remain green;
9. no prison model semantics are modified;
10. no claim is made that success on the authored pair establishes general theory-of-mind competence, natural-language understanding, or empirical validity.

## Expected next increment

After V1, the preferred next step is not broader story complexity immediately. It is an upstream parser experiment that attempts to recover the same canonical `NarrativeCaseV1` structure from the committed source texts and is scored against the manual annotations.

Only after the canonical parser boundary is measurable should the story domain expand to testimony, deception, uncertain perception, memory, or nested beliefs.
