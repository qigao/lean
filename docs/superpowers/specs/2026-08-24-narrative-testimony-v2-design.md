# Narrative Testimony V2 — Design

Date: 2026-08-24
Status: proposed design for review
Base branch: `proof/narrative-dynamics-v0`
Feature branch: `work/narrative-testimony-v2`
Implementation scope: Lean testimony/provenance semantics + Python canonical story/runtime extension

## Purpose

Narrative Microstory V1 proved one information boundary end to end:

```text
directly observed relocation
  → agent-specific subjective state
  → search decision
  → belief-sensitive prediction
```

V2 adds the smallest scientifically useful second information route:

```text
world event
  → another agent reports a proposition
  → recipient receives that report
  → report enters recipient-specific epistemic state with source provenance
  → search decision
```

The objective is not to model language understanding, persuasion, deception, or social trust in general. The objective is to represent communicated belief as a distinct evidence path from direct perception and to make its provenance explicit and testable.

The central question is:

> Can an agent update a decision-relevant belief from communicated testimony while preserving the distinction between objective truth, direct observation, report content, report reception, and the report's source?

V2 must preserve every V1 anti-leakage, fail-closed, runtime, and parameter-free boundary.

## Scientific contrast

V2 introduces one matched pair over the same objective relocation history.

### Case A: truthful testimony

Objective history:

```text
e1 @ t=1: Alice relocates key unknown → drawer
e2 @ t=2: Alice relocates key drawer → box
```

Bob directly observes `e1` but not `e2`.

Alice observes `e2`, then sends Bob a report whose content is `key@box`. Bob receives the report before deciding.

At decision time:

```text
objective(key)                  = box
Bob direct-perception state     = drawer
Bob testimony-supported state   = box
```

The testimony-aware model predicts `search_box`; the V1 direct-observation-only belief model continues to predict `search_drawer`; the omniscient baseline predicts `search_box`.

### Case B: stale / false testimony

The objective relocation history is identical.

Alice sends Bob a report whose content is `key@drawer`, and Bob receives it before deciding. The report is syntactically valid and genuinely received, but its proposition disagrees with current objective state.

At decision time:

```text
objective(key)                  = box
Bob direct-perception state     = drawer
Bob latest received testimony   = drawer
```

The testimony-aware model predicts `search_drawer`; the omniscient model predicts `search_box`.

This case is not labeled a psychological deception model. V2 only demonstrates that communicated content can diverge from narrator-level truth while remaining valid communicated evidence.

The pair isolates report content while holding objective event history, recipient, delivery path, decision, and action set fixed.

## Non-goals

V2 does not add:

- natural-language parsing or generation;
- hidden intentions, lying propensity, or speaker utility;
- learned trust weights;
- Bayesian reliability estimation;
- recursive theory of mind;
- conversation trees, dialogue acts, pronoun/coreference resolution, or free text semantics;
- empirical human data, population inference, or behavioral calibration;
- new simulation runtime, registry semantics, observation protocol, or prison-domain coupling;
- free parameters such as `beta`, source reliability, or attention probability.

A later version may study source reliability only after communicated evidence and its provenance are represented correctly without fitting knobs.

## Existing boundaries that remain authoritative

V2 reuses without changing:

- `WorldGraph.eventTime` as the only Lean event clock;
- information reachability and proof-carrying observation admission;
- provenance/support graph semantics;
- immutable Python `Scenario`, trace, manifest, and stable content-hash boundaries;
- `SimulationRunner` as the trusted execution path;
- V1 objective relocation continuity;
- V1 anti-leakage scenario projection and neutral payload-derived IDs;
- V1 distinction between objective replay and agent-specific epistemic replay.

V2 must not add a new `NodeKind`, change `TypedEdge.locatedAt`, or reinterpret V1 direct perception as testimony.

## Canonical semantic model

### Events remain objective

V1 `RelocationEventV1` remains the only world-state mutation in V2. Reports do not move objects and therefore cannot change objective replay.

### Direct observations remain direct observations

`DirectObservationV1(event, agent)` continues to mean that the agent directly perceived the referenced relocation event. It is not overloaded to mean hearing a report.

### New testimony values

V2 introduces two authored/runtime-visible semantic records.

```python
@dataclass(frozen=True)
class LocationReportV2:
    id: str
    logical_time: int
    speaker: str
    object: str
    location: str
    support_event: str
    kind: str = "report_object_location"

@dataclass(frozen=True)
class ReportReceptionV2:
    report: str
    recipient: str
    channel: str = "direct_testimony"
```

A `LocationReportV2` is an authored speech-act event with explicit propositional content. `support_event` records the relocation event the speaker is claiming as support/context for that report. It is provenance metadata inside the canonical semantics, not a gold correctness bit.

`ReportReceptionV2` states that the recipient actually receives that report. A report that exists but is not received must not affect the recipient's state.

The report's `location` is not required to equal objective truth. This is deliberate: requiring truth during schema validation would make false/stale testimony impossible and would leak narrator truth into the communication boundary.

### Time ordering

All relocation and report `logical_time` values participate in one strict narrative timeline. V2 must reject ambiguous equal-time ordering between world relocations, reports, and the decision.

The committed cases use:

```text
e1 relocation @ 1
e2 relocation @ 2
r1 report     @ 3
d1 decision   @ 4
```

No second clock is introduced in Lean; the concrete V2 witness maps those canonical times onto `WorldGraph.eventTime` for the corresponding world/report events.

### Report support constraints

A report is structurally valid only when:

- report ID is unique and non-empty;
- speaker, object, location references are declared;
- `support_event` references a declared relocation event of the same target object;
- `support_event` occurs strictly before the report;
- the speaker directly observed `support_event` before making the report;
- the report's logical time is strictly ordered in the global narrative timeline.

The last two rules make source provenance meaningful: a report cannot claim canonical support from an event that never entered the speaker's information boundary.

These rules do **not** assert that `report.location` equals the destination of `support_event`. Therefore a stale/false report is valid evidence with traceable provenance, not silently converted into truth.

### Reception constraints

A reception is valid only when:

- the referenced report exists;
- the recipient is a declared agent;
- duplicate `(report, recipient)` receptions are rejected;
- V2 supports exactly `channel="direct_testimony"`;
- report time is strictly before the decision when the reception is used for that decision actor.

V2 does not assume every report is broadcast or automatically heard.

## Epistemic replay

V2 keeps objective replay unchanged.

### Direct-perception state

The V1 `subjective_state(...)` contract remains unchanged and continues to replay only directly observed relocations. This preservation is important: existing callers retain the exact V1 meaning.

### Testimony-aware state

V2 adds a separate pure projection, conceptually:

```python
testimony_state(
    events,
    observations,
    reports,
    receptions,
    agent,
    *,
    at_time=None,
) -> Mapping[str, EpistemicLocationStateV2]
```

The returned state records not only location but evidence kind and provenance:

```python
@dataclass(frozen=True)
class EpistemicLocationStateV2:
    object_id: str
    location: str
    evidence_kind: str  # "direct_perception" | "testimony"
    supporting_id: str  # relocation event ID or report ID
    source_agent: str
    logical_time: int
```

Replay rules:

1. direct observations establish directly perceived object-location support at the relocation event time;
2. a received report establishes testimony support at the report time using the report's asserted location;
3. only evidence available at or before `at_time` participates;
4. the latest supported evidence for an object wins by logical time;
5. direct evidence does not receive an automatic priority bonus over later testimony, and testimony does not overwrite objective world state;
6. an unreceived report has no effect;
7. testimony from a report whose canonical provenance fails validation cannot enter replay;
8. there is no fallback from testimony-aware private state to objective state.

This deliberately models evidence chronology, not rational trust. Reliability weighting is deferred.

## Decision models

### Existing models stay unchanged

- `AgentBeliefSearchModel` remains V1 direct-observation-only.
- `OmniscientSearchModel` remains objective-state-only.

Changing either would destroy the clean competing-model baselines established in V1.

### New model

V2 adds:

```text
testimony-search
```

`TestimonySearchModel` uses the testimony-aware private state of the decision actor and otherwise emits the same deterministic one-hot action contract as V1.

It accepts exactly an empty parameter mapping. RNG is accepted only because it satisfies `SimulatorModel`; it is unused.

Outcome `epistemic_basis` is extended for this model with provenance fields while retaining stable core coordinates:

```text
{
  "kind": "testimony" | "direct_perception",
  "target_object": str,
  "resolved_location": str,
  "supporting_id": str,
  "source_agent": str
}
```

The action policy remains the common two-action `search_box` / `search_drawer` surface so `story_choice_metrics` can remain unchanged.

## Lean semantic extension

V2 adds a small module rather than modifying the V1 theorem surface in place:

```text
NarrativeDynamics/Core/Testimony.lean
```

Representative generic values:

```text
LocationReport Agent Event Object Location
reportAvailableTo
reportedLocation
```

The Lean layer must distinguish:

- a relocation/world event;
- the speaker having observed the support event;
- a report event carrying an asserted proposition;
- an information path from report event to recipient;
- the recipient having received the report.

A report proposition may disagree with objective latest location. Lean well-formedness proves information/provenance availability, not truthfulness.

Required theorem-level properties:

1. an unreceived report cannot alter a recipient's testimony-supported location;
2. a received report can update the recipient's latest supported location to the report content;
3. report reception requires an information path from the report event to the recipient under `WorldInvariant`;
4. a report cannot be canonically supported by a relocation event the speaker did not observe;
5. report truth is independent from report admissibility: an admissible report may assert a location different from objective latest state;
6. truthful testimony can make recipient belief converge with objective state;
7. stale/false testimony can leave recipient belief divergent from objective state;
8. the concrete matched pair yields `search_box` versus `search_drawer` for the testimony-aware model while omniscient remains `search_box` in both.

V2 should reuse existing provenance/observation primitives where possible. If a theorem can be expressed by specializing an existing information-path boundary, do that instead of creating a parallel epistemic kernel.

## Python module architecture

V2 should extend the existing `narrative_dynamics.story` subsystem rather than create a second story package.

Expected files are likely:

```text
narrative_dynamics/story/schema.py          # add V2 testimony dataclasses/validation
narrative_dynamics/story/replay.py          # add testimony-aware pure replay
narrative_dynamics/story/scenario.py        # extend exact visible V2 projection/decoder
narrative_dynamics/story/__init__.py         # export canonical testimony schema/replay only
narrative_dynamics/adapters/story_testimony_search.py
NarrativeDynamics/Core/Testimony.lean
NarrativeDynamics/Tests/Testimony.lean
fixtures/stories/key_location_truthful_testimony_v2.json
fixtures/stories/key_location_stale_testimony_v2.json
```

The implementation plan may choose version-specific files if that produces a cleaner compatibility boundary. It must not break V1 fixture loading or V1 scenario IDs.

## Schema versioning and V1 compatibility

V2 canonical fixtures use `schema_version = 2`.

V1 fixtures and runtime payloads remain valid and byte-for-byte identity stable. In particular:

- V1 `content_hash` values do not change;
- V1 scenario IDs do not change;
- V1 public dataclass behavior does not change;
- V1 models emit the same outcomes;
- V1 Python tests remain green without fixture rewrites.

A V2 fixture includes V1 fields plus:

```text
reports
receptions
```

V2 runtime payload includes exactly:

```text
entities
events
observations
reports
receptions
decision
```

It still excludes fixture name/version/source/provenance/source_text/oracle and any gold model answer.

The runtime ID is derived solely from this model-visible payload, using a version-neutral prefix that distinguishes V2, for example:

```text
story-v2-<payload sha256 hex>
```

V1 decoder must not accept a V2 payload accidentally, and V2 decoder must not silently reinterpret a V1 payload.

## Fixture oracle and research metadata

V2 fixtures may contain authored oracle fields for tests, but oracle content is never model-visible.

Recommended V2 oracle records:

```text
objective_location
actor_direct_location
actor_testimony_location
testimony_ranking
omniscient_ranking
```

As in V1, oracle values are references to declared locations/actions but are not trusted to establish replay truth. Production replay computes its result from canonical semantics.

Both V2 fixtures must state:

```text
synthetic = true
empirical_human_data = false
population_representative = false
```

## Anti-leakage requirements

Tests must prove all of the following:

- raw source text never enters `Scenario.payload`;
- authored oracle/rankings never enter `Scenario.payload`;
- fixture name/version/source/provenance do not affect runtime payload or runtime ID;
- no field such as `truthful`, `false`, `stale`, `correct`, or `gold` is present in model-visible payload;
- scenario ID cannot reveal the fixture label;
- changing only oracle/provenance/source text cannot change V2 runtime ID;
- changing report content does change runtime payload and ID because report content is genuinely model-visible semantics;
- manually constructed malformed V2 payloads cannot bypass speaker-observation support validation or reception validation.

## Error handling

V2 remains fail closed.

Reject, at minimum:

- unknown/missing V2 payload keys;
- duplicate report IDs;
- undeclared speaker/object/location/recipient;
- support event with wrong object;
- report at or before its support event;
- speaker lacking direct observation of the support event;
- duplicate reception pairs;
- unsupported reception channel;
- report at or after decision time when used by the decision actor;
- non-neutral or mismatched runtime ID;
- testimony-aware model with non-empty parameters;
- private state with no supported target location;
- resolved location without exactly one matching decision action.

No error path may fall back to objective state or oracle metadata.

## Trusted runtime and public API

`TestimonySearchModel` must run unchanged through existing `SimulationRunner` and produce canonical manifests/traces with empty parameters.

Canonical V2 testimony schema and replay helpers may be exported from `narrative_dynamics.story`.

Model adapters remain module-scoped under `narrative_dynamics.adapters`; package root `narrative_dynamics` remains unchanged.

No registry/runtime change is expected unless implementation proves an actual incompatibility. Any such incompatibility requires a focused RED test before changing shared runtime code.

## Testing strategy

Implementation remains strict RED → GREEN.

Minimum test families:

1. **Schema V2:** frozen values, exact keys, content hash, support references, global time order, speaker observation provenance, reception uniqueness/channel, V1 compatibility.
2. **Replay:** direct-only baseline preserved; truthful received testimony updates Bob to box; stale received testimony keeps/updates Bob to drawer; unreceived testimony has no effect; historical `at_time` queries are deterministic; no objective fallback.
3. **Scenario:** exact six-key V2 payload; neutral `story-v2-*` ID; metadata/oracle leakage resistance; malformed runtime payload cannot bypass shared validation.
4. **Models:** testimony-search behavior on matched pair; omniscient remains box in both; existing V1 belief model remains drawer on the V2 false-belief direct-perception history when run on an appropriate V1 projection or compatibility witness; empty parameters only; exact one-hot policy.
5. **Trusted runtime:** deterministic seed replay, manifest presence, common metrics compatibility, package export isolation, no prison imports.
6. **Lean:** generic testimony properties plus concrete truthful/stale witness and information-path/provenance checks.
7. **Regression:** all V1 fixture hashes, V1 scenario IDs, V1 tests, existing 311 Python tests, conformance, Lean build, and pre-existing theorem suite remain green.

## Integration boundaries

Development occurs on `work/narrative-testimony-v2` based on the exact current `proof/narrative-dynamics-v0` head.

Final integration, if the complete review is green, is a non-forced fast-forward of `proof/narrative-dynamics-v0` only.

`master` is not moved by this work.

## Acceptance criteria

V2 is complete only when:

- two committed V2 fixtures validate and have identical objective relocation histories;
- their only discriminating model-visible semantic difference is report content;
- Bob directly observes `e1` but not `e2` in both;
- Alice directly observes `e2` and is therefore a canonically supported report source in both;
- Bob receives the report in both;
- objective replay is `key@box` in both;
- direct-perception-only Bob state is `key@drawer` in both;
- testimony-aware Bob state is `key@box` for truthful testimony and `key@drawer` for stale/false testimony;
- testimony-search predicts box versus drawer across the pair;
- omniscient predicts box in both;
- report admissibility does not imply report truth;
- unreceived testimony has no effect;
- malformed runtime V2 scenarios cannot bypass provenance/reception validation;
- V1 fixtures, hashes, scenario IDs, APIs, and models are unchanged;
- no source/oracle/gold label leaks into runtime input or ID;
- both models remain parameter-free;
- all existing proof and Python gates remain green;
- final review records the exact SHA, proof run, Python test count, and explicit research limitations.

## Research claim boundary

A successful V2 supports only this repository-level claim:

> Narrative Dynamics can represent and execute a canonical distinction between direct perception, communicated testimony with explicit source provenance, objective state, and recipient-specific decision state.

It does not establish that humans reason this way, that speakers are trusted, that false statements are deception, or that the system understands unrestricted natural language.
