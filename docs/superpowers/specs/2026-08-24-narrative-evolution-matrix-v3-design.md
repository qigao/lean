# Narrative Evolution Matrix V3 Design

## Status

Design approved in conversation on 2026-08-24. This document freezes the first implementation scope before an implementation plan is written.

## Purpose

Narrative Testimony V2 can already represent and execute a distinction among objective world state, direct perception, received testimony with explicit source provenance, recipient-specific epistemic state, and final decision state.

V3 adds an analysis layer that reconstructs how those states evolve over logical time and compares the baseline trajectory with minimal single-variable counterfactual trajectories. The goal is to identify where two trajectories first diverge and which canonical state fields differ, without claiming that one observed action uniquely identifies one hidden cognitive mechanism.

The intended research question is:

> Given a validated testimony scenario, how do world events and information-path changes alter agent-specific epistemic state over time, and at which canonical step does a minimal counterfactual intervention first change the resulting trajectory?

This is a structural and counterfactual analysis capability. It is not an empirical claim about human cognition and does not establish unique causal identification from final behavior alone.

## Existing foundation

V3 is built on the existing V2 boundaries and must not redefine them:

- `NarrativeScenarioV2` is the canonical model-visible testimony scenario.
- `objective_state` and `subjective_state` provide objective and direct-perception replay.
- `testimony_state` provides recipient-specific direct-plus-testimony replay with provenance.
- V2 validation enforces declared entities, strict logical time, speaker support observation, report timing, and valid reception references.
- Report admissibility is intentionally not a truth predicate: a supported report may assert a location different from objective state.
- V2 runtime payload remains exactly `entities/events/observations/reports/receptions/decision`.
- V1 fixture hashes and V1 runtime IDs remain exact regression locks.

V3 consumes those semantics; it does not modify them.

## Architectural decision

### Chosen approach: independent evolution analysis layer

Add a new module:

`narrative_dynamics/story/evolution_v2.py`

This module accepts only an already validated `NarrativeScenarioV2`. It reconstructs snapshots, derives one baseline trajectory, generates a fixed set of minimal one-change-at-a-time counterfactuals, and compares valid counterfactual trajectories against the baseline.

This is preferred over extending `replay_v2.py` because replay should remain responsible only for state reconstruction. It is also preferred over extending `SimulationRunner` or common metrics because evolution analysis is story-domain logic, not generic runtime infrastructure.

### Dependency direction

The dependency direction must remain:

`schema/scenario -> replay -> evolution analysis`

The story package must not import adapters.

To avoid duplicating testimony action-selection semantics, extract a pure story-layer helper for deterministic testimony action resolution. `TestimonySearchModel` will reuse that helper. The helper must depend only on canonical story state and the decision action set, not on runtime traces, adapters, registries, or prison-domain code.

No new Lean theorem is required for V3. Lean remains the formal lower-layer boundary for information-path and testimony admissibility properties already established in V2.

## Scope of analysis

### Target object

V3 analyzes exactly the current decision target:

`story.decision.object`

The first version does not support arbitrary object overrides. This keeps the matrix aligned with the decision whose behavior is being explained.

### Tracked agents

By default, tracked agents are the smallest set relevant to the decision chain:

1. the decision actor; and
2. every speaker of a report about the decision target object.

Duplicates are removed while preserving deterministic ordering. The first version does not expand every declared agent and does not provide an override parameter.

This prevents the matrix from becoming a general world dump while retaining all agents required to interpret the target testimony chain.

### Snapshot times

A trajectory contains one snapshot at every canonical logical time relevant to the target analysis:

- every relocation event time;
- every report logical time; and
- the decision time.

Times are sorted strictly ascending and deduplicated.

Reception has no independent timestamp in V2. Therefore, consistent with existing V2 semantics, a declared reception makes the report available to that recipient at the report's own `logical_time`.

## Data model

All canonical V3 analysis values are immutable dataclasses and must contain only deterministic, JSON-serializable values or tuples/mappings of such values.

Two support records are implementation-internal and are not exported from `narrative_dynamics.story`: `_EvolutionAgentStateV2` and `_EvolutionDivergenceV2`. They exist only to keep the public objects typed without widening the approved package API.

### Internal `_EvolutionAgentStateV2`

Represents one tracked agent's state for the decision target at one snapshot.

Fields:

- `agent: str`
- `direct_location: str | None`
- `direct_supporting_id: str | None`
- `testimony_location: str | None`
- `evidence_kind: str | None`
- `supporting_id: str | None`
- `source_agent: str | None`
- `evidence_logical_time: int | None`

`None` means there is no supported state for that field at that time. Absence is represented explicitly rather than by falling back to objective truth.

### `EvolutionSnapshotV2`

Represents the canonical state at one logical time.

Fields:

- `logical_time: int`
- `trigger_kind: str`
- `trigger_ids: tuple[str, ...]`
- `objective_location: str | None`
- `agents: Mapping[str, _EvolutionAgentStateV2]`
- `selected_action: str | None`

`trigger_kind` is one of:

- `relocation`
- `report`
- `decision`
- `mixed`

The current V2 validator makes relocation and report times globally unique and requires reports before the decision, so `mixed` is defensive for future compatibility and should not occur in committed V2 fixtures.

`selected_action` is `None` except at decision time.

### `EvolutionTrajectoryV2`

Fields:

- `target_object: str`
- `tracked_agents: tuple[str, ...]`
- `snapshots: tuple[EvolutionSnapshotV2, ...]`
- `selected_action: str`

The trajectory is deterministic for a canonical scenario.

The trajectory deliberately does not contain a runtime `scenario_id`. `NarrativeScenarioV2` is a typed semantic value and does not carry the runtime `Scenario.id`; V3 must not duplicate or depend on private runtime-ID derivation merely to decorate a derived analysis artifact.

### `EvolutionInterventionV2`

Represents exactly one minimal intervention.

Fields:

- `kind: str`
- `subject_id: str`
- `agent: str | None`
- `from_value: str | None`
- `to_value: str | None`
- `logical_time: int | None`

Supported kinds in V3 are exactly:

- `remove_reception`
- `change_report_content`
- `remove_direct_observation`
- `remove_support_observation`

No compound intervention is allowed.

### Internal `_EvolutionDivergenceV2`

Fields:

- `logical_time: int`
- `changed_fields: tuple[str, ...]`
- `action_changed: bool`

`changed_fields` contains stable canonical paths, for example:

- `agents.bob.testimony_location`
- `agents.bob.evidence_kind`
- `agents.bob.supporting_id`
- `selected_action`

The first divergence is determined only from canonical snapshot state. Human-readable descriptions, fixture names, source text, provenance metadata, oracle values, and gold labels are never comparison inputs.

### `EvolutionCounterfactualV2`

Fields:

- `intervention: EvolutionInterventionV2`
- `status: str` where value is exactly `valid` or `rejected`
- `trajectory: EvolutionTrajectoryV2 | None`
- `first_divergence: _EvolutionDivergenceV2 | None`
- `rejection_stage: str | None`
- `rejection_reason: str | None`
- `rejection_logical_time: int | None`

`rejection_stage`, when present, is exactly one of:

- `scenario_validation`
- `action_resolution`

For a valid counterfactual, `trajectory` is present and rejection fields are `None`.

For a rejected counterfactual, `trajectory` and `first_divergence` are `None`. The intervention's relevant logical time is recorded as `rejection_logical_time` when it can be identified. Rejection is not labeled as a trajectory divergence because no valid alternate trajectory exists.

### `EvolutionAnalysisV2`

Fields:

- `baseline: EvolutionTrajectoryV2`
- `counterfactuals: tuple[EvolutionCounterfactualV2, ...]`
- `mechanism_uniqueness_claimed: bool`

`mechanism_uniqueness_claimed` is always `False` in V3. This is an explicit safety contract: the analysis records state-path differences and action changes but never claims that one final action uniquely identifies one mechanism.

## Public API

Export exactly these new names from `narrative_dynamics.story`, but not from package root:

- `EvolutionSnapshotV2`
- `EvolutionTrajectoryV2`
- `EvolutionInterventionV2`
- `EvolutionCounterfactualV2`
- `EvolutionAnalysisV2`
- `analyze_testimony_evolution`

Primary entry point:

```python
analyze_testimony_evolution(story: NarrativeScenarioV2) -> EvolutionAnalysisV2
```

The function must reject any value that is not a validated `NarrativeScenarioV2`.

No V3 API is exported from `narrative_dynamics` package root. The two internal support records are not added to `narrative_dynamics.story.__all__`.

## Baseline trajectory algorithm

For the decision target object:

1. Determine tracked agents from the decision actor plus relevant report speakers.
2. Build the sorted snapshot time set from relocation times, report times, and decision time.
3. For each snapshot time `t`:
   - derive objective target location from relocation events at or before `t`;
   - for each tracked agent, derive direct-perception target state at or before `t`;
   - for each tracked agent, derive testimony-aware target state using `testimony_state(..., at_time=t)`;
   - preserve evidence kind, supporting ID, source agent, and evidence logical time;
   - identify the canonical trigger IDs at `t`;
   - at decision time only, resolve the deterministic testimony-aware action using the shared pure action resolver.
4. Return the immutable ordered trajectory.

A missing direct or testimony-supported state is represented with `None`; it never falls back to objective state.

## Shared testimony action resolver

Introduce one pure story-layer function that contains the action-selection rule currently implemented by `TestimonySearchModel`:

1. obtain the decision actor's latest testimony-aware location at decision time;
2. require exactly one declared decision action whose location matches that epistemic location;
3. return that action ID;
4. fail closed if there is no supported epistemic location or if the location does not match exactly one action.

`TestimonySearchModel` must reuse this helper rather than maintain a separate copy of the selection rule.

This refactor must not change the model name, parameter contract, policy shape, event trace shape, outcome shape, determinism, or runtime integration.

## Counterfactual generation

V3 automatically generates only minimal single-variable interventions. It never generates combinations.

### 1. Remove decision-actor reception

For every report about the decision target that the decision actor receives, generate one counterfactual that removes only that `(report, recipient)` reception.

All events, reports, observations, decision fields, and other receptions remain unchanged.

If the modified scenario remains valid, produce a trajectory. If the decision actor still has direct support, testimony-aware state may fall back to that direct evidence through the existing replay rule.

### 2. Change report content

For every report about the decision target, generate one counterfactual for each alternate location represented by the decision's declared actions, excluding the report's current location.

Only `report.location` changes. Speaker, support event, report time, reception, events, observations, and decision remain unchanged.

This is a content intervention, not a truth-correction operation. V3 must not compare the new content with objective truth before admitting the counterfactual.

### 3. Remove decision-actor direct observation

For every direct observation by the decision actor of a relocation involving the decision target, generate one counterfactual that removes only that observation.

The modified scenario is then reconstructed through the ordinary canonical validator. If the V2 semantics reject it, record a rejected counterfactual rather than repairing, weakening, or bypassing validation.

For the two committed testimony fixtures, Bob has exactly one target observation, `e1`. Removing it is therefore deterministically rejected by the existing shared story validator with the boundary that the decision actor must have observed a relocation of the target object.

### 4. Remove speaker support observation

For every target report, generate one counterfactual that removes the speaker's direct observation of that report's declared support event.

Under current V2 semantics this is expected to be rejected because canonical report support requires the speaker to have observed the support event. The rejected result is important evidence that the support observation is a necessary admissibility boundary.

No validation bypass flag is allowed.

## Counterfactual validation pipeline

Every intervention follows the same pipeline:

1. construct a modified typed V2 scenario value from the baseline fields;
2. allow normal `NarrativeScenarioV2` validation to run;
3. if scenario validation fails because of the intervention, record `status="rejected"` with `rejection_stage="scenario_validation"`, exact reason, and intervention logical time when known;
4. if the modified scenario is valid, build a trajectory using the same baseline trajectory builder;
5. if deterministic action resolution fails for the otherwise valid modified scenario, record `status="rejected"` with `rejection_stage="action_resolution"` and the exact reason;
6. never catch a validation/action-boundary error and silently coerce the scenario into a valid trajectory.

The analysis layer must not consult authored oracle fields because `NarrativeScenarioV2` does not contain them.

Expected domain-boundary failures caused by a well-formed intervention become rejected counterfactual records. Unexpected programming/type errors unrelated to canonical validation or action resolution propagate as top-level errors.

## First-divergence algorithm

For valid counterfactuals only:

1. align baseline and counterfactual snapshots by canonical logical time;
2. compare snapshot state in ascending time order;
3. compare only:
   - objective target location;
   - each tracked agent's direct location and direct support ID;
   - each tracked agent's testimony location, evidence kind, supporting ID, source agent, and evidence logical time;
   - selected action at decision time;
4. the earliest snapshot containing any difference is the first divergence;
5. record all changed canonical field paths at that earliest time;
6. `action_changed` is true iff the final selected action differs, regardless of whether action difference occurs at the first divergence time.

A counterfactual may have an epistemic divergence without an action divergence. V3 must preserve that distinction.

If a valid counterfactual is canonically identical to the baseline across all compared fields, `first_divergence` is `None` and `action_changed` is false.

## Expected committed-fixture behavior

For the existing truthful and stale testimony fixtures:

### Baseline world trajectory

Objective target location:

`drawer -> box -> box -> box`

across `t=1,2,3,4`.

### Bob direct-perception trajectory

Bob sees `e1` but not `e2`, so Bob's direct target location remains `drawer` after `t=1` through decision time.

### Truthful testimony baseline

At report time `t=3`, Bob's testimony-aware state becomes:

- location `box`
- evidence kind `testimony`
- supporting ID `r1`
- source agent `alice`

At `t=4`, selected action is `search_box`.

### Stale testimony baseline

At report time `t=3`, Bob's testimony-aware state is:

- location `drawer`
- evidence kind `testimony`
- supporting ID `r1`
- source agent `alice`

At `t=4`, selected action is `search_drawer`.

### Reception removal

Removing Bob's reception of `r1` yields a valid scenario. Bob remains on direct evidence `drawer` at and after `t=3`. In the truthful fixture this changes the final action from `search_box` to `search_drawer`; in the stale fixture the final action remains `search_drawer` even though the evidence provenance changes from testimony to direct perception.

This explicitly demonstrates that identical final action does not imply identical mechanism.

### Report-content change

Changing only `r1.location` from `box` to `drawer` in the truthful fixture produces first divergence at `t=3` and changes the final action to `search_drawer`.

Changing only `r1.location` from `drawer` to `box` in the stale fixture produces first divergence at `t=3` and changes the final action to `search_box`.

### Decision-actor observation removal

Removing Bob's only target observation `e1` from either committed fixture is rejected during `scenario_validation` because the decision actor no longer has any observed relocation of the target object.

### Support-observation removal

Removing Alice's observation of `e2` makes the existing `r1` unsupported under V2 provenance validation and is rejected during `scenario_validation`.

## Mechanism-identification safety

V3 must encode a negative result as part of its contract:

> Final action alone does not uniquely identify the internal information mechanism.

The canonical witness is the stale fixture:

- testimony-aware baseline can select `search_drawer` using testimony from `r1`;
- removing reception can also select `search_drawer` using Bob's direct perception of `e1`.

Because the action is the same while evidence kind, support ID, and source differ, V3 must expose those provenance differences and must not emit labels such as `unique_cause`, `true_mechanism`, or `identified_mechanism`.

The matrix supports mechanism discrimination only when a scenario or counterfactual produces distinguishable predictions.

## Serialization and determinism

All analysis objects must provide a deterministic structural representation suitable for JSON serialization. Ordering is fixed by:

- snapshot logical time ascending;
- tracked agents in deterministic derived order;
- counterfactuals in a documented stable order by intervention kind, logical time, subject ID, and alternate value;
- changed field paths lexicographically sorted.

The first version does not require a new persisted content hash, runtime ID, or duplicated scenario ID for the analysis result. It is a derived analysis artifact, not a new simulation scenario identity.

## Error handling

Baseline analysis fails closed on:

- non-`NarrativeScenarioV2` input;
- missing decision-target epistemic support at action resolution;
- zero or multiple matching decision actions.

Counterfactual records are rejected, rather than aborting the whole analysis, when a well-formed single intervention causes:

- canonical V2 scenario validation failure; or
- deterministic action-resolution failure in an otherwise valid modified scenario.

Malformed intervention construction is an implementation error and must propagate rather than be mislabeled as scientific counterfactual rejection.

Rejected counterfactuals are data, not top-level analysis failures, only when the rejection is the expected result of applying a validly specified intervention to a valid baseline scenario.

Unexpected programming/type errors unrelated to canonical validation or action resolution must propagate.

## Public-boundary constraints

V3 must not modify:

- V1 fixture content or hashes;
- V1 runtime IDs;
- V1 schema, replay, or scenario semantics;
- V2 runtime payload key set or V2 runtime ID derivation;
- `SimulationRunner`;
- runtime contracts or model registry;
- `story_choice_metrics` semantics;
- prison adapters;
- package-root exports;
- Lean testimony semantics or theorem statements.

`TestimonySearchModel` may be minimally refactored only to reuse the shared action resolver. Its externally observable contract must remain exact.

## Test strategy

Implementation follows strict RED -> GREEN. No production V3 file or resolver refactor is written before the corresponding failing tests are observed on the exact feature head.

### Baseline evolution tests

Lock the truthful and stale matrices at `t=1,2,3,4`:

- objective `drawer -> box -> box -> box`;
- Bob direct state remains `drawer` after `t=1`;
- Alice direct state reaches `box` after observing `e2`;
- truthful Bob testimony state changes to `box` at `t=3`;
- stale Bob testimony state remains location `drawer` but provenance becomes testimony at `t=3`;
- final actions are `search_box` and `search_drawer` respectively.

### Counterfactual tests

Lock:

- remove Bob reception: valid; provenance changes to direct perception; truthful final action changes, stale final action does not;
- change report content: valid; first divergence exactly at report time; final action flips;
- remove Bob's only direct target observation: rejected at `scenario_validation` by the existing decision-actor observation boundary;
- remove Alice support observation: rejected at `scenario_validation` by speaker-support validation.

### Divergence tests

Assert:

- first divergence uses canonical state fields only;
- changed field paths are exact and stable;
- action change is reported separately from first epistemic divergence;
- canonically identical valid trajectories yield no first divergence.

### Mechanism-safety test

Assert that stale baseline and stale reception-removal counterfactual select the same action while evidence provenance differs, and that `mechanism_uniqueness_claimed` is false.

### Regression tests

Lock:

- exact V1 fixture hashes;
- exact V1 runtime IDs;
- V2 projection remains exactly six visible keys;
- V1 and V2 decoders remain mutually rejecting;
- all existing story runtime/model tests remain green;
- exact story package export set is updated with exactly the six approved V3 names and no internal helper types;
- package root still exposes none of the V2/V3 story-specific APIs;
- `TestimonySearchModel` output/event/policy contract remains byte-for-structure equivalent for the committed fixtures before and after shared-resolver refactor.

## Verification requirements

Before V3 can be integrated into the research branch:

1. exact-head GitHub Actions proof workflow must complete successfully;
2. full Lean build must remain green even though V3 adds no Lean code;
3. full Python unittest discovery must be green;
4. StoryState and Testimony Lean theorem gates must remain green;
5. complete diff must be reviewed against the exact pre-V3 research SHA;
6. no unplanned runtime, registry, prison, V1 semantic, or Lean changes may appear;
7. research branch may be advanced only by non-forced fast-forward after review and exact-head verification;
8. `master` must remain unchanged.

A standalone `python3 -m compileall -q narrative_dynamics` check may be reported only if it is actually run independently; CI success must not be described as standalone compileall evidence.

## Non-goals

V3 does not add:

- multi-intervention combinations;
- exhaustive power-set counterfactual search;
- learned trust or source reliability;
- Bayesian testimony weighting;
- deception or lying intent;
- speaker utility models;
- recursive Theory of Mind;
- free-text NLP parsing;
- empirical human data;
- population-level inference;
- automatic unique causal attribution;
- causal effect estimation from observational data;
- new runtime or registry infrastructure;
- new Lean semantics.

## Claim boundary

After V3, the strongest supported claim is:

> Narrative Dynamics can reconstruct a canonical temporal evolution of objective, direct-perception, and testimony-aware decision state for a validated testimony scenario, and can compare that trajectory with minimal single-variable counterfactuals to identify the earliest canonical state divergence or validation boundary.

It does not prove that a final observed action uniquely reveals the underlying cognitive mechanism, and it does not establish empirical human causal validity.
