# Narrative Testimony V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a canonical V2 story domain in which a decision actor can update a private object-location belief from received testimony with explicit source provenance, while preserving V1 direct-perception semantics, anti-leakage guarantees, deterministic model baselines, and research-branch isolation.

**Architecture:** Keep V1 implementation files unchanged and add version-specific V2 schema, scenario, and replay modules inside `narrative_dynamics.story`. A V2 authored case reuses V1 entity/relocation/direct-observation/decision dataclasses, adds reports and receptions, validates testimony provenance once in `schema_v2.py`, projects only six model-visible fields through `scenario_v2.py`, and computes recipient state through `replay_v2.py`. Lean adds a separate generic `Testimony` module that reuses `WorldGraph.observed`, `WorldGraph.eventTime`, and existing information-path invariants rather than introducing a parallel epistemic kernel.

**Tech Stack:** Lean 4.32 + mathlib; Python 3 standard library (`dataclasses`, `unittest`, immutable mappings); existing Narrative Dynamics contracts, stable content hashing, and `SimulationRunner`; GitHub Actions `proof` workflow.

**Spec:** `docs/superpowers/specs/2026-08-24-narrative-testimony-v2-design.md`

## Global Constraints

- Base research branch is exactly `proof/narrative-dynamics-v0`; feature branch is `work/narrative-testimony-v2`.
- Baseline final V1 research SHA is `e56331f8924c96308e671eda4d7238d6a1f16fe7`; do not move `master`.
- Baseline Python suite is 311 tests.
- V1 false-belief fixture hash must remain exactly `sha256:761f019271bfb097d95c7c77e80811496c22ee17089f586a433a1a8c21020420`.
- V1 informed fixture hash must remain exactly `sha256:2d63d1cc5a50a059cbcf9f425ba54275474bff4abb74ed37703514e069cff862`.
- V1 false-belief runtime scenario ID must remain exactly `story-v1-3b8036566da885c9f07136e141fe3a03f4d80844341dd6d97d109bcbdad4b38e`.
- V1 informed runtime scenario ID must remain exactly `story-v1-ecc5c96321e624e27ea4f07146875ba4a1878e1afadc09dd1337e169478f5368`.
- Do not modify `narrative_dynamics/story/schema.py`, `narrative_dynamics/story/replay.py`, `narrative_dynamics/story/scenario.py`, `narrative_dynamics/adapters/story_belief_search.py`, or `narrative_dynamics/adapters/story_omniscient_search.py` unless a focused RED proves an actual V1 incompatibility. The planned implementation requires no such modification.
- Do not modify `narrative_dynamics/__init__.py`, registry/runtime semantics, observation-protocol semantics, prison adapters, dependency files, `NodeKind`, or `TypedEdge.locatedAt`.
- V2 canonical fixtures use `schema_version = 2`; V1 fixture loaders and V1 scenario decoders remain V1-only.
- V2 reports are epistemic communication events only; `RelocationEventV1` remains the only object-world mutation.
- `DirectObservationV1` continues to mean direct perception of a relocation event and is never overloaded to mean report reception.
- All relocation/report logical times in a V2 case are unique; every model-visible report occurs strictly before the decision; each report support event occurs strictly before its report.
- Report admissibility does not require report content to equal objective truth.
- A report is canonically supported only when the speaker directly observed the referenced same-object support relocation.
- A report affects an agent only when a matching `ReportReceptionV2` exists for that recipient.
- V2 runtime payload contains exactly `entities`, `events`, `observations`, `reports`, `receptions`, and `decision` and uses `story-v2-<visible-payload-sha256>` as its neutral ID.
- No runtime payload/ID may contain fixture name/version/source/provenance/source text/oracle or labels such as `truthful`, `false`, `stale`, `correct`, or `gold`.
- `TestimonySearchModel`, `AgentBeliefSearchModel`, and `OmniscientSearchModel` remain parameter-free. Any non-empty parameter mapping fails closed.
- `story_choice_metrics` remains unchanged and continues to consume the common `policy` coordinates `search_box` and `search_drawer`.
- The two committed V2 fixtures have identical entities, objective relocation history, direct observations, report metadata other than asserted `location`, receptions, decision, and action set. Their only discriminating model-visible semantic field is `reports[0].location`.
- Both V2 fixtures are synthetic, `empirical_human_data=false`, and `population_representative=false`.
- Every production increment follows strict RED → verify intended failure → minimal GREEN → full regression verification → remote `proof` green.

---

## File Structure

New focused files:

```text
NarrativeDynamics/Core/Testimony.lean
NarrativeDynamics/Tests/Testimony.lean
narrative_dynamics/story/schema_v2.py
narrative_dynamics/story/scenario_v2.py
narrative_dynamics/story/replay_v2.py
narrative_dynamics/adapters/story_testimony_search.py
fixtures/stories/key_location_truthful_testimony_v2.json
fixtures/stories/key_location_stale_testimony_v2.json
tests/test_story_schema_v2.py
tests/test_story_scenario_v2.py
tests/test_story_replay_v2.py
tests/test_story_testimony_models.py
tests/test_story_testimony_runtime.py
```

Planned modifications:

```text
NarrativeDynamics.lean                    # import Testimony after GREEN
.github/workflows/proof.yml               # dedicated final Testimony theorem gate
narrative_dynamics/story/__init__.py       # append exact canonical V2 exports only
```

The version-specific Python split is intentional: V2 reuses V1 public dataclasses and the package-private V1 semantic validator, but V1 parsing/projection/replay code stays byte-identical.

---

### Task 1: Lean Testimony Semantics and Information-Path Boundary

**Files:**
- Create: `NarrativeDynamics/Tests/Testimony.lean`
- Create: `NarrativeDynamics/Core/Testimony.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `NarrativeDynamics.lean`

**Interfaces:**
- Consumes: `StoryRelocation`, `objectiveLocation`, `subjectiveLocation`, `WorldGraph`, `WorldInvariant`, `infoReachable`.
- Produces:
  - `LocationReport Agent Event Object Location`
  - `ReportSupported world report : Prop`
  - `reportAvailableTo world report recipient : Prop`
  - `reportedLocation world reports recipient object : Option Location`
  - theorem surface proving reception information paths, support observation, unreceived no-op, received update, and truth/admissibility independence.

- [ ] **Step 1: Write the Lean RED theorem witness**

Create `NarrativeDynamics/Tests/Testimony.lean` importing the not-yet-existing module:

```lean
import NarrativeDynamics.Core.Testimony

namespace NarrativeDynamics.Tests.Testimony

inductive Agent where
  | alice | bob
  deriving DecidableEq, Repr

inductive Event where
  | e1 | e2 | r1
  deriving DecidableEq, Repr

inductive Object where
  | key
  deriving DecidableEq, Repr

inductive Location where
  | drawer | box
  deriving DecidableEq, Repr

inductive Node where
  | event1 | event2 | report1 | alice | bob
  deriving DecidableEq, Repr

open Agent Event Object Location

private def eventNode : Event → Node
  | .e1 => .event1
  | .e2 => .event2
  | .r1 => .report1

private def agentNode : Agent → Node
  | .alice => .alice
  | .bob => .bob

private def eventTime : Event → Nat
  | .e1 => 1
  | .e2 => 2
  | .r1 => 3

private def testimonyWorld : WorldGraph Agent Event Node := {
  alive := fun _ => True
  canAct := fun _ => True
  info := fun _ _ => True
  eventNode := eventNode
  agentNode := agentNode
  observed := fun e a =>
    (e = .e1 ∧ a = .bob) ∨
    (e = .e2 ∧ a = .alice) ∨
    (e = .r1 ∧ a = .bob)
  causal := fun _ _ => False
  eventTime := eventTime
}

private def history : List (StoryRelocation Event Object Location) := [
  { event := .e1, object := .key, fromLocation := none, toLocation := .drawer },
  { event := .e2, object := .key, fromLocation := some .drawer, toLocation := .box }
]

private def truthfulReport : LocationReport Agent Event Object Location := {
  reportEvent := .r1
  supportEvent := .e2
  speaker := .alice
  object := .key
  location := .box
}

private def staleReport : LocationReport Agent Event Object Location := {
  reportEvent := .r1
  supportEvent := .e2
  speaker := .alice
  object := .key
  location := .drawer
}

example : ReportSupported testimonyWorld truthfulReport := by
  simp [ReportSupported, testimonyWorld, truthfulReport, eventTime]

example : ReportSupported testimonyWorld staleReport := by
  simp [ReportSupported, testimonyWorld, staleReport, eventTime]

example : objectiveLocation history .key = some .box := by
  rfl

example : subjectiveLocation testimonyWorld history .bob .key = some .drawer := by
  simp [subjectiveLocation, testimonyWorld, history, objectiveLocation,
    applyStoryRelocation]

example : reportedLocation testimonyWorld [truthfulReport] .bob .key = some .box := by
  simp [reportedLocation, reportAvailableTo, testimonyWorld, truthfulReport]

example : reportedLocation testimonyWorld [staleReport] .bob .key = some .drawer := by
  simp [reportedLocation, reportAvailableTo, testimonyWorld, staleReport]

example :
    ReportSupported testimonyWorld staleReport ∧
    reportAvailableTo testimonyWorld staleReport .bob ∧
    objectiveLocation history .key = some .box ∧
    reportedLocation testimonyWorld [staleReport] .bob .key = some .drawer := by
  simp [ReportSupported, reportAvailableTo, testimonyWorld, staleReport,
    history, eventTime, objectiveLocation, applyStoryRelocation, reportedLocation]

#check NarrativeDynamics.reportedLocation_append_unreceived
#check NarrativeDynamics.reportedLocation_append_received
#check NarrativeDynamics.received_report_has_info_path
#check NarrativeDynamics.unobserved_support_cannot_support_report

end NarrativeDynamics.Tests.Testimony
```

- [ ] **Step 2: Add the dedicated remote RED gate**

Append to `.github/workflows/proof.yml`, after the existing StoryState theorem step:

```yaml
      - name: Narrative testimony theorem tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Do not import `Testimony` from `NarrativeDynamics.lean` yet.

- [ ] **Step 3: Commit/push RED and create the draft PR**

```bash
git add NarrativeDynamics/Tests/Testimony.lean .github/workflows/proof.yml
git commit -m "test: add testimony semantics RED"
git push -u origin work/narrative-testimony-v2
```

Open a draft PR from `work/narrative-testimony-v2` to `proof/narrative-dynamics-v0`. Expected remote `proof`: conformance, full Lean build, all pre-existing theorem tests, all 311 Python tests, and StoryState theorem gate succeed; only `Narrative testimony theorem tests` fails because `NarrativeDynamics.Core.Testimony` does not exist.

- [ ] **Step 4: Implement the minimal generic Testimony module**

Create `NarrativeDynamics/Core/Testimony.lean`:

```lean
import NarrativeDynamics.Core.StoryState

namespace NarrativeDynamics

/-- A report event carries an asserted object location and identifies the
world event the speaker claims as its canonical support. -/
structure LocationReport (Agent Event Object Location : Type*) where
  reportEvent : Event
  supportEvent : Event
  speaker : Agent
  object : Object
  location : Location

/-- Canonical report support is epistemic and temporal, not a truth predicate. -/
def ReportSupported
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location) : Prop :=
  world.observed report.supportEvent report.speaker ∧
  world.eventTime report.supportEvent < world.eventTime report.reportEvent

/-- A recipient has the report only when the world records observation of the
report event itself. -/
def reportAvailableTo
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location)
    (recipient : Agent) : Prop :=
  world.observed report.reportEvent recipient

/-- Latest asserted location among reports actually received by one agent.
Authored report order is the semantic report order; Python V2 validation binds
that order to strict logical time. -/
noncomputable def reportedLocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent)
    (object : Object) : Option Location := by
  classical
  exact ((reports.filter fun report =>
    report.object = object ∧ reportAvailableTo world report recipient).getLast?).map
      LocationReport.location

/-- Appending an unreceived report cannot change the recipient's report state. -/
theorem reportedLocation_append_unreceived
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent) (object : Object)
    (report : LocationReport Agent Event Object Location)
    (hnot : ¬ reportAvailableTo world report recipient) :
    reportedLocation world (reports ++ [report]) recipient object =
      reportedLocation world reports recipient object := by
  classical
  simp [reportedLocation, reportAvailableTo, hnot]

/-- A received appended report about the queried object becomes the latest
reported location. -/
theorem reportedLocation_append_received
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (reports : List (LocationReport Agent Event Object Location))
    (recipient : Agent) (object : Object)
    (report : LocationReport Agent Event Object Location)
    (hreceived : reportAvailableTo world report recipient)
    (hobject : report.object = object) :
    reportedLocation world (reports ++ [report]) recipient object =
      some report.location := by
  classical
  simp [reportedLocation, reportAvailableTo, hreceived, hobject]

/-- Existing WorldInvariant information reachability remains authoritative for
report reception. -/
theorem received_report_has_info_path
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (report : LocationReport Agent Event Object Location)
    (recipient : Agent)
    (hreceived : reportAvailableTo world report recipient) :
    infoReachable world.info
      (world.eventNode report.reportEvent)
      (world.agentNode recipient) := by
  exact hworld.2.1 report.reportEvent recipient hreceived

/-- A report cannot satisfy canonical support when the speaker did not observe
its declared support event. -/
theorem unobserved_support_cannot_support_report
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (report : LocationReport Agent Event Object Location)
    (hnot : ¬ world.observed report.supportEvent report.speaker) :
    ¬ ReportSupported world report := by
  intro hsupported
  exact hnot hsupported.1

end NarrativeDynamics
```

If Lean syntax requires a local `classical` placement change, make only that syntax-level adjustment; do not alter the semantic surface above.

- [ ] **Step 5: Export the Lean module from the library root**

Add exactly:

```lean
import NarrativeDynamics.Core.Testimony
```

to `NarrativeDynamics.lean`, adjacent to the existing `StoryState` import.

- [ ] **Step 6: Verify Task 1 GREEN**

Run:

```bash
lake build NarrativeDynamics.Core.Testimony
lake env lean NarrativeDynamics/Tests/Testimony.lean
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake build
python3 -m unittest discover -s tests -v
```

Expected: all commands pass and Python remains exactly 311 tests at this task boundary.

- [ ] **Step 7: Commit/push GREEN and require remote proof**

```bash
git add NarrativeDynamics/Core/Testimony.lean NarrativeDynamics/Tests/Testimony.lean NarrativeDynamics.lean .github/workflows/proof.yml
git commit -m "feat: add testimony provenance semantics"
git push
```

Require remote `proof` green before Task 2.

---

### Task 2: Immutable V2 Canonical Schema and Matched Fixtures

**Files:**
- Create: `narrative_dynamics/story/schema_v2.py`
- Create: `tests/test_story_schema_v2.py`
- Create: `fixtures/stories/key_location_truthful_testimony_v2.json`
- Create: `fixtures/stories/key_location_stale_testimony_v2.json`

**Interfaces:**
- Consumes V1 public values: `StoryEntitiesV1`, `RelocationEventV1`, `DirectObservationV1`, `SearchActionV1`, `SearchDecisionV1`, plus package-private `_validate_story_semantics`, `_freeze_mapping`, `_thaw`, `_text`, `_integer`, `_exact_keys` from `story.schema` to preserve V1 structural semantics without editing the V1 module.
- Produces:
  - `LocationReportV2`
  - `ReportReceptionV2`
  - `NarrativeOracleV2`
  - `NarrativeCaseV2`
  - `load_narrative_case_v2(path) -> NarrativeCaseV2`
  - package-private `_validate_testimony_semantics(...)` reused by authored cases and runtime scenarios.

- [ ] **Step 1: Write schema V2 RED tests**

Create `tests/test_story_schema_v2.py`. Begin with exact baseline identity locks:

```python
import unittest

from narrative_dynamics.story.scenario import project_narrative_scenario
from narrative_dynamics.story.schema import load_narrative_case
from narrative_dynamics.story.schema_v2 import (
    LocationReportV2,
    NarrativeCaseV2,
    NarrativeOracleV2,
    ReportReceptionV2,
    load_narrative_case_v2,
)


class NarrativeStorySchemaV2Tests(unittest.TestCase):
    def test_v1_identity_regression_is_exact(self):
        false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )
        self.assertEqual(
            false_case.content_hash,
            "sha256:761f019271bfb097d95c7c77e80811496c22ee17089f586a433a1a8c21020420",
        )
        self.assertEqual(
            informed_case.content_hash,
            "sha256:2d63d1cc5a50a059cbcf9f425ba54275474bff4abb74ed37703514e069cff862",
        )
        self.assertEqual(
            project_narrative_scenario(false_case).id,
            "story-v1-3b8036566da885c9f07136e141fe3a03f4d80844341dd6d97d109bcbdad4b38e",
        )
        self.assertEqual(
            project_narrative_scenario(informed_case).id,
            "story-v1-ecc5c96321e624e27ea4f07146875ba4a1878e1afadc09dd1337e169478f5368",
        )
```

Add tests that require both V2 fixtures to:

- use `schema_version == 2`;
- round-trip through `to_dict()` / `from_dict()` and verify declared `content_hash`;
- have exactly identical `entities`, `events`, `observations`, `receptions`, and `decision`;
- differ in model-visible testimony semantics only at `reports[0].location` (`box` versus `drawer`);
- contain Bob observing `e1` but not `e2`, Alice observing `e2`, and Bob receiving `r1`;
- mark synthetic true, empirical human data false, population representative false.

Add focused rejection tests for:

```text
duplicate report id
undeclared speaker/object/location
support event missing
support event wrong object
support event not earlier than report
speaker did not directly observe support event
duplicate reception pair
unknown report in reception
undeclared recipient
unsupported reception channel
duplicate relocation/report logical time
report logical_time >= decision.time
unknown/missing V2 JSON keys
wrong schema_version
mismatched/missing declared content_hash
```

Also add a positive test proving schema validity does **not** require `report.location == support_event.to_location` by constructing a valid report supported by `e2` that asserts `drawer`.

- [ ] **Step 2: Verify, commit, and publish schema RED**

Run:

```bash
python3 -m unittest tests.test_story_schema_v2 -v
```

Expected: import failure only because `narrative_dynamics.story.schema_v2` does not exist.

Then:

```bash
git add tests/test_story_schema_v2.py
git commit -m "test: add testimony schema RED"
git push
```

Require remote `proof` to fail only on the new Python V2 schema import while Lean/conformance and all prior Python tests remain green up to that failure.

- [ ] **Step 3: Implement V2 frozen values and one V2 semantic validator**

Create `narrative_dynamics/story/schema_v2.py` with these exact public dataclasses:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

from narrative_dynamics.contracts import stable_content_hash
from narrative_dynamics.story.schema import (
    DirectObservationV1,
    RelocationEventV1,
    SearchDecisionV1,
    StoryEntitiesV1,
    _exact_keys,
    _freeze_mapping,
    _integer,
    _text,
    _thaw,
    _validate_story_semantics,
)


NARRATIVE_CASE_V2_SCHEMA_VERSION = 2


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


@dataclass(frozen=True)
class NarrativeOracleV2:
    objective_location: str
    actor_direct_location: str
    actor_testimony_location: str
    testimony_ranking: tuple[str, ...]
    omniscient_ranking: tuple[str, ...]


@dataclass(frozen=True)
class NarrativeCaseV2:
    name: str
    version: str
    source: Mapping[str, object]
    provenance: Mapping[str, object]
    source_text: str
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    reports: tuple[LocationReportV2, ...]
    receptions: tuple[ReportReceptionV2, ...]
    decision: SearchDecisionV1
    oracle: NarrativeOracleV2
    schema_version: int = field(default=NARRATIVE_CASE_V2_SCHEMA_VERSION)
```

Each dataclass must validate/freeze its own scalar fields in `__post_init__`, reject unsupported fixed `kind`/`channel`, provide strict `to_dict()` / `from_dict()` methods, and preserve tuple/mapping immutability exactly as V1 does.

Implement the shared V2 cross-field validator:

```python
def _validate_testimony_semantics(
    entities: StoryEntitiesV1,
    events: tuple[RelocationEventV1, ...],
    observations: tuple[DirectObservationV1, ...],
    reports: tuple[LocationReportV2, ...],
    receptions: tuple[ReportReceptionV2, ...],
    decision: SearchDecisionV1,
) -> None:
    _validate_story_semantics(
        entities,
        events,
        observations,
        decision,
        oracle=None,
    )

    if not reports:
        raise ValueError("V2 testimony story must contain at least one report")

    event_by_id = {event.id: event for event in events}
    observed_pairs = {(item.event, item.agent) for item in observations}
    occupied_times = {event.logical_time for event in events}
    report_by_id: dict[str, LocationReportV2] = {}
    previous_report_time: int | None = None

    for report in reports:
        if report.id in report_by_id:
            raise ValueError("report ids must be unique")
        if report.speaker not in entities.agents:
            raise ValueError("report speaker must reference a declared agent")
        if report.object not in entities.objects:
            raise ValueError("report object must reference a declared object")
        if report.location not in entities.locations:
            raise ValueError("report location must reference a declared location")
        support = event_by_id.get(report.support_event)
        if support is None:
            raise ValueError("report support_event must reference a declared relocation")
        if support.object != report.object:
            raise ValueError("report support_event must relocate the reported object")
        if support.logical_time >= report.logical_time:
            raise ValueError("report support_event must occur before the report")
        if (support.id, report.speaker) not in observed_pairs:
            raise ValueError("report speaker must have directly observed the support_event")
        if report.logical_time in occupied_times:
            raise ValueError("relocation and report logical times must be unique")
        if previous_report_time is not None and report.logical_time <= previous_report_time:
            raise ValueError("report logical time must be strictly increasing")
        if report.logical_time >= decision.time:
            raise ValueError("model-visible reports must occur before the decision")
        occupied_times.add(report.logical_time)
        previous_report_time = report.logical_time
        report_by_id[report.id] = report

    reception_pairs: set[tuple[str, str]] = set()
    for reception in receptions:
        if reception.report not in report_by_id:
            raise ValueError("reception report must reference a declared report")
        if reception.recipient not in entities.agents:
            raise ValueError("reception recipient must reference a declared agent")
        pair = (reception.report, reception.recipient)
        if pair in reception_pairs:
            raise ValueError("duplicate report-recipient reception is not allowed")
        reception_pairs.add(pair)
```

Do not compare `report.location` to `support.to_location`.

`NarrativeCaseV2.__post_init__` must freeze metadata, call `_validate_testimony_semantics`, and separately validate the oracle only for declared locations and exact decision-action ranking coverage. Do not trust oracle values as replay truth.

Use V1's typed stable-content-hash pattern:

```python
@property
def content_hash(self) -> str:
    return stable_content_hash(self.identity_payload())
```

`from_dict(..., verify_declared_hash=True)` requires and verifies `content_hash`; `False` skips only declared-hash verification, never structural or semantic validation.

- [ ] **Step 4: Generate the two committed V2 fixtures from production constructors**

After `schema_v2.py` exists, run a one-shot script using the production dataclasses. Use exactly these model-visible semantics in both cases:

```python
entities = StoryEntitiesV1(
    agents=("alice", "bob"),
    objects=("key",),
    locations=("drawer", "box"),
)

events = (
    RelocationEventV1(
        id="e1", logical_time=1, actor="alice", object="key",
        from_location=None, to_location="drawer",
    ),
    RelocationEventV1(
        id="e2", logical_time=2, actor="alice", object="key",
        from_location="drawer", to_location="box",
    ),
)

observations = (
    DirectObservationV1(event="e1", agent="bob"),
    DirectObservationV1(event="e2", agent="alice"),
)

receptions = (
    ReportReceptionV2(report="r1", recipient="bob"),
)

decision = SearchDecisionV1(
    id="d1",
    time=4,
    actor="bob",
    object="key",
    actions=(
        SearchActionV1(id="search_drawer", location="drawer"),
        SearchActionV1(id="search_box", location="box"),
    ),
)
```

Truthful case report:

```python
LocationReportV2(
    id="r1",
    logical_time=3,
    speaker="alice",
    object="key",
    location="box",
    support_event="e2",
)
```

Stale case report differs only in:

```python
location="drawer"
```

Use source text that describes the authored report, but remember it is fixture metadata. Use these oracle values:

```python
# truthful
NarrativeOracleV2(
    objective_location="box",
    actor_direct_location="drawer",
    actor_testimony_location="box",
    testimony_ranking=("search_box", "search_drawer"),
    omniscient_ranking=("search_box", "search_drawer"),
)

# stale
NarrativeOracleV2(
    objective_location="box",
    actor_direct_location="drawer",
    actor_testimony_location="drawer",
    testimony_ranking=("search_drawer", "search_box"),
    omniscient_ranking=("search_box", "search_drawer"),
)
```

Both fixtures use:

```python
source={"kind": "authored_microstory", "language": "en"}
provenance={
    "synthetic": True,
    "empirical_human_data": False,
    "population_representative": False,
}
version="2.0.0"
```

Write `case.to_json() + "\n"`; never hand-author the declared content hash.

- [ ] **Step 5: Verify schema/fixtures GREEN and V1 identity locks**

Run:

```bash
python3 -m unittest tests.test_story_schema_v2 -v
python3 -m unittest tests.test_story_schema -v
python3 -m unittest tests.test_story_scenario -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
```

Expected: all tests pass, both V1 hashes/IDs equal the four exact constants in Global Constraints, and the total Python count is greater than 311.

- [ ] **Step 6: Commit/push schema GREEN**

```bash
git add narrative_dynamics/story/schema_v2.py tests/test_story_schema_v2.py fixtures/stories/key_location_truthful_testimony_v2.json fixtures/stories/key_location_stale_testimony_v2.json
git commit -m "feat: add canonical testimony V2 schema"
git push
```

Require remote `proof` green before Task 3.

---

### Task 3: V2 Anti-Leakage Runtime Scenario Projection

**Files:**
- Create: `narrative_dynamics/story/scenario_v2.py`
- Create: `tests/test_story_scenario_v2.py`

**Interfaces:**
- Consumes: `NarrativeCaseV2`, V1 base schema dataclasses, V2 reports/receptions, `_validate_testimony_semantics`, `contracts.Scenario`, `stable_content_hash`.
- Produces:
  - `NarrativeScenarioV2`
  - `project_testimony_scenario(case: NarrativeCaseV2) -> Scenario`
  - `decode_testimony_scenario(scenario: Scenario) -> NarrativeScenarioV2`

- [ ] **Step 1: Write V2 scenario RED tests**

Create `tests/test_story_scenario_v2.py` with core assertions:

```python
class NarrativeScenarioProjectionV2Tests(unittest.TestCase):
    def setUp(self):
        self.truthful = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        self.stale = load_narrative_case_v2(
            "fixtures/stories/key_location_stale_testimony_v2.json"
        )

    def test_projection_contains_exact_v2_visible_keys(self):
        scenario = project_testimony_scenario(self.truthful)
        self.assertEqual(
            set(scenario.payload),
            {"entities", "events", "observations", "reports", "receptions", "decision"},
        )

    def test_runtime_id_is_v2_payload_digest_only(self):
        scenario = project_testimony_scenario(self.truthful)
        digest = stable_content_hash(scenario.payload).removeprefix("sha256:")
        self.assertEqual(scenario.id, f"story-v2-{digest}")
```

Add tests proving:

- fixture `name`, `version`, `source`, `provenance`, `source_text`, and oracle changes leave payload/ID unchanged;
- serialized payload and ID contain none of `oracle`, `source_text`, `truthful`, `false`, `stale`, `correct`, `gold`, `agent_belief_ranking`, `testimony_ranking`, `omniscient_ranking`;
- changing only `reports[0].location` changes payload and runtime ID;
- decoder round-trips canonical V2 scenarios;
- decoder rejects extra/missing visible keys and mismatched/non-neutral IDs;
- manually constructed payload without Alice's direct observation of `e2` fails speaker-support validation;
- unknown reception report and duplicate reception fail validation;
- report at/after decision fails validation;
- V1 `decode_narrative_scenario` rejects a V2 six-key scenario;
- V2 `decode_testimony_scenario` rejects a V1 four-key scenario.

- [ ] **Step 2: Verify, commit, and publish scenario RED**

```bash
python3 -m unittest tests.test_story_scenario_v2 -v
git add tests/test_story_scenario_v2.py
git commit -m "test: add testimony scenario leakage RED"
git push
```

Expected failure: missing `narrative_dynamics.story.scenario_v2` only.

- [ ] **Step 3: Implement one validated V2 model-facing scenario type**

Create `narrative_dynamics/story/scenario_v2.py`:

```python
@dataclass(frozen=True)
class NarrativeScenarioV2:
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    reports: tuple[LocationReportV2, ...]
    receptions: tuple[ReportReceptionV2, ...]
    decision: SearchDecisionV1

    def __post_init__(self) -> None:
        events = tuple(self.events)
        observations = tuple(self.observations)
        reports = tuple(self.reports)
        receptions = tuple(self.receptions)
        _validate_testimony_semantics(
            self.entities,
            events,
            observations,
            reports,
            receptions,
            self.decision,
        )
        object.__setattr__(self, "events", events)
        object.__setattr__(self, "observations", observations)
        object.__setattr__(self, "reports", reports)
        object.__setattr__(self, "receptions", receptions)
```

`from_case()` copies only the six model-visible semantic fields. `to_payload()` returns exactly those six fields. `from_payload()` reconstructs V1/V2 typed dataclasses and therefore executes the same `_validate_testimony_semantics` through `__post_init__`.

Use this neutral V2 ID only:

```python
def _runtime_id_v2(payload: Mapping[str, object]) -> str:
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return f"story-v2-{digest}"
```

Projection and decode:

```python
def project_testimony_scenario(case: NarrativeCaseV2) -> Scenario:
    story = NarrativeScenarioV2.from_case(case)
    payload = story.to_payload()
    return Scenario(id=_runtime_id_v2(payload), payload=payload)


def decode_testimony_scenario(scenario: Scenario) -> NarrativeScenarioV2:
    if not isinstance(scenario, Scenario):
        raise TypeError("testimony model scenario must be a Scenario")
    story = NarrativeScenarioV2.from_payload(scenario.payload)
    canonical_payload = story.to_payload()
    if stable_content_hash(scenario.payload) != stable_content_hash(canonical_payload):
        raise ValueError("testimony scenario payload is not canonical")
    if scenario.id != _runtime_id_v2(canonical_payload):
        raise ValueError("testimony scenario id must match the visible payload digest")
    return story
```

Do not expose a validation bypass flag.

- [ ] **Step 4: Verify and commit scenario GREEN**

```bash
python3 -m unittest tests.test_story_schema_v2 -v
python3 -m unittest tests.test_story_scenario_v2 -v
python3 -m unittest tests.test_story_scenario -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
git add narrative_dynamics/story/scenario_v2.py tests/test_story_scenario_v2.py
git commit -m "feat: add leak-safe testimony runtime projection"
git push
```

Require remote `proof` green before Task 4.

---

### Task 4: Pure Testimony-Aware Epistemic Replay

**Files:**
- Create: `narrative_dynamics/story/replay_v2.py`
- Create: `tests/test_story_replay_v2.py`

**Interfaces:**
- Consumes: validated `NarrativeScenarioV2`.
- Produces:
  - `EpistemicLocationStateV2`
  - `testimony_state(story, agent, *, at_time=None) -> Mapping[str, EpistemicLocationStateV2]`
  - `latest_epistemic_location(state, object_id) -> EpistemicLocationStateV2`

- [ ] **Step 1: Write replay RED tests**

Create `tests/test_story_replay_v2.py`:

```python
class TestimonyReplayV2Tests(unittest.TestCase):
    def setUp(self):
        truthful = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        stale = load_narrative_case_v2(
            "fixtures/stories/key_location_stale_testimony_v2.json"
        )
        self.truthful = NarrativeScenarioV2.from_case(truthful)
        self.stale = NarrativeScenarioV2.from_case(stale)

    def test_truthful_received_report_updates_bob_to_box(self):
        state = latest_epistemic_location(
            testimony_state(self.truthful, "bob"), "key"
        )
        self.assertEqual(
            (
                state.location,
                state.evidence_kind,
                state.supporting_id,
                state.source_agent,
                state.logical_time,
            ),
            ("box", "testimony", "r1", "alice", 3),
        )

    def test_stale_received_report_updates_bob_to_drawer(self):
        state = latest_epistemic_location(
            testimony_state(self.stale, "bob"), "key"
        )
        self.assertEqual(
            (state.location, state.evidence_kind, state.supporting_id),
            ("drawer", "testimony", "r1"),
        )
```

Add tests proving:

- V1 `subjective_state(events, observations, "bob")` is still drawer in both V2 cases;
- `objective_state(events)` is box in both V2 cases;
- removing Bob's reception of `r1` leaves testimony-aware state at direct `e1`/drawer with `evidence_kind="direct_perception"`, `supporting_id="e1"`, `source_agent="bob"`;
- `at_time=2` returns direct drawer and `at_time=3` includes the report;
- a report received by another agent does not update Bob;
- `latest_epistemic_location({}, "key")` fails closed rather than using objective state;
- invalid `at_time` and invalid object IDs fail closed.

- [ ] **Step 2: Verify, commit, and publish replay RED**

```bash
python3 -m unittest tests.test_story_replay_v2 -v
git add tests/test_story_replay_v2.py
git commit -m "test: add testimony replay RED"
git push
```

Expected failure: missing `narrative_dynamics.story.replay_v2` only.

- [ ] **Step 3: Implement replay over validated V2 scenarios**

Create `narrative_dynamics/story/replay_v2.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from narrative_dynamics.story.scenario_v2 import NarrativeScenarioV2


@dataclass(frozen=True)
class EpistemicLocationStateV2:
    object_id: str
    location: str
    evidence_kind: str
    supporting_id: str
    source_agent: str
    logical_time: int


def _at_time(value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("testimony replay at_time must be an integer or None")
    if value < 0:
        raise ValueError("testimony replay at_time must be non-negative")
    return value


def testimony_state(
    story: NarrativeScenarioV2,
    agent: str,
    *,
    at_time: int | None = None,
) -> Mapping[str, EpistemicLocationStateV2]:
    if not isinstance(story, NarrativeScenarioV2):
        raise TypeError("testimony replay requires a validated NarrativeScenarioV2")
    if not isinstance(agent, str) or not agent:
        raise ValueError("testimony replay agent must be a non-empty string")
    limit = _at_time(at_time)

    evidence: list[EpistemicLocationStateV2] = []
    event_by_id = {event.id: event for event in story.events}
    for observation in story.observations:
        if observation.agent != agent:
            continue
        event = event_by_id[observation.event]
        if limit is not None and event.logical_time > limit:
            continue
        evidence.append(EpistemicLocationStateV2(
            object_id=event.object,
            location=event.to_location,
            evidence_kind="direct_perception",
            supporting_id=event.id,
            source_agent=agent,
            logical_time=event.logical_time,
        ))

    report_by_id = {report.id: report for report in story.reports}
    for reception in story.receptions:
        if reception.recipient != agent:
            continue
        report = report_by_id[reception.report]
        if limit is not None and report.logical_time > limit:
            continue
        evidence.append(EpistemicLocationStateV2(
            object_id=report.object,
            location=report.location,
            evidence_kind="testimony",
            supporting_id=report.id,
            source_agent=report.speaker,
            logical_time=report.logical_time,
        ))

    state: dict[str, EpistemicLocationStateV2] = {}
    for item in sorted(evidence, key=lambda value: value.logical_time):
        state[item.object_id] = item
    return MappingProxyType(state)


def latest_epistemic_location(
    state: Mapping[str, EpistemicLocationStateV2],
    object_id: str,
) -> EpistemicLocationStateV2:
    if not isinstance(object_id, str) or not object_id:
        raise ValueError("testimony replay object id must be a non-empty string")
    try:
        value = state[object_id]
    except KeyError as error:
        raise ValueError(f"no supported testimony-aware location for object {object_id!r}") from error
    if not isinstance(value, EpistemicLocationStateV2):
        raise TypeError("testimony replay state must contain EpistemicLocationStateV2 values")
    return value
```

Do not accept raw unvalidated report/reception arrays; the validated `NarrativeScenarioV2` input is the replay provenance boundary.

- [ ] **Step 4: Verify and commit replay GREEN**

```bash
python3 -m unittest tests.test_story_replay_v2 -v
python3 -m unittest tests.test_story_replay -v
python3 -m unittest tests.test_story_scenario_v2 -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
git add narrative_dynamics/story/replay_v2.py tests/test_story_replay_v2.py
git commit -m "feat: add testimony-aware epistemic replay"
git push
```

Require remote `proof` green before Task 5.

---

### Task 5: Parameter-Free Testimony Search Model and Three-Model Contrast

**Files:**
- Create: `narrative_dynamics/adapters/story_testimony_search.py`
- Create: `tests/test_story_testimony_models.py`

**Interfaces:**
- Consumes: `decode_testimony_scenario`, `testimony_state`, `latest_epistemic_location`, V1 `NarrativeScenarioV1` only in tests for direct/omniscient compatibility witnesses.
- Produces: `TestimonySearchModel`, name exactly `testimony-search`.

Model outcome contract:

```text
epistemic_basis = {
  "kind": "testimony" | "direct_perception",
  "target_object": str,
  "resolved_location": str,
  "supporting_id": str,
  "source_agent": str
}
action_scores = {action_id: 0.0 | 1.0}
policy = {action_id: 0.0 | 1.0}
selected_action = action_id
```

- [ ] **Step 1: Write testimony-model RED tests**

Create `tests/test_story_testimony_models.py`. Test the matched pair:

```python
class TestimonyModelTests(unittest.TestCase):
    def setUp(self):
        self.truthful = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        self.stale = load_narrative_case_v2(
            "fixtures/stories/key_location_stale_testimony_v2.json"
        )

    def test_testimony_model_follows_received_report_content(self):
        model = TestimonySearchModel()
        truthful = model.simulate(
            project_testimony_scenario(self.truthful), {}, random.Random(7)
        )
        stale = model.simulate(
            project_testimony_scenario(self.stale), {}, random.Random(7)
        )
        self.assertEqual(truthful.outcome["selected_action"], "search_box")
        self.assertEqual(stale.outcome["selected_action"], "search_drawer")
        self.assertEqual(truthful.outcome["epistemic_basis"], {
            "kind": "testimony",
            "target_object": "key",
            "resolved_location": "box",
            "supporting_id": "r1",
            "source_agent": "alice",
        })
```

Add tests that:

- exact action-score/policy keys are `search_drawer` and `search_box`, probabilities are one-hot and normalized;
- non-empty parameters raise `ValueError` containing `empty parameter`;
- an unreceived-report V2 scenario uses Bob's direct `e1` evidence and selects drawer with basis kind `direct_perception`;
- a valid scenario whose resolved private location has no matching decision action fails closed;
- the model does not import/reference prison adapters.

For the three-model compatibility witness, construct a V1 four-key view in test code only:

```python
def _direct_v1_view(case: NarrativeCaseV2) -> Scenario:
    story = NarrativeScenarioV1(
        entities=case.entities,
        events=case.events,
        observations=case.observations,
        decision=case.decision,
    )
    payload = story.to_payload()
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return Scenario(id=f"story-v1-{digest}", payload=payload)
```

Then assert for both V2 cases:

```text
AgentBeliefSearchModel on direct V1 view -> search_drawer
OmniscientSearchModel on direct V1 view -> search_box
```

This demonstrates the contrast without modifying either V1 adapter or teaching V1 decoders to accept V2 payloads.

- [ ] **Step 2: Verify, commit, and publish model RED**

```bash
python3 -m unittest tests.test_story_testimony_models -v
git add tests/test_story_testimony_models.py
git commit -m "test: add testimony search model RED"
git push
```

Expected failure: missing `narrative_dynamics.adapters.story_testimony_search` only.

- [ ] **Step 3: Implement deterministic testimony search**

Create `narrative_dynamics/adapters/story_testimony_search.py`:

```python
from __future__ import annotations

from collections.abc import Mapping
import random

from narrative_dynamics.contracts import ModelRun, Scenario, TraceEvent
from narrative_dynamics.story.replay_v2 import (
    latest_epistemic_location,
    testimony_state,
)
from narrative_dynamics.story.scenario_v2 import decode_testimony_scenario


class TestimonySearchModel:
    name = "testimony-search"

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        if parameters:
            raise ValueError("testimony-search requires an empty parameter mapping")

        story = decode_testimony_scenario(scenario)
        decision = story.decision
        state = testimony_state(story, decision.actor, at_time=decision.time)
        location = latest_epistemic_location(state, decision.object)
        matching = [
            action.id
            for action in decision.actions
            if action.location == location.location
        ]
        if len(matching) != 1:
            raise ValueError(
                "private target location must match exactly one decision action"
            )

        selected = matching[0]
        scores = {
            action.id: 1.0 if action.id == selected else 0.0
            for action in decision.actions
        }
        policy = dict(scores)
        basis = {
            "kind": location.evidence_kind,
            "target_object": decision.object,
            "resolved_location": location.location,
            "supporting_id": location.supporting_id,
            "source_agent": location.source_agent,
        }

        return ModelRun(
            events=(
                TraceEvent(tick=0, kind="epistemic_basis_resolved", data=basis),
                TraceEvent(tick=1, kind="action_policy_computed", data={"policy": policy}),
                TraceEvent(tick=2, kind="action_selected", data={"action": selected}),
            ),
            outcome={
                "epistemic_basis": basis,
                "action_scores": scores,
                "policy": policy,
                "selected_action": selected,
            },
        )
```

The RNG argument is intentionally unused. Do not add reliability weights, stochastic trust, tie-breaking, or any hidden parameter.

- [ ] **Step 4: Verify and commit model GREEN**

```bash
python3 -m unittest tests.test_story_testimony_models -v
python3 -m unittest tests.test_story_models -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
git add narrative_dynamics/adapters/story_testimony_search.py tests/test_story_testimony_models.py
git commit -m "feat: add testimony-sensitive story search model"
git push
```

Require remote `proof` green before Task 6.

---

### Task 6: Trusted Runtime Acceptance and Public V2 Story API

**Files:**
- Modify: `narrative_dynamics/story/__init__.py`
- Create: `tests/test_story_testimony_runtime.py`

**Interfaces:**
- Consumes: V2 schema/scenario/replay, `TestimonySearchModel`, existing `SimulationRunner`, unchanged `story_choice_metrics`.
- Produces these additional public `narrative_dynamics.story` names exactly:
  - `LocationReportV2`
  - `ReportReceptionV2`
  - `NarrativeOracleV2`
  - `NarrativeCaseV2`
  - `load_narrative_case_v2`
  - `NarrativeScenarioV2`
  - `project_testimony_scenario`
  - `decode_testimony_scenario`
  - `EpistemicLocationStateV2`
  - `testimony_state`
  - `latest_epistemic_location`

Do not export `TestimonySearchModel` from `narrative_dynamics.story` or package root.

- [ ] **Step 1: Write trusted-runtime/public-API RED tests**

Create `tests/test_story_testimony_runtime.py`:

```python
class TestimonyRuntimeTests(unittest.TestCase):
    def test_testimony_model_runs_through_simulation_runner(self):
        case = load_narrative_case_v2(
            "fixtures/stories/key_location_truthful_testimony_v2.json"
        )
        scenario = project_testimony_scenario(case)
        traces = SimulationRunner().run_batch(
            TestimonySearchModel(), scenario, {}, seeds=(101, 102)
        )
        self.assertEqual(
            [trace.outcome["selected_action"] for trace in traces],
            ["search_box", "search_box"],
        )
        for trace in traces:
            self.assertEqual(trace.parameters, ())
            self.assertIsNotNone(trace.manifest)
            self.assertEqual(trace.scenario_id, scenario.id)

    def test_common_story_metrics_accept_testimony_trace(self):
        case = load_narrative_case_v2(
            "fixtures/stories/key_location_stale_testimony_v2.json"
        )
        trace = SimulationRunner().run_once(
            TestimonySearchModel(),
            project_testimony_scenario(case),
            {},
            seed=1,
        )
        self.assertEqual(story_choice_metrics(trace), {
            "choice.search_box": 0.0,
            "choice.search_drawer": 1.0,
        })
```

Also test:

- seed 7 and seed 999 produce identical outcomes but distinct trace seeds;
- all eleven additional V2 names are in `narrative_dynamics.story.__all__` and available as attributes;
- every V1 public story name from Task 7 of V1 remains exported;
- none of the new V2 story names are exported from `narrative_dynamics` package root;
- `TestimonySearchModel` is not exported from `narrative_dynamics.story` or root;
- source text of all three story adapters contains no `prison_pomdp` or `prison_reactive` reference.

- [ ] **Step 2: Verify, commit, and publish runtime/API RED**

```bash
python3 -m unittest tests.test_story_testimony_runtime -v
git add tests/test_story_testimony_runtime.py
git commit -m "test: add trusted testimony runtime RED"
git push
```

Expected: runtime/model/metrics tests pass via module imports, but public-API assertions fail because `narrative_dynamics.story.__init__` has not yet exported V2 canonical names.

- [ ] **Step 3: Append exact canonical V2 exports**

Modify only `narrative_dynamics/story/__init__.py`. Preserve every existing V1 import/export and append imports from `schema_v2`, `scenario_v2`, and `replay_v2`. The final `__all__` is the V1 list plus exactly these eleven V2 names:

```python
"LocationReportV2",
"ReportReceptionV2",
"NarrativeOracleV2",
"NarrativeCaseV2",
"load_narrative_case_v2",
"NarrativeScenarioV2",
"project_testimony_scenario",
"decode_testimony_scenario",
"EpistemicLocationStateV2",
"testimony_state",
"latest_epistemic_location",
```

Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 4: Verify Task 6 GREEN with all V1/V2 targeted suites**

Run:

```bash
python3 -m unittest tests.test_story_schema -v
python3 -m unittest tests.test_story_replay -v
python3 -m unittest tests.test_story_scenario -v
python3 -m unittest tests.test_story_models -v
python3 -m unittest tests.test_story_runtime -v
python3 -m unittest tests.test_story_schema_v2 -v
python3 -m unittest tests.test_story_scenario_v2 -v
python3 -m unittest tests.test_story_replay_v2 -v
python3 -m unittest tests.test_story_testimony_models -v
python3 -m unittest tests.test_story_testimony_runtime -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Expected: all pass; complete Python count is greater than 311; all V1 identity lock assertions remain exact.

- [ ] **Step 5: Commit/push trusted runtime GREEN**

```bash
git add narrative_dynamics/story/__init__.py tests/test_story_testimony_runtime.py
git commit -m "feat: exercise testimony stories through trusted runtime"
git push
```

Require remote `proof` green before Task 7.

---

### Task 7: Final Evidence, Review, and Research-Branch Integration

**Files:**
- Review complete diff against the exact pre-V2 research SHA `e56331f8924c96308e671eda4d7238d6a1f16fe7`.
- No production change is expected unless review finds a defect; every defect requires a focused RED before a fix.

**Interfaces:**
- Consumes: exact final V2 feature head and draft PR evidence.
- Produces: reviewed green V2 research head and a non-forced fast-forward of `proof/narrative-dynamics-v0` only.

- [ ] **Step 1: Run fresh complete verification on the exact final feature head**

Run the same gates as CI plus compile validation:

```bash
lake build \
  NarrativeDynamics.Core.Belief \
  NarrativeDynamics.Core.Drive \
  NarrativeDynamics.Core.Learning
mkdir -p .generated-conformance
lake env lean --run NarrativeDynamics/Conformance/ReferenceVectors.lean \
  > .generated-conformance/lean_reference_vectors.json
cmp conformance/lean_reference_vectors.json \
  .generated-conformance/lean_reference_vectors.json
lake build
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake env lean NarrativeDynamics/Tests/Testimony.lean
```

Also execute every pre-existing Lean theorem command listed in `.github/workflows/proof.yml`; do not substitute only `lake build` for theorem tests.

- [ ] **Step 2: Require a fresh remote final proof on the exact feature SHA**

The final `proof` run must show:

```text
Lean↔Python conformance             success
full Lean library build            success
all pre-existing Lean theorem tests success
complete Python suite              success, count > 311
Narrative StoryState theorem tests success
Narrative testimony theorem tests  success
```

Record exact feature SHA, workflow run number/ID, and final Python test count.

- [ ] **Step 3: Review the exact diff against V1**

Confirm all of the following:

- only one new Lean semantic module/test plus root import/workflow hook;
- `NarrativeDynamics/Core/WorldGraph.lean` and `TypedGraph.lean` are unchanged;
- no new `NodeKind`; `TypedEdge.locatedAt` remains unchanged;
- V2 Lean uses `WorldGraph.eventTime`, `WorldGraph.observed`, and `WorldInvariant` information reachability; no parallel clock or observation relation;
- V1 Python files `schema.py`, `replay.py`, `scenario.py`, `story_belief_search.py`, and `story_omniscient_search.py` are unchanged from base;
- V1 fixture content hashes equal the two exact constants in Global Constraints;
- V1 runtime scenario IDs equal the two exact constants in Global Constraints;
- V2 authored and runtime values call the same `_validate_testimony_semantics`;
- V2 runtime decoder cannot bypass speaker-observation support, report/reception references, time ordering, or exact-key validation;
- report location is never compared against support-event objective destination during validity checks;
- reports/receptions never enter V1 objective replay;
- direct V1 `subjective_state` remains direct-perception-only;
- testimony replay requires validated `NarrativeScenarioV2` and has no objective fallback;
- truthful/stale fixtures have identical model-visible semantics except report `location`;
- Bob observes `e1` only among target relocations, Alice observes `e2`, Bob receives `r1` in both;
- `TestimonySearchModel` is parameter-free and deterministic;
- V1 AgentBelief remains drawer on the direct view, Omniscient remains box, Testimony is box/drawer across the pair;
- V2 scenario payload has exactly six keys and neutral `story-v2-*` ID derived only from visible payload;
- fixture source/oracle/name/version/provenance and truth-label words never enter runtime payload/ID;
- `story_choice_metrics` and simulation/runtime infrastructure are unchanged;
- `narrative_dynamics.story` adds canonical V2 exports only; package root remains unchanged; no story model is root-exported;
- no prison adapter, observation-protocol, registry/runtime, dependency/lockfile, calibration, training, selection, final-test, or empirical-human-data code changed;
- no claim of deception, source trust, unrestricted NLP understanding, population behavior, or general theory of mind was added.

If review finds a defect: write focused RED → observe intended failure → make smallest fix → rerun all gates → obtain a new final proof run.

- [ ] **Step 4: Recheck ancestry immediately before integration**

Compare:

```text
base = proof/narrative-dynamics-v0
head = work/narrative-testimony-v2
```

Require `status=ahead`, `behind_by == 0`, and merge-base equal to the current research-base SHA. If the base moved independently, stop integration, re-evaluate the merge context, and rerun proof on the exact resulting head. Never force-update.

- [ ] **Step 5: Fast-forward only the research branch**

Move `proof/narrative-dynamics-v0` to the exact reviewed V2 SHA with `force=false`.

Do not move `master`.

- [ ] **Step 6: Verify identical refs after integration**

Compare feature and research refs and require:

```text
status = identical
ahead_by = 0
behind_by = 0
```

Fetch `master` and record its SHA to prove it was not moved by this work.

- [ ] **Step 7: Close feature metadata with exact evidence**

Update the V2 PR title/body to include:

- communicated testimony as a distinct evidence path from direct perception;
- explicit speaker support event and recipient reception provenance;
- truthful/stale matched pair with identical objective history and only report content differing model-visibly;
- unchanged V1 direct and omniscient baselines;
- parameter-free `testimony-search`;
- anti-leakage six-key V2 projection and neutral payload-derived ID;
- exact final SHA, proof run, and Python test count;
- exact V1 fixture hash/scenario-ID regression evidence;
- non-forced research-branch fast-forward and unchanged `master`;
- limits: authored canonical inputs, no NLP parsing, no learned trust/reliability, no empirical human data, no deception claim, no population/general theory-of-mind claim.

---

## Final Acceptance Checklist

- [ ] V1 fixture hashes are unchanged and equal the two exact baseline hashes.
- [ ] V1 runtime scenario IDs are unchanged and equal the two exact baseline IDs.
- [ ] V1 schema/replay/scenario implementation files and both V1 model adapters are unchanged.
- [ ] Two V2 fixtures validate, round-trip, and verify declared hashes.
- [ ] V2 fixtures have identical entities, relocations, direct observations, receptions, decision, and report metadata except asserted report location.
- [ ] Bob observes `e1` but not `e2`; Alice observes `e2`; Bob receives `r1`.
- [ ] V2 report support is invalid if Alice did not observe `e2`.
- [ ] A valid stale report can assert drawer while `e2` objectively moves the key to box.
- [ ] Objective state is key@box in both V2 cases.
- [ ] V1 direct-perception Bob state is key@drawer in both V2 cases.
- [ ] Testimony-aware Bob state is key@box for truthful and key@drawer for stale testimony.
- [ ] Removing Bob's reception makes the report irrelevant and leaves direct drawer support.
- [ ] Lean proves unreceived no-op, received update, speaker-support observation, report-reception information path, and admissibility/truth independence witness.
- [ ] `TestimonySearchModel` predicts box versus drawer; AgentBelief predicts drawer on both direct views; Omniscient predicts box on both direct views.
- [ ] All three story models reject non-empty parameter mappings.
- [ ] `story_choice_metrics` works unchanged on testimony traces.
- [ ] V2 model runs through existing `SimulationRunner` with manifests, deterministic replay, and empty canonical parameters.
- [ ] V2 payload contains exactly six model-visible keys and scenario ID is `story-v2-<payload digest>`.
- [ ] No oracle/source/provenance/gold/truth-label field leaks into runtime input or ID.
- [ ] Manually constructed malformed V2 `Scenario` values cannot bypass V2 semantic validation.
- [ ] `narrative_dynamics.story` exports canonical V2 values/replay only; package root and model exports remain unchanged.
- [ ] No prison, observation-protocol, registry/runtime, dependency, calibration, training, or empirical-evaluation semantics changed.
- [ ] Complete Python test count is greater than 311 and all remote proof gates are green.
- [ ] Final feature SHA is reviewed, research branch is fast-forwarded with `force=false`, feature/research refs are identical, and `master` is unchanged.
