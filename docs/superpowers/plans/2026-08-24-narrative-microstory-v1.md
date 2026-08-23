# Narrative Microstory V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first non-prison canonical story domain, prove objective-versus-subjective relocation semantics in Lean, and run two parameter-free competing search models through the existing trusted Python runtime on a matched false-belief/informed microstory pair.

**Architecture:** Add one generic Lean `StoryState` layer on top of the existing `WorldGraph` observation boundary. Add a Python `narrative_dynamics.story` package for authored schema, pure replay, anti-leakage runtime projection, and common metrics; keep `agent-belief-search` and `omniscient-search` as separate adapters that consume only projected canonical scenarios.

**Tech Stack:** Lean 4.32 + Mathlib; Python 3 standard library only; `unittest`; existing `narrative_dynamics` contracts, `stable_content_hash`, `SimulationRunner`, manifests, and metric infrastructure; GitHub Actions `proof` workflow.

**Spec:** `docs/superpowers/specs/2026-08-24-narrative-microstory-v1-design.md`

## Global Constraints

- Implementation base is `proof/narrative-dynamics-v0`; implementation work uses isolated branch `work/narrative-microstory-v1`.
- Do not merge or move `master` in this plan. Final integration, if green, is only a non-forced fast-forward of `proof/narrative-dynamics-v0`.
- Follow strict RED → verify RED → minimal GREEN → verify GREEN → commit for every production increment.
- Preserve observable RED evidence in the draft feature PR before the matching GREEN fix; unrelated existing behavior must stay green.
- V1 supports only authored canonical stories with `Agent`, `Object`, `Location`, `relocate_object`, `direct_perception`, and `search_object`.
- V1 is parameter-free. Both story models accept exactly an empty parameter mapping; do not add `beta`, calibration, training, selection, or final-test stages.
- Do not add NLP parsing, coreference, dialogue, testimony, deception, trust, noisy perception, Bayesian story confidence, memory, nested belief, emotion, personality, social norms, arbitrary planning, or a generic fluent language.
- Do not modify prison model semantics or import prison adapters from story code.
- Do not add dependencies or dependency/lock files.
- Do not add a new `NodeKind`; do not change existing `TypedEdge.locatedAt` semantics.
- Reuse `WorldGraph.observed`, `WorldInvariant`, and `ObservationEvidenceAdmissible`; do not define a second observation semantics in Lean.
- Python canonical `logical_time` maps to `WorldGraph.eventTime(event)` in the Lean instantiation. Lean does not introduce a second duplicate time field.
- Runtime `Scenario.payload` visible to models contains exactly `entities`, `events`, `observations`, and `decision`.
- Runtime `Scenario.id` is neutral and derived only from the model-visible payload digest. It must not contain fixture name/version, `false-belief`, `informed`, oracle, source text, provenance, or gold labels.
- `source_text`, fixture `name/version`, `source/provenance`, oracle, gold ranking/choice, and any post-decision continuation never enter model-visible input.
- Cross-field canonical validation is shared between authored fixtures and decoded runtime scenarios; a manually constructed `Scenario` may not bypass entity/reference/time/relocation/observation/decision invariants.
- `narrative_dynamics.story` is the public story surface. Do not add story APIs to package root `narrative_dynamics.__init__` in V1.
- Model-specific APIs remain module-scoped under `narrative_dynamics.adapters`.
- Both models must run through the existing `SimulationRunner`; no bespoke runner or trust boundary is added.
- The committed fixtures are synthetic authored examples only and must explicitly deny empirical-human-data and population-representativeness claims.
- Current pre-feature executable baseline is 269 Python tests. Final count must be greater than 269 and the complete suite must be green.

---

## File Structure

### Lean

- Create `NarrativeDynamics/Core/StoryState.lean` — generic temporal object-location fluent replay over existing `WorldGraph.observed`.
- Create `NarrativeDynamics/Tests/StoryState.lean` — generic and concrete false-belief/informed theorem obligations.
- Modify `NarrativeDynamics.lean` — import `NarrativeDynamics.Core.StoryState` after the new module exists.
- Modify `.github/workflows/proof.yml` — add a dedicated StoryState theorem step after the existing Python step so deliberate Lean RED still proves every pre-existing Lean/Python gate green.

### Python story domain

- Create `narrative_dynamics/story/__init__.py` — public V1 schema/projection/replay/metric exports only.
- Create `narrative_dynamics/story/schema.py` — immutable authored story schema, shared cross-field semantic validator, JSON round-trip, declared content-hash verification.
- Create `narrative_dynamics/story/replay.py` — pure objective/subjective relocation projection and support-event identity.
- Create `narrative_dynamics/story/scenario.py` — fixture-to-runtime anti-leakage projection plus strict model-facing scenario decoding using the same semantic validator as fixtures.
- Create `narrative_dynamics/story/metrics.py` — common two-action policy metric extractor.

### Python model adapters

- Create `narrative_dynamics/adapters/story_belief_search.py` — parameter-free actor-subjective search model.
- Create `narrative_dynamics/adapters/story_omniscient_search.py` — parameter-free objective-world baseline.

### Fixtures

- Create `fixtures/stories/key_location_false_belief_v1.json`.
- Create `fixtures/stories/key_location_informed_v1.json`.

### Tests

- Create `tests/test_story_schema.py`.
- Create `tests/test_story_replay.py`.
- Create `tests/test_story_scenario.py`.
- Create `tests/test_story_models.py`.
- Create `tests/test_story_runtime.py`.

Do not modify `narrative_dynamics/contracts.py`, `narrative_dynamics/simulation.py`, observation protocol modules, prison adapters, loss implementations, or model-comparison code unless a failing pre-existing contract proves a genuine incompatibility. If that happens, stop the current task, invoke `superpowers:systematic-debugging`, and add a focused RED test before touching shared runtime code.

---

### Task 1: Lean Story-State Semantics and False-Belief Theorems

**Files:**
- Create: `NarrativeDynamics/Tests/StoryState.lean`
- Create: `NarrativeDynamics/Core/StoryState.lean`
- Modify: `.github/workflows/proof.yml`
- Modify: `NarrativeDynamics.lean`

**Interfaces:**
- Consumes: existing `WorldGraph Agent Event Node`, `WorldInvariant`, `w.observed`, `w.eventTime`, `infoReachable`, and existing observation-admission theorems.
- Produces:
  - `StoryRelocation Event Object Location`
  - `applyStoryRelocation`
  - `objectiveLocation`
  - `subjectiveLocation`
  - `StoryHistoryCompatible`
  - `objectiveLocation_append_relocation`
  - `subjectiveLocation_append_unobserved`
  - `subjectiveLocation_append_observed_relocation`
  - `observed_story_event_has_info_path`
  - `no_info_path_story_event_unobserved`

- [ ] **Step 1: Start implementation in an isolated workspace**

At execution time use `superpowers:using-git-worktrees` before editing, based on the current `proof/narrative-dynamics-v0` head, and create branch:

```text
work/narrative-microstory-v1
```

Do not base it on `master`.

- [ ] **Step 2: Create the deliberate Lean RED theorem test**

Create `NarrativeDynamics/Tests/StoryState.lean` before the production module:

```lean
import NarrativeDynamics.Core.StoryState

namespace NarrativeDynamics.Tests.StoryState

inductive Agent where
  | bob
  deriving DecidableEq, Repr

inductive Event where
  | e1
  | e2
  deriving DecidableEq, Repr

inductive Object where
  | key
  deriving DecidableEq, Repr

inductive Location where
  | drawer
  | box
  deriving DecidableEq, Repr

inductive Node where
  | event1
  | event2
  | bob
  deriving DecidableEq, Repr

open Agent Event Object Location

private def eventNode : Event → Node
  | .e1 => .event1
  | .e2 => .event2

private def agentNode : Agent → Node
  | .bob => .bob

private def eventTime : Event → Nat
  | .e1 => 1
  | .e2 => 2

private def falseBeliefWorld : WorldGraph Agent Event Node := {
  alive := fun _ => True
  canAct := fun _ => True
  info := fun _ _ => True
  eventNode := eventNode
  agentNode := agentNode
  observed := fun e a => e = .e1 ∧ a = .bob
  causal := fun _ _ => False
  eventTime := eventTime
}

private def informedWorld : WorldGraph Agent Event Node := {
  falseBeliefWorld with
  observed := fun e a => (e = .e1 ∨ e = .e2) ∧ a = .bob
}

private def history : List (StoryRelocation Event Object Location) := [
  { event := .e1, object := .key, fromLocation := none, toLocation := .drawer },
  { event := .e2, object := .key, fromLocation := some .drawer, toLocation := .box }
]

example : StoryHistoryCompatible falseBeliefWorld history := by
  decide

example : objectiveLocation history .key = some .box := by
  rfl

example : subjectiveLocation falseBeliefWorld history .bob .key = some .drawer := by
  simp [subjectiveLocation, falseBeliefWorld, history, objectiveLocation,
    applyStoryRelocation]

example : subjectiveLocation informedWorld history .bob .key = some .box := by
  simp [subjectiveLocation, informedWorld, falseBeliefWorld, history,
    objectiveLocation, applyStoryRelocation]

private def score (known action : Location) : Nat :=
  if known = action then 1 else 0

example :
    score .drawer .box < score .drawer .drawer ∧
    score .box .drawer < score .box .box := by
  decide

#check NarrativeDynamics.no_info_path_no_admissible_observation

end NarrativeDynamics.Tests.StoryState
```

The concrete worlds deliberately make information reachability trivial because this test's purpose is fluent projection and ranking. Information-path denial itself remains covered by the generic theorem below and the pre-existing observation-admission theorem referenced by `#check`.

- [ ] **Step 3: Add a dedicated final StoryState CI step before implementation exists**

Do **not** put the deliberate RED command inside the existing `Lean theorem tests` block because that would skip the later Python step on failure. Append a new step after `Python numerical tests`:

```yaml
      - name: Narrative story theorem tests
        run: |
          export PATH="$HOME/.elan/bin:$PATH"
          lake env lean NarrativeDynamics/Tests/StoryState.lean
```

This ordering makes the intended remote RED precise: conformance, full build, all existing Lean theorem tests, and all existing Python tests run first; only the new StoryState step fails.

- [ ] **Step 4: Verify Lean RED locally**

Run:

```bash
lake build
python3 -m unittest discover -s tests -v
lake env lean NarrativeDynamics/Tests/StoryState.lean
```

Expected:

- existing Lean build succeeds;
- all pre-existing Python tests succeed;
- StoryState command fails because `NarrativeDynamics.Core.StoryState` does not exist.

- [ ] **Step 5: Commit and publish the precise RED**

```bash
git add .github/workflows/proof.yml NarrativeDynamics/Tests/StoryState.lean
git commit -m "test: add narrative story state RED"
git push -u origin work/narrative-microstory-v1
```

Open a draft PR to `proof/narrative-dynamics-v0`. Record the exact `proof` run and require:

- conformance success;
- full `lake build` success;
- all pre-existing Lean theorem tests success;
- Python suite success at the then-current count;
- failure only in `Narrative story theorem tests` because the new module is missing.

- [ ] **Step 6: Implement the minimal generic StoryState layer**

Create `NarrativeDynamics/Core/StoryState.lean`:

```lean
import NarrativeDynamics.Core.WorldGraph
import NarrativeDynamics.Core.ObservationAdmission

namespace NarrativeDynamics

structure StoryRelocation (Event Object Location : Type*) where
  event : Event
  object : Object
  fromLocation : Option Location
  toLocation : Location


def applyStoryRelocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (state : Object → Option Location)
    (relocation : StoryRelocation Event Object Location) :
    Object → Option Location :=
  fun object =>
    if object = relocation.object then some relocation.toLocation else state object


def objectiveLocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (history : List (StoryRelocation Event Object Location))
    (object : Object) : Option Location :=
  (history.foldl applyStoryRelocation (fun _ => none)) object


noncomputable def subjectiveLocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (object : Object) : Option Location := by
  classical
  exact objectiveLocation
    (history.filter (fun relocation => world.observed relocation.event agent))
    object


def storyHistoryCompatibleFrom
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (state : Object → Option Location)
    (previousTime : Option Nat) :
    List (StoryRelocation Event Object Location) → Prop
  | [] => True
  | relocation :: rest =>
      relocation.fromLocation = state relocation.object ∧
      (match previousTime with
       | none => True
       | some time => time < world.eventTime relocation.event) ∧
      storyHistoryCompatibleFrom world
        (applyStoryRelocation state relocation)
        (some (world.eventTime relocation.event))
        rest


def StoryHistoryCompatible
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location)) : Prop :=
  storyHistoryCompatibleFrom world (fun _ => none) none history


theorem objectiveLocation_append_relocation
    {Event Object Location : Type*}
    [DecidableEq Object]
    (history : List (StoryRelocation Event Object Location))
    (relocation : StoryRelocation Event Object Location) :
    objectiveLocation (history ++ [relocation]) relocation.object =
      some relocation.toLocation := by
  simp [objectiveLocation, List.foldl_append, applyStoryRelocation]


theorem subjectiveLocation_append_unobserved
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (object : Object)
    (relocation : StoryRelocation Event Object Location)
    (hnot : ¬ world.observed relocation.event agent) :
    subjectiveLocation world (history ++ [relocation]) agent object =
      subjectiveLocation world history agent object := by
  classical
  simp [subjectiveLocation, hnot]


theorem subjectiveLocation_append_observed_relocation
    {Agent Event Object Location Node : Type*}
    [DecidableEq Object]
    (world : WorldGraph Agent Event Node)
    (history : List (StoryRelocation Event Object Location))
    (agent : Agent)
    (relocation : StoryRelocation Event Object Location)
    (hobs : world.observed relocation.event agent) :
    subjectiveLocation world (history ++ [relocation]) agent relocation.object =
      some relocation.toLocation := by
  classical
  simp [subjectiveLocation, hobs, objectiveLocation_append_relocation]


theorem observed_story_event_has_info_path
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (relocation : StoryRelocation Event Object Location)
    (agent : Agent)
    (hobs : world.observed relocation.event agent) :
    infoReachable world.info
      (world.eventNode relocation.event)
      (world.agentNode agent) := by
  exact hworld.2.1 relocation.event agent hobs


theorem no_info_path_story_event_unobserved
    {Agent Event Object Location Node : Type*}
    (world : WorldGraph Agent Event Node)
    (hworld : WorldInvariant world)
    (relocation : StoryRelocation Event Object Location)
    (agent : Agent)
    (hno : ¬ infoReachable world.info
      (world.eventNode relocation.event)
      (world.agentNode agent)) :
    ¬ world.observed relocation.event agent := by
  intro hobs
  exact hno (observed_story_event_has_info_path world hworld relocation agent hobs)

end NarrativeDynamics
```

`StoryRelocation` deliberately does not duplicate canonical `logical_time`: the concrete story-to-Lean instantiation maps each Python event's `logical_time` to `world.eventTime relocation.event`. `StoryHistoryCompatible` validates strict event-time order plus objective `fromLocation` continuity.

If Lean requires a small syntax/proof adjustment, preserve these semantics and public theorem meanings; do not move object fluent state into `TypedEdge`.

- [ ] **Step 7: Add StoryState to the library root and verify GREEN**

Append to `NarrativeDynamics.lean`:

```lean
import NarrativeDynamics.Core.StoryState
```

Run:

```bash
lake env lean NarrativeDynamics/Tests/StoryState.lean
lake build
python3 -m unittest discover -s tests -v
```

Then execute every existing Lean theorem command from `.github/workflows/proof.yml`.

- [ ] **Step 8: Commit Lean GREEN**

```bash
git add NarrativeDynamics/Core/StoryState.lean NarrativeDynamics/Tests/StoryState.lean NarrativeDynamics.lean
git commit -m "feat: add narrative story state semantics"
git push
```

Require the draft PR `proof` run fully green before Task 2.

---

### Task 2: Immutable Canonical Story Schema and Authored Fixtures

**Files:**
- Create: `narrative_dynamics/story/schema.py`
- Create: `tests/test_story_schema.py`
- Create: `fixtures/stories/key_location_false_belief_v1.json`
- Create: `fixtures/stories/key_location_informed_v1.json`

**Interfaces:**
- Consumes: `narrative_dynamics.contracts.stable_content_hash`.
- Produces:
  - `StoryEntitiesV1`
  - `RelocationEventV1`
  - `DirectObservationV1`
  - `SearchActionV1`
  - `SearchDecisionV1`
  - `NarrativeOracleV1`
  - `NarrativeCaseV1`
  - `load_narrative_case(path: str | Path) -> NarrativeCaseV1`
  - private shared `_validate_story_semantics(entities, events, observations, decision) -> None`
  - `NarrativeCaseV1.identity_payload() -> dict[str, object]`
  - `NarrativeCaseV1.to_dict(include_content_hash: bool = True) -> dict[str, object]`
  - `NarrativeCaseV1.to_json() -> str`
  - `NarrativeCaseV1.from_dict(data, *, verify_declared_hash: bool = True) -> NarrativeCaseV1`

- [ ] **Step 1: Write schema RED tests**

Create `tests/test_story_schema.py` and import the not-yet-existing schema module. Define the canonical false-belief in-memory case exactly:

```python
def make_false_belief_case() -> NarrativeCaseV1:
    return NarrativeCaseV1(
        name="key-location-false-belief",
        version="1.0.0",
        source={"kind": "authored_microstory", "language": "en"},
        provenance={
            "synthetic": True,
            "empirical_human_data": False,
            "population_representative": False,
        },
        source_text=(
            "Alice and Bob are in a room. While Bob is watching, Alice puts a key "
            "in the drawer. Bob leaves the room. While Bob is away, Alice moves the "
            "key from the drawer to the box. Bob returns and wants to find the key. "
            "Where will Bob search first: the drawer or the box?"
        ),
        entities=StoryEntitiesV1(
            agents=("alice", "bob"),
            objects=("key",),
            locations=("drawer", "box"),
        ),
        events=(
            RelocationEventV1(
                id="e1", logical_time=1, actor="alice", object="key",
                from_location=None, to_location="drawer",
            ),
            RelocationEventV1(
                id="e2", logical_time=2, actor="alice", object="key",
                from_location="drawer", to_location="box",
            ),
        ),
        observations=(DirectObservationV1(event="e1", agent="bob"),),
        decision=SearchDecisionV1(
            id="d1", time=3, actor="bob", object="key",
            actions=(
                SearchActionV1(id="search_drawer", location="drawer"),
                SearchActionV1(id="search_box", location="box"),
            ),
        ),
        oracle=NarrativeOracleV1(
            objective_location="box",
            actor_subjective_location="drawer",
            agent_belief_ranking=("search_drawer", "search_box"),
            omniscient_ranking=("search_box", "search_drawer"),
        ),
    )
```

Required tests:

```python
class NarrativeStorySchemaTests(unittest.TestCase):
    def test_case_is_immutable_roundtrippable_and_hash_stable(self):
        case = make_false_belief_case()
        loaded = NarrativeCaseV1.from_dict(json.loads(case.to_json()))
        self.assertEqual(loaded, case)
        self.assertEqual(loaded.content_hash, case.content_hash)
        with self.assertRaises((FrozenInstanceError, TypeError)):
            case.name = "changed"

    def test_declared_hash_is_verified(self):
        payload = make_false_belief_case().to_dict()
        payload["content_hash"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(ValueError, "content hash"):
            NarrativeCaseV1.from_dict(payload)

    def test_objective_history_continuity_is_enforced(self):
        bad = make_false_belief_case().to_dict(include_content_hash=False)
        bad["events"][1]["from_location"] = "box"
        with self.assertRaisesRegex(ValueError, "from_location"):
            NarrativeCaseV1.from_dict(bad, verify_declared_hash=False)
```

Add explicit tests for duplicate entity/event/action IDs, undeclared actor/object/location, non-increasing event time, invalid first `from_location`, invalid observation event/agent, duplicate observation pair, unsupported observation channel, decision time not after events, duplicate action location, and decision actor lacking any observed relocation of the target object.

- [ ] **Step 2: Verify, commit, and publish schema RED**

Run:

```bash
python3 -m unittest tests.test_story_schema -v
```

Expected: import failure because `narrative_dynamics.story.schema` does not exist.

Then:

```bash
git add tests/test_story_schema.py
git commit -m "test: add canonical microstory schema RED"
git push
```

Remote `proof` must fail only on the new Python story-schema obligation; all Lean gates and prior Python tests must remain green up to that failure.

- [ ] **Step 3: Implement frozen schema values and one shared semantic validator**

Create `narrative_dynamics/story/schema.py` with these class signatures:

```python
@dataclass(frozen=True)
class StoryEntitiesV1:
    agents: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]

@dataclass(frozen=True)
class RelocationEventV1:
    id: str
    logical_time: int
    actor: str
    object: str
    from_location: str | None
    to_location: str
    kind: str = "relocate_object"

@dataclass(frozen=True)
class DirectObservationV1:
    event: str
    agent: str
    channel: str = "direct_perception"

@dataclass(frozen=True)
class SearchActionV1:
    id: str
    location: str

@dataclass(frozen=True)
class SearchDecisionV1:
    id: str
    time: int
    actor: str
    object: str
    actions: tuple[SearchActionV1, ...]
    kind: str = "search_object"

@dataclass(frozen=True)
class NarrativeOracleV1:
    objective_location: str
    actor_subjective_location: str
    agent_belief_ranking: tuple[str, ...]
    omniscient_ranking: tuple[str, ...]

@dataclass(frozen=True)
class NarrativeCaseV1:
    name: str
    version: str
    source: Mapping[str, object]
    provenance: Mapping[str, object]
    source_text: str
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    decision: SearchDecisionV1
    oracle: NarrativeOracleV1
    schema_version: int = 1
```

Implement a single private cross-field validator and call it from `NarrativeCaseV1.__post_init__`:

```python
def _validate_story_semantics(
    entities: StoryEntitiesV1,
    events: tuple[RelocationEventV1, ...],
    observations: tuple[DirectObservationV1, ...],
    decision: SearchDecisionV1,
) -> None:
    # Validate all cross references, strict event order, objective from_location
    # continuity, observation references/uniqueness, decision references/time,
    # unique actions/locations, known objective target, and at least one observed
    # relocation of the target for the decision actor.
```

The code body must implement each enumerated validation directly; do not duplicate this logic later in `scenario.py`.

Use recursive freezing patterns consistent with `narrative_dynamics/observations/dataset.py`, reject unknown/missing JSON keys, and use `stable_content_hash` for fixture identity:

```python
@property
def content_hash(self) -> str:
    return stable_content_hash(self.identity_payload())
```

`from_dict(..., verify_declared_hash=True)` requires a declared hash and verifies it. With `False`, the same structural/semantic validation still runs; only declared-hash checking is skipped.

- [ ] **Step 4: Generate exactly two committed fixtures from production constructors**

Use a one-shot Python command importing the production dataclasses and writing `case.to_json() + "\n"`; do not hand-author declared hashes.

The informed case is identical in entities/events/decision and adds:

```python
DirectObservationV1(event="e2", agent="bob")
```

with:

```python
NarrativeOracleV1(
    objective_location="box",
    actor_subjective_location="box",
    agent_belief_ranking=("search_box", "search_drawer"),
    omniscient_ranking=("search_box", "search_drawer"),
)
```

Extend the schema test:

```python
def test_committed_pair_has_identical_objective_events(self):
    false_case = load_narrative_case(
        "fixtures/stories/key_location_false_belief_v1.json"
    )
    informed_case = load_narrative_case(
        "fixtures/stories/key_location_informed_v1.json"
    )
    self.assertEqual(false_case.events, informed_case.events)
    self.assertNotEqual(false_case.observations, informed_case.observations)
    self.assertFalse(false_case.provenance["empirical_human_data"])
    self.assertFalse(false_case.provenance["population_representative"])
```

- [ ] **Step 5: Verify and commit schema GREEN**

Run:

```bash
python3 -m unittest tests.test_story_schema -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
```

Then:

```bash
git add narrative_dynamics/story/schema.py tests/test_story_schema.py fixtures/stories
git commit -m "feat: add canonical narrative microstory schema"
git push
```

Require remote `proof` green before Task 3.

---

### Task 3: Pure Objective and Subjective Story Replay

**Files:**
- Create: `narrative_dynamics/story/replay.py`
- Create: `tests/test_story_replay.py`

**Interfaces:**
- Consumes: `RelocationEventV1`, `DirectObservationV1`.
- Produces:
  - `ObjectLocationState`
  - `objective_state(events, *, at_time: int | None = None) -> Mapping[str, ObjectLocationState]`
  - `subjective_state(events, observations, agent, *, at_time: int | None = None) -> Mapping[str, ObjectLocationState]`
  - `latest_object_location(state, object_id) -> ObjectLocationState`

```python
@dataclass(frozen=True)
class ObjectLocationState:
    object_id: str
    location: str
    supporting_event_id: str
    logical_time: int
```

- [ ] **Step 1: Write replay RED tests**

Create `tests/test_story_replay.py`:

```python
class StoryReplayTests(unittest.TestCase):
    def setUp(self):
        self.false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        self.informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )

    def test_false_belief_has_objective_box_and_subjective_drawer(self):
        objective = latest_object_location(objective_state(self.false_case.events), "key")
        subjective = latest_object_location(
            subjective_state(self.false_case.events, self.false_case.observations, "bob"),
            "key",
        )
        self.assertEqual((objective.location, objective.supporting_event_id), ("box", "e2"))
        self.assertEqual((subjective.location, subjective.supporting_event_id), ("drawer", "e1"))

    def test_informed_observation_updates_subjective_state(self):
        subjective = latest_object_location(
            subjective_state(self.informed_case.events, self.informed_case.observations, "bob"),
            "key",
        )
        self.assertEqual((subjective.location, subjective.supporting_event_id), ("box", "e2"))
```

Also test:

- adding an unobserved later relocation changes objective state but not Bob's subjective state;
- adding Bob's observation of that event changes subjective state;
- `at_time=1` returns the historical drawer state;
- missing object support raises `ValueError` rather than falling back.

- [ ] **Step 2: Verify, commit, and publish replay RED**

```bash
python3 -m unittest tests.test_story_replay -v
git add tests/test_story_replay.py
git commit -m "test: add narrative replay RED"
git push
```

Expected local/remote failure is the missing replay module only.

- [ ] **Step 3: Implement replay with no runtime dependency**

Create `narrative_dynamics/story/replay.py`:

```python
@dataclass(frozen=True)
class ObjectLocationState:
    object_id: str
    location: str
    supporting_event_id: str
    logical_time: int


def objective_state(
    events: Iterable[RelocationEventV1],
    *,
    at_time: int | None = None,
) -> Mapping[str, ObjectLocationState]:
    state: dict[str, ObjectLocationState] = {}
    for event in events:
        if at_time is not None and event.logical_time > at_time:
            continue
        state[event.object] = ObjectLocationState(
            object_id=event.object,
            location=event.to_location,
            supporting_event_id=event.id,
            logical_time=event.logical_time,
        )
    return MappingProxyType(state)


def subjective_state(
    events: Iterable[RelocationEventV1],
    observations: Iterable[DirectObservationV1],
    agent: str,
    *,
    at_time: int | None = None,
) -> Mapping[str, ObjectLocationState]:
    observed_ids = {
        observation.event
        for observation in observations
        if observation.agent == agent
    }
    return objective_state(
        (event for event in events if event.id in observed_ids),
        at_time=at_time,
    )


def latest_object_location(
    state: Mapping[str, ObjectLocationState],
    object_id: str,
) -> ObjectLocationState:
    try:
        return state[object_id]
    except KeyError as error:
        raise ValueError(f"no supported location for object {object_id!r}") from error
```

No fallback from subjective to objective state is permitted.

- [ ] **Step 4: Verify and commit replay GREEN**

```bash
python3 -m unittest tests.test_story_replay -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/story/replay.py tests/test_story_replay.py
git commit -m "feat: add objective and subjective story replay"
git push
```

Require remote `proof` green before Task 4.

---

### Task 4: Anti-Leakage Runtime Scenario Projection

**Files:**
- Create: `narrative_dynamics/story/scenario.py`
- Create: `tests/test_story_scenario.py`
- Modify: `narrative_dynamics/story/schema.py` only if an import-safe private shared validator export is required; do not duplicate validation.

**Interfaces:**
- Consumes: `NarrativeCaseV1`, model-visible schema dataclasses, private `_validate_story_semantics`, `contracts.Scenario`, `stable_content_hash`.
- Produces:
  - `NarrativeScenarioV1`
  - `NarrativeScenarioV1.from_case(case) -> NarrativeScenarioV1`
  - `NarrativeScenarioV1.to_payload() -> dict[str, object]`
  - `NarrativeScenarioV1.from_payload(payload) -> NarrativeScenarioV1`
  - `project_narrative_scenario(case) -> Scenario`
  - `decode_narrative_scenario(scenario) -> NarrativeScenarioV1`

- [ ] **Step 1: Write anti-leakage and bypass-resistance RED tests**

Create `tests/test_story_scenario.py`:

```python
class NarrativeScenarioProjectionTests(unittest.TestCase):
    def test_projection_contains_only_model_visible_semantics(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        self.assertEqual(
            set(scenario.payload),
            {"entities", "events", "observations", "decision"},
        )
        serialized = repr(dict(scenario.payload)).lower()
        for forbidden in (
            "oracle", "source_text", "false-belief", "population_representative",
            "agent_belief_ranking", "omniscient_ranking", "observed_choice",
        ):
            self.assertNotIn(forbidden, serialized)
            self.assertNotIn(forbidden, scenario.id.lower())

    def test_runtime_id_is_derived_only_from_visible_payload(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        digest = stable_content_hash(scenario.payload).removeprefix("sha256:")
        self.assertEqual(scenario.id, f"story-v1-{digest}")
```

Add tests that:

- changing only fixture `name`, `version`, `source`, `provenance`, `source_text`, and oracle leaves projected payload and runtime ID unchanged;
- non-neutral scenario ID is rejected;
- extra/missing model-visible payload key is rejected;
- unsupported event kind/observation channel is rejected;
- **a manually constructed runtime payload with invalid `from_location` continuity is rejected**;
- **a manually constructed runtime payload with an undeclared entity reference is rejected**;
- **a manually constructed runtime payload where the decision actor has no observed target relocation is rejected**.

Those last three tests prove runtime decoding cannot bypass fixture-only cross-field validation.

- [ ] **Step 2: Verify, commit, and publish scenario RED**

```bash
python3 -m unittest tests.test_story_scenario -v
git add tests/test_story_scenario.py
git commit -m "test: add narrative scenario leakage RED"
git push
```

Expected failure is the missing scenario module only.

- [ ] **Step 3: Implement one validated model-facing scenario type**

Create:

```python
@dataclass(frozen=True)
class NarrativeScenarioV1:
    entities: StoryEntitiesV1
    events: tuple[RelocationEventV1, ...]
    observations: tuple[DirectObservationV1, ...]
    decision: SearchDecisionV1

    def __post_init__(self) -> None:
        _validate_story_semantics(
            self.entities,
            self.events,
            self.observations,
            self.decision,
        )
```

`from_case()` copies only those four fields. `from_payload()` reconstructs those exact typed values and therefore calls the same semantic validator through `__post_init__`.

The neutral ID algorithm is the only accepted one:

```python
def _runtime_id(payload: Mapping[str, object]) -> str:
    digest = stable_content_hash(payload).removeprefix("sha256:")
    return f"story-v1-{digest}"
```

Projection:

```python
def project_narrative_scenario(case: NarrativeCaseV1) -> Scenario:
    story = NarrativeScenarioV1.from_case(case)
    payload = story.to_payload()
    return Scenario(id=_runtime_id(payload), payload=payload)
```

Decoder:

```python
def decode_narrative_scenario(scenario: Scenario) -> NarrativeScenarioV1:
    story = NarrativeScenarioV1.from_payload(scenario.payload)
    canonical_payload = story.to_payload()
    if scenario.id != _runtime_id(canonical_payload):
        raise ValueError("narrative scenario id does not match canonical payload")
    return story
```

Do not expose a flag that disables the neutral-ID or semantic-validation checks.

- [ ] **Step 4: Verify and commit scenario GREEN**

```bash
python3 -m unittest tests.test_story_scenario -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/story/schema.py narrative_dynamics/story/scenario.py tests/test_story_scenario.py
git commit -m "feat: add leak-safe narrative runtime projection"
git push
```

Require remote `proof` green before Task 5.

---

### Task 5: Agent-Belief Search Model

**Files:**
- Create: `narrative_dynamics/adapters/story_belief_search.py`
- Create: `tests/test_story_models.py`

**Interfaces:**
- Consumes: `Scenario`, `ModelRun`, `TraceEvent`, `decode_narrative_scenario`, `subjective_state`, `latest_object_location`.
- Produces: `AgentBeliefSearchModel`, name exactly `agent-belief-search`.

Common outcome contract introduced here and reused unchanged by Task 6:

```text
epistemic_basis = {
  "kind": "subjective" | "objective",
  "target_object": str,
  "resolved_location": str,
  "supporting_event_id": str
}
action_scores = {action_id: 0.0 | 1.0}
policy = {action_id: 0.0 | 1.0}
selected_action = action_id
```

- [ ] **Step 1: Write belief-model RED tests**

```python
class StoryModelTests(unittest.TestCase):
    def setUp(self):
        self.false_case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        self.informed_case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )

    def test_belief_model_follows_actor_subjective_state(self):
        model = AgentBeliefSearchModel()
        false_run = model.simulate(
            project_narrative_scenario(self.false_case), {}, random.Random(7)
        )
        informed_run = model.simulate(
            project_narrative_scenario(self.informed_case), {}, random.Random(7)
        )
        self.assertEqual(false_run.outcome["selected_action"], "search_drawer")
        self.assertEqual(informed_run.outcome["selected_action"], "search_box")
        self.assertEqual(false_run.outcome["epistemic_basis"], {
            "kind": "subjective",
            "target_object": "key",
            "resolved_location": "drawer",
            "supporting_event_id": "e1",
        })

    def test_belief_model_rejects_any_parameter(self):
        with self.assertRaisesRegex(ValueError, "empty parameter"):
            AgentBeliefSearchModel().simulate(
                project_narrative_scenario(self.false_case),
                {"beta": 1.0},
                random.Random(7),
            )
```

Also assert exact action-score/policy keys, one-hot normalization, and fail-closed behavior when the known subjective location has no matching decision action. Do not allow an objective fallback.

- [ ] **Step 2: Verify, commit, and publish belief-model RED**

```bash
python3 -m unittest tests.test_story_models.StoryModelTests.test_belief_model_follows_actor_subjective_state -v
git add tests/test_story_models.py
git commit -m "test: add belief-sensitive story model RED"
git push
```

- [ ] **Step 3: Implement parameter-free belief search**

```python
class AgentBeliefSearchModel:
    name = "agent-belief-search"

    def simulate(
        self,
        scenario: Scenario,
        parameters: Mapping[str, float],
        rng: random.Random,
    ) -> ModelRun:
        if parameters:
            raise ValueError("agent-belief-search requires an empty parameter mapping")
        story = decode_narrative_scenario(scenario)
        decision = story.decision
        state = subjective_state(
            story.events,
            story.observations,
            decision.actor,
            at_time=decision.time,
        )
        location = latest_object_location(state, decision.object)
        matching = [
            action.id for action in decision.actions
            if action.location == location.location
        ]
        if len(matching) != 1:
            raise ValueError("subjective target location must match exactly one decision action")
        selected = matching[0]
        scores = {
            action.id: 1.0 if action.id == selected else 0.0
            for action in decision.actions
        }
        policy = dict(scores)
        basis = {
            "kind": "subjective",
            "target_object": decision.object,
            "resolved_location": location.location,
            "supporting_event_id": location.supporting_event_id,
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

The RNG argument is intentionally unused. Do not add stochastic behavior or a hidden tie-breaker.

- [ ] **Step 4: Verify and commit belief-model GREEN**

```bash
python3 -m unittest tests.test_story_models -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/adapters/story_belief_search.py tests/test_story_models.py
git commit -m "feat: add belief-sensitive story search model"
git push
```

Require remote `proof` green before Task 6.

---

### Task 6: Omniscient Baseline and Common Story Choice Metrics

**Files:**
- Create: `narrative_dynamics/adapters/story_omniscient_search.py`
- Create: `narrative_dynamics/story/metrics.py`
- Modify: `tests/test_story_models.py`

**Interfaces:**
- Consumes: common outcome contract from Task 5, `objective_state`, `SimulationTrace`.
- Produces:
  - `OmniscientSearchModel`, name exactly `omniscient-search`
  - `story_choice_metrics(trace) -> dict[str, float]`

V1 metric keys are exactly:

```text
choice.search_box
choice.search_drawer
```

- [ ] **Step 1: Add omniscient/metrics RED tests**

```python
def test_omniscient_model_follows_objective_state_in_both_cases(self):
    model = OmniscientSearchModel()
    for case in (self.false_case, self.informed_case):
        run = model.simulate(project_narrative_scenario(case), {}, random.Random(11))
        self.assertEqual(run.outcome["selected_action"], "search_box")
        self.assertEqual(run.outcome["epistemic_basis"], {
            "kind": "objective",
            "target_object": "key",
            "resolved_location": "box",
            "supporting_event_id": "e2",
        })


def test_false_belief_separates_models_and_informed_case_reunites_them(self):
    belief = AgentBeliefSearchModel()
    omniscient = OmniscientSearchModel()
    false_scenario = project_narrative_scenario(self.false_case)
    informed_scenario = project_narrative_scenario(self.informed_case)
    self.assertNotEqual(
        belief.simulate(false_scenario, {}, random.Random(1)).outcome["selected_action"],
        omniscient.simulate(false_scenario, {}, random.Random(1)).outcome["selected_action"],
    )
    self.assertEqual(
        belief.simulate(informed_scenario, {}, random.Random(1)).outcome["selected_action"],
        omniscient.simulate(informed_scenario, {}, random.Random(1)).outcome["selected_action"],
    )
```

Add metric validation tests for exact two-action coverage, numeric finite non-negative probabilities, and normalized mass.

- [ ] **Step 2: Verify, commit, and publish omniscient/metrics RED**

```bash
python3 -m unittest tests.test_story_models -v
git add tests/test_story_models.py
git commit -m "test: add omniscient story baseline RED"
git push
```

- [ ] **Step 3: Implement omniscient baseline**

Use the same result contract as Task 5, changing only projection:

```python
state = objective_state(story.events, at_time=story.decision.time)
location = latest_object_location(state, story.decision.object)
```

Basis kind is exactly `objective`; parameters must be empty; missing matching action fails closed.

- [ ] **Step 4: Implement common metric extractor**

```python
_EXPECTED_ACTIONS = ("search_box", "search_drawer")


def story_choice_metrics(trace: SimulationTrace) -> dict[str, float]:
    policy = trace.outcome.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("story trace outcome must contain a policy mapping")
    if set(policy) != set(_EXPECTED_ACTIONS):
        raise ValueError("story policy must contain exactly the V1 search actions")
    values: dict[str, float] = {}
    for action in _EXPECTED_ACTIONS:
        raw = policy[action]
        if isinstance(raw, bool):
            raise ValueError("story policy probabilities must be numeric")
        try:
            probability = float(raw)
        except (TypeError, ValueError) as error:
            raise ValueError("story policy probabilities must be numeric") from error
        if not math.isfinite(probability) or probability < 0.0:
            raise ValueError("story policy probabilities must be finite and non-negative")
        values[action] = probability
    if not math.isclose(sum(values.values()), 1.0, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("story policy must be normalized")
    return {f"choice.{action}": values[action] for action in _EXPECTED_ACTIONS}
```

- [ ] **Step 5: Verify and commit competing-model GREEN**

```bash
python3 -m unittest tests.test_story_models -v
python3 -m unittest discover -s tests -v
git add narrative_dynamics/adapters/story_omniscient_search.py narrative_dynamics/story/metrics.py tests/test_story_models.py
git commit -m "feat: add omniscient story baseline and metrics"
git push
```

Require remote `proof` green before Task 7.

---

### Task 7: Trusted Runtime Acceptance and Public Story API

**Files:**
- Create: `narrative_dynamics/story/__init__.py`
- Create: `tests/test_story_runtime.py`

**Interfaces:**
- Consumes: Task 2–6 story components and existing `SimulationRunner`.
- Produces public `narrative_dynamics.story` exports:
  - `StoryEntitiesV1`
  - `RelocationEventV1`
  - `DirectObservationV1`
  - `SearchActionV1`
  - `SearchDecisionV1`
  - `NarrativeOracleV1`
  - `NarrativeCaseV1`
  - `load_narrative_case`
  - `ObjectLocationState`
  - `objective_state`
  - `subjective_state`
  - `latest_object_location`
  - `NarrativeScenarioV1`
  - `project_narrative_scenario`
  - `decode_narrative_scenario`
  - `story_choice_metrics`

Do not export either model from package root.

- [ ] **Step 1: Write trusted-runtime/public-API RED tests**

Create `tests/test_story_runtime.py`:

```python
class StoryRuntimeTests(unittest.TestCase):
    def test_both_models_run_through_simulation_runner(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()
        belief = runner.run_batch(
            AgentBeliefSearchModel(), scenario, {}, seeds=(101, 102)
        )
        omniscient = runner.run_batch(
            OmniscientSearchModel(), scenario, {}, seeds=(101, 102)
        )
        self.assertEqual(
            [trace.outcome["selected_action"] for trace in belief],
            ["search_drawer", "search_drawer"],
        )
        self.assertEqual(
            [trace.outcome["selected_action"] for trace in omniscient],
            ["search_box", "search_box"],
        )
        for trace in belief + omniscient:
            self.assertEqual(trace.parameters, ())
            self.assertIsNotNone(trace.manifest)
            self.assertEqual(trace.scenario_id, scenario.id)

    def test_deterministic_model_is_seed_replay_compatible(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_informed_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()
        a = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=7)
        b = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=999)
        self.assertEqual(a.outcome, b.outcome)
        self.assertNotEqual(a.seed, b.seed)

    def test_common_metrics_work_for_both_models(self):
        case = load_narrative_case(
            "fixtures/stories/key_location_false_belief_v1.json"
        )
        scenario = project_narrative_scenario(case)
        runner = SimulationRunner()
        belief = runner.run_once(AgentBeliefSearchModel(), scenario, {}, seed=1)
        omniscient = runner.run_once(OmniscientSearchModel(), scenario, {}, seed=1)
        self.assertEqual(story_choice_metrics(belief), {
            "choice.search_box": 0.0,
            "choice.search_drawer": 1.0,
        })
        self.assertEqual(story_choice_metrics(omniscient), {
            "choice.search_box": 1.0,
            "choice.search_drawer": 0.0,
        })
```

Add API isolation:

```python
def test_story_api_is_exported_only_from_story_package(self):
    import narrative_dynamics
    import narrative_dynamics.story as story

    for name in (
        "NarrativeCaseV1",
        "project_narrative_scenario",
        "objective_state",
        "subjective_state",
        "story_choice_metrics",
    ):
        self.assertTrue(hasattr(story, name), name)
        self.assertFalse(hasattr(narrative_dynamics, name), name)
```

Also inspect both adapter module sources and assert they contain no `prison_pomdp` or `prison_reactive` import/reference.

- [ ] **Step 2: Verify, commit, and publish runtime/API RED**

```bash
python3 -m unittest tests.test_story_runtime -v
git add tests/test_story_runtime.py
git commit -m "test: add trusted microstory runtime RED"
git push
```

Expected failure is missing `narrative_dynamics.story` exports only.

- [ ] **Step 3: Add exact story package public exports**

Create `narrative_dynamics/story/__init__.py` importing and listing exactly the interface names above in `__all__`.

Do not modify `narrative_dynamics/__init__.py`.

- [ ] **Step 4: Verify and commit runtime/API GREEN**

```bash
python3 -m unittest tests.test_story_schema -v
python3 -m unittest tests.test_story_replay -v
python3 -m unittest tests.test_story_scenario -v
python3 -m unittest tests.test_story_models -v
python3 -m unittest tests.test_story_runtime -v
python3 -m unittest discover -s tests -v
python3 -m compileall -q narrative_dynamics
git add narrative_dynamics/story/__init__.py tests/test_story_runtime.py
git commit -m "feat: exercise narrative microstories through trusted runtime"
git push
```

Expected: all Python tests pass; total test count is greater than 269. Require remote `proof` green before Task 8.

---

### Task 8: Final Evidence, Review, and Research-Branch Integration

**Files:**
- Review the complete feature diff against the pre-feature `proof/narrative-dynamics-v0` SHA.
- No production modification is expected unless review finds a defect; every defect requires a failing test before a fix.

**Interfaces:**
- Consumes: final feature head and draft PR evidence.
- Produces: reviewed green feature head and, after verification, a non-forced fast-forward of `proof/narrative-dynamics-v0` only.

- [ ] **Step 1: Run fresh complete verification on the exact final feature head**

Run the same gates as CI:

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
```

Also run every pre-existing Lean theorem command from `.github/workflows/proof.yml`; do not substitute `lake build` for theorem tests.

- [ ] **Step 2: Require a fresh final feature `proof` run**

The final remote run must show:

- conformance success;
- full Lean library build success;
- all pre-existing theorem tests success;
- complete Python suite success with count greater than 269;
- dedicated `Narrative story theorem tests` success.

Record exact feature SHA and workflow/run ID.

- [ ] **Step 3: Review the exact diff**

Confirm:

- one new Lean semantic module plus root import/test/workflow hook only;
- no prison adapter changes;
- no observation-protocol semantic changes;
- no dependency/lockfile changes;
- no new `NodeKind`;
- unchanged `TypedEdge.locatedAt`;
- Python `logical_time` maps to Lean `WorldGraph.eventTime` rather than a second Lean clock;
- story code never imports prison adapters;
- replay code has no simulation/model-comparison dependency;
- authored fixture and runtime decoder call the same cross-field semantic validator;
- manually constructed malformed `Scenario` values cannot bypass continuity/reference/decision validation;
- belief model never falls back to objective state;
- both models reject any parameter;
- source text/oracle/name/version/source/provenance/gold outputs never enter `Scenario.payload`;
- runtime ID is only a digest of model-visible payload;
- scenario ID cannot reveal `false-belief` or `informed`;
- both fixture objective event histories are identical;
- informed case differs semantically only by Bob observing `e2`;
- `narrative_dynamics.story` exports the canonical story API and package root remains unchanged;
- no empirical validity, NLP-understanding, or general theory-of-mind claim was added.

If review finds a defect: add focused RED test → verify RED → smallest fix → rerun all gates.

- [ ] **Step 4: Verify ancestry before integration**

Compare:

```text
base = proof/narrative-dynamics-v0
head = work/narrative-microstory-v1
```

Require feature `ahead` and `behind_by == 0`. If base moved, re-evaluate the exact merge context and rerun proof; never force-update.

- [ ] **Step 5: Fast-forward only the research branch**

Move `proof/narrative-dynamics-v0` to the exact reviewed feature SHA with `force=false`.

Do not move `master`; any future main-branch PR is a separate user-authorized action.

- [ ] **Step 6: Verify identical refs and close feature metadata**

Require comparison of final feature SHA and `proof/narrative-dynamics-v0` to report `identical`, `ahead_by == 0`, `behind_by == 0`.

Update feature PR title/body with:

- first non-prison canonical story domain;
- false-belief/informed matched pair;
- objective versus agent-specific replay;
- Lean `StoryState` semantics;
- belief versus omniscient models;
- anti-leakage fixture→runtime projection;
- shared runtime semantic validation;
- neutral payload-derived scenario ID;
- parameter-free V1;
- final SHA/run/test count;
- explicit limits: authored canonical input, no NLP, no empirical human data, no general theory-of-mind claim.

---

## Final Acceptance Checklist

- [ ] Two authored fixtures validate, round-trip, and verify declared hashes.
- [ ] Both fixtures have identical objective relocation history.
- [ ] False-belief Bob observes only `e1`; informed Bob observes `e1` and `e2`.
- [ ] Objective replay is `key@box` in both cases.
- [ ] Bob subjective replay is `key@drawer` in false belief and `key@box` when informed.
- [ ] Lean proves an unobserved relocation leaves subjective projection unchanged and an observed later relocation updates it.
- [ ] Lean keeps information-path and observation-admission boundaries authoritative.
- [ ] `StoryHistoryCompatible` binds objective continuity and strict `WorldGraph.eventTime` order.
- [ ] Belief model selects drawer only in false belief.
- [ ] Omniscient model selects box in both cases.
- [ ] Both models accept exactly zero parameters.
- [ ] Both models use one shared result/metric contract.
- [ ] Both models run through `SimulationRunner` with manifests and canonical empty-parameter identity.
- [ ] Runtime scenario payload contains only `entities/events/observations/decision`.
- [ ] Runtime scenario ID is derived only from visible payload content.
- [ ] Fixture metadata changes cannot alter model-visible input unless canonical story semantics changed.
- [ ] Runtime decoding reuses fixture cross-field validation and rejects malformed hand-built scenarios.
- [ ] Source text/oracle/provenance/gold outputs are not model-visible.
- [ ] Story public API lives under `narrative_dynamics.story`, not package root.
- [ ] No prison semantics, dependency files, or generic observation semantics changed.
- [ ] All Lean conformance/build/theorem gates are green.
- [ ] Full Python suite is green with more than 269 tests.
- [ ] Integration, if performed, is a non-forced fast-forward of `proof/narrative-dynamics-v0` only.
