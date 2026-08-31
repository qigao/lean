# Perceptual Environment Graph V15 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic visual/auditory/interaction graph overlay and project objective V10 round events into sanitized private percepts without changing V10-V14 behavior or hashes.

**Architecture:** Frozen V15 contracts bind an exact `SituatedWorldModel` and describe directed perception edges, passage-state activation, per-agent thresholds, and per-action signals. A pure graph module derives reach from one exact world snapshot; a pure projector converts an existing `SituatedRoundResult` into content-addressed `SituatedPercept` values that disclose only the fields justified by access and fidelity. This foundation is query-only: V11-V14 continue using their existing observation path until a separate migration plan is reviewed.

**Tech Stack:** Python 3 standard library, frozen dataclasses, `Enum`, `MappingProxyType`, `heapq`, existing V10 situated contracts/events, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-simulated-story-production-architecture-design.md`

## Global Constraints

- V10-V14 public behavior and all existing content hashes remain unchanged.
- V15 wraps the exact V10 world model; it does not add fields to V10 contracts.
- Visibility and audibility are directed cumulative-cost graphs; interaction is direct and non-transitive.
- Every non-co-located perception edge is explicitly declared and optionally activated by one exact passage state.
- Partial percepts never embed a full `SituatedWorldEvent` or undisclosed event details.
- The actor always receives its own exact outcome; inspection remains exact and private to the actor.
- No LLM, embedding model, RNG, geometry engine, database migration, or third-party dependency is added.
- V15 artifacts are canonical and content-addressed; database paths and object identities never enter hashes.
- This plan does not feed V15 percepts into cognition, memory, social claims, or rendering.

---

## File map

- `narrative_dynamics/abm/situated_perception_contracts.py`: immutable graph, profile,
  signal, reach, percept, and projection contracts plus validation/hashing.
- `narrative_dynamics/abm/situated_perception.py`: active-edge selection,
  deterministic reach, interaction queries, and sanitized event projection.
- `tests/test_network_abm_situated_perception_contracts.py`: contract identity,
  canonicalization, invalid-reference, threshold, and privacy-shape tests.
- `tests/test_network_abm_situated_perception.py`: chain/graph reach, open/closed
  passage, partial hearing, combined-channel, privacy, and replay tests.
- `narrative_dynamics/abm/__init__.py`: exact V15 public imports and `__all__` entries.
- `tests/test_network_abm_public_api.py`: public-surface gate.
- `README.md`: executable graph-perception example and explicit query-only boundary.

## Task 1: Perception graph and sanitized percept contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_perception_contracts.py`
- Create: `tests/test_network_abm_situated_perception_contracts.py`

**Interfaces:**
- Consumes: `SituatedActionKind`, `ObservationChannel`, `EvidenceFact`,
  `SituatedWorldModel`, `SituatedWorldState`, and `SituatedRoundResult` identities.
- Produces: `SituatedPerceptionLayer`, `SituatedEdgeActivation`,
  `SituatedPerceptFidelity`, `SituatedPerceptionEdge`,
  `SituatedAgentPerceptionProfile`, `SituatedEventSignalProfile`,
  `SituatedPerceptionModel`, `SituatedPerceptionReach`, `SituatedPercept`, and
  `SituatedPerceptualProjection`.

- [ ] **Step 1: Write failing enum and edge/profile tests**

  Create tests that import the V15 module and construct these exact public values:

  ```python
  edge = SituatedPerceptionEdge(
      "meeting-door-closed-audio",
      SituatedPerceptionLayer.AUDITORY,
      "corridor",
      "meeting",
      25.0,
      SituatedEdgeActivation.PASSAGE_CLOSED,
      "meeting-door",
  )
  profile = SituatedAgentPerceptionProfile(
      "bob",
      max_visual_cost=2.0,
      minimum_detectable_sound=20.0,
      minimum_clear_sound=45.0,
  )
  signal = SituatedEventSignalProfile(
      SituatedActionKind.TELL,
      visually_observable=True,
      auditory_intensity=60.0,
  )
  ```

  Assert enum values are exactly `visibility`, `auditory`, `interaction`;
  activations are `always`, `passage_open`, `passage_closed`; fidelities are
  `detected`, `identified`, `exact`. Assert negative/non-finite edge costs fail,
  interaction cost other than `0.0` fails, clear threshold below detection fails,
  and non-finite signal intensity fails.

- [ ] **Step 2: Run Task 1 tests and verify RED**

  Run:

  ```text
  python -m unittest tests.test_network_abm_situated_perception_contracts -v
  ```

  Expected: import failure because `situated_perception_contracts.py` does not exist.

- [ ] **Step 3: Implement enums and leaf contracts**

  Add frozen dataclasses with these signatures:

  ```python
  class SituatedPerceptionLayer(str, Enum):
      VISIBILITY = "visibility"
      AUDITORY = "auditory"
      INTERACTION = "interaction"

  class SituatedEdgeActivation(str, Enum):
      ALWAYS = "always"
      PASSAGE_OPEN = "passage_open"
      PASSAGE_CLOSED = "passage_closed"

  class SituatedPerceptFidelity(str, Enum):
      DETECTED = "detected"
      IDENTIFIED = "identified"
      EXACT = "exact"

  @dataclass(frozen=True)
  class SituatedPerceptionEdge:
      edge_id: str
      layer: SituatedPerceptionLayer
      source_place_id: str
      target_place_id: str
      cost: float
      activation: SituatedEdgeActivation = SituatedEdgeActivation.ALWAYS
      passage_id: str | None = None

  @dataclass(frozen=True)
  class SituatedAgentPerceptionProfile:
      agent_id: str
      max_visual_cost: float
      minimum_detectable_sound: float
      minimum_clear_sound: float

  @dataclass(frozen=True)
  class SituatedEventSignalProfile:
      kind: SituatedActionKind
      visually_observable: bool
      auditory_intensity: float | None = None
  ```

  Canonicalize numbers to `float`. Require an `ALWAYS` edge to have no passage;
  require passage activations to have a passage ID. Interaction edges have cost
  exactly zero. Implement canonical `to_dict()` and `content_hash` on every leaf.

- [ ] **Step 4: Write failing model-reference and canonicalization tests**

  Build a four-place office world and assert `SituatedPerceptionModel` rejects an
  unknown endpoint, unknown passage, missing/duplicate agent profile, duplicate edge
  ID, duplicate signal kind, and an activation/passage mismatch. Construct inputs in
  reverse order and assert equality plus equal content hashes after canonicalization.

- [ ] **Step 5: Implement the exact model contract**

  Use this signature:

  ```python
  @dataclass(frozen=True)
  class SituatedPerceptionModel:
      model_id: str
      version: str
      world_model: SituatedWorldModel
      edges: tuple[SituatedPerceptionEdge, ...]
      agent_profiles: tuple[SituatedAgentPerceptionProfile, ...]
      signal_profiles: tuple[SituatedEventSignalProfile, ...]
  ```

  Require exactly one profile for every `world_model.agents` ID. Require edge place
  and passage references to exist. Require at most one signal profile per action
  kind; an omitted kind emits no non-self signal. Sort edges by `edge_id`, profiles
  by `agent_id`, and signals by `kind.value`. Serialize only
  `world_model.content_hash`, never the Python object or a path.

- [ ] **Step 6: Write failing reach/percept/projection shape tests**

  Require `SituatedPerceptionReach` to freeze sorted finite cost mappings and direct
  interaction place IDs. Require `SituatedPercept` to use this exact shape:

  ```python
  @dataclass(frozen=True)
  class SituatedPercept:
      percept_id: str
      round_index: int
      agent_id: str
      source_event_id: str
      source_event_hash: str
      channels: tuple[ObservationChannel, ...]
      fidelity: SituatedPerceptFidelity
      actor_agent_id: str | None = None
      kind: SituatedActionKind | None = None
      place_id: str | None = None
      outcome: str | None = None
      details: tuple[EvidenceFact, ...] = ()
  ```

  Assert `DETECTED` forbids actor/kind/place/outcome/details, `IDENTIFIED` requires
  actor/kind/place but forbids outcome/details, and `EXACT` requires all four
  identified fields plus outcome. Require non-empty unique channels and exact SHA-256
  source hashes. Confirm the contract contains no `SituatedWorldEvent` field.

- [ ] **Step 7: Implement reach, percept, and projection contracts**

  Define:

  ```python
  @dataclass(frozen=True)
  class SituatedPerceptionReach:
      model_hash: str
      state_hash: str
      source_place_id: str
      visual_costs: Mapping[str, float]
      auditory_losses: Mapping[str, float]
      interaction_place_ids: tuple[str, ...]

  @dataclass(frozen=True)
  class SituatedPerceptualProjection:
      model_id: str
      model_hash: str
      prior_state_hash: str
      round_result_hash: str
      percepts: tuple[SituatedPercept, ...]
  ```

  Freeze mappings with `MappingProxyType`, serialize keys lexically, sort channels
  by value, details by `(name, value)`, percepts by `(source_event_id, agent_id)`, and
  reject duplicate percept IDs or duplicate `(agent_id, source_event_id)` pairs.

- [ ] **Step 8: Run Task 1 tests and commit**

  Run the new contract suite plus V10 and V14 contract suites. Expected: all pass.
  Commit:

  ```text
  git add narrative_dynamics/abm/situated_perception_contracts.py tests/test_network_abm_situated_perception_contracts.py
  git commit -m "feat: define perceptual environment graph contracts"
  ```

## Task 2: Deterministic graph reach and interaction queries

**Files:**
- Create: `narrative_dynamics/abm/situated_perception.py`
- Create: `tests/test_network_abm_situated_perception.py`

**Interfaces:**
- Consumes: Task 1 `SituatedPerceptionModel`, `SituatedPerceptionReach`, exact
  `SituatedWorldState`.
- Produces: `derive_situated_perception_reach()` and
  `can_situated_agents_interact()`.

- [ ] **Step 1: Write failing chain and graph reach tests**

  Declare an office chain `records -> corridor -> meeting` plus a direct glass
  shortcut. Add both directions explicitly. Assert visual/auditory costs use the
  minimum cumulative path, not tuple order, and same-place costs are zero:

  ```python
  reach = derive_situated_perception_reach(
      perception_model,
      world_state,
      source_place_id="records",
  )
  self.assertEqual(reach.visual_costs["records"], 0.0)
  self.assertEqual(reach.visual_costs["meeting"], 2.0)
  self.assertEqual(reach.auditory_losses["meeting"], 15.0)
  ```

  Add a cycle and assert termination and stable lexical mapping order.

- [ ] **Step 2: Run focused reach test and verify RED**

  Run the named test. Expected: import failure because the runtime module does not
  exist.

- [ ] **Step 3: Implement passage activation and shortest-cost reach**

  Implement:

  ```python
  def derive_situated_perception_reach(
      model: SituatedPerceptionModel,
      state: SituatedWorldState,
      *,
      source_place_id: str,
  ) -> SituatedPerceptionReach:
      ...
  ```

  Validate `state.model_id/hash` against `model.world_model`. Select an edge when its
  activation is `ALWAYS`, or when the referenced `PassageState.open` matches the
  activation. Run separate `heapq` Dijkstra traversals for visibility and auditory
  layers using `(cost, place_id)` heap keys. Initialize the source at `0.0`. Never
  traverse interaction edges and never infer reverse edges.

- [ ] **Step 4: Write failing open/closed and interaction tests**

  For one meeting-room door, declare:

  - open visual edge cost `1.0`;
  - open auditory edge loss `5.0`;
  - closed auditory edge loss `25.0`;
  - open interaction edge cost `0.0`.

  Assert opening the passage changes all three derived relations; closing it removes
  visibility and interaction but retains the higher-loss acoustic path. Assert two
  consecutive interaction edges do not make endpoints interact.

- [ ] **Step 5: Implement direct interaction query**

  Implement:

  ```python
  def can_situated_agents_interact(
      model: SituatedPerceptionModel,
      state: SituatedWorldState,
      left_agent_id: str,
      right_agent_id: str,
  ) -> bool:
      ...
  ```

  Return true for distinct co-located agents. Otherwise require one active directed
  interaction edge from the left agent's current place to the right agent's place.
  Reject unknown agents and return false for the same agent.

- [ ] **Step 6: Add determinism/property-boundary tests and commit**

  Assert reversed edge input produces equal reach/hashes; closed-edge paths cannot
  leak through a cycle; unknown source place and mismatched state hash fail. Run:

  ```text
  python -m unittest tests.test_network_abm_situated_perception -v
  python -m unittest tests.test_network_abm_situated -v
  ```

  Commit:

  ```text
  git add narrative_dynamics/abm/situated_perception.py tests/test_network_abm_situated_perception.py
  git commit -m "feat: derive deterministic perceptual reach"
  ```

## Task 3: Sanitized event-to-percept projection

**Files:**
- Modify: `narrative_dynamics/abm/situated_perception.py`
- Modify: `tests/test_network_abm_situated_perception.py`

**Interfaces:**
- Consumes: Task 2 reach, Task 1 signal/profile/percept contracts, one exact existing
  `SituatedRoundResult`.
- Produces: `project_situated_percepts()` and `percepts_for_agent()`.

- [ ] **Step 1: Write failing actor/private-inspection tests**

  Resolve one existing V10 office round containing `INSPECT`. Project it and assert
  the actor receives one `EXACT` percept through `INSPECTION`, with exact actor, kind,
  place, outcome, and details. Assert every other agent receives no inspection
  percept even when a visual path exists.

- [ ] **Step 2: Run focused projection test and verify RED**

  Expected: `project_situated_percepts` is missing.

- [ ] **Step 3: Implement actor and signal selection skeleton**

  Add:

  ```python
  def project_situated_percepts(
      model: SituatedPerceptionModel,
      round_result: SituatedRoundResult,
  ) -> SituatedPerceptualProjection:
      ...

  def percepts_for_agent(
      projection: SituatedPerceptualProjection,
      agent_id: str,
  ) -> tuple[SituatedPercept, ...]:
      ...
  ```

  Validate the prior/next states and model hashes. Index bodies from
  `round_result.prior_state`. Use percept ID
  `f"{event.event_id}:p:{agent_id}"`. Actor `SELF` is exact for every event, except
  successful or failed `INSPECT` uses `INSPECTION`. Do not read or copy
  `round_result.observations`.

- [ ] **Step 4: Write failing open-door exact-hearing and visual tests**

  Resolve Alice's `TELL` in corridor with intensity `60.0`. Put Bob in meeting with
  an open door, acoustic loss `5.0`, clear threshold `45.0`, and visible cost `1.0`.
  Assert Bob gets one `EXACT` percept with channels `(AUDITORY, VISUAL)`, received
  sound `55.0` implied by the selected fidelity, and the exact message detail. Put
  Carol behind a visible-only glass edge and assert `IDENTIFIED` with actor/kind/place
  but no outcome or message.

- [ ] **Step 5: Implement strongest-fidelity channel merge**

  For each non-actor and event signal:

  ```text
  visible := signal.visually_observable and visual_cost <= max_visual_cost
  received := auditory_intensity - auditory_loss
  clear := received >= minimum_clear_sound
  detected := received >= minimum_detectable_sound
  ```

  Choose `EXACT` when clear, otherwise `IDENTIFIED` when visible, otherwise
  `DETECTED` when detected. Include every channel that actually crossed its access
  threshold, but disclose fields only to the chosen fidelity. Exact copies event
  outcome/details; identified copies actor/kind/place only; detected copies none.

- [ ] **Step 6: Write failing closed-door partial-hearing privacy test**

  Close the meeting door so acoustic loss becomes `25.0`: received sound is `35.0`,
  above Bob's detectable `20.0` but below clear `45.0`. Assert Bob's percept is
  `DETECTED`, has only `AUDITORY`, and contains none of actor, action kind, place,
  outcome, or the `message` detail. Assert Dana below threshold receives no percept.
  Recursively inspect `percept.to_dict()` and confirm the secret message string is
  absent.

- [ ] **Step 7: Implement partial projection and deterministic replay**

  Complete non-actor projection, canonical sorting, and immutable projection hashes.
  Compute reach once per distinct event source place in the round. Assert two equal
  models/states/rounds produce equal projections and hashes; edge tuple order and
  observer iteration order cannot affect output.

- [ ] **Step 8: Add mixed-event and no-signal regression tests**

  Assert `WAIT` with no signal profile remains actor-only; `MOVE`, `TAKE`, and `DROP`
  can be identified visually when declared; failed actions disclose only according
  to the same signal/fidelity rule; and one event yields at most one percept per
  observer even with two channels.

- [ ] **Step 9: Run Task 3 and V10-V14 regression tests, then commit**

  Run:

  ```text
  python -m unittest tests.test_network_abm_situated_perception -v
  python -m unittest discover -s tests -p "test_network_abm*.py"
  ```

  Expected: all existing hashes and tests remain unchanged. Commit:

  ```text
  git add narrative_dynamics/abm/situated_perception.py tests/test_network_abm_situated_perception.py
  git commit -m "feat: project sanitized private percepts"
  ```

## Task 4: Public API, executable example, review, and delivery

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: exact public V15 query/projection API and one executable office-door
  example.

- [ ] **Step 1: Add failing exact public-surface assertions**

  Add every Task 1 enum/dataclass and these runtime names to the expected ABM public
  API:

  ```text
  derive_situated_perception_reach
  can_situated_agents_interact
  project_situated_percepts
  percepts_for_agent
  ```

  Run the public API test and verify it fails only for missing V15 exports.

- [ ] **Step 2: Export the V15 surface**

  Import the exact symbols from the two new modules and add them once to `__all__`.
  Do not export private Dijkstra, activation, merge, or validation helpers.

- [ ] **Step 3: Add an executable README acceptance example**

  Build the open/closed meeting-door graph using public contracts. Demonstrate:

  ```python
  closed = project_situated_percepts(perception_model, tell_round)
  bob = percepts_for_agent(closed, "bob")
  assert bob[0].fidelity is SituatedPerceptFidelity.DETECTED
  assert bob[0].details == ()
  assert can_situated_agents_interact(
      perception_model, tell_round.prior_state, "alice", "bob"
  ) is False
  ```

  Explain that V15 is an analyst/query projection and does not yet replace V10-V14
  cognition or memory admission. Link the architecture spec and identify the next
  plan as percept-to-cognition/memory integration.

- [ ] **Step 4: Run complete fresh verification**

  Run:

  ```text
  python -m unittest discover -s tests -p "test_network_abm*.py"
  python -m compileall -q narrative_dynamics tests
  git diff --check
  ```

  Execute every README Python block independently. Record exact passing counts in
  the delivery note; do not claim the unrelated full repository CI is fixed.

- [ ] **Step 5: Review privacy and graph semantics**

  Review the exact V15 diff for full-event leakage in `DETECTED`/`IDENTIFIED`,
  reverse-edge inference, accidental transitive interaction, passage-state mismatch,
  non-finite costs, tuple-order dependence, duplicate observer/event percepts,
  V10-V14 hash compatibility, and query-only boundary violations. Fix every Critical
  or Important finding with a failing regression before implementation changes.

- [ ] **Step 6: Commit and update the active PR**

  Commit public integration as:

  ```text
  git add narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
  git commit -m "feat: expose perceptual environment graph"
  ```

  Push `feature/network-interaction-emergence-v1` and update PR #44 only after the
  user authorizes implementation/delivery. Preserve the linked worktree for review.

## Self-review

- Spec coverage: this plan covers only the first delivery-sequence item promised by
  the architecture spec: typed perception overlay, passage activation, deterministic
  visual/auditory reach, direct interaction, sanitized percepts, privacy, replay,
  public API, documentation, and review.
- Scope boundary: cognition/memory migration, semantic grounding, scene projection,
  rendering, and branch direction remain separate plans because each changes a
  different authority boundary.
- Placeholder scan: every executable step names its concrete input, output, command,
  expected result, and implementation boundary.
- Type consistency: Task 1 contracts are consumed without renaming by Tasks 2-4;
  `project_situated_percepts()` returns the exact projection accepted by
  `percepts_for_agent()`.
- Privacy check: the percept contract stores only an objective event ID/hash plus
  disclosed scalar fields/details; it has no full event field, and partial-hearing
  tests search serialized payloads for leaked message text.
