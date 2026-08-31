# Percept-to-Cognition and Memory V15.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make sanitized V15 percepts the authoritative private evidence for a new perception-aware cognition, SQLite memory, recall, and V14 social-claim pipeline without changing objective V10 events or legacy V10-V15 behavior.

**Architecture:** Add parallel V15.1 entry points rather than changing the semantics of legacy V11-V14 APIs. A deterministic perceptual story view feeds cognitive rule matching; a separate SQLite/FTS5 store persists only fields disclosed by each percept; memory recall reconstructs percepts rather than full events; and social evidence accepts only exact percepts. Existing objective stories remain the truth ledger, while every V15.1 result binds the exact perception-model hash.

**Tech Stack:** Python 3 standard library, frozen dataclasses, existing V10-V15 contracts, SQLite/FTS5, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-simulated-story-production-architecture-design.md`

## Global Constraints

- Objective `SituatedWorldEvent`, `SituatedRoundResult`, and `SituatedStory` values remain unchanged.
- Existing V10-V15 public entry points and all valid legacy content hashes remain unchanged.
- V15.1 cognition, memory, recall, and social evidence consume `SituatedPercept`; they never join a partial percept to a full objective event.
- `DETECTED` evidence cannot identify actor, action, place, outcome, details, or source testimony.
- `IDENTIFIED` evidence may use actor, action kind, and place, but never outcome or details.
- Only `EXACT` evidence may use outcome/details or create V14 testimony/verification evidence.
- The actor's own result remains exact; `INSPECT` remains exact, inspection-only, and actor-private.
- SQLite authoritative rows store exact perception/story hashes and only disclosed fields; FTS5 remains a rebuildable index.
- Database paths, SQLite row IDs, Python object identities, and FTS ranks never enter canonical hashes.
- Existing V12 memory tables remain readable and untouched; V15.1 uses separate versioned tables so no destructive migration is required.
- No LLM, embedding model, RNG, geometry engine, third-party dependency, natural-language grounding, narrative projection, or rendering is added.

---

## File map

- `narrative_dynamics/abm/situated_percept_cognition.py`: perceptual story views, rule matching, direct admission, source selection, and one perception-aware cognitive round.
- `narrative_dynamics/abm/situated_percept_memory_contracts.py`: fidelity policy, sanitized memory record, query, hit, and report contracts.
- `narrative_dynamics/abm/situated_percept_memory.py`: separate SQLite/FTS5 schema, ingestion, list/search, activation, and index rebuild.
- `narrative_dynamics/abm/situated_percept_memory_cognition.py`: bound perception/memory/cognition model, private recall, and one integrated round.
- `narrative_dynamics/abm/situated_percept_social_cognition.py`: exact-percept social evidence and V14 social transition orchestration.
- `tests/test_network_abm_situated_percept_cognition.py`: direct cognition and objective-story compatibility tests.
- `tests/test_network_abm_situated_percept_memory.py`: SQLite privacy, restart, idempotence, conflict, and FTS tests.
- `tests/test_network_abm_situated_percept_memory_cognition.py`: long-term recall and later-action tests.
- `tests/test_network_abm_situated_percept_social_cognition.py`: testimony/verification admission and trust tests.
- `narrative_dynamics/abm/__init__.py`, `tests/test_network_abm_public_api.py`, `README.md`: public surface and executable acceptance scenario.

## Task 1: Perceptual story view and cognition admission

**Files:**
- Create: `narrative_dynamics/abm/situated_percept_cognition.py`
- Create: `tests/test_network_abm_situated_percept_cognition.py`
- Modify: `narrative_dynamics/abm/situated_cognition.py`
- Modify: `narrative_dynamics/abm/situated_story.py`

**Interfaces:**
- Consumes: `SituatedPerceptionModel`, `SituatedPercept`, `SituatedStory`, `SituatedAgentCognitiveModel`, `SituatedCognitiveModel`, and `SituatedCognitiveState`.
- Produces: `SituatedPerceptCognitiveRoundResult`, `project_situated_story_percepts()`, `perceptual_timeline()`, `match_situated_percept_rule()`, `admit_situated_percepts()`, and `simulate_situated_percept_cognitive_round()`.

- [ ] **Step 1: Write failing fidelity-boundary tests**

  Create a real office story and perception graph. Assert:

  ```python
  detected = perceptual_timeline(perception, closed_story, "bob")[-1]
  assert detected.fidelity is SituatedPerceptFidelity.DETECTED
  assert match_situated_percept_rule(bob_model, detected) is None

  identified = perceptual_timeline(perception, glass_story, "carol")[-1]
  assert identified.fidelity is SituatedPerceptFidelity.IDENTIFIED
  assert match_situated_percept_rule(kind_only_model, identified).symbol_id == "saw_tell"
  assert match_situated_percept_rule(outcome_model, identified) is None

  exact = perceptual_timeline(perception, open_story, "bob")[-1]
  assert exact.fidelity is SituatedPerceptFidelity.EXACT
  assert match_situated_percept_rule(bob_model, exact).symbol_id == "approved"
  ```

  Also assert the timeline rejects an unknown agent, a story from another world hash, and a perception model whose exact world differs.

- [ ] **Step 2: Run the focused tests and verify RED**

  Run:

  ```text
  python -m unittest tests.test_network_abm_situated_percept_cognition -v
  ```

  Expected: import failure because `situated_percept_cognition.py` does not exist.

- [ ] **Step 3: Implement deterministic story projection and rule matching**

  Add these signatures:

  ```python
  def project_situated_story_percepts(
      model: SituatedPerceptionModel,
      story: SituatedStory,
  ) -> tuple[SituatedPerceptualProjection, ...]: ...

  def perceptual_timeline(
      model: SituatedPerceptionModel,
      story: SituatedStory,
      agent_id: str,
  ) -> tuple[SituatedPercept, ...]: ...

  def match_situated_percept_rule(
      model: SituatedAgentCognitiveModel,
      percept: SituatedPercept,
  ) -> SituatedObservationRule | None: ...
  ```

  Recompute every projection from the story's exact rounds; never accept a caller-supplied event join. `DETECTED` matches no current observation rule. `IDENTIFIED` may match only rules with no outcome/detail requirement. `EXACT` may match all disclosed fields. Preserve the legacy rule that an actor's own `TELL` through `SELF` does not update its belief.

- [ ] **Step 4: Write failing direct-admission privacy and replay tests**

  Assert `admit_situated_percepts()` accepts only percepts addressed to the mind's agent, stores `percept_id` in the legacy-compatible `processed_observation_ids`/`SituatedBeliefAdmission.observation_id` fields, and stores `source_event_id` in `observed_event_ids`. Replaying equal percepts must produce no second admission. A detected sound must be marked processed but must not change belief or authorize a TELL source.

- [ ] **Step 5: Implement percept admission**

  Add:

  ```python
  def admit_situated_percepts(
      model: SituatedAgentCognitiveModel,
      mind: SituatedAgentMindState,
      percepts: tuple[SituatedPercept, ...],
      *,
      _claim_topic_by_symbol: Mapping[str, str] | None = None,
      _consolidated_claim_keys: frozenset[tuple[str, str, str]] | None = None,
  ) -> SituatedBeliefAdmissionResult: ...
  ```

  Apply the existing likelihood matrix and consolidation semantics, but read only percept fields. A percept with no matching rule is still processed exactly once.

- [ ] **Step 6: Write a failing perception-aware cognitive-round test**

  From identical objective histories, assert Bob chooses different later actions when the meeting-room door yields `EXACT` versus `DETECTED` access. Assert the newly resolved objective round still comes from `advance_situated_story()` and has the same hash as resolving the selected intents through the legacy V10 transition. Assert the next mind's observed-event set is derived from V15 percepts, not the broader V10 observation projection.

- [ ] **Step 7: Factor decision source selection and implement the round**

  Extract the existing decision core in `situated_cognition.py` so legacy `decide_situated_action()` supplies `(event_id, kind)` candidates from `SituatedPerspectiveEvent`, while the new runtime supplies only percepts whose `kind` is disclosed. Do not change the legacy function's result or hash.

  Add an optional `perception_model` binding to `SituatedStory`. Omit it from serialization when absent so every legacy story hash remains unchanged. When present, story validation must re-derive each projection and authorize TELL source references only from prior percepts whose action kind was disclosed; a merely `DETECTED` event is not a valid source. The new cognitive round returns a perception-bound story while leaving every objective event and round unchanged.

  Define:

  ```python
  @dataclass(frozen=True)
  class SituatedPerceptCognitiveRoundResult:
      perception_model_id: str
      perception_model_hash: str
      prior_projection_hashes: tuple[str, ...]
      cognitive_round: SituatedCognitiveRoundResult
      next_projection: SituatedPerceptualProjection

  def simulate_situated_percept_cognitive_round(
      perception_model: SituatedPerceptionModel,
      cognitive_model: SituatedCognitiveModel,
      story: SituatedStory,
      state: SituatedCognitiveState,
  ) -> SituatedPerceptCognitiveRoundResult: ...
  ```

  Build decisions from percepts, resolve the objective round through the existing transition, then rebuild only `observed_event_ids` from `next_projection`. Include exact perception/projection hashes in the new result.

- [ ] **Step 8: Run Task 1 and legacy regressions, then commit**

  Run:

  ```text
  python -m unittest tests.test_network_abm_situated_percept_cognition -v
  python -m unittest tests.test_network_abm_situated_cognition tests.test_network_abm_situated_memory_cognition tests.test_network_abm_situated_social_cognition -v
  ```

  Commit:

  ```text
  git add docs/superpowers/plans/2026-09-01-percept-cognition-memory-integration-v15-1.md narrative_dynamics/abm/situated_story.py narrative_dynamics/abm/situated_cognition.py narrative_dynamics/abm/situated_percept_cognition.py tests/test_network_abm_situated_percept_cognition.py
  git commit -m "feat: admit private percepts into cognition"
  ```

## Task 2: Sanitized percept memory store

**Files:**
- Create: `narrative_dynamics/abm/situated_percept_memory_contracts.py`
- Create: `narrative_dynamics/abm/situated_percept_memory.py`
- Create: `tests/test_network_abm_situated_percept_memory.py`

**Interfaces:**
- Consumes: `SituatedPerceptionModel`, `SituatedPercept`, `SituatedPerceptFidelity`, and `SituatedStory`.
- Produces: `SituatedPerceptMemoryFidelityPolicy`, `SituatedPerceptMemoryPolicy`, `SituatedPerceptMemoryRecord`, `SituatedPerceptMemoryQuery`, `SituatedPerceptMemoryHit`, `SituatedPerceptMemoryWriteReport`, `SituatedPerceptMemoryIndexReport`, `standard_situated_percept_memory_policy()`, `ingest_situated_percept_story()`, `list_situated_percept_memories()`, `search_situated_percept_memories()`, `set_situated_percept_memory_active()`, and `rebuild_situated_percept_memory_index()`.

- [ ] **Step 1: Write failing policy and record-shape tests**

  Require exactly one confidence/salience policy per fidelity. Define `SituatedPerceptMemoryRecord` with exact provenance plus only the percept payload:

  ```python
  @dataclass(frozen=True)
  class SituatedPerceptMemoryRecord:
      memory_id: str
      agent_id: str
      percept_id: str
      perception_model_id: str
      perception_model_hash: str
      story_model_id: str
      story_model_hash: str
      projection_hash: str
      source_event_id: str
      source_event_hash: str
      round_index: int
      channels: tuple[ObservationChannel, ...]
      fidelity: SituatedPerceptFidelity
      actor_agent_id: str | None
      kind: SituatedActionKind | None
      place_id: str | None
      outcome: str | None
      details: tuple[EvidenceFact, ...]
      confidence: float
      salience: float
      policy_hash: str
      summary: str
      active: bool = True
  ```

  Require `memory_id == percept_id`, apply the same fidelity/privacy invariants as `SituatedPercept`, and prove there are no action ID, target, success, causes, full event, database path, or row-ID fields.

- [ ] **Step 2: Run contract tests and verify RED**

  Run the new test module; expect import failure for the missing contracts.

- [ ] **Step 3: Implement immutable memory contracts and queries**

  Add `SituatedPerceptMemoryQuery` filters for agent, text, actor, place, fidelities, event kinds, channels, round bounds, confidence, limit, story hash, included IDs, and excluded IDs. Canonicalize all tuples/mappings and ensure hashes exclude lexical rank and database identity.

- [ ] **Step 4: Write failing SQLite privacy/idempotence tests**

  Ingest the office story into one database for Alice, Bob, and Carol. Assert:

  - Alice's private inspection appears only in Alice's rows.
  - Bob's closed-door `DETECTED` row has no actor/kind/place/outcome/details and its `summary` plus raw authoritative SQLite row contain no secret message.
  - Carol's `IDENTIFIED` row has actor/kind/place but no outcome/details.
  - Reingestion is idempotent; the same `(agent, percept_id)` with conflicting content raises `SituatedPerceptMemoryConflictError`.
  - The legacy `memory_records` table, when present in the same database, is unchanged.

- [ ] **Step 5: Implement the separate schema and ingestion**

  Create `percept_memory_metadata`, `percept_memory_records`, and external-content `percept_memory_fts` with insert/delete/update triggers. Use schema version `1` under the separate metadata table. Store nullable disclosure columns and canonical JSON for channels/details. `ingest_situated_percept_story()` must derive projections itself and may not accept prejoined objective events.

- [ ] **Step 6: Write failing search/restart/rebuild tests**

  Assert every query is agent-scoped, channel matching uses any disclosed channel, `DETECTED` can be retrieved by a neutral summary but not by secret text, inactive rows are omitted by default, restart returns equal records/hashes, and rebuilding FTS preserves authoritative rows.

- [ ] **Step 7: Implement list/search/activation/rebuild and commit**

  Run the new suite plus legacy V12 storage tests. Commit:

  ```text
  git add narrative_dynamics/abm/situated_percept_memory_contracts.py narrative_dynamics/abm/situated_percept_memory.py tests/test_network_abm_situated_percept_memory.py
  git commit -m "feat: persist sanitized percept memories"
  ```

## Task 3: Percept-memory recall and later cognition

**Files:**
- Create: `narrative_dynamics/abm/situated_percept_memory_cognition.py`
- Create: `tests/test_network_abm_situated_percept_memory_cognition.py`

**Interfaces:**
- Consumes: Task 1 cognition APIs, Task 2 memory store, `SituatedAgentRecallPolicy`, and an existing `SituatedCognitiveModel`.
- Produces: `SituatedPerceptMemoryCognitiveModel`, `SituatedPerceptMemoryCognitiveRoundResult`, `recall_situated_percept_memories()`, `simulate_situated_percept_memory_cognitive_round()`, and `simulate_situated_percept_memory_cognition()`.

- [ ] **Step 1: Write failing exact-binding model tests**

  Define:

  ```python
  @dataclass(frozen=True)
  class SituatedPerceptMemoryCognitiveModel:
      model_id: str
      version: str
      perception_model: SituatedPerceptionModel
      cognitive_model: SituatedCognitiveModel
      memory_policy: SituatedPerceptMemoryPolicy
      agents: tuple[SituatedAgentRecallPolicy, ...]
  ```

  Require exact world-model equality/hash, exact agent-policy coverage, valid cue places, canonical order, and content hashing by child hashes.

- [ ] **Step 2: Implement the bound model and result contract**

  `SituatedPerceptMemoryCognitiveRoundResult` contains the exact model ID/hash, Task 1 percept cognitive round, and one `SituatedMemoryRecallResult` per agent. It delegates `decisions`, `next_story`, and `next_state` properties without embedding database paths.

- [ ] **Step 3: Write failing private recall tests**

  After a checkpoint, assert an earlier exact inspection can later update only Alice's belief and action. A detected sound can be recalled as an episode but matches no cognitive rule and cannot change belief. An identified TELL can match only a kind-only rule. Recalled exact TELL evidence is multiplied by the supplied directed source trust; direct inspection remains weight `1.0`.

- [ ] **Step 4: Implement recall from sanitized records**

  Reconstruct a `SituatedPercept` only from `SituatedPerceptMemoryRecord`, call `match_situated_percept_rule()`, and use the existing weighted Bayesian formula. Validate memories against the exact percept IDs/hashes re-derived from the current story and perception model. Never call `_memory_perspective()` or reconstruct `SituatedWorldEvent`.

- [ ] **Step 5: Write a failing integrated-round/restart test**

  Run control and recall-enabled models from the same checkpoint/database state. Assert the recalled model changes Alice's later selected action, stores the recalled percept ID once, and produces identical result hashes after closing and reopening SQLite. Assert Bob never recalls Alice's private inspection and the objective round equals legacy V10 resolution of the selected intents.

- [ ] **Step 6: Implement one-round and trajectory orchestration**

  Each round must:

  1. derive/ingest the exact private percept history;
  2. admit new direct percepts above the checkpoint floor;
  3. recall eligible older sanitized memories once;
  4. decide using only disclosed percept kinds as source candidates;
  5. resolve one objective V10 round;
  6. project the new round and update observed IDs from V15.

- [ ] **Step 7: Run Task 3 and V11-V13 regressions, then commit**

  Commit:

  ```text
  git add narrative_dynamics/abm/situated_percept_memory_cognition.py tests/test_network_abm_situated_percept_memory_cognition.py
  git commit -m "feat: recall percept memories into cognition"
  ```

## Task 4: Exact-percept social evidence

**Files:**
- Create: `narrative_dynamics/abm/situated_percept_social_cognition.py`
- Create: `tests/test_network_abm_situated_percept_social_cognition.py`

**Interfaces:**
- Consumes: `SituatedPerceptMemoryCognitiveModel`, existing `SituatedSocialMemoryModel`/state, Task 3 round results, and existing `advance_situated_social_memory()`.
- Produces: `SituatedPerceptSocialCognitiveRoundResult`, `recall_situated_percept_memories_with_social_trust()`, `simulate_situated_percept_social_cognitive_round()`, and `simulate_situated_percept_social_cognition()`.

- [ ] **Step 1: Write failing social-evidence entitlement tests**

  Assert an exact non-self TELL admission creates `TESTIMONY` with its disclosed actor as source. An exact INSPECT admission creates source-less `VERIFICATION`. `DETECTED` and `IDENTIFIED` percepts create no V14 evidence even if a generic cognitive rule matched. Repeat the same assertions for recalled percept memories.

- [ ] **Step 2: Implement exact-only evidence extraction**

  Resolve every admission ID against the agent's re-derived percept timeline or sanitized memory row. Reject missing, wrong-agent, wrong-event-hash, non-exact, or mismatched model evidence. Reuse existing `SituatedSocialEvidence` and `advance_situated_social_memory()`; do not change V14 claim/state contracts.

- [ ] **Step 3: Write failing trust/contradiction/replay tests**

  Run two exact conflicting TELL percepts and a later exact private verification. Assert claim supersession/history, confirmation or contradiction, and only the observer-to-source trust/affinity edge changes. Replaying the same percept IDs must be idempotent. Closing the acoustic edge so the TELL becomes merely detected must produce no claim and no relationship update.

- [ ] **Step 4: Implement round and trajectory orchestration**

  Validate that the existing social model's cognitive model equals the Task 3 model's cognitive model. Feed current directed trust into exact recalled TELL weighting, pass exact-only evidence to the existing V14 transition, and bind both model hashes in the new result.

- [ ] **Step 5: Run Task 4 and V14 regressions, then commit**

  Commit:

  ```text
  git add narrative_dynamics/abm/situated_percept_social_cognition.py tests/test_network_abm_situated_percept_social_cognition.py
  git commit -m "feat: ground social memory in exact percepts"
  ```

## Task 5: Public API, acceptance scenario, and delivery

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: the exact V15.1 public API and one executable closed/open-door cognition-memory-social comparison.

- [ ] **Step 1: Add failing exact public-surface assertions**

  Add only the new public dataclasses, errors, policies, query/store functions, cognition functions, recall functions, and social orchestration functions. Do not export SQLite SQL strings, row converters, evidence extractors, decision helpers, or legacy-private functions.

- [ ] **Step 2: Export the new surface**

  Import every specified public name once and add it once to `__all__`. Run the public API test and confirm GREEN.

- [ ] **Step 3: Add an executable README acceptance scenario**

  Extend the office-door example so the same TELL is `EXACT` with the door open and `DETECTED` when closed. Demonstrate that only the exact branch changes Bob's belief, creates a searchable message memory, and admits a V14 testimony claim; both branches retain the same objective truth rules. State explicitly that V16 semantic grounding remains the next phase for arbitrary natural language.

- [ ] **Step 4: Run fresh verification**

  Run:

  ```text
  python -m unittest discover -s tests -p "test_network_abm*.py"
  python -m compileall -q narrative_dynamics tests
  git diff --check
  ```

  Execute every README Python block independently. Repeat the full network-ABM suite under `PYTHONHASHSEED=1` and `PYTHONHASHSEED=8675309`. Record exact counts without claiming unrelated repository CI is fixed.

- [ ] **Step 5: Review privacy, compatibility, and replay**

  Inspect the complete branch diff for full-event joins from partial percepts, undisclosed SQLite/FTS fields, use of V10 observations in V15.1 paths, detected/identified social claims, unauthorized recall, database identity in hashes, tuple-order dependence, legacy hash changes, and branch/database cross-contamination. Fix each Critical or Important finding with a failing regression first.

- [ ] **Step 6: Commit public integration**

  Commit:

  ```text
  git add narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
  git commit -m "feat: expose percept-driven agent cognition"
  ```

## Self-review

- Spec coverage: Tasks 1-4 implement the delivery-sequence item “V15 cognition/memory integration” across direct cognition, durable private memory, recall, and social evidence while leaving objective events unchanged.
- Privacy boundary: no V15.1 API accepts a partial percept plus a full event; memory and social evidence are derived from sanitized fields only.
- Compatibility: legacy V10-V15 types, entry points, tables, and hashes remain available; the new path is explicit and content-addressed by the perception model.
- Placeholder scan: every task names concrete files, interfaces, tests, commands, expected failures, and commits; there are no deferred implementation placeholders.
- Type consistency: Task 1 percept IDs flow into Task 2 memory IDs, Task 3 recall admissions, and Task 4 evidence IDs; all stages retain source event and perception-model hashes.
- Scope boundary: arbitrary language grounding, LLM providers, narrative beats/scenes, rendering, and director branch search remain V16-V19.
