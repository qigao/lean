# Situated Memory-Augmented Cognition V13 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let restarted situated agents privately recall V12 memories and use them once as auditable evidence in V11 belief updates and POMDP action selection.

**Architecture:** New immutable recall contracts declare per-agent lexical cues and bounds. A separate memory-cognition runtime scopes V12 queries, reconstructs exact perspective events, applies confidence-weighted admissions, and delegates physical planning/resolution to V11. V11 state and decision contracts gain only the checkpoint and recall provenance needed to preserve one-count evidence semantics.

**Tech Stack:** Python 3 standard library, SQLite/FTS5 through V12 APIs, frozen dataclasses, existing `PlanningBeliefState`, V10 story world, V11 finite-horizon cognition, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-situated-memory-augmented-cognition-v13-design.md`

## Global Constraints

- Every recall query must include the requesting `agent_id`, exact world-model hash, and current state round ceiling.
- A memory already processed directly or recalled previously must not update belief again.
- Recall must not expose another agent's row, a future event, an inactive memory, or an event from another world model.
- V11 direct observations keep their existing one-round delay and exact Bayesian semantics.
- SQLite rows remain authoritative and unchanged by recall.
- Add no LLM, embeddings, LSTM, Transformer, or third-party dependency.
- Equal model, story, memory, and cue inputs must replay identically across database paths.

---

### Task 1: Recall and checkpoint contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_memory_cognition_contracts.py`
- Modify: `narrative_dynamics/abm/situated_cognition_contracts.py`
- Create: `tests/test_network_abm_situated_memory_cognition_contracts.py`

**Interfaces:**
- Consumes: `SituatedCognitiveModel`, `SituatedMemoryPolicy`, `SituatedActionKind`, `ObservationChannel`, and `SituatedStory`.
- Produces: `SituatedMemoryRecallCue`, `SituatedAgentRecallPolicy`, `SituatedMemoryCognitiveModel`, checkpoint-capable `SituatedAgentMindState`/`SituatedCognitiveState`, and `initialize_situated_memory_cognition()`.

- [ ] **Step 1: Write failing immutable-contract tests**

  Add literal tests proving cue filters and policy ordering are canonical, cue limits
  are bounded, every cognitive agent has exactly one recall policy, and the wrapper
  binds the exact cognitive and memory models. Add a late-story checkpoint test that
  asserts current body places, `observation_floor_round`, empty recalled/event sets,
  explicit checkpoint status, and no parent hash. These tests catch incomplete agent
  coverage, ambiguous cues, and fake normal-state ancestry.

- [ ] **Step 2: Run the contract tests and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory_cognition_contracts -v`.
  Expected: import failure because the V13 contracts module does not exist.

- [ ] **Step 3: Implement minimal frozen contracts and checkpoint state**

  Implement complete validation, canonical `to_dict()`, and content hashes. Extend
  `SituatedAgentMindState` with defaulted `recalled_memory_ids=()` and
  `observation_floor_round=0`. Extend `SituatedCognitiveState` with defaulted
  `checkpoint=False`; permit a non-zero parentless state only when checkpoint is
  true. Initialize V13 minds from declared priors and current body places without
  granting historical event IDs.

- [ ] **Step 4: Run contract and V11 regression tests and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory_cognition_contracts tests.test_network_abm_situated_cognition_contracts tests.test_network_abm_situated_cognition -v`.
  Expected: all V13 contracts and unchanged V11 behaviors pass.

- [ ] **Step 5: Commit the contracts**

  Run `git add docs/superpowers/specs/2026-08-31-situated-memory-augmented-cognition-v13-design.md docs/superpowers/plans/2026-08-31-situated-memory-augmented-cognition-v13.md narrative_dynamics/abm/situated_memory_cognition_contracts.py narrative_dynamics/abm/situated_cognition_contracts.py tests/test_network_abm_situated_memory_cognition_contracts.py && git commit -m "feat: define memory-augmented cognition contracts"`.

### Task 2: Scoped retrieval and one-count recall admission

**Files:**
- Modify: `narrative_dynamics/abm/situated_memory_contracts.py`
- Modify: `narrative_dynamics/abm/situated_memory.py`
- Create: `narrative_dynamics/abm/situated_memory_cognition.py`
- Create: `tests/test_network_abm_situated_memory_cognition.py`

**Interfaces:**
- Consumes: Task 1 contracts, `search_situated_memories`, `SituatedPerspectiveEvent`, and the V11 observation-rule matcher.
- Produces: `SituatedMemoryRecallAdmission`, `SituatedMemoryRecallResult`, `recall_situated_memories()`, and exact-world filtering in `SituatedMemoryQuery.story_model_hash`.

- [ ] **Step 1: Write failing exact-world query tests**

  Persist same-agent records from two literal world hashes and assert a query naming
  one hash returns only that world. The production change caught is cross-world
  memory entering a valid agent scope.

- [ ] **Step 2: Run the focused V12 query test and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory.SituatedMemoryRetrievalTests.test_story_model_hash_filter_is_authoritative -v`.
  Expected: `SituatedMemoryQuery` lacks `story_model_hash`.

- [ ] **Step 3: Implement exact-world structured filtering**

  Add a validated optional content hash to `SituatedMemoryQuery` and a parameterized
  `m.story_model_hash = ?` clause to V12 search. Keep existing query ordering and
  privacy behavior unchanged.

- [ ] **Step 4: Write failing recall-admission tests**

  With real SQLite rows, assert only the scoped agent/world/past/active memories are
  candidates; processed and already-recalled IDs are excluded; one inspection with
  confidence/salience one changes `0.5/0.5` to literal `0.9/0.1`; an auditory memory
  uses the declared tempered likelihood; unmatched recall changes no belief but adds
  the private event ID; and a second call is idempotent. These tests catch cross-scope
  leaks, future leakage, direct-plus-recall double counting, and repeated Bayes.

- [ ] **Step 5: Run recall tests and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory_cognition.SituatedMemoryRecallTests -v`.
  Expected: recall runtime symbols are unavailable.

- [ ] **Step 6: Implement retrieval, reconstruction, and tempered admission**

  Query cues canonically with the current round ceiling and exact model hash,
  reconstruct and hash-check each perspective event, deduplicate/cap results, and
  apply `tempered(h) = 1 - w + w*L(h)` where
  `w = confidence*salience`. Preserve complete recall provenance and update only the
  requesting mind's belief, recalled-memory IDs, and observed-event IDs.

- [ ] **Step 7: Run Task 2 tests and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory tests.test_network_abm_situated_memory_cognition -v`.
  Expected: all V12 retrieval and V13 recall tests pass.

- [ ] **Step 8: Commit retrieval and admission**

  Run `git add narrative_dynamics/abm/situated_memory_contracts.py narrative_dynamics/abm/situated_memory.py narrative_dynamics/abm/situated_memory_cognition.py tests/test_network_abm_situated_memory.py tests/test_network_abm_situated_memory_cognition.py && git commit -m "feat: recall private situated memories once"`.

### Task 3: Memory-augmented cognitive rounds and office acceptance

**Files:**
- Modify: `narrative_dynamics/abm/situated_cognition.py`
- Modify: `narrative_dynamics/abm/situated_memory_cognition.py`
- Modify: `tests/test_network_abm_situated_memory_cognition.py`
- Create: `tests/test_network_abm_situated_memory_cognition_story.py`

**Interfaces:**
- Consumes: Task 2 recall result, V11 direct admission/planning, and V10 synchronous story resolution.
- Produces: recall-aware decision provenance, `SituatedMemoryCognitiveRoundResult`, `SituatedMemoryCognitiveTrajectory`, `simulate_situated_memory_cognitive_round()`, and `simulate_situated_memory_cognition()`.

- [ ] **Step 1: Write failing decision-provenance and office control tests**

  Build an office history where Alice privately inspected the memo then moved open,
  persist it, and restart cognition at that checkpoint. Assert the no-cue control
  keeps belief `0.5/0.5` and chooses `wait`; the recall arm reports the exact memory,
  updates to `0.9/0.1`, restores only Alice's source event, and chooses `tell`; Bob
  and Dana report no recall. Assert the explanation separates direct evidence from
  recalled memory.

- [ ] **Step 2: Run the office acceptance test and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory_cognition_story -v`.
  Expected: memory-cognitive round/trajectory APIs are unavailable.

- [ ] **Step 3: Extend V11 decision provenance without changing V11 behavior**

  Add defaulted `recalled_memory_ids` and `recalled_symbol_ids` to decisions and
  explanations. Expose the existing observation-rule matcher for exact reuse. Extract
  the existing physical commit step into one internal helper used by both V11 and
  V13, preserving direct simulation hashes and ordering for equal V11 inputs.

- [ ] **Step 4: Implement the V13 round and trajectory orchestrators**

  For every agent: ingest its current perspective, filter direct observations above
  its floor, admit direct evidence, recall private memories, decide from the combined
  mind, then synchronously commit all intents. Wrap the V11 round with full recall
  admissions and chain wrappers into a deterministic trajectory. Do not include the
  database path in serialized artifacts.

- [ ] **Step 5: Add future, repeat, and replay acceptance tests**

  Assert later rows already present in SQLite do not enter an earlier checkpoint,
  the same memory does not update belief on the next round, Bob admits Alice's new
  telling only in the following round, and two fresh database paths yield equal
  trajectories and content hashes.

- [ ] **Step 6: Run Task 3 and V11 regression tests and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory_cognition_story tests.test_network_abm_situated_memory_cognition tests.test_network_abm_situated_cognition -v`.
  Expected: all V13 scenarios and existing V11 cognition pass.

- [ ] **Step 7: Commit round integration**

  Run `git add narrative_dynamics/abm/situated_cognition.py narrative_dynamics/abm/situated_memory_cognition.py tests/test_network_abm_situated_memory_cognition.py tests/test_network_abm_situated_memory_cognition_story.py && git commit -m "feat: let recalled memories change situated decisions"`.

### Task 4: Public API, documentation, verification, and PR update

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete V13 contracts and runtime.
- Produces: public V13 package surface, runnable user example, verified and pushed branch, and updated PR #44.

- [ ] **Step 1: Write the failing public-surface assertion**

  Update the exact package export test to the V13 surface before adding exports. It
  must fail on every missing recall contract/runtime symbol rather than merely count
  names.

- [ ] **Step 2: Run the public API test and verify RED**

  Run `python -m unittest tests.test_network_abm_public_api -v`.
  Expected: the package lacks the newly required V13 symbols.

- [ ] **Step 3: Export and document V13**

  Export all recall models, admissions, round/trajectory artifacts, initializer, and
  simulation functions. Add a README example comparing the same checkpoint with and
  without the restructuring cue and asserting the literal belief/action difference.
  State that V14, not V13, owns consolidation, contradiction, relationship learning,
  and forgetting.

- [ ] **Step 4: Run fresh scoped verification**

  Run `python -m unittest discover -s tests -p "test_network_abm*.py"`, execute every
  README Python block, run `python -m compileall -q narrative_dynamics tests`, and run
  `git diff --check`. Expected: all ABM tests and examples pass with no compile or
  whitespace error.

- [ ] **Step 5: Request independent code review and address findings**

  Review the full V13 diff against the spec, emphasizing double-count prevention,
  temporal/world/agent scope, checkpoint ancestry, deterministic replay, and V11
  compatibility. Fix every Critical or Important finding with its own RED→GREEN test.

- [ ] **Step 6: Commit, push, and update PR #44**

  Commit public integration as `feat: expose memory-augmented situated cognition`,
  push `feature/network-interaction-emergence-v1`, and update PR #44 with V13
  semantics and exact verification evidence. Preserve the linked worktree for PR
  feedback.

## Self-review

- Spec coverage: checkpoint initialization, cue model, exact agent/world/time scope,
  direct-vs-recall separation, confidence weighting, one-count evidence, event-source
  restoration, round ordering, audit artifacts, deterministic replay, public API,
  documentation, regression verification, and PR delivery each map to a task.
- Placeholder scan: no deferred implementation phrase or unspecified error handling
  remains; V14 work is explicitly out of scope.
- Type consistency: Task 1 models feed Task 2 queries/admissions; Task 2 recall results
  feed Task 3 decisions/rounds; Task 4 exports the exact symbols produced earlier.
