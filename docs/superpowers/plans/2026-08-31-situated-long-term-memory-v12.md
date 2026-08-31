# Situated Long-Term Memory V12 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, agent-private, SQLite-backed long-term memory with a rebuildable FTS5 trigram index for V10/V11 situated stories.

**Architecture:** Immutable contracts define policy, records, scoped queries, hits, and reports. A focused persistence module projects only `perspective_timeline(story, agent_id)` into authoritative SQLite rows and exposes scoped list/search/activation/rebuild functions. FTS5 accelerates literal text lookup but never becomes the source of truth or affects V11 decisions.

**Tech Stack:** Python 3 standard library (`dataclasses`, `enum`, `json`, `pathlib`, `sqlite3`), SQLite 3.45+ FTS5/trigram, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-situated-long-term-memory-v12-design.md`

## Global Constraints

- Persist only events returned by `perspective_timeline(story, agent_id)`.
- Require explicit `agent_id` scope on every ingestion, read, search, and activation operation.
- Use SQLite ordinary tables as authoritative storage and FTS5 only as a rebuildable index.
- Add no third-party dependency and no LSTM, Transformer, or embedding model.
- Do not inject recalled memories into V11 belief updates or decisions in V12.
- Preserve deterministic hashes and ordering for identical story, policy, and query inputs.

---

### Task 1: Immutable memory contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_memory_contracts.py`
- Create: `tests/test_network_abm_situated_memory_contracts.py`

**Interfaces:**
- Consumes: `ObservationChannel`, `SituatedActionKind`, `EvidenceFact`, and `stable_content_hash`.
- Produces: `MemoryChannelPolicy`, `SituatedMemoryPolicy`, `SituatedMemoryRecord`, `SituatedMemoryQuery`, `SituatedMemorySearchHit`, `SituatedMemoryWriteReport`, `SituatedMemoryIndexReport`, and `standard_situated_memory_policy()`.

- [ ] **Step 1: Write failing validation and canonicalization tests**

  Add tests with literal expected values proving that a policy must cover all four
  channels exactly once, weights and limits are bounded, query event kinds are
  canonicalized, record detail/cause ordering is canonical, and content hashes are
  stable. The production change caught is accepting incomplete/ambiguous policies
  or nondeterministic records.

- [ ] **Step 2: Run the contract test and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory_contracts -v`.
  Expected: import failure because `situated_memory_contracts` does not exist.

- [ ] **Step 3: Implement the minimal immutable contracts**

  Implement frozen dataclasses with the repository's validation style, complete
  `to_dict()` methods, and `content_hash` properties. Define the standard policy
  with the exact channel values in the design spec. A query requires a non-empty
  agent ID, accepts optional literal text and structured filters, rejects an empty
  string, validates `1 <= limit <= 1000`, and validates its round interval.

- [ ] **Step 4: Run the contract test and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory_contracts -v`.
  Expected: all contract tests pass without warnings.

- [ ] **Step 5: Commit the contracts**

  Run `git add narrative_dynamics/abm/situated_memory_contracts.py tests/test_network_abm_situated_memory_contracts.py && git commit -m "feat: define situated long-term memory contracts"`.

### Task 2: SQLite authoritative storage and FTS5 retrieval

**Files:**
- Create: `narrative_dynamics/abm/situated_memory.py`
- Create: `tests/test_network_abm_situated_memory.py`

**Interfaces:**
- Consumes: all Task 1 contracts and `SituatedPerspectiveEvent`.
- Produces: `initialize_situated_memory`, `ingest_situated_memory`, `list_situated_memories`, `search_situated_memories`, `set_situated_memory_active`, `rebuild_situated_memory_index`, `SituatedMemoryStorageError`, and `SituatedMemoryConflictError`.

- [ ] **Step 1: Write failing real-SQLite tests for schema and ingestion**

  Use `TemporaryDirectory` and a real file database. Assert initialization creates
  schema version 1; ingesting two literal perspective events writes two complete
  structured rows; a repeated ingest reports zero inserts and two existing rows;
  and an identity collision with changed content raises `SituatedMemoryConflictError`.
  The production changes caught are missing transactions, lost fields, duplicate
  memories, and silent history replacement.

- [ ] **Step 2: Run the focused storage tests and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory.SituatedMemoryStorageTests -v`.
  Expected: import or missing-function failure for the absent persistence module.

- [ ] **Step 3: Implement schema creation and transactional record writes**

  Create schema version 1, the external-content FTS5 trigram table, and sync
  triggers. Serialize details and causes using canonical JSON. Derive the summary
  from round, actor, kind, place, target, outcome, and every detail. Use parameterized
  SQL throughout. Compare stored and incoming hashes before deciding whether an
  observation is existing or conflicting.

- [ ] **Step 4: Run the storage tests and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory.SituatedMemoryStorageTests -v`.
  Expected: schema and ingestion tests pass.

- [ ] **Step 5: Write failing scoped retrieval, activation, and rebuild tests**

  Assert that every query returns only its requested agent, structured filters work,
  three-character Chinese and English substrings use lexical ranking, one-character
  literal queries fall back correctly, malformed inputs never become SQL, inactive
  rows are hidden by default, cross-agent activation is rejected, and rebuilding
  produces identical memory IDs. The production changes caught are privacy leaks,
  unsafe MATCH parsing, incorrect filtering, and treating FTS as authoritative.

- [ ] **Step 6: Run the focused retrieval tests and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory.SituatedMemoryRetrievalTests -v`.
  Expected: missing search/activation/rebuild behavior failures.

- [ ] **Step 7: Implement scoped list/search/activation/rebuild behavior**

  Join FTS rows to authoritative records with `agent_id` as a mandatory SQL filter.
  Quote text as a literal phrase, use a parameterized `LIKE` fallback below three
  characters, and rank FTS candidates with an agent-local literal-occurrence score
  that cannot change with another agent's corpus. Apply stable tie-breaking by
  salience, round, and memory ID.
  Activation must update exactly one row owned by the caller. Rebuild with the FTS5
  external-content `rebuild` command and report indexed authoritative row count.

- [ ] **Step 8: Run all Task 2 tests and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory -v`.
  Expected: all storage and retrieval tests pass without warnings.

- [ ] **Step 9: Commit persistence and retrieval**

  Run `git add narrative_dynamics/abm/situated_memory.py tests/test_network_abm_situated_memory.py && git commit -m "feat: persist and search private situated memories"`.

### Task 3: Story projection, public API, and office acceptance

**Files:**
- Modify: `narrative_dynamics/abm/situated_memory.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Create: `tests/test_network_abm_situated_memory_story.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: `perspective_timeline(story, agent_id)` and Task 2 persistence functions.
- Produces: `ingest_situated_story` and the complete V12 public package API.

- [ ] **Step 1: Write the failing office privacy and durability acceptance tests**

  Reuse the real office fixture and build the inspection/movement/telling story.
  Assert Alice can search the private inspection, Bob sees only the later audible
  telling, Dana sees neither, repeat ingestion is idempotent, close/reopen preserves
  hashes, and the round-one memory remains searchable after later wait rounds. The
  production changes caught are objective-ledger ingestion, cross-agent leakage,
  volatile storage, and time-based accidental loss.

- [ ] **Step 2: Run the story test and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_memory_story -v`.
  Expected: `ingest_situated_story` is unavailable or lacks the required projection.

- [ ] **Step 3: Implement perspective-only story ingestion**

  Validate the requested agent belongs to the story, call only
  `perspective_timeline(story, agent_id)`, derive one record per returned item using
  the declared channel policy, and delegate the atomic write to
  `ingest_situated_memory`. Never call `objective_timeline` in this module.

- [ ] **Step 4: Export and document the V12 API**

  Add all public contracts, errors, and functions to `narrative_dynamics.abm`.
  Add a runnable README example that persists Alice and Bob separately, retrieves
  the restructuring memory, demonstrates Bob cannot retrieve Alice's inspection,
  and states explicitly that V13—not V12—feeds recall into POMDP decisions.

- [ ] **Step 5: Run the story and README examples and verify GREEN**

  Run `python -m unittest tests.test_network_abm_situated_memory_story -v` and the
  repository's README Python-block checker. Expected: all story assertions and
  executable examples pass.

- [ ] **Step 6: Commit integration and documentation**

  Run `git add narrative_dynamics/abm/situated_memory.py narrative_dynamics/abm/__init__.py tests/test_network_abm_situated_memory_story.py README.md && git commit -m "feat: project situated stories into private memory"`.

### Task 4: Regression verification and PR update

**Files:**
- Modify only files required by failures caused by V12.

**Interfaces:**
- Consumes: complete V12 implementation.
- Produces: verified branch and updated existing pull request.

- [ ] **Step 1: Run focused and complete regression suites**

  Run `python -m unittest discover -s tests -p "test_network_abm*.py"`, then the
  repository's complete Python suite and `lake build`. Expected: V12 and all prior
  tests pass; any known CI-environment-only failure is recorded rather than hidden.

- [ ] **Step 2: Inspect repository state and diff**

  Run `git diff --check`, `git status --short`, and review `git diff HEAD^..HEAD`
  plus the branch diff. Confirm no generated database, cache, secret, unrelated
  user change, placeholder, or unscoped memory query is present.

- [ ] **Step 3: Push and update the existing pull request**

  Push `feature/network-interaction-emergence-v1`, update PR #44 title/body with
  V12 behavior and exact verification evidence, and report the commit and PR URL.

## Self-review

- Spec coverage: contracts, perspective-only provenance, SQLite authority, FTS5
  rebuild, idempotence/conflict handling, scoped retrieval, activation, determinism,
  durability, office acceptance, API exports, documentation, and regressions each
  map to a task above.
- Placeholder scan: no deferred implementation instruction or unspecified error
  handling remains; V13/V14 work is explicitly out of scope in the spec.
- Type consistency: Task 1 defines every value returned by Task 2; Task 2 defines
  the persistence primitives used by Task 3; the names exported in Task 3 match the
  produced interfaces.
