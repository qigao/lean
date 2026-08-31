# Situated Social Memory Revision V14 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic claim consolidation, contradiction history, directed source-trust/relationship learning, bounded forgetting, and trust-weighted later recall to situated agents.

**Architecture:** Frozen V14 contracts wrap the exact V13 model and state. A pure social transition consumes explicit finite-symbol evidence and produces a content-addressed relationship/claim chain. A thin V14 orchestrator reuses the V13 physical/cognitive round, supplies current trust multipliers for testimony recall, projects audited admissions into social evidence, and commits both chains without changing V13 defaults.

**Tech Stack:** Python 3 standard library, frozen dataclasses, SQLite/FTS5 through V12/V13, existing Bayesian/POMDP runtime, `unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-situated-social-memory-revision-v14-design.md`

## Global Constraints

- V13 public behavior and pure V11 golden hashes remain unchanged.
- Topics and symbols are finite and declarative; no free-text inference is added.
- Trust is directed and bounded in `[0, 1]`; affinity is bounded in `[-1, 1]`.
- Social evidence is idempotent and every state is parent-linked and content-addressed.
- Forgetting never deletes or deactivates authoritative V12 SQLite rows.
- Learned trust affects only later recalled external testimony.
- Add no LLM, embeddings, LSTM, Transformer, RNG, or third-party dependency.

---

### Task 1: Social memory contracts and checkpoint state

**Files:**
- Create: `narrative_dynamics/abm/situated_social_memory_contracts.py`
- Create: `tests/test_network_abm_situated_social_memory_contracts.py`

**Interfaces:**
- Consumes: `SituatedMemoryCognitiveModel`, `SituatedCognitiveState`.
- Produces: `SituatedClaimTopic`, `SituatedSocialMemoryPolicy`, `SituatedSocialMemoryModel`, `SituatedSourceRelationship`, `SituatedClaimStatus`, `SituatedConsolidatedClaim`, `SituatedSocialMemoryState`, `SituatedSocialEvidence`, and `initialize_situated_social_memory()`.

- [ ] **Step 1: Write failing contract tests**

  Assert canonical topic/policy/model ordering, unique symbol ownership, exact agent
  roster, complete ordered relationship pairs, bounded trust/affinity, valid claim
  status/history, checkpoint round/hash binding, and empty processed evidence.

- [ ] **Step 2: Run and verify RED**

  Run `python -m unittest tests.test_network_abm_situated_social_memory_contracts -v`.
  Expected: import failure because the V14 contracts module does not exist.

- [ ] **Step 3: Implement minimal frozen contracts**

  Implement complete validation and canonical `to_dict()`/`content_hash`. Use:

  ```python
  initialize_situated_social_memory(
      model: SituatedSocialMemoryModel,
      cognitive_state: SituatedCognitiveState,
  ) -> SituatedSocialMemoryState
  ```

  Initialize every `(observer, source)` pair where IDs differ with policy trust,
  affinity zero, and zero counters. Bind the V14 state to
  `cognitive_state.content_hash` and its round.

- [ ] **Step 4: Run and verify GREEN, then commit**

  Run the V14 contract tests plus V13 contract tests. Commit as
  `feat: define situated social memory contracts`.

### Task 2: Pure consolidation, contradiction, trust, and forgetting transition

**Files:**
- Create: `narrative_dynamics/abm/situated_social_memory.py`
- Create: `tests/test_network_abm_situated_social_memory.py`

**Interfaces:**
- Consumes: Task 1 contracts.
- Produces: `SituatedSocialMemoryUpdate`, `advance_situated_social_memory()`.

- [ ] **Step 1: Write failing consolidation and contradiction tests**

  Submit two literal same-symbol testimony items for Bob from Alice and assert one
  active claim, `support_count == 2`, and both event IDs. Submit a later opposite
  symbol and assert the first claim is `SUPERSEDED`, the second is active, and both
  remain auditable.

- [ ] **Step 2: Run and verify RED**

  Run the two focused tests. Expected: runtime module/symbol import failure.

- [ ] **Step 3: Implement canonical testimony transition**

  Process unseen testimony ordered by `(round_index, evidence_id)`. Consolidate the
  active identical `(observer, source, topic, symbol)` claim; otherwise supersede the
  prior active `(observer, source, topic)` claim and create a stable claim ID from the
  first evidence content.

- [ ] **Step 4: Write failing verification and directed-relationship tests**

  From trust `0.5`, confirmation rate `0.2` must yield `0.6`; contradiction rate
  `0.4` must yield `0.3`. Assert only Bob-to-Alice changes, affinity clamps, counters
  increment once, and replayed verification is idempotent.

- [ ] **Step 5: Implement verification learning and verify GREEN**

  Collect testimony before verification in each batch. Confirm/contradict active
  claims for the same observer/topic, update only their directed source links, mark
  terminal status, and record every evidence ID exactly once.

- [ ] **Step 6: Write RED forgetting tests and implement bounds**

  Assert unresolved age `max_age + 1` becomes `FORGOTTEN`, age exactly `max_age`
  remains active, and capacity forgets oldest active claims with lexical ID ties.
  Confirm SQLite is not accepted or mutated by this pure function.

- [ ] **Step 7: Run Task 2 suite and commit**

  Run V14 runtime/contract tests. Commit as
  `feat: evolve claims and source relationships`.

### Task 3: Trust-weighted recall and V14 round orchestration

**Files:**
- Modify: `narrative_dynamics/abm/situated_memory_cognition.py`
- Create: `narrative_dynamics/abm/situated_social_cognition.py`
- Create: `tests/test_network_abm_situated_social_cognition.py`

**Interfaces:**
- Consumes: V13 recall/round APIs and Task 2 transition.
- Produces: internal trust-aware V13 round hook, `SituatedSocialCognitiveRoundResult`, `SituatedSocialCognitiveTrajectory`, `simulate_situated_social_cognitive_round()`, and `simulate_situated_social_cognition()`.

- [ ] **Step 1: Write failing literal trust-weight test**

  For auditory confidence `0.7`, salience `0.75`, and source trust `0.5`, assert
  effective weight `0.2625` and approved posterior
  `0.5605187319884727` from a `0.5/0.5` prior. Assert V13 default trust keeps weight
  `0.525` and posterior `0.6423728813559322`.

- [ ] **Step 2: Run and verify RED**

  Expected: the recall API has no private source-trust multiplier path.

- [ ] **Step 3: Add minimal internal trust multiplier**

  Refactor the V13 orchestrator behind an internal function receiving canonical
  `source_trust_by_observer`. Multiply only external `TELL` memory weight by the
  current observer-to-actor trust. Keep public V13 signatures/default results and
  V11 hashes unchanged.

- [ ] **Step 4: Write failing V14 round projection tests**

  Run a physical office round with recalled testimony and assert the wrapper projects
  the exact observer/source/topic/symbol/event/memory evidence, advances social state
  after the decision, and binds it to the next cognitive state hash.

- [ ] **Step 5: Implement projection and orchestration**

  Resolve admitted IDs against the agent's exact private perspective, reuse the V11
  rule matcher, ignore self testimony and undeclared symbols, call
  `advance_situated_social_memory`, and wrap the V13 cognitive round plus social
  update. Chain rounds without serializing the database path.

- [ ] **Step 6: Add multi-round acceptance and replay tests**

  Assert learned trust changes only later recall, repeated evidence is idempotent,
  and two fresh database paths yield identical trajectories/content hashes.

- [ ] **Step 7: Run V14/V13/V11 regression tests and commit**

  Commit as `feat: let social memory shape later recall`.

### Task 4: Public API, README, review, verification, and PR update

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: the exact V14 package surface and an executable social-memory example.

- [ ] **Step 1: Update exact public API test and verify RED**

  Require every V14 contract, transition, result, initializer, and simulator symbol.
  Expected: missing exports.

- [ ] **Step 2: Export and document V14**

  Add imports/`__all__` and a README example showing repeated claims consolidate,
  direct verification changes Bob-to-Alice trust, and later auditory recall is
  trust-tempered. State the finite-symbol and deterministic-forgetting boundaries.

- [ ] **Step 3: Run fresh verification**

  Run `python -m unittest discover -s tests -p "test_network_abm*.py"`, all README
  Python blocks, `python -m compileall -q narrative_dynamics tests`, and
  `git diff --check`.

- [ ] **Step 4: Independent review and fixes**

  Review exact V14 diff for directed-link isolation, one-count evidence, temporal
  ordering, terminal-claim idempotency, destructive forgetting, V13 compatibility,
  and deterministic hashes. Fix all Critical/Important findings with RED-GREEN tests.

- [ ] **Step 5: Commit, push, and update PR #44**

  Commit public integration as `feat: expose situated social memory revision`, push
  `feature/network-interaction-emergence-v1`, update PR #44 to V14, and preserve the
  linked worktree for feedback.

## Self-review

- Spec coverage: contracts, categorical consolidation, contradiction history,
  verification-based directed learning, bounded non-destructive forgetting,
  trust-weighted later recall, physical-round integration, audit artifacts,
  deterministic replay, public API, documentation, review, and delivery each map to
  an explicit task.
- Placeholder scan: no deferred implementation phrase is used as an executable step;
  every deliberate limitation is in the design's deferred-work section.
- Type consistency: Task 1 state feeds Task 2 transition; Task 2 relationships feed
  Task 3 trust multipliers; Task 3 artifacts are exactly the symbols exported in
  Task 4.

