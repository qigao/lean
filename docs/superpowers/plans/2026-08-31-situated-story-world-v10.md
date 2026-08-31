# Situated Story World V10 Implementation Plan

> **For agentic workers:** Implement task-by-task with strict red/green TDD and run
> `superpowers:verification-before-completion` before any completion claim.

**Goal:** Add a deterministic physical world, independent embodied actions, private
observation projection, an objective event ledger, causal story queries, and replay.

**Architecture:** Frozen, content-hashed world contracts define places, passages,
agents, objects, and exact state. A synchronous resolver evaluates one action per
agent from a prior snapshot, resolves resource conflicts canonically, commits one
parent-linked next state, and projects local observations. A story aggregate checks
private evidence access and exposes objective, perspective, causal, explanation, and
replay queries without coupling V10 to V11 cognition.

**Tech Stack:** Python 3 standard library, frozen dataclasses, stable content hashes,
`unittest`.

**Spec:** `docs/superpowers/specs/2026-08-31-situated-story-world-v10-design.md`

## Global constraints

- No shared chatroom or global agent-visible state.
- One canonical action per agent per synchronous round; absent actions become waits.
- Preconditions read only the prior snapshot.
- Canonical conflict resolution and stable hashes make input order irrelevant.
- Each object has exactly one physical location or holder.
- Causal references are existing, strictly prior events.
- `TELL` may cite only events previously observed by its speaker.
- Perspective queries return only that agent's projected observations.
- V11 beliefs, goals, motives, and policies stay outside V10.

---

### Task 1: Situated world contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_contracts.py`
- Create: `tests/situated_fixtures.py`
- Create: `tests/test_network_abm_situated_contracts.py`

**Interfaces:**
- `PlaceSpec`, `PassageSpec`, `EmbodiedAgentSpec`, `WorldObjectSpec`
- `AgentBodyState`, `WorldObjectState`, `PassageState`
- `SituatedWorldModel`, `SituatedWorldState`, `initialize_situated_world`

- [ ] Write tests for canonical initialization, exact roster coverage, object
  exclusivity, passage endpoints, inventory capacity, parent linkage, and hashes.
- [ ] Run `python -m unittest tests.test_network_abm_situated_contracts -v` and
  confirm an import failure.
- [ ] Implement immutable validated contracts and round-zero initialization.
- [ ] Re-run the contract tests to green.
- [ ] Commit contracts, fixtures, specification, and plan.

### Task 2: Synchronous actions, event ledger, and local observations

**Files:**
- Create: `narrative_dynamics/abm/situated.py`
- Create: `tests/test_network_abm_situated.py`

**Interfaces:**
- `SituatedActionKind`, `SituatedActionIntent`
- `SituatedWorldEvent`, `SituatedObservation`, `SituatedRoundResult`
- `resolve_situated_round`

- [ ] Write failing tests for movement, closed passages, look/inspection privacy,
  taking/dropping, capacity, local telling, implicit waits, and canonical take
  conflicts.
- [ ] Confirm RED with `python -m unittest tests.test_network_abm_situated -v`.
- [ ] Implement prior-snapshot validation, canonical conflict selection, atomic
  state commit, deterministic ledger events, and channel-specific projection.
- [ ] Test reversed input ordering for exact result/hash identity.
- [ ] Re-run action tests to green and commit.

### Task 3: Story aggregate, causal queries, and replay

**Files:**
- Create: `narrative_dynamics/abm/situated_story.py`
- Create: `tests/test_network_abm_situated_story.py`

**Interfaces:**
- `SituatedStory`, `SituatedEventExplanation`
- `initialize_situated_story`, `advance_situated_story`, `replay_situated_story`
- `objective_timeline`, `perspective_timeline`, `direct_causes`,
  `causal_ancestors`, `information_chain`, `explain_situated_event`

- [ ] Write the office propagation case and failing query tests.
- [ ] Confirm RED with `python -m unittest tests.test_network_abm_situated_story -v`.
- [ ] Validate the round/state/event chain and speaker-private evidence references.
- [ ] Implement objective and perspective timelines plus deterministic causal graph
  traversal and grounded explanations.
- [ ] Replay the same schedule and assert exact story/hash identity.
- [ ] Re-run story tests to green and commit.

### Task 4: Public API, executable example, and full verification

**Files:**
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

- [ ] Export the complete V10 surface and update the exact public API test.
- [ ] Add an executable small-office README example demonstrating physical movement,
  private inspection, local propagation, perspective divergence, and causal tracing.
- [ ] Execute every README Python block independently.
- [ ] Run all network ABM tests and targeted narrative regressions.
- [ ] Run `python -m compileall -q narrative_dynamics tests` and a diff check.
- [ ] Inspect the final diff, commit, push the feature branch, and update PR 44.

## Plan self-review

- No placeholder implementations are permitted.
- Every public artifact must have deterministic `to_dict()` and `content_hash`.
- Tests must prove epistemic isolation, not merely event creation.
- The office case must prove a source chain across at least two tellings.
- The final diff must not modify unrelated user work.
