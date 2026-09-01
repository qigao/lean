# Situated Network Runtime V19 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one deterministic V19 snapshot, metric, and round runtime over the existing physical, perceptual, cognitive, memory, and directed social state.

**Architecture:** Add immutable V19 contracts without changing V10-V18 values, then derive a multiplex graph from exact current state and finally wrap the existing V15.1 social-cognitive transition in a parent-linked runtime. Network projection is read-only: physical events, sanitized percepts, private beliefs, and social trust keep their existing authorities.

**Tech Stack:** Python standard library, frozen dataclasses, SQLite/FTS5 through the existing memory runtime, `unittest`, content-addressed `stable_content_hash`.

**Spec:** `docs/superpowers/specs/2026-09-01-situated-network-runtime-v19-design.md`

## Global Constraints

- Do not modify V10-V18 contracts, transition semantics, or content hashes.
- The ABM core remains dependency-free and deterministic.
- V19 transmission records contain no message, event detail, memory text, prompt, or provider data.
- Every snapshot binds exact story, cognitive, and social hashes at one round.
- Physical lifecycle and V1-V9 rewiring remain explicit V19.1 non-goals.

---

### Task 1: Immutable V19 contracts

**Files:**
- Create: `narrative_dynamics/abm/situated_network_contracts.py`
- Create: `tests/test_network_abm_situated_network_contracts.py`

**Interfaces:**
- Consumes: `SituatedPerceptMemoryCognitiveModel`, `SituatedSocialMemoryModel`, `SituatedPerceptFidelity`, `ObservationChannel`, and `stable_content_hash`.
- Produces: `SituatedNetworkRuntimeModel`, `SituatedNetworkAgentNode`, `SituatedNetworkRelationshipEdge`, `SituatedNetworkAccessEdge`, `SituatedNetworkTransmission`, `SituatedNetworkSnapshot`, `SituatedNetworkEmergenceMetrics`, `SituatedNetworkRuntimeState`, `SituatedNetworkRoundResult`, and `SituatedNetworkTrajectory`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_runtime_model_rejects_missing_common_hypothesis(self):
    with self.assertRaisesRegex(ValueError, "tracked hypothesis"):
        SituatedNetworkRuntimeModel(
            "office-network", "1", memory_model, social_model,
            "missing", 0.7, 0.5,
        )

def test_snapshot_rejects_duplicate_directed_access_pair(self):
    edge = SituatedNetworkAccessEdge("alice", "bob", 1.0, 5.0, True)
    with self.assertRaisesRegex(ValueError, "access pairs must be unique"):
        snapshot_fixture(access_edges=(edge, edge))
```

The first test catches accepting a belief projection that some agent cannot supply. The second catches ambiguous duplicate physical reach.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_network_abm_situated_network_contracts.py -q`

Expected: collection fails because `situated_network_contracts` does not exist.

- [ ] **Step 3: Implement minimal immutable contracts**

Implement strict text/hash/probability validation, canonical tuple ordering, unique node/edge/transmission identities, exact round/hash chaining, `to_dict()`, and `content_hash`. `SituatedNetworkRuntimeModel.__post_init__` must verify exact memory/social cognitive binding and require `tracked_hypothesis_id` in every agent's hypothesis set.

- [ ] **Step 4: Run Task 1 tests GREEN**

Run: `python -m pytest tests/test_network_abm_situated_network_contracts.py -q`

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit Task 1**

```text
git add narrative_dynamics/abm/situated_network_contracts.py tests/test_network_abm_situated_network_contracts.py
git commit -m "feat(abm): add situated network runtime contracts"
```

### Task 2: Multiplex projection and emergence metrics

**Files:**
- Create: `narrative_dynamics/abm/situated_network.py`
- Create: `tests/test_network_abm_situated_network.py`

**Interfaces:**
- Consumes: every Task 1 model/node/edge/snapshot/metric contract plus `derive_situated_perception_reach`, `project_situated_percepts`, and exact V14 relationship/claim state.
- Produces: `project_situated_network_snapshot(model, story, cognitive_state, social_state) -> SituatedNetworkSnapshot` and `measure_situated_network_emergence(model, snapshot, social_state) -> SituatedNetworkEmergenceMetrics`.

- [ ] **Step 1: Write failing projection tests**

```python
def test_closed_door_projects_detected_tell_without_secret_payload(self):
    snapshot = project_situated_network_snapshot(
        runtime_model, closed_story, cognitive_state, social_state
    )
    bob = next(item for item in snapshot.transmissions if item.observer_agent_id == "bob")
    self.assertIs(bob.fidelity, SituatedPerceptFidelity.DETECTED)
    self.assertNotIn("message", snapshot.to_dict())

def test_open_door_projects_exact_tell_and_hand_checked_metrics(self):
    snapshot = project_situated_network_snapshot(
        runtime_model, open_story, cognitive_state, social_state
    )
    metrics = measure_situated_network_emergence(runtime_model, snapshot, social_state)
    self.assertEqual(metrics.population_size, 2)
    self.assertEqual(metrics.latest_tell_event_count, 1)
    self.assertEqual(metrics.exact_transmission_count, 1)
```

These tests catch leaking the TELL message into the network layer, losing sanitized fidelity, and miscounting actual latest-round reach.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_network_abm_situated_network.py -q`

Expected: import failure because the projection functions do not exist.

- [ ] **Step 3: Implement snapshot projection**

Validate exact round/model/hash alignment. Build nodes from the current world bodies and corresponding minds, relationship edges from V14 source relationships, access edges by deriving V15 reach from each source place, and latest-round TELL transmissions only from `project_situated_percepts` over the latest accepted round.

- [ ] **Step 4: Implement aggregate metrics**

Use literal denominators: adoption rate is adopted/population, tracked belief variance is population variance, mean trust is over all directed relationship edges, reached observers are unique non-source observers, and all latest-round counts are zero at round zero.

- [ ] **Step 5: Run Task 1-2 tests GREEN**

Run: `python -m pytest tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py -q`

Expected: all V19 tests pass.

- [ ] **Step 6: Commit Task 2**

```text
git add narrative_dynamics/abm/situated_network.py tests/test_network_abm_situated_network.py
git commit -m "feat(abm): project situated multiplex emergence"
```

### Task 3: Atomic runtime, trajectory, public API, and documentation

**Files:**
- Modify: `narrative_dynamics/abm/situated_network.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_situated_network.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 runtime state/round/trajectory contracts; Task 2 projection and metrics; `simulate_situated_percept_social_cognitive_round`.
- Produces: `initialize_situated_network_runtime`, `simulate_situated_network_round`, and `simulate_situated_network_runtime` as public `narrative_dynamics.abm` APIs.

- [ ] **Step 1: Write failing atomic-round and chain tests**

```python
def test_atomic_round_keeps_story_cognition_social_snapshot_and_metrics_synchronized(self):
    initial = initialize_situated_network_runtime(
        runtime_model, story, cognitive_state, social_state
    )
    result = simulate_situated_network_round(database, runtime_model, initial)
    self.assertEqual(result.next_state.round_index, initial.round_index + 1)
    self.assertEqual(result.next_state.story.content_hash, result.next_state.snapshot.story_hash)
    self.assertEqual(result.next_state.cognitive_state.content_hash, result.next_state.snapshot.cognitive_state_hash)
    self.assertEqual(result.next_state.social_state.content_hash, result.next_state.snapshot.social_state_hash)

def test_trajectory_rejects_a_broken_parent_chain(self):
    with self.assertRaisesRegex(ValueError, "exact chain"):
        replace(valid_trajectory, final_state=unrelated_state)
```

The tests catch partial commits where one subsystem advances without the others and trajectories that accept unrelated final values.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest tests/test_network_abm_situated_network.py -q`

Expected: import failure for the runtime functions.

- [ ] **Step 3: Implement initialization, one-round orchestration, and trajectory**

Initialization validates V15.1 and V14 state using their existing validators. A round calls the existing social-cognitive round exactly once, then derives one next snapshot and metric value. Multi-round simulation accepts only a positive integer count and returns the exact parent-linked state chain.

- [ ] **Step 4: Export the exact V19 surface and document the office case**

Add the Task 1 contracts and five Task 2-3 functions to `narrative_dynamics.abm.__all__`. README must state that V19 unifies observation/runtime state but does not yet implement physical lifecycle or V1-V9 rewiring feedback.

- [ ] **Step 5: Run focused and authoritative regression gates**

Run: `python -m pytest tests/test_network_abm_situated_network_contracts.py tests/test_network_abm_situated_network.py tests/test_network_abm_public_api.py -q`

Run: `python -m unittest discover -s tests -p "test_network_abm*.py"`

Run: `python -m compileall -q narrative_dynamics`

Expected: all focused tests pass, all authoritative network ABM tests pass, and compilation exits zero.

- [ ] **Step 6: Commit Task 3**

```text
git add README.md narrative_dynamics/abm/__init__.py narrative_dynamics/abm/situated_network.py tests/test_network_abm_situated_network.py tests/test_network_abm_public_api.py
git commit -m "feat(abm): run unified situated network rounds"
```

## Self-review

- Spec coverage: runtime binding, node/relationship/access/transmission projection, aggregates, atomic round, trajectory, privacy, public API, and documentation each map to Tasks 1-3.
- Placeholder scan: no unresolved marker, generic error-handling step, or undefined follow-up implementation remains.
- Type consistency: Task 2 produces the exact snapshot/metric values consumed by Task 3; every public runtime function accepts the Task 1 model/state types.
