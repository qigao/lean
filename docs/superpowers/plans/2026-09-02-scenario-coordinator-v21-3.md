# Scenario Coordinator V21.3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one synchronous, capability-scoped `ScenarioCoordinator` that controls an exact compiled scenario run, advances V19 atomically into V21.2 outputs, publishes filtered views, and checkpoints/forks local runs.

**Architecture:** Frozen command/query/checkpoint contracts and a compare-and-swap state-store seam surround one mutable coordinator authority. The coordinator pre-backs up SQLite, performs exactly one V19 transition, derives one canonical V21.2 batch, commits local state/history, then publishes through a synchronous audience-filtering bus. Local checkpoint storage snapshots SQLite atomically and restores exact state into a paused child coordinator without changing semantic identity.

**Tech Stack:** Python standard library (`dataclasses`, `enum`, `typing.Protocol`, `sqlite3`, `tempfile`, `pathlib`, `os`, `threading`), existing V19/V21.1/V21.2 ABM contracts, `unittest`/`pytest`.

**Spec:** `docs/superpowers/specs/2026-09-02-scenario-coordinator-v21-3-design.md`; `docs/superpowers/specs/2026-09-01-web-simulation-platform-v21-design.md`; `docs/superpowers/specs/2026-09-01-scenario-io-live-projection-v21-design.md`

## Global Constraints

- One `ScenarioCoordinator` is the sole writer for one run and one file-backed SQLite memory store.
- The coordinator is synchronous; `start` and `resume` create no thread and `step` advances exactly one V19 round.
- V19 remains the sole world-transition implementation; no direct state mutation or invented event is allowed.
- Every command binds exact run ID, compiled scenario hash, coordinator epoch, expected state hash, authority, command ID, and idempotency key.
- Credentials, capability tokens, database paths, checkpoint paths, subscriber callbacks, and deployment topology never enter canonical hashes.
- Duplicate idempotency keys return the exact cached result only for the exact same request and capability; conflicting reuse is rejected.
- Audience filtering occurs before callback or query return. Agent-private mind/social state is returned only to that Agent or `internal` capability.
- Accepted state/output/history becomes visible before callbacks; delivery failure never rolls back an accepted transition.
- SQLite, projection, output-limit, or compare-and-swap failure during `step` restores the exact prior store and exposes no partial coordinator change.
- Forks restore only an explicit checkpoint, use a distinct database/run/stream, start paused at sequence one, and increment the epoch.
- Add no HTTP, HTTP/2 server, WSS server, JSON-RPC server, browser dependency, X6, PixiJS, Monaco, Protobuf, gRPC, provider, retrieval, or Blender process.
- Do not run the stopped repository-wide suite; use focused V21.3 tests and `python -m unittest discover -s tests -p "test_network_abm*.py" -q`.

---

### Task 1: Command, query, checkpoint, fork, and state-store contracts

**Files:**
- Create: `narrative_dynamics/abm/scenario_coordinator_contracts.py`
- Create: `narrative_dynamics/abm/scenario_state_store.py`
- Create: `tests/test_network_abm_scenario_coordinator_contracts.py`

**Interfaces:**
- Consumes: `SimulationAudienceCapability`, `SituatedNetworkRuntimeState`, `SituatedNetworkSnapshot`, `SituatedNetworkEmergenceMetrics`, `SituatedAgentMindState`, `SituatedSourceRelationship`, `SituatedConsolidatedClaim`.
- Produces: coordinator enums/contracts, `ScenarioStateStore`, and `InMemoryScenarioStateStore` used by Tasks 3–5.

- [ ] **Step 1: Add failing literal contract and CAS tests**

Cover exact enum values:

```python
self.assertEqual(
    tuple(item.value for item in ScenarioRunStatus),
    ("created", "running", "paused", "stopped", "completed"),
)
self.assertEqual(
    tuple(item.value for item in ScenarioCommandKind),
    ("start", "pause", "resume", "step", "stop", "checkpoint"),
)
```

Add real constructor tests proving:

```python
request = ScenarioCommandRequest(
    "command-1", "idem-1", "run-1", SCENARIO_HASH, 1, STATE_HASH,
    "operator", ScenarioCommandKind.START,
)
self.assertEqual(request.to_dict()["kind"], "start")
self.assertEqual(request, dataclasses.replace(request))
self.assertRegex(request.content_hash, r"^sha256:[0-9a-f]{64}$")
```

Reject a checkpoint ID on non-checkpoint commands, missing checkpoint ID on a
checkpoint command, duplicate/unsorted command capability kinds, Agent capabilities
without an owner, malformed hashes, negative rounds, and checkpoint/fork artifacts
whose embedded state does not match their declared state/memory hash.

Add one-line mutation comments and tests proving `InMemoryScenarioStateStore`:

```python
store.initialize("run-1", initial)
self.assertIs(store.load("run-1"), initial)
store.compare_and_swap("run-1", initial.content_hash, advanced)
self.assertIs(store.load("run-1"), advanced)
with self.assertRaisesRegex(ValueError, "stale"):
    store.compare_and_swap("run-1", initial.content_hash, later)
```

Duplicate initialization and unknown-run load must fail without changing existing
state.

- [ ] **Step 2: Run Task 1 RED**

```text
python -m pytest tests/test_network_abm_scenario_coordinator_contracts.py -q
```

Expected: collection fails because the two production modules do not exist.

- [ ] **Step 3: Implement exact enums, schemas, and frozen contracts**

Use exact schemas:

```python
SCENARIO_COMMAND_REQUEST_SCHEMA = "narrative-dynamics.scenario-command-request/v1"
SCENARIO_COMMAND_RESULT_SCHEMA = "narrative-dynamics.scenario-command-result/v1"
SCENARIO_RUN_VIEW_SCHEMA = "narrative-dynamics.scenario-run-view/v1"
SCENARIO_PUBLIC_STATE_VIEW_SCHEMA = "narrative-dynamics.scenario-public-state-view/v1"
SCENARIO_AGENT_STATE_VIEW_SCHEMA = "narrative-dynamics.scenario-agent-state-view/v1"
SCENARIO_CHECKPOINT_SCHEMA = "narrative-dynamics.scenario-checkpoint/v1"
SCENARIO_FORK_REQUEST_SCHEMA = "narrative-dynamics.scenario-fork-request/v1"
SCENARIO_FORK_RESULT_SCHEMA = "narrative-dynamics.scenario-fork-result/v1"
```

Define exact enums:

```python
class ScenarioRunStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    COMPLETED = "completed"

class ScenarioCommandKind(str, Enum):
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    STEP = "step"
    STOP = "stop"
    CHECKPOINT = "checkpoint"

class ScenarioCommandReason(str, Enum):
    ACCEPTED = "accepted"
    UNAUTHORIZED = "unauthorized"
    RUN_MISMATCH = "run_mismatch"
    SCENARIO_MISMATCH = "scenario_mismatch"
    EPOCH_MISMATCH = "epoch_mismatch"
    STALE_STATE = "stale_state"
    INVALID_STATUS = "invalid_status"
    MAXIMUM_ROUNDS = "maximum_rounds"
```

Implement frozen, canonical, content-hashed values with these public signatures:

```python
ScenarioCommandCapability(authority_id, run_id, allowed_kinds, can_fork=False,
    can_read_all_audit=False)
ScenarioCommandRequest(command_id, idempotency_key, run_id, scenario_hash,
    coordinator_epoch, expected_state_hash, authority_id, kind,
    requested_checkpoint_id=None)
ScenarioCommandResult(command_id, idempotency_key, request_hash, capability_hash,
    run_id, scenario_hash, coordinator_epoch, kind, accepted, reason,
    prior_status, next_status, prior_state_hash, next_state_hash, round_index,
    output_batch_hash=None, checkpoint_hash=None)
ScenarioRunView(run_id, stream_id, scenario_hash, coordinator_epoch, status,
    round_index, state_hash, next_sequence, output_batch_hashes,
    checkpoint_hashes, parent_checkpoint_hash=None)
ScenarioPublicStateView(run_id, scenario_hash, round_index, state_hash,
    agent_places, passage_states, object_placements, metrics)
ScenarioAgentStateView(run_id, scenario_hash, round_index, state_hash,
    agent_id, place_id, mind, relationships, claims)
ScenarioCheckpoint(checkpoint_id, run_id, scenario_hash, coordinator_epoch,
    state, next_sequence, parent_checkpoint_hash=None)
ScenarioForkRequest(fork_id, idempotency_key, source_run_id, scenario_hash,
    source_epoch, checkpoint_hash, child_run_id, child_stream_id)
ScenarioForkResult(fork_id, idempotency_key, request_hash, capability_hash,
    source_run_id, child_run_id, child_stream_id, scenario_hash,
    child_epoch, checkpoint_hash, child_state_hash)
```

`ScenarioAgentStateView` requires relationships and claims whose
`observer_agent_id == agent_id`. `ScenarioCheckpoint.to_dict()` includes only
`state_hash` and `memory_store_hash`, never the state tree or path.

- [ ] **Step 4: Implement compare-and-swap storage seam**

Define runtime-checkable `ScenarioStateStore(Protocol)` with the three methods from
the design and an `InMemoryScenarioStateStore` that stores exact typed state objects,
rejects duplicate run initialization, rejects unknown runs, and compares the supplied
expected hash before replacement. Copying or persistence is not added.

- [ ] **Step 5: Run Task 1 GREEN and regression**

```text
python -m pytest tests/test_network_abm_scenario_coordinator_contracts.py tests/test_network_abm_simulation_output_contracts.py -q
```

Expected: all pass without warnings.

- [ ] **Step 6: Commit Task 1**

```text
git add narrative_dynamics/abm/scenario_coordinator_contracts.py narrative_dynamics/abm/scenario_state_store.py tests/test_network_abm_scenario_coordinator_contracts.py
git commit -m "feat(abm): define scenario coordinator contracts"
```

### Task 2: Capability-filtered synchronous output bus

**Files:**
- Create: `narrative_dynamics/abm/simulation_output_bus.py`
- Create: `tests/test_network_abm_simulation_output_bus.py`

**Interfaces:**
- Consumes: V21.2 `SimulationOutputBatch`, `SimulationOutputView`, `SimulationAudienceCapability`, and `SimulationOutputKind`.
- Produces: `SimulationOutputBus`, `SimulationOutputSubscription`, and `SimulationDeliveryReport` used as the Task 4 publisher.

- [ ] **Step 1: Add failing real-batch delivery tests**

Compile and advance the law-firm fixture once, project one real batch, then prove:

```python
public_seen = []
alice_seen = []
bus.subscribe(
    "public-sub", tuple(SimulationOutputKind),
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
    public_seen.append,
)
bus.subscribe(
    "alice-sub", tuple(SimulationOutputKind),
    SimulationAudienceCapability(SimulationOutputAudience.AGENT, "alice"),
    alice_seen.append,
)
report = bus.publish(batch)
self.assertTrue(all(record.audience is SimulationOutputAudience.PUBLIC
                    for record in public_seen[0].records))
self.assertTrue(all(record.audience is SimulationOutputAudience.PUBLIC
                    or record.owner_agent_id == "alice"
                    for record in alice_seen[0].records))
self.assertEqual(report.delivered_subscription_ids, ("alice-sub", "public-sub"))
```

Add tests proving kind filtering preserves source bounds/sequences, a callback exception
does not prevent later delivery, returned non-`None` becomes `non_none_return`, callback
text is absent from reports, reentrant publish is rejected, and unsubscribe during one
callback affects only the next publication. Use a mock only for no external boundary;
all callbacks and batches remain real.

- [ ] **Step 2: Run Task 2 RED**

```text
python -m pytest tests/test_network_abm_simulation_output_bus.py -q
```

Expected: import failure because the bus module does not exist.

- [ ] **Step 3: Implement immutable bus result contracts**

Use exact codes `callback_error`, `non_none_return`, and `reentrant_publish`.

```python
SimulationOutputSubscription(subscription_id, kinds, capability)
SimulationDeliveryFailure(subscription_id, code)
SimulationDeliveryReport(batch_hash, delivered_subscription_ids, failures)
```

All are frozen, sorted, canonical, and content-hashed. Reports contain no callback,
exception object, message, traceback, filesystem path, or subscriber return value.

- [ ] **Step 4: Implement `SimulationOutputBus`**

Public methods:

```python
subscribe(subscription_id, kinds, capability, callback)
    -> SimulationOutputSubscription
unsubscribe(subscription_id) -> None
publish(batch) -> SimulationDeliveryReport
```

Reject duplicate subscription IDs and empty kind sets. Snapshot and sort the internal
subscription/callback pairs before delivery. Derive `SimulationOutputView.from_batch`,
then create a new validated view containing only allowlisted kinds before callback.
Use a publication guard reset in `finally`. A nested call raises `RuntimeError` with
stable text; the outer callback records only `callback_error`.

- [ ] **Step 5: Run Task 2 GREEN**

```text
python -m pytest tests/test_network_abm_simulation_output_bus.py tests/test_network_abm_simulation_output.py -q
```

Expected: all pass without warnings.

- [ ] **Step 6: Commit Task 2**

```text
git add narrative_dynamics/abm/simulation_output_bus.py tests/test_network_abm_simulation_output_bus.py
git commit -m "feat(abm): publish scoped simulation outputs"
```

### Task 3: Capability-scoped query projection

**Files:**
- Create: `narrative_dynamics/abm/scenario_queries.py`
- Create: `tests/test_network_abm_scenario_queries.py`

**Interfaces:**
- Consumes: Task 1 views, existing runtime state, and V21.2 audience capabilities/batches.
- Produces: pure query helpers used by Task 4 without exposing a larger unfiltered state.

- [ ] **Step 1: Add failing query privacy tests**

Using a real initialized and one-round law-firm state, test:

```python
public = project_scenario_public_state("run-1", scenario.content_hash, state)
self.assertEqual(dict(public.agent_places)["alice"], "meeting")
self.assertEqual(public.metrics, state.metrics)

alice = project_scenario_agent_state(
    "run-1", scenario.content_hash, state, "alice",
    SimulationAudienceCapability(SimulationOutputAudience.AGENT, "alice"),
)
self.assertEqual(alice.mind.agent_id, "alice")
self.assertTrue(all(item.observer_agent_id == "alice" for item in alice.relationships))
self.assertTrue(all(item.observer_agent_id == "alice" for item in alice.claims))
```

Assert Bob's Agent capability cannot query Alice, public/analyst capabilities cannot
query raw Agent mind state, public/Agent capabilities cannot request the full network
snapshot, and objective/analyst/internal capabilities receive the exact snapshot.
Assert `project_scenario_output_view` delegates to V21.2 filtering and never returns
another Agent's records.

- [ ] **Step 2: Run Task 3 RED**

```text
python -m pytest tests/test_network_abm_scenario_queries.py -q
```

Expected: import failure because `scenario_queries` does not exist.

- [ ] **Step 3: Implement pure exact query helpers**

Use signatures:

```python
project_scenario_run_view(*, run_id, stream_id, scenario_hash,
    coordinator_epoch, status, state, next_sequence, output_batches,
    checkpoints, parent_checkpoint_hash=None) -> ScenarioRunView
project_scenario_public_state(run_id, scenario_hash, state)
    -> ScenarioPublicStateView
project_scenario_agent_state(run_id, scenario_hash, state, agent_id, capability)
    -> ScenarioAgentStateView
project_scenario_network_state(state, capability) -> SituatedNetworkSnapshot
project_scenario_output_view(batch, capability) -> SimulationOutputView
```

Public physical tuples come from the exact current story state and are sorted by stable
ID. Agent state looks up exactly one mind, its current placement, relationships where
the Agent is observer, and claims where the Agent is observer. Only matching Agent or
`internal` capability is accepted. Full network state accepts only `objective`,
`analyst`, or `internal` capability.

- [ ] **Step 4: Run Task 3 GREEN**

```text
python -m pytest tests/test_network_abm_scenario_queries.py tests/test_network_abm_scenario_coordinator_contracts.py -q
```

Expected: all pass without warnings.

- [ ] **Step 5: Commit Task 3**

```text
git add narrative_dynamics/abm/scenario_queries.py tests/test_network_abm_scenario_queries.py
git commit -m "feat(abm): project scoped scenario queries"
```

### Task 4: Coordinator lifecycle, idempotency, and atomic V19 step

**Files:**
- Create: `narrative_dynamics/abm/scenario_coordinator.py`
- Create: `tests/test_network_abm_scenario_coordinator.py`

**Interfaces:**
- Consumes: Tasks 1–3, `initialize_compiled_scenario`, `simulate_situated_network_round`, `project_simulation_output`, and V21.2 output contracts.
- Produces: `ScenarioCoordinator.create(...)`, command/query methods, retained audit/output history, and the internal checkpoint hooks completed by Task 5.

- [ ] **Step 1: Add failing lifecycle and idempotency tests**

Create a real coordinator over the law-firm fixture and exact capabilities. Assert:

```python
self.assertIs(coordinator.run_view().status, ScenarioRunStatus.CREATED)
started = coordinator.submit_command(start_request, operator)
self.assertTrue(started.accepted)
self.assertIs(coordinator.run_view().status, ScenarioRunStatus.RUNNING)
self.assertIs(coordinator.submit_command(start_request, operator), started)
```

Test the complete legal/illegal status matrix, unauthorized capability, run/scenario/
epoch mismatch, stale state hash, conflicting idempotency reuse, same command ID under a
different key, stopped/completed terminal behavior, and that rejected commands do not
change state/status/sequence/output history.

- [ ] **Step 2: Add failing real-step/output tests**

After `start`, submit one `step` and assert:

```python
self.assertTrue(result.accepted)
self.assertEqual(coordinator.state.round_index, 1)
self.assertEqual(result.next_state_hash, coordinator.state.content_hash)
self.assertIsNotNone(result.output_batch_hash)
view = coordinator.output_view(
    result.output_batch_hash,
    SimulationAudienceCapability(SimulationOutputAudience.INTERNAL),
)
self.assertEqual(view.first_sequence, 1)
self.assertEqual(view.next_state_hash, coordinator.state.content_hash)
self.assertTrue(any(record.kind is SimulationOutputKind.COMMAND_RESULT
                    for record in view.records))
```

Ensure `command.result` appears only when allowlisted, is first by canonical rank, and
binds the command request plus accepted V19 round hashes. A paused step remains paused.
At `maximum_rounds` the run becomes completed and a later step returns
`maximum_rounds`. Enforce `maximum_output_records` after the optional command record.

Add a subscriber test proving state/output history is already queryable inside the
callback and callback failure appears only in `last_delivery_report`.

- [ ] **Step 3: Add failing rollback tests**

Patch only the post-V19 projection boundary to raise after the real V19 function has
updated SQLite. Assert the command raises, the database logical memory hash equals the
pre-step hash, the coordinator state/status/next sequence/audit/output history are
unchanged, and retrying the same request after removing the patch succeeds. Repeat for
state-store compare-and-swap failure and output-record-limit failure. These mocks are
justified at explicit transaction seams; assertions remain on real SQLite/state bytes
and coordinator values.

- [ ] **Step 4: Run Task 4 RED**

```text
python -m pytest tests/test_network_abm_scenario_coordinator.py -q
```

Expected: import failure because the coordinator module does not exist.

- [ ] **Step 5: Implement coordinator creation, control commands, and queries**

Use signature:

```python
ScenarioCoordinator.create(
    database_path,
    scenario,
    *,
    run_id,
    stream_id,
    state_store=None,
    publisher=None,
    coordinator_epoch=1,
    parent_checkpoint_hash=None,
) -> ScenarioCoordinator
```

Creation requires an exact `CompiledSituatedScenario`, initializes the V21.1 database,
initializes the state store, begins at `created`, and records no command/output.

Public methods/properties:

```python
state
run_view()
public_state_view()
agent_state_view(agent_id, capability)
network_state(capability)
output_view(batch_hash, capability)
command_result(command_id, capability)
submit_command(request, capability) -> ScenarioCommandResult
```

Use an execution guard to reject reentrant `submit_command`. Cache by idempotency key
with exact request and capability hashes. Reject duplicate command IDs with different
keys. Implement the lifecycle exactly as the design state machine.

- [ ] **Step 6: Implement atomic step and canonical command record**

Before V19, back up the exact SQLite database into a temporary directory using
`sqlite3.Connection.backup`. After V19:

1. call `project_simulation_output` with current `next_sequence`;
2. if `command.result` is allowlisted, build `SimulationCommandResultPayload`, create a
   public `SimulationOutputRecord` whose source hashes are request and V19 round hashes,
   merge with the projected records, canonical-sort, resequence contiguously, and rebuild
   the `SimulationOutputBatch`;
3. enforce `maximum_output_records`;
4. compare-and-swap state;
5. append output/result histories and advance `next_sequence`;
6. publish after committed values are visible.

If V21.2 projection has no supported base record but `command.result` is allowlisted,
construct a valid command-only batch directly using the accepted V19 prior/next/round
hashes. If neither base nor command record can be produced, restore and raise the stable
projection error. Any exception before publish restores SQLite and leaves all mutable
coordinator fields unchanged.

- [ ] **Step 7: Run Task 4 GREEN and focused regressions**

```text
python -m pytest tests/test_network_abm_scenario_coordinator.py tests/test_network_abm_simulation_output_bus.py tests/test_network_abm_scenario_queries.py -q
```

Expected: all pass without warnings.

- [ ] **Step 8: Commit Task 4**

```text
git add narrative_dynamics/abm/scenario_coordinator.py tests/test_network_abm_scenario_coordinator.py
git commit -m "feat(abm): coordinate atomic scenario runs"
```

### Task 5: Atomic checkpoints, forked coordinators, exports, and documentation

**Files:**
- Create: `narrative_dynamics/abm/scenario_checkpoint_store.py`
- Create: `tests/test_network_abm_scenario_checkpoint_store.py`
- Modify: `narrative_dynamics/abm/scenario_coordinator.py`
- Modify: `tests/test_network_abm_scenario_coordinator.py`
- Modify: `narrative_dynamics/abm/__init__.py`
- Modify: `tests/test_network_abm_public_api.py`
- Modify: `README.md`

**Interfaces:**
- Consumes: Task 1 checkpoint/fork contracts and Task 4 coordinator.
- Produces: `LocalScenarioCheckpointStore`, manual/automatic checkpoint commands, idempotent `ScenarioCoordinator.fork(...)`, complete public V21.3 API/docs.

- [ ] **Step 1: Add failing atomic checkpoint-store tests**

From a real round-one coordinator database, create a typed checkpoint and assert:

```python
stored = store.create(checkpoint, source_database)
self.assertEqual(stored, checkpoint)
self.assertEqual(store.load(checkpoint.content_hash), checkpoint)
store.restore(checkpoint.content_hash, restored_database)
self.assertEqual(
    hash_situated_percept_memory_store(restored_database),
    checkpoint.state.memory_store_hash,
)
```

Test path-safe checkpoint IDs, conflicting ID reuse, unknown hash, restore to a
non-empty target, tampered source store, and `os.replace` failure preserving previous
snapshot bytes with no stage left. Assert exceptions contain no source/target/root path.

- [ ] **Step 2: Add failing coordinator checkpoint/fork tests**

Prove manual checkpoint returns its hash and repeated exact command returns the same
result. With `checkpoint_interval=2`, prove round two automatically creates exactly one
checkpoint and marks the output batch `checkpoint=True`.

Fork from the round-two checkpoint:

```python
child, forked = coordinator.fork(fork_request, fork_capability, child_database)
self.assertIs(child.run_view().status, ScenarioRunStatus.PAUSED)
self.assertEqual(child.state, checkpoint.state)
self.assertEqual(child.run_view().next_sequence, 1)
self.assertEqual(child.run_view().coordinator_epoch,
                 coordinator.run_view().coordinator_epoch + 1)
self.assertEqual(child.run_view().parent_checkpoint_hash, checkpoint.content_hash)
```

Advance parent and child independently and prove divergent state hashes do not change
the other database/state. Reject unauthorized fork, unknown checkpoint, source run/
scenario/epoch mismatch, duplicate child run/stream in one source coordinator,
conflicting fork-key reuse, existing/non-empty child database, and checkpoint restore
failure without returning/caching a child.

- [ ] **Step 3: Run Task 5 RED**

```text
python -m pytest tests/test_network_abm_scenario_checkpoint_store.py tests/test_network_abm_scenario_coordinator.py -q -k "checkpoint or fork"
```

Expected: missing store/import and coordinator checkpoint/fork failures.

- [ ] **Step 4: Implement `LocalScenarioCheckpointStore`**

Constructor accepts one root `str | Path`, creates it, resolves it once, and never puts
the root in artifacts/errors. Public methods:

```python
create(checkpoint, source_database_path) -> ScenarioCheckpoint
load(checkpoint_hash) -> ScenarioCheckpoint
restore(checkpoint_hash, target_database_path) -> None
```

Use a path-safe encoded filename derived from the checkpoint content hash, not the
human checkpoint ID. Validate the source logical memory hash before backup. Write via
same-directory stage, flush/fsync, `os.replace`, cleanup in `finally`, then validate the
published snapshot hash. Restore requires a missing target, writes through a
same-directory stage, validates, and atomically replaces.

- [ ] **Step 5: Integrate manual/automatic checkpoint and idempotent fork**

Extend `ScenarioCoordinator.create(..., checkpoint_store=None)`. A checkpoint command
requires a configured store, creates `ScenarioCheckpoint` from the exact current state
and sequence, then caches the accepted result. Automatic checkpoint uses deterministic
ID `"<run_id>-round-<round_index>"` after every accepted divisible round and rebuilds
the retained batch with `checkpoint=True` before publication/result hashing.

Add:

```python
fork(request, capability, child_database_path)
    -> tuple[ScenarioCoordinator, ScenarioForkResult]
```

Require `capability.can_fork`, exact source run/scenario/epoch, and an existing
checkpoint. Restore first; verify the child store hash; construct the child internally
without calling the empty-database initializer; initialize a fresh local state store
with the exact checkpoint state; begin paused with sequence one and incremented epoch.
Cache exact fork request/capability to child/result only after full success.

- [ ] **Step 6: Export and document the complete V21.3 flow**

Export every public V21.3 schema, enum, contract, state-store interface/implementation,
bus contract/class, query helper, checkpoint store, and coordinator from
`narrative_dynamics.abm`. Extend the public API test to assert object identity with each
source module.

Add a README V21.3 section containing a runnable law-firm outline that creates bus,
state/checkpoint stores, coordinator, capabilities, start/step/query/checkpoint/fork.
State explicitly that start/resume are synchronous status changes, callbacks cannot
issue commands, non-step results remain in the command audit ledger, and JSON-RPC/H2/
WSS, Web editor, story interventions, live Blender, providers, and remote Agents remain
later phases.

- [ ] **Step 7: Run Task 5 GREEN and authorized final gates**

Run exactly:

```text
python -m pytest tests/test_network_abm_scenario_coordinator_contracts.py tests/test_network_abm_simulation_output_bus.py tests/test_network_abm_scenario_queries.py tests/test_network_abm_scenario_coordinator.py tests/test_network_abm_scenario_checkpoint_store.py tests/test_network_abm_simulation_output_contracts.py tests/test_network_abm_simulation_output.py tests/test_network_abm_simulation_output_journal.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py" -q
python -m compileall -q narrative_dynamics tests
git diff --check
```

Expected: all executed tests pass; only the existing environment skip remains;
compileall and diff check exit zero. Do not run the stopped repository-wide suite.

- [ ] **Step 8: Commit Task 5**

```text
git add narrative_dynamics/abm/scenario_checkpoint_store.py tests/test_network_abm_scenario_checkpoint_store.py narrative_dynamics/abm/scenario_coordinator.py tests/test_network_abm_scenario_coordinator.py narrative_dynamics/abm/__init__.py tests/test_network_abm_public_api.py README.md
git commit -m "feat(abm): checkpoint and fork scenario runs"
```
