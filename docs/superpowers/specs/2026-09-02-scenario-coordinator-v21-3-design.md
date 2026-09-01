# Scenario Coordinator V21.3 Design

## Goal

Turn the compiled V21.1 scenario, V19 transition authority, and V21.2 output
projection into one controllable single-process run. A `ScenarioCoordinator` owns
exactly one run, serializes every state-changing request, advances only through the
existing V19 transaction, exposes capability-scoped typed queries, publishes complete
audience-filtered batches synchronously, and creates exact local checkpoints from
which independent runs can fork.

V21.3 is the application authority needed by the later JSON-RPC/H2/WSS service and
pure-Web editor. It is not itself a server and does not introduce a transport.

## Binding choices

- One coordinator is the sole writer for one `run_id` and one SQLite memory store.
- The coordinator is synchronous and single-process. `start` and `resume` change run
  status; they do not create a background thread. A `step` command performs one exact
  barriered V19 round.
- External transport remains JSON-RPC 2.0 over HTTPS/HTTP2 and WSS in a later phase.
  V21.3 adds no HTTP, WebSocket, Protobuf, or gRPC dependency.
- Every state-changing request carries `command_id`, `idempotency_key`, `run_id`,
  `scenario_hash`, `coordinator_epoch`, and `expected_state_hash`.
- Capabilities are passed separately from requests. Credentials and tokens never enter
  canonical request, result, state, batch, checkpoint, or fork identity.
- Duplicate idempotency keys return the exact prior result only when request and
  capability hashes match. Reusing a key for different input is rejected.
- A stale state hash or epoch is rejected; state is never merged.
- V19 remains the only world-transition implementation. V21.3 may make a pre-step
  SQLite backup so projection/record-limit failures can restore the prior store before
  the coordinator publishes a next state.
- State and output history become visible before subscriber callbacks run. Subscriber
  failure cannot roll back or change an accepted simulation transition.
- Control-command results are immutable synchronous results retained in the command
  audit ledger. An accepted `step` additionally emits `command.result` inside its V19
  round batch when that kind is allowed by the scenario run policy. Non-round control
  commands do not fabricate a V19 `round_result_hash` or a simulation batch.
- No arbitrary state-set command exists. V21.3 control commands are `start`, `pause`,
  `resume`, `step`, `stop`, and `checkpoint`.
- Forks start only from an explicit checkpoint artifact. A fork restores the exact
  checkpoint SQLite snapshot into a distinct database path, receives a new run and
  stream ID, starts paused, resets stream sequence to one, and increments the
  coordinator epoch.
- Checkpoint and database paths are launch configuration. They never enter semantic
  hashes or public errors.

## Run lifecycle

The finite state machine is:

```text
created --start--> running --pause--> paused --resume--> running
   |                  |                 |
   +------stop--------+------stop-------+----stop----> stopped
                      |
                      +--step until maximum_rounds--> completed

paused --step--> paused
```

`step` is legal in `running` and `paused`. A paused step advances exactly one round
and remains paused. A running step remains running unless it reaches
`ScenarioRunPolicy.maximum_rounds`, in which case it becomes `completed`. `stopped`
and `completed` are terminal. `checkpoint` is legal in `created`, `running`, `paused`,
and `completed`, but not `stopped`.

## Command boundary

`ScenarioCommandRequest` is a frozen, content-addressed value. It contains no arbitrary
parameter mapping. Only `checkpoint` may carry `requested_checkpoint_id`; all other
kinds require it to be null. `ScenarioCommandCapability` names one authority, one run,
an exact sorted command-kind allowlist, whether it may fork, and whether it is a host
capability allowed to read every command audit result.

`ScenarioCommandResult` records accepted/rejected status, a stable reason code, exact
prior/next run statuses and state hashes, the resulting round, and optional output or
checkpoint hash. Rejections are data, not exceptions. Malformed contracts, an
idempotency-key collision, local I/O failure, or an unexpected V19 failure remain
exceptions and do not enter the accepted audit ledger.

The reason vocabulary is:

- `accepted`
- `unauthorized`
- `run_mismatch`
- `scenario_mismatch`
- `epoch_mismatch`
- `stale_state`
- `invalid_status`
- `maximum_rounds`

## Typed query boundary

The coordinator exposes no generic dictionary query and never returns a larger object
for a caller to filter later.

- `run_view()` returns public run identity, lifecycle status, round, current state
  hash, coordinator epoch, next sequence, and retained output/checkpoint hashes.
- `public_state_view()` returns physical agent places, passage states, object
  placements, and aggregate emergence metrics.
- `agent_state_view(agent_id, capability)` returns one Agent's exact mind and only
  social relationships/claims observed by that Agent. It requires the matching Agent
  capability or `internal` capability.
- `network_state(capability)` returns the exact V19 network snapshot only to
  `objective`, `analyst`, or `internal` capability.
- `output_view(batch_hash, capability)` applies the existing V21.2 audience filter
  before returning a retained batch view.
- `command_result(command_id, capability)` returns an audit result only to the same
  authority or a host capability with `can_read_all_audit=True`.

## State storage and execution seams

The local coordinator uses replaceable semantic interfaces:

```python
class ScenarioStateStore(Protocol):
    def initialize(self, run_id: str, state: SituatedNetworkRuntimeState) -> None: ...
    def load(self, run_id: str) -> SituatedNetworkRuntimeState: ...
    def compare_and_swap(
        self,
        run_id: str,
        expected_state_hash: str,
        next_state: SituatedNetworkRuntimeState,
    ) -> None: ...

class OutputPublisher(Protocol):
    def publish(self, batch: SimulationOutputBatch) -> SimulationDeliveryReport: ...
```

`InMemoryScenarioStateStore` is the V21.3 implementation. Its compare-and-swap rule
preserves the same expected-hash contract needed by a future remote/durable adapter.
Agent execution and discovery remain inside the existing local V19 model for V21.3;
the future `AgentExecutor` and `AgentDirectory` wire seams remain reserved.

## Synchronous output bus

`SimulationOutputBus` stores subscriptions containing an output-kind allowlist and one
existing `SimulationAudienceCapability`. Publication snapshots the subscription table,
filters before callback, and invokes callbacks synchronously in subscription-ID order.

- A callback receives only `SimulationOutputView`.
- A callback must return `None`; any other value is a redacted delivery failure.
- Callback exceptions become stable `callback_error` failures without exception text.
- One failing callback does not stop later subscribers.
- Reentrant publication is rejected.
- Subscribe/unsubscribe during a callback affects only later publications.
- Delivery reports and subscription state are operational and do not enter world or
  output batch hashes.

## Atomic step

For one accepted `step`:

1. validate request, capability, epoch, state hash, status, and maximum round;
2. create a temporary SQLite backup of the exact current memory store;
3. call `simulate_situated_network_round` once;
4. project V21.2 records using the coordinator's next global sequence;
5. when allowlisted, prepend one public `command.result` record and canonically
   resequence the complete batch;
6. enforce `maximum_output_records`;
7. when the resulting round is divisible by `checkpoint_interval`, atomically create
   the checkpoint snapshot and mark the batch as a checkpoint batch;
8. compare-and-swap the exact next runtime state;
9. append command, output, and checkpoint history and advance the global sequence;
10. publish the immutable batch synchronously.

If steps 3–8 fail, the backup restores the prior SQLite store and no coordinator state,
sequence, command result, output history, or checkpoint remains visible. A checkpoint
created before a later compare-and-swap failure is discarded. Publication failures in
step 10 are reported but do not roll back steps 7–9.

## Checkpoint and fork

`ScenarioCheckpoint` binds checkpoint ID, source run, scenario, epoch, exact runtime
state hash/object, source sequence, parent checkpoint hash, and memory-store hash.
Its content hash excludes its file location.

`LocalScenarioCheckpointStore` writes an exact SQLite snapshot through a
same-directory stage, flush/fsync, and `os.replace`. It keeps typed artifact metadata
for the active process and restores by checkpoint hash. Duplicate checkpoint IDs are
idempotent only for the same artifact; conflicting reuse is rejected.
`discard(checkpoint_hash)` removes only an exact store-owned artifact and is used to
roll back an automatic checkpoint if the following state compare-and-swap fails.

`ScenarioForkRequest` binds source run/scenario/epoch/checkpoint plus child run and
stream identities. `ScenarioForkResult` binds the child initial state, epoch, and
parent checkpoint. Duplicate fork idempotency keys return the same child coordinator
and result only for the exact same request/capability.

The child receives the exact checkpoint `SituatedNetworkRuntimeState`; no facts,
memories, relationships, or events are re-created. The copied SQLite store must hash
to the checkpoint's `memory_store_hash` before the child is exposed.

## Limits and recovery

- `maximum_rounds`, `checkpoint_interval`, and `maximum_output_records` are enforced
  from the compiled run policy.
- Command and output history are bounded by those scenario limits for V21.3.
- No unbounded queue or background worker exists.
- A process crash after V19 publishes SQLite but before the coordinator records the
  next state remains the known V19 crash boundary. Durable coordinator metadata and
  process-restart recovery are later state-store work.
- A fork/checkpoint write failure leaves existing snapshot bytes unchanged and returns
  no artifact.

## Public API direction

```python
coordinator = ScenarioCoordinator.create(
    database_path,
    scenario,
    run_id="law-firm-run",
    stream_id="law-firm-stream",
    state_store=InMemoryScenarioStateStore(),
    publisher=SimulationOutputBus(),
    checkpoint_store=LocalScenarioCheckpointStore(checkpoint_root),
)

start = coordinator.submit_command(start_request, operator_capability)
step = coordinator.submit_command(step_request, operator_capability)
public = coordinator.output_view(
    step.output_batch_hash,
    SimulationAudienceCapability(SimulationOutputAudience.PUBLIC),
)
checkpoint = coordinator.submit_command(checkpoint_request, operator_capability)
child, fork_result = coordinator.fork(
    fork_request,
    operator_capability,
    child_database_path,
)
```

## Non-goals

- No HTTP/2, WSS, JSON-RPC server, browser client, X6, PixiJS, or Monaco bundle.
- No background run loop, scheduler thread, worker lease, remote Agent, discovery,
  message broker, consensus, distributed lock, or multi-writer state.
- No arbitrary world mutation, direct state setter, free-form command parameters, or
  developer console mutation on the source run.
- No story-scene coordinator, intervention validator, LLM/provider call, retrieval,
  Blender live connection, or asset fetch.
- No durable restart of command/output metadata and no private journal convenience.
