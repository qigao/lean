# Pure-Web Simulation Platform V21 Design

## Goal

Turn the existing data-authored situated simulation into a browser-native,
queryable, controllable platform while preserving one deterministic world authority.
The first delivery is the transport-neutral V21.2 output boundary. Later deliveries
add the JSON-RPC server and pure-Web editor without changing simulation identity.

## Binding product choices

- The editor is a pure Web application, not a VS Code extension.
- React Flow is not used.
- Graph editing uses AntV X6 under its MIT license.
- Physical-map rendering/editing uses PixiJS under its MIT license.
- Raw JSON inspection uses Monaco Editor under its MIT license.
- External APIs use JSON-RPC 2.0 only; Protobuf and gRPC are not used.
- Request/response JSON-RPC uses HTTPS over HTTP/2 at `POST /rpc`.
- Live subscriptions use JSON-RPC messages over WSS at `/v1/stream` with
  subprotocol `nd-jsonrpc-v1`.
- WSS may be bootstrapped separately from HTTP/2; RFC 8441 is optional deployment
  support, not a client requirement.
- Distributed execution is not delivered initially, but all runtime execution,
  storage, publication, and routing boundaries remain replaceable interfaces.

## Authority model

One `ScenarioCoordinator` is the sole writer for one run. Editors, Web clients,
Blender, providers, and future remote Agent workers submit requests or proposals;
none edits canonical state directly. Every accepted transition binds the exact
scenario hash, prior state hash, round, and coordinator epoch.

Agent independence is logical and privacy-scoped. Each Agent receives only its
sanitized percepts, private memory and belief state, local social view, grants, and
allowed actions. Initial delivery executes those turns locally in one Python process.
A future `RemoteAgentExecutor` may move computation to another machine while returning
the same typed proposal artifact to the same central resolver.

## Browser editor

The browser edits a server-owned `ScenarioDraft`, never an unversioned collection of
canvas objects. X6 renders social, institution, story-dependency, knowledge, and
entitlement graphs. PixiJS renders the Tiled-compatible physical map, agents, objects,
passages, and optional perception overlays. Monaco exposes the exact JSON documents.

Each user edit is a typed draft operation with `project_id`, `expected_revision`, and
stable entity IDs. The authoring service applies the operation, serializes canonical
scenario documents, refreshes manifest hashes, invokes the existing V21.1 compiler,
and returns the new revision plus sanitized diagnostics. Canvas layout metadata never
replaces domain IDs or enters runtime state unless the physical schema explicitly
declares it as authored geometry.

## JSON-RPC surface

The planned method namespaces are:

- `project.*`: create, open, read, patch, undo, redo, save;
- `scenario.*`: validate, compile, import, export;
- `run.*`: create, start, pause, resume, step, stop, fork, checkpoint;
- `state.*`, `agent.*`, `network.*`, `story.*`, `event.*`, `metrics.*`: scoped queries;
- `command.*`: validated state-changing requests;
- `stream.*`: subscribe, acknowledge, resume, unsubscribe;
- `worker.*` and `agent.turn.*`: reserved for future distributed executors.

State-changing methods always carry a JSON-RPC `id`, an application
`idempotency_key`, and `expected_state_hash`. JSON-RPC notifications are used only for
outputs whose delivery does not decide world authority. A JSON-RPC batch is not an
atomic world transaction; atomic multi-command submission requires the explicit
future `command.submitTransaction` method.

## V21.2 output boundary

V21.2 is transport-neutral. It produces immutable typed records and one atomic batch
from an accepted V19 round. Records bind stream ID, compiled scenario hash, sequence,
round, state hash, kind, audience, optional owner Agent, source hashes, typed payload,
and payload hash. Records are ordered canonically and the complete batch binds the
prior and next state hashes plus the accepted V19 round hash.

Audience filtering occurs before serialization or delivery:

- `public`: safe map/world deltas, safe event metadata, aggregate metrics;
- `objective`: analyst truth metadata that still excludes arbitrary private memory;
- `agent`: exactly one owner Agent's percept, decision, recall, and social view;
- `analyst`: privileged bounded diagnostics;
- `internal`: never emitted by public conveniences.

The public journal is canonical JSONL. Its first line is a header binding schema,
stream, scenario hash, and parent journal hash. Each later line is one public-only
batch. A write stages and fsyncs the complete replacement before `os.replace`, so a
crash does not expose half a batch. Replay validates schemas, sequence continuity,
state chaining, payload hashes, record hashes, and batch hashes without running an
Agent or provider.

## Distributed-ready interfaces

Initial implementations are local, but the following semantic interfaces are fixed:

```python
class AgentExecutor:
    def execute_turn(self, request: AgentTurnRequest) -> AgentProposal: ...

class ScenarioStateStore:
    def load(self, run_id: str, state_hash: str): ...
    def commit(self, run_id: str, expected_hash: str, next_state): ...

class OutputPublisher:
    def publish(self, batch): ...

class AgentDirectory:
    def resolve(self, run_id: str, agent_id: str): ...
```

Future wire messages reserve `protocol_version`, `run_id`, `scenario_hash`,
`round_index`, `prior_state_hash`, `agent_id`, `request_id`, `idempotency_key`,
`deadline`, `coordinator_epoch`, `capability`, `proposal_hash`, and `worker_id`.
Transport addresses, credentials, worker identities, and deployment topology never
enter compiled scenario identity.

## Failure and recovery rules

- A stale `expected_state_hash` is rejected rather than merged.
- Duplicate command or turn IDs return the already accepted result.
- A missing/late future worker resolves through a declared deterministic fallback,
  normally `WAIT`; late proposals are rejected.
- Reconnect uses `(stream_id, last_sequence, last_batch_hash)`.
- If retained batches cannot fill a gap, the server supplies an audience-scoped full
  snapshot before later deltas.
- Arbitrary developer mutation is allowed only on a forked run and remains an audited
  command.

## Delivery sequence

This sequence is the historical V21 plan. The delivered browser authoring, JSON-RPC
gateway, run console, packaging, and recovery contract is governed by the
[World Studio V22 design](2026-09-02-world-studio-v22-design.md), which is authoritative
for the current delivery.

1. V21.2 typed output contracts, deterministic projector, public journal and replay.
2. V21.3 single-process coordinator, command/query boundary, checkpoint and fork.
3. Pure-Web editor with X6, PixiJS, Monaco, and JSON-RPC client.
4. JSON-RPC gateway over HTTP/2 and WSS, including subscription recovery.
5. Blender live Web client and validated interventions.
6. Optional remote Agent executor and distributed state/publication adapters.

## Non-goals for V21.2

- No HTTP server, WSS server, browser bundle, X6, PixiJS, or Monaco dependency.
- No remote Agent, registry, discovery, lease service, distributed database, message
  broker, consensus, multi-writer state, or region sharding.
- No direct state-set RPC and no transport payload in canonical world identity.
- No provider, Blender process, network access, or asset retrieval during projection.

The V22 delivery also explicitly excludes React/React Flow, Protobuf/gRPC,
provider/LLM integration, remote worker execution, live Blender mutation, and any
arbitrary state-set RPC. Credentials, filesystem paths, callbacks, connection IDs,
and deployment topology remain outside semantic scenario and run identities.
