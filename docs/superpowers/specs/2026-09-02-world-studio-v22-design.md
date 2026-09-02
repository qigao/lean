# World Studio V22 Design

## Goal

Turn the merged V21.3 local scenario runtime into a browser-native authoring and
operations product without weakening deterministic simulation authority. V22 lets an
authorized user import or create a scenario project, edit its tree-structured
documents through graph/map/form/raw-JSON views, validate and compile a revision,
create and control local runs, inspect capability-scoped state and output, and resume
live subscriptions after reconnect.

V22 is the input, control, and observation layer. It does not add arbitrary state
mutation, free-form LLM actions, live Blender control, or remote Agent execution.

## Relationship to the V21 design

This specification supersedes only the delivery sequence in
`2026-09-01-web-simulation-platform-v21-design.md`. Its binding product choices remain:

- pure Web application, never a VS Code extension;
- no React Flow;
- AntV X6 for graph authoring;
- PixiJS for the physical map;
- Monaco Editor for exact JSON;
- JSON-RPC 2.0 over HTTPS/HTTP2 for request/response;
- JSON-RPC messages over WSS for subscriptions;
- no Protobuf or gRPC;
- distributed execution deferred, with replaceable execution/storage/publication
  boundaries preserved.

The current primary sources describe X6 as a data-driven HTML/SVG graph editing
engine, PixiJS as a WebGPU/WebGL 2D renderer, Starlette as an ASGI WebSocket
application layer, and Hypercorn as an ASGI server supporting HTTP/2 and WebSockets:

- https://x6.antv.antgroup.com/en/tutorial/about
- https://pixijs.com/
- https://www.starlette.io/websockets/
- https://hypercorn.readthedocs.io/en/latest/index.html

X6, PixiJS, and Monaco are MIT-licensed. The browser uses X6 directly without a
React adapter.

## Delivery boundary

V22 is delivered as five independently testable slices:

1. scenario project workspace and deterministic diagnostics;
2. transport-neutral application service and JSON-RPC dispatcher;
3. HTTP/2/WSS ASGI gateway with bounded subscription recovery;
4. pure-Web authoring editor;
5. run console, end-to-end integration, packaging, and hardening.

Each slice is usable without the later one. The Python ABM core remains importable
without server or browser dependencies.

## Authority model

There are two authorities and they never overlap:

- `ScenarioProjectWorkspace` owns editable draft revisions. Drafts are not runtime
  state and cannot affect an existing run.
- `ScenarioCoordinator` remains the sole writer for one compiled run. The Web client,
  JSON-RPC gateway, and future remote workers only submit typed requests.

A run is created from one exact compiled scenario hash. Editing or saving its source
project never changes that run. The user must compile a new revision and create a new
run or explicit fork.

No RPC sets arbitrary canonical state. V22 exposes only the existing V21.3 lifecycle,
step, checkpoint, and fork operations.

## Project workspace

### Storage

The server-owned workspace is a SQLite database, separate from each run's SQLite
percept-memory store. One transaction stores:

- project identity and metadata;
- canonical UTF-8 JSON documents keyed by document role and logical ID;
- raw document hash and semantic snapshot hash;
- monotonically increasing revision;
- accepted operation journal;
- undo/redo cursor;
- sanitized diagnostic report;
- optional compiled scenario identity;
- presentation-only layout metadata.

The workspace never executes Python or follows document URIs. Import delegates safe
filesystem reading to the V21.1 package loader. Export writes a new content-addressed
package directory and never overwrites an existing package. Credentials, workspace
paths, SQLite row IDs, and layout coordinates do not enter compiled scenario identity.

### Draft snapshot

`ScenarioDraftSnapshot` contains:

- `project_id`;
- positive `revision`;
- `snapshot_hash`;
- canonical document descriptors and exact JSON values;
- presentation layout hash;
- latest diagnostic report hash;
- optional compiled scenario hash.

Documents remain trees. Graph projections are derived views:

- physical: places, passages, perception edges, agents, objects;
- social: institutions, memberships, directed relationships, norms;
- story: acts, scenes, scene dependencies, scheduled actions;
- resources: knowledge/assets, entitlements, grants.

Graph node IDs are authored domain IDs. Canvas cell IDs and positions are
presentation metadata and never replace domain identity.

### Draft operations

Every state-changing draft request contains:

- `operation_id`;
- `idempotency_key`;
- `project_id`;
- `expected_revision`;
- `expected_snapshot_hash`;
- `document_role` and optional logical document ID;
- one closed operation kind;
- a bounded typed payload.

The V22 operation kinds are:

- `replace_document` — replace one complete JSON document after size/type validation;
- `set_value` — set one value at an RFC 6901 pointer;
- `insert_value` — insert into one object key or array index;
- `remove_value` — remove one object key or array element;
- `move_value` — move one array element within the same document;
- `set_layout` — update presentation metadata only.

Operations never address filesystem paths. Duplicate exact request/capability pairs
return the same result; conflicting reuse rejects. A stale revision/hash rejects
without merge. The browser reloads and presents a diff; V22 performs no silent
multi-user merge.

Undo and redo are server operations over the accepted journal. They create new
revisions rather than rewinding the revision number.

### Validation and compilation

After every accepted domain operation, the service reconstructs an in-memory
`ScenarioPackageSource`, refreshes manifest content hashes, invokes the existing
V21.1 loader/compiler validation path, and returns a deterministic
`ScenarioDiagnosticReport`.

Each diagnostic contains only:

- severity (`error` or `warning`);
- stable code;
- document role and logical ID;
- RFC 6901 pointer;
- optional related stable IDs;
- deterministic message key.

It contains no source value, host path, traceback, exception text, credential, or
provider response. Compilation succeeds only with zero errors and produces the exact
existing `CompiledSituatedScenario`; V22 introduces no alternative runtime model.

## Application service

`WorldStudioService` is transport-neutral. It receives a typed request plus a
separate `StudioCapability` derived by the host authentication layer. Authentication
tokens and transport connection IDs never enter request/result hashes.

The service owns replaceable registries:

```python
class ScenarioProjectStore(Protocol): ...
class ScenarioRunRegistry(Protocol): ...
class ScenarioCoordinatorFactory(Protocol): ...
class StudioOutputRouter(Protocol): ...
```

The initial implementations are local SQLite/in-memory registries. Interfaces use
stable project/run IDs and hashes so later process or machine boundaries do not
change application contracts.

## JSON-RPC 2.0 surface

### Project and compilation

- `project.create`
- `project.import`
- `project.snapshot`
- `project.apply`
- `project.undo`
- `project.redo`
- `project.export`
- `scenario.validate`
- `scenario.compile`

### Run authority

- `run.create`
- `run.command` — one existing `ScenarioCommandKind` request;
- `run.fork`
- `run.view`
- `run.list`

### Scoped queries

- `state.public`
- `state.agent`
- `state.network`
- `output.get`
- `command.get`

### Subscription control

- `stream.subscribe`
- `stream.acknowledge`
- `stream.resume`
- `stream.unsubscribe`

`worker.*`, `agent.turn.*`, arbitrary `state.set`, and transaction methods are
reserved and reject as method-not-found in V22.

### Envelope and errors

The dispatcher accepts JSON-RPC 2.0 single requests. A JSON-RPC batch is rejected in
V22 because it is not an atomic world transaction. State-changing notifications are
rejected; notifications are used only for server-to-client output delivery.

Standard JSON-RPC errors retain their standard codes. Application errors use stable
codes:

- `-32010` unauthorized;
- `-32011` stale revision/state;
- `-32012` validation/compilation failure;
- `-32013` invalid run lifecycle;
- `-32014` unknown project/run/artifact;
- `-32015` identity/idempotency conflict;
- `-32016` subscription history gap;
- `-32017` capacity limit.

Error data may contain stable diagnostic IDs, hashes, and JSON pointers only.

## HTTP/2 and WSS gateway

The optional ASGI integration uses Starlette for the application and WebSocket API and
Hypercorn as the deployment server. It is isolated under
`narrative_dynamics.integrations`; importing `narrative_dynamics.abm` or
`narrative_dynamics.studio` does not import either dependency.

Routes:

- `POST /rpc` — JSON-RPC request/response; HTTPS/HTTP2 in deployment;
- `GET /health` — bounded process health with no project/run data;
- `WSS /v1/stream` — JSON-RPC subscription messages using subprotocol
  `nd-jsonrpc-v1`;
- `/studio/*` — immutable hashed frontend assets and an index fallback.

The gateway enforces request/body/frame limits, message timeouts, origin policy,
authentication context, and maximum subscriptions per connection. TLS certificates,
tokens, client addresses, and deployment topology remain operational configuration.

## Subscription recovery

One subscription binds:

- subscription ID;
- run and stream IDs;
- audience capability and optional owner Agent;
- allowed output kinds;
- last acknowledged sequence and batch hash;
- bounded lease expiry.

The server filters through V21.2 capability projection before JSON serialization.
Reconnect supplies `(stream_id, last_sequence, last_batch_hash)`. If retained batches
cover the gap, the server replays them in order. Otherwise it returns `-32016` with a
capability-scoped snapshot token; the client fetches the matching state view and then
resumes from the current sequence. V22 does not promise exactly-once network delivery;
content hashes and acknowledgements provide deterministic deduplication.

## Browser editor

The frontend is TypeScript built with Vite and browser-native components. It does not
use React or React Flow.

The desktop layout contains:

- project/document tree;
- central mode canvas;
- property inspector and diagnostics;
- raw JSON tab;
- run toolbar and connection state;
- bottom event/output timeline.

Authoring modes:

- X6 physical topology graph;
- PixiJS Tiled-compatible physical geometry editor;
- X6 social/institution graph;
- X6 story dependency graph;
- X6 resource entitlement graph;
- Monaco exact JSON editor.

All edits produce a `project.apply` request. The client never treats X6/Pixi/Monaco
state as authoritative. It updates its local view only after the server returns the
next revision, or restores the prior view and shows a stale/conflict diagnostic.

Keyboard operation, visible focus, semantic form labels, reduced motion, and a
non-canvas document-tree/form fallback are release requirements. Monaco is not the
only way to edit any required field.

## Run console

The V22 run console can:

- create a run from one compiled revision;
- start, pause, resume, step, stop, checkpoint, and fork;
- show run status, round, state hash, epoch, and sequence;
- inspect public world state;
- inspect one authorized Agent view;
- inspect privileged network metrics when allowed;
- display audience-filtered output records on a timeline;
- reconnect and resume a subscription;
- open parent and child runs side by side.

It cannot modify a running world, create undeclared actions, invoke an LLM, or drive
Blender live. Typed story interventions are a later milestone.

## Concurrency and recovery

- Project mutations use revision/hash compare-and-swap inside one SQLite transaction.
- Run mutations delegate to the V21.3 coordinator operation gate.
- The gateway never holds a connection/subscription lock while calling project or run
  authority, preventing cross-layer lock inversion.
- Disconnect does not cancel an accepted command.
- Service restart reloads projects from the workspace SQLite store.
- V22 run-registry persistence is limited to metadata needed to report that an old
  process-local run is unavailable; full coordinator crash recovery is a separate
  milestone.

## Security and privacy

- Filesystem access is limited to configured import/export roots.
- Import reuses V21.1 traversal, symlink, duplicate-file, size, and hash defenses.
- Browser-supplied paths, Python names, module references, callbacks, and code are
  rejected.
- Audience filtering occurs before JSON encoding and before subscription retention.
- Private Agent data requires an owner-matching capability.
- Raw callback/provider/SQLite exceptions and local paths never enter JSON-RPC errors.
- Static assets use content hashes and a restrictive content-security policy.

## Distributed-ready boundaries

V22 runs locally, but every request includes protocol version, project/run ID,
scenario hash or revision, request ID, idempotency key, coordinator epoch, and expected
state hash where applicable. `ScenarioRunRegistry`, `ScenarioCoordinatorFactory`, and
`StudioOutputRouter` are replaceable.

V22 deliberately does not add worker discovery, leases, distributed clocks, remote
Agent proposals, consensus, multi-writer runs, or a message broker.

## Acceptance criteria

V22 is complete when an authorized browser can:

1. import the law-firm package into a persistent project;
2. edit a place, passage, relationship, scene dependency, and raw JSON field through
   the appropriate views;
3. receive deterministic pointer-addressed diagnostics and compile a valid revision;
4. create and step a V21.3 run through JSON-RPC;
5. inspect public and owner-Agent state without privacy leakage;
6. receive output over WSS, disconnect, and resume or recover by scoped snapshot;
7. checkpoint and fork, then inspect parent/child runs side by side;
8. restart the service and reopen the project revision;
9. pass Python, browser unit, ASGI integration, accessibility, and end-to-end tests;
10. demonstrate no gRPC/Protobuf, React Flow, remote Agent, live Blender, provider, or
    arbitrary state-set dependency in the delivered code.
