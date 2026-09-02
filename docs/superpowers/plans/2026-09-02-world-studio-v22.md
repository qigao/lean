# World Studio V22 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a persistent scenario-authoring workspace, transport-neutral JSON-RPC application service, HTTP/2/WSS gateway, and pure-Web editor/run console over the merged V21.3 coordinator.

**Architecture:** Keep editable `ScenarioProjectWorkspace` authority separate from immutable compiled runs. The Python studio core stores revisioned JSON documents in SQLite and delegates compilation and runtime transitions to existing V21.1/V21.3 APIs; optional ASGI and browser packages consume the same typed service contracts. The Web client renders derived graph/map views but every mutation is a revision-checked server operation.

**Tech Stack:** Python standard library and SQLite core; JSON-RPC 2.0; optional Starlette + Hypercorn ASGI adapter; TypeScript + Vite; AntV X6, PixiJS, Monaco Editor; Vitest and Playwright.

**Spec:** `docs/superpowers/specs/2026-09-02-world-studio-v22-design.md`

## Global Constraints

- The editor is pure Web and is not a VS Code extension.
- Do not use React Flow, Protobuf, or gRPC.
- X6, PixiJS, and Monaco render derived views; server project revisions remain authoritative.
- External request/response is JSON-RPC 2.0 at `POST /rpc`; live delivery is JSON-RPC over WSS `/v1/stream` with subprotocol `nd-jsonrpc-v1`.
- One V21.3 `ScenarioCoordinator` remains the sole writer for one run.
- Add no direct state-set RPC, free-form LLM action, live Blender mutation, remote Agent execution, worker discovery, broker, or distributed consensus.
- Credentials, paths, connection IDs, layout metadata, and deployment topology never enter compiled scenario or run identity.
- Capability filtering occurs before JSON serialization and subscription retention.
- The Python ABM/studio core imports no Starlette, Hypercorn, X6, PixiJS, Monaco, or Node package.
- Every behavior change follows RED → GREEN TDD with real SQLite/scenario/coordinator objects; mocks are limited to transport, clock, and explicit I/O failure seams.

---

### Task 1: Persistent scenario project workspace and diagnostics

**Files:**
- Create: `narrative_dynamics/studio/__init__.py`
- Create: `narrative_dynamics/studio/contracts.py`
- Create: `narrative_dynamics/studio/project_store.py`
- Create: `narrative_dynamics/studio/workspace.py`
- Create: `narrative_dynamics/studio/diagnostics.py`
- Create: `tests/test_world_studio_contracts.py`
- Create: `tests/test_world_studio_workspace.py`
- Create: `tests/test_world_studio_diagnostics.py`

**Interfaces:**
- Consumes: V21.1 `ScenarioPackageSource`, `ScenarioSourceDocument`, `ScenarioCompilationError`, `load_situated_scenario_package`, and `compile_situated_scenario_package`.
- Produces: frozen draft contracts, `ScenarioProjectStore`, `SQLiteScenarioProjectStore`, and `ScenarioProjectWorkspace` used by Tasks 2–5.

- [ ] **Step 1: Add failing contract tests**

Test exact enums and schemas:

```python
assert tuple(item.value for item in DraftOperationKind) == (
    "replace_document", "set_value", "insert_value",
    "remove_value", "move_value", "set_layout",
)
assert SCENARIO_DRAFT_SNAPSHOT_SCHEMA == "narrative-dynamics.scenario-draft-snapshot/v1"
assert SCENARIO_DRAFT_OPERATION_SCHEMA == "narrative-dynamics.scenario-draft-operation/v1"
assert SCENARIO_DIAGNOSTIC_REPORT_SCHEMA == "narrative-dynamics.scenario-diagnostic-report/v1"
```

Define constructor tests for frozen/canonical/content-addressed:

```python
operation = ScenarioDraftOperation(
    "op-1", "key-1", "project-1", 1, SNAPSHOT_HASH,
    "physical.world", None, DraftOperationKind.SET_VALUE,
    pointer="/places/0/name", value="Records archive",
)
assert operation.to_dict()["kind"] == "set_value"
assert operation.content_hash.startswith("sha256:")
```

Reject negative/zero revisions, malformed hashes/pointers, non-JSON values, unknown
operation fields, path-like document identities, payloads above the declared byte
limit, and layout operations that contain domain JSON.

- [ ] **Step 2: Run contract RED**

```text
python -m pytest tests/test_world_studio_contracts.py -q
```

Expected: collection fails because `narrative_dynamics.studio` does not exist.

- [ ] **Step 3: Implement frozen workspace contracts**

Implement:

```python
class DraftOperationKind(str, Enum): ...
class ScenarioDiagnosticSeverity(str, Enum): ...

@dataclass(frozen=True)
class ScenarioDraftDocument:
    role: str
    logical_id: str | None
    value: JsonValue
    content_hash: str

@dataclass(frozen=True)
class ScenarioDraftSnapshot:
    project_id: str
    revision: int
    documents: tuple[ScenarioDraftDocument, ...]
    layout: JsonValue
    diagnostic_report_hash: str
    compiled_scenario_hash: str | None

@dataclass(frozen=True)
class ScenarioDraftOperation: ...

@dataclass(frozen=True)
class ScenarioDraftOperationResult:
    operation_hash: str
    prior_revision: int
    next_snapshot: ScenarioDraftSnapshot
    diagnostic_report: ScenarioDiagnosticReport

@dataclass(frozen=True)
class ScenarioDiagnostic:
    severity: ScenarioDiagnosticSeverity
    code: str
    document_role: str
    logical_id: str | None
    pointer: str
    related_ids: tuple[str, ...] = ()

@dataclass(frozen=True)
class ScenarioDiagnosticReport: ...
```

Canonical JSON accepts only null/bool/int/finite float/string/list/object with string
keys. Sort document descriptors by `(role, logical_id or "")`, diagnostics by stable
location/code, and object keys during hashing. Never accept a path, callback, exception,
token, or Python object in a semantic contract.

- [ ] **Step 4: Add failing SQLite workspace tests**

Using a temporary real SQLite store, prove:

```python
workspace = ScenarioProjectWorkspace.open(database)
created = workspace.create_project("law-firm")
imported = workspace.import_package(
    "law-firm", Path("examples/law_firm_scenario"),
    expected_revision=created.revision,
    expected_snapshot_hash=created.content_hash,
)
reopened = ScenarioProjectWorkspace.open(database).snapshot("law-firm")
assert reopened == imported
```

Add mutation-focused tests for exact operation idempotency, both collision directions,
stale revision/hash rejection, `set/insert/remove/move/replace` JSON semantics, layout
identity exclusion, undo/redo creating increasing revisions, transaction rollback on
compiler/storage error, database reopen, project isolation, and bounded journal size.

- [ ] **Step 5: Run workspace RED**

```text
python -m pytest tests/test_world_studio_workspace.py -q
```

Expected: imports fail because store/workspace modules do not exist.

- [ ] **Step 6: Implement SQLite project storage**

Define a runtime-checkable protocol and implementation:

```python
class ScenarioProjectStore(Protocol):
    def create(self, project_id: str, snapshot: ScenarioDraftSnapshot) -> None: ...
    def load(self, project_id: str) -> ScenarioDraftSnapshot: ...
    def apply(self, operation: ScenarioDraftOperation,
              next_snapshot: ScenarioDraftSnapshot,
              report: ScenarioDiagnosticReport) -> ScenarioDraftOperationResult: ...
    def operation_result(self, project_id: str, idempotency_key: str): ...
    def undo(self, project_id: str, expected_revision: int,
             expected_hash: str) -> ScenarioDraftSnapshot: ...
    def redo(self, project_id: str, expected_revision: int,
             expected_hash: str) -> ScenarioDraftSnapshot: ...
```

Use one SQLite transaction for revision CAS, document rows, journal, cursor, diagnostics,
and idempotency. Store canonical JSON text plus hashes, not pickle. Enable foreign keys,
busy timeout, and WAL where supported. Normalize all SQLite failures to stable path-free
errors.

- [ ] **Step 7: Add failing diagnostic/compile/export tests**

Import the real law-firm package, deliberately set one dangling passage target, and
assert a deterministic report points to the exact role/pointer without including the
bad source value or package path. Undo, compile, and prove the compiled hash equals
direct V21.1 compilation.

Export to a missing target and prove direct V21.1 reload/compile equality. Reject an
existing target, symlink parent, root escape, and injected publication failure while
preserving the prior content-addressed export.

- [ ] **Step 8: Implement workspace orchestration and diagnostics**

Expose:

```python
ScenarioProjectWorkspace.open(database_path, *, import_roots=(), export_root=None)
create_project(project_id) -> ScenarioDraftSnapshot
import_package(project_id, source_root, *, expected_revision,
               expected_snapshot_hash) -> ScenarioDraftSnapshot
snapshot(project_id) -> ScenarioDraftSnapshot
apply(operation) -> ScenarioDraftOperationResult
undo(project_id, expected_revision, expected_snapshot_hash) -> ScenarioDraftSnapshot
redo(project_id, expected_revision, expected_snapshot_hash) -> ScenarioDraftSnapshot
validate(project_id) -> ScenarioDiagnosticReport
compile(project_id) -> CompiledSituatedScenario
export(project_id, target_name) -> ScenarioProjectExport
```

Build `ScenarioPackageSource` from exact draft documents and translate only known
package/compiler exceptions to stable diagnostics. Export a new content-addressed
directory beneath the configured root through a same-parent stage and no-clobber rename;
fsync files before publication and never overwrite.

- [ ] **Step 9: Run Task 1 GREEN and regressions**

```text
python -m pytest tests/test_world_studio_contracts.py tests/test_world_studio_workspace.py tests/test_world_studio_diagnostics.py tests/test_network_abm_scenario_package.py tests/test_network_abm_scenario_compiler.py -q
python -m compileall -q narrative_dynamics/studio tests/test_world_studio_*.py
git diff --check
```

- [ ] **Step 10: Commit Task 1**

```text
git add narrative_dynamics/studio tests/test_world_studio_contracts.py tests/test_world_studio_workspace.py tests/test_world_studio_diagnostics.py
git commit -m "feat(studio): add revisioned scenario workspace"
```

---

### Task 2: Transport-neutral service and JSON-RPC dispatcher

**Files:**
- Create: `narrative_dynamics/studio/capabilities.py`
- Create: `narrative_dynamics/studio/run_registry.py`
- Create: `narrative_dynamics/studio/service.py`
- Create: `narrative_dynamics/studio/jsonrpc.py`
- Modify: `narrative_dynamics/studio/__init__.py`
- Create: `tests/test_world_studio_service.py`
- Create: `tests/test_world_studio_jsonrpc.py`
- Modify: `tests/test_network_abm_public_api.py`

**Interfaces:**
- Consumes: Task 1 workspace, V21.3 `ScenarioCoordinator`, command/fork/query/output contracts.
- Produces: `StudioCapability`, `ScenarioRunRegistry`, `WorldStudioService`, and `JsonRpcDispatcher` used by Task 3 and the browser client contract.

- [ ] **Step 1: Add failing capability and service tests**

Define capabilities separate from payloads:

```python
capability = StudioCapability(
    "operator", project_ids=("law-firm",), run_ids=("run-1",),
    permissions=("project.read", "project.write", "scenario.compile",
                 "run.create", "run.command", "state.public"),
)
```

Test project create/import/snapshot/apply/undo/redo/export and scenario validate/compile
authorization. Create a real V21.3 coordinator through the service, submit start/step,
query public/Agent/network/output/audit views, checkpoint, and fork. Prove capabilities
are narrowed before the underlying query and unauthorized known/unknown IDs are
indistinguishable.

- [ ] **Step 2: Run service RED**

```text
python -m pytest tests/test_world_studio_service.py -q
```

- [ ] **Step 3: Implement run registry and service**

Define:

```python
class ScenarioRunRegistry(Protocol):
    def register(self, run_id: str, coordinator: ScenarioCoordinator) -> None: ...
    def resolve(self, run_id: str) -> ScenarioCoordinator: ...
    def list_for(self, capability: StudioCapability) -> tuple[ScenarioRunView, ...]: ...

class WorldStudioService:
    def invoke(self, method: str, params: JsonObject,
               capability: StudioCapability) -> JsonValue: ...
```

Use an explicit method table for the exact spec surface. Construct existing typed
commands/forks/capabilities from bounded validated JSON, invoke public V21.3 methods,
and serialize trusted typed results field-by-field. Never call caller-owned `to_dict`,
accept Python class names, or expose full objects for later filtering.

- [ ] **Step 4: Add failing JSON-RPC protocol tests**

Cover exact request/response behavior:

```python
response = dispatcher.dispatch(
    {"jsonrpc": "2.0", "id": "rpc-1", "method": "project.snapshot",
     "params": {"project_id": "law-firm"}},
    capability,
)
assert response["jsonrpc"] == "2.0"
assert response["id"] == "rpc-1"
assert response["result"]["project_id"] == "law-firm"
```

Test parse/invalid-request/method/params errors, stable application error mapping,
state-changing notification rejection, batch rejection, duplicate keys, depth/count/
byte limits, non-finite numbers, and raw exception/path/token redaction.

- [ ] **Step 5: Run dispatcher RED**

```text
python -m pytest tests/test_world_studio_jsonrpc.py -q
```

- [ ] **Step 6: Implement strict JSON-RPC dispatcher**

Expose:

```python
JsonRpcLimits(maximum_bytes=1_048_576, maximum_depth=64,
              maximum_members=20_000, maximum_string_bytes=262_144)
JsonRpcDispatcher(service, limits=JsonRpcLimits())
parse_and_dispatch(raw_utf8: bytes, capability: StudioCapability) -> bytes
dispatch(request: JsonObject, capability: StudioCapability) -> JsonObject | None
```

Reject arrays at the top level, duplicate JSON keys, bool IDs, unknown members, and
state-changing notifications. Encode with canonical separators/order. Suppress causes
when normalizing untrusted exceptions.

- [ ] **Step 7: Run Task 2 GREEN and regressions**

```text
python -m pytest tests/test_world_studio_service.py tests/test_world_studio_jsonrpc.py tests/test_network_abm_scenario_coordinator.py tests/test_network_abm_public_api.py -q
python -m compileall -q narrative_dynamics/studio
git diff --check
```

- [ ] **Step 8: Commit Task 2**

```text
git add narrative_dynamics/studio tests/test_world_studio_service.py tests/test_world_studio_jsonrpc.py tests/test_network_abm_public_api.py
git commit -m "feat(studio): expose typed JSON-RPC application service"
```

---

### Task 3: HTTP/2 and WSS ASGI gateway

**Files:**
- Create: `narrative_dynamics/integrations/world_studio_server.py`
- Create: `narrative_dynamics/studio/streaming.py`
- Create: `requirements-world-studio.txt`
- Create: `tests/test_world_studio_streaming.py`
- Create: `tests/test_world_studio_server.py`
- Modify: `narrative_dynamics/integrations/__init__.py`

**Interfaces:**
- Consumes: Task 2 dispatcher/service, V21.2 `SimulationOutputBus` and capability-filtered views.
- Produces: bounded resumable subscriptions and `create_world_studio_asgi_app(...)`.

- [ ] **Step 1: Add failing subscription recovery tests**

Use real V21.3 batches and prove:

```python
subscription = router.subscribe(
    "sub-1", "run-1", "stream-1", capability,
    tuple(SimulationOutputKind),
)
router.acknowledge("sub-1", first.last_sequence, first.content_hash)
replayed = router.resume("sub-1", first.last_sequence, first.content_hash)
assert tuple(item.source_batch_hash for item in replayed) == (second.content_hash,)
```

Test wrong capability, stale/forged ack, duplicate subscription ID, lease expiry,
per-connection/run capacity, retained-gap `SubscriptionHistoryGap` with scoped snapshot
token, and that private records never enter a public subscription buffer.

- [ ] **Step 2: Run streaming RED**

```text
python -m pytest tests/test_world_studio_streaming.py -q
```

- [ ] **Step 3: Implement bounded output router**

Define `StudioOutputRouter` and local implementation with methods `subscribe`,
`publish`, `acknowledge`, `resume`, and `unsubscribe`. Store only already-filtered
immutable views, bounded by batch count/record count/bytes and lease deadline. A clock
is injected only for deterministic expiry tests and is excluded from artifacts.

- [ ] **Step 4: Add failing ASGI route tests**

Using Starlette's test client, prove:

- `/health` returns only status/protocol version;
- `POST /rpc` dispatches one JSON-RPC request and preserves its ID;
- wrong content type, body limit, timeout, missing auth context, and raw exception are
  stable/redacted;
- `/v1/stream` requires `nd-jsonrpc-v1`, accepts subscribe/ack/resume/unsubscribe, and
  sends only JSON-RPC notifications;
- origin and maximum-connection/subscription policies fail closed;
- disconnect releases connection-owned subscription leases without cancelling commands.

- [ ] **Step 5: Run server RED**

```text
python -m pytest tests/test_world_studio_server.py -q
```

- [ ] **Step 6: Implement optional ASGI gateway**

Add optional dependencies to `requirements-world-studio.txt` and lazy imports in:

```python
create_world_studio_asgi_app(
    dispatcher,
    output_router,
    authenticate_http,
    authenticate_websocket,
    *,
    allowed_origins,
    limits,
    static_root=None,
)
```

Use Starlette routes and WebSocket primitives; deploy through Hypercorn for HTTP/2 and
WSS. Never hold a connection/router lock while calling the dispatcher/coordinator.
TLS material and auth tokens are constructor/deployment configuration only.

- [ ] **Step 7: Run Task 3 GREEN**

```text
python -m pytest tests/test_world_studio_streaming.py tests/test_world_studio_server.py tests/test_world_studio_jsonrpc.py -q
python -m compileall -q narrative_dynamics/integrations/world_studio_server.py narrative_dynamics/studio/streaming.py
git diff --check
```

- [ ] **Step 8: Commit Task 3**

```text
git add narrative_dynamics/integrations/world_studio_server.py narrative_dynamics/integrations/__init__.py narrative_dynamics/studio/streaming.py requirements-world-studio.txt tests/test_world_studio_streaming.py tests/test_world_studio_server.py
git commit -m "feat(studio): serve JSON-RPC over H2 and WSS"
```

---

### Task 4: Pure-Web scenario authoring editor

**Files:**
- Create: `world_studio_web/package.json`
- Create: `world_studio_web/package-lock.json`
- Create: `world_studio_web/tsconfig.json`
- Create: `world_studio_web/vite.config.ts`
- Create: `world_studio_web/index.html`
- Create: `world_studio_web/src/main.ts`
- Create: `world_studio_web/src/styles.css`
- Create: `world_studio_web/src/rpc/client.ts`
- Create: `world_studio_web/src/state/project-store.ts`
- Create: `world_studio_web/src/components/studio-shell.ts`
- Create: `world_studio_web/src/components/project-tree.ts`
- Create: `world_studio_web/src/components/property-inspector.ts`
- Create: `world_studio_web/src/components/diagnostic-list.ts`
- Create: `world_studio_web/src/editors/graph-editor.ts`
- Create: `world_studio_web/src/editors/map-editor.ts`
- Create: `world_studio_web/src/editors/json-editor.ts`
- Create: `world_studio_web/src/schema/studio-types.ts`
- Create: `world_studio_web/src/**/*.test.ts`

**Interfaces:**
- Consumes: Task 2 JSON-RPC method/result schemas and Task 3 `/rpc` route.
- Produces: persistent browser shell and graph/map/form/JSON editing views used by Task 5.

- [ ] **Step 1: Scaffold browser tests before production components**

Create a Vite/TypeScript/Vitest project with locked dependencies for X6, PixiJS,
Monaco, and test DOM tooling. Add failing tests proving the shell has semantic project
navigation, canvas mode tabs, inspector, diagnostics, raw JSON, run toolbar placeholder,
and timeline region; keyboard focus follows the visible reading order.

- [ ] **Step 2: Run shell RED**

```text
cd world_studio_web
npm ci
npm test -- --run
```

- [ ] **Step 3: Implement RPC client and authoritative project store**

Define exact TypeScript interfaces matching Task 2 JSON. `JsonRpcClient.call()` assigns
one request ID and never retries a state-changing request under a new idempotency key.
`ProjectStore.apply()` retains the prior view until the matching next revision returns;
stale errors reload `project.snapshot` and expose a conflict state instead of merging.

- [ ] **Step 4: Implement accessible application shell**

Use browser-native custom elements and semantic HTML. Provide skip link, labeled
landmarks, roving keyboard selection in the document tree, visible `:focus-visible`,
44px pointer targets, `aria-live` diagnostic/connection announcements, and
`prefers-reduced-motion` handling. CSS uses logical properties and container queries
for the inspector/timeline collapse.

- [ ] **Step 5: Add failing graph projection/edit tests**

For physical, social, story, and resource fixtures, assert stable domain IDs map to X6
nodes/edges and that create/connect/delete/property edits emit the exact
`project.apply` operation without treating X6 cell JSON as a scenario document.
Test keyboard node selection, edge creation through forms, and diagnostic pointer
selection.

- [ ] **Step 6: Implement X6 graph editor**

Render four explicit graph modes with shared selection and inspector adapters. Enable
pan/zoom, selection, snapline, minimap, and delete tools. Store X6 coordinates only via
`set_layout`; domain edits use RFC 6901 operations. Reject dangling edge creation in
the client and still rely on server diagnostics as authority.

- [ ] **Step 7: Add failing map/JSON synchronization tests**

Prove Pixi objects derive from physical documents/Tiled geometry, moving a place emits
only declared geometry/layout changes, passage endpoints retain stable IDs, Monaco
replacement emits `replace_document`, and a server diagnostic selects the matching
map object/property/Monaco pointer.

- [ ] **Step 8: Implement PixiJS and Monaco editors**

Use PixiJS for orthogonal map geometry, agents, objects, passage overlays, and optional
perception layers. Use Monaco's JSON model with server-returned diagnostics. Provide a
form/tree fallback for required map fields because canvas and Monaco alone are not an
accessible editing surface.

- [ ] **Step 9: Run Task 4 GREEN and production build**

```text
cd world_studio_web
npm test -- --run
npm run typecheck
npm run build
```

- [ ] **Step 10: Commit Task 4**

```text
git add world_studio_web
git commit -m "feat(studio-web): author scenario projects visually"
```

---

### Task 5: Run console, end-to-end flow, packaging, and release hardening

**Files:**
- Create: `world_studio_web/src/rpc/stream-client.ts`
- Create: `world_studio_web/src/state/run-store.ts`
- Create: `world_studio_web/src/components/run-toolbar.ts`
- Create: `world_studio_web/src/components/state-inspector.ts`
- Create: `world_studio_web/src/components/output-timeline.ts`
- Create: `world_studio_web/src/components/fork-comparison.ts`
- Create: `world_studio_web/src/**/*.test.ts`
- Create: `world_studio_web/e2e/law-firm.spec.ts`
- Create: `world_studio_web/playwright.config.ts`
- Create: `tools/run_world_studio.py`
- Create: `.github/workflows/world-studio.yml`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-01-web-simulation-platform-v21-design.md`

**Interfaces:**
- Consumes: Tasks 1–4 complete project/RPC/stream/browser surface.
- Produces: runnable V22 product, end-to-end law-firm proof, CI, and operator docs.

- [ ] **Step 1: Add failing run-store and stream-client tests**

Test create/run command/fork/query calls; one in-flight state-changing command per run;
stable idempotency across network retry; WSS subscribe/ack/resume; duplicate notification
deduplication by batch hash; gap recovery by scoped snapshot; and audience switching
that discards prior private records before requesting a new capability.

- [ ] **Step 2: Implement run state and WSS client**

`RunStore` owns run view, selected audience, public/Agent/network snapshot, command
receipts, output batches, connection status, and checkpoint/fork lineage. It never
optimistically advances canonical round/state. `StreamClient` validates subprotocol,
JSON-RPC notification shape, stream/run/hash binding, and monotonically increasing
acknowledgements.

- [ ] **Step 3: Add failing run-console component tests**

Prove legal lifecycle buttons reflect the current `ScenarioRunStatus`; step shows a
busy state; checkpoint/fork require capability; Agent selector never exposes an
unauthorized owner; output timeline preserves source sequences/gaps; parent/child
comparison displays exact state hashes and metrics; all controls are keyboard usable.

- [ ] **Step 4: Implement run console**

Build toolbar, state inspector, typed output timeline, connection/recovery status, and
side-by-side fork comparison. Disable controls from authoritative status/capability,
not local assumptions. Announce accepted/rejected commands and connection recovery via
bounded `aria-live` messages.

- [ ] **Step 5: Add the failing law-firm end-to-end test**

Playwright must:

1. import `examples/law_firm_scenario`;
2. edit one place label and one social relationship through visual/property views;
3. introduce and repair one dangling scene dependency using diagnostics;
4. compile the revision;
5. create, start, and step a run;
6. verify public output and owner-Agent privacy;
7. disconnect/reconnect the WSS stream and resume;
8. checkpoint and fork;
9. step parent/child and display their distinct hashes;
10. restart the service and reopen the saved project revision.

- [ ] **Step 6: Implement launcher and static packaging**

`tools/run_world_studio.py` accepts explicit workspace/import/export/static roots and
bind/TLS settings, constructs the local workspace/service/router, and starts Hypercorn.
It never reads `.env` implicitly. The frontend build uses hashed assets copied into the
configured static root; the ASGI app applies a restrictive CSP and immutable caching
for hashed assets.

- [ ] **Step 7: Add CI and documentation**

The workflow installs `requirements-world-studio.txt`, runs studio/network Python
tests, builds the frontend, runs unit/accessibility tests, and runs the Playwright
law-firm flow. README documents install, TLS/H2/WSS launch, import/edit/compile/run/
checkpoint/fork flow, capability model, recovery behavior, and explicit V22 non-goals.
Update the old V21 delivery sequence to link this V22 spec as the authoritative plan.

- [ ] **Step 8: Run complete V22 gates**

```text
python -m pytest tests/test_world_studio_*.py tests/test_network_abm_scenario_coordinator.py tests/test_network_abm_scenario_checkpoint_store.py tests/test_network_abm_simulation_output_bus.py tests/test_network_abm_public_api.py -q
python -m unittest discover -s tests -p "test_network_abm*.py" -q
python -m compileall -q narrative_dynamics tests tools/run_world_studio.py
cd world_studio_web
npm ci
npm test -- --run
npm run typecheck
npm run build
npx playwright test e2e/law-firm.spec.ts
cd ..
git diff --check
```

- [ ] **Step 9: Run forbidden-scope and identity audit**

Inspect added production dependencies and JSON schemas. Assert no React Flow,
Protobuf/gRPC, remote worker, provider, live Blender, arbitrary state-set, credential,
path, callback, or transport address enters a semantic contract/hash.

- [ ] **Step 10: Commit Task 5**

```text
git add world_studio_web tools/run_world_studio.py .github/workflows/world-studio.yml README.md docs/superpowers/specs/2026-09-01-web-simulation-platform-v21-design.md
git commit -m "feat(studio): deliver browser authoring and run console"
```

## GitHub tracking structure

The implementation is tracked by one `World Studio V22` milestone containing:

1. roadmap issue — scope, dependency graph, acceptance checklist, and release gates;
2. workspace issue — Task 1;
3. service issue — Task 2;
4. gateway issue — Task 3;
5. authoring UI issue — Task 4;
6. run console/release issue — Task 5.

Issues are executed in dependency order `workspace → service → gateway → authoring UI
→ run console/release`. UI mockups and interaction decisions may proceed in parallel
with backend work, but implementation acceptance never assumes an unavailable prior
contract.
