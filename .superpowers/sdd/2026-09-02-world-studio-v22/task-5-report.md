# Task 5 implementation report

Date: 2026-09-02

## Branch and revision

- Worktree: `C:\projects\lean\.worktrees\world-studio-v22-1-workspace`
- Branch: `feature/world-studio-v22-1-workspace`
- Required/base revision: `5727208b9e5340cf60c6b0f1356179b0299c5bc4`
- Pre-commit HEAD: `5727208b9e5340cf60c6b0f1356179b0299c5bc4`
- Focused commit: this report's enclosing `HEAD`, one commit above the recorded base,
  named `feat(studio): deliver browser authoring and run console`
- Initial commit before the report-metadata amendment: `dae44c01e000dd9ef77d28ec0a2cd5999f2df020`
- Final HEAD hash: reported in the handoff after amendment (a commit cannot embed its own
  content-derived hash)
- Push/merge/PR: intentionally not performed

## Preflight rulings applied

1. Task 3 released an active subscription and its retained views on disconnect, while
   Task 5 requires identical-subscription recreation followed by resume. The router now
   keeps a capacity- and lease-bounded released ledger containing only the immutable
   binding, already-filtered views, and cursor. It never retains callback or connection
   identity. An exact capability and full-binding match atomically claims the entry;
   mismatch, expiry, and capacity behavior are covered directly.
2. `ScenarioCoordinator` expects `publisher.publish(batch)` to return a
   `SimulationDeliveryReport`, while `StudioOutputRouter.publish(run_id, batch)` returns
   subscription IDs. The launcher uses a per-run adapter that constructs the exact
   report and leaves audience filtering/retention in the router.
3. Fork previously hard-coded a new `SimulationOutputBus`, preventing child WSS output.
   A keyword-only optional `child_publisher` keeps the old default and is validated
   before checkpoint side effects. Launcher forks receive a per-child router adapter;
   publisher/topology data is absent from request, checkpoint, result, and run identity.
4. `-32016` recovery adds no `/recovery` route and no `state.recover` method. The snapshot
   token remains opaque correlation. The client chooses only `state.public`,
   `state.agent`, or `state.network` from the immutable active binding, validates the
   returned scope, replaces that scope, resets to the reported cursor, and recreates the
   subscription.

No unresolved Task 1-4 contract contradiction required a broader redesign.

## Delivered behavior

- Exact TypeScript session, run, command, fork, scoped-state, stream, and discriminated
  output contracts; JSON-RPC callers can carry one explicit request ID across retries.
- `RunStore` accepts only validated authoritative run views, serializes state-changing
  work per run, retains exact command/fork identities across both request and subsequent
  view-refresh transport failures, bounds output bytes/records/batches and announcements,
  and clears private data before an audience request.
- `StreamClient` requires `nd-jsonrpc-v1`, validates strict JSON-RPC and output binding,
  rejects duplicate JSON keys/conflicts/regressions/gaps, bounds its dedup ledger,
  commits before monotonic acknowledgement, recreates an identical released binding
  before resume, and performs privacy-safe audience switching/scoped gap recovery.
- Accessible run toolbar, scoped state inspector, typed/owner-visible bounded timeline,
  gap/recovery markers, exact fork comparison, and persistent bounded announcement log
  are integrated into the existing single-main Task 4 shell. Semantic fallbacks, stable
  live regions, logical/RTL layout, 320 px reflow, forced colors, reduced motion, and
  target sizes remain covered.
- Static ASGI routing serves a revalidated index, immutable hashed assets, no-cache
  unhashed assets, non-cacheable authenticated session authority, and restrictive CSP
  without unsafe inline/eval; encoded traversal/backslash paths are rejected.
- The explicit launcher composes real SQLite project authority, run registry,
  coordinator, local checkpoint store, router, auth, and Hypercorn H2/WSS configuration.
  It accepts only configured roots/source IDs/bind/origin/limits/TLS, reads no `.env` or
  provider, and refuses trust-all outside loopback or non-loopback without auth and TLS.
- The Playwright law-firm flow performs all ten numbered steps in Chromium against a
  spawned real Hypercorn process, ASGI gateway, SQLite workspace, coordinator, output
  router, WSS, checkpoint store, and fork. Only process control/readiness is a transport
  seam; authority, persistence, compilation, privacy, and runtime are not mocked.
- Python/browser inputs are exact-pinned/lockfile-recorded. CI installs Chromium only,
  runs every focused gate, and uploads only three-day Playwright diagnostics on failure.
  README and the V21 design record persistent-project/process-local-run behavior,
  security/capability/recovery operation, the authoritative V22 link, and all non-goals.

## RED to GREEN evidence

All implementation slices began with behavioral tests using real typed objects and
SQLite/coordinator objects. Browser unit tests use only transport seams; clock control is
limited to deterministic router expiry, and I/O-failure seams are explicit.

- Recovery ledger RED: disconnect made identical resubscribe/resume impossible and the
  new expiry/capacity/mismatch tests failed. GREEN: exact replay, privacy mismatch,
  expiry/capacity eviction, and active-slot release pass in router and gateway suites.
- Child publication RED: a forked child always published to a private default bus.
  GREEN: default compatibility, invalid publisher before restore, child delivery, cached
  retry semantics, and unchanged identity tests pass with the optional child publisher.
- Run/stream browser RED: missing modules/contracts and strict-envelope behaviors failed.
  GREEN: authoritative gating, retry, dedup/conflict/gap/ack/reconnect/recovery/privacy
  tests pass with real Task 2/3 wire shapes.
- Launcher/static RED: no product composition/static route existed. GREEN: real
  SQLite/coordinator/checkpoint/fork/output composition and auth/TLS/root/cache/CSP/
  traversal tests pass.
- Real E2E RED history: direct-script module import failed; Hypercorn was absent; the
  repair initially sent an object to an exact scalar pointer; axe reported a roleless
  labelled canvas; and an early assertion assumed a private record in a round that
  deterministically produced only public metrics. Each failure was corrected at its
  actual boundary. The first complete real E2E then passed (`1 passed in 9.1s`).
- Session caching RED: `test_health_is_exact_and_rpc_preserves_request_id` raised
  `KeyError: cache-control`. GREEN: authenticated `/session` returns `no-store`.
- Packaging RED: the first complete `npm test -- --run` let Vitest discover the
  Playwright spec, producing one failed suite while all 67 unit tests passed. GREEN:
  Vitest now includes only `src/**/*.test.ts`.
- Timeline privacy RED: an Agent diagnostic rendered `diagnostic (agent)` without owner.
  GREEN: it renders `diagnostic (agent, owner alice)`, making the E2E no-Bob assertion
  inspect actual displayed owner metadata.
- Post-accept retry RED: two tests showed accepted command/fork followed by failed
  `run.view` refresh generated `command-2`/`fork-2` on caller retry. GREEN: identities
  remain `command-1`/`fork-1` until authoritative views validate (`17/17` focused).

## Final verification evidence

Host used for local evidence: Python 3.12.7, Node 24.5.0, npm 11.5.2, Windows.

- Locked Python install:
  `python -m pip install --disable-pip-version-check -r requirements-world-studio.txt`
  — every exact pin satisfied.
- Focused Python gate (PowerShell-resolved `tests/test_world_studio_*.py` plus every exact
  remaining path from the brief): `239 passed, 1 skipped, 89 subtests passed in 82.31s`.
  PowerShell does not expand an external-command wildcard; CI runs the brief's literal
  command under the Ubuntu shell. The one skip is the pre-existing Windows directory
  symlink privilege case.
- `python -m unittest discover -s tests -p "test_network_abm*.py" -q` —
  `Ran 697 tests in 157.703s`, `OK (skipped=1)`.
- `python -m compileall -q narrative_dynamics tests tools/run_world_studio.py` — exit 0.
- `npm ci` — 151 packages installed from the lockfile.
- Final `npm test -- --run` — 14 files, 70 tests passed. npm 11 warns that `--run` is an
  unknown CLI config; non-interactive Vitest ran once and exited successfully.
- Final `npm run typecheck` — exit 0.
- Final `npm run build` — 2,751 modules transformed, build passed in 2.53s. Only the
  existing lazy Monaco chunk-size warning remains.
- Final `npx playwright test e2e/law-firm.spec.ts` — `1 passed (11.5s)`.
- `git diff --check` — clean; Git reports only host LF-to-CRLF conversion notices.

## Forbidden-scope and identity audit

- Direct dependency inspection: no React/React Flow, Protobuf/gRPC, provider/LLM SDK,
  remote worker/broker, or Blender control dependency was added. New test dependencies
  are exactly `@playwright/test@1.62.1` and `@axe-core/playwright@4.13.0`.
- Production import/string scan over the changed/new Python and TypeScript surfaces is
  clean for React Flow, Protobuf/gRPC, state-set, remote worker, and provider/LLM.
  `blender.delta` is only the consumed V21 typed output discriminant; there is no live
  Blender connection or mutation path.
- Hash inspection: the only new direct hash construction is the opaque recovery token
  over schema, run ID, stream ID, current sequence/batch hash, and capability hash.
  Credentials, filesystem paths, callback/connection IDs, publisher objects, bind
  addresses, and deployment topology do not enter semantic request/run/fork/checkpoint
  schemas or hashes. Tests also compare fork identities across publisher choices.
- Static CSP contains neither `unsafe-inline` nor `unsafe-eval`; server and E2E tests
  exercise same-origin RPC/WSS and Monaco's minimum blob worker allowance.

## Files

Backend/runtime and tests:

- `narrative_dynamics/abm/scenario_coordinator.py`
- `narrative_dynamics/integrations/world_studio_server.py`
- `narrative_dynamics/studio/streaming.py`
- `tools/run_world_studio.py`
- `requirements-world-studio.txt`
- `tests/test_network_abm_scenario_coordinator.py`
- `tests/test_world_studio_launcher.py`
- `tests/test_world_studio_server.py`
- `tests/test_world_studio_streaming.py`

Browser product and tests:

- `world_studio_web/.gitignore`
- `world_studio_web/package.json`
- `world_studio_web/package-lock.json`
- `world_studio_web/playwright.config.ts`
- `world_studio_web/e2e/law-firm.spec.ts`
- `world_studio_web/vite.config.ts`
- `world_studio_web/src/main.ts`
- `world_studio_web/src/schema/studio-types.ts`
- `world_studio_web/src/rpc/client.ts`
- `world_studio_web/src/rpc/client.test.ts`
- `world_studio_web/src/rpc/stream-client.ts`
- `world_studio_web/src/rpc/stream-client.test.ts`
- `world_studio_web/src/state/run-store.ts`
- `world_studio_web/src/state/run-store.test.ts`
- `world_studio_web/src/components/run-toolbar.ts`
- `world_studio_web/src/components/state-inspector.ts`
- `world_studio_web/src/components/output-timeline.ts`
- `world_studio_web/src/components/fork-comparison.ts`
- `world_studio_web/src/components/studio-workflow.ts`
- `world_studio_web/src/components/studio-workflow.test.ts`
- `world_studio_web/src/components/run-console.test.ts`
- `world_studio_web/src/components/studio-shell.ts`
- `world_studio_web/src/components/studio-shell.integration.test.ts`
- `world_studio_web/src/editors/graph-editor.ts`
- `world_studio_web/src/editors/json-editor.ts`
- `world_studio_web/src/editors/map-editor.ts`
- `world_studio_web/src/styles.css`

Release/docs/report:

- `.github/workflows/world-studio.yml`
- `README.md`
- `docs/superpowers/specs/2026-09-01-web-simulation-platform-v21-design.md`
- `.superpowers/sdd/2026-09-02-world-studio-v22/task-5-report.md`

## Disclosed concerns and environment notes

- `npm audit --omit=dev` reports the already-ledgered Monaco-vendored DOMPurify chain:
  two findings (one low, one moderate). The proposed forced fix downgrades Monaco across
  a breaking boundary and does not establish that the vendored copy is patched. V22
  continues to prohibit untrusted custom HTML and enforces the restrictive CSP; this
  report does not claim those controls patch the dependency.
- The production build retains the known lazy Monaco chunk-size warning. X6, PixiJS,
  and Monaco remain lazy; no new production browser dependency was added.
- Chromium build 1234 for Playwright 1.62.1 installed successfully. No browser,
  Hypercorn, SQLite, or coordinator limitation remains. Earlier Hypercorn installation
  exposed an old shared-environment `httpcore`/`h11` conflict; exact `httpx==0.28.1`,
  `httpcore==1.0.9`, and `h11==0.16.0` alignment resolved it before the full Python gate.
- Generated `world_studio_web/test-results/` and `playwright-report/` are ignored. The
  generated `.last-run.json` was deleted before staging; no diagnostic artifact is part
  of the commit.

## Fix Round 1

Review base and pre-commit head: `64befdc8e1f91e54fd5bdc0ab008fdfcbe7c80ae` on
`feature/world-studio-v22-1-workspace`. This round addresses only the reviewer's one
Critical and eight Important findings; it does not take the deferred Task 4 Minor or
broaden V22's provider, worker, renderer, protocol, or state-mutation scope.

### RED to GREEN evidence

1. Browser authentication. RED tests demonstrated that token mode had no native-browser
   credential path because fetch and WebSocket construction could not supply the launch
   token. GREEN adds an accessible bootstrap form that sends the token once in an
   `Authorization` header to `POST /session`; the server stores only a random opaque,
   bounded, expiring session identifier and returns an HttpOnly, SameSite=Strict,
   Path=/ cookie (Secure under TLS). `GET /session`, `POST /rpc`, WSS, logout, expiry,
   deterministic capacity eviction, invalid credentials, and absence of token
   reflection are covered. Launcher token mode disables ambient authorization, while
   explicit loopback-only development trust-all remains compatible.
2. Closed stream audience. RED tests showed the wire subscription omitted audience and
   the router inferred analyst access from capability. GREEN requires exact
   `public|agent|analyst` on WSS, requires owner exactly for `agent`, explicitly checks
   analyst `state.network` and agent ownership, filters before retention, and binds the
   audience through subscribe echo, active/released identity, and reconnect. The UI's
   `network` label maps only at the boundary to wire `analyst`.
3. Ack loss. RED simulated a server-processed ack whose response was lost: the client
   deduplicated the replay without re-acking and could deadlock. GREEN separates the
   committed source cursor from the confirmed acknowledgement cursor, resumes from the
   safe committed boundary, and idempotently re-acks an exact committed duplicate until
   confirmation before accepting the following batch.
4. Released publish/rebind race. RED used a deterministic projection barrier to move a
   released subscription into the active map between publish snapshot and commit. The
   old commit path lost that batch. GREEN resolves the exact binding in either map while
   holding the commit lock, retains/delivers once, and accepts the next contiguous batch
   without a gap.
5. Strict command and fork authority. RED cases accepted unknown fields, mismatched
   command/idempotency IDs, swapped parent/child views, and incomplete refreshed-view
   bindings. GREEN uses exact-key validators and compares every request/result and
   authoritative run, stream, scenario, epoch, state, checkpoint, and lineage binding
   before replacing canonical state or releasing the stable retry identity. Accepted
   commands/forks whose refresh fails retain the same logical identity on retry.
6. Network binding and audience generations. RED showed network state lacked
   run/scenario/state identity and a delayed private response could overwrite a later
   public selection. GREEN extends the exact Task 2 response with its schema/run/scenario/
   state binding, validates it in `RunStore`, clears private data synchronously, disables
   the selector while pending, and serializes/generates switches so stale agent work can
   neither commit nor establish a stale subscription.
7. `/studio/*` packaging. RED cases showed literal path resolution could not serve the
   configured Vite base or a deep link. GREEN builds with base `/studio/`, strips that
   prefix safely, serves hashed assets immutable, unhashed assets no-cache, and the index
   revalidated for safe SPA paths. Missing assets, unprefixed assets, traversal,
   backslash, NUL, residual encoding, and double-encoded traversal return 404.
8. Production timeline markers. RED integration coverage showed the shell supplied only
   batches and no production path populated marker UI. GREEN records bounded per-run
   markers for detected gaps, scoped recovery start/completion/failure, and resume
   boundaries; it binds them through the real shell and preserves source ordering and
   bounded persistent aria-live announcements. Tests drive real stream transitions.
9. Real E2E strength. The first strengthened RED found that the selected inspector still
   displayed the pre-edit 0.8 relationship; the test now reselects through the UI and
   verifies 0.85. The next RED proved the authored output policy legally produced only
   public metrics, so the browser now authors the real private-output allowance through
   the Raw JSON editor. GREEN asserts an actual Alice-owned output, no Bob output, the
   relationship change and persistence, and exact numeric revision plus content hash
   equality across process restart, while retaining all ten real browser/ASGI/SQLite/
   coordinator steps.

Focused GREEN evidence includes 23 streaming tests, 16 RunStore tests, 12 StreamClient
tests, and the final combined browser unit suite below. All tests use real router,
gateway, store, scenario, coordinator, SQLite, or browser objects as appropriate; no
authority or persistence mock was introduced.

### Final verification evidence

Host used for this round: Python 3.12.7, Node 24.5.0, npm 11.5.2, Windows.

- `npm ci` — 151 packages installed from the exact lockfile.
- The brief's complete focused Python gate (with PowerShell expanding
  `tests/test_world_studio_*.py` before invoking pytest) — `246 passed, 1 skipped, 89
  subtests passed in 55.20s`. The skip remains the Windows directory-symlink privilege
  case.
- `python -m unittest discover -s tests -p "test_network_abm*.py" -q` — `Ran 697
  tests in 50.193s`, `OK (skipped=1)`.
- `python -m compileall -q narrative_dynamics tests tools/run_world_studio.py` — exit 0.
- `npm test -- --run` — 15 files, 83 tests passed.
- `npm run typecheck` — exit 0.
- `npm run build` — 2,752 modules transformed in 2.14s; only the known lazy Monaco
  chunk-size warning remains.
- `npx playwright test e2e/law-firm.spec.ts` against the production build and launcher
  token mode — `1 passed (9.1s)`.
- `git diff --check` — clean apart from host LF-to-CRLF notices.

### Fix-round files

- Runtime and gateway: `narrative_dynamics/integrations/world_studio_server.py`,
  `narrative_dynamics/studio/service.py`, `narrative_dynamics/studio/streaming.py`,
  `tools/run_world_studio.py`.
- Backend tests: `tests/test_world_studio_launcher.py`,
  `tests/test_world_studio_server.py`, `tests/test_world_studio_service.py`,
  `tests/test_world_studio_streaming.py`.
- Browser runtime: `world_studio_web/src/session-bootstrap.ts`,
  `world_studio_web/src/main.ts`, `world_studio_web/src/rpc/stream-client.ts`,
  `world_studio_web/src/state/run-store.ts`, the state/timeline/toolbar/shell components,
  schema, styles, index, and Vite configuration.
- Browser tests: `world_studio_web/src/session-bootstrap.test.ts`, StreamClient,
  RunStore, component/integration tests, and `world_studio_web/e2e/law-firm.spec.ts`.
- Documentation: `README.md` and this report.

### Audit and disclosed concerns

- The forbidden-scope scan remains clean: no React/React Flow, gRPC/Protobuf,
  provider/LLM integration, remote worker/broker, live Blender control, or state-set
  endpoint was added. Task 2's RPC set remains closed; no `/recovery` route or
  `state.recover` method exists.
- Identity inspection found no credential, cookie/session, filesystem/static path,
  callback/connection ID, publisher object, bind address, audience-switch generation,
  or deployment topology in semantic command, fork, run, checkpoint, content, or
  recovery identities. Browser session identifiers are random opaque transport
  credentials and contain no semantic identity.
- CSP still contains neither `unsafe-inline` nor `unsafe-eval`; session tokens are not
  placed in URLs, assets, request bodies, cookies, local/session storage, DOM messages,
  or returned payloads.
- `npm audit --omit=dev` still reports the previously disclosed Monaco-vendored
  DOMPurify chain (two findings: one low, one moderate). The offered forced remediation
  crosses a Monaco breaking downgrade, so this narrowly scoped fix does not apply it.
- The production build's known Monaco chunk warning remains. Playwright Chromium is
  installed and the real token-mode E2E has no environment blocker.
- `world_studio_web/test-results/` and `playwright-report/` are ignored; generated run
  artifacts are removed before the fix commit.

## Fix Round 2

Review base and pre-commit head: `6aeb8ec8b459e356c48069371820f8fdddd6ccb0` on
`feature/world-studio-v22-1-workspace`. The final commit hash is reported in the handoff
because a commit cannot embed its own resulting hash. This round changes only the two
remaining parts of review Finding 6.

### RED to GREEN evidence

1. Retiring audience delivery generation. The deterministic browser RED begins an
   Agent-to-Public switch, holds the unsubscribe response, and delivers an old
   Alice-private notification. Current production failed before the barrier assertion:
   `client.binding` still returned the Agent binding instead of `null` (`1 failed, 12
   passed`). GREEN invalidates the active delivery binding and cursors synchronously,
   before the clearing callback or any await. Every queued notification captures its
   delivery generation, so a retired generation returns before validation, delivery,
   acknowledgement, or marker creation. Retired subscription IDs are bounded by the
   existing client identity limit and ignored non-fatally, while control responses
   continue to resolve. Each audience switch derives a generation-unique wire
   subscription ID, so even a caller reusing the same ID cannot confuse a late Agent
   frame with the new Public subscription. The held-unsubscribe test now observes zero
   output deliveries, acknowledgements, markers, and protocol errors before and after
   Public activation, and the focused StreamClient suite is `13 passed`.
2. Atomic analyst state/run binding. The coordinator RED failed with `AttributeError`
   because there was no atomic pair API. The next service RED observed zero calls to the
   new pair and proved the handler still made separate reads. GREEN adds
   `ScenarioCoordinator.network_state_with_run_view`, which holds the existing outer
   `RLock` while reusing the existing network and run projections. A deterministic
   projector barrier starts a real STEP concurrently and proves `_submit_command`
   cannot enter between the network and run projectors; the returned pair remains round
   0 with its exact old state hash while the later transition advances the coordinator
   to round 1 and a different hash. The service now serializes only that returned pair.
   The coordinator and service single-test gates both pass.

No callback, publisher, or transport work was moved under the coordinator state lock;
the existing `network_state` and `run_view` public APIs and all semantic identities are
unchanged.

### Verification evidence

Host used for this round: Python 3.12.7, Node 24.5.0, npm 11.5.2, Windows.

- Focused `npx vitest run src/rpc/stream-client.test.ts src/state/run-store.test.ts` —
  2 files, 29 tests passed.
- Focused full service/coordinator regression — `73 passed, 30 subtests passed in
  39.79s`.
- `npm ci` — 151 packages installed from the unchanged exact lockfile.
- The complete brief Python pytest gate, with PowerShell expanding the Studio wildcard
  before invocation — `248 passed, 1 skipped, 89 subtests passed in 84.89s`. The skip
  remains the Windows directory-symlink privilege case.
- `python -m unittest discover -s tests -p "test_network_abm*.py" -q` — `Ran 698
  tests in 78.070s`, `OK (skipped=1)`.
- `python -m compileall -q narrative_dynamics tests tools/run_world_studio.py` — exit 0.
- `npm test -- --run` — 15 files, 84 tests passed.
- `npm run typecheck` — exit 0.
- `npm run build` — 2,752 modules transformed in 6.58s; only the known lazy Monaco
  chunk-size warning remains.
- `npx playwright test e2e/law-firm.spec.ts` against the production build and launcher
  token mode — `1 passed (13.9s)`.
- `git diff --check` — clean apart from host LF-to-CRLF notices.

### Files, scope, and audit

- Production: `world_studio_web/src/rpc/stream-client.ts`,
  `narrative_dynamics/abm/scenario_coordinator.py`, and
  `narrative_dynamics/studio/service.py`.
- Tests: `world_studio_web/src/rpc/stream-client.test.ts`,
  `tests/test_network_abm_scenario_coordinator.py`, and
  `tests/test_world_studio_service.py`.
- Report: this file only. No dependency, schema, RPC, route, component, launcher,
  persistence, or documentation contract changed.
- The forbidden-scope scan has no added React/React Flow, gRPC/Protobuf, provider/LLM,
  remote-worker, state-set, or Blender surface. The identity scan shows only the
  generation-bound transport subscription ID; no generation, connection, callback,
  credential, path, publisher, bind address, or topology value enters a command, fork,
  run, checkpoint, content, or recovery semantic identity/hash.
- `npm audit --omit=dev` still reports only the previously disclosed Monaco-vendored
  DOMPurify chain (two findings: one low and one moderate); the offered forced fix is a
  breaking Monaco downgrade. No production or development dependency changed.
- The Playwright `.last-run.json` artifact was removed before staging; generated result
  directories remain ignored.

## Fix Round 3

Review base and pre-commit head: `c7051e1feb9e8043fae9b61bceb6da6123fe4e3c` on
`feature/world-studio-v22-1-workspace`. The final commit hash is reported in the handoff
because a commit cannot embed its own resulting hash. This round changes only the
same-tick audience-switch retirement finding.

### RED to GREEN evidence

The exact RED creates a real `StreamClient` with a server-like control registry capped
at two active subscriptions, calls two audience switches synchronously without a
microtask flush, and delivers an old Alice-private notification while both requests are
pending. Current production failed with `1 failed, 13 passed`: the registry observed
zero unsubscribe requests for `subscription-original`, not the required one. The trace
confirmed the first deferred operation saw its stale generation and returned before
cleaning its captured original binding; the second call had already detached
`activeBinding` and therefore captured no retirement work.

GREEN makes cleanup of every captured active binding the first unconditional action in
the existing serialized audience-switch chain. Generation checks now gate only whether
the candidate may subscribe and become active. Cleanup IDs are exact-validated,
deduplicated, and bounded by the existing client identity limit; a rejected or malformed
cleanup is ambiguous authority and closes the transport before rejecting. Candidate
subscriptions that become stale after subscribe are retired through the same path.
Control responses still bypass the notification delivery chain, and the generation-
unique wire IDs from Fix Round 2 are unchanged.

The server-registry test now proves the original ID is unsubscribed exactly once, the
superseded candidate is never left subscribed, only the final Public generation remains
active, and no old private output, acknowledgement, marker, or protocol error is
produced. Six additional same-tick switch pairs keep registry occupancy at one and never
hit its capacity. A separate cleanup-ambiguity RED observed the socket remain open after
`unsubscribed: false`; GREEN closes it and leaves the client disconnected. Final focused
StreamClient evidence is `15 passed`.

### Verification, scope, and audit

Host used for this round: Node 24.5.0, npm 11.5.2, Windows.

- `npx vitest run src/rpc/stream-client.test.ts` — 1 file, 15 tests passed.
- The first full unit/build pass found no behavior failure (`86 tests passed`; production
  build passed) but typechecking correctly rejected four possibly undefined JSON values
  in the new server-registry test harness. The harness now validates and narrows those
  fields before echoing them; focused test and typecheck then passed.
- Final `npm test -- --run` — 15 files, 86 tests passed.
- Final `npm run typecheck` — exit 0.
- Final `npm run build` — 2,752 modules transformed in 4.99s; only the known lazy Monaco
  chunk-size warning remains.
- `npx playwright test e2e/law-firm.spec.ts` against the production build and launcher
  token mode — `1 passed (13.7s)`.
- `git diff --check` — clean apart from host LF-to-CRLF notices.
- Production changed only `world_studio_web/src/rpc/stream-client.ts`; coverage changed
  only `world_studio_web/src/rpc/stream-client.test.ts`; this report is the third file.
  No Python, RPC/schema, server, component, dependency, or persistence surface changed,
  so the proportionate gate is the complete browser/build/real-E2E surface.
- The forbidden and identity scan is clean. Only bounded transport subscription cleanup
  state was added; it does not enter semantic content, command, fork, run, checkpoint,
  recovery, or capability identities/hashes. Dependency manifests are unchanged.
- `npm audit --omit=dev` still reports only the previously disclosed Monaco-vendored
  DOMPurify chain (two findings: one low and one moderate); the offered forced fix is a
  breaking Monaco downgrade.
- The Playwright `.last-run.json` artifact was removed before staging; generated result
  directories remain ignored.
