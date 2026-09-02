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
