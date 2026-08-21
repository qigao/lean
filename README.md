# Lean Browser Runtime Safety Model

This repository formalizes safety properties for a stateful asynchronous CDP browser Driver in Lean 4.

The primary proof surface is now **interaction contracts**, not a second full implementation of the Browser Driver. The larger runtime model remains as an integration/reference model.

## Interaction contracts

`Browser/Interaction/` models the dangerous boundaries between components:

- CDP request/response ownership by `(ActionId, CdpCallId)`
- Deadline/timer ownership and stale timer rejection
- Human vs automation input ownership
- Terminal side-effect blocking after cancel/timeout/destroy
- Node epoch/incarnation isolation
- Context→Page and Page→Frame ownership
- Policy issue→deliver causality
- Typed authorized-effect composition

The compact single-lane contract can be checked with:

```text
lake exe browser-interaction-trace-check traces/interaction-contracts.jsonl
```

## Feedback and recovery contracts

The Driver is also modeled as an asynchronous feedback-control system rather than a monolithic state machine:

```text
Operation
  -> Feedback
  -> Decision
  -> minimal sufficient recovery when needed
  -> Healthy(new generation) | ExplicitFailure
```

`FeedbackClass` is intentionally small: `success`, `transient`, `stale`, `conflict`, `unavailable`, `timeout`, `terminal`, and `unknown`. `decideFeedback` is total; unknown and timed-out work fail safe instead of silently retrying an old attempt.

Recovery actions have an explicit least-to-most disruptive rank:

```text
retry
reResolve
rebindRuntime
reattachSession
recreatePage
recreateContext
restartBrowser
fail
```

`minimumRecovery` selects the least sufficient repair for each fault, and `minimum_recovery_is_minimal` proves that no lower-ranked sufficient action exists under this recovery relation.

Recovery state contains only status, generation, fault, current recovery action, and finite budget. Successful recovery always produces `generation + 1`; it never revives the failed incarnation. Timeout and unknown faults start directly in `failed` and cannot later be revived by a spurious success signal.

`runRecoveryFuel` abstracts environmental recovery with `available : RecoveryAction -> Bool`. It escalates while repair is unavailable and forces any still-recovering computation to explicit failure when fuel is exhausted. The theorem `run_recovery_fuel_is_terminal` proves for arbitrary finite fuel and arbitrary availability behavior that the internal recovery computation ends in `healthy` or `failed`, excluding internal recovery livelock.

The executable recovery scenario checks both directions:

```text
pageLost
  -> recreatePage unavailable
  -> recreateContext available
  -> Healthy(generation + 1)

sessionLost
  -> all repair levels unavailable
  -> bounded escalation
  -> Failed
```

Run it with:

```text
lake exe browser-recovery-check
```

This does not assume Chromium, the network, or the OS eventually recovers. The formal guarantee is conditional: when a repair level succeeds before the finite budget/fuel is exhausted, the recovered actor is a fresh generation; otherwise the Driver terminates recovery explicitly rather than hanging indefinitely.

## Driver-facing interaction journal

The Driver-facing journal is multiplexed. It starts with a version header, then every event carries a globally increasing `seq` and an opaque `lane`:

```json
{"kind":"interaction-journal","version":1}
{"kind":"interaction","seq":1,"lane":1,"event":"actionStarted","action":2}
{"kind":"interaction","seq":2,"lane":2,"event":"actionStarted","action":7}
```

`seq` is the total order in which the journal observed records. Each `lane` owns an independent compact `InteractionTraceState`, so events from different Pages/Actions may interleave without mutating each other's protocol state. The canonical fixture is `traces/driver-interaction-journal.jsonl` and can be checked with:

```text
lake exe browser-driver-interaction-journal-check traces/driver-interaction-journal.jsonl
```

The journal vocabulary currently includes:

```text
actionStarted(action)
actionFinished(action)
cdpRequest(action, call)
cdpResponse(action, call)
deadlineArmed(action, expiresAt)
timerExpired(action, now)
inputDispatch(action)
humanInput
policyIssued(eventId, correlationId, depth, page)
policyDelivered
epochRecreated
actorDestroyed
```

`inputDispatch` and `cdpRequest` are real external side effects. They are rejected when ownership/liveness contracts do not authorize them. A matching expired deadline propagates into terminal liveness, so later input/CDP side effects are rejected.

## C++ Driver journal API

The reference C++17 API lives under `driver/`:

```text
driver/include/browser/interaction_journal.hpp
driver/include/browser/interaction_boundary.hpp
driver/include/browser/jsonl_interaction_sink.hpp
driver/src/interaction_journal.cpp
driver/src/jsonl_interaction_sink.cpp
```

Driver components never construct JSON or allocate sequence numbers. `InteractionJournal` serializes sequence assignment and sink delivery under one mutex, so concurrent emitters observe one deterministic journal order. `JsonlInteractionSink` alone owns the Lean-compatible wire format.

`InteractionBoundary` is the integration point for real Driver code. Every method records the typed interaction immediately before invoking the supplied real Driver callable. The reference emitter already uses this path, so the CI round-trip is:

```text
Driver-like callable
        ↓
InteractionBoundary
        ↓ journal first
InteractionJournal
        ↓
JsonlInteractionSink
        ↓
/tmp/driver-interaction-journal.jsonl
        ↓
Lean verifyInteractionJournalJsonl
```

### Placement rules

The boundary must sit at the actual ownership or side-effect edge, not at a higher-level observer that later guesses what happened:

```text
ActionEngine
  before publishing start transition     → actionStarted
  before publishing terminal transition  → actionFinished

CdpSession / protocol routing
  immediately before transport send      → cdpRequest
  immediately before response delivery   → cdpResponse

Deadline / Timer
  immediately before arming scheduler    → deadlineArmed
  inside timer callback, before Action    → timerExpired

InputRouter / InputObserver
  immediately before Input.dispatch*     → inputDispatch
  before delivering observed human input → humanInput

PolicyEngine
  immediately before command enqueue     → policyIssued
  immediately before child delivery      → policyDelivered

RuntimeGraph / node lifecycle
  immediately before incarnation mutation → epochRecreated
  immediately before destruction mutation → actorDestroyed
```

A journal entry means **the Driver attempted or delivered that interaction at the boundary**. If the delegated transport/send/callback/mutation throws, the journal entry is intentionally not rolled back; the exception propagates unchanged. Success/failure diagnostics belong to the normal telemetry/flight-recorder channel unless success/failure itself later becomes part of a formal interaction contract.

The real Driver owns `lane` assignment. The intended default is one stable lane per Page interaction domain, so Action/Input/Deadline interactions for one Page share a lane while independent Pages may interleave globally through `seq`.

CI compiles both the concurrent journal smoke and the boundary-ordering smoke. The boundary smoke asserts from inside delegated callables that the expected record already exists, including the exception path.

## Integration/reference model

The existing Browser/Async model remains for composed regression testing:

- Browser / Context / Page / Frame / Action logical actors
- NodeEpoch and ActionId generations
- async Policy command emission and later delivery
- dynamic graph lifecycle
- duplicate/collision handling
- Driver JSONL replay

## Verification

Lean is pinned to 4.33.0. CI runs:

```text
C++17 InteractionJournal concurrent smoke test
C++17 InteractionBoundary ordering smoke test
C++17 JSONL emitter through InteractionBoundary
lake build --wfail
lake exe browser-interaction-check
lake exe browser-recovery-check
lake exe browser-driver-interaction-journal-check /tmp/driver-interaction-journal.jsonl
lake exe browser-driver-interaction-journal-check traces/driver-interaction-journal.jsonl
lake exe browser-interaction-trace-check traces/interaction-contracts.jsonl
lake exe browser-runtime-check
lake exe browser-driver-trace-check traces/generation-race.jsonl
lake exe browser-async-trace-check traces/async-runtime-race.jsonl
```

Design documents:

- `docs/superpowers/specs/2026-08-21-async-runtime-design.md`
- `docs/superpowers/plans/2026-08-21-async-runtime.md`
- `docs/superpowers/specs/2026-08-21-interaction-contracts-design.md`
- `docs/superpowers/plans/2026-08-21-interaction-contracts.md`
- `docs/superpowers/plans/2026-08-21-driver-interaction-journal.md`
- `docs/superpowers/plans/2026-08-21-driver-boundary-instrumentation.md`
- `docs/superpowers/specs/2026-08-21-feedback-recovery-contracts-design.md`
- `docs/superpowers/plans/2026-08-21-feedback-recovery-contracts.md`
