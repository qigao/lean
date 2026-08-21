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
driver/include/browser/jsonl_interaction_sink.hpp
driver/src/interaction_journal.cpp
driver/src/jsonl_interaction_sink.cpp
```

Driver components call typed methods such as `action_started`, `cdp_request`, `timer_expired`, `input_dispatch`, and `human_input`. They do not serialize JSON or allocate sequence numbers. `InteractionJournal` serializes sequence assignment and sink delivery under one mutex, so concurrent emitters observe one deterministic journal order. `JsonlInteractionSink` owns the Lean-compatible wire format.

CI compiles a concurrent C++ smoke test and also performs a real round trip:

```text
C++ InteractionJournal
        ↓
JsonlInteractionSink
        ↓
/tmp/driver-interaction-journal.jsonl
        ↓
Lean verifyInteractionJournalJsonl
```

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
C++17 InteractionJournal smoke test
C++17 JSONL emitter
lake build --wfail
lake exe browser-interaction-check
lake exe browser-driver-interaction-journal-check /tmp/driver-interaction-journal.jsonl
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
