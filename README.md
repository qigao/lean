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

A compact Driver EventJournal can be checked without reconstructing the full runtime:

```text
lake exe browser-interaction-trace-check traces/interaction-contracts.jsonl
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

`inputDispatch` and `cdpRequest` are treated as real external side effects. They are rejected when ownership/liveness contracts do not authorize them. A matching expired deadline propagates into terminal liveness, so later input/CDP side effects are rejected.

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
lake build --wfail
lake exe browser-interaction-check
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
