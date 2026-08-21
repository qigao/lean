# Lean Browser Runtime Safety Model

This repository formalizes safety properties for a stateful asynchronous CDP browser Driver in Lean 4.

The primary proof surface is **interaction + decoder/feedback/recovery contracts**, not a second full implementation of the Browser Driver. The larger runtime model remains as integration/reference evidence.

## Primary proof chain

```text
AdapterEvent
  -> decodeEvent
  -> RawObservation
  -> normalizeObservation
  -> NormalizedFeedback { FeedbackClass, FaultClass? }
  -> decideNormalized
  -> minimal sufficient recovery when required
  -> Healthy(new generation) | ExplicitFailure
```

`RawObservation` is a finite semantic vocabulary. Chromium/CDP version-specific JSON and error strings are parsed by a thin adapter before entering the proof core. Anything unknown, malformed, under-scoped, or ambiguous must become `unrecognized` rather than being silently dropped or optimistically guessed.

Representative normalized mappings:

```text
operationSucceeded          -> success / none
networkTransient            -> transient / none
humanInput                  -> conflict / none
elementDetached             -> stale / elementStale
executionContextDestroyed   -> stale / runtimeLost
frameDetached               -> stale / runtimeLost
sessionDetached             -> unavailable / sessionLost
pageClosed/pageCrashed      -> terminal / pageLost
contextDestroyed            -> terminal / contextLost
browserExited               -> terminal / browserLost
deadlineReached             -> timeout / timeoutFault
unrecognized                -> unknown / unknownFault
```

`FeedbackClass` alone does not choose recovery scope. If normalization identifies a `FaultClass`, `decideNormalized` uses that fault's `minimumRecovery`; only no-fault observations fall back to `decideFeedback`. Thus `elementDetached` selects `reResolve`, while `executionContextDestroyed` selects `rebindRuntime`, even though both are stale-style feedback.

`normalization_is_coherent` proves every raw observation produces an allowed feedback/fault pair. `recover_observation_fuel_is_resolved` composes normalization with bounded recovery and proves recovery is either not required or returns terminal `healthy`/`failed`, never an internally stuck `recovering` result.

Run the feedback/recovery check with:

```text
lake exe browser-feedback-recovery-check
```

## Decoder contract

`Browser/Interaction/Decoder.lean` formalizes the stable adapter boundary. It intentionally does not parse arbitrary Chromium JSON and does not choose recovery. It only converts structured adapter events into `RawObservation`.

Current critical CDP-style mappings include:

```text
Runtime.executionContextDestroyed -> executionContextDestroyed
Runtime.executionContextsCleared   -> executionContextDestroyed
Page.frameDetached                 -> frameDetached
Target.detachedFromTarget          -> sessionDetached
Inspector.detached                 -> sessionDetached
Target.targetCrashed(page)         -> pageCrashed
Target.targetDestroyed(page)       -> pageClosed
```

Target lifecycle events require an adapter-resolved scope. A worker/service-worker/unknown target crash or destroy is **not** promoted into a page fault; it decodes to `unrecognized` unless a higher layer has enough ownership information to classify it safely.

`Network.loadingFailed` and protocol command errors also require an adapter-resolved failure disposition:

```text
network transient       -> networkTransient
network cancelled       -> conditionPending
network blocked/CORS/?  -> unrecognized
protocol transient      -> protocolTransient
protocol sessionGone    -> sessionDetached
protocol unknown        -> unrecognized
```

This reflects a core rule: **method name alone is not always enough to select recovery scope**. Insufficient information must fail safe.

Decoder properties include:

- `decoder_output_safe`: scope/disposition ambiguity follows the fail-safe decoder rules.
- `decoded_normalization_is_coherent`: every decoded observation remains valid under the existing normalizer contract.
- malformed and unknown adapter events become `unrecognized -> unknownFault -> failSafe`.
- `recover_adapter_event_fuel_is_resolved`: the full decoder-to-recovery path is either unnecessary or terminal; it cannot internally livelock.

Run the end-to-end decoder check with:

```text
lake exe browser-decoder-recovery-check
```

The executable covers page/session recovery into a fresh generation, transient network waiting, and under-scoped/blocked/malformed inputs failing safe.

## Recovery contracts

Recovery actions are ranked from least to most disruptive:

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

`minimumRecovery` chooses the least sufficient repair for each fault, and `minimum_recovery_is_minimal` proves no lower-ranked sufficient action exists under that relation.

Successful recovery always produces `generation + 1`; it never revives the failed incarnation. Timeout and unknown faults begin directly in `failed` and cannot later be revived by a spurious success signal.

`runRecoveryFuel` abstracts environment availability with `RecoveryAction -> Bool`. The theorem `run_recovery_fuel_is_terminal` proves that for arbitrary finite fuel and arbitrary availability behavior, the internal recovery computation ends in `healthy` or `failed`, excluding recovery livelock.

This does not assert Chromium/network/OS eventually recover. The formal claim is conditional: if a repair succeeds before budget/fuel exhaustion, the Driver returns as a fresh generation; otherwise it fails explicitly in finite internal steps.

Run the lower-level recovery scenario with:

```text
lake exe browser-recovery-check
```

## Interaction contracts

`Browser/Interaction/` separately proves the dangerous cross-component boundaries:

- exact CDP `(ActionId, CdpCallId)` ownership
- deadline/timer ownership and stale timer rejection
- Human vs Automation input exclusivity
- terminal side-effect blocking after cancel/timeout/destroy
- node epoch/incarnation isolation
- Context→Page and Page→Frame ownership
- Policy issue→deliver causality
- typed authorized-effect composition

These contracts remain separate from decoding/normalization: identity, generation ownership, causal delivery, and side-effect authorization are not reconstructed in the feedback model.

## Driver-facing interaction journal

The multiplexed journal starts with a version header, then every event carries a globally increasing `seq` and opaque `lane`:

```json
{"kind":"interaction-journal","version":1}
{"kind":"interaction","seq":1,"lane":1,"event":"actionStarted","action":2}
{"kind":"interaction","seq":2,"lane":2,"event":"actionStarted","action":7}
```

Each lane owns an independent compact interaction state, while `seq` records total journal observation order.

```text
lake exe browser-driver-interaction-journal-check traces/driver-interaction-journal.jsonl
```

## C++ reference integration

The C++17 reference API lives under `driver/`:

```text
driver/include/browser/interaction_journal.hpp
driver/include/browser/interaction_boundary.hpp
driver/include/browser/jsonl_interaction_sink.hpp
driver/src/interaction_journal.cpp
driver/src/jsonl_interaction_sink.cpp
```

`InteractionBoundary` journals immediately before invoking the real Driver callable. This keeps formal evidence attached to the actual ownership/side-effect edge rather than reconstructing events afterward.

Intended placement:

```text
ActionEngine      -> before start/terminal transition publication
CdpSession        -> before transport send / response delivery
Deadline/Timer    -> before scheduler arm / Action timeout notification
InputRouter       -> before Input.dispatch*
InputObserver     -> before human-input delivery
PolicyEngine      -> before command enqueue / child delivery
RuntimeGraph      -> before incarnation/destruction mutation
```

## Integration/reference model

The larger `Browser/Async` model remains regression evidence for:

- Browser / Context / Page / Frame / Action logical actors
- NodeEpoch and ActionId generation isolation
- async Policy command emission and later delivery
- dynamic graph lifecycle
- duplicate/collision handling
- Driver JSONL replay

It is not the primary theorem surface.

## Verification

Lean is pinned to 4.33.0. CI runs:

```text
C++17 InteractionJournal concurrent smoke test
C++17 InteractionBoundary ordering smoke test
C++17 JSONL emitter through InteractionBoundary
lake build --wfail
lake exe browser-interaction-check
lake exe browser-recovery-check
lake exe browser-feedback-recovery-check
lake exe browser-decoder-recovery-check
lake exe browser-driver-interaction-journal-check /tmp/driver-interaction-journal.jsonl
lake exe browser-driver-interaction-journal-check traces/driver-interaction-journal.jsonl
lake exe browser-interaction-trace-check traces/interaction-contracts.jsonl
lake exe browser-runtime-check
lake exe browser-driver-trace-check traces/generation-race.jsonl
lake exe browser-async-trace-check traces/async-runtime-race.jsonl
```

## Design documents

- `docs/superpowers/specs/2026-08-21-async-runtime-design.md`
- `docs/superpowers/plans/2026-08-21-async-runtime.md`
- `docs/superpowers/specs/2026-08-21-interaction-contracts-design.md`
- `docs/superpowers/plans/2026-08-21-interaction-contracts.md`
- `docs/superpowers/plans/2026-08-21-driver-interaction-journal.md`
- `docs/superpowers/plans/2026-08-21-driver-boundary-instrumentation.md`
- `docs/superpowers/specs/2026-08-21-feedback-recovery-contracts-design.md`
- `docs/superpowers/plans/2026-08-21-feedback-recovery-contracts.md`
- `docs/superpowers/specs/2026-08-21-feedback-normalizer-design.md`
- `docs/superpowers/plans/2026-08-21-feedback-normalizer.md`
- `docs/superpowers/specs/2026-08-21-decoder-contract-design.md`
- `docs/superpowers/plans/2026-08-21-decoder-contract.md`
