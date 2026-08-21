# Lean Browser Runtime Safety Model

This repository formalizes safety properties for a stateful asynchronous CDP browser Driver in Lean 4.

The supported public proof chain is deliberately limited to:

```text
Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop
```

The larger whole-`AsyncRuntime` model, projection adapters, executable scenarios, JSONL replay, and historical whole-runtime theorems remain integration/reference evidence rather than a second public proof API.

## Import surfaces

Use the smallest entry point that matches the task:

```lean
import Browser.ProofAPI    -- supported formal proof surface
import Browser             -- executable/runtime model
import Browser.Integration -- historical proofs, projection, specs, replay/conformance
```

`Browser.ProofAPI` intentionally does not expose `Browser.AsyncRuntime`; CI enforces this with a negative compile guard.

## Public proof architecture

The Driver-facing control path is:

```text
AdapterEvent
  -> decodeEvent
  -> RawObservation
  -> normalizeObservation
  -> FaultClass?
  -> generation filtering
  -> aggregateRecovery
  -> bounded recovery
  -> Healthy(new generation) | ExplicitFailure
```

The five public layers have distinct responsibilities:

1. **Interaction** proves ownership, input exclusivity, terminal side-effect safety, epoch isolation, and causal delivery constraints.
2. **Feedback** decodes adapter input into a finite semantic vocabulary and fails safe on unknown, malformed, ambiguous, or under-scoped input.
3. **Recovery** selects the least sufficient repair and guarantees bounded internal resolution to healthy or explicit failure.
4. **Aggregation** combines concurrent finite faults into one duplicate-insensitive, permutation-invariant, least sufficient recovery requirement and connects runtime trace folding to that algebra.
5. **ClosedLoop** composes runtime selection, minimum recovery, aggregation, and bounded execution into the terminal system guarantee.

Representative top-level theorems include:

```text
minimum_recovery_is_minimal
aggregate_is_minimal_combined
healthy_trace_matches_aggregate
closed_loop_runtime_matches_selected
closed_loop_is_resolved
closed_loop_success_is_fresh
closed_loop_converges
```

Recovery actions are ordered from least to most disruptive:

```text
retry
< reResolve
< rebindRuntime
< reattachSession
< recreatePage
< recreateContext
< restartBrowser
< fail
```

The formal claim is about Driver behavior, not environmental liveness: if a sufficient repair succeeds within the finite budget, recovery returns healthy at a fresh generation; otherwise the Driver reaches explicit failure in finite internal steps.

## Driver-facing interaction evidence

The C++17 reference boundary lives under `driver/`:

```text
driver/include/browser/interaction_journal.hpp
driver/include/browser/interaction_boundary.hpp
driver/include/browser/jsonl_interaction_sink.hpp
driver/src/interaction_journal.cpp
driver/src/jsonl_interaction_sink.cpp
```

`InteractionBoundary` records typed interactions immediately before the real ownership/side-effect edge. The Lean journal verifier checks those observations without requiring production Driver components to serialize proof-specific JSON themselves.

## Integration/reference model

The larger `Browser/Async` model remains useful for regression evidence around:

- Browser / Context / Page / Frame / Action logical actors;
- NodeEpoch and ActionId generation isolation;
- asynchronous Policy command emission and later delivery;
- dynamic graph lifecycle;
- duplicate/collision behavior;
- Driver JSONL replay and conformance.

Historical whole-runtime proofs are intentionally kept out of `Browser.ProofAPI` and remain available through `Browser.Integration`.

## Verification

Lean is pinned to 4.33.0. CI runs the full proof and integration suite, including:

```text
lake build --wfail
public proof API boundary guard
lake exe browser-interaction-check
lake exe browser-recovery-check
lake exe browser-feedback-recovery-check
lake exe browser-decoder-recovery-check
lake exe browser-adversarial-recovery-check
lake exe browser-fault-aggregation-check
lake exe browser-trace-aggregation-check
lake exe browser-closed-loop-check
lake exe browser-driver-interaction-journal-check /tmp/driver-interaction-journal.jsonl
lake exe browser-driver-interaction-journal-check traces/driver-interaction-journal.jsonl
lake exe browser-interaction-trace-check traces/interaction-contracts.jsonl
lake exe browser-runtime-check
lake exe browser-driver-trace-check traces/generation-race.jsonl
lake exe browser-async-trace-check traces/async-runtime-race.jsonl
```

The workflow also compiles the C++ InteractionJournal, InteractionBoundary, and JSONL emitter smoke paths.

## Documentation

Start with `docs/README.md`. It is the canonical documentation index and explicitly separates **current architecture** from **historical/reference** design material.

For new formal work, the code-level source of truth is `Browser/ProofAPI.lean`; the final composition is `Browser/Interaction/ClosedLoop.lean`. Historical documents are retained for auditability and regression context, not as parallel architecture specifications.
