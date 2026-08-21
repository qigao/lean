# Feedback Normalizer Design

## Goal

Complete the Browser Driver feedback-control proof chain by normalizing typed raw observations into the compact `FeedbackClass + FaultClass` vocabulary already used by decision and recovery contracts.

The normalizer does not model Browser/Page/Frame/Action state. Identity/generation/ownership remain the responsibility of existing interaction contracts.

## Pipeline

```text
RawObservation
    ↓ normalizeObservation
NormalizedFeedback
├── feedback : FeedbackClass
└── fault    : Option FaultClass
    ↓ decideNormalized
Decision
    ↓ when fault exists
beginRecovery
    ↓ runRecoveryFuel
Healthy(new generation) | Failed
```

## Raw observation vocabulary

The adapter-facing vocabulary is intentionally semantic and finite:

- `operationSucceeded`
- `conditionPending`
- `navigationStarted`
- `networkTransient`
- `protocolTransient`
- `humanInput`
- `elementDetached`
- `executionContextDestroyed`
- `frameDetached`
- `sessionDetached`
- `pageClosed`
- `pageCrashed`
- `contextDestroyed`
- `browserExited`
- `deadlineReached`
- `unrecognized`

Raw protocol strings/codes are decoded into these values outside the proof core. Unknown decoder output must become `unrecognized`, never be dropped.

## Normalization

Representative mappings:

```text
operationSucceeded          -> success / none
conditionPending            -> transient / none
navigationStarted           -> transient / none
networkTransient            -> transient / none
protocolTransient           -> transient / none
humanInput                  -> conflict / none
elementDetached             -> stale / elementStale
executionContextDestroyed   -> stale / runtimeLost
frameDetached               -> stale / runtimeLost
sessionDetached             -> unavailable / sessionLost
pageClosed                  -> terminal / pageLost
pageCrashed                 -> terminal / pageLost
contextDestroyed            -> terminal / contextLost
browserExited               -> terminal / browserLost
deadlineReached             -> timeout / timeoutFault
unrecognized                -> unknown / unknownFault
```

## Decision rule

`FeedbackClass` alone is deliberately too coarse to choose recovery scope. If a normalized observation contains a `FaultClass`, the fault-specific `minimumRecovery` wins. If the minimum recovery is `fail`, the decision is `failSafe`. Only observations without a fault use `decideFeedback` directly.

Therefore `elementDetached` chooses `reResolve`, while `executionContextDestroyed` chooses `rebindRuntime` even though both are stale-style observations.

## Coherence

Define a small `normalizationCoherent` predicate describing allowed `(FeedbackClass, FaultClass?)` pairs. Prove every `RawObservation` normalizes coherently.

This catches category/fault contradictions without depending on Browser runtime state.

## Recovery composition

For fault-bearing observations, `recoverObservationFuel` composes:

```text
normalizeObservation
→ FaultClass
→ beginRecovery generation fault budget
→ runRecoveryFuel fuel available
```

The existing theorem `run_recovery_fuel_is_terminal` then gives the end-to-end property:

- if an attempted repair becomes available before bounds are exhausted, recovery can return `healthy` with a fresh generation;
- otherwise the bounded computation returns `failed`;
- timeout/unknown faults begin failed and cannot be revived.

## Required properties

1. `normalizeObservation` is total by exhaustive construction.
2. `normalization_is_coherent` for every raw observation.
3. No-fault observations use `decideFeedback` unchanged.
4. Fault-bearing observations use exactly `minimumRecovery`, unless it is `fail`, in which case they fail safe.
5. `unrecognized` normalizes to `unknown / unknownFault` and fails safe.
6. `deadlineReached` normalizes to `timeout / timeoutFault` and fails safe.
7. `recoverObservationFuel` is terminal for any finite fuel, budget, generation, and environment availability function.
8. A successful fault recovery creates a fresh generation through the existing Recovery contract.

## Non-goals

- Decoding every CDP error string or Chromium version-specific code inside Lean.
- Proving external dependencies eventually recover.
- Reintroducing full Browser/Page state into normalization.
- Replacing interaction identity/generation checks with feedback classification.
