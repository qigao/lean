# Decoder Contract Design

## Goal

Prove the last adapter boundary between actual Browser/CDP/DOM/Network/Human observations and the already-proven `RawObservation -> NormalizedFeedback -> Decision -> Recovery` control loop.

The decoder must be total and fail-safe: every adapter observation yields exactly one `RawObservation`; information that is unknown, malformed, version-specific, or insufficiently scoped yields `RawObservation.unrecognized` rather than being dropped or guessed.

## Boundary

The proof core does not parse arbitrary Chromium JSON or depend on unstable protocol payload layouts. A thin adapter first converts wire JSON into a stable `AdapterEvent`. This contract proves the stable adapter event is decoded correctly into `RawObservation`.

```text
CDP / DOM / Network / Human / Timer / Driver
                  |
          wire-specific parser
                  |
             AdapterEvent
                  |
             decodeEvent
                  |
           RawObservation
                  |
        normalizeObservation
                  |
             Decision
                  |
              Recovery
```

## Adapter event vocabulary

`AdapterEvent` distinguishes the cases where method name alone is insufficient:

- Runtime execution context destroyed / all contexts cleared
- Page frame detached
- Target session detached
- Target crashed / destroyed with resolved target scope
- Inspector detached
- Network loading failed with resolved failure disposition
- Protocol command failed with resolved failure disposition
- DOM element detached
- Human input
- Deadline reached
- Driver browser/context lifecycle observations
- operation success / condition pending / navigation started
- malformed or unknown signal

`Target.targetDestroyed` must include a resolved target scope before it may become `pageClosed`; otherwise it is `unrecognized`. `Network.loadingFailed` must include a resolved failure disposition; transient/cancelled cases may wait, while blocked/CORS/unknown cases fall back to `unrecognized` because the recovery policy cannot safely infer a transient fault.

## Safety properties

1. **Decoder totality**: every `AdapterEvent` returns a `RawObservation`.
2. **Known mapping correctness**: critical CDP lifecycle events map to the intended raw observation.
3. **Scope safety**: a non-page target crash/destroy is not promoted to a page fault.
4. **Failure-disposition safety**: ambiguous network/protocol errors do not get optimistic retry semantics.
5. **Unknown fallback**: malformed/future adapter input decodes to `unrecognized`.
6. **End-to-end fail-safe**: unknown/malformed decoder input therefore reaches `unknownFault -> failSafe -> Failed` through the existing normalizer/recovery contracts.

## Non-goals

- Proving Chromium itself emits correct events.
- Proving arbitrary JSON parsing libraries.
- Enumerating every CDP version-specific error string.
- Reconstructing Browser/Page/Action state inside the decoder.

The real implementation may have version-specific adapters, but they must terminate at this stable contract surface.