# Browser Runtime Formal Model

Lean 4 executable specification for a stateful CDP browser driver/runtime.

The model separates these concerns:

- **Runtime graph**: ownership/topology (`Browser -> Context -> Page -> Frame`).
- **State machines**: mutable page/runtime/input/action state.
- **Event scope**: page, browser-context, or browser.
- **Policy + commands**: the only legal route for cross-page reactions.
- **Causality + reaction budget**: stable correlation/parent/depth metadata and bounded response chains.
- **Action contracts**: explicit runtime dependencies plus post-event reconciliation.
- **Timeouts**: monotonic absolute deadlines with human/page/network/protocol causes.
- **Action generations**: stale timer, CDP response, and completion callbacks are rejected by `ActionId`.
- **Proofs + executable replay**: static safety theorems and concrete trace checking.

## Proven properties

The model covers page/context isolation, human/automation conflict handling,
parent/runtime invalidation, explicit policy routing, causal invariants and finite
reaction depth. Timeout and async-race properties additionally require:

1. Human/page/network/protocol waiting never extends an armed absolute deadline.
2. Timeout is terminal under generic pause/resume and page lifecycle changes.
3. Starting fresh work increments a page-local `ActionId` generation.
4. Deadlines and protocol waits are bound to the `ActionId` that created them.
5. A deadline callback carrying a stale ActionId is a no-op.
6. A CDP/protocol response carrying a stale ActionId is a no-op.
7. A completion callback carrying a stale ActionId cannot finish a newer action.
8. Sibling Page generations and async state remain isolated.

These are properties of the abstract driver model. They do **not** by themselves
prove a future C/C++ CDP driver implementation correct. The conformance layer is
trace based: the driver emits normalized events/actions and CI replays them
through this executable model.

## Async generation model

```text
Action A / id=1
  |-- timer(action=1)
  |-- CDP call(action=1, call=100)
  `-- finish(action=1)

Action B / id=2 starts

late timer(action=1)       -> ignored
late response(action=1)    -> ignored
late finish(action=1)      -> ignored
response(action=2)         -> may affect Action B
```

This prevents a timed-out or completed operation from later mutating a newer
browser operation when an OS timer, network response, renderer event, or CDP
response arrives late.

## Build

```bash
lake build --wfail
lake exe browser-runtime-check
```

The project is pinned to Lean 4.33.0. GitHub Actions builds the complete model
with `--wfail` and runs the executable verifier.

## Driver integration target

The C/C++ driver should normalize raw CDP/injected/timer events into the formal
vocabulary before runtime state mutation. Action-owned asynchronous work must
carry `PageId + ActionId`; protocol work additionally carries `CdpCallId`.
Cross-page effects remain explicit policy commands. Implementation traces will
be replayed through the same Lean transition functions for conformance checking.
