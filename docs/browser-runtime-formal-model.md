# Browser Runtime Formal Model

## Scope

This repository proves properties of the **normalized browser-driver runtime model**, not Chromium/CDP itself. Raw CDP and injected-page events must first be translated into `RuntimeEvent`; only `step` may mutate formal runtime state.

## Layers

1. `RuntimeGraph` models stable ownership/topology such as Page -> BrowserContext and Frame -> Page.
2. `PageState` models orthogonal lifecycle/runtime/input/action regions.
3. `RuntimeEvent` is the normalized event vocabulary.
4. `step` is the executable transition relation.
5. `Browser.Proofs` contains safety theorems over the same `step` function.
6. `Browser.Trace` replays concrete event streams through that specification.

## V1 safety properties

- Page-local transitions are isolated from sibling pages.
- Human input cannot silently coexist with automation ownership: it creates an explicit conflict and suspends the action.
- Destroyed execution contexts cannot continue primitive execution.
- Closed pages cannot continue primitive execution.
- Browser disconnect blocks execution for all descendant pages.

## Driver conformance contract

The future C/C++ driver should emit a normalized trace with stable IDs and causal metadata:

```text
CDP event / injected event
          |
          v
    RuntimeEvent
          |
          +----> runtime `step`
          |
          +----> telemetry/event journal
```

A conformance runner will replay that trace through the Lean model and reject any transition that cannot be produced by the specification. This complements theorem proving: Lean proves the model's invariants; trace replay checks that the implementation follows the model.

## Next proof layer

The next increment should add Context-level events and explicit Policy Commands so that cross-page effects are legal only through policy routing. Behavior-tree semantics should remain above primitive `ActionState` and consume proven-safe actions rather than bypassing them.
