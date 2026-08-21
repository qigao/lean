# Browser Runtime Formal Model

Lean 4 model for a stateful CDP browser driver/runtime.

The model separates four concerns:

- **Runtime graph**: ownership/topology (`Browser -> Context -> Page -> Frame`).
- **State machines**: mutable page/runtime/input/action state.
- **Events/transitions**: the only legal path for state mutation.
- **Proofs + executable replay**: static safety theorems and concrete trace checking.

## Proven in v1

The first model proves that:

1. A page-local event cannot mutate a sibling page.
2. Human input while automation owns a page enters `conflict` and suspends the action.
3. Destroying an execution context makes continued primitive execution impossible.
4. Closing a page makes continued primitive execution impossible.
5. Browser disconnect makes primitive execution impossible on every page.

These are properties of the abstract driver model. They do **not** by themselves prove a future C/C++ CDP driver implementation correct. The intended next layer is conformance: the driver emits normalized event/action traces, and CI replays those traces through this executable model.

## Build

```bash
lake build
lake exe browser-runtime-check
```

The project is pinned to Lean 4.33.0. GitHub Actions also runs `nanoda` with `sorry` disallowed.

## Driver integration target

The runtime driver should normalize raw CDP/injected events into the model vocabulary before state mutation:

```text
CDP / injected observer
        |
        v
 normalized RuntimeEvent
        |
        v
       step
        |
        +--> Runtime graph/state
        +--> Policy / ActionSM / BehaviorTree
        +--> telemetry trace
```

Cross-page effects should be explicit policy commands rather than hidden sibling-state mutation.
