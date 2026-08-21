# Browser Runtime Formal Model

## Scope

This repository proves properties of the **normalized browser-driver runtime model**, not Chromium/CDP itself. Raw CDP and injected-page events are translated into `RuntimeEvent` before they can affect formal runtime state.

## Layers

1. `RuntimeGraph` models stable ownership/topology such as Page -> BrowserContext and Frame -> Page.
2. `PageState` models orthogonal lifecycle/runtime/input/action regions.
3. `EventScope` classifies events as Page, Context, or Browser scoped.
4. `step` records the normalized event without implicit sibling-page mutation.
5. `Policy.commandsFor` turns parent-scope events into explicit `PageCommand` values.
6. `react` composes `step` with `applyCommands` and is the executable runtime transition used by trace replay.
7. `Browser.Proofs` proves safety and propagation boundaries over the same functions.
8. `Browser.Trace` checks every state in concrete event streams with `traceSafe`.

## Safety properties

### Page-local safety

- Page-local transitions are isolated from sibling pages.
- Human input cannot silently coexist with automation ownership: it creates an explicit conflict and suspends the action.
- Destroyed execution contexts cannot continue primitive execution.
- Closed pages cannot continue primitive execution.

### Parent-state safety

- Browser disconnect blocks execution for all descendant pages.
- Context unavailability blocks execution for every page belonging to that context.

### Explicit propagation safety

- A `PageCommand` mutates only its target page.
- Commands emitted for `contextUnavailable C` target only pages whose `RuntimeGraph.contextOf` is `C`.
- A list of commands cannot mutate page Q if no command targets Q.
- Therefore `react` for Context A cannot mutate page state in unrelated Context B.

This establishes the intended rule:

```text
Page A event ----X----> Page B direct mutation

Page/Context event
       |
       v
      step
       |
       v
     Policy
       |
       v
 explicit PageCommand(target = Page B)
       |
       v
 Page B state transition
```

Sibling effects are not implicit state propagation. They are explicit policy decisions represented as data and therefore observable in traces.

## Executable conformance contract

The future C/C++ driver should emit normalized events with stable IDs and causal metadata:

```text
CDP event / injected event
          |
          v
    RuntimeEvent
          |
          v
        react
          |
          +----> runtime state
          +----> telemetry/event journal
```

`replay` processes the same event stream through `react`. `traceSafe` checks the invariant after every event, so a transient illegal state is rejected even if a later event would recover it.

The current executable scenario contains two pages in different contexts. Page 1 starts automation in Context 10; `contextUnavailable 10` suspends Page 1 through policy while Page 2 in Context 20 remains unchanged. The scenario also exercises Browser disconnect/reconnect and policy pause/resume.

## Next proof layer

The next increment should formalize causal metadata (`EventId`, `CorrelationId`, parent cause), policy-loop prevention, and action precondition invalidation. Behavior-tree semantics should remain above primitive `ActionState` and consume proven-safe actions rather than bypassing them.
