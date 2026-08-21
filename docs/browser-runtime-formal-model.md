# Browser Runtime Formal Model

## Scope

This repository proves properties of the **normalized browser-driver runtime model**, not Chromium/CDP itself. Raw CDP and injected-page events are translated into semantic `RuntimeEvent` values and wrapped in causal `EventEnvelope` metadata before entering the bounded reactive runtime.

## Layers

1. `RuntimeGraph` models stable ownership/topology such as Page -> BrowserContext and Frame -> Page.
2. `PageState` models orthogonal lifecycle/runtime/input/action regions.
3. `EventScope` classifies events as Page, Context, or Browser scoped.
4. `EventEnvelope` adds `EventId`, `CorrelationId`, parent event, and causal depth without changing event semantics.
5. `step` records the normalized event without implicit sibling-page mutation.
6. `Policy.commandsFor` turns parent-scope events into explicit `PageCommand` values.
7. `issueCommandsFor` gives policy output a derived child `Cause`.
8. `react` composes semantic state transition with policy commands.
9. `reactEnvelope` adds the finite `ReactionBudget` gate before `react`.
10. `Browser.Proofs` proves safety, propagation, and causality boundaries over these same functions.
11. `Browser.Trace` checks semantic and causal invariants on executable traces.

## Safety properties

### Page-local safety

- Page-local transitions are isolated from sibling pages.
- Human input cannot silently coexist with automation ownership: it creates an explicit conflict and suspends the action.
- Destroyed execution contexts cannot continue primitive execution.
- Closed pages cannot continue primitive execution.

### Parent-state and propagation safety

- Browser disconnect blocks execution for all descendant pages.
- Context unavailability blocks execution for every page belonging to that context.
- A `PageCommand` mutates only its target page.
- Commands emitted for `contextUnavailable C` target only pages whose `RuntimeGraph.contextOf` is `C`.
- A list of commands cannot mutate page Q if no command targets Q.
- Therefore `react` for Context A cannot mutate page state in unrelated Context B.

### Causal safety

For an event envelope `E`, every command produced by policy is wrapped as an `IssuedCommand` with `childCause E`. Lean proves:

```text
issued.correlation = E.correlation
issued.parent      = some E.id
issued.depth       = E.depth + 1
```

`ReactionBudget.maxDepth` is checked before semantic state mutation. Lean proves that if `budget.maxDepth < envelope.depth`, `reactEnvelope` returns `none`. This provides a formal barrier against unbounded policy self-reaction.

## Explicit response model

```text
Page A event ----X----> Page B direct mutation

EventEnvelope
      |
      v
 budget gate
      |
      v
     step
      |
      v
    Policy
      |
      +----> IssuedCommand(correlation, parent, depth+1)
      |
      v
 explicit PageCommand(target = Page B)
      |
      v
 Page B state transition
```

Sibling effects are not implicit state propagation. They are explicit policy decisions represented as data, causally linked to the source event, bounded in depth, and observable in traces.

## Executable conformance contract

The future C/C++ driver should emit normalized records with stable IDs and causal metadata. `replay` validates semantic events through `react`; `replayEnvelopes` validates causal events through `reactEnvelope`; `traceSafe` checks every intermediate semantic state; `causalPolicySafe` checks every issued command's correlation, parent, and depth.

The executable scenario currently validates:

- Context 10 can pause Page 1 while Page 2 in Context 20 remains unchanged.
- Browser disconnect/reconnect passes through the explicit policy path.
- A Context event at depth 0 is accepted by a depth-3 reaction budget.
- Its policy-issued commands preserve correlation, parent linkage, and depth+1.
- A Context event at depth 4 is rejected by that same budget before state mutation.

## Next proof layer

The next increment should formalize action dependency invalidation: a running primitive action declares required Page/Runtime/Input predicates, and state transitions invalidate/suspend the action when a dependency stops holding. Behavior-tree semantics should remain above primitive actions and consume those proven-safe action results rather than bypassing them.
