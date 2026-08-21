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
9. `ActionContract` declares the dependencies a running action requires.
10. `reconcileKnownActions` revalidates running primitive actions after a reaction.
11. `reactSafe` composes `react` with dependency reconciliation.
12. `reactEnvelopeSafe` adds the finite `ReactionBudget` gate before `reactSafe`.
13. `Browser.Proofs` proves safety, propagation, causality, and action-invalidation properties over these functions.
14. `Browser.Trace` checks semantic, causal, and action-contract invariants on executable traces.

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

`ReactionBudget.maxDepth` is checked before state mutation. An over-budget envelope returns `none`.

### Action dependency safety

Primitive input actions currently declare five dependencies:

```text
BrowserAvailable
ContextAvailable
PageReady
RuntimeReady
AutomationOwned
```

`requirementsHold` evaluates the contract. `reconcileAction` proves the local safety rule:

```text
action = Executing
AND contract = false
        |
        v
reconcileAction
        |
        v
action = Suspended
input  = Idle
```

The contract is intentionally a list of reusable `ActionRequirement` values so future actions can declare different dependency sets rather than hard-coding one global actionability rule.

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
    react
      |
      v
 Action dependency reconciliation
      |
      v
  reconciled runtime state
```

Sibling effects are explicit policy decisions represented as data, causally linked to the source event, bounded in depth, and observable in traces. Action invalidation is likewise explicit state, not merely a failed guard hidden behind `canExecute`.

## Executable conformance contract

The future C/C++ driver should emit normalized records with stable IDs and causal metadata. `replay` now executes `reactSafe`; `replayEnvelopes` executes `reactEnvelopeSafe`; `traceSafe` checks every intermediate reconciled state; `causalPolicySafe` checks every issued command's correlation, parent, and depth.

The executable scenario validates:

- Context 10 can pause Page 1 while Page 2 in Context 20 remains unchanged.
- Browser disconnect/reconnect passes through the explicit policy path.
- A Context event at depth 0 is accepted by a depth-3 reaction budget.
- Its policy-issued commands preserve correlation, parent linkage, and depth+1.
- A Context event at depth 4 is rejected before state mutation.
- If automation is requested after Browser disconnect, reconciliation forces Page 1 to `suspended`/`idle`.
- If automation is requested while Context 10 is unavailable, reconciliation forces Page 1 to `suspended` without changing Page 2 in Context 20.

## Next proof layer

The next increment should formalize higher-level action lifecycle/resume semantics and Behavior Tree consumption of action results. BT nodes should observe `running/suspended/success/failure` and action dependencies rather than bypassing the proven `reactSafe` boundary.
