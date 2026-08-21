# Unified Async Browser Runtime Design

## Goal

Model the browser driver as an asynchronous message-processing runtime. Every externally observed or internally generated change is represented as a message delivery; cross-component synchronous mutation is not part of the formal semantics.

## Core rule

`deliver` is the only cross-component mutation boundary.

Existing `RuntimeEvent`, `step`, `Action` and `Policy` functions remain valid as actor-internal transition logic. They are not removed in the first migration. The new async layer wraps them and changes how effects move between actors.

## Actors

Logical actors are:

- Browser
- BrowserContext
- Page
- Frame
- Action(PageId, ActionId)

The Lean model does not simulate physical queues or threads. A mailbox is represented by the semantics of delivering one `AsyncEnvelope` to one actor at a time. This keeps proofs focused on observable ordering and isolation rather than scheduler implementation details.

## Identity and epochs

Every runtime-node target has a stable logical identity plus `NodeEpoch`.

- `PageId + NodeEpoch` identifies one Page incarnation.
- `FrameId + NodeEpoch` identifies one Frame incarnation.
- `PageId + PageEpoch + ActionId` identifies action-owned work.

A destroyed node retains its last epoch in the async runtime registry. Re-creation must advance the epoch. Messages carrying an old epoch are `stale` and cannot mutate current state.

## Envelope

```text
AsyncEnvelope
├── MessageId
├── Cause
│   ├── CorrelationId
│   ├── Parent MessageId?
│   └── depth
├── source : ActorAddress
├── target : ActorAddress
└── payload : AsyncPayload
```

`ActorAddress` contains an `ActorRef` and target `NodeEpoch`.

`AsyncPayload` initially supports:

- normalized `RuntimeEvent`
- explicit `PageCommand`
- dynamic graph lifecycle messages: Page create/destroy and Frame create/destroy

## Delivery disposition

Every delivery returns one disposition:

- `accepted`: target exists/current and transition ran
- `stale`: target epoch is old; state is unchanged
- `duplicate`: MessageId was already delivered; state is unchanged
- `orphan`: target is unknown or currently dead; state is unchanged
- `rejected`: causal/target/payload invariant failed; state is unchanged

Only `accepted` messages may mutate runtime state or emit child messages. `stale` messages may be recorded as observed deliveries for causal diagnostics but must not emit commands or state changes.

## Ordering model

There is no global happens-before relation across independent producers. The formal runtime only assumes:

1. actual delivery order supplied to `deliver`;
2. an accepted causal child references a previously observed parent;
3. correlation is preserved and causal depth is parent depth + 1;
4. node/action generation identity must match before actor-owned state may change.

Human input, CDP events/responses, page JS, network, timers, policy commands and action completion are therefore peers in the same asynchronous model.

## Policy becomes asynchronous

Before:

```text
Event -> Policy -> apply PageCommand synchronously
```

After:

```text
RuntimeEvent delivered to Browser/Context/Page
    -> actor-local state transition
    -> Policy computes PageCommand(s)
    -> emit AsyncEnvelope child message(s)
    -> later deliver command to target Page
    -> Page state transition
```

A Browser/Context parent can immediately constrain `canExecute` through parent state, but child `PageState` is not synchronously rewritten merely because policy produced a command.

## Dynamic RuntimeGraph

Graph topology changes are also messages.

```text
pageCreated(context, page, newEpoch)
pageDestroyed(page)
frameCreated(page, frame, newEpoch)
frameDestroyed(frame)
```

Creation validates parent actor identity and requires the new child epoch to advance monotonically. Destruction marks the node dead without resetting its epoch. An old message from a prior incarnation is therefore distinguishable from an event targeting a never-known node.

## Compatibility strategy

The first migration is additive:

- keep `RuntimeEvent`, `step`, `commandsFor`, `applyCommand` and existing proofs;
- add `AsyncRuntime`, `AsyncEnvelope`, `deliver` and dynamic graph helpers;
- call `step` only for an accepted runtime-event message targeted at the correct actor;
- emit policy commands instead of applying them from the async layer;
- call `applyCommand` only when a command message is later delivered to the target Page;
- keep existing JSONL trace support, then add an async trace vocabulary.

This avoids discarding already-proven page/action semantics while making the outer runtime accurately asynchronous.

## Required safety properties

The first async proof layer must establish:

- `stale_message_noop`
- `duplicate_delivery_noop`
- `orphan_delivery_noop`
- `old_node_epoch_cannot_mutate_new_node`
- `old_action_cannot_mutate_new_action` (existing ActionId theorem remains compatible)
- `page_delivery_isolated_from_other_pages`
- `policy_command_not_applied_before_delivery`
- `accepted_child_has_observed_parent`
- `delivery_preserves_graph_wellformed`

Executable scenarios must cover:

- duplicate delivery of the same MessageId;
- old Page epoch message after Page re-creation;
- stale ActionId timer/CDP callback inside the current Page epoch;
- context event emits Page commands while Page state remains unchanged until each command delivery;
- popup/new Page and Frame creation as asynchronous graph messages.

## C/C++ contract

The future driver event journal should emit the same identities directly:

```text
MessageId
CorrelationId
ParentMessageId
Source actor + epoch
Target actor + epoch
PageId
FrameId
ActionId
CdpCallId
payload
```

A synchronous-looking user API such as `await page.click()` is only a facade: internally it emits an action-start message and resolves its future when an action-completion message is delivered.
