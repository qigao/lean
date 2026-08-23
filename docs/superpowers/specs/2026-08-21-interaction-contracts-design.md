# Browser Interaction Contracts Design

## Goal

Reduce formal proof complexity by proving the safety contracts on browser-runtime interaction edges rather than re-proving the complete `Model`/`AsyncRuntime` implementation state.

The existing async runtime remains an executable reference/conformance model. A new small `Browser.Interaction` layer extracts the dangerous boundaries into independent protocols with tiny state spaces and composable guarantees.

## Principle

Lean should prove **what one component is allowed to do to another**, not duplicate every internal C/C++ Driver state.

The proof unit is an interaction contract:

```text
Assume(input/identity/ownership preconditions)
        ↓
Protocol transition
        ↓
Guarantee(effect confinement / stale no-op / exclusivity / terminal safety)
```

Full Driver conformance then projects journal entries into these interaction protocols.

## Interaction graph

```text
Context ──availability──> Page
Page    ──owns──────────> Frame
Page    ──owns──────────> Action
Action  ──request───────> CDP
CDP     ──response──────> Action
Timer   ──expiry────────> Action
Human   ──input─────────> InputOwner
Action  ──input─────────> InputOwner
Policy  ──command───────> Page
```

No implicit sibling mutation edge exists. A cross-page effect must be represented by an explicit policy-command interaction.

## Protocols

### 1. CDP response protocol

Minimal state:

```lean
structure CdpProtocol where
  currentAction : Option ActionId
  pending : Option (ActionId × CdpCallId)
```

Guarantees:
- only the exact `(ActionId, CdpCallId)` pair may consume a pending call;
- response from an old ActionId is a no-op;
- response with a wrong CallId is a no-op.

### 2. Deadline protocol

Minimal state:

```lean
structure DeadlineProtocol where
  currentAction : Option ActionId
  deadline : Option (ActionId × Time)
  timedOut : Bool
```

Guarantees:
- a timer owned by another ActionId cannot time out the current action;
- expiry before deadline is a no-op;
- a matching expired timer makes the action terminal.

### 3. Input ownership protocol

Minimal state:

```text
none | human | automation(ActionId) | conflict(ActionId)
```

Guarantees:
- Human and Automation cannot both hold dispatch ownership;
- human input while automation owns the input creates an explicit conflict;
- automation dispatch is forbidden while human/conflict owns the input.

### 4. Lifecycle/Epoch protocol

Minimal state:

```lean
structure EpochSlot where
  epoch : NodeEpoch
  alive : Bool
```

Guarantees:
- messages for an older epoch cannot mutate the current incarnation;
- future/unknown epochs cannot mutate the current incarnation;
- recreation advances epoch monotonically.

### 5. Context isolation protocol

Minimal state is Page -> Context ownership plus an effect target.

Guarantee:
- a Context-scoped effect can target only Pages owned by that Context.

### 6. Policy propagation protocol

Minimal state is a triggering interaction plus emitted command identities.

Guarantees:
- policy evaluation itself does not mutate the child Page;
- cross-actor Page mutation requires delivery of an emitted command;
- command inherits correlation and parent causality.

### 7. Graph ownership protocol

Minimal state contains ownership bindings only:

```text
Page -> Context
Frame -> Page
```

Guarantees:
- a Frame effect is authorized only through its owning Page;
- a Page effect is authorized only through its owning Context where required;
- there is no direct Page-A -> Page-B ownership path.

### 8. Terminal-effect protocol

Minimal state:

```text
live | cancelled | timedOut | destroyed
```

Guarantee:
- terminal actors cannot produce external side effects (CDP request, input dispatch, network mutation command).

## Composition model

The interaction layer defines a small common vocabulary:

```text
Interaction
├── source role
├── target role
├── correlation
├── generation/epoch ownership (optional)
└── effect kind
```

System-level proofs target five properties:

1. **AuthorizedEffect** — every side effect follows a legal interaction edge.
2. **GenerationIsolation** — stale NodeEpoch/ActionId interactions cannot mutate current state.
3. **InputExclusivity** — one Page never grants dispatch ownership to Human and Automation simultaneously.
4. **CausalIntegrity** — emitted reactions preserve correlation and parent linkage.
5. **TerminalSafety** — terminal actors cannot emit external side effects.

These properties compose independent protocol lemmas; they do not depend on the full browser state cross-product.

## Relationship to existing AsyncRuntime

`Browser.Async` remains useful as an executable integration/reference model. It is no longer the primary proof surface.

```text
existing AsyncRuntime
       │ project
       ▼
Interaction trace
       │
       ├── CdpProtocol
       ├── DeadlineProtocol
       ├── InputProtocol
       ├── EpochProtocol
       ├── ContextProtocol
       ├── PolicyProtocol
       ├── GraphProtocol
       └── TerminalProtocol
```

Old end-to-end theorems may remain as regression checks, but new correctness arguments should be expressed against the smallest relevant interaction protocol.

## Driver journal target

The C/C++ Driver should eventually emit interaction-oriented records, e.g.:

```text
ActionStarted(page=1, action=7)
CdpRequest(action=7, call=31)
CdpResponse(action=7, call=31)
HumanInput(page=1)
TimerExpired(action=6)
PolicyCommandIssued(parent=42, page=1)
PolicyCommandDelivered(command=43, page=1)
```

The verifier does not need a serialized copy of every internal Page/Graph field. It needs enough identity and causality data to validate the interaction contracts.

## Non-goals

- Proving Chromium/CDP implementation correctness.
- Proving every internal C/C++ object representation.
- Encoding every Page/Action state combination into one monolithic theorem.
- Imposing a global total order on independent async sources.
