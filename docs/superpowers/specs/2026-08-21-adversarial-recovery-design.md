# Adversarial Recovery Design

## Goal

Prove that the compact Driver feedback/recovery controller remains safe under adversarial asynchronous observations: stale/future generations, duplicate faults, reordered faults, ambiguous decoder input, faults arriving during recovery, repeated recovery failure, and late callbacks after successful recovery.

## Scope

This layer does not add Browser/Page/Frame state. It composes the already-proven Decoder, Normalizer, Recovery, and generation contracts around a tiny recovery supervisor.

## Recovery join

Recovery actions are already totally ranked:

`retry < reResolve < rebindRuntime < reattachSession < recreatePage < recreateContext < restartBrowser < fail`.

Define `joinRecovery a b` as the stronger action by rank. This is the least action sufficient for both requirements. Prove:

- idempotence: `joinRecovery a a = a`
- commutativity: arrival order does not matter
- associativity: grouping does not matter
- upper-bound: the join is at least as strong as both inputs
- least-upper-bound: any action strong enough for both is at least as strong as the join

This makes concurrent recovery requirements a small join-semilattice. Duplicate faults do not restart recovery. Weaker later faults do not downgrade it. Stronger later faults escalate directly to the minimum action that covers both faults.

## Tagged observations

Every adversarial observation is tagged with the generation it belongs to:

```text
TaggedObservation
├── generation
└── AdapterEvent
```

The supervisor only mutates on the current generation:

- old generation -> stale/no-op
- future generation -> orphan/no-op
- current generation -> decode and process

A successful recovery increments the generation. Therefore every callback from the failed incarnation becomes stale immediately.

## Supervisor

The supervisor keeps only:

```text
FaultSupervisor
├── generation
├── recoveryBudget
├── activeRecovery? : RecoveryState
└── failed
```

When healthy and a recoverable fault arrives, it starts `beginRecovery` using the fault's `minimumRecovery`.

When already recovering and another current-generation fault arrives:

- `minimumRecovery = fail` -> fail safe immediately
- otherwise update the current action with `joinRecovery currentAction minimumRecovery(newFault)`
- do not reset the recovery budget
- do not create a parallel recovery

Non-fault feedback is left to the ordinary feedback controller and does not restart recovery.

Recovery feedback still uses the existing `stepRecovery`; repeated failure consumes budget, insufficient repair escalates, recovered creates a fresh generation, and fatal failure is terminal.

## Adversarial properties

Prove/execute:

1. stale observation does not mutate supervisor state
2. future observation does not mutate supervisor state
3. duplicate fault is idempotent while recovering
4. two faults merged in either arrival order select the same recovery action
5. weaker fault cannot downgrade active recovery
6. stronger fault escalates only to the least combined sufficient action
7. unknown/ambiguous decoder events fail safe
8. repeated recovery failure is budget-bounded
9. successful recovery increments generation
10. old-generation observation after recovery is stale/no-op
11. a fatal/unknown fault arriving during recovery cannot be hidden by a later recovered signal

## Outcome

Under arbitrary finite adversarial traces, the controller can only:

- make safe progress without recovery,
- recover into a fresh generation,
- or fail explicitly.

The model does not assert external systems recover; it proves internal recovery coordination remains safe, minimal, order-independent, and bounded.
