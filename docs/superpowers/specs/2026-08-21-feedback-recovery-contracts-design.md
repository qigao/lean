# Feedback and Recovery Contracts Design

## Goal

Prove that an asynchronous browser Driver can classify operational feedback, choose a safe minimal recovery, and converge to either a fresh healthy incarnation or an explicit failure without reconstructing the complete Browser runtime state.

## Principle

The formal unit is a feedback/recovery contract, not a Browser/Page/Frame product state.

```text
Operation
  -> Feedback
  -> semantic classification
  -> Decision
  -> bounded recovery if required
  -> Healthy(new generation) | ExplicitFailure
```

The environment is not assumed reliable. Recovery liveness is conditional: if a recovery action in the permitted escalation path eventually succeeds before budget exhaustion, the Driver must return to a healthy fresh generation. If no recovery succeeds, the Driver must fail explicitly in finite steps.

## Feedback vocabulary

`FeedbackClass` is deliberately small:

- `success`
- `transient`
- `stale`
- `conflict`
- `unavailable`
- `timeout`
- `terminal`
- `unknown`

Every normalized feedback value is handled by `decideFeedback`; there is no implicit unhandled branch. `unknown` is fail-safe.

## Decisions

`Decision` is also small:

- `complete`
- `wait`
- `retry`
- `recover RecoveryAction`
- `suspend`
- `failSafe`

A decision does not directly mutate Browser runtime state. It describes the next allowed interaction.

## Recovery actions and order

Recovery actions form an explicit least-to-most disruptive order:

```text
retry
reResolve
rebindRuntime
reattachSession
recreatePage
recreateContext
restartBrowser
fail
```

`recoveryRank` is strictly increasing along escalation. The policy never jumps to a more destructive action when a cheaper sufficient action exists.

Faults are represented by `FaultClass`:

- `elementStale`
- `runtimeLost`
- `sessionLost`
- `pageLost`
- `contextLost`
- `browserLost`
- `timeoutFault`
- `unknownFault`

`minimumRecovery` maps each fault to its least sufficient recovery. `recoverySufficient fault action` defines the admissible repair set. The main optimality property is:

```text
minimumRecovery fault is sufficient
and
no lower-ranked action is sufficient for fault.
```

## Recovery state

The recovery protocol keeps only:

```text
status       : healthy | recovering | failed
generation   : Nat
fault        : Option FaultClass
action       : RecoveryAction
budget       : Nat
```

Starting recovery does not reuse the failed incarnation. A successful recovery increments `generation`, clears the fault, and returns to `healthy`.

## Recovery feedback

Each recovery attempt receives one of:

- `recovered`
- `retryableFailure`
- `insufficientRepair`
- `fatalFailure`

Rules:

- `recovered`: enter `healthy`, increment generation, clear fault;
- `retryableFailure`: consume one budget unit and retry the same recovery action;
- `insufficientRepair`: consume one budget unit and escalate exactly one recovery level;
- `fatalFailure`: enter `failed` immediately;
- any failure with zero remaining budget: enter `failed`.

## Required properties

### Total feedback handling

Every `FeedbackClass` produces a `Decision`. `unknown` produces `failSafe`.

### Minimum sufficient recovery

For every `FaultClass`, `minimumRecovery` is sufficient and no lower-ranked recovery action is sufficient.

### Recovery preserves generation isolation

A successful recovery creates `generation + 1`; it never revives the failed generation. Therefore messages tagged with the previous generation are stale relative to the recovered state.

### Bounded escalation

Every nonterminal recovery failure either decreases `budget` or terminates. `insufficientRepair` also increases recovery rank. Therefore the internal recovery loop cannot remain forever in the same `(budget, action)` state.

### Recoverable fault convergence

Under the explicit environment assumption that a recovery attempt reports `recovered` before budget exhaustion, replay reaches `healthy` with a strictly newer generation.

### Explicit failure

If recovery budget is exhausted, or the environment reports `fatalFailure`, recovery reaches `failed`; it never remains silently `recovering`.

### Scope isolation

This contract is lane-local. Existing multi-lane `InteractionJournal` conformance remains responsible for showing that recovery/feedback in one lane does not mutate another lane.

## Integration with interaction contracts

Recovery composes with existing safety contracts rather than replacing them:

```text
Feedback/Recovery
   -> chooses next interaction
Interaction contracts
   -> authorize that interaction
InteractionJournal
   -> records what was actually attempted/delivered
```

Recovery does not bypass `GenerationIsolation`, `InputExclusivity`, `CausalIntegrity`, or `TerminalSafety`.

## Non-goals

- Proving Chromium or the network eventually recovers.
- Predicting real-world latency or globally fastest wall-clock recovery.
- Reconstructing Browser/Page/Frame state in Lean.
- Modeling every CDP error code directly.

The formal meaning of "optimal recovery" is minimal sufficient disruption, not minimum elapsed time.
