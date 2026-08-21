# Browser Runtime Causality and Reaction Budget Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add explicit causal metadata to normalized runtime events and policy-issued commands, then prove that policy reactions preserve correlation and strictly consume a finite reaction budget.

**Architecture:** Keep `RuntimeEvent` as the semantic event payload. Add an `EventEnvelope` carrying stable event/correlation/parent/depth metadata. Policy responses become `IssuedCommand` values whose cause is derived from the triggering envelope. A bounded `reactEnvelope` accepts an event only when its depth is within budget and emits commands at depth+1, allowing Lean to prove correlation preservation, parent linkage, and bounded reaction chains.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, GitHub Actions.

**Spec:** `docs/browser-runtime-formal-model.md`

## Global Constraints

- Semantic state transitions remain defined by `RuntimeEvent`/`react`; causal metadata must not change state semantics.
- Policy-issued commands must preserve the trigger's `CorrelationId` and identify the triggering `EventId` as parent.
- Every policy reaction increments causal depth exactly once.
- A configurable finite `ReactionBudget` must reject envelopes deeper than the budget.
- Existing page/context isolation theorems must continue to compile unchanged.
- `lake build --wfail` and `lake exe browser-runtime-check` must pass.

---

### Task 1: Define causal envelopes

**Files:**
- Modify: `Browser/Types.lean`
- Create: `Browser/Causality.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `EventId`, `CorrelationId`, `Cause`, `EventEnvelope`, `ReactionBudget`, `withinBudget`.

- [ ] Add stable causal identifier aliases.
- [ ] Define envelope metadata independently of event semantics.
- [ ] Define finite budget checking.

### Task 2: Attach causality to policy commands

**Files:**
- Modify: `Browser/Command.lean`
- Modify: `Browser/Policy.lean`

**Interfaces:**
- Produces: `IssuedCommand`, `issueCommandsFor`, `childCause`.

- [ ] Wrap `PageCommand` with causal metadata.
- [ ] Derive command cause from trigger envelope with same correlation, parent=event id, depth+1.
- [ ] Preserve existing `commandsFor` semantic API for existing proofs.

### Task 3: Prove causal invariants and finite depth

**Files:**
- Modify: `Browser/Proofs.lean`

**Interfaces:**
- Produces theorems for correlation preservation, parent linkage, depth monotonicity, and over-budget rejection.

- [ ] Prove every issued policy command preserves correlation.
- [ ] Prove every issued policy command names the trigger as parent.
- [ ] Prove child depth is trigger depth + 1.
- [ ] Prove an event with depth greater than the finite budget cannot be accepted by bounded reaction.

### Task 4: Execute causal trace scenario

**Files:**
- Modify: `Browser/Trace.lean`
- Modify: `Main.lean`
- Modify: `README.md`
- Modify: `docs/browser-runtime-formal-model.md`

**Interfaces:**
- Produces executable causal checks alongside existing state safety checks.

- [ ] Add bounded envelope replay without changing semantic state results.
- [ ] Verify a context event issues only same-correlation child commands at depth+1.
- [ ] Verify an over-budget event is rejected.
- [ ] Run full CI verification.
