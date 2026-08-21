# Browser Runtime Action Dependency Invalidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Formalize primitive-action dependency contracts and guarantee that an executing action is suspended whenever required Browser/Context/Page/Runtime/Input state stops holding.

**Architecture:** Define reusable `ActionRequirement` values and `ActionContract`s above the existing runtime state. `requirementsHold` evaluates a contract against a `Model`. A reconciliation layer runs after the existing `react` transition and suspends any executing primitive action whose contract is invalid. Existing event/policy semantics stay intact; trace replay moves to the reconciled reaction so implementation conformance observes the same safety rule.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, GitHub Actions.

**Spec:** `docs/browser-runtime-formal-model.md`

## Global Constraints

- Keep Action dependency semantics separate from CDP/event semantics.
- A contract may explicitly require Browser availability, Context availability, Page ready, Runtime ready, and Automation input ownership.
- Existing Page/Context isolation and causal theorems must continue to compile.
- Reconciliation may change only action/input state; it must not invent browser/page lifecycle state.
- An executing action with any failed declared dependency must become non-executing.
- Trace replay must validate the reconciled state after every event.
- `lake build --wfail` and `lake exe browser-runtime-check` must pass.

---

### Task 1: Define reusable action contracts

**Files:**
- Create: `Browser/Action.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `ActionRequirement`, `ActionContract`, `requirementHolds`, `requirementsHold`, `primitiveContract`.

- [ ] Model the five current primitive dependencies explicitly.
- [ ] Make contract checking executable and theorem-friendly.

### Task 2: Add action reconciliation

**Files:**
- Modify: `Browser/Action.lean`
- Modify: `Browser/Transition.lean`

**Interfaces:**
- Produces: `reconcileAction`, `reconcileKnownActions`, `reactSafe`, `reactEnvelopeSafe`.

- [ ] Suspend an executing action if its declared requirements are invalid.
- [ ] Reconcile all known pages after `react`.
- [ ] Apply the same reconciliation after bounded envelope reaction.

### Task 3: Prove invalidation safety

**Files:**
- Modify: `Browser/Proofs.lean`

**Interfaces:**
- Produces theorems for Browser/Context/Runtime dependency invalidation and reconciled execution safety.

- [ ] Prove a failed dependency cannot remain executing after reconciliation.
- [ ] Prove Browser disconnected prevents a newly requested action from remaining executing.
- [ ] Prove Context unavailable prevents a newly requested action from remaining executing.
- [ ] Prove destroyed Runtime context prevents a running action from remaining executing.

### Task 4: Execute dependency invalidation scenarios

**Files:**
- Modify: `Browser/Trace.lean`
- Modify: `Main.lean`
- Modify: `README.md`
- Modify: `docs/browser-runtime-formal-model.md`

**Interfaces:**
- Trace replay validates reconciled reactions.

- [ ] Add a scenario that attempts `automationStarted` while Browser is disconnected and verify it is suspended.
- [ ] Add a scenario that attempts `automationStarted` while its Context is unavailable and verify it is suspended.
- [ ] Keep causal budget scenarios passing under the safe envelope path.
- [ ] Run full CI verification.
