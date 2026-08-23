> **Status: historical/reference plan.** This plan belongs to the earlier whole-runtime architecture and is retained for implementation history. New public proof work should start from `Browser.ProofAPI`; see `docs/README.md`.

# Browser Runtime Policy/Command Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the Lean browser-runtime model with scoped context events, explicit policy-generated page commands, and proofs that cross-page effects stay within the intended browser-context scope.

**Architecture:** Raw normalized events still enter through `step`, but context-level events only update parent availability state there. `Policy.commandsFor` converts a scoped event into explicit `PageCommand`s, and `react` applies those commands after `step`. Trace replay moves from `step` to `react`, so executable scenarios validate the same event→policy→command pipeline proved by Lean.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, GitHub Actions, no external Lean dependencies.

**Spec:** `docs/browser-runtime-formal-model.md`

## Global Constraints

- Keep `RuntimeGraph`, state, event, policy, command, transition, proof, and trace concerns in separate Lean modules.
- A page-local event may mutate only its source page.
- Cross-page mutation must occur only through explicit `PageCommand` values emitted by policy.
- Context-scoped policy commands may target only pages whose `contextOf` equals the event context.
- Browser disconnect remains a parent constraint that blocks execution globally.
- `lake build --wfail` and `lake exe browser-runtime-check` must pass.

---

### Task 1: Add event scope and context availability

**Files:**
- Modify: `Browser/Event.lean`
- Modify: `Browser/State.lean`

**Interfaces:**
- Produces: `EventScope`, `RuntimeEvent.contextUnavailable`, `RuntimeEvent.contextAvailable`, `scopeOf`, and `Model.contextAvailable`.

- [ ] Extend `RuntimeEvent` with context availability events and define `scopeOf`.
- [ ] Add `contextAvailable : ContextId → Bool` to `Model` and require it in `canExecute`.
- [ ] Build and confirm old proofs still compile after required updates.

### Task 2: Add explicit page commands and policy routing

**Files:**
- Create: `Browser/Command.lean`
- Create: `Browser/Policy.lean`
- Modify: `Browser/Transition.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `PageCommandKind`, `PageCommand`, `applyCommand`, `applyCommands`, `commandsFor`, and `react`.

- [ ] Define page-targeted commands for pausing and resuming automation.
- [ ] Implement functional command application so a command mutates only its target page.
- [ ] Generate context-scoped commands by filtering `knownPages` with `RuntimeGraph.contextOf`.
- [ ] Define `react m event := applyCommands (step m event) (commandsFor m event)`.

### Task 3: Prove scope and cross-page isolation

**Files:**
- Modify: `Browser/Proofs.lean`

**Interfaces:**
- Produces theorems that constrain command targeting and context propagation.

- [ ] Prove a single page command leaves every non-target page unchanged.
- [ ] Prove every command emitted for `contextUnavailable c` targets a page in context `c`.
- [ ] Prove `react` on `contextUnavailable c` leaves pages in other contexts unchanged.
- [ ] Prove `contextUnavailable c` blocks primitive execution for descendants in context `c`.

### Task 4: Extend executable trace verification

**Files:**
- Modify: `Browser/Trace.lean`
- Modify: `Main.lean`
- Modify: `README.md`
- Modify: `docs/browser-runtime-formal-model.md`

**Interfaces:**
- `replay` consumes `RuntimeEvent` using `react` rather than raw `step`.

- [ ] Replay events through `react`.
- [ ] Include context availability in executable safety checks.
- [ ] Add a scenario where context 10 pauses Page 1 without changing Page 2 in context 20.
- [ ] Document the event→state→policy→command pipeline and the new proofs.
- [ ] Run `lake build --wfail` and `lake exe browser-runtime-check`; GitHub Actions must be green.
