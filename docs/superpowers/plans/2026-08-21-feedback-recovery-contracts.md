# Feedback Recovery Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact Lean proof layer showing total feedback handling, minimal sufficient recovery, bounded recovery escalation, fresh-generation recovery, and explicit failure.

**Architecture:** Add `Feedback` and `Recovery` interaction modules that depend only on small protocol types. Do not import `Browser.Model` or `Browser.Async`. Add executable/spec scenarios, then compose the new recovery predicates into the existing interaction proof surface.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-21-feedback-recovery-contracts-design.md`

## Global Constraints

- Interaction contracts remain the primary proof surface.
- No new theorem may require the full Browser runtime product state.
- Unknown feedback must fail safe.
- A successful recovery creates a strictly newer generation.
- Recovery failure must consume budget, escalate, or terminate.
- Recovery optimality means minimum sufficient disruption, not minimum wall-clock latency.
- Existing interaction and async conformance checks must remain green.

---

### Task 1: Feedback decision contract

**Files:**
- Create: `Browser/Interaction/Feedback.lean`
- Create: `Browser/Interaction/FeedbackSpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `FeedbackClass`
- Produces: `Decision`
- Produces: `decideFeedback : FeedbackClass -> Decision`

- [ ] **Step 1: Write failing specs**

Create `FeedbackSpec.lean` importing the not-yet-existing `Browser.Interaction.Feedback` and assert all feedback constructors map to an explicit decision, including `unknown -> failSafe`.

- [ ] **Step 2: Verify RED**

Run the PR CI and require failure specifically because `Browser/Interaction/Feedback.lean` is absent.

- [ ] **Step 3: Implement minimal total decision function**

Implement the finite `FeedbackClass` and `Decision` enums plus total pattern matching in `decideFeedback`.

- [ ] **Step 4: Verify GREEN**

Require `lake build --wfail` and all existing executables to pass.

### Task 2: Recovery policy and optimality

**Files:**
- Create: `Browser/Interaction/Recovery.lean`
- Create: `Browser/Interaction/RecoverySpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `FaultClass`, `RecoveryAction`, `RecoveryStatus`, `RecoveryState`, `RecoveryFeedback`
- Produces: `recoveryRank`, `minimumRecovery`, `recoverySufficient`
- Produces: `beginRecovery`, `stepRecovery`

- [ ] **Step 1: Write failing recovery specs**

Assert examples for element/runtime/session/page/context/browser faults and general theorems that `minimumRecovery` is sufficient, lower-ranked actions are not sufficient, success increments generation, retry/escalation consumes budget, and exhausted/fatal recovery fails explicitly.

- [ ] **Step 2: Verify RED**

Require CI failure because `Browser.Interaction.Recovery` is absent.

- [ ] **Step 3: Implement minimal recovery protocol**

Use an explicit rank function and finite case analysis. Keep recovery state independent of Page/Frame/Browser implementation state.

- [ ] **Step 4: Verify GREEN**

Require Lean 4.33.0 `lake build --wfail` to pass with no warnings.

### Task 3: Bounded recovery replay and convergence scenario

**Files:**
- Create: `Browser/Interaction/RecoveryTrace.lean`
- Create: `Browser/Interaction/RecoveryTraceSpec.lean`
- Create: `RecoveryTraceMain.lean`
- Modify: `lakefile.toml`
- Modify: `.github/workflows/verify.yml`
- Modify: `README.md`

**Interfaces:**
- Produces: `replayRecovery : RecoveryState -> List RecoveryFeedback -> RecoveryState`
- Produces: `recoveryTerminal : RecoveryState -> Bool`
- Produces executable: `browser-recovery-check`

- [ ] **Step 1: Write failing trace spec**

Cover: session loss -> reattach -> recover; page loss -> insufficient repair -> escalate -> recover; stale old generation after recovery; repeated retryable failures -> explicit failure on budget exhaustion.

- [ ] **Step 2: Verify RED**

Require CI failure because `RecoveryTrace` / executable are absent.

- [ ] **Step 3: Implement replay and executable scenario**

Replay a finite feedback list. The executable must check one recoverable path and one exhausted path without reconstructing Browser runtime state.

- [ ] **Step 4: Run full verification**

Require C++ journal/boundary checks, `lake build --wfail`, interaction contracts, recovery executable, compact journal checks, runtime scenario, Driver trace, and unified async trace all to pass.

- [ ] **Step 5: Update PR description**

Describe Recovery Contract as a top-level proof surface and keep PR Draft.
