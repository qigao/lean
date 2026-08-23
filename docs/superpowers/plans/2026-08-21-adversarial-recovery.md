# Adversarial Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact adversarial recovery supervisor that proves duplicate/reordered faults merge optimally, stale/future generations are isolated, recovery remains bounded, and late callbacks cannot revive replaced generations.

**Architecture:** Reuse `Decoder`, `Normalizer`, `Recovery`, and `RecoveryTrace`. Add one small `AdversarialRecovery` module containing recovery-action join and a generation-tagged supervisor. Do not add Browser/Page/Frame state. Add an executable scenario and CI step.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, existing GitHub Actions workflow.

**Spec:** `docs/superpowers/specs/2026-08-21-adversarial-recovery-design.md`

## Global Constraints

- Keep interaction/recovery contracts as the primary proof surface.
- No new monolithic Browser runtime model.
- Unknown/ambiguous adapter events remain fail-safe.
- Recovery success always creates a fresh generation.
- Recovery loops remain finite through existing budget/fuel contracts.

---

### Task 1: Recovery join algebra

**Files:**
- Create: `Browser/Interaction/AdversarialRecovery.lean`
- Create: `Browser/Interaction/AdversarialRecoverySpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Consumes: `recoveryRank`, `minimumRecovery`, `RecoveryAction`.
- Produces: `joinRecovery`, idempotence/commutativity/associativity, upper-bound and least-upper-bound properties.

- [ ] **Step 1: Write failing spec** for join idempotence, commutativity, associativity, and representative recovery pairs.
- [ ] **Step 2: Run CI and verify RED** because `AdversarialRecovery.lean` does not exist.
- [ ] **Step 3: Implement minimal join algebra** using recovery rank.
- [ ] **Step 4: Run `lake build --wfail` via CI and verify GREEN.**
- [ ] **Step 5: Commit.**

### Task 2: Generation-tagged recovery supervisor

**Files:**
- Modify: `Browser/Interaction/AdversarialRecovery.lean`
- Modify: `Browser/Interaction/AdversarialRecoverySpec.lean`

**Interfaces:**
- Produces: `TaggedObservation`, `FaultSupervisor`, `observeAdversarial`, `applyRecoveryFeedback`.

- [ ] **Step 1: Add failing specs** proving stale/future no-op, duplicate fault idempotence, stronger-fault escalation, weaker-fault non-downgrade, and order-independent merge.
- [ ] **Step 2: Verify RED** against missing supervisor functions.
- [ ] **Step 3: Implement minimal supervisor** with one active `RecoveryState`; current-generation concurrent faults merge via `joinRecovery` without resetting budget.
- [ ] **Step 4: Verify GREEN** with `lake build --wfail` and existing suite.
- [ ] **Step 5: Commit.**

### Task 3: Recovery feedback and stale-after-recovery isolation

**Files:**
- Modify: `Browser/Interaction/AdversarialRecovery.lean`
- Modify: `Browser/Interaction/AdversarialRecoverySpec.lean`

**Interfaces:**
- Produces: feedback transition preserving terminality and fresh-generation isolation.

- [ ] **Step 1: Add failing specs** for repeated failure budget consumption, recovery success generation increment, old-generation event no-op after success, and fatal/unknown fault absorption.
- [ ] **Step 2: Verify RED.**
- [ ] **Step 3: Implement feedback transition by reusing `stepRecovery`.**
- [ ] **Step 4: Verify GREEN.**
- [ ] **Step 5: Commit.**

### Task 4: Executable adversarial scenarios and CI

**Files:**
- Create: `AdversarialRecoveryMain.lean`
- Modify: `lakefile.toml`
- Modify: `.github/workflows/verify.yml`
- Modify: `README.md`

**Interfaces:**
- Produces executable `browser-adversarial-recovery-check`.

- [ ] **Step 1: Add executable scenarios** for duplicate/reordered faults, page recovery followed by stale callback, repeated failure, human/transient input, and ambiguous decoder input.
- [ ] **Step 2: Add executable target and CI step.**
- [ ] **Step 3: Run full CI and require all prior C++/Lean integration checks to remain green.**
- [ ] **Step 4: Update README with the adversarial recovery semantics.**
- [ ] **Step 5: Commit and update Draft PR metadata only; do not merge.**
