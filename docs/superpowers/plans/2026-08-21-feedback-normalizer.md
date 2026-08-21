# Feedback Normalizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact raw-observation normalizer and compose fault-bearing observations into the existing bounded Recovery contract.

**Architecture:** `Normalizer.lean` owns only raw observation classification and normalized decision selection. `FeedbackRecovery.lean` composes normalized faults with existing `beginRecovery` and `runRecoveryFuel`; it does not introduce Browser/Page/Action runtime state.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-21-feedback-normalizer-design.md`

## Global Constraints

- Do not import or reconstruct the full Browser runtime model in the normalizer.
- Unknown raw observations must normalize to `unknown / unknownFault` and fail safe.
- Fault-specific `minimumRecovery` takes precedence over coarse `FeedbackClass` policy.
- Recovery remains bounded by existing fuel/budget contracts.
- Existing interaction, journal, runtime, driver-trace, and async checks must remain green.

---

### Task 1: Raw observation normalization

**Files:**
- Create: `Browser/Interaction/NormalizerSpec.lean`
- Create: `Browser/Interaction/Normalizer.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `RawObservation`
- Produces: `NormalizedFeedback`
- Produces: `normalizeObservation : RawObservation → NormalizedFeedback`
- Produces: `normalizationCoherent : NormalizedFeedback → Bool`
- Produces: `decideNormalized : NormalizedFeedback → Decision`

- [ ] **Step 1: Write the failing spec**

Create examples covering every `RawObservation`, plus explicit checks that `elementDetached` chooses `reResolve`, `executionContextDestroyed` chooses `rebindRuntime`, `deadlineReached` fails safe, and `unrecognized` fails safe.

- [ ] **Step 2: Run CI and confirm RED**

Expected: `lake build --wfail` fails because `Browser.Interaction.Normalizer` does not exist.

- [ ] **Step 3: Implement the minimal normalizer**

Implement exhaustive raw-to-normalized mapping, coherence predicate, `normalization_is_coherent`, and fault-aware `decideNormalized`.

- [ ] **Step 4: Run full CI and require GREEN**

Expected: build and all prior checks pass.

### Task 2: Fault normalization to bounded recovery

**Files:**
- Create: `Browser/Interaction/FeedbackRecoverySpec.lean`
- Create: `Browser/Interaction/FeedbackRecovery.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `recoverObservationFuel : Nat → Nat → Nat → (RecoveryAction → Bool) → RawObservation → Option RecoveryState`
- Produces: theorem that every fault-bearing observation returns a terminal RecoveryState.

- [ ] **Step 1: Write the failing spec**

Cover `pageCrashed` escalating to a later available repair and returning `healthy` with generation+1; cover `browserExited` with no available repair returning `failed`; cover `deadlineReached` and `unrecognized` remaining failed.

- [ ] **Step 2: Run CI and confirm RED**

Expected: missing `Browser.Interaction.FeedbackRecovery`.

- [ ] **Step 3: Implement the minimal composition**

Normalize the observation, return `none` for no-fault feedback, and otherwise call `runRecoveryFuel fuel (beginRecovery generation fault budget) available`.

- [ ] **Step 4: Prove terminality**

Use `run_recovery_fuel_is_terminal`; do not duplicate its induction.

- [ ] **Step 5: Run full CI and require GREEN**

Expected: all previous checks still pass.

### Task 3: Executable normalization/recovery scenario

**Files:**
- Create: `FeedbackRecoveryMain.lean`
- Modify: `lakefile.toml`
- Modify: `.github/workflows/verify.yml`
- Modify: `README.md`

**Interfaces:**
- Produces executable: `browser-feedback-recovery-check`.

- [ ] **Step 1: Add executable scenario**

Exercise raw observation → normalized decision → bounded recovery for recoverable, unrecoverable, timeout, unknown, conflict, and transient observations.

- [ ] **Step 2: Add CI step**

Run `lake exe browser-feedback-recovery-check` after `browser-recovery-check`.

- [ ] **Step 3: Update README**

Document the final proof chain: `RawObservation -> NormalizedFeedback -> Decision -> Recovery -> Healthy(new generation) | Failed`.

- [ ] **Step 4: Run latest-head full CI**

Require the full workflow to complete with `conclusion: success` before updating PR metadata.
