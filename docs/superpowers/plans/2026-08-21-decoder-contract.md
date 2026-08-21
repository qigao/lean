# Decoder Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a total, fail-safe adapter decoder from stable Browser/CDP/DOM/Network/Human observations into the existing `RawObservation -> NormalizedFeedback -> Recovery` proof chain.

**Architecture:** Keep wire parsing/version details outside the proof core. `Browser.Interaction.Decoder` defines a small structured `AdapterEvent`, decodes it to `RawObservation`, and proves scope/failure ambiguity falls back to `unrecognized`. Existing `Normalizer` and `FeedbackRecovery` are reused unchanged for policy and convergence.

**Tech Stack:** Lean 4.33.0, Lake 5.0.0, existing Browser.Interaction contracts.

**Spec:** `docs/superpowers/specs/2026-08-21-decoder-contract-design.md`

## Global Constraints

- Do not reintroduce complete Browser/Page/Action state into the decoder.
- Unknown, malformed, under-scoped, or ambiguous input must become `RawObservation.unrecognized`.
- Decoder must not choose recovery directly.
- Reuse existing normalizer/recovery theorems for end-to-end convergence.
- CI uses `lake build --wfail`; no Lean warnings are allowed.

---

### Task 1: Decoder semantics

**Files:**
- Create: `Browser/Interaction/DecoderSpec.lean`
- Create after RED: `Browser/Interaction/Decoder.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Consumes: `RawObservation` from `Browser.Interaction.Normalizer`.
- Produces: `AdapterEvent`, `decodeEvent : AdapterEvent -> RawObservation` and decoder safety theorems.

- [ ] **Step 1: Write the failing spec** covering execution-context destruction, frame detach, session detach, page crash/destroy scope, network/protocol disposition, human/timer signals, and unknown/malformed fallback.
- [ ] **Step 2: Run CI and verify RED** is caused by missing `Browser.Interaction.Decoder`.
- [ ] **Step 3: Implement minimal decoder** with structured scope and failure disposition; no Browser runtime state.
- [ ] **Step 4: Verify GREEN** with `lake build --wfail` and all existing CI checks.

### Task 2: End-to-end decoder recovery composition

**Files:**
- Create: `Browser/Interaction/DecoderRecoverySpec.lean`
- Create: `DecoderRecoveryMain.lean`
- Modify: `Browser.lean`
- Modify: `lakefile.toml`
- Modify: `.github/workflows/verify.yml`

**Interfaces:**
- Consumes: `decodeEvent`, `normalizeObservation`, `decideNormalized`, `recoverObservationFuel`.
- Produces: executable `browser-decoder-recovery-check`.

- [ ] **Step 1: Add RED spec** proving decoded page crash can recover as a fresh generation while malformed/unknown input fails safe.
- [ ] **Step 2: Compose decoder with existing recovery functions** without adding a second recovery state machine.
- [ ] **Step 3: Add executable scenarios** for known recoverable input, ambiguous input, and unknown input.
- [ ] **Step 4: Add CI command** `lake exe browser-decoder-recovery-check`.
- [ ] **Step 5: Run latest-head full CI** and require all existing and new checks to succeed.

### Task 3: Documentation and PR metadata

**Files:**
- Modify: `README.md` only if new content is materially needed.
- Update PR #1 metadata with GitHub PR API; do not create no-op file commits.

- [ ] **Step 1: Document the final wire-adapter boundary and fail-safe fallback.**
- [ ] **Step 2: Verify the final file-changing head with full CI.**
- [ ] **Step 3: Update Draft PR description via PR metadata only; do not merge.**
