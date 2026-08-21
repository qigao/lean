# Interaction Contracts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the browser Driver's dangerous cross-component interactions into small Lean protocols and prove their contracts independently from the full runtime model.

**Architecture:** Add `Browser/Interaction/` as a proof-focused layer with independent protocol state machines for CDP, deadlines, input ownership, epochs, context isolation, policy propagation, graph ownership and terminal effects. Reuse existing identifier types, but avoid importing `Browser.Async` except in a thin projection/conformance adapter.

**Tech Stack:** Lean 4.33.0, Lake, existing Browser formal model.

**Spec:** `docs/superpowers/specs/2026-08-21-interaction-contracts-design.md`

## Global Constraints

- Protocol theorem state must be minimal and independent of unrelated Page/Graph/Action fields.
- Existing `Browser.Async` remains an executable reference model and must continue to build.
- New correctness proofs should target protocol contracts rather than the complete runtime state cross-product.
- No global ordering assumption beyond explicit causal linkage or protocol-local ownership.
- CI remains `lake build --wfail` plus existing executable conformance checks.

---

### Task 1: Core interaction vocabulary and CDP/Deadline protocols

**Files:**
- Create: `Browser/Interaction/Types.lean`
- Create: `Browser/Interaction/Cdp.lean`
- Create: `Browser/Interaction/Deadline.lean`
- Create: `Browser/Interaction/CoreSpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces `InteractionRole`, `ExternalEffect`, `CdpProtocol`, `DeadlineProtocol`.
- Produces `receiveCdpResponse`, `expireDeadline` and local safety theorems.

- [ ] Add examples asserting stale ActionId CDP response is a no-op, wrong CallId is a no-op, stale timer is a no-op, early timer is a no-op, and matching expired timer becomes terminal.
- [ ] Run `lake build --wfail`; expected RED because protocol modules do not exist.
- [ ] Implement the minimal protocol states/transitions with no dependency on `Model` or `AsyncRuntime`.
- [ ] Prove the examples/theorems and rerun `lake build --wfail`.
- [ ] Commit `feat: extract cdp and deadline interaction contracts`.

### Task 2: Input ownership and terminal-effect protocols

**Files:**
- Create: `Browser/Interaction/Input.lean`
- Create: `Browser/Interaction/Terminal.lean`
- Create: `Browser/Interaction/SafetySpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces `InputOwner`, `onHumanInput`, `canAutomationDispatch`.
- Produces `ActorLiveness`, `mayEmitExternalEffect`.

- [ ] Add examples for automation->human conflict, dispatch denial under Human/Conflict, and external-effect denial for cancelled/timedOut/destroyed actors.
- [ ] Run build for RED.
- [ ] Implement minimal protocols and proofs.
- [ ] Run build for GREEN.
- [ ] Commit `feat: prove input and terminal interaction safety`.

### Task 3: Epoch, context and graph ownership protocols

**Files:**
- Create: `Browser/Interaction/Epoch.lean`
- Create: `Browser/Interaction/Ownership.lean`
- Create: `Browser/Interaction/OwnershipSpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces `EpochProtocol`, `classifyEpoch`, `recreateEpoch`.
- Produces `Ownership`, `contextOwnsPage`, `pageOwnsFrame`, `authorizedContextEffect`, `authorizedFrameEffect`.

- [ ] Add examples proving stale/future epochs cannot mutate, recreation increments epoch, Context A cannot authorize Page B, and Frame effects require Page ownership.
- [ ] Run build for RED.
- [ ] Implement independent ownership/epoch models and theorem checks.
- [ ] Run build for GREEN.
- [ ] Commit `feat: prove epoch and ownership interaction contracts`.

### Task 4: Policy causality and authorized-effect composition

**Files:**
- Create: `Browser/Interaction/Policy.lean`
- Create: `Browser/Interaction/Composition.lean`
- Create: `Browser/Interaction/CompositionSpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces `PolicyEmission`, `emitPolicyCommand`, `commandCausallyValid`.
- Produces `InteractionEdge`, `AuthorizedEffect`, `GenerationIsolation`, `CausalIntegrity`, `TerminalSafety` executable predicates/theorems.

- [ ] Add examples asserting policy emission preserves correlation/parent, no child mutation exists at emission time, and unauthorized Page-to-Page effects are rejected.
- [ ] Run build for RED.
- [ ] Implement minimal composition layer from protocol guarantees.
- [ ] Run build for GREEN.
- [ ] Commit `feat: compose interaction safety contracts`.

### Task 5: AsyncRuntime projection and interaction trace conformance

**Files:**
- Create: `Browser/Interaction/Projection.lean`
- Create: `Browser/Interaction/Trace.lean`
- Create: `Browser/Interaction/TraceSpec.lean`
- Modify: `Browser.lean`
- Modify: `.github/workflows/verify.yml`

**Interfaces:**
- Consumes existing `AsyncEnvelope`/Driver trace types only in the adapter.
- Produces `projectAsyncInteraction` and `verifyInteractionTrace`.

- [ ] Add a compact interaction trace with Action start, CDP request/response, stale response, Human input, timer expiry, policy issue/delivery and Page epoch recreation.
- [ ] Verify the trace using the small interaction protocols, not the full `AsyncRuntime` state cross-product.
- [ ] Keep the existing full async verifier as an integration regression check.
- [ ] Run `lake build --wfail`, `lake exe browser-runtime-check`, `lake exe browser-driver-trace-check traces/generation-race.jsonl`, and `lake exe browser-async-trace-check traces/async-runtime-race.jsonl`.
- [ ] Commit `feat: verify projected interaction traces`.

## Self-review

- The plan covers all eight protocols from the design spec.
- No protocol requires the entire `Model` state except the final projection adapter.
- The composition layer has exactly five system-level targets: AuthorizedEffect, GenerationIsolation, InputExclusivity, CausalIntegrity and TerminalSafety.
- Existing full runtime verification remains intact as regression evidence while the proof surface shrinks.
