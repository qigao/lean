# Unified Async Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make asynchronous message delivery the outer formal semantics of the browser runtime while preserving the already-proven actor-local Page/Action transitions.

**Architecture:** Add an `AsyncRuntime` wrapper around the existing `Model`, with actor epoch registry, seen-message history and message-id allocation. `deliver` validates causality/target epoch, executes actor-local transition logic only for accepted messages, and emits policy commands as child messages rather than synchronously applying them. Dynamic Page/Frame topology is represented by graph lifecycle payloads.

**Tech Stack:** Lean 4.33.0, Lake, existing Browser formal model, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-21-async-runtime-design.md`

## Global Constraints

- Preserve existing `RuntimeEvent`, ActionId, timeout and conformance semantics during the additive migration.
- No cross-actor state mutation outside `deliver`.
- No global ordering assumption beyond delivery order and explicit causal parent links.
- Old NodeEpoch/ActionId messages must be state no-ops.
- Policy commands are emitted first and mutate a Page only when separately delivered.
- CI remains `lake build --wfail` plus executable runtime/conformance scenarios.

---

### Task 1: Async identities and runtime registry

**Files:**
- Modify: `Browser/Types.lean`
- Modify: `Browser/Graph.lean`
- Create: `Browser/Async.lean`
- Create: `Browser/AsyncSpec.lean`

**Interfaces:**
- Produces `NodeEpoch`, `ActorRef`, `ActorAddress`, `NodeSlot`, `AsyncRuntime`, `AsyncPayload`, `AsyncEnvelope`, `DeliveryDisposition`, `DeliveryResult`.

- [ ] **Step 1: Write failing async identity examples** in `Browser/AsyncSpec.lean` proving bootstrap Page actors are alive at epoch 1 and an unknown Page is not alive.
- [ ] **Step 2: Run `lake build --wfail`** and confirm failure is caused by missing async types/functions.
- [ ] **Step 3: Implement minimal identity/registry types** and `asyncFromModel : Model → AsyncRuntime`.
- [ ] **Step 4: Run `lake build --wfail`** and confirm the identity examples pass.
- [ ] **Step 5: Commit** with `feat: add async actor registry`.

### Task 2: Deliver validation and stale/duplicate/orphan semantics

**Files:**
- Modify: `Browser/Async.lean`
- Create: `Browser/AsyncProofs.lean`

**Interfaces:**
- Produces `deliver : AsyncRuntime → AsyncEnvelope → DeliveryResult` and causal/epoch validation helpers.

- [ ] **Step 1: Add failing examples** for duplicate MessageId, stale Page epoch and orphan target.
- [ ] **Step 2: Run `lake build --wfail`** and confirm the examples fail for missing delivery behavior.
- [ ] **Step 3: Implement validation order:** duplicate → causal validity → target epoch/aliveness → payload-target validity → accepted delivery.
- [ ] **Step 4: Prove** `stale_message_noop`, `duplicate_delivery_noop`, and `orphan_delivery_noop`.
- [ ] **Step 5: Run `lake build --wfail`** and commit `feat: formalize async delivery dispositions`.

### Task 3: Async policy command emission

**Files:**
- Modify: `Browser/Async.lean`
- Modify: `Browser/AsyncSpec.lean`
- Modify: `Browser/AsyncProofs.lean`

**Interfaces:**
- Runtime-event delivery uses existing `step` for actor-local state.
- Existing `commandsFor` becomes emitted `AsyncPayload.pageCommand` children.
- Command delivery uses existing `applyCommand`.

- [ ] **Step 1: Add failing context scenario:** delivering `contextUnavailable` changes parent availability and emits Page pause messages but leaves child PageState unchanged.
- [ ] **Step 2: Run `lake build --wfail`** and confirm failure.
- [ ] **Step 3: Implement deterministic child-message allocation** using `nextMessageId`, inherited correlation, parent=current message, depth+1.
- [ ] **Step 4: Implement page-command delivery** and prove `policy_command_not_applied_before_delivery` plus page-target isolation.
- [ ] **Step 5: Run `lake build --wfail`** and commit `feat: make policy reactions asynchronous`.

### Task 4: Dynamic Page/Frame graph lifecycle

**Files:**
- Modify: `Browser/Graph.lean`
- Modify: `Browser/Async.lean`
- Modify: `Browser/AsyncSpec.lean`
- Modify: `Browser/AsyncProofs.lean`

**Interfaces:**
- Adds graph payloads `pageCreated`, `pageDestroyed`, `frameCreated`, `frameDestroyed`.
- Adds functional graph bind helpers for Page→Context and Frame→Page/parent.

- [ ] **Step 1: Add failing create/destroy/recreate scenario** with Page epoch 1 destroyed, Page epoch 2 created, then epoch-1 event delivered.
- [ ] **Step 2: Run `lake build --wfail`** and confirm failure.
- [ ] **Step 3: Implement monotonic node creation/destruction** with dead slots retaining their last epoch.
- [ ] **Step 4: Prove old epoch cannot mutate new Page/Frame incarnation and graph ownership remains well formed for accepted lifecycle messages.
- [ ] **Step 5: Run `lake build --wfail`** and commit `feat: model async runtime graph lifecycle`.

### Task 5: JSONL async conformance and CI

**Files:**
- Modify: `Browser/Conformance.lean`
- Create: `traces/async-runtime-race.jsonl`
- Modify: `DriverTraceMain.lean`
- Modify: `.github/workflows/verify.yml`
- Modify: `Browser.lean`

**Interfaces:**
- External trace maps directly to `AsyncEnvelope` and uses `deliver` rather than invoking runtime transitions directly.

- [ ] **Step 1: Add a failing JSONL trace** containing duplicate MessageId, Page recreation, old-epoch message, stale ActionId response, and async policy command delivery.
- [ ] **Step 2: Run the trace verifier** and confirm rejection/missing async path as expected.
- [ ] **Step 3: Decode source/target actor epochs and async payloads**, replay through `deliver`, and surface non-accepted dispositions deterministically.
- [ ] **Step 4: Run `lake build --wfail`, `lake exe browser-runtime-check`, and both conformance traces**.
- [ ] **Step 5: Commit** `feat: verify async driver traces` and update PR description.

## Self-review

- Spec coverage: message identity, causal order, NodeEpoch, ActionId compatibility, async policy, dynamic graph and external conformance each have a task.
- No placeholder/TODO steps remain.
- Type names are consistent across tasks: `AsyncRuntime`, `AsyncEnvelope`, `AsyncPayload`, `DeliveryResult`, `NodeEpoch`, `ActorRef`, `ActorAddress`.
