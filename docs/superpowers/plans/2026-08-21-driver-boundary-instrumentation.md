# Driver Interaction Boundary Instrumentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attach the existing typed `InteractionJournal` to real Driver side-effect/ownership boundaries without inventing a full browser Driver implementation.

**Architecture:** Add a small C++17 `InteractionBoundary` wrapper that takes the real boundary callable from `ActionEngine`, `CdpSession`, `InputRouter`, deadline/timer handling, `PolicyEngine`, or node lifecycle code. The wrapper writes the typed journal record immediately before invoking that callable. JSON remains isolated in `JsonlInteractionSink`, and Lean continues to validate only the resulting interaction trace.

**Tech Stack:** C++17, Lean 4.33.0, JSONL, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-21-interaction-contracts-design.md`

## Global Constraints

- Do not fabricate complete `ActionEngine`, `CdpSession`, `InputRouter`, `PolicyEngine`, or RuntimeGraph implementations in this repo.
- Business code must not construct JSON or allocate journal sequence numbers.
- `cdp_request` and `input_dispatch` are journaled immediately before the real external side-effect callable.
- Response/callback/lifecycle events are journaled immediately before delivery/mutation callables.
- A journal record means the Driver attempted/delivered that boundary interaction; it does not assert the delegated operation succeeded.
- `lane` remains opaque and is owned by the real Driver runtime; the reference integration uses one lane per Page interaction domain.

---

### Task 1: Typed boundary wrapper

**Files:**
- Create: `driver/include/browser/interaction_boundary.hpp`
- Create: `driver/examples/interaction_boundary_smoke.cpp`
- Modify: `.github/workflows/verify.yml`

**Interfaces:**
- Produces: `InteractionBoundary` with typed methods for action lifecycle, CDP request/response, deadline arm/expiry, input dispatch, human input, policy issue/delivery, epoch recreation, and actor destruction.
- Side-effect methods receive a callable and invoke it only after the corresponding journal entry is written.

- [ ] Write the smoke program against the desired API and assert the sink already contains the correct record from inside each delegated callable.
- [ ] Add a CI compile/run step and confirm RED because `browser/interaction_boundary.hpp` does not exist.
- [ ] Implement the minimal header-only typed wrapper.
- [ ] Confirm the C++ smoke and all existing checks are GREEN.

### Task 2: Route reference emitter through the boundary

**Files:**
- Modify: `driver/examples/emit_interaction_trace.cpp`

**Interfaces:**
- The emitter must no longer call `InteractionJournal` directly for boundary interactions; it must use `InteractionBoundary` and trivial callables that stand in for the real Driver effects.

- [ ] Change the emitter first and confirm compile failure if any boundary API is missing.
- [ ] Use `InteractionBoundary` for the multi-lane timeout and CDP/input scenario.
- [ ] Confirm C++ -> JSONL -> Lean round-trip remains accepted.

### Task 3: Document integration points

**Files:**
- Modify: `README.md`

- [ ] Document exact placement rules: before transport send, before input dispatch, before response delivery, inside timer callback before action notification, before policy enqueue/delivery, and before lifecycle mutation.
- [ ] Document that logging an attempt before a throwing delegate is intentional; success/failure telemetry belongs to a separate diagnostic channel unless it becomes a formal interaction requirement.
- [ ] Run the full workflow and require all C++ and Lean checks GREEN.
