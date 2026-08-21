# Driver Interaction Journal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define a compact multi-lane C/C++ Driver InteractionJournal contract and verify it directly with Lean without reconstructing the complete Browser runtime.

**Architecture:** Driver components emit typed interaction records into a journal. The journal assigns a global monotonically increasing sequence number and an opaque lane id. Lean verifies strict global observation order while keeping one small `InteractionTraceState` per lane. A C++17 reference API serializes the same contract through a pluggable sink.

**Tech Stack:** Lean 4.33.0, JSONL, C++17, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-21-interaction-contracts-design.md`

## Global Constraints

- Interaction contracts remain the primary proof surface; do not reintroduce the full `Browser.Model` into the compact journal verifier.
- Every journal event has a global `seq` and opaque `lane`.
- Global `seq` is strictly increasing in observed delivery order.
- Each lane owns an independent `InteractionTraceState`; events in one lane cannot mutate another lane.
- C++ components emit typed records; JSON serialization belongs to a sink/adapter, not Action/CDP/Input/Deadline/Policy business code.
- `inputDispatch` and `cdpRequest` are external side effects and must be rejected when ownership/liveness contracts forbid them.

---

### Task 1: Multi-lane Lean journal verifier

**Files:**
- Create: `Browser/Interaction/JournalConformance.lean`
- Create: `Browser/Interaction/JournalConformanceSpec.lean`
- Modify: `Browser.lean`

**Interfaces:**
- Produces: `verifyInteractionJournalJsonl : String → Except String InteractionJournalState`
- Produces: strict sequence ordering and independent lane replay.

- [ ] Write failing specs for two interleaved lanes, out-of-order `seq`, lane-local timeout, and lane isolation.
- [ ] Run CI and confirm failure is due to missing `Browser.Interaction.JournalConformance`.
- [ ] Implement minimal journal header/record parser and lane-state map.
- [ ] Run `lake build --wfail` and confirm specs pass.

### Task 2: C++17 typed journal API

**Files:**
- Create: `driver/include/browser/interaction_journal.hpp`
- Create: `driver/src/interaction_journal.cpp`
- Create: `driver/examples/interaction_journal_smoke.cpp`

**Interfaces:**
- Produces: `InteractionJournal`, `InteractionSink`, `InteractionRecord`, typed emit methods.
- `InteractionJournal` serializes sequence assignment and sink delivery so emitted order matches `seq`.

- [ ] Add compile smoke test first.
- [ ] Confirm CI fails because API files are missing.
- [ ] Implement typed record + thread-safe journal.
- [ ] Compile with `c++ -std=c++17 -Wall -Wextra -Werror`.

### Task 3: JSONL sink and Lean conformance

**Files:**
- Create: `driver/include/browser/jsonl_interaction_sink.hpp`
- Create: `driver/src/jsonl_interaction_sink.cpp`
- Create: `driver/examples/emit_interaction_trace.cpp`
- Create: `traces/driver-interaction-journal.jsonl`
- Modify: `.github/workflows/verify.yml`
- Modify: `README.md`

**Interfaces:**
- C++ example emits exactly the schema accepted by `verifyInteractionJournalJsonl`.

- [ ] Add CI command compiling/running the C++ example and validating its output with Lean.
- [ ] Confirm the test fails before the sink is implemented.
- [ ] Implement JSONL sink without external JSON dependencies.
- [ ] Run full CI and require all Lean and C++ checks green.
