# Browser formal model documentation

This index defines the documentation status for the browser Driver model. The code import boundary is authoritative when prose and implementation history disagree.

## Current architecture

The supported formal entry point is:

```lean
import Browser.ProofAPI
```

The public proof chain is intentionally limited to:

```text
Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop
```

`Browser/ProofAPI.lean` is the public proof surface. `Browser.lean` exposes the executable/runtime model. `Browser/Integration.lean` exposes historical whole-runtime proofs, projection adapters, specs, scenarios, replay, and conformance artifacts.

### Current design specs

These documents describe the current proof architecture:

- `superpowers/specs/2026-08-21-interaction-contracts-design.md`
- `superpowers/specs/2026-08-21-decoder-contract-design.md`
- `superpowers/specs/2026-08-21-feedback-normalizer-design.md`
- `superpowers/specs/2026-08-21-feedback-recovery-contracts-design.md`
- `superpowers/specs/2026-08-21-adversarial-recovery-design.md`
- `superpowers/specs/2026-08-21-finite-fault-aggregation-design.md`

The terminal composition is defined directly by `Browser/Interaction/ClosedLoop.lean`. A second prose design for ClosedLoop is intentionally not maintained; the theorem module is the canonical source for the final composition and prevents another duplicated architecture narrative.

### Current implementation plans

These plans record implementation work for the current proof chain:

- `superpowers/plans/2026-08-21-interaction-contracts.md`
- `superpowers/plans/2026-08-21-decoder-contract.md`
- `superpowers/plans/2026-08-21-feedback-normalizer.md`
- `superpowers/plans/2026-08-21-feedback-recovery-contracts.md`
- `superpowers/plans/2026-08-21-adversarial-recovery.md`
- `superpowers/plans/2026-08-21-finite-fault-aggregation.md`

Current driver-facing integration support is documented by:

- `superpowers/plans/2026-08-21-driver-interaction-journal.md`
- `superpowers/plans/2026-08-21-driver-boundary-instrumentation.md`

These integration plans support the formal boundary but do not enlarge the public theorem API.

## Historical/reference

The following documents are retained for design history, regression context, and compatibility. They are **not** the current proof architecture and should not be used as the starting point for new public theorems:

- `browser-runtime-formal-model.md` — earlier normalized whole-runtime architecture.
- `superpowers/specs/2026-08-21-async-runtime-design.md` — whole-`AsyncRuntime` actor/message design.
- `superpowers/plans/2026-08-21-async-runtime.md` — implementation plan for that whole-runtime model.
- `superpowers/plans/2026-08-21-browser-runtime-policy-command.md` — early synchronous runtime policy/command layer.
- `superpowers/plans/2026-08-21-browser-runtime-causality.md` — early runtime-envelope causality layer.
- `superpowers/plans/2026-08-21-browser-runtime-action-dependencies.md` — early whole-runtime action dependency reconciliation.

Their corresponding code remains useful as runtime/integration evidence and is reachable through `Browser.Integration` where appropriate.

## Rule for new work

New correctness arguments should extend the existing five-stage public chain. Add a new public module only when it introduces a genuinely new semantic boundary that cannot be expressed by the existing Interaction, Feedback, Recovery, Aggregation, or ClosedLoop layers. Runtime replay, JSONL conformance, scenario checks, projection adapters, and whole-`AsyncRuntime` theorems belong to integration/reference unless they establish a new public semantic contract.
