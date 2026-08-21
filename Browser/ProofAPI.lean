import Browser.Interaction.Trace
import Browser.Interaction.Ownership
import Browser.Interaction.ClosedLoop

/-!
# Browser public proof API

This is the supported proof entry point for the browser Driver model.

The public proof story is intentionally one directional chain:

`Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop`.

`Trace` plus `Ownership` provide the compact interaction contracts. `ClosedLoop`
transitively provides the feedback, recovery, finite aggregation, runtime/algebra
bridge, and terminal closed-loop guarantees.

New public proof obligations should extend this chain instead of creating a
second end-to-end theorem family.

Legacy whole-`AsyncRuntime` proofs, executable specs, scenarios, JSONL
conformance/replay checks, and `AsyncRuntime` projection adapters are regression
evidence. They are intentionally not re-exported here; import
`Browser.Integration` when those reference artifacts are needed.
-/
