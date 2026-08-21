import Browser.Interaction.Cdp
import Browser.Interaction.Deadline
import Browser.Interaction.Input
import Browser.Interaction.Terminal
import Browser.Interaction.Epoch
import Browser.Interaction.Ownership
import Browser.Interaction.Policy
import Browser.Interaction.Composition
import Browser.Interaction.Trace
import Browser.Interaction.Feedback
import Browser.Interaction.Normalizer
import Browser.Interaction.Decoder
import Browser.Interaction.Recovery
import Browser.Interaction.FeedbackRecovery
import Browser.Interaction.DecoderRecovery
import Browser.Interaction.AdversarialRecovery
import Browser.Interaction.FaultAggregation
import Browser.Interaction.TraceAggregation
import Browser.Interaction.ClosedLoop

/-!
# Browser public proof API

This is the supported proof entry point for the browser Driver model.

The public proof story is intentionally one directional chain:

`Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop`.

The small modules imported above remain useful implementation units, but new
public proof obligations should extend this chain instead of creating a second
end-to-end theorem family.

Legacy whole-`AsyncRuntime` proofs, executable specs, scenarios, JSONL
conformance/replay checks, and `AsyncRuntime` projection adapters are regression
evidence. They are intentionally not re-exported here; import
`Browser.Integration` when those reference artifacts are needed.
-/
