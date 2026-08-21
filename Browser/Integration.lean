import Browser.ProofAPI

import Browser.Proofs
import Browser.TimeoutProofs
import Browser.GenerationProofs
import Browser.PolicySpec
import Browser.CausalitySpec
import Browser.ActionSpec
import Browser.TimeoutSpec
import Browser.GenerationSpec

import Browser.AsyncProofs
import Browser.AsyncSpec
import Browser.AsyncScenario
import Browser.AsyncActionScenario
import Browser.Conformance
import Browser.ConformanceSpec
import Browser.AsyncConformance
import Browser.AsyncConformanceSpec

import Browser.Interaction.Projection
import Browser.Interaction.CoreSpec
import Browser.Interaction.SafetySpec
import Browser.Interaction.OwnershipSpec
import Browser.Interaction.CompositionSpec
import Browser.Interaction.TraceSpec
import Browser.Interaction.ConformanceSpec
import Browser.Interaction.JournalConformanceSpec
import Browser.Interaction.FeedbackSpec
import Browser.Interaction.RecoverySpec
import Browser.Interaction.RecoveryTraceSpec
import Browser.Interaction.NormalizerSpec
import Browser.Interaction.FeedbackRecoverySpec
import Browser.Interaction.DecoderSpec
import Browser.Interaction.DecoderRecoverySpec
import Browser.Interaction.AdversarialRecoverySpec
import Browser.Interaction.FaultAggregationSpec
import Browser.Interaction.TraceAggregationSpec
import Browser.Interaction.ClosedLoopSpec

/-!
# Browser integration/reference surface

This module keeps the historical whole-runtime proofs, AsyncRuntime projection,
and executable regression specifications available without making them part of
the supported proof API.

Use it for integration, replay, conformance, and regression work. New formal
contracts should normally be added through `Browser.ProofAPI` and the
`Interaction -> Feedback -> Recovery -> Aggregation -> ClosedLoop` chain.
-/
