import Browser.Policy
import Browser.Action

namespace Browser

/-- Pure transition for a single page-local event. -/
def applyLocal (kind : LocalEventKind) (s : PageState) : PageState :=
  match kind with
  | .humanInput =>
      if s.input = .automation then
        { s with input := .conflict, action := .suspended }
      else
        { s with input := .human }
  | .humanIdle =>
      match s.input with
      | .human | .conflict => { s with input := .idle }
      | _ => s
  | .automationStarted =>
      if s.lifecycle = .ready ∧ s.runtime = .ready ∧ s.input = .idle then
        { s with input := .automation, action := .executing }
      else
        { s with action := .waiting }
  | .automationFinished =>
      { s with input := .idle, action := .idle }
  | .navigationStarted =>
      { s with
        lifecycle := .loading
        runtime := .unavailable
        action := if s.action = .idle then .idle else .recovering }
  | .pageReady =>
      { s with lifecycle := .ready }
  | .executionContextDestroyed =>
      { s with
        runtime := .unavailable
        action := if s.action = .idle then .idle else .recovering }
  | .executionContextReady =>
      { s with
        runtime := .ready
        contextGeneration := s.contextGeneration + 1
        action := if s.action = .recovering then .waiting else s.action }
  | .pageCrashed =>
      { s with
        lifecycle := .crashed
        runtime := .unavailable
        input := .idle
        action := .suspended }
  | .pageClosed =>
      { s with
        lifecycle := .closed
        runtime := .unavailable
        input := .idle
        action := .suspended }

/-- Raw normalized state transition. Cross-page effects are intentionally absent. -/
def step (m : Model) (event : RuntimeEvent) : Model :=
  match event with
  | .local e => updatePage m e.page (applyLocal e.kind)
  | .contextUnavailable context => updateContextAvailability m context false
  | .contextAvailable context => updateContextAvailability m context true
  | .browserDisconnected => { m with browserAlive := false }
  | .browserConnected => { m with browserAlive := true }

/-- Full reactive transition: observe the event, then execute explicit policy commands. -/
def react (m : Model) (event : RuntimeEvent) : Model :=
  applyCommands (step m event) (commandsFor m event)

/-- Runtime-safe reaction additionally reconciles every running action contract. -/
def reactSafe (m : Model) (event : RuntimeEvent) : Model :=
  reconcileKnownActions (react m event)

/-- Bounded causal entry point for the semantic `react` path. -/
def reactEnvelope (budget : ReactionBudget) (m : Model) (envelope : EventEnvelope) : Option Model :=
  if envelope.cause.depth ≤ budget.maxDepth then
    some (react m envelope.event)
  else
    none

/-- Bounded causal entry point used by the action-safe runtime. -/
def reactEnvelopeSafe (budget : ReactionBudget) (m : Model) (envelope : EventEnvelope) : Option Model :=
  if envelope.cause.depth ≤ budget.maxDepth then
    some (reactSafe m envelope.event)
  else
    none

end Browser
