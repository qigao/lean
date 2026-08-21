import Browser.Policy

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

/-- Full reactive transition: first observe the world event, then execute only
    the explicit commands produced by policy. -/
def react (m : Model) (event : RuntimeEvent) : Model :=
  applyCommands (step m event) (commandsFor m event)

/-- Bounded causal entry point. Over-budget events are rejected before they can
    mutate semantic runtime state or trigger another policy reaction. -/
def reactEnvelope (budget : ReactionBudget) (m : Model) (envelope : EventEnvelope) : Option Model :=
  if envelope.cause.depth ≤ budget.maxDepth then
    some (react m envelope.event)
  else
    none

end Browser
