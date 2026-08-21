import Browser.Event

namespace Browser

/-- Functional page update. This is the formal boundary that prevents a
    page-local event from mutating sibling pages implicitly. -/
def updatePage (m : Model) (p : PageId) (f : PageState → PageState) : Model :=
  { m with
    page := fun q => if q = p then f (m.page q) else m.page q }

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

/-- The sole state transition entry point for the executable specification. -/
def step (m : Model) (event : RuntimeEvent) : Model :=
  match event with
  | .local e => updatePage m e.page (applyLocal e.kind)
  | .browserDisconnected => { m with browserAlive := false }
  | .browserConnected => { m with browserAlive := true }

end Browser
