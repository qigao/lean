import Browser.Policy
import Browser.Action
import Browser.Timeout

namespace Browser

/-- Preserve terminal timeout while putting all other active actions into recovery. -/
def recoveryAction (s : PageState) : ActionState :=
  match s.action with
  | .idle => .idle
  | .timedOut => .timedOut
  | _ => .recovering

/-- Page/runtime loss becomes the current wait cause only for a live action. -/
def pageWaitReason (s : PageState) : Option WaitReason :=
  match s.action with
  | .idle => s.waitingOn
  | .timedOut => s.waitingOn
  | _ => some .page

/-- Crash/close suspend a live action but cannot resurrect a timed-out one. -/
def suspendedOrTimedOut (s : PageState) : ActionState :=
  match s.action with
  | .timedOut => .timedOut
  | _ => .suspended

/-- Pure transition for a single page-local event. -/
def applyLocal (kind : LocalEventKind) (s : PageState) : PageState :=
  match kind with
  | .humanInput =>
      if s.input = .automation then
        { s with input := .conflict, action := .suspended, waitingOn := some .human }
      else
        { s with input := .human }
  | .humanIdle =>
      match s.input with
      | .human | .conflict => { s with input := .idle }
      | _ => s
  | .automationStarted =>
      let fresh := clearDeadlineState s
      if s.lifecycle = .ready ∧ s.runtime = .ready ∧ s.input = .idle then
        { fresh with input := .automation, action := .executing }
      else
        { fresh with action := .waiting }
  | .automationFinished =>
      let fresh := clearDeadlineState s
      { fresh with input := .idle, action := .idle }
  | .navigationStarted =>
      { s with
        lifecycle := .loading
        runtime := .unavailable
        waitingOn := pageWaitReason s
        action := recoveryAction s }
  | .pageReady =>
      { s with lifecycle := .ready }
  | .executionContextDestroyed =>
      { s with
        runtime := .unavailable
        waitingOn := pageWaitReason s
        action := recoveryAction s }
  | .executionContextReady =>
      { s with
        runtime := .ready
        contextGeneration := s.contextGeneration + 1
        action := if s.action = .recovering then .waiting else s.action }
  | .deadlineArmed expiresAt reason =>
      armDeadlineState s expiresAt reason
  | .deadlineReached now =>
      expirePageState s now
  | .pageCrashed =>
      { s with
        lifecycle := .crashed
        runtime := .unavailable
        input := .idle
        action := suspendedOrTimedOut s }
  | .pageClosed =>
      { s with
        lifecycle := .closed
        runtime := .unavailable
        input := .idle
        action := suspendedOrTimedOut s }

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
