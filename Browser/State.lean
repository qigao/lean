import Browser.Graph

namespace Browser

structure PageState where
  lifecycle : Lifecycle := .attaching
  runtime : RuntimeState := .unavailable
  input : InputState := .idle
  action : ActionState := .idle
  contextGeneration : Nat := 0
  deadline : Option Deadline := none
  waitingOn : Option WaitReason := none
  timeout : Option WaitReason := none
  deriving Repr

namespace PageState

def readyIdle : PageState := {
  lifecycle := .ready
  runtime := .ready
  input := .idle
  action := .idle
  contextGeneration := 1
}

/-- Safety invariant for primitive actions: an executing action must own
    automation input and must run only on a ready page/runtime. -/
def WellFormed (s : PageState) : Prop :=
  s.action = .executing →
    s.lifecycle = .ready ∧
    s.runtime = .ready ∧
    s.input = .automation

end PageState

structure Model where
  browserAlive : Bool := true
  graph : RuntimeGraph
  knownPages : List PageId := []
  contextAvailable : ContextId → Bool := fun _ => true
  page : PageId → PageState := fun _ => {}

/-- Functional page update. This is the formal boundary that prevents a
    page-targeted transition from mutating sibling pages implicitly. -/
def updatePage (m : Model) (p : PageId) (f : PageState → PageState) : Model :=
  { m with
    page := fun q => if q = p then f (m.page q) else m.page q }

/-- Functional parent-context availability update. -/
def updateContextAvailability (m : Model) (c : ContextId) (available : Bool) : Model :=
  { m with
    contextAvailable := fun q => if q = c then available else m.contextAvailable q }

/-- A primitive action may execute only when all browser/context/page/runtime/input
    preconditions hold simultaneously. A timed-out action is therefore terminal
    until a new action is explicitly started. -/
def canExecute (m : Model) (p : PageId) : Prop :=
  m.browserAlive = true ∧
  m.contextAvailable (m.graph.contextOf p) = true ∧
  (m.page p).lifecycle = .ready ∧
  (m.page p).runtime = .ready ∧
  (m.page p).input = .automation ∧
  (m.page p).action = .executing

end Browser
