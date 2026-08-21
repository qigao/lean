import Browser.Graph

namespace Browser

structure PageState where
  lifecycle : Lifecycle := .attaching
  runtime : RuntimeState := .unavailable
  input : InputState := .idle
  action : ActionState := .idle
  contextGeneration : Nat := 0
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
  page : PageId → PageState := fun _ => {}

/-- A primitive action may execute only when all browser/page/runtime/input
    preconditions hold simultaneously. -/
def canExecute (m : Model) (p : PageId) : Prop :=
  m.browserAlive = true ∧
  (m.page p).lifecycle = .ready ∧
  (m.page p).runtime = .ready ∧
  (m.page p).input = .automation ∧
  (m.page p).action = .executing

end Browser
