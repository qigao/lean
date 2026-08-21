import Browser.Proofs

namespace Browser

/-- Replay a concrete runtime event stream through the full reactive pipeline:
    normalized state transition followed by explicit policy commands. -/
def replay : Model → List RuntimeEvent → Model
  | m, [] => m
  | m, event :: rest => replay (react m event) rest

/-- Executable form of the primitive-action safety invariant for known pages. -/
def pageSafe (m : Model) (p : PageId) : Bool :=
  let s := m.page p
  match s.action with
  | .executing =>
      m.browserAlive &&
      m.contextAvailable (m.graph.contextOf p) &&
      (s.lifecycle == .ready) &&
      (s.runtime == .ready) &&
      (s.input == .automation)
  | _ => true

def wellFormed? (m : Model) : Bool :=
  m.knownPages.all (pageSafe m)

/-- Check every state reached by a concrete event trace, not only the final
    state. This is the executable conformance hook for future driver traces. -/
def traceSafe : Model → List RuntimeEvent → Bool
  | m, [] => wellFormed? m
  | m, event :: rest =>
      let next := react m event
      wellFormed? next && traceSafe next rest

end Browser
