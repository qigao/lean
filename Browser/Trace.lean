import Browser.Proofs

namespace Browser

/-- Replay a concrete runtime event stream against the same `step` function
    used by the proofs. -/
def replay : Model → List RuntimeEvent → Model
  | m, [] => m
  | m, event :: rest => replay (step m event) rest

/-- Executable form of the primitive-action safety invariant for known pages. -/
def pageSafe (m : Model) (p : PageId) : Bool :=
  let s := m.page p
  match s.action with
  | .executing =>
      m.browserAlive &&
      (s.lifecycle == .ready) &&
      (s.runtime == .ready) &&
      (s.input == .automation)
  | _ => true

def wellFormed? (m : Model) : Bool :=
  m.knownPages.all (pageSafe m)

end Browser
