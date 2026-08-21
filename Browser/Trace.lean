import Browser.Proofs

namespace Browser

/-- Replay a concrete runtime event stream through the full reactive pipeline:
    normalized state transition followed by explicit policy commands. -/
def replay : Model → List RuntimeEvent → Model
  | m, [] => m
  | m, event :: rest => replay (react m event) rest

/-- Replay causally wrapped events through the finite reaction-budget gate. -/
def replayEnvelopes (budget : ReactionBudget) : Model → List EventEnvelope → Option Model
  | m, [] => some m
  | m, envelope :: rest =>
      match reactEnvelope budget m envelope with
      | none => none
      | some next => replayEnvelopes budget next rest

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

/-- Check every semantic state reached by a concrete event trace. -/
def traceSafe : Model → List RuntimeEvent → Bool
  | m, [] => wellFormed? m
  | m, event :: rest =>
      let next := react m event
      wellFormed? next && traceSafe next rest

/-- Executable check corresponding to the causal theorems for one issued command. -/
def issuedCauseSafe (envelope : EventEnvelope) (issued : IssuedCommand) : Bool :=
  (issued.cause.correlation == envelope.cause.correlation) &&
  (issued.cause.parent == some envelope.id) &&
  (issued.cause.depth == envelope.cause.depth + 1)

/-- All commands emitted by policy for this event must carry the exact child cause. -/
def causalPolicySafe (m : Model) (envelope : EventEnvelope) : Bool :=
  (issueCommandsFor m envelope).all (issuedCauseSafe envelope)

end Browser
