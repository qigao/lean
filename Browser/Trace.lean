import Browser.Proofs

namespace Browser

/-- Replay a concrete runtime event stream through the fully reconciled runtime:
    normalized state transition, explicit policy commands, then action dependency
    reconciliation. -/
def replay : Model → List RuntimeEvent → Model
  | m, [] => m
  | m, event :: rest => replay (reactSafe m event) rest

/-- Replay causally wrapped events through both the finite reaction-budget gate
    and action dependency reconciliation. -/
def replayEnvelopes (budget : ReactionBudget) : Model → List EventEnvelope → Option Model
  | m, [] => some m
  | m, envelope :: rest =>
      match reactEnvelopeSafe budget m envelope with
      | none => none
      | some next => replayEnvelopes budget next rest

/-- Executable form of the primitive-action safety invariant for known pages. -/
def pageSafe (m : Model) (p : PageId) : Bool :=
  let s := m.page p
  match s.action with
  | .executing =>
      requirementsHold m (primitiveContract p)
  | _ => true

def wellFormed? (m : Model) : Bool :=
  m.knownPages.all (pageSafe m)

/-- Check every reconciled semantic state reached by a concrete event trace. -/
def traceSafe : Model → List RuntimeEvent → Bool
  | m, [] => wellFormed? m
  | m, event :: rest =>
      let next := reactSafe m event
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
