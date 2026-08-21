import Browser.Event

namespace Browser

/-- Causal metadata is deliberately orthogonal to semantic `RuntimeEvent` data. -/
structure Cause where
  correlation : CorrelationId
  parent : Option EventId := none
  depth : Nat := 0
  deriving Repr, DecidableEq, BEq

/-- Stable event identity plus causal metadata around a normalized event. -/
structure EventEnvelope where
  id : EventId
  event : RuntimeEvent
  cause : Cause
  deriving Repr

/-- Finite upper bound for policy/reaction chains. -/
structure ReactionBudget where
  maxDepth : Nat
  deriving Repr, DecidableEq, BEq

def WithinBudget (budget : ReactionBudget) (envelope : EventEnvelope) : Prop :=
  envelope.cause.depth ≤ budget.maxDepth

def withinBudget (budget : ReactionBudget) (envelope : EventEnvelope) : Bool :=
  decide (WithinBudget budget envelope)

/-- Causal metadata inherited by policy output from a triggering event. -/
def childCause (envelope : EventEnvelope) : Cause := {
  correlation := envelope.cause.correlation
  parent := some envelope.id
  depth := envelope.cause.depth + 1
}

end Browser
