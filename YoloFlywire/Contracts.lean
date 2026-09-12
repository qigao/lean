import YoloFlywire.Basic

namespace YoloFlywire

def sameObservationBoundary (a b : RunArm) : Prop :=
  a.observationSchemaHash = b.observationSchemaHash ∧
  a.splitHash = b.splitHash

def sameBudget (a b : RunArm) : Prop :=
  a.budget = b.budget

def hasRequiredRewiredControl (p : RunProtocol) : Prop :=
  ∃ fly rewired,
    fly ∈ p.arms ∧
    rewired ∈ p.arms ∧
    fly.family = .flywire ∧
    rewired.family = .rewiredFlywire ∧
    sameObservationBoundary fly rewired ∧
    sameBudget fly rewired

end YoloFlywire
