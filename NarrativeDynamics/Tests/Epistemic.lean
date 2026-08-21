import NarrativeDynamics.Core.Epistemic

open NarrativeDynamics

example {Node Observation Belief : Type}
    (g : InfoGraph Node) (event agent : Node)
    (payload : Observation) (prior : Belief)
    (revise : Belief → Observation → Belief)
    (h : ¬ infoReachable g event agent) :
    beliefUpdate revise prior (deliveredObservation g event agent payload) = prior := by
  exact no_epistemic_teleportation g event agent payload prior revise h
