import Mathlib

namespace NarrativeDynamics

/-- Directed information graph. Edges represent permitted observation or
information-transfer steps. -/
abbrev InfoGraph (Node : Type*) := Node → Node → Prop

/-- Information is reachable when there is a finite (possibly empty) chain
of permitted information edges from source to destination. -/
def infoReachable {Node : Type*} (g : InfoGraph Node) (src dst : Node) : Prop :=
  Relation.ReflTransGen g src dst

/-- An observation is delivered only when the information graph contains a
reachable path from the event/source node to the receiving agent node. -/
noncomputable def deliveredObservation {Node Observation : Type*}
    (g : InfoGraph Node) (src dst : Node) (payload : Observation) : Option Observation := by
  classical
  exact if infoReachable g src dst then some payload else none

/-- Belief revision consumes only delivered observations. No observation
leaves the prior belief unchanged. -/
def beliefUpdate {Observation Belief : Type*}
    (revise : Belief → Observation → Belief) (prior : Belief) : Option Observation → Belief
  | none => prior
  | some obs => revise prior obs

/-- No epistemic teleportation: if the event has no permitted information
path to an agent, that event cannot directly alter the agent's belief through
this update channel. -/
theorem no_epistemic_teleportation {Node Observation Belief : Type*}
    (g : InfoGraph Node) (event agent : Node)
    (payload : Observation) (prior : Belief)
    (revise : Belief → Observation → Belief)
    (h : ¬ infoReachable g event agent) :
    beliefUpdate revise prior (deliveredObservation g event agent payload) = prior := by
  simp [deliveredObservation, h, beliefUpdate]

end NarrativeDynamics
