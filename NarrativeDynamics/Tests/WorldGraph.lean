import NarrativeDynamics.Core.WorldGraph

open NarrativeDynamics

example {Agent Event Node : Type*} [DecidableEq Agent] [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w) (a : Agent) :
    WorldInvariant (killAgent w a) := by
  exact killAgent_preserves_invariant w h a

example {Agent Event Node : Type*} [DecidableEq Agent] [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w)
    (e : Event) (a : Agent)
    (hr : infoReachable w.info (w.eventNode e) (w.agentNode a)) :
    WorldInvariant (addObservation w e a hr) := by
  exact addObservation_preserves_invariant w h e a hr

example {Agent Event Node : Type*} [DecidableEq Agent] [DecidableEq Event]
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w)
    (u v : Event) (ht : w.eventTime u < w.eventTime v) :
    WorldInvariant (addCausalEdge w u v ht) := by
  exact addCausalEdge_preserves_invariant w h u v ht

example {Agent Event Node : Type*}
    (rewrites : List (WorldGraph Agent Event Node → WorldGraph Agent Event Node))
    (hpres : ∀ r ∈ rewrites, PreservesWorldInvariant r)
    (w : WorldGraph Agent Event Node) (hw : WorldInvariant w) :
    WorldInvariant (rewrites.foldl (fun state r => r state) w) := by
  exact rewriteSequence_preserves_invariant rewrites hpres w hw
