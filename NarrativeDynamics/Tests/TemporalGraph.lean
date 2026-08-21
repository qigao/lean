import NarrativeDynamics.Core.WorldGraph

open NarrativeDynamics

example {Agent Event Node : Type*}
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w)
    {u v : Event} (hp : causalPath w u v) :
    w.eventTime u < w.eventTime v := by
  exact causalPath_time_increasing w h hp

example {Agent Event Node : Type*}
    (w : WorldGraph Agent Event Node) (h : WorldInvariant w) (e : Event) :
    ¬ causalPath w e e := by
  exact causalGraph_acyclic w h e
