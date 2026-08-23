import NarrativeDynamics.Core.TypedGraph

open NarrativeDynamics

example {Agent Event Object Location Institution Concept : Type*}
    (e : TypedEdge Agent Event Object Location Institution Concept) :
    WellTypedEdge e := by
  exact edge_is_well_typed e

example {Agent Event Object Location Institution Concept : Type*}
    (e : TypedEdge Agent Event Object Location Institution Concept)
    (h : edgeKind e = EdgeKind.causal) :
    nodeKind (edgeSource e) = NodeKind.event ∧
    nodeKind (edgeTarget e) = NodeKind.event := by
  exact causal_edge_endpoints_are_events e h
