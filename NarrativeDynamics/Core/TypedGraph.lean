import Mathlib

namespace NarrativeDynamics

/-- Kinds of heterogeneous nodes used by the minimal narrative world. -/
inductive NodeKind where
  | agent
  | event
  | object
  | location
  | institution
  | concept
  deriving DecidableEq, Repr

/-- A heterogeneous node keeps the payload type associated with its node kind. -/
inductive HNode (Agent Event Object Location Institution Concept : Type*) where
  | agent : Agent → HNode Agent Event Object Location Institution Concept
  | event : Event → HNode Agent Event Object Location Institution Concept
  | object : Object → HNode Agent Event Object Location Institution Concept
  | location : Location → HNode Agent Event Object Location Institution Concept
  | institution : Institution → HNode Agent Event Object Location Institution Concept
  | concept : Concept → HNode Agent Event Object Location Institution Concept

/-- Semantic edge families admitted by the minimal typed graph. -/
inductive EdgeKind where
  | causal
  | observes
  | locatedAt
  | owns
  | memberOf
  deriving DecidableEq, Repr

/-- Typed edge constructors encode endpoint legality directly.
An ill-typed edge such as `causal : Agent → Event` has no constructor. -/
inductive TypedEdge (Agent Event Object Location Institution Concept : Type*) where
  | causal : Event → Event → TypedEdge Agent Event Object Location Institution Concept
  | observes : Event → Agent → TypedEdge Agent Event Object Location Institution Concept
  | locatedAt : Agent → Location → TypedEdge Agent Event Object Location Institution Concept
  | owns : Agent → Object → TypedEdge Agent Event Object Location Institution Concept
  | memberOf : Agent → Institution → TypedEdge Agent Event Object Location Institution Concept

/-- Recover the coarse node kind from a heterogeneous node. -/
def nodeKind {Agent Event Object Location Institution Concept : Type*} :
    HNode Agent Event Object Location Institution Concept → NodeKind
  | .agent _ => .agent
  | .event _ => .event
  | .object _ => .object
  | .location _ => .location
  | .institution _ => .institution
  | .concept _ => .concept

/-- Recover an edge's semantic kind. -/
def edgeKind {Agent Event Object Location Institution Concept : Type*} :
    TypedEdge Agent Event Object Location Institution Concept → EdgeKind
  | .causal _ _ => .causal
  | .observes _ _ => .observes
  | .locatedAt _ _ => .locatedAt
  | .owns _ _ => .owns
  | .memberOf _ _ => .memberOf

/-- Source endpoint as a heterogeneous node. -/
def edgeSource {Agent Event Object Location Institution Concept : Type*} :
    TypedEdge Agent Event Object Location Institution Concept →
      HNode Agent Event Object Location Institution Concept
  | .causal u _ => .event u
  | .observes e _ => .event e
  | .locatedAt a _ => .agent a
  | .owns a _ => .agent a
  | .memberOf a _ => .agent a

/-- Target endpoint as a heterogeneous node. -/
def edgeTarget {Agent Event Object Location Institution Concept : Type*} :
    TypedEdge Agent Event Object Location Institution Concept →
      HNode Agent Event Object Location Institution Concept
  | .causal _ v => .event v
  | .observes _ a => .agent a
  | .locatedAt _ l => .location l
  | .owns _ o => .object o
  | .memberOf _ i => .institution i

/-- Runtime-level endpoint specification corresponding to the constructors. -/
def allowedEndpoints : EdgeKind → NodeKind → NodeKind → Prop
  | .causal, .event, .event => True
  | .observes, .event, .agent => True
  | .locatedAt, .agent, .location => True
  | .owns, .agent, .object => True
  | .memberOf, .agent, .institution => True
  | _, _, _ => False

/-- A typed edge satisfies the endpoint schema recovered from its constructors. -/
def WellTypedEdge {Agent Event Object Location Institution Concept : Type*}
    (e : TypedEdge Agent Event Object Location Institution Concept) : Prop :=
  allowedEndpoints (edgeKind e) (nodeKind (edgeSource e)) (nodeKind (edgeTarget e))

/-- Every representable edge is well typed by construction. -/
theorem edge_is_well_typed {Agent Event Object Location Institution Concept : Type*}
    (e : TypedEdge Agent Event Object Location Institution Concept) :
    WellTypedEdge e := by
  cases e <;> simp [WellTypedEdge, edgeKind, edgeSource, edgeTarget,
    nodeKind, allowedEndpoints]

/-- If an edge reports semantic kind `causal`, both endpoints are event nodes. -/
theorem causal_edge_endpoints_are_events
    {Agent Event Object Location Institution Concept : Type*}
    (e : TypedEdge Agent Event Object Location Institution Concept)
    (h : edgeKind e = EdgeKind.causal) :
    nodeKind (edgeSource e) = NodeKind.event ∧
    nodeKind (edgeTarget e) = NodeKind.event := by
  cases e <;> simp [edgeKind, edgeSource, edgeTarget, nodeKind] at h ⊢

end NarrativeDynamics
