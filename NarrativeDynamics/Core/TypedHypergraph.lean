import NarrativeDynamics.Core.TypedGraph

namespace NarrativeDynamics

/-- A heterogeneous node indexed by its `NodeKind`. The index carries the
endpoint type at compile time. -/
inductive KindNode (Agent Event Object Location Institution Concept : Type*) : NodeKind → Type where
  | agent : Agent → KindNode Agent Event Object Location Institution Concept .agent
  | event : Event → KindNode Agent Event Object Location Institution Concept .event
  | object : Object → KindNode Agent Event Object Location Institution Concept .object
  | location : Location → KindNode Agent Event Object Location Institution Concept .location
  | institution : Institution → KindNode Agent Event Object Location Institution Concept .institution
  | concept : Concept → KindNode Agent Event Object Location Institution Concept .concept

/-- Erase the dependent index while preserving the ordinary heterogeneous node. -/
def eraseKindNode {Agent Event Object Location Institution Concept : Type*}
    {k : NodeKind} :
    KindNode Agent Event Object Location Institution Concept k →
      HNode Agent Event Object Location Institution Concept
  | .agent a => .agent a
  | .event e => .event e
  | .object o => .object o
  | .location l => .location l
  | .institution i => .institution i
  | .concept c => .concept c

/-- Erasing an indexed node recovers exactly the kind encoded in its type. -/
theorem eraseKindNode_kind {Agent Event Object Location Institution Concept : Type*}
    {k : NodeKind}
    (n : KindNode Agent Event Object Location Institution Concept k) :
    nodeKind (eraseKindNode n) = k := by
  cases n <;> rfl

/-- A dependent list of premises whose type index is the exact sequence of
node kinds required by a hyperedge signature. -/
inductive TypedPremises (Agent Event Object Location Institution Concept : Type*) :
    List NodeKind → Type where
  | nil : TypedPremises Agent Event Object Location Institution Concept []
  | cons {k : NodeKind} {ks : List NodeKind} :
      KindNode Agent Event Object Location Institution Concept k →
      TypedPremises Agent Event Object Location Institution Concept ks →
      TypedPremises Agent Event Object Location Institution Concept (k :: ks)

/-- Erase dependent premises into ordinary heterogeneous nodes. -/
def erasePremises {Agent Event Object Location Institution Concept : Type*}
    {ks : List NodeKind} :
    TypedPremises Agent Event Object Location Institution Concept ks →
      List (HNode Agent Event Object Location Institution Concept)
  | .nil => []
  | .cons head tail => eraseKindNode head :: erasePremises tail

/-- Erasing a typed premise list preserves its complete kind sequence. -/
theorem erasePremises_kinds {Agent Event Object Location Institution Concept : Type*}
    {ks : List NodeKind}
    (ps : TypedPremises Agent Event Object Location Institution Concept ks) :
    (erasePremises ps).map nodeKind = ks := by
  induction ps with
  | nil => rfl
  | cons head tail ih =>
      simp [erasePremises, eraseKindNode_kind, ih]

/-- Type signature of a causal hyperedge: an ordered sequence of premise node
kinds and exactly one conclusion node kind. -/
structure HyperedgeSignature where
  premiseKinds : List NodeKind
  conclusionKind : NodeKind
  deriving DecidableEq, Repr

/-- A hyperedge whose premise arity/kinds and conclusion kind are enforced by
the dependent types in its signature. -/
structure TypedHyperedge
    (Agent Event Object Location Institution Concept : Type*)
    (sig : HyperedgeSignature) where
  inputs : TypedPremises Agent Event Object Location Institution Concept sig.premiseKinds
  output : KindNode Agent Event Object Location Institution Concept sig.conclusionKind

/-- Every representable hyperedge has exactly the premise-kind sequence stated
by its signature. -/
theorem hyperedge_premises_match_signature
    {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    (erasePremises h.inputs).map nodeKind = sig.premiseKinds := by
  exact erasePremises_kinds h.inputs

/-- Every representable hyperedge has exactly the conclusion kind stated by
its signature. -/
theorem hyperedge_conclusion_matches_signature
    {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    nodeKind (eraseKindNode h.output) = sig.conclusionKind := by
  exact eraseKindNode_kind h.output

/-- The number of represented premises is exactly the signature arity. -/
theorem hyperedge_arity_matches_signature
    {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    (erasePremises h.inputs).length = sig.premiseKinds.length := by
  have hk := hyperedge_premises_match_signature h
  exact List.length_map (f := nodeKind) (erasePremises h.inputs) ▸ congrArg List.length hk

end NarrativeDynamics
