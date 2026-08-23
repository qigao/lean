import NarrativeDynamics.Core.Closure

namespace NarrativeDynamics

universe uP uA uE uO uL uI uC

/-- A provenance graph assigns each proof identifier a concluded fact, optional
rule origin, ordered support proofs, and a rank. Support edges must point from
lower rank to higher rank, which makes the proof graph acyclic by construction
once `ProvenanceInvariant` holds. -/
structure ProvenanceDAG
    (ProofId : Type uP)
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) where
  fact : ProofId → HNode Agent Event Object Location Institution Concept
  origin : ProofId → Option (SomeTypedHyperedge Agent Event Object Location Institution Concept)
  parents : ProofId → List ProofId
  rank : ProofId → Nat

/-- A support edge goes from a premise proof to the proof that consumes it. -/
def provenanceSupport
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (parent child : ProofId) : Prop :=
  parent ∈ g.parents child

/-- Well-formed provenance requires:
1. every support edge strictly increases proof rank;
2. asserted facts (`origin = none`) have no support parents;
3. derived proofs conclude exactly their rule output, and their ordered parent
   facts match exactly the rule's typed premise list. -/
def ProvenanceInvariant
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept) : Prop :=
  (∀ parent child, provenanceSupport g parent child → g.rank parent < g.rank child) ∧
  (∀ p, g.origin p = none → g.parents p = []) ∧
  (∀ p rule, g.origin p = some rule →
    g.fact p = eraseKindNode rule.edge.output ∧
      (g.parents p).map g.fact = erasePremises rule.edge.inputs)

/-- Every direct provenance support edge strictly raises rank. -/
theorem provenanceSupport_rank_lt
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    {parent child : ProofId}
    (hs : provenanceSupport g parent child) :
    g.rank parent < g.rank child := by
  exact h.1 parent child hs

/-- Rank increases strictly along every nonempty provenance support path. -/
theorem provenancePath_rank_increasing
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    {source target : ProofId}
    (path : Relation.TransGen (provenanceSupport g) source target) :
    g.rank source < g.rank target := by
  induction path with
  | single hab =>
      exact provenanceSupport_rank_lt g h hab
  | tail hab hbc ih =>
      exact Nat.lt_trans ih (provenanceSupport_rank_lt g h hbc)

/-- A well-formed provenance graph cannot contain a directed support cycle. -/
theorem provenance_acyclic
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId) :
    ¬ Relation.TransGen (provenanceSupport g) p p := by
  intro cycle
  have hrank : g.rank p < g.rank p := provenancePath_rank_increasing g h cycle
  exact Nat.lt_irrefl _ hrank

/-- A derived proof record must agree exactly with the typed rule that produced
it: same conclusion and same ordered premise facts. -/
theorem derived_provenance_matches_rule
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId)
    (rule : SomeTypedHyperedge Agent Event Object Location Institution Concept)
    (horigin : g.origin p = some rule) :
    g.fact p = eraseKindNode rule.edge.output ∧
      (g.parents p).map g.fact = erasePremises rule.edge.inputs := by
  exact h.2.2 p rule horigin

/-- An asserted proof node is a root in the provenance graph. -/
theorem asserted_provenance_has_no_parents
    {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId)
    (horigin : g.origin p = none) :
    g.parents p = [] := by
  exact h.2.1 p horigin

end NarrativeDynamics
