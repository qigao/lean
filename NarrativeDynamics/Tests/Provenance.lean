import NarrativeDynamics.Core.Provenance

open NarrativeDynamics

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    {parent child : ProofId}
    (hs : provenanceSupport g parent child) :
    g.rank parent < g.rank child := by
  exact provenanceSupport_rank_lt g h hs

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId) :
    ¬ Relation.TransGen (provenanceSupport g) p p := by
  exact provenance_acyclic g h p

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId)
    (rule : SomeTypedHyperedge Agent Event Object Location Institution Concept)
    (horigin : g.origin p = some rule) :
    g.fact p = eraseKindNode rule.edge.output ∧
      (g.parents p).map g.fact = erasePremises rule.edge.inputs := by
  exact derived_provenance_matches_rule g h p rule horigin

example {ProofId Agent Event Object Location Institution Concept : Type*}
    (g : ProvenanceDAG ProofId Agent Event Object Location Institution Concept)
    (h : ProvenanceInvariant g)
    (p : ProofId)
    (horigin : g.origin p = none) :
    g.parents p = [] := by
  exact asserted_provenance_has_no_parents g h p horigin
