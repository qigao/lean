import NarrativeDynamics.Core.TypedHypergraph

open NarrativeDynamics

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    (erasePremises h.inputs).map nodeKind = sig.premiseKinds := by
  exact hyperedge_premises_match_signature h

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    nodeKind (eraseKindNode h.output) = sig.conclusionKind := by
  exact hyperedge_conclusion_matches_signature h

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    (erasePremises h.inputs).length = sig.premiseKinds.length := by
  exact hyperedge_arity_matches_signature h
