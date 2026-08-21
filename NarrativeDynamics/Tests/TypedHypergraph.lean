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

/-- Concrete instance only for stress-testing the generic type system:
shared grievance + coalition + leader + opportunity -> rebellion event. -/
def rebellionSignature : HyperedgeSignature where
  premiseKinds := [.concept, .institution, .agent, .event]
  conclusionKind := .event

def rebellionPremises :
    TypedPremises String String String String String String rebellionSignature.premiseKinds :=
  .cons (.concept "shared_grievance")
    (.cons (.institution "rebel_coalition")
      (.cons (.agent "leader")
        (.cons (.event "opportunity") .nil)))

def rebellionHyperedge :
    TypedHyperedge String String String String String String rebellionSignature where
  inputs := rebellionPremises
  output := .event "rebellion"

example :
    (erasePremises rebellionHyperedge.inputs).map nodeKind =
      [.concept, .institution, .agent, .event] := by
  exact hyperedge_premises_match_signature rebellionHyperedge

example : nodeKind (eraseKindNode rebellionHyperedge.output) = .event := by
  exact hyperedge_conclusion_matches_signature rebellionHyperedge
