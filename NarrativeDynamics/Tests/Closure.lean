import NarrativeDynamics.Core.Closure

open NarrativeDynamics

example {Agent Event Object Location Institution Concept : Type*}
    (p q : FactSet Agent Event Object Location Institution Concept)
    (hsub : FactSubset p q)
    {ks : List NodeKind}
    (premises : TypedPremises Agent Event Object Location Institution Concept ks) :
    PremisesHold p premises → PremisesHold q premises := by
  exact premisesHold_mono p q hsub premises

example {Agent Event Object Location Institution Concept : Type*}
    (holds : FactSet Agent Event Object Location Institution Concept)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept)) :
    FactSubset holds (applyRuleSetOnce holds rules) := by
  exact applyRuleSetOnce_extensive holds rules

example {Agent Event Object Location Institution Concept : Type*}
    (p q : FactSet Agent Event Object Location Institution Concept)
    (hsub : FactSubset p q)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept)) :
    FactSubset (applyRuleSetOnce p rules) (applyRuleSetOnce q rules) := by
  exact applyRuleSetOnce_mono p q hsub rules

example {Agent Event Object Location Institution Concept : Type*}
    (holds : FactSet Agent Event Object Location Institution Concept)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept))
    (n : Nat) :
    FactSubset (iterateRuleSet rules n holds) (iterateRuleSet rules (n + 1) holds) := by
  exact iterateRuleSet_ascending rules holds n

example {Agent Event Object Location Institution Concept : Type*}
    (holds : FactSet Agent Event Object Location Institution Concept)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept))
    (n : Nat) :
    FactSubset holds (iterateRuleSet rules n holds) := by
  exact iterateRuleSet_extensive rules holds n
