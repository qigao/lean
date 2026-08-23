import NarrativeDynamics.Core.TypedHypergraph

namespace NarrativeDynamics

universe uA uE uO uL uI uC

/-- All premises of a typed hyperedge are true in the current fact predicate. -/
def PremisesHold
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    {ks : List NodeKind} :
    TypedPremises Agent Event Object Location Institution Concept ks → Prop
  | .nil => True
  | .cons head tail =>
      holds (eraseKindNode head) ∧ PremisesHold holds tail

/-- Apply one typed hyperedge to a fact predicate. Existing facts remain true;
the conclusion is added exactly when every typed premise holds. -/
def applyHyperedge
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    HNode Agent Event Object Location Institution Concept → Prop :=
  fun n =>
    holds n ∨
      (PremisesHold holds h.inputs ∧ n = eraseKindNode h.output)

/-- If all premises hold, firing the hyperedge makes its typed conclusion true. -/
theorem applyHyperedge_derives_conclusion
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (hp : PremisesHold holds h.inputs) :
    applyHyperedge holds h (eraseKindNode h.output) := by
  right
  exact ⟨hp, rfl⟩

/-- Hyperedge application is monotone in the sense that no existing fact is
lost when a rule fires or fails to fire. -/
theorem applyHyperedge_preserves_fact
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (n : HNode Agent Event Object Location Institution Concept)
    (hn : holds n) :
    applyHyperedge holds h n := by
  left
  exact hn

/-- If a premise is missing and the conclusion was not already true, applying
this hyperedge cannot create the conclusion spontaneously. -/
theorem applyHyperedge_no_spontaneous_conclusion
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    {sig : HyperedgeSignature}
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (hmissing : ¬ PremisesHold holds h.inputs)
    (hout : ¬ holds (eraseKindNode h.output)) :
    ¬ applyHyperedge holds h (eraseKindNode h.output) := by
  intro happ
  rcases happ with hold | hnew
  · exact hout hold
  · exact hmissing hnew.1

end NarrativeDynamics
