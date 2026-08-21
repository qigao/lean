import NarrativeDynamics.Core.TypedInference

open NarrativeDynamics

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (hp : PremisesHold holds h.inputs) :
    applyHyperedge holds h (eraseKindNode h.output) := by
  exact applyHyperedge_derives_conclusion holds h hp

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (n : HNode Agent Event Object Location Institution Concept)
    (hn : holds n) :
    applyHyperedge holds h n := by
  exact applyHyperedge_preserves_fact holds h n hn

example {Agent Event Object Location Institution Concept : Type*}
    {sig : HyperedgeSignature}
    (holds : HNode Agent Event Object Location Institution Concept → Prop)
    (h : TypedHyperedge Agent Event Object Location Institution Concept sig)
    (hmissing : ¬ PremisesHold holds h.inputs)
    (hout : ¬ holds (eraseKindNode h.output)) :
    ¬ applyHyperedge holds h (eraseKindNode h.output) := by
  exact applyHyperedge_no_spontaneous_conclusion holds h hmissing hout
