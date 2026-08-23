import NarrativeDynamics.Core.TypedInference

namespace NarrativeDynamics

universe uA uE uO uL uI uC

/-- A fact set is a predicate over the heterogeneous narrative node universe. -/
abbrev FactSet
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) :=
  HNode Agent Event Object Location Institution Concept → Prop

/-- Pointwise inclusion between fact predicates. -/
def FactSubset
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (p q : FactSet Agent Event Object Location Institution Concept) : Prop :=
  ∀ n, p n → q n

/-- Package a typed hyperedge with its dependent signature hidden existentially,
so heterogeneous rules can live in one ordinary list. -/
structure SomeTypedHyperedge
    (Agent : Type uA) (Event : Type uE) (Object : Type uO)
    (Location : Type uL) (Institution : Type uI) (Concept : Type uC) where
  sig : HyperedgeSignature
  edge : TypedHyperedge Agent Event Object Location Institution Concept sig

/-- Premise satisfaction is monotone: adding facts cannot invalidate an already
satisfied typed premise list. -/
theorem premisesHold_mono
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (p q : FactSet Agent Event Object Location Institution Concept)
    (hsub : FactSubset p q)
    {ks : List NodeKind}
    (premises : TypedPremises Agent Event Object Location Institution Concept ks) :
    PremisesHold p premises → PremisesHold q premises := by
  induction premises with
  | nil =>
      intro _
      trivial
  | cons head tail ih =>
      intro hp
      exact ⟨hsub (eraseKindNode head) hp.1, ih hp.2⟩

/-- Applying one typed hyperedge is monotone in its input fact set. -/
theorem applyHyperedge_mono
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (p q : FactSet Agent Event Object Location Institution Concept)
    (hsub : FactSubset p q)
    {sig : HyperedgeSignature}
    (edge : TypedHyperedge Agent Event Object Location Institution Concept sig) :
    FactSubset (applyHyperedge p edge) (applyHyperedge q edge) := by
  intro n hn
  rcases hn with hold | hnew
  · left
    exact hsub n hold
  · right
    exact ⟨premisesHold_mono p q hsub edge.inputs hnew.1, hnew.2⟩

/-- Apply one existentially packaged rule. -/
def applySomeHyperedge
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : FactSet Agent Event Object Location Institution Concept)
    (rule : SomeTypedHyperedge Agent Event Object Location Institution Concept) :
    FactSet Agent Event Object Location Institution Concept :=
  applyHyperedge holds rule.edge

/-- One forward-chaining pass applies rules sequentially in list order. Facts
created by an earlier rule are available to later rules in the same pass. -/
def applyRuleSetOnce
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : FactSet Agent Event Object Location Institution Concept) :
    List (SomeTypedHyperedge Agent Event Object Location Institution Concept) →
      FactSet Agent Event Object Location Institution Concept
  | [] => holds
  | rule :: rest => applyRuleSetOnce (applySomeHyperedge holds rule) rest

/-- One rule-set pass never retracts facts. -/
theorem applyRuleSetOnce_extensive
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (holds : FactSet Agent Event Object Location Institution Concept)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept)) :
    FactSubset holds (applyRuleSetOnce holds rules) := by
  induction rules generalizing holds with
  | nil =>
      intro n hn
      exact hn
  | cons rule rest ih =>
      intro n hn
      apply ih (applySomeHyperedge holds rule)
      exact applyHyperedge_preserves_fact holds rule.edge n hn

/-- One rule-set pass is monotone in its input facts. -/
theorem applyRuleSetOnce_mono
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (p q : FactSet Agent Event Object Location Institution Concept)
    (hsub : FactSubset p q)
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept)) :
    FactSubset (applyRuleSetOnce p rules) (applyRuleSetOnce q rules) := by
  induction rules generalizing p q with
  | nil =>
      exact hsub
  | cons rule rest ih =>
      apply ih
      exact applyHyperedge_mono p q hsub rule.edge

/-- Apply the rule-set exactly `n` times. This is deliberately finite iteration;
no generic termination or fixed-point theorem is assumed here. -/
def iterateRuleSet
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept)) :
    Nat → FactSet Agent Event Object Location Institution Concept →
      FactSet Agent Event Object Location Institution Concept
  | 0, holds => holds
  | n + 1, holds => applyRuleSetOnce (iterateRuleSet rules n holds) rules

/-- Successive forward-chaining iterations form an ascending chain. -/
theorem iterateRuleSet_ascending
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept))
    (holds : FactSet Agent Event Object Location Institution Concept)
    (n : Nat) :
    FactSubset (iterateRuleSet rules n holds) (iterateRuleSet rules (n + 1) holds) := by
  simpa [iterateRuleSet] using
    (applyRuleSetOnce_extensive (iterateRuleSet rules n holds) rules)

/-- Every finite iteration preserves all initial facts. -/
theorem iterateRuleSet_extensive
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    (rules : List (SomeTypedHyperedge Agent Event Object Location Institution Concept))
    (holds : FactSet Agent Event Object Location Institution Concept)
    (n : Nat) :
    FactSubset holds (iterateRuleSet rules n holds) := by
  induction n with
  | zero =>
      intro node hn
      exact hn
  | succ n ih =>
      intro node hn
      have hstep : iterateRuleSet rules n holds node := ih node hn
      exact applyRuleSetOnce_extensive (iterateRuleSet rules n holds) rules node hstep

end NarrativeDynamics
