import NarrativeDynamics.Core.GroundedGoalSoftmax

open NarrativeDynamics

universe uG uH uP uA uE uO uL uI uC uN

/-- A finite nonempty goal softmax is a normalized choice distribution. -/
example
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (witness : Goal) :
    (∑ goal, finiteSoftmax β score goal) = 1 := by
  exact finiteSoftmax_sum_one β score witness

/-- Positive inverse temperature preserves strict score ordering inside one
finite softmax distribution. -/
example
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (goalA goalB : Goal)
    (hβ : 0 < β) (hscore : score goalA < score goalB) :
    finiteSoftmax β score goalA < finiteSoftmax β score goalB := by
  exact finiteSoftmax_lt_of_score_lt β score goalA goalB hβ hscore

/-- A covariance-driven score reversal becomes a softmax preference reversal
within an arbitrary finite goal set. -/
example
    {Goal : Type uG} [Fintype Goal]
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (goalA goalB : Goal)
    (instrumentality : Goal → ι → ℝ)
    (cost risk : Goal → ℝ)
    (β pressure : ℝ)
    (hβ : 0 < β)
    (hpressure : 0 < pressure)
    (hmass : 0 < evidenceMass g.competition)
    (hprior :
      groundedPriorGoalScoreFamily pressure g instrumentality cost risk goalA <
        groundedPriorGoalScoreFamily pressure g instrumentality cost risk goalB)
    (hmargin :
      groundedPriorGoalScoreFamily pressure g instrumentality cost risk goalB -
          groundedPriorGoalScoreFamily pressure g instrumentality cost risk goalA <
        pressure *
          (groundedLikelihoodInstrumentalityCovariance g (instrumentality goalA) -
            groundedLikelihoodInstrumentalityCovariance g (instrumentality goalB)) /
          evidenceMass g.competition) :
    finiteSoftmax β
        (groundedPriorGoalScoreFamily pressure g instrumentality cost risk) goalA <
      finiteSoftmax β
        (groundedPriorGoalScoreFamily pressure g instrumentality cost risk) goalB ∧
    groundedGoalChoiceProbability β pressure g instrumentality cost risk goalB <
      groundedGoalChoiceProbability β pressure g instrumentality cost risk goalA := by
  exact grounded_covariance_reversal_induces_softmax_preference_reversal
    g goalA goalB instrumentality cost risk β pressure
    hβ hpressure hmass hprior hmargin
