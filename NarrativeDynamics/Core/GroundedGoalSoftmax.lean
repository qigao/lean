import NarrativeDynamics.Core.GroundedGoalRankingReversal

namespace NarrativeDynamics

open scoped BigOperators

universe uG uH uP uA uE uO uL uI uC uN

/-- Finite softmax choice probability over an arbitrary goal type. -/
noncomputable def finiteSoftmax
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (goal : Goal) : ℝ :=
  Real.exp (β * score goal) /
    ∑ rival, Real.exp (β * score rival)

/-- The finite-softmax denominator is strictly positive whenever a goal
witness exists. -/
theorem finiteSoftmax_denominator_pos
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (witness : Goal) :
    0 < ∑ goal, Real.exp (β * score goal) := by
  classical
  apply Finset.sum_pos
  · intro goal _
    exact Real.exp_pos (β * score goal)
  · exact ⟨witness, Finset.mem_univ witness⟩

/-- A finite nonempty softmax distribution is normalized. -/
theorem finiteSoftmax_sum_one
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (witness : Goal) :
    (∑ goal, finiteSoftmax β score goal) = 1 := by
  classical
  unfold finiteSoftmax
  rw [← Finset.sum_div]
  change
    (∑ goal, Real.exp (β * score goal)) /
        (∑ goal, Real.exp (β * score goal)) = 1
  exact div_self (ne_of_gt (finiteSoftmax_denominator_pos β score witness))

/-- At positive inverse temperature, strict score order is exactly preserved
inside one finite softmax distribution. -/
theorem finiteSoftmax_lt_of_score_lt
    {Goal : Type uG} [Fintype Goal]
    (β : ℝ) (score : Goal → ℝ) (goalA goalB : Goal)
    (hβ : 0 < β) (hscore : score goalA < score goalB) :
    finiteSoftmax β score goalA < finiteSoftmax β score goalB := by
  unfold finiteSoftmax
  have hden : 0 < ∑ goal, Real.exp (β * score goal) :=
    finiteSoftmax_denominator_pos β score goalA
  apply (div_lt_div_iff_of_pos_right hden).2
  exact (Real.exp_lt_exp).2 (mul_lt_mul_of_pos_left hscore hβ)

/-- Goal-score family before the common evidence reweights the finite grounded
hypothesis space. -/
noncomputable def groundedPriorGoalScoreFamily
    {Goal : Type uG}
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (pressure : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : Goal → ι → ℝ)
    (cost risk : Goal → ℝ)
    (goal : Goal) : ℝ :=
  groundedPriorEpistemicGoalScore pressure (cost goal) (risk goal)
    g (instrumentality goal)

/-- Goal-score family after the admitted evidence has reweighted the grounded
hypothesis space. -/
noncomputable def groundedGoalScoreFamily
    {Goal : Type uG}
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (pressure : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : Goal → ι → ℝ)
    (cost risk : Goal → ℝ)
    (goal : Goal) : ℝ :=
  groundedEpistemicGoalScore pressure (cost goal) (risk goal)
    g (instrumentality goal)

/-- Posterior softmax probability for one goal after finite grounded epistemic
scoring. -/
noncomputable def groundedGoalChoiceProbability
    {Goal : Type uG} [Fintype Goal]
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (β pressure : ℝ)
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (instrumentality : Goal → ι → ℝ)
    (cost risk : Goal → ℝ)
    (goal : Goal) : ℝ :=
  finiteSoftmax β
    (groundedGoalScoreFamily pressure g instrumentality cost risk)
    goal

/-- A covariance-driven score reversal induces a softmax preference reversal
inside any finite goal set. The proof first ranks the prior scores, then uses
the grounded covariance margin to reverse the posterior scores, and finally
uses positive-temperature softmax order preservation. -/
theorem grounded_covariance_reversal_induces_softmax_preference_reversal
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
  have hprior' :
      groundedPriorEpistemicGoalScore pressure (cost goalA) (risk goalA)
          g (instrumentality goalA) <
        groundedPriorEpistemicGoalScore pressure (cost goalB) (risk goalB)
          g (instrumentality goalB) := by
    simpa [groundedPriorGoalScoreFamily] using hprior
  have hmargin' :
      groundedPriorEpistemicGoalScore pressure (cost goalB) (risk goalB)
            g (instrumentality goalB) -
          groundedPriorEpistemicGoalScore pressure (cost goalA) (risk goalA)
            g (instrumentality goalA) <
        pressure *
          (groundedLikelihoodInstrumentalityCovariance g (instrumentality goalA) -
            groundedLikelihoodInstrumentalityCovariance g (instrumentality goalB)) /
          evidenceMass g.competition := by
    simpa [groundedPriorGoalScoreFamily] using hmargin
  have hpost :
      groundedEpistemicGoalScore pressure (cost goalB) (risk goalB)
          g (instrumentality goalB) <
        groundedEpistemicGoalScore pressure (cost goalA) (risk goalA)
          g (instrumentality goalA) :=
    covariance_advantage_reverses_unequal_prior_goal_ranking
      g (instrumentality goalA) (instrumentality goalB)
      pressure (cost goalA) (risk goalA) (cost goalB) (risk goalB)
      hpressure hmass hprior' hmargin'
  constructor
  · exact finiteSoftmax_lt_of_score_lt β
      (groundedPriorGoalScoreFamily pressure g instrumentality cost risk)
      goalA goalB hβ hprior
  · unfold groundedGoalChoiceProbability
    apply finiteSoftmax_lt_of_score_lt β
      (groundedGoalScoreFamily pressure g instrumentality cost risk)
      goalB goalA hβ
    simpa [groundedGoalScoreFamily] using hpost

end NarrativeDynamics
