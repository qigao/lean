import Mathlib

namespace NarrativeDynamics

/-- Bayesian posterior for a binary hypothesis after observing evidence with
likelihoods under the hypothesis and its negation. -/
noncomputable def bayesPosterior
    (prior likelihoodH likelihoodNotH : ℝ) : ℝ :=
  (prior * likelihoodH) /
    (prior * likelihoodH + (1 - prior) * likelihoodNotH)

/-- Positive evidence raises the posterior: if the evidence is strictly more
likely under `H` than under `¬H`, and the prior is nondegenerate, then the
posterior probability of `H` is strictly greater than the prior. -/
theorem bayesPosterior_gt_prior_of_positive_evidence
    {prior likelihoodH likelihoodNotH : ℝ}
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    prior < bayesPosterior prior likelihoodH likelihoodNotH := by
  unfold bayesPosterior
  have hlikeH : 0 < likelihoodH :=
    lt_of_le_of_lt hnot hevidence
  have hq : 0 < 1 - prior := sub_pos.mpr hprior1
  have hfirst : 0 < prior * likelihoodH :=
    mul_pos hprior0 hlikeH
  have hsecond : 0 ≤ (1 - prior) * likelihoodNotH :=
    mul_nonneg (le_of_lt hq) hnot
  have hden : 0 < prior * likelihoodH + (1 - prior) * likelihoodNotH := by
    exact add_pos_of_pos_of_nonneg hfirst hsecond
  apply (lt_div_iff₀ hden).2
  have hdiff : 0 < likelihoodH - likelihoodNotH :=
    sub_pos.mpr hevidence
  have hprod : 0 < prior * (1 - prior) * (likelihoodH - likelihoodNotH) :=
    mul_pos (mul_pos hprior0 hq) hdiff
  have hid :
      prior * likelihoodH -
          prior * (prior * likelihoodH + (1 - prior) * likelihoodNotH) =
        prior * (1 - prior) * (likelihoodH - likelihoodNotH) := by
    ring
  nlinarith

end NarrativeDynamics
