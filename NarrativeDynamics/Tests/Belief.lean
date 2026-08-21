import NarrativeDynamics.Core.Belief

open NarrativeDynamics

example {prior likelihoodH likelihoodNotH : ℝ}
    (hprior0 : 0 < prior)
    (hprior1 : prior < 1)
    (hnot : 0 ≤ likelihoodNotH)
    (hevidence : likelihoodNotH < likelihoodH) :
    prior < bayesPosterior prior likelihoodH likelihoodNotH := by
  exact bayesPosterior_gt_prior_of_positive_evidence
    hprior0 hprior1 hnot hevidence
