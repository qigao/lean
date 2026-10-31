import NarrativeDynamics.Core.NativeTrustWeightedTell

open NarrativeDynamics.NativeTrustWeightedTell

-- R2.M5g.3 finite source model: positive evidence mass and normalized
-- posterior, no fictitious Agent action or general human influence theorem.
example (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 < effectiveLikelihood t (18 / 25 : ℝ) +
      effectiveLikelihood t (2 / 25 : ℝ) := by
  exact evidenceMass_pos t h0 h1

example (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    approvedPosterior t + deniedPosterior t = 1 := by
  exact posterior_sum_one t h0 h1

example (s t : ℝ) (hs : 0 ≤ s) (hst : s < t) (ht : t ≤ 1) :
    approvedPosterior s < approvedPosterior t := by
  exact approvedPosterior_strict_mono s t hs hst ht

example (t likelihood : ℝ) (hp : 0 < effectiveLikelihood t likelihood) :
    effectiveLikelihood t likelihood /
      (effectiveLikelihood t likelihood +
        effectiveLikelihood t likelihood) = (1 / 2 : ℝ) := by
  exact equal_likelihood_uninformative t likelihood hp

example : approvedPosterior 0 = (1 / 2 : ℝ) := posterior_trust_zero
example : approvedPosterior (3 / 10 : ℝ) = (229 / 410 : ℝ) :=
  posterior_trust_three_tenths
example : approvedPosterior (3 / 5 : ℝ) = (13 / 20 : ℝ) :=
  posterior_trust_three_fifths
example : approvedPosterior 1 = (9 / 10 : ℝ) := posterior_trust_one
