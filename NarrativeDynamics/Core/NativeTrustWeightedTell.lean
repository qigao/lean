import Mathlib

namespace NarrativeDynamics.NativeTrustWeightedTell

/-!
R2.M5g.3: theorem about the declared ORIGINAL V19 direct-EXACT-TELL
two-hypothesis Bayesian likelihood update, not a social-network law.

The original source uses the interpolation (1 - trust) + trust * likelihood.
Here the source-owned neutral-audit model's target symbol has likelihood
18/25 for approved and 2/25 for denied; Bob's source prior is 1/2 each.
The equal priors cancel in the normalized two-coordinate posterior.
The V19/SQLite source engine remains authoritative for actual admissions,
action values, selected actions, events, trust and provenance.
-/

/-- This is the existing source cognitive likelihood interpolation, not
a new planner, estimator, social state or inference runtime. -/
def effectiveLikelihood (trust likelihood : ℝ) : ℝ :=
  (1 - trust) + trust * likelihood

/-- Equal-prior, source-declared approved posterior for a newly admitted
EXACT external TELL (18/25 approved vs 2/25 denied). -/
def approvedPosterior (trust : ℝ) : ℝ :=
  effectiveLikelihood trust (18 / 25 : ℝ) /
    (effectiveLikelihood trust (18 / 25 : ℝ) +
      effectiveLikelihood trust (2 / 25 : ℝ))

def deniedPosterior (trust : ℝ) : ℝ :=
  effectiveLikelihood trust (2 / 25 : ℝ) /
    (effectiveLikelihood trust (18 / 25 : ℝ) +
      effectiveLikelihood trust (2 / 25 : ℝ))

/-- The model-local evidence denominator is strictly positive for every
valid social trust value. -/
theorem evidenceMass_pos (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    0 < effectiveLikelihood t (18 / 25 : ℝ) +
      effectiveLikelihood t (2 / 25 : ℝ) := by
  unfold effectiveLikelihood
  nlinarith

/-- The two posterior hypothesis coordinates remain normalized.
No source admission or percept provenance is manufactured by this lemma. -/
theorem posterior_sum_one (t : ℝ) (h0 : 0 ≤ t) (h1 : t ≤ 1) :
    approvedPosterior t + deniedPosterior t = 1 := by
  have hm : effectiveLikelihood t (18 / 25 : ℝ) +
      effectiveLikelihood t (2 / 25 : ℝ) ≠ 0 :=
    ne_of_gt (evidenceMass_pos t h0 h1)
  unfold approvedPosterior deniedPosterior
  rw [← add_div]
  exact div_self hm

/-- Closed form for this specific declared V19 likelihood pair, not for
arbitrary speakers, network structures or cognitive models. -/
theorem approvedPosterior_closed (t : ℝ) :
    approvedPosterior t =
      (1 - (7 / 25 : ℝ) * t) / (2 - (6 / 5 : ℝ) * t) := by
  have hn : effectiveLikelihood t (18 / 25 : ℝ) =
      1 - (7 / 25 : ℝ) * t := by
    unfold effectiveLikelihood
    ring
  have hd : effectiveLikelihood t (18 / 25 : ℝ) +
      effectiveLikelihood t (2 / 25 : ℝ) =
      2 - (6 / 5 : ℝ) * t := by
    unfold effectiveLikelihood
    ring
  unfold approvedPosterior
  rw [hd, hn]

/-- The registered finite software update has a strictly increasing
approved posterior when trust increases, provided both trust inputs
are in [0,1]. This says nothing about selected actions or society. -/
theorem approvedPosterior_strict_mono
    (s t : ℝ) (hs : 0 ≤ s) (hst : s < t) (ht : t ≤ 1) :
    approvedPosterior s < approvedPosterior t := by
  have hs1 : s ≤ 1 := le_trans (le_of_lt hst) ht
  have ht0 : 0 ≤ t := le_trans hs (le_of_lt hst)
  have hds : 0 < 2 - (6 / 5 : ℝ) * s := by
    nlinarith
  have hdt : 0 < 2 - (6 / 5 : ℝ) * t := by
    nlinarith
  rw [approvedPosterior_closed s, approvedPosterior_closed t]
  apply (div_lt_div_iff₀ hds hdt).2
  nlinarith

/-- No information when the likelihoods are equal and positive.
Trust cannot move a 1/2 prior under a nondiscriminating testimony. -/
theorem equal_likelihood_uninformative
    (t likelihood : ℝ) (hpositive : 0 < effectiveLikelihood t likelihood) :
    effectiveLikelihood t likelihood /
      (effectiveLikelihood t likelihood +
       effectiveLikelihood t likelihood) = (1 / 2 : ℝ) := by
  have hn : effectiveLikelihood t likelihood ≠ 0 := ne_of_gt hpositive
  have hd : effectiveLikelihood t likelihood +
      effectiveLikelihood t likelihood =
      (2 : ℝ) * effectiveLikelihood t likelihood := by
    ring
  rw [hd]
  apply (div_eq_iff (mul_ne_zero (by norm_num : (2 : ℝ) ≠ 0) hn)).2
  ring

theorem posterior_trust_zero : approvedPosterior 0 = (1 / 2 : ℝ) := by
  norm_num [approvedPosterior, effectiveLikelihood]

theorem posterior_trust_three_tenths :
    approvedPosterior (3 / 10 : ℝ) = (229 / 410 : ℝ) := by
  norm_num [approvedPosterior, effectiveLikelihood]

theorem posterior_trust_three_fifths :
    approvedPosterior (3 / 5 : ℝ) = (13 / 20 : ℝ) := by
  norm_num [approvedPosterior, effectiveLikelihood]

theorem posterior_trust_one : approvedPosterior 1 = (9 / 10 : ℝ) := by
  norm_num [approvedPosterior, effectiveLikelihood]

end NarrativeDynamics.NativeTrustWeightedTell
