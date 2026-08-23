import Mathlib

namespace NarrativeDynamics

open scoped BigOperators

universe u

/-- A finite Bayesian competition between mutually exclusive/exhaustive
interpretive hypotheses. `prior` and `likelihood` are kept separate so the
same evidence can reweight an existing hypothesis distribution. -/
structure HypothesisCompetition (ι : Type u) where
  prior : ι → ℝ
  likelihood : ι → ℝ

/-- Unnormalized Bayesian weight for one hypothesis. -/
noncomputable def hypothesisWeight {ι : Type u}
    (c : HypothesisCompetition ι) (i : ι) : ℝ :=
  c.prior i * c.likelihood i

/-- Total probability mass assigned to the observed evidence by the finite
hypothesis space. -/
noncomputable def evidenceMass {ι : Type u} [Fintype ι]
    (c : HypothesisCompetition ι) : ℝ :=
  ∑ i, hypothesisWeight c i

/-- Normalized posterior probability of one hypothesis after the evidence. -/
noncomputable def posterior {ι : Type u} [Fintype ι]
    (c : HypothesisCompetition ι) (i : ι) : ℝ :=
  hypothesisWeight c i / evidenceMass c

/-- Whenever the observed evidence has strictly positive total mass, the
finite posterior distribution is normalized. -/
theorem posterior_sum_one {ι : Type u} [Fintype ι]
    (c : HypothesisCompetition ι)
    (hmass : 0 < evidenceMass c) :
    (∑ i, posterior c i) = 1 := by
  classical
  unfold posterior
  rw [← Finset.sum_div]
  change evidenceMass c / evidenceMass c = 1
  exact div_self (ne_of_gt hmass)

/-- Nonnegative priors and likelihoods produce nonnegative posterior mass when
the evidence normalizer is positive. -/
theorem posterior_nonneg {ι : Type u} [Fintype ι]
    (c : HypothesisCompetition ι)
    (hprior : ∀ i, 0 ≤ c.prior i)
    (hlikelihood : ∀ i, 0 ≤ c.likelihood i)
    (hmass : 0 < evidenceMass c)
    (i : ι) :
    0 ≤ posterior c i := by
  unfold posterior hypothesisWeight
  exact div_nonneg
    (mul_nonneg (hprior i) (hlikelihood i))
    (le_of_lt hmass)

/-- With equal positive priors and a common evidence normalizer, the hypothesis
that makes the evidence more likely receives the larger posterior. -/
theorem higher_likelihood_wins_equal_prior {ι : Type u} [Fintype ι]
    (c : HypothesisCompetition ι)
    (i j : ι)
    (hprior : c.prior i = c.prior j)
    (hpriorPos : 0 < c.prior i)
    (hlikelihood : c.likelihood j < c.likelihood i)
    (hmass : 0 < evidenceMass c) :
    posterior c j < posterior c i := by
  unfold posterior hypothesisWeight
  apply (div_lt_div_iff_of_pos_right hmass).2
  have hpj : 0 < c.prior j := by
    rw [← hprior]
    exact hpriorPos
  rw [hprior]
  exact mul_lt_mul_of_pos_left hlikelihood hpj

end NarrativeDynamics
