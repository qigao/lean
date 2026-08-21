import NarrativeDynamics.Core.HypothesisCompetition

open NarrativeDynamics
open scoped BigOperators

example {ι : Type*} [Fintype ι]
    (c : HypothesisCompetition ι)
    (hmass : 0 < evidenceMass c) :
    (∑ i, posterior c i) = 1 := by
  exact posterior_sum_one c hmass

example {ι : Type*} [Fintype ι]
    (c : HypothesisCompetition ι)
    (hprior : ∀ i, 0 ≤ c.prior i)
    (hlikelihood : ∀ i, 0 ≤ c.likelihood i)
    (hmass : 0 < evidenceMass c)
    (i : ι) :
    0 ≤ posterior c i := by
  exact posterior_nonneg c hprior hlikelihood hmass i

example {ι : Type*} [Fintype ι]
    (c : HypothesisCompetition ι)
    (i j : ι)
    (hprior : c.prior i = c.prior j)
    (hpriorPos : 0 < c.prior i)
    (hlikelihood : c.likelihood j < c.likelihood i)
    (hmass : 0 < evidenceMass c) :
    posterior c j < posterior c i := by
  exact higher_likelihood_wins_equal_prior
    c i j hprior hpriorPos hlikelihood hmass
