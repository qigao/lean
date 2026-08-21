import NarrativeDynamics.Core.InterpretationCommitment

namespace NarrativeDynamics

open scoped BigOperators

universe uH uP uA uE uO uL uI uC uN

/-- Nonnegative grounded priors and likelihoods induce nonnegative posterior
mass for every interpretation when the common evidence has positive mass. -/
theorem grounded_posterior_nonneg
    {ι : Type uH} [Fintype ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hprior : ∀ k, 0 ≤ g.prior k)
    (hlikelihood : ∀ k, 0 ≤ (g.candidate k).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    (i : ι) :
    0 ≤ posterior g.competition i := by
  have hp : ∀ k, 0 ≤ g.competition.prior k := by
    intro k
    simpa [GroundedHypothesisSpace.competition] using hprior k
  have hl : ∀ k, 0 ≤ g.competition.likelihood k := by
    intro k
    simpa [GroundedHypothesisSpace.competition] using hlikelihood k
  exact posterior_nonneg g.competition hp hl hmass i

/-- Any two distinct coordinates of one normalized, nonnegative grounded
posterior distribution have total mass at most one. -/
theorem pair_grounded_posteriors_le_one
    {ι : Type uH} [Fintype ι] [DecidableEq ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (hprior : ∀ k, 0 ≤ g.prior k)
    (hlikelihood : ∀ k, 0 ≤ (g.candidate k).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    {i j : ι} (hneq : i ≠ j) :
    posterior g.competition i + posterior g.competition j ≤ 1 := by
  classical
  have hnonneg : ∀ k, 0 ≤ posterior g.competition k :=
    fun k => grounded_posterior_nonneg g hprior hlikelihood hmass k
  have hpair :
      (∑ k in ({i, j} : Finset ι), posterior g.competition k) ≤
        ∑ k, posterior g.competition k := by
    apply Finset.sum_le_sum_of_subset_of_nonneg
    · intro k hk
      simp
    · intro k _hk _hnotmem
      exact hnonneg k
  have htotal : (∑ k, posterior g.competition k) = 1 :=
    grounded_posterior_sum_one g hmass
  rw [htotal] at hpair
  simpa [hneq] using hpair

/-- At a majority threshold, two distinct grounded interpretations cannot both
be committed. Normalization and nonnegativity make commitment unique whenever
it exists; thresholds below one half deliberately do not have this guarantee. -/
theorem majority_interpretation_commitment_unique
    {ι : Type uH} [Fintype ι] [DecidableEq ι] {ProofId : Type uP}
    {Agent : Type uA} {Event : Type uE} {Object : Type uO}
    {Location : Type uL} {Institution : Type uI} {Concept : Type uC}
    {Node : Type uN}
    {w : WorldGraph Agent Event Node} {event : Event}
    (g : GroundedHypothesisSpace ι ProofId Agent Event Object Location Institution Concept Node w event)
    (threshold : ℝ)
    (hthreshold : (1 : ℝ) / 2 ≤ threshold)
    (hprior : ∀ k, 0 ≤ g.prior k)
    (hlikelihood : ∀ k, 0 ≤ (g.candidate k).likelihoodH)
    (hmass : 0 < evidenceMass g.competition)
    {i j : ι}
    (hi : InterpretationCommittedAt g.active threshold g i)
    (hj : InterpretationCommittedAt g.active threshold g j) :
    i = j := by
  by_contra hneq
  have hpair := pair_grounded_posteriors_le_one
    g hprior hlikelihood hmass hneq
  have hiHalf : (1 : ℝ) / 2 < posterior g.competition i :=
    lt_of_le_of_lt hthreshold hi.2
  have hjHalf : (1 : ℝ) / 2 < posterior g.competition j :=
    lt_of_le_of_lt hthreshold hj.2
  nlinarith

end NarrativeDynamics
