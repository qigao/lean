import NarrativeDynamics.Core.FiniteTimeVaryingConsensus

open NarrativeDynamics
open NarrativeDynamics.FiniteConsensus
open NarrativeDynamics.FiniteTimeVaryingConsensus
open Filter Topology

private def K0 : Kernel (Fin 2) := !![3/4, 1/4; 1/4, 3/4]
private def K1 : Kernel (Fin 2) := !![2/3, 1/3; 1/3, 2/3]
private def sched : KernelSchedule (Fin 2) := fun k => if k % 2 = 0 then K0 else K1
private def x0 : Fin 2 → Rat := ![1, 0]

private theorem K0_averaging : AveragingKernel K0 := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num [K0]
  · intro i
    fin_cases i <;> norm_num [K0]

private theorem K1_averaging : AveragingKernel K1 := by
  constructor
  · intro i j
    fin_cases i <;> fin_cases j <;> norm_num [K1]
  · intro i
    fin_cases i <;> norm_num [K1]

example :
    applyKernel (windowKernel sched 0 2) x0 =
      varyingTrajectory sched x0 2 := by
  exact apply_windowKernel sched x0 0 2

example (k : Nat) :
    coordRange (varyingTrajectory sched x0 (k + 1)) ≤
      coordRange (varyingTrajectory sched x0 k) := by
  apply coordRange_varying_le
  intro t
  by_cases h : t % 2 = 0
  · simpa [sched, h] using K0_averaging
  · simpa [sched, h] using K1_averaging

example {ι : Type*} [Fintype ι] [Nonempty ι] [DecidableEq ι]
    (K : KernelSchedule ι) (hK : ∀ k, AveragingKernel (K k))
    (x : ι → Rat) (b : Nat) (hb : 0 < b)
    (δ : Rat) (hδ0 : 0 < δ) (hδ1 : δ < 1)
    (hc : UniformBlockCommonColumn K b δ) :
    ∃ c : Real, ∀ i,
      Tendsto (fun k => (varyingTrajectory K x k i : Real)) atTop (nhds c) := by
  exact block_contraction_consensus_exists K hK x b hb δ hδ0 hδ1 hc

#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.apply_windowKernel
#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.coordRange_tendsto_zero
#print axioms NarrativeDynamics.FiniteTimeVaryingConsensus.block_contraction_consensus_exists
