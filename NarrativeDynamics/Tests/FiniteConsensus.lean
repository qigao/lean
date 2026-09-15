import NarrativeDynamics.Core.FiniteConsensus

open NarrativeDynamics.FiniteConsensus

private def pairKernel : Kernel (Fin 2) := fun _ _ => 1/2
private def pairWeights : Fin 2 → Rat := fun _ => 1/2

example : applyKernel pairKernel ![0, 1] = ![1/2, 1/2] := by
  decide_cbv

example : weightedMean pairWeights ![0, 1] = 1/2 := by
  decide_cbv

example (x : Fin 2 → Rat) : 0 ≤ coordRange x := coordRange_nonneg x

example
    (hK : AveragingKernel pairKernel)
    (hπ : StationaryWeights pairKernel pairWeights)
    (hcommon : CommonColumnMass (pairKernel ^ 1) (1/2))
    (x : Fin 2 → Rat) (i : Fin 2) :
    Tendsto
      (fun k : Nat => (kernelTrajectory pairKernel x k i : Real))
      atTop
      (nhds (weightedMean pairWeights x : Real)) :=
  block_contraction_tendsto
    pairKernel pairWeights hK hπ 1 (by omega)
    (1/2) (by norm_num) (by norm_num) hcommon x i
