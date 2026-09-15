import NarrativeDynamics.Core.FiniteConsensus

open NarrativeDynamics.FiniteConsensus

private def pairKernel : Kernel (Fin 2) := fun _ _ => 1/2
private def pairWeights : Fin 2 → Rat := fun _ => 1/2

example : applyKernel pairKernel ![0, 1] = ![1/2, 1/2] := by
  decide_cbv

example : weightedMean pairWeights ![0, 1] = 1/2 := by
  decide_cbv

example (x : Fin 2 → Rat) : 0 ≤ coordRange x := coordRange_nonneg x
