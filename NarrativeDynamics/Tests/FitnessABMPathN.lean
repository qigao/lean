import NarrativeDynamics.Core.FitnessABMPathN

open NarrativeDynamics.FitnessABMPathN

example : pathAdj 6 (0 : Fin 6) 1 := by
  decide

example : degree 6 0 = 1 := by
  decide_cbv

example : degree 6 3 = 2 := by
  decide_cbv

example : pathKernel 6 0 0 = 1/2 := by
  decide_cbv

example : pathKernel 6 0 1 = 1/2 := by
  decide_cbv

example : pathKernel 6 3 2 = 1/4 := by
  decide_cbv
