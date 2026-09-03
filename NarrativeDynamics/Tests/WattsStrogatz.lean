import NarrativeDynamics.Core.WattsStrogatz

open NarrativeDynamics

example (nodeCount radius : Nat) :
    SymmetricMesh (wattsStrogatzRing nodeCount radius) := by
  exact wattsStrogatzRing_symmetric nodeCount radius

example (nodeCount radius : Nat) :
    LooplessMesh (wattsStrogatzRing nodeCount radius) := by
  exact wattsStrogatzRing_loopless nodeCount radius

example : localClusteringCoefficient
    (wattsStrogatzRing 6 2) (0 : Fin 6) = (2 : ℚ) / 3 := by
  native_decide
