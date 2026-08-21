import Browser.Interaction.FaultAggregation

namespace Browser.Interaction

private def faultsA : List FaultClass :=
  [.elementStale, .sessionLost, .pageLost]

private def faultsB : List FaultClass :=
  [.pageLost, .elementStale, .sessionLost]

example : aggregateRecovery faultsA = .recreatePage := by rfl
example : aggregateRecovery faultsB = .recreatePage := by rfl

example :
    aggregateRecovery [.pageLost, .pageLost, .sessionLost] =
      aggregateRecovery [.pageLost, .sessionLost] := by
  exact aggregate_duplicate_invariant .pageLost [.sessionLost]

private def faultsPermuted : FaultPermutation faultsA faultsB := by
  apply FaultPermutation.trans
  · exact FaultPermutation.swap .elementStale .sessionLost [.pageLost]
  · apply FaultPermutation.trans
    · exact FaultPermutation.cons .sessionLost (FaultPermutation.swap .elementStale .pageLost [])
    · apply FaultPermutation.trans
      · exact FaultPermutation.swap .sessionLost .pageLost [.elementStale]
      · exact FaultPermutation.cons .pageLost (FaultPermutation.swap .sessionLost .elementStale [])

example : aggregateRecovery faultsA = aggregateRecovery faultsB := by
  exact aggregate_permutation_invariant faultsPermuted

example : CoversFaults faultsA (aggregateRecovery faultsA) := by
  exact aggregate_covers_faults faultsA

example (candidate : RecoveryAction)
    (h : CoversFaults faultsA candidate) :
    recoveryRank (aggregateRecovery faultsA) ≤ recoveryRank candidate := by
  exact aggregate_least_upper faultsA candidate h

example : MinimalCombined faultsA (aggregateRecovery faultsA) := by
  exact aggregate_is_minimal_combined faultsA

example : aggregateRecovery [.runtimeLost, .contextLost, .sessionLost] = .recreateContext := by
  rfl

example : aggregateRecovery [.pageLost, .timeoutFault] = .fail := by
  rfl

end Browser.Interaction
