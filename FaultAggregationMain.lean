import Browser.Interaction.FaultAggregation

open Browser.Interaction

private def scenarioVerified : Bool :=
  (aggregateRecovery [.elementStale, .sessionLost, .pageLost] == .recreatePage) &&
  (aggregateRecovery [.pageLost, .elementStale, .sessionLost] == .recreatePage) &&
  (aggregateRecovery [.pageLost, .pageLost, .sessionLost] == .recreatePage) &&
  (aggregateRecovery [.runtimeLost, .contextLost, .sessionLost] == .recreateContext) &&
  (aggregateRecovery [.pageLost, .timeoutFault] == .fail) &&
  (aggregateRecovery [] == .retry)

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "finite fault aggregation: permutation/duplicate/minimal recovery verified"
    return 0
  else
    IO.eprintln "finite fault aggregation: violation"
    return 1
