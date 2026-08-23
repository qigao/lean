import Browser.Interaction.TraceAggregation

open Browser.Interaction

private def mixedTrace : List TaggedObservation := [
  { generation := 10, event := .targetCrashed .page },
  { generation := 9, event := .browserProcessExited },
  { generation := 10, event := .networkLoadingFailed .transient },
  { generation := 11, event := .contextDestroyed },
  { generation := 10, event := .targetSessionDetached },
  { generation := 10, event := .targetCrashed .page }
]

private def reorderedTrace : List TaggedObservation := [
  { generation := 10, event := .targetSessionDetached },
  { generation := 10, event := .targetCrashed .page },
  { generation := 10, event := .domElementDetached }
]

private def fatalTrace : List TaggedObservation := [
  { generation := 10, event := .targetCrashed .page },
  { generation := 10, event := .malformed },
  { generation := 10, event := .targetSessionDetached }
]

private def runtimeRequirement (trace : List TaggedObservation) : RecoveryAction :=
  supervisorRequirement (observeTrace (healthySupervisor 10 5) trace)

private def mathematicalRequirement (trace : List TaggedObservation) : RecoveryAction :=
  aggregateRecovery (effectiveFaults 10 trace)

private def scenarioVerified : Bool :=
  (runtimeRequirement mixedTrace == mathematicalRequirement mixedTrace) &&
  (runtimeRequirement mixedTrace == .recreatePage) &&
  (runtimeRequirement reorderedTrace == mathematicalRequirement reorderedTrace) &&
  (runtimeRequirement reorderedTrace == .recreatePage) &&
  (runtimeRequirement fatalTrace == mathematicalRequirement fatalTrace) &&
  (runtimeRequirement fatalTrace == .fail) &&
  (effectiveFaults 10 mixedTrace == [.pageLost, .sessionLost, .pageLost])

def main : IO UInt32 := do
  if scenarioVerified then
    IO.println "runtime trace aggregation: incremental recovery equals finite aggregate"
    return 0
  else
    IO.eprintln "runtime trace aggregation: equivalence violation"
    return 1
