import Browser.Interaction.TraceAggregation

namespace Browser.Interaction

private def mixedTrace : List TaggedObservation := [
  { generation := 10, event := .targetCrashed .page },
  { generation := 9, event := .browserProcessExited },
  { generation := 10, event := .networkLoadingFailed .transient },
  { generation := 11, event := .contextDestroyed },
  { generation := 10, event := .targetSessionDetached },
  { generation := 10, event := .targetCrashed .page }
]

example : effectiveFaults 10 mixedTrace = [.pageLost, .sessionLost, .pageLost] := by
  rfl

example : aggregateRecovery (effectiveFaults 10 mixedTrace) = .recreatePage := by
  rfl

example :
    supervisorRequirement (observeTrace (healthySupervisor 10 5) mixedTrace) =
      aggregateRecovery (effectiveFaults 10 mixedTrace) := by
  exact healthy_trace_matches_aggregate 10 5 mixedTrace

example :
    supervisorRequirement
      (observeTrace (healthySupervisor 10 5)
        [{ generation := 10, event := .malformed }]) = .fail := by
  rfl

example :
    effectiveFaults 10 [
      { generation := 9, event := .targetCrashed .page },
      { generation := 11, event := .targetCrashed .page },
      { generation := 10, event := .operationSucceeded }
    ] = [] := by
  rfl

example (supervisor : FaultSupervisor) (trace : List TaggedObservation) :
    supervisorRequirement (observeTrace supervisor trace) =
      joinRecovery
        (supervisorRequirement supervisor)
        (aggregateRecovery (effectiveFaults supervisor.generation trace)) := by
  exact observe_trace_requirement supervisor trace

end Browser.Interaction
