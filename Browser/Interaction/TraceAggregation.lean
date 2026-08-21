import Browser.Interaction.FaultAggregation

namespace Browser.Interaction

/-- Recovery requirement currently represented by the supervisor. Terminal
    failure is the top recovery requirement; no active recovery is `retry`. -/
def supervisorRequirement (supervisor : FaultSupervisor) : RecoveryAction :=
  if supervisor.failed then
    .fail
  else
    match supervisor.active with
    | none => .retry
    | some active => active.action

/-- Recovery requirement contributed by one observation to a specific runtime
    generation. Stale/future observations and no-fault observations contribute
    the join identity `retry`. -/
def observationRequirement
    (generation : Nat) (observation : TaggedObservation) : RecoveryAction :=
  if observation.generation = generation then
    match (normalizeObservation (decodeEvent observation.event)).fault with
    | none => .retry
    | some fault => minimumRecovery fault
  else
    .retry

/-- Pure recovery fold for a finite observation trace. -/
def traceRequirement (generation : Nat) : List TaggedObservation → RecoveryAction
  | [] => .retry
  | observation :: rest =>
      joinRecovery
        (observationRequirement generation observation)
        (traceRequirement generation rest)

/-- Extract only current-generation semantic faults, preserving arrival order.
    Stale/future and ordinary no-fault feedback are filtered out. -/
def effectiveFaults (generation : Nat) : List TaggedObservation → List FaultClass
  | [] => []
  | observation :: rest =>
      if observation.generation = generation then
        match (normalizeObservation (decodeEvent observation.event)).fault with
        | none => effectiveFaults generation rest
        | some fault => fault :: effectiveFaults generation rest
      else
        effectiveFaults generation rest

/-- Runtime-style sequential observation fold. -/
def observeTrace : FaultSupervisor → List TaggedObservation → FaultSupervisor
  | supervisor, [] => supervisor
  | supervisor, observation :: rest =>
      observeTrace (observeAdversarial supervisor observation) rest

private theorem join_retry_left (action : RecoveryAction) :
    joinRecovery .retry action = action := by
  cases action <;> rfl

private theorem join_retry_right (action : RecoveryAction) :
    joinRecovery action .retry = action := by
  cases action <;> rfl

private theorem join_fail_left (action : RecoveryAction) :
    joinRecovery .fail action = .fail := by
  cases action <;> rfl

private theorem join_fail_right (action : RecoveryAction) :
    joinRecovery action .fail = .fail := by
  cases action <;> rfl

/-- Observation delivery never changes the runtime generation; only successful
    recovery feedback may advance it. -/
theorem observe_adversarial_preserves_generation
    (supervisor : FaultSupervisor) (observation : TaggedObservation) :
    (observeAdversarial supervisor observation).generation = supervisor.generation := by
  by_cases hfailed : supervisor.failed = true
  · simp [observeAdversarial, hfailed]
  · have hfailedFalse : supervisor.failed = false := Bool.eq_false_of_not_eq_true hfailed
    by_cases hgeneration : observation.generation = supervisor.generation
    · cases hfault : (normalizeObservation (decodeEvent observation.event)).fault with
      | none =>
          simp [observeAdversarial, hfailedFalse, hgeneration, hfault]
      | some fault =>
          by_cases hrequired : minimumRecovery fault = .fail
          · simp [observeAdversarial, hfailedFalse, hgeneration, hfault, hrequired]
          · cases hactive : supervisor.active with
            | none =>
                simp [observeAdversarial, hfailedFalse, hgeneration, hfault,
                  hrequired, hactive, beginRecovery]
            | some active =>
                simp [observeAdversarial, hfailedFalse, hgeneration, hfault,
                  hrequired, hactive]
    · simp [observeAdversarial, hfailedFalse, hgeneration]

/-- One delivered observation changes the represented recovery requirement by
    exactly joining in that observation's current-generation requirement. -/
theorem observe_requirement_step
    (supervisor : FaultSupervisor) (observation : TaggedObservation) :
    supervisorRequirement (observeAdversarial supervisor observation) =
      joinRecovery
        (supervisorRequirement supervisor)
        (observationRequirement supervisor.generation observation) := by
  by_cases hfailed : supervisor.failed = true
  · simp [observeAdversarial, supervisorRequirement, observationRequirement,
      hfailed, join_fail_left]
  · have hfailedFalse : supervisor.failed = false := Bool.eq_false_of_not_eq_true hfailed
    by_cases hgeneration : observation.generation = supervisor.generation
    · cases hfault : (normalizeObservation (decodeEvent observation.event)).fault with
      | none =>
          simp [observeAdversarial, supervisorRequirement, observationRequirement,
            hfailedFalse, hgeneration, hfault, join_retry_right]
      | some fault =>
          by_cases hrequired : minimumRecovery fault = .fail
          · simp [observeAdversarial, supervisorRequirement, observationRequirement,
              hfailedFalse, hgeneration, hfault, hrequired, join_fail_right]
          · cases hactive : supervisor.active with
            | none =>
                simp [observeAdversarial, supervisorRequirement, observationRequirement,
                  hfailedFalse, hgeneration, hfault, hrequired, hactive,
                  beginRecovery, join_retry_left, join_retry_right]
            | some active =>
                simp [observeAdversarial, supervisorRequirement, observationRequirement,
                  hfailedFalse, hgeneration, hfault, hrequired, hactive,
                  join_retry_right, joinRecovery]
    · simp [observeAdversarial, supervisorRequirement, observationRequirement,
        hfailedFalse, hgeneration, join_retry_right]

/-- The pure trace requirement is exactly the finite-fault aggregate after
    filtering stale/future/no-fault observations. -/
theorem trace_requirement_matches_aggregate
    (generation : Nat) (trace : List TaggedObservation) :
    traceRequirement generation trace =
      aggregateRecovery (effectiveFaults generation trace) := by
  induction trace with
  | nil =>
      rfl
  | cons observation rest ih =>
      by_cases hgeneration : observation.generation = generation
      · cases hfault : (normalizeObservation (decodeEvent observation.event)).fault with
        | none =>
            simp [traceRequirement, observationRequirement, effectiveFaults,
              hgeneration, hfault, ih, join_retry_left]
        | some fault =>
            simp [traceRequirement, observationRequirement, effectiveFaults,
              aggregateRecovery, hgeneration, hfault, ih]
      · simp [traceRequirement, observationRequirement, effectiveFaults,
          hgeneration, ih, join_retry_left]

/-- General runtime/algebra bridge. Sequential adversarial delivery is exactly
    the join of the supervisor's existing requirement and the mathematical
    aggregate of all effective faults in the finite trace. -/
theorem observe_trace_requirement
    (supervisor : FaultSupervisor) (trace : List TaggedObservation) :
    supervisorRequirement (observeTrace supervisor trace) =
      joinRecovery
        (supervisorRequirement supervisor)
        (aggregateRecovery (effectiveFaults supervisor.generation trace)) := by
  rw [← trace_requirement_matches_aggregate]
  induction trace generalizing supervisor with
  | nil =>
      simp [observeTrace, traceRequirement, join_retry_right]
  | cons observation rest ih =>
      rw [observeTrace]
      rw [ih (supervisor := observeAdversarial supervisor observation)]
      rw [observe_adversarial_preserves_generation]
      rw [observe_requirement_step]
      simp only [traceRequirement]
      exact join_recovery_associative
        (supervisorRequirement supervisor)
        (observationRequirement supervisor.generation observation)
        (traceRequirement supervisor.generation rest)

/-- Top-level theorem for a fresh lane: the runtime supervisor's final recovery
    requirement is exactly the least combined recovery of the effective finite
    fault collection, independent of stale/future/no-fault observations. -/
theorem healthy_trace_matches_aggregate
    (generation budget : Nat) (trace : List TaggedObservation) :
    supervisorRequirement
        (observeTrace (healthySupervisor generation budget) trace) =
      aggregateRecovery (effectiveFaults generation trace) := by
  rw [observe_trace_requirement]
  simp [healthySupervisor, supervisorRequirement, join_retry_left]

end Browser.Interaction
