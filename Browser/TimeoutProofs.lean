import Browser.Proofs

namespace Browser

/-- Navigation cannot resurrect a terminal timed-out action. -/
theorem timed_out_terminal_under_navigation
    (s : PageState) (h : s.action = .timedOut) :
    (applyLocal .navigationStarted s).action = .timedOut := by
  simp [applyLocal, recoveryAction, h]

/-- Execution-context loss cannot resurrect a terminal timed-out action. -/
theorem timed_out_terminal_under_context_loss
    (s : PageState) (h : s.action = .timedOut) :
    (applyLocal .executionContextDestroyed s).action = .timedOut := by
  simp [applyLocal, recoveryAction, h]

/-- Crash preserves the timeout terminal result while updating page health. -/
theorem timed_out_terminal_under_crash
    (s : PageState) (h : s.action = .timedOut) :
    (applyLocal .pageCrashed s).action = .timedOut := by
  simp [applyLocal, suspendedOrTimedOut, h]

/-- Close preserves the timeout terminal result while updating lifecycle. -/
theorem timed_out_terminal_under_close
    (s : PageState) (h : s.action = .timedOut) :
    (applyLocal .pageClosed s).action = .timedOut := by
  simp [applyLocal, suspendedOrTimedOut, h]

/-- A deadline notification before the absolute expiry instant is a no-op. -/
theorem before_deadline_does_not_timeout
    (s : PageState) (deadline now : Time) (reason : WaitReason)
    (hbefore : now < deadline) :
    expirePageState (armDeadlineState s deadline reason) now =
      armDeadlineState s deadline reason := by
  simp [expirePageState, armDeadlineState, Nat.not_le.mpr hbefore]

/-- Explicitly starting a new action clears the previous action's deadline and
    timeout result. It is the only supported way to leave the terminal timeout
    result and begin fresh work. -/
theorem new_action_clears_timeout
    (s : PageState)
    (hlife : s.lifecycle = .ready)
    (hruntime : s.runtime = .ready)
    (hinput : s.input = .idle) :
    (applyLocal .automationStarted s).deadline = none ∧
    (applyLocal .automationStarted s).waitingOn = none ∧
    (applyLocal .automationStarted s).timeout = none := by
  simp [applyLocal, clearDeadlineState, hlife, hruntime, hinput]

end Browser
