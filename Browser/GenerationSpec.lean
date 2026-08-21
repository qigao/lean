import Browser.Generation

namespace Browser

/-- Starting a new action advances the page-local action generation and binds
    the new ActionId to that generation. -/
example (s : PageState) :
    (startActionState s).actionGeneration = s.actionGeneration + 1 ∧
    (startActionState s).currentAction = some (s.actionGeneration + 1) := by
  simp [startActionState]

/-- A timer created by an older action generation cannot arm a deadline on the
    current action. -/
example (s : PageState) (action : ActionId) (expiresAt : Time) (reason : WaitReason)
    (hstale : s.currentAction ≠ some action) :
    armActionDeadlineState s action expiresAt reason = s := by
  simp [armActionDeadlineState, hstale]

/-- A late deadline callback from an older action generation cannot time out a
    newer action. -/
example (s : PageState) (action : ActionId) (now : Time)
    (hstale : s.currentAction ≠ some action) :
    expireActionDeadlineState s action now = s := by
  simp [expireActionDeadlineState, hstale]

/-- A late CDP response from an older action cannot clear the current action's
    pending protocol wait. -/
example (s : PageState) (action : ActionId) (call : CdpCallId)
    (hstale : s.currentAction ≠ some action) :
    receiveProtocolResponseState s action call = s := by
  simp [receiveProtocolResponseState, hstale]

end Browser
