import Browser.Generation

/-!
# Action-generation integration/reference spec

Executable examples for the earlier whole-runtime ActionId generation model.
They remain regression evidence through `Browser.Integration`; current public
freshness/ownership contracts belong to `Browser.ProofAPI`.
-/

namespace Browser

/-- Starting a new action advances the page-local action generation and binds
    the new ActionId to that generation. -/
example (s : PageState) :
    (startActionState s).actionGeneration = s.actionGeneration + 1 ∧
    (startActionState s).currentAction = some (s.actionGeneration + 1) := by
  simp [startActionState, nextActionId, clearDeadlineState]

/-- A completion callback from an older action generation cannot finish the
    current action. -/
example (s : PageState) (action : ActionId)
    (hstale : s.currentAction ≠ some action) :
    finishActionState s action = s := by
  simp [finishActionState, hstale]

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
