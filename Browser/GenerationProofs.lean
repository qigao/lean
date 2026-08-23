import Browser.Proofs

/-!
# Action-generation integration/reference proofs

These theorems preserve the earlier whole-runtime ActionId/generation model for
regression and compatibility. They are not part of `Browser.ProofAPI`; new
public ownership and freshness contracts should be expressed through the
current Interaction/Recovery chain.
-/

namespace Browser

/-- Action generations are monotonic and the newly started action is bound to
    the successor generation. -/
theorem new_action_advances_generation (s : PageState) :
    (startActionState s).actionGeneration = s.actionGeneration + 1 ∧
    (startActionState s).currentAction = some (s.actionGeneration + 1) := by
  simp [startActionState, nextActionId, clearDeadlineState]

/-- A stale completion callback from an old ActionId cannot finish newer work. -/
theorem stale_action_finish_ignored
    (s : PageState) (action : ActionId)
    (hstale : s.currentAction ≠ some action) :
    finishActionState s action = s := by
  simp [finishActionState, hstale]

/-- A stale timer registration from an old ActionId is a state no-op. -/
theorem stale_deadline_arm_ignored
    (s : PageState) (action : ActionId) (expiresAt : Time) (reason : WaitReason)
    (hstale : s.currentAction ≠ some action) :
    armActionDeadlineState s action expiresAt reason = s := by
  simp [armActionDeadlineState, hstale]

/-- A stale deadline callback from an old ActionId cannot time out a newer action. -/
theorem stale_deadline_callback_ignored
    (s : PageState) (action : ActionId) (now : Time)
    (hstale : s.currentAction ≠ some action) :
    expireActionDeadlineState s action now = s := by
  simp [expireActionDeadlineState, hstale]

/-- A late protocol response from an old ActionId cannot clear or otherwise
    mutate the current action's pending protocol state. -/
theorem stale_protocol_response_ignored
    (s : PageState) (action : ActionId) (call : CdpCallId)
    (hstale : s.currentAction ≠ some action) :
    receiveProtocolResponseState s action call = s := by
  simp [receiveProtocolResponseState, hstale]

/-- After starting a new generation, a callback carrying any different ActionId
    is rejected before it reaches timeout semantics. -/
theorem new_action_rejects_old_deadline_callback
    (s : PageState) (oldAction : ActionId) (now : Time)
    (hstale : some (s.actionGeneration + 1) ≠ some oldAction) :
    expireActionDeadlineState (startActionState s) oldAction now = startActionState s := by
  simp [expireActionDeadlineState, startActionState, nextActionId, clearDeadlineState, hstale]

end Browser
