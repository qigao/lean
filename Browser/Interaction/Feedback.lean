namespace Browser.Interaction

/-- Recovery actions are ordered later by the Recovery contract. They live in
    the feedback vocabulary because `Decision.recover` must name the requested
    repair without importing the recovery state machine. -/
inductive RecoveryAction where
  | retry
  | reResolve
  | rebindRuntime
  | reattachSession
  | recreatePage
  | recreateContext
  | restartBrowser
  | fail
  deriving Repr, DecidableEq, BEq

/-- Semantic feedback classes. Raw CDP/DOM/network/timer events are normalized
    into this compact vocabulary before policy selection. -/
inductive FeedbackClass where
  | success
  | transient
  | stale
  | conflict
  | unavailable
  | timeout
  | terminal
  | unknown
  deriving Repr, DecidableEq, BEq

inductive Decision where
  | complete
  | wait
  | retry
  | recover (action : RecoveryAction)
  | suspend
  | failSafe
  deriving Repr, DecidableEq, BEq

/-- Total feedback policy. No feedback class is silently ignored. Unknown and
    timed-out work fail safe rather than guessing or reviving an old attempt. -/
def decideFeedback : FeedbackClass → Decision
  | .success => .complete
  | .transient => .wait
  | .stale => .recover .reResolve
  | .conflict => .suspend
  | .unavailable => .recover .reattachSession
  | .timeout => .failSafe
  | .terminal => .recover .recreatePage
  | .unknown => .failSafe

theorem unknown_feedback_fails_safe :
    decideFeedback .unknown = .failSafe := by
  rfl

theorem timeout_feedback_does_not_retry_old_attempt :
    decideFeedback .timeout = .failSafe := by
  rfl

end Browser.Interaction
