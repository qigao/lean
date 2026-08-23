import Browser.Interaction.Types

namespace Browser.Interaction

structure DeadlineProtocol where
  currentAction : Option ActionId := none
  deadline : Option (ActionId × Time) := none
  timedOut : Bool := false
  deriving Repr, DecidableEq, BEq

/-- Expiry is action-owned. A stale ActionId or early timer is a no-op; only a
    matching action whose absolute deadline has elapsed becomes terminal. -/
def expireDeadline
    (s : DeadlineProtocol) (action : ActionId) (now : Time) : DeadlineProtocol :=
  if s.currentAction = some action then
    match s.deadline with
    | some (owner, expiresAt) =>
        if owner = action ∧ expiresAt ≤ now then
          { s with timedOut := true }
        else
          s
    | none => s
  else
    s

theorem stale_timer_noop
    (s : DeadlineProtocol) (action : ActionId) (now : Time)
    (h : s.currentAction ≠ some action) :
    expireDeadline s action now = s := by
  simp [expireDeadline, h]

theorem early_timer_noop
    (s : DeadlineProtocol) (action : ActionId) (expiresAt now : Time)
    (hcurrent : s.currentAction = some action)
    (hdeadline : s.deadline = some (action, expiresAt))
    (hearly : now < expiresAt) :
    expireDeadline s action now = s := by
  simp [expireDeadline, hcurrent, hdeadline, Nat.not_le.mpr hearly]

theorem matching_expired_timer_times_out
    (s : DeadlineProtocol) (action : ActionId) (expiresAt now : Time)
    (hcurrent : s.currentAction = some action)
    (hdeadline : s.deadline = some (action, expiresAt))
    (hexpired : expiresAt ≤ now) :
    (expireDeadline s action now).timedOut = true := by
  simp [expireDeadline, hcurrent, hdeadline, hexpired]

end Browser.Interaction
