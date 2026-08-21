import Browser.State

namespace Browser

/-- Arm or replace the absolute deadline for the current action phase. -/
def armDeadlineState (s : PageState) (expiresAt : Time) (reason : WaitReason) : PageState :=
  { s with
    deadline := some { expiresAt := expiresAt }
    waitingOn := some reason
    timeout := none }

/-- Starting a new action must discard timeout state from any previous action. -/
def clearDeadlineState (s : PageState) : PageState :=
  { s with deadline := none, waitingOn := none, timeout := none }

/-- If no more specific phase is registered, expiry is attributed to the action
    itself. -/
def timeoutReason (s : PageState) : WaitReason :=
  match s.waitingOn with
  | some reason => reason
  | none => .action

/-- Expire an armed absolute deadline. Waiting, recovery, suspension and human
    takeover do not move the deadline; they merely change why the action is
    waiting. -/
def expirePageState (s : PageState) (now : Time) : PageState :=
  match s.deadline with
  | none => s
  | some deadline =>
      if deadline.expiresAt ≤ now then
        { s with
          input := .idle
          action := .timedOut
          timeout := some (timeoutReason s) }
      else
        s

def armDeadline (m : Model) (p : PageId) (expiresAt : Time) (reason : WaitReason) : Model :=
  updatePage m p fun state => armDeadlineState state expiresAt reason

def expirePage (m : Model) (p : PageId) (now : Time) : Model :=
  updatePage m p fun state => expirePageState state now

end Browser
