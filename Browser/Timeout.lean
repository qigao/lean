import Browser.State

namespace Browser

/-- Arm or replace the absolute deadline for one action generation. Callers must
    still validate that `action` is the current ActionId before invoking this
    primitive; `Browser.Generation` provides the guarded API used by events. -/
def armDeadlineState
    (s : PageState) (action : ActionId) (expiresAt : Time) (reason : WaitReason) : PageState :=
  { s with
    deadline := some { action := action, expiresAt := expiresAt }
    waitingOn := some reason
    timeout := none }

/-- Clear deadline/timeout phase state without changing the page's ActionId. -/
def clearDeadlineState (s : PageState) : PageState :=
  { s with deadline := none, waitingOn := none, timeout := none }

/-- If no more specific phase is registered, expiry is attributed to the action
    itself. -/
def timeoutReason (s : PageState) : WaitReason :=
  match s.waitingOn with
  | some reason => reason
  | none => .action

/-- Expire only the deadline bound to the supplied ActionId. -/
def expirePageState (s : PageState) (action : ActionId) (now : Time) : PageState :=
  match s.deadline with
  | none => s
  | some deadline =>
      if deadline.action = action ∧ deadline.expiresAt ≤ now then
        { s with
          input := .idle
          action := .timedOut
          timeout := some (timeoutReason s) }
      else
        s

def armDeadline
    (m : Model) (p : PageId) (action : ActionId) (expiresAt : Time) (reason : WaitReason) : Model :=
  updatePage m p fun state => armDeadlineState state action expiresAt reason

def expirePage (m : Model) (p : PageId) (action : ActionId) (now : Time) : Model :=
  updatePage m p fun state => expirePageState state action now

end Browser
