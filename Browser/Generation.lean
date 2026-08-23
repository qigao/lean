import Browser.Timeout

namespace Browser

/-- The next page-local ActionId is the successor of the last generation. -/
def nextActionId (s : PageState) : ActionId :=
  s.actionGeneration + 1

/-- Begin a fresh action generation. The new action owns a new identity while
    deadline/timeout/pending-protocol state from the previous generation is
    discarded. Page/runtime/input state is intentionally preserved so the
    caller can decide whether the new action executes immediately or waits. -/
def startActionState (s : PageState) : PageState :=
  let action := nextActionId s
  let fresh := clearDeadlineState s
  { fresh with
    actionGeneration := action
    currentAction := some action
    pendingProtocol := none }

/-- Finish only the action generation named by the completion callback. A late
    completion from an older action is ignored instead of clearing newer work. -/
def finishActionState (s : PageState) (action : ActionId) : PageState :=
  if s.currentAction = some action then
    let fresh := clearDeadlineState s
    { fresh with
      input := .idle
      action := .idle
      currentAction := none
      pendingProtocol := none }
  else
    s

/-- Action-aware deadline arming. A delayed timer-registration callback from an
    older action generation is ignored. -/
def armActionDeadlineState
    (s : PageState) (action : ActionId) (expiresAt : Time) (reason : WaitReason) : PageState :=
  if s.currentAction = some action then
    armDeadlineState s action expiresAt reason
  else
    s

/-- Action-aware deadline expiry. A callback belonging to an older generation
    cannot time out the current action. -/
def expireActionDeadlineState (s : PageState) (action : ActionId) (now : Time) : PageState :=
  if s.currentAction = some action then
    expirePageState s action now
  else
    s

/-- Register one protocol/CDP wait only for the current live action. -/
def startProtocolWaitState
    (s : PageState) (action : ActionId) (call : CdpCallId) : PageState :=
  if s.currentAction = some action then
    match s.action with
    | .timedOut => s
    | _ =>
        { s with
          pendingProtocol := some { action := action, call := call }
          waitingOn := some .protocol
          action := .waiting }
  else
    s

/-- Consume a protocol response only when both its ActionId and CdpCallId match
    the current pending wait. Late responses from older actions are no-ops. A
    response after terminal timeout is also ignored. -/
def receiveProtocolResponseState
    (s : PageState) (action : ActionId) (call : CdpCallId) : PageState :=
  if s.currentAction = some action then
    match s.action with
    | .timedOut => s
    | _ =>
        match s.pendingProtocol with
        | some pending =>
            if pending.action = action ∧ pending.call = call then
              { s with pendingProtocol := none, waitingOn := none }
            else
              s
        | none => s
  else
    s

end Browser
