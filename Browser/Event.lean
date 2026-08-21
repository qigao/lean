import Browser.State

namespace Browser

inductive EventScope where
  | page (page : PageId)
  | context (context : ContextId)
  | browser
  deriving Repr, DecidableEq, BEq

inductive LocalEventKind where
  | humanInput
  | humanIdle
  | automationStarted
  | automationFinished
  | navigationStarted
  | pageReady
  | executionContextDestroyed
  | executionContextReady
  | deadlineArmed (action : ActionId) (expiresAt : Time) (reason : WaitReason)
  | deadlineReached (action : ActionId) (now : Time)
  | protocolWaitStarted (action : ActionId) (call : CdpCallId)
  | protocolResponse (action : ActionId) (call : CdpCallId)
  | pageCrashed
  | pageClosed
  deriving Repr, DecidableEq, BEq

structure LocalEvent where
  page : PageId
  kind : LocalEventKind
  deriving Repr

inductive RuntimeEvent where
  | local (event : LocalEvent)
  | contextUnavailable (context : ContextId)
  | contextAvailable (context : ContextId)
  | browserDisconnected
  | browserConnected
  deriving Repr

def scopeOf : RuntimeEvent → EventScope
  | .local event => .page event.page
  | .contextUnavailable context => .context context
  | .contextAvailable context => .context context
  | .browserDisconnected => .browser
  | .browserConnected => .browser

end Browser
