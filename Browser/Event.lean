import Browser.State

namespace Browser

inductive LocalEventKind where
  | humanInput
  | humanIdle
  | automationStarted
  | automationFinished
  | navigationStarted
  | pageReady
  | executionContextDestroyed
  | executionContextReady
  | pageCrashed
  | pageClosed
  deriving Repr, DecidableEq, BEq

structure LocalEvent where
  page : PageId
  kind : LocalEventKind
  deriving Repr

inductive RuntimeEvent where
  | local (event : LocalEvent)
  | browserDisconnected
  | browserConnected
  deriving Repr

end Browser
