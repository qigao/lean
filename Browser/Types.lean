namespace Browser

abbrev BrowserId := Nat
abbrev ContextId := Nat
abbrev PageId := Nat
abbrev FrameId := Nat
abbrev CdpSessionId := Nat
abbrev EventId := Nat
abbrev CorrelationId := Nat
abbrev ActionId := Nat
abbrev CdpCallId := Nat
abbrev Time := Nat

/-- Why a running browser action is currently waiting. The same value is
    retained as the timeout cause when its absolute deadline expires. -/
inductive WaitReason where
  | action
  | human
  | page
  | network
  | protocol
  deriving Repr, DecidableEq, BEq

/-- Absolute monotonic deadline bound to the ActionId that created it. Driver
    implementations should map `Time` to a monotonic clock rather than wall-clock time. -/
structure Deadline where
  action : ActionId
  expiresAt : Time
  deriving Repr, DecidableEq, BEq

/-- One outstanding protocol/CDP wait is bound to both its action generation
    and command id so late responses can be rejected deterministically. -/
structure ProtocolWait where
  action : ActionId
  call : CdpCallId
  deriving Repr, DecidableEq, BEq

inductive Lifecycle where
  | attaching
  | loading
  | ready
  | crashed
  | closed
  deriving Repr, DecidableEq, BEq

inductive RuntimeState where
  | unavailable
  | creating
  | ready
  deriving Repr, DecidableEq, BEq

inductive InputState where
  | idle
  | automation
  | human
  | conflict
  deriving Repr, DecidableEq, BEq

inductive ActionState where
  | idle
  | executing
  | waiting
  | recovering
  | suspended
  | timedOut
  deriving Repr, DecidableEq, BEq

end Browser
