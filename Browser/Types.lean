namespace Browser

abbrev BrowserId := Nat
abbrev ContextId := Nat
abbrev PageId := Nat
abbrev FrameId := Nat
abbrev CdpSessionId := Nat
abbrev EventId := Nat
abbrev CorrelationId := Nat
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

/-- Absolute monotonic deadline. Driver implementations should map `Time` to a
    monotonic clock rather than wall-clock time. -/
structure Deadline where
  expiresAt : Time
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
