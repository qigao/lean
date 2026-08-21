namespace Browser

abbrev BrowserId := Nat
abbrev ContextId := Nat
abbrev PageId := Nat
abbrev FrameId := Nat
abbrev CdpSessionId := Nat
abbrev EventId := Nat
abbrev CorrelationId := Nat

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
  deriving Repr, DecidableEq, BEq

end Browser
