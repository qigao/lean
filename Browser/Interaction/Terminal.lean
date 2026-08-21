import Browser.Interaction.Types

namespace Browser.Interaction

inductive ActorLiveness where
  | live
  | cancelled
  | timedOut
  | destroyed
  deriving Repr, DecidableEq, BEq

/-- External side effects are permitted only for live actors. The particular
    effect kind is retained in the contract vocabulary for later composition. -/
def mayEmitExternalEffect (state : ActorLiveness) (_effect : ExternalEffect) : Bool :=
  state == .live

theorem cancelled_cannot_emit (effect : ExternalEffect) :
    mayEmitExternalEffect .cancelled effect = false := by
  rfl

theorem timed_out_cannot_emit (effect : ExternalEffect) :
    mayEmitExternalEffect .timedOut effect = false := by
  rfl

theorem destroyed_cannot_emit (effect : ExternalEffect) :
    mayEmitExternalEffect .destroyed effect = false := by
  rfl

theorem live_actor_may_emit (effect : ExternalEffect) :
    mayEmitExternalEffect .live effect = true := by
  rfl

end Browser.Interaction
