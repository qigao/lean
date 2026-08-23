import Browser.Interaction.Types

namespace Browser.Interaction

structure InteractionCause where
  id : EventId
  correlation : CorrelationId
  depth : Nat := 0
  deriving Repr, DecidableEq, BEq

structure EmittedCause where
  correlation : CorrelationId
  parent : Option EventId
  depth : Nat
  deriving Repr, DecidableEq, BEq

structure PolicyEmission where
  cause : EmittedCause
  page : PageId
  deriving Repr, DecidableEq, BEq

/-- The only state retained here is an abstract child-effect version. Emission
    must not change it; actual command delivery may. -/
structure PolicyProtocol where
  childVersion : Nat := 0
  deriving Repr, DecidableEq, BEq

def emitPolicyCommand
    (state : PolicyProtocol) (trigger : InteractionCause) (page : PageId) :
    PolicyProtocol × PolicyEmission :=
  (state, {
    cause := {
      correlation := trigger.correlation
      parent := some trigger.id
      depth := trigger.depth + 1
    }
    page := page
  })

def deliverPolicyCommand (state : PolicyProtocol) (_emission : PolicyEmission) : PolicyProtocol :=
  { state with childVersion := state.childVersion + 1 }

def commandCausallyValid (trigger : InteractionCause) (emission : PolicyEmission) : Bool :=
  decide (
    emission.cause.correlation = trigger.correlation ∧
    emission.cause.parent = some trigger.id ∧
    emission.cause.depth = trigger.depth + 1)

theorem policy_emission_does_not_mutate_child
    (state : PolicyProtocol) (trigger : InteractionCause) (page : PageId) :
    (emitPolicyCommand state trigger page).1.childVersion = state.childVersion := by
  rfl

theorem policy_emission_preserves_cause
    (state : PolicyProtocol) (trigger : InteractionCause) (page : PageId) :
    commandCausallyValid trigger (emitPolicyCommand state trigger page).2 = true := by
  simp [commandCausallyValid, emitPolicyCommand]

end Browser.Interaction
