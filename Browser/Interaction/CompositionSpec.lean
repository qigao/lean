import Browser.Interaction.Policy
import Browser.Interaction.Composition

namespace Browser.Interaction

private def policy : PolicyProtocol := { childVersion := 5 }
private def trigger : InteractionCause := {
  id := 10
  correlation := 99
  depth := 2
}

example :
    let (next, emission) := emitPolicyCommand policy trigger 1
    next.childVersion = 5 ∧
    emission.cause.correlation = 99 ∧
    emission.cause.parent = some 10 ∧
    emission.cause.depth = 3 := by
  native_decide

example :
    let (_, emission) := emitPolicyCommand policy trigger 1
    (deliverPolicyCommand policy emission).childVersion = 6 := by
  native_decide

example : AuthorizedEffect .action .cdp .cdpRequest = true := by native_decide
example : AuthorizedEffect .policy .page .pageCommand = true := by native_decide
example : AuthorizedEffect .page .page .pageCommand = false := by native_decide
example : AuthorizedEffect .page .cdp .cdpRequest = false := by native_decide

example :
    let (_, emission) := emitPolicyCommand policy trigger 1
    CausalIntegrity trigger emission = true := by
  native_decide

end Browser.Interaction
