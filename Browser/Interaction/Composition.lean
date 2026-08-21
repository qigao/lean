import Browser.Interaction.Policy
import Browser.Interaction.Input
import Browser.Interaction.Terminal
import Browser.Interaction.Epoch

namespace Browser.Interaction

/-- Whitelist of cross-component effect edges. Absence from this relation means
    the effect is not authorized by the interaction graph. -/
def AuthorizedEffect
    (source target : InteractionRole) (effect : ExternalEffect) : Bool :=
  match source, target, effect with
  | .action, .cdp, .cdpRequest => true
  | .action, .inputRouter, .inputDispatch => true
  | .policy, .page, .pageCommand => true
  | _, _, _ => false

/-- A generation-sensitive interaction is valid only when both node incarnation
    and ActionId ownership are current. -/
def GenerationIsolation
    (slot : EpochProtocol) (incomingEpoch : NodeEpoch)
    (currentAction : Option ActionId) (incomingAction : ActionId) : Bool :=
  epochAccepts slot incomingEpoch && decide (currentAction = some incomingAction)

/-- The ownership representation is exclusive by construction; this predicate
    exposes the relevant automation-dispatch guarantee for composition. -/
def InputExclusivity (owner : InputOwner) (action : ActionId) : Bool :=
  match owner with
  | .human | .conflict _ => !canAutomationDispatch owner action
  | _ => true

/-- Policy causality is the system-level causal integrity check. -/
def CausalIntegrity (trigger : InteractionCause) (emission : PolicyEmission) : Bool :=
  commandCausallyValid trigger emission

/-- If an external effect is attempted, its actor must be live. -/
def TerminalSafety
    (state : ActorLiveness) (effect : ExternalEffect) (attempted : Bool := true) : Bool :=
  if attempted then mayEmitExternalEffect state effect else true

theorem direct_page_to_page_command_rejected :
    AuthorizedEffect .page .page .pageCommand = false := by
  rfl

theorem action_to_cdp_request_authorized :
    AuthorizedEffect .action .cdp .cdpRequest = true := by
  rfl

theorem terminal_timeout_blocks_effect (effect : ExternalEffect) :
    TerminalSafety .timedOut effect true = false := by
  rfl

end Browser.Interaction
