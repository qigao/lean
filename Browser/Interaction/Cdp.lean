import Browser.Interaction.Types

namespace Browser.Interaction

structure CdpProtocol where
  currentAction : Option ActionId := none
  pending : Option (ActionId × CdpCallId) := none
  deriving Repr, DecidableEq, BEq

/-- Consume a CDP response only when both ActionId and CallId match the
    currently pending request. Any stale/mismatched response is a no-op. -/
def receiveCdpResponse
    (s : CdpProtocol) (action : ActionId) (call : CdpCallId) : CdpProtocol :=
  if s.currentAction = some action ∧ s.pending = some (action, call) then
    { s with pending := none }
  else
    s

theorem stale_cdp_response_noop
    (s : CdpProtocol) (action : ActionId) (call : CdpCallId)
    (h : s.currentAction ≠ some action) :
    receiveCdpResponse s action call = s := by
  simp [receiveCdpResponse, h]

theorem wrong_cdp_call_noop
    (s : CdpProtocol) (action : ActionId) (call : CdpCallId)
    (hcurrent : s.currentAction = some action)
    (hcall : s.pending ≠ some (action, call)) :
    receiveCdpResponse s action call = s := by
  simp [receiveCdpResponse, hcurrent, hcall]

theorem matching_cdp_response_consumes
    (s : CdpProtocol) (action : ActionId) (call : CdpCallId)
    (hcurrent : s.currentAction = some action)
    (hpending : s.pending = some (action, call)) :
    (receiveCdpResponse s action call).pending = none := by
  simp [receiveCdpResponse, hcurrent, hpending]

end Browser.Interaction
