import Browser.Action
import Browser.Transition

namespace Browser

/-- Browser availability is a declared primitive-action dependency. -/
example (m : Model) (p : PageId) (h : m.browserAlive = false) :
    requirementsHold m (primitiveContract p) = false := by
  simp [requirementsHold, primitiveContract, requirementHolds, h]

/-- Reconciliation must suspend an executing action when its contract fails. -/
example (m : Model) (p : PageId)
    (hexec : (m.page p).action = .executing)
    (hreq : requirementsHold m (primitiveContract p) = false) :
    ((reconcileAction m (primitiveContract p)).page p).action = .suspended := by
  simp [reconcileAction, primitiveContract, hexec, hreq, updatePage]

end Browser
