import Browser.Types

namespace Browser.Interaction

inductive InteractionRole where
  | browser
  | context
  | page
  | frame
  | action
  | cdp
  | timer
  | human
  | inputRouter
  | policy
  deriving Repr, DecidableEq, BEq

inductive ExternalEffect where
  | cdpRequest
  | inputDispatch
  | networkMutation
  | pageCommand
  deriving Repr, DecidableEq, BEq

end Browser.Interaction
