import Browser.Interaction.Types

namespace Browser.Interaction

inductive InputOwner where
  | none
  | human
  | automation (action : ActionId)
  | conflict (action : ActionId)
  deriving Repr, DecidableEq, BEq

/-- Human input never co-owns automation dispatch. If automation currently owns
    the input channel, human input makes the conflict explicit. -/
def onHumanInput : InputOwner → InputOwner
  | .automation action => .conflict action
  | .conflict action => .conflict action
  | _ => .human

/-- Automation may dispatch only when the exact ActionId owns the input. -/
def canAutomationDispatch (owner : InputOwner) (action : ActionId) : Bool :=
  decide (owner = .automation action)

theorem human_input_conflicts_with_automation (action : ActionId) :
    onHumanInput (.automation action) = .conflict action := by
  rfl

theorem human_owner_blocks_automation (action : ActionId) :
    canAutomationDispatch .human action = false := by
  rfl

theorem conflict_blocks_automation (owner action : ActionId) :
    canAutomationDispatch (.conflict owner) action = false := by
  rfl

theorem exact_owner_may_dispatch (action : ActionId) :
    canAutomationDispatch (.automation action) action = true := by
  simp [canAutomationDispatch]

end Browser.Interaction
