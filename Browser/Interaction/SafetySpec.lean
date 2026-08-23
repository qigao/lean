import Browser.Interaction.Input
import Browser.Interaction.Terminal

namespace Browser.Interaction

example :
    onHumanInput (.automation 7) = .conflict 7 := by
  native_decide

example :
    canAutomationDispatch .human 7 = false ∧
    canAutomationDispatch (.conflict 7) 7 = false ∧
    canAutomationDispatch (.automation 7) 7 = true := by
  native_decide

example :
    mayEmitExternalEffect .live .cdpRequest = true ∧
    mayEmitExternalEffect .cancelled .cdpRequest = false ∧
    mayEmitExternalEffect .timedOut .inputDispatch = false ∧
    mayEmitExternalEffect .destroyed .networkMutation = false := by
  native_decide

end Browser.Interaction
