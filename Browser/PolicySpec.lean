import Browser.Policy
import Browser.Transition

namespace Browser

/-- A page command is a strictly page-local mutation. -/
example (m : Model) (cmd : PageCommand) (q : PageId) (h : q ≠ cmd.page) :
    (applyCommand m cmd).page q = m.page q := by
  simp [applyCommand, updatePage, h]

/-- Making a context unavailable must block execution for every descendant page. -/
example (m : Model) (c : ContextId) (p : PageId)
    (hctx : m.graph.contextOf p = c) :
    ¬ canExecute (step m (.contextUnavailable c)) p := by
  simp [canExecute, step, updateContextAvailability, hctx]

end Browser
