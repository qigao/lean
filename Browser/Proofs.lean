import Browser.Transition

namespace Browser

/-- A page-local event cannot mutate an unrelated sibling page. -/
theorem local_event_isolated
    (m : Model) (e : LocalEvent) (q : PageId) (h : q ≠ e.page) :
    (step m (.local e)).page q = m.page q := by
  simp [step, updatePage, h]

/-- Human input arriving while automation owns the page enters the explicit
    conflict state and suspends the running action. -/
theorem human_input_conflicts_with_automation
    (m : Model) (p : PageId)
    (h : (m.page p).input = .automation) :
    ((step m (.local { page := p, kind := .humanInput })).page p).input = .conflict ∧
    ((step m (.local { page := p, kind := .humanInput })).page p).action = .suspended := by
  simp [step, updatePage, applyLocal, h]

/-- Once an execution context is destroyed, no primitive action is allowed to
    keep executing against that stale context. -/
theorem destroyed_context_blocks_execution
    (m : Model) (p : PageId) :
    ¬ canExecute (step m (.local { page := p, kind := .executionContextDestroyed })) p := by
  simp [canExecute, step, updatePage, applyLocal]

/-- Closing a page immediately makes primitive execution impossible. -/
theorem closed_page_blocks_execution
    (m : Model) (p : PageId) :
    ¬ canExecute (step m (.local { page := p, kind := .pageClosed })) p := by
  simp [canExecute, step, updatePage, applyLocal]

/-- Parent browser availability constrains all descendant pages. -/
theorem browser_disconnect_blocks_all_pages
    (m : Model) (p : PageId) :
    ¬ canExecute (step m .browserDisconnected) p := by
  simp [canExecute, step]

/-- An explicit page command cannot mutate any sibling page. -/
theorem page_command_isolated
    (m : Model) (cmd : PageCommand) (q : PageId) (h : q ≠ cmd.page) :
    (applyCommand m cmd).page q = m.page q := by
  simp [applyCommand, updatePage, h]

/-- Parent-context unavailability blocks primitive execution for every page in
    that context before any recovery policy is considered. -/
theorem context_unavailable_blocks_descendants
    (m : Model) (c : ContextId) (p : PageId)
    (hctx : m.graph.contextOf p = c) :
    ¬ canExecute (step m (.contextUnavailable c)) p := by
  simp [canExecute, step, updateContextAvailability, hctx]

end Browser
