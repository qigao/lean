import Browser.Timeout
import Browser.Proofs

/-!
# Timeout integration/reference spec

Executable examples for the earlier whole-runtime deadline/timeout model.
Retained as regression evidence through `Browser.Integration`; the supported
public recovery surface is `Browser.ProofAPI`.
-/

namespace Browser

/-- Human suspension must not extend an already armed absolute deadline. -/
example (m : Model) (p : PageId) (action : ActionId) (deadline : Time)
    (hinput : (m.page p).input = .automation) :
    ((step (armDeadline m p action deadline .human) (.local { page := p, kind := .humanInput })).page p).deadline =
      some { action := action, expiresAt := deadline } := by
  simp [armDeadline, armDeadlineState, step, updatePage, applyLocal, hinput]

/-- An expired action cannot remain executable. -/
example (m : Model) (p : PageId) (action : ActionId) (deadline now : Time)
    (hexpired : deadline ≤ now) :
    ¬ canExecute (expirePage (armDeadline m p action deadline .network) p action now) p := by
  exact expired_page_blocks_execution m p action deadline now .network hexpired

/-- Timeout cause remains observable so policy can distinguish human/page/network/protocol stalls. -/
example (m : Model) (p : PageId) (action : ActionId) (deadline now : Time)
    (hexpired : deadline ≤ now) :
    ((expirePage (armDeadline m p action deadline .network) p action now).page p).timeout = some .network := by
  exact expired_page_records_reason m p action deadline now .network hexpired

/-- Expiring one Page is isolated from sibling Page state. -/
example (m : Model) (p q : PageId) (action : ActionId) (now : Time) (hne : q ≠ p) :
    (expirePage m p action now).page q = m.page q := by
  exact expire_page_isolated m p q action now hne

/-- Timeout is terminal under ordinary pause/resume policy commands. -/
example (s : PageState) (h : s.action = .timedOut) :
    (applyPageCommand .pauseAutomation s).action = .timedOut ∧
    (applyPageCommand .resumeAutomation s).action = .timedOut := by
  constructor
  · exact timed_out_terminal_under_pause s h
  · exact timed_out_terminal_under_resume s h

/-- Deadline events use the same page-local event path as browser/CDP events and
    require the ActionId that owns the timer. -/
example (m : Model) (p : PageId) (action : ActionId) (deadline now : Time)
    (hcurrent : (m.page p).currentAction = some action)
    (hexpired : deadline ≤ now) :
    ¬ canExecute
      (step
        (step m (.local { page := p, kind := .deadlineArmed action deadline .protocol }))
        (.local { page := p, kind := .deadlineReached action now })) p := by
  simp [canExecute, step, updatePage, applyLocal, armActionDeadlineState,
    expireActionDeadlineState, armDeadlineState, expirePageState, timeoutReason,
    hcurrent, hexpired]

end Browser
