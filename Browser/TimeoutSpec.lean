import Browser.Timeout
import Browser.Proofs

namespace Browser

/-- Human suspension must not extend an already armed absolute deadline. -/
example (m : Model) (p : PageId) (deadline : Time)
    (hinput : (m.page p).input = .automation) :
    ((step (armDeadline m p deadline .human) (.local { page := p, kind := .humanInput })).page p).deadline =
      some { expiresAt := deadline } := by
  simp [armDeadline, step, updatePage, applyLocal, hinput]

/-- An expired action cannot remain executable. -/
example (m : Model) (p : PageId) (deadline now : Time)
    (hexpired : deadline ≤ now) :
    ¬ canExecute (expirePage (armDeadline m p deadline .network) p now) p := by
  exact expired_page_blocks_execution m p deadline now .network hexpired

/-- Timeout cause remains observable so policy can distinguish human/page/network/protocol stalls. -/
example (m : Model) (p : PageId) (deadline now : Time)
    (hexpired : deadline ≤ now) :
    ((expirePage (armDeadline m p deadline .network) p now).page p).timeout = some .network := by
  exact expired_page_records_reason m p deadline now .network hexpired

/-- Expiring one Page is isolated from sibling Page state. -/
example (m : Model) (p q : PageId) (now : Time) (hne : q ≠ p) :
    (expirePage m p now).page q = m.page q := by
  exact expire_page_isolated m p q now hne

end Browser
