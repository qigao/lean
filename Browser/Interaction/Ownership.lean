import Browser.Interaction.Types

namespace Browser.Interaction

/-- Only the ownership edges needed for interaction authorization. -/
structure Ownership where
  pageContext : PageId → ContextId
  framePage : FrameId → PageId

/-- Context-scoped effects may address only Pages owned by that Context. -/
def authorizedContextEffect
    (ownership : Ownership) (context : ContextId) (page : PageId) : Bool :=
  decide (ownership.pageContext page = context)

/-- Frame-scoped effects may be authorized only through the owning Page. -/
def authorizedFrameEffect
    (ownership : Ownership) (page : PageId) (frame : FrameId) : Bool :=
  decide (ownership.framePage frame = page)

theorem foreign_context_page_rejected
    (ownership : Ownership) (context : ContextId) (page : PageId)
    (h : ownership.pageContext page ≠ context) :
    authorizedContextEffect ownership context page = false := by
  simp [authorizedContextEffect, h]

theorem owning_context_page_authorized
    (ownership : Ownership) (context : ContextId) (page : PageId)
    (h : ownership.pageContext page = context) :
    authorizedContextEffect ownership context page = true := by
  simp [authorizedContextEffect, h]

theorem foreign_page_frame_rejected
    (ownership : Ownership) (page : PageId) (frame : FrameId)
    (h : ownership.framePage frame ≠ page) :
    authorizedFrameEffect ownership page frame = false := by
  simp [authorizedFrameEffect, h]

theorem owning_page_frame_authorized
    (ownership : Ownership) (page : PageId) (frame : FrameId)
    (h : ownership.framePage frame = page) :
    authorizedFrameEffect ownership page frame = true := by
  simp [authorizedFrameEffect, h]

end Browser.Interaction
