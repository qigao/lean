import Browser.Types

namespace Browser

/-- Runtime topology. The formal model intentionally keeps identity and
    ownership separate from mutable page state. -/
structure RuntimeGraph where
  contextOf : PageId → ContextId
  pageOfFrame : FrameId → PageId
  parentOfFrame : FrameId → Option FrameId

/-- Functional Page -> Context rebinding used by accepted graph-lifecycle
    messages. Historical mappings may remain for dead nodes; NodeEpoch decides
    whether a later message is allowed to use them. -/
def bindPageContext (g : RuntimeGraph) (page : PageId) (context : ContextId) : RuntimeGraph :=
  { g with contextOf := fun q => if q = page then context else g.contextOf q }

/-- Functional Frame -> Page/parent rebinding for a new Frame incarnation. -/
def bindFrame
    (g : RuntimeGraph) (frame : FrameId) (page : PageId) (parent : Option FrameId) : RuntimeGraph :=
  { g with
    pageOfFrame := fun q => if q = frame then page else g.pageOfFrame q
    parentOfFrame := fun q => if q = frame then parent else g.parentOfFrame q }

/-- Pages are related at account/browser-context scope when they belong to
    the same browser context. -/
def sameContext (g : RuntimeGraph) (a b : PageId) : Prop :=
  g.contextOf a = g.contextOf b

end Browser
