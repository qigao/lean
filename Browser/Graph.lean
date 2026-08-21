import Browser.Types

namespace Browser

/-- Runtime topology. The formal model intentionally keeps identity and
    ownership separate from mutable page state. -/
structure RuntimeGraph where
  contextOf : PageId → ContextId
  pageOfFrame : FrameId → PageId
  parentOfFrame : FrameId → Option FrameId

/-- Pages are related at account/browser-context scope when they belong to
    the same browser context. -/
def sameContext (g : RuntimeGraph) (a b : PageId) : Prop :=
  g.contextOf a = g.contextOf b

end Browser
